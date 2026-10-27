"""dinlat/run.py SEED EPSJSON EMB OFFSET OUTJSON — entrena y evalúa los brazos de un seed.
Cadena: texto del frame → Gemma-2-2B (capa 13, media de tokens; gemma-embed) → PCA(d=16, ajustada SOLO
con frames train del seed, sin etiquetas) → rotación ortogonal aleatoria fija Q_rand → s_t.
Brazos: PERS, AR2 (ridge), ORB_LIBRE (Koopman denso + corrector Brenier, contexto inferido de la historia),
ORB_SESGO (bloques rotación-escala de model.rs + contexto one-hot familia+p VERDADERO).
Controles (re-entrenando AR2 y ORB_LIBRE): C1 permutado entre episodios, C2 barajado temporal."""
import sys, json, math
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
DIM, DL, T, TMIN = 2304, 16, 9, 2      # dim Gemma, dim latente, frames, historia mínima para evaluar
FAMS = ["rotar", "trasladar", "escalar", "logistica", "modsuma", "regla_signo", "cizalla", "tienda", "rodar_suma"]
PR = {"rotar": (0.15, 0.6), "trasladar": (-0.08, 0.08), "escalar": (0.88, 1.08), "cizalla": (0.1, 0.35),
      "logistica": (3.6, 3.99), "tienda": (1.6, 1.99), "modsuma": (0.1, 0.5), "regla_signo": (0.1, 0.4), "rodar_suma": (0.1, 0.5)}
STEPS, LR, WD = 2500, 1e-3, 1e-3
def ctx_true(e):
    c = np.zeros(len(FAMS) + 1); c[FAMS.index(e["fam"])] = 1; lo, hi = PR[e["fam"]]; c[-1] = (e["p"] - lo) / (hi - lo); return c
# ---------------- modelos ----------------
class SetCtx(nn.Module):                       # contexto inferido de transiciones observadas (sin etiquetas)
    def __init__(s, d=DL, h=32, k=8): super().__init__(); s.f = nn.Sequential(nn.Linear(2 * d, h), nn.Tanh(), nn.Linear(h, k))
    def forward(s, S, t):                      # S: B×T×d ; usa transiciones τ<t (τ+1≤t)
        a, b = S[:, :t], S[:, 1:t + 1]; return s.f(torch.cat([a, b - a], -1)).mean(1)
class Brenier(nn.Module):                      # u ← u + η Wᵀ(softplus(a)⊙σ(Wu+b)), r pasos (gradiente de convexa)
    def __init__(s, d=DL, k=32, r=2): super().__init__(); s.W = nn.Parameter(0.1 * torch.randn(k, d)); s.b = nn.Parameter(torch.zeros(k)); s.a = nn.Parameter(torch.zeros(k)); s.eta = nn.Parameter(torch.tensor(0.1)); s.r = r
    def forward(s, u):
        for _ in range(s.r): u = u + s.eta * (torch.sigmoid(u @ s.W.T + s.b) * nn.functional.softplus(s.a)) @ s.W
        return u
class OrbLibre(nn.Module):                     # z=[s; g(s,c)], y = K(c) z + β(c), K denso sin estructura
    def __init__(s, d=DL, m=16, k=8, nb=8):
        super().__init__(); n = d + m; s.d, s.n, s.nb = d, n, nb; s.ctx = SetCtx(d, 32, k)
        s.g = nn.Sequential(nn.Linear(d + k, 32), nn.Tanh(), nn.Linear(32, m), nn.Tanh())
        s.K0 = nn.Parameter(torch.eye(n)); s.Kb = nn.Parameter(0.01 * torch.randn(nb, n, n)); s.mix = nn.Linear(k, nb); s.beta = nn.Linear(k, n); s.cor = Brenier(d)
    def step(s, x, c):
        z = torch.cat([x, s.g(torch.cat([x, c], -1))], -1); K = s.K0 + torch.einsum("bk,kij->bij", s.mix(c), s.Kb)
        y = torch.einsum("bij,bj->bi", K, z) + s.beta(c); return s.cor(y[:, :s.d])
    def forward(s, S, t): return s.step(S[:, t], s.ctx(S, t))
    def ctxof(s, S, t): return s.ctx(S, t)
def cayley(A): I = torch.eye(A.shape[-1]); return torch.linalg.solve(I + A, I - A)
class OrbSesgo(nn.Module):                     # model.rs: y = Q·blk(ρ_j R(ω_j))·Qᵀ z + β, Q Cayley, contexto one-hot+p
    def __init__(s, d=DL, m=0, cdim=len(FAMS) + 1):
        super().__init__(); n = d; s.d, s.nb = d, d // 2; s.Sk = nn.Parameter(0.01 * torch.randn(n, n))
        s.h = nn.Sequential(nn.Linear(cdim, 32), nn.Tanh(), nn.Linear(32, 2 * s.nb + n)); s.cor = Brenier(d)
    def step(s, x, c):
        o = s.h(c); w, lr, beta = o[:, :s.nb], o[:, s.nb:2 * s.nb], o[:, 2 * s.nb:]
        Q = cayley(s.Sk - s.Sk.T); u = x @ Q; u2 = u.view(-1, s.nb, 2); rho = torch.exp(lr)
        cw, sw = torch.cos(w), torch.sin(w); a, b = u2[..., 0], u2[..., 1]
        v = torch.stack([rho * (cw * a - sw * b), rho * (sw * a + cw * b)], -1).view(-1, s.d)
        return s.cor(v @ Q.T + beta)
def _loss(mdl, S, C):
    L = 0
    for t in range(1, T - 1):
        pred = mdl(S, t) if C is None else mdl.step(S[:, t], C)
        L = L + ((pred - S[:, t + 1]) ** 2).mean()
    return L / (T - 2)
def train(mdl, S, C, seed):                    # parada temprana: 25% de episodios train como validación interna
    g = np.random.default_rng(seed + 7); idx = g.permutation(len(S)); nv = len(S) // 4; va, tr = idx[:nv], idx[nv:]
    Ct = None if C is None else C[tr]; Cv = None if C is None else C[va]
    opt = torch.optim.Adam(mdl.parameters(), lr=LR, weight_decay=WD); best = (float("inf"), None)
    for it in range(STEPS):
        opt.zero_grad(); _loss(mdl, S[tr], Ct).backward(); opt.step()
        if it % 50 == 49:
            with torch.no_grad(): v = float(_loss(mdl, S[va], Cv))
            if v < best[0]: best = (v, {k: x.clone() for k, x in mdl.state_dict().items()})
    if best[1] is not None: mdl.load_state_dict(best[1])
    return mdl
def ar_fit(S, a=1.0):
    X = np.concatenate([np.concatenate([S[:, t], S[:, t - 1], np.ones((len(S), 1))], 1) for t in range(1, T - 1)])
    Y = np.concatenate([S[:, t + 1] for t in range(1, T - 1)]); R = a * np.eye(X.shape[1]); R[-1, -1] = 0
    return np.linalg.solve(X.T @ X + R, X.T @ Y)
def ar_pred(W, S, t): return np.concatenate([S[:, t], S[:, t - 1], np.ones((len(S), 1))], 1) @ W
# ---------------- evaluación ----------------
def errs(predf, S):                            # SE por episodio, promediado sobre t=TMIN..T-2 (objetivo s_{t+1})
    return np.mean([((predf(t) - S[:, t + 1]) ** 2).sum(1) for t in range(TMIN, T - 1)], 0)
def roll3(stepf, S):                           # secundario: desde t=TMIN, 3 pasos autoregresivos, contexto fijo
    x = S[:, TMIN]; e = 0
    for h in range(1, 4): x = stepf(x); e = e + ((x - S[:, TMIN + h]) ** 2).sum(1)
    return e / 3
def run(seed, eps, E):
    sd = int(seed, 16); rng = np.random.default_rng(sd + 99)
    sp = np.array([e["split"] for e in eps]); E = E.reshape(len(eps), T, DIM)
    tr = sp == "train"; Xtr = E[tr].reshape(-1, DIM); mu = Xtr.mean(0)
    U, sv, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); P = Vt[:DL]; scale = sv[:DL].max() / math.sqrt(len(Xtr))
    Qr, _ = np.linalg.qr(rng.standard_normal((DL, DL)))          # rotación ortogonal aleatoria fija
    lat = lambda X: ((X - mu) @ P.T / scale) @ Qr
    S = lat(E.reshape(-1, DIM)).reshape(len(eps), T, DL)
    expl = float((sv[:DL] ** 2).sum() / (sv ** 2).sum())
    Ctr = np.array([ctx_true(e) for e in eps])
    out = {"seed": seed, "pca_var_explicada": expl, "n_eps": {k: int((sp == k).sum()) for k in ("train", "ho", "tf")}}
    def fit_eval(Strain, Stests, tag, sesgo=False):
        res = {}; tS = torch.tensor(Strain, dtype=torch.float32)
        W = ar_fit(Strain)
        torch.manual_seed(sd); lib = train(OrbLibre(), tS, None, sd)
        if sesgo: torch.manual_seed(sd + 1); ses = train(OrbSesgo(), tS, torch.tensor(Ctr[tr], dtype=torch.float32), sd)
        for name, (St, Ct) in Stests.items():
            tt = torch.tensor(St, dtype=torch.float32)
            with torch.no_grad():
                r = {"PERS": errs(lambda t: St[:, t], St), "AR2": errs(lambda t: ar_pred(W, St, t), St),
                     "ORB_LIBRE": errs(lambda t: lib(tt, t).numpy(), St)}
                c = lib.ctxof(tt, TMIN); r["ORB_LIBRE_h3"] = roll3(lambda x: lib.step(torch.tensor(x, dtype=torch.float32), c).numpy(), St)
                prev = {"p": St[:, TMIN - 1]}
                def arstep(x):
                    y = np.concatenate([x, prev["p"], np.ones((len(x), 1))], 1) @ W; prev["p"] = x; return y
                r["AR2_h3"] = roll3(arstep, St); r["PERS_h3"] = roll3(lambda x: x, St)
                if sesgo:
                    cc = torch.tensor(Ct, dtype=torch.float32)
                    r["ORB_SESGO"] = errs(lambda t: ses.step(tt[:, t], cc).numpy(), St)
                    r["ORB_SESGO_h3"] = roll3(lambda x: ses.step(torch.tensor(x, dtype=torch.float32), cc).numpy(), St)
            res[name] = {k: v.tolist() for k, v in r.items()}
        out[tag] = res
    fams = [e["fam"] for e in eps]
    fit_eval(S[tr], {"ho": (S[sp == "ho"], Ctr[sp == "ho"]), "tf": (S[sp == "tf"], Ctr[sp == "tf"])}, "main", sesgo=True)
    out["fam"] = {k: [f for f, s_ in zip(fams, sp) if s_ == k] for k in ("ho", "tf")}
    # C1: permutado entre episodios (cada t toma el frame de otro episodio) — rompe la transición
    def c1(X):
        Y = X.copy()
        for t in range(T): Y[:, t] = X[rng.permutation(len(X)), t]
        return Y
    # C2: barajado temporal dentro del episodio
    def c2(X): return np.stack([x[rng.permutation(T)] for x in X])
    for tag, f in (("C1_permutado", c1), ("C2_barajado", c2)):
        fit_eval(f(S[tr]), {"ho": (f(S[sp == "ho"]), None)}, tag)
    return out
if __name__ == "__main__":
    seed, ej, emb, off, oj = sys.argv[1:6]; eps = json.load(open(ej)); n = len(eps) * T
    E = np.fromfile(emb, dtype="<f4", count=n * DIM, offset=int(off) * DIM * 4).astype(np.float64)
    assert E.size == n * DIM, (E.size, n * DIM)
    json.dump(run(seed, eps, E), open(oj, "w"))
