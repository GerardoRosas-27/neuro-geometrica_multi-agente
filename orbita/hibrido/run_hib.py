"""hibrido/run_hib.py SEED EPSJSON EMB OFFSET OUTJSON — híbridos ÓRBITA×AR2 sobre los latentes de dinlat.
Reutiliza modelos/latentes de dinlat/run.py (mismo código, mismos seeds ⇒ mismos ORB_LIBRE/ORB_SESGO/AR2; se ABORTA si
los errores recalculados no coinciden con res_<seed>.json de dinlat cuando se pasa DINLAT_RES).
Brazos nuevos:
  MIX50      0.5·AR2 + 0.5·ORB_LIBRE (control sin aprendizaje)
  HIB_GATE   softmax(W·f_t + b) sobre {AR2, ORB_LIBRE}; f_t = errores PASADOS de cada experto (τ<t) + t/T
  HIB_HEDGE  pesos ∝ exp(−η·err_pasado_k), η elegido en grilla sobre predicciones cross-fit de train
  HIB_RES    AR2 + R(S,t), R = OrbLibre entrenado sobre residuos de AR2 (inicialización ≈ 0)
  HIB_GATE_S compuerta sobre {AR2, ORB_LIBRE, ORB_SESGO} (ORB_SESGO usa etiquetas verdaderas → secundario)
  ORACLE_FAM techo: por familia, el mejor de {AR2, ORB_LIBRE} usando errores del propio split (NO es un modelo)
La compuerta se entrena SOLO con predicciones fuera-de-pliegue (K=4 cross-fit por episodio) sobre train."""
import sys, os, json
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dinlat"))
import run as D
T, TMIN, DL = D.T, D.TMIN, D.DL
K_FOLD, GATE_STEPS, GATE_WD, EPS = 4, 1500, 1e-3, 1e-6
HEDGE_GRID = [0.0, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
# ---------- predicciones de un paso (todas t=1..T-2; objetivo s_{t+1}) ----------
def preds_ar(W, S): return {t: D.ar_pred(W, S, t) for t in range(1, T - 1)}
def preds_torch(f, S):
    tt = torch.tensor(S, dtype=torch.float32)
    with torch.no_grad(): return {t: f(tt, t).numpy().astype(np.float64) for t in range(1, T - 1)}
class ResOrb(nn.Module):                       # residuo: R(S,t) = OrbLibre(S,t) − s_t  (K0=I ⇒ R≈0 al inicio)
    def __init__(s): super().__init__(); s.o = D.OrbLibre()
    def forward(s, S, t): return s.o(S, t) - S[:, t]
def train_res(S, W, sd):
    """Entrena R sobre residuos s_{t+1} − AR2_t con la misma pérdida/parada temprana que dinlat (25% val interna)."""
    A = {t: torch.tensor(D.ar_pred(W, S, t), dtype=torch.float32) for t in range(1, T - 1)}
    class Wrap(nn.Module):                     # m(S,t) = AR2_t[idx] + R(S,t)  ⇒  D._loss ajusta R sobre residuos
        def __init__(s): super().__init__(); s.r = ResOrb(); s.idx = None
        def forward(s, Sx, t): return A[t][s.idx] + s.r(Sx, t)
    torch.manual_seed(sd + 2); m = Wrap(); St = torch.tensor(S, dtype=torch.float32)
    g = np.random.default_rng(sd + 7); perm = g.permutation(len(S)); nv = len(S) // 4; va, tr = perm[:nv], perm[nv:]
    opt = torch.optim.Adam(m.r.parameters(), lr=D.LR, weight_decay=D.WD); best = (float("inf"), None)
    def loss(ix): m.idx = torch.tensor(ix); return D._loss(m, St[ix], None)
    for it in range(D.STEPS):
        opt.zero_grad(); loss(tr).backward(); opt.step()
        if it % 50 == 49:
            with torch.no_grad(): v = float(loss(va))
            if v < best[0]: best = (v, {k: x.clone() for k, x in m.r.state_dict().items()})
    if best[1] is not None: m.r.load_state_dict(best[1])
    return m.r
# ---------- características de la compuerta: SOLO pasado ----------
def past_feats(P, S):
    """P: lista de expertos, cada uno dict t→pred (B×d). Devuelve dict t→F (B×(2K+1)) para t=TMIN..T-2.
    En t se usan errores de transiciones τ=1..t-1 (pred_τ vs s_{τ+1}, con τ+1 ≤ t): nada de s_{t+1} en adelante."""
    out = {}
    for t in range(TMIN, T - 1):
        cols = []
        for p in P:
            e = np.stack([((p[tau] - S[:, tau + 1]) ** 2).sum(1) for tau in range(1, t)], 1)
            cols += [np.log(e.mean(1) + EPS), np.log(e[:, -1] + EPS)]
        out[t] = np.stack(cols + [np.full(len(S), t / T)], 1)
    return out
class Gate(nn.Module):
    def __init__(s, nf, k): super().__init__(); s.l = nn.Linear(nf, k); nn.init.zeros_(s.l.weight); nn.init.zeros_(s.l.bias)
    def forward(s, F): return torch.softmax(s.l(F), -1)
def mix(w, P, t): return sum(w[:, k:k + 1] * P[k][t] for k in range(len(P)))
def fit_gate(P, S, sd):
    F = past_feats(P, S); torch.manual_seed(sd + 3); g = Gate(F[TMIN].shape[1], len(P))
    tP = [{t: torch.tensor(p[t]) for t in p} for p in P]; tS = torch.tensor(S); tF = {t: torch.tensor(F[t]) for t in F}
    opt = torch.optim.Adam(g.parameters(), lr=1e-2, weight_decay=GATE_WD); g.double()
    for _ in range(GATE_STEPS):
        opt.zero_grad()
        L = sum(((mix(g(tF[t]), tP, t) - tS[:, t + 1]) ** 2).sum(1).mean() for t in F) / len(F); L.backward(); opt.step()
    return g
def apply_gate(g, P, S):
    F = past_feats(P, S)
    with torch.no_grad(): return {t: mix(g(torch.tensor(F[t])).numpy(), P, t) for t in F}
def hedge_w(P, S, eta):
    F = past_feats(P, S); return {t: (lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))(-eta * np.exp(F[t][:, 0::2][:, :len(P)])) for t in F}
def apply_hedge(P, S, eta):
    W = hedge_w(P, S, eta); return {t: mix(W[t], P, t) for t in W}
def errs_from(pr, S): return np.mean([((pr[t] - S[:, t + 1]) ** 2).sum(1) for t in range(TMIN, T - 1)], 0)
# ---------- cross-fit sobre train para entrenar la compuerta ----------
def crossfit(S, C, sd, sesgo):
    n = len(S); folds = np.random.default_rng(sd + 11).permutation(n) % K_FOLD
    P = {"AR2": {}, "ORB_LIBRE": {}, "ORB_SESGO": {}}
    for k in range(K_FOLD):
        a, b = folds != k, folds == k; W = D.ar_fit(S[a]); torch.manual_seed(sd + 100 + k)
        lib = D.train(D.OrbLibre(), torch.tensor(S[a], dtype=torch.float32), None, sd + 100 + k)
        parts = {"AR2": preds_ar(W, S[b]), "ORB_LIBRE": preds_torch(lib, S[b])}
        if sesgo:
            torch.manual_seed(sd + 200 + k); ses = D.train(D.OrbSesgo(), torch.tensor(S[a], dtype=torch.float32), torch.tensor(C[a], dtype=torch.float32), sd + 200 + k)
            cb = torch.tensor(C[b], dtype=torch.float32); parts["ORB_SESGO"] = preds_torch(lambda x, t: ses.step(x[:, t], cb), S[b])
        for name, pr in parts.items():
            for t, v in pr.items(): P[name].setdefault(t, np.zeros((n, DL)))[b] = v
    return {k: v for k, v in P.items() if v}
def latents(seed, eps, E):
    """Idéntico a dinlat/run.py y vstimesfm/latents.py (PCA train, Q_rand, rng de C1/C2 en el mismo orden)."""
    import math
    sd = int(seed, 16); rng = np.random.default_rng(sd + 99)
    sp = np.array([e["split"] for e in eps]); E = E.reshape(len(eps), T, D.DIM)
    tr = sp == "train"; Xtr = E[tr].reshape(-1, D.DIM); mu = Xtr.mean(0)
    U, sv, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); P = Vt[:DL]; scale = sv[:DL].max() / math.sqrt(len(Xtr))
    Qr, _ = np.linalg.qr(rng.standard_normal((DL, DL)))
    S = (((E.reshape(-1, D.DIM) - mu) @ P.T / scale) @ Qr).reshape(len(eps), T, DL)
    def c1(X):
        Y = X.copy()
        for t in range(T): Y[:, t] = X[rng.permutation(len(X)), t]
        return Y
    def c2(X): return np.stack([x[rng.permutation(T)] for x in X])
    out = {"main": {"train": S[tr], "ho": S[sp == "ho"], "tf": S[sp == "tf"]}}
    for tag, f in (("C1_permutado", c1), ("C2_barajado", c2)):
        a = f(S[tr]); b = f(S[sp == "ho"]); out[tag] = {"train": a, "ho": b}
    return out
def run(seed, eps, E):
    sd = int(seed, 16); L = latents(seed, eps, E)
    Ctr = np.array([D.ctx_true(e) for e in eps]); sp = np.array([e["split"] for e in eps]); fams = [e["fam"] for e in eps]
    out = {"seed": seed, "fam": {k: [f for f, s_ in zip(fams, sp) if s_ == k] for k in ("ho", "tf")}}
    def fit_eval(Str, Ctrain, tests, tag, sesgo):
        W = D.ar_fit(Str); tS = torch.tensor(Str, dtype=torch.float32)
        torch.manual_seed(sd); lib = D.train(D.OrbLibre(), tS, None, sd)
        if sesgo: torch.manual_seed(sd + 1); ses = D.train(D.OrbSesgo(), tS, torch.tensor(Ctrain, dtype=torch.float32), sd)
        R = train_res(Str, W, sd)
        CF = crossfit(Str, Ctrain, sd, sesgo); names2 = ["AR2", "ORB_LIBRE"]
        g2 = fit_gate([CF[n] for n in names2], Str, sd)
        eta = min(HEDGE_GRID, key=lambda e: errs_from(apply_hedge([CF[n] for n in names2], Str, e), Str).mean())
        g3 = fit_gate([CF[n] for n in names2 + ["ORB_SESGO"]], Str, sd) if sesgo else None
        res = {"_eta_hedge": eta}
        for name, (St, Ct) in tests.items():
            P = {"AR2": preds_ar(W, St), "ORB_LIBRE": preds_torch(lib, St)}
            with torch.no_grad(): rr = preds_torch(R, St)
            r = {k: errs_from(v, St) for k, v in P.items()}
            r["PERS"] = errs_from({t: St[:, t] for t in range(1, T - 1)}, St)
            r["MIX50"] = errs_from({t: 0.5 * P["AR2"][t] + 0.5 * P["ORB_LIBRE"][t] for t in P["AR2"]}, St)
            r["HIB_GATE"] = errs_from(apply_gate(g2, [P[n] for n in names2], St), St)
            r["HIB_HEDGE"] = errs_from(apply_hedge([P[n] for n in names2], St, eta), St)
            r["HIB_RES"] = errs_from({t: P["AR2"][t] + rr[t] for t in P["AR2"]}, St)
            if sesgo:
                cc = torch.tensor(Ct, dtype=torch.float32); P["ORB_SESGO"] = preds_torch(lambda x, t: ses.step(x[:, t], cc), St)
                r["ORB_SESGO"] = errs_from(P["ORB_SESGO"], St)
                r["HIB_GATE_S"] = errs_from(apply_gate(g3, [P[n] for n in names2 + ["ORB_SESGO"]], St), St)
            res[name] = {k: v.tolist() for k, v in r.items()}
        out[tag] = res
    m = L["main"]
    fit_eval(m["train"], Ctr[sp == "train"], {"ho": (m["ho"], Ctr[sp == "ho"]), "tf": (m["tf"], Ctr[sp == "tf"])}, "main", True)
    for tag in ("C1_permutado", "C2_barajado"):
        fit_eval(L[tag]["train"], None, {"ho": (L[tag]["ho"], None)}, tag, False)
    ref = os.environ.get("DINLAT_RES")
    if ref:                                    # verificación de identidad con dinlat (mismos modelos base)
        d = json.load(open(ref))
        for sp_ in ("ho", "tf"):
            for k in ("PERS", "AR2", "ORB_LIBRE", "ORB_SESGO"):
                assert np.allclose(d["main"][sp_][k], out["main"][sp_][k], rtol=1e-6, atol=1e-9), ("no coincide con dinlat", sp_, k)
    return out
if __name__ == "__main__":
    seed, ej, emb, off, oj = sys.argv[1:6]; eps = json.load(open(ej)); n = len(eps) * T
    E = np.fromfile(emb, dtype="<f4", count=n * D.DIM, offset=int(off) * D.DIM * 4).astype(np.float64)
    assert E.size == n * D.DIM, (E.size, n * D.DIM)
    json.dump(run(seed, eps, E), open(oj, "w"))
