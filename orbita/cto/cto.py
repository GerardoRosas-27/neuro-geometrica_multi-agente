"""Cognitive Transformer + ÓRBITA (docs/cognitive_transformer_orbita.md; prerregistro docs/prereg_cognitive_transformer_orbita.md).
Atención sobre slots de estado no lingüísticos (6 puntos x latente ÓRBITA 8-d). Q desde el estado ÓRBITA, K/V desde memoria
(observaciones del episodio + CDT-RQM consolidada por error de predicción), Δz residual, después ÓRBITA (congelada) evoluciona z_{t+1}.
El LLM (Gemma 2 2B, D2/numnorm) es periferia: sólo entra como contexto c ya extraído (cache de exp/orbita-yo crece/ctx).
Uso: cto.py train PH SEED ARM | stream PH SEED | calib | analyze PH | smoke
"""
import sys, os, json, math, time, csv, hashlib
import numpy as np
import torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(f"{HERE}/../..")
DATA = "/workspace/orbita-data"; CRE = f"{DATA}/crece"; OUT = os.environ.get("CTO_OUT", f"{DATA}/cto")
ART = f"{ROOT}/artifacts/orbita/cognitive_transformer"
SEEDS = {"dev": [f"0xA4{i:02X}" for i in range(16)], "confirm": [f"0xB4{i:02X}" for i in range(16)]}
KS = (2, 4, 8); CK = (0, 150, 600, 2400); NF = 12; P = 6; M = 8; NB, NT = 4, 2
TRAIN_STEPS = int(os.environ.get("CTO_STEPS", "4000")); BATCH = 32; LR = 3e-3
BANK_TRAIN = 512; RET_R = 4; CDT_CAP = 600; P_CORRUPT = 0.3
ARMS_TRAINED = ("CT", "SELF", "MLP", "NOORB")

# ---------------- reglas (port exacto de src/v4_dataset.rs apply_point y orbita/src/fam.rs rule) ----------------
def rot(t, x, y): return np.cos(t) * x - np.sin(t) * y, np.sin(t) * x + np.cos(t) * y
def rule(f, p, X):
    """f:int, p:[2], X:[...,12] -> [...,12]"""
    x, y = X[..., 0::2], X[..., 1::2]
    if f == 0: a, b = x + 0.5 * p[0], y + 0.5 * p[1]
    elif f == 1: a, b = rot(0.6 * p[0] + 0.15 * p[1], x, y)
    elif f == 2:
        t = 0.8 * p[0] + 0.2 * p[1]; c, s = np.cos(2 * t), np.sin(2 * t); a, b = c * x + s * y, s * x - c * y
    elif f == 3: a, b = np.exp(0.3 * p[0]) * x, np.exp(0.3 * p[1]) * y
    elif f == 4: a, b = x + 0.5 * p[0] * y, y + 0.3 * p[1] * x
    elif f == 5: a, b = x + 0.3 * (p[0] * x + p[1] * y) + 0.3 * p[1], y + 0.3 * (-p[1] * x + p[0] * y) + 0.3 * p[0]
    elif f == 6:
        u, w = rot(0.6 * p[0], x, y); sc = np.exp(0.3 * p[1]); a, b = sc * u, sc * w
    elif f == 7:
        g = 1 + 0.3 * p[0] * np.tanh(x * x + y * y - 1); a, b = g * x + 0.2 * p[1] * np.sin(y), g * y + 0.2 * p[1] * np.sin(x)
    elif f == 8:
        u, w = rot(0.6 * p[0], x, y); a, b = u + 0.25 * p[1] * np.sin(2 * w), w + 0.25 * p[1] * np.sin(2 * u)
    elif f == 9:
        r = np.sqrt(x * x + y * y); u, w = rot(0.4 * p[0] + 0.8 * p[1] * (r - 0.7), x, y); s = np.exp(0.1 * p[1]); a, b = s * u, s * w
    elif f == 10:
        r, th = np.sqrt(x * x + y * y), np.arctan2(y, x); r2 = r * (1 + 0.15 * p[1] * np.sin(3 * th)); t2 = th + 0.5 * p[0]
        a, b = r2 * np.cos(t2), r2 * np.sin(t2)
    else:
        u = x + 0.3 * p[0] * np.tanh(2 * y); a, b = u, y + 0.3 * p[1] * np.tanh(2 * u)
    out = np.empty_like(X); out[..., 0::2] = a; out[..., 1::2] = b; return out
def traj(f, p, x0, T):
    ys = [x0]
    for _ in range(T): ys.append(rule(f, p, ys[-1]))
    return np.stack(ys)                                  # [T+1,12], ys[0]=x0
def context_of(f, p):
    c = np.zeros(10); c[f] = 1; c[8], c[9] = p; return c
def make_object(rng, cls):
    cx, cy, r, ph = rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(.5, 1), rng.uniform(0, 2 * math.pi); v = np.zeros(12)
    for i in range(6):
        if cls == 0: a = ph + 2 * math.pi * (i + rng.uniform(-.3, .3)) / 6; rr = r * rng.uniform(.6, 1.2); px, py = rr * math.cos(a), rr * math.sin(a)
        elif cls == 1: a = ph + 2 * math.pi * i / 6; px, py = r * math.cos(a), r * math.sin(a)
        else:
            t = 3 * i / 6; e = math.floor(t); fr = t - e; a0 = ph + 2 * math.pi * e / 3; a1 = ph + 2 * math.pi * (e + 1) / 3
            px, py = r * ((1 - fr) * math.cos(a0) + fr * math.cos(a1)), r * ((1 - fr) * math.sin(a0) + fr * math.sin(a1))
        v[2 * i], v[2 * i + 1] = cx + px, cy + py
    return v
def rel_err(a, b):
    d = np.sqrt(((a - b) ** 2).sum(-1)) / np.maximum(np.sqrt((b ** 2).sum(-1)), 1e-9)
    return np.where(np.isfinite(d), d, 1e9)

# ---------------- ÓRBITA congelada (misma matemática que orbita/train_orbita.py / src/model.rs) ----------------
class Orb:
    def __init__(s, path):
        w = json.load(open(path)); T = lambda k: torch.tensor(w[k])
        s.w1, s.b1, s.w2, s.b2, s.w3, s.b3, s.w4, s.b4, s.wc = [T(k) for k in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "wc")]
        s.r, s.eta, s.kc = w["r"], w["eta"], w["kc"]; s.n_params = w["n_params"]; s.sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        iu = torch.triu_indices(M, M, 1); s.iu0, s.iu1 = iu[0], iu[1]
    def ctx(s, c):
        o = torch.tanh(c @ s.w1.T + s.b1) @ s.w2.T + s.b2; k = 0
        om = o[:, :NB]; sv = o[:, NB:2 * NB]; k = 2 * NB
        rho = torch.cat([torch.exp(sv[:, :NT]), torch.exp(-torch.nn.functional.softplus(sv[:, NT:]))], 1)
        ns = M * (M - 1) // 2; a = o[:, k:k + ns]; k += ns
        S = torch.zeros(c.shape[0], M, M); S[:, s.iu0, s.iu1] = a; S = S - S.transpose(1, 2); I = torch.eye(M).expand_as(S)
        Q = torch.linalg.solve(I - 0.5 * S, I + 0.5 * S)
        beta = o[:, k:k + M]; k += M; bk = o[:, k:k + s.kc]; k += s.kc; ak = torch.nn.functional.softplus(o[:, k:k + s.kc])
        return dict(Q=Q, rho=rho, om=om, beta=beta, bk=bk, ak=ak, c=c)
    def encode(s, x, cp):                                  # x [B,12] -> z [B,6,8]
        q = x.view(x.shape[0], P, 2); c = cp["c"][:, None, :].expand(-1, P, -1)
        g = torch.tanh(torch.cat([q, c], -1) @ s.w3.T + s.b3) @ s.w4.T + s.b4
        return torch.cat([q, g], -1)
    def evolve(s, z, cp):                                  # core + corrector, un paso
        Q, rho, om, beta = cp["Q"], cp["rho"], cp["om"], cp["beta"]
        y = torch.einsum("bji,bpj->bpi", Q, z).reshape(z.shape[0], P, NB, 2)
        cs, sn, r = torch.cos(om)[:, None], torch.sin(om)[:, None], rho[:, None]
        y0, y1 = y[..., 0], y[..., 1]
        y = torch.stack([r * (cs * y0 - sn * y1), r * (sn * y0 + cs * y1)], -1).reshape(z.shape)
        u = torch.einsum("bij,bpj->bpi", Q, y) + beta[:, None, :]
        for _ in range(s.r):
            gate = cp["ak"][:, None, :] * torch.sigmoid(u @ s.wc.T + cp["bk"][:, None, :]); u = u + s.eta * gate @ s.wc
        return u
    def step(s, x, cp): return s.evolve(s.encode(x, cp), cp)[..., :2].reshape(x.shape)

# ---------------- memoria: entradas por (transición, slot) ----------------
DK_IN, DV_IN, DSIG = 8 + 7, 6, 7
def episode_entries(orb, obs, cp):
    """obs [B,k+1,12] (x_0..x_k). Devuelve keys_in [B,6k,15], vals_in [B,6k,6], sig [B,7], surprise [B]"""
    B, k1, _ = obs.shape; k = k1 - 1
    X = obs[:, :-1].reshape(B * k, 12); Y = obs[:, 1:].reshape(B * k, 12)
    cpk = {n: v.repeat_interleave(k, 0) for n, v in cp.items()}
    z = orb.encode(X, cpk); pr = orb.evolve(z, cpk)[..., :2]
    xs, ys = X.view(B * k, P, 2), Y.view(B * k, P, 2); e = ys - pr; dx = ys - xs
    surp = (torch.sqrt(((pr.reshape(B * k, 12) - Y) ** 2).sum(-1)) / torch.sqrt((Y ** 2).sum(-1)).clamp_min(1e-9)).view(B, k).mean(1)
    sig = torch.cat([e.view(B, -1, 2).mean(1), e.abs().view(B, -1, 2).mean(1), dx.view(B, -1, 2).mean(1), torch.log(surp.clamp_min(1e-6))[:, None]], 1)
    kin = z.reshape(B, k * P, 8)                          # z = [x, g(x,c)] latente ÓRBITA del slot
    vin = torch.cat([dx, e, xs], -1).view(B, k * P, 6)
    return kin, vin, sig, surp
def full_keys(kin, sig_e): return torch.cat([kin, sig_e[:, None, :].expand(-1, kin.shape[1], -1)], -1)  # [.,.,15]

class Attn(torch.nn.Module):
    def __init__(s, dq, dk, dv, H=4, dh=8, dout=8):
        super().__init__(); s.H, s.dh = H, dh
        s.q = torch.nn.Sequential(torch.nn.Linear(dq, 32), torch.nn.Tanh(), torch.nn.Linear(32, H * dh))
        s.k = torch.nn.Sequential(torch.nn.Linear(dk, 32), torch.nn.Tanh(), torch.nn.Linear(32, H * dh))
        s.v = torch.nn.Sequential(torch.nn.Linear(dv, 32), torch.nn.Tanh(), torch.nn.Linear(32, H * dh))
        s.o = torch.nn.Linear(H * dh, dout)
        with torch.no_grad(): s.o.weight.zero_(); s.o.bias.zero_()
    def forward(s, qi, ki, vi, mask):                     # qi [B,6,dq], ki [B,N,dk], vi [B,N,dv], mask [B,N] bool válido
        B, N = ki.shape[:2]
        Q = s.q(qi).view(B, P, s.H, s.dh); K = s.k(ki).view(B, N, s.H, s.dh); V = s.v(vi).view(B, N, s.H, s.dh)
        a = torch.einsum("bphd,bnhd->bhpn", Q, K) / math.sqrt(s.dh)
        a = a.masked_fill(~mask[:, None, None, :], -1e9); w = torch.softmax(a, -1)
        A = torch.einsum("bhpn,bnhd->bphd", w, V).reshape(B, P, s.H * s.dh)
        return s.o(A), w

class Net(torch.nn.Module):
    """arm: CT (atención estado→memoria + ÓRBITA) | SELF (auto-atención sobre la historia de estados, sin memoria de transiciones)
    | MLP (Δz por MLP con la media de la memoria, sin atención; parámetros emparejados) | NOORB (misma atención, sin ÓRBITA: dinámica libre)"""
    def __init__(s, arm):
        super().__init__(); s.arm = arm; dq = 8 + 2 + DSIG
        if arm == "CT": s.att = Attn(dq, DK_IN, DV_IN)
        elif arm == "SELF": s.att = Attn(8 + 2, 8 + 1, 8)
        elif arm == "MLP":
            s.mlp = torch.nn.Sequential(torch.nn.Linear(dq + DK_IN + DV_IN, 101), torch.nn.Tanh(), torch.nn.Linear(101, 8))
            with torch.no_grad(): s.mlp[2].weight.zero_(); s.mlp[2].bias.zero_()
        elif arm == "NOORB":
            s.att = Attn(dq, DK_IN, DV_IN)
            s.dyn = torch.nn.Sequential(torch.nn.Linear(2 + 10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 8))
            s.dout = torch.nn.Linear(8, 2)
    def n_params(s): return sum(p.numel() for p in s.parameters())

def build_mem(net, orb, obs, cp, bank, mode="real", gen=None):
    """Memoria para un lote de episodios. bank: dict(keys [E,Nmax,17], vals [E,Nmax,6], mask [E,Nmax], sig [E,7]) o None.
    mode: real | rand | none(sin CDT). Devuelve dict con ki, vi, mask, sig."""
    kin, vin, sig, surp = episode_entries(orb, obs, cp)
    ki = full_keys(kin, sig); vi = vin; mask = torch.ones(ki.shape[:2], dtype=torch.bool)
    if bank is not None and bank["sig"].shape[0] > 0:
        d = torch.cdist(sig, bank["sig"]); R = min(RET_R, bank["sig"].shape[0])
        if "self_idx" in bank: d[torch.arange(len(sig)), bank["self_idx"]] = float("inf")
        idx = d.topk(R, largest=False).indices          # [B,R]
        bk, bv, bm = bank["keys"][idx].flatten(1, 2), bank["vals"][idx].flatten(1, 2), bank["mask"][idx].flatten(1, 2)
        ki = torch.cat([ki, bk], 1); vi = torch.cat([vi, bv], 1); mask = torch.cat([mask, bm], 1)
    if mode == "rand":
        g = gen if gen is not None else torch.Generator().manual_seed(0)
        ki = ki.mean((0, 1)) + ki.std((0, 1)) * torch.randn(ki.shape, generator=g)
        vi = vi.mean((0, 1)) + vi.std((0, 1)) * torch.randn(vi.shape, generator=g)
        sig = sig.mean(0) + sig.std(0) * torch.randn(sig.shape, generator=g)
    return dict(ki=ki, vi=vi, mask=mask, sig=sig, surp=surp, obs=obs)

def net_step(net, orb, x, cp, mem):
    B = x.shape[0]
    if net.arm == "NOORB":
        q = x.view(B, P, 2); h = torch.tanh(net.dyn[0](torch.cat([q, cp["c"][:, None].expand(-1, P, -1)], -1))); zf = net.dyn[2](h)
        base = q + net.dout(zf); qi = torch.cat([q, zf[..., :6], base - q, mem["sig"][:, None].expand(-1, P, -1)], -1)
        D, _ = net.att(qi, mem["ki"], mem["vi"], mem["mask"]); return (base + D[..., :2]).reshape(B, 12)
    z = orb.encode(x, cp); u0 = orb.evolve(z, cp); q = x.view(B, P, 2)
    if net.arm == "SELF":
        qi = torch.cat([z, u0[..., :2] - q], -1)
        D, _ = net.att(qi, mem["hk"], mem["hv"], mem["hmask"])
    else:
        qi = torch.cat([z, u0[..., :2] - q, mem["sig"][:, None].expand(-1, P, -1)], -1)
        if net.arm == "CT": D, _ = net.att(qi, mem["ki"], mem["vi"], mem["mask"])
        else:
            mf = mem["mask"].double()[..., None]; mk = (mem["ki"] * mf).sum(1) / mf.sum(1); mv = (mem["vi"] * mf).sum(1) / mf.sum(1)
            D = net.mlp(torch.cat([qi, mk[:, None].expand(-1, P, -1), mv[:, None].expand(-1, P, -1)], -1))
    return orb.evolve(z + D, cp)[..., :2].reshape(B, 12)

def hist_mem(orb, obs, cp):
    """SELF: tokens = slots de los estados observados x_0..x_k (sin Δ/errores de transición, sin CDT)."""
    B, k1, _ = obs.shape; X = obs.reshape(B * k1, 12); cpk = {n: v.repeat_interleave(k1, 0) for n, v in cp.items()}
    z = orb.encode(X, cpk); t = (torch.arange(k1, dtype=torch.float64) / 8).repeat(B).view(B * k1, 1, 1).expand(-1, P, 1)
    hk = torch.cat([z, t], -1).view(B, k1 * P, 9); hv = z.view(B, k1 * P, 8)
    return hk, hv, torch.ones(B, k1 * P, dtype=torch.bool)

def rollout(net, orb, obs, cp, mem, T, arm_kind):
    """desde x_k hasta el paso T (absoluto). arm_kind: net | orb | oep | knn. Devuelve [B,T-k,12] predicciones para pasos k+1..T"""
    k = obs.shape[1] - 1; x = obs[:, -1]; out = []
    if arm_kind == "oep": W = ls_fit(obs)
    for _ in range(T - k):
        if arm_kind == "orb": x = orb.step(x, cp)
        elif arm_kind == "oep": x = ls_step(W, x)
        elif arm_kind == "knn": x = knn_step(orb, x, cp, mem)
        else: x = net_step(net, orb, x, cp, mem)
        x = torch.nan_to_num(x, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6); out.append(x)
    return torch.stack(out, 1)

def ls_fit(obs):                                          # ORACLE_EP: afín 2x3 compartido por puntos (como fam.rs ls_fit), mismos datos k
    B, k1, _ = obs.shape; X = obs[:, :-1].reshape(B, -1, 2); Y = obs[:, 1:].reshape(B, -1, 2)
    U = torch.cat([X, torch.ones_like(X[..., :1])], -1); A = U.transpose(1, 2) @ U + 1e-9 * torch.eye(3)
    return torch.linalg.solve(A, U.transpose(1, 2) @ Y)    # [B,3,2]
def ls_step(W, x):
    B = x.shape[0]; q = x.view(B, P, 2); return (torch.cat([q, torch.ones_like(q[..., :1])], -1) @ W).reshape(B, 12)
KNN_H = float(os.environ.get("CTO_KNN_H", "0.3"))
def knn_step(orb, x, cp, mem):                            # ORACLE_KNN: atención NO aprendida (núcleo gaussiano en x) sobre las MISMAS entradas
    B = x.shape[0]; q = x.view(B, P, 2); u = orb.step(x, cp).view(B, P, 2)
    kx, e = mem["ki"][..., :2], mem["vi"][..., 2:4]
    d = ((q[:, :, None, :] - kx[:, None]) ** 2).sum(-1) / KNN_H ** 2
    d = d.masked_fill(~mem["mask"][:, None, :], 1e9); w = torch.softmax(-d, -1)
    return (u + w @ e).reshape(B, 12)

# ---------------- entrenamiento (sólo familias 0–7, generador propio; nunca H/S/TEST) ----------------
def gen_episodes(rng, n, ks=KS):
    E = []
    for _ in range(n):
        f = int(rng.integers(0, 8)); p = rng.uniform(-1, 1, 2); x0 = make_object(rng, int(rng.integers(0, 3))); k = int(rng.choice(ks))
        c = context_of(f, p)
        if rng.random() < P_CORRUPT: c = context_of(int(rng.integers(0, 8)), rng.uniform(-1, 1, 2))   # contexto erróneo (como la periferia con familias desconocidas)
        else: c[8:] += rng.normal(0, 0.02, 2)
        E.append((f, p, x0, k, c))
    return E
def make_bank(orb, eps, max_k=8):
    """CDT-RQM: banco de episodios consolidados (entradas por slot-transición) con padding a 6*max_k."""
    keys, vals, masks, sigs = [], [], [], []
    for k in sorted({e[3] for e in eps}):
        G = [e for e in eps if e[3] == k]
        obs = torch.tensor(np.stack([traj(e[0], e[1], e[2], k) for e in G])); cp = orb.ctx(torch.tensor(np.stack([e[4] for e in G])))
        with torch.no_grad(): kin, vin, sig, _ = episode_entries(orb, obs, cp)
        ki = full_keys(kin, sig); pad = 6 * max_k - ki.shape[1]
        keys.append(torch.nn.functional.pad(ki, (0, 0, 0, pad))); vals.append(torch.nn.functional.pad(vin, (0, 0, 0, pad)))
        m = torch.zeros(len(G), 6 * max_k, dtype=torch.bool); m[:, :ki.shape[1]] = True; masks.append(m); sigs.append(sig)
    return dict(keys=torch.cat(keys), vals=torch.cat(vals), mask=torch.cat(masks), sig=torch.cat(sigs))
def wpath(ph, s): return f"{DATA}/{ph}/w_{s}.json"
def train(ph, s, arm):
    out = f"{OUT}/{ph}/net_{arm}_{s}.pt"
    if os.path.exists(out): return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    si = int(s, 16); torch.manual_seed(si ^ (1009 * (ARMS_TRAINED.index(arm) + 1))); rng = np.random.default_rng([si, 0xC7, ARMS_TRAINED.index(arm)])
    orb = Orb(wpath(ph, s)); h0 = orb.sha; net = Net(arm); opt = torch.optim.Adam(net.parameters(), lr=LR)
    bank = make_bank(orb, gen_episodes(np.random.default_rng([si, 0xBA]), BANK_TRAIN)) if arm in ("CT", "MLP", "NOORB") else None
    t0 = time.time(); log = open(out + ".log", "w")
    for st in range(TRAIN_STEPS):
        for g in opt.param_groups: g["lr"] = 3e-5 + 0.5 * (LR - 3e-5) * (1 + math.cos(math.pi * st / TRAIN_STEPS))
        k = int(rng.choice(KS)); eps = gen_episodes(rng, BATCH, (k,))
        tr = torch.tensor(np.stack([traj(e[0], e[1], e[2], 16) for e in eps])); cp = orb.ctx(torch.tensor(np.stack([e[4] for e in eps])))
        obs = tr[:, :k + 1]
        with torch.no_grad():
            if arm == "SELF": mem = {}; mem["hk"], mem["hv"], mem["hmask"] = hist_mem(orb, obs, cp)
            else: mem = build_mem(net, orb, obs, cp, bank)
        pr = rollout(net, orb, obs, cp, mem, 16, "net"); tg = tr[:, k + 1:]
        l = (((pr - tg) ** 2).sum(-1) / (tg ** 2).sum(-1).clamp_min(1e-6)); loss = torch.log1p(l).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if (st + 1) % 250 == 0: print(f"step {st+1} loss={float(loss):.4e} {time.time()-t0:.0f}s", file=log, flush=True)
    assert Orb(wpath(ph, s)).sha == h0
    torch.save({"state": net.state_dict(), "arm": arm, "n_params": net.n_params(), "train_s": time.time() - t0, "steps": TRAIN_STEPS, "orb_sha": h0}, out + ".tmp")
    os.replace(out + ".tmp", out); print("trained", ph, s, arm, net.n_params(), f"{time.time()-t0:.0f}s", flush=True)
def load_net(ph, s, arm):
    d = torch.load(f"{OUT}/{ph}/net_{arm}_{s}.pt", weights_only=False); n = Net(arm); n.load_state_dict(d["state"]); n.eval(); return n, d

# ---------------- conjuntos H / S y contexto Gemma (cache de exp/orbita-yo, crece) ----------------
def load_set(ph, nm):
    R = list(csv.DictReader(open(f"{CRE}/{ph}/{nm}_set.csv")))
    return {int(r["idx"]): (int(r["fam"]), np.array([float(r["p1"]), float(r["p2"])]), np.array([float(v) for v in r["x0"].split(";")])) for r in R}
def load_ctx(ph, s, nm):
    return [(int(r["idx"]), r["cond"], np.array([float(r[f"c{i}"]) for i in range(10)])) for r in csv.DictReader(open(f"{CRE}/{ph}/ctx/{nm}_{s}.csv"))]
def taus(): return {int(k): v for k, v in json.load(open(f"{OUT}/tau.json"))["tau"].items()}

HZ = (8, 16, 32, 64, 128, 256)                            # horizontes absolutos evaluados (acc16 primario)
def eval_H(ph, s, nets, bank, tag, rows, orb, Hs, Hc, knn_h=None):
    """Evalúa todos los brazos en H (192 episodios x k) con el banco CDT congelado `bank` (o None)."""
    global KNN_H
    if knn_h is not None: KNN_H = knn_h
    idxs = [i for i, _, _ in Hc]; C = torch.tensor(np.stack([c for _, _, c in Hc])); fam = [Hs[i][0] for i in idxs]
    TR = torch.tensor(np.stack([traj(Hs[i][0], Hs[i][1], Hs[i][2], max(HZ)) for i in idxs]))
    cp = orb.ctx(C); n = len(idxs); g = torch.Generator().manual_seed(int(s, 16) ^ 0x5EED)
    perm = np.random.default_rng(int(s, 16) ^ 0x5F).permutation(n)
    while (perm == np.arange(n)).any(): perm = np.random.default_rng(int(perm[0]) + 7).permutation(n)
    for k in KS:
        obs = TR[:, :k + 1]; res = {}
        with torch.no_grad():
            mem = build_mem(None, orb, obs, cp, bank); mem0 = build_mem(None, orb, obs, cp, None)
            memr = build_mem(None, orb, obs, cp, bank, "rand", g)
            memsh = build_mem(None, orb, obs[perm], cp[perm] if False else {kk: v[perm] for kk, v in cp.items()}, bank)
            hk, hv, hm = hist_mem(orb, obs, cp)
            res["ORB"] = rollout(None, orb, obs, cp, None, max(HZ), "orb")
            res["ORACLE_EP"] = rollout(None, orb, obs, cp, None, max(HZ), "oep")
            res["ORACLE_KNN"] = rollout(None, orb, obs, cp, mem, max(HZ), "knn")
            for arm, net in nets.items():
                if arm == "SELF": res["SELF"] = rollout(net, orb, obs, cp, dict(hk=hk, hv=hv, hmask=hm), max(HZ), "net"); continue
                res[arm] = rollout(net, orb, obs, cp, mem, max(HZ), "net")
                if arm == "CT":
                    res["CT_NOCDT"] = rollout(net, orb, obs, cp, mem0, max(HZ), "net")
                    res["CT_RAND"] = rollout(net, orb, obs, cp, memr, max(HZ), "net")
                    res["CT_SHUF"] = rollout(net, orb, obs, cp, memsh, max(HZ), "net")
                    # sorpresa CT: error del último paso observado prediciendo con memoria de los pasos previos (+CDT); k>=2
                    mp = build_mem(None, orb, obs[:, :-1], cp, bank); xp = net_step(net, orb, obs[:, -2], cp, mp)
                    ct_surp = rel_err(xp.numpy(), obs[:, -1].numpy())
        for j in range(n):
            r = {"tag": tag, "idx": idxs[j], "fam": fam[j], "k": k, "orb_surp": float(mem["surp"][j]), "ct_surp": float(ct_surp[j]) if "CT" in nets else ""}
            for a, pr in res.items():
                for h in HZ:
                    e = rel_err(pr[j, h - k - 1].numpy(), TR[j, h].numpy()); r[f"{a}_{h}"] = int(e < 0.05)
                r[f"{a}_re16"] = float(min(rel_err(pr[j, 16 - k - 1].numpy(), TR[j, 16].numpy()), 1e3))
            rows.append(r)

def stream(ph, s):
    """Flujo S (2400 episodios, k del cache): consolidación CDT-RQM por error de predicción; evaluación de H en los puntos de control."""
    out = f"{OUT}/{ph}/eval_{s}.csv"
    if os.path.exists(out): return
    orb = Orb(wpath(ph, s)); h0 = orb.sha; nets = {a: load_net(ph, s, a)[0] for a in ARMS_TRAINED}
    Hs, Ss = load_set(ph, "H"), load_set(ph, "S"); Hc, Sc = load_ctx(ph, s, "H"), load_ctx(ph, s, "S"); tau = taus()
    knn_h = json.load(open(f"{OUT}/selected.json"))["knn_h"] if os.path.exists(f"{OUT}/selected.json") else KNN_H
    keys, vals, masks, sigs, sps, fams = [], [], [], [], [], []; rows = []; ev = []; t0 = time.time(); tev = 0.0
    def bank():
        if not keys: return None
        return dict(keys=torch.stack(keys), vals=torch.stack(vals), mask=torch.stack(masks), sig=torch.stack(sigs))
    for n in range(CK[-1] + 1):
        if n in CK:
            t1 = time.time(); eval_H(ph, s, nets, bank(), n, rows, orb, Hs, Hc, knn_h); tev += time.time() - t1
            ev.append({"ckpt": n, "bank": len(keys), "new_frac": float(np.mean([f >= 8 for f in fams])) if fams else None})
        if n == CK[-1]: break
        i, cond, c = Sc[n]; f, p, x0 = Ss[i]; k = int(cond[1:])
        obs = torch.tensor(traj(f, p, x0, k))[None]; cp = orb.ctx(torch.tensor(c)[None])
        with torch.no_grad(): kin, vin, sig, surp = episode_entries(orb, obs, cp)
        if float(surp[0]) > tau[k]:                          # consolidación dirigida por el error de predicción
            ki = full_keys(kin, sig)[0]; pad = 48 - ki.shape[0]
            keys.append(torch.nn.functional.pad(ki, (0, 0, 0, pad))); vals.append(torch.nn.functional.pad(vin[0], (0, 0, 0, pad)))
            m = torch.zeros(48, dtype=torch.bool); m[:ki.shape[0]] = True; masks.append(m); sigs.append(sig[0]); sps.append(float(surp[0])); fams.append(f)
            if len(keys) > CDT_CAP:                          # desalojo: la entrada de menor error
                j = int(np.argmin(sps)); [L.pop(j) for L in (keys, vals, masks, sigs, sps, fams)]
    assert Orb(wpath(ph, s)).sha == h0
    cols = list(rows[0].keys()); tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
    os.replace(tmp, out)
    json.dump({"events": ev, "orb_sha_before": h0, "orb_sha_after": Orb(wpath(ph, s)).sha, "stream_eval_s": time.time() - t0, "eval_s": tev,
               "n_params": {a: n.n_params() for a, n in nets.items()}, "orb_params": orb.n_params,
               "train_s": {a: load_net(ph, s, a)[1]["train_s"] for a in ARMS_TRAINED}, "queries_test": 0}, open(f"{OUT}/{ph}/meta_{s}.json", "w"), indent=1)
    print("stream ok", ph, s, f"{time.time()-t0:.0f}s", flush=True)

def calib():
    """τ_k = P95 de la sorpresa ÓRBITA en H DEV, familias conocidas 1–2 (misma regla que crece), 16 seeds."""
    v = {k: [] for k in KS}
    for s in SEEDS["dev"]:
        orb = Orb(wpath("dev", s)); Hs = load_set("dev", "H"); Hc = load_ctx("dev", s, "H")
        idx = [i for i, _, _ in Hc if Hs[i][0] in (1, 2)]; C = torch.tensor(np.stack([c for i, _, c in Hc if Hs[i][0] in (1, 2)]))
        for k in KS:
            obs = torch.tensor(np.stack([traj(*Hs[i], k) for i in idx]))
            with torch.no_grad(): v[k] += episode_entries(orb, obs, orb.ctx(C))[3].tolist()
    tau = {k: float(np.percentile(v[k], 95)) for k in KS}; print({k: (len(v[k]), tau[k]) for k in KS})
    json.dump({"tau": tau, "rule": "P95 sorpresa ÓRBITA (rel_err medio en pasos observados), H DEV fam 1-2, 16 seeds"}, open(f"{OUT}/tau.json", "w"), indent=1)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train": train(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "stream": stream(sys.argv[2], sys.argv[3])
    elif cmd == "calib": calib()
