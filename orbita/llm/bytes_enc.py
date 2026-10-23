"""ÓRBITA-Bytes: encoder sin tokenizador (bytes UTF-8 → parches por entropía (BLT) → VAE robusto por parche (CALM)
→ concepto por frase (LCM) → c (8 familia + 2 params) para `demo-step`). Prerregistro: docs/prereg_orbita_bytes.md.
  bytes_enc.py train PH SEED [VARIANT]   entrena S0 (LM entropía) → S1 (autoencoder) → S2 (concepto+cabezas+destilación+JEPA+trayectoria)
VARIANT ∈ {MAIN, NOVAE, NOHASH, NONCE, FIXPATCH}. Checkpoint del mejor paso de validación en cada etapa."""
import sys, os, json, time, math, copy, zlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as P0
torch.set_num_threads(int(os.environ.get("BYTES_THREADS", "2")))
BD = os.environ.get("BYTES_DATA", f"{P0.DATA}/bytes"); D2D = f"{P0.DATA}/llm_d2"
# ---- hiperparámetros FIJOS (prerregistro) ----
MAXLEN, D, NH, NHASH, HB, NGR = 256, 128, 4, 8192, (3, 4, 5, 6, 7, 8), None
PATCH_TARGET, PATCH_MAX, THETA_R, ZD = 5.0, 16, 1.0, 32
BETA, KL_CLIP, ZDROP, BMASK = 1e-3, 0.25, 0.15, 0.15
LR, WD, WARM, CLIP, BATCH = 4e-4, 0.1, 200, 1.0, 32
MAXS = {"s0": 3000, "s1": 4000, "s2": 3000}; EVERY, PATIENCE, MIN_DELTA = 250, 4, 1e-3
LAM_P, LAM_NCE, LAM_JEPA, LAM_T, TAU, EMA, NX, KS = 10.0, 0.5, 1.0, 1.0, 0.1, 0.99, 8, [1, 2, 4, 8]
if os.environ.get("BYTES_SMOKE"): MAXS, EVERY = {k: int(os.environ["BYTES_SMOKE"]) for k in MAXS}, max(1, int(os.environ["BYTES_SMOKE"]) // 3)   # sólo humo/coste
VAL_I = (14, 15)          # contextos pool{fam}_{14,15} → validación (16 contextos, 192 textos)

def tob(t): return list(t.encode("utf-8"))[:MAXLEN]
def hashes(b):
    """RollPolyHash de n-gramas de bytes que terminan en cada posición, n=3..8 → [L,6] en [0,NHASH)"""
    H = np.zeros((len(b), len(HB)), np.int64)
    for j, n in enumerate(HB):
        for t in range(len(b)):
            if t + 1 < n: H[t, j] = 0; continue
            h = 0
            for x in b[t - n + 1:t + 1]: h = (h * 1000003 + x + 1) % (1 << 61)
            H[t, j] = 1 + h % (NHASH - 1)
    return H
def pad(seqs, v=0):
    L = max(len(s) for s in seqs); X = np.full((len(seqs), L), v, np.int64); M = np.zeros((len(seqs), L), bool)
    for i, s in enumerate(seqs): X[i, :len(s)] = s; M[i, :len(s)] = True
    return torch.from_numpy(X), torch.from_numpy(M)

class Block(nn.Module):
    def __init__(s, d=D, ff=4 * D):
        super().__init__(); s.n1 = nn.LayerNorm(d); s.a = nn.MultiheadAttention(d, NH, batch_first=True); s.n2 = nn.LayerNorm(d); s.f = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
    def forward(s, x, mask=None, kpm=None):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=mask, key_padding_mask=kpm, need_weights=False)[0]; return x + s.f(s.n2(x))

class EntropyLM(nn.Module):           # S0: LM de bytes diminuto (2 capas, d=128, contexto = la línea)
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(257, D); s.p = nn.Embedding(MAXLEN + 1, D); s.b = nn.ModuleList([Block() for _ in range(2)]); s.n = nn.LayerNorm(D); s.o = nn.Linear(D, 256)
    def forward(s, X, M):             # entrada desplazada: BOS(256) + bytes[:-1]
        inp = torch.cat([torch.full_like(X[:, :1], 256), X[:, :-1]], 1); T = X.shape[1]
        h = s.e(inp) + s.p(torch.arange(T))[None]; cm = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for b in s.b: h = b(h, cm)
        return s.o(s.n(h))
def entropies(lm, seqs):
    X, M = pad(seqs)
    with torch.no_grad(): lp = F.log_softmax(lm(X, M), -1); H = -(lp.exp() * lp).sum(-1)
    return [H[i, :len(q)].numpy() for i, q in enumerate(seqs)]
def patch_starts(H, theta_g, fixed=False):
    st = [0]
    for t in range(1, len(H)):
        if t - st[-1] >= PATCH_MAX or (fixed and t - st[-1] >= PATCH_TARGET) or (not fixed and (H[t] > theta_g or H[t] - H[t - 1] > THETA_R)): st.append(t)
    return st
def patch_ids(n, st):
    pid = np.zeros(n, np.int64)
    for k, a in enumerate(st): pid[a:] = k
    return pid

class Batch:                           # tensores de un lote de textos ya parcheados
    def __init__(s, items):
        s.X, s.M = pad([it["b"] for it in items]); L = s.X.shape[1]
        s.Hh = torch.zeros(len(items), L, len(HB), dtype=torch.long)
        for i, it in enumerate(items): s.Hh[i, :len(it["b"])] = torch.from_numpy(it["h"])
        s.pid = torch.zeros(len(items), L, dtype=torch.long); s.pos = torch.zeros(len(items), L, dtype=torch.long)
        s.np_ = torch.tensor([len(it["st"]) for it in items]); P = int(s.np_.max())
        s.PM = torch.arange(P)[None] < s.np_[:, None]
        s.slot = torch.full((len(items), P, PATCH_MAX), -1, dtype=torch.long)
        for i, it in enumerate(items):
            n = len(it["b"]); pid = patch_ids(n, it["st"]); s.pid[i, :n] = torch.from_numpy(pid)
            for k, a in enumerate(it["st"]):
                e = it["st"][k + 1] if k + 1 < len(it["st"]) else n
                s.slot[i, k, :e - a] = torch.arange(a, e); s.pos[i, a:e] = torch.arange(e - a)

class AE(nn.Module):                   # S1: encoder local BLT + VAE CALM por parche + decoder local
    def __init__(s, v):
        super().__init__(); s.v = v
        s.e = nn.Embedding(257, D); s.hash = nn.ModuleList([nn.Embedding(NHASH, D) for _ in HB]) if v != "NOHASH" else None
        s.p = nn.Embedding(MAXLEN + 1, D); s.enc = Block(); s.xq = nn.Linear(D, D); s.xa = nn.MultiheadAttention(D, NH, batch_first=True); s.xn = nn.LayerNorm(D)
        s.vae = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, 2 * ZD))
        s.de = nn.Embedding(257, D); s.dz = nn.Linear(ZD, D); s.dp = nn.Embedding(PATCH_MAX + 1, D); s.dec = nn.ModuleList([Block() for _ in range(2)]); s.dn = nn.LayerNorm(D); s.do = nn.Linear(D, 256)
    def encode(s, B, mask_in=False):
        X = B.X.clone()
        if mask_in: X[(torch.rand(X.shape) < BMASK) & B.M] = 256
        T = X.shape[1]; h = s.e(X) + s.p(torch.arange(T))[None]
        if s.hash is not None: h = h + sum(t(B.Hh[..., j]) for j, t in enumerate(s.hash)) / len(HB)
        i = torch.arange(T); lm = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= PATCH_MAX)   # atención local causal por bloques
        h = s.enc(h, lm)
        Bn, P, S = B.slot.shape; sl = B.slot.clamp_min(0); g = torch.gather(h, 1, sl.view(Bn, P * S, 1).expand(-1, -1, D)).view(Bn, P, S, D)
        sm = B.slot >= 0; q = s.xq(g.masked_fill(~sm[..., None], -1e9).max(2).values.masked_fill(~B.PM[..., None], 0))   # consulta = max-pool del parche
        kv = s.xn(g.view(Bn * P, S, D)); kpm = ~sm.view(Bn * P, S); kpm[:, 0] = False
        o = s.xa(q.view(Bn * P, 1, D), kv, kv, key_padding_mask=kpm, need_weights=False)[0].view(Bn, P, D)
        mu, lv = s.vae(q + o).chunk(2, -1)
        if s.v == "NOVAE": lv = torch.zeros_like(lv)
        return mu, lv.clamp(-10, 5)
    def sample(s, mu, lv, train):
        if s.v == "NOVAE" or not train: return mu
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return F.dropout(z, ZDROP, True)
    def decode_logits(s, B, z):        # cada byte ve sólo el latente de su parche y los bytes previos del mismo parche
        inp = torch.cat([torch.full_like(B.X[:, :1], 256), B.X[:, :-1]], 1); inp = torch.where(B.pos == 0, torch.full_like(inp, 256), inp)
        zb = torch.gather(z, 1, B.pid[..., None].expand(-1, -1, ZD)); h = s.de(inp) + s.dz(zb) + s.dp(B.pos)
        T = inp.shape[1]; same = B.pid[:, :, None] == B.pid[:, None, :]; i = torch.arange(T); m = ~(same & (i[None, :, None] >= i[None, None, :]))
        m = m.repeat_interleave(NH, 0)
        for b in s.dec: h = b(h, m)
        return s.do(s.dn(h))
    def kl(s, mu, lv, PM):
        if s.v == "NOVAE": return torch.zeros(())
        k = 0.5 * (mu ** 2 + lv.exp() - 1 - lv); k = (k * PM[..., None]).sum((0, 1)) / PM.sum()
        return k.clamp_min(KL_CLIP).sum()

class Concept(nn.Module):              # S2: pooler LCM + cabezas + proyección NCE + predictor JEPA
    def __init__(s):
        super().__init__(); s.i = nn.Linear(ZD, D); s.p = nn.Embedding(MAXLEN + 1, D); s.b = nn.ModuleList([Block() for _ in range(2)]); s.q = nn.Parameter(torch.randn(1, 1, D) / D ** 0.5)
        s.pa = nn.MultiheadAttention(D, NH, batch_first=True); s.n = nn.LayerNorm(D); s.f = nn.Linear(D, 8); s.pp = nn.Linear(D, 2)
        s.proj = nn.Linear(D, 128); s.jp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
    def forward(s, mu, PM):
        h = s.i(mu) + s.p(torch.arange(mu.shape[1]))[None]
        for b in s.b: h = b(h, None, ~PM)
        u = s.n(s.pa(s.q.expand(len(mu), -1, -1), h, h, key_padding_mask=~PM, need_weights=False)[0][:, 0]); return u
    def heads(s, u):
        lg = s.f(u); pi = F.softmax(lg, -1); oh = F.one_hot(pi.argmax(-1), 8).to(pi.dtype)
        return torch.cat([oh + pi - pi.detach(), torch.tanh(s.pp(u))], -1), lg

def sched(opt, step):
    for g in opt.param_groups: g["lr"] = LR * min(1.0, (step + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(step, 1e9) / MAXS[g["stage"]]))
def mkopt(params, stage):
    o = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WD); o.param_groups[0]["stage"] = stage; return o
class Stopper:
    """mejor-validación desde el inicio: guarda el estado del mejor eval; para si no mejora > MIN_DELTA (relativo) en PATIENCE evals"""
    def __init__(s, mods): s.mods, s.best, s.state, s.bad, s.hist, s.best_step = mods, float("inf"), None, 0, [], -1
    def __call__(s, step, val, extra=None):
        s.hist.append([step, val] + ([extra] if extra is not None else []))
        if val < s.best - MIN_DELTA * abs(s.best if math.isfinite(s.best) else 1): s.best, s.bad, s.best_step = val, 0, step; s.state = [copy.deepcopy(m.state_dict()) for m in s.mods]
        else: s.bad += 1
        return s.bad >= PATIENCE
    def restore(s):
        for m, st in zip(s.mods, s.state): m.load_state_dict(st)
    def info(s, stopped, last): return {"best_step": s.best_step, "best_val": s.best, "stopped_by_patience": stopped, "last_step": last, "hist": s.hist}

def load_pool():
    rows = P0.tsv(f"{D2D}/pool2.tsv"); return rows, [int(r[0].split("_")[1]) in VAL_I for r in rows]
def prep(texts, lm, theta, fixed):
    bs = [tob(t) for t in texts]; Hs = entropies(lm, bs) if lm is not None else [np.zeros(len(b)) for b in bs]
    return [{"b": b, "h": hashes(b), "st": patch_starts(H, theta, fixed)} for b, H in zip(bs, Hs)]

def teacher_u(ph, seed, rows):
    """destilación: u (128) del encoder D2 (Gemma capa 13) de la misma seed sobre pool2 — calculado una vez, fuera de línea"""
    f = f"{BD}/teacher/{ph}_{seed}.npy"
    if os.path.exists(f): return np.load(f)
    import d2 as D2m; torch.set_default_dtype(torch.float32); torch.set_num_threads(int(os.environ.get("BYTES_THREADS", "2")))
    st = np.load(f"{D2D}/stats.npz"); tok = D2m.Tok(f"{D2D}/pool2.tok"); enc = D2m.D2().double(); enc.load_state_dict(torch.load(f"{D2D}/{ph}/enc_D2_{seed}.pt"))
    U = []
    with torch.no_grad():
        for a in range(0, len(rows), 64):
            X, M = tok.batch(list(range(a, min(a + 64, len(rows)))), st["mu"], st["sd"]); U.append(enc(X.double(), M)[2].numpy())
    U = np.concatenate(U); os.makedirs(os.path.dirname(f), exist_ok=True); np.save(f, U); return U

def train(ph, seed, v="MAIN"):
    t0 = time.time(); out = f"{BD}/models/{ph}/{v}/{seed}"; os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/done.json"): print("ya", out); return
    sd = int(seed, 16) ^ zlib.crc32(v.encode()); torch.manual_seed(sd); rng = np.random.default_rng(sd)
    rows, isval = load_pool(); texts = [r[6] for r in rows]; TR = [i for i in range(len(rows)) if not isval[i]]; VA = [i for i in range(len(rows)) if isval[i]]
    log = {"variant": v, "seed": seed, "phase": ph, "n_train": len(TR), "n_val": len(VA)}
    # S0 LM de entropía
    lm = EntropyLM(); bs = [tob(t) for t in texts]
    if v != "FIXPATCH":
        opt = mkopt(lm.parameters(), "s0"); stp = Stopper([lm]); step = 0; stopped = False
        Xv, Mv = pad([bs[i] for i in VA])
        while step < MAXS["s0"]:
            ids = rng.choice(TR, BATCH); X, M = pad([bs[i] for i in ids]); lm.train()
            loss = (F.cross_entropy(lm(X, M).transpose(1, 2), X, reduction="none") * M).sum() / M.sum()
            sched(opt, step); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(lm.parameters(), CLIP); opt.step(); step += 1
            if step % EVERY == 0 or step == MAXS["s0"]:
                lm.eval()
                with torch.no_grad(): vl = float((F.cross_entropy(lm(Xv, Mv).transpose(1, 2), Xv, reduction="none") * Mv).sum() / Mv.sum())
                if stp(step, vl): stopped = True; break
        stp.restore(); lm.eval(); log["s0"] = stp.info(stopped, step); log["t_s0"] = time.time() - t0
        Htr = entropies(lm, [bs[i] for i in TR]); lo, hi = 0.0, 8.0
        for _ in range(40):          # θg por bisección: tamaño medio de parche = PATCH_TARGET sobre TRAIN
            mid = (lo + hi) / 2; mean = np.mean([len(H) / len(patch_starts(H, mid)) for H in Htr])
            lo, hi = (mid, hi) if mean < PATCH_TARGET else (lo, mid)
        theta = (lo + hi) / 2
    else: theta, lm = None, None
    items = prep(texts, lm, theta, v == "FIXPATCH"); log["theta_g"] = theta; log["mean_patch_train"] = float(np.mean([len(items[i]["b"]) / len(items[i]["st"]) for i in TR]))
    # S1 autoencoder
    ae = AE(v); opt = mkopt(ae.parameters(), "s1"); stp = Stopper([ae]); step = 0; stopped = False; Bv = Batch([items[i] for i in VA])
    while step < MAXS["s1"]:
        B = Batch([items[i] for i in rng.choice(TR, BATCH)]); ae.train(); mu, lv = ae.encode(B, True); z = ae.sample(mu, lv, True)
        rec = (F.cross_entropy(ae.decode_logits(B, z).transpose(1, 2), B.X, reduction="none") * B.M).sum() / B.M.sum(); loss = rec + BETA * ae.kl(mu, lv, B.PM)
        sched(opt, step); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ae.parameters(), CLIP); opt.step(); step += 1
        if step % EVERY == 0 or step == MAXS["s1"]:
            ae.eval()
            with torch.no_grad():
                mu, lv = ae.encode(Bv); lg = ae.decode_logits(Bv, mu); vl = float((F.cross_entropy(lg.transpose(1, 2), Bv.X, reduction="none") * Bv.M).sum() / Bv.M.sum())
                acc = float(((lg.argmax(-1) == Bv.X) & Bv.M).sum() / Bv.M.sum())
            if stp(step, vl, acc): stopped = True; break
    stp.restore(); ae.eval(); log["s1"] = stp.info(stopped, step); log["t_s1"] = time.time() - t0
    for p in ae.parameters(): p.requires_grad_(False)
    # S2 concepto + cabezas + NCE (teacher Gemma/D2) + JEPA (EMA, stop-grad) + trayectoria ÓRBITA congelada
    import d1 as D1; torch.set_default_dtype(torch.float32); torch.set_num_threads(int(os.environ.get("BYTES_THREADS", "2")))
    m = D1.load_orbita(f"{P0.DATA}/{ph}/w_{seed}.json").double(); meta = json.load(open(f"{P0.DATA}/{ph}/{seed}/meta.json")); zd, cd = meta["zdim"], meta["cdim"]
    XT = torch.from_numpy(np.fromfile(f"{P0.DATA}/{ph}/{seed}/train.bin", dtype="<f8").reshape(-1, zd + cd + zd)[:, :zd].copy())
    U = torch.from_numpy(teacher_u(ph, seed, rows)).float(); FAM = torch.tensor([int(r[1]) for r in rows]); PT = torch.tensor([[float(r[2]), float(r[3])] for r in rows]).float()
    Ctrue = torch.tensor(np.array([P0.ctx_true(int(r[1]), float(r[2]), float(r[3])) for r in rows]))
    byc = {}
    for i in TR: byc.setdefault(rows[i][0], []).append(i)
    cids = sorted(byc); cm = Concept(); tgt = copy.deepcopy(cm)
    for p in tgt.parameters(): p.requires_grad_(False)
    with torch.no_grad(): MU = [None] * len(rows)
    def enc_mu(ids):
        B = Batch([items[i] for i in ids]); mu, lv = ae.encode(B); return mu, lv, B.PM
    opt = mkopt(cm.parameters(), "s2"); stp = Stopper([cm]); step = 0; stopped = False; Gn = torch.Generator().manual_seed(sd)
    with torch.no_grad(): muv, _, pmv = enc_mu(VA)
    while step < MAXS["s2"]:
        cb = torch.randint(0, len(cids), (BATCH,), generator=Gn).tolist(); a, b = [], []
        for j in cb:
            l = byc[cids[j]]; pr = torch.randperm(len(l), generator=Gn)[:2].tolist(); a.append(l[pr[0]]); b.append(l[pr[1]])
        with torch.no_grad(): mua, lva, pma = enc_mu(a); mub, _, pmb = enc_mu(b)
        za = ae.sample(mua, lva, True) if v != "NOVAE" else mua
        cm.train(); u = cm(za, pma); ch, lg = cm.heads(u); A = torch.tensor(a)
        lce = F.cross_entropy(lg, FAM[A]); lp = ((ch[:, 8:] - PT[A]) ** 2).sum(1).mean()
        lnce = F.cross_entropy(F.normalize(cm.proj(u), dim=-1) @ F.normalize(U[A], dim=-1).T / TAU, torch.arange(BATCH)) if v != "NONCE" else torch.zeros(())
        with torch.no_grad(): ub = tgt(mub, pmb)
        lj = (1 - F.cosine_similarity(cm.jp(u), ub, dim=-1)).mean()
        xi = XT[torch.randint(0, len(XT), (BATCH * NX,), generator=Gn)]; chx = ch.double().repeat_interleave(NX, 0); cx = Ctrue[A].repeat_interleave(NX, 0); lt, zh, zt, prev = 0.0, xi, xi, 0
        for k in KS:
            zh = D1.rollout(m, zh, chx, k - prev)
            with torch.no_grad(): zt = D1.rollout(m, zt, cx, k - prev)
            prev = k; lt = lt + 0.25 * (((zh - zt) ** 2).sum(1) / (zt ** 2).sum(1).clamp_min(1e-9)).clamp(max=10.0).mean()
        loss = lce + LAM_P * lp + LAM_NCE * lnce + LAM_JEPA * lj + LAM_T * lt.float()
        sched(opt, step); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(cm.parameters(), CLIP); opt.step(); step += 1
        with torch.no_grad():
            for pt, po in zip(tgt.parameters(), cm.parameters()): pt.mul_(EMA).add_(po, alpha=1 - EMA)
        if step % EVERY == 0 or step == MAXS["s2"]:
            cm.eval()
            with torch.no_grad():
                chv, lgv = cm.heads(cm(muv, pmv)); Av = torch.tensor(VA)
                vl = float(F.cross_entropy(lgv, FAM[Av]) + LAM_P * ((chv[:, 8:] - PT[Av]) ** 2).sum(1).mean()); acc = float((lgv.argmax(-1) == FAM[Av]).float().mean())
            if stp(step, vl, acc): stopped = True; break
    stp.restore(); cm.eval(); log["s2"] = stp.info(stopped, step)
    with torch.no_grad():
        mu, _, pm = enc_mu(TR); Ctr = cm(mu, pm).numpy()
    med = np.median(Ctr, 0); iqr = np.percentile(Ctr, 75, 0) - np.percentile(Ctr, 25, 0) + 1e-6   # robust scaler (LCM)
    torch.save({"lm": lm.state_dict() if lm is not None else None, "theta": theta, "ae": ae.state_dict(), "cm": cm.state_dict(), "med": med, "iqr": iqr, "variant": v}, f"{out}/model.pt")
    log["wall_s"] = time.time() - t0; log["n_params"] = {k: sum(p.numel() for p in mod.parameters()) for k, mod in (("lm", lm), ("ae", ae), ("concept", cm)) if mod is not None}
    json.dump(log, open(f"{out}/done.json", "w")); print(ph, v, seed, "wall_s", round(log["wall_s"], 1), "best", log["s0"]["best_step"] if "s0" in log else "-", log["s1"]["best_step"], log["s2"]["best_step"], flush=True)

class Encoder:
    """inferencia: textos → (c̃ 10 [onehot+p head], logits familia, concepto robust-escalado, n_parches)"""
    def __init__(s, path):
        torch.set_default_dtype(torch.float32); ck = torch.load(path, weights_only=False); s.v = ck["variant"]; s.theta = ck["theta"]
        s.lm = None
        if ck["lm"] is not None: s.lm = EntropyLM(); s.lm.load_state_dict(ck["lm"]); s.lm.eval()
        s.ae = AE(s.v); s.ae.load_state_dict(ck["ae"]); s.ae.eval(); s.cm = Concept(); s.cm.load_state_dict(ck["cm"]); s.cm.eval(); s.med, s.iqr = ck["med"], ck["iqr"]
    def __call__(s, texts, bs=64):
        C, LG, U, NP = [], [], [], []
        with torch.no_grad():
            for a in range(0, len(texts), bs):
                it = prep(texts[a:a + bs], s.lm, s.theta, s.v == "FIXPATCH"); B = Batch(it); mu, _ = s.ae.encode(B); u = s.cm(mu, B.PM); c, lg = s.cm.heads(u)
                C.append(c.numpy()); LG.append(lg.numpy()); U.append((u.numpy() - s.med) / s.iqr); NP += [len(x["st"]) for x in it]
        return np.concatenate(C), np.concatenate(LG), np.concatenate(U), NP
    def recon(s, texts):
        """fidelidad (H7): exactitud de bytes con decodificación voraz parche a parche desde μ (longitudes de parche conocidas)"""
        it = prep(texts, s.lm, s.theta, s.v == "FIXPATCH"); B = Batch(it); ok = tot = ex = 0
        with torch.no_grad():
            mu, _ = s.ae.encode(B); X = B.X.clone(); X[B.M] = 0; L = int(B.pos.max()) + 1
            for t in range(L):       # posición dentro del parche t: todos los parches en paralelo
                B2 = copy.copy(B); B2.X = X; lg = s.ae.decode_logits(B2, mu); sel = (B.pos == t) & B.M; X[sel] = lg.argmax(-1)[sel]
        for i in range(len(texts)):
            n = int(B.M[i].sum()); e = (X[i, :n] == B.X[i, :n]); ok += int(e.sum()); tot += n; ex += int(e.all())
        return ok / tot, ex / len(texts)

if __name__ == "__main__":
    if sys.argv[1] == "train": train(*sys.argv[2:])
