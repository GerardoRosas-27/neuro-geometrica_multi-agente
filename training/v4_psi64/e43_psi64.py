"""§18 — E43/E44 on a NEW frozen Ψ64 encoder trained on Gemma hidden (E45 recipe).

Usage: python e43_psi64.py run <smoke|dev|confirm>
       python e43_psi64.py reeval <phase> <seed>     (fresh-process check)
Pre-registration: docs/preregistro_v4.md §18.
"""
import hashlib, json, os, subprocess, sys, time
import numpy as np, jax, jax.numpy as jnp, optax

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
A = lambda p: os.path.join(ROOT, p)
TASK = json.load(open(A("artifacts/v4/psi64/lang_task.json")))
LANGS, PAIRS, ANCHORS = TASK["langs"], TASK["pairs"], TASK["anchors"]
SEEDS = {"dev": list(range(0xA400, 0xA410)), "confirm": list(range(0xB400, 0xB410)), "smoke": [0x5A00]}
# pre-registered constants (§18)
N_PSI, ENC_H, ENC_STEPS, ENC_LR = 64, 256, 1500, 2e-3
D_H, D_STEPS, D_BATCH, LR0, LR1, WD, TAU, REPLAY = 136, 24000, 8, 3e-3, 3e-5, 1e-4, 0.01, 0.5


def sha_arrays(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode()); h.update(np.ascontiguousarray(np.asarray(d[k], np.float64)).tobytes())
    return h.hexdigest()


def load_emb():
    g = json.load(open(A("artifacts/v4/lang/emb_gemma.json")))
    q = json.load(open(A("artifacts/v4/lang/emb_qwen.json")))
    r = json.load(open(A("artifacts/v4/psi64/emb_random512.json")))
    f = lambda e: {k: np.asarray(v, np.float64) for k, v in e.items()}
    return f(g), f(q), f(r)


def split(seed):
    tc = set(TASK["train_concepts"][f"0x{seed:X}"])
    train = [p for p in PAIRS if p["lang"] == 0 and p["concept"] in tc]
    novel = [p for p in PAIRS if p["concept"] not in tc]
    return tc, train, novel


# ───────── Ψ64 encoder, E45 recipe (gelu MLP H→256→64, decoder CE + linear transition A_r) ─────────
def gelu(x):
    return jax.nn.gelu(x, approximate=True)


def enc_apply(p, x):
    return gelu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b"]


def train_encoder(emb, train, seed):
    words = sorted({w for p in train for w in (p["src"], p["dst"])})
    X = np.stack([emb[w] for w in words])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    concepts = sorted({p["concept"] for p in train})
    ci = {c: i for i, c in enumerate(concepts)}
    wi = {w: i for i, w in enumerate(words)}
    src = np.array([wi[p["src"]] for p in train]); dst = np.array([wi[p["dst"]] for p in train])
    rel = np.array([p["rel"] for p in train]); cc = np.array([ci[p["concept"]] for p in train])
    Xs = jnp.asarray((X - mu) / sd)
    H = X.shape[1]
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    p = {"W1": jax.random.normal(k[0], (H, ENC_H)) / np.sqrt(H), "b1": jnp.zeros(ENC_H),
         "W2": jax.random.normal(k[1], (ENC_H, N_PSI)) / 16.0, "b": jnp.zeros(N_PSI),
         "Dc": jax.random.normal(k[2], (N_PSI, len(concepts))) / np.sqrt(N_PSI), "Dcb": jnp.zeros(len(concepts)),
         "Df": jax.random.normal(k[3], (N_PSI, 2)) / np.sqrt(N_PSI), "Dfb": jnp.zeros(2),
         "A": jnp.tile(jnp.eye(N_PSI), (2, 1, 1)) + 0.01 * jax.random.normal(k[4], (2, N_PSI, N_PSI))}

    def ce(lg, y):
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], -1))

    def dec(p, s):
        return s @ p["Dc"] + p["Dcb"], s @ p["Df"] + p["Dfb"]

    def loss(p):
        psi = enc_apply(p, Xs)
        ps, pd = psi[src], psi[dst]
        pred = jnp.einsum("bn,bnm->bm", ps, p["A"][rel])
        l = 0.0
        for s, form in ((ps, 0), (pd, 1), (pred, 1)):
            lc, lf = dec(p, s)
            l = l + ce(lc, cc) + ce(lf, jnp.full(len(cc), form))
        return l + 0.1 * jnp.mean((pred - jax.lax.stop_gradient(pd)) ** 2)

    opt = optax.adam(ENC_LR); st = opt.init(p)

    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    l0 = float(loss(p))
    for _ in range(ENC_STEPS):
        p, st, l = step(p, st)
    psi = enc_apply(p, Xs)
    lc, _ = dec(p, psi[src]); pred = jnp.einsum("bn,bnm->bm", psi[src], p["A"][rel])
    lcp, lfp = dec(p, pred)
    stats = {"n_words": len(words), "n_pairs": len(train), "n_concepts": len(concepts), "loss0": l0, "loss_final": float(l),
             "train_concept_acc": float(jnp.mean(jnp.argmax(lc, -1) == cc)),
             "train_transition_form_acc": float(jnp.mean(jnp.argmax(lfp, -1) == 1))}
    frozen = {"W1": np.asarray(p["W1"]), "b1": np.asarray(p["b1"]), "W2": np.asarray(p["W2"]), "b": np.asarray(p["b"]), "mu": mu, "sd": sd}
    return frozen, words, stats


def psi_of(fr, x):
    return np.asarray(enc_apply({k: jnp.asarray(v) for k, v in fr.items()}, jnp.asarray((x - fr["mu"]) / fr["sd"])))


# ───────── Dφ (§16 E3 adapted): gated diagonal local-linear head + trusted replay ─────────
def init_d(key, kind, d):
    out = 2 * d if kind in ("gated", "mlp_gated") else d
    k1, k2 = jax.random.split(key)
    return {"W1": jax.random.normal(k1, (d + 2, D_H)) / np.sqrt(d + 2), "b1": jnp.zeros(D_H),
            "W2": jax.random.normal(k2, (D_H, out)) * 0.1 / np.sqrt(D_H), "b2": jnp.zeros(out)}


def d_apply(kind, p, z, r):
    c = jax.nn.one_hot(r, 2)
    o = jnp.tanh(jnp.concatenate([z, c], -1) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    d = z.shape[-1]
    if kind == "plain":
        return z + o
    m, b = o[..., :d], o[..., d:]
    return (z if kind == "gated" else 0.0) + m * z + b


def ridge(X, Y, lam):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)


def teacher(Zs, R, Zd):
    """Consolidated per-relation linear map + leave-one-out relative residual."""
    W, loo = {}, {}
    for r in (0, 1):
        m = R == r
        X, Y = Zs[m], Zd[m]
        lam = 1e-3 * len(X)
        W[r] = ridge(X, Y, lam)
        errs = []
        for i in range(len(X)):
            keep = np.arange(len(X)) != i
            w = ridge(X[keep], Y[keep], lam)
            errs.append(np.linalg.norm(np.append(X[i], 1) @ w - Y[i]) / max(np.linalg.norm(Y[i]), 1e-9))
        loo[r] = float(np.mean(errs))
    return W, loo


def train_d(kind, Zs, R, Zd, seed, trust):
    rng = np.random.default_rng(seed)
    W, loo = trust
    trusted = [r for r in (0, 1) if loo[r] < TAU]
    allz = np.vstack([Zs, Zd]); mu, sd = allz.mean(0), allz.std(0)
    n, d = len(Zs), Zs.shape[1]
    T = D_STEPS * D_BATCH
    idx = rng.integers(0, n, T)
    xs, rs, ys = Zs[idx].copy(), R[idx].copy(), Zd[idx].copy()
    if trusted:
        rep = rng.random(T) < REPLAY
        rr = rng.choice(trusted, T)
        xr = mu + sd * rng.uniform(-1.7, 1.7, (T, d))
        yr = np.stack([np.append(x, 1) @ W[r] for x, r in zip(xr[rep], rr[rep])]) if rep.any() else None
        xs[rep], rs[rep] = xr[rep], rr[rep]
        if yr is not None:
            ys[rep] = yr
    xs, rs, ys = (jnp.asarray(a.reshape(D_STEPS, D_BATCH, *a.shape[1:])) for a in (xs, rs, ys))
    p = init_d(jax.random.PRNGKey(seed), kind, d)
    sched = optax.cosine_decay_schedule(LR0, D_STEPS, alpha=LR1 / LR0)
    opt = optax.adam(sched); st = opt.init(p)

    def loss(p, x, r, y):
        yh = d_apply(kind, p, x, r)
        rel = jnp.sum((yh - y) ** 2, -1) / jnp.maximum(jnp.sum(y ** 2, -1), 1e-6)
        return jnp.mean(rel) + 0.5 * WD * sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(p))

    def body(c, b):
        p, st = c
        g = jax.grad(loss)(p, *b)
        u, st = opt.update(g, st, p)
        return (optax.apply_updates(p, u), st), None

    (p, _), _ = jax.lax.scan(body, (p, st), (xs, rs, ys))
    return {k: np.asarray(v) for k, v in p.items()}, trusted


# ───────── evaluation ─────────
def nearest(z, cands):
    W, V = cands
    s = V @ z / (np.linalg.norm(V, axis=1) * max(np.linalg.norm(z), 1e-12) + 1e-12)
    return W[int(np.argmax(s))]


def evaluate(enc, novel, methods):
    """enc: word -> z. methods: name -> fn(z, rel) -> z'. Returns {method: [acc per lang]}, pooled."""
    out = {m: [] for m in methods}
    pooled = {m: [0, 0] for m in methods}
    for l in range(4):
        ws = sorted({w for p in PAIRS if p["lang"] == l for w in (p["src"], p["dst"])})
        cands = (ws, np.stack([enc(w) for w in ws]))
        lp = [p for p in novel if p["lang"] == l]
        for m, f in methods.items():
            ok = sum(nearest(f(enc(p["src"]), p["rel"]), cands) == p["dst"] for p in lp)
            out[m].append(ok / max(len(lp), 1))
            pooled[m][0] += ok; pooled[m][1] += len(lp)
    return out, {m: a / max(b, 1) for m, (a, b) in pooled.items()}


def anchor_map(src_emb, tgt_z):
    X = np.stack([src_emb[w] for w in ANCHORS]); mu = X.mean(0); Xc = X - mu
    Y = np.stack([tgt_z(w) for w in ANCHORS])
    K = Xc @ Xc.T; lam = 1e-3 * np.trace(K) / len(K)
    alpha = np.linalg.solve(K + lam * np.eye(len(K)), Y)
    return lambda x: ((x - mu) @ Xc.T) @ alpha


def fapply(kind, p):
    pj = {k: jnp.asarray(v) for k, v in p.items()}
    return lambda z, r: np.asarray(d_apply(kind, pj, jnp.asarray(z), r))


def build_eval(seed, g, q, rnd, ck):
    """Everything downstream of the frozen artefacts (shared by run and reeval)."""
    _, train, novel = split(seed)
    zg = lambda w: psi_of(ck["enc_g"], g[w]) / ck["scale_g"]
    zr = lambda w: psi_of(ck["enc_r"], rnd[w]) / ck["scale_r"]
    mg = {"E3_dphi_gated_trusted": fapply("gated", ck["d_g_gated"]), "M_mlp_gated_trusted": fapply("mlp_gated", ck["d_g_mlp"]),
          "P_dphi_plain_trusted": fapply("plain", ck["d_g_plain"]),
          "offset": lambda z, r: z + ck["off_g"][r], "static": lambda z, r: z}
    mr = {"E3_dphi_gated_trusted": fapply("gated", ck["d_r_gated"])}
    accg, poolg = evaluate(zg, novel, mg)
    accr, poolr = evaluate(zr, novel, mr)
    e3 = mg["E3_dphi_gated_trusted"]
    mq = anchor_map(q, zg); mrr = anchor_map(rnd, zg); mga = anchor_map(g, zg)
    a44 = {}
    for tag, emb, mp in (("qwen_aligned", q, mq), ("random_aligned", rnd, mrr), ("gemma_aligned", g, mga)):
        a44[tag] = evaluate(lambda w, e=emb, f=mp: f(e[w]), novel, {"E3": e3})
    return accg, poolg, accr, poolr, a44


def ckpt_path(phase, seed):
    return A(f"artifacts/v4/psi64/{phase}/ckpt/seed_0x{seed:X}.npz")


def save_ck(path, ck):
    flat = {}
    for k, v in ck.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}::{kk}"] = np.asarray(vv)
        else:
            flat[k] = np.asarray(v)
    np.savez(path, **flat)


def load_ck(path):
    d = np.load(path); ck = {}
    for k in d.files:
        if "::" in k:
            a, b = k.split("::")
            ck.setdefault(a, {})[int(b) if b.isdigit() else b] = d[k]
        else:
            ck[k] = d[k]
    ck["scale_g"] = float(ck["scale_g"]); ck["scale_r"] = float(ck["scale_r"])
    return ck


def run_seed(seed, phase, g, q, rnd):
    tc, train, novel = split(seed)
    novel_words = {w for p in novel for w in (p["src"], p["dst"])}
    enc_g, words, st_g = train_encoder(g, train, seed)
    enc_r, _, st_r = train_encoder(rnd, train, seed)
    leakage = len(novel_words & set(words))
    ck = {"enc_g": enc_g, "enc_r": enc_r}
    rows, man = [], {"seed": f"0x{seed:X}", "train_words": words, "leakage_novel_words_in_training": leakage,
                     "enc_gemma_sha256": sha_arrays(enc_g), "enc_random_sha256": sha_arrays(enc_r), "stats_gemma": st_g, "stats_random": st_r}
    for tag, emb, enc in (("g", g, enc_g), ("r", rnd, enc_r)):
        P = lambda w: psi_of(enc, emb[w])
        tr_psi = np.stack([P(w) for w in words])
        scale = float(np.sqrt(np.mean(tr_psi ** 2)))
        ck[f"scale_{tag}"] = scale
        Zs = np.stack([P(p["src"]) / scale for p in train]); Zd = np.stack([P(p["dst"]) / scale for p in train])
        R = np.array([p["rel"] for p in train])
        trust = teacher(Zs, R, Zd)
        man[f"teacher_loo_{tag}"] = trust[1]
        kinds = ("gated", "mlp", "plain") if tag == "g" else ("gated",)
        for kd in kinds:
            kind = {"gated": "gated", "mlp": "mlp_gated", "plain": "plain"}[kd]
            p, trusted = train_d(kind, Zs, R, Zd, seed ^ 0xD0, trust)
            ck[f"d_{tag}_{kd}"] = p
            man[f"trusted_{tag}"] = trusted
        if tag == "g":
            ck["off_g"] = {r: (Zd[R == r] - Zs[R == r]).mean(0) for r in (0, 1)}
    os.makedirs(os.path.dirname(ckpt_path(phase, seed)), exist_ok=True)
    save_ck(ckpt_path(phase, seed), ck)
    ck = load_ck(ckpt_path(phase, seed))
    accg, poolg, accr, poolr, a44 = build_eval(seed, g, q, rnd, ck)
    for m, v in accg.items():
        for l in range(4):
            rows.append(f"0x{seed:X},E43,gemma_psi64,{LANGS[l]},novel,{m},{v[l]:.4f}")
    for l in range(4):
        rows.append(f"0x{seed:X},E43,random_psi64,{LANGS[l]},novel,E3_dphi_gated_trusted,{accr['E3_dphi_gated_trusted'][l]:.4f}")
    for tag, (acc, pool) in a44.items():
        for l in range(4):
            rows.append(f"0x{seed:X},E44,{tag},{LANGS[l]},novel,E3,{acc['E3'][l]:.4f}")
    e3 = accg["E3_dphi_gated_trusted"]; st = accg["static"]; rr = accr["E3_dphi_gated_trusted"]
    wins = sum(e3[l] >= max(st[l], rr[l]) + 0.05 for l in (1, 2, 3))
    pg = poolg["E3_dphi_gated_trusted"]; pq = a44["qwen_aligned"][1]["E3"]; pr = a44["random_aligned"][1]["E3"]
    e44 = pq >= pr + 0.05 and pq >= 0.5 * pg
    # fresh process
    out = subprocess.run([sys.executable, __file__, "reeval", phase, str(seed)], capture_output=True, text=True).stdout.strip().splitlines()
    fr = json.loads(out[-1]) if out else {}
    fresh_ok = fr.get("pooled_gemma_E3") == pg and fr.get("pooled_qwen_E3") == pq and fr.get("pid") != os.getpid()
    clean = leakage == 0
    man.update({"e43_pass": bool(wins >= 2 and clean and fresh_ok), "e44_pass": bool(e44 and clean and fresh_ok), "wins": int(wins),
                "pooled": {"gemma_E3": pg, "qwen_aligned": pq, "random_aligned": pr, "gemma_aligned": a44["gemma_aligned"][1]["E3"]},
                "fresh_process": {"ok": bool(fresh_ok), "pid": fr.get("pid")}, "queries": 0})
    return rows, man


def main():
    if sys.argv[1] == "reeval":
        phase, seed = sys.argv[2], int(sys.argv[3])
        g, q, rnd = load_emb()
        ck = load_ck(ckpt_path(phase, seed))
        accg, poolg, accr, poolr, a44 = build_eval(seed, g, q, rnd, ck)
        print(json.dumps({"pooled_gemma_E3": poolg["E3_dphi_gated_trusted"], "pooled_qwen_E3": a44["qwen_aligned"][1]["E3"], "pid": os.getpid()}))
        return
    phase = sys.argv[2]
    g, q, rnd = load_emb()
    outd = A(f"artifacts/v4/psi64/{phase}"); os.makedirs(outd, exist_ok=True)
    t0 = time.time(); rows, mans = [], []
    for s in SEEDS[phase]:
        r, mn = run_seed(s, phase, g, q, rnd)
        rows += r; mans.append(mn)
        print(f"seed 0x{s:X} done {time.time()-t0:.0f}s e43={mn['e43_pass']} e44={mn['e44_pass']} pooled={mn['pooled']}", flush=True)
    open(f"{outd}/metrics.csv", "w").write("seed,exp,encoder,lang,subset,method,acc\n" + "\n".join(rows) + "\n")
    with open(f"{outd}/encoder_manifest.jsonl", "w") as f:
        for mn in mans:
            f.write(json.dumps(mn) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    sha = lambda p: hashlib.sha256(open(A(p), "rb").read()).hexdigest()
    json.dump({"git_sha": git, "git_dirty": dirty, "phase": phase, "seeds": [f"0x{s:X}" for s in SEEDS[phase]], "wall_s": time.time() - t0,
               "emb_gemma_sha256": sha("artifacts/v4/lang/emb_gemma.json"), "emb_qwen_sha256": sha("artifacts/v4/lang/emb_qwen.json"),
               "emb_random_sha256": sha("artifacts/v4/psi64/emb_random512.json"), "task_sha256": sha("artifacts/v4/psi64/lang_task.json"),
               "constants": {"N_PSI": N_PSI, "ENC_H": ENC_H, "ENC_STEPS": ENC_STEPS, "ENC_LR": ENC_LR, "D_H": D_H, "D_STEPS": D_STEPS,
                             "D_BATCH": D_BATCH, "LR0": LR0, "LR1": LR1, "WD": WD, "TAU": TAU, "REPLAY": REPLAY}}, open(f"{outd}/config.json", "w"), indent=1)
    p43 = sum(m["e43_pass"] for m in mans); p44 = sum(m["e44_pass"] for m in mans)
    print(f"E43 PASS {p43}/{len(mans)}  E44 PASS {p44}/{len(mans)}")


if __name__ == "__main__":
    main()
