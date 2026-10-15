"""Fase 0 LLM→ÓRBITA (prerregistrado en docs/preregistro_orbita_llm.md).
  pipeline.py texts PHASE        → ctx-dump por seed + textos de evaluación (plantillas 2-3)
  pipeline.py fitP               → ridge (GEMMA y NGRAM) sobre el pool (plantillas 0-1), alpha por CV 5-fold
  pipeline.py ctx PHASE          → CSV de contextos por seed: TRUE, GEMMA, GEMMA_SWAP, NGRAM, RAND
  pipeline.py eval PHASE         → orbita llm-eval por seed
  pipeline.py analyze PHASE      → tablas + IC95 (bootstrap pareado por seed, 10000, semilla 12345)"""
import sys, os, csv, json, subprocess, zlib
import numpy as np
D = "/workspace/orbita-data/llm"; R = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(R))
B = f"{ROOT}/orbita/target/release/orbita"; DIM = 2304
SEEDS = {"dev": [f"0xA4{i:02X}" for i in range(16)], "confirm": [f"0xB4{i:02X}" for i in range(16)]}
ART = f"{ROOT}/artifacts/orbita/llm"
def tsv(p): return [l.rstrip("\n").split("\t") for l in open(p)]
def emb(p, n): x = np.fromfile(p, dtype="<f4"); assert x.size == n * DIM, (p, x.size, n); return x.reshape(n, DIM).astype(np.float64)
def ngram(texts):
    X = np.zeros((len(texts), DIM))
    for i, t in enumerate(texts):
        t = f" {t.lower()} "
        for n in (1, 2, 3):
            for j in range(len(t) - n + 1): X[i, zlib.crc32(t[j:j+n].encode()) % DIM] += 1
    return X / np.linalg.norm(X, axis=1, keepdims=True)
def ctx_true(fam, p1, p2):
    c = np.zeros(10); c[fam] = 1; c[8] = p1; c[9] = p2; return c
ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1000]
def ridge_fit(X, Y, a):
    mu, sd = X.mean(0), X.std(0) + 1e-8; Z = (X - mu) / sd; ym = Y.mean(0)
    K = Z @ Z.T; A = np.linalg.solve(K + a * np.eye(len(Z)), Y - ym)
    return {"mu": mu, "sd": sd, "W": Z.T @ A, "ym": ym}
def ridge_pred(P, X): return ((X - P["mu"]) / P["sd"]) @ P["W"] + P["ym"]
def fitP():
    rows = tsv(f"{D}/pool.tsv"); n = len(rows)
    Y = np.array([ctx_true(int(r[1]), float(r[2]), float(r[3])) for r in rows])
    groups = np.array([hash_ctx(r[0]) for r in rows]) % 5      # folds por contexto (no por texto)
    out = {}
    for name, X in (("GEMMA", emb(f"{D}/pool.emb", n)), ("NGRAM", ngram([r[6] for r in rows]))):
        cv = {}
        for a in ALPHAS:
            err = []
            for k in range(5):
                tr, te = groups != k, groups == k
                err.append(((ridge_pred(ridge_fit(X[tr], Y[tr], a), X[te]) - Y[te]) ** 2).mean())
            cv[a] = float(np.mean(err))
        a = min(cv, key=cv.get); P = ridge_fit(X, Y, a)
        np.savez(f"{D}/P_{name}.npz", **P); out[name] = {"alpha": a, "cv_mse": cv}
    os.makedirs(ART, exist_ok=True); json.dump(out, open(f"{ART}/fitP.json", "w"), indent=1); print(json.dumps(out))
def hash_ctx(s): return zlib.crc32(s.encode())
def texts(ph):
    allr = []
    for s in SEEDS[ph]:
        cp = f"{D}/{ph}/ctx_{s}.csv"; os.makedirs(os.path.dirname(cp), exist_ok=True)
        open(cp, "w").write(subprocess.check_output([B, "ctx-dump", "--seed", s]).decode())
        subprocess.check_call(["python3", f"{R}/make_texts.py", "test", cp, f"{D}/{ph}/texts_{s}.tsv"], stdout=subprocess.DEVNULL)
        allr += [[s] + r for r in tsv(f"{D}/{ph}/texts_{s}.tsv")]
    with open(f"{D}/{ph}/all.tsv", "w") as f:
        for r in allr: f.write("\t".join(r) + "\n")
    with open(f"{D}/{ph}/all.txt", "w") as f:
        for r in allr: f.write(r[7] + "\n")
    print(len(allr), "textos", ph)
def ctx(ph):
    rows = tsv(f"{D}/{ph}/all.tsv"); n = len(rows)
    PG = dict(np.load(f"{D}/P_GEMMA.npz")); PN = dict(np.load(f"{D}/P_NGRAM.npz"))
    CG = ridge_pred(PG, emb(f"{D}/{ph}/all.emb", n)); CN = ridge_pred(PN, ngram([r[7] for r in rows]))
    rng = np.random.default_rng(0xBAD)
    hdr = "idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))
    for s in SEEDS[ph]:
        I = [i for i, r in enumerate(rows) if r[0] == s]
        ex = {}
        for i in I: ex.setdefault(rows[i][1], []).append(i)
        keys = list(ex)
        L = [hdr]
        fmt = lambda v: ",".join(repr(float(x)) for x in v)
        for k in keys:
            r0 = rows[ex[k][0]]; L.append(f"{k},TRUE,0," + fmt(ctx_true(int(r0[2]), float(r0[3]), float(r0[4]))))
            same = [j for j in keys if rows[ex[j][0]][2] == r0[2]]; nxt = same[(same.index(k) + 1) % len(same)]
            for t, i in enumerate(ex[k]):
                tid = f"{rows[i][5]}-{rows[i][6]}"
                L.append(f"{k},GEMMA,{tid}," + fmt(CG[i])); L.append(f"{k},NGRAM,{tid}," + fmt(CN[i]))
                c = np.zeros(10); c[rng.integers(8)] = 1; c[8:] = rng.uniform(-1, 1, 2); L.append(f"{k},RAND,{tid}," + fmt(c))
                L.append(f"{k},GEMMA_SWAP,{tid}," + fmt(CG[ex[nxt][t]]))
        open(f"{D}/{ph}/ctxin_{s}.csv", "w").write("\n".join(L) + "\n")
    print("ctx ok", ph)
def ev(ph):
    os.makedirs(f"{ART}/{ph}", exist_ok=True)
    for s in SEEDS[ph]:
        w = f"/workspace/orbita-data/{ph}/w_{s}.json"
        p = subprocess.run([B, "llm-eval", "--seed", s, "--weights", w, "--ctx", f"{D}/{ph}/ctxin_{s}.csv"], capture_output=True, check=True)
        open(f"{ART}/{ph}/llm_eval_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/queries_{s}.txt", "wb").write(p.stderr)
    print("eval ok", ph)
def boot(x, B_=10000, seed=12345):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B_, len(x)))].mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(x)
def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y, bool)
    if y.all() or (~y).all(): return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    for v in np.unique(s): r[s == v] = r[s == v].mean()
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
def analyze(ph):
    import collections
    per = collections.defaultdict(dict); T = []
    for s in SEEDS[ph]:
        R_ = list(csv.DictReader(open(f"{ART}/{ph}/llm_eval_{s}.csv"))); g = lambda c: [r for r in R_ if r["cond"] == c]
        for c in ("TRUE", "GEMMA", "NGRAM", "RAND", "GEMMA_SWAP"):
            for h in (1, 2, 4, 8, 16, 32): per[(c, h)][s] = np.mean([int(r[f"ok{h}"]) for r in g(c)])
        for lang in ("es", "en", "fr"): per[("GEMMA_" + lang, 16)][s] = np.mean([int(r["ok16"]) for r in g("GEMMA") if r["text_id"].startswith(lang)])
        # H-L2
        z = {(r["idx"], r["cond"], r["text_id"]): np.array([float(v) for v in r["z16"].split(";")]) for r in R_}
        ynorm = {}
        dsame, ddiff = [], []
        for idx in {r["idx"] for r in R_}:
            tz = [z[k] for k in z if k[0] == idx and k[1] == "GEMMA"]; sw = [z[k] for k in sorted(z) if k[0] == idx and k[1] == "GEMMA_SWAP"]
            own = [z[k] for k in sorted(z) if k[0] == idx and k[1] == "GEMMA"]
            ref = np.linalg.norm(z[(idx, "TRUE", "0")]) + 1e-12
            dsame.append(np.median([np.linalg.norm(a - b) / ref for i, a in enumerate(tz) for b in tz[i+1:]]))
            ddiff.append(np.median([np.linalg.norm(a - b) / ref for a, b in zip(own, sw)]))
        per[("Dsame", 16)][s] = np.median(dsame); per[("Ddiff", 16)][s] = np.median(ddiff)
        # H-C1
        G = g("GEMMA")
        per[("fam_teacher", 0)][s] = np.mean([r["fam_hat_teacher"] == r["fam"] for r in G])
        per[("fam_ctx", 0)][s] = np.mean([r["fam_hat_ctx"] == r["fam"] for r in G])
        per[("auroc", 0)][s] = auroc([float(r["teacher_div16"]) for r in G], [r["ok16"] == "0" for r in G])
        per[("auroc_TRUE", 0)][s] = auroc([float(r["teacher_div16"]) for r in g("TRUE")], [r["ok16"] == "0" for r in g("TRUE")])
        per[("fail_rate", 0)][s] = np.mean([r["ok16"] == "0" for r in G])
    S = SEEDS[ph]; v = lambda k: np.array([per[k][s] for s in S]); out = []
    P = lambda name, x: out.append(f"| {name} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos TEST iterables × 6 textos por seed)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for c in ("TRUE", "GEMMA", "NGRAM", "RAND", "GEMMA_SWAP"):
        for h in (1, 8, 16, 32): P(f"acc h{h} {c}", boot(v((c, h))))
    for l in ("es", "en", "fr"): P(f"acc h16 GEMMA_{l}", boot(v(("GEMMA_" + l, 16))))
    P("**H-L1** acc16 GEMMA − TRUE", boot(v(("GEMMA", 16)) - v(("TRUE", 16))))
    P("**H-L2** Dsame16 − Ddiff16", boot(v(("Dsame", 16)) - v(("Ddiff", 16)))); P("H-L2 Dsame16", boot(v(("Dsame", 16)))); P("H-L2 Ddiff16", boot(v(("Ddiff", 16))))
    for a_, b_ in (("es", "en"), ("es", "fr"), ("en", "fr")): P(f"**H-L3a** acc16 {a_} − {b_}", boot(v(("GEMMA_" + a_, 16)) - v(("GEMMA_" + b_, 16))))
    P("**H-L3b** acc16 GEMMA − NGRAM", boot(v(("GEMMA", 16)) - v(("NGRAM", 16)))); P("**H-L3b** acc16 GEMMA − RAND", boot(v(("GEMMA", 16)) - v(("RAND", 16))))
    P("**H-C1a** fam teacher − 1/8", boot(v(("fam_teacher", 0)) - 0.125)); P("fam teacher − fam argmax ĉ", boot(v(("fam_teacher", 0)) - v(("fam_ctx", 0))))
    P("fam argmax ĉ", boot(v(("fam_ctx", 0)))); P("**H-C1b** AUROC(fallo h16) − 0.5, GEMMA", boot(v(("auroc", 0)) - 0.5)); P("AUROC − 0.5, TRUE", boot(v(("auroc_TRUE", 0)) - 0.5)); P("tasa de fallo h16 GEMMA", boot(v(("fail_rate", 0))))
    q = sum(int(open(f"{ART}/{ph}/queries_{s}.txt").read().split("queries=")[1].split()[0]) for s in S)
    out.append(f"\nQueries totales a memoria/TEST durante llm-eval: {q}")
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"texts": texts, "fitP": fitP, "ctx": ctx, "eval": ev, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
