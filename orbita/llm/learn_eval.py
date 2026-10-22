"""Aprendizaje online de dinámicas (docs/prereg_orbita_aprendizaje_online.md).
  learn_eval.py texts PH | ctx PH | calib | grid | run PH | analyze PH"""
import sys, os, csv, json, collections, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E, fam_eval as FE
O = "/workspace/orbita-data/learn"; ART = f"{P0.ROOT}/artifacts/orbita/aprendizaje_online"
RNG = {"dev": "0xFA50", "confirm": "0xFA51"}
GRID = [(d, l) for d in (2, 3) for l in ("0.0001", "0.001", "0.01", "0.1", "1")]
GROUPS = {"nuevas_8_9": [8, 9], "conocidas_1_2": [1, 2], "v4_no_iterables": [0, 3, 4, 5, 6, 7], "conocidas_0_7": list(range(8))}
NPAR = int(os.environ.get("NPAR", "6"))
def D(ph): return f"{O}/{ph}"
def texts(ph):
    os.makedirs(D(ph), exist_ok=True); st = subprocess.check_output([P0.B, "fam-set", "--rng", RNG[ph]]).decode(); open(f"{D(ph)}/set.csv", "w").write(st); out = []
    for r in csv.DictReader(st.splitlines()):
        f, p1, p2 = int(r["fam"]), float(r["p1"]), float(r["p2"]); k = int(r["idx"]) % 2
        for li, lang in enumerate(E.LANGS):
            fa, fb = E.fmts(r["idx"], k, li); art, nm = FE.name(lang, f)
            out.append([r["idx"], str(f), r["p1"], r["p2"], lang, str(6 + k), f"{NN.FMT_NAMES[fa]}+{NN.FMT_NAMES[fb]}", E.T[lang][k].format(art=art, f=nm, a=NN.render(p1, lang, fa), b=NN.render(p2, lang, fb))])
    open(f"{D(ph)}/texts.tsv", "w").write("".join("\t".join(x) + "\n" for x in out)); open(f"{D(ph)}/texts.txt", "w").write("".join(x[7] + "\n" for x in out)); print(len(out), "textos", ph)
def ctx(ph):
    rows = P0.tsv(f"{D(ph)}/texts.tsv"); os.makedirs(f"{D(ph)}/ctx", exist_ok=True)
    for s in P0.SEEDS[ph]:
        dn = f"{D(ph)}/ctx/{s}.done"
        if os.path.exists(dn): continue
        si = int(s, 16); C = E.d2_predict(f"{D(ph)}/texts.tok", ph, s, list(range(len(rows))))
        ex = sorted({int(r[0]) for r in rows}); order = np.random.default_rng(si).permutation(len(ex))
        L = ["idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))]; M = ["pos,idx,lang,fmt,parse_ok"]
        for pos, oi in enumerate(order):
            idx = ex[oi]; lang = E.LANGS[(idx + si) % 3]; j = next(j for j, r in enumerate(rows) if int(r[0]) == idx and r[4] == lang); r = rows[j]
            c, ok = E.build_c(C[j], r[7]); L.append(f"{r[0]},E2E,{r[4]}-{r[5]}," + ",".join(repr(float(x)) for x in c)); M.append(f"{pos},{idx},{lang},{r[6]},{int(ok)}")
        open(f"{D(ph)}/ctx/ctx_{s}.csv", "w").write("\n".join(L) + "\n"); open(f"{D(ph)}/ctx/meta_{s}.csv", "w").write("\n".join(M) + "\n")
        open(dn, "w").write("ok\n"); print(ph, s, "ctx ok", flush=True)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def one(ph, s, k, deg, lam, tau, tag):
    out = f"{D(ph)}/eval/{tag}/k{k}_{s}.csv"; dn = out + ".done"
    if os.path.exists(dn): return
    os.makedirs(os.path.dirname(out), exist_ok=True); w = f"/workspace/orbita-data/{ph}/w_{s}.json"; h0 = sha(w)
    p = subprocess.run([P0.B, "learn-eval", "--weights", w, "--set", f"{D(ph)}/set.csv", "--ctx", f"{D(ph)}/ctx/ctx_{s}.csv", "--k", str(k), "--deg", str(deg), "--lam", lam, "--tau", repr(tau)], capture_output=True, check=True)
    open(out, "wb").write(p.stdout); open(out[:-4] + ".err", "w").write(p.stderr.decode() + f"\nsha256_before={h0}\nsha256_after={sha(w)}\n")
    assert sha(w) == h0; open(dn, "w").write("ok\n")
def pool(jobs):
    with ThreadPoolExecutor(NPAR) as ex: [f.result() for f in [ex.submit(one, *j) for j in jobs]]
def load(ph, tag, k, s): return list(csv.DictReader(open(f"{D(ph)}/eval/{tag}/k{k}_{s}.csv")))
def calib():
    pool([("dev", s, k, 2, "1", 1e9, "calib") for s in P0.SEEDS["dev"] for k in (1, 2)]); tau = {}
    for k in (1, 2):
        v = [float(r["s_known"]) for s in P0.SEEDS["dev"] for r in load("dev", "calib", k, s) if int(r["fam"]) in (1, 2)]
        tau[k] = float(np.percentile(v, 95)); print(f"k={k} n={len(v)} tau={tau[k]:.6f}")
    json.dump({"tau": tau, "rule": "P95 S_known DEV fam 1-2"}, open(f"{O}/tau.json", "w"), indent=1)
def taus(): return {int(k): v for k, v in json.load(open(f"{O}/tau.json"))["tau"].items()}
def acc(R, col, fams=None): v = [int(r[col]) for r in R if fams is None or int(r["fam"]) in fams]; return np.mean(v)
def grid():
    t = taus(); pool([("dev", s, 2, d, l, t[2], f"d{d}_l{l}") for d, l in GRID for s in P0.SEEDS["dev"]]); sel = {}; tab = []
    for cond in ("learn", "ofam", "oep"):
        sc = {(d, l): np.mean([acc(load("dev", f"d{d}_l{l}", 2, s), f"{cond}_ok16") for s in P0.SEEDS["dev"]]) for d, l in GRID}
        best = max(sc.values()); cands = [g for g in GRID if sc[g] == best]; cands.sort(key=lambda g: (g[0], -float(g[1]))); sel[cond] = list(cands[0])
        tab += [f"{cond} D={d} λ={l} acc16_dev={sc[(d, l)]:.4f}" for d, l in GRID]
    json.dump({"selected": sel, "table": tab}, open(f"{O}/selected.json", "w"), indent=1, ensure_ascii=False); print("\n".join(tab)); print(sel)
def sel(): return json.load(open(f"{O}/selected.json"))["selected"]
def run(ph):
    t = taus(); S = sel(); cfgs = {tuple(v) for v in S.values()}
    pool([(ph, s, k, d, l, t[k], f"d{d}_l{l}") for d, l in cfgs for k in (1, 2) for s in P0.SEEDS[ph]])
def analyze(ph):
    S = P0.SEEDS[ph]; Sel = sel(); out = []; per = collections.defaultdict(dict); t = taus()
    tagof = {c: f"d{Sel[c][0]}_l{Sel[c][1]}" for c in Sel}
    src = {"base": "learn", "obs": "learn", "learn": "learn", "lep": "learn", "final": "learn", "ls": "learn", "ofam": "ofam", "oep": "oep"}
    info = collections.defaultdict(list); qs = []
    for s in S:
        meta = {int(r["pos"]): r for r in csv.DictReader(open(f"{D(ph)}/ctx/meta_{s}.csv"))}
        for k in (1, 2):
            R = {c: load(ph, tagof[src[c]], k, s) for c in src}
            err = open(f"{D(ph)}/eval/{tagof['learn']}/k{k}_{s}.err").read(); qs.append(err.split()[0]); info[(k, "nmod")].append(int(err.split()[1].split("=")[1]))
            info[(k, "sha_ok")].append(err.count(err.split("sha256_before=")[1].split()[0]) == 2)
            lib = json.loads(err.split("library=")[1].split("\n")[0])
            for c in src:
                for g, fams in list(GROUPS.items()) + [(f"fam{f}", [f]) for f in range(10)]:
                    for h in (8, 16): per[(k, g, c, h)][s] = acc(R[c], f"{c}_ok{h}", fams)
            L = R["learn"]
            for g, fams in GROUPS.items():
                G = [r for r in L if int(r["fam"]) in fams]
                per[(k, g, "trigger")][s] = np.mean([r["learn_action"] != "known" for r in G])
            N = [r for r in L if int(r["fam"]) >= 8]
            per[(k, "reuse")][s] = np.mean([r["learn_action"] in ("lib", "join") for r in N])
            per[(k, "nmod")][s] = len(lib)
            mem = [mm["members"] for mm in lib]; per[(k, "purity")][s] = (sum(collections.Counter(x).most_common(1)[0][1] for x in mem) / sum(map(len, mem))) if mem else np.nan
            per[(k, "nmod_newfam_majority")][s] = sum(collections.Counter(x).most_common(1)[0][0] >= 8 for x in mem)
            sk = np.array([float(r["s_known"]) for r in L]); fam = np.array([int(r["fam"]) for r in L])
            per[(k, "auroc_new_vs_0_7")][s] = P0.auroc(sk, fam >= 8)
            m12 = (fam >= 8) | (fam == 1) | (fam == 2); per[(k, "auroc_new_vs_1_2")][s] = P0.auroc(sk[m12], fam[m12] >= 8)
            rank = collections.Counter(); bins = collections.defaultdict(list)
            for r in L:
                f = int(r["fam"]); rank[f] += 1
                if f >= 8: bins["1-4" if rank[f] <= 4 else "5-8" if rank[f] <= 8 else "9-16"].append(int(r["learn_ok16"]))
            for b, v in bins.items(): per[(k, "curve", b)][s] = np.mean(v)
            for lang in E.LANGS:
                for c in ("obs", "learn"):
                    v = [int(r[f"{c}_ok16"]) for r in L if r["text_id"].startswith(lang) and int(r["fam"]) >= 8]; per[(k, "lang", lang, c)][s] = np.mean(v)
            for fm in ("palabras", "otros"):
                for c in ("obs", "learn"):
                    v = [int(r[f"{c}_ok16"]) for r in L if int(r["fam"]) >= 8 and (("palabras" in meta[int(r["pos"])]["fmt"]) == (fm == "palabras"))]; per[(k, "fmt", fm, c)][s] = np.mean(v) if v else np.nan
            per[(k, "parse_fail")][s] = sum(1 - int(m["parse_ok"]) for m in meta.values())
    v = lambda *key: np.array([per[key].get(s, np.nan) for s in S])
    def Pr(n, x):
        x = np.asarray(x, float)
        if np.isnan(x).all(): return
        b = P0.boot(x[~np.isnan(x)]); out.append(f"| {n} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} |")
    out += [f"## {ph.upper()} (16 seeds × 160 episodios; τ = {t}; selección {Sel})", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    names = {"base": "BASE", "obs": "OBS", "learn": "LEARN", "lep": "LEARN_EP", "final": "FINAL", "ofam": "ORACLE_FAM", "oep": "ORACLE_EP", "ls": "ORACLE_LS"}
    for k in (2, 1):
        out.append(f"| **k = {k}** | | | |")
        for g in list(GROUPS) + [f"fam{f}" for f in range(10)]:
            for c in names:
                Pr(f"{g} acc16 {names[c]}{k}", v(k, g, c, 16))
                if g in GROUPS: Pr(f"{g} acc8 {names[c]}{k}", v(k, g, c, 8))
        H = [("HL1", "nuevas_8_9", "learn", "obs"), ("HL2", "conocidas_0_7", "learn", "obs"), ("HL2", "conocidas_1_2", "learn", "obs"),
             ("HL2", "conocidas_0_7", "final", "obs"), ("HL2", "conocidas_1_2", "final", "obs"), ("HL3", "nuevas_8_9", "learn", "lep"),
             ("HL4", "nuevas_8_9", "learn", "ofam"), ("HL4b", "nuevas_8_9", "learn", "oep"), ("—", "nuevas_8_9", "final", "learn"),
             ("—", "v4_no_iterables", "learn", "obs"), ("—", "nuevas_8_9", "learn", "ls")]
        for h, g, a, b in H:
            Pr(f"**{h}** acc16 {names[a]}{k} − {names[b]}{k} ({g})", v(k, g, a, 16) - v(k, g, b, 16)); Pr(f"  acc8 {names[a]}{k} − {names[b]}{k} ({g})", v(k, g, a, 8) - v(k, g, b, 8))
        for g in GROUPS: Pr(f"tasa de disparo LEARN{k} ({g})", v(k, g, "trigger"))
        for q in ("reuse", "nmod", "nmod_newfam_majority", "purity", "auroc_new_vs_0_7", "auroc_new_vs_1_2", "parse_fail"): Pr(f"{q} (k={k})", v(k, q))
        for b in ("1-4", "5-8", "9-16"): Pr(f"curva acc16 LEARN{k} nuevas, aparición {b}", v(k, "curve", b))
        for lang in E.LANGS:
            for c in ("obs", "learn"): Pr(f"nuevas acc16 {names[c]}{k} idioma {lang}", v(k, "lang", lang, c))
        for fm in ("palabras", "otros"):
            for c in ("obs", "learn"): Pr(f"nuevas acc16 {names[c]}{k} formato {fm}", v(k, "fmt", fm, c))
        out.append(f"\nsha256 pesos ÓRBITA idéntico antes/después (k={k}): {all(info[(k, 'sha_ok')])}\n")
    out.append(f"\nQueries v4 (todas las ejecuciones): {sorted(set(qs))}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
def explo(ph):
    """EXPLORATORIO (post hoc, no prerregistrado): horizonte de pasos consecutivos correctos tras k (binario con columnas *_hz añadidas).
    Verifica que las columnas comunes coinciden con la ejecución principal."""
    t = taus(); d, l = sel()["learn"]; pool([(ph, s, k, d, l, t[k], "explo_hz") for k in (1, 2) for s in P0.SEEDS[ph]]); out = []
    same = True; per = collections.defaultdict(dict)
    for s in P0.SEEDS[ph]:
        for k in (1, 2):
            A = load(ph, f"d{d}_l{l}", k, s); Bx = load(ph, "explo_hz", k, s)
            same &= all(all(a[c] == b[c] for c in a) for a, b in zip(A, Bx)) and len(A) == len(Bx)
            for g, fams in GROUPS.items():
                for c in ("obs", "learn", "lep", "oep", "ofam", "ls"):
                    per[(k, g, c)][s] = np.mean([int(r[f"{c}_hz"]) for r in Bx if int(r["fam"]) in fams])
    out += [f"## EXPLORATORIO {ph.upper()}: horizonte medio (pasos consecutivos correctos tras y_k; máx 16 − k)", f"Columnas comunes idénticas a la ejecución principal: {same}", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for k in (2, 1):
        for g in GROUPS:
            for c in ("obs", "learn", "lep", "oep", "ofam", "ls"):
                b = P0.boot(np.array([per[(k, g, c)][s] for s in P0.SEEDS[ph]])); out.append(f"| k={k} {g} horizonte {c.upper()} | {b[0]:.3f} | [{b[1]:.3f}, {b[2]:.3f}] | {b[3]} |")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/explo_horizonte_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"texts": texts, "ctx": ctx, "calib": calib, "grid": grid, "run": run, "analyze": analyze, "explo": explo}[sys.argv[1]](*sys.argv[2:])
