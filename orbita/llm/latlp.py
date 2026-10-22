"""Latente de Koopman k>2 (Exp. 1) + flujo largo (Exp. 2). docs/prereg_orbita_latente_largo_plazo.md
  latlp.py texts PH | ctx PH | calib | grid | run1 PH | analyze1 PH | run2 PH | analyze2 PH"""
import sys, os, csv, json, collections, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E, fam_eval as FE
O = f"{P0.DATA}/latlp"; ART = f"{P0.ROOT}/artifacts/orbita/latente_largo_plazo"
RNG = {"dev": ("0xFA61", "0xFA60"), "confirm": ("0xFA63", "0xFA62")}  # (H, S)
KS = (2, 4, 8); L0 = ("0.0001", "0.001", "0.01", "0.1", "1"); LM = ("0.001", "0.01", "0.1", "1", "10")
OEP = [(2, "0.0001"), (2, "0.001"), (2, "0.01"), (2, "0.1"), (2, "1"), (3, "0.0001"), (3, "0.001"), (3, "0.01"), (3, "0.1"), (3, "1")]
OMS = [(2, "0.0001"), (2, "0.001"), (2, "0.01"), (2, "0.1"), (3, "0.0001"), (3, "0.001"), (3, "0.01"), (3, "0.1")]
CK = [0, 125, 250, 500, 1000, 2000]
GROUPS = {"nuevas_8_9": [8, 9], "conocidas_1_2": [1, 2], "v4_no_iterables": [0, 3, 4, 5, 6, 7], "conocidas_0_7": list(range(8))}
NPAR = int(os.environ.get("NPAR", "6"))
def D(ph): return f"{O}/{ph}"
def colname(kind, d, l): return f"{kind}_d{d}_l{float(l)!r}".replace("'", "") if False else f"{kind}_d{d}_l{rust_f(l)}"
def rust_f(l): v = float(l); return {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1.0: "1"}[v]
def mk_text(r, lang, li):
    f, p1, p2 = int(r["fam"]), float(r["p1"]), float(r["p2"]); k = int(r["idx"]) % 2; fa, fb = E.fmts(r["idx"], k, li); art, nm = FE.name(lang, f)
    return [r["idx"], str(f), r["p1"], r["p2"], lang, str(6 + k), f"{NN.FMT_NAMES[fa]}+{NN.FMT_NAMES[fb]}", E.T[lang][k].format(art=art, f=nm, a=NN.render(p1, lang, fa), b=NN.render(p2, lang, fb))]
def texts(ph):
    os.makedirs(D(ph), exist_ok=True)
    for name, rng, n in (("H", RNG[ph][0], 16), ("S", RNG[ph][1], 200)):
        st = subprocess.check_output([P0.B, "fam-set", "--rng", rng, "--n", str(n)]).decode(); open(f"{D(ph)}/{name}_set.csv", "w").write(st); out = []
        for r in csv.DictReader(st.splitlines()):
            if name == "H": out += [mk_text(r, lang, li) for li, lang in enumerate(E.LANGS)]
            else: li = int(r["idx"]) % 3; out.append(mk_text(r, E.LANGS[li], li))
        open(f"{D(ph)}/{name}.tsv", "w").write("".join("\t".join(x) + "\n" for x in out)); open(f"{D(ph)}/{name}.txt", "w").write("".join(x[7] + "\n" for x in out))
        print(ph, name, len(out), "textos")
def d2_chunks(tok, ph, s, n, ch=200):
    return np.concatenate([E.d2_predict(tok, ph, s, list(range(i, min(n, i + ch)))) for i in range(0, n, ch)])
HDR = "idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))
def ctx(ph):
    H = P0.tsv(f"{D(ph)}/H.tsv"); S = P0.tsv(f"{D(ph)}/S.tsv"); os.makedirs(f"{D(ph)}/ctx", exist_ok=True)
    for s in P0.SEEDS[ph]:
        dn = f"{D(ph)}/ctx/{s}.done"
        if os.path.exists(dn): continue
        si = int(s, 16); CH = d2_chunks(f"{D(ph)}/H.tok", ph, s, len(H)); CS = d2_chunks(f"{D(ph)}/S.tok", ph, s, len(S))
        ex = sorted({int(r[0]) for r in H}); order = np.random.default_rng(si).permutation(len(ex)); L = [HDR]; Mt = ["pos,idx,lang,fmt,parse_ok"]
        for pos, oi in enumerate(order):
            idx = ex[oi]; lang = E.LANGS[(idx + si) % 3]; j = next(j for j, r in enumerate(H) if int(r[0]) == idx and r[4] == lang); r = H[j]
            c, ok = E.build_c(CH[j], r[7]); L.append(f"{r[0]},E2E,{r[4]}-{r[5]}," + ",".join(repr(float(x)) for x in c)); Mt.append(f"{pos},{idx},{lang},{r[6]},{int(ok)}")
        open(f"{D(ph)}/ctx/H_{s}.csv", "w").write("\n".join(L) + "\n"); open(f"{D(ph)}/ctx/Hmeta_{s}.csv", "w").write("\n".join(Mt) + "\n")
        rng = np.random.default_rng(si); order = rng.permutation(len(S)); ks = rng.choice(KS, len(S)); L = [HDR]; Mt = ["pos,idx,k,lang,fmt,parse_ok"]
        for pos, j in enumerate(order):
            r = S[j]; c, ok = E.build_c(CS[j], r[7]); L.append(f"{r[0]},k{ks[pos]},{r[4]}-{r[5]}," + ",".join(repr(float(x)) for x in c)); Mt.append(f"{pos},{r[0]},{ks[pos]},{r[4]},{r[6]},{int(ok)}")
        open(f"{D(ph)}/ctx/S_{s}.csv", "w").write("\n".join(L) + "\n"); open(f"{D(ph)}/ctx/Smeta_{s}.csv", "w").write("\n".join(Mt) + "\n")
        open(dn, "w").write("ok\n"); print(ph, s, "ctx ok", flush=True)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def job(out, argv, w):
    dn = out + ".done"
    if os.path.exists(dn): return
    os.makedirs(os.path.dirname(out), exist_ok=True); h0 = sha(w); p = subprocess.run([P0.B] + argv, capture_output=True, check=True)
    open(out, "wb").write(p.stdout); open(out[:-4] + ".err", "w").write(p.stderr.decode() + f"\nsha256_before={h0}\nsha256_after={sha(w)}\n"); assert sha(w) == h0; open(dn, "w").write("ok\n")
def e1(ph, s, k, tau, l0, lm, tag):
    w = f"{P0.DATA}/{ph}/w_{s}.json"
    return (f"{D(ph)}/e1/{tag}/k{k}_{s}.csv", ["lat-eval", "exp1", "--weights", w, "--set", f"{D(ph)}/H_set.csv", "--ctx", f"{D(ph)}/ctx/H_{s}.csv", "--k", str(k), "--tau", repr(tau), "--lam0", l0, "--lamm", lm], w)
def pool(jobs):
    with ThreadPoolExecutor(NPAR) as ex: [f.result() for f in [ex.submit(job, *j) for j in jobs]]
def load(p): return list(csv.DictReader(open(p)))
def calib():
    pool([e1("dev", s, k, 1e9, "0.01", "0.1", "calib") for s in P0.SEEDS["dev"] for k in KS]); tau = {}
    for k in KS:
        v = [float(r["s_known"]) for s in P0.SEEDS["dev"] for r in load(f"{D('dev')}/e1/calib/k{k}_{s}.csv") if int(r["fam"]) in (1, 2)]
        tau[k] = float(np.percentile(v, 95)); print(f"k={k} n={len(v)} tau={tau[k]!r}")
    json.dump({"tau": tau, "rule": "P95 S_known H DEV fam 1-2"}, open(f"{O}/tau.json", "w"), indent=1)
def taus(): return {int(k): v for k, v in json.load(open(f"{O}/tau.json"))["tau"].items()}
def acc(R, col, fams=None): return np.mean([int(r[col]) for r in R if fams is None or int(r["fam"]) in fams])
def grid():
    t = taus(); pool([e1("dev", s, k, t[k], l0, lm, f"l0{l0}_lm{lm}") for l0 in L0 for lm in LM for k in KS for s in P0.SEEDS["dev"]])
    R = lambda tag, k, s: load(f"{D('dev')}/e1/{tag}/k{k}_{s}.csv")
    score = lambda tag, col: np.mean([np.mean([acc(R(tag, k, s), col) for s in P0.SEEDS["dev"]]) for k in KS])
    tab = []; tags = [(l0, lm) for l0 in L0 for lm in LM]
    sl = {(l0, lm): score(f"l0{l0}_lm{lm}", "latmem16") for l0, lm in tags}; se = {l0: score(f"l0{l0}_lm{LM[0]}", "latep16") for l0 in L0}
    pick = lambda d, key: sorted([g for g in d if d[g] == max(d.values())], key=key)[0]
    lat = pick(sl, lambda g: (-float(g[0]), -float(g[1]))); lep = pick(se, lambda g: -float(g))
    ref = f"l0{lat[0]}_lm{lat[1]}"; so = {g: score(ref, f"oep_d{g[0]}_l{g[1]}") for g in OEP}; sm = {g: score(ref, f"oms_d{g[0]}_l{g[1]}") for g in OMS}
    oep = pick(so, lambda g: (g[0], -float(g[1]))); oms = pick(sm, lambda g: (g[0], -float(g[1])))
    tab += [f"LAT λ0={a} λm={b} score={v:.4f}" for (a, b), v in sl.items()] + [f"LAT_EP λ0={a} score={v:.4f}" for a, v in se.items()]
    tab += [f"ORACLE_EP D={a} λ={b} score={v:.4f}" for (a, b), v in so.items()] + [f"ORACLE_MS D={a} λ={b} score={v:.4f}" for (a, b), v in sm.items()]
    S = {"lat": list(lat), "latep_lam0": lep, "oep": list(oep), "oms": list(oms)}
    json.dump({"selected": S, "table": tab}, open(f"{O}/selected.json", "w"), indent=1, ensure_ascii=False); print("\n".join(tab)); print(S)
def sel(): return json.load(open(f"{O}/selected.json"))["selected"]
def run1(ph):
    t = taus(); S = sel(); tags = {(S["lat"][0], S["lat"][1]), (S["latep_lam0"], LM[0])}
    pool([e1(ph, s, k, t[k], a, b, f"l0{a}_lm{b}") for a, b in tags for k in KS for s in P0.SEEDS[ph]])
def errinfo(p):
    e = open(p).read(); h = e.split("sha256_before=")[1].split()[0]; return e.split()[0], e.count(h) == 2
def analyze1(ph):
    S = sel(); Ss = P0.SEEDS[ph]; out = []; per = collections.defaultdict(dict); qs = []; shas = []
    tl = f"l0{S['lat'][0]}_lm{S['lat'][1]}"; te = f"l0{S['latep_lam0']}_lm{LM[0]}"
    cols = {"BASE": (tl, "base16"), "OBS": (tl, "obs16"), "RES": (tl, "res16"), "RES_EP": (tl, "resep16"), "LAT_EP": (te, "latep16"), "LAT": (tl, "latmem16"),
            "FINAL": (tl, "final16"), "ORACLE_EP": (tl, f"oep_d{S['oep'][0]}_l{S['oep'][1]}"), "ORACLE_MS": (tl, f"oms_d{S['oms'][0]}_l{S['oms'][1]}"), "ORACLE_LS": (tl, "ls16")}
    for s in Ss:
        for k in KS:
            R = {tg: load(f"{D(ph)}/e1/{tg}/k{k}_{s}.csv") for tg in {tl, te}}
            for tg in {tl, te}: q, ok = errinfo(f"{D(ph)}/e1/{tg}/k{k}_{s}.err"); qs.append(q); shas.append(ok)
            for c, (tg, col) in cols.items():
                for g, fams in list(GROUPS.items()) + [(f"fam{f}", [f]) for f in range(10)]: per[(k, g, c)][s] = acc(R[tg], col, fams)
            L = R[tl]; N = [r for r in L if int(r["fam"]) >= 8]
            for g, fams in GROUPS.items(): per[(k, g, "trigger")][s] = np.mean([r["latmem_action"] != "known" for r in L if int(r["fam"]) in fams])
            per[(k, "lib_rate_new")][s] = np.mean([r["latmem_action"] == "lib" for r in N])
            pu = [float(r["latmem_purity"]) for r in N if r["latmem_action"] == "lib"]; per[(k, "purity_new")][s] = np.mean(pu) if pu else np.nan
            sk = np.array([float(r["s_known"]) for r in L]); fam = np.array([int(r["fam"]) for r in L]); per[(k, "auroc")][s] = P0.auroc(sk, fam >= 8)
    v = lambda *key: np.array([per[key].get(s, np.nan) for s in Ss])
    def Pr(n, x):
        x = np.asarray(x, float)
        if np.isnan(x).all(): return
        b = P0.boot(x[~np.isnan(x)]); out.append(f"| {n} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} |")
    out += [f"## EXP1 {ph.upper()} (16 seeds × 160 episodios H; τ = {taus()}; selección {S})", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    H = [("H1a", "nuevas_8_9", "LAT_EP", "RES_EP"), ("H1b", "nuevas_8_9", "LAT", "RES"), ("H2", "nuevas_8_9", "LAT", "LAT_EP"), ("H3", "nuevas_8_9", "LAT", "OBS"),
         ("H4", "nuevas_8_9", "LAT", "ORACLE_EP"), ("H4", "nuevas_8_9", "LAT", "ORACLE_MS"), ("H5", "conocidas_0_7", "LAT", "OBS"), ("H5", "conocidas_1_2", "LAT", "OBS"),
         ("H5", "conocidas_0_7", "FINAL", "OBS"), ("H5", "conocidas_1_2", "FINAL", "OBS"), ("—", "nuevas_8_9", "LAT", "ORACLE_LS"), ("—", "nuevas_8_9", "RES", "OBS"), ("—", "v4_no_iterables", "LAT", "OBS")]
    for k in KS:
        out.append(f"| **k = {k}** | | | |")
        for g in list(GROUPS) + [f"fam{f}" for f in range(10)]:
            for c in cols: Pr(f"{g} acc16 {c}{k}", v(k, g, c))
        for h, g, a, b in H: Pr(f"**{h}** acc16 {a}{k} − {b}{k} ({g})", v(k, g, a) - v(k, g, b))
        for g in GROUPS: Pr(f"tasa de disparo LAT{k} ({g})", v(k, g, "trigger"))
        for q in ("lib_rate_new", "purity_new", "auroc"): Pr(f"{q} (k={k})", v(k, q))
    out.append(f"\nsha256 pesos idéntico en todas: {all(shas)}; queries: {sorted(set(qs))}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_exp1_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
def run2(ph):
    t = taus(); S = sel(); jobs = []
    for s in P0.SEEDS[ph]:
        w = f"{P0.DATA}/{ph}/w_{s}.json"; out = f"{D(ph)}/e2/stream_{s}.csv"
        jobs.append((out, ["lat-eval", "long", "--weights", w, "--set", f"{D(ph)}/S_set.csv", "--ctx", f"{D(ph)}/ctx/S_{s}.csv", "--held", f"{D(ph)}/ctx/H_{s}.csv", "--hset", f"{D(ph)}/H_set.csv",
                     "--taus", ",".join(repr(t[k]) for k in KS), "--lam0", S["lat"][0], "--lamm", S["lat"][1], "--oep-d", str(S["oep"][0]), "--oep-l", S["oep"][1], "--oms-d", str(S["oms"][0]), "--oms-l", S["oms"][1],
                     "--ckpts", ",".join(map(str, CK)), "--held-out", f"{D(ph)}/e2/held_{s}.csv"], w))
    os.makedirs(f"{D(ph)}/e2", exist_ok=True); pool(jobs)
def analyze2(ph):
    Ss = P0.SEEDS[ph]; out = []; per = collections.defaultdict(dict); qs = []; shas = []
    for s in Ss:
        q, ok = errinfo(f"{D(ph)}/e2/stream_{s}.err"); qs.append(q); shas.append(ok)
        Hh = load(f"{D(ph)}/e2/held_{s}.csv"); St = load(f"{D(ph)}/e2/stream_{s}.csv")
        for k in KS:
            for g, fams in GROUPS.items():
                for n in CK:
                    G = [r for r in Hh if int(r["ckpt"]) == n and int(r["k"]) == k and int(r["fam"]) in fams]
                    per[(k, g, "LAT", n)][s] = np.mean([int(r["lat16"]) for r in G])
                    if n == 0:
                        for c in ("obs16", "oep16", "oms16"): per[(k, g, c)][s] = np.mean([int(r[c]) for r in G])
                    if g == "nuevas_8_9":
                        pu = [float(r["purity"]) for r in G if r["action"] == "lib"]; per[(k, "purity", n)][s] = np.mean(pu) if pu else np.nan
                        per[(k, "librate", n)][s] = np.mean([r["action"] == "lib" for r in G])
                    if g == "conocidas_0_7": per[(k, "trigger_known", n)][s] = np.mean([r["action"] != "known" for r in G])
            x = np.log2(CK[1:]); y = np.array([per[(k, "nuevas_8_9", "LAT", n)][s] for n in CK[1:]]); per[(k, "slope")][s] = np.polyfit(x, y, 1)[0]
        for n in CK: per[("libsize", n)][s] = int(St[n - 1]["lib_size"]) if n > 0 else 0
        for b in range(0, 2000, 250):
            B = [r for r in St[b:b + 250] if int(r["fam"]) >= 8]
            for c in ("lat16", "latep16", "obs16"): per[("stream", b, c)][s] = np.mean([int(r[c]) for r in B])
            Bk = [r for r in St[b:b + 250] if int(r["fam"]) < 8]; per[("stream_known", b)][s] = np.mean([int(r["lat16"]) - int(r["obs16"]) for r in Bk])
    v = lambda *key: np.array([per[key].get(s, np.nan) for s in Ss])
    def Pr(n, x):
        x = np.asarray(x, float)
        if np.isnan(x).all(): return
        b = P0.boot(x[~np.isnan(x)]); out.append(f"| {n} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} |")
    out += [f"## EXP2 {ph.upper()} (16 seeds; flujo 2000 episodios S; held-out H 160 episodios, memoria congelada en cada punto de control)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for k in KS:
        out.append(f"| **k_eval = {k}** | | | |")
        for g in GROUPS:
            for n in CK: Pr(f"{g} acc16 LAT n={n}", v(k, g, "LAT", n))
            for c in ("obs16", "oep16", "oms16"): Pr(f"{g} acc16 {c[:-2].upper()}", v(k, g, c))
        N = "nuevas_8_9"; K = "conocidas_0_7"
        Pr(f"**L1** pendiente acc16 nuevas vs log2 n (k={k})", v(k, "slope"))
        Pr(f"**L2** acc16 nuevas n=2000 − n=0 (k={k})", v(k, N, "LAT", 2000) - v(k, N, "LAT", 0))
        Pr(f"**L3** acc16 nuevas n=2000 − ORACLE_EP (k={k})", v(k, N, "LAT", 2000) - v(k, N, "oep16"))
        Pr(f"**L3** acc16 nuevas n=2000 − ORACLE_MS (k={k})", v(k, N, "LAT", 2000) - v(k, N, "oms16"))
        Pr(f"**L4** acc16 conocidas 0–7 n=2000 − OBS (k={k})", v(k, K, "LAT", 2000) - v(k, K, "obs16"))
        Pr(f"**L4** acc16 conocidas 0–7 n=2000 − n=125 (k={k})", v(k, K, "LAT", 2000) - v(k, K, "LAT", 125))
        Pr(f"  acc16 conocidas 1–2 n=2000 − OBS (k={k})", v(k, "conocidas_1_2", "LAT", 2000) - v(k, "conocidas_1_2", "obs16"))
        for n in CK: Pr(f"pureza recuperación nuevas n={n} (k={k})", v(k, "purity", n)); Pr(f"tasa uso memoria nuevas n={n} (k={k})", v(k, "librate", n)); Pr(f"tasa disparo conocidas n={n} (k={k})", v(k, "trigger_known", n))
    for n in CK: Pr(f"tamaño memoria n={n}", v("libsize", n))
    for b in range(0, 2000, 250):
        for c in ("lat16", "latep16", "obs16"): Pr(f"flujo nuevas acc16 {c[:-2].upper()} episodios {b}–{b + 249}", v("stream", b, c))
        Pr(f"flujo conocidas acc16 LAT − OBS episodios {b}–{b + 249}", v("stream_known", b))
    out.append(f"\nsha256 pesos idéntico en todas: {all(shas)}; queries: {sorted(set(qs))}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_exp2_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"texts": texts, "ctx": ctx, "calib": calib, "grid": grid, "run1": run1, "analyze1": analyze1, "run2": run2, "analyze2": analyze2}[sys.argv[1]](*sys.argv[2:])
