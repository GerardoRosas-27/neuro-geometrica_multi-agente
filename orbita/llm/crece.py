"""ÓRBITA que crece (expertos añadidos por sorpresa persistente). docs/prereg_orbita_crece.md
  crece.py texts PH | ctx PH | calib | grid | confirm | analyze PH"""
import sys, os, csv, json, collections, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E, fam_eval as FE
O = "/workspace/orbita-data/crece"; ART = f"{P0.ROOT}/artifacts/orbita/orbita_crece"
RNG = {"dev": ("0xFA71", "0xFA70"), "confirm": ("0xFA73", "0xFA72")}  # (H, S)
NF = 12; KS = (2, 4, 8); CES = ("1", "2", "4"); CK = [0, 150, 300, 600, 1200, 2400]
NEW = {"es": (("un", "pulso radial"), ("una", "cizalla saturada")), "en": (("a", "radial pulse"), ("a", "saturated shear")), "fr": (("une", "pulsation radiale"), ("un", "cisaillement saturé"))}
GROUPS = {"nuevas_8_11": [8, 9, 10, 11], "nuevas_8_9": [8, 9], "nuevas_10_11": [10, 11], "conocidas_0_7": list(range(8)), "conocidas_1_2": [1, 2], "todas": list(range(NF))}
LAT_SEL = {"l0": "0.0001", "lm": "0.001"}  # fijado por 9166f1f (latlp/selected.json)
NPAR = int(os.environ.get("NPAR", "6"))
def D(ph): return f"{O}/{ph}"
def name(lang, f): return FE.name(lang, f) if f < 10 else NEW[lang][f - 10]
def mk_text(r, lang, li):
    f, p1, p2 = int(r["fam"]), float(r["p1"]), float(r["p2"]); k = int(r["idx"]) % 2; fa, fb = E.fmts(r["idx"], k, li); art, nm = name(lang, f)
    return [r["idx"], str(f), r["p1"], r["p2"], lang, str(6 + k), f"{NN.FMT_NAMES[fa]}+{NN.FMT_NAMES[fb]}", E.T[lang][k].format(art=art, f=nm, a=NN.render(p1, lang, fa), b=NN.render(p2, lang, fb))]
def texts(ph):
    os.makedirs(D(ph), exist_ok=True)
    for nm_, rng, n in (("H", RNG[ph][0], 16), ("S", RNG[ph][1], 200)):
        st = subprocess.check_output([P0.B, "fam-set", "--rng", rng, "--n", str(n), "--nf", str(NF)]).decode(); open(f"{D(ph)}/{nm_}_set.csv", "w").write(st); out = []
        for r in csv.DictReader(st.splitlines()):
            if nm_ == "H": out += [mk_text(r, lang, li) for li, lang in enumerate(E.LANGS)]
            else: li = int(r["idx"]) % 3; out.append(mk_text(r, E.LANGS[li], li))
        open(f"{D(ph)}/{nm_}.tsv", "w").write("".join("\t".join(x) + "\n" for x in out)); open(f"{D(ph)}/{nm_}.txt", "w").write("".join(x[7] + "\n" for x in out))
        print(ph, nm_, len(out), "textos")
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
    print("job ok", out, flush=True)
def pool(jobs):
    with ThreadPoolExecutor(NPAR) as ex: [f.result() for f in [ex.submit(job, *j) for j in jobs]]
def load(p): return list(csv.DictReader(open(p)))
def W(ph, s): return f"/workspace/orbita-data/{ph}/w_{s}.json"
def calib():
    ph = "dev"; jobs = [(f"{D(ph)}/calib/k{k}_{s}.csv", ["lat-eval", "exp1", "--weights", W(ph, s), "--set", f"{D(ph)}/H_set.csv", "--ctx", f"{D(ph)}/ctx/H_{s}.csv", "--k", str(k), "--tau", "1e9",
            "--lam0", LAT_SEL["l0"], "--lamm", LAT_SEL["lm"]], W(ph, s)) for s in P0.SEEDS[ph] for k in KS]
    pool(jobs); tau = {}
    for k in KS:
        v = [float(r["s_known"]) for s in P0.SEEDS[ph] for r in load(f"{D(ph)}/calib/k{k}_{s}.csv") if int(r["fam"]) in (1, 2)]
        tau[k] = float(np.percentile(v, 95)); print(f"k={k} n={len(v)} tau={tau[k]!r}")
    json.dump({"tau": tau, "rule": "P95 S_known H DEV fam 1-2"}, open(f"{O}/tau.json", "w"), indent=1)
def taus(): return {int(k): v for k, v in json.load(open(f"{O}/tau.json"))["tau"].items()}
def gjob(ph, s, ce):
    t = taus(); b = f"{D(ph)}/run/ce{ce}"
    return (f"{b}/stream_{s}.csv", ["grow-eval", "--weights", W(ph, s), "--set", f"{D(ph)}/S_set.csv", "--ctx", f"{D(ph)}/ctx/S_{s}.csv", "--held", f"{D(ph)}/ctx/H_{s}.csv", "--hset", f"{D(ph)}/H_set.csv",
            "--taus", ",".join(repr(t[k]) for k in KS), "--ce", ce, "--rng-seed", str(int(s, 16)), "--ckpts", ",".join(map(str, CK)), "--held-out", f"{b}/held_{s}.csv", "--events", f"{b}/events_{s}.jsonl"], W(ph, s))
def score(ph, ce):
    R = [r for s in P0.SEEDS[ph] for r in load(f"{D(ph)}/run/ce{ce}/held_{s}.csv") if int(r["ckpt"]) == CK[-1]]; return np.mean([int(r["grow16"]) for r in R])
def grid():
    pool([gjob("dev", s, ce) for ce in CES for s in P0.SEEDS["dev"]])
    sc = {ce: score("dev", ce) for ce in CES}; best = max(sc.values()); ce = sorted([c for c in CES if sc[c] == best], key=float)[0]
    json.dump({"ce": ce, "scores": sc, "rule": "max acc16 GROW held-out DEV final, media todas familias y k; empate -> menor ce"}, open(f"{O}/selected.json", "w"), indent=1); print(sc, "->", ce)
def sel(): return json.load(open(f"{O}/selected.json"))["ce"]
def confirm(): pool([gjob("confirm", s, sel()) for s in P0.SEEDS["confirm"]])
def errinfo(p):
    e = open(p).read(); h = e.split("sha256_before=")[1].split()[0]; return e.split()[0], e.count(h) == 2
def analyze(ph):
    ce = sel(); Ss = P0.SEEDS[ph]; out = []; per = collections.defaultdict(dict); qs = []; shas = []; b = f"{D(ph)}/run/ce{ce}"; fin = CK[-1]
    for s in Ss:
        q, ok = errinfo(f"{b}/stream_{s}.err"); qs.append(q); shas.append(ok)
        Hh = load(f"{b}/held_{s}.csv"); St = load(f"{b}/stream_{s}.csv"); EV = [json.loads(l) for l in open(f"{b}/events_{s}.jsonl")]
        for k in KS:
            for g, fams in GROUPS.items():
                for n in CK:
                    G = [r for r in Hh if int(r["ckpt"]) == n and int(r["k"]) == k and int(r["fam"]) in fams]
                    for c in ("grow16", "mono16", "famnet16"): per[(k, g, c, n)][s] = np.mean([int(r[c]) for r in G])
                    if n in (0, fin): per[(k, g, "lat16", n)][s] = np.mean([int(r["lat16"]) for r in G])
                    if n == 0:
                        for c in ("obs16", "latep16", "oep16", "oms16"): per[(k, g, c)][s] = np.mean([int(r[c]) for r in G])
                    if g == "nuevas_8_11":
                        per[(k, "netrate", n)][s] = np.mean([r["grow_action"] == "net" for r in G])
                        pu = [int(r["grow_expert_fam"]) == int(r["fam"]) for r in G if r["grow_action"] == "net"]; per[(k, "purity", n)][s] = np.mean(pu) if pu else np.nan
                    if g == "conocidas_0_7": per[(k, "netrate_known", n)][s] = np.mean([r["grow_action"] == "net" for r in G])
            x = np.log2(CK[1:]); y = np.array([per[(k, "nuevas_8_11", "grow16", n)][s] for n in CK[1:]]); per[(k, "slope")][s] = np.polyfit(x, y, 1)[0]
        for n in CK[1:]: per[("nexp", n)][s] = int(St[n - 1]["n_experts"])
        per[("grow_ev",)][s] = sum(e["event"] == "grow" for e in EV); per[("merge_ev",)][s] = sum(e["event"] == "merge" for e in EV)
        per[("evict_ev",)][s] = sum(e["event"] == "evict" for e in EV); per[("fail_ev",)][s] = sum(e["event"] == "grow_fail" for e in EV)
        gr = [f for e in EV if e["event"] == "grow" for f in e["fams"]]; per[("claim_new_frac",)][s] = np.mean([f >= 8 for f in gr]) if gr else np.nan
        for bb in range(0, 2400, 300):
            B = [r for r in St[bb:bb + 300] if int(r["fam"]) >= 8]
            for c in ("grow16", "mono16", "lat16", "latep16", "obs16"): per[("stream", bb, c)][s] = np.mean([int(r[c]) for r in B])
            B2 = [r for r in B if int(r["k"]) == 2]; per[("stream_k2", bb)][s] = np.mean([int(r["grow16"]) - int(r["latep16"]) for r in B2]) if B2 else np.nan
    v = lambda *key: np.array([per[key].get(s, np.nan) for s in Ss])
    def Pr(n, x):
        x = np.asarray(x, float)
        if np.isnan(x).all(): return
        bt = P0.boot(x[~np.isnan(x)]); out.append(f"| {n} | {bt[0]:+.4f} | [{bt[1]:+.4f}, {bt[2]:+.4f}] | {bt[3]} |")
    out += [f"## {ph.upper()} (16 seeds; flujo 2400 episodios S, 12 familias; held-out H 192 episodios, estado congelado en cada punto de control; ce={ce}; τ={taus()})", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    N = "nuevas_8_11"; K = "conocidas_0_7"
    for k in KS:
        out.append(f"| **k_eval = {k}** | | | |")
        Pr(f"**G1** pendiente acc16 GROW nuevas vs log2 n (k={k})", v(k, "slope"))
        Pr(f"**G2** acc16 GROW nuevas n={fin} − n=0 (k={k})", v(k, N, "grow16", fin) - v(k, N, "grow16", 0))
        Pr(f"**G4** acc16 GROW nuevas n={fin} − ORACLE_EP (k={k})", v(k, N, "grow16", fin) - v(k, N, "oep16"))
        Pr(f"**G4** acc16 GROW nuevas n={fin} − ORACLE_MS (k={k})", v(k, N, "grow16", fin) - v(k, N, "oms16"))
        Pr(f"**G6** acc16 GROW conocidas 0–7 n={fin} − OBS (k={k})", v(k, K, "grow16", fin) - v(k, K, "obs16"))
        Pr(f"**G6** acc16 GROW conocidas 0–7 n={fin} − n={CK[1]} (k={k})", v(k, K, "grow16", fin) - v(k, K, "grow16", CK[1]))
        Pr(f"**H_LAT** acc16 GROW − LAT nuevas n={fin} (k={k})", v(k, N, "grow16", fin) - v(k, N, "lat16", fin))
        Pr(f"**H_MONO** acc16 GROW − MONO nuevas n={fin} (k={k})", v(k, N, "grow16", fin) - v(k, N, "mono16", fin))
        Pr(f"  acc16 GROW − ORACLE_FAMNET nuevas n={fin} (k={k})", v(k, N, "grow16", fin) - v(k, N, "famnet16", fin))
        Pr(f"  acc16 GROW conocidas 0–7 n={fin} − LAT n={fin} (k={k})", v(k, K, "grow16", fin) - v(k, K, "lat16", fin))
        for g in GROUPS:
            for n in CK:
                for c in ("grow16", "mono16", "famnet16"): Pr(f"{g} acc16 {c[:-2].upper()} n={n} (k={k})", v(k, g, c, n))
            Pr(f"{g} acc16 LAT n={fin} (k={k})", v(k, g, "lat16", fin))
            for c in ("obs16", "latep16", "oep16", "oms16"): Pr(f"{g} acc16 {c[:-2].upper()} (k={k})", v(k, g, c))
        for n in CK: Pr(f"tasa ruta experto nuevas n={n} (k={k})", v(k, "netrate", n)); Pr(f"pureza experto (familia mayoritaria = real) nuevas n={n} (k={k})", v(k, "purity", n)); Pr(f"tasa ruta experto conocidas n={n} (k={k})", v(k, "netrate_known", n))
    for n in CK[1:]: Pr(f"nº expertos tras episodio {n}", v("nexp", n))
    for q in ("grow_ev", "merge_ev", "evict_ev", "fail_ev", "claim_new_frac"): Pr(f"eventos {q}", v(q))
    for bb in range(0, 2400, 300):
        for c in ("grow16", "mono16", "lat16", "latep16", "obs16"): Pr(f"flujo (precuencial) nuevas acc16 {c[:-2].upper()} episodios {bb}–{bb + 299}", v("stream", bb, c))
        Pr(f"flujo nuevas k=2 acc16 GROW − LAT_EP episodios {bb}–{bb + 299}", v("stream_k2", bb))
    out.append(f"\nsha256 pesos idéntico en todas: {all(shas)}; queries: {sorted(set(qs))}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"texts": texts, "ctx": ctx, "calib": calib, "grid": grid, "confirm": confirm, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
