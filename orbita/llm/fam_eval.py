"""Otras familias (docs/preregistro_otras_familias.md).  fam_eval.py texts | run PH | analyze PH"""
import sys, os, csv, collections, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0, numnorm as NN, e2e as E
from make_texts import FAM, ART as ARTI
O = "/workspace/orbita-data/demo_fam"; ART = f"{P0.ROOT}/artifacts/orbita/otras_familias"
NEWN = {"es": (["un", "una"], ["giro ondulado", "espiral"]), "en": (["a", "a"], ["wavy turn", "spiral"]), "fr": (["un", "une"], ["virage ondulé", "spirale"])}
GROUPS = {"conocidas_1_2": [1, 2], "v4_no_iterables": [0, 3, 4, 5, 6, 7], "nuevas_8_9": [8, 9]}
def name(lang, f): return (ARTI[lang][f], FAM[lang][f]) if f < 8 else (NEWN[lang][0][f - 8], NEWN[lang][1][f - 8])
def texts():
    os.makedirs(O, exist_ok=True); st = subprocess.check_output([P0.B, "fam-set"]).decode(); open(f"{O}/set.csv", "w").write(st); out = []
    for r in csv.DictReader(st.splitlines()):
        f, p1, p2 = int(r["fam"]), float(r["p1"]), float(r["p2"]); k = int(r["idx"]) % 2
        for li, lang in enumerate(E.LANGS):
            fa, fb = E.fmts(r["idx"], k, li); art, nm = name(lang, f)
            out.append([r["idx"], str(f), r["p1"], r["p2"], lang, str(6 + k), f"{NN.FMT_NAMES[fa]}+{NN.FMT_NAMES[fb]}", E.T[lang][k].format(art=art, f=nm, a=NN.render(p1, lang, fa), b=NN.render(p2, lang, fb))])
    open(f"{O}/texts.tsv", "w").write("".join("\t".join(x) + "\n" for x in out)); open(f"{O}/texts.txt", "w").write("".join(x[7] + "\n" for x in out)); print(len(out), "textos")
def run(ph):
    os.makedirs(f"{O}/{ph}", exist_ok=True); rows = P0.tsv(f"{O}/texts.tsv")
    for s in P0.SEEDS[ph]:
        dn = f"{O}/{ph}/{s}.done"
        if os.path.exists(dn): continue
        C = E.d2_predict(f"{O}/texts.tok", ph, s, list(range(len(rows)))); L = ["idx,cond,text_id," + ",".join(f"c{i}" for i in range(10))]; nf = 0
        for j, r in enumerate(rows):
            c, ok = E.build_c(C[j], r[7]); nf += not ok; L.append(f"{r[0]},E2E,{r[4]}-{r[5]}," + ",".join(repr(float(x)) for x in c))
        open(f"{O}/{ph}/ctx_{s}.csv", "w").write("\n".join(L) + "\n")
        p = subprocess.run([P0.B, "fam-eval", "--weights", f"/workspace/orbita-data/{ph}/w_{s}.json", "--set", f"{O}/set.csv", "--ctx", f"{O}/{ph}/ctx_{s}.csv"], capture_output=True, check=True)
        open(f"{O}/{ph}/eval_{s}.csv", "wb").write(p.stdout); open(f"{O}/{ph}/q_{s}.txt", "w").write(p.stderr.decode() + f" parse_fail={nf}\n")
        open(dn, "w").write("ok\n"); print(ph, s, "ok", flush=True)
def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []
    for s in S:
        R = list(csv.DictReader(open(f"{O}/{ph}/eval_{s}.csv")))
        for gname, fams in list(GROUPS.items()) + [(f"fam{f}", [f]) for f in range(10)]:
            G = [r for r in R if int(r["fam"]) in fams]
            for c in ("base", "obs1", "obs2", "ls1", "ls2", "true"):
                for h in (8, 16):
                    vals = [int(r[f"{c}_ok{h}"]) for r in G if r[f"{c}_ok{h}"] != ""]
                    if vals: per[(gname, c, h)][s] = np.mean(vals)
            per[(gname, "reid1_known")][s] = np.mean([int(r["obs1_fam"]) == int(r["fam"]) for r in G]) if max(fams) < 8 else np.nan
    v = lambda *k: np.array([per[k].get(s, np.nan) for s in S])
    def Pr(n, x):
        x = np.asarray(x, float)
        if np.isnan(x).all(): return
        b = P0.boot(x[~np.isnan(x)]); out.append(f"| {n} | {b[0]:+.4f} | [{b[1]:+.4f}, {b[2]:+.4f}] | {b[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos × 3 textos por familia)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for g in list(GROUPS) + [f"fam{f}" for f in range(10)]:
        for c in ("base", "obs1", "obs2", "ls1", "ls2", "true"):
            Pr(f"{g} acc16 {c.upper()}", v(g, c, 16)); Pr(f"{g} acc8 {c.upper()}", v(g, c, 8))
        Pr(f"{g} familia re-identificada = verdadera (OBS1)", v(g, "reid1_known"))
    for k in (1, 2):
        for tag, g in (("a", "nuevas_8_9"), ("b", "v4_no_iterables")):
            Pr(f"**HF{k}{tag}** acc16 OBS{k} − BASE ({g})", v(g, f"obs{k}", 16) - v(g, "base", 16))
            Pr(f"  acc16 OBS{k} − ORACLE_LS{k} ({g})", v(g, f"obs{k}", 16) - v(g, f"ls{k}", 16))
    for c in ("base", "obs1", "obs2"):
        for g in ("nuevas_8_9", "v4_no_iterables"): Pr(f"brecha acc16 conocidas − {g} ({c.upper()})", v("conocidas_1_2", c, 16) - v(g, c, 16))
    q = [open(f"{O}/{ph}/q_{s}.txt").read().strip() for s in S]; out.append(f"\nQueries v4 / parse_fail por seed: {q}")
    os.makedirs(ART, exist_ok=True); open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"texts": texts, "run": run, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
