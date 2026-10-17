"""Pruebas causales CDT → ÓRBITA (docs/preregistro_cdt_causal.md).  causal.py eval PH | analyze PH"""
import sys, os, csv, collections, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P0
O = "/workspace/orbita-data/llm_cdt2"; E2E = "/workspace/orbita-data/llm_e2e"; ART = f"{P0.ROOT}/artifacts/orbita/llm_cdt2"
CTRL = {"OBS": ["BASE", "OBS_CORRUPT", "OBS_SHUF", "ORACLE_ORB", "ORACLE_LS"], "CORR": ["BASE", "CORR_CORRUPT", "CORR_SHUF"],
        "OBSCORR": ["BASE", "OBSCORR_CORRUPT", "OBSCORR_SHUF", "ORACLE_ORB", "ORACLE_LS"]}
def ev(ph):
    os.makedirs(f"{ART}/{ph}", exist_ok=True)
    for s in P0.SEEDS[ph]:
        p = subprocess.run([P0.B, "cdt-causal", "--seed", s, "--weights", f"/workspace/orbita-data/{ph}/w_{s}.json", "--ctx", f"{E2E}/{ph}/ctx_{s}.csv"], capture_output=True, check=True)
        open(f"{ART}/{ph}/causal_{s}.csv", "wb").write(p.stdout); open(f"{ART}/{ph}/log_{s}.txt", "wb").write(p.stderr)
    print("eval ok", ph)
def analyze(ph):
    S = P0.SEEDS[ph]; per = collections.defaultdict(dict); out = []
    for s in S:
        R = list(csv.DictReader(open(f"{ART}/{ph}/causal_{s}.csv")))
        for c in sorted({r["cond"] for r in R}):
            G = [r for r in R if r["cond"] == c]
            for h in (1, 8, 16): per[(c, h)][s] = np.mean([int(r[f"ok{h}"]) for r in G])
            per[(c, "au")][s] = P0.auroc([float(r["score"]) for r in G], [r["ok16"] == "0" for r in G])
    conds = sorted({k[0] for k in per})
    v = lambda c, m: np.array([per[(c, m)][s] for s in S]); Pr = lambda n, x: out.append(f"| {n} | {x[0]:+.4f} | [{x[1]:+.4f}, {x[2]:+.4f}] | {x[3]} |")
    out += [f"## {ph.upper()} (16 seeds; 16 ejemplos TEST iterables × 6 textos E2E_R; K_OBS=2, α=0,5)", "", "| cantidad | media | IC95 | n |", "|---|---|---|---|"]
    for c in conds:
        for h in (1, 8, 16): Pr(f"acc h{h} {c}", P0.boot(v(c, h)))
        Pr(f"AUROC score→fallo h16 {c}", P0.boot(v(c, "au")))
    for fam, ctl in CTRL.items():
        for via in ("DIR", "SLEEP"):
            V = f"{fam}_{via}"
            for m, lab in ((16, "acc16"), ("au", "AUROC")):
                mx = np.max(np.stack([v(c, m) for c in ctl]), 0); nm = f"**H-{V} {lab}**" if m == 16 else f"H-{V} {lab}"
                Pr(f"{nm} {V} − max({', '.join(ctl)})", P0.boot(v(V, m) - mx))
                for c in ctl: Pr(f"  {V} − {c} ({lab})", P0.boot(v(V, m) - v(c, m)))
        Pr(f"**H-SD-{fam}** acc16 {fam}_SLEEP − {fam}_DIR", P0.boot(v(f"{fam}_SLEEP", 16) - v(f"{fam}_DIR", 16)))
    q, sz = 0, set()
    for s in S:
        t = open(f"{ART}/{ph}/log_{s}.txt").read(); q += int(t.split("queries=")[1].split()[0]); sz.add(t.split("sizes=")[1].split()[0])
    out.append(f"\nQueries a TEST auditado (cdt-causal): {q}\nTamaños de señal (números antes->después): {sorted(sz)}")
    open(f"{ART}/summary_{ph}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
if __name__ == "__main__":
    {"eval": ev, "analyze": analyze}[sys.argv[1]](*sys.argv[2:])
