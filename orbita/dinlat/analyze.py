"""dinlat/analyze.py PHASE OUTDIR — tablas e IC95 (bootstrap pareado por seed, 10000, semilla 12345).
G(A vs B) = log(MSE_B / MSE_A) por seed (MSE medio sobre episodios); >0 ⇒ A mejor."""
import sys, json, glob, numpy as np
DELTA = 0.05
GEOM = {"rotar", "trasladar", "escalar", "cizalla"}
def boot(x):
    x = np.asarray(x); r = np.random.default_rng(12345); i = r.integers(0, len(x), (10000, len(x)))
    m = x[i].mean(1); return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def G(R, tag, split, a, b, mask=None):
    out = []
    for r in R:
        A, B = np.array(r[tag][split][a]), np.array(r[tag][split][b])
        if mask: k = np.array([mask(f) for f in r["fam"][split]]); A, B = A[k], B[k]
        out.append(np.log(B.mean() / A.mean()))
    return out
def main(ph, d):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/res_*.json"))]
    T = {"n_seeds": len(R), "pca_var_explicada_media": float(np.mean([r["pca_var_explicada"] for r in R]))}
    rows = {
      "H1 ORB_LIBRE vs AR2 (ho)": G(R, "main", "ho", "ORB_LIBRE", "AR2"),
      "H2 ORB_LIBRE vs PERS (ho)": G(R, "main", "ho", "ORB_LIBRE", "PERS"),
      "H3 ORB_LIBRE vs ORB_SESGO (ho)": G(R, "main", "ho", "ORB_LIBRE", "ORB_SESGO"),
      "H4 ORB_LIBRE vs PERS (tf)": G(R, "main", "tf", "ORB_LIBRE", "PERS"),
      "S ORB_LIBRE vs AR2 (tf)": G(R, "main", "tf", "ORB_LIBRE", "AR2"),
      "S ORB_SESGO vs PERS (ho)": G(R, "main", "ho", "ORB_SESGO", "PERS"),
      "S ORB_SESGO vs PERS (tf)": G(R, "main", "tf", "ORB_SESGO", "PERS"),
      "S AR2 vs PERS (ho)": G(R, "main", "ho", "AR2", "PERS"),
      "S AR2 vs PERS (tf)": G(R, "main", "tf", "AR2", "PERS"),
      "S ORB_LIBRE vs AR2 (ho, geométricas)": G(R, "main", "ho", "ORB_LIBRE", "AR2", lambda f: f in GEOM),
      "S ORB_LIBRE vs AR2 (ho, no geométricas)": G(R, "main", "ho", "ORB_LIBRE", "AR2", lambda f: f not in GEOM),
      "S ORB_LIBRE vs PERS (ho, no geométricas)": G(R, "main", "ho", "ORB_LIBRE", "PERS", lambda f: f not in GEOM),
      "S h3 ORB_LIBRE vs AR2 (ho)": G(R, "main", "ho", "ORB_LIBRE_h3", "AR2_h3"),
      "S h3 ORB_LIBRE vs PERS (ho)": G(R, "main", "ho", "ORB_LIBRE_h3", "PERS_h3"),
      "C1 ORB_LIBRE vs PERS (permutado)": G(R, "C1_permutado", "ho", "ORB_LIBRE", "PERS"),
      "C2 ORB_LIBRE vs PERS (barajado)": G(R, "C2_barajado", "ho", "ORB_LIBRE", "PERS"),
      "C2 ORB_LIBRE vs AR2 (barajado)": G(R, "C2_barajado", "ho", "ORB_LIBRE", "AR2"),
    }
    rows["C1 ORB_LIBRE vs AR2 (permutado)"] = G(R, "C1_permutado", "ho", "ORB_LIBRE", "AR2")
    T["filas"] = {k: dict(zip(("media", "ic_lo", "ic_hi"), boot(v))) for k, v in rows.items()}
    f = T["filas"]
    T["criterios"] = {"H1": f["H1 ORB_LIBRE vs AR2 (ho)"]["ic_lo"] > DELTA, "H2": f["H2 ORB_LIBRE vs PERS (ho)"]["ic_lo"] > DELTA,
        "H3_no_inferior": f["H3 ORB_LIBRE vs ORB_SESGO (ho)"]["ic_lo"] > -DELTA, "H4": f["H4 ORB_LIBRE vs PERS (tf)"]["ic_lo"] > DELTA,
        "V1_sin_ganancia_espuria_C1": f["C1 ORB_LIBRE vs AR2 (permutado)"]["ic_hi"] < DELTA, "V2_sin_ganancia_espuria_C2": f["C2 ORB_LIBRE vs AR2 (barajado)"]["ic_hi"] < DELTA}
    if ph == "dev":
        T["gate_dev_a_confirm"] = bool(f["H1 ORB_LIBRE vs AR2 (ho)"]["media"] > 0 and f["H2 ORB_LIBRE vs PERS (ho)"]["ic_lo"] > 0)
    json.dump(T, open(f"{d}/analysis.json", "w"), indent=1, ensure_ascii=False)
    for k, v in f.items(): print(f"{k:45s} {v['media']:+.4f} [{v['ic_lo']:+.4f}, {v['ic_hi']:+.4f}]")
    print(json.dumps({k: v for k, v in T.items() if k != "filas"}, ensure_ascii=False))
if __name__ == "__main__": main(*sys.argv[1:3])
