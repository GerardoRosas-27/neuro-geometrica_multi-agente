"""hibrido/analyze.py PHASE OUTDIR — mismo bootstrap que dinlat (pareado por seed, 10000, semilla 12345). δ=0.05.
G(A vs B) = log(MSE_B/MSE_A); >0 ⇒ A mejor. BEST_IND = min por seed de MSE(AR2), MSE(ORB_LIBRE) (elegido a posteriori ⇒ conservador)."""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dinlat"))
from analyze import boot, GEOM, DELTA
PRIM = "HIB_GATE"                                  # brazo primario fijado antes de DEV (ver PREREG.md)
def mse(r, tag, sp, k, mask=None):
    if k == "BEST_IND": return min(mse(r, tag, sp, "AR2", mask), mse(r, tag, sp, "ORB_LIBRE", mask))
    if k == "ORACLE_FAM":                          # techo: por familia, mejor experto con errores del propio split
        A, B = np.array(r[tag][sp]["AR2"]), np.array(r[tag][sp]["ORB_LIBRE"]); f = np.array(r["fam"][sp]); e = np.empty_like(A)
        for u in set(f): k_ = f == u; e[k_] = A[k_] if A[k_].mean() <= B[k_].mean() else B[k_]
        x = e
    else: x = np.array(r[tag][sp][k])
    if mask: x = x[np.array([mask(u) for u in r["fam"][sp]])]
    return float(x.mean())
def G(R, tag, sp, a, b, mask=None): return [np.log(mse(r, tag, sp, b, mask) / mse(r, tag, sp, a, mask)) for r in R]
def main(ph, d):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/res_*.json"))]; geo = lambda f: f in GEOM; ng = lambda f: f not in GEOM
    rows = {
      "H1 HIB_GATE vs BEST_IND (ho)": G(R, "main", "ho", PRIM, "BEST_IND"),
      "H2 HIB_GATE vs AR2 (ho)": G(R, "main", "ho", PRIM, "AR2"),
      "H3 HIB_GATE vs AR2 (tf)": G(R, "main", "tf", PRIM, "AR2"),
      "H4 HIB_GATE vs MIX50 (ho)": G(R, "main", "ho", PRIM, "MIX50"),
      "S HIB_GATE vs ORB_LIBRE (ho)": G(R, "main", "ho", PRIM, "ORB_LIBRE"),
      "S HIB_GATE vs ORB_LIBRE (tf)": G(R, "main", "tf", PRIM, "ORB_LIBRE"),
      "S HIB_GATE vs BEST_IND (tf)": G(R, "main", "tf", PRIM, "BEST_IND"),
      "S HIB_GATE vs AR2 (ho, geométricas)": G(R, "main", "ho", PRIM, "AR2", geo),
      "S HIB_GATE vs AR2 (ho, no geométricas)": G(R, "main", "ho", PRIM, "AR2", ng),
      "S HIB_GATE vs ORB_LIBRE (ho, geométricas)": G(R, "main", "ho", PRIM, "ORB_LIBRE", geo),
      "S HIB_GATE vs ORB_LIBRE (ho, no geométricas)": G(R, "main", "ho", PRIM, "ORB_LIBRE", ng),
      "S HIB_HEDGE vs AR2 (ho)": G(R, "main", "ho", "HIB_HEDGE", "AR2"),
      "S HIB_RES vs AR2 (ho)": G(R, "main", "ho", "HIB_RES", "AR2"),
      "S HIB_RES vs AR2 (tf)": G(R, "main", "tf", "HIB_RES", "AR2"),
      "S MIX50 vs AR2 (ho)": G(R, "main", "ho", "MIX50", "AR2"),
      "S HIB_GATE_S vs ORB_SESGO (ho)": G(R, "main", "ho", "HIB_GATE_S", "ORB_SESGO"),
      "S HIB_GATE_S vs ORB_SESGO (tf)": G(R, "main", "tf", "HIB_GATE_S", "ORB_SESGO"),
      "S ORACLE_FAM vs HIB_GATE (ho) [brecha al techo]": G(R, "main", "ho", "ORACLE_FAM", PRIM),
      "S ORB_LIBRE vs AR2 (ho) [réplica dinlat]": G(R, "main", "ho", "ORB_LIBRE", "AR2"),
      "C1 HIB_GATE vs AR2 (permutado)": G(R, "C1_permutado", "ho", PRIM, "AR2"),
      "C2 HIB_GATE vs AR2 (barajado)": G(R, "C2_barajado", "ho", PRIM, "AR2"),
      "C2 HIB_RES vs AR2 (barajado)": G(R, "C2_barajado", "ho", "HIB_RES", "AR2"),
    }
    f = {k: dict(zip(("media", "ic_lo", "ic_hi"), boot(v))) for k, v in rows.items()}
    T = {"n_seeds": len(R), "filas": f, "eta_hedge": [r["main"]["_eta_hedge"] for r in R], "criterios": {
      "H1": f["H1 HIB_GATE vs BEST_IND (ho)"]["ic_lo"] > DELTA, "H2": f["H2 HIB_GATE vs AR2 (ho)"]["ic_lo"] > DELTA,
      "H3": f["H3 HIB_GATE vs AR2 (tf)"]["ic_lo"] > DELTA, "H4": f["H4 HIB_GATE vs MIX50 (ho)"]["ic_lo"] > DELTA,
      "V1_sin_ganancia_espuria_C1": f["C1 HIB_GATE vs AR2 (permutado)"]["ic_hi"] < DELTA,
      "V2_sin_ganancia_espuria_C2": f["C2 HIB_GATE vs AR2 (barajado)"]["ic_hi"] < DELTA}}
    if ph == "dev": T["gate_dev_a_confirm"] = bool(f["H2 HIB_GATE vs AR2 (ho)"]["media"] > 0)
    json.dump(T, open(f"{d}/analysis.json", "w"), indent=1, ensure_ascii=False)
    for k, v in f.items(): print(f"{k:52s} {v['media']:+.4f} [{v['ic_lo']:+.4f}, {v['ic_hi']:+.4f}]")
    print(json.dumps({k: v for k, v in T.items() if k != "filas"}, ensure_ascii=False))
if __name__ == "__main__": main(*sys.argv[1:3])
