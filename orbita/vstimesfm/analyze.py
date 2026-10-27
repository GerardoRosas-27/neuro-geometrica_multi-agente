"""vstimesfm/analyze.py PHASE DINLAT_DIR TFM_DIR — mismo G e IC95 que dinlat/analyze.py (bootstrap pareado por seed,
10000 réplicas, semilla 12345). G(A vs B)=log(MSE_B/MSE_A) por seed; >0 ⇒ A mejor."""
import sys, json, glob, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dinlat"))
from analyze import boot, DELTA, GEOM
def merge(dd, td):
    R = []
    for f in sorted(glob.glob(f"{td}/tfm_*.json")):
        t = json.load(open(f)); r = json.load(open(f"{dd}/res_{t['seed']}.json"))
        for tag in t:
            if tag == "seed": continue
            for sp in t[tag]: r[tag][sp].update(t[tag][sp])
        R.append(r)
    return R
def G(R, tag, split, a, b, mask=None):
    out = []
    for r in R:
        A, B = np.array(r[tag][split][a]), np.array(r[tag][split][b])
        if mask: k = np.array([mask(f) for f in r["fam"][split]]); A, B = A[k], B[k]
        out.append(np.log(B.mean() / A.mean()))
    return out
geo, ngeo = (lambda f: f in GEOM), (lambda f: f not in GEOM)
def main(ph, dd, td):
    R = merge(dd, td); T = {"n_seeds": len(R)}; rows = {}
    rows["H1 ORB_SESGO vs TIMESFM (ho)"] = G(R, "main", "ho", "ORB_SESGO", "TIMESFM")
    rows["H1b ORB_LIBRE vs TIMESFM (ho)"] = G(R, "main", "ho", "ORB_LIBRE", "TIMESFM")
    rows["H2 TIMESFM vs PERS (ho)"] = G(R, "main", "ho", "TIMESFM", "PERS")
    rows["H3 ORB_LIBRE vs TIMESFM (tf)"] = G(R, "main", "tf", "ORB_LIBRE", "TIMESFM")
    rows["S ORB_SESGO vs TIMESFM (tf)"] = G(R, "main", "tf", "ORB_SESGO", "TIMESFM")
    for sp in ("ho", "tf"):
        rows[f"S TIMESFM vs AR2 ({sp})"] = G(R, "main", sp, "TIMESFM", "AR2")
        rows[f"S TIMESFM vs VAR2_OLS ({sp})"] = G(R, "main", sp, "TIMESFM", "VAR2_OLS")
        rows[f"S AR2 vs VAR2_OLS ({sp})"] = G(R, "main", sp, "AR2", "VAR2_OLS")
    rows["S TIMESFM vs PERS (tf)"] = G(R, "main", "tf", "TIMESFM", "PERS")
    for nm, m in (("geométricas", geo), ("no geométricas", ngeo)):
        for a in ("ORB_SESGO", "ORB_LIBRE", "AR2", "PERS"):
            rows[f"S {a} vs TIMESFM (ho, {nm})"] = G(R, "main", "ho", a, "TIMESFM", m)
    for a in ("ORB_SESGO", "ORB_LIBRE", "AR2", "PERS"):
        rows[f"S h3 {a} vs TIMESFM (ho)"] = G(R, "main", "ho", f"{a}_h3", "TIMESFM_h3")
    for c in ("C1_permutado", "C2_barajado"):
        rows[f"C TIMESFM vs PERS ({c})"] = G(R, c, "ho", "TIMESFM", "PERS")
        rows[f"C ORB_LIBRE vs TIMESFM ({c})"] = G(R, c, "ho", "ORB_LIBRE", "TIMESFM")
    T["filas"] = {k: dict(zip(("media", "ic_lo", "ic_hi"), boot(v))) for k, v in rows.items()}; f = T["filas"]
    T["criterios"] = {"H1": f["H1 ORB_SESGO vs TIMESFM (ho)"]["ic_lo"] > DELTA, "H1b": f["H1b ORB_LIBRE vs TIMESFM (ho)"]["ic_lo"] > DELTA,
        "H2": f["H2 TIMESFM vs PERS (ho)"]["ic_lo"] > DELTA, "H3": f["H3 ORB_LIBRE vs TIMESFM (tf)"]["ic_lo"] > DELTA,
        "TIMESFM_mejor_que_ORB_SESGO": f["H1 ORB_SESGO vs TIMESFM (ho)"]["ic_hi"] < -DELTA,
        "TIMESFM_mejor_que_ORB_LIBRE": f["H1b ORB_LIBRE vs TIMESFM (ho)"]["ic_hi"] < -DELTA}
    if ph == "dev":
        T["gate_dev_a_confirm"] = bool(f["H1 ORB_SESGO vs TIMESFM (ho)"]["media"] > 0 and f["H2 TIMESFM vs PERS (ho)"]["ic_lo"] > 0)
    json.dump(T, open(f"{td}/analysis.json", "w"), indent=1, ensure_ascii=False)
    for k, v in f.items(): print(f"{k:50s} {v['media']:+.4f} [{v['ic_lo']:+.4f}, {v['ic_hi']:+.4f}]")
    print(json.dumps({k: v for k, v in T.items() if k != "filas"}, ensure_ascii=False))
if __name__ == "__main__": main(*sys.argv[1:4])
