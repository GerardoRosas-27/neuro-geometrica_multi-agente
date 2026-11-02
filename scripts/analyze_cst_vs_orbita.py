#!/usr/bin/env python3
"""Análisis CST vs ÓRBITA-eq (docs/cst_vs_orbita/PREREG.md + ENMIENDA_1.md).
Bootstrap pareado 10000, semilla 12345, IC95 percentil, δ=0.05.
  --stopcheck dev : regla de paro (NaN o colapso en >2/16 semillas → exit 1)
  --md            : salida markdown"""
import json, glob, sys, os
import numpy as np
A = "artifacts/cst_vs_orbita"; DELTA = 0.05; NB = 10000

def load(ph):
    return {os.path.basename(f)[8:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{A}/{ph}/metrics_0x*.json"))}

def boot(d):
    d = np.asarray(d, float); rng = np.random.default_rng(12345)
    m = d[rng.integers(0, len(d), (NB, len(d)))].mean(1)
    return {"n": len(d), "mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}

def get(r, arm, key):
    a = r["arms"].get(arm)
    if a is None: return None
    if key == "e33_k4_16": return None if "e33_k4" not in a else a["e33_k4"]["acc"][a["e33_k4"]["deltas"].index(16)]
    if key == "e33_ctx_16": return a["e33_ctx"]["acc"][a["e33_ctx"]["horizons"].index(16)] if "e33_ctx" in a else None
    return a.get(key)

def pair(R, x, y, key):
    v = [(get(r, x, key), get(r, y, key)) for r in R.values()]
    v = [(a, b) for a, b in v if a is not None and b is not None]
    return boot([a - b for a, b in v]) if v else None

def phase(ph):
    R = load(ph); out = {"n_seeds": len(R)}
    if not R: return out
    out["H1_new_med_log_cst_minus_eqlat"] = h1 = pair(R, "cst", "orbita_eq_lat", "new_med_log")
    out["H2_e33k4_acc16_cst_minus_eqlat"] = h2 = pair(R, "cst", "orbita_eq_lat", "e33_k4_16")
    out["H3_known_acc16_cst_minus_eqlat"] = h3 = pair(R, "cst", "orbita_eq_lat", "known_acc16")
    out["H1_pass"] = h1["hi"] < 0; out["H2_pass"] = h2["lo"] > 0; out["H3_pass"] = h3["lo"] > -DELTA
    inf = {}
    for x, y in [("cst", "orbita_eq"), ("cst", "orbita_v1"), ("cst", "orbita_v1_lat"), ("cst", "cst_no_fast"), ("cst", "persist"), ("orbita_eq_lat", "orbita_v1_lat")]:
        for k in ["new_med_log", "known_med_log", "known_acc16", "new_acc16", "e33_k4_16", "onestep_ctx_acc"]:
            p = pair(R, x, y, k)
            if p: inf[f"{x}-{y}:{k}"] = p
    inf["cst_prior-orbita_eq:e33_ctx_16"] = pair(R, "cst_prior", "orbita_eq", "e33_ctx_16")
    out["H4_informativo"] = inf
    arms = sorted({a for r in R.values() for a in r["arms"]})
    out["medias"] = {a: {k: float(np.mean([get(r, a, k) for r in R.values() if get(r, a, k) is not None]))
                         for k in ["new_med_log", "known_med_log", "known_acc16", "new_acc16", "e33_k4_16", "e33_ctx_16", "onestep_ctx_acc"]
                         if any(get(r, a, k) is not None for r in R.values())} for a in arms}
    out["cst_train_samples"] = sorted({r["arms"]["cst"]["train_samples"] for r in R.values() if "cst" in r["arms"]})
    out["nan_or_collapse"] = [s for s, r in R.items() for a in ("cst", "cst_no_fast") if a in r["arms"] and (r["arms"][a]["nan"] or r["arms"][a]["collapse"]["collapsed"])]
    return out

def main():
    if "--stopcheck" in sys.argv:
        o = phase(sys.argv[sys.argv.index("--stopcheck") + 1]); bad = len(set(o.get("nan_or_collapse", [])))
        o["stop"] = bad > 2 or o["n_seeds"] < 16; print(json.dumps(o, indent=1)); sys.exit(1 if o["stop"] else 0)
    res = {ph: phase(ph) for ph in ("dev", "confirm")}
    ok = all(res[p].get(k) for p in res for k in ("H1_pass", "H3_pass"))
    res["veredicto"] = "CST gana (H1∧H3 en DEV y CONFIRM)" if ok else "CST NO gana (H1∧H3 no se cumple en ambas fases)"
    if "--md" not in sys.argv: print(json.dumps(res, indent=1)); return
    print("# CST vs ÓRBITA-eq — resultados (generado por scripts/analyze_cst_vs_orbita.py)\n")
    print(f"**Veredicto:** {res['veredicto']}\n")
    for ph in ("dev", "confirm"):
        o = res[ph]; print(f"## {ph.upper()} (n={o['n_seeds']})\n")
        if not o.get("n_seeds"): continue
        for h in ("H1_new_med_log_cst_minus_eqlat", "H2_e33k4_acc16_cst_minus_eqlat", "H3_known_acc16_cst_minus_eqlat"):
            b = o[h]; print(f"- {h}: {b['mean']:+.4f} IC95 [{b['lo']:+.4f}, {b['hi']:+.4f}] → {'PASA' if o[h[:2]+'_pass'] else 'NO pasa'}")
        print("\n| brazo | new_med_log | known_med_log | known_acc16 | new_acc16 | E33-k4 acc16 | E33-ctx acc16 | 1-paso acc |\n|---|---|---|---|---|---|---|---|")
        for a, m in o["medias"].items():
            print(f"| {a} | " + " | ".join(f"{m[k]:.4f}" if k in m else "—" for k in ["new_med_log", "known_med_log", "known_acc16", "new_acc16", "e33_k4_16", "e33_ctx_16", "onestep_ctx_acc"]) + " |")
        print(f"\nNaN/colapso: {o['nan_or_collapse']}; muestras CST: {o['cst_train_samples']}\n")

main()
