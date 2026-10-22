"""Comparación pareada ÓRBITA vs v4 §16 (E3_gated_trusted / R_E3_V0_pure, control M3) por seed.
Uso: python3 orbita/analyze_orbita.py dev|confirm  → imprime tabla markdown y escribe artifacts/orbita/<fase>/paired_stats.md"""
import csv, glob, random, sys, statistics as st, os
ph = sys.argv[1]
V4 = f"artifacts/v4/distill_e/{ph}/metrics.csv"
def rows(paths):
    for p in paths:
        with open(p) as f:
            yield from csv.DictReader(f)
v4 = list(rows([V4])); orb = list(rows(sorted(glob.glob(f"artifacts/orbita/{ph}/metrics_0x*.csv"))))
def get(rs, cond, field="accuracy"):
    return {r["seed"]: float(r[field]) for r in rs if r["condition"] == cond}
def gate(rs, part):
    return {r["seed"]: r["status"] == "PASS" for r in rs if r["condition"] == "GATE" and r["partition"] == part}
def region(rs, part):
    import re
    return {r["seed"]: int(re.search(r"region=(\d+)", r["notes"]).group(1)) for r in rs if r["condition"] == "GATE" and r["partition"] == part}
arms = {
  "v4_E3": dict(one=get(v4, "E3_gated_trusted"), rel=get(v4, "E3_gated_trusted", "stability"), h=lambda t: get(v4, f"R_E3_V0_pure_h{t}"), g=gate(v4, "R_E3_V0_pure"), reg=region(v4, "R_E3_V0_pure")),
  "v4_M3_mlp": dict(one=get(v4, "M3_mlp_gated_trusted"), rel=get(v4, "M3_mlp_gated_trusted", "stability"), h=lambda t: get(v4, f"R_M3_V0_h{t}"), g=gate(v4, "R_M3_V0"), reg=region(v4, "R_M3_V0")),
  "ORBITA": dict(one=get(orb, "ORBITA"), rel=get(orb, "ORBITA", "stability"), h=lambda t: get(orb, f"R_ORBITA_V0_h{t}"), g=gate(orb, "R_ORBITA_V0"), reg=region(orb, "R_ORBITA_V0")),
}
seeds = sorted(set(arms["ORBITA"]["one"]) & set(arms["v4_E3"]["one"]))
def boot(d, n=10000, seed=12345):
    rng = random.Random(seed); m = []
    for _ in range(n):
        s = [d[rng.randrange(len(d))] for _ in d]; m.append(sum(s) / len(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n) - 1]
H = [1, 2, 4, 8, 16, 32, 64]
out = [f"# ÓRBITA vs v4 — fase {ph} (n={len(seeds)} seeds pareadas)\n",
       "| Brazo | params | acc 1 paso | rel_err₁ | gate E33 (región≥8, limpio, proceso nuevo) | región mediana | " + " | ".join(f"acc h{t}" for t in H) + " |",
       "|---|---|---|---|---|---|" + "---|" * len(H)]
params = {"v4_E3": 2590, "v4_M3_mlp": 2590, "ORBITA": int(next(r["params"] for r in orb if r["condition"] == "ORBITA")) if orb else 0}
for a, A in arms.items():
    hs = {t: A["h"](t) for t in H}
    out.append(f"| {a} | {params[a]} | {st.mean(A['one'][s] for s in seeds):.4f} | {st.mean(A['rel'][s] for s in seeds):.4f} | {sum(A['g'][s] for s in seeds)}/{len(seeds)} | {st.median(A['reg'][s] for s in seeds)} | " +
               " | ".join(f"{st.mean(hs[t][s] for s in seeds):.3f}" for t in H) + " |")
out.append("\n## Diferencias pareadas (IC95 bootstrap, 10 000 remuestreos, semilla 12345)\n")
out.append("| Comparación | métrica | media | IC95 | seeds a favor |\n|---|---|---|---|---|")
for b in ["v4_E3", "v4_M3_mlp"]:
    for name, fa, fb in [("acc h16", arms["ORBITA"]["h"](16), arms[b]["h"](16)), ("región", arms["ORBITA"]["reg"], arms[b]["reg"]), ("acc h8", arms["ORBITA"]["h"](8), arms[b]["h"](8)), ("acc 1 paso", arms["ORBITA"]["one"], arms[b]["one"])]:
        d = [fa[s] - fb[s] for s in seeds]
        lo, hi = boot(d)
        out.append(f"| ORBITA − {b} | {name} | {st.mean(d):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {sum(x > 0 for x in d)}/{len(d)} |")
txt = "\n".join(out); print(txt)
open(f"artifacts/orbita/{ph}/paired_stats.md", "w").write(txt + "\n")
