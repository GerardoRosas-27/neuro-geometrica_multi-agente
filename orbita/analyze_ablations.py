"""Análisis de ablaciones (docs/ablaciones_orbita.md). Uso:
  python3 orbita/analyze_ablations.py D            → curva E(h) ÓRBITA vs v4 E3 (extrapolación)
  python3 orbita/analyze_ablations.py ABC dev|confirm → ÓRBITA vs A/B/C vs E3/M3"""
import csv, glob, random, re, statistics as st, sys
def boot(d, n=10000, seed=12345):
    rng = random.Random(seed); m = []
    for _ in range(n):
        s = [d[rng.randrange(len(d))] for _ in d]; m.append(sum(s) / len(s))
    m.sort(); return m[int(0.025 * n)], m[int(0.975 * n) - 1]
def fmt(d):
    lo, hi = boot(d); return f"{st.mean(d):+.4f} | {st.median(d):+.4f} | {st.pstdev(d):.4f} | [{lo:+.4f}, {hi:+.4f}] | {sum(x>0 for x in d)}/{sum(x<0 for x in d)}"
mode = sys.argv[1]
if mode == "D":
    out = ["# Experimento D — extrapolación (ÓRBITA vs v4 E3, TEST iterable rotate+reflect, free-run)\n"]
    for ph in ["dev", "confirm"]:
        R = list(csv.DictReader(open(f"artifacts/orbita/ablations/D_extrap/extrap_{ph}.csv")))
        hs = sorted({int(r["h"]) for r in R})
        g = lambda mdl, h, k: {r["seed"]: float(r[k]) for r in R if r["model"] == mdl and int(r["h"]) == h}
        out.append(f"\n## {ph.upper()} (n=16)\n\n| h | acc ÓRBITA | acc E3 | acc teacher | rel_err med ÓRBITA | rel_err med E3 | norma ÓRBITA | norma E3 | colapso ÓRBITA | colapso E3 | div. teacher ÓRBITA | div. teacher E3 | Δacc ÓRBITA−E3: media | mediana | sd | IC95 | seeds +/− |\n|" + "---|" * 16)
        for h in hs:
            a, b = g("ORBITA", h, "acc"), g("v4_E3", h, "acc"); s = sorted(a)
            m = lambda d: st.mean(d[x] for x in s)
            out.append(f"| {h} | {m(a):.3f} | {m(b):.3f} | {m(g('ORBITA',h,'teacher_acc')):.3f} | {m(g('ORBITA',h,'rel_err_median')):.3f} | {m(g('v4_E3',h,'rel_err_median')):.3f} | {m(g('ORBITA',h,'norm_ratio_median')):.3f} | {m(g('v4_E3',h,'norm_ratio_median')):.3f} | {m(g('ORBITA',h,'collapse_rate')):.3f} | {m(g('v4_E3',h,'collapse_rate')):.3f} | {m(g('ORBITA',h,'teacher_div_median')):.3f} | {m(g('v4_E3',h,'teacher_div_median')):.3f} | " + fmt([a[x] - b[x] for x in s]) + " |")
        # error relativo pareado en h no vistos
        for h in [128, 256]:
            a, b = g("ORBITA", h, "rel_err_median"), g("v4_E3", h, "rel_err_median"); s = sorted(a)
            out.append(f"\nrel_err mediano h{h}, ÓRBITA − E3 (negativo = ÓRBITA mejor): media | mediana | sd | IC95 | seeds +/−: " + fmt([a[x] - b[x] for x in s]))
    print("\n".join(out)); open("artifacts/orbita/ablations/D_extrap/summary.md", "w").write("\n".join(out) + "\n")
else:
    ph = sys.argv[2]
    def rows(ps):
        for p in ps: yield from csv.DictReader(open(p))
    v4 = list(rows([f"artifacts/v4/distill_e/{ph}/metrics.csv"]))
    def arm(rs, one, roll):
        d = {}
        for r in rs:
            s = r["seed"]; d.setdefault(s, {})
            if r["condition"] == one: d[s]["acc1"] = float(r["accuracy"]); d[s]["rel1"] = float(r["stability"]); d[s]["clean"] = r["status"] == "CLEAN"; d[s]["params"] = int(r["params"])
            m = re.match(rf"{roll}_h(\d+)$", r["condition"])
            if m: d[s][f"h{m.group(1)}"] = float(r["accuracy"])
            if r["condition"] == "GATE" and r["partition"] == roll:
                d[s]["gate"] = r["status"] == "PASS"; d[s]["reg"] = int(re.search(r"region=(\d+)", r["notes"]).group(1))
        return {k: v for k, v in d.items() if "acc1" in v}
    A = {"ÓRBITA": arm(rows(sorted(glob.glob(f"artifacts/orbita/{ph}/metrics_0x*.csv"))), "ORBITA", "R_ORBITA_V0")}
    for v in ["A", "B", "C"]:
        fs = sorted(glob.glob(f"artifacts/orbita/ablations/{v}/{ph}/metrics_0x*.csv"))
        if fs: A[{"A":"A corrector-only","B":"B1 K libre","C":"C sin corrector"}[v]] = arm(rows(fs), f"ABL_{v}", f"R_ABL_{v}_V0")
    A["v4 E3"] = arm(v4, "E3_gated_trusted", "R_E3_V0_pure"); A["MLP M3"] = arm(v4, "M3_mlp_gated_trusted", "R_M3_V0")
    H = [1, 2, 4, 8, 16, 32, 64]
    out = [f"# Ablaciones A/B1/C — {ph.upper()}\n", "| Variante | n | params | acc1 | rel1 | gate E33 | región med | " + " | ".join(f"h{t}" for t in H) + " | limpio |", "|" + "---|" * (8 + len(H))]
    for n, d in A.items():
        s = sorted(d); mm = lambda k: st.mean(d[x][k] for x in s)
        out.append(f"| {n} | {len(s)} | {d[s[0]]['params']} | {mm('acc1'):.4f} | {mm('rel1'):.4f} | {sum(d[x]['gate'] for x in s)}/{len(s)} | {st.median(d[x]['reg'] for x in s)} | " + " | ".join(f"{mm(f'h{t}'):.3f}" for t in H) + f" | {sum(d[x]['clean'] for x in s)}/{len(s)} |")
    out.append("\n## Diferencias pareadas ÓRBITA − variante (IC95 bootstrap 10 000, semilla 12345)\n\n| Comparación | métrica | media | mediana | sd | IC95 | seeds ÓRBITA mejor / peor |\n|---|---|---|---|---|---|---|")
    o = A["ÓRBITA"]
    for n, d in A.items():
        if n == "ÓRBITA": continue
        s = sorted(set(o) & set(d))
        for k in ["h8", "h16", "reg", "acc1"]:
            out.append(f"| ÓRBITA − {n} (n={len(s)}) | {k} | " + fmt([o[x][k] - d[x][k] for x in s]) + " |")
    txt = "\n".join(out); print(txt)
    import os; os.makedirs(f"artifacts/orbita/ablations", exist_ok=True); open(f"artifacts/orbita/ablations/summary_{ph}.md", "w").write(txt + "\n")
