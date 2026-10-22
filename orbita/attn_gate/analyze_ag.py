"""Análisis ÓRBITA + corrector de atención gateado (docs/prereg_orbita_attn_gate.md). Uso: analyze_ag.py dev|confirm → markdown (stdout)
y per_seed_{ph}.json. IC95 bootstrap pareado por seed (10 000 remuestreos, numpy default_rng(12345)), n = 16. τ fijo de tau.json (DEV)."""
import csv, json, sys, os, numpy as np
ph = sys.argv[1]; W = "/workspace/orbita-data/attn-gate"; A = sys.argv[2] if len(sys.argv) > 2 else "."
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
pre = "0xA4" if ph == "dev" else "0xB4"; SEEDS = [f"{pre}{i:02X}" for i in range(16)]; DELTA = 0.05
TAU = json.load(open(f"{W}/tau.json")); TFPE = json.load(open(f"{R}/artifacts/orbita/transformer_fe/per_seed_{ph}.json"))
assert TFPE["seeds"] == SEEDS
L = lambda p: list(csv.DictReader(open(p)))
V = {}; put = lambda k, v: V.setdefault(k, []).append(float(v)); checks = []
for i, s in enumerate(SEEDS):
    ev = L(f"{W}/{ph}/AG/eval_{s}.csv")
    for k in (2, 4, 8):
        lat = {r["idx"]: r for r in L(f"{W}/{ph}/LAT/lat_{s}_k{k}.csv")}
        fam = np.array([int(r["fam"]) for r in ev]); sk = np.array([float(r[f"s_k{k}"]) for r in ev])
        obs = np.array([float(r[f"obs_k{k}"]) for r in ev]); alw = np.array([float(r[f"alw_k{k}"]) for r in ev])
        lobs = np.array([float(lat[r["idx"]]["obs16"]) for r in ev]); latep = np.array([float(lat[r["idx"]]["latep16"]) for r in ev])
        checks.append(int((lobs != obs).sum()))
        arms = {"ORBITA": obs, "ALWAYS": alw, "LAT_EP": latep}
        for q in (0.90, 0.95, 0.99):
            g = sk > TAU[f"k{k}_q{q:.2f}"]; arms["GATE" if q == 0.95 else f"GATE_q{q:.2f}"] = np.where(g, alw, obs)
            for grp, msk in (("nuevas", fam >= 8), ("conocidas", fam < 8)): put(f"disparo GATE_q{q:.2f} {grp} k={k}", g[msk].mean())
        for grp, msk in (("nuevas", fam >= 8), ("conocidas", fam < 8)):
            for a, v in arms.items(): put(f"{a} {grp} k={k}", v[msk].mean())
            put(f"TF-PE {grp} k={k}", TFPE["per_seed"][f"TF-PE {grp} k={k}"][i])
            tl = TFPE["per_seed"][f"ORBITA-LAT_EP {grp} k={k}"][i]; checks.append(int(abs(tl - latep[msk].mean()) > 1e-12))
M = {k: np.array(v) for k, v in V.items()}
rng = np.random.default_rng(12345); BI = rng.integers(0, 16, (10000, 16))
def ci(x): b = x[BI].mean(1); return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"; g = lambda t: f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"
out = [f"# ÓRBITA + corrector de atención gateado — análisis {ph.upper()}", "",
       f"Comprobaciones: discrepancias OBS Python vs Rust y LAT_EP vs artefacto dcaadad = {sum(checks)} (debe ser 0). τ: {json.dumps(TAU)}", ""]
HY = [("H1", "GATE", "LAT_EP", "nuevas", 0.0), ("H2", "GATE", "ORBITA", "conocidas", -DELTA), ("H3", "GATE", "ALWAYS", "conocidas", 0.0),
      ("H4", "GATE", "TF-PE", "conocidas", 0.0), ("S1", "GATE", "ORBITA", "nuevas", 0.0), ("S2", "ALWAYS", "LAT_EP", "nuevas", 0.0),
      ("S3", "GATE", "TF-PE", "nuevas", 0.0), ("S4", "GATE", "LAT_EP", "conocidas", 0.0), ("S5", "ALWAYS", "ORBITA", "nuevas", 0.0),
      ("S6", "ALWAYS", "ORBITA", "conocidas", -DELTA)]
res = {}
for k in (2, 4, 8):
    out += [f"## k = {k}", "", "| id | A − B | grupo | media [IC95] | umbral (inf. IC95 >) | cumple |", "|---|---|---|---|---|---|"]
    for hid, a, b, grp, thr in HY:
        t = ci(M[f"{a} {grp} k={k}"] - M[f"{b} {grp} k={k}"]); res[f"{hid} k={k}"] = t
        out.append(f"| {hid} | {a} − {b} | {grp} | {f(t)} | {thr:+.2f} | {'sí' if t[1] > thr else ('opuesto' if t[2] < thr else 'no')} |")
    out += ["", "| brazo | nuevas acc16 | conocidas acc16 |", "|---|---|---|"]
    for a in ("ORBITA", "TF-PE", "GATE", "GATE_q0.90", "GATE_q0.99", "ALWAYS", "LAT_EP"):
        out.append(f"| {a} | {g(ci(M[f'{a} nuevas k={k}']))} | {g(ci(M[f'{a} conocidas k={k}']))} |")
    out += ["", "| tasa de disparo | nuevas | conocidas |", "|---|---|---|"]
    for q in ("0.90", "0.95", "0.99"): out.append(f"| GATE_q{q} | {g(ci(M[f'disparo GATE_q{q} nuevas k={k}']))} | {g(ci(M[f'disparo GATE_q{q} conocidas k={k}']))} |")
    out.append("")
# convergencia
conv = []
for s in SEEDS:
    cv = L(f"{W}/{ph}/AG/c_{s}.pt.curve.csv"); val = np.array([float(r["val"]) for r in cv]); tr = np.array([float(r["train"]) for r in cv[1:]])
    n = len(tr); w = max(1, n // 10); c1 = val[-1] <= 1.05 * val.min(); c2 = abs(tr[-w:].mean() - tr[-2 * w:-w].mean()) / tr[-2 * w:-w].mean() < 0.05
    c3 = val[-1] < 0.9 * val[0]; c4 = int(float(cv[-1]["skipped"])) == 0 and np.isfinite(val).all()
    conv.append((s, val[0], val.min(), val[-1], tr[-w:].mean(), c1, c2, c3, c4))
out += ["## Convergencia del corrector (criterios C1–C4 del prerregistro)", "", "| seed | val(0) | val mín | val final | train último 10 % | C1 | C2 | C3 | C4 |", "|---|---|---|---|---|---|---|---|---|"]
for r in conv: out.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | " + " | ".join("✔" if x else "✘" for x in r[5:]) + " |")
allc = all(all(r[5:]) for r in conv); out += ["", f"**Convergencia: {'TODAS las seeds cumplen C1–C4' if allc else 'FALLA en alguna seed'}**"]
print("\n".join(out))
json.dump({"seeds": SEEDS, "per_seed": V, "ci": res, "converged": allc, "checks": sum(checks)}, open(f"{A}/per_seed_{ph}.json", "w"), indent=0)
open(f"{W}/{ph}/converged" if allc else f"{W}/{ph}/not_converged", "w").write("1")
