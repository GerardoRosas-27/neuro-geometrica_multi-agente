"""Tablas por semilla (ÓRBITA, v4 E3, M3) desde los artefactos. Uso: python3 orbita/per_seed_tables.py dev|confirm"""
import csv, glob, json, re, sys
ph = sys.argv[1]
def rows(ps):
    for p in ps:
        yield from csv.DictReader(open(p))
v4 = list(rows([f"artifacts/v4/distill_e/{ph}/metrics.csv"])); orb = list(rows(sorted(glob.glob(f"artifacts/orbita/{ph}/metrics_0x*.csv"))))
H = [1, 2, 4, 8, 16]
def pick(rs, one, roll):
    d = {}
    for r in rs:
        s = r["seed"]; d.setdefault(s, {})
        if r["condition"] == one:
            d[s]["acc1"] = float(r["accuracy"]); d[s]["rel1"] = float(r["stability"]); d[s]["leak"] = int(r["leakage"])
            d[s]["q"] = sum(int(r[k]) for k in ["cdt_queries","rqm_queries","table_queries","nn_queries","attractor_queries","direct_memory_queries"])
            m = re.search(r"fresh_ok=(\w+)", r["notes"])
            if m: d[s]["fresh"] = m.group(1)
            d[s]["status"] = r["status"]
        for t in H:
            if r["condition"] == f"{roll}_h{t}": d[s][f"h{t}"] = float(r["accuracy"])
        if r["condition"] == "GATE" and r["partition"] == roll:
            d[s]["gate"] = r["status"]; d[s]["reg"] = int(re.search(r"region=(\d+)", r["notes"]).group(1))
            m = re.search(r"fresh_E3=([0-9.]+)\(q=([0-9.]+)\)", r["notes"])
            if m and "fresh" not in d[s]: d[s]["fresh"] = f"acc={m.group(1)},q={m.group(2)}"
    return d
A = {"ÓRBITA": pick(orb, "ORBITA", "R_ORBITA_V0"), "v4 E3": pick(v4, "E3_gated_trusted", "R_E3_V0_pure"), "MLP M3": pick(v4, "M3_mlp_gated_trusted", "R_M3_V0")}
seeds = sorted(A["ÓRBITA"])
par = {s: json.load(open(f"artifacts/orbita/{ph}/parity_{s}.json"))["max_l2_diff"] for s in seeds}
print(f"### {ph.upper()} — por semilla\n")
print("| seed | modelo | gate E33 | región | acc1 | rel1 | " + " | ".join(f"h{t}" for t in H) + " | estado | leak | queries | proceso nuevo | paridad Rust↔Py |")
print("|---|---|---|---|---|---|" + "---|" * len(H) + "---|---|---|---|---|")
for s in seeds:
    for n, d in A.items():
        x = d[s]
        p = f"{par[s]:.1e}" if n == "ÓRBITA" else "—"
        fr = x.get("fresh", "—") if n != "MLP M3" else "—"
        print(f"| {s} | {n} | {x['gate']} | {x['reg']} | {x['acc1']:.4f} | {x['rel1']:.4f} | " + " | ".join(f"{x[f'h{t}']:.3f}" for t in H) + f" | {x['status']} | {x['leak']} | {x['q']} | {fr} | {p} |")
print()
