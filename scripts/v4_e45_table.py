#!/usr/bin/env python3
import csv, re, sys, statistics as st
base = sys.argv[1]
for ph in sys.argv[2:]:
    rows = [r for r in csv.DictReader(open(f"{base}/{ph}/metrics.csv")) if r["condition"] != "GATE"]
    print(f"## {ph}\n| brain | acc TEST | set1 tras p2 | olvido | set2 | retención episodios borrados | consolidación ms | params cambiados | bytes | latencia µs | updates | queries |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in ["L1_liquid_gated", "L2_liquid_lowrank", "T1_thermo_cdt", "H_liquid_inf_cdt_cons"]:
        R = [r for r in rows if r["condition"] == b]
        g = lambda k: st.mean(float(re.search(k + r"=(-?[0-9.]+)", r["notes"]).group(1)) for r in R)
        q = sum(int(r[c]) for r in R for c in ["cdt_queries","rqm_queries","table_queries","nn_queries","attractor_queries","direct_memory_queries"]) + sum(int(r["leakage"]) for r in R)
        print(f"| {b} | {st.mean(float(r['accuracy']) for r in R):.3f} | {g('set1_after_p2'):.3f} | {g('forgetting'):.3f} | {g('set2'):.3f} | {g('retention_deleted_episodes'):.3f} | {g('consolidation_ms'):.1f} | {g('params_changed'):.0f} | {g('bytes'):.0f} | {g('latency_us'):.2f} | {g('updates'):.0f} | {q} |")
