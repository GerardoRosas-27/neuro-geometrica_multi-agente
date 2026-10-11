#!/usr/bin/env python3
"""DIST_D (E33 on one-step model) summary per variant/horizon."""
import csv, re, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
V = ["V0_pure_C1", "V1_reproj_C1", "V3_reproj_spectral_C1", "V0_pure_M1_mlp", "V0_pure_C0_lock"]
for v in V:
    g = [r for r in rows if r["condition"] == "GATE" and r["partition"] == v]
    regs = [int(re.search(r"region=(\d+)", r["notes"]).group(1)) for r in g]
    sig = [float(re.search(r"sigma_med=([0-9.]+)", r["notes"]).group(1)) for r in g]
    print(f"{v:<24} PASS {sum(r['status']=='PASS' for r in g)}/{len(g)} region mean={st.mean(regs):.2f} max={max(regs)} sigma_med={st.mean(sig):.3f}")
    for h in [1, 2, 4, 8, 16, 32, 64]:
        rs = [r for r in rows if r["condition"] == f"{v}_h{h}"]
        f = lambda k: st.mean(float(r[k]) for r in rs)
        nr = st.mean(float(re.search(r"norm_ratio=([0-9.]+)", r["notes"]).group(1)) for r in rs)
        print(f"   h{h:<3} acc={f('accuracy'):.3f} rel={f('stability'):.3f} norm={nr:.3f} energy={f('energy'):.3f} manif={f('manifold_distance'):.3f} pert_growth={f('cosine'):.3f}")
