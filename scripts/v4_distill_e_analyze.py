#!/usr/bin/env python3
"""DIST_E (§16) summary: one-step per family + E33 per variant."""
import csv, re, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
M = ["E0_faseC_H64", "E1_plain_trusted", "E2_gated_consol", "E3_gated_trusted", "M3_mlp_gated_trusted"]
names = ["translate","rotate","reflect","scale","shear","affine","compose_rot_scale","radial_warp","in_dist","interp_hole"]
for c in M:
    rs = [r for r in rows if r["condition"] == c]
    f = lambda k: st.mean(float(r[k]) for r in rs)
    print(f"{c:<22} params={rs[0]['params']} acc={f('accuracy'):.4f} rel_err1={f('stability'):.4f} leak={sum(int(r['leakage']) for r in rs)} q={sum(sum(int(r[k]) for k in ['cdt_queries','rqm_queries','table_queries','nn_queries','attractor_queries','direct_memory_queries']) for r in rs)}")
    fam = []
    for nm in names:
        v = [float(re.search(re.escape(nm) + r"(?:\(n=\d+\))?:rel=([0-9.]+)", r["notes"]).group(1)) for r in rs]
        fam.append(f"{nm}={st.mean(v):.4f}")
    print("    " + " ".join(fam))
V = ["R_E0_V0", "R_E1_V0", "R_E2_V0", "R_E3_V0_pure", "R_E3_V1_reproj", "R_M3_V0"]
for v in V:
    g = [r for r in rows if r["condition"] == "GATE" and r["partition"] == v]
    regs = [int(re.search(r"region=(\d+)", r["notes"]).group(1)) for r in g]
    fr = [re.search(r"fresh_E3=([0-9.]+)\(q=([0-9.]+)", r["notes"]) for r in g]
    print(f"{v:<16} PASS {sum(r['status']=='PASS' for r in g)}/{len(g)} region mean={st.mean(regs):.2f} median={st.median(regs)} regions={regs}")
    line = []
    for h in [1, 2, 4, 8, 16, 32, 64]:
        rs = [r for r in rows if r["condition"] == f"{v}_h{h}"]
        ro = st.mean(float(re.search(r"acc_rot=([0-9.]+)", r["notes"]).group(1)) for r in rs)
        rf = st.mean(float(re.search(r"acc_refl=([0-9.]+)", r["notes"]).group(1)) for r in rs)
        line.append(f"h{h}:acc={st.mean(float(r['accuracy']) for r in rs):.3f}(rot {ro:.2f}/refl {rf:.2f}) rel={st.mean(float(r['stability']) for r in rs):.3f} pert={st.mean(float(r['cosine']) for r in rs):.2f}")
    print("    " + "\n    ".join(line))
g = [r for r in rows if r["condition"] == "GATE" and r["partition"] == "R_E3_V0_pure"]
e3 = {r["seed"]: float(r["accuracy"]) for r in rows if r["condition"] == "E3_gated_trusted"}
ok = 0
for r in g:
    m = re.search(r"fresh_E3=([0-9.]+)\(q=([0-9.]+)", r["notes"])
    ok += bool(m and abs(float(m.group(1)) - e3[r["seed"]]) < 1e-9 and float(m.group(2)) == 0)
print(f"fresh-process E3 OK {ok}/{len(g)}")
