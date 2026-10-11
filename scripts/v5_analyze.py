#!/usr/bin/env python3
"""v5 P2 summary: one-step per arm and E33 region per variant."""
import csv, re, statistics as st, sys

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
arms = ["P2_gated", "P2_mlp", "P2_shuffled"]
names = [
    "translate", "rotate", "reflect", "scale", "shear",
    "affine", "compose_rot_scale", "radial_warp",
]
for c in arms:
    rs = [r for r in rows if r["condition"] == c]
    if not rs:
        continue
    f = lambda k: st.mean(float(r[k]) for r in rs)
    print(
        f"{c:<14} params={rs[0]['params']} acc={f('accuracy'):.4f} "
        f"rel_err1={f('stability'):.4f} leak={sum(int(r['leakage']) for r in rs)}"
    )
    fam = []
    for nm in names:
        v = []
        for r in rs:
            m = re.search(re.escape(nm) + r":rel=([0-9.]+)", r["notes"])
            if m:
                v.append(float(m.group(1)))
        if v:
            fam.append(f"{nm}={st.mean(v):.4f}")
    print("    " + " ".join(fam))

for v in ["R_P2_gated", "R_P2_mlp", "R_P2_shuffled"]:
    g = [r for r in rows if r["condition"] == "GATE" and r["partition"] == v]
    if not g:
        continue
    regs = [int(re.search(r"region=(\d+)", r["notes"]).group(1)) for r in g]
    print(
        f"{v:<16} PASS {sum(r['status']=='PASS' for r in g)}/{len(g)} "
        f"region mean={st.mean(regs):.2f} median={st.median(regs)} regions={regs}"
    )
    line = []
    for h in [1, 2, 4, 8, 16, 32, 64]:
        rs = [r for r in rows if r["condition"] == f"{v}_h{h}"]
        if not rs:
            continue
        ro = st.mean(float(re.search(r"acc_rot=([0-9.]+)", r["notes"]).group(1)) for r in rs)
        rf = st.mean(float(re.search(r"acc_refl=([0-9.]+)", r["notes"]).group(1)) for r in rs)
        line.append(
            f"h{h}:acc={st.mean(float(r['accuracy']) for r in rs):.3f}"
            f"(rot {ro:.2f}/refl {rf:.2f}) rel={st.mean(float(r['stability']) for r in rs):.3f}"
        )
    print("    " + "\n    ".join(line))
