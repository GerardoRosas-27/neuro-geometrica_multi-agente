#!/usr/bin/env python3
"""Paired stats for DIST_* metrics.csv. Usage: v4_distill_analyze.py metrics.csv"""
import csv, random, statistics as st, sys, re
rows = list(csv.DictReader(open(sys.argv[1])))
exps = sorted({r["experiment"] for r in rows})
def col(e, c, k="accuracy"):
    return {r["seed"]: float(r[k]) for r in rows if r["experiment"] == e and r["condition"] == c}
def note(e, c, key):
    out = {}
    for r in rows:
        if r["experiment"] == e and r["condition"] == c:
            m = re.search(key + r"=([-0-9.eE]+)", r["notes"])
            if m: out[r["seed"]] = float(m.group(1))
    return out
def paired(a, b):
    d = [b[s] - a[s] for s in a if s in b]
    random.seed(0xB007)
    ms = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(4000))
    return st.mean(d), ms[100], ms[3899], sum(x > 0 for x in d), len(d)
for e in exps:
    conds = [c for c in dict.fromkeys(r["condition"] for r in rows if r["experiment"] == e) if c != "GATE"]
    print(f"== {e}")
    for c in conds:
        a = col(e, c); re_ = col(e, c, "stability"); te = note(e, c, "teacher_to_model_rel"); sg = note(e, c, "sigma_med"); w = note(e, c, "wall_s")
        p = [r["params"] for r in rows if r["experiment"] == e and r["condition"] == c][0]
        print(f"  {c:<30} params={p:<6} acc={st.mean(a.values()):.4f} rel_err1={st.mean(re_.values()):.4f}"
              + (f" t2m={st.mean(te.values()):.4f}" if te else "") + (f" sigma={st.mean(sg.values()):.3f}" if sg else "")
              + (f" wall_s={st.mean(w.values()):.2f}" if w else ""))
    leak = sum(int(r["leakage"]) for r in rows if r["experiment"] == e)
    q = sum(sum(int(r[k]) for k in ["cdt_queries","rqm_queries","table_queries","nn_queries","attractor_queries","direct_memory_queries"]) for r in rows if r["experiment"] == e)
    g = [r for r in rows if r["experiment"] == e and r["condition"] == "GATE"]
    print(f"  leakage={leak} queries={q} gate_seeds_PASS={sum(r['status']=='PASS' for r in g)}/{len(g)}")
    pairs = {"DIST_A": [("B_consolidation_route", "D1_teacher_distill"), ("A_train_only", "D1_teacher_distill"), ("M1_mlp_direct_distill", "D1_teacher_distill"), ("D1_teacher_distill", "D4_teacher_distill_4xsteps")],
             "DIST_C": [("C0_lock_budget", "C1_onestep_recipe"), ("M1_mlp_onestep_recipe", "C1_onestep_recipe")],
             "DIST_B": [("DPHI_H64", f"DPHI_H{h}") for h in (136, 272, 544)] + [(f"MLP_H{h}", f"DPHI_H{h}") for h in (64, 136, 272, 544)]}.get(e, [])
    for a, b in pairs:
        d, lo, hi, w, n = paired(col(e, a), col(e, b))
        print(f"  {b} - {a}: diff={d:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] wins={w}/{n}")
    if e == "DIST_A":
        fr = [re.search(r"fresh_D1=([0-9.]+)\(q=([0-9.]+)", r["notes"]) for r in g]
        d1 = col(e, "D1_teacher_distill")
        ok = sum(1 for r, m in zip(g, fr) if m and abs(float(m.group(1)) - d1[r["seed"]]) < 1e-9 and float(m.group(2)) == 0)
        print(f"  fresh-process OK {ok}/{len(g)}")
    if e == "DIST_C":
        for c in ["C1_onestep_recipe", "M1_mlp_onestep_recipe"]:
            re_ = list(col(e, c, "stability").values())
            random.seed(0xB007)
            ms = sorted(st.mean(random.choices(re_, k=len(re_))) for _ in range(4000))
            print(f"  {c} rel_err1 mean={st.mean(re_):.4f} CI95=[{ms[100]:.4f},{ms[3899]:.4f}] seeds<0.05: {sum(x<0.05 for x in re_)}/{len(re_)}")
        names = ["translate","rotate","reflect","scale","shear","affine","compose_rot_scale","radial_warp","in_dist","interp_hole"]
        for c in ["C0_lock_budget", "C1_onestep_recipe"]:
            print(f"  breakdown {c}:")
            for nm in names:
                vals = {k: [] for k in ["rel","abs","cos","manif"]}
                for r in rows:
                    if r["experiment"] == e and r["condition"] == c:
                        m = re.search(re.escape(nm) + r"(?:\(n=\d+\))?:rel=([0-9.]+),abs=([0-9.]+),cos=([-0-9.]+),manif=([0-9.]+)", r["notes"])
                        for k, v in zip(vals, m.groups()): vals[k].append(float(v))
                print(f"    {nm:<18} " + " ".join(f"{k}={st.mean(v):.4f}" for k, v in vals.items()))
