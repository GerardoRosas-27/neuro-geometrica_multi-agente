#!/usr/bin/env python3
"""Paired stats for v4 metrics.csv (E35 A vs B, E33 horizons, E42 per-N)."""
import csv, random, statistics as st, sys, re
p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
def acc(e, c, part="TEST"):
    return {r["seed"]: float(r["accuracy"]) for r in rows if r["experiment"] == e and r["condition"] == c and r["partition"] == part}
def paired(e, ca, cb, part="TEST"):
    a, b = acc(e, ca, part), acc(e, cb, part)
    d = [b[s] - a[s] for s in a if s in b]
    random.seed(0xB007)
    ms = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(4000))
    sd = st.stdev(d) if len(d) > 1 else 0
    return dict(n=len(d), meanA=st.mean(a.values()), meanB=st.mean(b.values()), diff=st.mean(d), median=st.median(d),
                ci=(ms[100], ms[3899]), dz=(st.mean(d) / sd if sd else 0), wins=sum(x > 0 for x in d))
fmt = lambda r: f"n={r['n']} A={r['meanA']:.4f} B={r['meanB']:.4f} diff={r['diff']:+.4f} med={r['median']:+.4f} CI95=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] dz={r['dz']:.2f} B>A in {r['wins']}"
print("E35 TEST   ", fmt(paired("E35", "A_control", "B_cdt_experience")))
print("E35 DEV    ", fmt(paired("E35", "A_control", "B_cdt_experience", "DEV")))
print("E36 C-Braw ", fmt(paired("E36", "B_raw_experience", "C_consolidated")))
print("E36 Braw-A ", fmt(paired("E36", "A_no_experience", "B_raw_experience")))
print("E39        ", fmt(paired("E39", "A_no_consolidation", "B_consolidated", "TEST_triangle")))
for n in [8, 16, 32, 64, 128]:
    print(f"E42 N={n:<4}", fmt(paired("E42", f"A_N{n}", f"B_N{n}")))
leak = sum(int(r["leakage"]) for r in rows if r["experiment"] in ("E35", "E36", "E39", "E42"))
q = sum(int(r["cdt_queries"]) + int(r["rqm_queries"]) + int(r["table_queries"]) + int(r["nn_queries"]) + int(r["attractor_queries"]) + int(r["direct_memory_queries"]) for r in rows if r["experiment"] in ("E35", "E36", "E39", "E42"))
print(f"E35/36/39/42 total leaked predictions={leak} total memory queries in TEST={q}")
for h in [1, 2, 4, 8, 16, 32, 64]:
    it = [float(re.search(r"iterable_acc=([0-9.]+)", r["notes"]).group(1)) for r in rows if r["experiment"] == "E33" and r["condition"] == f"DPHI_h{h}"]
    al = [float(r["accuracy"]) for r in rows if r["experiment"] == "E33" and r["condition"] == f"DPHI_h{h}"]
    re_ = [float(r["stability"]) for r in rows if r["experiment"] == "E33" and r["condition"] == f"DPHI_h{h}"]
    print(f"E33 h{h:<3} iterable_acc={st.mean(it):.3f} all_acc={st.mean(al):.3f} mean_rel_err={st.mean(re_):.3f}")
for c in ["DPHI_h1", "MLP_DIRECT_h1", "DPHI_h4_iter", "MLP_DIRECT_h4_iter", "STATIC"]:
    print(f"E34 {c:<20} acc={st.mean(acc('E34', c).values()):.3f}")
for c in ["DPHI", "STATIC", "LINEAR", "RANDOM_DYN", "SHUFFLED_LABEL", "NN_LOOKUP"]:
    print(f"E31 {c:<15} acc={st.mean(acc('E31', c).values()):.3f}")
for c in ["A_sequential", "B_cdt_replay", "D_corrupt_replay"]:
    print(f"E40 {c:<17} acc={st.mean(acc('E40', c).values()):.3f}")
