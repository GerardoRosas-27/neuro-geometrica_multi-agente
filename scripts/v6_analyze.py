#!/usr/bin/env python3
"""v6 paired analysis (docs/preregistro_v6.md §5). Usage: v6_analyze.py artifacts/v6/dev"""
import csv, re, sys, random, statistics as st
from collections import defaultdict

d = sys.argv[1]
g = defaultdict(dict)  # label -> seed -> dict
for r in csv.DictReader(open(f"{d}/metrics.csv")):
    if r["condition"] != "GATE":
        continue
    kv = dict(re.findall(r"(\w+)=([^\s]+)", r["notes"]))
    kv["status"] = r["status"]
    g[r["partition"]][r["seed"]] = kv

def f(lbl, key):
    return {s: float(v[key]) for s, v in g.get(lbl, {}).items() if v.get(key) not in (None, "NaN", "nan")}

def ci(diffs, n=10000):
    rng = random.Random(0x6B00)
    bs = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]

def pair(a, b, key="acc_h16"):
    A, B = f(a, key), f(b, key)
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    return ci([A[x] - B[x] for x in s])

def show(name, r):
    print(f"{name:55s} " + ("n/a" if r is None else f"{r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]"))

print("## gates")
for lbl in sorted(g):
    p = sum(v["status"] == "PASS" for v in g[lbl].values())
    clean = all(v.get("clean") == "true" and v.get("fresh_ok") == "true" for v in g[lbl].values())
    print(f"{lbl:28s} PASS {p}/{len(g[lbl])} clean+fresh={clean} acc_h16={st.mean(f(lbl,'acc_h16').values() or [float('nan')]):.4f}")
print("\n## pre-registered contrasts (acc_h16 unless noted; mean [IC95])")
show("H1  H1_bias - E3_v4", pair("R_H1_bias", "R_E3_v4"))
show("H1  acc1 H1_bias - E3_v4", pair("R_H1_bias", "R_E3_v4", "acc1"))
show("H2* H2_prm_eqc - H2_mh_eqc (PENDING)", pair("R_H2_prm_eqc", "R_H2_mh_eqc"))
for m in ["0.5", "1", "2", "4"]:
    show(f"H3  rel_err1 H3_x{m} - E3_v4", pair(f"R_H3_x{m}", "R_E3_v4", "rel_err1"))
a, b = pair("R_H4_sobolev", "R_E3_v4"), pair("R_H4_sobolev_mlp", "R_E3_mlp")
A1, B1, A2, B2 = (f(x, "acc_h16") for x in ["R_H4_sobolev", "R_E3_v4", "R_H4_sobolev_mlp", "R_E3_mlp"])
s = sorted(set(A1) & set(B1) & set(A2) & set(B2))
show("H4  interaction (field gain - mlp gain)", ci([(A1[x]-B1[x])-(A2[x]-B2[x]) for x in s]) if len(s) > 1 else None)
show("H5a H5_soft - E3_v4", pair("R_H5_soft", "R_E3_v4"))
for e in ["1", "2", "4", "8"]:
    v = f(f"R_H5_eta{e}", "dstep_vs_E3")
    print(f"H5b eta={e}% dstep_vs_E3 mean={st.mean(v.values()) if v else float('nan'):.5f}")
show("H6  H6_semigroup - E3_mlp", pair("R_H6_semigroup", "R_E3_mlp"))
show("H6  H6_semigroup - E3_v4", pair("R_H6_semigroup", "R_E3_v4"))
show("H7  H7_normal(V0) - E3_v4(V1reproj)", pair("R_H7_normal", "R_E3_v4_V1reproj"))
show("H7  H7_normal - E3_v4", pair("R_H7_normal", "R_E3_v4"))
show("v6  V6_all - E3_v4", pair("R_V6_all", "R_E3_v4"))
show("v6  acc1 V6_all - E3_v4", pair("R_V6_all", "R_E3_v4", "acc1"))
show("v6  V6_all - V6_all_mlp", pair("R_V6_all", "R_V6_all_mlp"))
show("v6* V6_all_H2 - V6_all (PENDING)", pair("R_V6_all_H2", "R_V6_all"))
