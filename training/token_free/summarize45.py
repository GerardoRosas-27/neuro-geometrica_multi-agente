import json, sys, numpy as np, collections
f, split = sys.argv[1], sys.argv[2]
rows = json.load(open(f))["rows"]
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["method"], r["N"])].append(r["res"][split])
for (m, N), v in sorted(agg.items()):
    a = lambda k: np.mean([x[k] for x in v])
    per = [round(x["relational_exact"], 4) for x in v]
    print(f"{m:14s} {N:4d} dec={a('decode_exact'):.4f} rel={a('relational_exact'):.4f} per={per} surfR2={a('surface_r2'):.3f}")
