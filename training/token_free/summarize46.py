import json, sys, numpy as np, collections
rows = json.load(open(sys.argv[1]))["rows"]
by = collections.defaultdict(list)
for r in rows: by[r["model"]].append(r)
for m, v in by.items():
    out = {"params": v[0]["params"], "train_s": round(np.mean([r["train_s"] for r in v]), 1)}
    for s in v[0]["res"]:
        for k in v[0]["res"][s]:
            vals = [r["res"][s][k] for r in v]
            out[f"{s}.{k}"] = (round(float(np.mean(vals)), 4), [round(x, 4) for x in vals])
    print(m, json.dumps(out))
