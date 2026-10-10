import json, sys, numpy as np, collections
d = json.load(open(sys.argv[1])); rows = d["rows"]
print("encoder_train_s", round(d["encoder_train_s"], 1), "enc_loss", [round(x, 3) for x in d["encoder_loss"]])
g = collections.defaultdict(list)
for r in rows: g[r["model"]].append(r)
for m, v in g.items():
    res = v[0]["res"]; out = {"params": v[0]["params"], "train_s": round(v[0]["train_s_per_seed"], 1)}
    for s in res:
        for k in res[s]:
            vals = [r["res"][s][k] for r in v]; out[f"{s}.{k}"] = round(float(np.mean(vals)), 4)
    if "DEV" in res: ax = [ (r["res"]["DEV"]["h4"] + r["res"]["LONG_HORIZON"]["h16"]) / 2 for r in v]
    else: ax = [(r["res"]["OOD"]["h4"] + r["res"]["COMPOSITION"]["h4"] + r["res"]["LONG_HORIZON"]["h16"]) / 3 for r in v]
    out["AXIS1"] = round(float(np.mean(ax)), 4); out["axis1_per_seed"] = [round(x, 4) for x in ax]
    print(m, json.dumps(out))
