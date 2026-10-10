import json, sys, numpy as np, collections
r = json.load(open(sys.argv[1]))
g = collections.defaultdict(list)
for x in r:
    key = (x["exp"], x.get("model", "liquid_ss"), x.get("k", x.get("K", x.get("eps"))), x.get("mode", "topk"))
    g[key].append(x)
for k, v in g.items():
    par = all(x["rust"]["step1"]["max_abs"] <= 1e-4 for x in v)
    print(k, "axis1=%.4f" % np.mean([x["q"]["axis1"] for x in v]), "per", [round(x["q"]["axis1"], 4) for x in v],
          "lat_mean=%.2f" % np.mean([x["rust"]["latency_us_step"]["mean"] for x in v]), "p50=%.2f" % np.mean([x["rust"]["latency_us_step"]["p50"] for x in v]),
          "sub=%.2f" % np.mean([x["rust"]["substeps_per_op_mean"] for x in v]), "par1", par, "max16=%.1e" % max(x["rust"]["step16"]["max_abs"] for x in v), "disc16", min(x["rust"]["step16"]["discrete_match"] for x in v),
          {s: round(float(np.mean([x["q"][s] for x in v])), 4) for s in ("OOD", "COMPOSITION") if s in v[0]["q"]})
