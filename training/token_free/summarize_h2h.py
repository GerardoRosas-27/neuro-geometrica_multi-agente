import json, sys, numpy as np, collections
C = "/workspace/ngm-tokenfree-cache"
for ph in ["dev", "conf"]:
    v4 = json.load(open(f"{C}/h2h_v4_{ph}.json")); tf = json.load(open(f"artifacts/token_free/h2h_tf_{ph}.json"))
    print(f"== {ph} ({len(v4)} semillas)")
    for arm in "AB":
        x = [s["v4"][arm] for s in v4]
        print(f"v4 Dphi {arm}: h1={np.mean([a['acc_h1'] for a in x]):.4f} roll={np.round(np.mean([a['roll_iterable'] for a in x],0),3).tolist()} comp2={np.mean([a['comp2'] for a in x]):.3f} lat_p50={np.median([a['lat_us']['p50'] for a in x]):.2f} params={x[0]['params']} wall_E35={np.mean([s['e35_core_wall_s'] for s in v4]):.2f}s mem_q={sum(s['test_memory_queries_B'] for s in v4)}")
    g = collections.defaultdict(list)
    for r in tf: g[(r["model"], r["arm"])].append(r)
    for (m, a), v in g.items():
        par1 = max(r["parity"]["roll_step1"]["max_abs"] for r in v); par64 = max(r["parity"]["roll_step64"]["max_abs"] for r in v)
        print(f"TF {m} {a}: h1={np.mean([r['acc_h1'] for r in v]):.4f} roll={np.round(np.mean([r['roll_iterable'] for r in v],0),3).tolist()} comp2={np.mean([r['comp2'] for r in v]):.3f} lat_p50={np.median([r['lat_us']['p50'] for r in v]):.2f} params={v[0]['params']} train={np.mean([r['train_s'] for r in v]):.2f}s parity_max_step1={par1:.1e} step64={par64:.1e} mem_q={sum(r['memory_queries'] for r in v)}")
    # diferencia pareada mejor TF_B (elegido en DEV: ssm) vs v4 B
    b4 = np.array([s["v4"]["B"]["acc_h1"] for s in v4])
    tfb = np.array([r["acc_h1"] for r in tf if r["model"] == "ssm" and r["arm"] == "B"])
    d = tfb - b4; rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    print(f"TF_ssm_B - v4_B: {d.mean():+.4f} IC95 [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
