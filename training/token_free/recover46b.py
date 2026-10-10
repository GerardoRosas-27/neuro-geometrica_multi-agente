"""Reconstruye e46b_<fase>.json desde el log + exportaciones (el proceso DEV murió por OOM tras exportar todo)."""
import json, sys, numpy as np
ph = sys.argv[1]; C = "/workspace/ngm-tokenfree-cache"; rows = []
for l in open(f"{C}/e46b_{ph}.log"):
    if not l.startswith('{"model"'): continue
    r = json.loads(l); a = json.load(open(f"{C}/e46b_{r['model']}_s{r['seed']}.json"))
    r["params"] = int(sum(np.prod(v["shape"]) for k, v in a["params"].items() if k.startswith("model.")))
    r["train_s_per_seed"] = a["meta"]["train_s"]; r["params_sha256"] = a["params_sha256"]
    rows.append(r)
json.dump({"manifest": json.load(open(f"{C}/e46b_{rows[0]['model']}_s{rows[0]['seed']}.json"))["meta"]["dataset"],
           "encoder_train_s": float("nan"), "encoder_loss": [], "recovered_from_log": True, "rows": rows},
          open(f"../../artifacts/token_free/e46b_{ph}.json", "w"), indent=1)
