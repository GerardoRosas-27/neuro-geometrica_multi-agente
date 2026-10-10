"""Paridad + latencia Rust de los 36 modelos E46b (corpus: 256 inicios DEV × 16 ops, semilla 9)."""
import json, numpy as np
from e46b import build
from e47_48 import load, rust_bench
_, d, _ = build(); out = []
for s in range(6):
    for kind in ["mlp", "v4res", "liquid", "liquid_ss", "ssm", "linear"]:
        p, Dw, Db, P = load(kind, s)
        r = rust_bench(kind, s, p, P, d, [], f"par46b_{kind}_s{s}")
        out.append({"model": kind, "seed": s, **r}); print(kind, s, r["step1"]["max_abs"], r["step16"]["max_abs"], r["step16"]["discrete_match"], r["latency_us_step"]["p50"], flush=True)
json.dump(out, open("../../artifacts/token_free/e46b_parity.json", "w"), indent=1)
