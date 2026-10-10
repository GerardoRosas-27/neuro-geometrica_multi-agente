"""E47 (sparse top-k) y E48 (parada adaptativa) sobre modelos E46b congelados."""
import json, sys, subprocess, numpy as np, jax, jax.numpy as jnp
import models2 as M
from e46b import build, NOPS, N, S, V, dec_logits

C_DIR = "/workspace/ngm-tokenfree-cache"
RUNNER = "../../target/release/run_token_free_field"
KS = [1, 2, 4, 7, 13, 32, 64]
KFIX = [1, 2, 4, 8, 16, 32, 64]
EPS = [0.1, 0.03, 0.01, 0.003]


def load(kind, s):
    a = json.load(open(f"{C_DIR}/e46b_{kind}_s{s}.json"))
    t = {k: np.array(v["data"], np.float32).reshape(v["shape"]) for k, v in a["params"].items()}
    p = {k[6:]: jnp.asarray(v) for k, v in t.items() if k.startswith("model.")}
    return p, t["decoder.Dw"], t["decoder.Db"], np.load(f"{C_DIR}/e46b_psi_s{s}.npy")


def adaptive_step(p, psi, c, eps):
    """Devuelve (Ψ', sub-pasos acumulados) por muestra; misma regla que el runner Rust."""
    outs = {K: M.step("liquid_ss", p, psi, c, None, K) for K in KFIX}
    done = jnp.zeros(psi.shape[0], bool); res = outs[1]; cost = jnp.ones(psi.shape[0])
    prev = outs[1]
    for K in KFIX[1:]:
        cur = outs[K]
        active = ~done
        cost = cost + jnp.where(active, K, 0)
        res = jnp.where(active[:, None], cur, res)
        rel = jnp.linalg.norm(cur - prev, axis=-1) / (jnp.linalg.norm(cur, axis=-1) + 1e-6)
        done = done | (rel < eps)
        prev = cur
    return res, cost


def roll(kind, p, psi0, ops, k=None, K=4, eps=None):
    C = jax.nn.one_hot(jnp.asarray(ops), NOPS)
    cur, out, costs = psi0, [], []
    for t in range(C.shape[1]):
        if eps is None:
            cur = M.step(kind, p, cur, C[:, t], k, K); costs.append(jnp.full(cur.shape[0], K if kind == "liquid_ss" else 1))
        else:
            cur, c = adaptive_step(p, cur, C[:, t], eps); costs.append(c)
        out.append(cur)
    return jnp.stack(out, 1), float(jnp.mean(jnp.stack(costs)))


def quality(kind, p, Dw, Db, P, d, splits, **kw):
    r = {}
    for sp in splits:
        D = d[sp]; L = 16 if sp == "LONG_HORIZON" else 4
        pr, cost = roll(kind, p, jnp.asarray(P[D["tidx"][:, 0]]), D["ops"][:, :L], **kw)
        acc = np.asarray((dec_logits(Dw, Db, pr).argmax(-1) == D["z"][:, 1:L + 1]).all(-1).mean(0))
        r[sp] = float(acc[L - 1]); r[sp + "_cost"] = cost
    keys = ["DEV", "LONG_HORIZON"] if "DEV" in splits else ["OOD", "COMPOSITION", "LONG_HORIZON"]
    r["axis1"] = float(np.mean([r[k] for k in keys]))
    return r


def rust_bench(kind, s, p, P, d, flags, tag, **kw):
    D = d["DEV"]; ops = np.random.default_rng(9).integers(NOPS, size=(256, 16))
    psi0 = jnp.asarray(P[D["tidx"][:256, 0]])
    ref, _ = roll(kind, p, psi0, ops, **kw)
    cp = f"{C_DIR}/e47_{tag}.corpus"
    json.dump({"psi0": np.asarray(psi0).tolist(), "ops": ops.tolist(), "reference": np.asarray(ref).tolist(), "slots": S}, open(cp, "w"))
    o = subprocess.run([RUNNER, f"{C_DIR}/e46b_{kind}_s{s}.json", cp, "20", *flags], capture_output=True, text=True, check=True)
    return json.loads(o.stdout)


def run(phase, best):
    seeds = [0, 1, 2] if phase == "dev" else [3, 4, 5]
    splits = ["DEV", "LONG_HORIZON"] if phase == "dev" else ["OOD", "COMPOSITION", "LONG_HORIZON"]
    _, d, _ = build()
    rows = []
    for s in seeds:
        for kind in sorted({"liquid_ss", best}):
            p, Dw, Db, P = load(kind, s)
            for k in KS:
                q = quality(kind, p, Dw, Db, P, d, splits, k=k)
                rb = rust_bench(kind, s, p, P, d, ["--topk", str(k)], f"{kind}_s{s}_k{k}", k=k)
                rows.append({"exp": "E47", "model": kind, "seed": s, "k": k, "q": q, "rust": rb})
                print(json.dumps({"E47": kind, "s": s, "k": k, "axis1": q["axis1"], "lat": rb["latency_us_step"]["p50"]}), flush=True)
        p, Dw, Db, P = load("liquid_ss", s)
        for K in KFIX:
            q = quality("liquid_ss", p, Dw, Db, P, d, splits, K=K)
            rb = rust_bench("liquid_ss", s, p, P, d, ["--substeps", str(K)], f"ss_s{s}_K{K}", K=K)
            rows.append({"exp": "E48", "mode": "fixed", "K": K, "seed": s, "q": q, "rust": rb})
            print(json.dumps({"E48": "fixed", "s": s, "K": K, "axis1": q["axis1"], "lat": rb["latency_us_step"]["mean"]}), flush=True)
        for e in EPS:
            q = quality("liquid_ss", p, Dw, Db, P, d, splits, eps=e)
            rb = rust_bench("liquid_ss", s, p, P, d, ["--adaptive", str(e)], f"ss_s{s}_e{e}", eps=e)
            rows.append({"exp": "E48", "mode": "adaptive", "eps": e, "seed": s, "q": q, "rust": rb})
            print(json.dumps({"E48": "adaptive", "s": s, "eps": e, "axis1": q["axis1"], "lat": rb["latency_us_step"]["mean"], "sub": rb["substeps_per_op_mean"]}), flush=True)
    json.dump(rows, open(f"../../artifacts/token_free/e47_48_{phase}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
