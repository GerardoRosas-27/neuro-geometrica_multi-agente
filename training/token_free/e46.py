"""E46: Liquid vs MLP recurrente vs SSM vs lineal local, sobre Ψ del encoder E45 (congelado)."""
import json, sys, time, hashlib, platform
import numpy as np, jax, jax.numpy as jnp, optax
import data as D
from e45 import enc_apply, dec_logits, ce

MODELS = ["mlp", "liquid", "ssm", "linear"]
H_HID = {"mlp": 242, "liquid": 208, "ssm": 208}


def init(kind, N, key):
    ks = jax.random.split(key, 6)
    I = N + D.NOPS
    g = lambda k, s, fan: (jax.random.normal(k, s) / np.sqrt(fan)).astype(jnp.float32)
    if kind == "linear":
        return {"A": jnp.tile(jnp.eye(N), (D.NOPS, 1, 1)) + g(ks[0], (D.NOPS, N, N), N) * 0.05,
                "b": jnp.zeros((D.NOPS, N))}
    H = H_HID[kind]
    p = {"W1": g(ks[0], (I, H), I), "b1": jnp.zeros(H), "W2": g(ks[1], (H, N), H) * 0.5, "b2": jnp.zeros(N)}
    if kind in ("liquid", "ssm"):
        p["Wg"] = g(ks[2], (I, N), I); p["bg"] = jnp.full((N,), 1.0 if kind == "ssm" else 0.0)
    return p


def step(kind, p, psi, c):
    """Una transición Ψ→Ψ'. c: one-hot de op. Semántica exacta replicada en Rust."""
    x = jnp.concatenate([psi, c], -1)
    if kind == "linear":
        o = jnp.argmax(c, -1)
        return jnp.einsum("bn,bnm->bm", psi, p["A"][o]) + p["b"][o]
    if kind == "mlp":
        return jax.nn.gelu(x @ p["W1"] + p["b1"], approximate=True) @ p["W2"] + p["b2"]
    if kind == "liquid":
        tau = 0.5 + jax.nn.softplus(x @ p["Wg"] + p["bg"])
        f = jnp.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        return psi + 1.0 * (-psi / tau + f)
    a = jax.nn.sigmoid(x @ p["Wg"] + p["bg"])
    return a * psi + jax.nn.gelu(x @ p["W1"] + p["b1"], approximate=True) @ p["W2"] + p["b2"]


def rollout(kind, p, psi0, ops):
    C = jax.nn.one_hot(ops, D.NOPS)
    def f(psi, c):
        n = step(kind, p, psi, c)
        return n, n
    _, tr = jax.lax.scan(f, psi0, jnp.swapaxes(C, 0, 1))
    return jnp.swapaxes(tr, 0, 1)  # B,L,N


def nparams(p):
    return int(sum(np.prod(v.shape) for v in p.values()))


def train(kind, enc, N, seed, tr, steps=3000):
    p = init(kind, N, jax.random.PRNGKey(seed * 10 + MODELS.index(kind)))
    psi_all = enc_apply(enc, jnp.asarray(tr["h"]))
    z = jnp.asarray(tr["z"]); ops = jnp.asarray(tr["ops"])
    opt = optax.adam(2e-3); st = opt.init(p)

    def loss(p, b):
        pr = rollout(kind, p, psi_all[b, 0], ops[b])
        return jnp.mean((pr - psi_all[b, 1:]) ** 2) + ce(dec_logits(enc, pr), z[b, 1:])

    @jax.jit
    def upd(p, st, k):
        b = jax.random.randint(k, (256,), 0, z.shape[0])
        l, g = jax.value_and_grad(loss)(p, b)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rk = jax.random.PRNGKey(seed + 777)
    t0 = time.time()
    for i in range(steps):
        rk, k = jax.random.split(rk)
        p, st, l = upd(p, st, k)
    jax.block_until_ready(p)
    return p, time.time() - t0, float(l)


def exact(enc, pr, z):
    return np.asarray((dec_logits(enc, pr).argmax(-1) == z).all(-1).mean(0))  # per-h


def evaluate(kind, p, enc, d, splits):
    out = {}
    for s in splits:
        sp = d[s]
        psi0 = enc_apply(enc, jnp.asarray(sp["h"][:, 0]))
        pr = jax.jit(lambda a, o: rollout(kind, p, a, o))(psi0, jnp.asarray(sp["ops"]))
        acc = exact(enc, pr, jnp.asarray(sp["z"][:, 1:]))
        if s == "LONG_HORIZON":
            out[s] = {f"h{h}": float(acc[h - 1]) for h in [1, 2, 4, 8, 16, 32, 64]}
        else:
            out[s] = {f"h{h}": float(acc[h - 1]) for h in [1, 2, 3, 4]}
    # estabilidad: amplificación de perturbación a h=16 y norma espectral del Jacobiano de 1 paso
    sp = d["LONG_HORIZON"]
    psi0 = enc_apply(enc, jnp.asarray(sp["h"][:200, 0])); o = jnp.asarray(sp["ops"][:200, :16])
    dl = 1e-3 * jax.random.normal(jax.random.PRNGKey(5), psi0.shape)
    a = rollout(kind, p, psi0, o)[:, -1]; b = rollout(kind, p, psi0 + dl, o)[:, -1]
    amp = float(jnp.mean(jnp.linalg.norm(a - b, axis=-1) / jnp.linalg.norm(dl, axis=-1)))
    c0 = jax.nn.one_hot(o[:, 0], D.NOPS)
    J = jax.vmap(jax.jacfwd(lambda x, c: step(kind, p, x[None], c[None])[0]))(psi0, c0)
    spec = float(jnp.mean(jnp.linalg.norm(J, ord=2, axis=(1, 2))))
    out["stability"] = {"perturb_amp_h16": amp, "jacobian_spec_mean": spec}
    return out


def export(kind, p, enc, N, seed, meta, path):
    arrs = {"model." + k: np.asarray(v, np.float32) for k, v in p.items()}
    arrs.update({"decoder.Dw": np.asarray(enc["Dw"], np.float32), "decoder.Db": np.asarray(enc["Db"], np.float32)})
    body = {k: {"shape": list(v.shape), "data": v.ravel().tolist()} for k, v in sorted(arrs.items())}
    h = hashlib.sha256()
    for k, v in sorted(arrs.items()):
        h.update(k.encode()); h.update(v.tobytes())
    art = {"schema": "tfl-v1", "kind": kind, "N": N, "ops": D.NOPS, "seed": seed,
           "dtype": "f32", "layout": "row-major; x@W con W[in,out]; x=[psi, onehot(op)]",
           "activations": {"mlp": "gelu-tanh", "liquid": "tanh; tau=0.5+softplus; dt=1 Euler",
                           "ssm": "a=sigmoid; gelu-tanh", "linear": "A[op]^T psi (psi@A[op]) + b[op]"}[kind],
           "params_sha256": h.hexdigest(), "framework": {"jax": jax.__version__, "optax": optax.__version__,
           "python": platform.python_version()}, "meta": meta, "params": body}
    json.dump(art, open(path, "w"))
    return h.hexdigest()


def run(seeds, splits, N, outpath):
    from common import load
    d, man = load()
    rows = []
    for seed in seeds:
        enc = {k: jnp.asarray(v) for k, v in np.load(f"/workspace/ngm-tokenfree-cache/enc_mlp_N{N}_s{seed}.npz").items()}
        for kind in MODELS:
            p, ts, fl = train(kind, enc, N, seed, d["TRAIN"])
            res = evaluate(kind, p, enc, d, splits)
            sha = export(kind, p, enc, N, seed, {"train_s": ts, "final_loss": fl, "dataset": man["splits"]},
                         f"/workspace/ngm-tokenfree-cache/e46_{kind}_N{N}_s{seed}.json")
            rows.append({"model": kind, "seed": seed, "N": N, "params": nparams(p), "train_s": ts,
                         "final_loss": fl, "params_sha256": sha, "res": res})
            print(json.dumps(rows[-1]), flush=True)
    json.dump({"manifest": man, "rows": rows, "device": str(jax.devices())}, open(outpath, "w"), indent=1)


if __name__ == "__main__":
    role, N = sys.argv[1], int(sys.argv[2])
    if role == "dev":
        run([0, 1, 2], ["DEV", "LONG_HORIZON"], N, "../../artifacts/token_free/e46_dev.json")
    else:
        run([3, 4, 5], ["TEST", "OOD", "COMPOSITION", "LONG_HORIZON"], N, "../../artifacts/token_free/e46_conf.json")
