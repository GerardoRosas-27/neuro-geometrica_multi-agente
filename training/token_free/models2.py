"""Dinámicas ciclo 2 (semántica replicada en src/token_free_field.rs::step_ctx)."""
import numpy as np, jax, jax.numpy as jnp

GATED = ("liquid", "liquid_ss", "ssm")


def init(kind, N, C, H, key):
    ks = jax.random.split(key, 4)
    I = N + C
    g = lambda k, s, fan: (jax.random.normal(k, s) / np.sqrt(fan)).astype(jnp.float32)
    if kind == "linear":
        return {"A": jnp.tile(jnp.eye(N), (C, 1, 1)) + g(ks[0], (C, N, N), N) * 0.05, "b": jnp.zeros((C, N))}
    p = {"W1": g(ks[0], (I, H), I), "b1": jnp.zeros(H), "W2": g(ks[1], (H, N), H) * 0.5, "b2": jnp.zeros(N)}
    if kind in GATED:
        p["Wg"] = g(ks[2], (I, N), I)
        p["bg"] = jnp.full((N,), 1.0 if kind == "ssm" else 0.0)
    return p


def nparams(p):
    return int(sum(np.prod(v.shape) for v in p.values()))


def topk_mask(psi, k):
    N = psi.shape[-1]
    if k is None or k >= N:
        return jnp.ones_like(psi)
    order = jnp.argsort(-jnp.abs(psi), axis=-1, stable=True)
    ranks = jnp.argsort(order, axis=-1, stable=True)
    return (ranks < k).astype(psi.dtype)


def _inner(kind, p, cur, c, m, dt):
    xm = cur * m
    x = jnp.concatenate([xm, c], -1)
    act = jnp.tanh if kind in ("liquid", "liquid_ss") else (lambda v: jax.nn.gelu(v, approximate=True))
    f = act(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    if kind == "mlp":
        new = f
    elif kind == "v4res":
        new = xm + f
    else:
        gg = x @ p["Wg"] + p["bg"]
        if kind == "ssm":
            new = jax.nn.sigmoid(gg) * xm + f
        else:
            tau = 0.5 + jax.nn.softplus(gg)
            new = xm + dt * (-xm / tau + f)
    return m * new + (1 - m) * cur


def step(kind, p, psi, c, k=None, K=4):
    m = topk_mask(psi, k)
    if kind == "linear":
        o = jnp.argmax(c, -1)
        new = jnp.einsum("bn,bnm->bm", psi * m, p["A"][o]) + p["b"][o]
        return m * new + (1 - m) * psi
    if kind == "liquid_ss":
        cur = psi
        for _ in range(K):
            cur = _inner(kind, p, cur, c, m, 1.0 / K)
        return cur
    return _inner(kind, p, psi, c, m, 1.0)


def rollout(kind, p, psi0, C, k=None, K=4):
    """C: [B, L, ctx]."""
    def f(psi, c):
        n = step(kind, p, psi, c, k, K)
        return n, n
    _, tr = jax.lax.scan(f, psi0, jnp.swapaxes(C, 0, 1))
    return jnp.swapaxes(tr, 0, 1)


def export(kind, p, N, C, seed, meta, path, decoder=None, K=4):
    import hashlib, json, platform, optax
    arrs = {"model." + k: np.asarray(v, np.float32) for k, v in p.items()}
    if decoder is not None:
        arrs["decoder.Dw"] = np.asarray(decoder[0], np.float32)
        arrs["decoder.Db"] = np.asarray(decoder[1], np.float32)
    else:  # decoder identidad (tarea v4): se exporta explícito para el runner
        arrs["decoder.Dw"] = np.eye(N, dtype=np.float32)
        arrs["decoder.Db"] = np.zeros(N, np.float32)
    h = hashlib.sha256()
    for kk, v in sorted(arrs.items()):
        h.update(kk.encode()); h.update(v.tobytes())
    art = {"schema": "tfl-v1", "kind": kind, "N": N, "ops": C, "ctx_dim": C, "substeps": K if kind == "liquid_ss" else 1,
           "seed": seed, "dtype": "f32", "layout": "row-major; x@W, W[in,out]; x=[psi*mask, ctx]",
           "params_sha256": h.hexdigest(),
           "framework": {"jax": jax.__version__, "optax": optax.__version__, "python": platform.python_version()},
           "meta": meta, "params": {kk: {"shape": list(v.shape), "data": v.ravel().tolist()} for kk, v in sorted(arrs.items())}}
    json.dump(art, open(path, "w"))
    return h.hexdigest()
