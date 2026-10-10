"""E45 bottleneck token-free."""
import json, sys, time
import numpy as np, jax, jax.numpy as jnp, optax
import data as D
from common import load, ridge, rapply, decode_ridge

DIMS = [16, 32, 64, 128, 256, 512]


def gelu(x):
    return jax.nn.gelu(x, approximate=True)


def init_enc(key, kind, N):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    p = {}
    if kind == "lin":
        p["W"] = jax.random.normal(k1, (D.HID, N)) / np.sqrt(D.HID)
    else:
        p["W1"] = jax.random.normal(k1, (D.HID, 256)) / np.sqrt(D.HID)
        p["b1"] = jnp.zeros(256)
        p["W2"] = jax.random.normal(k2, (256, N)) / 16.0
    p["b"] = jnp.zeros(N)
    p["Dw"] = jax.random.normal(k3, (N, D.S * D.V)) / np.sqrt(N)
    p["Db"] = jnp.zeros(D.S * D.V)
    p["A"] = jnp.tile(jnp.eye(N), (D.NOPS, 1, 1)) + 0.01 * jax.random.normal(k4, (D.NOPS, N, N))
    return p


def enc_apply(p, h):
    if "W" in p:
        return h @ p["W"] + p["b"]
    return gelu(h @ p["W1"] + p["b1"]) @ p["W2"] + p["b"]


def dec_logits(p, psi):
    return (psi @ p["Dw"] + p["Db"]).reshape(psi.shape[:-1] + (D.S, D.V))


def ce(logits, z):
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), z[..., None], -1))


def train_encoder(kind, N, seed, tr, steps=1500):
    key = jax.random.PRNGKey(seed)
    p = init_enc(key, kind, N)
    opt = optax.adam(2e-3)
    st = opt.init(p)
    h = jnp.asarray(tr["h"][:, :2]); z = jnp.asarray(tr["z"][:, :2]); op = jnp.asarray(tr["ops"][:, 0])

    def loss(p, hb, zb, ob):
        psi = enc_apply(p, hb)
        pred = jnp.einsum("bn,bnm->bm", psi[:, 0], p["A"][ob])
        return (ce(dec_logits(p, psi), zb) + ce(dec_logits(p, pred), zb[:, 1])
                + 0.1 * jnp.mean((pred - jax.lax.stop_gradient(psi[:, 1])) ** 2))

    @jax.jit
    def step(p, st, k):
        idx = jax.random.randint(k, (256,), 0, h.shape[0])
        l, g = jax.value_and_grad(loss)(p, h[idx], z[idx], op[idx])
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rk = jax.random.PRNGKey(seed + 1000)
    t0 = time.time()
    for i in range(steps):
        rk, k = jax.random.split(rk)
        p, st, l = step(p, st, k)
    return p, time.time() - t0


def evaluate(fe, d, splits):
    tr = d["TRAIN"]
    psi0 = fe(tr["h"][:, 0]); psi1 = fe(tr["h"][:, 1])
    Y = D.onehot(tr["z"][:, 0])
    probe = ridge(psi0, Y)
    trans = {}
    for o in range(D.NOPS):
        m = tr["ops"][:, 0] == o
        trans[o] = ridge(psi0[m], psi1[m])
    surf_probe = ridge(psi0, tr["surf"][:, 0])
    out = {}
    for s in splits:
        sp = d[s]
        x0 = fe(sp["h"][:, 0])
        dec = decode_ridge(probe, x0)
        pred = np.zeros_like(x0)
        for o in range(D.NOPS):
            m = sp["ops"][:, 0] == o
            pred[m] = rapply(trans[o], x0[m])
        rel = decode_ridge(probe, pred)
        sres = rapply(surf_probe, x0) - sp["surf"][:, 0]
        r2 = 1 - sres.var() / sp["surf"][:, 0].var()
        out[s] = {"decode_exact": float((dec == sp["z"][:, 0]).all(-1).mean()),
                  "relational_exact": float((rel == sp["z"][:, 1]).all(-1).mean()),
                  "surface_r2": float(r2)}
    return out


def run(seeds, splits, outpath):
    d, man = load()
    rows = []
    hraw = lambda x: x
    for seed in seeds:
        rows.append(dict(method="raw", N=512, seed=seed, train_s=0, **{"res": evaluate(hraw, d, splits)}))
        rng = np.random.default_rng(seed)
        h0 = d["TRAIN"]["h"][:, 0]; mu = h0.mean(0)
        U = np.linalg.svd(h0[:5000] - mu, full_matrices=False)[2]
        for N in DIMS:
            R = rng.normal(0, 1 / np.sqrt(D.HID), (D.HID, N)).astype(np.float32)
            rows.append(dict(method="random", N=N, seed=seed, train_s=0, res=evaluate(lambda x: x @ R, d, splits)))
            P = U[:N].T.astype(np.float32)
            rows.append(dict(method="pca", N=N, seed=seed, train_s=0, res=evaluate(lambda x: (x - mu) @ P, d, splits)))
            for kind in ["lin", "mlp"]:
                p, ts = train_encoder(kind, N, seed, d["TRAIN"])
                fe = jax.jit(lambda x: enc_apply(p, x))
                f = lambda x: np.asarray(fe(jnp.asarray(x)))
                rows.append(dict(method="trained_" + kind, N=N, seed=seed, train_s=ts, res=evaluate(f, d, splits)))
                if kind == "mlp":
                    np.savez(f"/workspace/ngm-tokenfree-cache/enc_mlp_N{N}_s{seed}.npz",
                             **{k: np.asarray(v) for k, v in p.items()})
            print(seed, N, json.dumps(rows[-1]["res"]), flush=True)
    json.dump({"manifest": man, "rows": rows}, open(outpath, "w"), indent=1)


if __name__ == "__main__":
    role = sys.argv[1]
    if role == "dev":
        run([0, 1, 2], ["DEV"], "../../artifacts/token_free/e45_dev.json")
    else:
        run([3, 4, 5], ["TEST", "OOD"], "../../artifacts/token_free/e45_conf.json")
