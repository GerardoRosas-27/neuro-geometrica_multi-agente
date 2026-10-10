"""E46b: tarea no lineal (tfl-gen-v2) sobre hidden real de Gemma (pre-registro ciclo 2)."""
import json, sys, time, hashlib, numpy as np, jax, jax.numpy as jnp, optax
import models2 as M, trainlib

C_DIR = "/workspace/ngm-tokenfree-cache"
S, V, NOPS, N = 3, 10, 6, 64
FORBIDDEN = {(0, 5), (5, 0), (1, 2), (3, 0)}
HID = {"mlp": 245, "v4res": 245, "liquid": 212, "liquid_ss": 212, "ssm": 212, "linear": 0}
MODELS = list(HID)


def apply_op(z, op):
    a, b, c = z
    if op == 0: a = a * b % V
    elif op == 1: b = (b * b + 1) % V
    elif op == 2: c = (a + c * c) % V
    elif op == 3: a, c = c, a
    elif op == 4: a = (a + 1) % V
    else: b = (a * c + b) % V
    return (a, b, c)


def is_ood(z):
    return z[0] == z[2]


def split(name, n, L, seed):
    r = np.random.default_rng(seed)
    Z, O = [], []
    while len(Z) < n:
        z0 = tuple(int(v) for v in r.integers(V, size=3))
        if name == "OOD":
            z0 = (z0[0], z0[1], z0[0])
        elif is_ood(z0):
            continue
        ops = r.integers(NOPS, size=L)
        bi = {(int(ops[t]), int(ops[t + 1])) for t in range(L - 1)}
        has = bool(bi & FORBIDDEN)
        if (name == "COMPOSITION") != has and name != "LONG_HORIZON":
            continue
        traj = [z0]
        for o in ops:
            traj.append(apply_op(traj[-1], int(o)))
        if name == "TRAIN" and any(is_ood(z) for z in traj):
            continue
        Z.append(traj); O.append(ops)
    Z = np.array(Z); O = np.array(O)
    T = r.integers(3, size=Z.shape[:2])  # plantilla por aparición
    idx = T * 1000 + Z[..., 0] * 100 + Z[..., 1] * 10 + Z[..., 2]
    return {"z": Z, "ops": O, "tidx": idx}


SPLITS = {"TRAIN": (20000, 4, 1100), "DEV": (2000, 4, 1200), "TEST": (2000, 4, 1300), "OOD": (2000, 4, 1400),
          "COMPOSITION": (2000, 4, 1500), "LONG_HORIZON": (1000, 32, 1600)}


def build():
    raw = np.fromfile(f"{C_DIR}/gemma_v2.f32", np.float32)
    H = raw.reshape(3000, 2304)
    d = {k: split(k, *v) for k, v in SPLITS.items()}
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    man = {"generator": "tfl-gen-v2", "gemma": "gemma-2-2b-it-Q3_K_L.gguf mean final hidden", "hidden_sha256": sha(H),
           "splits": {k: {"n": v[0], "len": v[1], "seed": v[2], "sha256": sha(np.concatenate([d[k]["z"].ravel(), d[k]["ops"].ravel(), d[k]["tidx"].ravel()]))} for k, v in SPLITS.items()},
           "train_ood_states": int(sum(is_ood(z) for t in d["TRAIN"]["z"] for z in t)),
           "train_forbidden_bigrams": int(sum((int(a), int(b)) in FORBIDDEN for o in d["TRAIN"]["ops"] for a, b in zip(o[:-1], o[1:])))}
    return H, d, man


def onehot_z(z):
    return jax.nn.one_hot(z, V)


def train_encoders(H, d, seeds):
    tr_idx = np.unique(d["TRAIN"]["tidx"].ravel())  # textos de estados TRAIN (nunca OOD)
    mu, sd = H[tr_idx].mean(0), H[tr_idx].std(0) + 1e-5
    Hn = jnp.asarray((H - mu) / sd)
    zall = np.array([[(i % 1000) // 100, (i % 100) // 10, i % 10] for i in range(3000)])
    Xi, Zi = Hn[tr_idx], jnp.asarray(zall[tr_idx])

    def init(k):
        k1, k2, k3 = jax.random.split(k, 3)
        return {"W1": jax.random.normal(k1, (2304, 256)) / 48.0, "b1": jnp.zeros(256),
                "W2": jax.random.normal(k2, (256, N)) / 16.0, "b2": jnp.zeros(N),
                "Dw": jax.random.normal(k3, (N, S * V)) / 8.0, "Db": jnp.zeros(S * V)}

    def enc(p, x):
        return jax.nn.gelu(x @ p["W1"] + p["b1"], approximate=True) @ p["W2"] + p["b2"]

    def loss(p, b):
        lg = (enc(p, Xi[b]) @ p["Dw"] + p["Db"]).reshape(-1, S, V)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(lg), Zi[b][..., None], -1))

    ps = jax.tree.map(lambda *x: jnp.stack(x), *[init(jax.random.PRNGKey(1000 + s)) for s in seeds])
    t0 = time.time()
    ps, ls = trainlib.fit_many(loss, ps, len(tr_idx), 2000, [2000 + s for s in seeds])
    t_enc = time.time() - t0
    out = []
    for i, s in enumerate(seeds):
        p = jax.tree.map(lambda a: a[i], ps)
        out.append({"psi_all": np.asarray(enc(p, Hn)), "Dw": np.asarray(p["Dw"]), "Db": np.asarray(p["Db"])})
    return out, t_enc, ls


def dec_logits(Dw, Db, psi):
    return (psi @ Dw + Db).reshape(psi.shape[:-1] + (S, V))


def run(phase):
    seeds = [0, 1, 2] if phase == "dev" else [3, 4, 5]
    splits = ["DEV", "LONG_HORIZON"] if phase == "dev" else ["TEST", "OOD", "COMPOSITION", "LONG_HORIZON"]
    H, d, man = build()
    encs, t_enc, enc_loss = train_encoders(H, d, seeds)
    rows = []
    tr = d["TRAIN"]
    for kind in MODELS:
        def loss_for(e):
            P = jnp.asarray(e["psi_all"]); Dw, Db = jnp.asarray(e["Dw"]), jnp.asarray(e["Db"])
            return P, Dw, Db
        # vmap sobre semillas: cada semilla con su encoder → apilar Ψ
        P = jnp.stack([jnp.asarray(e["psi_all"]) for e in encs]); DW = jnp.stack([jnp.asarray(e["Dw"]) for e in encs])
        DB = jnp.stack([jnp.asarray(e["Db"]) for e in encs])
        tidx, z, ops = jnp.asarray(tr["tidx"]), jnp.asarray(tr["z"]), jnp.asarray(tr["ops"])
        C = jax.nn.one_hot(ops, NOPS)

        def loss(pp, b):
            p, Pi, Dw, Db = pp["m"], pp["P"], pp["Dw"], pp["Db"]
            Pi, Dw, Db = jax.lax.stop_gradient(Pi), jax.lax.stop_gradient(Dw), jax.lax.stop_gradient(Db)
            psi = Pi[tidx[b]]
            pr = M.rollout(kind, p, psi[:, 0], C[b])
            lg = jax.nn.log_softmax(dec_logits(Dw, Db, pr))
            return jnp.mean((pr - psi[:, 1:]) ** 2) - jnp.mean(jnp.take_along_axis(lg, z[b, 1:][..., None], -1))

        ms = jax.tree.map(lambda *x: jnp.stack(x), *[M.init(kind, N, NOPS, HID[kind], jax.random.PRNGKey(s * 10 + MODELS.index(kind))) for s in seeds])
        t0 = time.time()
        out, ls = trainlib.fit_many(loss, {"m": ms, "P": P, "Dw": DW, "Db": DB}, z.shape[0], 3000, [777 + s for s in seeds])
        ts = (time.time() - t0) / len(seeds)
        for i, s in enumerate(seeds):
            p = jax.tree.map(lambda a: a[i], out["m"]); e = encs[i]
            res = {}
            for sp in splits:
                D = d[sp]; Pi = jnp.asarray(e["psi_all"])
                pr = jax.jit(lambda a, c: M.rollout(kind, p, a, c))(Pi[D["tidx"][:, 0]], jax.nn.one_hot(jnp.asarray(D["ops"]), NOPS))
                acc = np.asarray((dec_logits(e["Dw"], e["Db"], pr).argmax(-1) == D["z"][:, 1:]).all(-1).mean(0))
                hs = [1, 2, 4, 8, 16, 32] if sp == "LONG_HORIZON" else [1, 2, 3, 4]
                res[sp] = {f"h{h}": float(acc[h - 1]) for h in hs}
            D = d["LONG_HORIZON"]; Pi = jnp.asarray(e["psi_all"])
            a0 = Pi[D["tidx"][:200, 0]]; Cc = jax.nn.one_hot(jnp.asarray(D["ops"][:200, :16]), NOPS)
            dl = 1e-3 * jax.random.normal(jax.random.PRNGKey(5), a0.shape)
            A_ = M.rollout(kind, p, a0, Cc)[:, -1]; B_ = M.rollout(kind, p, a0 + dl, Cc)[:, -1]
            res["stability"] = {"perturb_amp_h16": float(jnp.mean(jnp.linalg.norm(A_ - B_, axis=-1) / jnp.linalg.norm(dl, axis=-1)))}
            path = f"{C_DIR}/e46b_{kind}_s{s}.json"
            sha = M.export(kind, p, N, NOPS, s, {"train_s": ts, "dataset": man["splits"]}, path, decoder=(e["Dw"], e["Db"]))
            np.save(f"{C_DIR}/e46b_psi_s{s}.npy", e["psi_all"])
            rows.append({"model": kind, "seed": s, "params": M.nparams(p), "train_s_per_seed": ts, "final_loss": ls[i], "params_sha256": sha, "res": res})
            print(json.dumps({"model": kind, "seed": s, "res": res}), flush=True)
    json.dump({"manifest": man, "encoder_train_s": t_enc, "encoder_loss": enc_loss, "rows": rows},
              open(f"../../artifacts/token_free/e46b_{phase}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1])
