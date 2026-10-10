"""H2H token-free sobre la tarea sellada de v4 (pre-registro ciclo 2)."""
import json, sys, time, subprocess, numpy as np, jax, jax.numpy as jnp, optax
import models2 as M

C_DIR = "/workspace/ngm-tokenfree-cache"
RUNNER = "../../target/release/run_token_free_field"
MODELS = {"mlp": 27, "liquid_ss": 19, "ssm": 19}
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
ITER = (1, 2)


def apply_point(f, p, x, y):
    if f == 0: return x + .5 * p[0], y + .5 * p[1]
    if f == 1:
        t = .6 * p[0] + .15 * p[1]; c, s = np.cos(t), np.sin(t); return c * x - s * y, s * x + c * y
    if f == 2:
        a = .8 * p[0] + .2 * p[1]; c, s = np.cos(2 * a), np.sin(2 * a); return c * x + s * y, s * x - c * y
    if f == 3: return np.exp(.3 * p[0]) * x, np.exp(.3 * p[1]) * y
    if f == 4: return x + .5 * p[0] * y, y + .3 * p[1] * x
    if f == 5: return x + .3 * (p[0] * x + p[1] * y) + .3 * p[1], y + .3 * (-p[1] * x + p[0] * y) + .3 * p[0]
    if f == 6:
        t = .6 * p[0]; sc = np.exp(.3 * p[1]); c, s = np.cos(t), np.sin(t); return sc * (c * x - s * y), sc * (s * x + c * y)
    r2 = x * x + y * y; g = 1 + .3 * p[0] * np.tanh(r2 - 1)
    return g * x + .2 * p[1] * np.sin(y), g * y + .2 * p[1] * np.sin(x)


def apply_rule(f, p, o):
    o = np.asarray(o, np.float64); out = np.empty_like(o)
    out[0::2], out[1::2] = apply_point(f, p, o[0::2], o[1::2])
    return out


def correct(z, y):
    r = np.linalg.norm(np.asarray(z, np.float64) - y) / max(np.linalg.norm(y), 1e-9)
    return bool(np.isfinite(r) and r < 0.05)


def train(kind, H, X, Cx, Y, seed, arm):
    p = M.init(kind, 12, 10, H, jax.random.PRNGKey(seed % (2**31)))
    opt = optax.adam(3e-3); st = opt.init(p)
    X, Cx, Y = map(jnp.asarray, (X, Cx, Y)); n = X.shape[0]

    def loss(p, b):
        return jnp.mean((M.step(kind, p, X[b], Cx[b]) - Y[b]) ** 2)

    def one(carry, k):
        p, st = carry
        k1, k2, k3 = jax.random.split(k, 3)
        b = jax.random.randint(k1, (16,), 0, n)
        if arm == "B":  # 50 % del almacén episódico (mismo contenido que TRAIN)
            b2 = jax.random.randint(k2, (16,), 0, n)
            b = jnp.where(jax.random.uniform(k3, (16,)) < 0.5, b2, b)
        l, g = jax.value_and_grad(loss)(p, b)
        u, st = opt.update(g, st, p)
        return (optax.apply_updates(p, u), st), l

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(seed % (2**31) + 7), 8000)
    (p, st), ls = jax.jit(lambda p, st, k: jax.lax.scan(one, (p, st), k))(p, st, keys)
    jax.block_until_ready(p)
    return p, time.time() - t0


def rust(art, corpus, dump):
    o = subprocess.run([RUNNER, art, corpus, "5", "--dump", dump], capture_output=True, text=True, check=True)
    return json.loads(o.stdout), json.load(open(dump))


def run(phase):
    data = json.load(open(f"{C_DIR}/h2h_v4_{phase}.json"))
    rows = []
    for sd in data:
        seed = sd["seed"]
        tr, te = sd["train"], sd["test"]
        X = np.array([e["x"] for e in tr], np.float32); Cx = np.array([e["c"] for e in tr], np.float32)
        Y = np.array([e["y"] for e in tr], np.float32)
        Xt = np.array([e["x"] for e in te], np.float32); Ct = np.array([e["c"] for e in te], np.float32)
        # verificación del re-implementado apply_rule contra targets v4
        assert all(np.allclose(apply_rule(e["fam"], e["p"], e["x"]), e["y"], atol=1e-9) for e in te)
        for kind, H in MODELS.items():
            for arm in ("A", "B"):
                p, ts = train(kind, H, X, Cx, Y, seed, arm)
                tag = f"{C_DIR}/h2h_{phase}_{kind}_{arm}_{seed}"
                sha = M.export(kind, p, 12, 10, seed, {"phase": phase, "arm": arm, "train_s": ts}, tag + ".json")
                # corpus rollout 64 pasos (contexto repetido) y composición 2 pasos
                Cr = np.repeat(Ct[:, None], 64, 1)
                ref = np.asarray(M.rollout(kind, p, jnp.asarray(Xt), jnp.asarray(Cr)))
                json.dump({"psi0": Xt.tolist(), "ctx": Cr.tolist(), "reference": ref.tolist(), "slots": 0},
                          open(tag + "_roll.corpus", "w"))
                Cc = np.stack([Ct[[i for i, _ in sd["comp"]]], Ct[[j for _, j in sd["comp"]]]], 1)
                X0c = Xt[[i for i, _ in sd["comp"]]]
                refc = np.asarray(M.rollout(kind, p, jnp.asarray(X0c), jnp.asarray(Cc)))
                json.dump({"psi0": X0c.tolist(), "ctx": Cc.tolist(), "reference": refc.tolist(), "slots": 0},
                          open(tag + "_comp.corpus", "w"))
                rep, traj = rust(tag + ".json", tag + "_roll.corpus", tag + "_roll.dump")
                repc, trajc = rust(tag + ".json", tag + "_comp.corpus", tag + "_comp.dump")
                acc_h1 = np.mean([correct(traj[q][0], np.asarray(te[q]["y"])) for q in range(len(te))])
                it = [q for q in range(len(te)) if te[q]["fam"] in ITER]
                roll = []
                for h in HORIZONS:
                    ok = 0
                    for q in it:
                        y = np.asarray(te[q]["x"], np.float64)
                        for _ in range(h): y = apply_rule(te[q]["fam"], te[q]["p"], y)
                        ok += correct(traj[q][h - 1], y)
                    roll.append(ok / len(it))
                comp = np.mean([correct(trajc[q][1], apply_rule(te[j]["fam"], te[j]["p"], te[i]["y"]))
                                for q, (i, j) in enumerate(sd["comp"])])
                parity = {"roll_step1": rep["step1"], "roll_step64": rep["step64"], "comp_step2": repc["step2"]}
                rows.append({"seed": seed, "model": kind, "arm": arm, "params": M.nparams(p), "train_s": ts,
                             "sha256": sha, "acc_h1": float(acc_h1), "roll_iterable": roll, "comp2": float(comp),
                             "lat_us": rep["latency_us_step"], "memory_queries": rep["memory_queries"], "parity": parity})
                print(json.dumps({k: rows[-1][k] for k in ("seed", "model", "arm", "acc_h1", "comp2", "train_s")}), flush=True)
    json.dump(rows, open(f"../../artifacts/token_free/h2h_tf_{phase}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1])
