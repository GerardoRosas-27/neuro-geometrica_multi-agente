import time, json, numpy as np, jax, jax.numpy as jnp
from common import load
from e45 import enc_apply, dec_logits, ce
import e46, trainlib
d, _ = load(); tr = d["TRAIN"]
enc = {k: jnp.asarray(v) for k, v in np.load("/workspace/ngm-tokenfree-cache/enc_mlp_N64_s0.npz").items()}
psi = enc_apply(enc, jnp.asarray(tr["h"])); z = jnp.asarray(tr["z"]); ops = jnp.asarray(tr["ops"])
def loss(p, b):
    pr = e46.rollout("ssm", p, psi[b, 0], ops[b])
    return jnp.mean((pr - psi[b, 1:]) ** 2) + ce(dec_logits(enc, pr), z[b, 1:])
ps = jax.tree.map(lambda *x: jnp.stack(x), *[e46.init("ssm", 64, jax.random.PRNGKey(s)) for s in range(3)])
t0 = time.time(); p, l = trainlib.fit_many(loss, ps, z.shape[0], 3000, [0, 1, 2]); jax.block_until_ready(p); t = time.time() - t0
o = json.load(open("../../artifacts/token_free/speedup.json"))
o["ssm_vmap_3seeds"] = {"s_total": t, "s_per_seed": t / 3, "speedup_vs_python_loop_per_seed": o["ssm"]["python_loop_s"] / (t / 3), "losses": l}
print(o["ssm_vmap_3seeds"]); json.dump(o, open("../../artifacts/token_free/speedup.json", "w"), indent=1)
