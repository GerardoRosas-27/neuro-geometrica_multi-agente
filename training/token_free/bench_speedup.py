"""Speedup del bucle compilado vs bucle Python (mismo modelo/datos/pasos, E46 SSM N=64 s0)."""
import time, json, numpy as np, jax, jax.numpy as jnp
import data as D
from common import load
from e45 import enc_apply, dec_logits, ce
import e46, trainlib
d, _ = load(); tr = d["TRAIN"]
enc = {k: jnp.asarray(v) for k, v in np.load("/workspace/ngm-tokenfree-cache/enc_mlp_N64_s0.npz").items()}
out = {}
for kind in ["ssm", "liquid"]:
    t0 = time.time(); p, _, l0 = e46.train(kind, enc, 64, 0, tr); t_old = time.time() - t0
    psi = enc_apply(enc, jnp.asarray(tr["h"])); z = jnp.asarray(tr["z"]); ops = jnp.asarray(tr["ops"])
    def loss(p, b):
        pr = e46.rollout(kind, p, psi[b, 0], ops[b])
        return jnp.mean((pr - psi[b, 1:]) ** 2) + ce(dec_logits(enc, pr), z[b, 1:])
    p0 = e46.init(kind, 64, jax.random.PRNGKey(0))
    t0 = time.time(); p1, l1 = trainlib.fit(loss, p0, z.shape[0], 3000, 777); jax.block_until_ready(p1); t_new = time.time() - t0
    out[kind] = {"python_loop_s": t_old, "compiled_scan_s": t_new, "speedup": t_old / t_new, "loss_old": l0, "loss_new": l1}
    print(kind, out[kind], flush=True)
json.dump(out, open("../../artifacts/token_free/speedup.json", "w"), indent=1)
