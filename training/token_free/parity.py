"""Corpus de paridad (inicios DEV, ops aleatorias semilla 9, 16 pasos) y referencia JAX."""
import json, sys, numpy as np, jax, jax.numpy as jnp
import data as D
from common import load
from e45 import enc_apply
from e46 import rollout


def corpus(kind, N, seed, path):
    d, _ = load()
    art = json.load(open(f"/workspace/ngm-tokenfree-cache/e46_{kind}_N{N}_s{seed}.json"))
    p = {k[6:]: jnp.asarray(np.array(v["data"], np.float32).reshape(v["shape"]))
         for k, v in art["params"].items() if k.startswith("model.")}
    enc = {k: jnp.asarray(v) for k, v in np.load(f"/workspace/ngm-tokenfree-cache/enc_mlp_N{N}_s{seed}.npz").items()}
    psi0 = enc_apply(enc, jnp.asarray(d["DEV"]["h"][:512, 0]))
    ops = np.random.default_rng(9).integers(D.NOPS, size=(512, 16))
    ref = rollout(kind, p, psi0, jnp.asarray(ops))
    json.dump({"psi0": np.asarray(psi0).tolist(), "ops": ops.tolist(),
               "reference": np.asarray(ref).tolist(), "slots": D.S}, open(path, "w"))


if __name__ == "__main__":
    corpus(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
