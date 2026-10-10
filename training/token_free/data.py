"""Generador tfl-gen-v1: tarea sintética de slots con encoder congelado proxy."""
import hashlib, json
import numpy as np

GEN_VERSION = "tfl-gen-v1"
S, V, NOPS, NSURF, HID = 6, 10, 8, 16, 512
FORBIDDEN = {(6, 7), (7, 6), (0, 6), (6, 0), (3, 7)}


def apply_op(z, op):
    z = z.copy()
    if op < 6:
        z[..., op] = (z[..., op] + 1) % V
    elif op == 6:
        z = np.roll(z, -1, axis=-1)
    else:
        z = z[..., [1, 0, 3, 2, 5, 4]]
    return z


def is_ood(z):
    return z[..., 0] == z[..., 2]


def onehot(z):
    out = np.zeros(z.shape[:-1] + (S * V,), np.float32)
    idx = np.arange(S) * V + z
    np.put_along_axis(out, idx, 1.0, axis=-1)
    return out


class FrozenEncoder:
    """Proxy fijo de un encoder lingüístico congelado (semilla fija 12345)."""

    def __init__(self):
        r = np.random.default_rng(12345)
        self.w1 = r.normal(0, 1 / np.sqrt(S * V), (S * V, 256)).astype(np.float32)
        self.ws = r.normal(0, 1 / np.sqrt(NSURF), (NSURF, 256)).astype(np.float32)
        self.w2 = r.normal(0, 1 / np.sqrt(256), (256, HID)).astype(np.float32)

    def __call__(self, z, surf, noise_rng):
        a = onehot(z) @ self.w1 + 0.7 * surf @ self.ws
        a = a * 0.5 * (1 + np.tanh(0.79788456 * (a + 0.044715 * a**3)))
        h = np.tanh(a @ self.w2)
        return (h + noise_rng.normal(0, 0.05, h.shape)).astype(np.float32)


def _sample_ops(rng, L, mode):
    ops = np.zeros(L, np.int64)
    while True:
        for t in range(L):
            ops[t] = rng.integers(NOPS)
        bigr = {(ops[t], ops[t + 1]) for t in range(L - 1)}
        has = bool(bigr & FORBIDDEN)
        if mode == "clean" and not has:
            return ops.copy()
        if mode == "comp" and has:
            return ops.copy()
        if mode == "any":
            return ops.copy()


def make_split(name, n, L, seed, enc):
    rng = np.random.default_rng(seed)
    zs, opss = [], []
    while len(zs) < n:
        z0 = rng.integers(V, size=S)
        mode = {"COMPOSITION": "comp", "LONG_HORIZON": "any"}.get(name, "clean")
        if name == "OOD":
            z0[2] = z0[0]
        elif is_ood(z0):
            continue
        ops = _sample_ops(rng, L, mode)
        traj = [z0]
        for op in ops:
            traj.append(apply_op(traj[-1], op))
        traj = np.stack(traj)
        if name == "TRAIN" and is_ood(traj).any():
            continue
        zs.append(traj)
        opss.append(ops)
    z = np.stack(zs)
    surf = rng.normal(0, 1, (n, L + 1, NSURF)).astype(np.float32)
    h = enc(z, surf, rng)
    return {"z": z, "ops": np.stack(opss), "surf": surf, "h": h}


SPLITS = {  # name: (n, L, base seed)
    "TRAIN": (20000, 4, 100),
    "DEV": (2000, 4, 200),
    "TEST": (2000, 4, 300),
    "OOD": (2000, 4, 400),
    "COMPOSITION": (2000, 4, 500),
    "LONG_HORIZON": (1000, 64, 600),
}


def sha(d):
    m = hashlib.sha256()
    for k in sorted(d):
        m.update(k.encode())
        m.update(np.ascontiguousarray(d[k]).tobytes())
    return m.hexdigest()


def build_all():
    enc = FrozenEncoder()
    data = {k: make_split(k, n, L, s, enc) for k, (n, L, s) in SPLITS.items()}
    manifest = {
        "generator": GEN_VERSION,
        "slots": S, "values": V, "ops": NOPS, "surface_vars": NSURF, "hidden": HID,
        "frozen_encoder_seed": 12345, "noise_sigma": 0.05,
        "ood_rule": "z0 == z2", "forbidden_bigrams": sorted(FORBIDDEN),
        "splits": {k: {"n": n, "len": L, "seed": s, "sha256": sha(data[k])}
                   for k, (n, L, s) in SPLITS.items()},
        "leak_check": {
            "train_states_in_ood_region": int(is_ood(data["TRAIN"]["z"]).sum()),
            "train_forbidden_bigrams": int(sum(
                (a, b) in FORBIDDEN for o in data["TRAIN"]["ops"] for a, b in zip(o[:-1], o[1:]))),
            "train_dev_identical_start_states_frac": float(np.mean(
                np.isin(_key(data["DEV"]["z"][:, 0]), _key(data["TRAIN"]["z"][:, 0])))),
        },
    }
    return data, manifest


def _key(z):
    return (z * (V ** np.arange(S))).sum(-1)


if __name__ == "__main__":
    _, m = build_all()
    print(json.dumps(m, indent=1))
