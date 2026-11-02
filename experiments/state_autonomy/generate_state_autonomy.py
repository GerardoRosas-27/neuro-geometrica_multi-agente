#!/usr/bin/env python3
"""SA-1 dataset generator.

Generates 8-D latent dynamical systems with no geometric rule families.
The learner sees x, a 10-D support-trajectory fingerprint c, and y only in TRAIN.
"""
import argparse, hashlib, json, math, os
import numpy as np

Z, C = 8, 10


def stable_matrix(rng, scale=0.78):
    a = rng.normal(size=(Z, Z))
    eig = np.max(np.abs(np.linalg.eigvals(a)))
    return a * (scale / max(float(eig), 1e-9))


def make_mechanism(seed):
    rng = np.random.default_rng(seed)
    return {
        "A": stable_matrix(rng),
        "U": rng.normal(0, 0.20, (Z, Z)),
        "V": rng.normal(0, 0.22, (Z, Z)),
        "b": rng.normal(0, 0.08, Z),
    }


def step(m, x):
    h = np.tanh(m["V"] @ x)
    return np.tanh(0.82 * (m["A"] @ x) + 0.16 * (m["U"] @ h) + m["b"])


def fingerprint(states):
    d = np.diff(states, axis=0)
    dd = np.diff(d, axis=0) if len(d) > 1 else np.zeros((1, Z))
    # 8 dimensions of mean velocity + two scalar dynamics descriptors.
    c = np.concatenate([
        d.mean(axis=0),
        [np.linalg.norm(d, axis=1).mean(), np.linalg.norm(dd, axis=1).mean()],
    ])
    assert c.shape == (C,)
    return c


def episode(m, rng, support=5, query_h=16):
    x = rng.normal(0, 0.65, Z)
    states = [x]
    for _ in range(support + query_h):
        x = step(m, x)
        states.append(x)
    states = np.asarray(states)
    c = fingerprint(states[: support + 1])
    return states[support - 1].copy(), c, states[support:].copy()


def write_rows(path, rows):
    os.makedirs(path, exist_ok=True)
    arr = np.asarray(rows, dtype=np.float64)
    np.asarray(arr, dtype="<f8").tofile(os.path.join(path, "data.bin"))
    meta = {
        "zdim": Z,
        "cdim": C,
        "rows": int(len(rows)),
        "dtype": "float64-le",
        "layout": "x[8], c[10], y[8]",
        "sha256": hashlib.sha256(arr.astype("<f8").tobytes()).hexdigest(),
    }
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build(out, seed, n_mechanisms, episodes_per_mechanism, split_name):
    rows = []
    mechanism_ids = []
    for i in range(n_mechanisms):
        mid = seed + i * 0x9E3779B1
        m = make_mechanism(mid)
        mechanism_ids.append(int(mid))
        rng = np.random.default_rng(mid ^ 0x51A7)
        for _ in range(episodes_per_mechanism):
            x, c, future = episode(m, rng)
            rows.append(np.concatenate([x, c, future[0]]))
    write_rows(out, rows)
    with open(os.path.join(out, "mechanisms.json"), "w", encoding="utf-8") as f:
        json.dump({"split": split_name, "seed": seed, "mechanisms": mechanism_ids}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=lambda s: int(s, 0), required=True)
    ap.add_argument("--mechanisms", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    build(args.out, args.seed, args.mechanisms, args.episodes, args.split)


if __name__ == "__main__":
    main()
