"""dinlat/gen.py — mundos sintéticos para la cadena Gemma→latente→ÓRBITA (exp/orbita-dinamica-latente).
Estado x∈[-1,1]^8 (4 «puntos» o 8 escalares). Las familias se usan SOLO para generar datos y para el
brazo con sesgo (ORB-sesgo); ningún modelo libre las ve. Uso: gen.py PHASE OUTDIR"""
import sys, os, json, zlib
import numpy as np
SEEDS = {"dev": [f"0xA4{i:02X}" for i in range(16)], "confirm": [f"0xB4{i:02X}" for i in range(16)]}
D, T = 8, 9                       # dim estado, frames por episodio
TRAIN_FAMS = ["rotar", "trasladar", "escalar", "logistica", "modsuma", "regla_signo"]
TRANSFER_FAMS = ["cizalla", "tienda", "rodar_suma"]
N_TR, N_HO, N_TF = 8, 4, 4        # episodios por familia: train, held-out, transferencia
def pts(x): return x.reshape(4, 2)
def step(f, x, p):
    if f == "rotar":
        a = p; c, s = np.cos(a), np.sin(a); return (pts(x) @ np.array([[c, -s], [s, c]]).T).ravel()
    if f == "trasladar": return np.clip(x + np.tile([p, -p / 2], 4), -1, 1)
    if f == "escalar": return np.clip(x * p, -1, 1)
    if f == "cizalla": q = pts(x).copy(); q[:, 0] = np.clip(q[:, 0] + p * q[:, 1], -1, 1); return q.ravel()
    if f == "logistica": u = (x + 1) / 2; return 2 * (p * u * (1 - u)) - 1
    if f == "tienda": u = (x + 1) / 2; return 2 * (p * np.minimum(u, 1 - u)) - 1
    if f == "modsuma": return ((x + p + 1) % 2) - 1
    if f == "regla_signo": return np.where(x > 0, x - p, -x * 0.9)
    if f == "rodar_suma": y = np.roll(x, 1); y[0] = ((y[0] + p + 1) % 2) - 1; return y
    raise ValueError(f)
PRANGE = {"rotar": (0.15, 0.6), "trasladar": (-0.08, 0.08), "escalar": (0.88, 1.08), "cizalla": (0.1, 0.35),
          "logistica": (3.6, 3.99), "tienda": (1.6, 1.99), "modsuma": (0.1, 0.5), "regla_signo": (0.1, 0.4),
          "rodar_suma": (0.1, 0.5)}
ALLF = TRAIN_FAMS + TRANSFER_FAMS
def episode(rng, f):
    p = rng.uniform(*PRANGE[f]); x = rng.uniform(-0.9, 0.9, D)
    if f in ("logistica", "tienda"): x = rng.uniform(-0.8, 0.8, D)
    X = [x]
    for _ in range(T - 1): X.append(step(f, X[-1], p))
    return np.array(X), p
def text(x): return "estado: " + " ".join(f"{v:+.2f}" for v in x)
def seed_int(s): return int(s, 16)
def gen_seed(s):
    rng = np.random.default_rng(seed_int(s) * 7919 + 1); eps = []
    for split, fams, n in (("train", TRAIN_FAMS, N_TR), ("ho", TRAIN_FAMS, N_HO), ("tf", TRANSFER_FAMS, N_TF)):
        for f in fams:
            for _ in range(n):
                X, p = episode(rng, f); eps.append({"split": split, "fam": f, "p": float(p), "X": X.round(2).tolist()})
    return eps
if __name__ == "__main__":
    ph, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    with open(f"{out}/texts.txt", "w") as ft:
        for s in SEEDS[ph]:
            eps = gen_seed(s); json.dump(eps, open(f"{out}/eps_{s}.json", "w"))
            for e in eps:
                for x in e["X"]: ft.write(text(np.array(x)) + "\n")
    print("ok", ph, sum(1 for _ in open(f"{out}/texts.txt")))
