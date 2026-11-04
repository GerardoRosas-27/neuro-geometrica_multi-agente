"""Paso 2: codificador congelado PCA (blanqueado) H -> d_field, ajustado SOLO en train.
Escribe <out>/<split>_z.f32 + <split>_z.json (formato del crate Rust) y encoder.npz.
Paso 2b (decodificador lineal z -> H): ridge en train, decoder.npz."""
import argparse, os
import numpy as np
from common import load_json, save_json


def lengths(dirp, split):
    return [len(t["ids"]) for t in load_json(os.path.join(dirp, f"{split}_tokens.json"))]


def encode(H, enc):
    return ((H.astype(np.float64) - enc["mean"]) @ enc["comp"].T / enc["scale"]).astype(np.float32)


def decode(Z, dec):
    return (np.asarray(Z, np.float64) @ dec["W"] + dec["b"]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True, help="salida de extract.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-field", type=int, default=64)
    ap.add_argument("--fit-max", type=int, default=200000, help="máx. tokens de train para ajustar PCA")
    ap.add_argument("--dec-lambda", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Htr = np.load(os.path.join(a.hdir, "train_h.npy"))
    rng = np.random.default_rng(a.seed)
    sel = rng.permutation(len(Htr))[: a.fit_max]
    # PCA por covarianza acumulada en bloques (evita SVD de una matriz N x 2304 en f64)
    Hd = Htr.shape[1]
    mean = np.zeros(Hd); C = np.zeros((Hd, Hd)); n = 0
    for i in range(0, len(sel), 8192):
        Xb = Htr[np.sort(sel[i:i + 8192])].astype(np.float64)
        mean += Xb.sum(0); C += Xb.T @ Xb; n += len(Xb)
    mean /= n
    C = (C - n * np.outer(mean, mean)) / (n - 1)
    evals, evecs = np.linalg.eigh(C)
    o = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[o], 1e-12, None), evecs[:, o]
    k = min(a.d_field, Hd)
    comp = evecs[:, :k].T
    scale = np.sqrt(evals[:k])
    ev = evals / evals.sum()
    X = sel
    enc = {"mean": mean, "comp": comp, "scale": scale}
    np.savez(os.path.join(a.out, "encoder.npz"), **enc)
    Ztr = encode(Htr, enc)
    # decodificador lineal z -> H (ridge, solo train)
    Zb = np.concatenate([Ztr.astype(np.float64), np.ones((len(Ztr), 1))], 1)
    A = Zb.T @ Zb + a.dec_lambda * len(Zb) * np.eye(Zb.shape[1])
    Wb = np.linalg.solve(A, Zb.T @ Htr.astype(np.float64))
    dec = {"W": Wb[:-1], "b": Wb[-1]}
    np.savez(os.path.join(a.out, "decoder.npz"), **dec)
    info = {"d_field": k, "H": int(Htr.shape[1]), "var_explicada": float(ev[:k].sum()), "fit_tokens": int(len(X))}
    for split in ["train", "val", "test"]:
        H = Htr if split == "train" else np.load(os.path.join(a.hdir, f"{split}_h.npy"))
        Z = encode(H, enc)
        L = lengths(a.hdir, split)
        assert sum(L) == len(Z)
        Z.astype("<f4").tofile(os.path.join(a.out, f"{split}_z.f32"))
        save_json(os.path.join(a.out, f"{split}_z.json"), {"d": k, "lengths": L})
        rec = decode(Z, dec)
        Hf = H.astype(np.float32)
        info[f"{split}_recon_rel_mse_centrado"] = float(((rec - Hf) ** 2).sum() / ((Hf - mean.astype(np.float32)) ** 2).sum())
    save_json(os.path.join(a.out, "encoder_info.json"), info)
    print(info)


if __name__ == "__main__":
    main()
