"""Paso 2: PCA blanqueado de estados de turno -> z (d), ajustado SOLO en train y congelado (SHA-256).
Escribe <out>/<split>_z.f32 + <split>_z.json en el formato del crate Rust (lengths = nº de turnos por conversación)
y <split>_roles.json."""
import argparse, os
import numpy as np
from tc_common import load_json, save_json, sha256_file


def fit(H, d):
    H = H.astype(np.float64); mean = H.mean(0)
    C = np.cov((H - mean).T) if len(H) > 1 else np.eye(H.shape[1])
    ev, V = np.linalg.eigh(np.atleast_2d(C)); o = np.argsort(ev)[::-1]; ev, V = np.clip(ev[o], 1e-12, None), V[:, o]
    k = min(d, H.shape[1], len(H) - 1)
    return {"mean": mean, "comp": V[:, :k].T, "scale": np.sqrt(ev[:k]), "var_expl": np.array(ev[:k].sum() / ev.sum())}


def encode(H, enc):
    return ((H.astype(np.float64) - enc["mean"]) @ enc["comp"].T / enc["scale"]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--d-field", type=int, default=64); ap.add_argument("--frozen", default=None, help="dir con encoder.npz ya ajustado")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.frozen:
        enc = dict(np.load(os.path.join(a.frozen, "encoder.npz")))
    else:
        enc = fit(np.load(os.path.join(a.hdir, "train_turns_h.npy")), a.d_field)
        np.savez(os.path.join(a.out, "encoder.npz"), **enc)
    encp = os.path.join(a.frozen or a.out, "encoder.npz")
    info = {"d_field": int(len(enc["scale"])), "var_explicada": float(enc["var_expl"]), "encoder_sha256": sha256_file(encp)}
    for split in ["train", "val", "test"]:
        p = os.path.join(a.hdir, f"{split}_turns_h.npy")
        if not os.path.exists(p):
            continue
        recs = load_json(os.path.join(a.hdir, f"{split}_convs.json"))
        Z = encode(np.load(p), enc); L = [len(r["turns"]) for r in recs]; assert sum(L) == len(Z)
        Z.astype("<f4").tofile(os.path.join(a.out, f"{split}_z.f32"))
        save_json(os.path.join(a.out, f"{split}_z.json"), {"d": int(Z.shape[1]), "lengths": L})
        save_json(os.path.join(a.out, f"{split}_roles.json"), [[t["role"] for t in r["turns"]] for r in recs])
        info[f"{split}_turnos"] = int(len(Z)); info[f"{split}_z_var_media"] = float(Z.var(0).mean())
    save_json(os.path.join(a.out, "encoder_info.json"), info); print(info)


if __name__ == "__main__":
    main()
