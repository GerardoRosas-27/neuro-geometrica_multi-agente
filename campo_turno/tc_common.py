"""Utilidades del experimento campo-estado-por-turno. Reutiliza gemma_campo/python/common.py
(carga del LLM o del Gemma2 diminuto aleatorio para smoke, tokenizador de bytes, json, lectura de z)."""
import os, sys, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gemma_campo", "python"))
from common import load_model, save_json, load_json, read_z, GEMMA_ID, ByteTok  # noqa: E402,F401

B, SEED = 10000, 12345


def role_name(r):
    return "User" if r == "user" else ("Assistant" if r == "assistant" else r.title())


def build_conv(tok, messages, max_turn_tokens=128):
    """Tokeniza una conversación turno a turno (mismo formato plano que gemma_campo: 'Role: texto\\n').
    Devuelve ids y, por turno, dict(role, hdr=(s,e) cabecera 'Role:', content=(s,e) contenido ' texto\\n').
    Cada pieza se tokeniza por separado para que los límites sean exactos."""
    ids = list(tok.encode("", add_bos=True))  # BOS (o vacío si el tokenizador no tiene)
    turns = []
    for m in messages:
        if m["role"] not in ("user", "assistant"):
            continue
        h = tok.encode(f"{role_name(m['role'])}:", add_bos=False)
        c = tok.encode(" " + m["content"].strip(), add_bos=False)[:max_turn_tokens] + tok.encode("\n", add_bos=False)
        s = len(ids); ids += h; e = len(ids); ids += c
        turns.append({"role": m["role"], "hdr": [s, e], "content": [e, len(ids)], "text": m["content"].strip()})
    return ids, turns


def pool(H, span, mode="mean"):
    """H: [T, Hd]; span=(s,e) del contenido. mean = media de los tokens; last = último token."""
    s, e = span
    assert e > s, "turno vacío"
    return H[e - 1] if mode == "last" else H[s:e].mean(0)


def boot(x, y=None, b=B, seed=SEED):
    """media e IC95 bootstrap pareado (percentiles) de x - y, unidad = conversación."""
    x = np.asarray(x, float)
    d = x - (np.asarray(y, float) if y is not None else 0.0)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(b, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"media": float(d.mean()), "ic95": [float(lo), float(hi)], "p_le0": float((bs <= 0).mean()), "n": int(len(d))}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def state_dict_sha(model):
    """Hash de todos los pesos del LLM (para demostrar que no se tocan)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def ctx_features(Zc, t, ctx):
    """[z_{t-1}, z_{t-2}, ..., z_{t-ctx}] para predecir z_t (antes del inicio se repite z_0, como features.rs)."""
    return np.concatenate([Zc[max(t - 1 - c, 0)] for c in range(ctx)])


def asst_positions(roles):
    return [t for t, r in enumerate(roles) if r == "assistant" and t >= 1]


def derangement(n, seed=SEED):
    """permutación sin puntos fijos (n>=2)"""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p
