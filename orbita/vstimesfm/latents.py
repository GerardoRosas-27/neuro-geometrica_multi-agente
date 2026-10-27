"""vstimesfm/latents.py — reconstruye EXACTAMENTE los latentes s_t de dinlat/run.py (misma PCA, misma Q_rand,
mismo rng para controles C1/C2) a partir de los embeddings Gemma cacheados. No re-embebe."""
import json, math, numpy as np
DIM, DL, T, TMIN = 2304, 16, 9, 2
def load(seed, ej, emb, off):
    eps = json.load(open(ej)); n = len(eps) * T
    E = np.fromfile(emb, dtype="<f4", count=n * DIM, offset=int(off) * DIM * 4).astype(np.float64)
    assert E.size == n * DIM
    sd = int(seed, 16); rng = np.random.default_rng(sd + 99)
    sp = np.array([e["split"] for e in eps]); E = E.reshape(len(eps), T, DIM)
    tr = sp == "train"; Xtr = E[tr].reshape(-1, DIM); mu = Xtr.mean(0)
    U, sv, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); P = Vt[:DL]; scale = sv[:DL].max() / math.sqrt(len(Xtr))
    Qr, _ = np.linalg.qr(rng.standard_normal((DL, DL)))
    S = (((E.reshape(-1, DIM) - mu) @ P.T / scale) @ Qr).reshape(len(eps), T, DL)
    def c1(X):
        Y = X.copy()
        for t in range(T): Y[:, t] = X[rng.permutation(len(X)), t]
        return Y
    def c2(X): return np.stack([x[rng.permutation(T)] for x in X])
    out = {"main": {"train": S[tr], "ho": S[sp == "ho"], "tf": S[sp == "tf"]}}
    for tag, f in (("C1_permutado", c1), ("C2_barajado", c2)):
        a = f(S[tr]); b = f(S[sp == "ho"]); out[tag] = {"train": a, "ho": b}
    return out
