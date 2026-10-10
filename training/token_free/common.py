import os, pickle, numpy as np
import data as D

CACHE = "/workspace/ngm-tokenfree-cache/data.pkl"


def load():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    d, m = D.build_all()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump((d, m), f)
    return d, m


def ridge(X, Y, lam=1e-2):
    X1 = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1).astype(np.float64)
    A = X1.T @ X1 + lam * len(X) * np.eye(X1.shape[1])
    return np.linalg.solve(A, X1.T @ Y.astype(np.float64)).astype(np.float32)


def rapply(W, X):
    return X @ W[:-1] + W[-1]


def decode_ridge(W, X):
    return rapply(W, X).reshape(len(X), D.S, D.V).argmax(-1)
