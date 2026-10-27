"""vstimesfm/run_tfm.py SEED EPSJSON EMB OFFSET RESJSON OUTJSON
TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) zero-shot, univariado por dimensión latente.
Mismo contexto que ÓRBITA: un paso desde historia s_0..s_t (t=TMIN..T-2) → s_{t+1}; h3: contexto s_0..s_TMIN → 3 pasos.
VAR2_OLS: VAR(2) multivariado sin regularización (ajustado en train del mismo seed). Verifica que PERS/AR2 recalculados
coinciden con res_SEED.json de dinlat antes de escribir (garantiza mismos latentes)."""
import sys, json, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from latents import load, T, TMIN
torch.set_num_threads(int(__import__("os").environ.get("TFM_THREADS", "4")))
_M = None
def model():
    global _M
    if _M is None:
        import timesfm
        _M = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        _M.compile(timesfm.ForecastConfig(max_context=64, max_horizon=8, normalize_inputs=True, use_continuous_quantile_head=False,
                   force_flip_invariance=True, infer_is_positive=False, fix_quantile_crossing=True))
    return _M
def fc(series, h):
    pt, _ = model().forecast(horizon=h, inputs=[np.asarray(s, dtype=np.float32) for s in series]); return np.asarray(pt)[:, :h]
def tfm_errs(S):
    n, _, d = S.shape; e = np.zeros(n)
    for t in range(TMIN, T - 1):
        ser = [S[i, :t + 1, j] for i in range(n) for j in range(d)]
        p = fc(ser, 1)[:, 0].reshape(n, d); e += ((p - S[:, t + 1]) ** 2).sum(1)
    return e / (T - 1 - TMIN)
def tfm_h3(S):
    n, _, d = S.shape; p = fc([S[i, :TMIN + 1, j] for i in range(n) for j in range(d)], 3).reshape(n, d, 3)
    return np.mean([((p[:, :, h - 1] - S[:, TMIN + h]) ** 2).sum(1) for h in (1, 2, 3)], 0)
def var_fit(S, a):
    X = np.concatenate([np.concatenate([S[:, t], S[:, t - 1], np.ones((len(S), 1))], 1) for t in range(1, T - 1)])
    Y = np.concatenate([S[:, t + 1] for t in range(1, T - 1)]); R = a * np.eye(X.shape[1]); R[-1, -1] = 0
    return np.linalg.lstsq(X.T @ X + R, X.T @ Y, rcond=None)[0]
def var_errs(W, S):
    return np.mean([((np.concatenate([S[:, t], S[:, t - 1], np.ones((len(S), 1))], 1) @ W - S[:, t + 1]) ** 2).sum(1) for t in range(TMIN, T - 1)], 0)
if __name__ == "__main__":
    seed, ej, emb, off, rj, oj = sys.argv[1:7]; L = load(seed, ej, emb, off); R = json.load(open(rj)); out = {"seed": seed}
    for tag, D in L.items():
        Wa = var_fit(D["train"], 1.0); W0 = var_fit(D["train"], 0.0); out[tag] = {}
        for sp in [k for k in D if k != "train"]:
            S = D[sp]; pers = np.mean([((S[:, t] - S[:, t + 1]) ** 2).sum(1) for t in range(TMIN, T - 1)], 0)
            assert np.allclose(pers, R[tag][sp]["PERS"], rtol=1e-6, atol=1e-8), ("PERS no coincide", tag, sp)
            assert np.allclose(var_errs(Wa, S), R[tag][sp]["AR2"], rtol=1e-5, atol=1e-7), ("AR2 no coincide", tag, sp)
            r = {"TIMESFM": tfm_errs(S).tolist(), "VAR2_OLS": var_errs(W0, S).tolist()}
            if tag == "main": r["TIMESFM_h3"] = tfm_h3(S).tolist()
            out[tag][sp] = r; print(tag, sp, "ok", flush=True)
    json.dump(out, open(oj, "w"))
