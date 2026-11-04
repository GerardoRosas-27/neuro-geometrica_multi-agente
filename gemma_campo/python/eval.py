"""Paso 6: evaluación en TEST (solo test; λ, η, γ ya se eligieron en val).
Condiciones (todas sustituyen la salida de la capa L en posiciones t>=1; t=0 queda intacto):
  gemma   : Gemma sin tocar (control)
  oracle  : D(E(h_t))         -> coste del cuello de botella d_field (techo del campo)
  repeat  : D(E(h_{t-1}))     -> control "repetir el último estado"
  campo   : D(ẑ_t), ẑ_t = predicción 1-paso del campo dada z_{<=t-1}
Métricas: z-MSE 1 paso y k pasos (desde Rust zmse.json), ΔNLL/token y perplejidad, por conversación,
bootstrap pareado 10000 remuestreos, semilla 12345. Textos de muestra: argmax por condición."""
import argparse, os, json, math
import numpy as np
import torch
from common import load_model, load_json, save_json, logits_with_replacement, nll_per_token, read_z, GEMMA_ID
from encoder import decode

B, SEED = 10000, 12345


def boot(x, y=None):
    """media y IC95 bootstrap pareado de (x - y) por conversación"""
    x = np.asarray(x, float)
    dlt = x - (np.asarray(y, float) if y is not None else 0.0)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(dlt), size=(B, len(dlt)))
    bs = dlt[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"media": float(dlt.mean()), "ic95": [float(lo), float(hi)], "p_le0": float((bs <= 0).mean()), "n": int(len(dlt))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True); ap.add_argument("--zdir", required=True); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gemma", default=GEMMA_ID); ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32"); ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = load_json(os.path.join(a.hdir, "extract_meta.json"))
    L = meta["layer"]
    model, tok, mm = load_model(a.gemma, a.tiny_random, a.dtype, a.device)
    dec = dict(np.load(os.path.join(a.zdir, "decoder.npz")))
    z, lens, offs = read_z(a.zdir, "test")
    pred = np.fromfile(os.path.join(a.model_dir, "test_pred1.f32"), dtype="<f4").reshape(z.shape)
    toks = load_json(os.path.join(a.hdir, "test_tokens.json"))
    zm = load_json(os.path.join(a.model_dir, "test_zmse.json"))
    assert [len(t["ids"]) for t in toks] == list(lens)
    res = {c: [] for c in ["gemma", "oracle", "repeat", "campo"]}
    samples = []
    # sanity: sustituir por el propio h de Gemma debe dar logits idénticos (hook correcto)
    H0 = np.load(os.path.join(a.hdir, "test_h.npy"), mmap_mode="r")
    for i, t in enumerate(toks):
        ids = t["ids"]; T = len(ids)
        zc, pc = z[offs[i]:offs[i + 1]], pred[offs[i]:offs[i + 1]]
        mask = np.ones(T, bool); mask[0] = False
        rep = np.concatenate([zc[:1], zc[:-1]], 0)
        conds = {"gemma": None, "oracle": decode(zc, dec), "repeat": decode(rep, dec), "campo": decode(pc, dec)}
        lg_all = {}
        for c, hr in conds.items():
            lg = logits_with_replacement(model, ids, L, None if hr is None else torch.from_numpy(hr), mask)
            nll = nll_per_token(lg, ids)[1:]  # predicciones desde posiciones t>=1 (donde hay sustitución)
            res[c].append(float(nll.mean()))
            lg_all[c] = lg
        if i == 0:
            hs = torch.from_numpy(np.asarray(H0[offs[0]:offs[1]], np.float32))
            lg_id = logits_with_replacement(model, ids, L, hs, mask)
            res["sanity_identity_maxdiff"] = float((lg_id - lg_all["gemma"]).abs().max())
        if i < a.n_samples:
            n = min(T, 48)
            samples.append({"conv": t["conv"], "texto_real": tok.decode(ids[1:n]),
                            **{f"argmax_{c}": tok.decode(lg_all[c][:n - 1].argmax(-1).tolist()) for c in lg_all}})
    out = {"AVISO": "SMOKE (modelo diminuto aleatorio): NO son resultados" if a.tiny_random else "test split, resultados reales",
           "gemma": mm["model"], "layer": L, "n_conv_test": len(toks), "bootstrap": {"B": B, "seed": SEED},
           "sanity_identity_maxdiff": res.pop("sanity_identity_maxdiff", None)}
    out["nll_por_token_medio"] = {c: float(np.mean(v)) for c, v in res.items()}
    out["ppl"] = {c: math.exp(v) for c, v in out["nll_por_token_medio"].items()}
    out["delta_nll"] = {"campo-gemma": boot(res["campo"], res["gemma"]), "campo-repeat": boot(res["campo"], res["repeat"]),
                        "campo-oracle": boot(res["campo"], res["oracle"]), "oracle-gemma": boot(res["oracle"], res["gemma"]),
                        "repeat-gemma": boot(res["repeat"], res["gemma"])}
    cv = zm["convs"]
    out["zmse_1paso"] = {"campo": boot([c["field1"] for c in cv]), "lenta": boot([c["slow1"] for c in cv]),
                         "repeat": boot([c["repeat1"] for c in cv]),
                         "campo-repeat": boot([c["field1"] for c in cv], [c["repeat1"] for c in cv]),
                         "campo-lenta": boot([c["field1"] for c in cv], [c["slow1"] for c in cv])}
    out["zmse_k"] = {}
    for k in range(zm["kmax"]):
        ok = [c for c in cv if c["nk"][k] > 0]
        if ok:
            out["zmse_k"][k + 1] = {"campo": float(np.mean([c["fieldk"][k] for c in ok])), "repeat": float(np.mean([c["repeatk"][k] for c in ok])),
                                    "campo-repeat": boot([c["fieldk"][k] for c in ok], [c["repeatk"][k] for c in ok])}
    out["muestras"] = samples
    os.makedirs(a.out, exist_ok=True)
    save_json(os.path.join(a.out, "eval_test.json"), out)
    print(json.dumps({k: out[k] for k in ["AVISO", "sanity_identity_maxdiff", "ppl", "delta_nll"]}, indent=1))
    print("zmse_1paso:", json.dumps(out["zmse_1paso"]["campo-repeat"]))


if __name__ == "__main__":
    main()
