"""Paso 5: evaluación en TEST (una sola vez). Hipótesis T1–T4 + no-daño del PLAN.md.
z-MSE en turnos assistant (t>=1) por conversación; NLL/token de la respuesta real con condiciones
none / campo (adapter(ẑ_campo)) / true (adapter(z_true)) / shuf (adapter(z_true de otra conversación)) [+ mlp opcional].
Bootstrap pareado por conversación B=10000 semilla 12345."""
import argparse, os
import numpy as np
import torch
from tc_common import load_model, load_json, save_json, read_z, boot, derangement, asst_positions, state_dict_sha, GEMMA_ID
from tc_decode import Adapter, freeze, examples, response_nll, generate


def zmse_by_conv(z, P, offs, roles):
    out = []
    for c, r in enumerate(roles):
        ts = asst_positions(r)
        out.append(float(np.mean([((P[offs[c] + t] - z[offs[c] + t]) ** 2).mean() for t in ts])) if ts else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True); ap.add_argument("--zdir", required=True)
    ap.add_argument("--field-dir", required=True, help="dir del modelo Rust con test_pred1.f32")
    ap.add_argument("--base-dir", required=True, help="salida de turno_baselines.py"); ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=GEMMA_ID); ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--dtype", default="float32"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ctx-tokens", type=int, default=384); ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=60); ap.add_argument("--max-eval", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sp = a.split
    z, lens, offs = read_z(a.zdir, sp); roles = load_json(os.path.join(a.zdir, f"{sp}_roles.json"))
    preds = {"campo": np.fromfile(os.path.join(a.field_dir, f"{sp}_pred1.f32"), dtype="<f4").reshape(z.shape)}
    for k in ["repeat", "repeat_asst", "mean", "ridge", "mlp"]:
        preds[k] = np.fromfile(os.path.join(a.base_dir, f"{sp}_pred_{k}.f32"), dtype="<f4").reshape(z.shape)
    zm = {k: zmse_by_conv(z, P, offs, roles) for k, P in preds.items()}
    ok = ~np.isnan(zm["campo"])
    res = {"split": sp, "n_conv": int(ok.sum()), "zmse_media": {k: float(np.nanmean(v)) for k, v in zm.items()}}
    res["T1"] = {"campo_menos_repeat": boot(zm["campo"][ok], zm["repeat"][ok]), "campo_menos_mean": boot(zm["campo"][ok], zm["mean"][ok]),
                 "campo_menos_repeat_asst": boot(zm["campo"][ok], zm["repeat_asst"][ok]), "campo_menos_ridge": boot(zm["campo"][ok], zm["ridge"][ok])}
    res["T1"]["pasa"] = res["T1"]["campo_menos_repeat"]["ic95"][1] < 0 and res["T1"]["campo_menos_mean"]["ic95"][1] < 0
    t2 = boot(zm["campo"][ok], zm["mlp"][ok]); thr = 0.05 * float(np.mean(zm["mlp"][ok]))
    res["T2"] = {"campo_menos_mlp": t2, "umbral": thr, "pasa": t2["ic95"][1] <= thr}
    # ---- decodificación ----
    if a.device == "dml":
        import torch_directml; a.device = torch_directml.device()
    model, tok, mm = load_model(a.model, a.tiny_random, a.dtype, a.device); freeze(model)
    dev = next(model.parameters()).device; sha0 = state_dict_sha(model)
    ck = torch.load(os.path.join(a.adapter_dir, "adapter.pt"), map_location="cpu")
    ad = Adapter(**{k: ck["cfg"][k] for k in ["d", "hidden", "P", "width", "emb_rms"]}); ad.load_state_dict(ck["state"]); ad.to(dev).eval()
    recs = load_json(os.path.join(a.hdir, f"{sp}_convs.json"))
    bos = tok.t.bos_token_id if hasattr(tok, "t") else getattr(tok, "bos_token_id", None)
    ex = examples(recs, a.ctx_tokens, bos)
    if a.max_eval: ex = ex[:a.max_eval]
    # control barajado: z_true de un turno assistant de OTRA conversación (permutación sin puntos fijos sobre ejemplos de convs distintas)
    perm = derangement(len(ex))
    for i in range(len(ex)):  # reparar coincidencias de conversación
        if ex[perm[i]][0] == ex[i][0]:
            for j in range(len(ex)):
                if ex[perm[j]][0] != ex[i][0] and ex[perm[i]][0] != ex[j][0] and j != i:
                    perm[i], perm[j] = perm[j], perm[i]; break
    zrow = lambda c, t, P=None: (z if P is None else P)[offs[c] + t]
    conds = ["none", "campo", "true", "shuf", "mlp"]
    per = {k: {} for k in conds}; ident = 0.0; samples = []
    with torch.no_grad():
        for i, (c, t, ctx, resp) in enumerate(ex):
            zs = {"campo": zrow(c, t, preds["campo"]), "true": zrow(c, t), "shuf": zrow(ex[perm[i]][0], ex[perm[i]][1]), "mlp": zrow(c, t, preds["mlp"])}
            n0 = float(response_nll(model, ctx, resp))
            if i < 50: ident = max(ident, abs(n0 - float(response_nll(model, ctx, resp, use_embeds=False))))
            per["none"].setdefault(c, []).append(n0)
            softs = {k: ad(torch.tensor(v[None], dtype=torch.float32, device=dev)) for k, v in zs.items()}
            for k, s in softs.items():
                per[k].setdefault(c, []).append(float(response_nll(model, ctx, resp, s)))
            if len(samples) < a.n_samples:
                smp = {"conv": int(recs[c]["conv"]), "turno": t, "contexto_final": tok.decode(ctx[-120:]), "real": tok.decode(resp)}
                for k in ["none", "campo", "true", "shuf"]:
                    smp[k] = tok.decode(generate(model, ctx, softs.get(k), a.max_new, getattr(tok, "eos_token_id", None), tok.encode("\n", add_bos=False)[-1:]))
                samples.append(smp)
    cs = sorted(per["none"]); M = {k: np.array([np.mean(per[k][c]) for c in cs]) for k in conds}
    res["nll_media"] = {k: float(v.mean()) for k, v in M.items()}; res["ppl"] = {k: float(np.exp(v.mean())) for k, v in M.items()}
    res["n_conv_nll"] = len(cs); res["n_turnos_nll"] = len(ex)
    res["T3"] = {"campo_menos_none": boot(M["campo"], M["none"])}; res["T3"]["pasa"] = res["T3"]["campo_menos_none"]["ic95"][1] < 0
    res["T4"] = {"campo_menos_true": boot(M["campo"], M["true"]), "campo_menos_shuf": boot(M["campo"], M["shuf"]),
                 "true_menos_none": boot(M["true"], M["none"]), "mlp_menos_campo": boot(M["mlp"], M["campo"])}
    res["T4"]["pasa"] = res["T4"]["campo_menos_shuf"]["ic95"][1] < 0
    lim = 1e-4 if a.dtype == "float32" else 0.05
    res["no_dano"] = {"identidad_maxdiff_embeds_vs_ids": ident, "limite": lim, "identidad_ok": ident < lim,
                      "llm_sha": sha0, "llm_intacto_tras_eval": state_dict_sha(model) == sha0,
                      "llm_sha_adapter_train": load_json(os.path.join(a.adapter_dir, "adapter_info.json")).get("llm_sha_after")}
    res["muestras"] = samples
    save_json(os.path.join(a.out, f"eval_{sp}.json"), res)
    for k in ["zmse_media", "T1", "T2", "nll_media", "ppl", "T3", "T4", "no_dano"]:
        print(k, res[k])
    for s in samples:
        print("----", s)


if __name__ == "__main__":
    main()
