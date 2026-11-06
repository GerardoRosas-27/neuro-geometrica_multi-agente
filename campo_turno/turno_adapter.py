"""Paso 4: entrena SOLO el adaptador z -> soft-prompt con el z VERDADERO de la respuesta (+ ruido). LLM congelado:
se comprueba que el hash de sus pesos no cambia. Selección de época por NLL en val (z_true)."""
import argparse, os, time
import numpy as np
import torch
from tc_common import load_model, load_json, save_json, read_z, state_dict_sha, GEMMA_ID
from tc_decode import Adapter, freeze, emb_rms, examples, response_nll


def zrow(z, offs, c, t):
    return z[offs[c] + t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True); ap.add_argument("--zdir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=GEMMA_ID); ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--dtype", default="float32"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--P", type=int, default=4); ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=8); ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--ctx-tokens", type=int, default=384); ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=300); ap.add_argument("--loss-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.device == "dml":
        import torch_directml; a.device = torch_directml.device()
    model, tok, mm = load_model(a.model, a.tiny_random, a.dtype, a.device); freeze(model)
    dev = next(model.parameters()).device
    sha0 = state_dict_sha(model)
    bos = getattr(tok, "bos_token_id", None) if not hasattr(tok, "t") else tok.t.bos_token_id
    ztr, _, otr = read_z(a.zdir, "train"); zva, _, ova = read_z(a.zdir, "val")
    extr = examples(load_json(os.path.join(a.hdir, "train_convs.json")), a.ctx_tokens, bos)
    exva = examples(load_json(os.path.join(a.hdir, "val_convs.json")), a.ctx_tokens, bos)
    if a.max_train: extr = [extr[i] for i in rng.permutation(len(extr))[:a.max_train]]
    exva = exva[:a.max_val]
    ad = Adapter(ztr.shape[1], mm["hidden"], a.P, a.width, emb_rms(model)).to(dev)
    opt = torch.optim.AdamW(ad.parameters(), a.lr, weight_decay=1e-4)
    lf = open(os.path.join(a.out, "adapter_log.txt"), "w", encoding="utf-8")
    def log(s): print(s); lf.write(s + "\n"); lf.flush()

    def val_nll():
        ad.eval(); v = []
        with torch.no_grad():
            for c, t, ctx, resp in exva:
                v.append(float(response_nll(model, ctx, resp, ad(torch.tensor(zrow(zva, ova, c, t)[None], device=dev)))))
        ad.train(); return float(np.mean(v))
    with torch.no_grad():
        base = float(np.mean([float(response_nll(model, ctx, resp)) for _, _, ctx, resp in exva]))
    best = val_nll(); log(f"train {len(extr)} ej., val {len(exva)} ej.; val NLL sin cond={base:.4f}; adaptador init (z_true)={best:.4f}; sha LLM={sha0}")
    torch.save({"cfg": ad.cfg, "state": ad.state_dict()}, os.path.join(a.out, "adapter.pt")); best_ep = 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); tot = 0.0; opt.zero_grad()
        for i, j in enumerate(rng.permutation(len(extr))):
            c, t, ctx, resp = extr[j]
            zz = torch.tensor(zrow(ztr, otr, c, t)[None], device=dev)
            if a.noise > 0: zz = zz + a.noise * torch.randn_like(zz)
            l = response_nll(model, ctx, resp, ad(zz)); (l * a.loss_scale / a.accum).backward(); tot += float(l.detach())
            if (i + 1) % a.accum == 0 or i == len(extr) - 1:
                if a.loss_scale != 1.0:
                    for p in ad.parameters():
                        if p.grad is not None: p.grad /= a.loss_scale
                opt.step(); opt.zero_grad()
        v = val_nll(); log(f"epoch {ep}: train NLL={tot / len(extr):.4f} val NLL(z_true)={v:.4f} ({time.time() - t0:.0f}s)")
        if v < best:
            best, best_ep = v, ep; torch.save({"cfg": ad.cfg, "state": ad.state_dict()}, os.path.join(a.out, "adapter.pt"))
    sha1 = state_dict_sha(model)
    log(f"mejor época={best_ep} val NLL={best:.4f}; LLM intacto={sha0 == sha1}")
    save_json(os.path.join(a.out, "adapter_info.json"), {"best_epoch": best_ep, "val_nll_true": best, "val_nll_none": base,
              "llm_sha_before": sha0, "llm_sha_after": sha1, "llm_intacto": sha0 == sha1, **vars(a), "device": str(a.device)})
    assert sha0 == sha1, "¡los pesos del LLM cambiaron!"


if __name__ == "__main__":
    main()
