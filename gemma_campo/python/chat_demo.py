"""Demo de chat: prompt -> Gemma capas<=L -> E (PCA) -> campo predice ẑ_t + innovación token actual -> D -> capas>L + LM head -> token.
--mode campo : en cada paso el subespacio contextual de la ÚLTIMA posición se sustituye por D(ẑ).
--mode gemma : control, Gemma sola. Codificador de entrada (capas<=L) siempre es Gemma congelado."""
import argparse, os, sys
import numpy as np
import torch
from common import (load_model, load_json, hidden_at, logits_with_replacement,
                    fit_or_load_token_res, apply_token_res, GEMMA_ID)
from encoder import encode, decode
from campo import Campo


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--zdir", required=True); ap.add_argument("--model-dir", required=True); ap.add_argument("--hdir", required=True)
    ap.add_argument("--gemma", default=GEMMA_ID); ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--mode", choices=["campo", "gemma"], default="campo")
    ap.add_argument("--prompt", default="Hello, how are you today?")
    ap.add_argument("--max-new", type=int, default=40); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--k-skip", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    L = load_json(os.path.join(a.hdir, "extract_meta.json"))["layer"]
    model, tok, mm = load_model(a.gemma, a.tiny_random, a.dtype, a.device)
    enc, dec = dict(np.load(os.path.join(a.zdir, "encoder.npz"))), dict(np.load(os.path.join(a.zdir, "decoder.npz")))
    campo = Campo(a.model_dir)
    W_res = fit_or_load_token_res(model, a.hdir, a.zdir, a.model_dir) if a.mode == "campo" else None
    emb_w = model.model.embed_tokens.weight.detach().float().cpu().numpy() if a.mode == "campo" else None
    k_skip = min(a.k_skip, enc["comp"].shape[0] // 4)
    V_ctx = enc["comp"][k_skip:].T.astype(np.float32)
    ids = tok.encode(f"User: {a.prompt.strip()}\nAssistant:")
    n0 = len(ids)
    for _ in range(a.max_new):
        if a.mode == "gemma":
            lg = logits_with_replacement(model, ids, L)[-1]
        else:
            z = encode(hidden_at(model, ids, L).cpu().numpy(), enc)
            zp = apply_token_res(campo.one_step_all(z), ids, emb_w, W_res)
            hr = torch.from_numpy(decode(zp, dec))
            mask = np.zeros(len(ids), bool); mask[-1] = len(ids) > 1
            lg = logits_with_replacement(model, ids, L, hr, mask, proj_V=V_ctx)[-1]
        if a.temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(lg / a.temperature, -1), 1))
        else:
            nxt = int(lg.argmax())
        ids.append(nxt)
        if nxt == tok.eos_token_id or tok.decode(ids[n0:]).endswith("\nUser:"):
            break
    tag = "[SMOKE modelo aleatorio] " if a.tiny_random else ""
    print(f"{tag}[{a.mode}] Assistant:{tok.decode(ids[n0:])}")


if __name__ == "__main__":
    main()
