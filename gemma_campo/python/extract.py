"""Paso 1: extraer estados ocultos de la capa media de Gemma 2 2B congelado, por token.
Split train/val/test POR CONVERSACIÓN (antes de tokenizar), semilla fija.

Dataset por defecto: HuggingFaceTB/everyday-conversations-llama3.1-2k (diálogos cortos, público, ~2.2k conv).
Tamaño por defecto pequeño: 600 conversaciones (480/60/60), máx. 384 tokens por conversación.
"""
import argparse, os, json
import numpy as np
import torch
from common import load_model, format_conv, hidden_at, save_json, GEMMA_ID


def load_convs(a):
    if a.jsonl:
        return [json.loads(l)["messages"] for l in open(a.jsonl) if l.strip()]
    from datasets import load_dataset
    ds = load_dataset(a.dataset, split=a.hf_split, token=os.environ.get("HF_TOKEN"))
    return [r["messages"] for r in ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GEMMA_ID)
    ap.add_argument("--tiny-random", action="store_true", help="SMOKE: Gemma2 diminuto aleatorio + tokenizador de bytes")
    ap.add_argument("--layer", type=int, default=13, help="hidden_states[layer]; gemma-2-2b tiene 26 capas, d=2304")
    ap.add_argument("--dataset", default="HuggingFaceTB/everyday-conversations-llama3.1-2k")
    ap.add_argument("--hf-split", default="train_sft")
    ap.add_argument("--jsonl", default=None, help='archivo local con {"messages":[{"role","content"}...]} por línea')
    ap.add_argument("--n-conv", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--frac", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    # modo continuo: held-out fijo (n-test, n-val primeros de la permutación), el resto es el flujo de train en shards
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--n-val", type=int, default=60); ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shard-size", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    convs = load_convs(a)
    rng = np.random.default_rng(a.seed)
    # dedup por hash del texto: conversaciones idénticas comparten id y caen en un solo split
    import hashlib
    hashes = [hashlib.sha256(format_conv(c).encode()).hexdigest()[:16] for c in convs]
    first = {}
    for i, h in enumerate(hashes):
        first.setdefault(h, i)
    uniq = np.array(sorted(first.values()))
    if a.stream:
        order = uniq[rng.permutation(len(uniq))]
        test, val, pool = order[:a.n_test], order[a.n_test:a.n_test + a.n_val], order[a.n_test + a.n_val:]
        sh = pool[a.shard * a.shard_size:(a.shard + 1) * a.shard_size]
        if len(sh) == 0:
            raise SystemExit(f"shard {a.shard} vacío: flujo agotado ({len(pool)} conv de train)")
        splits = {"train": sh, "val": val, "test": test} if a.shard == 0 else {"train": sh}
        n = len(sh)
    else:
        order = uniq[rng.permutation(len(uniq))][: a.n_conv]
        n = len(order)
        ntr, nva = int(round(a.frac[0] * n)), int(round(a.frac[1] * n))
        splits = {"train": order[:ntr], "val": order[ntr:ntr + nva], "test": order[ntr + nva:]}
    model, tok, meta = load_model(a.model, a.tiny_random, a.dtype, a.device)
    assert 1 <= a.layer < meta["layers"], "la capa debe dejar capas por encima para decodificar"
    for name, ids_c in splits.items():
        hs, toks = [], []
        for ci in ids_c:
            ids = tok.encode(format_conv(convs[ci]))[: a.max_tokens]
            if len(ids) < 3:
                continue
            h = hidden_at(model, ids, a.layer).cpu().numpy().astype(np.float16)
            hs.append(h)
            toks.append({"conv": int(ci), "ids": ids})
        H = np.concatenate(hs, 0)
        np.save(os.path.join(a.out, f"{name}_h.npy"), H)
        save_json(os.path.join(a.out, f"{name}_tokens.json"), toks)
        print(f"{name}: {len(toks)} conv, {H.shape[0]} tokens, H={H.shape[1]}")
    for s1 in splits:
        for s2 in splits:
            if s1 < s2:
                assert not set(splits[s1].tolist()) & set(splits[s2].tolist())
    save_json(os.path.join(a.out, "extract_meta.json"), {**meta, "layer": a.layer, "dataset": a.jsonl or a.dataset,
              "n_conv": n, "seed": a.seed, "max_tokens": a.max_tokens, "dtype": a.dtype,
              "split_conv_ids": {k: v.tolist() for k, v in splits.items()},
              "split_conv_hashes": {k: [hashes[i] for i in v] for k, v in splits.items()},
              "stream": a.stream, "shard": a.shard if a.stream else None, "shard_size": a.shard_size if a.stream else None})


if __name__ == "__main__":
    main()
