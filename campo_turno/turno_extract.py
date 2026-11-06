"""Paso 1: estados por TURNO. El LLM congelado lee cada conversación completa una vez; para cada turno se agrega
hidden_states[L] sobre los tokens de su contenido (mean o last). Split POR CONVERSACIÓN (dedup por hash), semilla fija."""
import argparse, os, json, hashlib
import numpy as np
import torch
from tc_common import load_model, save_json, build_conv, pool, GEMMA_ID


def load_convs(a):
    if a.jsonl:
        return [json.loads(l)["messages"] for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    from datasets import load_dataset
    return [r["messages"] for r in load_dataset(a.dataset, split=a.hf_split, token=os.environ.get("HF_TOKEN"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GEMMA_ID); ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--layer", type=int, default=13); ap.add_argument("--pool", choices=["mean", "last"], default="mean")
    ap.add_argument("--dataset", default="HuggingFaceTB/everyday-conversations-llama3.1-2k"); ap.add_argument("--hf-split", default="train_sft")
    ap.add_argument("--jsonl", default=None); ap.add_argument("--n-conv", type=int, default=2000)
    ap.add_argument("--max-turn-tokens", type=int, default=128); ap.add_argument("--frac", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    convs = load_convs(a)
    txt = lambda c: "\n".join(f"{m['role']}: {m['content'].strip()}" for m in c)
    hashes = [hashlib.sha256(txt(c).encode()).hexdigest()[:16] for c in convs]
    first = {}
    for i, h in enumerate(hashes):
        first.setdefault(h, i)
    uniq = np.array(sorted(first.values()))
    order = uniq[np.random.default_rng(a.seed).permutation(len(uniq))][: a.n_conv]
    n = len(order); ntr, nva = int(round(a.frac[0] * n)), int(round(a.frac[1] * n))
    splits = {"train": order[:ntr], "val": order[ntr:ntr + nva], "test": order[ntr + nva:]}
    for s1 in splits:
        for s2 in splits:
            if s1 < s2:
                assert not set(splits[s1].tolist()) & set(splits[s2].tolist())
    if a.device == "dml":
        import torch_directml; a.device = torch_directml.device()
    model, tok, meta = load_model(a.model, a.tiny_random, a.dtype, a.device)
    dev = next(model.parameters()).device
    assert 1 <= a.layer <= meta["layers"]
    for name, idx in splits.items():
        Hs, recs = [], []
        for ci in idx:
            ids, turns = build_conv(tok, convs[ci], a.max_turn_tokens)
            if sum(t["role"] == "assistant" for t in turns) == 0 or len(turns) < 2:
                continue
            with torch.no_grad():
                hs = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states[a.layer][0].float().cpu().numpy()
            for t in turns:
                Hs.append(pool(hs, t["content"], a.pool))
            recs.append({"conv": int(ci), "hash": hashes[ci], "ids": ids, "turns": turns})
        H = np.stack(Hs).astype(np.float32)
        np.save(os.path.join(a.out, f"{name}_turns_h.npy"), H)
        save_json(os.path.join(a.out, f"{name}_convs.json"), recs)
        print(f"{name}: {len(recs)} conv, {len(H)} turnos, H={H.shape[1]}")
    save_json(os.path.join(a.out, "extract_meta.json"), {**meta, "layer": a.layer, "pool": a.pool, "dataset": a.jsonl or a.dataset,
              "n_conv": n, "seed": a.seed, "max_turn_tokens": a.max_turn_tokens, "dtype": a.dtype, "device": str(a.device),
              "split_conv_ids": {k: v.tolist() for k, v in splits.items()},
              "split_conv_hashes": {k: [hashes[i] for i in v] for k, v in splits.items()}})


if __name__ == "__main__":
    main()
