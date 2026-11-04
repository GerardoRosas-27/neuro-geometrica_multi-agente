"""Utilidades compartidas: carga de Gemma 2 (o modelo diminuto aleatorio para SMOKE), formato de chat,
hook que sustituye el estado oculto de la capa L y ejecuta las capas restantes + LM head."""
import json, os
import numpy as np
import torch

GEMMA_ID = "google/gemma-2-2b"


class ByteTok:
    """Tokenizador de bytes SOLO para smoke (el de Gemma está bajo licencia)."""
    vocab_size = 259
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, s, add_bos=True):
        return ([1] if add_bos else []) + [b + 3 for b in s.encode("utf-8")]

    def decode(self, ids):
        return bytes([i - 3 for i in ids if i >= 3]).decode("utf-8", errors="replace")


class HFTok:
    def __init__(self, tok):
        self.t = tok
        self.eos_token_id = tok.eos_token_id

    def encode(self, s, add_bos=True):
        return self.t(s, add_special_tokens=add_bos)["input_ids"]

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=True)


def load_model(model_id=GEMMA_ID, tiny_random=False, dtype="float32", device="cpu"):
    """Devuelve (model, tok, meta). Gemma 2 2B requiere aceptar la licencia y HF_TOKEN en el entorno."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, Gemma2ForCausalLM
    if tiny_random:
        torch.manual_seed(0)
        cfg = Gemma2Config(vocab_size=ByteTok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=1024,
                           sliding_window=256, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        cfg._attn_implementation = "eager"
        m = Gemma2ForCausalLM(cfg).eval()
        return m, ByteTok(), {"model": "TINY-RANDOM-SMOKE", "hidden": 64, "layers": 4}
    token = os.environ.get("HF_TOKEN")  # nunca hardcodear
    if token is None:
        print("[aviso] HF_TOKEN no definido: gemma-2-2b está bajo licencia y probablemente fallará la descarga")
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    m = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype=getattr(torch, dtype),
                                             attn_implementation="eager").to(device).eval()
    return m, HFTok(tok), {"model": model_id, "hidden": m.config.hidden_size, "layers": m.config.num_hidden_layers}


def format_conv(messages):
    """Formato plano para el modelo BASE gemma-2-2b (sin plantilla de instrucciones)."""
    out = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else ("Assistant" if msg["role"] == "assistant" else msg["role"].title())
        out.append(f"{role}: {msg['content'].strip()}")
    return "\n".join(out) + "\n"


def layers_of(model):
    return model.model.layers


@torch.no_grad()
def hidden_at(model, ids, layer):
    """Estado oculto tras la capa `layer` (hidden_states[layer]; 0 = embeddings), [T, H] float32."""
    x = torch.tensor([ids], device=model.device)
    out = model(input_ids=x, output_hidden_states=True)
    return out.hidden_states[layer][0].float()


@torch.no_grad()
def logits_with_replacement(model, ids, layer, h_repl=None, mask=None):
    """Logits [T,V] cuando la salida de la capa `layer` se sustituye por h_repl en las posiciones mask (bool [T]).
    h_repl=None -> Gemma sin tocar. Las capas > layer y el LM head se ejecutan normalmente."""
    x = torch.tensor([ids], device=model.device)
    handle = None
    if h_repl is not None:
        assert layer >= 1
        hr = torch.as_tensor(h_repl, device=model.device)
        mk = torch.as_tensor(mask, device=model.device)

        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            new = hs.clone()
            new[0, mk] = hr[mk].to(hs.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        handle = layers_of(model)[layer - 1].register_forward_hook(hook)
    try:
        out = model(input_ids=x)
    finally:
        if handle is not None:
            handle.remove()
    return out.logits[0].float()


def nll_per_token(logits, ids):
    """NLL de ids[1:] dados logits[:-1]; devuelve tensor [T-1]."""
    lp = torch.log_softmax(logits[:-1], -1)
    tgt = torch.tensor(ids[1:], device=logits.device)
    return -lp.gather(1, tgt[:, None])[:, 0]


def save_json(p, o):
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def read_z(dirp, split):
    idx = load_json(os.path.join(dirp, f"{split}_z.json"))
    z = np.fromfile(os.path.join(dirp, f"{split}_z.f32"), dtype="<f4").reshape(-1, idx["d"])
    offs = np.concatenate([[0], np.cumsum(idx["lengths"])]).astype(int)
    return z, idx["lengths"], offs
