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


class _StopForward(Exception):
    pass


@torch.no_grad()
def hidden_at(model, ids, layer):
    """Estado oculto tras la capa `layer` (hidden_states[layer]; 0 = embeddings), [T, H] float32."""
    x = torch.tensor([ids], device=model.device)
    if layer >= 1:
        captured = []
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured.append(hs[0].detach().float().cpu())
            raise _StopForward()
        h = layers_of(model)[layer - 1].register_forward_hook(hook)
        try:
            model(input_ids=x)
        except _StopForward:
            pass
        finally:
            h.remove()
        return captured[0]
    out = model(input_ids=x, output_hidden_states=True)
    return out.hidden_states[layer][0].float()


@torch.no_grad()
def logits_with_replacement(model, ids, layer, h_repl=None, mask=None, proj_V=None):
    """Logits [T,V] cuando la salida de la capa `layer` se sustituye por h_repl en las posiciones mask (bool [T]).
    Si proj_V ([H, k] ortonormal) se especifica, reemplaza únicamente el subespacio contextual V del campo
    preservando el complemento ortogonal local del token actual: h_new = h + ((h_repl - h) @ V) @ V.T.
    h_repl=None -> Gemma sin tocar. Las capas > layer y el LM head se ejecutan normalmente."""
    x = torch.tensor([ids], device=model.device)
    handle = None
    if h_repl is not None:
        assert layer >= 1
        hr_np = h_repl.detach().float().cpu().numpy() if isinstance(h_repl, torch.Tensor) else np.asarray(h_repl, np.float32)
        mk_np = mask.detach().cpu().numpy().astype(bool) if isinstance(mask, torch.Tensor) else np.asarray(mask, bool)
        V_np = np.asarray(proj_V, np.float32) if proj_V is not None else None

        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            new_cpu = hs[0].detach().float().cpu().numpy().copy()
            if V_np is not None:
                diff = hr_np[mk_np] - new_cpu[mk_np]
                new_cpu[mk_np] = new_cpu[mk_np] + (diff @ V_np) @ V_np.T
            else:
                new_cpu[mk_np] = hr_np[mk_np]
            new = torch.from_numpy(new_cpu).unsqueeze(0).to(device=hs.device, dtype=hs.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        handle = layers_of(model)[layer - 1].register_forward_hook(hook)
    try:
        out = model(input_ids=x)
    finally:
        if handle is not None:
            handle.remove()
    return out.logits[0].float().cpu()


def fit_or_load_token_res(model, hdir, zdir, model_dir, lam=5.0):
    """Ajusta (solo en train) o carga W_res: innovación causal del token actual e(x_t) -> r_t = z_t - ẑ_t^ctx."""
    p = os.path.join(model_dir, "token_res.npz")
    if os.path.exists(p):
        return np.load(p)["W_res"]
    from campo import Campo
    campo = Campo(model_dir)
    emb_w = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    Hd = emb_w.shape[1]
    F_tok = Hd + 1
    XtX = np.zeros((F_tok, F_tok), dtype=np.float64)
    XtY = None
    # Si hdir pertenece a shards/0/h de un run continuo, acumula todos los shards de train disponibles
    shards_root = os.path.dirname(os.path.dirname(os.path.abspath(hdir)))
    pairs = []
    if os.path.basename(os.path.dirname(os.path.abspath(hdir))) == "0" and os.path.basename(shards_root) == "shards":
        s = 0
        while True:
            sh_h = os.path.join(shards_root, str(s), "h")
            sh_z = os.path.join(shards_root, str(s), "z")
            if os.path.exists(os.path.join(sh_h, "train_tokens.json")) and os.path.exists(os.path.join(sh_z, "train_z.f32")):
                pairs.append((sh_h, sh_z))
                s += 1
            else:
                break
    if not pairs:
        pairs = [(hdir, zdir)]
    for sh_h, sh_z in pairs:
        toks = load_json(os.path.join(sh_h, "train_tokens.json"))
        Z, lens, offs = read_z(sh_z, "train")
        if XtY is None:
            XtY = np.zeros((F_tok, Z.shape[1]), dtype=np.float64)
        P = np.vstack([campo.one_step_all(Z[offs[i]:offs[i + 1]]) for i in range(len(lens))]).astype(np.float64)
        ids = np.concatenate([t["ids"] for t in toks])
        mask = np.ones(len(ids), bool)
        mask[offs[:-1]] = False
        R = Z[mask].astype(np.float64) - P[mask]
        Eb = np.hstack([emb_w[ids[mask]].astype(np.float64), np.ones((int(mask.sum()), 1))])
        XtX += Eb.T @ Eb
        XtY += Eb.T @ R
    W_res = np.linalg.solve(XtX + lam * np.eye(F_tok), XtY).astype(np.float32)
    np.savez(p, W_res=W_res, lam=np.float32(lam), n_shards=np.int32(len(pairs)))
    return W_res


def apply_token_res(pc, ids, emb_w, W_res):
    """Suma la innovación causal del token actual x_t (t>=1) a la predicción contextual pc."""
    T = len(ids)
    ev = emb_w[ids].astype(np.float32)
    ev_b = np.hstack([ev, np.ones((T, 1), dtype=np.float32)])
    corr = ev_b @ W_res
    corr[0] = 0.0
    return (pc + corr).astype(np.float32)


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
