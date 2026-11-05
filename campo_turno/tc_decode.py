"""Decodificador: adaptador z -> P vectores de soft-prompt prefijados a los embeddings. El LLM queda congelado."""
import torch


class Adapter(torch.nn.Module):
    def __init__(self, d, hidden, P=4, width=512, emb_rms=1.0):
        super().__init__()
        self.P, self.H = P, hidden
        self.net = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.GELU(), torch.nn.Linear(width, P * hidden))
        torch.nn.init.normal_(self.net[2].weight, std=0.02); torch.nn.init.zeros_(self.net[2].bias)
        self.register_buffer("emb_rms", torch.tensor(float(emb_rms)))
        self.cfg = {"d": d, "hidden": hidden, "P": P, "width": width, "emb_rms": float(emb_rms)}

    def forward(self, z):  # z [B,d] -> [B,P,H]
        return self.net(z).view(-1, self.P, self.H) * self.emb_rms


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def emb_rms(model):
    E = model.get_input_embeddings().weight.detach().float()
    return float(E.pow(2).mean().sqrt())


def examples(recs, ctx_tokens=384, bos=None):
    """Por cada turno assistant t>=1: (conv_idx, t, ids_contexto (incluye 'Assistant:'), ids_respuesta)."""
    out = []
    for c, r in enumerate(recs):
        ids = r["ids"]
        for t, tu in enumerate(r["turns"]):
            if tu["role"] != "assistant" or t < 1:
                continue
            he = tu["hdr"][1]; cs, ce = tu["content"]
            lo = max(0, he - ctx_tokens)
            ctx = ids[lo:he]
            if lo > 0 and bos is not None:
                ctx = [bos] + ctx[1:]
            out.append((c, t, ctx, ids[cs:ce]))
    return out


def response_nll(model, ctx, resp, soft=None, use_embeds=True):
    """NLL media/token de resp dado [soft] + ctx. soft: [1,P,H] o None. Devuelve tensor escalar (con grad si soft lo tiene)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([ctx + resp], device=dev)
    if use_embeds:
        e = model.get_input_embeddings()(ids)
        if soft is not None:
            e = torch.cat([soft.to(e.dtype), e], 1)
        logits = model(inputs_embeds=e).logits[0]
    else:
        assert soft is None
        logits = model(input_ids=ids).logits[0]
    off = 0 if soft is None else soft.shape[1]
    n = len(resp); start = off + len(ctx)
    lp = torch.log_softmax(logits[start - 1:start - 1 + n].float(), -1)
    return -lp.gather(1, torch.tensor(resp, device=dev)[:, None]).mean()


@torch.no_grad()
def generate(model, ctx, soft=None, max_new=60, eos=None, stop_ids=()):
    """greedy sin caché (simple y portable a directml)."""
    dev = next(model.parameters()).device
    ids = list(ctx); out = []
    for _ in range(max_new):
        e = model.get_input_embeddings()(torch.tensor([ids], device=dev))
        if soft is not None:
            e = torch.cat([soft.to(e.dtype), e], 1)
        nx = int(model(inputs_embeds=e).logits[0, -1].argmax())
        if nx == eos or nx in stop_ids:
            break
        ids.append(nx); out.append(nx)
    return out
