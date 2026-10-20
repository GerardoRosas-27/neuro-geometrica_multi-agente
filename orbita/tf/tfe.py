"""ÓRBITA-TF: Transformer sobre estados continuos (puntos 2-D como elementos, sin tokens de lenguaje) + energía libre.
Brazos: TFE (latente gaussiano q(mu|c), F = error de predicción + KL; inferencia iterativa en test), TF (mismo, mu determinista, sin KL),
MLP (control). Prerregistro: docs/prereg_orbita_transformer_fe.md. Entrenamiento con la receta de ÓRBITA (train_orbita.py)."""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_orbita import load, KS
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
D, HEADS, FF, ML = 12, 2, 44, 8      # ancho, cabezas, FFN, dim. del latente mu

class TFNet(torch.nn.Module):
    """Elementos de atención: 6 puntos (x,y) + 1 elemento latente mu. Una capa pre-LN; salida y = x + Δ por punto."""
    def __init__(s, fe=True, cdim=10):
        super().__init__(); s.fe = fe
        s.pin = torch.nn.Linear(2, D); s.min = torch.nn.Linear(ML, D)
        s.ln1 = torch.nn.LayerNorm(D); s.qkv = torch.nn.Linear(D, 3 * D); s.o = torch.nn.Linear(D, D)
        s.ln2 = torch.nn.LayerNorm(D); s.f1 = torch.nn.Linear(D, FF); s.f2 = torch.nn.Linear(FF, D)
        s.out = torch.nn.Linear(D, 2)
        s.p1 = torch.nn.Linear(cdim, 24); s.p2 = torch.nn.Linear(24, 2 * ML if fe else ML)
        with torch.no_grad(): s.out.weight.mul_(0.1); s.out.bias.zero_()
    def n_params(s): return sum(p.numel() for p in s.parameters())
    def prior(s, c):
        o = s.p2(torch.tanh(s.p1(c)))
        if s.fe: return o[:, :ML], o[:, ML:].clamp(-6, 2)   # media, log sigma
        return o, None
    def step(s, x, mu):                              # x [B,12], mu [B,ML]
        B = x.shape[0]; q = x.view(B, -1, 2)
        h = torch.cat([s.pin(q), s.min(mu)[:, None, :]], 1)       # [B,7,D]
        a = s.ln1(h); Q, K, V = s.qkv(a).chunk(3, -1)
        sp = lambda t: t.view(B, -1, HEADS, D // HEADS).transpose(1, 2)
        att = torch.softmax(sp(Q) @ sp(K).transpose(-1, -2) / math.sqrt(D // HEADS), -1) @ sp(V)
        h = h + s.o(att.transpose(1, 2).reshape(B, -1, D))
        h = h + s.f2(F.gelu(s.f1(s.ln2(h))))
        return (q + s.out(h[:, :-1])).reshape(B, -1)
    def kl(s, m, ls):                                # KL(N(m,σ²) || N(0,I))
        return 0.5 * (m * m + torch.exp(2 * ls) - 1 - 2 * ls).sum(1)

class MLPNet(torch.nn.Module):
    """Control: MLP por punto (como Dφ/M3) sobre [q, c]; y = q + Δ."""
    def __init__(s, cdim=10, h=44):
        super().__init__(); s.fe = False
        s.l1 = torch.nn.Linear(2 + cdim, h); s.l2 = torch.nn.Linear(h, h); s.l3 = torch.nn.Linear(h, 2)
        with torch.no_grad(): s.l3.weight.mul_(0.1); s.l3.bias.zero_()
    def n_params(s): return sum(p.numel() for p in s.parameters())
    def prior(s, c): return c, None                   # "mu" = c
    def step(s, x, c):
        B = x.shape[0]; q = x.view(B, -1, 2); cc = c[:, None, :].expand(B, q.shape[1], c.shape[1])
        return (q + s.l3(torch.tanh(s.l2(torch.tanh(s.l1(torch.cat([q, cc], -1))))))).reshape(B, -1)

def make(arm):
    return {"TFE": lambda: TFNet(fe=True), "TF": lambda: TFNet(fe=False), "MLP": lambda: MLPNet()}[arm]()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True); ap.add_argument("--steps", type=int, default=256000)
    ap.add_argument("--beta", type=float, default=1e-3); ap.add_argument("--lam_h", type=float, default=0.01)
    ap.add_argument("--ckpt_every", type=int, default=10000)
    a = ap.parse_args(); seed = int(a.seed, 16)
    meta, (tx, tc, ty), (rx, rc, ry, rw) = load(a.data)
    torch.manual_seed(seed ^ 0x7F); model = make(a.arm)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed ^ 0x7A); gz = torch.Generator().manual_seed(seed ^ 0x2E)
    rprob = rw / rw.sum(); any_t = bool((rw > 1e-12).any()); ck = a.out + ".ckpt"; start = 0
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"]); gz.set_state(st["gz"]); start = st["step"]
    t0 = time.time(); bs = 32
    for step in range(start, a.steps):
        t = step / a.steps
        for g in opt.param_groups: g["lr"] = 3e-5 + 0.5 * (3e-3 - 3e-5) * (1 + math.cos(math.pi * t))
        isrep = (torch.rand(bs, generator=gen) < meta["replay_fraction"]) & any_t; nrep = int(isrep.sum())
        it = torch.randint(0, tx.shape[0], (bs - nrep,), generator=gen)
        ir = torch.multinomial(rprob, nrep, replacement=True, generator=gen) if nrep else torch.zeros(0, dtype=torch.long)
        x = torch.cat([tx[it], rx[ir]]); c = torch.cat([tc[it], rc[ir]]); y = torch.cat([ty[it], ry[ir, 0]])
        m, ls = model.prior(c)
        mu = m + torch.exp(ls) * torch.randn(m.shape, generator=gz) if model.fe else m
        yh = model.step(x, mu)
        l1 = (((yh - y) ** 2).sum(1) / (y * y).sum(1).clamp_min(1e-6)).mean()
        dp = (yh - y).view(-1, 2); lb = (dp.mean(0) ** 2).sum() / (y.view(-1, 2) ** 2).sum(1).mean()
        lk = model.kl(m, ls).mean() if model.fe else torch.zeros(())
        lh = torch.zeros(())
        if a.lam_h > 0 and nrep:
            sl = slice(bs - nrep, bs); z = x[sl]; prev = 0
            for j, k in enumerate(KS):
                for _ in range(k - prev): z = model.step(z, mu[sl])
                prev = k; yk = ry[ir, j]
                lh = lh + 0.9 ** k * (((z - yk) ** 2).sum(1) / (yk * yk).sum(1).clamp_min(1e-6)).clamp(max=10.0).mean() * nrep / bs
        loss = l1 + a.lam_h * lh + lb + a.beta * lk
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % a.ckpt_every == 0 or step + 1 == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "gz": gz.get_state(), "step": step + 1}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            print(f"step {step+1} l1={float(l1):.3e} lh={float(lh):.3e} kl={float(lk):.3e} {time.time()-t0:.0f}s", flush=True)
    torch.save({"arm": a.arm, "state": model.state_dict(), "n_params": model.n_params(), "train": vars(a)}, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    print("done", model.n_params(), flush=True)

if __name__ == "__main__": main()
