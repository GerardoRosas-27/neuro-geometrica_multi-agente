"""ÓRBITA — entrenamiento (PyTorch, float64, 1 hilo, determinista por seed).
Receta fija (preregistro docs/preregistro_orbita.md): mismo presupuesto que v4 §16 E3:
256 000 pasos Adam, batch 32, lr coseno 3e-3→3e-5, pérdida relativa, 50 % replay del teacher.
Uso: python train_orbita.py --data DIR --out W.json --seed 0xA400 [--steps N] [--lam_h 0.1]
"""
import argparse, json, math, time, os, sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
M, NB, NT = 8, 4, 2
NSKEW = M * (M - 1) // 2
KS = [1, 2, 4, 8]


class Orbita(torch.nn.Module):
    def __init__(s, cdim=10, hh=24, hg=16, kc=16, r=2, eta=0.5):
        super().__init__()
        s.hh, s.hg, s.kc, s.r, s.eta = hh, hg, kc, r, eta
        s.nout = NB + NB + NSKEW + M + kc + kc
        s.l1 = torch.nn.Linear(cdim, hh)
        s.l2 = torch.nn.Linear(hh, s.nout)
        s.l3 = torch.nn.Linear(2 + cdim, hg)
        s.l4 = torch.nn.Linear(hg, M - 2)
        s.wc = torch.nn.Parameter(torch.randn(kc, M) * (1.0 / math.sqrt(M)))
        with torch.no_grad():
            s.l2.weight.mul_(0.1); s.l2.bias.zero_()
            s.l2.bias[-kc:] = -4.0          # a_k ≈ softplus(-4): corrector pequeño al inicio
            s.l4.weight.mul_(0.1); s.l4.bias.zero_()
        iu = torch.triu_indices(M, M, 1)
        s.register_buffer("iu0", iu[0]); s.register_buffer("iu1", iu[1])

    def n_params(s):
        return sum(p.numel() for p in s.parameters())

    def ctx(s, c):                                   # c: [B,10]
        o = s.l2(torch.tanh(s.l1(c)))
        k = 0
        om = o[:, k:k + NB]; k += NB
        sv = o[:, k:k + NB]; k += NB
        rho = torch.cat([torch.exp(sv[:, :NT]), torch.exp(-F.softplus(sv[:, NT:]))], 1)
        a = o[:, k:k + NSKEW]; k += NSKEW
        S = torch.zeros(c.shape[0], M, M, dtype=c.dtype)
        S[:, s.iu0, s.iu1] = a
        S = S - S.transpose(1, 2)
        I = torch.eye(M, dtype=c.dtype).expand_as(S)
        Q = torch.linalg.solve(I - 0.5 * S, I + 0.5 * S)   # Cayley: ortogonal
        beta = o[:, k:k + M]; k += M
        bk = o[:, k:k + s.kc]; k += s.kc
        ak = F.softplus(o[:, k:k + s.kc])
        return Q, rho, om, beta, bk, ak

    def encode(s, q, c):                             # q: [B,P,2], c:[B,10]
        cc = c[:, None, :].expand(q.shape[0], q.shape[1], c.shape[1])
        g = s.l4(torch.tanh(s.l3(torch.cat([q, cc], -1))))
        return torch.cat([q, g], -1)                 # [B,P,8]

    def core(s, cp, z, k=1):
        Q, rho, om, beta, _, _ = cp
        for _ in range(k):
            y = torch.einsum("bji,bpj->bpi", Q, z)   # Qᵀ z
            y = y.view(*y.shape[:-1], NB, 2)
            cs, sn = torch.cos(om)[:, None, :], torch.sin(om)[:, None, :]
            r = rho[:, None, :]
            y0, y1 = y[..., 0], y[..., 1]
            y = torch.stack([r * (cs * y0 - sn * y1), r * (sn * y0 + cs * y1)], -1).reshape(z.shape)
            z = torch.einsum("bij,bpj->bpi", Q, y) + beta[:, None, :]
        return z

    def correct(s, cp, u):
        _, _, _, _, bk, ak = cp
        for _ in range(s.r):
            pre = torch.einsum("km,bpm->bpk", s.wc, u) + bk[:, None, :]
            gate = ak[:, None, :] * torch.sigmoid(pre)
            u = u + s.eta * torch.einsum("km,bpk->bpm", s.wc, gate)
        return u

    def forward(s, x, c):                            # x: [B,12]
        q = x.view(x.shape[0], -1, 2)
        cp = s.ctx(c)
        z = s.encode(q, c)
        u0 = s.core(cp, z)
        u = s.correct(cp, u0)
        return u[..., :2].reshape(x.shape), u, u0, z, cp

    def to_json(s):
        t = lambda p: p.detach().tolist()
        return {"arch": "ORBITA-v1", "hh": s.hh, "hg": s.hg, "kc": s.kc, "r": s.r, "eta": s.eta,
                "w1": t(s.l1.weight), "b1": t(s.l1.bias), "w2": t(s.l2.weight), "b2": t(s.l2.bias),
                "w3": t(s.l3.weight), "b3": t(s.l3.bias), "w4": t(s.l4.weight), "b4": t(s.l4.bias),
                "wc": t(s.wc), "n_params": s.n_params()}


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z, cd = meta["zdim"], meta["cdim"]
    tr = np.fromfile(os.path.join(d, "train.bin"), dtype="<f8").reshape(-1, z + cd + z)
    rp = np.fromfile(os.path.join(d, "replay.bin"), dtype="<f8").reshape(-1, z + cd + z * len(KS) + 1)
    T = torch.from_numpy(tr.copy()); R = torch.from_numpy(rp.copy())
    return meta, (T[:, :z], T[:, z:z + cd], T[:, z + cd:]), (R[:, :z], R[:, z:z + cd], R[:, z + cd:z + cd + z * len(KS)].reshape(-1, len(KS), z), R[:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True); ap.add_argument("--steps", type=int, default=256000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_final", type=float, default=3e-5)
    ap.add_argument("--lam_h", type=float, default=0.1); ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lam_b", type=float, default=1.0); ap.add_argument("--lam_c", type=float, default=0.01)
    ap.add_argument("--mh_x", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=20000)
    a = ap.parse_args()
    seed = int(a.seed, 16) if a.seed.startswith("0x") else int(a.seed)
    meta, (tx, tc, ty), (rx, rc, ry, rw) = load(a.data)
    torch.manual_seed(seed ^ 0x0B)
    model = Orbita()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(seed ^ 0x7A)
    any_trusted = bool((rw > 1e-12).any())
    rprob = rw / rw.sum()
    ck = a.out + ".ckpt"
    start = 0
    if os.path.exists(ck):                            # reanudación determinista
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); gen.set_state(st["gen"]); start = st["step"]
    t0 = time.time(); last = 0.0
    for step in range(start, a.steps):
        t = step / max(a.steps, 1)
        for g in opt.param_groups:
            g["lr"] = a.lr_final + 0.5 * (a.lr - a.lr_final) * (1 + math.cos(math.pi * t))
        isrep = (torch.rand(a.batch, generator=gen) < meta["replay_fraction"]) & any_trusted
        nrep = int(isrep.sum())
        it = torch.randint(0, tx.shape[0], (a.batch - nrep,), generator=gen)
        ir = torch.multinomial(rprob, nrep, replacement=True, generator=gen) if nrep else torch.zeros(0, dtype=torch.long)
        x = torch.cat([tx[it], rx[ir]]); c = torch.cat([tc[it], rc[ir]]); y = torch.cat([ty[it], ry[ir, 0]])
        yh, u, u0, z, cp = model(x, c)
        scale = 1.0 / (y * y).sum(1).clamp_min(1e-6)
        l1 = (((yh - y) ** 2).sum(1) * scale).mean()
        dp = (yh - y).view(-1, 2)
        lb = (dp.mean(0) ** 2).sum() / (y.view(-1, 2) ** 2).sum(1).mean()
        lc = ((u - u0) ** 2).sum(-1).mean()
        lh = torch.zeros(())
        if a.lam_h > 0 and nrep:
            sl = slice(a.batch - nrep, a.batch)
            cpr = tuple(p[sl] for p in cp)
            zr = z[sl]; zk = zr; prev = 0
            for j, k in enumerate(KS):
                zk = model.core(cpr, zk, k - prev); prev = k
                yk = ry[ir, j]
                with torch.no_grad():
                    tgt = model.encode(yk.view(nrep, -1, 2), c[sl])
                sk = 1.0 / (yk * yk).sum(1).clamp_min(1e-6)
                dz = (zk - tgt)[..., :2] if a.mh_x else (zk - tgt)
                lh = lh + a.gamma ** k * ((dz ** 2).sum((1, 2)) * sk).mean() * nrep / a.batch
        loss = l1 + a.lam_h * lh + a.lam_b * lb + a.lam_c * lc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(l1.detach())
        if (step + 1) % a.ckpt_every == 0 or step + 1 == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "step": step + 1}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            print(f"step {step+1} l1={last:.3e} lh={float(lh):.3e} {time.time()-t0:.0f}s", flush=True)
    w = model.to_json(); w["train"] = vars(a); w["final_l1"] = last
    json.dump(w, open(a.out, "w"))
    # sonda de paridad Rust↔PyTorch
    with torch.no_grad():
        idx = torch.arange(0, 16)
        yp = model(tx[idx], tc[idx])[0]
    json.dump([{"x": tx[i].tolist(), "c": tc[i].tolist(), "y": yp[i].tolist()} for i in range(16)], open(a.out + ".probe.json", "w"))
    print("done", model.n_params(), flush=True)


if __name__ == "__main__":
    main()
