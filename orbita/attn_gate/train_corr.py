"""Entrenamiento del corrector de atención sobre ÓRBITA-v1 congelado (docs/prereg_orbita_attn_gate.md).
Uso: train_corr.py --orbita W.json --data TRAINDIR --out C.pt --seed 0xA400 [--steps 20000]
Sólo familias conocidas 0–7 (episodios propios de gen_eps.py train, sin ds.test). Base ÓRBITA por episodio y k ∈ {2,4,8}: re-identificación con
k observaciones («reid») o, con prob. p_mis = 0.5, re-identificación EXCLUYENDO la familia verdadera («mis»: simula una familia nueva).
Pérdida = error de predicción relativo² (1 paso con teacher forcing + free-run k+1..16), recortado a 10 por paso. Reanudable (.ckpt);
curva de pérdida (train y validación en 8 episodios/familia reservados) en OUT.curve.csv."""
import argparse, math, os, time, json, numpy as np, torch
from orb import load_orbita, reid, step, episode_memory, Corrector
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
KS = (2, 4, 8)
def roll_loss(m, C, T, c, cp, k):
    mem = episode_memory(m, C, T, c, cp, k); tf = 0.0; fr = 0.0; x = T[:, k]; n = 16 - k
    for t in range(k + 1, 17):
        y = T[:, t]; sc = (y * y).sum(1).clamp_min(1e-6); xt = T[:, t - 1]
        pt = C(xt, step(m, xt, c, cp), mem); tf = tf + (((pt - y) ** 2).sum(1) / sc).clamp(max=10.0).mean()
        x = C(x, step(m, x, c, cp), mem)
        e = ((x - y) ** 2).sum(1) / sc; e = torch.where(torch.isfinite(e), e, torch.full_like(e, 10.0)); fr = fr + e.clamp(max=10.0).mean()
    return tf / n, fr / n
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbita", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lr_final", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--p_mis", type=float, default=0.5); ap.add_argument("--every", type=int, default=500)
    a = ap.parse_args(); seed = int(a.seed, 16)
    m = load_orbita(a.orbita); T = torch.from_numpy(np.load(f"{a.data}/traj.npy"))
    fam = torch.tensor([int(l.split(",")[1]) for l in open(f"{a.data}/set.csv").read().strip().split("\n")[1:]])
    idx_tr = torch.tensor([i for i in range(len(fam)) if i % 72 < 64]); idx_va = torch.tensor([i for i in range(len(fam)) if i % 72 >= 64])
    cache = a.out + ".ctx.pt"
    if os.path.exists(cache): CT = torch.load(cache)
    else:
        CT = {}
        for k in KS: CT[(k, "reid")] = reid(m, T, k)[0]; CT[(k, "mis")] = reid(m, T, k, excl=fam)[0]
        torch.save(CT, cache + ".tmp"); os.replace(cache + ".tmp", cache)
    CPs = {key: m.ctx(v) for key, v in CT.items()}
    torch.manual_seed(seed ^ 0xA77); C = Corrector(); opt = torch.optim.Adam(C.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(seed ^ 0xC0); ck = a.out + ".ckpt"; start = 0; curve = []
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False); C.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); gen.set_state(st["gen"]); start = st["step"]; curve = st["curve"]
    def val():
        out = {}
        with torch.no_grad():
            for k in KS:
                for mode in ("reid", "mis"):
                    out[f"k{k}_{mode}"] = float(roll_loss(m, C, T[idx_va], CT[(k, mode)][idx_va], tuple(p[idx_va] for p in CPs[(k, mode)]), k)[1])
        out["val"] = float(np.mean(list(out.values()))); return out
    if start == 0: curve.append({"step": 0, "train": float("nan"), "train_tf": float("nan"), "lr": 0.0, "skipped": 0, **val()}); print("step 0", json.dumps(curve[-1]), flush=True)
    t0 = time.time(); acc = []; acct = []; skipped = 0
    for it in range(start, a.steps):
        lr = a.lr * (it + 1) / a.warmup if it < a.warmup else a.lr_final + 0.5 * (a.lr - a.lr_final) * (1 + math.cos(math.pi * (it - a.warmup) / (a.steps - a.warmup)))
        for g in opt.param_groups: g["lr"] = lr
        k = KS[int(torch.randint(0, 3, (1,), generator=gen))]
        b = idx_tr[torch.randint(0, len(idx_tr), (a.bs,), generator=gen)]; mis = torch.rand(a.bs, generator=gen) < a.p_mis
        c = torch.where(mis[:, None], CT[(k, "mis")][b], CT[(k, "reid")][b])
        cp = tuple(torch.where(mis.view(-1, *([1] * (pm.dim() - 1))), pm[b], pr[b]) for pm, pr in zip(CPs[(k, "mis")], CPs[(k, "reid")]))
        ltf, lfr = roll_loss(m, C, T[b], c, cp, k); loss = ltf + lfr
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(C.parameters(), a.clip)
        if torch.isfinite(gn): opt.step()
        else: skipped += 1
        acc.append(float(lfr.detach())); acct.append(float(ltf.detach()))
        if (it + 1) % a.every == 0 or it + 1 == a.steps:
            curve.append({"step": it + 1, "train": float(np.mean(acc)), "train_tf": float(np.mean(acct)), "lr": lr, "skipped": skipped, **val()}); acc = []; acct = []
            torch.save({"model": C.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "step": it + 1, "curve": curve}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
            print(f"step {it+1} train={curve[-1]['train']:.4e} val={curve[-1]['val']:.4e} {time.time()-t0:.0f}s", flush=True)
    with open(a.out + ".curve.csv", "w") as f:
        keys = list(curve[-1]); f.write(",".join(keys) + "\n")
        for r in curve: f.write(",".join(f"{r[k_]:.6g}" for k_ in keys) + "\n")
    torch.save({"state": C.state_dict(), "n_params": C.n_params(), "train": vars(a), "curve": curve}, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    print("done", C.n_params(), flush=True)
if __name__ == "__main__": main()
