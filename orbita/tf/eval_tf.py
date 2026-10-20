"""Evaluación ÓRBITA-TF (sin ds.test; episodios de gen_eps.py). Uso: eval_tf.py ARM W.pt EPDIR OUT.csv
Por episodio: true_ok8/16 (c verdadero, sólo 0-7); para k∈{2,4,8} con c desconocido: OBS (re-identificación sobre 8 familias × p∈{0}∪malla 9×9,
igual que ÓRBITA), y para TF* inferencia iterativa sobre mu: FE (TFE: error de predicción con precisión 1/(2·0.05²) + complejidad gaussiana
respecto al prior elegido) o PE (TF: sólo error de predicción). 100 pasos Adam lr 0.03, se guarda la mejor iteración en F."""
import sys, time, math, numpy as np, torch
from tfe import make, ML
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
arm, wp, ed, out = sys.argv[1:5]
ck = torch.load(wp, weights_only=False); mdl = make(arm); mdl.load_state_dict(ck["state"]); mdl.eval()
for p in mdl.parameters(): p.requires_grad_(False)
T = torch.from_numpy(np.load(f"{ed}/traj.npy")); N = T.shape[0]
rows = [l.split(",") for l in open(f"{ed}/set.csv").read().strip().split("\n")[1:]]
fam = [int(r[1]) for r in rows]; P = [(float(r[2]), float(r[3])) for r in rows]
def ctx(g, p): c = torch.zeros(10); c[g] = 1; c[8], c[9] = p; return c
def rel(a, b): r = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-9); return torch.where(torch.isfinite(r), r, torch.full_like(r, 1e9))
def roll(x, mu, n):
    for _ in range(n): x = mdl.step(x, mu)
    return x
mean = lambda c: mdl.prior(c)[0]
res = {}
t0 = time.time()
with torch.no_grad():   # c verdadero (sólo familias conocidas)
    C = torch.stack([ctx(f, p) if f < 8 else torch.zeros(10) for f, p in zip(fam, P)]); mu = mean(C); x = T[:, 0]; o8 = o16 = None
    for t in range(1, 17):
        x = mdl.step(x, mu)
        if t == 8: o8 = rel(x, T[:, 8]) < 0.05
    o16 = rel(x, T[:, 16]) < 0.05
    res["true_ok8"], res["true_ok16"] = o8.long(), o16.long()
t_true = time.time() - t0
grid = [(0.0, 0.0)] + [(-1 + 0.25 * i, -1 + 0.25 * j) for i in range(9) for j in range(9)]
CAND = torch.stack([ctx(g, p) for g in range(8) for p in grid]); NC = CAND.shape[0]
with torch.no_grad():
    Mc, LSc = mdl.prior(CAND)
tim = {}
for k in (2, 4, 8):
    t0 = time.time()
    S = torch.zeros(N, NC)
    with torch.no_grad():
        for i0 in range(0, N, 16):
            ii = slice(i0, min(N, i0 + 16)); n = ii.stop - ii.start
            x = T[ii, 0].repeat_interleave(NC, 0); mu = Mc.repeat(n, 1); s = torch.zeros(n * NC)
            for t in range(1, k + 1):
                x = mdl.step(x, mu); s = s + rel(x, T[ii, t].repeat_interleave(NC, 0)).clamp(max=10.0)
            S[ii] = s.view(n, NC)
    best = S.argmin(1); m0 = Mc[best]; s_obs = S[torch.arange(N), best] / k
    with torch.no_grad():
        z = roll(T[:, k], m0, 16 - k); res[f"obs_k{k}"] = (rel(z, T[:, 16]) < 0.05).long(); res[f"sobs_k{k}"] = s_obs
    tim[f"obs_k{k}"] = time.time() - t0
    if arm in ("TFE", "TF"):
        modes = ["FE", "PE"] if arm == "TFE" else ["PE"]
        for mode in modes:
            t0 = time.time()
            mu = m0.clone().requires_grad_(True); opt = torch.optim.Adam([mu], lr=0.03)
            sig = torch.exp(LSc[best]) if arm == "TFE" else None
            bestF = torch.full((N,), float("inf")); bestmu = m0.clone(); bestpe = torch.zeros(N)
            for it in range(101):
                x = T[:, 0]; pe2 = torch.zeros(N); pe = torch.zeros(N)
                for t in range(1, k + 1):
                    x = mdl.step(x, mu); r = rel(x, T[:, t]).clamp(max=10.0); pe2 = pe2 + r * r; pe = pe + r
                Fv = pe2 / (2 * 0.05 ** 2)
                if mode == "FE": Fv = Fv + 0.5 * (((mu - m0) / sig) ** 2).sum(1)
                with torch.no_grad():
                    upd = Fv < bestF; bestF = torch.where(upd, Fv, bestF); bestmu[upd] = mu[upd].detach(); bestpe = torch.where(upd, pe / k, bestpe)
                if it == 100: break
                opt.zero_grad(); Fv.sum().backward(); opt.step()
            with torch.no_grad():
                z = roll(T[:, k], bestmu, 16 - k)
            res[f"{mode.lower()}_k{k}"] = (rel(z, T[:, 16]) < 0.05).long(); res[f"F{mode.lower()}_k{k}"] = bestF.detach(); res[f"s{mode.lower()}_k{k}"] = bestpe.detach()
            tim[f"{mode.lower()}_k{k}"] = time.time() - t0
keys = list(res)
with open(out, "w") as f:
    f.write("idx,fam," + ",".join(keys) + "\n")
    for i in range(N): f.write(f"{rows[i][0]},{fam[i]}," + ",".join(f"{float(res[k_][i]):.8g}" for k_ in keys) + "\n")
open(out + ".time.json", "w").write(__import__("json").dumps({"true": t_true, **tim, "n_params": ck["n_params"]}))
print("eval ok", arm, out, {k_: round(v, 2) for k_, v in tim.items()})
