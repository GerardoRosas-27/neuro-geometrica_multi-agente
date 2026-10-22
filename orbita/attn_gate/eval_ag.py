"""Evaluación (sin ds.test): ÓRBITA-OBS congelado y ÓRBITA + corrector siempre (ALWAYS). GATE se forma en el análisis con s_known y τ fijado.
Uso: eval_ag.py ORBITA.json CORR.pt EPDIR OUT.csv → por k ∈ {2,4,8}: s_k (sorpresa), g_k, obs_k, alw_k (acierto t=16, rel. err. < 0.05)."""
import sys, os, json, time, numpy as np, torch
from orb import load_orbita, reid, step, rel, episode_memory, Corrector
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
wo, wc, ed, out = sys.argv[1:5]
m = load_orbita(wo); C = Corrector(); C.load_state_dict(torch.load(wc, weights_only=False)["state"]); C.eval()
T = torch.from_numpy(np.load(f"{ed}/traj.npy")); N = T.shape[0]
rows = [l.split(",") for l in open(f"{ed}/set.csv").read().strip().split("\n")[1:]]
ok = lambda x: (torch.isfinite(x).all(1) & (rel(x, T[:, 16]) < 0.05)).long()
res = {}; tim = {}
with torch.no_grad():
    for k in (2, 4, 8):
        t0 = time.time(); c, sk, g = reid(m, T, k); cp = m.ctx(c); mem = episode_memory(m, C, T, c, cp, k); xo = T[:, k]; xa = T[:, k]
        for _ in range(16 - k): xo = step(m, xo, c, cp); xa = C(xa, step(m, xa, c, cp), mem)
        res[f"s_k{k}"] = sk; res[f"g_k{k}"] = g; res[f"obs_k{k}"] = ok(xo); res[f"alw_k{k}"] = ok(xa); tim[f"k{k}"] = time.time() - t0
with open(out + ".tmp", "w") as f:
    keys = list(res); f.write("idx,fam," + ",".join(keys) + "\n")
    for i in range(N): f.write(f"{rows[i][0]},{rows[i][1]}," + ",".join(f"{float(res[k_][i]):.10g}" for k_ in keys) + "\n")
os.replace(out + ".tmp", out); open(out + ".time.json", "w").write(json.dumps(tim)); print("eval ok", out, tim)
