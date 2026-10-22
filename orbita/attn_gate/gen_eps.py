"""Episodios de estado (sin texto, sin ds.test) para ÓRBITA + corrector de atención gateado (docs/prereg_orbita_attn_gate.md).
Uso: gen_eps.py dev|confirm|tune OUT   → 12 familias × 64 (idéntico a orbita/tf/gen_eps.py: mismo RNG por fase)
     gen_eps.py train SEED OUT          → familias conocidas 0–7 × 72 (64 entrenamiento + 8 validación por familia), RNG = SEED ^ 0x5A17
Escribe set.csv y traj.npy [N,17,12] con la regla verdadera (`orbita demo-rule`)."""
import subprocess, sys, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
B = os.environ.get("ORBITA_BIN", "/workspace/ngm-attn-gate/orbita/target/release/orbita")
if sys.argv[1] == "train":
    seed = int(sys.argv[2], 16); out = sys.argv[3]; rng = hex(seed ^ 0x5A17); nf, n = 8, 72
    assert rng not in ("0xd7f0", "0xc7f0", "0xe7f0")
else:
    out = sys.argv[2]; rng = {"dev": "0xD7F0", "confirm": "0xC7F0", "tune": "0xE7F0"}[sys.argv[1]]; nf, n = 12, 64
os.makedirs(out, exist_ok=True)
s = subprocess.run([B, "fam-set", "--nf", str(nf), "--n", str(n), "--rng", rng], capture_output=True, text=True, check=True).stdout
open(f"{out}/set.csv", "w").write(s)
rows = [l.split(",") for l in s.strip().split("\n")[1:]]
def traj(r):
    f, p, x = r[1], f"{r[2]},{r[3]}", r[4].replace(";", ","); t = [x]
    for _ in range(16):
        x = subprocess.run([B, "demo-rule", "--fam", f, "--p", p, "--x", x], capture_output=True, text=True, check=True).stdout.strip(); t.append(x)
    return [[float(v) for v in z.split(",")] for z in t]
with ThreadPoolExecutor(int(os.environ.get("GEN_THREADS", "4"))) as ex: T = np.array(list(ex.map(traj, rows)))
np.save(f"{out}/traj.tmp.npy", T); os.replace(f"{out}/traj.tmp.npy", f"{out}/traj.npy")
if sys.argv[1] != "train":
    for name, true in (("ctx_true", True), ("ctx_none", False)):
        with open(f"{out}/{name}.csv", "w") as fo:
            fo.write("idx,cond,text_id," + ",".join(f"c{i}" for i in range(10)) + "\n")
            for r in rows:
                c = [0.0] * 10; f = int(r[1])
                if true and f < 8: c[f] = 1.0; c[8] = float(r[2]); c[9] = float(r[3])
                fo.write(f"{r[0]},k,state,{','.join(repr(v) for v in c)}\n")
print("ok", T.shape)
