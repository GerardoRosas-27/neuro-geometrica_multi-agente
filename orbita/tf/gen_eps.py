"""Conjunto de episodios de estado (sin texto, sin ds.test): fam-set --nf 12 --n 64 con RNG propio por fase; trayectorias de 16 pasos con la
regla verdadera (demo-rule de ÓRBITA). Escribe set.csv, traj.npy [N,17,12], ctx_true.csv (c verdadero en 0-7; 0 en 8-11) y ctx_none.csv (c = 0)."""
import subprocess, sys, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
B = "/workspace/ngm-orbita/orbita/target/release/orbita"
ph, out = sys.argv[1], sys.argv[2]; rng = {"dev": "0xD7F0", "confirm": "0xC7F0", "tune": "0xE7F0"}[ph]
os.makedirs(out, exist_ok=True)
s = subprocess.run([B, "fam-set", "--nf", "12", "--n", "64", "--rng", rng], capture_output=True, text=True, check=True).stdout
open(f"{out}/set.csv", "w").write(s)
rows = [l.split(",") for l in s.strip().split("\n")[1:]]
def traj(r):
    f, p, x = r[1], f"{r[2]},{r[3]}", r[4].replace(";", ","); t = [x]
    for _ in range(16):
        x = subprocess.run([B, "demo-rule", "--fam", f, "--p", p, "--x", x], capture_output=True, text=True, check=True).stdout.strip(); t.append(x)
    return [[float(v) for v in z.split(",")] for z in t]
with ThreadPoolExecutor(8) as ex: T = np.array(list(ex.map(traj, rows)))
np.save(f"{out}/traj.npy", T)
for name, true in (("ctx_true", True), ("ctx_none", False)):
    with open(f"{out}/{name}.csv", "w") as fo:
        fo.write("idx,cond,text_id," + ",".join(f"c{i}" for i in range(10)) + "\n")
        for r in rows:
            c = [0.0] * 10; f = int(r[1])
            if true and f < 8: c[f] = 1.0; c[8] = float(r[2]); c[9] = float(r[3])
            fo.write(f"{r[0]},k,state,{','.join(repr(v) for v in c)}\n")
print("ok", T.shape)
