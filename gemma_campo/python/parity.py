"""Test de paridad: gemelo numpy (campo.py) == Rust `predict` (test_pred1.f32)."""
import argparse, os
import numpy as np
from campo import Campo
from common import read_z

ap = argparse.ArgumentParser()
ap.add_argument("--zdir", required=True); ap.add_argument("--model", required=True); ap.add_argument("--split", default="test")
a = ap.parse_args()
z, L, offs = read_z(a.zdir, a.split)
pr = np.fromfile(os.path.join(a.model, f"{a.split}_pred1.f32"), dtype="<f4").reshape(z.shape)
c = Campo(a.model)
worst = 0.0
for i in range(len(L)):
    p = c.one_step_all(z[offs[i]:offs[i + 1]])
    worst = max(worst, float(np.abs(p - pr[offs[i]:offs[i + 1]]).max() / (1 + np.abs(p).max())))
print(f"paridad numpy vs Rust: max err relativo = {worst:.2e}")
assert worst < 1e-4, "PARIDAD FALLA"
print("PARIDAD OK")
