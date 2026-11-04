"""Gemelo numpy del predictor Rust (lenta + rápida) para el demo de chat y el test de paridad."""
import json, os
import numpy as np


class Campo:
    def __init__(self, model_dir):
        m = json.load(open(os.path.join(model_dir, "model.json")))
        c = m["cfg"]
        self.d, self.ctx, self.quad = c["d"], c["ctx"], c["quad"]
        self.eta, self.gamma = m["fast_eta"], m["fast_gamma"]
        self.F = 1 + self.ctx * self.d + (self.d * (self.d + 1) // 2 if self.quad else 0)
        self.W = np.fromfile(os.path.join(model_dir, "model.bin"), dtype="<f4").reshape(self.F, self.d).astype(np.float64)
        self.iu = np.triu_indices(self.d)
        self.reset()

    def reset(self):
        self.A = np.zeros((self.d, self.d))

    def phi(self, seq, t):
        parts = [np.ones(1)] + [seq[max(t - c, 0)] for c in range(self.ctx)]
        if self.quad:
            z = seq[t].astype(np.float32)
            parts.append(np.outer(z, z)[self.iu])
        return np.concatenate([np.asarray(p, np.float32) for p in parts]).astype(np.float64)

    def slow(self, seq, t):
        return seq[t].astype(np.float64) + self.phi(seq, t) @ self.W

    def predict(self, seq, t):
        """predicción (campo) de z_{t+1} dado seq[:t+1]"""
        s = self.slow(seq, t)
        return s + (self.A @ seq[t] if self.eta else 0.0), s

    def observe(self, seq, t, slow_pred):
        """actualización NLMS tras ver z_{t+1}=seq[t+1]"""
        if not self.eta:
            return
        z = seq[t].astype(np.float64)
        r = seq[t + 1] - slow_pred
        e = r - self.A @ z
        self.A = self.gamma * self.A + self.eta * np.outer(e, z) / (z @ z + 1e-6)

    def one_step_all(self, seq):
        """fila t = predicción de z_t (fila 0 = z_0). Igual que Rust predict."""
        self.reset()
        out = np.array(seq, np.float64, copy=True)
        for t in range(len(seq) - 1):
            p, s = self.predict(seq, t)
            out[t + 1] = p
            self.observe(seq, t, s)
        return out
