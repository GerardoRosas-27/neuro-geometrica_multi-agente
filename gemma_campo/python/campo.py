"""Gemelo numpy del predictor Rust (lenta + rápida v4/BQ10 M1+M2+M6+M7) para el demo de chat y el test de paridad."""
import json, os
import numpy as np

BQ10_DA = 10.0
BQ10_MAX_WIN = 32
LIB_S = np.float32(0.3)


class Campo:
    def __init__(self, model_dir):
        m = json.load(open(os.path.join(model_dir, "model.json")))
        c = m["cfg"]
        self.d, self.ctx, self.quad = c["d"], c["ctx"], c["quad"]
        self.bq10 = bool(c.get("bq10", False))
        self.eta, self.gamma = m["fast_eta"], m["fast_gamma"]
        self.bq10_lam = float(m.get("bq10_lam", 0.0))
        self.F = (
            1
            + self.ctx * self.d
            + (self.d * (self.d + 1) // 2 if self.quad else 0)
            + (6 * self.d if self.bq10 else 0)
        )
        self.W = np.fromfile(os.path.join(model_dir, "model.bin"), dtype="<f4").reshape(self.F, self.d).astype(np.float64)
        self.iu = np.triu_indices(self.d)
        deg = [0] + [1] * (self.ctx * self.d)
        if self.quad:
            deg += [2] * (self.d * (self.d + 1) // 2)
        if self.bq10:
            deg += [1] * (2 * self.d) + [2] * (2 * self.d) + [3] * (2 * self.d)
        self.colscale = np.array([1.0 if g <= 1 else BQ10_DA ** (-(g - 1) / 2.0) for g in deg], dtype=np.float64)
        self.reset()

    def reset(self):
        self.A = np.zeros((self.d, self.d), dtype=np.float64)
        self.psi_hist = []
        self.res_hist = []
        self.cum_err_v4 = 0.0
        self.cum_err_m2m7 = 0.0
        self.steps = 0
        self._last_phi = None
        self._last_fp_v4 = None
        self._last_fp_bq = None

    def phi(self, seq, t):
        parts = [np.ones(1, dtype=np.float32)] + [ np.asarray(seq[max(t - c, 0)], dtype=np.float32) for c in range(self.ctx) ]
        z = np.asarray(seq[t], dtype=np.float32)
        if self.quad:
            parts.append(np.outer(z, z)[self.iu])
        if self.bq10:
            r2 = np.float32((z * z).sum()) / np.float32(max(self.d, 1))
            r = np.float32(np.sqrt(r2))
            two_z = np.float32(2.0) * z
            zc = np.clip(z, np.float32(-4.0), np.float32(4.0))
            r2c = np.minimum(r2, np.float32(16.0))
            parts.extend([
                LIB_S * np.sin(two_z),
                LIB_S * np.tanh(two_z),
                LIB_S * (np.cos(two_z) - np.float32(1.0)),
                LIB_S * r * z,
                LIB_S * (zc * zc * zc),
                LIB_S * r2c * zc,
            ])
        return np.concatenate([np.asarray(p, np.float32) for p in parts]).astype(np.float64)

    def _scaled_psi(self, phi_vec):
        v = phi_vec * self.colscale
        nrm = max(float(np.linalg.norm(v)), 1e-6)
        return v / nrm

    def _predict_delta_bq(self, phi_vec):
        m = len(self.psi_hist)
        if self.bq10_lam <= 0.0 or m == 0:
            return np.zeros(self.d, dtype=np.float64)
        psi = self._scaled_psi(phi_vec)
        P = np.asarray(self.psi_hist, dtype=np.float64)
        R = np.asarray(self.res_hist, dtype=np.float64)
        K = P @ P.T + self.bq10_lam * np.eye(m, dtype=np.float64)
        rhs = P @ psi
        try:
            alpha = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(self.d, dtype=np.float64)
        return alpha @ R

    def _use_m2m7(self):
        return self.bq10_lam > 0.0 and self.steps >= 2 and self.cum_err_m2m7 < self.cum_err_v4

    def slow(self, seq, t):
        ph = self.phi(seq, t)
        self._last_phi = ph
        return seq[t].astype(np.float64) + ph @ self.W

    def predict(self, seq, t):
        """predicción (campo) de z_{t+1} dado seq[:t+1]"""
        s = self.slow(seq, t)
        fp_v4 = s + (self.A @ seq[t].astype(np.float64) if self.eta else 0.0)
        self._last_fp_v4 = fp_v4
        if self.bq10_lam > 0.0:
            fp_bq = s + self._predict_delta_bq(self._last_phi)
            self._last_fp_bq = fp_bq
            chosen = fp_bq if self._use_m2m7() else fp_v4
            return chosen, s
        return fp_v4, s

    def observe(self, seq, t, slow_pred):
        """actualización causal NLMS + BQ10 (M2+M7+M6) tras ver z_{t+1}=seq[t+1]"""
        tgt = seq[t + 1].astype(np.float64)
        r = tgt - slow_pred
        if self.bq10_lam > 0.0 and self._last_phi is not None:
            err_v4 = float(np.mean((self._last_fp_v4 - tgt) ** 2))
            err_bq = float(np.mean((self._last_fp_bq - tgt) ** 2))
            self.cum_err_v4 = 0.95 * self.cum_err_v4 + err_v4
            self.cum_err_m2m7 = 0.95 * self.cum_err_m2m7 + err_bq
            self.steps += 1
            psi = self._scaled_psi(self._last_phi)
            if len(self.psi_hist) == BQ10_MAX_WIN:
                self.psi_hist.pop(0)
                self.res_hist.pop(0)
            self.psi_hist.append(psi)
            self.res_hist.append(r)
        if self.eta:
            z = seq[t].astype(np.float64)
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
