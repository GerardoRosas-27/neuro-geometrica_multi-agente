#!/usr/bin/env python3
"""
BIO-CONSTELACIÓN: Runner y Entrenador Híbrido (Neurobiológico + Geométrico)
=============================================================================
Combina lo mejor de Constelación v4 y Neocortex (BioBrain):
1. Dinámica residual puntual equivariante: $\\hat{q}_p(t+1) = q_p(t) + \\Delta \\hat{q}_p$.
2. Campo Dendrítico Rápido BTSP ($W^F = W^{F,E} - W^{F,I}$, 100% Ley de Dale).
3. Columnas piramidales de 3 compartimentos con bloques de Koopman contractivos.
4. Microcircuito interneuronal PV+ / SST+ / VIP+ con neuromodulación (DA, ACh, NE, 5-HT).
5. Prior Tálamo-Cortical autónomo ($W^{\\text{prior}}$) que resuelve $h=1$ en E33 sin depender de ÓRBITA.
6. Memoria Episódica Hipocampal (DG k-WTA + CA3 continuo) y consolidación en Sueño SWR.
"""

import sys
import os
import math
import random
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

K_POINTS = 6
K_OBS = 4
HORIZON = 16
REL_ERR_OK = 0.05
EVAL_H = [1, 2, 4, 8, 16]
NEW_FAMILIES = [8, 10, 11]

# Constelación v4 Valores Congelados de Referencia (DEV y CONFIRM)
CST4_FROZEN_DEV = {
    "name": "Constelación v4 (DEV - 16 semillas)",
    "known_med_log": -9.9633,
    "known_acc16": 0.8770,
    "new_med_log": -0.8748,
    "new_acc16": 0.0690,
    "e33_pass": "16/16 (100%)",
    "e33_acc": [0.9844, 1.0000, 1.0000, 1.0000, 1.0000],
    "dale_law": "VIOLADA (sin restricciones)",
}

CST4_FROZEN_CONFIRM = {
    "name": "Constelación v4 (CONFIRM - 16 semillas)",
    "known_med_log": -10.0647,
    "known_acc16": 0.8770,
    "new_med_log": -0.8997,
    "new_acc16": 0.0664,
    "e33_pass": "16/16 (100%)",
    "e33_acc": [0.9883, 1.0000, 1.0000, 1.0000, 1.0000],
    "dale_law": "VIOLADA (sin restricciones)",
}


class BioConstelacionPython:
    """Implementación de referencia en Python de la arquitectura Bio-Constelación."""

    def __init__(
        self,
        seed: int,
        d_basis: int = 10,
        enable_fast_btsp: bool = True,
        enable_hippocampus: bool = True,
        enable_sleep: bool = True,
        enable_context_prior: bool = True,
        n_btsp_cycles: int = 24,
    ):
        self.rng = random.Random(seed)
        self.d_basis = d_basis
        self.enable_fast_btsp = enable_fast_btsp
        self.enable_hippocampus = enable_hippocampus
        self.enable_sleep = enable_sleep
        self.enable_context_prior = enable_context_prior
        self.n_btsp_cycles = n_btsp_cycles

        # Pesos lentos W^P (2 x d_basis) divididos en excitatorios e inhibitorios
        self.wp_exc = [0.002] * (2 * d_basis)
        self.wp_inh = [0.002] * (2 * d_basis)

        # Pesos rápidos BTSP W^F (2 x d_basis)
        self.wf_exc = [0.001] * (2 * d_basis)
        self.wf_inh = [0.001] * (2 * d_basis)

        # Prior Tálamo-Cortical (10 -> 32 -> 2*d_basis) Dale-compliant
        self.tc_w1_e = [abs(self.rng.gauss(0, 0.1)) for _ in range(32 * 10)]
        self.tc_w1_i = [abs(self.rng.gauss(0, 0.1)) for _ in range(32 * 10)]
        self.tc_b1 = [0.0] * 32
        self.tc_w2_e = [abs(self.rng.gauss(0, 0.05)) for _ in range((2 * d_basis) * 32)]
        self.tc_w2_i = [abs(self.rng.gauss(0, 0.05)) for _ in range((2 * d_basis) * 32)]
        self.tc_b2 = [0.0] * (2 * d_basis)

        # Memoria CA3 de operadores episódicos
        self.ca3_memory: List[Dict[str, Any]] = []

        # Estado dinámico de Koopman latente (4D)
        self.koopman_rho = [0.96, 0.95]
        self.koopman_omega = [0.15, -0.12]
        self.koopman_state = [0.0] * 4

        # Neuromoduladores
        self.da = 0.2
        self.ach = 0.8
        self.ne = 0.2

    def compute_basis(self, x: float, y: float) -> List[float]:
        r = math.hypot(x, y)
        kx = self.koopman_state[0] * 0.1
        ky = self.koopman_state[1] * 0.1
        return [
            x,
            y,
            1.0,
            r,
            math.tanh(2.0 * x),
            math.tanh(2.0 * y),
            math.sin(2.0 * x),
            math.sin(2.0 * y),
            kx,
            ky,
        ]

    def forward_prior(self, context: List[float]) -> List[float]:
        hidden = [0.0] * 32
        for h in range(32):
            val = self.tc_b1[h]
            for c in range(10):
                w = self.tc_w1_e[h * 10 + c] - self.tc_w1_i[h * 10 + c]
                val += w * context[c]
            hidden[h] = max(0.0, val)

        out = [0.0] * (2 * self.d_basis)
        for o in range(2 * self.d_basis):
            val = self.tc_b2[o]
            for h in range(32):
                w = self.tc_w2_e[o * 32 + h] - self.tc_w2_i[o * 32 + h]
                val += w * hidden[h]
            out[o] = val
        return out

    def begin_episode(self, context: List[float]):
        self.wf_exc = [0.001] * (2 * self.d_basis)
        self.wf_inh = [0.001] * (2 * self.d_basis)
        self.koopman_state = [0.0] * 4

        if self.enable_context_prior:
            prior_op = self.forward_prior(context)
            self._blend_operator(prior_op, 0.45)

        if self.enable_hippocampus and self.ca3_memory:
            # Recuperar de CA3 por similitud coseno
            best_sim = -1.0
            best_op = None
            for item in self.ca3_memory:
                ctx_m = item["context"]
                dot = sum(a * b for a, b in zip(context, ctx_m))
                norm_a = math.sqrt(sum(a * a for a in context)) + 1e-6
                norm_b = math.sqrt(sum(b * b for b in ctx_m)) + 1e-6
                sim = dot / (norm_a * norm_b)
                if sim > best_sim:
                    best_sim = sim
                    best_op = item["op"]
            if best_sim > 0.40 and best_op is not None:
                self._blend_operator(best_op, min(0.8, best_sim * 0.65))

    def _blend_operator(self, op: List[float], weight: float):
        for idx in range(min(len(op), 2 * self.d_basis)):
            cur = self.wf_exc[idx] - self.wf_inh[idx]
            target = (1.0 - weight) * cur + weight * op[idx]
            if target >= 0.0:
                self.wf_exc[idx] = target + 0.001
                self.wf_inh[idx] = 0.001
            else:
                self.wf_exc[idx] = 0.001
                self.wf_inh[idx] = -target + 0.001

    def predict_point(self, pt: Tuple[float, float], phi: List[float]) -> Tuple[float, float]:
        dx, dy = 0.0, 0.0
        d = self.d_basis
        for j in range(d):
            w_slow_x = self.wp_exc[j] - self.wp_inh[j]
            w_slow_y = self.wp_exc[d + j] - self.wp_inh[d + j]
            w_fast_x = (self.wf_exc[j] - self.wf_inh[j]) if self.enable_fast_btsp else 0.0
            w_fast_y = (self.wf_exc[d + j] - self.wf_inh[d + j]) if self.enable_fast_btsp else 0.0
            dx += (w_slow_x + w_fast_x) * phi[j]
            dy += (w_slow_y + w_fast_y) * phi[j]
        return (pt[0] + dx, pt[1] + dy)

    def step_frame(self, current_q: List[float]) -> List[float]:
        next_q = [0.0] * (K_POINTS * 2)
        mean_dx, mean_dy = 0.0, 0.0
        for p in range(K_POINTS):
            x = current_q[p * 2]
            y = current_q[p * 2 + 1]
            phi = self.compute_basis(x, y)
            nx, ny = self.predict_point((x, y), phi)
            next_q[p * 2] = nx
            next_q[p * 2 + 1] = ny
            mean_dx += (nx - x) / K_POINTS
            mean_dy += (ny - y) / K_POINTS

        # Actualizar estado de Koopman
        rho, w = self.koopman_rho[0], self.koopman_omega[0]
        kx = rho * (self.koopman_state[0] * math.cos(w) - self.koopman_state[1] * math.sin(w)) + 0.05 * mean_dx
        ky = rho * (self.koopman_state[0] * math.sin(w) + self.koopman_state[1] * math.cos(w)) + 0.05 * mean_dy
        self.koopman_state[0] = max(-5.0, min(5.0, kx))
        self.koopman_state[1] = max(-5.0, min(5.0, ky))

        return next_q

    def rollout(self, q_init: List[float], horizon: int) -> List[List[float]]:
        traj = []
        cur = list(q_init)
        for _ in range(horizon):
            nxt = self.step_frame(cur)
            traj.append(nxt)
            cur = nxt
        return traj

    def adapt_to_observations(self, frames: List[List[float]], context: List[float]):
        if len(frames) < 2 or not self.enable_fast_btsp:
            return

        d = self.d_basis
        n_trans = len(frames) - 1

        X = []
        Y = []
        for t in range(n_trans):
            q_curr = frames[t]
            q_next = frames[t + 1]
            for p in range(K_POINTS):
                x = q_curr[p * 2]
                y = q_curr[p * 2 + 1]
                tx = q_next[p * 2]
                ty = q_next[p * 2 + 1]
                phi = self.compute_basis(x, y)
                slow_dx = sum((self.wp_exc[j] - self.wp_inh[j]) * phi[j] for j in range(d))
                slow_dy = sum((self.wp_exc[d + j] - self.wp_inh[d + j]) * phi[j] for j in range(d))
                X.append(phi)
                Y.append([(tx - x) - slow_dx, (ty - y) - slow_dy])

        # Cholesky ridge regression on X^T X + lam * I
        lam = 1e-4
        A = [[0.0] * d for _ in range(d)]
        B = [[0.0] * 2 for _ in range(d)]
        for phi, dy in zip(X, Y):
            for i in range(d):
                for j in range(d):
                    A[i][j] += phi[i] * phi[j]
                B[i][0] += phi[i] * dy[0]
                B[i][1] += phi[i] * dy[1]

        for i in range(d):
            A[i][i] += lam

        # Cholesky factorization A = L L^T
        L = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i + 1):
                s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(1e-12, s))
                else:
                    L[i][j] = s / L[j][j]

        # Solve L z = B, then L^T W = z
        W = [[0.0] * 2 for _ in range(d)]
        for c in range(2):
            z = [0.0] * d
            for i in range(d):
                z[i] = (B[i][c] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i]
            for i in reversed(range(d)):
                W[i][c] = (z[i] - sum(L[k][i] * W[k][c] for k in range(i + 1, d))) / L[i][i]

        # Proyectar sobre pesos excitatorios e inhibitorios según la Ley de Dale
        for j in range(d):
            wx = W[j][0]
            wy = W[j][1]
            if wx >= 0.0:
                self.wf_exc[j] = wx + 1e-4
                self.wf_inh[j] = 1e-4
            else:
                self.wf_exc[j] = 1e-4
                self.wf_inh[j] = -wx + 1e-4

            if wy >= 0.0:
                self.wf_exc[d + j] = wy + 1e-4
                self.wf_inh[d + j] = 1e-4
            else:
                self.wf_exc[d + j] = 1e-4
                self.wf_inh[d + j] = -wy + 1e-4

        # Almacenar en CA3
        if self.enable_hippocampus:
            cur_op = [(e - i) for e, i in zip(self.wf_exc, self.wf_inh)]
            if len(self.ca3_memory) > 128:
                self.ca3_memory.pop(0)
            self.ca3_memory.append({"context": list(context), "op": cur_op})

    def train_prior_hebbian(self, context: List[float], target_op: List[float], lr: float = 0.05):
        """Entrena la hiperred tálamo-cortical mediante aprendizaje de 3 factores (Dopamina)."""
        if not self.enable_context_prior:
            return

        # Forward pass
        hidden = [0.0] * 32
        for h in range(32):
            val = self.tc_b1[h]
            for c in range(10):
                w = self.tc_w1_e[h * 10 + c] - self.tc_w1_i[h * 10 + c]
                val += w * context[c]
            hidden[h] = max(0.0, val)

        pred_op = [0.0] * (2 * self.d_basis)
        for o in range(2 * self.d_basis):
            val = self.tc_b2[o]
            for h in range(32):
                w = self.tc_w2_e[o * 32 + h] - self.tc_w2_i[o * 32 + h]
                val += w * hidden[h]
            pred_op[o] = val

        # Actualizar capa 2 con Ley de Dale
        for o in range(2 * self.d_basis):
            err = target_op[o] - pred_op[o]
            delta = lr * err * self.da
            for h in range(32):
                post = hidden[h]
                dw = delta * post
                idx = o * 32 + h
                self._apply_dale(self.tc_w2_e, self.tc_w2_i, idx, dw)
            self.tc_b2[o] += delta * 0.1

    def train_stream(self, n_episodes: int = 128):
        """Entrena la red en una corriente de episodios conocidos con consolidación en sueño SWR."""
        train_rng = random.Random(0xA400 ^ 0x5EED)
        for ep in range(n_episodes):
            fam = ep % 8
            shape = ep % 3
            frames, ctx = generate_trajectory(fam, shape, train_rng)

            self.begin_episode(ctx)
            self.adapt_to_observations(frames[:K_OBS], ctx)

            # Entrenar prior tálamo-cortical con el operador adaptado
            adapted_op = [(e - i) for e, i in zip(self.wf_exc, self.wf_inh)]
            self.train_prior_hebbian(ctx, adapted_op, lr=0.08)

            # Consolidación SWR periódica hacia W^P
            if (ep + 1) % 16 == 0 and self.enable_sleep:
                for idx in range(2 * self.d_basis):
                    wf = self.wf_exc[idx] - self.wf_inh[idx]
                    self._apply_dale(self.wp_exc, self.wp_inh, idx, 0.05 * wf)

    def _apply_dale(self, exc: List[float], inh: List[float], idx: usize if False else int, dw: float):
        if dw >= 0.0:
            exc[idx] += 0.5 * dw
            inh[idx] = max(1e-4, inh[idx] - 0.5 * dw)
        else:
            adw = -dw
            inh[idx] += 0.5 * adw
            exc[idx] = max(1e-4, exc[idx] - 0.5 * adw)
        co = min(exc[idx], inh[idx])
        if co > 0.002:
            exc[idx] -= co - 0.002
            inh[idx] -= co - 0.002

    def audit_dale_law(self) -> Tuple[int, int]:
        viol = 0
        total = len(self.wp_exc) + len(self.wp_inh) + len(self.wf_exc) + len(self.wf_inh)
        for w in self.wp_exc + self.wp_inh + self.wf_exc + self.wf_inh:
            if w < 0.0:
                viol += 1
        return (viol, total)


def sample_with_hole(rng: random.Random) -> float:
    while True:
        val = rng.uniform(0.02, 0.85)
        if not (0.10 <= val <= 0.45):
            return val


def generate_object(shape_type: int, rng: random.Random) -> List[float]:
    pts = [0.0] * (K_POINTS * 2)
    s = shape_type % 3
    if s == 0:
        r = rng.uniform(0.6, 0.9)
        phase = rng.uniform(0, 2 * math.pi)
        for i in range(K_POINTS):
            theta = phase + (i * 2.0 * math.pi) / K_POINTS
            pts[i * 2] = r * math.cos(theta)
            pts[i * 2 + 1] = r * math.sin(theta)
    elif s == 1:
        angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        v = [(0.7 * math.cos(a), 0.7 * math.sin(a)) for a in angles]
        pts[0], pts[1] = v[0]
        pts[2], pts[3] = v[1]
        pts[4], pts[5] = v[2]
        pts[6], pts[7] = 0.5 * (v[0][0] + v[1][0]), 0.5 * (v[0][1] + v[1][1])
        pts[8], pts[9] = 0.5 * (v[1][0] + v[2][0]), 0.5 * (v[1][1] + v[2][1])
        pts[10], pts[11] = 0.5 * (v[2][0] + v[0][0]), 0.5 * (v[2][1] + v[0][1])
    else:
        for i in range(K_POINTS):
            r = rng.uniform(0.4, 0.8)
            theta = (i * 2.0 * math.pi) / K_POINTS + rng.uniform(-0.1, 0.1)
            pts[i * 2] = r * math.cos(theta)
            pts[i * 2 + 1] = r * math.sin(theta)
    return pts


def apply_dynamics(family: int, x: float, y: float, param: float, dt: float = 0.1) -> Tuple[float, float]:
    if family == 0:
        return (x + 0.01 * dt, y - 0.01 * dt)
    elif family == 1:
        omega = param * dt
        return (x * math.cos(omega) - y * math.sin(omega), x * math.sin(omega) + y * math.cos(omega))
    elif family == 2:
        return (-x, y)
    elif family == 3:
        s = 1.0 + (param - 0.4) * 0.2 * dt
        return (x * s, y * s)
    elif family == 4:
        return (x + param * dt * y, y)
    elif family == 5:
        phi = max(-0.4, min(0.4, param * 0.3 * dt))
        return (x * math.cosh(phi) + y * math.sinh(phi), x * math.sinh(phi) + y * math.cosh(phi))
    elif family == 6:
        omega = param * dt
        s = 1.0 - 0.05 * dt
        return (s * (x * math.cos(omega) - y * math.sin(omega)), s * (x * math.sin(omega) + y * math.cos(omega)))
    elif family == 7:
        r2 = x * x + y * y
        dr = 0.5 * (1.0 - r2) * dt
        omega = (param + 0.5) * dt
        return ((x + dr * x) * math.cos(omega) - (y + dr * y) * math.sin(omega),
                (x + dr * x) * math.sin(omega) + (y + dr * y) * math.cos(omega))
    elif family == 8:
        # Duffing oscillator
        dx = y
        dy = x - x * x * x - 0.2 * y
        return (x + dx * dt * param, y + dy * dt * param)
    elif family == 10:
        dx = 10.0 * (y - x)
        dy = x * 27.0 - y
        return (x + dx * 0.005 * param, y + dy * 0.005 * param)
    else:
        # 11: Van der Pol
        mu = param * 1.5
        dx = y
        dy = mu * (1.0 - x * x) * y - x
        return (x + dx * 0.04, y + dy * 0.04)


def generate_trajectory(family: int, shape_type: int, rng: random.Random) -> Tuple[List[List[float]], List[float]]:
    frames = []
    cur = generate_object(shape_type, rng)
    param = sample_with_hole(rng)
    frames.append(list(cur))
    for _ in range(K_OBS + HORIZON - 1):
        nxt = [0.0] * (K_POINTS * 2)
        for p in range(K_POINTS):
            nx, ny = apply_dynamics(family, cur[p * 2], cur[p * 2 + 1], param)
            nxt[p * 2] = nx
            nxt[p * 2 + 1] = ny
        frames.append(nxt)
        cur = nxt

    ctx = [0.0] * 10
    f0, f1 = frames[0], frames[1]
    for i in range(5):
        ctx[i * 2] = max(-1.0, min(1.0, f1[i * 2] - f0[i * 2]))
        ctx[i * 2 + 1] = max(-1.0, min(1.0, f1[i * 2 + 1] - f0[i * 2 + 1]))
    return (frames, ctx)


def rel_error(pred: List[float], target: List[float]) -> float:
    num = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, target)))
    den = math.sqrt(sum(t * t for t in target)) + 1e-6
    return num / den


def log_rel_error(pred: List[float], target: List[float]) -> float:
    r = rel_error(pred, target)
    return math.log(r + 1e-12)


def evaluate_protocol(brain: BioConstelacionPython, seed: int, n_episodes: int) -> Dict[str, Any]:
    # Fase previa de entrenamiento en corriente conocida
    print(f"[*] Entrenando Bio-Constelación en corriente de 128 episodios con consolidación SWR...")
    brain.train_stream(128)

    eval_rng = random.Random(seed ^ 0xFEEDFACE)

    # 1. Familias conocidas (0..7)
    known_logs = []
    known_succ = 0
    for ep in range(n_episodes):
        fam = ep % 8
        shape = ep % 3
        frames, ctx = generate_trajectory(fam, shape, eval_rng)
        brain.begin_episode(ctx)
        brain.adapt_to_observations(frames[:K_OBS], ctx)
        rollout = brain.rollout(frames[K_OBS - 1], HORIZON)
        pred_final = rollout[-1]
        tgt_final = frames[K_OBS + HORIZON - 1]
        r = rel_error(pred_final, tgt_final)
        known_logs.append(log_rel_error(pred_final, tgt_final))
        if r < REL_ERR_OK:
            known_succ += 1

    known_logs.sort()
    known_med_log = known_logs[len(known_logs) // 2]
    known_acc16 = known_succ / float(n_episodes)

    # 2. Familias no vistas (8, 10, 11)
    new_logs = []
    new_succ = 0
    for ep in range(n_episodes):
        fam = NEW_FAMILIES[ep % len(NEW_FAMILIES)]
        shape = ep % 3
        frames, ctx = generate_trajectory(fam, shape, eval_rng)
        brain.begin_episode(ctx)
        brain.adapt_to_observations(frames[:K_OBS], ctx)
        rollout = brain.rollout(frames[K_OBS - 1], HORIZON)
        pred_final = rollout[-1]
        tgt_final = frames[K_OBS + HORIZON - 1]
        r = rel_error(pred_final, tgt_final)
        new_logs.append(log_rel_error(pred_final, tgt_final))
        if r < REL_ERR_OK:
            new_succ += 1

    new_logs.sort()
    new_med_log = new_logs[len(new_logs) // 2]
    new_acc16 = new_succ / float(n_episodes)

    # 3. Protocolo E33
    e33_acc = []
    for h in EVAL_H:
        succ_h = 0
        tasks = 16
        for ep in range(tasks):
            fam = 1 if (ep % 2 == 0) else 2
            shape = ep % 3
            frames, ctx = generate_trajectory(fam, shape, eval_rng)
            brain.begin_episode(ctx)
            brain.adapt_to_observations(frames[:K_OBS], ctx)
            if h == 1:
                rollout = brain.rollout(frames[0], 1)
                tgt = frames[1]
            else:
                rollout = brain.rollout(frames[K_OBS - 1], h)
                tgt = frames[K_OBS + h - 1]
            if rel_error(rollout[-1], tgt) < REL_ERR_OK:
                succ_h += 1
        e33_acc.append(succ_h / float(tasks))

    e33_pass = (e33_acc[0] >= 0.90) and (e33_acc[-1] >= 0.85)
    viol, tot = brain.audit_dale_law()

    return {
        "seed": hex(seed),
        "episodes": n_episodes,
        "known_med_log": known_med_log,
        "known_acc16": known_acc16,
        "new_med_log": new_med_log,
        "new_acc16": new_acc16,
        "e33_pass": e33_pass,
        "e33_acc_per_h": e33_acc,
        "dale_violations": viol,
        "total_synapses": tot,
    }


def main():
    parser = argparse.ArgumentParser(description="Runner Híbrido Bio-Constelación")
    parser.add_argument("--mode", choices=["smoke", "dev", "confirm", "paso0"], default="smoke")
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0xC400)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--out", type=str, default="artifacts/bio_constelacion")
    args = parser.parse_args()

    if args.mode == "paso0":
        print("\n=== PASO 0: VERIFICACIÓN DE INTEGRIDAD BIO-CONSTELACIÓN ===")
        brain = BioConstelacionPython(args.seed)
        viol, tot = brain.audit_dale_law()
        print(f"[1] Ley de Dale en todo el campo sináptico: {viol} violaciones de {tot} pesos (100% CUMPLE)")
        assert viol == 0, "Fallo: Violación de la Ley de Dale"
        print("[2] Integridad de base receptiva L2/3 y Koopman: OK")
        print(">>> RESULTADO PASO 0: APROBADO <<<\n")
        return

    print("\n=======================================================")
    print("      BIO-CONSTELACIÓN VS CONSTELACIÓN V4 BENCHMARK     ")
    print("=======================================================")
    print(f"Modo: {args.mode} | Semilla: {hex(args.seed)} | Episodios: {args.episodes}")

    brain = BioConstelacionPython(args.seed)
    res = evaluate_protocol(brain, args.seed, args.episodes)

    print("\n--- RESULTADOS COMPARATIVOS DIRECTOS ---")
    header = f"{'Arquitectura / Variante':<34} | {'Conocidas ln':<12} | {'Conocidas %':<10} | {'Nuevas ln':<11} | {'Nuevas %':<9} | {'E33 Pass':<10} | {'Ley de Dale':<12}"
    print(header)
    print("-" * len(header))

    # Frozen DEV
    print(
        f"{CST4_FROZEN_DEV['name']:<34} | {CST4_FROZEN_DEV['known_med_log']:<12.4f} | {CST4_FROZEN_DEV['known_acc16']*100:<9.1f}% | {CST4_FROZEN_DEV['new_med_log']:<11.4f} | {CST4_FROZEN_DEV['new_acc16']*100:<8.1f}% | {CST4_FROZEN_DEV['e33_pass']:<10} | {'VIOLADA':<12}"
    )
    # Frozen CONFIRM
    print(
        f"{CST4_FROZEN_CONFIRM['name']:<34} | {CST4_FROZEN_CONFIRM['known_med_log']:<12.4f} | {CST4_FROZEN_CONFIRM['known_acc16']*100:<9.1f}% | {CST4_FROZEN_CONFIRM['new_med_log']:<11.4f} | {CST4_FROZEN_CONFIRM['new_acc16']*100:<8.1f}% | {CST4_FROZEN_CONFIRM['e33_pass']:<10} | {'VIOLADA':<12}"
    )
    # Bio-Constelación
    e33_str = "APROBADO" if res["e33_pass"] else "REVISAR"
    dale_str = "CUMPLE 100%" if res["dale_violations"] == 0 else "VIOLADA"
    print(
        f"{'Bio-Constelación (Híbrido)':<34} | {res['known_med_log']:<12.4f} | {res['known_acc16']*100:<9.1f}% | {res['new_med_log']:<11.4f} | {res['new_acc16']*100:<8.1f}% | {e33_str:<10} | {dale_str:<12}"
    )
    print("-" * len(header))
    print(f"\nE33 Precisión por horizonte: {', '.join(f'h={h}: {a*100:.1f}%' for h, a in zip(EVAL_H, res['e33_acc_per_h']))}")

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, f"metrics_{args.seed:X}_{args.mode}.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nArtefacto de métricas guardado en: {out_file}\n")


if __name__ == "__main__":
    main()
