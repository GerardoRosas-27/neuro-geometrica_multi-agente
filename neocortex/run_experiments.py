#!/usr/bin/env python3
"""
Ejecutor espejo exacto (f64) de la arquitectura Rust `neocortex` (`BioBrain`).
Permite correr los 6 experimentos neurobiológicos y exportar telemetría JSON
en entornos donde no esté instalado el linker C++ de Visual Studio (`link.exe`),
manteniendo paridad ecuación por ecuación con `neocortex/src/*.rs`.
"""

import json
import math
from typing import List, Optional, Tuple


class Xoshiro256StarStar:
    """PRNG determinista idéntico a `neocortex/src/rng.rs`."""

    def __init__(self, seed: int) -> None:
        self.s = [0, 0, 0, 0]
        state = seed & 0xFFFFFFFFFFFFFFFF
        for i in range(4):
            state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            self.s[i] = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) & 0xFFFFFFFFFFFFFFFF) | (x >> (64 - k))

    def next_u64(self) -> int:
        s1_5 = (self.s[1] * 5) & 0xFFFFFFFFFFFFFFFF
        result = (self._rotl(s1_5, 7) * 9) & 0xFFFFFFFFFFFFFFFF
        t = (self.s[1] << 17) & 0xFFFFFFFFFFFFFFFF

        self.s[2] ^= self.s[0]
        self.s[3] ^= self.s[1]
        self.s[1] ^= self.s[2]
        self.s[0] ^= self.s[3]

        self.s[2] ^= t
        self.s[3] = self._rotl(self.s[3], 45)
        return result

    def next_f64(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def gen_range(self, low: float, high: float) -> float:
        return low + (high - low) * self.next_f64()

    def randn(self) -> float:
        u1 = self.gen_range(1e-12, 1.0)
        u2 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(math.tau * u2)


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PyramidalNeuron:
    """Neurona piramidal de 3 compartimentos (Basal, Apical, Soma) — `dendritic_neuron.rs`."""

    def __init__(self) -> None:
        self.v_basal = 0.0
        self.v_apical = 0.0
        self.u_soma = 0.0
        self.ca_plateau = 0.5
        self.event_rate = 0.0
        self.burst_prob = 0.5
        self.burst_prob_ema = 0.5
        self.burst_rate = 0.0
        self.dendritic_teaching_signal = 0.0
        self.bcm_threshold = 0.1
        self.intrinsic_excitability = 0.0
        self.avg_activity = 0.25

    def reset_fast_state(self) -> None:
        self.v_basal = 0.0
        self.v_apical = 0.0
        self.u_soma = 0.0
        self.ca_plateau = 0.5
        self.event_rate = 0.0
        self.burst_prob = 0.5
        self.burst_rate = 0.0
        self.dendritic_teaching_signal = 0.0

    def step(
        self,
        basal_drive: float,
        apical_top_down: float,
        sst_apical_inhibition: float,
        pv_somatic_inhibition: float,
        vip_disinhibition_gate: float,
    ) -> None:
        dt_over_tau_m = 0.5
        target_v_basal = basal_drive + self.intrinsic_excitability
        self.v_basal += dt_over_tau_m * (target_v_basal - self.v_basal)

        effective_sst = sst_apical_inhibition / (1.0 + 0.5 * max(0.0, vip_disinhibition_gate))
        net_apical = (apical_top_down - effective_sst) * (1.0 + 0.3 * vip_disinhibition_gate)
        self.v_apical += dt_over_tau_m * (net_apical - self.v_apical)

        self.ca_plateau = sigmoid(2.0 * self.v_apical)
        bac_term = 0.35 * math.tanh(self.v_basal) * (2.0 * self.ca_plateau - 1.0)

        denom = max(1e-6, 0.15 + 1.0 + 0.45 + max(0.0, pv_somatic_inhibition))
        u_eq = (
            1.0 * self.v_basal
            + 0.45 * self.v_apical
            + bac_term
            - 0.5 * pv_somatic_inhibition
        ) / denom
        self.u_soma += dt_over_tau_m * (u_eq - self.u_soma)

        basal_rate = math.tanh(self.v_basal)
        soma_rate = math.tanh(self.u_soma)
        self.event_rate = 0.7 * basal_rate + 0.3 * soma_rate
        self.burst_prob = self.ca_plateau
        self.burst_rate = self.event_rate * self.burst_prob

        phi_prime = max(0.05, 1.0 - basal_rate * basal_rate)
        self.dendritic_teaching_signal = self.v_apical * phi_prime
        self.burst_prob_ema += 0.15 * (self.burst_prob - self.burst_prob_ema)

    def update_homeostasis(self) -> None:
        act_abs = abs(self.event_rate)
        self.avg_activity += 0.05 * (act_abs - self.avg_activity)
        r_sq = self.event_rate * self.event_rate
        self.bcm_threshold = clamp(self.bcm_threshold + 0.05 * (r_sq - self.bcm_threshold), 0.01, 0.95)
        rate_error = 0.25 - self.avg_activity
        self.intrinsic_excitability = clamp(self.intrinsic_excitability + 0.005 * rate_error, -0.5, 0.5)


class BioSynapseMatrix:
    """Matriz sináptica sujeta a Ley de Dale, STP de Tsodyks-Markram y regla de 3 factores — `synapse_stp.rs`."""

    def __init__(self, n_out: int, n_in: int, scale: float, u0: float, seed: int) -> None:
        self.n_out = n_out
        self.n_in = n_in
        self.u0 = u0
        rng = Xoshiro256StarStar(seed)
        std = scale / math.sqrt(max(1, n_in))
        total = n_out * n_in
        self.w_exc = [0.0] * total
        self.w_inh = [0.0] * total
        self.w_consolidated = [0.0] * total
        baseline = 0.05 * std
        for idx in range(total):
            w = rng.randn() * std
            if w >= 0.0:
                self.w_exc[idx] = w + baseline
                self.w_inh[idx] = baseline
            else:
                self.w_exc[idx] = baseline
                self.w_inh[idx] = -w + baseline
            self.w_consolidated[idx] = w
        self.bias = [0.0] * n_out
        self.bias_consolidated = [0.0] * n_out
        self.spine_maturity = [0.0] * total
        self.eligibility = [0.0] * total
        self.bias_eligibility = [0.0] * n_out
        self.synaptic_tag_acc = [0.0] * total
        self.stp_u = [u0] * n_in
        self.stp_x = [1.0] * n_in

    def reset_short_term_state(self) -> None:
        self.stp_u = [self.u0] * self.n_in
        self.stp_x = [1.0] * self.n_in
        self.eligibility = [0.0] * (self.n_out * self.n_in)
        self.bias_eligibility = [0.0] * self.n_out

    def forward(self, pre_rates: List[float]) -> List[float]:
        effective_pre = [0.0] * self.n_in
        for j in range(self.n_in):
            act = clamp(abs(pre_rates[j]), 0.0, 1.0)
            u = self.stp_u[j]
            x = self.stp_x[j]
            u_next = u + 0.08 * (self.u0 - u) + self.u0 * (1.0 - u) * act
            x_next = x + 0.15 * (1.0 - x) - u_next * x * act
            self.stp_u[j] = clamp(u_next, 0.05, 0.99)
            self.stp_x[j] = clamp(x_next, 0.05, 1.0)
            stp_gain = (self.stp_u[j] * self.stp_x[j]) / self.u0 if self.u0 > 1e-6 else 1.0
            mod_gain = 1.0 + 0.25 * (stp_gain - 1.0)
            effective_pre[j] = pre_rates[j] * mod_gain

        out = [0.0] * self.n_out
        for i in range(self.n_out):
            row = i * self.n_in
            s = self.bias[i]
            for j in range(self.n_in):
                s += (self.w_exc[row + j] - self.w_inh[row + j]) * effective_pre[j]
            out[i] = s
        return out

    def forward_stateless(self, pre_rates: List[float]) -> List[float]:
        out = [0.0] * self.n_out
        for i in range(self.n_out):
            row = i * self.n_in
            s = self.bias[i]
            for j in range(self.n_in):
                stp_gain = (self.stp_u[j] * self.stp_x[j]) / self.u0 if self.u0 > 1e-6 else 1.0
                mod_gain = 1.0 + 0.25 * (stp_gain - 1.0)
                s += (self.w_exc[row + j] - self.w_inh[row + j]) * pre_rates[j] * mod_gain
            out[i] = s
        return out

    def backward_project(self, post_signals: List[float]) -> List[float]:
        back = [0.0] * self.n_in
        for i in range(self.n_out):
            sig = post_signals[i]
            if abs(sig) < 1e-12:
                continue
            row = i * self.n_in
            for j in range(self.n_in):
                back[j] += (self.w_exc[row + j] - self.w_inh[row + j]) * sig
        return back

    def update_three_factor(
        self,
        pre_rates: List[float],
        post_teaching: List[float],
        post_rates: List[float],
        third_factor_mod: float,
        lr: float,
    ) -> None:
        eff_lr = lr * third_factor_mod
        if abs(eff_lr) < 1e-12:
            return
        for i in range(self.n_out):
            row = i * self.n_in
            teach_i = clamp(post_teaching[i], -3.0, 3.0)
            r_post_sq = post_rates[i] * post_rates[i]
            self.bias_eligibility[i] = 0.50 * self.bias_eligibility[i] + teach_i
            bias_pull = 0.02 * (self.bias_consolidated[i] - self.bias[i])
            self.bias[i] = clamp(self.bias[i] + eff_lr * self.bias_eligibility[i] + lr * bias_pull, -4.0, 4.0)

            for j in range(self.n_in):
                idx = row + j
                hebb = teach_i * pre_rates[j]
                self.eligibility[idx] = 0.50 * self.eligibility[idx] + hebb
                self.synaptic_tag_acc[idx] += abs(hebb)
                maturity = self.spine_maturity[idx]
                plasticity_gate = 1.0 / (1.0 + maturity)
                w_eff = self.w_exc[idx] - self.w_inh[idx]
                cons_force = 0.02 * maturity * (self.w_consolidated[idx] - w_eff)
                oja_term = -1e-4 * r_post_sq * w_eff
                dw = eff_lr * plasticity_gate * self.eligibility[idx] + lr * (cons_force + oja_term)

                if dw >= 0.0:
                    self.w_exc[idx] += 0.5 * dw
                    self.w_inh[idx] = max(0.001, self.w_inh[idx] - 0.5 * dw)
                else:
                    abs_dw = -dw
                    self.w_inh[idx] += 0.5 * abs_dw
                    self.w_exc[idx] = max(0.001, self.w_exc[idx] - 0.5 * abs_dw)

                min_co = min(self.w_exc[idx], self.w_inh[idx])
                if min_co > 0.02:
                    excess = min_co - 0.02
                    self.w_exc[idx] -= excess
                    self.w_inh[idx] -= excess

                new_eff = self.w_exc[idx] - self.w_inh[idx]
                if new_eff > 4.0:
                    self.w_exc[idx] = 4.02
                    self.w_inh[idx] = 0.02
                elif new_eff < -4.0:
                    self.w_exc[idx] = 0.02
                    self.w_inh[idx] = 4.02

    def consolidate_synapses(self, consolidation_rate: float, shy_downscale: float) -> None:
        max_tag = max([1e-8] + self.synaptic_tag_acc)
        for idx in range(len(self.w_exc)):
            w_eff = self.w_exc[idx] - self.w_inh[idx]
            norm_tag = clamp(self.synaptic_tag_acc[idx] / max_tag, 0.0, 1.0)
            alpha = clamp(consolidation_rate * (0.25 + 0.75 * norm_tag), 0.0, 1.0)
            self.w_consolidated[idx] += alpha * (w_eff - self.w_consolidated[idx])
            self.spine_maturity[idx] = clamp(self.spine_maturity[idx] + 1.5 * norm_tag, 0.0, 12.0)
            if norm_tag < 0.15 and shy_downscale > 0.0:
                scale = 1.0 - shy_downscale * (1.0 - norm_tag)
                self.w_exc[idx] = max(0.001, self.w_exc[idx] * scale)
                self.w_inh[idx] = max(0.001, self.w_inh[idx] * scale)
            self.synaptic_tag_acc[idx] = 0.0
        for i in range(self.n_out):
            self.bias_consolidated[i] += consolidation_rate * (self.bias[i] - self.bias_consolidated[i])

    def n_params(self) -> int:
        return self.n_out * self.n_in + self.n_out


class CorticalColumn:
    """Columna Cortical Laminada (L4, L2/3, L5/6 + PV/SST/VIP) — `cortical_column.rs`."""

    def __init__(self, layer_index: int, n_in: int, n_hidden: int, u0: float, seed: int) -> None:
        self.layer_index = layer_index
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.syn_l4_to_l23_basal = BioSynapseMatrix(n_hidden, n_in, 1.15, u0, seed + 101)
        self.syn_l23_to_l56 = BioSynapseMatrix(n_hidden, n_hidden, 0.90, u0, seed + 202)
        self.syn_l56_topdown_pred = BioSynapseMatrix(n_in, n_hidden, 0.85, u0, seed + 303)
        self.l23_pyramidal = [PyramidalNeuron() for _ in range(n_hidden)]
        self.l56_pyramidal = [PyramidalNeuron() for _ in range(n_hidden)]
        self.pv_activity = [0.0] * n_hidden
        self.sst_activity = [0.0] * n_hidden
        self.vip_activity = [0.0] * n_hidden
        self.w_pyr_to_sst = [0.05] * n_hidden
        self.last_bottom_up_input = [0.0] * n_in
        self.top_down_prediction_out = [0.0] * n_in
        self.local_prediction_error = [0.0] * n_in
        self.error_precision = [1.0] * n_in

    def reset_short_term_state(self) -> None:
        self.syn_l4_to_l23_basal.reset_short_term_state()
        self.syn_l23_to_l56.reset_short_term_state()
        self.syn_l56_topdown_pred.reset_short_term_state()
        for n in self.l23_pyramidal:
            n.reset_fast_state()
        for n in self.l56_pyramidal:
            n.reset_fast_state()

    def step_column(
        self,
        bottom_up_input: List[float],
        apical_feedback: List[float],
        ach_level: float,
        ne_level: float,
        learn_microcircuit: bool,
    ) -> List[float]:
        self.last_bottom_up_input = list(bottom_up_input)
        l23_basal_drive = self.syn_l4_to_l23_basal.forward(bottom_up_input)
        prev_l23_rates = [n.event_rate for n in self.l23_pyramidal]

        mean_abs = sum(abs(r) for r in prev_l23_rates) / max(1, self.n_hidden)
        abs_sorted = sorted((abs(r) for r in prev_l23_rates), reverse=True)
        k_idx = max(1, min(self.n_hidden - 1, round(self.n_hidden * 0.65)))
        kwta_thresh = abs_sorted[k_idx]
        vip_drive = clamp(0.6 * ach_level + 0.4 * ne_level, 0.0, 2.0)

        for i in range(self.n_hidden):
            self.vip_activity[i] = vip_drive
            raw_sst = self.w_pyr_to_sst[i] * prev_l23_rates[i]
            self.sst_activity[i] = raw_sst / (1.0 + 0.8 * self.vip_activity[i])
            if learn_microcircuit and i < len(apical_feedback):
                apical_res = apical_feedback[i] - raw_sst
                self.w_pyr_to_sst[i] = clamp(
                    self.w_pyr_to_sst[i] + 0.015 * apical_res * prev_l23_rates[i], -2.0, 2.0
                )
            sub_kwta = max(0.0, kwta_thresh - abs(prev_l23_rates[i]))
            self.pv_activity[i] = 0.25 * mean_abs + 0.60 * sub_kwta

        l23_rates = [0.0] * self.n_hidden
        for i in range(self.n_hidden):
            ap_in = apical_feedback[i] if i < len(apical_feedback) else 0.0
            self.l23_pyramidal[i].step(
                l23_basal_drive[i], ap_in, self.sst_activity[i], self.pv_activity[i], self.vip_activity[i]
            )
            l23_rates[i] = self.l23_pyramidal[i].event_rate

        l56_drive = self.syn_l23_to_l56.forward(l23_rates)
        l56_rates = [0.0] * self.n_hidden
        for i in range(self.n_hidden):
            ap_in = apical_feedback[i] if i < len(apical_feedback) else 0.0
            combined_basal = 0.65 * l23_rates[i] + 0.35 * l56_drive[i]
            self.l56_pyramidal[i].step(
                combined_basal, ap_in, 0.0, 0.5 * self.pv_activity[i], self.vip_activity[i]
            )
            l56_rates[i] = self.l56_pyramidal[i].event_rate

        self.top_down_prediction_out = self.syn_l56_topdown_pred.forward_stateless(l56_rates)
        for j in range(self.n_in):
            raw_err = bottom_up_input[j] - self.top_down_prediction_out[j]
            var_est = max(0.05, 1.0 / self.error_precision[j])
            new_var = clamp(0.95 * var_est + 0.05 * raw_err * raw_err, 0.05, 4.0)
            self.error_precision[j] = 1.0 / new_var
            self.local_prediction_error[j] = raw_err * math.sqrt(self.error_precision[j])
        return l56_rates

    def local_plasticity_step(self, third_factor_mod: float, lr: float) -> List[float]:
        l23_rates = [n.event_rate for n in self.l23_pyramidal]
        l56_rates = [n.event_rate for n in self.l56_pyramidal]
        l56_teaching = [n.dendritic_teaching_signal for n in self.l56_pyramidal]
        intra_back = self.syn_l23_to_l56.backward_project(l56_teaching)
        pred_err_back = self.syn_l56_topdown_pred.backward_project(self.local_prediction_error)

        l23_teaching = [0.0] * self.n_hidden
        for i in range(self.n_hidden):
            phi_p = max(0.05, 1.0 - l23_rates[i] * l23_rates[i])
            l23_teaching[i] = (
                0.65 * self.l23_pyramidal[i].dendritic_teaching_signal
                + 0.25 * intra_back[i] * phi_p
                + 0.10 * pred_err_back[i] * phi_p
            )

        feedback_to_lower = self.syn_l4_to_l23_basal.backward_project(l23_teaching)
        self.syn_l4_to_l23_basal.update_three_factor(
            self.last_bottom_up_input, l23_teaching, l23_rates, third_factor_mod, lr
        )
        self.syn_l23_to_l56.update_three_factor(
            l23_rates, l56_teaching, l56_rates, third_factor_mod, 0.6 * lr
        )
        self.syn_l56_topdown_pred.update_three_factor(
            l56_rates,
            self.local_prediction_error,
            self.top_down_prediction_out,
            max(0.25, abs(third_factor_mod)),
            0.4 * lr,
        )
        for n in self.l23_pyramidal:
            n.update_homeostasis()
        for n in self.l56_pyramidal:
            n.update_homeostasis()
        return feedback_to_lower


class HippocampalSystem:
    """Circuito Trisináptico Hipocampal (EC -> DG -> CA3 -> CA1) — `hippocampus_cls.rs`."""

    def __init__(
        self,
        ec_dim: int,
        dg_dim: int,
        target_dim: int,
        dg_sparsity: float,
        ca3_inverse_temp: float,
        max_engrams: int,
        seed: int,
    ) -> None:
        self.ec_dim = ec_dim
        self.dg_dim = dg_dim
        self.target_dim = target_dim
        self.dg_sparsity = clamp(dg_sparsity, 0.04, 0.25)
        self.ca3_inverse_temp = max(1.0, ca3_inverse_temp)
        self.max_engrams = max(16, max_engrams)
        rng = Xoshiro256StarStar(seed)
        scale = 1.0 / math.sqrt(max(1, ec_dim))
        self.w_perforant = [rng.randn() * scale for _ in range(dg_dim * ec_dim)]
        self.ca3_engrams = []

    def dentate_gyrus_pattern_separation(self, ec_input: List[float]) -> List[float]:
        raw = [0.0] * self.dg_dim
        for i in range(self.dg_dim):
            row = i * self.ec_dim
            s = sum(self.w_perforant[row + j] * ec_input[j] for j in range(self.ec_dim))
            raw[i] = max(0.0, s)
        k = max(1, min(self.dg_dim, round(self.dg_dim * self.dg_sparsity)))
        indices = sorted(range(self.dg_dim), key=lambda idx: raw[idx], reverse=True)
        sparse = [0.0] * self.dg_dim
        norm_sq = 0.0
        for idx in indices[:k]:
            sparse[idx] = raw[idx]
            norm_sq += raw[idx] * raw[idx]
        norm = math.sqrt(norm_sq)
        if norm > 1e-9:
            sparse = [v / norm for v in sparse]
        return sparse

    def ca3_pattern_completion(self, ec_query: List[float]) -> Tuple[List[float], List[float], float]:
        if not self.ca3_engrams:
            return list(ec_query), [0.0] * self.target_dim, 1.0
        q_dg = self.dentate_gyrus_pattern_separation(ec_query)
        q_norm = max(1e-8, math.sqrt(sum(x * x for x in ec_query)))
        scores = []
        max_sim = -1e9
        for eng in self.ca3_engrams:
            e_norm = max(1e-8, math.sqrt(sum(x * x for x in eng["ec_input"])))
            cos_ec = sum(a * b for a, b in zip(ec_query, eng["ec_input"])) / (q_norm * e_norm)
            cos_dg = sum(a * b for a, b in zip(q_dg, eng["dg_sparse_code"]))
            sim = 0.65 * cos_ec + 0.35 * cos_dg
            max_sim = max(max_sim, sim)
            scores.append(self.ca3_inverse_temp * sim)
        m_sc = max(scores)
        weights = [math.exp(s - m_sc) for s in scores]
        sw = max(1e-12, sum(weights))
        weights = [w / sw for w in weights]
        comp_ec = [0.0] * self.ec_dim
        rec_ca1 = [0.0] * self.target_dim
        for eng, w in zip(self.ca3_engrams, weights):
            for j in range(self.ec_dim):
                comp_ec[j] += w * eng["ec_input"][j]
            for j in range(self.target_dim):
                rec_ca1[j] += w * eng["ca1_target"][j]
        novelty = clamp(1.0 - max_sim, 0.0, 1.0)
        return comp_ec, rec_ca1, novelty

    def encode_episode(self, ec_input: List[float], ca1_target: List[float], salience: float) -> None:
        dg_code = self.dentate_gyrus_pattern_separation(ec_input)
        q_norm = max(1e-8, math.sqrt(sum(x * x for x in ec_input)))
        for eng in self.ca3_engrams:
            e_norm = max(1e-8, math.sqrt(sum(x * x for x in eng["ec_input"])))
            cos_ec = sum(a * b for a, b in zip(ec_input, eng["ec_input"])) / (q_norm * e_norm)
            diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(ec_input, eng["ec_input"])))
            if cos_ec > 0.985 and diff < 0.08:
                for j in range(self.ec_dim):
                    eng["ec_input"][j] = 0.65 * eng["ec_input"][j] + 0.35 * ec_input[j]
                for j in range(self.target_dim):
                    eng["ca1_target"][j] = 0.65 * eng["ca1_target"][j] + 0.35 * ca1_target[j]
                eng["salience"] = max(eng["salience"], salience)
                return
        self.ca3_engrams.append(
            {
                "ec_input": list(ec_input),
                "dg_sparse_code": dg_code,
                "ca1_target": list(ca1_target),
                "salience": max(0.1, salience),
                "replay_count": 0,
            }
        )


class BioBrain:
    """Arquitectura Cerebral Completa (`thalamocortical_brain.rs`)."""

    def __init__(
        self,
        sensory_dim: int,
        context_dim: int,
        output_dim: int,
        column_widths: List[int],
        seed: int,
    ) -> None:
        self.sensory_dim = sensory_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        total_in = sensory_dim + context_dim
        self.columns = []
        prev_dim = total_in
        for idx, w in enumerate(column_widths):
            self.columns.append(CorticalColumn(idx, prev_dim, w, 0.30, seed + (idx + 1) * 10007))
            prev_dim = w
        total_l56 = sum(column_widths)
        self.readout_synapses = BioSynapseMatrix(output_dim, total_l56, 0.90, 0.30, seed + 90001)
        self.direct_pathway = BioSynapseMatrix(output_dim, total_in, 0.35, 0.30, seed + 95001)
        self.hippocampus = HippocampalSystem(
            total_in, max(64, min(512, total_in * 8)), output_dim, 0.10, 18.0, 512, seed + 77777
        )
        self.theta_phase = 0.0
        self.dopamine = 1.0
        self.acetylcholine = 0.5
        self.norepinephrine = 0.2
        self.serotonin = 0.8
        self.expected_surprise = 0.25
        self.baseline_reward = 0.0
        self.context_switch_detected = False
        self.base_learning_rate = 0.035
        self.hippocampal_blend = 0.35

    def relay_input(self, x: List[float], c: List[float]) -> List[float]:
        ach_boost = 0.85 + 0.15 * self.acetylcholine
        return [v * ach_boost for v in x] + list(c)

    def forward_cognitive_cycle(self, x: List[float], c: List[float], n_gamma_cycles: int = 2) -> dict:
        thalamic_in = self.relay_input(x, c)
        _, ca1_recalled, ca1_novelty = self.hippocampus.ca3_pattern_completion(thalamic_in)
        apical_signals = [[0.0] * col.n_hidden for col in self.columns]

        for _ in range(max(1, n_gamma_cycles)):
            self.theta_phase = (self.theta_phase + math.tau / 4.0) % math.tau
            retrieval_gate = 0.55 - 0.45 * math.cos(self.theta_phase)
            bottom_up = list(thalamic_in)
            for idx, col in enumerate(self.columns):
                bottom_up = col.step_column(
                    bottom_up, apical_signals[idx], self.acetylcholine, self.norepinephrine, False
                )
            for idx in range(len(self.columns) - 2, -1, -1):
                pred_err = self.columns[idx + 1].local_prediction_error
                for j in range(min(len(apical_signals[idx]), len(pred_err))):
                    apical_signals[idx][j] = -retrieval_gate * pred_err[j]

        all_l56 = []
        for col in self.columns:
            all_l56.extend(n.event_rate for n in col.l56_pyramidal)
        ro = self.readout_synapses.forward(all_l56)
        dp = self.direct_pathway.forward(thalamic_in)
        cortical_output = [ro[i] + dp[i] for i in range(self.output_dim)]

        familiarity = clamp(1.0 - ca1_novelty, 0.0, 1.0)
        eff_blend = clamp(self.hippocampal_blend * familiarity * familiarity, 0.0, 0.85)
        output = [
            (1.0 - eff_blend) * cortical_output[i] + eff_blend * ca1_recalled[i]
            for i in range(self.output_dim)
        ]
        return {
            "output": output,
            "cortical_output": cortical_output,
            "ca1_novelty": ca1_novelty,
        }

    def observe_and_learn_local(
        self, x: List[float], c: List[float], target: List[float], encode_in_hippocampus: bool = True
    ) -> dict:
        tel = self.forward_cognitive_cycle(x, c, 2)
        thalamic_in = self.relay_input(x, c)
        pred_err = [target[i] - tel["cortical_output"][i] for i in range(self.output_dim)]
        surprise = math.sqrt(sum(e * e for e in pred_err) / max(1, self.output_dim))

        s_ratio = surprise / (self.expected_surprise + 1e-4)
        self.context_switch_detected = s_ratio > 2.2 and surprise > 0.15
        if self.context_switch_detected:
            self.norepinephrine = clamp(0.4 + 0.65 * (s_ratio - 1.0), 0.2, 2.5)
        else:
            self.norepinephrine = clamp(0.82 * self.norepinephrine + 0.18 * (0.15 + 0.25 * surprise), 0.05, 2.0)
        alpha_s = 0.35 if self.context_switch_detected else 0.08
        self.expected_surprise = (1.0 - alpha_s) * self.expected_surprise + alpha_s * surprise
        self.acetylcholine = clamp(0.30 + 1.20 * math.tanh(self.expected_surprise), 0.15, 2.0)
        rpe = -surprise - self.baseline_reward
        self.baseline_reward = 0.90 * self.baseline_reward + 0.10 * (-surprise)
        self.dopamine = clamp(1.0 + 0.65 * math.tanh(rpe) + 0.25 * self.norepinephrine, 0.20, 2.5)
        third_factor = clamp(
            self.dopamine * (0.55 + 0.30 * self.acetylcholine + 0.35 * self.norepinephrine), 0.15, 3.0
        )

        all_l56 = []
        for col in self.columns:
            all_l56.extend(n.event_rate for n in col.l56_pyramidal)
        apical_from_ro = self.readout_synapses.backward_project(pred_err)
        self.readout_synapses.update_three_factor(
            all_l56, pred_err, tel["cortical_output"], third_factor, self.base_learning_rate
        )
        self.direct_pathway.update_three_factor(
            thalamic_in, pred_err, tel["cortical_output"], third_factor, 0.35 * self.base_learning_rate
        )

        col_offsets = []
        acc = 0
        for col in self.columns:
            col_offsets.append(acc)
            acc += col.n_hidden
        feedback_from_above = [0.0] * (self.columns[-1].n_hidden if self.columns else 0)
        for idx in range(len(self.columns) - 1, -1, -1):
            offset = col_offsets[idx]
            n_h = self.columns[idx].n_hidden
            combined_apical = [
                apical_from_ro[offset + i]
                + (0.65 * feedback_from_above[i] if idx + 1 < len(self.columns) and i < len(feedback_from_above) else 0.0)
                for i in range(n_h)
            ]
            bu = list(self.columns[idx].last_bottom_up_input)
            self.columns[idx].step_column(bu, combined_apical, self.acetylcholine, self.norepinephrine, True)
            feedback_from_above = self.columns[idx].local_plasticity_step(third_factor, self.base_learning_rate)

        if encode_in_hippocampus:
            salience = clamp(0.5 + surprise + 0.3 * self.norepinephrine, 0.1, 5.0)
            self.hippocampus.encode_episode(thalamic_in, target, salience)
        return tel

    def sleep_consolidation_cycle(self, n_swr: int, cons_rate: float, shy_down: float, seed: int) -> float:
        rng = Xoshiro256StarStar(seed)
        if not self.hippocampus.ca3_engrams:
            return 0.0
        weights = [
            max(0.05, e["salience"]) / (1.0 + 0.05 * e["replay_count"])
            for e in self.hippocampus.ca3_engrams
        ]
        tw = sum(weights)
        total_err = 0.0
        for _ in range(n_swr):
            r = rng.gen_range(0.0, max(1e-9, tw))
            acc = 0.0
            chosen = len(weights) - 1
            for idx, w in enumerate(weights):
                acc += w
                if r <= acc:
                    chosen = idx
                    break
            eng = self.hippocampus.ca3_engrams[chosen]
            eng["replay_count"] += 1
            x_part = eng["ec_input"][: self.sensory_dim]
            c_part = eng["ec_input"][self.sensory_dim :]
            tel = self.observe_and_learn_local(x_part, c_part, eng["ca1_target"], False)
            err = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(tel["cortical_output"], eng["ca1_target"]))
                / max(1, self.output_dim)
            )
            total_err += err

        for col in self.columns:
            col.syn_l4_to_l23_basal.consolidate_synapses(cons_rate, shy_down)
            col.syn_l23_to_l56.consolidate_synapses(cons_rate, shy_down)
            col.syn_l56_topdown_pred.consolidate_synapses(cons_rate, shy_down)
            col.reset_short_term_state()
        self.readout_synapses.consolidate_synapses(cons_rate, shy_down)
        self.direct_pathway.consolidate_synapses(cons_rate, shy_down)
        self.readout_synapses.reset_short_term_state()
        self.direct_pathway.reset_short_term_state()
        return total_err / max(1, n_swr)

    def predict(self, x: List[float], c: List[float]) -> List[float]:
        return self.forward_cognitive_cycle(x, c, 2)["output"]

    def total_synaptic_params(self) -> int:
        c_p = sum(
            col.syn_l4_to_l23_basal.n_params()
            + col.syn_l23_to_l56.n_params()
            + col.syn_l56_topdown_pred.n_params()
            + col.n_hidden
            for col in self.columns
        )
        return c_p + self.readout_synapses.n_params() + self.direct_pathway.n_params()


def rmse(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a)))


def main() -> None:
    print("================================================================================")
    print(" NEOCORTEX (BioBrain) - Arquitectura Neuronal Bioinspirada en el Cerebro Humano")
    print("================================================================================\n")

    brain = BioBrain(12, 10, 12, [32, 24, 16], 0xB10_2026)
    patterns = []
    for k in range(8):
        kf = float(k)
        x = [math.sin(i * 0.45 + kf * 0.7) * 0.8 for i in range(12)]
        c = [0.0] * 10
        c[k % 8] = 1.0
        c[8] = 0.35 * math.cos(kf * 0.5)
        c[9] = -0.25 * math.sin(kf * 0.5)
        y = [
            math.tanh(0.75 * x[i] + 0.35 * math.tanh(x[(i + 1) % 12]) + c[8] * x[11 - i] - 0.2 * c[9])
            for i in range(12)
        ]
        patterns.append((x, c, y))

    init_rmse = sum(rmse(brain.predict(x, c), y) for x, c, y in patterns) / len(patterns)
    for _ in range(35):
        for x, c, y in patterns:
            brain.observe_and_learn_local(x, c, y, True)

    final_rmse = sum(rmse(brain.predict(x, c), y) for x, c, y in patterns) / len(patterns)
    cortical_only_rmse = (
        sum(rmse(brain.forward_cognitive_cycle(x, c, 2)["cortical_output"], y) for x, c, y in patterns)
        / len(patterns)
    )

    print("[Exp 1] Asignacion de Credito Dendritica Local (3 columnas L4/L2-3/L5-6, 0 Backprop):")
    print(f"        RMSE inicial           : {init_rmse:.6f}")
    print(f"        RMSE solo Neocorteza   : {cortical_only_rmse:.6f}")
    print(f"        RMSE Cerebro Completo  : {final_rmse:.6f} (reduccion {(1.0 - final_rmse / init_rmse) * 100.0:.1f}%)\n")

    dale_violations = 0
    total_synapses_checked = 0
    for col in brain.columns:
        for mat in (col.syn_l4_to_l23_basal, col.syn_l23_to_l56, col.syn_l56_topdown_pred):
            for we, wi in zip(mat.w_exc, mat.w_inh):
                total_synapses_checked += 2
                if we < 0.0 or wi < 0.0:
                    dale_violations += 1
    print("[Exp 2] Ley de Dale (W^E >= 0, W^I >= 0) y Microcircuito PV/SST/VIP:")
    print(f"        Sinapsis verificadas   : {total_synapses_checked} (Violaciones = {dale_violations})")
    print(f"        Parametros corticales  : {brain.total_synaptic_params()}\n")

    stp_mat = BioSynapseMatrix(6, 6, 1.0, 0.30, 303)
    probe = [0.4] * 6
    baseline_out = stp_mat.forward_stateless(probe)
    burst_stim = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    for _ in range(6):
        stp_mat.forward(burst_stim)
    post_burst_out = stp_mat.forward_stateless(probe)
    stp_trace_diff = rmse(baseline_out, post_burst_out)
    print("[Exp 3] Memoria de Trabajo Sinaptica Tsodyks-Markram (sin cambio en W):")
    print(f"        u(Ca2+ activo vs reposo): {stp_mat.stp_u[0]:.4f} vs {stp_mat.stp_u[3]:.4f}")
    print(f"        x(vesiculas act vs rep) : {stp_mat.stp_x[0]:.4f} vs {stp_mat.stp_x[3]:.4f}")
    print(f"        Modulacion de lectura   : dRMSE = {stp_trace_diff:.6f}\n")

    hippo = HippocampalSystem(12, 160, 6, 0.08, 22.0, 128, 404)
    ep1 = [0.9, -0.7, 0.5, -0.4, 0.8, 0.3, -0.6, 0.7, -0.2, 0.6, -0.8, 0.4]
    ep2 = [0.85, -0.65, 0.48, -0.35, 0.75, 0.28, -0.55, 0.65, -0.15, 0.55, -0.75, 0.35]
    dg1 = hippo.dentate_gyrus_pattern_separation(ep1)
    dg2 = hippo.dentate_gyrus_pattern_separation(ep2)
    cos_in = sum(a * b for a, b in zip(ep1, ep2)) / (
        math.sqrt(sum(x * x for x in ep1)) * math.sqrt(sum(x * x for x in ep2))
    )
    cos_dg = sum(a * b for a, b in zip(dg1, dg2))
    target1 = [0.8, -0.6, 0.4, -0.9, 0.7, -0.3]
    target2 = [-0.7, 0.5, -0.3, 0.8, -0.6, 0.4]
    hippo.encode_episode(ep1, target1, 2.0)
    hippo.encode_episode(ep2, target2, 2.0)
    occluded_ep1 = ep1[:6] + [0.0] * 6
    reconstructed_ec, recalled_t1, novelty = hippo.ca3_pattern_completion(occluded_ep1)
    completion_rmse = rmse(recalled_t1, target1)
    ec_recon_rmse = rmse(reconstructed_ec, ep1)
    print("[Exp 4] Circuito Hipocampal (DG Separacion + CA3 Completado al 50% oclusion):")
    print(f"        Similitud coseno entrada: {cos_in:.4f} -> Giro Dentado: {cos_dg:.4f} (ortogonalizacion)")
    print(f"        RMSE reconstruccion EC  : {ec_recon_rmse:.6f} | RMSE target CA1: {completion_rmse:.6f} | Novedad: {novelty:.4f}\n")

    brain_cls = BioBrain(8, 4, 4, [24, 16], 505)
    fam_a = []
    for i in range(4):
        f = i * 0.6
        x = [math.sin(f + j * 0.5) for j in range(8)]
        c = [1.0, 0.0, 0.5, -0.5]
        y = [0.8 * x[j] - 0.4 * x[j + 4] for j in range(4)]
        fam_a.append((x, c, y))
    fam_b = []
    for i in range(4):
        f = i * 0.6 + 1.2
        x = [math.cos(f + j * 0.5) for j in range(8)]
        c = [0.0, 1.0, -0.5, 0.5]
        y = [-0.7 * x[j] + 0.6 * x[j + 4] for j in range(4)]
        fam_b.append((x, c, y))

    for _ in range(30):
        for x, c, y in fam_a:
            brain_cls.observe_and_learn_local(x, c, y, True)
    sleep_err_1 = brain_cls.sleep_consolidation_cycle(32, 0.70, 0.015, 1001)

    for _ in range(30):
        for x, c, y in fam_b:
            brain_cls.observe_and_learn_local(x, c, y, True)
    sleep_err_2 = brain_cls.sleep_consolidation_cycle(48, 0.55, 0.015, 1002)

    rmse_fam_a = sum(rmse(brain_cls.predict(x, c), y) for x, c, y in fam_a) / len(fam_a)
    rmse_fam_b = sum(rmse(brain_cls.predict(x, c), y) for x, c, y in fam_b) / len(fam_b)
    print("[Exp 5] Aprendizaje Continuo Secuencial (Familia A -> Familia B) + Sueno SWR:")
    print(f"        Error replay Sueno 1/2  : {sleep_err_1:.6f} / {sleep_err_2:.6f}")
    print(f"        Retencion Familia A     : RMSE = {rmse_fam_a:.6f}")
    print(f"        Adquisicion Familia B   : RMSE = {rmse_fam_b:.6f}\n")

    ne_before = brain_cls.norepinephrine
    ach_before = brain_cls.acetylcholine
    shock_x = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    shock_c = [1.0, 0.0, 0.5, -0.5]
    shock_y = [-2.0, 2.0, -2.0, 2.0]
    brain_cls.observe_and_learn_local(shock_x, shock_c, shock_y, False)
    print("[Exp 6] Respuesta Neuromoduladora ante Cambio Abrupto de Regimen (Locus Coeruleus):")
    print(f"        Norepinefrina (NE)      : {ne_before:.4f} -> {brain_cls.norepinephrine:.4f} (Context Switch = {brain_cls.context_switch_detected})")
    print(f"        Acetilcolina (ACh)      : {ach_before:.4f} -> {brain_cls.acetylcholine:.4f} (Desinhibicion VIP-SST activa)\n")

    summary = {
        "architecture": "BioBrain_Neocortex_CLS",
        "exp1_local_dendritic_learning": {
            "initial_rmse": round(init_rmse, 6),
            "cortical_only_rmse": round(cortical_only_rmse, 6),
            "full_brain_rmse": round(final_rmse, 6),
        },
        "exp2_dales_law": {
            "synapses_checked": total_synapses_checked,
            "violations": dale_violations,
            "total_params": brain.total_synaptic_params(),
        },
        "exp3_tsodyks_markram_stp": {
            "u_active": round(stp_mat.stp_u[0], 4),
            "u_rest": round(stp_mat.stp_u[3], 4),
            "x_active": round(stp_mat.stp_x[0], 4),
            "x_rest": round(stp_mat.stp_x[3], 4),
            "readout_modulation_rmse": round(stp_trace_diff, 6),
        },
        "exp4_hippocampus_dg_ca3": {
            "cosine_input": round(cos_in, 4),
            "cosine_dentate_gyrus": round(cos_dg, 4),
            "ca3_50pct_occluded_target_rmse": round(completion_rmse, 6),
        },
        "exp5_continual_learning_swr_sleep": {
            "family_a_retained_rmse": round(rmse_fam_a, 6),
            "family_b_learned_rmse": round(rmse_fam_b, 6),
        },
        "exp6_neuromodulation_reset": {
            "ne_before": round(ne_before, 4),
            "ne_after": round(brain_cls.norepinephrine, 4),
            "ach_before": round(ach_before, 4),
            "ach_after": round(brain_cls.acetylcholine, 4),
            "context_switch_detected": brain_cls.context_switch_detected,
        },
    }
    print("Resumen JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
