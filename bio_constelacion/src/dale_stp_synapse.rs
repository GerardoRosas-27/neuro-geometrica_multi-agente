//! Proyecciones Sinápticas Híbridas: Ley de Dale Estricta ($W^E \ge 0, W^I \ge 0$),
//! Plasticidad de Corto Plazo Tsodyks-Markram, Poda/Sinaptogénesis Estructural (`tau`, `cm`),
//! Consolidación Metaplástica en Sueño y Campo Dendrítico Rápido BTSP (`FastDendriticBtspField`).
//!
//! Combina:
//! 1. **De CONSTELACIÓN v4**: constantes temporales adaptativas por sinapsis ($\tau_{ij}$),
//!    utilidad acumulada ($u_{ij}$), máscara estructural de poda y recrecimiento (`cm`),
//!    acotamiento de norma por fila ($b_{\text{in}}$) y adaptación rápida compartida entre puntos.
//! 2. **De NEOCORTEX (`BioBrain`)**: cumplimiento estricto de la **Ley de Dale** vía descomposición
//!    glutamatérgica/GABAérgica disináptica ($W^E \ge 0, W^I \ge 0$), dinámica vesicular y de
//!    $\text{Ca}^{2+}$ presináptico (**Tsodyks & Markram, 1997**), trazas de elegibilidad de
//!    3 factores (**Gerstner et al., 2018**), plasticidad rápida de meseta dendrítica **BTSP**
//!    (**Bittner et al., 2017**, *Science*) con normalización divisiva **PV+** (**Carandini & Heeger, 2012**)
//!    y consolidación estructural (**Frey & Morris, 1997; Benna & Fusi, 2016**).

#![allow(clippy::needless_range_loop)]

use crate::rng::Xoshiro256StarStar;

/// Capa sináptica cortical lenta con Ley de Dale, STP, poda/recrecimiento y consolidación.
#[derive(Clone, Debug)]
pub struct DaleSynapseLayer {
    pub n_out: usize,
    pub n_in: usize,
    /// Pesos excitatorios AMPA/NMDA ($W^E \ge 0$).
    pub w_exc: Vec<f64>,
    /// Pesos inhibitorios GABAérgicos disinápticos ($W^I \ge 0$).
    pub w_inh: Vec<f64>,
    /// Máscara de conectividad estructural (`1.0` espina activa, `0.0` podada).
    pub conn_mask: Vec<f64>,
    /// Constante de tiempo metaplástica por sinapsis ($\tau_{ij} \in [\tau_{\min}, \tau_{\max}]$).
    pub tau: Vec<f64>,
    /// Utilidad acumulada por sinapsis ($u_{ij}$) para poda estructural.
    pub utility: Vec<f64>,
    /// Contador de ciclos con baja utilidad antes de podar.
    pub low_count: Vec<u8>,
    /// Actividad correlacional potencial para sinaptogénesis (recrecimiento de espinas).
    pub potential_act: Vec<f64>,
    /// Traza de elegibilidad Neo-Hebbiana de 3 factores ($e_{ij}$).
    pub eligibility: Vec<f64>,
    /// Andamiaje estructural consolidado durante el sueño ($W^{\text{cons}}_{ij}$).
    pub w_consolidated: Vec<f64>,
    /// Madurez/rigidez metaplástica de la espina ($\Omega_{ij} \ge 0$).
    pub spine_maturity: Vec<f64>,
    /// Etiqueta sináptica acumulada (*Synaptic Tagging*, Frey & Morris 1997).
    pub tag_acc: Vec<f64>,
    /// Estado de corto plazo Tsodyks-Markram: facilitación de calcio $u_j$ y vesículas $x_j$.
    pub stp_u: Vec<f64>,
    pub stp_x: Vec<f64>,
    pub u0: f64,
    pub n_pruned: usize,
    pub n_grown: usize,
    pub n_row_rescaled: usize,
}

impl DaleSynapseLayer {
    pub fn new(
        n_out: usize,
        n_in: usize,
        w_init_scale: f64,
        tau_min: f64,
        u0: f64,
        rng: &mut Xoshiro256StarStar,
    ) -> Self {
        let total = n_out * n_in;
        let std = w_init_scale / (n_in.max(1) as f64).sqrt();
        let co_tone = 0.02 * std.max(1e-3);

        let mut w_exc = vec![0.0; total];
        let mut w_inh = vec![0.0; total];
        let mut w_consolidated = vec![0.0; total];

        for idx in 0..total {
            let w = rng.randn() * std;
            if w >= 0.0 {
                w_exc[idx] = w + co_tone;
                w_inh[idx] = co_tone;
            } else {
                w_exc[idx] = co_tone;
                w_inh[idx] = -w + co_tone;
            }
            w_consolidated[idx] = w;
        }

        Self {
            n_out,
            n_in,
            w_exc,
            w_inh,
            conn_mask: vec![1.0; total],
            tau: vec![tau_min; total],
            utility: vec![0.0; total],
            low_count: vec![0; total],
            potential_act: vec![0.0; total],
            eligibility: vec![0.0; total],
            w_consolidated,
            spine_maturity: vec![0.0; total],
            tag_acc: vec![0.0; total],
            stp_u: vec![u0; n_in],
            stp_x: vec![1.0; n_in],
            u0,
            n_pruned: 0,
            n_grown: 0,
            n_row_rescaled: 0,
        }
    }

    #[inline]
    pub fn effective_weight(&self, idx: usize) -> f64 {
        (self.w_exc[idx] - self.w_inh[idx]) * self.conn_mask[idx]
    }

    pub fn reset_episode_stp(&mut self) {
        self.stp_u.fill(self.u0);
        self.stp_x.fill(1.0);
        self.eligibility.fill(0.0);
    }

    /// Paso presináptico de corto plazo Tsodyks-Markram (actualiza `stp_u` y `stp_x` una vez por cuadro).
    pub fn step_stp(&mut self, mean_pre_abs: &[f64], stp_strength: f64) -> Vec<f64> {
        let mut gain_vec = vec![1.0; self.n_in];
        if stp_strength <= 0.0 {
            return gain_vec;
        }
        for j in 0..self.n_in {
            let act = mean_pre_abs[j].clamp(0.0, 1.0);
            let u = self.stp_u[j];
            let x = self.stp_x[j];
            let u_next = u + 0.08 * (self.u0 - u) + self.u0 * (1.0 - u) * act;
            let x_next = x + 0.15 * (1.0 - x) - u_next * x * act;
            self.stp_u[j] = u_next.clamp(0.05, 0.99);
            self.stp_x[j] = x_next.clamp(0.05, 1.0);
            let raw_gain = if self.u0 > 1e-6 {
                (self.stp_u[j] * self.stp_x[j]) / self.u0
            } else {
                1.0
            };
            gain_vec[j] = 1.0 + stp_strength * (raw_gain - 1.0);
        }
        gain_vec
    }

    /// Proyección hacia abajo / adelante $y = W_{\text{eff}} x$.
    pub fn forward_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_out];
        for i in 0..self.n_out {
            let row = i * self.n_in;
            let mut s = 0.0;
            for j in 0..self.n_in {
                s += self.effective_weight(row + j) * x[j];
            }
            out[i] = s;
        }
        out
    }

    /// Proyección transpuesta $g = W_{\text{eff}}^\top e$ para relajación dendrítica local.
    pub fn transpose_vec(&self, err: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_in];
        for i in 0..self.n_out {
            let e = err[i];
            if e.abs() < 1e-15 {
                continue;
            }
            let row = i * self.n_in;
            for j in 0..self.n_in {
                out[j] += self.effective_weight(row + j) * e;
            }
        }
        out
    }

    /// Curvatura diagonal local $\sum_i \Pi_i W_{\text{eff}, ij}^2$ (precondicionador de Constelación v4).
    pub fn diagonal_curvature(&self, pi_out: &[f64]) -> Vec<f64> {
        let mut curv = vec![0.0; self.n_in];
        for i in 0..self.n_out {
            let p = pi_out[i];
            let row = i * self.n_in;
            for j in 0..self.n_in {
                let w = self.effective_weight(row + j);
                curv[j] += p * w * w;
            }
        }
        curv
    }

    /// Actualización Neo-Hebbiana de 3 factores respetando la Ley de Dale ($W^E \ge 0, W^I \ge 0$),
    /// constante metaplástica $\tau_{ij}$, protección de espinas consolidadas $\Omega_{ij}$ y cota $b_{\text{in}}$.
    #[allow(clippy::too_many_arguments)]
    pub fn update_dale_three_factor(
        &mut self,
        grad_hebb: &[f64],
        third_factor: f64,
        eta_w: f64,
        lambda_decay: f64,
        energy_reduction: f64,
        b_in_bound: f64,
        tau_min: f64,
        tau_max: f64,
        enable_prune_tau: bool,
    ) {
        let eff_lr = eta_w * third_factor;
        let zeta = 0.01;
        let rho_tau = 0.01;

        for idx in 0..self.w_exc.len() {
            let g = grad_hebb[idx];
            self.potential_act[idx] = (1.0 - zeta) * self.potential_act[idx] + zeta * g.abs();
            if self.conn_mask[idx] == 0.0 {
                continue;
            }

            self.eligibility[idx] = 0.45 * self.eligibility[idx] + g;
            self.tag_acc[idx] += g.abs();

            let tau = if enable_prune_tau { self.tau[idx] } else { 1.0 };
            let maturity_gate = 1.0 / (1.0 + 0.35 * self.spine_maturity[idx]);
            let w_eff = self.w_exc[idx] - self.w_inh[idx];
            let cons_pull = 0.015 * self.spine_maturity[idx] * (self.w_consolidated[idx] - w_eff);

            let dw = (eff_lr / tau) * maturity_gate * self.eligibility[idx] + eta_w * cons_pull
                - lambda_decay * w_eff;

            self.apply_dale_delta(idx, dw);

            if enable_prune_tau {
                let w_new = (self.w_exc[idx] - self.w_inh[idx]).abs();
                self.utility[idx] = (1.0 - zeta) * self.utility[idx]
                    + zeta * dw.abs() * w_new * energy_reduction.max(1e-4);
            }
        }

        if enable_prune_tau {
            let mut sum_u = 0.0;
            let mut cnt = 0usize;
            for idx in 0..self.utility.len() {
                if self.conn_mask[idx] > 0.0 {
                    sum_u += self.utility[idx];
                    cnt += 1;
                }
            }
            let u_mean = sum_u / (cnt.max(1) as f64);
            if u_mean > 0.0 {
                for idx in 0..self.tau.len() {
                    if self.conn_mask[idx] > 0.0 {
                        self.tau[idx] = (self.tau[idx]
                            * (1.0 + rho_tau * (self.utility[idx] / u_mean - 1.0)))
                            .clamp(tau_min, tau_max);
                    }
                }
            }
        }

        // Cota homeostática por fila (equivalente a escalado sináptico de Turrigiano + b_in de Constelación)
        for r in 0..self.n_out {
            let row = r * self.n_in;
            let mut row_l1 = 0.0;
            for j in 0..self.n_in {
                row_l1 += self.effective_weight(row + j).abs();
            }
            if row_l1 > b_in_bound && row_l1 > 1e-9 {
                let scale = b_in_bound / row_l1;
                for j in 0..self.n_in {
                    self.w_exc[row + j] = (self.w_exc[row + j] * scale).max(1e-5);
                    self.w_inh[row + j] = (self.w_inh[row + j] * scale).max(1e-5);
                }
                self.n_row_rescaled += 1;
            }
        }
    }

    #[inline]
    fn apply_dale_delta(&mut self, idx: usize, dw: f64) {
        if dw >= 0.0 {
            self.w_exc[idx] += 0.5 * dw;
            self.w_inh[idx] = (self.w_inh[idx] - 0.5 * dw).max(1e-4);
        } else {
            let adw = -dw;
            self.w_inh[idx] += 0.5 * adw;
            self.w_exc[idx] = (self.w_exc[idx] - 0.5 * adw).max(1e-4);
        }
        // Renormalizar co-tono basal E/I para evitar deriva ilimitada
        let co = self.w_exc[idx].min(self.w_inh[idx]);
        if co > 0.005 {
            let ex = co - 0.005;
            self.w_exc[idx] -= ex;
            self.w_inh[idx] -= ex;
        }
    }

    /// Poda estructural de espinas inactivas y sinaptogénesis guiada por actividad (`prune_and_grow`).
    pub fn structural_prune_and_grow(&mut self, w_min: f64, u_min_rel: f64, tau_min: f64) {
        let mut sum_u = 0.0;
        let mut cnt = 0usize;
        for idx in 0..self.utility.len() {
            if self.conn_mask[idx] > 0.0 {
                sum_u += self.utility[idx];
                cnt += 1;
            }
        }
        let u_mean = sum_u / (cnt.max(1) as f64);

        for idx in 0..self.conn_mask.len() {
            if self.conn_mask[idx] == 0.0 {
                continue;
            }
            let w_abs = (self.w_exc[idx] - self.w_inh[idx]).abs();
            if w_abs < w_min
                && self.utility[idx] < u_min_rel * u_mean
                && self.spine_maturity[idx] < 0.5
            {
                self.low_count[idx] += 1;
                if self.low_count[idx] >= 2 {
                    self.conn_mask[idx] = 0.0;
                    self.low_count[idx] = 0;
                    self.n_pruned += 1;
                }
            } else {
                self.low_count[idx] = 0;
            }
        }

        // Sinaptogénesis: reactivar la conexión podada con mayor correlación potencial
        let mut best_idx: Option<usize> = None;
        let mut best_act = 0.0;
        for idx in 0..self.conn_mask.len() {
            if self.conn_mask[idx] == 0.0 && self.potential_act[idx] > best_act {
                best_act = self.potential_act[idx];
                best_idx = Some(idx);
            }
        }
        if let Some(bi) = best_idx {
            self.conn_mask[bi] = 1.0;
            self.w_exc[bi] = 0.005;
            self.w_inh[bi] = 0.005;
            self.tau[bi] = tau_min;
            self.utility[bi] = 0.0;
            self.n_grown += 1;
        }
    }

    /// Consolidación durante sueño NREM (Frey & Morris 1997 + Tononi & Cirelli 2014 SHY).
    pub fn consolidate_sleep(&mut self, rate: f64, shy_downscale: f64) {
        let max_tag = self.tag_acc.iter().copied().fold(1e-9_f64, f64::max);
        for idx in 0..self.w_exc.len() {
            let w_eff = self.w_exc[idx] - self.w_inh[idx];
            let norm_tag = (self.tag_acc[idx] / max_tag).clamp(0.0, 1.0);
            let alpha = (rate * (0.20 + 0.80 * norm_tag)).clamp(0.0, 1.0);
            self.w_consolidated[idx] += alpha * (w_eff - self.w_consolidated[idx]);
            self.spine_maturity[idx] = (self.spine_maturity[idx] + 1.2 * norm_tag).clamp(0.0, 10.0);

            if norm_tag < 0.10 && shy_downscale > 0.0 {
                let s = 1.0 - shy_downscale * (1.0 - norm_tag);
                self.w_exc[idx] = (self.w_exc[idx] * s).max(1e-4);
                self.w_inh[idx] = (self.w_inh[idx] * s).max(1e-4);
            }
            self.tag_acc[idx] = 0.0;
        }
    }

    /// Verifica el cumplimiento estricto de la Ley de Dale: devuelve `(sinapsis_totales, violaciones)`.
    pub fn verify_dales_law(&self) -> (usize, usize) {
        let mut viol = 0usize;
        for (&we, &wi) in self.w_exc.iter().zip(self.w_inh.iter()) {
            if we < 0.0 || wi < 0.0 {
                viol += 1;
            }
        }
        (self.w_exc.len() * 2, viol)
    }

    /// Devuelve `(violaciones, sinapsis_totales)` para la auditoría.
    pub fn audit_dale_law(&self) -> (usize, usize) {
        let (total, viol) = self.verify_dales_law();
        (viol, total)
    }
}

/// Campo Sináptico Rápido de Meseta Dendrítica (`FastDendriticBtspField`) que respeta la Ley de Dale.
///
/// Reemplaza el optimizador algebraico sin signo de Constelación v4 por **Plasticidad Sináptica de
/// Escala Conductual (BTSP, Bittner et al., 2017 *Science*)** con **Normalización Divisiva de
/// Conductancia PV+ (Carandini & Heeger, 2012)**:
/// - Pesos rápidos excitatorios $W^{F, E} \ge 0$ e inhibitorios $W^{F, I} \ge 0$.
/// - Pesos lentos corticales $W^{P, E} \ge 0$ y $W^{P, I} \ge 0$.
#[derive(Clone, Debug)]
pub struct FastDendriticBtspField {
    pub feat_dim: usize,
    /// Pesos lentos corticales L5/6 ($2 \times d$, sujetos a Ley de Dale).
    pub wp_exc: Vec<f64>,
    pub wp_inh: Vec<f64>,
    pub wp_cons: Vec<f64>,
    pub wp_maturity: Vec<f64>,
    /// Pesos rápidos episódicos BTSP ($2 \times d$, sujetos a Ley de Dale).
    pub wf_exc: Vec<f64>,
    pub wf_inh: Vec<f64>,
    /// Precisión sensorial de transición en L2/3 ($\Pi^P \in \mathbb{R}^2$).
    pub pi_pred: [f64; 2],
}

impl FastDendriticBtspField {
    pub fn new(feat_dim: usize) -> Self {
        let total = 2 * feat_dim;
        Self {
            feat_dim,
            wp_exc: vec![0.002; total],
            wp_inh: vec![0.002; total],
            wp_cons: vec![0.0; total],
            wp_maturity: vec![0.0; total],
            wf_exc: vec![0.001; total],
            wf_inh: vec![0.001; total],
            pi_pred: [1.0, 1.0],
        }
    }

    pub fn reset_fast_field(&mut self) {
        self.wf_exc.fill(0.001);
        self.wf_inh.fill(0.001);
    }

    /// Inicializa el campo rápido a partir de un operador prior contextual/hipocampal (respetando Dale).
    pub fn set_prior_operator(&mut self, op_eff: &[f64], blend: f64) {
        for idx in 0..self.wf_exc.len().min(op_eff.len()) {
            let cur = self.wf_exc[idx] - self.wf_inh[idx];
            let target = (1.0 - blend) * cur + blend * op_eff[idx];
            if target >= 0.0 {
                self.wf_exc[idx] = target + 0.001;
                self.wf_inh[idx] = 0.001;
            } else {
                self.wf_exc[idx] = 0.001;
                self.wf_inh[idx] = -target + 0.001;
            }
        }
    }

    pub fn current_fast_operator(&self) -> Vec<f64> {
        self.wf_exc
            .iter()
            .zip(&self.wf_inh)
            .map(|(e, i)| e - i)
            .collect()
    }

    /// Predicción residual por punto: $\hat{q}(t+1) = q(t) + (W^P + W^F) \phi(t)$.
    pub fn predict_point(&self, q_prev: &[f64; 2], phi: &[f64], use_fast: bool) -> [f64; 2] {
        let d = self.feat_dim;
        let mut out = *q_prev;
        for c in 0..2 {
            let row = c * d;
            let mut delta = 0.0;
            for j in 0..d {
                let wp = self.wp_exc[row + j] - self.wp_inh[row + j];
                let wf = if use_fast {
                    self.wf_exc[row + j] - self.wf_inh[row + j]
                } else {
                    0.0
                };
                delta += (wp + wf) * phi[j];
            }
            out[c] += delta;
        }
        out
    }

    /// Actualiza los pesos rápidos BTSP (modulados por desinhibición VIP-SST / ACh-NE)
    /// y los pesos corticales lentos de 3 factores respetando estrictamente la Ley de Dale.
    #[allow(clippy::too_many_arguments)]
    pub fn update_btsp_and_slow(
        &mut self,
        q_prevs: &[[f64; 2]],
        phis: &[Vec<f64>],
        q_targets: &[[f64; 2]],
        train_slow: bool,
        eta_slow: f64,
        third_factor: f64,
        enable_fast_btsp: bool,
        eta_fast: f64,
        vip_ach_gate: f64,
        n_btsp_cycles: usize,
    ) {
        let n_pts = phis.len();
        if n_pts == 0 {
            return;
        }
        let d = self.feat_dim;

        // 1. Actualización lenta de 3 factores en W^P (antes de que W^F absorba la transición nueva)
        if train_slow && eta_slow > 0.0 {
            let mut grad_slow = vec![0.0; 2 * d];
            let mut err_sq = [0.0; 2];
            for p in 0..n_pts {
                let pred = self.predict_point(&q_prevs[p], &phis[p], enable_fast_btsp);
                for c in 0..2 {
                    let e = q_targets[p][c] - pred[c];
                    err_sq[c] += e * e / (n_pts as f64);
                    let pe = self.pi_pred[c] * e;
                    for j in 0..d {
                        grad_slow[c * d + j] += pe * phis[p][j] / (n_pts as f64);
                    }
                }
            }
            for c in 0..2 {
                self.pi_pred[c] = (self.pi_pred[c]
                    + 0.001 * 0.5 * (1.0 / self.pi_pred[c] - err_sq[c]))
                    .clamp(0.1, 10.0);
            }
            let geom = (self.pi_pred[0] * self.pi_pred[1]).sqrt().max(1e-6);
            self.pi_pred[0] /= geom;
            self.pi_pred[1] /= geom;

            let eff_lr = eta_slow * third_factor;
            for idx in 0..2 * d {
                let gate = 1.0 / (1.0 + 0.4 * self.wp_maturity[idx]);
                let wp = self.wp_exc[idx] - self.wp_inh[idx];
                let pull = 0.01 * self.wp_maturity[idx] * (self.wp_cons[idx] - wp);
                let dw = eff_lr * gate * grad_slow[idx] + eta_slow * pull - 1e-6 * wp;
                apply_dale_pair(&mut self.wp_exc[idx], &mut self.wp_inh[idx], dw);
            }
        }

        // 2. Plasticidad Rápida de Meseta Dendrítica (Bio-BTSP, Bittner et al., 2017)
        // Normalizada divisivamente por la conductancia dendrítica total (PV+, Carandini & Heeger 2012)
        // y abierta por la compuerta de desinhibición VIP+/ACh.
        if enable_fast_btsp && eta_fast > 0.0 && n_btsp_cycles > 0 {
            let pv_conductance: f64 = phis
                .iter()
                .map(|phi| phi.iter().map(|v| v * v).sum::<f64>())
                .sum::<f64>()
                + 1e-6;
            let eff_eta_f =
                (eta_fast * (0.65 + 0.35 * vip_ach_gate.clamp(0.0, 2.0))).clamp(0.05, 1.35);

            for _ in 0..n_btsp_cycles {
                let mut g_fast = vec![0.0; 2 * d];
                for p in 0..n_pts {
                    let pred = self.predict_point(&q_prevs[p], &phis[p], true);
                    for c in 0..2 {
                        let e = q_targets[p][c] - pred[c];
                        for j in 0..d {
                            g_fast[c * d + j] += e * phis[p][j];
                        }
                    }
                }
                for idx in 0..2 * d {
                    // Ligera regularización homeostática para evitar sobreajuste en características ruidosas
                    let wf = self.wf_exc[idx] - self.wf_inh[idx];
                    let dw = eff_eta_f * g_fast[idx] / pv_conductance - 1e-5 * wf;
                    apply_dale_pair(&mut self.wf_exc[idx], &mut self.wf_inh[idx], dw);
                }
            }
        }
    }

    pub fn consolidate_sleep(&mut self, rate: f64) {
        for idx in 0..self.wp_exc.len() {
            let wp = self.wp_exc[idx] - self.wp_inh[idx];
            self.wp_cons[idx] += rate * (wp - self.wp_cons[idx]);
            self.wp_maturity[idx] = (self.wp_maturity[idx] + 0.5 * wp.abs()).clamp(0.0, 8.0);
        }
    }

    pub fn verify_dales_law(&self) -> (usize, usize) {
        let mut viol = 0usize;
        for (&we, &wi) in self
            .wp_exc
            .iter()
            .zip(&self.wp_inh)
            .chain(self.wf_exc.iter().zip(&self.wf_inh))
        {
            if we < 0.0 || wi < 0.0 {
                viol += 1;
            }
        }
        ((self.wp_exc.len() + self.wf_exc.len()) * 2, viol)
    }

    pub fn audit_dale_law(&self) -> (usize, usize) {
        let (total, viol) = self.verify_dales_law();
        (viol, total)
    }
}

#[inline]
pub fn apply_dale_pair(w_exc: &mut f64, w_inh: &mut f64, dw: f64) {
    if dw >= 0.0 {
        *w_exc += 0.5 * dw;
        *w_inh = (*w_inh - 0.5 * dw).max(1e-4);
    } else {
        let adw = -dw;
        *w_inh += 0.5 * adw;
        *w_exc = (*w_exc - 0.5 * adw).max(1e-4);
    }
    let co = (*w_exc).min(*w_inh);
    if co > 0.002 {
        let ex = co - 0.002;
        *w_exc -= ex;
        *w_inh -= ex;
    }
}
