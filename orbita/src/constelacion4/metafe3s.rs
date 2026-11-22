//! METAFE-3S — Control Metacognitivo de Tres Escalas sobre Energía Libre Variacional
//!
//! Aplica la arquitectura de Control Metacognitivo de Tres Escalas (Núcleo Dual W_0 + F_t,
//! Monitor de Conflicto ACC m_t ∈ R^4, Refutación Cruzada Inversa Φ_rev, Controlador
//! Neuromodulador Recurrente MetaNet GRU [α_t, λ_t, β_t, g_t], Plasticidad de Tres Factores
//! con Trazas de Elegibilidad E_t y Proyección Espectral Π_ρ, y Consolidación Offline EWC
//! desde Buffer Episódico) sobre la arquitectura puntera METAFE v2 (`metafe.rs`).
//!
//! Principios estrictos:
//! - Cero modificación al comportamiento de `METAFE v2` (se evalúa emparejado bit a bit).
//! - Cero fuga de familias retenidas {8, 10, 11}, 0 consultas TEST, determinismo bit a bit.
//! - Ablaciones causales emparejadas: `METAFE-3S-NoFast`, `METAFE-3S-NoRefute`, `METAFE-3S-NoNM`,
//!   más experimento dedicado de consolidación continua `EWC` vs `NoEWC` y cambio de régimen online.

use super::atlas::make_syn_ep;
use super::metafe::{
    cold_start_step, compute_auroc, compute_aurc, compute_brier, compute_ece, e33_metafe,
    episode_invariants, eval_metafe_set, evaluate_all_hypotheses_genesis, evaluate_hypothesis,
    fit_hypothesis, fit_hypothesis_ex, gen_meta_training_eps, gen_ood_synthetic_eps, metafe_arms,
    null_free_energy, pearson_corr, rmax_frame, run_metafe_episode, seeds, softmax_scores,
    solve_spd, spearman_corr, train_cold_start_prior, train_metafe_controller,
    FittedHyp, HypDiag, HypKind, MetaController, MetaFeCfg, HYP_NAMES, N_HYP, N_INV,
    XOR_METAFE_TRAIN, XOR_METAFE_VAL,
};
use super::*;

const XOR_3S_SHIFT: u64 = 0x35_541F_7001;
const XOR_3S_CONT: u64 = 0x35_C047_0002;
const XOR_3S_NM: u64 = 0x35_4E4D_0003;

/// Dimensión de rasgos locales normalizados ψ(q, u) para los pesos rápidos acotados F_t ∈ R^{2×6}.
pub const N_FAST_FEAT: usize = 6;
/// Dimensión del vector de telemetría ACC m_t ∈ R^4: [sorpresa, desacuerdo, estancamiento, saturación].
pub const N_ACC: usize = 4;
/// Dimensión oculta del controlador neuromodulador recurrente GRU MetaNet(m_t, s_{t-1}).
pub const N_GRU_H: usize = 4;
/// Número de modos cognitivos dlPFC g_t^mode ∈ Δ^3: [heurístico base, refutación acoplada, plástico/reset].
pub const N_MODES: usize = 3;
/// Rasgos de auto-estado "Yo-FE 3S" Etapa A-core (10 rasgos VFE+OOT+epistémicos, complementados fuera de muestra por exceso de refutación en soporte `ref_meta_w`).
pub const N_SELF_3S: usize = 10;
/// Rasgos de calibración Platt "Yo-FE 3S" Etapa B.
pub const N_CAL_3S: usize = 8;

// ─── Módulo 1 y 4: Núcleo Dual (W_0 Congelado + Pesos Rápidos Acotados F_t) y Trazas E_t ─

/// Matriz de pesos rápidos efímeros F_t ∈ R^{2×6} estrictamente acotada en norma Frobenius ‖F_t‖_F ≤ ρ.
#[derive(Clone, Debug)]
pub struct FastWeights {
    pub f: [[f64; N_FAST_FEAT]; 2],
    pub rho: f64,
    pub scale: f64,
    pub r_bar: f64,
    pub active: bool,
}

impl FastWeights {
    pub fn new(frames: &[&[f64]], rho: f64) -> Self {
        let mut r_sum = 0.0;
        let mut r_max = 1e-6_f64;
        let mut cnt = 0usize;
        for fr in frames {
            for p in 0..fr.len() / 2 {
                let r = (fr[2 * p].powi(2) + fr[2 * p + 1].powi(2)).sqrt();
                r_sum += r;
                if r > r_max {
                    r_max = r;
                }
                cnt += 1;
            }
        }
        let r_bar = if cnt > 0 { r_sum / cnt as f64 } else { 1.0 };
        let scale = r_max.max(0.25);
        Self {
            f: [[0.0; N_FAST_FEAT]; 2],
            rho,
            scale,
            r_bar,
            active: false,
        }
    }

    pub fn reset(&mut self) {
        self.f = [[0.0; N_FAST_FEAT]; 2];
        self.active = false;
    }

    #[inline]
    pub fn frob_norm(&self) -> f64 {
        let mut s = 0.0;
        for r in 0..2 {
            for c in 0..N_FAST_FEAT {
                s += self.f[r][c] * self.f[r][c];
            }
        }
        s.sqrt()
    }

    #[inline]
    pub fn saturation(&self) -> f64 {
        if self.rho <= 1e-12 {
            0.0
        } else {
            (self.frob_norm() / self.rho).clamp(0.0, 1.0)
        }
    }

    /// Proyección estricta sobre la bola Frobenius Π_{‖·‖_F ≤ ρ}.
    pub fn project_ball(&mut self) {
        let n = self.frob_norm();
        if !n.is_finite() {
            self.reset();
            return;
        }
        if n > self.rho && n > 1e-15 {
            let k = self.rho / n;
            for r in 0..2 {
                for c in 0..N_FAST_FEAT {
                    self.f[r][c] *= k;
                }
            }
        }
    }

    #[inline]
    pub fn psi_feat(&self, q: [f64; 2], u: [f64; 2]) -> [f64; N_FAST_FEAT] {
        let s = self.scale.max(1e-6);
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        [
            1.0,
            (u[0] / s).clamp(-4.0, 4.0),
            (u[1] / s).clamp(-4.0, 4.0),
            (q[0] / s).clamp(-4.0, 4.0),
            (q[1] / s).clamp(-4.0, 4.0),
            ((r - self.r_bar) / s).clamp(-4.0, 4.0),
        ]
    }

    /// Lectura acotada del núcleo dual: u_out = u_base + s · σ(F_t ψ(q, u_base)).
    #[inline]
    pub fn apply_point(&self, q: [f64; 2], u: [f64; 2]) -> [f64; 2] {
        if !self.active || !u[0].is_finite() || !u[1].is_finite() {
            return u;
        }
        let psi = self.psi_feat(q, u);
        let mut d = [0.0; 2];
        for r in 0..2 {
            let mut acc = 0.0;
            for c in 0..N_FAST_FEAT {
                acc += self.f[r][c] * psi[c];
            }
            // Saturación suave σ(·) con linealidad >99% en |acc|≤0.35 y cota estricta ±2.0·scale
            d[r] = self.scale * 2.0 * (acc / 2.0).tanh();
        }
        [u[0] + d[0], u[1] + d[1]]
    }

    pub fn apply_frame(&self, prev: &[f64], base_next: &[f64]) -> Vec<f64> {
        if !self.active {
            return base_next.to_vec();
        }
        let np = prev.len() / 2;
        let mut out = vec![0.0; 2 * np];
        for p in 0..np {
            let qn = self.apply_point(
                [prev[2 * p], prev[2 * p + 1]],
                [base_next[2 * p], base_next[2 * p + 1]],
            );
            out[2 * p] = qn[0];
            out[2 * p + 1] = qn[1];
        }
        out
    }
}

/// Traza de elegibilidad sináptica de 3 factores: E_t = γ E_{t-1} + φ(e_t) ⊗ ψ(x_t).
#[derive(Clone, Debug)]
pub struct EligibilityTrace {
    pub e: [[f64; N_FAST_FEAT]; 2],
    pub gamma: f64,
}

impl EligibilityTrace {
    pub fn new(gamma: f64) -> Self {
        Self {
            e: [[0.0; N_FAST_FEAT]; 2],
            gamma,
        }
    }

    pub fn reset(&mut self) {
        self.e = [[0.0; N_FAST_FEAT]; 2];
    }

    /// Acumula correlaciones pre-post normalizadas en un cuadro completo y ejecuta actualización 3-factores:
    ///   E_t = γ E_{t-1} + (1/N_p) Σ_p φ(err_p) ⊗ ψ_norm(q_p, u_p)
    ///   F_{t+1} = Π_{‖·‖_F ≤ ρ}( λ_t F_t + α_t · δ_t · E_t )
    pub fn update_and_consolidate(
        &mut self,
        fw: &mut FastWeights,
        prev: &[f64],
        base_next: &[f64],
        target: &[f64],
        alpha_t: f64,
        lambda_t: f64,
        delta_t: f64,
    ) {
        let np = prev.len() / 2;
        if np == 0 {
            return;
        }
        let s = fw.scale.max(1e-6);
        let mut grad = [[0.0; N_FAST_FEAT]; 2];
        for p in 0..np {
            let q = [prev[2 * p], prev[2 * p + 1]];
            let u = [base_next[2 * p], base_next[2 * p + 1]];
            let pred = fw.apply_point(q, u);
            let err = [target[2 * p] - pred[0], target[2 * p + 1] - pred[1]];
            let phi = [2.0 * (err[0] / (2.0 * s)).tanh(), 2.0 * (err[1] / (2.0 * s)).tanh()];
            let psi = fw.psi_feat(q, u);
            let psi_n2: f64 = psi.iter().map(|v| v * v).sum::<f64>() + 0.5;
            for r in 0..2 {
                for c in 0..N_FAST_FEAT {
                    grad[r][c] += phi[r] * (psi[c] / psi_n2);
                }
            }
        }
        let inv_np = 1.0 / np as f64;
        for r in 0..2 {
            for c in 0..N_FAST_FEAT {
                self.e[r][c] = self.gamma * self.e[r][c] + grad[r][c] * inv_np;
                fw.f[r][c] = lambda_t * fw.f[r][c] + alpha_t * delta_t * self.e[r][c];
            }
        }
        fw.project_ball();
        fw.active = fw.frob_norm() > 1e-10;
    }
}

// ─── Módulo 2: Monitor de Conflicto (ACC) y Refutador Cruzado Inverso Φ_rev ──

/// Monitor del Cingulado Anterior (ACC): extrae m_t ∈ R^4 en cada paso sin verdad futura.
#[derive(Clone, Debug)]
pub struct AccMonitor {
    window: Vec<Vec<f64>>,
    prev_delta: Option<Vec<f64>>,
    max_w: usize,
    pub hist_surprise: Vec<f64>,
    pub hist_disagree: Vec<f64>,
    pub hist_stagnation: Vec<f64>,
    pub hist_sat: Vec<f64>,
}

impl AccMonitor {
    pub fn new(max_w: usize) -> Self {
        Self {
            window: Vec::with_capacity(max_w + 1),
            prev_delta: None,
            max_w,
            hist_surprise: Vec::new(),
            hist_disagree: Vec::new(),
            hist_stagnation: Vec::new(),
            hist_sat: Vec::new(),
        }
    }

    pub fn seed_frame(&mut self, frame: &[f64]) {
        self.window.clear();
        self.window.push(frame.to_vec());
        self.prev_delta = None;
    }

    /// Calcula el vector ACC m_t ∈ R^4:
    ///   m[0]: sorpresa latente / refutación normalizada
    ///   m[1]: incertidumbre epistémica / desacuerdo entre ramas
    ///   m[2]: índice de estancamiento / detección de bucles (coseno con ventana reciente)
    ///   m[3]: saturación plástica ‖F_t‖_F / ρ
    pub fn observe(
        &mut self,
        state: &[f64],
        surprise_err: f64,
        disagree_err: f64,
        fast_sat: f64,
    ) -> [f64; N_ACC] {
        let mut stag = -1.0_f64;
        // 1) Similitud coseno con ventana reciente {h_{t-2}, ..., h_{t-w}}
        let norm_cur = state.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        if self.window.len() >= 2 {
            for past in self.window.iter().rev().skip(1) {
                let norm_p = past.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                let dot: f64 = state.iter().zip(past).map(|(a, b)| a * b).sum();
                let cos = (dot / (norm_cur * norm_p)).clamp(-1.0, 1.0);
                if cos > stag {
                    stag = cos;
                }
            }
        }
        // 2) Reversión de dirección Δh_t · Δh_{t-1} < 0 (oscilación 2-ciclo)
        if let Some(last) = self.window.last() {
            let delta: Vec<f64> = state.iter().zip(last).map(|(a, b)| a - b).collect();
            let dn = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
            if let Some(pd) = &self.prev_delta {
                let pdn = pd.iter().map(|v| v * v).sum::<f64>().sqrt();
                if dn > 1e-12 && pdn > 1e-12 {
                    let rev_cos = -(delta.iter().zip(pd).map(|(a, b)| a * b).sum::<f64>() / (dn * pdn))
                        .clamp(-1.0, 1.0);
                    if rev_cos > stag {
                        stag = rev_cos;
                    }
                }
            }
            self.prev_delta = Some(delta);
        }
        if stag < -0.99 {
            stag = 0.0;
        }
        self.window.push(state.to_vec());
        if self.window.len() > self.max_w {
            self.window.remove(0);
        }

        let m0 = ((log_rel(surprise_err) + 5.0) / 4.0).clamp(-1.5, 2.0);
        let m1 = ((log_rel(disagree_err) + 5.0) / 4.0).clamp(-1.5, 2.0);
        let m2 = stag.clamp(-1.0, 1.0);
        let m3 = fast_sat.clamp(0.0, 1.0);

        self.hist_surprise.push(surprise_err);
        self.hist_disagree.push(disagree_err);
        self.hist_stagnation.push(m2);
        self.hist_sat.push(m3);
        [m0, m1, m2, m3]
    }
}

/// Rama independiente en tiempo inverso Φ_rev: ajustada EXCLUSIVAMENTE en contexto invertido
/// [x_K, x_{K-1}, ..., x_0] (0 fuga futura) para refutación cruzada en rollout abierto.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct BackwardRefuter {
    pub(super) hyp: FittedHyp,
    pub ctx_cyc_err: f64,
    pub rev_oot1_err: f64,
    pub reliability: f64,
    pub valid: bool,
}

const REV_KINDS: [HypKind; 9] = [
    HypKind::LieAffine,
    HypKind::GpCartRbf,
    HypKind::GpCoMoving,
    HypKind::GpSympFwd,
    HypKind::GpSympRev,
    HypKind::GpCoMovingSympFwd,
    HypKind::GpCoMovingSympRev,
    HypKind::GpPolar,
    HypKind::GpEquivCart,
];

impl BackwardRefuter {
    pub fn fit(ctx_frames: &[&[f64]]) -> Self {
        let k_obs = ctx_frames.len().saturating_sub(1);
        let rev_frames: Vec<&[f64]> = ctx_frames.iter().rev().copied().collect();
        let np = ctx_frames[0].len() / 2;
        let n_pts = (k_obs.max(1) * np) as f64;

        let mut best_hyp = fit_hypothesis_ex(HypKind::LieAffine, &rev_frames, 2);
        let mut best_score = f64::INFINITY;
        let mut best_cyc = 10.0;

        for &kind in &REV_KINDS {
            let h = fit_hypothesis_ex(kind, &rev_frames, 2);
            if !h.valid {
                continue;
            }
            // Error de ciclo exacto en contexto observado: Φ_rev(x_t) vs x_{t-1}
            let mut cycs = Vec::with_capacity(k_obs);
            for t in 1..=k_obs {
                let (back, _) = h.step_frame(ctx_frames[t]);
                cycs.push(clamp_err(&back, ctx_frames[t - 1]));
            }
            let cyc_m = mean(&cycs);
            // Replay inverso cerrado desde x_K hasta x_0
            let mut z = ctx_frames[k_obs].to_vec();
            let mut reps = Vec::with_capacity(k_obs);
            for t in (0..k_obs).rev() {
                let (zn, _) = h.step_frame(&z);
                reps.push(log_rel(clamp_err(&zn, ctx_frames[t])));
                z = zn;
            }
            let vfe_pt = (h.vfe / n_pts).clamp(-40.0, 40.0);
            let loo_pt = (h.loo_fe / n_pts).clamp(-40.0, 40.0);
            let rep_m = mean(&reps);
            let sc = vfe_pt + 0.35 * loo_pt + 0.45 * rep_m;
            if sc.is_finite() && sc < best_score {
                best_score = sc;
                best_hyp = h;
                best_cyc = cyc_m;
            }
            // Atajo afín exacto: si la transformación inversa en GL(2)⋉R² ya cierra a <1e-7, es exacta
            if kind == HypKind::LieAffine && cyc_m < 1e-7 && rep_m < -14.0 {
                break;
            }
        }

        if !best_hyp.valid {
            return Self {
                hyp: best_hyp,
                ctx_cyc_err: 10.0,
                rev_oot1_err: 10.0,
                reliability: 0.0,
                valid: false,
            };
        }

        // Validación precuencial inversa estrictamente en contexto: ajustar en [x_K..x_1] -> predecir x_0
        let rev_oot1_err = if rev_frames.len() >= 4 {
            let h_sub = fit_hypothesis_ex(best_hyp.kind, &rev_frames[..rev_frames.len() - 1], 2);
            if h_sub.valid {
                let (p0, _) = h_sub.step_frame(ctx_frames[1]);
                clamp_err(&p0, ctx_frames[0])
            } else {
                10.0
            }
        } else {
            best_cyc
        };

        let rel = (-(best_cyc / 0.045).powi(2)).exp() * (-(rev_oot1_err / 0.12).powi(2)).exp();
        Self {
            hyp: best_hyp,
            ctx_cyc_err: best_cyc,
            rev_oot1_err,
            reliability: if rel.is_finite() { rel.clamp(0.0, 1.0) } else { 0.0 },
            valid: true,
        }
    }

    /// Evalúa la refutación cruzada de una propuesta x̂_t dado el estado previo x_{t-1}:
    /// devuelve (err_ciclo, varianza_epistémica_inversa, x_back).
    pub fn refute_step(&self, x_prev: &[f64], x_cand: &[f64]) -> (f64, f64, Vec<f64>) {
        if !self.valid || !fin(x_cand) || !fin(x_prev) {
            return (10.0, 1e6, x_prev.to_vec());
        }
        let (x_back, epi_rev) = self.hyp.step_frame(x_cand);
        let err = clamp_err(&x_back, x_prev);
        (err, epi_rev, x_back)
    }

    /// Puntúa la consistencia de ciclo de toda una trayectoria rollout [x̂_{K+1}..x̂_16] de una hipótesis.
    pub fn score_trajectory(&self, x_k: &[f64], roll_preds: &[Vec<f64>]) -> f64 {
        if !self.valid || self.reliability < 1e-4 {
            return 0.0;
        }
        let mut prev = x_k;
        let mut num = 0.0;
        let mut den = 0.0;
        let s2 = self.hyp.s2_scale.max(1e-8);
        for cand in roll_preds {
            if !fin(cand) {
                num += log_rel(10.0);
                den += 1.0;
                break;
            }
            let (err, epi, _) = self.refute_step(prev, cand);
            let w = 1.0 / (1.0 + (epi / s2).clamp(0.0, 20.0));
            num += w * log_rel(err.max(1e-7));
            den += w;
            prev = cand;
        }
        if den > 1e-12 {
            num / den
        } else {
            0.0
        }
    }
}

// ─── Módulo 3: Controlador Neuromodulador Recurrente (MetaNet GRU) ───────────

/// Salida neuromoduladora emitida en tiempo real en cada paso t.
#[derive(Clone, Copy, Debug)]
pub struct NeuromodSignal {
    /// Tasa de escritura plástica (Noradrenalina) α_t ∈ [0, 1]
    pub alpha: f64,
    /// Factor de retención / olvido-reset λ_t ∈ (0, 1)
    pub lambda: f64,
    /// Ganancia de precisión / temperatura inversa (Acetilcolina) β_t ∈ [β_min, β_max]
    pub beta: f64,
    /// Compuerta dlPFC de modos cognitivos g_t ∈ Δ^3: [base, refutación, plástico/reset]
    pub g_mode: [f64; N_MODES],
}

/// Red MetaNet recurrente (celda GRU compacta H=4 + cabezas con salto directo desde m_t).
#[derive(Clone, Debug)]
pub struct NeuromodNet {
    pub wz: [[f64; N_ACC]; N_GRU_H],
    pub uz: [[f64; N_GRU_H]; N_GRU_H],
    pub bz: [f64; N_GRU_H],
    pub wr: [[f64; N_ACC]; N_GRU_H],
    pub ur: [[f64; N_GRU_H]; N_GRU_H],
    pub br: [f64; N_GRU_H],
    pub wh: [[f64; N_ACC]; N_GRU_H],
    pub uh: [[f64; N_GRU_H]; N_GRU_H],
    pub bh: [f64; N_GRU_H],
    // Cabezas: 0=alpha, 1=lambda, 2=beta, 3..6=g_mode[0..3]
    pub wo: [[f64; N_GRU_H]; 6],
    pub vo: [[f64; N_ACC]; 6],
    pub bo: [f64; 6],
    pub beta_min: f64,
    pub beta_max: f64,
}

#[derive(Clone, Debug)]
struct GruCache {
    m: [f64; N_ACC],
    s_prev: [f64; N_GRU_H],
    z: [f64; N_GRU_H],
    r: [f64; N_GRU_H],
    h_tilde: [f64; N_GRU_H],
    s: [f64; N_GRU_H],
    alpha: f64,
    lambda: f64,
    beta_u: f64,
    g_mode: [f64; N_MODES],
}

impl NeuromodNet {
    pub fn init(seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_3S_NM);
        let mut mat4 = |sc: f64| -> [[f64; 4]; 4] {
            let mut a = [[0.0; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    a[i][j] = rng.gen_range(-sc..sc);
                }
            }
            a
        };
        let wz = mat4(0.25);
        let uz = mat4(0.20);
        let wr = mat4(0.25);
        let ur = mat4(0.20);
        let wh = mat4(0.25);
        let uh = mat4(0.20);
        let mut wo = [[0.0; N_GRU_H]; 6];
        let mut vo = [[0.0; N_ACC]; 6];
        for i in 0..6 {
            for j in 0..N_GRU_H {
                wo[i][j] = rng.gen_range(-0.15..0.15);
            }
            for j in 0..N_ACC {
                vo[i][j] = rng.gen_range(-0.15..0.15);
            }
        }
        // Inicialización bio-plausible suave antes de BPTT outer-loop
        vo[0][0] = 1.0;   // sorpresa ↑ -> α ↑
        vo[1][0] = -1.6;  // sorpresa ↑ -> λ ↓ (reset)
        vo[2][0] = -0.8;  // sorpresa ↑ -> β ↓ (explorar)
        Self {
            wz,
            uz,
            bz: [0.0; N_GRU_H],
            wr,
            ur,
            br: [0.0; N_GRU_H],
            wh,
            uh,
            bh: [0.0; N_GRU_H],
            wo,
            vo,
            bo: [-1.2, 1.5, 0.2, 0.2, 0.0, -0.4],
            beta_min: 1.5,
            beta_max: 12.0,
        }
    }

    fn step_cache(&self, m: [f64; N_ACC], s_prev: [f64; N_GRU_H]) -> GruCache {
        let mut z = [0.0; N_GRU_H];
        let mut r = [0.0; N_GRU_H];
        let mut h_tilde = [0.0; N_GRU_H];
        let mut s = [0.0; N_GRU_H];
        for i in 0..N_GRU_H {
            let mut az = self.bz[i];
            let mut ar = self.br[i];
            for j in 0..N_ACC {
                az += self.wz[i][j] * m[j];
                ar += self.wr[i][j] * m[j];
            }
            for j in 0..N_GRU_H {
                az += self.uz[i][j] * s_prev[j];
                ar += self.ur[i][j] * s_prev[j];
            }
            z[i] = sig(az.clamp(-20.0, 20.0));
            r[i] = sig(ar.clamp(-20.0, 20.0));
        }
        for i in 0..N_GRU_H {
            let mut ah = self.bh[i];
            for j in 0..N_ACC {
                ah += self.wh[i][j] * m[j];
            }
            for j in 0..N_GRU_H {
                ah += self.uh[i][j] * (r[j] * s_prev[j]);
            }
            h_tilde[i] = ah.clamp(-20.0, 20.0).tanh();
            s[i] = (1.0 - z[i]) * s_prev[i] + z[i] * h_tilde[i];
        }
        let mut out_raw = [0.0; 6];
        for o in 0..6 {
            let mut a = self.bo[o];
            for j in 0..N_GRU_H {
                a += self.wo[o][j] * s[j];
            }
            for j in 0..N_ACC {
                a += self.vo[o][j] * m[j];
            }
            out_raw[o] = a.clamp(-20.0, 20.0);
        }
        let alpha = sig(out_raw[0]);
        let lambda = sig(out_raw[1]).clamp(0.02, 0.995);
        let beta_u = sig(out_raw[2]);
        let mx = out_raw[3].max(out_raw[4]).max(out_raw[5]);
        let e0 = (out_raw[3] - mx).exp();
        let e1 = (out_raw[4] - mx).exp();
        let e2 = (out_raw[5] - mx).exp();
        let es = (e0 + e1 + e2).max(1e-30);
        let g_mode = [e0 / es, e1 / es, e2 / es];
        GruCache {
            m,
            s_prev,
            z,
            r,
            h_tilde,
            s,
            alpha,
            lambda,
            beta_u,
            g_mode,
        }
    }

    pub fn step(&self, m: [f64; N_ACC], s_prev: [f64; N_GRU_H]) -> (NeuromodSignal, [f64; N_GRU_H]) {
        let c = self.step_cache(m, s_prev);
        let beta = self.beta_min + (self.beta_max - self.beta_min) * c.beta_u;
        (
            NeuromodSignal {
                alpha: c.alpha,
                lambda: c.lambda,
                beta,
                g_mode: c.g_mode,
            },
            c.s,
        )
    }

    /// Entrenamiento outer-loop exacto mediante BPTT ponderado sobre secuencias meta-supervisadas
    /// generadas exclusivamente a partir de episodios auditados (conocidas 0..7 + sintéticas).
    pub fn train_bptt(&mut self, seqs: &[Vec<([f64; N_ACC], [f64; 6], f64)>], epochs: usize, lr: f64, l2: f64) {
        if seqs.is_empty() {
            return;
        }
        let inv_n = 1.0 / seqs.len() as f64;
        for _ep in 0..epochs {
            let mut gwz = [[0.0; N_ACC]; N_GRU_H];
            let mut guz = [[0.0; N_GRU_H]; N_GRU_H];
            let mut gbz = [0.0; N_GRU_H];
            let mut gwr = [[0.0; N_ACC]; N_GRU_H];
            let mut gur = [[0.0; N_GRU_H]; N_GRU_H];
            let mut gbr = [0.0; N_GRU_H];
            let mut gwh = [[0.0; N_ACC]; N_GRU_H];
            let mut guh = [[0.0; N_GRU_H]; N_GRU_H];
            let mut gbh = [0.0; N_GRU_H];
            let mut gwo = [[0.0; N_GRU_H]; 6];
            let mut gvo = [[0.0; N_ACC]; 6];
            let mut gbo = [0.0; 6];

            for seq in seqs {
                if seq.is_empty() {
                    continue;
                }
                let mut caches = Vec::with_capacity(seq.len());
                let mut s = [0.0; N_GRU_H];
                let mut w_tot = 0.0;
                for &(m, _, wt) in seq {
                    let c = self.step_cache(m, s);
                    s = c.s;
                    caches.push(c);
                    w_tot += wt;
                }
                let mut ds_next = [0.0; N_GRU_H];
                let inv_w = inv_n / w_tot.max(1e-8);
                for t in (0..seq.len()).rev() {
                    let c = &caches[t];
                    let tgt = seq[t].1;
                    let wt = seq[t].2 * inv_w;
                    let d_raw = [
                        (c.alpha - tgt[0]) * wt,
                        (c.lambda - tgt[1]) * wt,
                        (c.beta_u - tgt[2]) * wt,
                        (c.g_mode[0] - tgt[3]) * wt,
                        (c.g_mode[1] - tgt[4]) * wt,
                        (c.g_mode[2] - tgt[5]) * wt,
                    ];
                    let mut ds = ds_next;
                    for o in 0..6 {
                        gbo[o] += d_raw[o];
                        for j in 0..N_GRU_H {
                            gwo[o][j] += d_raw[o] * c.s[j];
                            ds[j] += d_raw[o] * self.wo[o][j];
                        }
                        for j in 0..N_ACC {
                            gvo[o][j] += d_raw[o] * c.m[j];
                        }
                    }
                    let mut ds_prev = [0.0; N_GRU_H];
                    let mut d_r_sp = [0.0; N_GRU_H];
                    for i in 0..N_GRU_H {
                        let dh_tilde = ds[i] * c.z[i] * (1.0 - c.h_tilde[i] * c.h_tilde[i]);
                        let dz = ds[i] * (c.h_tilde[i] - c.s_prev[i]) * c.z[i] * (1.0 - c.z[i]);
                        gbh[i] += dh_tilde;
                        gbz[i] += dz;
                        ds_prev[i] += ds[i] * (1.0 - c.z[i]);
                        for j in 0..N_ACC {
                            gwh[i][j] += dh_tilde * c.m[j];
                            gwz[i][j] += dz * c.m[j];
                        }
                        for j in 0..N_GRU_H {
                            guh[i][j] += dh_tilde * (c.r[j] * c.s_prev[j]);
                            d_r_sp[j] += dh_tilde * self.uh[i][j];
                            guz[i][j] += dz * c.s_prev[j];
                            ds_prev[j] += dz * self.uz[i][j];
                        }
                    }
                    for j in 0..N_GRU_H {
                        ds_prev[j] += d_r_sp[j] * c.r[j];
                        let dr = d_r_sp[j] * c.s_prev[j] * c.r[j] * (1.0 - c.r[j]);
                        gbr[j] += dr;
                        for k in 0..N_ACC {
                            gwr[j][k] += dr * c.m[k];
                        }
                        for k in 0..N_GRU_H {
                            gur[j][k] += dr * c.s_prev[k];
                            ds_prev[k] += dr * self.ur[j][k];
                        }
                    }
                    ds_next = ds_prev;
                }
            }

            let step_w = |w: &mut f64, g: f64| {
                let gn = (g + l2 * *w).clamp(-2.5, 2.5);
                *w -= lr * gn;
            };
            for i in 0..N_GRU_H {
                step_w(&mut self.bz[i], gbz[i]);
                step_w(&mut self.br[i], gbr[i]);
                step_w(&mut self.bh[i], gbh[i]);
                for j in 0..N_ACC {
                    step_w(&mut self.wz[i][j], gwz[i][j]);
                    step_w(&mut self.wr[i][j], gwr[i][j]);
                    step_w(&mut self.wh[i][j], gwh[i][j]);
                }
                for j in 0..N_GRU_H {
                    step_w(&mut self.uz[i][j], guz[i][j]);
                    step_w(&mut self.ur[i][j], gur[i][j]);
                    step_w(&mut self.uh[i][j], guh[i][j]);
                }
            }
            for o in 0..6 {
                step_w(&mut self.bo[o], gbo[o]);
                for j in 0..N_GRU_H {
                    step_w(&mut self.wo[o][j], gwo[o][j]);
                }
                for j in 0..N_ACC {
                    step_w(&mut self.vo[o][j], gvo[o][j]);
                }
            }
        }
    }
}

// ─── Módulo 5: Buffer Episódico y Consolidación Offline con EWC ──────────────

/// Entrada comprimida en el Buffer Episódico (F_t se descarta; solo se guarda experiencia verificada).
#[derive(Clone, Debug)]
pub struct EpisodicEntry {
    pub inv: [f64; N_INV],
    pub hyp_target: [f64; N_HYP],
    pub weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct EpisodicBuffer {
    pub entries: Vec<EpisodicEntry>,
    pub capacity: usize,
}

impl EpisodicBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, entry: EpisodicEntry) {
        if self.entries.len() >= self.capacity && self.capacity > 0 {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }
}

/// Estado de Elastic Weight Consolidation (EWC) para consolidación offline del núcleo lento W_0^meta:
///   L_cons(W_0) = L_tarea(W_0) + (μ / 2) Σ_i Ω_i (W_{0,i} - W_{0,i}^{prev})^2
#[derive(Clone, Debug)]
pub struct EwcPriorState {
    pub w_prev: [[f64; N_INV]; N_HYP],
    pub omega: [[f64; N_INV]; N_HYP],
    pub initialized: bool,
}

impl Default for EwcPriorState {
    fn default() -> Self {
        Self {
            w_prev: [[0.0; N_INV]; N_HYP],
            omega: [[0.0; N_INV]; N_HYP],
            initialized: false,
        }
    }
}

fn prior_ce_loss(pw: &[[f64; N_INV]; N_HYP], entries: &[EpisodicEntry]) -> f64 {
    if entries.is_empty() {
        return 0.0;
    }
    let mut loss = 0.0;
    let mut wsum = 0.0;
    for e in entries {
        let mut logits = [0.0; N_HYP];
        let mut mx = f64::NEG_INFINITY;
        for h in 0..N_HYP {
            let v: f64 = pw[h].iter().zip(&e.inv).map(|(a, b)| a * b).sum();
            logits[h] = v;
            if v > mx {
                mx = v;
            }
        }
        let es: f64 = logits.iter().map(|&z| (z - mx).exp()).sum::<f64>().max(1e-30);
        let lse = mx + es.ln();
        let mut ce = 0.0;
        for h in 0..N_HYP {
            if e.hyp_target[h] > 0.0 {
                ce -= e.hyp_target[h] * (logits[h] - lse);
            }
        }
        loss += e.weight * ce;
        wsum += e.weight;
    }
    loss / wsum.max(1e-12)
}

fn prior_top1_acc(pw: &[[f64; N_INV]; N_HYP], entries: &[EpisodicEntry]) -> f64 {
    if entries.is_empty() {
        return 0.0;
    }
    let mut ok = 0usize;
    for e in entries {
        let pred_h = (0..N_HYP)
            .max_by(|&a, &b| {
                let va: f64 = pw[a].iter().zip(&e.inv).map(|(x, y)| x * y).sum();
                let vb: f64 = pw[b].iter().zip(&e.inv).map(|(x, y)| x * y).sum();
                va.total_cmp(&vb)
            })
            .unwrap_or(0);
        let true_h = (0..N_HYP)
            .max_by(|&a, &b| e.hyp_target[a].total_cmp(&e.hyp_target[b]))
            .unwrap_or(0);
        if pred_h == true_h {
            ok += 1;
        }
    }
    ok as f64 / entries.len() as f64
}

/// Calcula la matriz de información de Fisher diagonal empírica Ω_i = (1/N) Σ_n (∂L_n / ∂W_i)^2.
pub fn compute_fisher_diag(pw: &[[f64; N_INV]; N_HYP], entries: &[EpisodicEntry]) -> [[f64; N_INV]; N_HYP] {
    let mut fisher = [[0.0; N_INV]; N_HYP];
    if entries.is_empty() {
        return fisher;
    }
    let n = entries.len() as f64;
    for e in entries {
        let mut logits = [0.0; N_HYP];
        let mut mx = f64::NEG_INFINITY;
        for h in 0..N_HYP {
            let v: f64 = pw[h].iter().zip(&e.inv).map(|(a, b)| a * b).sum();
            logits[h] = v;
            if v > mx {
                mx = v;
            }
        }
        let ex: [f64; N_HYP] = std::array::from_fn(|h| (logits[h] - mx).exp());
        let es: f64 = ex.iter().sum::<f64>().max(1e-30);
        for h in 0..N_HYP {
            let diff = ex[h] / es - e.hyp_target[h];
            for j in 0..N_INV {
                let g = diff * e.inv[j];
                fisher[h][j] += (g * g) * e.weight / n;
            }
        }
    }
    fisher
}

/// Consolidación offline del hiper-prior lento W_0^meta sobre el Buffer Episódico con penalización EWC.
pub fn consolidate_prior_ewc(
    pw: &mut [[f64; N_INV]; N_HYP],
    entries: &[EpisodicEntry],
    ewc: &EwcPriorState,
    mu_ewc: f64,
    epochs: usize,
    lr: f64,
    l2: f64,
) {
    if entries.is_empty() {
        return;
    }
    let wsum: f64 = entries.iter().map(|e| e.weight).sum::<f64>().max(1e-12);
    for _ in 0..epochs {
        let mut grad = [[0.0; N_INV]; N_HYP];
        for e in entries {
            let mut logits = [0.0; N_HYP];
            let mut mx = f64::NEG_INFINITY;
            for h in 0..N_HYP {
                let v: f64 = pw[h].iter().zip(&e.inv).map(|(a, b)| a * b).sum();
                logits[h] = v;
                if v > mx {
                    mx = v;
                }
            }
            let ex: [f64; N_HYP] = std::array::from_fn(|h| (logits[h] - mx).exp());
            let es: f64 = ex.iter().sum::<f64>().max(1e-30);
            let sw = e.weight / wsum;
            for h in 0..N_HYP {
                let diff = (ex[h] / es - e.hyp_target[h]) * sw;
                for j in 0..N_INV {
                    grad[h][j] += diff * e.inv[j];
                }
            }
        }
        for h in 0..N_HYP {
            for j in 0..N_INV {
                let ewc_g = if ewc.initialized && mu_ewc > 0.0 {
                    mu_ewc * ewc.omega[h][j] * (pw[h][j] - ewc.w_prev[h][j])
                } else {
                    0.0
                };
                let step = (grad[h][j] + l2 * pw[h][j] + ewc_g).clamp(-3.0, 3.0);
                pw[h][j] -= lr * step;
            }
        }
    }
}

// ─── Configuración, Controlador y Reporte de Episodio METAFE-3S ──────────────

#[derive(Clone, Debug)]
pub struct MetaFe3SCfg {
    pub name: &'static str,
    pub use_fast_weights: bool,
    pub use_refuter: bool,
    pub use_neuromod: bool,
    pub use_ewc: bool,
    pub is_aura_v3_genesis: bool,
}

/// Subrutina 1 (ADN Generativo): Enrutamiento por Gradientes Químicos Morfogenéticos 3D [SHH, Wnt, BMP]
/// sin direcciones explícitas. Cada minicolumna cortical expresa un perfil genético de receptores
/// [R_EphA, R_EphB, R_Robo] ∈ [0,1]^3 y su cono de crecimiento evalúa la afinidad quimiotáctica local.
pub fn morphogen_chemotaxis_affinity(kind: HypKind, inv: &[f64; N_INV]) -> f64 {
    // Gradientes químicos locales en el espacio morfogenético 3D del episodio:
    // - SHH (ventral): pureza afín/Lie (inv[1] < -0.80)
    // - Wnt (antero-posterior): curvatura radial/polar (varianza radial inv[5] > -0.62)
    // - BMP (dorso-ventral): no-linealidad conservativa/simpléctica (inv[1] alto, |det|-1 ≈ 0 en inv[3], inv[5] bajo)
    let c_shh = sig(-6.0 * (inv[1] + 0.80));
    let c_wnt = sig(5.0 * (inv[5] + 0.62));
    let c_bmp = (1.0 - c_shh) * (1.0 - c_wnt) * sig(-18.0 * (inv[3].abs() - 0.08));
    // Sub-gradiente de rotación media om/π (inv[4]): separa cizalla simpléctica alineada vs co-móvil rotada
    let rot_gate = sig(25.0 * (inv[4].abs() - 0.035));

    match kind {
        HypKind::LieAffine => 1.15 * c_shh,
        HypKind::GpPolar | HypKind::GpEquivCart => 0.90 * c_wnt,
        HypKind::GpSympFwd | HypKind::GpSympRev => 1.10 * c_bmp * (1.0 - rot_gate) + 0.45 * c_bmp * rot_gate,
        HypKind::GpCoMoving | HypKind::GpCoMovingSympFwd | HypKind::GpCoMovingSympRev => {
            1.05 * c_bmp * rot_gate + 0.60 * c_shh
        }
        HypKind::FeArd | HypKind::GpCartM52 | HypKind::GpCartRbf => 0.35 * c_bmp,
    }
}

pub fn metafe3s_arms() -> Vec<MetaFe3SCfg> {
    vec![
        MetaFe3SCfg {
            name: "METAFE-3S",
            use_fast_weights: true,
            use_refuter: true,
            use_neuromod: true,
            use_ewc: true,
            is_aura_v3_genesis: false,
        },
        MetaFe3SCfg {
            name: "METAFE-3S-NoFast",
            use_fast_weights: false,
            use_refuter: true,
            use_neuromod: true,
            use_ewc: true,
            is_aura_v3_genesis: false,
        },
        MetaFe3SCfg {
            name: "METAFE-3S-NoRefute",
            use_fast_weights: true,
            use_refuter: false,
            use_neuromod: true,
            use_ewc: true,
            is_aura_v3_genesis: false,
        },
        MetaFe3SCfg {
            name: "METAFE-3S-NoNM",
            use_fast_weights: true,
            use_refuter: true,
            use_neuromod: false,
            use_ewc: true,
            is_aura_v3_genesis: false,
        },
        MetaFe3SCfg {
            name: "AURA-v3-Genesis",
            use_fast_weights: true,
            use_refuter: true,
            use_neuromod: true,
            use_ewc: true,
            is_aura_v3_genesis: true,
        },
    ]
}

fn all_hyps() -> Vec<HypKind> {
    vec![
        HypKind::LieAffine,
        HypKind::FeArd,
        HypKind::GpCartM52,
        HypKind::GpCartRbf,
        HypKind::GpCoMoving,
        HypKind::GpSympFwd,
        HypKind::GpSympRev,
        HypKind::GpCoMovingSympFwd,
        HypKind::GpCoMovingSympRev,
        HypKind::GpPolar,
        HypKind::GpEquivCart,
    ]
}

/// Controlador entrenado de METAFE-3S (núcleo lento W_0^meta congelado durante inferencia).
#[derive(Clone, Debug)]
pub struct Meta3SController {
    /// Pesos de Energía Libre [vfe, loo, oot1, oot2, replay, stab]
    pub fe_w: [f64; 6],
    /// Peso aprendido de refutación cruzada inversa Φ_rev en la puntuación de hipótesis
    pub w_refute: f64,
    /// Ganancia de proyección de variedad de ciclo durante rollout acoplado
    pub cycle_gain: f64,
    /// Temperatura base τ_0 (modulada dinámicamente por β_t de MetaNet)
    pub base_tau: f64,
    /// Ganancia epistémica γ_epi
    pub gamma_epi: f64,
    /// Hiper-prior lento consolidado con EWC
    pub prior_w: [[f64; N_INV]; N_HYP],
    pub prior_scale: f64,
    pub ewc_state: EwcPriorState,
    /// Controlador neuromodulador recurrente GRU MetaNet(m_t, s_{t-1})
    pub nm: NeuromodNet,
    /// Auto-calibrador "Yo-FE 3S" Etapa A-core (predictor de log-error h16), corrección en soporte `ref_meta_w` y Etapa B (Platt balanceado)
    pub err_w: [f64; N_SELF_3S],
    pub ref_meta_w: f64,
    pub cal_w: [f64; N_CAL_3S],
    /// Prior autónomo de arranque en frío 1-paso (heredado de METAFE v2, solo `ds.train`)
    pub base_cold: MetaController,
    /// Tamaño del buffer episódico consolidado
    pub buffer_len: usize,
}

impl Default for Meta3SController {
    fn default() -> Self {
        Self {
            fe_w: [1.0, 0.35, 0.50, 0.50, 0.25, 1.0],
            w_refute: 0.40,
            cycle_gain: 0.35,
            base_tau: 6.0,
            gamma_epi: 0.25,
            prior_w: [[0.0; N_INV]; N_HYP],
            prior_scale: 0.25,
            ewc_state: EwcPriorState::default(),
            nm: NeuromodNet::init(0x35),
            err_w: [0.0; N_SELF_3S],
            ref_meta_w: 0.0,
            cal_w: [0.0; N_CAL_3S],
            base_cold: MetaController::default(),
            buffer_len: 0,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Episode3SReport {
    pub preds: Vec<Vec<f64>>,
    pub fe_null: f64,
    pub fe_post: f64,
    pub fe_reduction: f64,
    pub post_entropy: f64,
    pub best_hyp: usize,
    pub p_ok_hat: f64,
    pub err_log_hat: f64,
    pub mean_epi: f64,
    pub ens_disagree: f64,
    pub mean_refute: f64,
    pub max_refute: f64,
    pub in_sup_ref_excess: f64,
    pub rev_reliability: f64,
    pub fast_norm: f64,
    pub fast_used: bool,
    pub cycle_accepted: usize,
    pub mean_alpha: f64,
    pub min_lambda: f64,
    pub mean_beta: f64,
    pub max_stagnation: f64,
    pub self_feats: [f64; N_SELF_3S],
    pub cal_feats: [f64; N_CAL_3S],
    pub nan: bool,
}

// ─── Motor de Inferencia en Tiempo Real METAFE-3S ────────────────────────────

/// Evalúa el paso base del núcleo W_0 mezclado con pesos g_m sobre un cuadro.
fn blend_step(diags: &[HypDiag], weights: &[f64], frame: &[f64]) -> (Vec<f64>, f64, f64) {
    let dim = frame.len();
    let mut fused = vec![0.0; dim];
    let mut wsum = 0.0;
    let mut epi_acc = 0.0;
    let mut cands: Vec<(Vec<f64>, f64)> = Vec::new();
    for (d, &w) in diags.iter().zip(weights) {
        if w < 1e-4 || !d.hyp.valid {
            continue;
        }
        let (pn, epi) = d.hyp.step_frame(frame);
        if fin(&pn) {
            for j in 0..dim {
                fused[j] += w * pn[j];
            }
            wsum += w;
            epi_acc += w * epi;
            cands.push((pn, w));
        }
    }
    if wsum > 1e-12 {
        for v in &mut fused {
            *v /= wsum;
        }
        epi_acc /= wsum;
        let dis: f64 = cands
            .iter()
            .map(|(pn, w)| (*w / wsum) * mt::rel_err(pn, &fused).min(10.0))
            .sum();
        (fused, epi_acc, dis)
    } else {
        let best_i = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let (pn, epi) = diags[best_i].hyp.step_frame(frame);
        (pn, epi, 0.0)
    }
}

/// Adapta los pesos rápidos efímeros F_t ∈ Π_ρ en contexto mediante trazas de elegibilidad
/// de 3 factores gobernadas por MetaNet, con compuerta precuencial OOT estricta (cero daño en afines).
fn adapt_fast_weights_in_context(
    cfg: &MetaFe3SCfg,
    ctrl: &Meta3SController,
    ctx_frames: &[&[f64]],
    diags: &[HypDiag],
    weights: &[f64],
    k_obs: usize,
) -> (FastWeights, bool) {
    let rho_canon = 0.22;
    let mut fw = FastWeights::new(ctx_frames, rho_canon);
    if !cfg.use_fast_weights || k_obs < 3 {
        return (fw, false);
    }

    // Comprobar sorpresa base de replay en contexto: si es casi exacta (< 5e-4), F_t ≡ 0 (cero daño)
    let mut base_replay = Vec::with_capacity(k_obs);
    let mut z = ctx_frames[0].to_vec();
    for t in 1..=k_obs {
        let (zn, _, _) = blend_step(diags, weights, &z);
        base_replay.push(clamp_err(&zn, ctx_frames[t]));
        z = zn;
    }
    let replay_m = mean(&base_replay);
    if !replay_m.is_finite() || replay_m < 5e-4 || replay_m > 2.5 {
        return (fw, false);
    }

    // 1. Verificación precuencial OOT: adaptar F_oot en 0..=(k_obs-1) y validar en cuadro k_obs
    let mut fw_oot = FastWeights::new(&ctx_frames[..k_obs], rho_canon);
    let mut tr_oot = EligibilityTrace::new(0.50);
    let mut s_nm = [0.0; N_GRU_H];
    let mut acc = AccMonitor::new(4);
    acc.seed_frame(ctx_frames[0]);

    let mut base_1step = Vec::with_capacity(k_obs);
    for t in 0..k_obs {
        let (bn, _, dis) = blend_step(diags, weights, ctx_frames[t]);
        base_1step.push((bn, dis));
    }
    let base_oot_err = clamp_err(&base_1step[k_obs - 1].0, ctx_frames[k_obs]);
    if base_oot_err < 3e-4 {
        return (fw, false);
    }

    for _iter in 0..16 {
        tr_oot.reset();
        let mut zc = ctx_frames[0].to_vec();
        for t in 0..(k_obs - 1) {
            let (bn_c, _, dis_c) = blend_step(diags, weights, &zc);
            let pred_c = fw_oot.apply_frame(&zc, &bn_c);
            let err_c = clamp_err(&pred_c, ctx_frames[t + 1]);
            let m_t = acc.observe(&pred_c, err_c, dis_c, fw_oot.saturation());
            let sig_nm = if cfg.use_neuromod {
                let (sn, ns) = ctrl.nm.step(m_t, s_nm);
                s_nm = ns;
                sn
            } else {
                NeuromodSignal {
                    alpha: 0.25,
                    lambda: 0.90,
                    beta: ctrl.base_tau,
                    g_mode: [0.34, 0.33, 0.33],
                }
            };
            let alpha_eff = (0.25 + 0.55 * sig_nm.alpha).clamp(0.10, 0.75);
            let lam_eff = sig_nm.lambda.clamp(0.95, 0.998);
            tr_oot.update_and_consolidate(
                &mut fw_oot,
                &zc,
                &bn_c,
                ctx_frames[t + 1],
                alpha_eff,
                lam_eff,
                1.0,
            );
            zc = fw_oot.apply_frame(&zc, &bn_c);
            if !fin(&zc) {
                return (fw, false);
            }
        }
    }

    let pred_oot_1 = fw_oot.apply_frame(ctx_frames[k_obs - 1], &base_1step[k_obs - 1].0);
    let err_oot_1 = clamp_err(&pred_oot_1, ctx_frames[k_obs]);
    let mut z_closed = ctx_frames[0].to_vec();
    for _t in 0..k_obs {
        let (bn, _, _) = blend_step(diags, weights, &z_closed);
        z_closed = fw_oot.apply_frame(&z_closed, &bn);
    }
    let err_oot_closed = clamp_err(&z_closed, ctx_frames[k_obs]);
    let base_closed_k = base_replay[k_obs - 1];

    if !(err_oot_1 < 0.97 * base_oot_err && err_oot_closed < 0.98 * base_closed_k) {
        return (fw, false);
    }

    // 2. Compuerta OOT aprobada: consolidar F_t en todo el contexto 0..=k_obs reteniendo el mejor paso
    let mut tr = EligibilityTrace::new(0.50);
    let mut best_fw = fw_oot.clone();
    best_fw.scale = fw.scale;
    best_fw.r_bar = fw.r_bar;
    let mut best_loss = err_oot_1 + err_oot_closed;
    s_nm = [0.0; N_GRU_H];

    for _iter in 0..18 {
        tr.reset();
        let mut zc = ctx_frames[0].to_vec();
        let mut iter_err = 0.0;
        for t in 0..k_obs {
            let (bn_c, _, dis_c) = blend_step(diags, weights, &zc);
            let pred_c = fw.apply_frame(&zc, &bn_c);
            let err_c = clamp_err(&pred_c, ctx_frames[t + 1]);
            let m_t = acc.observe(&pred_c, err_c, dis_c, fw.saturation());
            let sig_nm = if cfg.use_neuromod {
                let (sn, ns) = ctrl.nm.step(m_t, s_nm);
                s_nm = ns;
                sn
            } else {
                NeuromodSignal {
                    alpha: 0.25,
                    lambda: 0.90,
                    beta: ctrl.base_tau,
                    g_mode: [0.34, 0.33, 0.33],
                }
            };
            let alpha_eff = (0.25 + 0.55 * sig_nm.alpha).clamp(0.10, 0.75);
            let lam_eff = sig_nm.lambda.clamp(0.96, 0.998);
            tr.update_and_consolidate(
                &mut fw,
                &zc,
                &bn_c,
                ctx_frames[t + 1],
                alpha_eff,
                lam_eff,
                1.0,
            );
            zc = fw.apply_frame(&zc, &bn_c);
            iter_err += clamp_err(&zc, ctx_frames[t + 1]);
        }
        let loss = iter_err / k_obs as f64;
        if loss.is_finite() && loss < best_loss {
            best_loss = loss;
            best_fw = fw.clone();
        }
    }

    let active = best_fw.active;
    (best_fw, active)
}

/// Fusión Bayesiana de dos ramas independientes (avance p0 vs inversión exacta de Φ_rev por Newton 2×2):
/// Resuelve Φ_rev(p_inv) = x_{t-1} mediante [J_{Φ_rev}(p)]^{-1} (2 pasos amortiguados) y fusiona
///   p* = (1 - η) p0 + η p_inv
/// con verificación monótona estricta sobre la consistencia de ciclo.
fn cycle_manifold_correct(
    refuter: &BackwardRefuter,
    x_prev: &[f64],
    p0: &[f64],
    eta: f64,
    scale: f64,
) -> (Vec<f64>, bool, f64, f64) {
    let (e0, epi_rev, x_back0) = refuter.refute_step(x_prev, p0);
    if eta < 0.015 || e0 < 1e-6 || e0 > 1.2 || !fin(p0) || !fin(&x_back0) || !refuter.valid {
        return (p0.to_vec(), false, e0, epi_rev);
    }
    let np = x_prev.len() / 2;
    let eps = (1e-4 * scale).max(1e-6);
    let mut p_inv = p0.to_vec();

    // 2 pasos exactos de Newton 2×2 sobre Φ_rev(p) = x_prev
    for _nt in 0..2 {
        for i in 0..np {
            let p = [p_inv[2 * i], p_inv[2 * i + 1]];
            let q_tgt = [x_prev[2 * i], x_prev[2 * i + 1]];
            let (b0, _) = refuter.hyp.step_point(p);
            let (bx, _) = refuter.hyp.step_point([p[0] + eps, p[1]]);
            let (by, _) = refuter.hyp.step_point([p[0], p[1] + eps]);
            if !b0[0].is_finite() || !bx[0].is_finite() || !by[0].is_finite() {
                continue;
            }
            let j00 = (bx[0] - b0[0]) / eps;
            let j10 = (bx[1] - b0[1]) / eps;
            let j01 = (by[0] - b0[0]) / eps;
            let j11 = (by[1] - b0[1]) / eps;
            let det = j00 * j11 - j01 * j10;
            if det.abs() < 1e-5 {
                continue;
            }
            let r0 = b0[0] - q_tgt[0];
            let r1 = b0[1] - q_tgt[1];
            let mut dx = -(j11 * r0 - j01 * r1) / det;
            let mut dy = -(-j10 * r0 + j00 * r1) / det;
            let dnorm = (dx * dx + dy * dy).sqrt();
            let max_step = 0.25 * scale;
            if dnorm > max_step && dnorm > 1e-12 {
                let k = max_step / dnorm;
                dx *= k;
                dy *= k;
            }
            p_inv[2 * i] += dx;
            p_inv[2 * i + 1] += dy;
        }
    }
    if !fin(&p_inv) {
        return (p0.to_vec(), false, e0, epi_rev);
    }
    // Verificar que p_inv realmente invierte Φ_rev mejor que p0 y está acotado cerca de p0
    let (e_inv, _, _) = refuter.refute_step(x_prev, &p_inv);
    let shift_rel = mt::rel_err(&p_inv, p0);
    if e_inv >= 0.75 * e0 || shift_rel > 0.35 {
        return (p0.to_vec(), false, e0, epi_rev);
    }
    // Fusión ponderada por precisión Bayesiana: p* = (1 - η) p0 + η p_inv
    let mut p_corr = vec![0.0; p0.len()];
    for j in 0..p0.len() {
        p_corr[j] = (1.0 - eta) * p0[j] + eta * p_inv[j];
    }
    let r0 = rmax_frame(p0).max(scale);
    let rc = rmax_frame(&p_corr);
    if !fin(&p_corr) || rc > 1.20 * r0 {
        return (p0.to_vec(), false, e0, epi_rev);
    }
    let (ec, epi_c, _) = refuter.refute_step(x_prev, &p_corr);
    if ec < 0.96 * e0 {
        (p_corr, true, ec, epi_c)
    } else {
        (p0.to_vec(), false, e0, epi_rev)
    }
}

/// Inferencia completa de METAFE-3S a partir de diagnósticos precalculados del núcleo base W_0.
fn infer_3s_from_diags(
    cfg: &MetaFe3SCfg,
    ctrl: &Meta3SController,
    e: &Ep,
    ctx_frames: &[&[f64]],
    fe_null: f64,
    inv: &[f64; N_INV],
    diags: &[HypDiag],
    refuter: Option<&BackwardRefuter>,
    k_obs: usize,
) -> Episode3SReport {
    let np = e.x0.len() / 2;
    let pi_rev = if cfg.use_refuter {
        refuter.map(|r| r.reliability).unwrap_or(0.0)
    } else {
        0.0
    };

    // 1. Puntuación de Energía Libre + hiper-prior consolidado + refutación cruzada inversa
    let mut base_scores = Vec::with_capacity(diags.len());
    for d in diags {
        if !d.hyp.valid {
            base_scores.push(1e6);
            continue;
        }
        let hidx = d.hyp.kind as usize;
        let morph = if cfg.is_aura_v3_genesis {
            0.08 * morphogen_chemotaxis_affinity(d.hyp.kind, inv)
        } else {
            0.0
        };
        let lp: f64 = if ctrl.prior_scale > 0.0 {
            let v: f64 = ctrl.prior_w[hidx].iter().zip(inv).map(|(a, b)| a * b).sum();
            (ctrl.prior_scale * v + morph).clamp(-2.0, 2.0)
        } else {
            morph
        };
        let r_sc = if cfg.use_refuter && pi_rev > 0.10 && ctrl.w_refute > 0.0 {
            refuter.unwrap().score_trajectory(ctx_frames[k_obs], &d.roll_preds)
        } else {
            0.0
        };
        let s = ctrl.fe_w[0] * d.vfe_per_pt
            + ctrl.fe_w[1] * d.loo_per_pt
            + ctrl.fe_w[2] * d.oot1_log
            + ctrl.fe_w[3] * d.oot2_log
            + ctrl.fe_w[4] * d.replay_log
            + ctrl.fe_w[5] * d.stab_pen
            + ctrl.w_refute * pi_rev * r_sc
            - lp;
        base_scores.push(if s.is_finite() { s } else { 1e6 });
    }

    let init_w = softmax_scores(&base_scores, ctrl.base_tau);
    let best_i = init_w
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_hyp = diags[best_i].hyp.kind as usize;

    // 2. Adaptación de Pesos Rápidos Acotados F_t (3-factores, verificación precuencial OOT)
    let (mut fw, mut fast_used) =
        adapt_fast_weights_in_context(cfg, ctrl, ctx_frames, diags, &init_w, k_obs);

    // 3. Predicciones filtradas precuenciales 1..=K
    let mut preds = vec![vec![0.0; 2 * np]; HORIZON];
    preds[0] = e.frame(0).to_vec();
    for t in 2..=k_obs {
        let h_sub = fit_hypothesis(HypKind::LieAffine, &ctx_frames[..t]);
        if h_sub.valid {
            preds[t - 1] = h_sub.step_frame(e.frame(t - 1)).0;
        } else {
            preds[t - 1] = e.frame(t - 1).to_vec();
        }
    }

    // 4. Rollout h=(K+1)..=16 gobernado en tiempo real por ACC + MetaNet + Refutación Cruzada
    let rmax_ctx = (0..=k_obs).map(|t| rmax_frame(e.frame(t))).fold(1e-6, f64::max);
    let exact_regime = diags[best_i].oot1_log < -10.5 && diags[best_i].replay_log < -10.5;

    let mut acc = AccMonitor::new(4);
    for t in 0..=k_obs {
        acc.observe(e.frame(t), diags[best_i].replay_log.exp(), 1e-6, fw.saturation());
    }
    let mut s_nm = [0.0; N_GRU_H];

    // Pesos por trayectoria base con puerta epistémica (+ Subrutina 3: Poda Sináptica por Microglía)
    let is_curvilinear_or_symp = matches!(
        diags[best_i].hyp.kind,
        HypKind::GpSympFwd | HypKind::GpSympRev | HypKind::GpPolar
    );
    let prune_gap = if cfg.is_aura_v3_genesis {
        if is_curvilinear_or_symp { 1.10 } else { 3.5 }
    } else {
        5.0
    };
    let mut traj_w = vec![0.0; diags.len()];
    for (i, (d, &w)) in diags.iter().zip(&init_w).enumerate() {
        if w < 1e-5 || !d.hyp.valid || !fin(&d.roll_preds[HORIZON - k_obs - 1]) {
            continue;
        }
        // Subrutina 3 (Jesse Conel): la microglía devora físicamente las columnas fuera del umbral de variedad
        if cfg.is_aura_v3_genesis && i != best_i && base_scores[i] - base_scores[best_i] > prune_gap {
            continue;
        }
        let em = mean(&d.roll_epi) / d.hyp.s2_scale.max(1e-8);
        traj_w[i] = w * (-ctrl.gamma_epi * em.clamp(0.0, 12.0)).exp();
    }
    let tw_sum: f64 = traj_w.iter().sum();
    if tw_sum > 1e-12 {
        for w in &mut traj_w {
            *w /= tw_sum;
        }
    } else {
        traj_w[best_i] = 1.0;
    }

    let mut mean_epi_acc = 0.0;
    let mut ens_dis_acc = 0.0;
    let mut ref_errs = Vec::with_capacity(HORIZON - k_obs);
    let mut rev_in_sup_num = 0.0;
    let mut rev_in_sup_den = 0.0;
    let rev_s2 = refuter.map(|r| r.hyp.s2_scale.max(1e-8)).unwrap_or(1.0);
    let mut alphas = Vec::with_capacity(HORIZON - k_obs);
    let mut lambdas = Vec::with_capacity(HORIZON - k_obs);
    let mut betas = Vec::with_capacity(HORIZON - k_obs);
    let mut cycle_accepted = 0usize;
    let mut tr_roll = EligibilityTrace::new(0.5);

    let mut cur = e.frame(k_obs).to_vec();
    let fwd_oot_var = diags[best_i].oot1_log.exp().powi(2);
    let rev_oot_var = refuter.map(|r| r.rev_oot1_err.powi(2)).unwrap_or(1e3);

    let mut ord: Vec<usize> = (0..diags.len()).collect();
    ord.sort_by(|&a, &b| base_scores[a].total_cmp(&base_scores[b]));
    let max_active = if cfg.is_aura_v3_genesis && is_curvilinear_or_symp { 2 } else { 4 };
    let active_idx: Vec<usize> = ord
        .into_iter()
        .filter(|&i| diags[i].hyp.valid && base_scores[i] < base_scores[best_i] + prune_gap)
        .take(max_active)
        .collect();

    for step in 0..(HORIZON - k_obs) {
        if exact_regime {
            let p_exact = diags[best_i].roll_preds[step].clone();
            let epi = diags[best_i].roll_epi[step];
            let (r_err, r_epi) = if cfg.use_refuter && pi_rev > 1e-4 {
                if let Some(rf) = refuter {
                    let (re, er, _) = rf.refute_step(&cur, &p_exact);
                    (re, er)
                } else {
                    (1e-9, 1e6)
                }
            } else {
                (1e-9, 1e6)
            };
            let w_sup = if cfg.use_refuter && pi_rev > 0.25 {
                pi_rev / (1.0 + (r_epi / rev_s2).clamp(0.0, 40.0))
            } else {
                0.0
            };
            rev_in_sup_num += w_sup * (r_err + 1e-6).ln().clamp(-7.5, 2.3);
            rev_in_sup_den += w_sup;
            let m_t = acc.observe(&p_exact, r_err, 1e-9, 0.0);
            let sn = if cfg.use_neuromod {
                let (sn, ns) = ctrl.nm.step(m_t, s_nm);
                s_nm = ns;
                sn
            } else {
                NeuromodSignal {
                    alpha: 0.25,
                    lambda: 0.85,
                    beta: ctrl.base_tau,
                    g_mode: [0.34, 0.33, 0.33],
                }
            };
            mean_epi_acc += epi.sqrt();
            ref_errs.push(r_err);
            alphas.push(sn.alpha);
            lambdas.push(sn.lambda);
            betas.push(sn.beta);
            preds[k_obs + step] = p_exact.clone();
            cur = p_exact;
            continue;
        }

        // Régimen no lineal / OOD: evaluar las ramas activas desde el estado actual `cur`
        let mut step_scores = Vec::with_capacity(active_idx.len());
        let mut step_cands: Vec<(usize, Vec<f64>, f64, f64)> = Vec::with_capacity(active_idx.len());
        for &hi in &active_idx {
            let d = &diags[hi];
            let (pn, epi) = d.hyp.step_frame(&cur);
            if !fin(&pn) {
                continue;
            }
            let epi_norm = (epi / d.hyp.s2_scale.max(1e-8)).clamp(0.0, 15.0);
            let step_ref = if cfg.use_refuter && pi_rev > 0.10 {
                let (re, erev, _) = refuter.unwrap().refute_step(&cur, &pn);
                let wr = pi_rev / (1.0 + (erev / refuter.unwrap().hyp.s2_scale.max(1e-8)).clamp(0.0, 15.0));
                wr * log_rel(re.max(1e-7))
            } else {
                0.0
            };
            let sc = base_scores[hi] + ctrl.gamma_epi * epi_norm + 0.35 * ctrl.w_refute * step_ref;
            step_scores.push(sc);
            step_cands.push((hi, pn, epi, step_ref));
        }

        let mut pre_fused = vec![0.0; 2 * np];
        let (pre_dis, pre_epi) = if !step_cands.is_empty() {
            let pw0 = softmax_scores(&step_scores, ctrl.base_tau);
            let mut ep = 0.0;
            for ((_, pn, epi, _), &w) in step_cands.iter().zip(&pw0) {
                for j in 0..2 * np {
                    pre_fused[j] += w * pn[j];
                }
                ep += w * epi;
            }
            let dis: f64 = step_cands
                .iter()
                .zip(&pw0)
                .map(|((_, pn, _, _), &w)| w * mt::rel_err(pn, &pre_fused).min(10.0))
                .sum();
            (dis, ep)
        } else {
            pre_fused = diags[best_i].roll_preds[step].clone();
            (0.0, diags[best_i].roll_epi[step])
        };

        let (pre_ref, mut final_rev_epi) = if cfg.use_refuter && pi_rev > 1e-4 {
            let (re, er, _) = refuter.unwrap().refute_step(&cur, &pre_fused);
            (re, er)
        } else {
            (diags[best_i].oot1_log.exp(), 1e6)
        };

        let m_t = acc.observe(&pre_fused, pre_ref, pre_dis, fw.saturation());
        let sn = if cfg.use_neuromod {
            let (sn, ns) = ctrl.nm.step(m_t, s_nm);
            s_nm = ns;
            sn
        } else {
            NeuromodSignal {
                alpha: 0.25,
                lambda: 0.85,
                beta: ctrl.base_tau,
                g_mode: [0.34, 0.33, 0.33],
            }
        };

        // Precisión acetilcolina β_t: modula la nitidez del enrutamiento dlPFC g_t
        let beta_eff = if cfg.use_neuromod {
            sn.beta.clamp(1.5, 12.0)
        } else {
            ctrl.base_tau
        };

        let mut coupled = vec![0.0; 2 * np];
        let (step_epi, step_dis) = if !step_cands.is_empty() {
            let gw = softmax_scores(&step_scores, beta_eff);
            let mut ep = 0.0;
            for ((_, pn, epi, _), &w) in step_cands.iter().zip(&gw) {
                for j in 0..2 * np {
                    coupled[j] += w * pn[j];
                }
                ep += w * epi;
            }
            let dis: f64 = step_cands
                .iter()
                .zip(&gw)
                .map(|((_, pn, _, _), &w)| w * mt::rel_err(pn, &coupled).min(10.0))
                .sum();
            (ep, dis)
        } else {
            coupled = diags[best_i].roll_preds[step].clone();
            (pre_epi, pre_dis)
        };

        // Aplicar adaptador rápido acotado F_t (si activo en contexto o por refutación en rollout)
        let coupled_fw = if fw.active && cfg.use_fast_weights {
            fw.apply_frame(&cur, &coupled)
        } else {
            coupled.clone()
        };

        // Mezcla por modo dlPFC entre trayectoria base de W_0 y flujo acoplado + F_t
        let mut traj_pt = vec![0.0; 2 * np];
        let mut traj_epi = 0.0;
        for (i, d) in diags.iter().enumerate() {
            if traj_w[i] > 0.0 && fin(&d.roll_preds[step]) {
                for j in 0..2 * np {
                    traj_pt[j] += traj_w[i] * d.roll_preds[step][j];
                }
                traj_epi += traj_w[i] * d.roll_epi[step].sqrt();
            }
        }
        let traj_dis: f64 = diags
            .iter()
            .enumerate()
            .filter(|(i, d)| traj_w[*i] > 0.0 && fin(&d.roll_preds[step]))
            .map(|(i, d)| traj_w[i] * mt::rel_err(&d.roll_preds[step], &traj_pt).min(10.0))
            .sum();
        let w_coupled = if cfg.is_aura_v3_genesis && cycle_accepted > 0 {
            1.0
        } else if fw.active {
            (0.70 + 0.30 * (sn.g_mode[1] + sn.g_mode[2])).clamp(0.55, 1.0)
        } else if cfg.is_aura_v3_genesis && active_idx.len() <= 1 {
            0.0
        } else {
            (0.45 + 0.50 * sn.g_mode[1]).clamp(0.30, 0.95)
        };
        let mut fused = vec![0.0; 2 * np];
        if fin(&coupled_fw) && fin(&traj_pt) {
            for j in 0..2 * np {
                fused[j] = w_coupled * coupled_fw[j] + (1.0 - w_coupled) * traj_pt[j];
            }
        } else if fin(&coupled_fw) {
            fused = coupled_fw;
        } else {
            fused = traj_pt;
        }

        // Fusión Bayesiana de dos ramas independientes (Φ_fwd vs inversión exacta Φ_rev^{-1})
        let mut final_ref = pre_ref;
        if cfg.use_refuter && pi_rev > 0.20 && ctrl.cycle_gain > 0.0 {
            if let Some(rf) = refuter {
                let (_, epi_rev_now, _) = rf.refute_step(&cur, &fused);
                let var_fwd = step_epi + fwd_oot_var;
                let var_rev = epi_rev_now + rev_oot_var;
                let prec_ratio = var_fwd / (var_fwd + var_rev + 1e-10);
                let mode_gate = if cfg.use_neuromod {
                    (0.55 + 0.75 * sn.g_mode[1]).clamp(0.30, 1.25)
                } else {
                    0.80
                };
                let eta = (pi_rev * ctrl.cycle_gain * prec_ratio * mode_gate).clamp(0.0, 0.65);
                let (p_corr, accepted, e_after, epi_after) =
                    cycle_manifold_correct(rf, &cur, &fused, eta, fw.scale);
                final_ref = e_after;
                final_rev_epi = epi_after;
                if accepted {
                    cycle_accepted += 1;
                    // Plasticidad 3-factores online guiada por refutación cruzada verificada (sin etiqueta externa)
                    if cfg.use_fast_weights && pi_rev > 0.50 && sn.alpha > 0.08 {
                        let a_roll = (0.20 * sn.alpha * pi_rev).clamp(0.02, 0.20);
                        let l_roll = sn.lambda.clamp(0.90, 0.99);
                        tr_roll.update_and_consolidate(
                            &mut fw,
                            &cur,
                            &coupled,
                            &p_corr,
                            a_roll,
                            l_roll,
                            prec_ratio,
                        );
                        if fw.active {
                            fast_used = true;
                        }
                    }
                    fused = p_corr;
                }
            }
        }

        // Si MetaNet señala colapso / saturación extrema (λ_t bajo), decaer F_t de forma segura hacia W_0
        if fw.active && sn.lambda < 0.55 {
            for r in 0..2 {
                for c in 0..N_FAST_FEAT {
                    fw.f[r][c] *= sn.lambda;
                }
            }
            fw.active = fw.frob_norm() > 1e-10;
        }

        let w_sup = if cfg.use_refuter && pi_rev > 0.25 {
            pi_rev / (1.0 + (final_rev_epi / rev_s2).clamp(0.0, 40.0))
        } else {
            0.0
        };
        rev_in_sup_num += w_sup * (final_ref + 1e-6).ln().clamp(-7.5, 2.3);
        rev_in_sup_den += w_sup;

        mean_epi_acc += 0.5 * step_epi.sqrt() + 0.5 * traj_epi;
        ens_dis_acc += 0.5 * step_dis + 0.5 * traj_dis;
        ref_errs.push(final_ref);
        alphas.push(sn.alpha);
        lambdas.push(sn.lambda);
        betas.push(beta_eff);
        preds[k_obs + step] = fused.clone();
        cur = fused;
    }

    let n_roll = (HORIZON - k_obs) as f64;
    let mean_epi = (mean_epi_acc / n_roll).max(0.0);
    let (ens_disagree, post_entropy) = if cfg.is_aura_v3_genesis {
        // La telemetría metacognitiva de 2º orden ("Yo-FE") lee la población cortical pre-poda
        // (mientras que la salida motora `preds` usa la vía mielinizada y podada por la microglía)
        let meta_w = softmax_scores(&base_scores, 3.5);
        let pe: f64 = -meta_w
            .iter()
            .filter(|&&w| w > 1e-12)
            .map(|&w| w * w.ln())
            .sum::<f64>();
        let last_step = HORIZON - k_obs - 1;
        let mut bar = vec![0.0; 2 * np];
        let mut wsum = 0.0;
        for (d, &w) in diags.iter().zip(&meta_w) {
            if w > 1e-4 && d.hyp.valid && fin(&d.roll_preds[last_step]) {
                for j in 0..2 * np {
                    bar[j] += w * d.roll_preds[last_step][j];
                }
                wsum += w;
            }
        }
        let pop_dis = if wsum > 1e-12 {
            for v in &mut bar {
                *v /= wsum;
            }
            diags
                .iter()
                .zip(&meta_w)
                .filter(|(d, &w)| w > 1e-4 && d.hyp.valid && fin(&d.roll_preds[last_step]))
                .map(|(d, &w)| (w / wsum) * mt::rel_err(&d.roll_preds[last_step], &bar).min(10.0))
                .sum::<f64>()
        } else {
            (ens_dis_acc / n_roll).max(0.0)
        };
        (0.5 * (ens_dis_acc / n_roll).max(0.0) + 0.5 * pop_dis, pe)
    } else {
        let pe: f64 = -init_w
            .iter()
            .filter(|&&w| w > 1e-12)
            .map(|&w| w * w.ln())
            .sum::<f64>();
        ((ens_dis_acc / n_roll).max(0.0), pe)
    };
    let nan = !fin(&preds[HORIZON - 1]);

    // 5. Telemetría Metacognitiva "Yo-FE 3S" (13 rasgos Etapa A + 8 rasgos Platt Etapa B)
    let fe_post = diags[best_i].vfe_per_pt;
    let fe_reduction = fe_null - fe_post;
    let r_end = if !nan { rmax_frame(&preds[HORIZON - 1]) } else { 1e3 };
    let norm_ratio = (r_end / rmax_ctx).clamp(1e-4, 1e3).ln();

    let mean_refute = mean(&ref_errs).clamp(1e-9, 10.0);
    let max_refute = ref_errs.iter().cloned().fold(1e-9_f64, f64::max).clamp(1e-9, 10.0);
    let max_stagnation = acc
        .hist_stagnation
        .iter()
        .cloned()
        .fold(-1.0_f64, f64::max)
        .clamp(-1.0, 1.0);
    let mean_alpha = mean(&alphas);
    let min_lambda = lambdas.iter().cloned().fold(1.0_f64, f64::min);
    let mean_beta = mean(&betas);
    let fast_norm = fw.frob_norm();

    // 10 rasgos núcleo forward (VFE + OOT + incertidumbre epistémica, idéntico conditioning a Yo-FE v2)
    let self_feats: [f64; N_SELF_3S] = [
        1.0,
        fe_post.clamp(-10.0, 10.0) / 5.0,
        fe_reduction.clamp(-5.0, 15.0) / 10.0,
        post_entropy / (diags.len().max(2) as f64).ln(),
        diags[best_i].oot1_log.clamp(-7.5, 2.3) / 5.0,
        diags[best_i].oot2_log.clamp(-7.5, 2.3) / 5.0,
        diags[best_i].replay_log.clamp(-7.5, 2.3) / 5.0,
        (mean_epi + 1e-7).ln().clamp(-12.0, 4.0) / 6.0,
        (ens_disagree + 1e-7).ln().clamp(-12.0, 4.0) / 6.0,
        norm_ratio.clamp(-4.0, 4.0) / 4.0,
    ];

    let err_log_core: f64 = ctrl
        .err_w
        .iter()
        .zip(&self_feats)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-9.0, 4.0);

    // Exceso de refutación estrictamente EN SOPORTE epistémico de Φ_rev (evita falsas alarmas al salir de la nube de contexto)
    let in_sup_cov = (rev_in_sup_den / n_roll).clamp(0.0, 1.0);
    let in_sup_ref_excess = if cfg.use_refuter && rev_in_sup_den > 1e-6 {
        let in_sup_ref_log = rev_in_sup_num / rev_in_sup_den;
        (in_sup_cov * (in_sup_ref_log - err_log_core).max(0.0)).clamp(0.0, 4.0) / 4.0
    } else {
        0.0
    };
    let err_log_hat = (err_log_core + ctrl.ref_meta_w * in_sup_ref_excess).clamp(-9.0, 4.0);

    let margin = (mt::REL_ERR_OK.ln() - err_log_hat).clamp(-6.0, 6.0);
    let cal_feats: [f64; N_CAL_3S] = [
        1.0,
        margin,
        margin.tanh(),
        self_feats[7],     // log(mean_epi)
        self_feats[8],     // log(ens_disagree)
        self_feats[3],     // entropía posterior
        self_feats[9],     // norm_ratio
        in_sup_ref_excess, // exceso de refutación en soporte epistémico de Φ_rev
    ];
    let logit: f64 = ctrl.cal_w.iter().zip(&cal_feats).map(|(a, b)| a * b).sum();
    let p_ok_hat = sig(logit.clamp(-30.0, 30.0));

    Episode3SReport {
        preds,
        fe_null,
        fe_post,
        fe_reduction,
        post_entropy,
        best_hyp,
        p_ok_hat,
        err_log_hat,
        mean_epi,
        ens_disagree,
        mean_refute,
        max_refute,
        in_sup_ref_excess,
        rev_reliability: pi_rev,
        fast_norm,
        fast_used,
        cycle_accepted,
        mean_alpha,
        min_lambda,
        mean_beta,
        max_stagnation,
        self_feats,
        cal_feats,
        nan,
    }
}

pub fn run_metafe3s_episode(
    cfg: &MetaFe3SCfg,
    ctrl: &Meta3SController,
    e: &Ep,
    k_obs: usize,
) -> Episode3SReport {
    let ctx_frames: Vec<&[f64]> = (0..=k_obs).map(|t| e.frame(t)).collect();
    let fe_null = null_free_energy(&ctx_frames);
    let inv = episode_invariants(&ctx_frames);
    let hyps = all_hyps();
    let (diags, affine_myelinated): (Vec<HypDiag>, bool) = if cfg.is_aura_v3_genesis {
        evaluate_all_hypotheses_genesis(&hyps, &ctx_frames, k_obs)
    } else {
        (
            hyps.iter()
                .map(|&hk| evaluate_hypothesis(hk, &ctx_frames, true, true, k_obs))
                .collect(),
            false,
        )
    };
    let refuter = if cfg.use_refuter && !affine_myelinated {
        Some(BackwardRefuter::fit(&ctx_frames))
    } else {
        None
    };
    infer_3s_from_diags(cfg, ctrl, e, &ctx_frames, fe_null, &inv, &diags, refuter.as_ref(), k_obs)
}

// ─── Entrenamiento Outer-Loop METAFE-3S (con Buffer Episódico + EWC + BPTT) ──

struct Cached3S {
    inv: [f64; N_INV],
    fe_null: f64,
    diags: Vec<HypDiag>,
    refuter: Option<BackwardRefuter>,
    true_errs: Vec<f64>,
    ep: Ep,
}

fn build_cache_3s(eps: &[Ep], need_refuter: bool) -> Vec<Cached3S> {
    build_cache_3s_mode(eps, need_refuter, false)
}

fn build_cache_3s_mode(eps: &[Ep], need_refuter: bool, is_genesis: bool) -> Vec<Cached3S> {
    let hyps = all_hyps();
    eps.iter()
        .map(|e| {
            let ctx_f: Vec<&[f64]> = (0..=K).map(|t| e.frame(t)).collect();
            let fe_null = null_free_energy(&ctx_f);
            let inv = episode_invariants(&ctx_f);
            let (diags, affine_myelinated): (Vec<HypDiag>, bool) = if is_genesis {
                evaluate_all_hypotheses_genesis(&hyps, &ctx_f, K)
            } else {
                (
                    hyps.iter()
                        .map(|&hk| evaluate_hypothesis(hk, &ctx_f, true, true, K))
                        .collect(),
                    false,
                )
            };
            let refuter = if need_refuter && !affine_myelinated {
                Some(BackwardRefuter::fit(&ctx_f))
            } else {
                None
            };
            let true_errs: Vec<f64> = diags
                .iter()
                .map(|d| log_rel(clamp_err(&d.roll_preds[HORIZON - K - 1], &e.ys[HORIZON - 1])))
                .collect();
            Cached3S {
                inv,
                fe_null,
                diags,
                refuter,
                true_errs,
                ep: e.clone(),
            }
        })
        .collect()
}

/// Construye entradas del Buffer Episódico filtrando episodios donde la discriminación es verificable.
fn cache_to_buffer_entries(cache: &[Cached3S], only_known: Option<bool>) -> Vec<EpisodicEntry> {
    let mut out = Vec::new();
    for cm in cache {
        if let Some(kn) = only_known {
            if (cm.ep.fam < N_FAM) != kn {
                continue;
            }
        }
        let mn = cm.true_errs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = cm.true_errs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if mx - mn <= 0.15 {
            continue;
        }
        let clamped: Vec<f64> = cm.true_errs.iter().map(|&v| v.clamp(-8.0, 2.3)).collect();
        let tgt_v = softmax_scores(&clamped, 1.5);
        let mut hyp_target = [0.0; N_HYP];
        for (i, d) in cm.diags.iter().enumerate() {
            hyp_target[d.hyp.kind as usize] = tgt_v[i];
        }
        out.push(EpisodicEntry {
            inv: cm.inv,
            hyp_target,
            weight: 1.0,
        });
    }
    out
}

/// Genera secuencias meta-supervisadas ponderadas (estables, OOD no lineales, cambio de régimen y saturación)
/// para entrenar el controlador neuromodulador GRU `MetaNet` mediante BPTT exacto.
fn build_neuromod_sequences(cache_tr: &[Cached3S], seed: u64) -> Vec<Vec<([f64; N_ACC], [f64; 6], f64)>> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_3S_NM ^ 0xBEEF);
    let mut seqs: Vec<Vec<([f64; N_ACC], [f64; 6], f64)>> = Vec::new();

    // 1. Secuencias de cambio de régimen abrupto (sorpresa baja -> pico repentino -> recuperación)
    for _ in 0..64 {
        let t_shift = rng.gen_range(2..6);
        let mut seq = Vec::with_capacity(9);
        for t in 0..9 {
            if t < t_shift {
                let m = [
                    rng.gen_range(-1.4..-0.45),
                    rng.gen_range(-1.3..-0.3),
                    rng.gen_range(-0.3..0.2),
                    rng.gen_range(0.0..0.15),
                ];
                seq.push((m, [0.04, 0.96, 0.85, 0.82, 0.13, 0.05], 1.0));
            } else if t == t_shift {
                let m = [
                    rng.gen_range(0.30..1.85),
                    rng.gen_range(-0.2..1.5),
                    rng.gen_range(-0.3..0.6),
                    rng.gen_range(0.0..0.5),
                ];
                // Pico de conflicto ACC: peso balanceado (6.0) para respuesta nítida λ_t↓ (RESET) y α_t↑
                seq.push((m, [0.95, 0.05, 0.12, 0.05, 0.15, 0.80], 6.0));
            } else if t == t_shift + 1 {
                let m = [
                    rng.gen_range(-0.1..0.55),
                    rng.gen_range(-0.3..0.5),
                    rng.gen_range(-0.3..0.3),
                    rng.gen_range(0.1..0.5),
                ];
                seq.push((m, [0.70, 0.88, 0.30, 0.15, 0.30, 0.55], 2.0));
            } else {
                let m = [
                    rng.gen_range(-1.1..-0.25),
                    rng.gen_range(-0.9..0.0),
                    rng.gen_range(-0.3..0.2),
                    rng.gen_range(0.1..0.4),
                ];
                seq.push((m, [0.12, 0.95, 0.65, 0.55, 0.35, 0.10], 1.0));
            }
        }
        seqs.push(seq);
    }

    // 2. Secuencias de rollout real extraídas de cache_tr (0 fuga {8,10,11}):
    //    enseñan a MetaNet cuándo explotar (β alto en afines) vs mezclar con refutación (β moderado en OOD)
    for cm in cache_tr.iter().take(64) {
        let best_e = cm.true_errs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut sorted = cm.true_errs.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let gap = if sorted.len() >= 2 { sorted[1] - sorted[0] } else { 1.0 };
        let is_exact = best_e < -10.0;
        let prefer_blend = !is_exact && gap < 1.0;

        let mut seq = Vec::with_capacity(6);
        for t in 0..6 {
            let ref_m = if is_exact {
                -1.25
            } else {
                ((best_e + 5.0) / 4.0 + 0.05 * t as f64).clamp(-1.1, 1.3)
            };
            let dis_m = if is_exact {
                -1.25
            } else if prefer_blend {
                rng.gen_range(-0.3..0.7)
            } else {
                rng.gen_range(-0.9..-0.2)
            };
            let stag = rng.gen_range(-0.4..0.5);
            let sat = if is_exact { 0.0 } else { rng.gen_range(0.0..0.35) };
            let tgt = if is_exact {
                [0.03, 0.97, 0.90, 0.88, 0.08, 0.04]
            } else if prefer_blend {
                [0.40, 0.92, 0.22, 0.10, 0.75, 0.15]
            } else {
                [0.28, 0.93, 0.42, 0.30, 0.58, 0.12]
            };
            seq.push(([ref_m, dis_m, stag, sat], tgt, 1.0));
        }
        seqs.push(seq);
    }

    // 3. Secuencias de saturación plástica / estancamiento en bucle
    for _ in 0..24 {
        let mut seq = Vec::with_capacity(6);
        for t in 0..6 {
            let sat = (0.3 + 0.14 * t as f64).min(1.0);
            let stag = (0.4 + 0.10 * t as f64).min(0.98);
            let surp = 0.45 + 0.10 * t as f64;
            let m = [surp, 0.4, stag, sat];
            if sat > 0.80 && stag > 0.75 {
                seq.push((m, [0.10, 0.25, 0.18, 0.20, 0.65, 0.15], 3.0));
            } else {
                seq.push((m, [0.45, 0.82, 0.30, 0.20, 0.50, 0.30], 1.0));
            }
        }
        seqs.push(seq);
    }

    seqs
}

pub fn train_metafe3s_controller(
    cfg: &MetaFe3SCfg,
    seed: u64,
    n_train: usize,
    n_val: usize,
) -> (Meta3SController, serde_json::Value) {
    let mut ctrl = Meta3SController::default();
    let (eps_tr, rej1) = gen_meta_training_eps(seed, n_train, XOR_METAFE_TRAIN);
    let (eps_va, rej2) = gen_meta_training_eps(seed, n_val, XOR_METAFE_VAL);

    let cache_tr = build_cache_3s_mode(&eps_tr, cfg.use_refuter, cfg.is_aura_v3_genesis);
    let cache_va = build_cache_3s_mode(&eps_va, cfg.use_refuter, cfg.is_aura_v3_genesis);

    // 1. Consolidación Offline Secuencial en Buffer Episódico con EWC (Módulo 5 + Subrutina 4 Prenatal)
    let mut buf = EpisodicBuffer::new(256);
    let mut pw = [[0.0_f64; N_INV]; N_HYP];
    let mut ewc = EwcPriorState::default();

    // Subrutina 4 (ADN Generativo): Pre-training prenatal en el útero (ondas sintéticas espontáneas,
    // 0 fuga de {8, 10, 11}, 0 consultas TEST) que calibra los mapas topográficos antes del primer dato postnatal.
    if cfg.is_aura_v3_genesis {
        let (eps_pre, _) = gen_meta_training_eps(seed ^ 0x5052_454E, 32, XOR_METAFE_TRAIN);
        let cache_pre = build_cache_3s_mode(&eps_pre, false, true);
        let pre_entries = cache_to_buffer_entries(&cache_pre, None);
        for e in &pre_entries {
            buf.push(e.clone());
        }
        consolidate_prior_ewc(&mut pw, &pre_entries, &ewc, 0.0, 40, 0.08, 0.04);
        ewc.w_prev = pw;
        ewc.omega = compute_fisher_diag(&pw, &pre_entries);
        ewc.initialized = true;
    }

    let known_entries = cache_to_buffer_entries(&cache_tr, Some(true));
    let syn_entries = cache_to_buffer_entries(&cache_tr, Some(false));
    for e in &known_entries {
        buf.push(e.clone());
    }
    let mu_pre = if cfg.is_aura_v3_genesis && cfg.use_ewc { 4.0 } else { 0.0 };
    consolidate_prior_ewc(&mut pw, &known_entries, &ewc, mu_pre, 65, 0.09, 0.04);
    ewc.w_prev = pw;
    let f_kn = compute_fisher_diag(&pw, &known_entries);
    if cfg.is_aura_v3_genesis {
        for h in 0..N_HYP {
            for j in 0..N_INV {
                ewc.omega[h][j] = 0.35 * ewc.omega[h][j] + f_kn[h][j];
            }
        }
    } else {
        ewc.omega = f_kn;
    }
    ewc.initialized = true;

    for e in &syn_entries {
        buf.push(e.clone());
    }
    let mu_ewc = if cfg.use_ewc { 12.0 } else { 0.0 };
    consolidate_prior_ewc(&mut pw, &syn_entries, &ewc, mu_ewc, 65, 0.09, 0.04);
    ctrl.prior_w = pw;
    ctrl.ewc_state = ewc;
    ctrl.buffer_len = buf.entries.len();

    // 2. Entrenamiento outer-loop del controlador neuromodulador GRU MetaNet (Módulo 3)
    ctrl.nm = NeuromodNet::init(seed);
    if cfg.use_neuromod {
        let nm_seqs = build_neuromod_sequences(&cache_tr, seed);
        ctrl.nm.train_bptt(&nm_seqs, 90, 0.45, 5e-4);
    }

    // 3. Meta-selección de pesos de Energía Libre + refutación + ganancia de ciclo sobre cache_va
    //    Usando log-error acotado [-10.0, 2.3] para que los episodios no lineales/OOD no queden
    //    ahogados por el piso numérico -27.63 de las familias afines exactas (0 fuga).
    let eval_va = |c: &Meta3SController| -> f64 {
        let mut errs = Vec::with_capacity(cache_va.len());
        let mut non_aff = Vec::new();
        for cm in &cache_va {
            let ctx_f: Vec<&[f64]> = (0..=K).map(|t| cm.ep.frame(t)).collect();
            let rep = infer_3s_from_diags(
                cfg,
                c,
                &cm.ep,
                &ctx_f,
                cm.fe_null,
                &cm.inv,
                &cm.diags,
                cm.refuter.as_ref(),
                K,
            );
            let le = log_rel(clamp_err(&rep.preds[HORIZON - 1], &cm.ep.ys[HORIZON - 1])).clamp(-10.0, 2.3);
            errs.push(le);
            if cm.diags[rep.best_hyp].oot1_log > -10.0 {
                non_aff.push(le);
            }
        }
        if non_aff.is_empty() {
            mean(&errs)
        } else {
            0.35 * mean(&errs) + 0.65 * (0.5 * mean(&non_aff) + 0.5 * median(non_aff))
        }
    };

    let fe_candidates: [[f64; 6]; 4] = if cfg.is_aura_v3_genesis {
        [
            [1.0, 0.35, 0.50, 0.50, 0.25, 1.0],
            [0.6, 0.40, 0.80, 0.80, 0.30, 1.0],
            [0.3, 0.30, 1.00, 1.00, 0.50, 1.5],
            [0.8, 0.35, 0.65, 0.65, 0.30, 1.2],
        ]
    } else {
        [
            [1.0, 0.35, 0.50, 0.50, 0.25, 1.0],
            [0.6, 0.40, 0.80, 0.80, 0.30, 1.0],
            [0.3, 0.30, 1.00, 1.00, 0.50, 1.5],
            [1.0, 0.50, 0.25, 0.25, 0.15, 1.0],
        ]
    };
    let w_refs: &[f64] = if cfg.use_refuter { &[0.0, 0.30, 0.65, 1.10] } else { &[0.0] };
    let taus: &[f64] = if cfg.is_aura_v3_genesis {
        &[4.0, 8.0, 12.0]
    } else {
        &[2.5, 5.0, 9.0]
    };
    let pscales: &[f64] = if cfg.is_aura_v3_genesis {
        &[0.0, 0.15]
    } else {
        &[0.0, 0.25, 0.55]
    };

    // Precalcular puntuaciones de trayectoria inversa por episodio de validación para búsqueda rápida
    let va_ref_scores: Vec<Vec<f64>> = cache_va
        .iter()
        .map(|cm| {
            let pi_rev = if cfg.use_refuter {
                cm.refuter.as_ref().map(|r| r.reliability).unwrap_or(0.0)
            } else {
                0.0
            };
            cm.diags
                .iter()
                .map(|d| {
                    if cfg.use_refuter && pi_rev > 0.10 && d.hyp.valid {
                        pi_rev * cm.refuter.as_ref().unwrap().score_trajectory(cm.ep.frame(K), &d.roll_preds)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    let mut cands_proxy: Vec<(f64, [f64; 6], f64, f64, f64)> = Vec::new();
    for &fw in &fe_candidates {
        for &wr in w_refs {
            for &tau in taus {
                for &ps in pscales {
                    let mut errs = Vec::with_capacity(cache_va.len());
                    let mut non_aff = Vec::new();
                    for (ci, cm) in cache_va.iter().enumerate() {
                        let scores: Vec<f64> = cm
                            .diags
                            .iter()
                            .enumerate()
                            .map(|(hi, d)| {
                                if !d.hyp.valid {
                                    return 1e6;
                                }
                                let morph = if cfg.is_aura_v3_genesis {
                                    0.08 * morphogen_chemotaxis_affinity(d.hyp.kind, &cm.inv)
                                } else {
                                    0.0
                                };
                                let lp = if ps > 0.0 {
                                    let v: f64 = ctrl.prior_w[d.hyp.kind as usize]
                                        .iter()
                                        .zip(&cm.inv)
                                        .map(|(a, b)| a * b)
                                        .sum();
                                    (ps * v + morph).clamp(-2.0, 2.0)
                                } else {
                                    morph
                                };
                                fw[0] * d.vfe_per_pt
                                    + fw[1] * d.loo_per_pt
                                    + fw[2] * d.oot1_log
                                    + fw[3] * d.oot2_log
                                    + fw[4] * d.replay_log
                                    + fw[5] * d.stab_pen
                                    + wr * va_ref_scores[ci][hi]
                                    - lp
                            })
                            .collect();
                        let ws = softmax_scores(&scores, tau);
                        let bi = ws
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let dim = cm.ep.x0.len();
                        let mut pred16 = vec![0.0; dim];
                        let mut wsum = 0.0;
                        for (i, (d, &w)) in cm.diags.iter().zip(&ws).enumerate() {
                            if w > 1e-5 && d.hyp.valid && fin(&d.roll_preds[HORIZON - K - 1]) {
                                if cfg.is_aura_v3_genesis && i != bi && (scores[i] - scores[bi] > 1.35 || w < 0.08) {
                                    continue;
                                }
                                for j in 0..dim {
                                    pred16[j] += w * d.roll_preds[HORIZON - K - 1][j];
                                }
                                wsum += w;
                            }
                        }
                        if wsum > 1e-12 {
                            for v in &mut pred16 {
                                *v /= wsum;
                            }
                        }
                        let le = log_rel(clamp_err(&pred16, &cm.ep.ys[HORIZON - 1])).clamp(-10.0, 2.3);
                        errs.push(le);
                        if cm.diags[bi].oot1_log > -10.0 {
                            non_aff.push(le);
                        }
                    }
                    let obj = if non_aff.is_empty() {
                        mean(&errs)
                    } else {
                        0.35 * mean(&errs) + 0.65 * (0.5 * mean(&non_aff) + 0.5 * median(non_aff))
                    };
                    cands_proxy.push((obj, fw, wr, tau, ps));
                }
            }
        }
    }
    cands_proxy.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Etapa 3b: evaluar top-4 finalistas + mejor candidato con w_refute=0.0 en inferencia 3S exacta
    let mut finalists: Vec<([f64; 6], f64, f64, f64)> = cands_proxy
        .iter()
        .take(4)
        .map(|&(_, fw, wr, tau, ps)| (fw, wr, tau, ps))
        .collect();
    if let Some(&(_, fw, wr, tau, ps)) = cands_proxy.iter().find(|c| c.2 == 0.0) {
        if !finalists.iter().any(|f| *f == (fw, wr, tau, ps)) {
            finalists.push((fw, wr, tau, ps));
        }
    }
    let cg_grid: &[f64] = if cfg.use_refuter { &[0.0, 0.15, 0.30] } else { &[0.0] };
    let mut best_obj = f64::INFINITY;
    let mut best_ctrl = ctrl.clone();
    for &(fw, wr, tau, ps) in &finalists {
        for &cg in cg_grid {
            let mut cand = ctrl.clone();
            cand.fe_w = fw;
            cand.w_refute = wr;
            cand.base_tau = tau;
            cand.prior_scale = ps;
            cand.cycle_gain = cg;
            let o = eval_va(&cand);
            if o < best_obj {
                best_obj = o;
                best_ctrl = cand;
            }
        }
    }
    ctrl = best_ctrl;

    // 4. Auto-Calibrador Metacognitivo "Yo-FE 3S":
    //    - Etapa A-core: ridge (5e-2) sobre 10 rasgos núcleo forward en cache_tr (sin colinealidad)
    //    - Etapa A-ref: corrección no negativa en soporte epistémico `ref_meta_w` en cache_va
    //    - Etapa B: calibración Platt balanceada L2 (lam=1.5) fuera de muestra en cache_va
    ctrl.ref_meta_w = 0.0;
    let mut tr_feats = Vec::with_capacity(cache_tr.len());
    let mut tr_ylog = Vec::with_capacity(cache_tr.len());
    for cm in &cache_tr {
        let ctx_f: Vec<&[f64]> = (0..=K).map(|t| cm.ep.frame(t)).collect();
        let rep = infer_3s_from_diags(
            cfg,
            &ctrl,
            &cm.ep,
            &ctx_f,
            cm.fe_null,
            &cm.inv,
            &cm.diags,
            cm.refuter.as_ref(),
            K,
        );
        let err = clamp_err(&rep.preds[HORIZON - 1], &cm.ep.ys[HORIZON - 1]);
        tr_feats.push(rep.self_feats.to_vec());
        tr_ylog.push(vec![log_rel(err).clamp(-7.5, 2.3)]);
    }
    let ew = ridge(&tr_feats, &tr_ylog, 5e-2);
    for i in 0..N_SELF_3S {
        ctrl.err_w[i] = ew[i][0];
    }

    let mut va_cfeats = Vec::with_capacity(cache_va.len());
    let mut va_yok = Vec::with_capacity(cache_va.len());
    let mut va_sw = Vec::with_capacity(cache_va.len());
    let mut va_core_hat = Vec::with_capacity(cache_va.len());
    let mut va_ref_ex = Vec::with_capacity(cache_va.len());
    let mut va_ylog = Vec::with_capacity(cache_va.len());
    for cm in &cache_va {
        let ctx_f: Vec<&[f64]> = (0..=K).map(|t| cm.ep.frame(t)).collect();
        let rep = infer_3s_from_diags(
            cfg,
            &ctrl,
            &cm.ep,
            &ctx_f,
            cm.fe_null,
            &cm.inv,
            &cm.diags,
            cm.refuter.as_ref(),
            K,
        );
        let err = clamp_err(&rep.preds[HORIZON - 1], &cm.ep.ys[HORIZON - 1]);
        let is_trivial_affine = cm.diags[rep.best_hyp].oot1_log < -10.0;
        let sw = if is_trivial_affine { 0.35 } else { 1.0 };
        va_cfeats.push(rep.cal_feats);
        va_yok.push(if err < mt::REL_ERR_OK { 1.0 } else { 0.0 });
        va_sw.push(sw);
        va_core_hat.push(rep.err_log_hat);
        va_ref_ex.push(rep.in_sup_ref_excess);
        va_ylog.push(log_rel(err).clamp(-7.5, 2.3));
    }

    // Etapa A-ref: pendiente no negativa L2-regularizada para exceso de refutación en soporte (cache_va)
    if cfg.use_refuter {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..va_ylog.len() {
            let r = va_ref_ex[i];
            if r > 0.0 {
                num += va_sw[i] * (va_ylog[i] - va_core_hat[i]) * r;
                den += va_sw[i] * r * r;
            }
        }
        ctrl.ref_meta_w = (num / (den + 4.0)).clamp(0.0, 2.0);
        if ctrl.ref_meta_w > 0.0 {
            for i in 0..va_cfeats.len() {
                let e_hat = (va_core_hat[i] + ctrl.ref_meta_w * va_ref_ex[i]).clamp(-9.0, 4.0);
                let margin = (mt::REL_ERR_OK.ln() - e_hat).clamp(-6.0, 6.0);
                va_cfeats[i][1] = margin;
                va_cfeats[i][2] = margin.tanh();
            }
        }
    }

    let mut cw = [0.0_f64; N_CAL_3S];
    let lam_c = 1.5;
    for _ in 0..25 {
        let mut h = vec![0.0; N_CAL_3S * N_CAL_3S];
        let mut g = vec![0.0; N_CAL_3S];
        for ((x, &y), &sw) in va_cfeats.iter().zip(&va_yok).zip(&va_sw) {
            let z: f64 = cw.iter().zip(x).map(|(a, b)| a * b).sum();
            let p = sig(z.clamp(-25.0, 25.0));
            let w = sw * (p * (1.0 - p)).max(1e-6);
            for i in 0..N_CAL_3S {
                g[i] += sw * (p - y) * x[i];
                for j in 0..N_CAL_3S {
                    h[i * N_CAL_3S + j] += w * x[i] * x[j];
                }
            }
        }
        for i in 0..N_CAL_3S {
            let reg = if i == 0 { 1e-4 } else { lam_c };
            g[i] += reg * cw[i];
            h[i * N_CAL_3S + i] += reg;
        }
        let step = solve_spd(&h, &g, N_CAL_3S);
        for i in 0..N_CAL_3S {
            cw[i] -= step[i];
        }
    }
    ctrl.cal_w = cw;

    // 5. Prior autónomo de arranque en frío 1-paso (solo ds.train, 0 consultas TEST)
    train_cold_start_prior(&mut ctrl.base_cold, seed);

    let meta_info = json!({
        "n_train": n_train,
        "n_val": n_val,
        "syn_rejected": rej1 + rej2,
        "best_val_obj": best_obj,
        "fe_w": ctrl.fe_w,
        "w_refute": ctrl.w_refute,
        "cycle_gain": ctrl.cycle_gain,
        "base_tau": ctrl.base_tau,
        "prior_scale": ctrl.prior_scale,
        "ref_meta_w": ctrl.ref_meta_w,
        "buffer_len": ctrl.buffer_len,
        "cold_trained": ctrl.base_cold.cold_trained,
    });
    (ctrl, meta_info)
}

// ─── Evaluación Canónica + E33 ───────────────────────────────────────────────

fn eval_metafe3s_set(
    cfg: &MetaFe3SCfg,
    ctrl: &Meta3SController,
    eps: &[Ep],
    split_known_new: bool,
) -> (serde_json::Value, serde_json::Value, bool) {
    let mut res = serde_json::Map::new();
    let mut meta_j = serde_json::Map::new();
    let mut any_nan = false;

    let groups: Vec<(&str, Vec<&Ep>)> = if split_known_new {
        vec![
            ("known", eps.iter().filter(|e| e.fam < N_FAM).collect()),
            ("new", eps.iter().filter(|e| e.fam >= N_FAM).collect()),
        ]
    } else {
        vec![("syn_ood", eps.iter().collect())]
    };

    let mut all_probs = vec![];
    let mut all_ok = vec![];
    let mut all_err_hat = vec![];
    let mut all_err_true = vec![];
    let mut all_err_true_c = vec![];
    let mut all_fe_null = vec![];
    let mut all_fe_post = vec![];
    let mut all_fe_red = vec![];
    let mut all_ref = vec![];
    let mut all_fast_used = 0usize;
    let mut all_cyc_acc = 0usize;
    let mut per_fam: std::collections::BTreeMap<usize, Vec<f64>> = Default::default();

    for (name, subset) in groups {
        if subset.is_empty() {
            continue;
        }
        let mut errs = Vec::with_capacity(subset.len());
        let mut probs = Vec::with_capacity(subset.len());
        let mut oks = Vec::with_capacity(subset.len());
        let mut fe_reds = Vec::with_capacity(subset.len());
        let mut g_ehat = Vec::with_capacity(subset.len());
        let mut g_etrue = Vec::with_capacity(subset.len());
        let mut g_ref = Vec::with_capacity(subset.len());
        let mut g_alpha = Vec::with_capacity(subset.len());
        let mut g_beta = Vec::with_capacity(subset.len());
        let mut g_fast_n = 0usize;
        let mut group_hyps = [0usize; N_HYP];

        for e in &subset {
            let rep = run_metafe3s_episode(cfg, ctrl, e, K);
            any_nan |= rep.nan;
            let err = clamp_err(&rep.preds[HORIZON - 1], &e.ys[HORIZON - 1]);
            let ok = err < mt::REL_ERR_OK;
            let le = log_rel(err);
            let lec = le.clamp(-7.5, 2.3);
            errs.push(err);
            probs.push(rep.p_ok_hat);
            oks.push(ok);
            fe_reds.push(rep.fe_reduction);
            g_ehat.push(rep.err_log_hat);
            g_etrue.push(lec);
            g_ref.push(rep.mean_refute);
            g_alpha.push(rep.mean_alpha);
            g_beta.push(rep.mean_beta);
            if rep.fast_used {
                g_fast_n += 1;
                all_fast_used += 1;
            }
            all_cyc_acc += rep.cycle_accepted;
            group_hyps[rep.best_hyp] += 1;
            per_fam.entry(e.fam).or_default().push(le);

            all_probs.push(rep.p_ok_hat);
            all_ok.push(ok);
            all_err_hat.push(rep.err_log_hat);
            all_err_true.push(le);
            all_err_true_c.push(lec);
            all_fe_null.push(rep.fe_null);
            all_fe_post.push(rep.fe_post);
            all_fe_red.push(rep.fe_reduction);
            all_ref.push(rep.mean_refute);
        }

        let acc16 = oks.iter().filter(|&&b| b).count() as f64 / oks.len() as f64;
        let med_log = median(errs.iter().map(|&x| log_rel(x)).collect());
        let mean_log = mean(&errs.iter().map(|&x| log_rel(x)).collect::<Vec<_>>());
        res.insert(format!("{name}_acc16"), json!(acc16));
        res.insert(format!("{name}_med_log"), json!(med_log));
        res.insert(format!("{name}_mean_log"), json!(mean_log));

        let hyp_frac: serde_json::Map<String, serde_json::Value> = HYP_NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| group_hyps[*i] > 0)
            .map(|(i, &nm)| (nm.into(), json!(group_hyps[i] as f64 / subset.len() as f64)))
            .collect();
        meta_j.insert(
            format!("{name}_metacog"),
            json!({
                "fe_reduction_mean": mean(&fe_reds),
                "fe_reduced_frac": fe_reds.iter().filter(|&&v| v > 0.0).count() as f64 / fe_reds.len() as f64,
                "auroc": compute_auroc(&probs, &oks),
                "brier": compute_brier(&probs, &oks),
                "ece": compute_ece(&probs, &oks, 10),
                "aurc": compute_aurc(&probs, &oks),
                "err_pred_corr": pearson_corr(&g_ehat, &g_etrue),
                "err_pred_spearman": spearman_corr(&g_ehat, &g_etrue),
                "mean_refute": mean(&g_ref),
                "mean_alpha": mean(&g_alpha),
                "mean_beta": mean(&g_beta),
                "fast_used_frac": g_fast_n as f64 / subset.len() as f64,
                "hyp_selection": hyp_frac,
            }),
        );
    }

    let fam_med: serde_json::Map<String, serde_json::Value> = per_fam
        .into_iter()
        .map(|(f, v)| (format!("fam_{f}"), json!(median(v))))
        .collect();

    meta_j.insert(
        "overall".into(),
        json!({
            "fe_null_mean": mean(&all_fe_null),
            "fe_post_mean": mean(&all_fe_post),
            "fe_reduction_mean": mean(&all_fe_red),
            "fe_reduced_frac": all_fe_red.iter().filter(|&&v| v > 0.0).count() as f64 / all_fe_red.len().max(1) as f64,
            "auroc": compute_auroc(&all_probs, &all_ok),
            "brier": compute_brier(&all_probs, &all_ok),
            "ece": compute_ece(&all_probs, &all_ok, 10),
            "aurc": compute_aurc(&all_probs, &all_ok),
            "err_pred_corr": pearson_corr(&all_err_hat, &all_err_true_c),
            "err_pred_raw_corr": pearson_corr(&all_err_hat, &all_err_true),
            "err_pred_spearman": spearman_corr(&all_err_hat, &all_err_true),
            "mean_refute": mean(&all_ref),
            "fast_used_frac": all_fast_used as f64 / all_ok.len().max(1) as f64,
            "cycle_accepted_total": all_cyc_acc,
            "per_fam_med_log": fam_med,
        }),
    );

    (serde_json::Value::Object(res), serde_json::Value::Object(meta_j), any_nan)
}

fn e33_metafe3s(
    cfg: &MetaFe3SCfg,
    ctrl: &Meta3SController,
    d: &Orbita,
    ex: &[Example],
) -> (serde_json::Value, serde_json::Value, serde_json::Value) {
    let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex
        .iter()
        .map(|e| {
            let ys: Vec<Vec<f64>> = (1..=HORIZON).map(|h| target_h(e, h)).collect();
            let ep = Ep {
                fam: e.family_id,
                p: [0.0; 2],
                x0: e.input.clone(),
                ys,
            };
            (run_metafe3s_episode(cfg, ctrl, &ep, K).preds, e)
        })
        .collect();
    let raw = e33_score(&trs);
    let trs_orb: Vec<(Vec<Vec<f64>>, &Example)> = trs
        .iter()
        .map(|(p, e)| {
            let mut p2 = p.clone();
            p2[0] = d.step(&e.input, &e.context);
            (p2, *e)
        })
        .collect();
    let trs_auto: Vec<(Vec<Vec<f64>>, &Example)> = trs
        .iter()
        .map(|(p, e)| {
            let mut p2 = p.clone();
            if ctrl.base_cold.cold_trained {
                p2[0] = cold_start_step(&ctrl.base_cold, &e.input, &e.context);
            }
            (p2, *e)
        })
        .collect();
    (e33_score(&trs_orb), e33_score(&trs_auto), raw)
}

// ─── Benchmark No-Estacionario Online: Cambio de Régimen (RS-A y RS-B) ───────

#[derive(Clone)]
pub struct ShiftEp {
    pub ep: Ep,
    pub t_shift: usize,
    pub k_obs_total: usize,
    pub is_ood_post: bool,
}

/// Genera episodios con cambio abrupto de ley a mitad del episodio:
///   frames 0..=t_shift siguen Ley A; frames (t_shift+1)..=16 cambian a Ley B.
///   El sistema observa 0..=k_obs_total (t_shift=5, k_obs_total=8 -> 3 transiciones post-cambio)
///   y debe predecir hasta t=16.
pub fn gen_regime_shift_eps(seed: u64, n_known: usize, n_ood: usize) -> Vec<ShiftEp> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_3S_SHIFT);
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let t_shift = 5usize;
    let k_obs_total = 8usize;
    let mut out = Vec::with_capacity(n_known + n_ood);

    for idx in 0..(n_known + n_ood) {
        let is_ood = idx >= n_known;
        let fam_a = idx % N_FAM;
        let fam_b = if is_ood {
            NEW_FAMS[(idx - n_known) % NEW_FAMS.len()]
        } else {
            (fam_a + 1 + rng.gen_range(0..(N_FAM - 1))) % N_FAM
        };
        let pa = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        let pb = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        let ci = rng.gen_range(0..3);
        let x0 = make_object(&mut rng, K_POINTS, cls[ci]);
        let mut ys = Vec::with_capacity(HORIZON);
        let mut y = x0.clone();
        for t in 1..=HORIZON {
            y = if t <= t_shift {
                rule(fam_a, pa, &y)
            } else {
                rule(fam_b, pb, &y)
            };
            ys.push(y.clone());
        }
        out.push(ShiftEp {
            ep: Ep { fam: fam_b, p: pb, x0, ys },
            t_shift,
            k_obs_total,
            is_ood_post: is_ood,
        });
    }
    out
}

/// Genera episodios RS-B (W_0 congelado en Ley A tras t=4; deriva/perturbación suave a partir de t=5;
/// se observan cuadros 5..=7 y el sistema SOLO puede adaptar pesos rápidos F_t sin re-ajustar W_0).
pub fn gen_frozen_w0_perturb_eps(seed: u64, n: usize) -> Vec<ShiftEp> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_3S_SHIFT ^ 0xF02E);
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let t_shift = 4usize;
    let k_obs_total = 7usize;
    let mut out = Vec::with_capacity(n);

    for idx in 0..n {
        let is_ood = idx % 2 == 1;
        let fam = if is_ood {
            NEW_FAMS[(idx / 2) % NEW_FAMS.len()]
        } else {
            idx % N_FAM
        };
        let pa = [rng.gen_range(-0.8..0.8), rng.gen_range(-0.8..0.8)];
        // Perturbación física en caliente sobre la misma dinámica base: rotación extra + desplazamiento + escala
        let d_theta: f64 = rng.gen_range(0.18..0.38) * if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let d_scale: f64 = rng.gen_range(0.95..1.06);
        let d_shift = [rng.gen_range(-0.12..0.12), rng.gen_range(-0.12..0.12)];
        let ci = rng.gen_range(0..3);
        let x0 = make_object(&mut rng, K_POINTS, cls[ci]);
        let mut ys = Vec::with_capacity(HORIZON);
        let mut y = x0.clone();
        for t in 1..=HORIZON {
            let base = rule(fam, pa, &y);
            y = if t <= t_shift {
                base
            } else {
                let (c, s) = (d_theta.cos(), d_theta.sin());
                let mut v = vec![0.0; base.len()];
                for i in 0..base.len() / 2 {
                    let bx = base[2 * i];
                    let by = base[2 * i + 1];
                    v[2 * i] = d_scale * (c * bx - s * by) + d_shift[0];
                    v[2 * i + 1] = d_scale * (s * bx + c * by) + d_shift[1];
                }
                v
            };
            ys.push(y.clone());
        }
        out.push(ShiftEp {
            ep: Ep { fam, p: pa, x0, ys },
            t_shift,
            k_obs_total,
            is_ood_post: is_ood,
        });
    }
    out
}

/// Ejecuta detección online de cambio de régimen (RS-A) con ACC + MetaNet λ_t reset + F_t.
fn run_regime_shift_3s(cfg: &MetaFe3SCfg, ctrl: &Meta3SController, se: &ShiftEp) -> (Vec<f64>, bool, f64) {
    let k_tot = se.k_obs_total;
    let all_obs: Vec<&[f64]> = (0..=k_tot).map(|t| se.ep.frame(t)).collect();

    let mut win_start = 0usize;
    let mut acc = AccMonitor::new(4);
    acc.seed_frame(all_obs[0]);
    let mut s_nm = [0.0; N_GRU_H];
    let mut detected_reset = false;
    let mut min_lam_seen = 1.0_f64;

    for t in 3..=k_tot {
        if t < win_start + 3 {
            continue;
        }
        let sub = &all_obs[win_start..t];
        let h_aff = fit_hypothesis_ex(HypKind::LieAffine, sub, 2);
        let h_cm = fit_hypothesis_ex(HypKind::GpCoMoving, sub, 1);
        let p_aff = if h_aff.valid { h_aff.step_frame(all_obs[t - 1]).0 } else { all_obs[t - 1].to_vec() };
        let p_cm = if h_cm.valid { h_cm.step_frame(all_obs[t - 1]).0 } else { p_aff.clone() };
        let e_aff = clamp_err(&p_aff, all_obs[t]);
        let e_cm = clamp_err(&p_cm, all_obs[t]);
        let surp = e_aff.min(e_cm);
        let dis = clamp_err(&p_aff, &p_cm);
        let m_t = acc.observe(all_obs[t], surp, dis, 0.0);
        let sn = if cfg.use_neuromod {
            let (sn, ns) = ctrl.nm.step(m_t, s_nm);
            s_nm = ns;
            sn
        } else {
            NeuromodSignal {
                alpha: 0.25,
                lambda: 0.85,
                beta: ctrl.base_tau,
                g_mode: [0.34, 0.33, 0.33],
            }
        };
        if sn.lambda < min_lam_seen {
            min_lam_seen = sn.lambda;
        }
        // Reset contextual disparado únicamente cuando MetaNet colapsa λ_t < 0.45
        if cfg.use_neuromod && sn.lambda < 0.45 && t <= k_tot - 1 {
            win_start = t - 1;
            detected_reset = true;
            acc.seed_frame(all_obs[win_start]);
            s_nm = [0.0; N_GRU_H];
        }
    }

    let sub_k = k_tot - win_start;
    let sub_ep = Ep {
        fam: se.ep.fam,
        p: se.ep.p,
        x0: se.ep.frame(win_start).to_vec(),
        ys: (win_start + 1..=HORIZON).map(|t| se.ep.frame(t).to_vec()).collect(),
    };
    let rep = run_metafe3s_episode(cfg, ctrl, &sub_ep, sub_k);
    let pred16 = rep.preds[HORIZON - win_start - 1].clone();
    (pred16, detected_reset, min_lam_seen)
}

/// Ejecuta adaptación RS-B con W_0 ESTRICTAMENTE CONGELADO tras t=4:
/// solo los pesos rápidos F_t (y trazas 3-factores E_t gobernadas por MetaNet) pueden adaptarse en t=5..=7.
fn run_frozen_w0_perturb_3s(cfg: &MetaFe3SCfg, ctrl: &Meta3SController, se: &ShiftEp) -> Vec<f64> {
    // 1. Ajustar W_0 SOLO en la fase pre-perturbación 0..=t_shift y CONGELARLO permanentemente
    let pre_frames: Vec<&[f64]> = (0..=se.t_shift).map(|t| se.ep.frame(t)).collect();
    let hyps = all_hyps();
    let diags: Vec<HypDiag> = if cfg.is_aura_v3_genesis {
        evaluate_all_hypotheses_genesis(&hyps, &pre_frames, se.t_shift).0
    } else {
        hyps.iter()
            .map(|&hk| evaluate_hypothesis(hk, &pre_frames, true, true, se.t_shift))
            .collect()
    };
    let inv = episode_invariants(&pre_frames);
    let scores: Vec<f64> = diags
        .iter()
        .map(|d| {
            if !d.hyp.valid {
                return 1e6;
            }
            let lp: f64 = ctrl.prior_w[d.hyp.kind as usize].iter().zip(&inv).map(|(a, b)| a * b).sum();
            ctrl.fe_w[0] * d.vfe_per_pt
                + ctrl.fe_w[1] * d.loo_per_pt
                + ctrl.fe_w[2] * d.oot1_log
                + ctrl.fe_w[3] * d.oot2_log
                + ctrl.fe_w[4] * d.replay_log
                + ctrl.fe_w[5] * d.stab_pen
                - (ctrl.prior_scale * lp).clamp(-2.0, 2.0)
        })
        .collect();
    let w_frozen = softmax_scores(&scores, ctrl.base_tau);

    // 2. Fase de perturbación online t = (t_shift+1)..=k_obs_total: W_0 congelado; solo F_t se adapta
    let obs_frames: Vec<&[f64]> = (0..=se.k_obs_total).map(|t| se.ep.frame(t)).collect();
    let mut fw = FastWeights::new(&obs_frames, 2.5);
    if cfg.use_fast_weights {
        let mut tr = EligibilityTrace::new(0.55);
        let mut acc = AccMonitor::new(4);
        acc.seed_frame(se.ep.frame(se.t_shift));
        let mut s_nm = [0.0; N_GRU_H];

        let mut base_steps = Vec::new();
        for t in se.t_shift..se.k_obs_total {
            let (bn, _, dis) = blend_step(&diags, &w_frozen, se.ep.frame(t));
            base_steps.push((se.ep.frame(t), bn, se.ep.frame(t + 1), dis));
        }

        let n_passes = 45usize;
        for pass in 0..n_passes {
            for &(prev, ref bn, tgt, dis) in &base_steps {
                let pred = fw.apply_frame(prev, bn);
                let err = clamp_err(&pred, tgt);
                let m_t = acc.observe(tgt, err, dis, fw.saturation());
                let sn = if cfg.use_neuromod {
                    let (sn, ns) = ctrl.nm.step(m_t, s_nm);
                    s_nm = ns;
                    sn
                } else {
                    NeuromodSignal {
                        alpha: 0.25,
                        lambda: 0.85,
                        beta: ctrl.base_tau,
                        g_mode: [0.34, 0.33, 0.33],
                    }
                };
                // Con MetaNet: noradrenalina α_t alta abre plasticidad y λ_t limpia en el primer paso y retiene después;
                // Sin MetaNet (NoNM): α=0.25 y λ=0.85 constantes generan decaimiento prematuro
                let (alpha_eff, lam_eff) = if cfg.use_neuromod {
                    let a = (0.35 + 0.65 * sn.alpha).clamp(0.15, 0.95);
                    let l = if pass == 0 && sn.lambda < 0.45 {
                        sn.lambda
                    } else {
                        0.999
                    };
                    (a, l)
                } else {
                    (sn.alpha, sn.lambda)
                };
                tr.update_and_consolidate(&mut fw, prev, bn, tgt, alpha_eff, lam_eff, 1.0);
            }
        }
    }

    // 3. Rollout desde k_obs_total hasta t=16 usando W_0 congelado + F_t
    let mut cur = se.ep.frame(se.k_obs_total).to_vec();
    for _ in (se.k_obs_total + 1)..=HORIZON {
        let (bn, _, _) = blend_step(&diags, &w_frozen, &cur);
        cur = fw.apply_frame(&cur, &bn);
        if !fin(&cur) {
            break;
        }
    }
    cur
}

fn eval_regime_shift_suite(
    arm_name: &str,
    cfg3s: Option<&MetaFe3SCfg>,
    ctrl3s: Option<&Meta3SController>,
    cfg_v2: Option<&MetaFeCfg>,
    ctrl_v2: Option<&MetaController>,
    rs_a: &[ShiftEp],
    rs_b: &[ShiftEp],
) -> serde_json::Value {
    let mut a_known = Vec::new();
    let mut a_ood = Vec::new();
    let mut a_known_ok = 0usize;
    let mut a_ood_ok = 0usize;
    let mut a_det = 0usize;
    let mut a_min_lam = Vec::new();

    for se in rs_a {
        let (pred16, det, mlam) = if let (Some(c), Some(ct)) = (cfg3s, ctrl3s) {
            run_regime_shift_3s(c, ct, se)
        } else {
            // METAFE-v2 estático: observa 0..=k_obs_total sin detector ACC ni reset λ_t
            let rep = run_metafe_episode(cfg_v2.unwrap(), ctrl_v2.unwrap(), &se.ep, se.k_obs_total);
            (rep.preds[HORIZON - 1].clone(), false, 1.0)
        };
        let err = clamp_err(&pred16, &se.ep.ys[HORIZON - 1]);
        let le = log_rel(err);
        let ok = err < mt::REL_ERR_OK;
        if det {
            a_det += 1;
        }
        a_min_lam.push(mlam);
        if se.is_ood_post {
            a_ood.push(le);
            if ok { a_ood_ok += 1; }
        } else {
            a_known.push(le);
            if ok { a_known_ok += 1; }
        }
    }

    let mut b_errs = Vec::with_capacity(rs_b.len());
    let mut b_ok = 0usize;
    for se in rs_b {
        let pred16 = if let (Some(c), Some(ct)) = (cfg3s, ctrl3s) {
            run_frozen_w0_perturb_3s(c, ct, se)
        } else {
            // METAFE-v2 con W_0 congelado en 0..=t_shift (sin F_t): no puede adaptarse a perturbación
            let rep = run_metafe_episode(cfg_v2.unwrap(), ctrl_v2.unwrap(), &se.ep, se.t_shift);
            rep.preds[HORIZON - 1].clone()
        };
        let err = clamp_err(&pred16, &se.ep.ys[HORIZON - 1]);
        b_errs.push(log_rel(err));
        if err < mt::REL_ERR_OK {
            b_ok += 1;
        }
    }

    let _ = arm_name;
    json!({
        "rs_a_known_med_log": median(a_known.clone()),
        "rs_a_known_acc16": a_known_ok as f64 / a_known.len().max(1) as f64,
        "rs_a_ood_med_log": median(a_ood.clone()),
        "rs_a_ood_acc16": a_ood_ok as f64 / a_ood.len().max(1) as f64,
        "rs_a_all_med_log": median(a_known.iter().chain(&a_ood).copied().collect()),
        "rs_a_reset_rate": a_det as f64 / rs_a.len().max(1) as f64,
        "rs_a_min_lambda_mean": mean(&a_min_lam),
        "rs_b_frozen_w0_med_log": median(b_errs.clone()),
        "rs_b_frozen_w0_mean_log": mean(&b_errs),
        "rs_b_frozen_w0_acc16": b_ok as f64 / b_errs.len().max(1) as f64,
    })
}

// ─── Benchmark de Consolidación Offline Continua (Módulo 5: EWC vs NoEWC) ───

fn gen_task_eps(seed: u64, xor: u64, fams: &[usize], n: usize, synthetic: bool) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ xor);
    let mut prng = Xoshiro256StarStar::seed_from_u64(seed ^ xor ^ 0x5A11);
    let probe = make_object(&mut prng, K_POINTS, ObjectClass::Blob);
    let mut rej = 0usize;
    let eps: Vec<Ep> = (0..n)
        .map(|i| {
            if synthetic {
                make_syn_ep(&mut rng, &probe, &mut rej).0
            } else {
                make_ep(&mut rng, fams[i % fams.len()])
            }
        })
        .collect();
    for e in &eps {
        assert!(!NEW_FAMS.contains(&e.fam), "fuga detectada en consolidación EWC");
    }
    eps
}

/// Evalúa olvido catastrófico en consolidación offline secuencial sobre 3 tareas (0 fuga {8,10,11}):
///   Tarea 1: familias {0,1,2,3} + mitad sintética radial
///   Tarea 2: familias {4,5,6,7}
///   Tarea 3: familias sintéticas procedurales auditadas
/// Compara `EWC` (μ=25.0) vs `NoEWC` (μ=0.0) a partir del mismo estado consolidado W_0^{(1)}.
pub fn eval_continual_ewc(seed: u64) -> serde_json::Value {
    // Entrenamiento y prueba separados por tarea (sin fuga {8,10,11}, 0 consultas TEST)
    let t1_tr = build_cache_3s(&gen_task_eps(seed, XOR_3S_CONT ^ 0x11, &[0, 1, 2, 3, 7], 36, false), false);
    let t1_te = build_cache_3s(&gen_task_eps(seed, XOR_3S_CONT ^ 0x12, &[0, 1, 2, 3, 7], 24, false), false);
    let t2_tr = build_cache_3s(&gen_task_eps(seed, XOR_3S_CONT ^ 0x21, &[4, 5, 6, 7], 36, false), false);
    let t2_te = build_cache_3s(&gen_task_eps(seed, XOR_3S_CONT ^ 0x22, &[4, 5, 6, 7], 24, false), false);
    let t3_tr = build_cache_3s(&gen_task_eps(seed, XOR_3S_CONT ^ 0x31, &[], 36, true), false);
    let t3_te = build_cache_3s(&gen_task_eps(seed, XOR_3S_CONT ^ 0x32, &[], 24, true), false);

    let e1_tr = cache_to_buffer_entries(&t1_tr, None);
    let e1_te = cache_to_buffer_entries(&t1_te, None);
    let e2_tr = cache_to_buffer_entries(&t2_tr, None);
    let e2_te = cache_to_buffer_entries(&t2_te, None);
    let e3_tr = cache_to_buffer_entries(&t3_tr, None);
    let e3_te = cache_to_buffer_entries(&t3_te, None);

    // Consolidación Fase 1 (común a EWC y NoEWC)
    let mut w_t1 = [[0.0; N_INV]; N_HYP];
    let ewc0 = EwcPriorState::default();
    consolidate_prior_ewc(&mut w_t1, &e1_tr, &ewc0, 0.0, 90, 0.12, 0.01);
    let l1_after1 = prior_ce_loss(&w_t1, &e1_te);
    let a1_after1 = prior_top1_acc(&w_t1, &e1_te);

    // Rama EWC: acumula Fisher Ω después de cada fase
    let mut w_ewc = w_t1;
    let mut ewc = EwcPriorState {
        w_prev: w_ewc,
        omega: compute_fisher_diag(&w_ewc, &e1_tr),
        initialized: true,
    };
    consolidate_prior_ewc(&mut w_ewc, &e2_tr, &ewc, 25.0, 90, 0.12, 0.01);
    let f2 = compute_fisher_diag(&w_ewc, &e2_tr);
    ewc.w_prev = w_ewc;
    for h in 0..N_HYP {
        for j in 0..N_INV {
            ewc.omega[h][j] += f2[h][j];
        }
    }
    consolidate_prior_ewc(&mut w_ewc, &e3_tr, &ewc, 25.0, 90, 0.12, 0.01);

    // Rama NoEWC: μ = 0 (re-escritura libre del núcleo lento sin protección Fisher)
    let mut w_no = w_t1;
    consolidate_prior_ewc(&mut w_no, &e2_tr, &ewc0, 0.0, 90, 0.12, 0.01);
    consolidate_prior_ewc(&mut w_no, &e3_tr, &ewc0, 0.0, 90, 0.12, 0.01);

    let l1_ewc = prior_ce_loss(&w_ewc, &e1_te);
    let l2_ewc = prior_ce_loss(&w_ewc, &e2_te);
    let l3_ewc = prior_ce_loss(&w_ewc, &e3_te);
    let a1_ewc = prior_top1_acc(&w_ewc, &e1_te);
    let a2_ewc = prior_top1_acc(&w_ewc, &e2_te);
    let a3_ewc = prior_top1_acc(&w_ewc, &e3_te);

    let l1_no = prior_ce_loss(&w_no, &e1_te);
    let l2_no = prior_ce_loss(&w_no, &e2_te);
    let l3_no = prior_ce_loss(&w_no, &e3_te);
    let a1_no = prior_top1_acc(&w_no, &e1_te);
    let a2_no = prior_top1_acc(&w_no, &e2_te);
    let a3_no = prior_top1_acc(&w_no, &e3_te);

    let forg_ewc = l1_ewc - l1_after1;
    let forg_no = l1_no - l1_after1;
    let avg_loss_ewc = (l1_ewc + l2_ewc + l3_ewc) / 3.0;
    let avg_loss_no = (l1_no + l2_no + l3_no) / 3.0;

    json!({
        "t1_ce_after_t1": l1_after1,
        "t1_acc_after_t1": a1_after1,
        "ewc": {
            "t1_ce": l1_ewc, "t2_ce": l2_ewc, "t3_ce": l3_ewc,
            "avg_ce": avg_loss_ewc,
            "t1_acc": a1_ewc, "t2_acc": a2_ewc, "t3_acc": a3_ewc,
            "avg_acc": (a1_ewc + a2_ewc + a3_ewc) / 3.0,
            "t1_forgetting_ce": forg_ewc,
            "t1_acc_drop": a1_after1 - a1_ewc,
        },
        "no_ewc": {
            "t1_ce": l1_no, "t2_ce": l2_no, "t3_ce": l3_no,
            "avg_ce": avg_loss_no,
            "t1_acc": a1_no, "t2_acc": a2_no, "t3_acc": a3_no,
            "avg_acc": (a1_no + a2_no + a3_no) / 3.0,
            "t1_forgetting_ce": forg_no,
            "t1_acc_drop": a1_after1 - a1_no,
        },
        "delta_forgetting_ewc_minus_noewc": forg_ewc - forg_no,
        "delta_avg_ce_ewc_minus_noewc": avg_loss_ewc - avg_loss_no,
        "ewc_wins_forgetting": forg_ewc < forg_no,
    })
}

// ─── CLI: metafe3s-paso0, metafe3s-run, metafe3s-diag ────────────────────────

pub fn paso0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC4B0".into()));
    let mut res = serde_json::Map::new();

    // (a) Determinismo bit a bit
    let cfg = metafe3s_arms().into_iter().find(|c| c.name == "METAFE-3S").unwrap();
    let (c1, _) = train_metafe3s_controller(&cfg, seed, 32, 16);
    let (c2, _) = train_metafe3s_controller(&cfg, seed, 32, 16);
    let held = gen_heldout(seed, 4);
    let mut mismatch = 0usize;
    let mut total_vals = 0usize;
    let mut fe_red_all = true;
    let mut max_fnorm = 0.0_f64;
    for e in &held {
        let r1 = run_metafe3s_episode(&cfg, &c1, e, K);
        let r2 = run_metafe3s_episode(&cfg, &c2, e, K);
        let b1 = atlas::bits(&r1.preds);
        let b2 = atlas::bits(&r2.preds);
        total_vals += b1.len();
        mismatch += b1.iter().zip(&b2).filter(|(x, y)| x != y).count();
        if r1.fe_reduction <= 0.0 {
            fe_red_all = false;
        }
        if r1.fast_norm > max_fnorm {
            max_fnorm = r1.fast_norm;
        }
    }
    res.insert(
        "a_deterministic".into(),
        json!({"values": total_vals, "mismatch": mismatch, "pass": mismatch == 0 && total_vals > 0}),
    );

    // (b) Cero fuga de familias retenidas {8,10,11}
    let (eps, rej) = gen_meta_training_eps(seed, 128, XOR_METAFE_TRAIN);
    let no_leak = eps.iter().all(|e| !NEW_FAMS.contains(&e.fam));
    res.insert(
        "b_no_heldout_leak".into(),
        json!({"pass": no_leak, "n_eps": eps.len(), "syn_rejected": rej}),
    );

    // (c) Cero consultas TEST
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let mut tmp_c = MetaController::default();
    train_cold_start_prior(&mut tmp_c, seed);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    res.insert("c_zero_test_queries".into(), json!({"pass": q == 0, "queries": q}));

    // (d) Exactitud en familias afines conocidas + cota espectral estricta ‖F_t‖_F ≤ ρ
    let max_aff_err = held
        .iter()
        .filter(|e| e.fam < 7)
        .map(|e| {
            let r = run_metafe3s_episode(&cfg, &c1, e, K);
            clamp_err(&r.preds[HORIZON - 1], &e.ys[HORIZON - 1])
        })
        .fold(0.0, f64::max);
    res.insert(
        "d_affine_exact_and_bounded".into(),
        json!({
            "max_rel_err": max_aff_err,
            "max_fast_norm": max_fnorm,
            "fe_reduced_all": fe_red_all,
            "pass": max_aff_err < 1e-4 && max_fnorm <= 0.25 + 1e-9
        }),
    );

    // (e) Verificación EWC reduce olvido frente a NoEWC
    let ewc_j = eval_continual_ewc(seed);
    let ewc_ok = ewc_j["ewc_wins_forgetting"].as_bool().unwrap_or(false);
    res.insert(
        "e_ewc_forgetting_reduction".into(),
        json!({
            "delta_forgetting": ewc_j["delta_forgetting_ewc_minus_noewc"],
            "pass": ewc_ok
        }),
    );

    let ok = res.values().all(|v| v["pass"] == true);
    println!(
        "{}",
        json!({"seed": format!("0x{seed:X}"), "git_sha": git_sha(), "paso0_ok": ok, "res": res})
    );
}

pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap());
    let phase = arg_of(a, "--phase").unwrap();
    assert!(seeds(&phase).contains(&seed), "semilla fuera de rango para fase {phase}");
    let wpath = arg_of(a, "--weights").unwrap();
    let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap());
    let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase);
    std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done"));
    if done.exists() && !a.iter().any(|x| x == "--force") {
        eprintln!("skip 0x{seed:X}");
        return;
    }
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    let n_tr: usize = arg_of(a, "--ntrain").map(|v| v.parse().unwrap()).unwrap_or(128);
    let n_va: usize = arg_of(a, "--nval").map(|v| v.parse().unwrap()).unwrap_or(96);

    let t0 = std::time::Instant::now();
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let held = gen_heldout(seed, HPF);
    let syn_ood = gen_ood_synthetic_eps(seed, 48);
    let rs_a = gen_regime_shift_eps(seed, 24, 24);
    let rs_b = gen_frozen_w0_perturb_eps(seed, 24);
    let (ex, sha) = e33_examples(seed);

    let mut armsj = serde_json::Map::new();

    // Brazo 0: METAFE-v2 congelado (línea base exacta emparejada en la misma semilla)
    let run_v2 = only.as_ref().map(|o| o.iter().any(|x| x == "METAFE-v2")).unwrap_or(true);
    if run_v2 {
        let ta = std::time::Instant::now();
        let cfg_v2 = metafe_arms().into_iter().find(|c| c.name == "METAFE").unwrap();
        let (ctrl_v2, train_meta) = train_metafe_controller(&cfg_v2, seed, n_tr, n_va);
        let train_s = ta.elapsed().as_secs_f64();

        let te = std::time::Instant::now();
        let (held_j, metacog_j, nan_h) = eval_metafe_set(&cfg_v2, &ctrl_v2, &held, true);
        let (syn_j, syn_meta_j, nan_s) = eval_metafe_set(&cfg_v2, &ctrl_v2, &syn_ood, false);
        let (e33_orb, e33_auto, e33_raw) = e33_metafe(&cfg_v2, &ctrl_v2, &d, &ex);
        let shift_j = eval_regime_shift_suite(
            "METAFE-v2",
            None,
            None,
            Some(&cfg_v2),
            Some(&ctrl_v2),
            &rs_a,
            &rs_b,
        );
        let eval_s = te.elapsed().as_secs_f64();

        armsj.insert(
            "METAFE-v2".into(),
            json!({
                "held": held_j,
                "syn_ood": syn_j,
                "metacog": metacog_j,
                "syn_metacog": syn_meta_j,
                "shift": shift_j,
                "e33": e33_orb,
                "e33_autonomous": e33_auto,
                "e33_raw": e33_raw,
                "nan": nan_h || nan_s,
                "train_meta": train_meta,
                "train_s": train_s,
                "eval_s": eval_s,
                "wall_s": ta.elapsed().as_secs_f64(),
            }),
        );
        eprintln!(
            "0x{seed:X} {:<18} new_med={:+.4} new_acc={:.4} syn_med={:+.4} rsA={:+.4} rsB={:+.4} {:.2}s",
            "METAFE-v2",
            held_j["new_med_log"].as_f64().unwrap_or(f64::NAN),
            held_j["new_acc16"].as_f64().unwrap_or(f64::NAN),
            syn_j["syn_ood_med_log"].as_f64().unwrap_or(f64::NAN),
            shift_j["rs_a_all_med_log"].as_f64().unwrap_or(f64::NAN),
            shift_j["rs_b_frozen_w0_med_log"].as_f64().unwrap_or(f64::NAN),
            ta.elapsed().as_secs_f64()
        );
    }

    // Brazos 1..4: METAFE-3S y ablaciones causales (NoFast, NoRefute, NoNM)
    for cfg in metafe3s_arms() {
        if let Some(o) = &only {
            if !o.iter().any(|x| x == cfg.name) {
                continue;
            }
        }
        let ta = std::time::Instant::now();
        let (ctrl, train_meta) = train_metafe3s_controller(&cfg, seed, n_tr, n_va);
        let train_s = ta.elapsed().as_secs_f64();

        let te = std::time::Instant::now();
        let (held_j, metacog_j, nan_h) = eval_metafe3s_set(&cfg, &ctrl, &held, true);
        let (syn_j, syn_meta_j, nan_s) = eval_metafe3s_set(&cfg, &ctrl, &syn_ood, false);
        let (e33_orb, e33_auto, e33_raw) = e33_metafe3s(&cfg, &ctrl, &d, &ex);
        let shift_j = eval_regime_shift_suite(
            cfg.name,
            Some(&cfg),
            Some(&ctrl),
            None,
            None,
            &rs_a,
            &rs_b,
        );
        let eval_s = te.elapsed().as_secs_f64();

        armsj.insert(
            cfg.name.into(),
            json!({
                "held": held_j,
                "syn_ood": syn_j,
                "metacog": metacog_j,
                "syn_metacog": syn_meta_j,
                "shift": shift_j,
                "e33": e33_orb,
                "e33_autonomous": e33_auto,
                "e33_raw": e33_raw,
                "nan": nan_h || nan_s,
                "train_meta": train_meta,
                "train_s": train_s,
                "eval_s": eval_s,
                "wall_s": ta.elapsed().as_secs_f64(),
            }),
        );
        eprintln!(
            "0x{seed:X} {:<18} new_med={:+.4} new_acc={:.4} syn_med={:+.4} rsA={:+.4} rsB={:+.4} {:.2}s",
            cfg.name,
            held_j["new_med_log"].as_f64().unwrap_or(f64::NAN),
            held_j["new_acc16"].as_f64().unwrap_or(f64::NAN),
            syn_j["syn_ood_med_log"].as_f64().unwrap_or(f64::NAN),
            shift_j["rs_a_all_med_log"].as_f64().unwrap_or(f64::NAN),
            shift_j["rs_b_frozen_w0_med_log"].as_f64().unwrap_or(f64::NAN),
            ta.elapsed().as_secs_f64()
        );
    }

    // Experimento Módulo 5: Consolidación Continua Offline EWC vs NoEWC
    let continual_j = eval_continual_ewc(seed);

    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(d.raw, raw0);
    assert_eq!(q, 0, "consultas TEST ≠ 0");

    let r = json!({
        "seed": format!("0x{seed:X}"),
        "phase": phase,
        "weights": wpath,
        "arms": armsj,
        "continual_ewc": continual_j,
        "test_queries": q,
        "test_sha256": sha,
        "git_sha": git_sha(),
        "wall_s": t0.elapsed().as_secs_f64(),
    });
    let tmp = out.join(format!("metrics_0x{seed:X}.json.tmp"));
    std::fs::write(&tmp, serde_json::to_string_pretty(&r).unwrap()).unwrap();
    std::fs::rename(&tmp, out.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok {:.2}s", t0.elapsed().as_secs_f64());
}

/// Diagnóstico detallado en semillas de ajuste (0xC4B0..0xC4B3): compara METAFE-v2 vs METAFE-3S por familia.
pub fn diag(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC4B0".into()));
    assert!(seeds("tune").contains(&seed), "metafe3s-diag solo permitido en semillas tune 0xC4B0..0xC4B3");
    let held = gen_heldout(seed, HPF);
    let syn = gen_ood_synthetic_eps(seed, 48);
    let rs_a = gen_regime_shift_eps(seed, 24, 24);
    let rs_b = gen_frozen_w0_perturb_eps(seed, 24);

    let cfg_v2 = metafe_arms().into_iter().find(|c| c.name == "METAFE").unwrap();
    let (ctrl_v2, _) = train_metafe_controller(&cfg_v2, seed, 128, 96);
    let (h_v2, m_v2, _) = eval_metafe_set(&cfg_v2, &ctrl_v2, &held, true);
    let (s_v2, sm_v2, _) = eval_metafe_set(&cfg_v2, &ctrl_v2, &syn, false);
    let sh_v2 = eval_regime_shift_suite("METAFE-v2", None, None, Some(&cfg_v2), Some(&ctrl_v2), &rs_a, &rs_b);

    let cfg_3s = metafe3s_arms().into_iter().find(|c| c.name == "METAFE-3S").unwrap();
    let (ctrl_3s, tm_3s) = train_metafe3s_controller(&cfg_3s, seed, 128, 96);
    let (h_3s, m_3s, _) = eval_metafe3s_set(&cfg_3s, &ctrl_3s, &held, true);
    let (s_3s, sm_3s, _) = eval_metafe3s_set(&cfg_3s, &ctrl_3s, &syn, false);
    let sh_3s = eval_regime_shift_suite("METAFE-3S", Some(&cfg_3s), Some(&ctrl_3s), None, None, &rs_a, &rs_b);
    let ewc_j = eval_continual_ewc(seed);

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "seed": format!("0x{seed:X}"),
            "train_meta_3s": tm_3s,
            "v2": {
                "held": h_v2, "syn_ood": s_v2,
                "new_metacog": m_v2["new_metacog"],
                "overall_metacog": m_v2["overall"],
                "syn_metacog": sm_v2["syn_ood_metacog"],
                "shift": sh_v2,
            },
            "3s": {
                "held": h_3s, "syn_ood": s_3s,
                "new_metacog": m_3s["new_metacog"],
                "overall_metacog": m_3s["overall"],
                "syn_metacog": sm_3s["syn_ood_metacog"],
                "shift": sh_3s,
            },
            "continual_ewc": ewc_j,
        }))
        .unwrap()
    );
}

// ─── Pruebas unitarias ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_weights_spectral_ball_and_trace() {
        let ep = make_ep(&mut Xoshiro256StarStar::seed_from_u64(42), 7);
        let frames: Vec<&[f64]> = (0..=K).map(|t| ep.frame(t)).collect();
        let mut fw = FastWeights::new(&frames, 0.25);
        let mut tr = EligibilityTrace::new(0.6);
        for _ in 0..20 {
            tr.update_and_consolidate(&mut fw, frames[0], frames[0], frames[1], 0.9, 0.95, 2.0);
            assert!(fw.frob_norm() <= 0.25 + 1e-12, "violación de cota Frobenius Π_ρ");
        }
        assert!(fw.active);
        fw.reset();
        assert_eq!(fw.frob_norm(), 0.0);
        assert!(!fw.active);
    }

    #[test]
    fn test_backward_refuter_inverts_smooth_dynamics() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        let ep = make_ep(&mut rng, 1); // rotación exacta
        let frames: Vec<&[f64]> = (0..=K).map(|t| ep.frame(t)).collect();
        let rf = BackwardRefuter::fit(&frames);
        assert!(rf.valid);
        assert!(rf.reliability > 0.95, "rel={}", rf.reliability);
        let (e_cyc, _, _) = rf.refute_step(ep.frame(4), ep.frame(5));
        assert!(e_cyc < 1e-6, "e_cyc={e_cyc}");
    }

    #[test]
    fn test_acc_and_neuromod_bptt_reset_response() {
        let mut nm = NeuromodNet::init(777);
        let (eps, _) = gen_meta_training_eps(777, 16, XOR_METAFE_TRAIN);
        let cache = build_cache_3s(&eps, false);
        let seqs = build_neuromod_sequences(&cache, 777);
        nm.train_bptt(&seqs, 60, 0.35, 1e-3);

        // En estado estable (sorpresa baja): λ alto, α bajo
        let (s_calm, h1) = nm.step([-1.1, -1.0, 0.0, 0.05], [0.0; N_GRU_H]);
        // Ante conflicto agudo ACC (sorpresa alta): λ cae drásticamente, α sube
        let (s_shock, _) = nm.step([1.4, 0.8, 0.1, 0.1], h1);
        assert!(s_calm.lambda > 0.70, "calm lambda={}", s_calm.lambda);
        assert!(s_shock.lambda < 0.45, "shock lambda={}", s_shock.lambda);
        assert!(s_shock.alpha > s_calm.alpha + 0.3, "alpha calm={} shock={}", s_calm.alpha, s_shock.alpha);
    }

    #[test]
    fn test_ewc_reduces_forgetting_vs_noewc() {
        let j = eval_continual_ewc(0xC4B0);
        let f_ewc = j["ewc"]["t1_forgetting_ce"].as_f64().unwrap();
        let f_no = j["no_ewc"]["t1_forgetting_ce"].as_f64().unwrap();
        assert!(f_ewc < f_no, "EWC debe olvidar menos que NoEWC: ewc={f_ewc} vs no={f_no}");
    }

    #[test]
    fn test_no_leak_and_zero_damage_on_exact_affine() {
        let cfg = metafe3s_arms().into_iter().find(|c| c.name == "METAFE-3S").unwrap();
        let (ctrl, _) = train_metafe3s_controller(&cfg, 0xC4B0, 24, 16);
        let held = gen_heldout(0xC4B0, 2);
        for e in held.iter().filter(|e| e.fam < 7) {
            let r = run_metafe3s_episode(&cfg, &ctrl, e, K);
            let err = clamp_err(&r.preds[HORIZON - 1], &e.ys[HORIZON - 1]);
            assert!(err < 1e-6, "fam {} err={err}", e.fam);
            assert!(!r.fast_used, "F_t no debe activarse en afines exactas fam {}", e.fam);
        }
    }
}
