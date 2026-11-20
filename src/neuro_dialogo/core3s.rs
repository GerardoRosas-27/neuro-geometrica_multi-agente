//! Núcleo Cognitivo Neuro-Geométrico de Tres Escalas con Refutación Adversarial Dual (METAFE-3S-Dial).
//!
//! Resuelve el cuello de botella crítico del razonamiento abierto sin feedback externo:
//! - Si δ_t solo premia reducir entropía propia (`DeltaMode::NaiveEntropy`), los pesos rápidos
//!   colapsan rápidamente en un atractor dogmático rígido de alta confianza falsa.
//! - `DeltaMode::AdversarialCrossRefute` acopla **dos ramas con pesos rápidos independientes y
//!   ortogonales** (`F_t^A ⊥ F_t^B`, ambos en la bola Frobenius Π_ρ) sobre un núcleo lento `W_0`
//!   congelado durante inferencia, generando la señal de tres factores:
//!     δ_t^real = Δ(E_cyc) + λ_cf · S_cf(x_t, x_cf) − λ_adv · V_ref(A vs B) − λ_dogma · dogma_t + λ_sym · δ_sym
//!   junto con monitor ACC m_t ∈ R⁴, controlador recurrente GRU MetaNet (α_t, λ_t, β_t, g_t),
//!   trazas de elegibilidad E_t y consolidación offline EWC (diagonal de Fisher Ω_i) desde buffer episódico.

#![allow(clippy::needless_range_loop)]

use crate::neuro_dialogo::belief::{EpistemicStance, GroundingVerdict};
use crate::neuro_dialogo::encoder::{DialogueIntent, SemanticFrame, D_SEM, N_INTENTS};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const H_LAT: usize = 32;
pub const N_STANCES: usize = 5;
pub const N_ACC: usize = 4;
pub const N_GRU: usize = 4;
pub const N_MODES: usize = 3;

/// Modo de cálculo de la señal δ_t (permite ablaciones causales directas contra el colapso dogmático).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaMode {
    /// Solución completa: dos ramas rápidas ortogonales (F_t^A ⊥ F_t^B) + ciclo Φ_rev + contrafactual + anti-dogma.
    AdversarialCrossRefute,
    /// Control negativo exacto del cuello de botella: δ_t = −ΔH(p_t^A) (reduce entropía propia sin refutador).
    NaiveEntropy,
    /// Ablación sin rama B adversaria (solo rama A + ciclo).
    NoRefute,
    /// Ablación sin pesos rápidos (F_t ≡ 0, solo núcleo lento W_0).
    NoFast,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuromodSignal {
    pub alpha: f64,
    pub lambda: f64,
    pub beta: f64,
    pub g_mode: [f64; N_MODES],
}

/// Controlador neuromodulador recurrente GRU (Módulo 3, reutiliza diseño exacto de METAFE-3S).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuromodGru {
    pub wz: [[f64; N_ACC]; N_GRU],
    pub uz: [[f64; N_GRU]; N_GRU],
    pub bz: [f64; N_GRU],
    pub wr: [[f64; N_ACC]; N_GRU],
    pub ur: [[f64; N_GRU]; N_GRU],
    pub br: [f64; N_GRU],
    pub wh: [[f64; N_ACC]; N_GRU],
    pub uh: [[f64; N_GRU]; N_GRU],
    pub bh: [f64; N_GRU],
    pub w_out_h: [[f64; N_GRU]; 6], // [alpha, lambda, beta, g0, g1, g2]
    pub w_out_m: [[f64; N_ACC]; 6],
    pub b_out: [f64; 6],
}

impl NeuromodGru {
    pub fn init(seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x35_4E4D);
        let mut s = Self {
            wz: [[0.0; N_ACC]; N_GRU],
            uz: [[0.0; N_GRU]; N_GRU],
            bz: [-0.4; N_GRU],
            wr: [[0.0; N_ACC]; N_GRU],
            ur: [[0.0; N_GRU]; N_GRU],
            br: [0.3; N_GRU],
            wh: [[0.0; N_ACC]; N_GRU],
            uh: [[0.0; N_GRU]; N_GRU],
            bh: [0.0; N_GRU],
            w_out_h: [[0.0; N_GRU]; 6],
            w_out_m: [[0.0; N_ACC]; 6],
            b_out: [-0.8, 1.8, 0.3, 0.4, 0.1, -0.5],
        };
        for r in 0..N_GRU {
            for c in 0..N_ACC {
                s.wz[r][c] = rng.gen_range(-0.25..0.25);
                s.wr[r][c] = rng.gen_range(-0.25..0.25);
                s.wh[r][c] = rng.gen_range(-0.35..0.35);
            }
            for c in 0..N_GRU {
                s.uz[r][c] = rng.gen_range(-0.20..0.20);
                s.ur[r][c] = rng.gen_range(-0.20..0.20);
                s.uh[r][c] = rng.gen_range(-0.20..0.20);
            }
        }
        // Sesgo inicial biológicamente coherente (refinado por BPTT outer-loop):
        // m = [sorpresa/ciclo, desacuerdo_adv, bucle_dogma, saturación_F]
        // α (noradrenalina): sube con sorpresa/desacuerdo, baja en saturación/bucle
        s.w_out_m[0] = [1.4, 0.9, -0.8, -0.7];
        // λ (retención/reset): cae fuerte ante choque/contradicción o bucle dogmático
        s.w_out_m[1] = [-2.2, -1.5, -2.5, -0.8];
        // β (precisión acetilcolina): baja cuando hay desacuerdo adversarial o bucle dogmático!
        s.w_out_m[2] = [-1.4, -2.4, -2.2, -0.5];
        // g_mode: [0]=rápido heurístico, [1]=refutación profunda, [2]=reset/exploración
        s.w_out_m[3] = [-1.2, -1.6, -1.4, -0.3];
        s.w_out_m[4] = [0.6, 1.9, 0.5, 0.2];
        s.w_out_m[5] = [1.4, 0.8, 2.2, 1.0];
        for o in 0..6 {
            for h in 0..N_GRU {
                s.w_out_h[o][h] = rng.gen_range(-0.15..0.15);
            }
        }
        s
    }

    pub fn step(&self, m: [f64; N_ACC], h_prev: [f64; N_GRU]) -> (NeuromodSignal, [f64; N_GRU]) {
        let mut z = [0.0; N_GRU];
        let mut r = [0.0; N_GRU];
        let mut h_tilde = [0.0; N_GRU];
        let mut h_new = [0.0; N_GRU];
        for i in 0..N_GRU {
            let mut az = self.bz[i];
            let mut ar = self.br[i];
            for j in 0..N_ACC {
                az += self.wz[i][j] * m[j];
                ar += self.wr[i][j] * m[j];
            }
            for j in 0..N_GRU {
                az += self.uz[i][j] * h_prev[j];
                ar += self.ur[i][j] * h_prev[j];
            }
            z[i] = sig(az);
            r[i] = sig(ar);
        }
        for i in 0..N_GRU {
            let mut ah = self.bh[i];
            for j in 0..N_ACC {
                ah += self.wh[i][j] * m[j];
            }
            for j in 0..N_GRU {
                ah += self.uh[i][j] * (r[j] * h_prev[j]);
            }
            h_tilde[i] = ah.tanh();
            h_new[i] = (1.0 - z[i]) * h_prev[i] + z[i] * h_tilde[i];
        }
        let mut raw = [0.0; 6];
        for o in 0..6 {
            let mut v = self.b_out[o];
            for j in 0..N_GRU {
                v += self.w_out_h[o][j] * h_new[j];
            }
            for j in 0..N_ACC {
                v += self.w_out_m[o][j] * m[j];
            }
            raw[o] = v;
        }
        let alpha = sig(raw[0]).clamp(0.01, 0.95);
        let lambda = (0.08 + 0.91 * sig(raw[1])).clamp(0.08, 0.99);
        let beta = (0.6 + 3.4 * sig(raw[2])).clamp(0.6, 4.0);
        let gm = softmax3([raw[3], raw[4], raw[5]]);
        (
            NeuromodSignal {
                alpha,
                lambda,
                beta,
                g_mode: gm,
            },
            h_new,
        )
    }

    /// Entrenamiento outer-loop de MetaNet sobre secuencias sintéticas de estabilidad, choque,
    /// desacuerdo adversarial y bucle dogmático (gradiente en cabezas + proyecciones directas).
    pub fn train_outer_loop(&mut self, seed: u64, epochs: usize) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x007E_A001);
        // Objetivos biológicos por régimen: (m_t, target_alpha, target_lambda, target_beta_norm, target_mode)
        let lr = 0.08;
        for _ in 0..epochs {
            let mut h = [0.0; N_GRU];
            for step in 0..32 {
                let regime = (step + rng.gen_range(0..4)) % 4;
                let (m, ta, tl, tb, tm) = match regime {
                    // 0: Régimen estable verificado -> α bajo, λ alto, β alto, modo 0
                    0 => (
                        [
                            rng.gen_range(-0.6..0.05),
                            rng.gen_range(0.0..0.12),
                            rng.gen_range(0.0..0.10),
                            rng.gen_range(0.0..0.35),
                        ],
                        0.08,
                        0.96,
                        0.85,
                        [0.85, 0.10, 0.05],
                    ),
                    // 1: Cambio de tema / sorpresa fuerte -> α alto, λ bajo (reset), β medio, modo 2
                    1 => (
                        [
                            rng.gen_range(0.65..1.4),
                            rng.gen_range(0.25..0.60),
                            rng.gen_range(0.0..0.25),
                            rng.gen_range(0.2..0.7),
                        ],
                        0.82,
                        0.22,
                        0.38,
                        [0.05, 0.25, 0.70],
                    ),
                    // 2: Desacuerdo adversarial A vs B (refutación!) -> β muy bajo (no bluffear!), modo 1
                    2 => (
                        [
                            rng.gen_range(0.2..0.7),
                            rng.gen_range(0.55..1.0),
                            rng.gen_range(0.05..0.30),
                            rng.gen_range(0.2..0.8),
                        ],
                        0.55,
                        0.50,
                        0.12,
                        [0.05, 0.80, 0.15],
                    ),
                    // 3: Bucle / colapso dogmático -> λ muy bajo (limpiar F_t!), β muy bajo, α bajo, modo 2
                    _ => (
                        [
                            rng.gen_range(0.3..1.0),
                            rng.gen_range(0.2..0.8),
                            rng.gen_range(0.55..1.0),
                            rng.gen_range(0.65..1.0),
                        ],
                        0.12,
                        0.10,
                        0.08,
                        [0.03, 0.17, 0.80],
                    ),
                };
                let (sn, h_new) = self.step(m, h);
                let pa = sn.alpha;
                let pl = (sn.lambda - 0.08) / 0.91;
                let pb = (sn.beta - 0.6) / 3.4;
                let d_raw = [
                    (pa - ta) * pa * (1.0 - pa),
                    (pl - tl) * pl * (1.0 - pl),
                    (pb - tb) * pb * (1.0 - pb),
                    sn.g_mode[0] - tm[0],
                    sn.g_mode[1] - tm[1],
                    sn.g_mode[2] - tm[2],
                ];
                let w_step = if regime >= 2 { 2.5 } else { 1.0 };
                for o in 0..6 {
                    let g = (w_step * d_raw[o]).clamp(-2.0, 2.0);
                    self.b_out[o] -= lr * (g + 1e-3 * self.b_out[o]);
                    for j in 0..N_GRU {
                        self.w_out_h[o][j] -= lr * (g * h_new[j] + 1e-3 * self.w_out_h[o][j]);
                    }
                    for j in 0..N_ACC {
                        self.w_out_m[o][j] -= lr * (g * m[j] + 1e-3 * self.w_out_m[o][j]);
                    }
                }
                // Actualización recurrente ligera en wh para propagar memoria de conflicto
                for i in 0..N_GRU {
                    let dh: f64 = (0..6).map(|o| w_step * d_raw[o] * self.w_out_h[o][i]).sum();
                    let g = (dh * (1.0 - h_new[i] * h_new[i])).clamp(-1.0, 1.0);
                    self.bh[i] -= 0.25 * lr * g;
                    for j in 0..N_ACC {
                        self.wh[i][j] -= 0.25 * lr * (g * m[j] + 1e-3 * self.wh[i][j]);
                    }
                }
                h = h_new;
            }
        }
    }
}

/// Matriz de pesos rápidos efímeros `F_t ∈ R^{H_LAT × D_SEM}` + sesgo rápido `b_f ∈ R^{H_LAT}`
/// con trazas de elegibilidad `E_t` y proyección estricta sobre la bola Frobenius `Π_{‖·‖_F ≤ ρ}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FastBranch {
    pub f: Vec<f64>,  // H_LAT * D_SEM
    pub e: Vec<f64>,  // H_LAT * D_SEM (traza de elegibilidad)
    #[serde(default = "default_bf")]
    pub bf: Vec<f64>, // H_LAT (sesgo rápido efímero)
    #[serde(default = "default_bf")]
    pub eb: Vec<f64>, // H_LAT (traza del sesgo rápido)
    pub rho: f64,
    pub gamma_e: f64,
}

fn default_bf() -> Vec<f64> {
    vec![0.0; H_LAT]
}

impl FastBranch {
    pub fn new(rho: f64, gamma_e: f64) -> Self {
        Self {
            f: vec![0.0; H_LAT * D_SEM],
            e: vec![0.0; H_LAT * D_SEM],
            bf: vec![0.0; H_LAT],
            eb: vec![0.0; H_LAT],
            rho,
            gamma_e,
        }
    }

    pub fn reset(&mut self) {
        for v in &mut self.f {
            *v = 0.0;
        }
        for v in &mut self.e {
            *v = 0.0;
        }
        for v in &mut self.bf {
            *v = 0.0;
        }
        for v in &mut self.eb {
            *v = 0.0;
        }
    }

    pub fn frob_norm(&self) -> f64 {
        let s_f: f64 = self.f.iter().map(|v| v * v).sum();
        let s_b: f64 = self.bf.iter().map(|v| v * v).sum();
        (s_f + s_b).sqrt()
    }

    pub fn saturation(&self) -> f64 {
        if self.rho <= 1e-12 {
            0.0
        } else {
            (self.frob_norm() / self.rho).clamp(0.0, 1.0)
        }
    }

    pub fn project_ball(&mut self) {
        let n = self.frob_norm();
        if !n.is_finite() {
            self.reset();
            return;
        }
        if n > self.rho && n > 1e-15 {
            let k = self.rho / n;
            for v in &mut self.f {
                *v *= k;
            }
            for v in &mut self.bf {
                *v *= k;
            }
        }
    }

    /// Proyección ortogonal estricta de esta rama respecto a `other` (`⟨(F_B, bf_B), (F_A, bf_A)⟩ = 0`):
    /// impide que la rama refutadora colapse en una copia de la rama proponente.
    pub fn orthogonalize_against(&mut self, other: &FastBranch) {
        let dot_f: f64 = self.f.iter().zip(&other.f).map(|(a, b)| a * b).sum();
        let dot_b: f64 = self.bf.iter().zip(&other.bf).map(|(a, b)| a * b).sum();
        let n2_f: f64 = other.f.iter().map(|v| v * v).sum();
        let n2_b: f64 = other.bf.iter().map(|v| v * v).sum();
        let n2 = n2_f + n2_b;
        if n2 > 1e-10 {
            let coeff = (dot_f + dot_b) / n2;
            for (a, &b) in self.f.iter_mut().zip(&other.f) {
                *a -= coeff * b;
            }
            for (a, &b) in self.bf.iter_mut().zip(&other.bf) {
                *a -= coeff * b;
            }
        }
        self.project_ball();
    }

    /// Regla de plasticidad de tres factores:
    ///   E_t = γ E_{t-1} + φ(err_h) ⊗ [x_t, 1]
    ///   (F_{t+1}, bf_{t+1}) = Π_ρ(λ_t (F_t, bf_t) + α_t · δ_t · E_t)
    pub fn update_three_factor(
        &mut self,
        err_h: &[f64; H_LAT],
        x: &[f64; D_SEM],
        alpha: f64,
        lambda: f64,
        delta: f64,
        lr_scale: f64,
    ) {
        let xn2: f64 = x.iter().map(|v| v * v).sum::<f64>() + 0.25;
        let step_k = (lr_scale * alpha * delta).clamp(-0.35, 0.35);
        for r in 0..H_LAT {
            let pe = 2.0 * (err_h[r] / 2.0).tanh();
            let row = r * D_SEM;
            for c in 0..D_SEM {
                let idx = row + c;
                let et = self.gamma_e * self.e[idx] + pe * (x[c] / xn2);
                self.e[idx] = et.clamp(-2.5, 2.5);
                self.f[idx] = lambda * self.f[idx] + step_k * self.e[idx];
            }
            let ebt = self.gamma_e * self.eb[r] + 0.5 * pe;
            self.eb[r] = ebt.clamp(-2.5, 2.5);
            self.bf[r] = lambda * self.bf[r] + step_k * self.eb[r];
        }
        self.project_ball();
    }
}

/// Núcleo lento estructural W_0 (congelado durante conversación; solo se consolida offline con EWC).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlowCoreW0 {
    pub w_enc: Vec<f64>,    // H_LAT * D_SEM
    pub b_enc: Vec<f64>,    // H_LAT
    pub u_h: Vec<f64>,      // H_LAT * H_LAT (memoria de trabajo recurrente)
    pub b_mode: Vec<f64>,   // H_LAT * N_MODES (sesgo ejecutivo dlPFC B·g_t)
    pub w_dec: Vec<f64>,    // D_SEM * H_LAT (predicción de estado respuesta z_t)
    pub w_rev: Vec<f64>,    // D_SEM * D_SEM (reconstructor inverso de premisa Φ_rev(z_t) ≈ x_t)
    pub w_intent: Vec<f64>, // N_INTENTS * H_LAT
    pub b_intent: Vec<f64>, // N_INTENTS
    pub w_stance: Vec<f64>, // N_STANCES * H_LAT
    pub b_stance: Vec<f64>, // N_STANCES
    /// Sesgo escéptico independiente de la Rama B para romper simetría desde el turno 0.
    pub b_skeptic: Vec<f64>, // H_LAT
}

impl SlowCoreW0 {
    pub fn init(seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5100_0000);
        let s_enc = (1.0 / D_SEM as f64).sqrt();
        let s_hid = (1.0 / H_LAT as f64).sqrt();
        let mut w_enc = vec![0.0; H_LAT * D_SEM];
        for v in &mut w_enc {
            *v = rng.gen_range(-s_enc..s_enc);
        }
        let mut u_h = vec![0.0; H_LAT * H_LAT];
        for r in 0..H_LAT {
            for c in 0..H_LAT {
                u_h[r * H_LAT + c] = if r == c { 0.18 } else { rng.gen_range(-0.04..0.04) };
            }
        }
        let mut b_mode = vec![0.0; H_LAT * N_MODES];
        for v in &mut b_mode {
            *v = rng.gen_range(-0.15..0.15);
        }
        let mut w_dec = vec![0.0; D_SEM * H_LAT];
        for v in &mut w_dec {
            *v = rng.gen_range(-s_hid..s_hid);
        }
        // Inicializar Φ_rev cerca de la identidad suave para que la reconstrucción de premisa arranque estable
        let mut w_rev = vec![0.0; D_SEM * D_SEM];
        for r in 0..D_SEM {
            for c in 0..D_SEM {
                w_rev[r * D_SEM + c] = if r == c { 0.75 } else { rng.gen_range(-0.03..0.03) };
            }
        }
        let mut w_intent = vec![0.0; N_INTENTS * H_LAT];
        for v in &mut w_intent {
            *v = rng.gen_range(-s_hid..s_hid);
        }
        let mut w_stance = vec![0.0; N_STANCES * H_LAT];
        for v in &mut w_stance {
            *v = rng.gen_range(-s_hid..s_hid);
        }
        let b_skeptic: Vec<f64> = (0..H_LAT)
            .map(|i| if i % 2 == 0 { 0.28 } else { -0.28 })
            .collect();
        Self {
            w_enc,
            b_enc: vec![0.0; H_LAT],
            u_h,
            b_mode,
            w_dec,
            w_rev,
            w_intent,
            b_intent: vec![0.0; N_INTENTS],
            w_stance,
            b_stance: vec![0.0; N_STANCES],
            b_skeptic,
        }
    }

    pub fn flat_params(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.param_count());
        v.extend_from_slice(&self.w_enc);
        v.extend_from_slice(&self.b_enc);
        v.extend_from_slice(&self.w_dec);
        v.extend_from_slice(&self.w_rev);
        v.extend_from_slice(&self.w_intent);
        v.extend_from_slice(&self.b_intent);
        v.extend_from_slice(&self.w_stance);
        v.extend_from_slice(&self.b_stance);
        v
    }

    pub fn param_count(&self) -> usize {
        self.w_enc.len()
            + self.b_enc.len()
            + self.w_dec.len()
            + self.w_rev.len()
            + self.w_intent.len()
            + self.b_intent.len()
            + self.w_stance.len()
            + self.b_stance.len()
    }
}

/// Estado EWC (diagonal exacta de Fisher Ω_i y ancla W_0^prev).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EwcAnchor {
    pub initialized: bool,
    pub w_prev: Vec<f64>,
    pub omega: Vec<f64>,
    pub consolidations: usize,
}

/// Episodio verificado guardado en el Buffer Episódico (solo episodios que pasaron refutación y anti-dogma).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifiedEpisode {
    pub x: Vec<f64>,
    pub x_cf: Vec<f64>,
    pub target_z: Vec<f64>,
    pub intent_idx: usize,
    pub stance_idx: usize,
    pub weight: f64,
    pub domain: u8,
}

/// Telemetría completa de un paso cognitivo (glass-box).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveStepReport {
    pub intent: DialogueIntent,
    pub stance: EpistemicStance,
    pub intent_probs: Vec<f64>,
    pub stance_probs: Vec<f64>,
    pub z_a: Vec<f64>,
    pub z_b: Vec<f64>,
    pub cycle_err_a: f64,
    pub cycle_err_b: f64,
    pub counterfactual_sens: f64,
    pub cross_disagree: f64,
    pub refute_vuln: f64,
    pub cross_refute_margin: f64,
    pub loop_index: f64,
    pub dogma_score: f64,
    pub delta_real: f64,
    pub free_energy_null: f64,
    pub free_energy_post: f64,
    pub free_energy_delta: f64,
    pub acc_m: [f64; N_ACC],
    pub neuromod: NeuromodSignal,
    pub confidence: f64,
    pub fast_norm_a: f64,
    pub fast_norm_b: f64,
    pub branch_orthogonality: f64,
    pub active_branch: String,
    pub entropy_norm: f64,
}

/// Motor Cognitivo completo METAFE-3S-Dial (los 5 módulos integrados).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Core3S {
    pub delta_mode: DeltaMode,
    pub w0: SlowCoreW0,
    pub branch_a: FastBranch,
    pub branch_b: FastBranch,
    pub nm: NeuromodGru,
    pub ewc: EwcAnchor,
    pub ewc_mu: f64,
    pub buffer: Vec<VerifiedEpisode>,
    pub buffer_cap: usize,
    /// Calibrador metacognitivo "Yo-FE 3S" (Platt L2 sobre rasgos de telemetría ACC/refutación).
    pub cal_w: [f64; 7],
    /// Estado dinámico de trabajo en conversación
    pub h_prev: [f64; H_LAT],
    pub nm_state: [f64; N_GRU],
    pub recent_h: VecDeque<Vec<f64>>,
    pub recent_x: VecDeque<Vec<f64>>,
    pub prev_cycle_err: f64,
    pub prev_entropy: f64,
    /// Estadísticas acumuladas
    pub total_turns: u64,
    pub total_train_episodes: u64,
    pub total_sleeps: u64,
    pub total_dogma_blocks: u64,
    pub total_resets: u64,
}

impl Core3S {
    pub fn new(seed: u64, delta_mode: DeltaMode, ewc_mu: f64) -> Self {
        let mut nm = NeuromodGru::init(seed);
        nm.train_outer_loop(seed, 45);
        Self {
            delta_mode,
            w0: SlowCoreW0::init(seed),
            branch_a: FastBranch::new(0.65, 0.55),
            branch_b: FastBranch::new(0.65, 0.55),
            nm,
            ewc: EwcAnchor::default(),
            ewc_mu,
            buffer: Vec::new(),
            buffer_cap: 512,
            // Pesos iniciales de calibración "Yo-FE 3S": [1, max_p, -cycle_err, -refute_vuln, cf_sens, -dogma, grounded]
            cal_w: [-0.4, 2.2, -1.8, -2.6, 1.1, -3.0, 1.4],
            h_prev: [0.0; H_LAT],
            nm_state: [0.0; N_GRU],
            recent_h: VecDeque::with_capacity(8),
            recent_x: VecDeque::with_capacity(8),
            prev_cycle_err: 0.5,
            prev_entropy: 0.8,
            total_turns: 0,
            total_train_episodes: 0,
            total_sleeps: 0,
            total_dogma_blocks: 0,
            total_resets: 0,
        }
    }

    pub fn reset_episode_fast_state(&mut self) {
        self.branch_a.reset();
        self.branch_b.reset();
        self.h_prev = [0.0; H_LAT];
        self.nm_state = [0.0; N_GRU];
        self.recent_h.clear();
        self.recent_x.clear();
        self.prev_cycle_err = 0.5;
        self.prev_entropy = 0.8;
    }

    fn forward_branch(
        &self,
        x: &[f64; D_SEM],
        fast: Option<&FastBranch>,
        beta: f64,
        g_mode: &[f64; N_MODES],
        skeptic: bool,
    ) -> ([f64; H_LAT], [f64; D_SEM], [f64; N_INTENTS], [f64; N_STANCES]) {
        let mut h = [0.0; H_LAT];
        for r in 0..H_LAT {
            let mut acc = self.w0.b_enc[r] + fast.map(|fb| fb.bf[r]).unwrap_or(0.0);
            let row = r * D_SEM;
            for c in 0..D_SEM {
                let w = self.w0.w_enc[row + c] + fast.map(|fb| fb.f[row + c]).unwrap_or(0.0);
                acc += w * x[c];
            }
            let mut rec = 0.0;
            let urow = r * H_LAT;
            for c in 0..H_LAT {
                rec += self.w0.u_h[urow + c] * self.h_prev[c];
            }
            let mut mode_b = 0.0;
            for m in 0..N_MODES {
                mode_b += self.w0.b_mode[r * N_MODES + m] * g_mode[m];
            }
            let sk = if skeptic { self.w0.b_skeptic[r] } else { 0.0 };
            h[r] = (beta * acc + 0.25 * rec + mode_b + sk).tanh();
        }
        let mut z = [0.0; D_SEM];
        for d in 0..D_SEM {
            let row = d * H_LAT;
            let mut acc = 0.0;
            for r in 0..H_LAT {
                acc += self.w0.w_dec[row + r] * h[r];
            }
            z[d] = acc;
        }
        let mut l_int = [0.0; N_INTENTS];
        for c in 0..N_INTENTS {
            let row = c * H_LAT;
            let mut acc = self.w0.b_intent[c];
            for r in 0..H_LAT {
                acc += self.w0.w_intent[row + r] * h[r];
            }
            l_int[c] = beta * acc;
        }
        let mut l_st = [0.0; N_STANCES];
        for c in 0..N_STANCES {
            let row = c * H_LAT;
            let mut acc = self.w0.b_stance[c];
            for r in 0..H_LAT {
                acc += self.w0.w_stance[row + r] * h[r];
            }
            l_st[c] = beta * acc;
        }
        (h, z, l_int, l_st)
    }

    /// Reconstructor inverso de premisa Φ_rev(z) ≈ x_t y error de ciclo normalizado.
    pub fn cycle_error(&self, z: &[f64; D_SEM], x: &[f64; D_SEM]) -> (f64, [f64; D_SEM]) {
        let mut x_hat = [0.0; D_SEM];
        for r in 0..D_SEM {
            let row = r * D_SEM;
            let mut acc = 0.0;
            for c in 0..D_SEM {
                acc += self.w0.w_rev[row + c] * z[c];
            }
            x_hat[r] = acc;
        }
        let mut diff2 = 0.0;
        let mut xn2 = 1e-8;
        for i in 0..D_SEM {
            let d = x_hat[i] - x[i];
            diff2 += d * d;
            xn2 += x[i] * x[i];
        }
        ((diff2 / xn2).min(4.0), x_hat)
    }

    /// Paso cognitivo online completo (con `W_0` ESTRICTAMENTE CONGELADO).
    ///
    /// Si `verdict` es `None`, el núcleo opera en **razonamiento abierto sin feedback externo**:
    /// aquí es donde `NaiveEntropy` colapsa dogmáticamente mientras `AdversarialCrossRefute`
    /// usa el verificador cruzado dual `F_t^A ⊥ F_t^B` + ciclo + contrafactual + anti-bucle.
    pub fn think_step(
        &mut self,
        frame: &SemanticFrame,
        verdict: Option<&GroundingVerdict>,
        allow_fast_update: bool,
    ) -> CognitiveStepReport {
        self.total_turns += 1;
        let x = &frame.vec;
        let x_cf = &frame.counterfactual_vec;

        let use_fast = self.delta_mode != DeltaMode::NoFast;
        let use_branch_b = self.delta_mode == DeltaMode::AdversarialCrossRefute;

        // 1. Paso preliminar de Rama A con señal neuromoduladora neutra para medir ACC m_t
        let init_beta = 1.8;
        let init_g = [0.5, 0.35, 0.15];
        let (h_a0, z_a0, _l_int0, l_st0) = self.forward_branch(
            x,
            use_fast.then_some(&self.branch_a),
            init_beta,
            &init_g,
            false,
        );
        let (cyc_a0, _) = self.cycle_error(&z_a0, x);

        // Sensibilidad contrafactual inmediata de Rama A sobre x_cf
        let (_, z_cf0, _, l_st_cf0) = self.forward_branch(
            x_cf,
            use_fast.then_some(&self.branch_a),
            init_beta,
            &init_g,
            false,
        );
        let cf_sens0 = rel_dist(&z_a0, &z_cf0)
            + 0.35 * l1_prob_diff(&softmax_slice(&l_st0), &softmax_slice(&l_st_cf0));

        // 2. Refutación Adversarial Independiente por Rama B (F_t^B ⊥ F_t^A):
        //    Rama B ejecuta K=3 pasos rápidos buscando minimizar E_cyc(z_B, x) mientras
        //    MAXIMIZA la divergencia respecto a z_A (contra-hipótesis válida que refuta a A!).
        let mut cyc_b = cyc_a0;
        let mut z_b = z_a0;
        let mut cross_disagree = 0.0;
        let mut refute_vuln = 0.0;

        if use_branch_b {
            let mut fb_work = self.branch_b.clone();
            fb_work.orthogonalize_against(&self.branch_a);
            for _ in 0..3 {
                let (hb, zb, _, _) = self.forward_branch(x, Some(&fb_work), 1.6, &[0.1, 0.8, 0.1], true);
                let (cb, x_hat_b) = self.cycle_error(&zb, x);
                // Gradiente aproximado en z_B: bajar error de ciclo y alejarse de z_A0
                let mut dz = [0.0; D_SEM];
                for j in 0..D_SEM {
                    let mut g_cyc = 0.0;
                    for i in 0..D_SEM {
                        g_cyc += self.w0.w_rev[i * D_SEM + j] * (x[i] - x_hat_b[i]);
                    }
                    let rep = (zb[j] - z_a0[j]).clamp(-0.6, 0.6);
                    dz[j] = 0.85 * g_cyc + 0.35 * rep;
                }
                let mut err_hb = [0.0; H_LAT];
                for r in 0..H_LAT {
                    let mut acc = 0.0;
                    for j in 0..D_SEM {
                        acc += self.w0.w_dec[j * H_LAT + r] * dz[j];
                    }
                    err_hb[r] = acc * (1.0 - hb[r] * hb[r]);
                }
                fb_work.update_three_factor(&err_hb, x, 0.65, 0.92, 1.0, 0.14);
                fb_work.orthogonalize_against(&self.branch_a);
                cyc_b = cb;
                z_b = zb;
            }
            if allow_fast_update {
                self.branch_b = fb_work;
            }
            cross_disagree = rel_dist(&z_a0, &z_b);
            let cyc_excess_b = (cyc_b - cyc_a0).max(0.0);
            refute_vuln = (cross_disagree * (-2.2 * cyc_excess_b).exp()).clamp(0.0, 1.0);
        }

        // 3. Índice preliminar de bucle / dogmatismo para alimentar el monitor ACC m_t
        let compute_loop = |h_cur: &[f64; H_LAT], recent_h: &VecDeque<Vec<f64>>, recent_x: &VecDeque<Vec<f64>>| -> f64 {
            let mut mx = 0.0_f64;
            for (hp, xp) in recent_h.iter().zip(recent_x.iter()) {
                let cos_h = cosine_slices(h_cur, hp);
                let cos_x = cosine_slices(x, xp);
                let excess_rep = (cos_h - cos_x.max(0.15)).max(0.0);
                if excess_rep > mx {
                    mx = excess_rep;
                }
            }
            mx.clamp(0.0, 1.0)
        };
        let loop_pre = compute_loop(&h_a0, &self.recent_h, &self.recent_x);
        let p_st0 = softmax_slice(&l_st0);
        let overconf_pre = (1.0 - norm_entropy(&p_st0)).clamp(0.0, 1.0);
        let cf_def_pre = (0.30 - cf_sens0).max(0.0) / 0.30;
        let cyc_bad_pre = (cyc_a0 - 0.25).max(0.0).clamp(0.0, 1.0);
        let dogma_pre = (overconf_pre * loop_pre.max(cyc_bad_pre).max(cf_def_pre)).clamp(0.0, 1.0);

        // 4. Monitor ACC m_t ∈ R⁴ y Controlador Neuromodulador MetaNet GRU
        let sym_conflict = verdict.map(|v| if v.belief_conflict { 0.85 } else { 0.0 }).unwrap_or(0.0);
        let sym_revised = verdict.map(|v| if v.belief_revised { 0.75 } else { 0.0 }).unwrap_or(0.0);
        let surprise_m = ((cyc_a0 + 0.05).ln() + 1.2 + sym_conflict + sym_revised).clamp(-1.5, 2.0);
        let disagree_m = (refute_vuln + 0.6 * sym_conflict).clamp(0.0, 1.5);
        let sat_m = self.branch_a.saturation().max(self.branch_b.saturation());
        let acc_m = [surprise_m, disagree_m, loop_pre.max(dogma_pre), sat_m];

        let (nm_sig, next_nm_state) = self.nm.step(acc_m, self.nm_state);
        self.nm_state = next_nm_state;

        // En modo NaiveEntropy no existe regulación anti-dogma (ganancia fija alta, persistencia casi unitaria).
        // En modo completo, MetaNet reduce β_t cuando hay desacuerdo adversarial/dogma y dispara reset λ_t ante bucle.
        let (beta_eff, lambda_eff, alpha_eff) = match self.delta_mode {
            DeltaMode::NaiveEntropy => (3.2, 0.985, 0.80),
            _ => {
                let b_adj = if dogma_pre > 0.35 || sym_conflict > 0.5 || refute_vuln > 0.40 {
                    (nm_sig.beta * 0.60).clamp(0.55, 1.55)
                } else {
                    nm_sig.beta
                };
                let l_adj = if dogma_pre > 0.42 || sym_revised > 0.5 || sym_conflict > 0.5 {
                    self.total_resets += 1;
                    nm_sig.lambda.min(0.28)
                } else {
                    nm_sig.lambda
                };
                (b_adj, l_adj, nm_sig.alpha)
            }
        };

        // 5. Inferencia final con ganancia β_t y enrutamiento g_t de MetaNet
        let (h_a, z_a, l_int, l_st) = self.forward_branch(
            x,
            use_fast.then_some(&self.branch_a),
            beta_eff,
            &nm_sig.g_mode,
            false,
        );
        let (cyc_a, x_hat_a) = self.cycle_error(&z_a, x);
        let (_, z_cf, _, l_st_cf) = self.forward_branch(
            x_cf,
            use_fast.then_some(&self.branch_a),
            beta_eff,
            &nm_sig.g_mode,
            false,
        );
        let cf_sens = (rel_dist(&z_a, &z_cf)
            + 0.35 * l1_prob_diff(&softmax_slice(&l_st), &softmax_slice(&l_st_cf)))
        .clamp(0.0, 2.0);

        let p_int = softmax_slice(&l_int);
        let p_st = softmax_slice(&l_st);
        let ent_final = norm_entropy(&p_st);

        // Índice de dogmatismo POST-neuromodulación (100% simétrico para todos los brazos):
        //   sobreconfianza emitida (1 - H_final) × max(bucle_estado, fallo_ciclo, insensibilidad_contrafactual)
        let loop_index = compute_loop(&h_a, &self.recent_h, &self.recent_x);
        let overconf_post = (1.0 - ent_final).clamp(0.0, 1.0);
        let cf_deficit = (0.30 - cf_sens).max(0.0) / 0.30;
        let cyc_bad = (cyc_a - 0.25).max(0.0).clamp(0.0, 1.0);
        let dogma_score = (overconf_post * loop_index.max(cyc_bad).max(cf_deficit)).clamp(0.0, 1.0);

        // Energía Libre Variacional (sorpresa de reconstrucción vs modelo nulo persistente + complejidad KL)
        let fe_null = 0.5 * (1.0 + 1.0_f64.ln() + std::f64::consts::TAU.ln());
        let kl_fast = 0.15 * (self.branch_a.frob_norm().powi(2));
        let fe_post = 0.5 * ((cyc_a.max(1e-4)).ln() + 1.0 + std::f64::consts::TAU.ln()) + kl_fast;
        let fe_delta = fe_null - fe_post;

        // 6. Cálculo de δ_t según el modo (AQUÍ ESTÁ LA SOLUCIÓN AL CUELLO DE BOTELLA!)
        let cyc_improv = (self.prev_cycle_err - cyc_a).clamp(-1.0, 1.0);
        let cf_reward = (cf_sens - 0.25).clamp(-0.6, 0.8);
        let sym_delta = verdict.map(|v| v.symbolic_delta).unwrap_or(0.0);
        let cross_refute_margin = (cyc_b - cyc_a) - 0.5 * refute_vuln;

        let delta_real = match self.delta_mode {
            DeltaMode::NaiveEntropy => {
                // EL FALLO CLÁSICO: premiar exclusivamente reducir la propia entropía (auto-refuerzo ciego!)
                (self.prev_entropy - ent_final) + 0.65 * (1.0 - ent_final)
            }
            DeltaMode::NoRefute => {
                0.70 * cyc_improv + 0.45 * cf_reward - 0.70 * dogma_score + 0.50 * sym_delta
            }
            DeltaMode::AdversarialCrossRefute => {
                0.75 * cyc_improv
                    + 0.55 * cf_reward
                    - 0.85 * refute_vuln
                    - 1.10 * dogma_score
                    + 0.55 * sym_delta
            }
            DeltaMode::NoFast => 0.0,
        };

        // 7. Plasticidad de tres factores online en F_t^A (nunca toca W_0!)
        if allow_fast_update && use_fast {
            match self.delta_mode {
                DeltaMode::NaiveEntropy => {
                    // Gradiente analítico exacto de minimizar entropía propia −∇H(p_st) (Tent / auto-entrenamiento ciego):
                    //   −∂H/∂l_c = p_c · (ln p_c + H(p))
                    let h_raw: f64 = -p_st
                        .iter()
                        .filter(|&&p| p > 1e-12)
                        .map(|&p| p * p.ln())
                        .sum::<f64>();
                    let mut err_h = [0.0; H_LAT];
                    for r in 0..H_LAT {
                        let mut g = 0.0;
                        for c in 0..N_STANCES {
                            let pc = p_st[c].max(1e-12);
                            let neg_dh_dl = pc * (pc.ln() + h_raw);
                            g += neg_dh_dl * self.w0.w_stance[c * H_LAT + r];
                        }
                        err_h[r] = beta_eff * g * (1.0 - h_a[r] * h_a[r]);
                    }
                    self.branch_a.update_three_factor(
                        &err_h,
                        x,
                        alpha_eff,
                        lambda_eff,
                        delta_real.max(0.35),
                        0.38,
                    );
                }
                DeltaMode::AdversarialCrossRefute | DeltaMode::NoRefute => {
                    // Dirección de gradiente fundamentada en coherencia de ciclo (reconstruir premisa x_t)
                    let mut err_h = [0.0; H_LAT];
                    let mut dz = [0.0; D_SEM];
                    for j in 0..D_SEM {
                        let mut g_cyc = 0.0;
                        for i in 0..D_SEM {
                            g_cyc += self.w0.w_rev[i * D_SEM + j] * (x[i] - x_hat_a[i]);
                        }
                        dz[j] = g_cyc;
                    }
                    for r in 0..H_LAT {
                        let mut acc = 0.0;
                        for j in 0..D_SEM {
                            acc += self.w0.w_dec[j * H_LAT + r] * dz[j];
                        }
                        err_h[r] = acc * (1.0 - h_a[r] * h_a[r]);
                    }
                    if dogma_score > 0.40 {
                        self.total_dogma_blocks += 1;
                        self.branch_a.update_three_factor(&err_h, x, 0.0, lambda_eff, 0.0, 0.0);
                    } else {
                        self.branch_a.update_three_factor(
                            &err_h,
                            x,
                            alpha_eff,
                            lambda_eff,
                            delta_real.clamp(-0.8, 1.0),
                            0.12,
                        );
                    }
                    if use_branch_b {
                        self.branch_b.orthogonalize_against(&self.branch_a);
                    }
                }
                DeltaMode::NoFast => {}
            }
        }

        // Actualizar estado de trabajo reciente
        self.h_prev = h_a;
        self.prev_cycle_err = cyc_a;
        self.prev_entropy = ent_final;
        if self.recent_h.len() >= 6 {
            self.recent_h.pop_front();
            self.recent_x.pop_front();
        }
        self.recent_h.push_back(h_a.to_vec());
        self.recent_x.push_back(x.to_vec());

        // 8. Decisión final de intención y postura epistémica + calibración "Yo-FE 3S"
        let neural_intent = DialogueIntent::from_idx(argmax_slice(&p_int));
        let neural_stance = stance_from_idx(argmax_slice(&p_st));

        // Combinar decisión neuronal con ancla simbólica verificada (si existe) y guardia anti-dogma
        let (final_intent, mut final_stance) = if let Some(v) = verdict {
            if v.grounded {
                (frame.intent_hint, v.stance)
            } else {
                let st = if p_st[argmax_slice(&p_st)] > 0.45 {
                    neural_stance
                } else {
                    v.stance
                };
                (neural_intent, st)
            }
        } else {
            (neural_intent, neural_stance)
        };

        // Guardia anti-dogmatismo en modo completo: si Branch B encuentra un contramodelo competitivo o hay dogma alto sin ancla simbólica, jamás afirmar ciegamente!
        let active_branch = if self.delta_mode == DeltaMode::AdversarialCrossRefute {
            let grounded_ok = verdict.map(|v| v.grounded).unwrap_or(false);
            let competitive_refute = refute_vuln > 0.55 && cyc_b <= cyc_a + 0.12;
            if !grounded_ok && (competitive_refute || dogma_score > 0.42) {
                if final_stance == EpistemicStance::Assert {
                    final_stance = if refute_vuln > 0.65 {
                        EpistemicStance::NeedClarification
                    } else {
                        EpistemicStance::Qualified
                    };
                }
                "Rama B refutó certeza → postura degradada/calibrada".to_string()
            } else if dogma_score > 0.42 {
                "ACC bloqueó bucle dogmático (reset λ_t)".to_string()
            } else {
                "Rama A sobrevivió refutación de Rama B".to_string()
            }
        } else if self.delta_mode == DeltaMode::NaiveEntropy {
            "Rama única (entropía ingenua sin refutador)".to_string()
        } else {
            "Rama A activa".to_string()
        };

        // Confianza calibrada "Yo-FE 3S"
        let max_p = p_st.iter().cloned().fold(0.0_f64, f64::max);
        let grounded_f = if verdict.map(|v| v.grounded).unwrap_or(false) { 1.0 } else { 0.0 };
        let cal_x = [
            1.0,
            max_p,
            cyc_a.clamp(0.0, 2.0),
            refute_vuln,
            cf_sens.clamp(0.0, 1.5),
            dogma_score,
            grounded_f,
        ];
        let confidence = if self.delta_mode == DeltaMode::NaiveEntropy {
            max_p.clamp(0.05, 0.995)
        } else {
            let z_cal: f64 = self.cal_w.iter().zip(&cal_x).map(|(a, b)| a * b).sum();
            sig(z_cal.clamp(-12.0, 12.0))
        };

        // Ortogonalidad real entre Rama A y Rama B (1.0 = perfectamente ortogonales)
        let orth = {
            let na = self.branch_a.frob_norm();
            let nb = self.branch_b.frob_norm();
            if na > 1e-9 && nb > 1e-9 {
                let d_f: f64 = self.branch_a.f.iter().zip(&self.branch_b.f).map(|(a, b)| a * b).sum();
                let d_b: f64 = self.branch_a.bf.iter().zip(&self.branch_b.bf).map(|(a, b)| a * b).sum();
                1.0 - ((d_f + d_b) / (na * nb)).abs().clamp(0.0, 1.0)
            } else {
                1.0
            }
        };

        CognitiveStepReport {
            intent: final_intent,
            stance: final_stance,
            intent_probs: p_int,
            stance_probs: p_st,
            z_a: z_a.to_vec(),
            z_b: z_b.to_vec(),
            cycle_err_a: cyc_a,
            cycle_err_b: cyc_b,
            counterfactual_sens: cf_sens,
            cross_disagree,
            refute_vuln,
            cross_refute_margin,
            loop_index,
            dogma_score,
            delta_real,
            free_energy_null: fe_null,
            free_energy_post: fe_post,
            free_energy_delta: fe_delta,
            acc_m,
            neuromod: nm_sig,
            confidence,
            fast_norm_a: self.branch_a.frob_norm(),
            fast_norm_b: self.branch_b.frob_norm(),
            branch_orthogonality: orth,
            active_branch,
            entropy_norm: ent_final,
        }
    }

    /// Añade un episodio al Buffer Episódico SOLO si superó la compuerta de coherencia y anti-dogma.
    pub fn push_verified_episode(&mut self, ep: VerifiedEpisode, dogma_score: f64, delta_real: f64) -> bool {
        if self.delta_mode == DeltaMode::AdversarialCrossRefute
            && self.total_sleeps >= 2
            && (dogma_score > 0.44 || delta_real < -0.45)
        {
            return false;
        }
        if self.buffer.len() >= self.buffer_cap {
            self.buffer.remove(0);
        }
        self.buffer.push(ep);
        true
    }

    /// Módulo 5 — Consolidación Offline ("Sueño") con EWC (diagonal de Fisher Ω_i):
    /// Destila episodios verificados del buffer hacia `W_0`, actualiza Fisher Ω_i y descarta `F_t^A, F_t^B`.
    pub fn sleep_consolidate_ewc(&mut self, epochs: usize, lr: f64) -> SleepConsolidationReport {
        if self.buffer.is_empty() {
            self.reset_episode_fast_state();
            return SleepConsolidationReport::default();
        }
        let n_eps = self.buffer.len();
        let n_p = self.w0.param_count();
        let mut loss_before = 0.0;
        let mut loss_after = 0.0;
        let g_neutral = [0.5, 0.3, 0.2];

        let mut vel = vec![0.0; n_p];
        let mom = 0.65;

        for ep_i in 0..epochs {
            let mut g_enc = vec![0.0; self.w0.w_enc.len()];
            let mut g_benc = vec![0.0; H_LAT];
            let mut g_dec = vec![0.0; self.w0.w_dec.len()];
            let mut g_rev = vec![0.0; self.w0.w_rev.len()];
            let mut g_int = vec![0.0; self.w0.w_intent.len()];
            let mut g_bint = vec![0.0; N_INTENTS];
            let mut g_st = vec![0.0; self.w0.w_stance.len()];
            let mut g_bst = vec![0.0; N_STANCES];
            let mut tot_loss = 0.0;
            let mut w_tot = 0.0;

            for item in &self.buffer {
                let mut x = [0.0; D_SEM];
                let mut x_cf = [0.0; D_SEM];
                let mut tz = [0.0; D_SEM];
                x.copy_from_slice(&item.x[..D_SEM]);
                x_cf.copy_from_slice(&item.x_cf[..D_SEM]);
                tz.copy_from_slice(&item.target_z[..D_SEM]);
                let sw = item.weight.max(0.1);
                w_tot += sw;

                // Forward limpio sobre W_0 puro (sin F_t)
                let (h, z, l_int, l_st) = self.forward_branch(&x, None, 1.8, &g_neutral, false);
                let (cyc_err, x_hat) = self.cycle_error(&z, &x);
                let p_int = softmax_slice(&l_int);
                let p_st = softmax_slice(&l_st);

                let ce_int = -p_int[item.intent_idx].max(1e-9).ln();
                let ce_st = -p_st[item.stance_idx].max(1e-9).ln();
                let mut z_mse = 0.0;
                for d in 0..D_SEM {
                    z_mse += (z[d] - tz[d]).powi(2);
                }
                z_mse /= D_SEM as f64;

                // Término contrafactual en W_0: exigir separación en h cuando x se invierte a x_cf
                let (h_cf, _, _, _) = self.forward_branch(&x_cf, None, 1.8, &g_neutral, false);
                let mut cf_d2 = 0.0;
                for r in 0..H_LAT {
                    cf_d2 += (h[r] - h_cf[r]).powi(2);
                }
                let cf_margin_pen = (0.35 - cf_d2.sqrt() / (H_LAT as f64).sqrt()).max(0.0);

                tot_loss += sw * (ce_int + 1.2 * ce_st + 0.8 * z_mse + 0.8 * cyc_err + 0.6 * cf_margin_pen);

                // Backprop analítico exacto
                let mut dh = [0.0; H_LAT];
                // 1) Intent CE (l_int = 1.8 * (b_int + W_int · h))
                for c in 0..N_INTENTS {
                    let dl = 1.8 * sw * (p_int[c] - if c == item.intent_idx { 1.0 } else { 0.0 });
                    g_bint[c] += dl;
                    let row = c * H_LAT;
                    for r in 0..H_LAT {
                        g_int[row + r] += dl * h[r];
                        dh[r] += dl * self.w0.w_intent[row + r];
                    }
                }
                // 2) Stance CE (l_st = 1.8 * (b_st + W_st · h))
                for c in 0..N_STANCES {
                    let dl = 1.8 * 1.2 * sw * (p_st[c] - if c == item.stance_idx { 1.0 } else { 0.0 });
                    g_bst[c] += dl;
                    let row = c * H_LAT;
                    for r in 0..H_LAT {
                        g_st[row + r] += dl * h[r];
                        dh[r] += dl * self.w0.w_stance[row + r];
                    }
                }
                // 3) Reconstrucción de ciclo Φ_rev(z) ≈ x y objetivo z ≈ tz
                let mut dz = [0.0; D_SEM];
                for d in 0..D_SEM {
                    dz[d] += 1.6 * sw * (z[d] - tz[d]) / D_SEM as f64;
                }
                let inv_xn2 = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-8);
                for i in 0..D_SEM {
                    let dx_hat = 1.6 * sw * (x_hat[i] - x[i]) * inv_xn2;
                    let row = i * D_SEM;
                    for j in 0..D_SEM {
                        g_rev[row + j] += dx_hat * z[j];
                        dz[j] += dx_hat * self.w0.w_rev[row + j];
                    }
                }
                for d in 0..D_SEM {
                    let row = d * H_LAT;
                    for r in 0..H_LAT {
                        g_dec[row + r] += dz[d] * h[r];
                        dh[r] += dz[d] * self.w0.w_dec[row + r];
                    }
                }
                // 4) Contraste contrafactual suave (evita colapso de representación)
                if cf_margin_pen > 0.0 {
                    let s = 0.6 * sw / (cf_d2.sqrt().max(1e-5) * (H_LAT as f64).sqrt());
                    for r in 0..H_LAT {
                        dh[r] -= s * (h[r] - h_cf[r]);
                    }
                }
                // 5) Propagar hacia w_enc
                for r in 0..H_LAT {
                    let da = dh[r] * (1.0 - h[r] * h[r]) * 1.8;
                    g_benc[r] += da;
                    let row = r * D_SEM;
                    for c in 0..D_SEM {
                        g_enc[row + c] += da * x[c];
                    }
                }
            }

            let inv_w = 1.0 / w_tot.max(1e-9);
            let avg_loss = tot_loss * inv_w;
            if ep_i == 0 {
                loss_before = avg_loss;
            }
            loss_after = avg_loss;

            // Paso proximal semi-implícito de EWC con momento suave sobre TODOS los pesos y sesgos:
            //   v_i ← β_m v_i + (1 - β_m) g_data,i
            //   w_{k+1} = w_k - η · (v_i + μ·Ω_i·(w_k - w_prev)) / (1 + η·μ·Ω_i)
            let mut offset = 0usize;
            let apply_block = |w: &mut [f64],
                               g: &[f64],
                               l2: f64,
                               off: &mut usize,
                               vel: &mut [f64],
                               ewc: &EwcAnchor,
                               mu: f64| {
                for i in 0..w.len() {
                    let idx = *off + i;
                    let g_data = (g[i] * inv_w + l2 * w[i]).clamp(-2.5, 2.5);
                    vel[idx] = mom * vel[idx] + (1.0 - mom) * g_data;
                    let g_eff = vel[idx];
                    if ewc.initialized && mu > 0.0 && idx < ewc.omega.len() {
                        let k = mu * ewc.omega[idx];
                        let w_star = ewc.w_prev[idx];
                        let num = lr * (g_eff + k * (w[i] - w_star));
                        let den = 1.0 + lr * k;
                        w[i] -= num / den;
                    } else {
                        w[i] -= lr * g_eff;
                    }
                }
                *off += w.len();
            };

            apply_block(&mut self.w0.w_enc, &g_enc, 2e-4, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
            apply_block(&mut self.w0.b_enc, &g_benc, 0.0, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
            apply_block(&mut self.w0.w_dec, &g_dec, 2e-4, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
            apply_block(&mut self.w0.w_rev, &g_rev, 2e-4, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
            apply_block(&mut self.w0.w_intent, &g_int, 2e-4, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
            apply_block(&mut self.w0.b_intent, &g_bint, 0.0, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
            apply_block(&mut self.w0.w_stance, &g_st, 2e-4, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
            apply_block(&mut self.w0.b_stance, &g_bst, 0.0, &mut offset, &mut vel, &self.ewc, self.ewc_mu);
        }

        // Actualizar diagonal exacta de Fisher Ω_i (Online EWC con decaimiento 0.85) y ancla W_0^prev
        let fisher_new = self.compute_fisher_diag();
        if !self.ewc.initialized || self.ewc.omega.len() != n_p {
            self.ewc.omega = fisher_new;
            self.ewc.initialized = true;
        } else {
            for (o, f) in self.ewc.omega.iter_mut().zip(&fisher_new) {
                *o = 0.85 * (*o) + f;
            }
        }
        self.ewc.w_prev = self.w0.flat_params();
        self.ewc.consolidations += 1;

        // Re-calibrar "Yo-FE 3S" (Platt L2) sobre el buffer actual
        self.recalibrate_yofe();

        // Descartar pesos rápidos F_t^A, F_t^B al concluir el sueño (regla estricta Módulo 5)
        self.reset_episode_fast_state();
        self.total_sleeps += 1;
        self.total_train_episodes += n_eps as u64;

        let fisher_mean = self.ewc.omega.iter().sum::<f64>() / self.ewc.omega.len().max(1) as f64;
        SleepConsolidationReport {
            episodes_consolidated: n_eps,
            loss_before,
            loss_after,
            fisher_mean,
            ewc_mu: self.ewc_mu,
            consolidations: self.ewc.consolidations,
        }
    }

    /// Diagonal exacta de la matriz de información de Fisher / Gauss-Newton esperada:
    ///   Ω_i = E_{x ~ D} E_{y ~ p_θ(y|x)} [(∂ log p_θ(y|x) / ∂W_i)²]
    /// A diferencia del Fisher empírico `(p - y_true)²`, NO colapsa a cero al converger y
    /// protege tanto pesos como sesgos (`b_enc`, `b_intent`, `b_stance`).
    fn compute_fisher_diag(&self) -> Vec<f64> {
        let mut f_enc = vec![0.0; self.w0.w_enc.len()];
        let mut f_benc = vec![0.0; H_LAT];
        let mut f_dec = vec![0.0; self.w0.w_dec.len()];
        let mut f_rev = vec![0.0; self.w0.w_rev.len()];
        let mut f_int = vec![0.0; self.w0.w_intent.len()];
        let mut f_bint = vec![0.0; N_INTENTS];
        let mut f_st = vec![0.0; self.w0.w_stance.len()];
        let mut f_bst = vec![0.0; N_STANCES];
        let g_neutral = [0.5, 0.3, 0.2];
        let n = self.buffer.len().max(1) as f64;
        let beta = 1.8_f64;
        let beta2 = beta * beta;

        for item in &self.buffer {
            let mut x = [0.0; D_SEM];
            x.copy_from_slice(&item.x[..D_SEM]);
            let (h, z, l_int, l_st) = self.forward_branch(&x, None, beta, &g_neutral, false);
            let p_int = softmax_slice(&l_int);
            let p_st = softmax_slice(&l_st);

            let mut cur_h = [0.0; H_LAT];
            // 1) Fisher exacto de cabeza de intención (E_{y~p}[(β(p_c - 1_{y=c}))²] = β² p_c (1 - p_c))
            for c in 0..N_INTENTS {
                let var_c = beta2 * p_int[c] * (1.0 - p_int[c]);
                f_bint[c] += var_c;
                let row = c * H_LAT;
                for r in 0..H_LAT {
                    f_int[row + r] += var_c * h[r] * h[r];
                    cur_h[r] += var_c * self.w0.w_intent[row + r].powi(2);
                }
            }
            // 2) Fisher exacto de cabeza de postura epistémica
            for c in 0..N_STANCES {
                let var_c = 1.2 * beta2 * p_st[c] * (1.0 - p_st[c]);
                f_bst[c] += var_c;
                let row = c * H_LAT;
                for r in 0..H_LAT {
                    f_st[row + r] += var_c * h[r] * h[r];
                    cur_h[r] += var_c * self.w0.w_stance[row + r].powi(2);
                }
            }
            // 3) Curvatura Gauss-Newton del reconstructor de ciclo Φ_rev(z) ≈ x y decodificador W_dec
            let inv_xn2 = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-8);
            let mut cur_z = [1.6 / D_SEM as f64; D_SEM];
            for i in 0..D_SEM {
                let row = i * D_SEM;
                for j in 0..D_SEM {
                    f_rev[row + j] += 1.6 * inv_xn2 * z[j] * z[j];
                    cur_z[j] += 1.6 * inv_xn2 * self.w0.w_rev[row + j].powi(2);
                }
            }
            for d in 0..D_SEM {
                let row = d * H_LAT;
                for r in 0..H_LAT {
                    f_dec[row + r] += cur_z[d] * h[r] * h[r];
                    cur_h[r] += cur_z[d] * self.w0.w_dec[row + r].powi(2);
                }
            }
            // 4) Propagación diagonal de curvatura hacia w_enc y b_enc
            for r in 0..H_LAT {
                let dtanh = (1.0 - h[r] * h[r]) * beta;
                let cur_a = cur_h[r] * dtanh * dtanh;
                f_benc[r] += cur_a;
                let row = r * D_SEM;
                for c in 0..D_SEM {
                    f_enc[row + c] += cur_a * x[c] * x[c];
                }
            }
        }

        let mut out = Vec::with_capacity(self.w0.param_count());
        for blk in [&f_enc, &f_benc, &f_dec, &f_rev, &f_int, &f_bint, &f_st, &f_bst] {
            for &v in blk.iter() {
                out.push((v / n).min(25.0));
            }
        }
        out
    }

    fn recalibrate_yofe(&mut self) {
        if self.buffer.len() < 12 {
            return;
        }
        let g_neutral = [0.5, 0.3, 0.2];
        let use_refute = self.delta_mode == DeltaMode::AdversarialCrossRefute;
        let mut xs = Vec::with_capacity(self.buffer.len() * 2);
        let mut ys = Vec::with_capacity(self.buffer.len() * 2);
        for item in &self.buffer {
            let mut x = [0.0; D_SEM];
            let mut x_cf = [0.0; D_SEM];
            x.copy_from_slice(&item.x[..D_SEM]);
            x_cf.copy_from_slice(&item.x_cf[..D_SEM]);
            let (_, z, _, l_st) = self.forward_branch(&x, None, 1.8, &g_neutral, false);
            let (_, z_cf, _, l_st_cf) = self.forward_branch(&x_cf, None, 1.8, &g_neutral, false);
            let (cyc, _) = self.cycle_error(&z, &x);
            let p_st = softmax_slice(&l_st);
            let pred = argmax_slice(&p_st);
            let max_p = p_st[pred];
            let cf_s = (rel_dist(&z, &z_cf)
                + 0.35 * l1_prob_diff(&p_st, &softmax_slice(&l_st_cf)))
            .clamp(0.0, 1.5);
            let refute_v = if use_refute {
                let (_, zb, _, _) = self.forward_branch(&x, None, 1.6, &[0.1, 0.8, 0.1], true);
                let (cyb, _) = self.cycle_error(&zb, &x);
                let dis = rel_dist(&z, &zb);
                (dis * (-2.2 * (cyb - cyc).max(0.0)).exp()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let overconf = (1.0 - norm_entropy(&p_st)).clamp(0.0, 1.0);
            let dogma_est = (overconf * ((cyc - 0.25).max(0.0).clamp(0.0, 1.0)).max(((0.30 - cf_s).max(0.0)) / 0.30))
                .clamp(0.0, 1.0);
            let ok = if pred == item.stance_idx { 1.0 } else { 0.0 };
            // Muestra real en modo abierto (grounded = 0.0) con rasgos reales de refutación y dogma
            xs.push([1.0, max_p, cyc.clamp(0.0, 2.0), refute_v, cf_s, dogma_est, 0.0]);
            ys.push(ok);
            // Muestra pareada en modo fundamentado (grounded = 1.0)
            xs.push([1.0, max_p, cyc.clamp(0.0, 2.0), refute_v, cf_s, dogma_est, 1.0]);
            ys.push(0.95_f64.max(ok));
        }
        // Gradiente logístico L2 calibrado en rasgos reales conservando monotonicidad anti-refutación/anti-dogma
        let lr = 0.08;
        let lam = 0.08;
        let prior = [-0.2, 2.4, -1.2, -0.9, 0.8, -1.8, 0.9];
        for _ in 0..30 {
            let mut g = [0.0; 7];
            for (feat, &y) in xs.iter().zip(&ys) {
                let z: f64 = self.cal_w.iter().zip(feat).map(|(a, b)| a * b).sum();
                let p = sig(z.clamp(-15.0, 15.0));
                for k in 0..7 {
                    g[k] += (p - y) * feat[k];
                }
            }
            let inv_n = 1.0 / xs.len() as f64;
            for k in 0..7 {
                self.cal_w[k] -= lr * (g[k] * inv_n + lam * (self.cal_w[k] - prior[k]));
            }
            // Monotonicidad protectora estricta: mayor refutación o dogma NUNCA puede subir confianza
            self.cal_w[3] = self.cal_w[3].min(-0.20);
            self.cal_w[5] = self.cal_w[5].min(-0.35);
        }
    }

    /// Evalúa CE media de clasificación (`intent + stance`) y error de ciclo sobre un conjunto dado (para medir olvido EWC).
    pub fn eval_loss_on(&self, items: &[VerifiedEpisode]) -> (f64, f64, f64) {
        if items.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let g_neutral = [0.5, 0.3, 0.2];
        let mut ce = 0.0;
        let mut int_ok = 0usize;
        let mut st_ok = 0usize;
        for item in items {
            let mut x = [0.0; D_SEM];
            x.copy_from_slice(&item.x[..D_SEM]);
            let (_, _, l_int, l_st) = self.forward_branch(&x, None, 1.8, &g_neutral, false);
            let p_int = softmax_slice(&l_int);
            let p_st = softmax_slice(&l_st);
            ce += -p_int[item.intent_idx].max(1e-9).ln() - p_st[item.stance_idx].max(1e-9).ln();
            if argmax_slice(&p_int) == item.intent_idx {
                int_ok += 1;
            }
            if argmax_slice(&p_st) == item.stance_idx {
                st_ok += 1;
            }
        }
        let n = items.len() as f64;
        (ce / n, int_ok as f64 / n, st_ok as f64 / n)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SleepConsolidationReport {
    pub episodes_consolidated: usize,
    pub loss_before: f64,
    pub loss_after: f64,
    pub fisher_mean: f64,
    pub ewc_mu: f64,
    pub consolidations: usize,
}

pub fn stance_to_idx(s: EpistemicStance) -> usize {
    match s {
        EpistemicStance::Assert => 0,
        EpistemicStance::Qualified => 1,
        EpistemicStance::ConflictDetected => 2,
        EpistemicStance::NeedClarification => 3,
        EpistemicStance::Abstain => 4,
    }
}

pub fn stance_from_idx(i: usize) -> EpistemicStance {
    match i {
        0 => EpistemicStance::Assert,
        1 => EpistemicStance::Qualified,
        2 => EpistemicStance::ConflictDetected,
        3 => EpistemicStance::NeedClarification,
        _ => EpistemicStance::Abstain,
    }
}

#[inline]
fn sig(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softmax3(v: [f64; 3]) -> [f64; 3] {
    let mx = v[0].max(v[1]).max(v[2]);
    let e = [(v[0] - mx).exp(), (v[1] - mx).exp(), (v[2] - mx).exp()];
    let s = (e[0] + e[1] + e[2]).max(1e-12);
    [e[0] / s, e[1] / s, e[2] / s]
}

pub fn softmax_slice(v: &[f64]) -> Vec<f64> {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = v.iter().map(|&x| (x - mx).clamp(-40.0, 0.0).exp()).collect();
    let s: f64 = ex.iter().sum::<f64>().max(1e-12);
    ex.into_iter().map(|e| e / s).collect()
}

pub fn argmax_slice(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn norm_entropy(p: &[f64]) -> f64 {
    let k = (p.len().max(2) as f64).ln();
    let h: f64 = -p
        .iter()
        .filter(|&&x| x > 1e-12)
        .map(|&x| x * x.ln())
        .sum::<f64>();
    (h / k).clamp(0.0, 1.0)
}

pub fn rel_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut d2 = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        d2 += (x - y) * (x - y);
        na += x * x;
        nb += y * y;
    }
    d2.sqrt() / (na.sqrt() + nb.sqrt() + 1e-6)
}

pub fn cosine_slices(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let den = (na.sqrt() * nb.sqrt()).max(1e-9);
    (dot / den).clamp(-1.0, 1.0)
}

fn l1_prob_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f64>() * 0.5
}
