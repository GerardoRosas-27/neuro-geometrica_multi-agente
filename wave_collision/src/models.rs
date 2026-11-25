//! 768-Wave Self-Calibrating Colliding-Wave Interferometric Network (`WaveCollisionCore`)
//! and capacity-matched conventional neural baseline (`MlpCore`, 3,832 core params vs 3,328 wave).
//!
//! Architecture highlights:
//!   - ZERO dense weight matrix `W_pair` inside `WaveCollisionCore`:
//!     100% of the 3,328 core parameters are wave frequencies (2,304) and phases/angles (1,024)!
//!   - Stage 1 (512 primary colliding waves = 16 banks x 16 collisions):
//!       C_m(u, v) = cos(beta_m) * cos(s_g * w1_m . (v - u) + phi1_m + delta_g)
//!                 + sin(beta_m) * cos(s_g * w1_m . u + phi1_m) * cos(s_g * w2_m . v + phi2_m + delta_g)
//!       h_g(u, v) = (1 / sqrt(16)) * sum_{k=0..15} C_{16g+k}(u, v)
//!   - Stage 2 (256 secondary coupled heterodyne wave-wave colliding waves, exact zero vacuum):
//!       z_g(u, v) = h_g(u, v) + (rho_g / sqrt(16)) * sum_{j=0..15} sin(kappa_{g,j} * h_j) * cos(kappa_{j,g} * h_g + psi_{g,j})
//!   - Resonant Cavity Auto-Calibration (+48 knobs: s_g[16], delta_g[16], rho_g[16]) while all
//!     3,328 core wave frequencies & phases stay 100% bit-exact frozen (verified via IEEE-754 checksum).
//!   - Exact 1:1 fairness on `MlpCore`: matching +48 FiLM auto-calibration knobs
//!     (film_in_scale[8], film_in_shift[8], film_z_scale[16], film_z_shift[16]).

use crate::domains::DomainData;
use crate::rng::{checksum_f64_slices, riemann_zeros_256, Rng64};
use std::f64::consts::PI;

pub const LATENT_D: usize = 4;
pub const N_BANKS: usize = 16;
pub const K_PER_BANK: usize = 16;
pub const N_COLLISIONS: usize = N_BANKS * K_PER_BANK; // 256 collisions = 512 primary waves
pub const REL_DIM: usize = N_BANKS; // 16 resonant channels
pub const MLP_H1: usize = 64;
pub const MLP_H2: usize = 40;

pub const R_MIN: f64 = 0.25;
pub const R_MAX: f64 = 1.70;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumKind {
    Adaptive,
    RiemannFixed,
    FourierFixed,
    PoissonFixed,
    DysonFixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairingMode {
    FullCollision,
    AdditiveOnly,
    PermutedPairing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreTrainMode {
    /// Train the full core on source domain (or from scratch on target domain); cal knobs locked at identity.
    FullTrain,
    /// Freeze ALL core wave parameters AND lock cal knobs at identity (0 core/cal params updated).
    FullyFrozen,
    /// Freeze ALL 3,328 core wave parameters bit-exact; train ONLY the 48 resonant auto-calibration knobs.
    AutoCalOnly,
}

#[derive(Clone, Debug)]
pub struct AdamBuf {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl AdamBuf {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    pub fn step(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        lr: f64,
        wd: f64,
        t: usize,
        grad_clip: f64,
    ) {
        let b1: f64 = 0.9;
        let b2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let n2: f64 = grads.iter().map(|g| g * g).sum();
        let scale = if n2 > grad_clip * grad_clip {
            grad_clip / n2.sqrt()
        } else {
            1.0
        };
        for i in 0..params.len() {
            let g = grads[i] * scale;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            params[i] -= lr * (mh / (vh.sqrt() + eps) + wd * params[i]);
        }
    }

    /// Damped harmonic spring step anchored at `anchor` (no Adam 1/sqrt(v) noise amplification):
    /// F = -(g + k_spring * (p - anchor)), keeping knobs smoothly locked when residual gradient is noise
    /// while gliding cleanly into resonance under coherent domain mismatch.
    pub fn step_spring(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        lr: f64,
        k_spring: f64,
        anchor: f64,
        grad_clip: f64,
    ) {
        let n2: f64 = grads.iter().map(|g| g * g).sum();
        let scale = if n2 > grad_clip * grad_clip {
            grad_clip / n2.sqrt()
        } else {
            1.0
        };
        for i in 0..params.len() {
            let g_eff = grads[i] * scale + k_spring * (params[i] - anchor);
            self.m[i] = 0.80 * self.m[i] + g_eff;
            params[i] -= lr * self.m[i];
        }
    }
}

/// Linear sensory encoder E_dom: R^obs_dim -> R^4.
#[derive(Clone, Debug)]
pub struct DomainEncoder {
    pub obs_dim: usize,
    pub w: Vec<f64>,
    pub b: Vec<f64>,
}

impl DomainEncoder {
    pub fn new(obs_dim: usize, rng: &mut Rng64) -> Self {
        let scale = (1.0 / (obs_dim as f64)).sqrt() * 0.85;
        let w = (0..(LATENT_D * obs_dim))
            .map(|_| rng.normal() * scale)
            .collect();
        let b = vec![0.0; LATENT_D];
        Self { obs_dim, w, b }
    }

    pub fn param_count(&self) -> usize {
        self.w.len() + self.b.len()
    }

    #[inline]
    pub fn forward(&self, x: &[f64]) -> [f64; LATENT_D] {
        let mut u = [0.0; LATENT_D];
        for k in 0..LATENT_D {
            let row = &self.w[k * self.obs_dim..(k + 1) * self.obs_dim];
            let mut acc = self.b[k];
            for j in 0..self.obs_dim {
                acc += row[j] * x[j];
            }
            u[k] = acc;
        }
        u
    }

    #[inline]
    pub fn backward(&self, x: &[f64], du: &[f64; LATENT_D], gw: &mut [f64], gb: &mut [f64]) {
        for k in 0..LATENT_D {
            let dk = du[k];
            gb[k] += dk;
            let row = &mut gw[k * self.obs_dim..(k + 1) * self.obs_dim];
            for j in 0..self.obs_dim {
                row[j] += dk * x[j];
            }
        }
    }
}

/// Linear readout head H_dom: R^16 -> (reg_dim, cls_dim).
#[derive(Clone, Debug)]
pub struct DomainHead {
    pub reg_dim: usize,
    pub cls_dim: usize,
    pub w_reg: Vec<f64>,
    pub b_reg: Vec<f64>,
    pub w_cls: Vec<f64>,
    pub b_cls: Vec<f64>,
}

impl DomainHead {
    pub fn new(reg_dim: usize, cls_dim: usize, rng: &mut Rng64) -> Self {
        let s = (1.0 / (REL_DIM as f64)).sqrt() * 0.5;
        Self {
            reg_dim,
            cls_dim,
            w_reg: (0..(reg_dim * REL_DIM)).map(|_| rng.normal() * s).collect(),
            b_reg: vec![0.0; reg_dim],
            w_cls: (0..(cls_dim * REL_DIM)).map(|_| rng.normal() * s).collect(),
            b_cls: vec![0.0; cls_dim],
        }
    }

    pub fn param_count(&self) -> usize {
        self.w_reg.len() + self.b_reg.len() + self.w_cls.len() + self.b_cls.len()
    }

    pub fn forward(&self, z: &[f64; REL_DIM]) -> (Vec<f64>, Vec<f64>) {
        let mut reg = vec![0.0; self.reg_dim];
        for r in 0..self.reg_dim {
            let mut acc = self.b_reg[r];
            let row = &self.w_reg[r * REL_DIM..(r + 1) * REL_DIM];
            for j in 0..REL_DIM {
                acc += row[j] * z[j];
            }
            reg[r] = acc;
        }
        let mut logits = vec![0.0; self.cls_dim];
        for c in 0..self.cls_dim {
            let mut acc = self.b_cls[c];
            let row = &self.w_cls[c * REL_DIM..(c + 1) * REL_DIM];
            for j in 0..REL_DIM {
                acc += row[j] * z[j];
            }
            logits[c] = acc;
        }
        (reg, logits)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn loss_and_backward(
        &self,
        z: &[f64; REL_DIM],
        reg_pred: &[f64],
        logits: &[f64],
        reg_true: &[f64],
        cls_true: usize,
        w_mse: f64,
        w_ce: f64,
        scale: f64,
        gw_reg: &mut [f64],
        gb_reg: &mut [f64],
        gw_cls: &mut [f64],
        gb_cls: &mut [f64],
    ) -> (f64, [f64; REL_DIM]) {
        let mut loss = 0.0;
        let mut dz = [0.0; REL_DIM];

        if self.reg_dim > 0 && w_mse > 0.0 {
            let inv_r = 1.0 / (self.reg_dim as f64);
            for r in 0..self.reg_dim {
                let err = reg_pred[r] - reg_true[r];
                loss += w_mse * 0.5 * err * err * inv_r;
                let g = scale * w_mse * err * inv_r;
                gb_reg[r] += g;
                let row_g = &mut gw_reg[r * REL_DIM..(r + 1) * REL_DIM];
                let row_w = &self.w_reg[r * REL_DIM..(r + 1) * REL_DIM];
                for j in 0..REL_DIM {
                    row_g[j] += g * z[j];
                    dz[j] += g * row_w[j];
                }
            }
        }

        if self.cls_dim > 0 && w_ce > 0.0 {
            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut probs = vec![0.0; self.cls_dim];
            let mut sum_e = 0.0;
            for c in 0..self.cls_dim {
                let e = (logits[c] - max_l).exp();
                probs[c] = e;
                sum_e += e;
            }
            for c in 0..self.cls_dim {
                probs[c] /= sum_e;
            }
            loss += -w_ce * probs[cls_true].max(1e-15).ln();
            for c in 0..self.cls_dim {
                let target = if c == cls_true { 1.0 } else { 0.0 };
                let g = scale * w_ce * (probs[c] - target);
                gb_cls[c] += g;
                let row_g = &mut gw_cls[c * REL_DIM..(c + 1) * REL_DIM];
                let row_w = &self.w_cls[c * REL_DIM..(c + 1) * REL_DIM];
                for j in 0..REL_DIM {
                    row_g[j] += g * z[j];
                    dz[j] += g * row_w[j];
                }
            }
        }

        (loss, dz)
    }
}

/// Canonical 16D domain-agnostic pair invariants (identical for WaveCollisionCore and MlpCore).
fn canonical_pair_invariants(u: &[f64; LATENT_D], v: &[f64; LATENT_D]) -> [f64; REL_DIM] {
    let mut t = [0.0; REL_DIM];
    for d in 0..LATENT_D {
        let diff = v[d] - u[d];
        t[d] = diff;
        t[4 + d] = 2.0 * (-0.75 * diff * diff).exp() - 1.0;
        t[8 + d] = (0.75 * (u[d] + v[d])).cos();
    }
    t[12] = (u[0] * v[1] - u[1] * v[0]).tanh();
    t[13] = (u[2] * v[3] - u[3] * v[2]).tanh();
    t[14] = (u[0] * v[2] - u[2] * v[0]).tanh();
    t[15] = (u[1] * v[3] - u[3] * v[1]).tanh();
    t
}

/// Full-rank VICReg variance-covariance decorrelation on batch Z (identical for Wave & MLP).
pub fn batch_var_cov_loss_and_grad(
    zs: &[[f64; REL_DIM]],
    lambda_var: f64,
    lambda_cov: f64,
    dzs: &mut [[f64; REL_DIM]],
) -> f64 {
    let n = zs.len();
    if n < 2 || (lambda_var == 0.0 && lambda_cov == 0.0) {
        return 0.0;
    }
    let nf = n as f64;
    let mut mean = [0.0; REL_DIM];
    for z in zs {
        for r in 0..REL_DIM {
            mean[r] += z[r];
        }
    }
    for r in 0..REL_DIM {
        mean[r] /= nf;
    }

    let mut cov = [[0.0; REL_DIM]; REL_DIM];
    for z in zs {
        for r in 0..REL_DIM {
            let dr = z[r] - mean[r];
            for s in r..REL_DIM {
                cov[r][s] += dr * (z[s] - mean[s]);
            }
        }
    }
    for r in 0..REL_DIM {
        for s in r..REL_DIM {
            cov[r][s] /= nf;
            cov[s][r] = cov[r][s];
        }
    }

    let mut loss = 0.0;
    let inv_r = 1.0 / (REL_DIM as f64);
    let inv_off = 1.0 / ((REL_DIM * (REL_DIM - 1)) as f64);
    let mut dcov = [[0.0; REL_DIM]; REL_DIM];

    if lambda_var > 0.0 {
        let target_var = 0.80;
        let lambda_mean = 8.0 * lambda_var;
        let mean_grad_scale = 2.0 * lambda_mean * inv_r / nf;
        for r in 0..REL_DIM {
            let err = cov[r][r] - target_var;
            loss += lambda_var * inv_r * err * err;
            dcov[r][r] += 2.0 * lambda_var * inv_r * err;
            loss += lambda_mean * inv_r * mean[r] * mean[r];
        }
        for dz in dzs.iter_mut() {
            for r in 0..REL_DIM {
                dz[r] += mean_grad_scale * mean[r];
            }
        }
    }
    if lambda_cov > 0.0 {
        for r in 0..REL_DIM {
            for s in 0..REL_DIM {
                if r == s {
                    continue;
                }
                let c = cov[r][s];
                loss += lambda_cov * inv_off * c * c;
                dcov[r][s] += 2.0 * lambda_cov * inv_off * c;
            }
        }
    }

    let scale = 2.0 / nf;
    for (i, z) in zs.iter().enumerate() {
        for r in 0..REL_DIM {
            let mut acc = 0.0;
            for s in 0..REL_DIM {
                acc += dcov[r][s] * (z[s] - mean[s]);
            }
            dzs[i][r] += scale * acc;
        }
    }
    loss
}

fn make_derangement(n: usize, rng: &mut Rng64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    loop {
        rng.shuffle(&mut perm);
        if perm.iter().enumerate().all(|(i, &p)| i != p) {
            return perm;
        }
    }
}

fn sample_dyson_radii(n: usize, r_min: f64, r_max: f64, rng: &mut Rng64) -> Vec<f64> {
    let mut spacings = Vec::with_capacity(n.saturating_sub(1));
    let c_const = 4.0 / PI;
    let peak = (1.0 / c_const) * (-1.0f64).exp();
    for _ in 0..n.saturating_sub(1) {
        let s = loop {
            let x = rng.uniform_range(0.0, 3.2);
            let y = rng.uniform_range(0.0, peak * 1.05);
            if y <= x * x * (-c_const * x * x).exp() {
                break x;
            }
        };
        spacings.push(s);
    }
    let mut cum = vec![0.0; n];
    for i in 1..n {
        cum[i] = cum[i - 1] + spacings[i - 1];
    }
    let span = cum[n - 1].max(1e-9);
    cum.into_iter()
        .map(|v| r_min + (r_max - r_min) * (v / span))
        .collect()
}

fn sample_poisson_radii(n: usize, r_min: f64, r_max: f64, rng: &mut Rng64) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..n).map(|_| rng.uniform_range(r_min, r_max)).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    vals
}

fn make_fourier_lattice_4d(n: usize, r_min: f64, r_max: f64) -> Vec<[f64; LATENT_D]> {
    let mut pts: Vec<([i32; 4], i32)> = Vec::new();
    for a in -3..=3 {
        for b in -3..=3 {
            for c in -3..=3 {
                for d in -3..=3 {
                    let n2 = a * a + b * b + c * c + d * d;
                    if n2 == 0 {
                        continue;
                    }
                    let first_nz = [a, b, c, d].into_iter().find(|&v| v != 0).unwrap();
                    if first_nz < 0 {
                        continue;
                    }
                    pts.push(([a, b, c, d], n2));
                }
            }
        }
    }
    pts.sort_by_key(|&(v, n2)| (n2, v));
    pts.truncate(n);
    let max_r = (pts.last().unwrap().1 as f64).sqrt().max(1.0);
    let min_r = (pts.first().unwrap().1 as f64).sqrt();
    let span = (max_r - min_r).max(1e-6);
    pts.into_iter()
        .map(|(v, n2)| {
            let r = (n2 as f64).sqrt();
            let target_r = r_min + (r_max - r_min) * (r - min_r) / span;
            let scale = target_r / r;
            [
                v[0] as f64 * scale,
                v[1] as f64 * scale,
                v[2] as f64 * scale,
                v[3] as f64 * scale,
            ]
        })
        .collect()
}

/// 768-Wave Self-Calibrating Colliding-Wave Core (3,328 core params = 100% frequencies & phases,
/// ZERO dense weight matrix `W_pair`!) + 48 resonant cavity auto-calibration knobs.
#[derive(Clone, Debug)]
pub struct WaveCollisionCore {
    pub kind: SpectrumKind,
    pub pairing: PairingMode,
    pub zero_phase_lock: bool,
    pub disable_stage2: bool,

    // Stage 1: 256 primary colliding wave pairs (= 512 waves), 2,816 params
    pub omega1: Vec<f64>, // 256 * 4 = 1,024 (past/incident wavevectors)
    pub omega2: Vec<f64>, // 256 * 4 = 1,024 (future/counter-propagating wavevectors)
    pub phi1: Vec<f64>,   // 256 (incident phases)
    pub phi2: Vec<f64>,   // 256 (future phases)
    pub beta: Vec<f64>,   // 256 (conjugate vs heterodyne mixing angle)

    // Stage 2: 256 secondary wave-wave colliding waves, 512 params
    pub kappa: Vec<f64>, // 16 * 16 = 256 (secondary collision frequencies)
    pub psi: Vec<f64>,   // 16 * 16 = 256 (secondary collision phases)

    // Resonant Cavity Auto-Calibration knobs (48 params; locked at identity during Domain A & FullyFrozen)
    pub cal_scale: Vec<f64>, // 16 (bank resonant wavelength scaling, init 1.0)
    pub cal_phase: Vec<f64>, // 16 (bank resonant phase detuning, init 0.0)
    pub cal_rho: Vec<f64>,   // 16 (bank Stage-2 coupling gain, init 1.0)

    pub perm_pi: Vec<usize>,
    pub uses_riemann_zeros: bool,
}

/// Precomputed per-step sin/cos of beta[256] (independent of sample u, v).
#[derive(Clone, Debug)]
pub struct StepTrigCache {
    pub cos_beta: [f64; N_COLLISIONS],
    pub sin_beta: [f64; N_COLLISIONS],
}

impl StepTrigCache {
    pub fn from_core(core: &WaveCollisionCore) -> Self {
        let mut cos_beta = [0.0; N_COLLISIONS];
        let mut sin_beta = [0.0; N_COLLISIONS];
        for m in 0..N_COLLISIONS {
            let (s, c) = core.beta[m].sin_cos();
            sin_beta[m] = s;
            cos_beta[m] = c;
        }
        Self { cos_beta, sin_beta }
    }
}

#[derive(Clone, Debug)]
pub struct WaveForwardCache {
    pub u: [f64; LATENT_D],
    pub v: [f64; LATENT_D],
    pub cos_diff: [f64; N_COLLISIONS],
    pub sin_diff: [f64; N_COLLISIONS],
    pub cos_u: [f64; N_COLLISIONS],
    pub sin_u: [f64; N_COLLISIONS],
    pub cos_v: [f64; N_COLLISIONS],
    pub sin_v: [f64; N_COLLISIONS],
    pub cos_u2: [f64; N_COLLISIONS],
    pub sin_u2: [f64; N_COLLISIONS],
    pub cos_v2: [f64; N_COLLISIONS],
    pub sin_v2: [f64; N_COLLISIONS],
    pub h: [f64; REL_DIM],
    pub sin_s2_a1: [f64; REL_DIM * REL_DIM],
    pub cos_s2_a1: [f64; REL_DIM * REL_DIM],
    pub sin_s2_a2: [f64; REL_DIM * REL_DIM],
    pub cos_s2_a2: [f64; REL_DIM * REL_DIM],
    pub z_rel: [f64; REL_DIM],
}

#[derive(Clone, Debug)]
pub struct WaveCoreGrads {
    pub omega1: Vec<f64>,
    pub omega2: Vec<f64>,
    pub phi1: Vec<f64>,
    pub phi2: Vec<f64>,
    pub beta: Vec<f64>,
    pub kappa: Vec<f64>,
    pub psi: Vec<f64>,
    pub cal_scale: Vec<f64>,
    pub cal_phase: Vec<f64>,
    pub cal_rho: Vec<f64>,
}

impl WaveCoreGrads {
    pub fn zeros() -> Self {
        Self {
            omega1: vec![0.0; N_COLLISIONS * LATENT_D],
            omega2: vec![0.0; N_COLLISIONS * LATENT_D],
            phi1: vec![0.0; N_COLLISIONS],
            phi2: vec![0.0; N_COLLISIONS],
            beta: vec![0.0; N_COLLISIONS],
            kappa: vec![0.0; REL_DIM * REL_DIM],
            psi: vec![0.0; REL_DIM * REL_DIM],
            cal_scale: vec![0.0; REL_DIM],
            cal_phase: vec![0.0; REL_DIM],
            cal_rho: vec![0.0; REL_DIM],
        }
    }

    pub fn clear(&mut self) {
        self.omega1.fill(0.0);
        self.omega2.fill(0.0);
        self.phi1.fill(0.0);
        self.phi2.fill(0.0);
        self.beta.fill(0.0);
        self.kappa.fill(0.0);
        self.psi.fill(0.0);
        self.cal_scale.fill(0.0);
        self.cal_phase.fill(0.0);
        self.cal_rho.fill(0.0);
    }
}

impl WaveCollisionCore {
    pub fn new(
        kind: SpectrumKind,
        pairing: PairingMode,
        zero_phase_lock: bool,
        disable_stage2: bool,
        rng: &mut Rng64,
    ) -> Self {
        let mut omega1 = vec![0.0; N_COLLISIONS * LATENT_D];
        let mut omega2 = vec![0.0; N_COLLISIONS * LATENT_D];
        let mut uses_riemann_zeros = false;

        // Shuffle mode-to-bank assignment so every bank gets full multi-scale frequency coverage
        // rather than bank 0 getting only tiny frequencies and bank 15 only high frequencies!
        let mut bank_order: Vec<usize> = (0..N_COLLISIONS).collect();
        rng.shuffle(&mut bank_order);

        match kind {
            SpectrumKind::Adaptive | SpectrumKind::PoissonFixed => {
                // Pure random Poisson initialization — ZERO Riemann zeros, ZERO primes!
                let r1 = sample_poisson_radii(N_COLLISIONS, R_MIN, R_MAX, rng);
                let r2 = sample_poisson_radii(N_COLLISIONS, R_MIN, R_MAX, rng);
                for (i, &m) in bank_order.iter().enumerate() {
                    let d1 = rng.unit_vec(LATENT_D);
                    let d2 = rng.unit_vec(LATENT_D);
                    for d in 0..LATENT_D {
                        omega1[m * LATENT_D + d] = d1[d] * r1[i];
                        omega2[m * LATENT_D + d] = d2[d] * r2[i];
                    }
                }
            }
            SpectrumKind::RiemannFixed => {
                uses_riemann_zeros = true;
                let rz = riemann_zeros_256();
                let g_min = rz[0];
                let g_max = rz[N_COLLISIONS - 1];
                let span = g_max - g_min;
                let mut order2: Vec<usize> = (0..N_COLLISIONS).collect();
                rng.shuffle(&mut order2);
                for i in 0..N_COLLISIONS {
                    let r = R_MIN + (R_MAX - R_MIN) * (rz[i] - g_min) / span;
                    let m1 = bank_order[i];
                    let m2 = order2[i];
                    let d1 = rng.unit_vec(LATENT_D);
                    let d2 = rng.unit_vec(LATENT_D);
                    for d in 0..LATENT_D {
                        omega1[m1 * LATENT_D + d] = d1[d] * r;
                        omega2[m2 * LATENT_D + d] = d2[d] * r;
                    }
                }
            }
            SpectrumKind::FourierFixed => {
                let lat = make_fourier_lattice_4d(N_COLLISIONS, R_MIN, R_MAX);
                let mut order2: Vec<usize> = (0..N_COLLISIONS).collect();
                rng.shuffle(&mut order2);
                for i in 0..N_COLLISIONS {
                    let m1 = bank_order[i];
                    let m2 = order2[i];
                    for d in 0..LATENT_D {
                        omega1[m1 * LATENT_D + d] = lat[i][d];
                        omega2[m2 * LATENT_D + d] = lat[i][d];
                    }
                }
            }
            SpectrumKind::DysonFixed => {
                let r1 = sample_dyson_radii(N_COLLISIONS, R_MIN, R_MAX, rng);
                let r2 = sample_dyson_radii(N_COLLISIONS, R_MIN, R_MAX, rng);
                for (i, &m) in bank_order.iter().enumerate() {
                    let d1 = rng.unit_vec(LATENT_D);
                    let d2 = rng.unit_vec(LATENT_D);
                    for d in 0..LATENT_D {
                        omega1[m * LATENT_D + d] = d1[d] * r1[i];
                        omega2[m * LATENT_D + d] = d2[d] * r2[i];
                    }
                }
            }
        }

        let (phi1, phi2, psi) = if zero_phase_lock {
            (
                vec![0.0; N_COLLISIONS],
                vec![0.0; N_COLLISIONS],
                vec![0.0; REL_DIM * REL_DIM],
            )
        } else {
            (
                (0..N_COLLISIONS).map(|_| rng.uniform_range(-PI, PI)).collect(),
                (0..N_COLLISIONS).map(|_| rng.uniform_range(-PI, PI)).collect(),
                (0..(REL_DIM * REL_DIM)).map(|_| rng.uniform_range(-PI, PI)).collect(),
            )
        };
        let beta = (0..N_COLLISIONS)
            .map(|_| rng.uniform_range(-PI, PI))
            .collect();
        let kappa = (0..(REL_DIM * REL_DIM))
            .map(|_| rng.normal() * 0.36)
            .collect();

        let cal_scale = vec![1.0; REL_DIM];
        let cal_phase = vec![0.0; REL_DIM];
        let cal_rho = vec![if disable_stage2 { 0.0 } else { 1.0 }; REL_DIM];
        let perm_pi = make_derangement(N_COLLISIONS, rng);

        Self {
            kind,
            pairing,
            zero_phase_lock,
            disable_stage2,
            omega1,
            omega2,
            phi1,
            phi2,
            beta,
            kappa,
            psi,
            cal_scale,
            cal_phase,
            cal_rho,
            perm_pi,
            uses_riemann_zeros,
        }
    }

    pub fn assert_no_riemann_injection(&self, attempted_riemann_or_prime_inject: bool) {
        if self.kind == SpectrumKind::Adaptive
            && (attempted_riemann_or_prime_inject || self.uses_riemann_zeros)
        {
            panic!("PROTOCOL VIOLATION: Adaptive WaveCollisionCore forbids Riemann zeros or prime frequency injection!");
        }
    }

    /// Core wave parameters (excluding the 48 auto-calibration knobs): 3,328 exact.
    pub fn core_param_count(&self) -> usize {
        self.omega1.len()
            + self.omega2.len()
            + self.phi1.len()
            + self.phi2.len()
            + self.beta.len()
            + self.kappa.len()
            + self.psi.len()
    }

    pub fn autocal_param_count(&self) -> usize {
        self.cal_scale.len() + self.cal_phase.len() + self.cal_rho.len()
    }

    /// Exact IEEE-754 checksum over all 3,328 core wave frequencies & phases.
    pub fn frozen_waves_checksum(&self) -> u64 {
        checksum_f64_slices(&[
            &self.omega1,
            &self.omega2,
            &self.phi1,
            &self.phi2,
            &self.beta,
            &self.kappa,
            &self.psi,
        ])
    }

    /// Full checksum including the 48 auto-calibration knobs (for FullyFrozen check).
    pub fn full_checksum(&self) -> u64 {
        checksum_f64_slices(&[
            &self.omega1,
            &self.omega2,
            &self.phi1,
            &self.phi2,
            &self.beta,
            &self.kappa,
            &self.psi,
            &self.cal_scale,
            &self.cal_phase,
            &self.cal_rho,
        ])
    }

    pub fn reset_autocal(&mut self) {
        self.cal_scale.fill(1.0);
        self.cal_phase.fill(0.0);
        self.cal_rho.fill(if self.disable_stage2 { 0.0 } else { 1.0 });
    }

    pub fn with_scrambled_phases(&self, rng: &mut Rng64) -> Self {
        let mut c = self.clone();
        for p in &mut c.phi1 {
            *p = rng.uniform_range(-PI, PI);
        }
        for p in &mut c.phi2 {
            *p = rng.uniform_range(-PI, PI);
        }
        for p in &mut c.psi {
            *p = rng.uniform_range(-PI, PI);
        }
        c
    }

    pub fn forward(&self, u: [f64; LATENT_D], v: [f64; LATENT_D]) -> WaveForwardCache {
        let trig = StepTrigCache::from_core(self);
        self.forward_with_trig(u, v, &trig)
    }

    pub fn forward_with_trig(
        &self,
        u: [f64; LATENT_D],
        v: [f64; LATENT_D],
        trig: &StepTrigCache,
    ) -> WaveForwardCache {
        let mut cos_diff = [0.0; N_COLLISIONS];
        let mut sin_diff = [0.0; N_COLLISIONS];
        let mut cos_u = [0.0; N_COLLISIONS];
        let mut sin_u = [0.0; N_COLLISIONS];
        let mut cos_v = [0.0; N_COLLISIONS];
        let mut sin_v = [0.0; N_COLLISIONS];
        let mut cos_u2 = [0.0; N_COLLISIONS];
        let mut sin_u2 = [0.0; N_COLLISIONS];
        let mut cos_v2 = [0.0; N_COLLISIONS];
        let mut sin_v2 = [0.0; N_COLLISIONS];
        let mut h = [0.0; REL_DIM];

        let inv_sqrt_k = 1.0 / (K_PER_BANK as f64).sqrt();
        let mut diff = [0.0; LATENT_D];
        for d in 0..LATENT_D {
            diff[d] = v[d] - u[d];
        }

        match self.pairing {
            PairingMode::FullCollision => {
                for g in 0..N_BANKS {
                    let sg = self.cal_scale[g];
                    let dg = self.cal_phase[g];
                    let mut acc = 0.0;
                    for k in 0..K_PER_BANK {
                        let m = g * K_PER_BANK + k;
                        let w1 = &self.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                        let w2 = &self.omega2[m * LATENT_D..(m + 1) * LATENT_D];
                        let mut dot_diff = 0.0;
                        let mut dot_u = 0.0;
                        let mut dot_v = 0.0;
                        for d in 0..LATENT_D {
                            dot_diff += w1[d] * diff[d];
                            dot_u += w1[d] * u[d];
                            dot_v += w2[d] * v[d];
                        }
                        let ad = sg * dot_diff + self.phi1[m] + dg;
                        let au = sg * dot_u + self.phi1[m];
                        let av = sg * dot_v + self.phi2[m] + dg;
                        let (sd, cd) = ad.sin_cos();
                        let (su, cu) = au.sin_cos();
                        let (sv, cv) = av.sin_cos();
                        cos_diff[m] = cd;
                        sin_diff[m] = sd;
                        cos_u[m] = cu;
                        sin_u[m] = su;
                        cos_v[m] = cv;
                        sin_v[m] = sv;
                        acc += trig.cos_beta[m] * cd + trig.sin_beta[m] * cu * cv;
                    }
                    h[g] = acc * inv_sqrt_k;
                }
            }
            PairingMode::AdditiveOnly => {
                // Strictly separable non-colliding waves: zero multiplicative/difference interaction!
                for g in 0..N_BANKS {
                    let sg = self.cal_scale[g];
                    let dg = self.cal_phase[g];
                    let mut acc = 0.0;
                    for k in 0..K_PER_BANK {
                        let m = g * K_PER_BANK + k;
                        let w1 = &self.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                        let w2 = &self.omega2[m * LATENT_D..(m + 1) * LATENT_D];
                        let mut w1u = 0.0;
                        let mut w1v = 0.0;
                        let mut w2u = 0.0;
                        let mut w2v = 0.0;
                        for d in 0..LATENT_D {
                            w1u += w1[d] * u[d];
                            w1v += w1[d] * v[d];
                            w2u += w2[d] * u[d];
                            w2v += w2[d] * v[d];
                        }
                        let au1 = sg * w1u + self.phi1[m];
                        let av1 = sg * w1v + self.phi1[m] + dg;
                        let au2 = sg * w2u + self.phi2[m];
                        let av2 = sg * w2v + self.phi2[m] + dg;
                        let (su1, cu1) = au1.sin_cos();
                        let (sv1, cv1) = av1.sin_cos();
                        let (su2, cu2) = au2.sin_cos();
                        let (sv2, cv2) = av2.sin_cos();
                        cos_u[m] = cu1;
                        sin_u[m] = su1;
                        cos_v[m] = cv1;
                        sin_v[m] = sv1;
                        cos_u2[m] = cu2;
                        sin_u2[m] = su2;
                        cos_v2[m] = cv2;
                        sin_v2[m] = sv2;
                        acc += 0.5 * trig.cos_beta[m] * (cv1 + cu1)
                            + 0.5 * trig.sin_beta[m] * (cv2 - cu2);
                    }
                    h[g] = acc * inv_sqrt_k;
                }
            }
            PairingMode::PermutedPairing => {
                for g in 0..N_BANKS {
                    let sg = self.cal_scale[g];
                    let dg = self.cal_phase[g];
                    let mut acc = 0.0;
                    for k in 0..K_PER_BANK {
                        let m = g * K_PER_BANK + k;
                        let p = self.perm_pi[m];
                        let w1 = &self.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                        let w1p = &self.omega1[p * LATENT_D..(p + 1) * LATENT_D];
                        let w2p = &self.omega2[p * LATENT_D..(p + 1) * LATENT_D];
                        let mut w1v = 0.0;
                        let mut w1pu = 0.0;
                        let mut w1u = 0.0;
                        let mut w2pv = 0.0;
                        for d in 0..LATENT_D {
                            w1v += w1[d] * v[d];
                            w1pu += w1p[d] * u[d];
                            w1u += w1[d] * u[d];
                            w2pv += w2p[d] * v[d];
                        }
                        let ad = sg * (w1v - w1pu) + self.phi1[m] + dg;
                        let au = sg * w1u + self.phi1[m];
                        let av = sg * w2pv + self.phi2[m] + dg;
                        let (sd, cd) = ad.sin_cos();
                        let (su, cu) = au.sin_cos();
                        let (sv, cv) = av.sin_cos();
                        cos_diff[m] = cd;
                        sin_diff[m] = sd;
                        cos_u[m] = cu;
                        sin_u[m] = su;
                        cos_v[m] = cv;
                        sin_v[m] = sv;
                        acc += trig.cos_beta[m] * cd + trig.sin_beta[m] * cu * cv;
                    }
                    h[g] = acc * inv_sqrt_k;
                }
            }
        }

        // Stage 2: 256 secondary coupled heterodyne wave-wave colliding waves (h_g collides with h_j),
        // exact zero vacuum: h = 0 => sin(0) * cos(psi) = 0 with zero DC bias.
        let mut sin_s2_a1 = [0.0; REL_DIM * REL_DIM];
        let mut cos_s2_a1 = [0.0; REL_DIM * REL_DIM];
        let mut sin_s2_a2 = [0.0; REL_DIM * REL_DIM];
        let mut cos_s2_a2 = [0.0; REL_DIM * REL_DIM];
        let mut z_rel = h;
        let active_s2 = !self.disable_stage2 && self.pairing != PairingMode::AdditiveOnly;
        if active_s2 {
            let inv_sqrt_r = 1.0 / (REL_DIM as f64).sqrt();
            for g in 0..REL_DIM {
                let rho = self.cal_rho[g];
                let hg = h[g];
                let mut sum2 = 0.0;
                let base = g * REL_DIM;
                for j in 0..REL_DIM {
                    let idx = base + j;
                    let tidx = j * REL_DIM + g;
                    let a1 = self.kappa[idx] * h[j];
                    let a2 = self.kappa[tidx] * hg + self.psi[idx];
                    let (s1, c1) = a1.sin_cos();
                    let (s2, c2) = a2.sin_cos();
                    sin_s2_a1[idx] = s1;
                    cos_s2_a1[idx] = c1;
                    sin_s2_a2[idx] = s2;
                    cos_s2_a2[idx] = c2;
                    sum2 += s1 * c2;
                }
                z_rel[g] = hg + rho * inv_sqrt_r * sum2;
            }
        }

        WaveForwardCache {
            u,
            v,
            cos_diff,
            sin_diff,
            cos_u,
            sin_u,
            cos_v,
            sin_v,
            cos_u2,
            sin_u2,
            cos_v2,
            sin_v2,
            h,
            sin_s2_a1,
            cos_s2_a1,
            sin_s2_a2,
            cos_s2_a2,
            z_rel,
        }
    }

    #[allow(dead_code)]
    pub fn backward(
        &self,
        cache: &WaveForwardCache,
        dz_rel: &[f64; REL_DIM],
        mode: CoreTrainMode,
        grads: Option<&mut WaveCoreGrads>,
    ) -> ([f64; LATENT_D], [f64; LATENT_D]) {
        let trig = StepTrigCache::from_core(self);
        self.backward_with_trig(cache, &trig, dz_rel, mode, grads)
    }

    pub fn backward_with_trig(
        &self,
        cache: &WaveForwardCache,
        trig: &StepTrigCache,
        dz_rel: &[f64; REL_DIM],
        mode: CoreTrainMode,
        mut grads: Option<&mut WaveCoreGrads>,
    ) -> ([f64; LATENT_D], [f64; LATENT_D]) {
        let update_core = mode == CoreTrainMode::FullTrain;
        let update_autocal = mode == CoreTrainMode::AutoCalOnly;
        let update_freq = update_core && self.kind == SpectrumKind::Adaptive;
        let update_phi = update_core && !self.zero_phase_lock;

        // Backward through coupled heterodyne Stage 2 (ZERO trig calls!)
        let mut dh = *dz_rel;
        let active_s2 = !self.disable_stage2 && self.pairing != PairingMode::AdditiveOnly;
        if active_s2 {
            let inv_sqrt_r = 1.0 / (REL_DIM as f64).sqrt();
            for g in 0..REL_DIM {
                let dzg = dz_rel[g];
                let rho = self.cal_rho[g];
                let coeff = dzg * rho * inv_sqrt_r;
                let hg = cache.h[g];
                let base = g * REL_DIM;
                let mut sum_prod = 0.0;
                for j in 0..REL_DIM {
                    let idx = base + j;
                    let tidx = j * REL_DIM + g;
                    let s1 = cache.sin_s2_a1[idx];
                    let c1 = cache.cos_s2_a1[idx];
                    let s2 = cache.sin_s2_a2[idx];
                    let c2 = cache.cos_s2_a2[idx];
                    sum_prod += s1 * c2;
                    let d_a1 = coeff * c1 * c2;
                    let d_a2 = -coeff * s1 * s2;
                    dh[j] += d_a1 * self.kappa[idx];
                    dh[g] += d_a2 * self.kappa[tidx];
                    if let Some(ref mut gr) = grads {
                        if update_freq {
                            gr.kappa[idx] += d_a1 * cache.h[j];
                            gr.kappa[tidx] += d_a2 * hg;
                        }
                        if update_phi {
                            gr.psi[idx] += d_a2;
                        }
                    }
                }
                if update_autocal {
                    if let Some(ref mut gr) = grads {
                        gr.cal_rho[g] += dzg * inv_sqrt_r * sum_prod;
                    }
                }
            }
        }

        // Backward through Stage 1 (512 primary colliding waves, ZERO trig calls!)
        let mut du = [0.0; LATENT_D];
        let mut dv = [0.0; LATENT_D];
        let inv_sqrt_k = 1.0 / (K_PER_BANK as f64).sqrt();
        let mut diff = [0.0; LATENT_D];
        for d in 0..LATENT_D {
            diff[d] = cache.v[d] - cache.u[d];
        }

        match self.pairing {
            PairingMode::FullCollision => {
                for g in 0..N_BANKS {
                    let dc = dh[g] * inv_sqrt_k;
                    let sg = self.cal_scale[g];
                    let mut g_sg = 0.0;
                    let mut g_dg = 0.0;
                    for k in 0..K_PER_BANK {
                        let m = g * K_PER_BANK + k;
                        let w1 = &self.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                        let w2 = &self.omega2[m * LATENT_D..(m + 1) * LATENT_D];
                        let cb = trig.cos_beta[m];
                        let sb = trig.sin_beta[m];

                        let cd = cache.cos_diff[m];
                        let sd = cache.sin_diff[m];
                        let cu = cache.cos_u[m];
                        let su = cache.sin_u[m];
                        let cv = cache.cos_v[m];
                        let sv = cache.sin_v[m];

                        let d_ad = -dc * cb * sd;
                        let d_au = -dc * sb * su * cv;
                        let d_av = -dc * sb * cu * sv;

                        let mut dot_diff = 0.0;
                        let mut dot_u = 0.0;
                        let mut dot_v = 0.0;
                        for d in 0..LATENT_D {
                            dot_diff += w1[d] * diff[d];
                            dot_u += w1[d] * cache.u[d];
                            dot_v += w2[d] * cache.v[d];
                            du[d] += sg * w1[d] * (d_au - d_ad);
                            dv[d] += sg * (w1[d] * d_ad + w2[d] * d_av);
                        }

                        g_sg += d_ad * dot_diff + d_au * dot_u + d_av * dot_v;
                        g_dg += d_ad + d_av;

                        if let Some(ref mut gr) = grads {
                            if update_core {
                                gr.beta[m] += dc * (-sb * cd + cb * cu * cv);
                            }
                            if update_phi {
                                gr.phi1[m] += d_ad + d_au;
                                gr.phi2[m] += d_av;
                            }
                            if update_freq {
                                let gw1 = &mut gr.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                                let gw2 = &mut gr.omega2[m * LATENT_D..(m + 1) * LATENT_D];
                                for d in 0..LATENT_D {
                                    gw1[d] += sg * (d_ad * diff[d] + d_au * cache.u[d]);
                                    gw2[d] += sg * (d_av * cache.v[d]);
                                }
                            }
                        }
                    }
                    if update_autocal {
                        if let Some(ref mut gr) = grads {
                            gr.cal_scale[g] += g_sg;
                            gr.cal_phase[g] += g_dg;
                        }
                    }
                }
            }
            PairingMode::AdditiveOnly => {
                for g in 0..N_BANKS {
                    let dc = dh[g] * inv_sqrt_k;
                    let sg = self.cal_scale[g];
                    let mut g_sg = 0.0;
                    let mut g_dg = 0.0;
                    for k in 0..K_PER_BANK {
                        let m = g * K_PER_BANK + k;
                        let w1 = &self.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                        let w2 = &self.omega2[m * LATENT_D..(m + 1) * LATENT_D];
                        let cb = trig.cos_beta[m];
                        let sb = trig.sin_beta[m];

                        let cu1 = cache.cos_u[m];
                        let su1 = cache.sin_u[m];
                        let cv1 = cache.cos_v[m];
                        let sv1 = cache.sin_v[m];
                        let cu2 = cache.cos_u2[m];
                        let su2 = cache.sin_u2[m];
                        let cv2 = cache.cos_v2[m];
                        let sv2 = cache.sin_v2[m];

                        let d_au1 = -0.5 * dc * cb * su1;
                        let d_av1 = -0.5 * dc * cb * sv1;
                        let d_au2 = 0.5 * dc * sb * su2;
                        let d_av2 = -0.5 * dc * sb * sv2;

                        let mut w1u = 0.0;
                        let mut w1v = 0.0;
                        let mut w2u = 0.0;
                        let mut w2v = 0.0;
                        for d in 0..LATENT_D {
                            w1u += w1[d] * cache.u[d];
                            w1v += w1[d] * cache.v[d];
                            w2u += w2[d] * cache.u[d];
                            w2v += w2[d] * cache.v[d];
                            du[d] += sg * (w1[d] * d_au1 + w2[d] * d_au2);
                            dv[d] += sg * (w1[d] * d_av1 + w2[d] * d_av2);
                        }
                        g_sg += d_au1 * w1u + d_av1 * w1v + d_au2 * w2u + d_av2 * w2v;
                        g_dg += d_av1 + d_av2;

                        if let Some(ref mut gr) = grads {
                            if update_core {
                                gr.beta[m] +=
                                    dc * (-0.5 * sb * (cv1 + cu1) + 0.5 * cb * (cv2 - cu2));
                            }
                            if update_phi {
                                gr.phi1[m] += d_au1 + d_av1;
                                gr.phi2[m] += d_au2 + d_av2;
                            }
                            if update_freq {
                                let gw1 = &mut gr.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                                let gw2 = &mut gr.omega2[m * LATENT_D..(m + 1) * LATENT_D];
                                for d in 0..LATENT_D {
                                    gw1[d] += sg * (d_au1 * cache.u[d] + d_av1 * cache.v[d]);
                                    gw2[d] += sg * (d_au2 * cache.u[d] + d_av2 * cache.v[d]);
                                }
                            }
                        }
                    }
                    if update_autocal {
                        if let Some(ref mut gr) = grads {
                            gr.cal_scale[g] += g_sg;
                            gr.cal_phase[g] += g_dg;
                        }
                    }
                }
            }
            PairingMode::PermutedPairing => {
                for g in 0..N_BANKS {
                    let dc = dh[g] * inv_sqrt_k;
                    let sg = self.cal_scale[g];
                    let mut g_sg = 0.0;
                    let mut g_dg = 0.0;
                    for k in 0..K_PER_BANK {
                        let m = g * K_PER_BANK + k;
                        let p = self.perm_pi[m];
                        let w1 = &self.omega1[m * LATENT_D..(m + 1) * LATENT_D];
                        let w1p = &self.omega1[p * LATENT_D..(p + 1) * LATENT_D];
                        let w2p = &self.omega2[p * LATENT_D..(p + 1) * LATENT_D];
                        let cb = trig.cos_beta[m];
                        let sb = trig.sin_beta[m];

                        let cd = cache.cos_diff[m];
                        let sd = cache.sin_diff[m];
                        let cu = cache.cos_u[m];
                        let su = cache.sin_u[m];
                        let cv = cache.cos_v[m];
                        let sv = cache.sin_v[m];

                        let d_ad = -dc * cb * sd;
                        let d_au = -dc * sb * su * cv;
                        let d_av = -dc * sb * cu * sv;

                        let mut w1v = 0.0;
                        let mut w1pu = 0.0;
                        let mut w1u = 0.0;
                        let mut w2pv = 0.0;
                        for d in 0..LATENT_D {
                            w1v += w1[d] * cache.v[d];
                            w1pu += w1p[d] * cache.u[d];
                            w1u += w1[d] * cache.u[d];
                            w2pv += w2p[d] * cache.v[d];
                            dv[d] += sg * (w1[d] * d_ad + w2p[d] * d_av);
                            du[d] += sg * (-w1p[d] * d_ad + w1[d] * d_au);
                        }
                        g_sg += d_ad * (w1v - w1pu) + d_au * w1u + d_av * w2pv;
                        g_dg += d_ad + d_av;

                        if let Some(ref mut gr) = grads {
                            if update_core {
                                gr.beta[m] += dc * (-sb * cd + cb * cu * cv);
                            }
                            if update_phi {
                                gr.phi1[m] += d_ad + d_au;
                                gr.phi2[m] += d_av;
                            }
                            if update_freq {
                                for d in 0..LATENT_D {
                                    gr.omega1[m * LATENT_D + d] +=
                                        sg * (d_ad * cache.v[d] + d_au * cache.u[d]);
                                    gr.omega1[p * LATENT_D + d] += sg * (-d_ad * cache.u[d]);
                                    gr.omega2[p * LATENT_D + d] += sg * (d_av * cache.v[d]);
                                }
                            }
                        }
                    }
                    if update_autocal {
                        if let Some(ref mut gr) = grads {
                            gr.cal_scale[g] += g_sg;
                            gr.cal_phase[g] += g_dg;
                        }
                    }
                }
            }
        }

        (du, dv)
    }

    /// Fast global short-range Gaussian radial repulsion + intra-bank 4D vector repulsion
    /// + confining bandwidth annulus [R_MIN, R_MAX] on all 512 primary waves (0 Riemann zeros).
    pub fn spectral_regularizer_grad(
        &self,
        lambda_vec: f64,
        lambda_rad: f64,
        lambda_bw: f64,
        g_omega1: &mut [f64],
        g_omega2: &mut [f64],
    ) -> f64 {
        let mut loss = 0.0;
        let sigma_r = 1.45 * (R_MAX - R_MIN) / (N_COLLISIONS as f64); // ~0.0074
        let cutoff = 5.5 * sigma_r;
        let inv_2sr2 = 1.0 / (2.0 * sigma_r * sigma_r);
        let rad_scale = 1.0 / (N_COLLISIONS as f64);

        // Apply identical short-range radial repulsion & bandwidth confining box to omega1 and omega2
        let apply_band = |omega: &[f64], g_om: &mut [f64]| -> f64 {
            let mut l = 0.0;
            let mut radii = [0.0; N_COLLISIONS];
            for m in 0..N_COLLISIONS {
                let wm = &omega[m * LATENT_D..(m + 1) * LATENT_D];
                radii[m] = (wm.iter().map(|x| x * x).sum::<f64>() + 1e-12).sqrt();
            }

            // Global short-range Gaussian radial repulsion across all 256 modes (fast via cutoff skip)
            if lambda_rad > 0.0 {
                for m in 0..N_COLLISIONS {
                    let rm = radii[m];
                    let inv_rm = 1.0 / rm;
                    let wm = &omega[m * LATENT_D..(m + 1) * LATENT_D];
                    for n in (m + 1)..N_COLLISIONS {
                        let dr = rm - radii[n];
                        if dr.abs() > cutoff {
                            continue;
                        }
                        let e = (-dr * dr * inv_2sr2).exp();
                        l += lambda_rad * rad_scale * e;
                        let d_dr = -2.0 * lambda_rad * rad_scale * inv_2sr2 * dr * e;
                        let inv_rn = 1.0 / radii[n];
                        let wn = &omega[n * LATENT_D..(n + 1) * LATENT_D];
                        for d in 0..LATENT_D {
                            g_om[m * LATENT_D + d] += d_dr * wm[d] * inv_rm;
                            g_om[n * LATENT_D + d] -= d_dr * wn[d] * inv_rn;
                        }
                    }
                }
            }

            // Intra-bank 4D vector repulsion so the 16 waves inside each bank span distinct directions
            if lambda_vec > 0.0 {
                let eps_v = 0.35;
                let pair_scale = 2.0 / ((N_BANKS * K_PER_BANK * (K_PER_BANK - 1)) as f64);
                for g in 0..N_BANKS {
                    let base = g * K_PER_BANK;
                    for a in 0..K_PER_BANK {
                        let m = base + a;
                        let wm = &omega[m * LATENT_D..(m + 1) * LATENT_D];
                        for b in (a + 1)..K_PER_BANK {
                            let n = base + b;
                            let wn = &omega[n * LATENT_D..(n + 1) * LATENT_D];
                            let mut d2 = 0.0;
                            let mut diff = [0.0; LATENT_D];
                            for d in 0..LATENT_D {
                                diff[d] = wm[d] - wn[d];
                                d2 += diff[d] * diff[d];
                            }
                            let denom = d2 + eps_v;
                            l += lambda_vec * pair_scale / denom;
                            let coeff = -2.0 * lambda_vec * pair_scale / (denom * denom);
                            for d in 0..LATENT_D {
                                let gd = coeff * diff[d];
                                g_om[m * LATENT_D + d] += gd;
                                g_om[n * LATENT_D + d] -= gd;
                            }
                        }
                    }
                }
            }

            // Confining bandwidth annulus [R_MIN, R_MAX]
            if lambda_bw > 0.0 {
                let inv_m = 1.0 / (N_COLLISIONS as f64);
                for m in 0..N_COLLISIONS {
                    let r = radii[m];
                    let wm = &omega[m * LATENT_D..(m + 1) * LATENT_D];
                    if r < R_MIN {
                        let viol = R_MIN - r;
                        l += lambda_bw * inv_m * viol * viol;
                        let coeff = -2.0 * lambda_bw * inv_m * viol / r;
                        for d in 0..LATENT_D {
                            g_om[m * LATENT_D + d] += coeff * wm[d];
                        }
                    } else if r > R_MAX {
                        let viol = r - R_MAX;
                        l += lambda_bw * inv_m * viol * viol;
                        let coeff = 2.0 * lambda_bw * inv_m * viol / r;
                        for d in 0..LATENT_D {
                            g_om[m * LATENT_D + d] += coeff * wm[d];
                        }
                    }
                }
            }

            l
        };

        loss += apply_band(&self.omega1, g_omega1);
        loss += apply_band(&self.omega2, g_omega2);
        loss
    }
}

/// Matched-capacity MLP Core (3,832 trunk params: 8 -> 64 -> 40 -> 16)
/// + exact 1:1 matching 48 FiLM auto-calibration knobs (8+8+16+16 = 48).
#[derive(Clone, Debug)]
pub struct MlpCore {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub w3: Vec<f64>,
    pub b3: Vec<f64>,
    // 48 matching FiLM auto-calibration parameters:
    pub film_in_scale: Vec<f64>, // 8, init 1.0
    pub film_in_shift: Vec<f64>, // 8, init 0.0
    pub film_z_scale: Vec<f64>,  // 16, init 1.0
    pub film_z_shift: Vec<f64>,  // 16, init 0.0
}

#[derive(Clone, Debug)]
pub struct MlpForwardCache {
    pub raw_inp: [f64; 2 * LATENT_D],
    pub inp: [f64; 2 * LATENT_D],
    pub h1: [f64; MLP_H1],
    pub h2: [f64; MLP_H2],
    pub z_raw: [f64; REL_DIM],
    pub z_rel: [f64; REL_DIM],
}

#[derive(Clone, Debug)]
pub struct MlpCoreGrads {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub w3: Vec<f64>,
    pub b3: Vec<f64>,
    pub film_in_scale: Vec<f64>,
    pub film_in_shift: Vec<f64>,
    pub film_z_scale: Vec<f64>,
    pub film_z_shift: Vec<f64>,
}

impl MlpCoreGrads {
    pub fn zeros() -> Self {
        Self {
            w1: vec![0.0; MLP_H1 * 2 * LATENT_D],
            b1: vec![0.0; MLP_H1],
            w2: vec![0.0; MLP_H2 * MLP_H1],
            b2: vec![0.0; MLP_H2],
            w3: vec![0.0; REL_DIM * MLP_H2],
            b3: vec![0.0; REL_DIM],
            film_in_scale: vec![0.0; 2 * LATENT_D],
            film_in_shift: vec![0.0; 2 * LATENT_D],
            film_z_scale: vec![0.0; REL_DIM],
            film_z_shift: vec![0.0; REL_DIM],
        }
    }

    pub fn clear(&mut self) {
        self.w1.fill(0.0);
        self.b1.fill(0.0);
        self.w2.fill(0.0);
        self.b2.fill(0.0);
        self.w3.fill(0.0);
        self.b3.fill(0.0);
        self.film_in_scale.fill(0.0);
        self.film_in_shift.fill(0.0);
        self.film_z_scale.fill(0.0);
        self.film_z_shift.fill(0.0);
    }
}

impl MlpCore {
    pub fn new(rng: &mut Rng64) -> Self {
        let s1 = (2.0 / ((2 * LATENT_D + MLP_H1) as f64)).sqrt();
        let s2 = (2.0 / ((MLP_H1 + MLP_H2) as f64)).sqrt();
        let s3 = (2.0 / ((MLP_H2 + REL_DIM) as f64)).sqrt();
        Self {
            w1: (0..(MLP_H1 * 2 * LATENT_D)).map(|_| rng.normal() * s1).collect(),
            b1: vec![0.0; MLP_H1],
            w2: (0..(MLP_H2 * MLP_H1)).map(|_| rng.normal() * s2).collect(),
            b2: vec![0.0; MLP_H2],
            w3: (0..(REL_DIM * MLP_H2)).map(|_| rng.normal() * s3).collect(),
            b3: vec![0.0; REL_DIM],
            film_in_scale: vec![1.0; 2 * LATENT_D],
            film_in_shift: vec![0.0; 2 * LATENT_D],
            film_z_scale: vec![1.0; REL_DIM],
            film_z_shift: vec![0.0; REL_DIM],
        }
    }

    pub fn core_param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.w3.len() + self.b3.len()
    }

    pub fn autocal_param_count(&self) -> usize {
        self.film_in_scale.len()
            + self.film_in_shift.len()
            + self.film_z_scale.len()
            + self.film_z_shift.len()
    }

    pub fn frozen_trunk_checksum(&self) -> u64 {
        checksum_f64_slices(&[&self.w1, &self.b1, &self.w2, &self.b2, &self.w3, &self.b3])
    }

    pub fn full_checksum(&self) -> u64 {
        checksum_f64_slices(&[
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w3,
            &self.b3,
            &self.film_in_scale,
            &self.film_in_shift,
            &self.film_z_scale,
            &self.film_z_shift,
        ])
    }

    pub fn reset_autocal(&mut self) {
        self.film_in_scale.fill(1.0);
        self.film_in_shift.fill(0.0);
        self.film_z_scale.fill(1.0);
        self.film_z_shift.fill(0.0);
    }

    pub fn forward(&self, u: [f64; LATENT_D], v: [f64; LATENT_D]) -> MlpForwardCache {
        let mut raw_inp = [0.0; 2 * LATENT_D];
        let mut inp = [0.0; 2 * LATENT_D];
        for d in 0..LATENT_D {
            raw_inp[d] = u[d];
            raw_inp[LATENT_D + d] = v[d];
        }
        for i in 0..(2 * LATENT_D) {
            inp[i] = self.film_in_scale[i] * raw_inp[i] + self.film_in_shift[i];
        }
        let mut h1 = [0.0; MLP_H1];
        for i in 0..MLP_H1 {
            let mut acc = self.b1[i];
            let row = &self.w1[i * 2 * LATENT_D..(i + 1) * 2 * LATENT_D];
            for j in 0..(2 * LATENT_D) {
                acc += row[j] * inp[j];
            }
            h1[i] = acc.tanh();
        }
        let mut h2 = [0.0; MLP_H2];
        for i in 0..MLP_H2 {
            let mut acc = self.b2[i];
            let row = &self.w2[i * MLP_H1..(i + 1) * MLP_H1];
            for j in 0..MLP_H1 {
                acc += row[j] * h1[j];
            }
            h2[i] = acc.tanh();
        }
        let mut z_raw = [0.0; REL_DIM];
        let mut z_rel = [0.0; REL_DIM];
        for i in 0..REL_DIM {
            let mut acc = self.b3[i];
            let row = &self.w3[i * MLP_H2..(i + 1) * MLP_H2];
            for j in 0..MLP_H2 {
                acc += row[j] * h2[j];
            }
            z_raw[i] = acc;
            z_rel[i] = self.film_z_scale[i] * acc + self.film_z_shift[i];
        }
        MlpForwardCache {
            raw_inp,
            inp,
            h1,
            h2,
            z_raw,
            z_rel,
        }
    }

    pub fn backward(
        &self,
        cache: &MlpForwardCache,
        dz_rel: &[f64; REL_DIM],
        mode: CoreTrainMode,
        mut grads: Option<&mut MlpCoreGrads>,
    ) -> ([f64; LATENT_D], [f64; LATENT_D]) {
        let update_core = mode == CoreTrainMode::FullTrain;
        let update_autocal = mode == CoreTrainMode::AutoCalOnly;

        let mut dz_raw = [0.0; REL_DIM];
        for i in 0..REL_DIM {
            dz_raw[i] = dz_rel[i] * self.film_z_scale[i];
            if update_autocal {
                if let Some(ref mut g) = grads {
                    g.film_z_scale[i] += dz_rel[i] * cache.z_raw[i];
                    g.film_z_shift[i] += dz_rel[i];
                }
            }
        }

        let mut dh2 = [0.0; MLP_H2];
        for i in 0..REL_DIM {
            let dt = dz_raw[i];
            let row_w = &self.w3[i * MLP_H2..(i + 1) * MLP_H2];
            if update_core {
                if let Some(ref mut g) = grads {
                    g.b3[i] += dt;
                    let row_g = &mut g.w3[i * MLP_H2..(i + 1) * MLP_H2];
                    for j in 0..MLP_H2 {
                        row_g[j] += dt * cache.h2[j];
                        dh2[j] += dt * row_w[j];
                    }
                    continue;
                }
            }
            for j in 0..MLP_H2 {
                dh2[j] += dt * row_w[j];
            }
        }

        let mut dh1 = [0.0; MLP_H1];
        for i in 0..MLP_H2 {
            let dt = dh2[i] * (1.0 - cache.h2[i] * cache.h2[i]);
            let row_w = &self.w2[i * MLP_H1..(i + 1) * MLP_H1];
            if update_core {
                if let Some(ref mut g) = grads {
                    g.b2[i] += dt;
                    let row_g = &mut g.w2[i * MLP_H1..(i + 1) * MLP_H1];
                    for j in 0..MLP_H1 {
                        row_g[j] += dt * cache.h1[j];
                        dh1[j] += dt * row_w[j];
                    }
                    continue;
                }
            }
            for j in 0..MLP_H1 {
                dh1[j] += dt * row_w[j];
            }
        }

        let mut dinp = [0.0; 2 * LATENT_D];
        for i in 0..MLP_H1 {
            let dt = dh1[i] * (1.0 - cache.h1[i] * cache.h1[i]);
            let row_w = &self.w1[i * 2 * LATENT_D..(i + 1) * 2 * LATENT_D];
            if update_core {
                if let Some(ref mut g) = grads {
                    g.b1[i] += dt;
                    let row_g = &mut g.w1[i * 2 * LATENT_D..(i + 1) * 2 * LATENT_D];
                    for j in 0..(2 * LATENT_D) {
                        row_g[j] += dt * cache.inp[j];
                        dinp[j] += dt * row_w[j];
                    }
                    continue;
                }
            }
            for j in 0..(2 * LATENT_D) {
                dinp[j] += dt * row_w[j];
            }
        }

        let mut draw = [0.0; 2 * LATENT_D];
        for i in 0..(2 * LATENT_D) {
            draw[i] = dinp[i] * self.film_in_scale[i];
            if update_autocal {
                if let Some(ref mut g) = grads {
                    g.film_in_scale[i] += dinp[i] * cache.raw_inp[i];
                    g.film_in_shift[i] += dinp[i];
                }
            }
        }

        let mut du = [0.0; LATENT_D];
        let mut dv = [0.0; LATENT_D];
        for d in 0..LATENT_D {
            du[d] = draw[d];
            dv[d] = draw[LATENT_D + d];
        }
        (du, dv)
    }
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr_encoder: f64,
    pub lr_core: f64,
    pub lr_autocal: f64,
    pub lr_head: f64,
    pub wd_encoder: f64,
    pub wd_core: f64,
    pub wd_autocal: f64,
    pub wd_head: f64,
    pub w_mse: f64,
    pub w_ce: f64,
    pub w_ssl: f64,
    pub lambda_vec: f64,
    pub lambda_rad: f64,
    pub lambda_bw: f64,
    pub lambda_var: f64,
    pub lambda_cov: f64,
    pub latent_norm_reg: f64,
    pub latent_jitter: f64,
    pub core_mode: CoreTrainMode,
}

#[derive(Clone, Debug)]
pub struct WaveModel {
    pub encoder: DomainEncoder,
    pub core: WaveCollisionCore,
    pub head: DomainHead,
}

impl WaveModel {
    pub fn new(
        obs_dim: usize,
        reg_dim: usize,
        cls_dim: usize,
        core: WaveCollisionCore,
        rng: &mut Rng64,
    ) -> Self {
        Self {
            encoder: DomainEncoder::new(obs_dim, rng),
            core,
            head: DomainHead::new(reg_dim, cls_dim, rng),
        }
    }

    pub fn predict_pair(&self, xa: &[f64], xb: &[f64]) -> (Vec<f64>, Vec<f64>, [f64; REL_DIM]) {
        let u = self.encoder.forward(xa);
        let v = self.encoder.forward(xb);
        let cache = self.core.forward(u, v);
        let (reg, logits) = self.head.forward(&cache.z_rel);
        (reg, logits, cache.z_rel)
    }

    pub fn train(&mut self, domain: &DomainData, train_indices: &[usize], cfg: &TrainConfig, rng: &mut Rng64) {
        self.core.assert_no_riemann_injection(false);
        let pre_waves_hash = self.core.frozen_waves_checksum();
        let pre_full_hash = self.core.full_checksum();

        let mut opt_ew = AdamBuf::new(self.encoder.w.len());
        let mut opt_eb = AdamBuf::new(self.encoder.b.len());
        let mut opt_hw_r = AdamBuf::new(self.head.w_reg.len());
        let mut opt_hb_r = AdamBuf::new(self.head.b_reg.len());
        let mut opt_hw_c = AdamBuf::new(self.head.w_cls.len());
        let mut opt_hb_c = AdamBuf::new(self.head.b_cls.len());

        let mut ssl_head = DomainHead::new(REL_DIM, 0, &mut rng.fork(0x5511));
        let mut opt_sw = AdamBuf::new(ssl_head.w_reg.len());
        let mut opt_sb = AdamBuf::new(ssl_head.b_reg.len());
        let mut gw_s = vec![0.0; ssl_head.w_reg.len()];
        let mut gb_s = vec![0.0; ssl_head.b_reg.len()];
        let mut dummy_c1 = Vec::new();
        let mut dummy_c2 = Vec::new();

        let mut opt_om1 = AdamBuf::new(self.core.omega1.len());
        let mut opt_om2 = AdamBuf::new(self.core.omega2.len());
        let mut opt_p1 = AdamBuf::new(self.core.phi1.len());
        let mut opt_p2 = AdamBuf::new(self.core.phi2.len());
        let mut opt_beta = AdamBuf::new(self.core.beta.len());
        let mut opt_kap = AdamBuf::new(self.core.kappa.len());
        let mut opt_psi = AdamBuf::new(self.core.psi.len());
        let mut opt_cs = AdamBuf::new(self.core.cal_scale.len());
        let mut opt_cp = AdamBuf::new(self.core.cal_phase.len());
        let mut opt_cr = AdamBuf::new(self.core.cal_rho.len());

        let mut gw_e = vec![0.0; self.encoder.w.len()];
        let mut gb_e = vec![0.0; self.encoder.b.len()];
        let mut gw_r = vec![0.0; self.head.w_reg.len()];
        let mut gb_r = vec![0.0; self.head.b_reg.len()];
        let mut gw_c = vec![0.0; self.head.w_cls.len()];
        let mut gb_c = vec![0.0; self.head.b_cls.len()];
        let mut g_core = WaveCoreGrads::zeros();

        let n_tr = train_indices.len();
        let bs = cfg.batch_size.min(n_tr).max(1);
        let mut order: Vec<usize> = train_indices.to_vec();
        let mut cursor = order.len();

        let mut batch_ids = Vec::with_capacity(bs);
        let mut batch_caches = Vec::with_capacity(bs);
        let mut batch_zs = Vec::with_capacity(bs);
        let mut batch_dzs = Vec::with_capacity(bs);

        let full_train = cfg.core_mode == CoreTrainMode::FullTrain;
        let need_core_grads = cfg.core_mode != CoreTrainMode::FullyFrozen;
        let mut trig = StepTrigCache::from_core(&self.core);

        for step in 1..=cfg.steps {
            if full_train && step > 1 {
                trig = StepTrigCache::from_core(&self.core);
            }
            let frac = (step as f64) / (cfg.steps as f64);
            let mult = if step <= 18 {
                (step as f64) / 18.0
            } else {
                0.10 + 0.90 * 0.5 * (1.0 + (PI * frac).cos())
            };

            gw_e.fill(0.0);
            gb_e.fill(0.0);
            gw_r.fill(0.0);
            gb_r.fill(0.0);
            gw_c.fill(0.0);
            gb_c.fill(0.0);
            gw_s.fill(0.0);
            gb_s.fill(0.0);
            if need_core_grads {
                g_core.clear();
            }

            batch_ids.clear();
            batch_caches.clear();
            batch_zs.clear();
            batch_dzs.clear();

            let inv_bs = 1.0 / (bs as f64);
            for _ in 0..bs {
                if cursor >= order.len() {
                    rng.shuffle(&mut order);
                    cursor = 0;
                }
                let idx = order[cursor];
                cursor += 1;
                let pair = &domain.pairs[idx];
                let xa = &domain.obs[pair.a];
                let xb = &domain.obs[pair.b];

                let mut u = self.encoder.forward(xa);
                let mut v = self.encoder.forward(xb);
                if cfg.latent_jitter > 0.0 {
                    for d in 0..LATENT_D {
                        u[d] += rng.normal() * cfg.latent_jitter;
                        v[d] += rng.normal() * cfg.latent_jitter;
                    }
                }
                let cache = self.core.forward_with_trig(u, v, &trig);
                let (reg_pred, logits) = self.head.forward(&cache.z_rel);

                let (_loss, mut dz) = self.head.loss_and_backward(
                    &cache.z_rel,
                    &reg_pred,
                    &logits,
                    &pair.reg,
                    pair.cls,
                    cfg.w_mse,
                    cfg.w_ce,
                    inv_bs,
                    &mut gw_r,
                    &mut gb_r,
                    &mut gw_c,
                    &mut gb_c,
                );

                if full_train && cfg.w_ssl > 0.0 {
                    let ssl_target = canonical_pair_invariants(&u, &v);
                    let (ssl_pred, _) = ssl_head.forward(&cache.z_rel);
                    let (_, dz_ssl) = ssl_head.loss_and_backward(
                        &cache.z_rel,
                        &ssl_pred,
                        &[],
                        &ssl_target,
                        0,
                        cfg.w_ssl,
                        0.0,
                        inv_bs,
                        &mut gw_s,
                        &mut gb_s,
                        &mut dummy_c1,
                        &mut dummy_c2,
                    );
                    for r in 0..REL_DIM {
                        dz[r] += dz_ssl[r];
                    }
                }

                batch_ids.push(idx);
                batch_zs.push(cache.z_rel);
                batch_dzs.push(dz);
                batch_caches.push(cache);
            }

            if full_train && (cfg.lambda_var > 0.0 || cfg.lambda_cov > 0.0) {
                batch_var_cov_loss_and_grad(&batch_zs, cfg.lambda_var, cfg.lambda_cov, &mut batch_dzs);
            }

            for b in 0..bs {
                let pair = &domain.pairs[batch_ids[b]];
                let xa = &domain.obs[pair.a];
                let xb = &domain.obs[pair.b];
                let cache = &batch_caches[b];
                let (mut du, mut dv) = self.core.backward_with_trig(
                    cache,
                    &trig,
                    &batch_dzs[b],
                    cfg.core_mode,
                    if need_core_grads { Some(&mut g_core) } else { None },
                );

                if cfg.latent_norm_reg > 0.0 {
                    let target_r2 = 1.44;
                    let ru2: f64 = cache.u.iter().map(|x| x * x).sum();
                    let rv2: f64 = cache.v.iter().map(|x| x * x).sum();
                    let cu = inv_bs * cfg.latent_norm_reg * 2.0 * (ru2 - target_r2);
                    let cv = inv_bs * cfg.latent_norm_reg * 2.0 * (rv2 - target_r2);
                    for d in 0..LATENT_D {
                        du[d] += cu * cache.u[d];
                        dv[d] += cv * cache.v[d];
                    }
                }

                self.encoder.backward(xa, &du, &mut gw_e, &mut gb_e);
                self.encoder.backward(xb, &dv, &mut gw_e, &mut gb_e);
            }

            if full_train && self.core.kind == SpectrumKind::Adaptive {
                self.core.spectral_regularizer_grad(
                    cfg.lambda_vec,
                    cfg.lambda_rad,
                    cfg.lambda_bw,
                    &mut g_core.omega1,
                    &mut g_core.omega2,
                );
            }

            opt_ew.step(&mut self.encoder.w, &gw_e, cfg.lr_encoder * mult, cfg.wd_encoder, step, 3.0);
            opt_eb.step(&mut self.encoder.b, &gb_e, cfg.lr_encoder * mult, 0.0, step, 3.0);
            if self.head.reg_dim > 0 {
                opt_hw_r.step(&mut self.head.w_reg, &gw_r, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                opt_hb_r.step(&mut self.head.b_reg, &gb_r, cfg.lr_head * mult, 0.0, step, 3.0);
            }
            if self.head.cls_dim > 0 {
                opt_hw_c.step(&mut self.head.w_cls, &gw_c, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                opt_hb_c.step(&mut self.head.b_cls, &gb_c, cfg.lr_head * mult, 0.0, step, 3.0);
            }

            match cfg.core_mode {
                CoreTrainMode::FullTrain => {
                    if cfg.w_ssl > 0.0 {
                        opt_sw.step(&mut ssl_head.w_reg, &gw_s, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                        opt_sb.step(&mut ssl_head.b_reg, &gb_s, cfg.lr_head * mult, 0.0, step, 3.0);
                    }
                    if self.core.kind == SpectrumKind::Adaptive {
                        opt_om1.step(&mut self.core.omega1, &g_core.omega1, cfg.lr_core * mult, 0.0, step, 3.0);
                        opt_om2.step(&mut self.core.omega2, &g_core.omega2, cfg.lr_core * mult, 0.0, step, 3.0);
                        opt_kap.step(&mut self.core.kappa, &g_core.kappa, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                    }
                    if !self.core.zero_phase_lock {
                        opt_p1.step(&mut self.core.phi1, &g_core.phi1, cfg.lr_core * mult, 0.0, step, 3.0);
                        opt_p2.step(&mut self.core.phi2, &g_core.phi2, cfg.lr_core * mult, 0.0, step, 3.0);
                        opt_psi.step(&mut self.core.psi, &g_core.psi, cfg.lr_core * mult, 0.0, step, 3.0);
                    }
                    opt_beta.step(&mut self.core.beta, &g_core.beta, cfg.lr_core * mult, 0.0, step, 3.0);
                }
                CoreTrainMode::AutoCalOnly => {
                    let cal_start = (cfg.steps * 50) / 100;
                    if step > cal_start {
                        opt_cs.step_spring(&mut self.core.cal_scale, &g_core.cal_scale, cfg.lr_autocal * mult, cfg.wd_autocal, 1.0, 2.0);
                        opt_cp.step_spring(&mut self.core.cal_phase, &g_core.cal_phase, cfg.lr_autocal * mult, cfg.wd_autocal, 0.0, 2.0);
                        opt_cr.step_spring(&mut self.core.cal_rho, &g_core.cal_rho, cfg.lr_autocal * mult, cfg.wd_autocal, 1.0, 2.0);
                    }
                }
                CoreTrainMode::FullyFrozen => {}
            }
        }

        match cfg.core_mode {
            CoreTrainMode::AutoCalOnly => {
                assert_eq!(
                    pre_waves_hash,
                    self.core.frozen_waves_checksum(),
                    "FROZEN WAVE CORE VIOLATION during AutoCalOnly!"
                );
            }
            CoreTrainMode::FullyFrozen => {
                assert_eq!(
                    pre_full_hash,
                    self.core.full_checksum(),
                    "FROZEN CORE VIOLATION during FullyFrozen!"
                );
            }
            CoreTrainMode::FullTrain => {}
        }
    }
}

#[derive(Clone, Debug)]
pub struct MlpModel {
    pub encoder: DomainEncoder,
    pub core: MlpCore,
    pub head: DomainHead,
}

impl MlpModel {
    pub fn new(
        obs_dim: usize,
        reg_dim: usize,
        cls_dim: usize,
        core: MlpCore,
        rng: &mut Rng64,
    ) -> Self {
        Self {
            encoder: DomainEncoder::new(obs_dim, rng),
            core,
            head: DomainHead::new(reg_dim, cls_dim, rng),
        }
    }

    pub fn predict_pair(&self, xa: &[f64], xb: &[f64]) -> (Vec<f64>, Vec<f64>, [f64; REL_DIM]) {
        let u = self.encoder.forward(xa);
        let v = self.encoder.forward(xb);
        let cache = self.core.forward(u, v);
        let (reg, logits) = self.head.forward(&cache.z_rel);
        (reg, logits, cache.z_rel)
    }

    pub fn train(&mut self, domain: &DomainData, train_indices: &[usize], cfg: &TrainConfig, rng: &mut Rng64) {
        let pre_trunk_hash = self.core.frozen_trunk_checksum();
        let pre_full_hash = self.core.full_checksum();

        let mut opt_ew = AdamBuf::new(self.encoder.w.len());
        let mut opt_eb = AdamBuf::new(self.encoder.b.len());
        let mut opt_hw_r = AdamBuf::new(self.head.w_reg.len());
        let mut opt_hb_r = AdamBuf::new(self.head.b_reg.len());
        let mut opt_hw_c = AdamBuf::new(self.head.w_cls.len());
        let mut opt_hb_c = AdamBuf::new(self.head.b_cls.len());

        let mut ssl_head = DomainHead::new(REL_DIM, 0, &mut rng.fork(0x5511));
        let mut opt_sw = AdamBuf::new(ssl_head.w_reg.len());
        let mut opt_sb = AdamBuf::new(ssl_head.b_reg.len());
        let mut gw_s = vec![0.0; ssl_head.w_reg.len()];
        let mut gb_s = vec![0.0; ssl_head.b_reg.len()];
        let mut dummy_c1 = Vec::new();
        let mut dummy_c2 = Vec::new();

        let mut opt_w1 = AdamBuf::new(self.core.w1.len());
        let mut opt_b1 = AdamBuf::new(self.core.b1.len());
        let mut opt_w2 = AdamBuf::new(self.core.w2.len());
        let mut opt_b2 = AdamBuf::new(self.core.b2.len());
        let mut opt_w3 = AdamBuf::new(self.core.w3.len());
        let mut opt_b3 = AdamBuf::new(self.core.b3.len());
        let mut opt_fis = AdamBuf::new(self.core.film_in_scale.len());
        let mut opt_fib = AdamBuf::new(self.core.film_in_shift.len());
        let mut opt_fzs = AdamBuf::new(self.core.film_z_scale.len());
        let mut opt_fzb = AdamBuf::new(self.core.film_z_shift.len());

        let mut gw_e = vec![0.0; self.encoder.w.len()];
        let mut gb_e = vec![0.0; self.encoder.b.len()];
        let mut gw_r = vec![0.0; self.head.w_reg.len()];
        let mut gb_r = vec![0.0; self.head.b_reg.len()];
        let mut gw_c = vec![0.0; self.head.w_cls.len()];
        let mut gb_c = vec![0.0; self.head.b_cls.len()];
        let mut g_core = MlpCoreGrads::zeros();

        let n_tr = train_indices.len();
        let bs = cfg.batch_size.min(n_tr).max(1);
        let mut order: Vec<usize> = train_indices.to_vec();
        let mut cursor = order.len();

        let mut batch_ids = Vec::with_capacity(bs);
        let mut batch_caches = Vec::with_capacity(bs);
        let mut batch_zs = Vec::with_capacity(bs);
        let mut batch_dzs = Vec::with_capacity(bs);

        let full_train = cfg.core_mode == CoreTrainMode::FullTrain;
        let need_core_grads = cfg.core_mode != CoreTrainMode::FullyFrozen;

        for step in 1..=cfg.steps {
            let frac = (step as f64) / (cfg.steps as f64);
            let mult = if step <= 18 {
                (step as f64) / 18.0
            } else {
                0.10 + 0.90 * 0.5 * (1.0 + (PI * frac).cos())
            };

            gw_e.fill(0.0);
            gb_e.fill(0.0);
            gw_r.fill(0.0);
            gb_r.fill(0.0);
            gw_c.fill(0.0);
            gb_c.fill(0.0);
            gw_s.fill(0.0);
            gb_s.fill(0.0);
            if need_core_grads {
                g_core.clear();
            }

            batch_ids.clear();
            batch_caches.clear();
            batch_zs.clear();
            batch_dzs.clear();

            let inv_bs = 1.0 / (bs as f64);
            for _ in 0..bs {
                if cursor >= order.len() {
                    rng.shuffle(&mut order);
                    cursor = 0;
                }
                let idx = order[cursor];
                cursor += 1;
                let pair = &domain.pairs[idx];
                let xa = &domain.obs[pair.a];
                let xb = &domain.obs[pair.b];

                let mut u = self.encoder.forward(xa);
                let mut v = self.encoder.forward(xb);
                if cfg.latent_jitter > 0.0 {
                    for d in 0..LATENT_D {
                        u[d] += rng.normal() * cfg.latent_jitter;
                        v[d] += rng.normal() * cfg.latent_jitter;
                    }
                }
                let cache = self.core.forward(u, v);
                let (reg_pred, logits) = self.head.forward(&cache.z_rel);

                let (_loss, mut dz) = self.head.loss_and_backward(
                    &cache.z_rel,
                    &reg_pred,
                    &logits,
                    &pair.reg,
                    pair.cls,
                    cfg.w_mse,
                    cfg.w_ce,
                    inv_bs,
                    &mut gw_r,
                    &mut gb_r,
                    &mut gw_c,
                    &mut gb_c,
                );

                if full_train && cfg.w_ssl > 0.0 {
                    let ssl_target = canonical_pair_invariants(&u, &v);
                    let (ssl_pred, _) = ssl_head.forward(&cache.z_rel);
                    let (_, dz_ssl) = ssl_head.loss_and_backward(
                        &cache.z_rel,
                        &ssl_pred,
                        &[],
                        &ssl_target,
                        0,
                        cfg.w_ssl,
                        0.0,
                        inv_bs,
                        &mut gw_s,
                        &mut gb_s,
                        &mut dummy_c1,
                        &mut dummy_c2,
                    );
                    for r in 0..REL_DIM {
                        dz[r] += dz_ssl[r];
                    }
                }

                batch_ids.push(idx);
                batch_zs.push(cache.z_rel);
                batch_dzs.push(dz);
                batch_caches.push(cache);
            }

            if full_train && (cfg.lambda_var > 0.0 || cfg.lambda_cov > 0.0) {
                batch_var_cov_loss_and_grad(&batch_zs, cfg.lambda_var, cfg.lambda_cov, &mut batch_dzs);
            }

            for b in 0..bs {
                let pair = &domain.pairs[batch_ids[b]];
                let xa = &domain.obs[pair.a];
                let xb = &domain.obs[pair.b];
                let cache = &batch_caches[b];
                let (mut du, mut dv) = self.core.backward(
                    cache,
                    &batch_dzs[b],
                    cfg.core_mode,
                    if need_core_grads { Some(&mut g_core) } else { None },
                );

                if cfg.latent_norm_reg > 0.0 {
                    let target_r2 = 1.44;
                    let ru2: f64 = cache.raw_inp[0..LATENT_D].iter().map(|x| x * x).sum();
                    let rv2: f64 = cache.raw_inp[LATENT_D..2 * LATENT_D].iter().map(|x| x * x).sum();
                    let cu = inv_bs * cfg.latent_norm_reg * 2.0 * (ru2 - target_r2);
                    let cv = inv_bs * cfg.latent_norm_reg * 2.0 * (rv2 - target_r2);
                    for d in 0..LATENT_D {
                        du[d] += cu * cache.raw_inp[d];
                        dv[d] += cv * cache.raw_inp[LATENT_D + d];
                    }
                }

                self.encoder.backward(xa, &du, &mut gw_e, &mut gb_e);
                self.encoder.backward(xb, &dv, &mut gw_e, &mut gb_e);
            }

            opt_ew.step(&mut self.encoder.w, &gw_e, cfg.lr_encoder * mult, cfg.wd_encoder, step, 3.0);
            opt_eb.step(&mut self.encoder.b, &gb_e, cfg.lr_encoder * mult, 0.0, step, 3.0);
            if self.head.reg_dim > 0 {
                opt_hw_r.step(&mut self.head.w_reg, &gw_r, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                opt_hb_r.step(&mut self.head.b_reg, &gb_r, cfg.lr_head * mult, 0.0, step, 3.0);
            }
            if self.head.cls_dim > 0 {
                opt_hw_c.step(&mut self.head.w_cls, &gw_c, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                opt_hb_c.step(&mut self.head.b_cls, &gb_c, cfg.lr_head * mult, 0.0, step, 3.0);
            }

            match cfg.core_mode {
                CoreTrainMode::FullTrain => {
                    if cfg.w_ssl > 0.0 {
                        opt_sw.step(&mut ssl_head.w_reg, &gw_s, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                        opt_sb.step(&mut ssl_head.b_reg, &gb_s, cfg.lr_head * mult, 0.0, step, 3.0);
                    }
                    opt_w1.step(&mut self.core.w1, &g_core.w1, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                    opt_b1.step(&mut self.core.b1, &g_core.b1, cfg.lr_core * mult, 0.0, step, 3.0);
                    opt_w2.step(&mut self.core.w2, &g_core.w2, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                    opt_b2.step(&mut self.core.b2, &g_core.b2, cfg.lr_core * mult, 0.0, step, 3.0);
                    opt_w3.step(&mut self.core.w3, &g_core.w3, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                    opt_b3.step(&mut self.core.b3, &g_core.b3, cfg.lr_core * mult, 0.0, step, 3.0);
                }
                CoreTrainMode::AutoCalOnly => {
                    let cal_start = (cfg.steps * 50) / 100;
                    if step > cal_start {
                        opt_fis.step_spring(&mut self.core.film_in_scale, &g_core.film_in_scale, cfg.lr_autocal * mult, cfg.wd_autocal, 1.0, 2.0);
                        opt_fib.step_spring(&mut self.core.film_in_shift, &g_core.film_in_shift, cfg.lr_autocal * mult, cfg.wd_autocal, 0.0, 2.0);
                        opt_fzs.step_spring(&mut self.core.film_z_scale, &g_core.film_z_scale, cfg.lr_autocal * mult, cfg.wd_autocal, 1.0, 2.0);
                        opt_fzb.step_spring(&mut self.core.film_z_shift, &g_core.film_z_shift, cfg.lr_autocal * mult, cfg.wd_autocal, 0.0, 2.0);
                    }
                }
                CoreTrainMode::FullyFrozen => {}
            }
        }

        match cfg.core_mode {
            CoreTrainMode::AutoCalOnly => {
                assert_eq!(
                    pre_trunk_hash,
                    self.core.frozen_trunk_checksum(),
                    "FROZEN MLP TRUNK VIOLATION during AutoCalOnly!"
                );
            }
            CoreTrainMode::FullyFrozen => {
                assert_eq!(
                    pre_full_hash,
                    self.core.full_checksum(),
                    "FROZEN MLP VIOLATION during FullyFrozen!"
                );
            }
            CoreTrainMode::FullTrain => {}
        }
    }
}
