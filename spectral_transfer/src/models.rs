//! General Spectral Interference Field (`SpectralCore` + `DomainEncoder` + `DomainHead`)
//! and exact capacity-matched neural baseline (`MlpCore`, 2,008 core params vs 1,936 spectral).
//!
//! Includes exact analytical backpropagation, AdamW optimizer, gentle Dyson spectral
//! self-organization, symmetric VICReg full-rank covariance decorrelation (identical for
//! SpectralCore and MlpCore), strict frozen-core transfer with IEEE-754 checksum verification,
//! and a panic guard preventing Riemann/prime injection into Adaptive mode.

use crate::domains::DomainData;
use crate::rng::{checksum_f64_slices, Rng64, RIEMANN_ZEROS_48};
use std::f64::consts::PI;

pub const LATENT_D: usize = 4;
pub const N_MODES: usize = 48;
pub const HALF_MODES: usize = 24;
pub const REL_DIM: usize = 16;
pub const MLP_H1: usize = 48;
pub const MLP_H2: usize = 24;

/// Multi-scale Nyquist-safe bandwidth:
///   [0.28, 0.75] = sub-Nyquist monotonic ramps (hierarchies, partial orders, directed drives)
///   [0.75, 1.55] = resonant interferometric band (cycles, beating, sharp partner disambiguation)
pub const R_MIN: f64 = 0.28;
pub const R_MAX: f64 = 1.55;

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
    FullInterferometric,
    AdditiveOnly,
    PermutedPairing,
}

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Deterministic AdamW optimizer state for a flat parameter buffer.
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
}

/// Lightweight linear sensory interface E_dom: R^obs_dim -> R^4.
#[derive(Clone, Debug)]
pub struct DomainEncoder {
    pub obs_dim: usize,
    pub w: Vec<f64>, // LATENT_D * obs_dim
    pub b: Vec<f64>, // LATENT_D
}

impl DomainEncoder {
    pub fn new(obs_dim: usize, rng: &mut Rng64) -> Self {
        let scale = (1.0 / (obs_dim as f64)).sqrt() * 0.85;
        let mut w = Vec::with_capacity(LATENT_D * obs_dim);
        for _ in 0..(LATENT_D * obs_dim) {
            w.push(rng.normal() * scale);
        }
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

/// Lightweight linear readout head H_dom: R^16 -> (reg_dim, cls_dim).
#[derive(Clone, Debug)]
pub struct DomainHead {
    pub reg_dim: usize,
    pub cls_dim: usize,
    pub w_reg: Vec<f64>, // reg_dim * REL_DIM
    pub b_reg: Vec<f64>, // reg_dim
    pub w_cls: Vec<f64>, // cls_dim * REL_DIM
    pub b_cls: Vec<f64>, // cls_dim
}

impl DomainHead {
    pub fn new(reg_dim: usize, cls_dim: usize, rng: &mut Rng64) -> Self {
        let s = (1.0 / (REL_DIM as f64)).sqrt() * 0.5;
        let w_reg = (0..(reg_dim * REL_DIM)).map(|_| rng.normal() * s).collect();
        let b_reg = vec![0.0; reg_dim];
        let w_cls = (0..(cls_dim * REL_DIM)).map(|_| rng.normal() * s).collect();
        let b_cls = vec![0.0; cls_dim];
        Self {
            reg_dim,
            cls_dim,
            w_reg,
            b_reg,
            w_cls,
            b_cls,
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

/// Full-rank VICReg-style variance + off-diagonal covariance decorrelation on a batch of
/// relational representations Z (N x 16). Applied IDENTICALLY to SpectralModel and MlpModel
/// during Domain A training so neither core suffers dimensional rank collapse before transfer.
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

    // Centered Z and covariance matrix C[r][s] = (1/N) sum_i (z_ir - mean_r)(z_is - mean_s)
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

    // dL / dC[r][s]
    let mut dcov = [[0.0; REL_DIM]; REL_DIM];
    if lambda_var > 0.0 {
        let target_var = 0.75;
        for r in 0..REL_DIM {
            let err = cov[r][r] - target_var;
            loss += lambda_var * inv_r * err * err;
            dcov[r][r] += 2.0 * lambda_var * inv_r * err;
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

    // Backprop through C = (1/N) Z_c^T Z_c: dL/dz_{ir} = (1/N) sum_s (dcov[r][s] + dcov[s][r]) * (z_{is} - mean_s)
    // Wait: for diagonal r==s, dC_rr / dz_ir = (2/N)(z_ir - mean_r), matching (dcov[r][r] + dcov[r][r])/N!
    // And for off-diagonal r!=s, both C_rs and C_sr appear in loss, and dcov is symmetric,
    // so dL/dz_ir = (2/N) sum_s dcov[r][s] (z_is - mean_s)!
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

/// Deterministic derangement pi[m] != m for the PermutedPairing ablation.
fn make_derangement(rng: &mut Rng64) -> [usize; N_MODES] {
    let mut perm = [0usize; N_MODES];
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    loop {
        rng.shuffle(&mut perm);
        if perm.iter().enumerate().all(|(i, &p)| i != p) {
            return perm;
        }
    }
}

/// Sample radial norms for Dyson GUE spacing distribution P(s) ~ s^2 exp(-4 s^2 / pi).
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
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                for d in -2..=2 {
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

/// Universal Spectral Interference Core (exact 1,936 params, domain-agnostic):
///   - 48 conjugate difference modes:  a_d * cos(w_m . (v - u) + phi_d)
///   - 24 compositional sum modes:     a_s * cos(w_m . (u + v) + phi_s)         (m = 0..24)
///   - 24 chiral quadrature products:  a_s * cos(w_m . u + phi_d) * cos(w_{m^1} . v + phi_s) (m = 24..48)
///   - Pure linear interferometric superposition: z_rel = W_pair * feat + b_pair
///     (NO hidden tanh MLP layer behind feat — ensuring AdditiveOnly is 100% strictly separable!)
#[derive(Clone, Debug)]
pub struct SpectralCore {
    pub kind: SpectrumKind,
    pub pairing: PairingMode,
    pub zero_phase_lock: bool,
    pub omega: Vec<f64>,      // 48 * 4 = 192
    pub phi_diff: Vec<f64>,   // 48
    pub phi_sum: Vec<f64>,    // 48
    pub alpha_diff: Vec<f64>, // 48
    pub alpha_sum: Vec<f64>,  // 48
    pub w_pair: Vec<f64>,     // 16 * 96 = 1,536
    pub b_pair: Vec<f64>,     // 16
    pub perm_pi: [usize; N_MODES],
    pub uses_riemann_zeros: bool,
}

#[derive(Clone, Debug)]
pub struct SpectralForwardCache {
    pub u: [f64; LATENT_D],
    pub v: [f64; LATENT_D],
    pub feat: [f64; 2 * N_MODES],
    pub z_rel: [f64; REL_DIM],
    pub arg_d: [f64; N_MODES],
    pub arg_s: [f64; N_MODES],
    pub arg_u_d: [f64; N_MODES],
    pub arg_v_d: [f64; N_MODES],
    pub arg_u_s: [f64; N_MODES],
    pub arg_v_s: [f64; N_MODES],
    pub a_diff: [f64; N_MODES],
    pub a_sum: [f64; N_MODES],
}

#[derive(Clone, Debug)]
pub struct SpectralCoreGrads {
    pub omega: Vec<f64>,
    pub phi_diff: Vec<f64>,
    pub phi_sum: Vec<f64>,
    pub alpha_diff: Vec<f64>,
    pub alpha_sum: Vec<f64>,
    pub w_pair: Vec<f64>,
    pub b_pair: Vec<f64>,
}

impl SpectralCoreGrads {
    pub fn zeros() -> Self {
        Self {
            omega: vec![0.0; N_MODES * LATENT_D],
            phi_diff: vec![0.0; N_MODES],
            phi_sum: vec![0.0; N_MODES],
            alpha_diff: vec![0.0; N_MODES],
            alpha_sum: vec![0.0; N_MODES],
            w_pair: vec![0.0; REL_DIM * 2 * N_MODES],
            b_pair: vec![0.0; REL_DIM],
        }
    }

    pub fn clear(&mut self) {
        self.omega.fill(0.0);
        self.phi_diff.fill(0.0);
        self.phi_sum.fill(0.0);
        self.alpha_diff.fill(0.0);
        self.alpha_sum.fill(0.0);
        self.w_pair.fill(0.0);
        self.b_pair.fill(0.0);
    }
}

impl SpectralCore {
    pub fn new(kind: SpectrumKind, pairing: PairingMode, zero_phase_lock: bool, rng: &mut Rng64) -> Self {
        let mut omega = vec![0.0; N_MODES * LATENT_D];
        let mut uses_riemann_zeros = false;

        match kind {
            SpectrumKind::Adaptive | SpectrumKind::PoissonFixed => {
                // Pure random Poisson initialization — ZERO Riemann zeros, ZERO primes!
                let radii = sample_poisson_radii(N_MODES, R_MIN, R_MAX, rng);
                for (m, &r) in radii.iter().enumerate() {
                    let dir = rng.unit_vec(LATENT_D);
                    for d in 0..LATENT_D {
                        omega[m * LATENT_D + d] = dir[d] * r;
                    }
                }
            }
            SpectrumKind::RiemannFixed => {
                uses_riemann_zeros = true;
                let g_min = RIEMANN_ZEROS_48[0];
                let g_max = RIEMANN_ZEROS_48[N_MODES - 1];
                let span = g_max - g_min;
                for m in 0..N_MODES {
                    let r = R_MIN + (R_MAX - R_MIN) * (RIEMANN_ZEROS_48[m] - g_min) / span;
                    let dir = rng.unit_vec(LATENT_D);
                    for d in 0..LATENT_D {
                        omega[m * LATENT_D + d] = dir[d] * r;
                    }
                }
            }
            SpectrumKind::FourierFixed => {
                let lat = make_fourier_lattice_4d(N_MODES, R_MIN, R_MAX);
                for m in 0..N_MODES {
                    for d in 0..LATENT_D {
                        omega[m * LATENT_D + d] = lat[m][d];
                    }
                }
            }
            SpectrumKind::DysonFixed => {
                let radii = sample_dyson_radii(N_MODES, R_MIN, R_MAX, rng);
                for (m, &r) in radii.iter().enumerate() {
                    let dir = rng.unit_vec(LATENT_D);
                    for d in 0..LATENT_D {
                        omega[m * LATENT_D + d] = dir[d] * r;
                    }
                }
            }
        }

        let (phi_diff, phi_sum) = if zero_phase_lock {
            (vec![0.0; N_MODES], vec![0.0; N_MODES])
        } else {
            let pd = (0..N_MODES).map(|_| rng.uniform_range(-PI, PI)).collect();
            let ps = (0..N_MODES).map(|_| rng.uniform_range(-PI, PI)).collect();
            (pd, ps)
        };

        let alpha_diff = vec![0.55; N_MODES];
        let alpha_sum = vec![0.55; N_MODES];

        let s = (1.0 / ((2 * N_MODES) as f64)).sqrt() * 1.1;
        let w_pair = (0..(REL_DIM * 2 * N_MODES))
            .map(|_| rng.normal() * s)
            .collect();
        let b_pair = vec![0.0; REL_DIM];
        let perm_pi = make_derangement(rng);

        Self {
            kind,
            pairing,
            zero_phase_lock,
            omega,
            phi_diff,
            phi_sum,
            alpha_diff,
            alpha_sum,
            w_pair,
            b_pair,
            perm_pi,
            uses_riemann_zeros,
        }
    }

    pub fn assert_no_riemann_injection(&self, attempted_riemann_or_prime_inject: bool) {
        if self.kind == SpectrumKind::Adaptive
            && (attempted_riemann_or_prime_inject || self.uses_riemann_zeros)
        {
            panic!("PROTOCOL VIOLATION: Adaptive SpectralCore forbids Riemann zeros or prime frequency injection!");
        }
    }

    pub fn param_count(&self) -> usize {
        self.omega.len()
            + self.phi_diff.len()
            + self.phi_sum.len()
            + self.alpha_diff.len()
            + self.alpha_sum.len()
            + self.w_pair.len()
            + self.b_pair.len()
    }

    pub fn checksum(&self) -> u64 {
        checksum_f64_slices(&[
            &self.omega,
            &self.phi_diff,
            &self.phi_sum,
            &self.alpha_diff,
            &self.alpha_sum,
            &self.w_pair,
            &self.b_pair,
        ])
    }

    pub fn with_scrambled_phases(&self, rng: &mut Rng64) -> Self {
        let mut c = self.clone();
        for p in &mut c.phi_diff {
            *p = rng.uniform_range(-PI, PI);
        }
        for p in &mut c.phi_sum {
            *p = rng.uniform_range(-PI, PI);
        }
        c
    }

    pub fn forward(&self, u: [f64; LATENT_D], v: [f64; LATENT_D]) -> SpectralForwardCache {
        let mut feat = [0.0; 2 * N_MODES];
        let mut arg_d = [0.0; N_MODES];
        let mut arg_s = [0.0; N_MODES];
        let mut arg_u_d = [0.0; N_MODES];
        let mut arg_v_d = [0.0; N_MODES];
        let mut arg_u_s = [0.0; N_MODES];
        let mut arg_v_s = [0.0; N_MODES];
        let mut a_diff = [0.0; N_MODES];
        let mut a_sum = [0.0; N_MODES];

        match self.pairing {
            PairingMode::FullInterferometric => {
                let mut diff = [0.0; LATENT_D];
                let mut sum = [0.0; LATENT_D];
                for d in 0..LATENT_D {
                    diff[d] = v[d] - u[d];
                    sum[d] = u[d] + v[d];
                }
                for m in 0..N_MODES {
                    let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                    let ad = softplus(self.alpha_diff[m]);
                    let as_ = softplus(self.alpha_sum[m]);
                    a_diff[m] = ad;
                    a_sum[m] = as_;

                    let mut dd = self.phi_diff[m];
                    for d in 0..LATENT_D {
                        dd += wm[d] * diff[d];
                    }
                    arg_d[m] = dd;
                    feat[m] = ad * dd.cos();

                    if m < HALF_MODES {
                        let mut ds = self.phi_sum[m];
                        for d in 0..LATENT_D {
                            ds += wm[d] * sum[d];
                        }
                        arg_s[m] = ds;
                        feat[N_MODES + m] = as_ * ds.cos();
                    } else {
                        // Chiral Past x Future quadrature product pairing adjacent modes (m, m^1)
                        let q = m ^ 1;
                        let wq = &self.omega[q * LATENT_D..(q + 1) * LATENT_D];
                        let mut au = self.phi_diff[m];
                        let mut av = self.phi_sum[m];
                        for d in 0..LATENT_D {
                            au += wm[d] * u[d];
                            av += wq[d] * v[d];
                        }
                        arg_u_s[m] = au;
                        arg_v_s[m] = av;
                        feat[N_MODES + m] = as_ * au.cos() * av.cos();
                    }
                }
            }
            PairingMode::AdditiveOnly => {
                for m in 0..N_MODES {
                    let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                    let ad = softplus(self.alpha_diff[m]);
                    let as_ = softplus(self.alpha_sum[m]);
                    a_diff[m] = ad;
                    a_sum[m] = as_;
                    let mut wu = 0.0;
                    let mut wv = 0.0;
                    for d in 0..LATENT_D {
                        wu += wm[d] * u[d];
                        wv += wm[d] * v[d];
                    }
                    arg_u_d[m] = wu + self.phi_diff[m];
                    arg_v_d[m] = wv + self.phi_diff[m];
                    arg_u_s[m] = wu + self.phi_sum[m];
                    arg_v_s[m] = wv + self.phi_sum[m];
                    // Strictly separable additive superposition: zero cross-interference between u and v!
                    feat[m] = 0.5 * ad * (arg_v_d[m].cos() + arg_u_d[m].cos());
                    feat[N_MODES + m] = 0.5 * as_ * (arg_v_s[m].cos() - arg_u_s[m].cos());
                }
            }
            PairingMode::PermutedPairing => {
                for m in 0..N_MODES {
                    let p = self.perm_pi[m];
                    let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                    let wp = &self.omega[p * LATENT_D..(p + 1) * LATENT_D];
                    let ad = softplus(self.alpha_diff[m]);
                    let as_ = softplus(self.alpha_sum[m]);
                    a_diff[m] = ad;
                    a_sum[m] = as_;
                    let mut wv = 0.0;
                    let mut wu = 0.0;
                    for d in 0..LATENT_D {
                        wv += wm[d] * v[d];
                        wu += wp[d] * u[d];
                    }
                    arg_d[m] = wv - wu + self.phi_diff[m];
                    arg_s[m] = wv + wu + self.phi_sum[m];
                    feat[m] = ad * arg_d[m].cos();
                    feat[N_MODES + m] = as_ * arg_s[m].cos();
                }
            }
        }

        // Pure linear interferometric superposition (NO extra MLP tanh!)
        let mut z_rel = [0.0; REL_DIM];
        for r in 0..REL_DIM {
            let row = &self.w_pair[r * 2 * N_MODES..(r + 1) * 2 * N_MODES];
            let mut acc = self.b_pair[r];
            for j in 0..(2 * N_MODES) {
                acc += row[j] * feat[j];
            }
            z_rel[r] = acc;
        }

        SpectralForwardCache {
            u,
            v,
            feat,
            z_rel,
            arg_d,
            arg_s,
            arg_u_d,
            arg_v_d,
            arg_u_s,
            arg_v_s,
            a_diff,
            a_sum,
        }
    }

    pub fn backward(
        &self,
        cache: &SpectralForwardCache,
        dz_rel: &[f64; REL_DIM],
        update_core: bool,
        grads: Option<&mut SpectralCoreGrads>,
    ) -> ([f64; LATENT_D], [f64; LATENT_D]) {
        let mut dfeat = [0.0; 2 * N_MODES];
        let mut grads_opt = if update_core { grads } else { None };

        for r in 0..REL_DIM {
            let dt = dz_rel[r];
            let row_w = &self.w_pair[r * 2 * N_MODES..(r + 1) * 2 * N_MODES];
            if let Some(ref mut g) = grads_opt {
                g.b_pair[r] += dt;
                let row_g = &mut g.w_pair[r * 2 * N_MODES..(r + 1) * 2 * N_MODES];
                for j in 0..(2 * N_MODES) {
                    row_g[j] += dt * cache.feat[j];
                    dfeat[j] += dt * row_w[j];
                }
            } else {
                for j in 0..(2 * N_MODES) {
                    dfeat[j] += dt * row_w[j];
                }
            }
        }

        let mut du = [0.0; LATENT_D];
        let mut dv = [0.0; LATENT_D];
        let update_omega = update_core && self.kind == SpectrumKind::Adaptive;
        let update_phi = update_core && !self.zero_phase_lock;

        match self.pairing {
            PairingMode::FullInterferometric => {
                for m in 0..N_MODES {
                    let df_d = dfeat[m];
                    let df_s = dfeat[N_MODES + m];
                    let cos_d = cache.arg_d[m].cos();
                    let sin_d = cache.arg_d[m].sin();
                    let d_arg_d = -df_d * cache.a_diff[m] * sin_d;

                    let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                    for d in 0..LATENT_D {
                        du[d] -= wm[d] * d_arg_d;
                        dv[d] += wm[d] * d_arg_d;
                    }
                    if let Some(ref mut g) = grads_opt {
                        g.alpha_diff[m] += df_d * cos_d * sigmoid(self.alpha_diff[m]);
                        if update_phi {
                            g.phi_diff[m] += d_arg_d;
                        }
                        if update_omega {
                            let gw = &mut g.omega[m * LATENT_D..(m + 1) * LATENT_D];
                            for d in 0..LATENT_D {
                                gw[d] += d_arg_d * (cache.v[d] - cache.u[d]);
                            }
                        }
                    }

                    if m < HALF_MODES {
                        let cos_s = cache.arg_s[m].cos();
                        let sin_s = cache.arg_s[m].sin();
                        let d_arg_s = -df_s * cache.a_sum[m] * sin_s;
                        for d in 0..LATENT_D {
                            du[d] += wm[d] * d_arg_s;
                            dv[d] += wm[d] * d_arg_s;
                        }
                        if let Some(ref mut g) = grads_opt {
                            g.alpha_sum[m] += df_s * cos_s * sigmoid(self.alpha_sum[m]);
                            if update_phi {
                                g.phi_sum[m] += d_arg_s;
                            }
                            if update_omega {
                                let gw = &mut g.omega[m * LATENT_D..(m + 1) * LATENT_D];
                                for d in 0..LATENT_D {
                                    gw[d] += d_arg_s * (cache.u[d] + cache.v[d]);
                                }
                            }
                        }
                    } else {
                        let q = m ^ 1;
                        let wq = &self.omega[q * LATENT_D..(q + 1) * LATENT_D];
                        let cu = cache.arg_u_s[m].cos();
                        let su = cache.arg_u_s[m].sin();
                        let cv = cache.arg_v_s[m].cos();
                        let sv = cache.arg_v_s[m].sin();

                        let d_au = -df_s * cache.a_sum[m] * su * cv;
                        let d_av = -df_s * cache.a_sum[m] * cu * sv;
                        for d in 0..LATENT_D {
                            du[d] += wm[d] * d_au;
                            dv[d] += wq[d] * d_av;
                        }
                        if let Some(ref mut g) = grads_opt {
                            g.alpha_sum[m] += df_s * (cu * cv) * sigmoid(self.alpha_sum[m]);
                            if update_phi {
                                g.phi_diff[m] += d_au;
                                g.phi_sum[m] += d_av;
                            }
                            if update_omega {
                                for d in 0..LATENT_D {
                                    g.omega[m * LATENT_D + d] += d_au * cache.u[d];
                                    g.omega[q * LATENT_D + d] += d_av * cache.v[d];
                                }
                            }
                        }
                    }
                }
            }
            PairingMode::AdditiveOnly => {
                for m in 0..N_MODES {
                    let df_d = dfeat[m];
                    let df_s = dfeat[N_MODES + m];
                    let cu_d = cache.arg_u_d[m].cos();
                    let cv_d = cache.arg_v_d[m].cos();
                    let su_d = cache.arg_u_d[m].sin();
                    let sv_d = cache.arg_v_d[m].sin();

                    let cu_s = cache.arg_u_s[m].cos();
                    let cv_s = cache.arg_v_s[m].cos();
                    let su_s = cache.arg_u_s[m].sin();
                    let sv_s = cache.arg_v_s[m].sin();

                    let du_d = -0.5 * df_d * cache.a_diff[m] * su_d;
                    let dv_d = -0.5 * df_d * cache.a_diff[m] * sv_d;
                    let du_s = 0.5 * df_s * cache.a_sum[m] * su_s;
                    let dv_s = -0.5 * df_s * cache.a_sum[m] * sv_s;

                    let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                    for d in 0..LATENT_D {
                        du[d] += wm[d] * (du_d + du_s);
                        dv[d] += wm[d] * (dv_d + dv_s);
                    }

                    if let Some(ref mut g) = grads_opt {
                        g.alpha_diff[m] +=
                            0.5 * df_d * (cv_d + cu_d) * sigmoid(self.alpha_diff[m]);
                        g.alpha_sum[m] +=
                            0.5 * df_s * (cv_s - cu_s) * sigmoid(self.alpha_sum[m]);
                        if update_phi {
                            g.phi_diff[m] += du_d + dv_d;
                            g.phi_sum[m] += du_s + dv_s;
                        }
                        if update_omega {
                            let gw = &mut g.omega[m * LATENT_D..(m + 1) * LATENT_D];
                            for d in 0..LATENT_D {
                                gw[d] += (du_d + du_s) * cache.u[d] + (dv_d + dv_s) * cache.v[d];
                            }
                        }
                    }
                }
            }
            PairingMode::PermutedPairing => {
                for m in 0..N_MODES {
                    let p = self.perm_pi[m];
                    let df_d = dfeat[m];
                    let df_s = dfeat[N_MODES + m];
                    let cos_d = cache.arg_d[m].cos();
                    let sin_d = cache.arg_d[m].sin();
                    let cos_s = cache.arg_s[m].cos();
                    let sin_s = cache.arg_s[m].sin();

                    let d_arg_d = -df_d * cache.a_diff[m] * sin_d;
                    let d_arg_s = -df_s * cache.a_sum[m] * sin_s;

                    let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                    let wp = &self.omega[p * LATENT_D..(p + 1) * LATENT_D];
                    for d in 0..LATENT_D {
                        dv[d] += wm[d] * (d_arg_d + d_arg_s);
                        du[d] += wp[d] * (d_arg_s - d_arg_d);
                    }

                    if let Some(ref mut g) = grads_opt {
                        g.alpha_diff[m] += df_d * cos_d * sigmoid(self.alpha_diff[m]);
                        g.alpha_sum[m] += df_s * cos_s * sigmoid(self.alpha_sum[m]);
                        if update_phi {
                            g.phi_diff[m] += d_arg_d;
                            g.phi_sum[m] += d_arg_s;
                        }
                        if update_omega {
                            for d in 0..LATENT_D {
                                g.omega[m * LATENT_D + d] += (d_arg_d + d_arg_s) * cache.v[d];
                                g.omega[p * LATENT_D + d] += (d_arg_s - d_arg_d) * cache.u[d];
                            }
                        }
                    }
                }
            }
        }

        (du, dv)
    }

    /// Gentle Dyson-like spectral self-organization regularizer (0 Riemann zeros, 0 primes):
    ///   - Uses smooth log-barrier / soft Riesz repulsion so task gradients lead while
    ///     level spacing <r> rises organically from Poisson (~0.38) into GUE (~0.55..0.68)
    ///     without collapsing into a rigid crystal.
    pub fn spectral_regularizer_grad(
        &self,
        lambda_vec: f64,
        lambda_rad: f64,
        lambda_alias: f64,
        lambda_bw: f64,
        g_omega: &mut [f64],
    ) -> f64 {
        let mut loss = 0.0;
        let mut radii = [0.0; N_MODES];
        for m in 0..N_MODES {
            let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
            let r2: f64 = wm.iter().map(|x| x * x).sum();
            radii[m] = (r2 + 1e-12).sqrt();
        }

        let pair_scale = 2.0 / ((N_MODES * (N_MODES - 1)) as f64);
        let eps_v = 0.35;
        // Short-range local Gaussian radial repulsion at natural mean spacing scale sigma_r ~ (R_MAX-R_MIN)/N_MODES:
        // prevents mode bunching / frequency collisions WITHOUT exerting outward explosion force on boundary modes!
        let sigma_r = 1.25 * (R_MAX - R_MIN) / (N_MODES as f64); // ~0.033
        let inv_2sr2 = 1.0 / (2.0 * sigma_r * sigma_r);
        let rad_scale = 1.0 / (N_MODES as f64);

        for m in 0..N_MODES {
            let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
            for n in (m + 1)..N_MODES {
                let wn = &self.omega[n * LATENT_D..(n + 1) * LATENT_D];
                if lambda_vec > 0.0 {
                    let mut d2 = 0.0;
                    let mut diff = [0.0; LATENT_D];
                    for d in 0..LATENT_D {
                        diff[d] = wm[d] - wn[d];
                        d2 += diff[d] * diff[d];
                    }
                    let denom = d2 + eps_v;
                    loss += lambda_vec * pair_scale / denom;
                    let coeff = -2.0 * lambda_vec * pair_scale / (denom * denom);
                    for d in 0..LATENT_D {
                        let gd = coeff * diff[d];
                        g_omega[m * LATENT_D + d] += gd;
                        g_omega[n * LATENT_D + d] -= gd;
                    }
                }
                if lambda_rad > 0.0 {
                    let dr = radii[m] - radii[n];
                    let e = (-dr * dr * inv_2sr2).exp();
                    loss += lambda_rad * rad_scale * e;
                    let d_dr = -2.0 * lambda_rad * rad_scale * inv_2sr2 * dr * e;
                    let inv_rm = 1.0 / radii[m];
                    let inv_rn = 1.0 / radii[n];
                    for d in 0..LATENT_D {
                        g_omega[m * LATENT_D + d] += d_dr * wm[d] * inv_rm;
                        g_omega[n * LATENT_D + d] -= d_dr * wn[d] * inv_rn;
                    }
                }
            }
        }

        if lambda_alias > 0.0 {
            let inv_pairs = 1.0 / ((N_MODES * (N_MODES - 1)) as f64);
            let inv_2s2 = 1.0 / (2.0 * 0.25 * 0.25);
            for m in 0..N_MODES {
                let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                for n in 0..N_MODES {
                    if m == n {
                        continue;
                    }
                    let wn = &self.omega[n * LATENT_D..(n + 1) * LATENT_D];
                    let mut d2 = 0.0;
                    let mut diff = [0.0; LATENT_D];
                    for d in 0..LATENT_D {
                        diff[d] = wm[d] - 2.0 * wn[d];
                        d2 += diff[d] * diff[d];
                    }
                    let e = (-d2 * inv_2s2).exp();
                    loss += lambda_alias * inv_pairs * e;
                    let coeff = -2.0 * lambda_alias * inv_pairs * inv_2s2 * e;
                    for d in 0..LATENT_D {
                        g_omega[m * LATENT_D + d] += coeff * diff[d];
                        g_omega[n * LATENT_D + d] -= 2.0 * coeff * diff[d];
                    }
                }
            }
        }

        if lambda_bw > 0.0 {
            let inv_m = 1.0 / (N_MODES as f64);
            for m in 0..N_MODES {
                let r = radii[m];
                let wm = &self.omega[m * LATENT_D..(m + 1) * LATENT_D];
                if r < R_MIN {
                    let viol = R_MIN - r;
                    loss += lambda_bw * inv_m * viol * viol;
                    let coeff = -2.0 * lambda_bw * inv_m * viol / r;
                    for d in 0..LATENT_D {
                        g_omega[m * LATENT_D + d] += coeff * wm[d];
                    }
                } else if r > R_MAX {
                    let viol = r - R_MAX;
                    loss += lambda_bw * inv_m * viol * viol;
                    let coeff = 2.0 * lambda_bw * inv_m * viol / r;
                    for d in 0..LATENT_D {
                        g_omega[m * LATENT_D + d] += coeff * wm[d];
                    }
                }
            }
        }

        loss
    }
}

/// Capacity-matched conventional neural core (2,008 params: 8 -> 48 -> 24 -> 16).
#[derive(Clone, Debug)]
pub struct MlpCore {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub w3: Vec<f64>,
    pub b3: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct MlpForwardCache {
    pub inp: [f64; 2 * LATENT_D],
    pub h1: [f64; MLP_H1],
    pub h2: [f64; MLP_H2],
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
        }
    }

    pub fn clear(&mut self) {
        self.w1.fill(0.0);
        self.b1.fill(0.0);
        self.w2.fill(0.0);
        self.b2.fill(0.0);
        self.w3.fill(0.0);
        self.b3.fill(0.0);
    }
}

impl MlpCore {
    pub fn new(rng: &mut Rng64) -> Self {
        let s1 = (2.0 / ((2 * LATENT_D + MLP_H1) as f64)).sqrt();
        let s2 = (2.0 / ((MLP_H1 + MLP_H2) as f64)).sqrt();
        let s3 = (2.0 / ((MLP_H2 + REL_DIM) as f64)).sqrt();
        Self {
            w1: (0..(MLP_H1 * 2 * LATENT_D))
                .map(|_| rng.normal() * s1)
                .collect(),
            b1: vec![0.0; MLP_H1],
            w2: (0..(MLP_H2 * MLP_H1)).map(|_| rng.normal() * s2).collect(),
            b2: vec![0.0; MLP_H2],
            w3: (0..(REL_DIM * MLP_H2))
                .map(|_| rng.normal() * s3)
                .collect(),
            b3: vec![0.0; REL_DIM],
        }
    }

    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.w3.len() + self.b3.len()
    }

    pub fn checksum(&self) -> u64 {
        checksum_f64_slices(&[&self.w1, &self.b1, &self.w2, &self.b2, &self.w3, &self.b3])
    }

    pub fn forward(&self, u: [f64; LATENT_D], v: [f64; LATENT_D]) -> MlpForwardCache {
        let mut inp = [0.0; 2 * LATENT_D];
        for d in 0..LATENT_D {
            inp[d] = u[d];
            inp[LATENT_D + d] = v[d];
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
        // Linear output z_rel matching SpectralCore's linear z_rel output cleanly!
        let mut z_rel = [0.0; REL_DIM];
        for i in 0..REL_DIM {
            let mut acc = self.b3[i];
            let row = &self.w3[i * MLP_H2..(i + 1) * MLP_H2];
            for j in 0..MLP_H2 {
                acc += row[j] * h2[j];
            }
            z_rel[i] = acc;
        }
        MlpForwardCache { inp, h1, h2, z_rel }
    }

    pub fn backward(
        &self,
        cache: &MlpForwardCache,
        dz_rel: &[f64; REL_DIM],
        update_core: bool,
        grads: Option<&mut MlpCoreGrads>,
    ) -> ([f64; LATENT_D], [f64; LATENT_D]) {
        let mut grads_opt = if update_core { grads } else { None };
        let mut dh2 = [0.0; MLP_H2];
        for i in 0..REL_DIM {
            let dt = dz_rel[i];
            let row_w = &self.w3[i * MLP_H2..(i + 1) * MLP_H2];
            if let Some(ref mut g) = grads_opt {
                g.b3[i] += dt;
                let row_g = &mut g.w3[i * MLP_H2..(i + 1) * MLP_H2];
                for j in 0..MLP_H2 {
                    row_g[j] += dt * cache.h2[j];
                    dh2[j] += dt * row_w[j];
                }
            } else {
                for j in 0..MLP_H2 {
                    dh2[j] += dt * row_w[j];
                }
            }
        }

        let mut dh1 = [0.0; MLP_H1];
        for i in 0..MLP_H2 {
            let dt = dh2[i] * (1.0 - cache.h2[i] * cache.h2[i]);
            let row_w = &self.w2[i * MLP_H1..(i + 1) * MLP_H1];
            if let Some(ref mut g) = grads_opt {
                g.b2[i] += dt;
                let row_g = &mut g.w2[i * MLP_H1..(i + 1) * MLP_H1];
                for j in 0..MLP_H1 {
                    row_g[j] += dt * cache.h1[j];
                    dh1[j] += dt * row_w[j];
                }
            } else {
                for j in 0..MLP_H1 {
                    dh1[j] += dt * row_w[j];
                }
            }
        }

        let mut dinp = [0.0; 2 * LATENT_D];
        for i in 0..MLP_H1 {
            let dt = dh1[i] * (1.0 - cache.h1[i] * cache.h1[i]);
            let row_w = &self.w1[i * 2 * LATENT_D..(i + 1) * 2 * LATENT_D];
            if let Some(ref mut g) = grads_opt {
                g.b1[i] += dt;
                let row_g = &mut g.w1[i * 2 * LATENT_D..(i + 1) * 2 * LATENT_D];
                for j in 0..(2 * LATENT_D) {
                    row_g[j] += dt * cache.inp[j];
                    dinp[j] += dt * row_w[j];
                }
            } else {
                for j in 0..(2 * LATENT_D) {
                    dinp[j] += dt * row_w[j];
                }
            }
        }

        let mut du = [0.0; LATENT_D];
        let mut dv = [0.0; LATENT_D];
        for d in 0..LATENT_D {
            du[d] = dinp[d];
            dv[d] = dinp[LATENT_D + d];
        }
        (du, dv)
    }
}

/// Compute the 16 canonical domain-agnostic relational invariants of any pair (u, v) in R^4:
///   [0..4]   antisymmetric displacement: v_d - u_d
///   [4..8]   symmetric even proximity:   2 * exp(-0.75 * (v_d - u_d)^2) - 1
///   [8..12]  symmetric sum binding:      cos(0.75 * (u_d + v_d))
///   [12..16] chiral bivector (u ^ v):    tanh(u_0 v_1 - u_1 v_0), tanh(u_2 v_3 - u_3 v_2),
///                                        tanh(u_0 v_2 - u_2 v_0), tanh(u_1 v_3 - u_3 v_1)
/// Computed purely from detached (u, v) — zero target-domain info, identical for Spectral & MLP.
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

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr_encoder: f64,
    pub lr_core: f64,
    pub lr_head: f64,
    pub wd_encoder: f64,
    pub wd_core: f64,
    pub wd_head: f64,
    pub w_mse: f64,
    pub w_ce: f64,
    pub w_ssl: f64,
    pub lambda_vec: f64,
    pub lambda_rad: f64,
    pub lambda_alias: f64,
    pub lambda_bw: f64,
    pub lambda_var: f64,
    pub lambda_cov: f64,
    pub latent_norm_reg: f64,
    pub latent_jitter: f64,
    pub freeze_core: bool,
}

#[derive(Clone, Debug)]
pub struct SpectralModel {
    pub encoder: DomainEncoder,
    pub core: SpectralCore,
    pub head: DomainHead,
}

impl SpectralModel {
    pub fn new(
        obs_dim: usize,
        reg_dim: usize,
        cls_dim: usize,
        core: SpectralCore,
        rng: &mut Rng64,
    ) -> Self {
        Self {
            encoder: DomainEncoder::new(obs_dim, rng),
            core,
            head: DomainHead::new(reg_dim, cls_dim, rng),
        }
    }

    pub fn predict_pair(
        &self,
        xa: &[f64],
        xb: &[f64],
    ) -> (Vec<f64>, Vec<f64>, [f64; REL_DIM], [f64; 2 * N_MODES]) {
        let u = self.encoder.forward(xa);
        let v = self.encoder.forward(xb);
        let cache = self.core.forward(u, v);
        let (reg, logits) = self.head.forward(&cache.z_rel);
        (reg, logits, cache.z_rel, cache.feat)
    }

    pub fn train(&mut self, domain: &DomainData, train_indices: &[usize], cfg: &TrainConfig, rng: &mut Rng64) {
        self.core.assert_no_riemann_injection(false);
        let pre_checksum = self.core.checksum();

        let mut opt_ew = AdamBuf::new(self.encoder.w.len());
        let mut opt_eb = AdamBuf::new(self.encoder.b.len());
        let mut opt_hw_r = AdamBuf::new(self.head.w_reg.len());
        let mut opt_hb_r = AdamBuf::new(self.head.b_reg.len());
        let mut opt_hw_c = AdamBuf::new(self.head.w_cls.len());
        let mut opt_hb_c = AdamBuf::new(self.head.b_cls.len());

        // Auxiliary self-supervised canonical relational invariants head (16 -> 16),
        // discarded after Domain A training; identical for SpectralCore and MlpCore.
        let mut ssl_head = DomainHead::new(REL_DIM, 0, &mut rng.fork(0x5511));
        let mut opt_sw = AdamBuf::new(ssl_head.w_reg.len());
        let mut opt_sb = AdamBuf::new(ssl_head.b_reg.len());
        let mut gw_s = vec![0.0; ssl_head.w_reg.len()];
        let mut gb_s = vec![0.0; ssl_head.b_reg.len()];
        let mut dummy_c1 = Vec::new();
        let mut dummy_c2 = Vec::new();

        let mut opt_om = AdamBuf::new(self.core.omega.len());
        let mut opt_pd = AdamBuf::new(self.core.phi_diff.len());
        let mut opt_ps = AdamBuf::new(self.core.phi_sum.len());
        let mut opt_ad = AdamBuf::new(self.core.alpha_diff.len());
        let mut opt_as = AdamBuf::new(self.core.alpha_sum.len());
        let mut opt_wp = AdamBuf::new(self.core.w_pair.len());
        let mut opt_bp = AdamBuf::new(self.core.b_pair.len());

        let mut gw_e = vec![0.0; self.encoder.w.len()];
        let mut gb_e = vec![0.0; self.encoder.b.len()];
        let mut gw_r = vec![0.0; self.head.w_reg.len()];
        let mut gb_r = vec![0.0; self.head.b_reg.len()];
        let mut gw_c = vec![0.0; self.head.w_cls.len()];
        let mut gb_c = vec![0.0; self.head.b_cls.len()];
        let mut g_core = SpectralCoreGrads::zeros();

        let n_tr = train_indices.len();
        let bs = cfg.batch_size.min(n_tr).max(1);
        let mut order: Vec<usize> = train_indices.to_vec();
        let mut cursor = order.len();

        let mut batch_ids = Vec::with_capacity(bs);
        let mut batch_caches = Vec::with_capacity(bs);
        let mut batch_zs = Vec::with_capacity(bs);
        let mut batch_dzs = Vec::with_capacity(bs);

        for step in 1..=cfg.steps {
            let frac = (step as f64) / (cfg.steps as f64);
            let mult = if step <= 20 {
                (step as f64) / 20.0
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
            if !cfg.freeze_core {
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

                if !cfg.freeze_core && cfg.w_ssl > 0.0 {
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

            if !cfg.freeze_core && (cfg.lambda_var > 0.0 || cfg.lambda_cov > 0.0) {
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
                    !cfg.freeze_core,
                    if cfg.freeze_core { None } else { Some(&mut g_core) },
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

            if !cfg.freeze_core && self.core.kind == SpectrumKind::Adaptive {
                self.core.spectral_regularizer_grad(
                    cfg.lambda_vec,
                    cfg.lambda_rad,
                    cfg.lambda_alias,
                    cfg.lambda_bw,
                    &mut g_core.omega,
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

            if !cfg.freeze_core {
                if cfg.w_ssl > 0.0 {
                    opt_sw.step(&mut ssl_head.w_reg, &gw_s, cfg.lr_head * mult, cfg.wd_head, step, 3.0);
                    opt_sb.step(&mut ssl_head.b_reg, &gb_s, cfg.lr_head * mult, 0.0, step, 3.0);
                }
                if self.core.kind == SpectrumKind::Adaptive {
                    opt_om.step(&mut self.core.omega, &g_core.omega, cfg.lr_core * mult, 0.0, step, 3.0);
                }
                if !self.core.zero_phase_lock {
                    opt_pd.step(&mut self.core.phi_diff, &g_core.phi_diff, cfg.lr_core * mult, 0.0, step, 3.0);
                    opt_ps.step(&mut self.core.phi_sum, &g_core.phi_sum, cfg.lr_core * mult, 0.0, step, 3.0);
                }
                opt_ad.step(&mut self.core.alpha_diff, &g_core.alpha_diff, cfg.lr_core * mult, 1e-4, step, 3.0);
                opt_as.step(&mut self.core.alpha_sum, &g_core.alpha_sum, cfg.lr_core * mult, 1e-4, step, 3.0);
                opt_wp.step(&mut self.core.w_pair, &g_core.w_pair, cfg.lr_core * mult, cfg.wd_core, step, 3.0);
                opt_bp.step(&mut self.core.b_pair, &g_core.b_pair, cfg.lr_core * mult, 0.0, step, 3.0);
            }
        }

        if cfg.freeze_core {
            assert_eq!(
                pre_checksum,
                self.core.checksum(),
                "FROZEN CORE VIOLATION: SpectralCore bytes changed during frozen-core transfer!"
            );
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
        let pre_checksum = self.core.checksum();

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

        for step in 1..=cfg.steps {
            let frac = (step as f64) / (cfg.steps as f64);
            let mult = if step <= 20 {
                (step as f64) / 20.0
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
            if !cfg.freeze_core {
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

                if !cfg.freeze_core && cfg.w_ssl > 0.0 {
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

            if !cfg.freeze_core && (cfg.lambda_var > 0.0 || cfg.lambda_cov > 0.0) {
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
                    !cfg.freeze_core,
                    if cfg.freeze_core { None } else { Some(&mut g_core) },
                );

                if cfg.latent_norm_reg > 0.0 {
                    let target_r2 = 1.44;
                    let ru2: f64 = cache.inp[0..LATENT_D].iter().map(|x| x * x).sum();
                    let rv2: f64 = cache.inp[LATENT_D..2 * LATENT_D].iter().map(|x| x * x).sum();
                    let cu = inv_bs * cfg.latent_norm_reg * 2.0 * (ru2 - target_r2);
                    let cv = inv_bs * cfg.latent_norm_reg * 2.0 * (rv2 - target_r2);
                    for d in 0..LATENT_D {
                        du[d] += cu * cache.inp[d];
                        dv[d] += cv * cache.inp[LATENT_D + d];
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

            if !cfg.freeze_core {
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
        }

        if cfg.freeze_core {
            assert_eq!(
                pre_checksum,
                self.core.checksum(),
                "FROZEN CORE VIOLATION: MlpCore bytes changed during frozen-core transfer!"
            );
        }
    }
}
