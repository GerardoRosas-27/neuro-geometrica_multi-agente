//! Adaptive Spectral Collapse loaded onto frozen ÓRBITA-v1 + Prime-Frequency Spectral Sieve.
//!
//! Strictly isolated module (`exp/orbita-adaptive-spectral`):
//!   - Never modifies `orbita/src/model.rs` or any v4 files; verifies `d_after.raw == d.raw`.
//!   - Trains Suite O1 strictly on `s.ds.train` ONLY (`queries == 0`, `leaked == 0`, `CLEAN`, `fresh_ok`).
//!   - Runs Suites O1 (Official §16 TEST + E33), O2 (Unknown-Context & Compositional Rule-Pair Collapse),
//!     O3 (Continual Few-Shot Learning of ÓRBITA's 4 Brand-New Nonlinear Families 8..11 vs LAT/OBS/BASE/MLP),
//!     and O4 (Prime-Frequency Spectral Sieve: extracting ONLY exact prime / Riemann / log-prime frequencies
//!     from the already-trained inference field vs matched-size composite/even/random controls).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::constelacion4::{gen_heldout, rule as c4_rule, Ep as C4Ep};
use crate::fam::reid;
use crate::latent::{ctx8, lat_decide, lat_step, Entry as LatEntry, Ep as LatEp};
use crate::model::Orbita;
use cdt_rqm_epr::field_autonomy_v4::{
    collect_experience, evaluate, target_h, AuditFlags, Auditor, ExpMode, HORIZONS, ITERABLE_FAMS,
    K_POINTS,
};
use cdt_rqm_epr::v4_controls::Predictor;
use cdt_rqm_epr::v4_dataset::{
    context_of, generate_and_seal, make_object, DatasetConfig, Example, ObjectClass, Sealed, N_FAM,
};
use cdt_rqm_epr::v4_metrics as m;
use cdt_rqm_epr::v4_provenance::snapshot;
use rand_xoshiro::rand_core::SeedableRng;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// ─── Constants (Preregistered in docs/orbita_adaptive_spectral_v1/PLAN.md) ───

pub const M_DYN: usize = 16; // Modes per family in Suite O1 (16*7 + 2 = 114 params/fam -> 912 total -> 3,514)
pub const MLP_H_O1: usize = 16; // 4->16->2 = 114 params/fam -> 912 total -> 3,514 (exact 1:1 parity!)
pub const O3_H1: usize = 32; // Layer-1 spectral wave / MLP width in Suite O3
pub const O3_H2: usize = 32; // Layer-2 spectral interference / MLP width in Suite O3 (1,282 params/expert -> 5,128 total)

pub const K_RULES_O2: usize = 12; // 0..7 known + 8..11 new nonlinear families
pub const DIM_SIG: usize = 16; // Opaque geometric transition signature dimension
pub const DIM_PHASE: usize = 3; // 3D spectral collapse cavity
pub const M1D_O2: usize = 18; // 1D wave modes per cavity axis
pub const M3D_O2: usize = 64; // 3D coherence wave modes
pub const CAVITY_LIM: f64 = 1.75;
pub const DC_3D: f64 = -0.35;

pub const NEW_FAMS_ALL: [usize; 4] = [8, 9, 10, 11];
pub const PRIMES_10: [usize; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
// Bandwidth- and cardinality-matched [2..29] controls (10 modes each, identical spectral span!)
pub const COMPOSITES_10: [usize; 10] = [4, 6, 8, 9, 12, 15, 18, 21, 24, 28];
pub const EVENS_10: [usize; 10] = [2, 4, 6, 10, 12, 16, 18, 22, 26, 28];

// ─── Deterministic PRNG (SplitMix64 + Xoshiro256**) & Adam ──────────────────

#[derive(Clone, Debug)]
pub struct Rng64 {
    s: [u64; 4],
    spare: Option<f64>,
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng64 {
    pub fn from_seed(seed: u64, stream: u64) -> Self {
        let mut sm = seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
        Self {
            s: [
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
            ],
            spare: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let res = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        res
    }

    #[inline]
    pub fn u01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.u01()
    }

    #[inline]
    pub fn usize_range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let r = (-2.0 * self.u01().ln()).sqrt();
        let th = 2.0 * PI * self.u01();
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}

#[derive(Clone, Debug)]
pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    step: usize,
}

impl Adam {
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
        }
    }

    pub fn update(&mut self, p: &mut [f64], g: &[f64], lr: f64, wd: f64) {
        self.step += 1;
        let b1: f64 = 0.9;
        let b2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);
        for i in 0..p.len() {
            let gi = g[i] + wd * p[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            p[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
}

// ─── Exact Riemann Zeros (Euler-Maclaurin Hardy Z(t) + Brent) & Spectra ────

fn ln_gamma_complex(re: f64, im: f64) -> (f64, f64) {
    let mut x = re;
    let y = im;
    let mut sub_re = 0.0;
    let mut sub_im = 0.0;
    while x < 10.0 {
        sub_re += 0.5 * (x * x + y * y).ln();
        sub_im += y.atan2(x);
        x += 1.0;
    }
    let r2 = x * x + y * y;
    let ln_r = 0.5 * r2.ln();
    let phi = y.atan2(x);
    let mut res_re = (x - 0.5) * ln_r - y * phi - x + 0.5 * (2.0 * PI).ln();
    let mut res_im = y * ln_r + (x - 0.5) * phi - y;
    let inv_re = x / r2;
    let inv_im = -y / r2;
    let inv2_re = inv_re * inv_re - inv_im * inv_im;
    let inv2_im = 2.0 * inv_re * inv_im;
    let coeffs = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360360.0,
    ];
    let mut pow_re = inv_re;
    let mut pow_im = inv_im;
    for &c in &coeffs {
        res_re += c * pow_re;
        res_im += c * pow_im;
        let nr = pow_re * inv2_re - pow_im * inv2_im;
        let ni = pow_re * inv2_im + pow_im * inv2_re;
        pow_re = nr;
        pow_im = ni;
    }
    (res_re - sub_re, res_im - sub_im)
}

fn riemann_siegel_theta(t: f64) -> f64 {
    let (_, im) = ln_gamma_complex(0.25, 0.5 * t);
    im - 0.5 * t * PI.ln()
}

fn c_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn c_div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn zeta_critical(t: f64) -> (f64, f64) {
    let n_cut: usize = ((t.abs() * 0.65).ceil() as usize).clamp(24, 120);
    let mut sum = (0.0, 0.0);
    for n in 1..n_cut {
        let nf = n as f64;
        let mag = nf.powf(-0.5);
        let ang = -t * nf.ln();
        sum.0 += mag * ang.cos();
        sum.1 += mag * ang.sin();
    }
    let nf = n_cut as f64;
    let n_pow = {
        let mag = nf.powf(-0.5);
        let ang = -t * nf.ln();
        (mag * ang.cos(), mag * ang.sin())
    };
    sum.0 += 0.5 * n_pow.0;
    sum.1 += 0.5 * n_pow.1;
    let n_1ms = {
        let mag = nf.powf(0.5);
        let ang = -t * nf.ln();
        (mag * ang.cos(), mag * ang.sin())
    };
    let tail = c_div(n_1ms, (-0.5, t));
    sum.0 += tail.0;
    sum.1 += tail.1;

    let b2k = [
        1.0 / 6.0,
        -1.0 / 30.0,
        1.0 / 42.0,
        -1.0 / 30.0,
        5.0 / 66.0,
        -691.0 / 2730.0,
        7.0 / 6.0,
    ];
    let inv_n2 = 1.0 / (nf * nf);
    let mut fact = 1.0_f64;
    let mut poch = (1.0, 0.0);
    let mut n_term = (n_pow.0 / nf, n_pow.1 / nf);
    for (idx, &bk) in b2k.iter().enumerate() {
        let k = idx + 1;
        let m1 = 2 * k - 1;
        if k == 1 {
            poch = (0.5, t);
            fact = 2.0;
        } else {
            let f1 = ((m1 - 2) as f64 + 0.5, t);
            let f2 = ((m1 - 1) as f64 + 0.5, t);
            poch = c_mul(poch, c_mul(f1, f2));
            fact *= ((2 * k - 1) * (2 * k)) as f64;
            n_term.0 *= inv_n2;
            n_term.1 *= inv_n2;
        }
        let coeff = bk / fact;
        let prod = c_mul(poch, n_term);
        sum.0 += coeff * prod.0;
        sum.1 += coeff * prod.1;
    }
    sum
}

fn hardy_z(t: f64) -> f64 {
    let th = riemann_siegel_theta(t);
    let z = zeta_critical(t);
    th.cos() * z.0 - th.sin() * z.1
}

fn brent_root<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let mut fa = f(a);
    let mut fb = f(b);
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    for _ in 0..80 {
        if fb.abs() < tol || (b - a).abs() < tol {
            return b;
        }
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 {
            return b;
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if (a - c).abs() < 1e-15 {
                (2.0 * xm * s, 1.0 - s)
            } else {
                let q1 = fa / fc;
                let r1 = fb / fc;
                (
                    s * (2.0 * xm * q1 * (q1 - r1) - (b - a) * (r1 - 1.0)),
                    (q1 - 1.0) * (r1 - 1.0) * (s - 1.0),
                )
            };
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };
            if 2.0 * p < (3.0 * xm * q - (tol1 * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b += d;
        } else {
            b += if xm >= 0.0 { tol1 } else { -tol1 };
        }
        fb = f(b);
    }
    b
}

pub fn compute_riemann_zeros(n_zeros: usize) -> Vec<f64> {
    let mut zeros: Vec<f64> = Vec::with_capacity(n_zeros);
    let mut t = 12.0;
    let mut z_prev = hardy_z(t);
    let step = 0.08;
    while zeros.len() < n_zeros {
        let t_next = t + step;
        let z_next = hardy_z(t_next);
        if z_prev * z_next <= 0.0 {
            let root = brent_root(hardy_z, t, t_next, 1e-12);
            if zeros.last().map_or(true, |&prev| (root - prev).abs() > 1e-4) {
                zeros.push(root);
            }
        }
        t = t_next;
        z_prev = z_next;
    }
    zeros
}

pub fn scaled_riemann(zeros: &[f64], n: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let z0 = zeros[0];
    let z1 = zeros[n - 1];
    let span = (z1 - z0).max(1e-9);
    (0..n)
        .map(|i| fmin + (fmax - fmin) * (zeros[i] - z0) / span)
        .collect()
}

pub fn mean_spacing_r(levels: &[f64]) -> f64 {
    if levels.len() < 3 {
        return 0.0;
    }
    let mut s: Vec<f64> = levels.iter().map(|v| v.abs()).collect();
    s.sort_by(|a, b| a.total_cmp(b));
    let gaps: Vec<f64> = s.windows(2).map(|w| (w[1] - w[0]).max(1e-12)).collect();
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for w in gaps.windows(2) {
        let lo = w[0].min(w[1]);
        let hi = w[0].max(w[1]);
        if hi > 1e-12 {
            sum += lo / hi;
            cnt += 1;
        }
    }
    if cnt > 0 { sum / cnt as f64 } else { 0.0 }
}

// ─── Linear Algebra Utility (Cholesky / Ridge) ──────────────────────────────

fn solve_ridge(a: &[f64], b: &[f64], n: usize, lam: f64) -> Vec<f64> {
    let mut mat = a.to_vec();
    for i in 0..n {
        mat[i * n + i] += lam;
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = mat[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = s.max(1e-18).sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * z[k];
        }
        z[i] = s / l[i * n + i];
    }
    let mut w = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in i + 1..n {
            s -= l[k * n + i] * w[k];
        }
        w[i] = s / l[i * n + i];
    }
    w
}

// ─── Spectral Arm Enum ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecMode {
    Adaptive,
    RiemannFixed,
    FourierFixed,
    PoissonFixed,
}

impl SpecMode {
    pub fn name(self) -> &'static str {
        match self {
            SpecMode::Adaptive => "ORBITA_ASC",
            SpecMode::RiemannFixed => "ORBITA_RIEMANN_FIXED",
            SpecMode::FourierFixed => "ORBITA_FOURIER_FIXED",
            SpecMode::PoissonFixed => "ORBITA_POISSON_FIXED",
        }
    }
    pub fn adapts_freq(self) -> bool {
        matches!(self, SpecMode::Adaptive)
    }
}

// ─── Component 1: 4D Adaptive Spectral Wave Field (for Suites O1, O3, O4a) ─

/// Planar wave superposition over 4D phase space u = [x, y, p1, p2] -> Δq ∈ R^2:
///   Δq_d(u) = b_d + Σ_{m=1..M} a_{d,m} cos(k_m · u + phi_m)
/// Exact parameter count: 7*M + 2 (M=16 -> 114 params; M=24 -> 170 params).
#[derive(Clone, Debug)]
pub struct WaveField4D {
    pub n_modes: usize,
    pub mode: SpecMode,
    pub k: Vec<[f64; 4]>,
    pub a: Vec<[f64; 2]>,
    pub phi: Vec<f64>,
    pub bias: [f64; 2],
    pub init_r: f64,
}

impl WaveField4D {
    pub fn new(
        n_modes: usize,
        mode: SpecMode,
        zeros: Option<&[f64]>,
        k_min: f64,
        k_max: f64,
        rng: &mut Rng64,
    ) -> Self {
        if mode == SpecMode::Adaptive {
            assert!(
                zeros.is_none(),
                "AdaptiveSpectral must NEVER receive Riemann zeros!"
            );
        }
        let mut k = Vec::with_capacity(n_modes);
        let mut phi = Vec::with_capacity(n_modes);
        let init_r = if mode == SpecMode::FourierFixed {
            let w0 = (k_max - k_min) * 0.45;
            for m in 0..n_modes {
                let n0 = ((m % 3) as f64) - 1.0;
                let n1 = (((m / 3) % 3) as f64) - 1.0;
                let n2 = (((m / 9) % 2) as f64) - 0.5;
                let n3 = (((m / 18) % 2) as f64) - 0.5;
                let mut kv = [n0 * w0, n1 * w0, n2 * w0, n3 * w0];
                if kv.iter().all(|v| v.abs() < 1e-9) {
                    kv[0] = 2.0 * w0;
                }
                k.push(kv);
                phi.push(rng.uniform(0.0, 2.0 * PI));
            }
            let norms: Vec<f64> = k
                .iter()
                .map(|km| (km[0] * km[0] + km[1] * km[1] + km[2] * km[2] + km[3] * km[3]).sqrt())
                .collect();
            mean_spacing_r(&norms)
        } else {
            let radii: Vec<f64> = match mode {
                SpecMode::Adaptive | SpecMode::PoissonFixed => {
                    let mut v: Vec<f64> = (0..n_modes).map(|_| rng.uniform(k_min, k_max)).collect();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v
                }
                SpecMode::RiemannFixed => {
                    let zs = zeros.expect("RiemannFixed needs exact zeros");
                    scaled_riemann(zs, n_modes, k_min, k_max)
                }
                SpecMode::FourierFixed => unreachable!(),
            };
            for m in 0..n_modes {
                let mut dir = [rng.normal(), rng.normal(), rng.normal(), rng.normal()];
                // Slightly gentler initial p-components for smooth parameter interpolation
                dir[2] *= 0.7;
                dir[3] *= 0.7;
                let nrm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] + dir[3] * dir[3])
                    .sqrt()
                    .max(1e-9);
                let r = radii[m];
                k.push([
                    r * dir[0] / nrm,
                    r * dir[1] / nrm,
                    r * dir[2] / nrm,
                    r * dir[3] / nrm,
                ]);
                phi.push(rng.uniform(0.0, 2.0 * PI));
            }
            mean_spacing_r(&radii)
        };
        Self {
            n_modes,
            mode,
            k,
            a: vec![[0.0, 0.0]; n_modes], // Zero init -> ORBITA_ASC == ORBITA_BASE bit-exact at step 0!
            phi,
            bias: [0.0, 0.0],
            init_r,
        }
    }

    pub fn n_params(&self) -> usize {
        7 * self.n_modes + 2
    }

    #[inline]
    pub fn eval(&self, u: [f64; 4]) -> [f64; 2] {
        let mut o0 = self.bias[0];
        let mut o1 = self.bias[1];
        for m in 0..self.n_modes {
            let km = &self.k[m];
            let ang = km[0] * u[0] + km[1] * u[1] + km[2] * u[2] + km[3] * u[3] + self.phi[m];
            let c = ang.cos();
            o0 += self.a[m][0] * c;
            o1 += self.a[m][1] * c;
        }
        [o0, o1]
    }

    pub fn learned_r(&self) -> f64 {
        let norms: Vec<f64> = self
            .k
            .iter()
            .map(|km| (km[0] * km[0] + km[1] * km[1] + km[2] * km[2] + km[3] * km[3]).sqrt())
            .collect();
        mean_spacing_r(&norms)
    }

    #[cfg(test)]
    fn pack(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        v.push(self.bias[0]);
        v.push(self.bias[1]);
        for m in 0..self.n_modes {
            v.extend_from_slice(&self.k[m]);
            v.extend_from_slice(&self.a[m]);
            v.push(self.phi[m]);
        }
        v
    }

    #[cfg(test)]
    fn unpack(&mut self, v: &[f64]) {
        self.bias[0] = v[0];
        self.bias[1] = v[1];
        let mut idx = 2;
        for m in 0..self.n_modes {
            self.k[m] = [v[idx], v[idx + 1], v[idx + 2], v[idx + 3]];
            self.a[m] = [v[idx + 4], v[idx + 5]];
            self.phi[m] = v[idx + 6];
            idx += 7;
        }
    }

    /// Accumulates analytical gradients for a single (u, target, weight) residual pair.
    #[cfg(test)]
    pub fn accum_grad(&self, u: [f64; 4], dout: [f64; 2], grads: &mut [f64]) {
        grads[0] += dout[0];
        grads[1] += dout[1];
        let adapt_k = self.mode.adapts_freq();
        for m in 0..self.n_modes {
            let base = 2 + 7 * m;
            let km = &self.k[m];
            let ang = km[0] * u[0] + km[1] * u[1] + km[2] * u[2] + km[3] * u[3] + self.phi[m];
            let c = ang.cos();
            let s = ang.sin();
            grads[base + 4] += dout[0] * c;
            grads[base + 5] += dout[1] * c;
            let dang = -(dout[0] * self.a[m][0] + dout[1] * self.a[m][1]) * s;
            grads[base + 6] += dang;
            if adapt_k {
                grads[base] += dang * u[0];
                grads[base + 1] += dang * u[1];
                grads[base + 2] += dang * u[2];
                grads[base + 3] += dang * u[3];
            }
        }
    }

    /// Solves exact weighted ridge regression for `[bias, a_0..a_{M-1}]` given fixed `(k, phi)`.
    fn solve_optimal_amplitudes(&mut self, samples: &[([f64; 4], [f64; 2], f64)], lam: f64) {
        let nf = self.n_modes + 1; // [1.0, cos(k_0·u + phi_0), ...]
        let mut ata = vec![0.0; nf * nf];
        let mut atb0 = vec![0.0; nf];
        let mut atb1 = vec![0.0; nf];
        let mut feat = vec![0.0; nf];
        let n_f = samples.len().max(1) as f64;
        for &(u, tgt, w) in samples {
            let sw = w / n_f;
            feat[0] = 1.0;
            for m in 0..self.n_modes {
                let km = &self.k[m];
                feat[m + 1] =
                    (km[0] * u[0] + km[1] * u[1] + km[2] * u[2] + km[3] * u[3] + self.phi[m]).cos();
            }
            for r in 0..nf {
                let fr = feat[r] * sw;
                for c in r..nf {
                    ata[r * nf + c] += fr * feat[c];
                }
                atb0[r] += fr * tgt[0];
                atb1[r] += fr * tgt[1];
            }
        }
        for r in 0..nf {
            for c in 0..r {
                ata[r * nf + c] = ata[c * nf + r];
            }
        }
        let w0 = solve_ridge(&ata, &atb0, nf, lam);
        let w1 = solve_ridge(&ata, &atb1, nf, lam);
        self.bias = [w0[0], w1[0]];
        for m in 0..self.n_modes {
            self.a[m] = [w0[m + 1], w1[m + 1]];
        }
    }

    /// Trains via Variable Projection: alternates closed-form optimal ridge amplitudes with
    /// Adam analytical gradient adaptation on wavevectors `k_m` (when Adaptive) and phases `phi_m`.
    pub fn fit(
        &mut self,
        samples: &[([f64; 4], [f64; 2], f64)],
        steps: usize,
        lr_amp: f64,
        lr_freq: f64,
        wd_amp: f64,
        kq_max: f64,
        kp_max: f64,
        repel_w: f64,
    ) {
        if samples.is_empty() {
            return;
        }
        let _ = lr_amp;
        let adapt_k = self.mode.adapts_freq();
        let lam = wd_amp.max(1e-5);
        // Scale spectral regularization strictly proportionally to target residual variance so
        // tiny residuals (e.g. O1 ~0.006) are NEVER overwhelmed by unscaled repulsion gradients.
        let target_var = (samples
            .iter()
            .map(|&(_, tgt, w)| w * (tgt[0] * tgt[0] + tgt[1] * tgt[1]))
            .sum::<f64>()
            / (samples.len() as f64))
            .max(1e-10);
        // Exact closed-form amplitude projection right from step 0!
        self.solve_optimal_amplitudes(samples, lam);

        let n_cycles = (steps / 25).clamp(6, 28);
        let inner_steps = 12usize;
        let np_nl = self.n_modes * 5; // [kx, ky, kp1, kp2, phi] per mode
        let mut p_nl = vec![0.0; np_nl];
        for m in 0..self.n_modes {
            p_nl[5 * m..5 * m + 4].copy_from_slice(&self.k[m]);
            p_nl[5 * m + 4] = self.phi[m];
        }
        let mut adam = Adam::new(np_nl);
        let n_tot = samples.len();
        let batch_sz = if n_tot <= 320 { n_tot } else { 240 };
        let mut brng = Rng64::from_seed(n_tot as u64 ^ (self.n_modes as u64), 0xBEEF);

        for cyc in 0..n_cycles {
            for ist in 0..inner_steps {
                for m in 0..self.n_modes {
                    self.k[m] = [p_nl[5 * m], p_nl[5 * m + 1], p_nl[5 * m + 2], p_nl[5 * m + 3]];
                    self.phi[m] = p_nl[5 * m + 4];
                }
                let mut g_nl = vec![0.0; np_nl];
                let n_f = batch_sz as f64;
                for bi in 0..batch_sz {
                    let &(u, tgt, w) = if batch_sz == n_tot {
                        &samples[bi]
                    } else {
                        &samples[brng.usize_range(n_tot)]
                    };
                    let pred = self.eval(u);
                    let d0 = 2.0 * w * (pred[0] - tgt[0]) / n_f;
                    let d1 = 2.0 * w * (pred[1] - tgt[1]) / n_f;
                    for m in 0..self.n_modes {
                        let km = &self.k[m];
                        let ang =
                            km[0] * u[0] + km[1] * u[1] + km[2] * u[2] + km[3] * u[3] + self.phi[m];
                        let dang = -(d0 * self.a[m][0] + d1 * self.a[m][1]) * ang.sin();
                        g_nl[5 * m + 4] += dang;
                        if adapt_k {
                            g_nl[5 * m] += dang * u[0];
                            g_nl[5 * m + 1] += dang * u[1];
                            g_nl[5 * m + 2] += dang * u[2];
                            g_nl[5 * m + 3] += dang * u[3];
                        }
                    }
                }
                if adapt_k {
                    let mut norms = Vec::with_capacity(self.n_modes);
                    for m in 0..self.n_modes {
                        let km = &self.k[m];
                        let rq2 = km[0] * km[0] + km[1] * km[1];
                        let rp2 = km[2] * km[2] + km[3] * km[3];
                        if rq2 > kq_max * kq_max {
                            let c = 0.08 * target_var * (rq2 - kq_max * kq_max);
                            g_nl[5 * m] += c * km[0];
                            g_nl[5 * m + 1] += c * km[1];
                        }
                        if rp2 > kp_max * kp_max {
                            let c = 0.14 * target_var * (rp2 - kp_max * kp_max);
                            g_nl[5 * m + 2] += c * km[2];
                            g_nl[5 * m + 3] += c * km[3];
                        }
                        norms.push(((rq2 + rp2).sqrt(), m));
                    }
                    if repel_w > 0.0 && self.n_modes >= 3 {
                        norms.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let tgap = (kq_max - 0.3) / (self.n_modes as f64);
                        for i in 0..self.n_modes - 1 {
                            let (r1, m1) = norms[i];
                            let (r2, m2) = norms[i + 1];
                            if r2 - r1 < tgap * 0.92 {
                                let push = repel_w * target_var * (tgap * 0.92 - (r2 - r1));
                                let inv1 = 1.0 / r1.max(1e-6);
                                let inv2 = 1.0 / r2.max(1e-6);
                                for j in 0..4 {
                                    g_nl[5 * m1 + j] += push * self.k[m1][j] * inv1;
                                    g_nl[5 * m2 + j] -= push * self.k[m2][j] * inv2;
                                }
                            }
                        }
                    }
                }
                let prog = (cyc * inner_steps + ist) as f64 / ((n_cycles * inner_steps) as f64);
                let decay = 0.15 + 0.85 * 0.5 * (1.0 + (PI * prog).cos());
                let prev = p_nl.clone();
                adam.update(&mut p_nl, &g_nl, lr_freq * decay, 0.0);
                if !adapt_k {
                    // Strictly preserve frozen wavevectors k_m
                    for m in 0..self.n_modes {
                        p_nl[5 * m..5 * m + 4].copy_from_slice(&prev[5 * m..5 * m + 4]);
                    }
                }
            }
            for m in 0..self.n_modes {
                self.k[m] = [p_nl[5 * m], p_nl[5 * m + 1], p_nl[5 * m + 2], p_nl[5 * m + 3]];
                self.phi[m] = p_nl[5 * m + 4];
            }
            self.solve_optimal_amplitudes(samples, lam);
        }
    }
}

// ─── Matched-Capacity Conventional MLP Residual Adapter (Exact 1:1 Parity) ──

/// Conventional 2-layer MLP: 4 -> H (tanh) -> 2.
/// Exact parameter count: H*4 + H + 2*H + 2 = 7*H + 2 (H=16 -> 114; H=24 -> 170).
#[derive(Clone, Debug)]
pub struct MlpRes4D {
    pub h: usize,
    pub w1: Vec<[f64; 4]>,
    pub b1: Vec<f64>,
    pub w2: Vec<[f64; 2]>,
    pub b2: [f64; 2],
}

impl MlpRes4D {
    pub fn new(h: usize, rng: &mut Rng64) -> Self {
        let sc = (1.0 / 4.0_f64).sqrt();
        let w1 = (0..h)
            .map(|_| {
                [
                    sc * rng.normal(),
                    sc * rng.normal(),
                    sc * rng.normal(),
                    sc * rng.normal(),
                ]
            })
            .collect();
        let b1 = vec![0.0; h];
        // Zero output layer init (same exact identity-residual start as WaveField4D!)
        let w2 = vec![[0.0, 0.0]; h];
        Self {
            h,
            w1,
            b1,
            w2,
            b2: [0.0, 0.0],
        }
    }

    pub fn n_params(&self) -> usize {
        7 * self.h + 2
    }

    #[inline]
    pub fn eval(&self, u: [f64; 4]) -> [f64; 2] {
        let mut o0 = self.b2[0];
        let mut o1 = self.b2[1];
        for i in 0..self.h {
            let z = self.b1[i]
                + self.w1[i][0] * u[0]
                + self.w1[i][1] * u[1]
                + self.w1[i][2] * u[2]
                + self.w1[i][3] * u[3];
            let a = z.tanh();
            o0 += self.w2[i][0] * a;
            o1 += self.w2[i][1] * a;
        }
        [o0, o1]
    }

    #[cfg(test)]
    fn pack(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        v.push(self.b2[0]);
        v.push(self.b2[1]);
        for i in 0..self.h {
            v.extend_from_slice(&self.w1[i]);
            v.push(self.b1[i]);
            v.extend_from_slice(&self.w2[i]);
        }
        v
    }

    #[cfg(test)]
    fn unpack(&mut self, v: &[f64]) {
        self.b2[0] = v[0];
        self.b2[1] = v[1];
        let mut idx = 2;
        for i in 0..self.h {
            self.w1[i] = [v[idx], v[idx + 1], v[idx + 2], v[idx + 3]];
            self.b1[i] = v[idx + 4];
            self.w2[i] = [v[idx + 5], v[idx + 6]];
            idx += 7;
        }
    }

    #[cfg(test)]
    pub fn accum_grad(&self, u: [f64; 4], dout: [f64; 2], grads: &mut [f64]) {
        grads[0] += dout[0];
        grads[1] += dout[1];
        for i in 0..self.h {
            let base = 2 + 7 * i;
            let z = self.b1[i]
                + self.w1[i][0] * u[0]
                + self.w1[i][1] * u[1]
                + self.w1[i][2] * u[2]
                + self.w1[i][3] * u[3];
            let a = z.tanh();
            grads[base + 5] += dout[0] * a;
            grads[base + 6] += dout[1] * a;
            let da = dout[0] * self.w2[i][0] + dout[1] * self.w2[i][1];
            let dz = da * (1.0 - a * a);
            grads[base + 4] += dz;
            grads[base] += dz * u[0];
            grads[base + 1] += dz * u[1];
            grads[base + 2] += dz * u[2];
            grads[base + 3] += dz * u[3];
        }
    }

    fn solve_optimal_out(&mut self, samples: &[([f64; 4], [f64; 2], f64)], lam: f64) {
        let nf = self.h + 1;
        let mut ata = vec![0.0; nf * nf];
        let mut atb0 = vec![0.0; nf];
        let mut atb1 = vec![0.0; nf];
        let mut feat = vec![0.0; nf];
        let n_f = samples.len().max(1) as f64;
        for &(u, tgt, w) in samples {
            let sw = w / n_f;
            feat[0] = 1.0;
            for i in 0..self.h {
                let z = self.b1[i]
                    + self.w1[i][0] * u[0]
                    + self.w1[i][1] * u[1]
                    + self.w1[i][2] * u[2]
                    + self.w1[i][3] * u[3];
                feat[i + 1] = z.tanh();
            }
            for r in 0..nf {
                let fr = feat[r] * sw;
                for c in r..nf {
                    ata[r * nf + c] += fr * feat[c];
                }
                atb0[r] += fr * tgt[0];
                atb1[r] += fr * tgt[1];
            }
        }
        for r in 0..nf {
            for c in 0..r {
                ata[r * nf + c] = ata[c * nf + r];
            }
        }
        let w0 = solve_ridge(&ata, &atb0, nf, lam);
        let w1 = solve_ridge(&ata, &atb1, nf, lam);
        self.b2 = [w0[0], w1[0]];
        for i in 0..self.h {
            self.w2[i] = [w0[i + 1], w1[i + 1]];
        }
    }

    pub fn fit(&mut self, samples: &[([f64; 4], [f64; 2], f64)], steps: usize, lr: f64, wd: f64) {
        if samples.is_empty() {
            return;
        }
        let lam = wd.max(1e-5);
        self.solve_optimal_out(samples, lam);
        let n_cycles = (steps / 25).clamp(6, 28);
        let inner_steps = 12usize;
        let np_h = self.h * 5;
        let mut p_h = vec![0.0; np_h];
        for i in 0..self.h {
            p_h[5 * i..5 * i + 4].copy_from_slice(&self.w1[i]);
            p_h[5 * i + 4] = self.b1[i];
        }
        let mut adam = Adam::new(np_h);
        let n_tot = samples.len();
        let batch_sz = if n_tot <= 320 { n_tot } else { 240 };
        let mut brng = Rng64::from_seed(n_tot as u64 ^ (self.h as u64), 0xCAFE);
        for cyc in 0..n_cycles {
            for ist in 0..inner_steps {
                for i in 0..self.h {
                    self.w1[i] = [p_h[5 * i], p_h[5 * i + 1], p_h[5 * i + 2], p_h[5 * i + 3]];
                    self.b1[i] = p_h[5 * i + 4];
                }
                let mut g_h = vec![0.0; np_h];
                let n_f = batch_sz as f64;
                for bi in 0..batch_sz {
                    let &(u, tgt, w) = if batch_sz == n_tot {
                        &samples[bi]
                    } else {
                        &samples[brng.usize_range(n_tot)]
                    };
                    let pred = self.eval(u);
                    let d0 = 2.0 * w * (pred[0] - tgt[0]) / n_f;
                    let d1 = 2.0 * w * (pred[1] - tgt[1]) / n_f;
                    for i in 0..self.h {
                        let z = self.b1[i]
                            + self.w1[i][0] * u[0]
                            + self.w1[i][1] * u[1]
                            + self.w1[i][2] * u[2]
                            + self.w1[i][3] * u[3];
                        let a = z.tanh();
                        let da = d0 * self.w2[i][0] + d1 * self.w2[i][1];
                        let dz = da * (1.0 - a * a);
                        g_h[5 * i + 4] += dz;
                        for j in 0..4 {
                            g_h[5 * i + j] += dz * u[j];
                        }
                    }
                }
                let prog = (cyc * inner_steps + ist) as f64 / ((n_cycles * inner_steps) as f64);
                let decay = 0.15 + 0.85 * 0.5 * (1.0 + (PI * prog).cos());
                adam.update(&mut p_h, &g_h, lr * decay, 1e-4);
            }
            for i in 0..self.h {
                self.w1[i] = [p_h[5 * i], p_h[5 * i + 1], p_h[5 * i + 2], p_h[5 * i + 3]];
                self.b1[i] = p_h[5 * i + 4];
            }
            self.solve_optimal_out(samples, lam);
        }
    }
}

// ─── Component 1b: 2-Layer Adaptive Spectral Interference Net (for Suite O3 & O4a) ─

/// 2-layer Spectral Wave Interference Network (`4 -> O3_H1 cos(K1 u + φ1) -> O3_H2 sin(K2 h + φ2) -> 2`)
/// or exact matched-capacity 2-layer tanh MLP (`4 -> O3_H1 tanh -> O3_H2 tanh -> 2`).
/// Exact 1:1 parameter count: `4*32 + 32 + 32*28 + 28 + 2*28 + 2 = 1,138` params/expert (`4,552` for 4 experts).
/// In `SpecMode::Adaptive`: `K1` starts uniform random (0 primes, 0 Riemann zeros) and adapts with Adam;
/// in `RiemannFixed`/`FourierFixed`/`PoissonFixed`: input wavevectors `K1` stay strictly frozen.
#[derive(Clone, Debug)]
pub struct SpectralWaveNet2 {
    pub mode: SpecMode,
    pub is_mlp: bool,
    pub w1: [[f64; 4]; O3_H1],
    pub b1: [f64; O3_H1],
    pub w2: [[f64; O3_H1]; O3_H2],
    pub b2: [f64; O3_H2],
    pub w3: [[f64; O3_H2]; 2],
    pub b3: [f64; 2],
}

impl SpectralWaveNet2 {
    pub fn new(mode: SpecMode, is_mlp: bool, zeros: Option<&[f64]>, rng: &mut Rng64) -> Self {
        if !is_mlp && mode == SpecMode::Adaptive {
            assert!(
                zeros.is_none(),
                "AdaptiveSpectral must NEVER receive Riemann zeros!"
            );
        }
        let mut w1 = [[0.0; 4]; O3_H1];
        let mut b1 = [0.0; O3_H1];
        let mut w2 = [[0.0; O3_H1]; O3_H2];
        let mut b2 = [0.0; O3_H2];
        let w3 = [[0.0; O3_H2]; 2];
        let b3 = [0.0; 2];

        let radii: Vec<f64> = if is_mlp || mode == SpecMode::FourierFixed {
            Vec::new()
        } else {
            match mode {
                SpecMode::Adaptive | SpecMode::PoissonFixed => {
                    let mut v: Vec<f64> = (0..O3_H1).map(|_| rng.uniform(0.3, 2.0)).collect();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v
                }
                SpecMode::RiemannFixed => {
                    let zs = zeros.expect("RiemannFixed needs exact zeros");
                    scaled_riemann(zs, O3_H1, 0.3, 2.0)
                }
                SpecMode::FourierFixed => unreachable!(),
            }
        };

        for i in 0..O3_H1 {
            if is_mlp {
                for c in 0..4 {
                    w1[i][c] = 0.5 * rng.normal();
                }
                b1[i] = 0.0;
            } else if mode == SpecMode::FourierFixed {
                // Regular 4D integer Fourier harmonic lattice
                let n0 = ((i % 3) as f64) - 1.0;
                let n1 = (((i / 3) % 3) as f64) - 1.0;
                let n2 = (((i / 9) % 3) as f64) - 1.0;
                let n3 = (((i / 27) % 2) as f64) - 0.5;
                let mut kv = [n0 * 0.9, n1 * 0.9, n2 * 0.9, n3 * 0.9];
                if kv.iter().all(|v| v.abs() < 1e-9) {
                    kv[0] = 1.8;
                }
                w1[i] = kv;
                b1[i] = rng.uniform(0.0, 2.0 * PI);
            } else {
                let d = [rng.normal(), rng.normal(), rng.normal(), rng.normal()];
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
                    .sqrt()
                    .max(1e-9);
                let r = radii[i];
                w1[i] = [r * d[0] / n, r * d[1] / n, r * d[2] / n, r * d[3] / n];
                b1[i] = rng.uniform(0.0, 2.0 * PI);
            }
        }
        let s2 = (6.0 / O3_H1 as f64).sqrt() * 0.5;
        let s2_mlp = (1.0 / O3_H1 as f64).sqrt();
        for i in 0..O3_H2 {
            for j in 0..O3_H1 {
                w2[i][j] = if is_mlp {
                    s2_mlp * rng.normal()
                } else {
                    rng.uniform(-s2, s2)
                };
            }
            b2[i] = if is_mlp { 0.0 } else { rng.uniform(-0.5, 0.5) };
        }
        Self {
            mode,
            is_mlp,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        }
    }

    pub fn n_params(&self) -> usize {
        O3_H1 * 4 + O3_H1 + O3_H2 * O3_H1 + O3_H2 + 2 * O3_H2 + 2
    }

    #[inline]
    pub fn eval(&self, u: [f64; 4]) -> [f64; 2] {
        let mut h1 = [0.0; O3_H1];
        for i in 0..O3_H1 {
            let z = self.b1[i]
                + self.w1[i][0] * u[0]
                + self.w1[i][1] * u[1]
                + self.w1[i][2] * u[2]
                + self.w1[i][3] * u[3];
            h1[i] = if self.is_mlp { z.tanh() } else { z.cos() };
        }
        let mut h2 = [0.0; O3_H2];
        for i in 0..O3_H2 {
            let mut z = self.b2[i];
            for j in 0..O3_H1 {
                z += self.w2[i][j] * h1[j];
            }
            h2[i] = if self.is_mlp { z.tanh() } else { z.sin() };
        }
        let mut o = self.b3;
        for d in 0..2 {
            for i in 0..O3_H2 {
                o[d] += self.w3[d][i] * h2[i];
            }
        }
        o
    }

    #[cfg(test)]
    fn pack(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for i in 0..O3_H1 {
            v.extend_from_slice(&self.w1[i]);
        }
        v.extend_from_slice(&self.b1);
        for i in 0..O3_H2 {
            v.extend_from_slice(&self.w2[i]);
        }
        v.extend_from_slice(&self.b2);
        for d in 0..2 {
            v.extend_from_slice(&self.w3[d]);
        }
        v.extend_from_slice(&self.b3);
        v
    }

    #[cfg(test)]
    fn unpack(&mut self, v: &[f64]) {
        let mut idx = 0;
        for i in 0..O3_H1 {
            self.w1[i] = [v[idx], v[idx + 1], v[idx + 2], v[idx + 3]];
            idx += 4;
        }
        for i in 0..O3_H1 {
            self.b1[i] = v[idx];
            idx += 1;
        }
        for i in 0..O3_H2 {
            for j in 0..O3_H1 {
                self.w2[i][j] = v[idx];
                idx += 1;
            }
        }
        for i in 0..O3_H2 {
            self.b2[i] = v[idx];
            idx += 1;
        }
        for d in 0..2 {
            for i in 0..O3_H2 {
                self.w3[d][i] = v[idx];
                idx += 1;
            }
        }
        self.b3 = [v[idx], v[idx + 1]];
    }

    /// Accumulates exact analytical backprop gradients for one `(u, dout)` pair.
    #[cfg(test)]
    pub fn accum_grad(&self, u: [f64; 4], dout: [f64; 2], g: &mut [f64]) {
        let adapt_k1 = self.is_mlp || self.mode.adapts_freq();
        let np = self.n_params();
        let o_b3 = np - 2;
        let o_w3 = o_b3 - 2 * O3_H2;
        let o_b2 = o_w3 - O3_H2;
        let o_w2 = o_b2 - O3_H2 * O3_H1;
        let o_b1 = o_w2 - O3_H1;

        let mut z1 = [0.0; O3_H1];
        let mut h1 = [0.0; O3_H1];
        for i in 0..O3_H1 {
            let z = self.b1[i]
                + self.w1[i][0] * u[0]
                + self.w1[i][1] * u[1]
                + self.w1[i][2] * u[2]
                + self.w1[i][3] * u[3];
            z1[i] = z;
            h1[i] = if self.is_mlp { z.tanh() } else { z.cos() };
        }
        let mut z2 = [0.0; O3_H2];
        let mut h2 = [0.0; O3_H2];
        for i in 0..O3_H2 {
            let mut z = self.b2[i];
            for j in 0..O3_H1 {
                z += self.w2[i][j] * h1[j];
            }
            z2[i] = z;
            h2[i] = if self.is_mlp { z.tanh() } else { z.sin() };
        }

        g[o_b3] += dout[0];
        g[o_b3 + 1] += dout[1];
        let mut dh2 = [0.0; O3_H2];
        for i in 0..O3_H2 {
            g[o_w3 + i] += dout[0] * h2[i];
            g[o_w3 + O3_H2 + i] += dout[1] * h2[i];
            dh2[i] = dout[0] * self.w3[0][i] + dout[1] * self.w3[1][i];
        }
        let mut dh1 = [0.0; O3_H1];
        for i in 0..O3_H2 {
            let dz = dh2[i]
                * if self.is_mlp {
                    1.0 - h2[i] * h2[i]
                } else {
                    z2[i].cos()
                };
            g[o_b2 + i] += dz;
            let row = o_w2 + i * O3_H1;
            for j in 0..O3_H1 {
                g[row + j] += dz * h1[j];
                dh1[j] += dz * self.w2[i][j];
            }
        }
        for i in 0..O3_H1 {
            let dz = dh1[i]
                * if self.is_mlp {
                    1.0 - h1[i] * h1[i]
                } else {
                    -z1[i].sin()
                };
            g[o_b1 + i] += dz;
            if adapt_k1 {
                for c in 0..4 {
                    g[4 * i + c] += dz * u[c];
                }
            }
        }
    }

    pub fn fit(
        &mut self,
        samples: &[([f64; 4], [f64; 2], f64)],
        steps: usize,
        bsz: usize,
        rng: &mut Rng64,
    ) {
        if samples.is_empty() {
            return;
        }
        let adapt_k1 = self.is_mlp || self.mode.adapts_freq();
        let np = self.n_params();
        let o_b3 = np - 2;
        let o_w3 = o_b3 - 2 * O3_H2;
        let o_b2 = o_w3 - O3_H2;
        let o_w2 = o_b2 - O3_H2 * O3_H1;
        let o_b1 = o_w2 - O3_H1;
        let mut m = vec![0.0; np];
        let mut v = vec![0.0; np];
        let n_tot = samples.len();
        let batch = bsz.min(n_tot);
        let b1_a: f64 = 0.9;
        let b2_a: f64 = 0.999;

        for st in 1..=steps {
            let mut g = vec![0.0; np];
            let bf = batch as f64;
            for _ in 0..batch {
                let &(u, tgt, w) = &samples[rng.usize_range(n_tot)];
                let mut z1 = [0.0; O3_H1];
                let mut h1 = [0.0; O3_H1];
                for i in 0..O3_H1 {
                    let z = self.b1[i]
                        + self.w1[i][0] * u[0]
                        + self.w1[i][1] * u[1]
                        + self.w1[i][2] * u[2]
                        + self.w1[i][3] * u[3];
                    z1[i] = z;
                    h1[i] = if self.is_mlp { z.tanh() } else { z.cos() };
                }
                let mut z2 = [0.0; O3_H2];
                let mut h2 = [0.0; O3_H2];
                for i in 0..O3_H2 {
                    let mut z = self.b2[i];
                    for j in 0..O3_H1 {
                        z += self.w2[i][j] * h1[j];
                    }
                    z2[i] = z;
                    h2[i] = if self.is_mlp { z.tanh() } else { z.sin() };
                }
                let mut o = self.b3;
                for d in 0..2 {
                    for i in 0..O3_H2 {
                        o[d] += self.w3[d][i] * h2[i];
                    }
                }
                let d0 = 2.0 * w * (o[0] - tgt[0]) / bf;
                let d1 = 2.0 * w * (o[1] - tgt[1]) / bf;
                g[o_b3] += d0;
                g[o_b3 + 1] += d1;
                let mut dh2 = [0.0; O3_H2];
                for i in 0..O3_H2 {
                    g[o_w3 + i] += d0 * h2[i];
                    g[o_w3 + O3_H2 + i] += d1 * h2[i];
                    dh2[i] = d0 * self.w3[0][i] + d1 * self.w3[1][i];
                }
                let mut dh1 = [0.0; O3_H1];
                for i in 0..O3_H2 {
                    let dz = dh2[i]
                        * if self.is_mlp {
                            1.0 - h2[i] * h2[i]
                        } else {
                            z2[i].cos()
                        };
                    g[o_b2 + i] += dz;
                    let row = o_w2 + i * O3_H1;
                    for j in 0..O3_H1 {
                        g[row + j] += dz * h1[j];
                        dh1[j] += dz * self.w2[i][j];
                    }
                }
                for i in 0..O3_H1 {
                    let dz = dh1[i]
                        * if self.is_mlp {
                            1.0 - h1[i] * h1[i]
                        } else {
                            -z1[i].sin()
                        };
                    g[o_b1 + i] += dz;
                    if adapt_k1 {
                        for c in 0..4 {
                            g[4 * i + c] += dz * u[c];
                        }
                    }
                }
            }
            let prog = st as f64 / steps as f64;
            let lr = 0.010 * (0.02 + 0.98 * 0.5 * (1.0 + (PI * prog).cos()));
            let bc1 = 1.0 - b1_a.powi(st as i32);
            let bc2 = 1.0 - b2_a.powi(st as i32);
            let mut idx = 0usize;
            for i in 0..O3_H1 {
                for c in 0..4 {
                    if adapt_k1 {
                        let gi = g[idx] + 1e-6 * self.w1[i][c];
                        m[idx] = b1_a * m[idx] + (1.0 - b1_a) * gi;
                        v[idx] = b2_a * v[idx] + (1.0 - b2_a) * gi * gi;
                        self.w1[i][c] -= lr * (m[idx] / bc1) / ((v[idx] / bc2).sqrt() + 1e-8);
                    }
                    idx += 1;
                }
            }
            for i in 0..O3_H1 {
                let gi = g[idx];
                m[idx] = b1_a * m[idx] + (1.0 - b1_a) * gi;
                v[idx] = b2_a * v[idx] + (1.0 - b2_a) * gi * gi;
                self.b1[i] -= lr * (m[idx] / bc1) / ((v[idx] / bc2).sqrt() + 1e-8);
                idx += 1;
            }
            for i in 0..O3_H2 {
                for j in 0..O3_H1 {
                    let gi = g[idx] + 1e-6 * self.w2[i][j];
                    m[idx] = b1_a * m[idx] + (1.0 - b1_a) * gi;
                    v[idx] = b2_a * v[idx] + (1.0 - b2_a) * gi * gi;
                    self.w2[i][j] -= lr * (m[idx] / bc1) / ((v[idx] / bc2).sqrt() + 1e-8);
                    idx += 1;
                }
            }
            for i in 0..O3_H2 {
                let gi = g[idx];
                m[idx] = b1_a * m[idx] + (1.0 - b1_a) * gi;
                v[idx] = b2_a * v[idx] + (1.0 - b2_a) * gi * gi;
                self.b2[i] -= lr * (m[idx] / bc1) / ((v[idx] / bc2).sqrt() + 1e-8);
                idx += 1;
            }
            for d in 0..2 {
                for i in 0..O3_H2 {
                    let gi = g[idx] + 1e-6 * self.w3[d][i];
                    m[idx] = b1_a * m[idx] + (1.0 - b1_a) * gi;
                    v[idx] = b2_a * v[idx] + (1.0 - b2_a) * gi * gi;
                    self.w3[d][i] -= lr * (m[idx] / bc1) / ((v[idx] / bc2).sqrt() + 1e-8);
                    idx += 1;
                }
            }
            for d in 0..2 {
                let gi = g[idx];
                m[idx] = b1_a * m[idx] + (1.0 - b1_a) * gi;
                v[idx] = b2_a * v[idx] + (1.0 - b2_a) * gi * gi;
                self.b3[d] -= lr * (m[idx] / bc1) / ((v[idx] / bc2).sqrt() + 1e-8);
                idx += 1;
            }
        }
    }
}

// ─── Predictor Wrappers Around Frozen ÓRBITA for Official §16 Evaluation ────

pub struct OrbitaWithWave {
    pub base: Orbita,
    pub fields: Vec<WaveField4D>, // 8 families
    pub label: String,
}

impl OrbitaWithWave {
    pub fn step(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        let mut y = self.base.step(x, c);
        let fam = (0..N_FAM)
            .max_by(|&a, &b| c[a].total_cmp(&c[b]))
            .unwrap_or(0);
        let p = [c[N_FAM], c[N_FAM + 1]];
        let wf = &self.fields[fam];
        for i in 0..x.len() / 2 {
            let dq = wf.eval([x[2 * i], x[2 * i + 1], p[0], p[1]]);
            y[2 * i] += dq[0];
            y[2 * i + 1] += dq[1];
        }
        y
    }
}

impl Predictor for OrbitaWithWave {
    fn name(&self) -> &str {
        &self.label
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        self.step(x, c)
    }
    fn params(&self) -> usize {
        self.base.n_params() + self.fields.iter().map(|f| f.n_params()).sum::<usize>()
    }
}

pub struct OrbitaWithMlp {
    pub base: Orbita,
    pub mlps: Vec<MlpRes4D>,
}

impl OrbitaWithMlp {
    pub fn step(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        let mut y = self.base.step(x, c);
        let fam = (0..N_FAM)
            .max_by(|&a, &b| c[a].total_cmp(&c[b]))
            .unwrap_or(0);
        let p = [c[N_FAM], c[N_FAM + 1]];
        let ml = &self.mlps[fam];
        for i in 0..x.len() / 2 {
            let dq = ml.eval([x[2 * i], x[2 * i + 1], p[0], p[1]]);
            y[2 * i] += dq[0];
            y[2 * i + 1] += dq[1];
        }
        y
    }
}

impl Predictor for OrbitaWithMlp {
    fn name(&self) -> &str {
        "ORBITA_MLP_RES"
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        self.step(x, c)
    }
    fn params(&self) -> usize {
        self.base.n_params() + self.mlps.iter().map(|m| m.n_params()).sum::<usize>()
    }
}

// ─── Component 2: 3D Adaptive Spectral Collapse Cavity (for Suites O2, O3, O4b) ─

/// Extracts a 16D opaque geometric dynamical signature from an observed transition x0 -> y1
/// (ZERO family labels, ZERO primes, ZERO Riemann zeros!).
pub fn dynamical_signature_16d(x0: &[f64], y1: &[f64]) -> [f64; DIM_SIG] {
    let np = x0.len() / 2;
    let nf = np.max(1) as f64;
    let (mut mx, mut my, mut mdx, mut mdy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..np {
        mx += x0[2 * i];
        my += x0[2 * i + 1];
        mdx += y1[2 * i] - x0[2 * i];
        mdy += y1[2 * i + 1] - x0[2 * i + 1];
    }
    mx /= nf;
    my /= nf;
    mdx /= nf;
    mdy /= nf;

    // Fit affine map y1 ≈ A x0 + t by 3x3 normal equations
    let mut ata = [0.0; 9];
    let mut atbx = [0.0; 3];
    let mut atby = [0.0; 3];
    for i in 0..np {
        let u = [x0[2 * i], x0[2 * i + 1], 1.0];
        for r in 0..3 {
            for c in 0..3 {
                ata[r * 3 + c] += u[r] * u[c];
            }
            atbx[r] += u[r] * y1[2 * i];
            atby[r] += u[r] * y1[2 * i + 1];
        }
    }
    let wx = solve_ridge(&ata, &atbx, 3, 1e-6);
    let wy = solve_ridge(&ata, &atby, 3, 1e-6);
    let (jxx, jxy, tx) = (wx[0], wx[1], wx[2]);
    let (jyx, jyy, ty) = (wy[0], wy[1], wy[2]);
    let curl = jyx - jxy;
    let div = (jxx - 1.0) + (jyy - 1.0);
    let det_m1 = (jxx * jyy - jxy * jyx) - 1.0;
    let shear_a = jxx - jyy;
    let shear_b = jxy + jyx;

    let mut nl_err2 = 0.0;
    let mut rad_twist = 0.0;
    let mut trefoil_h = 0.0;
    let mut wavy_h = 0.0;
    let mut tanh_h = 0.0;
    let mut dr_mean = 0.0;
    for i in 0..np {
        let (x, y) = (x0[2 * i], x0[2 * i + 1]);
        let (nx, ny) = (y1[2 * i], y1[2 * i + 1]);
        let px = jxx * x + jxy * y + tx;
        let py = jyx * x + jyy * y + ty;
        let rx = nx - px;
        let ry = ny - py;
        nl_err2 += rx * rx + ry * ry;
        let r0 = (x * x + y * y).sqrt().max(1e-6);
        let r1 = (nx * nx + ny * ny).sqrt();
        let dr = r1 - r0;
        dr_mean += dr;
        let th0 = y.atan2(x);
        let th1 = ny.atan2(nx);
        let mut dth = th1 - th0;
        while dth > PI {
            dth -= 2.0 * PI;
        }
        while dth < -PI {
            dth += 2.0 * PI;
        }
        rad_twist += (r0 - 0.7) * dth;
        trefoil_h += (dr / r0) * (3.0 * th0).sin();
        wavy_h += rx * (2.0 * y).sin() + ry * (2.0 * x).sin();
        tanh_h += rx * (2.0 * y).tanh() + ry * (2.0 * x).tanh();
    }
    let nl_rms = (nl_err2 / nf).sqrt();
    let mut raw = [
        tx,
        ty,
        mdx,
        mdy,
        jxx - 1.0,
        jxy,
        jyx,
        jyy - 1.0,
        curl,
        div,
        det_m1,
        shear_a + 0.5 * shear_b,
        nl_rms * 4.0,
        (rad_twist / nf) * 3.0,
        (trefoil_h / nf) * 4.0 + (wavy_h / nf) * 4.0,
        (tanh_h / nf) * 4.0 + (dr_mean / nf) + 0.05 * (mx + my),
    ];
    let nrm = raw.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6);
    for v in &mut raw {
        *v /= nrm;
    }
    raw
}

/// 1D wave channel for the 3D Spectral Collapse Cavity.
#[derive(Clone, Debug)]
pub struct CavWave1D {
    pub m: usize,
    pub mode: SpecMode,
    pub w: Vec<f64>,
    pub a: Vec<f64>,
    pub phi: Vec<f64>,
}

impl CavWave1D {
    pub fn new(m: usize, mode: SpecMode, zeros: Option<&[f64]>, rng: &mut Rng64) -> Self {
        let w = match mode {
            SpecMode::Adaptive | SpecMode::PoissonFixed => {
                let mut v: Vec<f64> = (0..m).map(|_| rng.uniform(1.2, 6.2)).collect();
                v.sort_by(|a, b| a.total_cmp(b));
                v
            }
            SpecMode::RiemannFixed => scaled_riemann(zeros.unwrap(), m, 1.2, 6.2),
            SpecMode::FourierFixed => (1..=m).map(|i| 2.0 * PI * (i as f64) / 4.8).collect(),
        };
        let a = (0..m).map(|_| 0.06 * rng.normal()).collect();
        let phi = (0..m).map(|_| rng.uniform(0.0, 2.0 * PI)).collect();
        Self { m, mode, w, a, phi }
    }

    #[inline]
    pub fn eval(&self, u: f64) -> f64 {
        let mut s = -0.25;
        for i in 0..self.m {
            s += self.a[i] * (self.w[i] * u + self.phi[i]).cos();
        }
        s
    }

    pub fn accum_grad(&self, u: f64, dout: f64, gw: &mut [f64], ga: &mut [f64], gp: &mut [f64]) {
        let adapt = self.mode.adapts_freq();
        for i in 0..self.m {
            let ang = self.w[i] * u + self.phi[i];
            ga[i] += dout * ang.cos();
            let da = -dout * self.a[i] * ang.sin();
            gp[i] += da;
            if adapt {
                gw[i] += da * u;
            }
        }
    }
}

/// 3D coherence wave for the Spectral Collapse Cavity.
#[derive(Clone, Debug)]
pub struct CavWave3D {
    pub m: usize,
    #[allow(dead_code)]
    pub mode: SpecMode,
    pub k: Vec<[f64; 3]>,
    pub a: Vec<f64>,
    pub phi: Vec<f64>,
}

impl CavWave3D {
    pub fn new(m: usize, mode: SpecMode, zeros: Option<&[f64]>, rng: &mut Rng64) -> Self {
        let mut k = Vec::with_capacity(m);
        if mode == SpecMode::FourierFixed {
            // Genuine regular 3D Fourier integer harmonic lattice (nx, ny, nz) * 2π / L
            let w0 = 2.0 * PI / 4.2;
            let mut lat: Vec<(f64, [f64; 3])> = Vec::new();
            for nx in -2i32..=2 {
                for ny in -2i32..=2 {
                    for nz in -2i32..=2 {
                        if nx == 0 && ny == 0 && nz == 0 {
                            continue;
                        }
                        // Keep one half-space to avoid trivial +k / -k duplicates
                        if nx < 0 || (nx == 0 && ny < 0) || (nx == 0 && ny == 0 && nz < 0) {
                            continue;
                        }
                        let kv = [nx as f64 * w0, ny as f64 * w0, nz as f64 * w0];
                        let n = (kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2]).sqrt();
                        lat.push((n, kv));
                    }
                }
            }
            lat.sort_by(|a, b| a.0.total_cmp(&b.0));
            for i in 0..m {
                k.push(lat[i % lat.len()].1);
            }
        } else {
            let norms: Vec<f64> = match mode {
                SpecMode::Adaptive | SpecMode::PoissonFixed => {
                    let mut v: Vec<f64> = (0..m).map(|_| rng.uniform(1.2, 4.8)).collect();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v
                }
                SpecMode::RiemannFixed => scaled_riemann(zeros.unwrap(), m, 1.2, 4.8),
                SpecMode::FourierFixed => unreachable!(),
            };
            for &r in &norms {
                let mut d = [rng.normal(), rng.normal(), rng.normal()];
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
                for v in &mut d {
                    *v = *v * r / n;
                }
                k.push(d);
            }
        }
        let a = (0..m).map(|_| 0.05 * rng.normal()).collect();
        let phi = (0..m).map(|_| rng.uniform(0.0, 2.0 * PI)).collect();
        Self { m, mode, k, a, phi }
    }

    #[inline]
    pub fn eval(&self, u: [f64; 3]) -> f64 {
        let mut s = DC_3D;
        for i in 0..self.m {
            let ki = &self.k[i];
            s += self.a[i] * (ki[0] * u[0] + ki[1] * u[1] + ki[2] * u[2] + self.phi[i]).cos();
        }
        s
    }

    #[inline]
    pub fn eval_grad_u(&self, u: [f64; 3]) -> (f64, [f64; 3]) {
        let mut s = DC_3D;
        let mut g = [0.0; 3];
        for i in 0..self.m {
            let ki = &self.k[i];
            let ang = ki[0] * u[0] + ki[1] * u[1] + ki[2] * u[2] + self.phi[i];
            s += self.a[i] * ang.cos();
            let d = -self.a[i] * ang.sin();
            g[0] += d * ki[0];
            g[1] += d * ki[1];
            g[2] += d * ki[2];
        }
        (s, g)
    }
}

/// Complete 3D Adaptive Spectral Collapse Cavity (`W_enc: 3x16` + 3x1D + 3D + Past x Future Collapse).
#[derive(Clone, Debug)]
pub struct SpectralCavity3D {
    pub mode: SpecMode,
    pub w_enc: [[f64; DIM_SIG]; DIM_PHASE],
    pub w1d: [CavWave1D; DIM_PHASE],
    pub w3d: CavWave3D,
    pub states: Vec<[f64; DIM_PHASE]>,
    pub init_r: f64,
}

impl SpectralCavity3D {
    pub fn new(mode: SpecMode, zeros: Option<&[f64]>, rng: &mut Rng64) -> Self {
        if mode == SpecMode::Adaptive {
            assert!(
                zeros.is_none(),
                "AdaptiveSpectral cavity must NEVER receive Riemann zeros!"
            );
        }
        let mut w_enc = [[0.0; DIM_SIG]; DIM_PHASE];
        let sc = (1.0 / DIM_SIG as f64).sqrt();
        for d in 0..DIM_PHASE {
            for j in 0..DIM_SIG {
                w_enc[d][j] = sc * rng.normal();
            }
        }
        let w1d = [
            CavWave1D::new(M1D_O2, mode, zeros, rng),
            CavWave1D::new(M1D_O2, mode, zeros, rng),
            CavWave1D::new(M1D_O2, mode, zeros, rng),
        ];
        let w3d = CavWave3D::new(M3D_O2, mode, zeros, rng);
        let init_r = mean_spacing_r(
            &w3d.k
                .iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                .collect::<Vec<_>>(),
        );
        Self {
            mode,
            w_enc,
            w1d,
            w3d,
            states: Vec::new(),
            init_r,
        }
    }

    pub fn n_params(&self) -> usize {
        DIM_PHASE * DIM_SIG + 3 * (3 * M1D_O2) + 5 * M3D_O2
    }

    #[inline]
    pub fn encode(&self, x: &[f64; DIM_SIG]) -> [f64; DIM_PHASE] {
        let mut z = [0.0; DIM_PHASE];
        for d in 0..DIM_PHASE {
            let mut s = 0.0;
            for j in 0..DIM_SIG {
                s += self.w_enc[d][j] * x[j];
            }
            z[d] = s;
        }
        z
    }

    pub fn learned_r(&self) -> f64 {
        let norms: Vec<f64> = self
            .w3d
            .k
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .collect();
        mean_spacing_r(&norms)
    }

    /// Trains W_enc for collision-free Sidon state separation on known atomic prototypes + train_pairs,
    /// then trains 1D + 3D wave fields by experience (0 pair table, 0 Riemann zeros in Adaptive!).
    pub fn fit(
        &mut self,
        prototypes: &[[f64; DIM_SIG]],
        train_pairs: &[(usize, usize)],
        rng: &mut Rng64,
    ) {
        let k_cnt = prototypes.len();
        // Stage 1: Autoorganization of W_enc (Sidon collision-free geometry)
        let mut wflat = vec![0.0; DIM_PHASE * DIM_SIG];
        for d in 0..DIM_PHASE {
            for j in 0..DIM_SIG {
                wflat[d * DIM_SIG + j] = self.w_enc[d][j];
            }
        }
        let mut adam_w = Adam::new(wflat.len());
        let s1_steps = 280usize;
        for step in 0..s1_steps {
            for d in 0..DIM_PHASE {
                for j in 0..DIM_SIG {
                    self.w_enc[d][j] = wflat[d * DIM_SIG + j];
                }
            }
            let zs: Vec<[f64; 3]> = prototypes.iter().map(|x| self.encode(x)).collect();
            let mut gz = vec![[0.0; 3]; k_cnt];

            // Atomic state repulsion
            for i in 0..k_cnt {
                for j in (i + 1)..k_cnt {
                    let dx = [
                        zs[i][0] - zs[j][0],
                        zs[i][1] - zs[j][1],
                        zs[i][2] - zs[j][2],
                    ];
                    let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                    let rep = -0.75 * (-d2 / 0.70).exp();
                    for d in 0..3 {
                        gz[i][d] += rep * dx[d];
                        gz[j][d] -= rep * dx[d];
                    }
                }
                // Cavity containment [-1.45, 1.45]^3
                for d in 0..3 {
                    let v = zs[i][d];
                    if v.abs() > 1.38 {
                        let ex = v.abs() - 1.38;
                        gz[i][d] += 2.5 * ex * v.signum();
                    }
                }
            }
            // Sidon pair-sum repulsion on train_pairs
            let np = train_pairs.len();
            for _ in 0..120 {
                let p1 = train_pairs[rng.usize_range(np)];
                let p2 = train_pairs[rng.usize_range(np)];
                if p1 == p2 {
                    continue;
                }
                let dx = [
                    (zs[p1.0][0] + zs[p1.1][0]) - (zs[p2.0][0] + zs[p2.1][0]),
                    (zs[p1.0][1] + zs[p1.1][1]) - (zs[p2.0][1] + zs[p2.1][1]),
                    (zs[p1.0][2] + zs[p1.1][2]) - (zs[p2.0][2] + zs[p2.1][2]),
                ];
                let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                let rep = -0.35 * (-d2 / 0.28).exp();
                for d in 0..3 {
                    gz[p1.0][d] += rep * dx[d];
                    gz[p1.1][d] += rep * dx[d];
                    gz[p2.0][d] -= rep * dx[d];
                    gz[p2.1][d] -= rep * dx[d];
                }
            }
            let mut gw = vec![0.0; wflat.len()];
            for i in 0..k_cnt {
                for d in 0..3 {
                    for j in 0..DIM_SIG {
                        gw[d * DIM_SIG + j] += gz[i][d] * prototypes[i][j] / (k_cnt as f64);
                    }
                }
            }
            let lr = 0.035 * (0.2 + 0.8 * (1.0 - step as f64 / s1_steps as f64));
            adam_w.update(&mut wflat, &gw, lr, 0.008);
        }
        for d in 0..DIM_PHASE {
            for j in 0..DIM_SIG {
                self.w_enc[d][j] = wflat[d * DIM_SIG + j];
            }
        }
        self.states = prototypes.iter().map(|x| self.encode(x)).collect();

        // Stage 2: Fit 1D waves (for fast axis peak proposals)
        for ax in 0..DIM_PHASE {
            let np1 = 3 * M1D_O2;
            let mut p1 = vec![0.0; np1];
            for i in 0..M1D_O2 {
                p1[i] = self.w1d[ax].w[i];
                p1[M1D_O2 + i] = self.w1d[ax].a[i];
                p1[2 * M1D_O2 + i] = self.w1d[ax].phi[i];
            }
            let mut adam1 = Adam::new(np1);
            for step in 0..160 {
                for i in 0..M1D_O2 {
                    self.w1d[ax].w[i] = p1[i];
                    self.w1d[ax].a[i] = p1[M1D_O2 + i];
                    self.w1d[ax].phi[i] = p1[2 * M1D_O2 + i];
                }
                let mut gw = vec![0.0; M1D_O2];
                let mut ga = vec![0.0; M1D_O2];
                let mut gp = vec![0.0; M1D_O2];
                for s in &self.states {
                    let v = self.w1d[ax].eval(s[ax]);
                    if v < 1.6 {
                        self.w1d[ax].accum_grad(
                            s[ax],
                            -2.2 * (1.6 - v) / (k_cnt as f64),
                            &mut gw,
                            &mut ga,
                            &mut gp,
                        );
                    }
                }
                for _ in 0..36 {
                    let u = rng.uniform(-CAVITY_LIM, CAVITY_LIM);
                    let near = self.states.iter().any(|s| (s[ax] - u).abs() < 0.09);
                    if near {
                        continue;
                    }
                    let v = self.w1d[ax].eval(u);
                    if v > -0.4 {
                        self.w1d[ax].accum_grad(u, 2.4 * (v + 0.4) / 36.0, &mut gw, &mut ga, &mut gp);
                    }
                }
                let mut g = vec![0.0; np1];
                for i in 0..M1D_O2 {
                    g[i] = gw[i];
                    g[M1D_O2 + i] = ga[i];
                    g[2 * M1D_O2 + i] = gp[i];
                }
                let lr = 0.03 * (0.2 + 0.8 * (1.0 - step as f64 / 160.0));
                let w_before: Vec<f64> = p1[..M1D_O2].to_vec();
                adam1.update(&mut p1, &g, lr, 0.001);
                if self.mode.adapts_freq() {
                    for i in 0..M1D_O2 {
                        p1[i] = p1[i].clamp(0.8, 7.5);
                    }
                } else {
                    p1[..M1D_O2].copy_from_slice(&w_before);
                }
            }
            for i in 0..M1D_O2 {
                self.w1d[ax].w[i] = p1[i];
                self.w1d[ax].a[i] = p1[M1D_O2 + i];
                self.w1d[ax].phi[i] = p1[2 * M1D_O2 + i];
            }
        }

        // Stage 3: Fit 3D Coherence Wave with residual distractors (z_a + z_b - z_c)
        let np3 = 5 * M3D_O2;
        let mut p3 = vec![0.0; np3];
        for i in 0..M3D_O2 {
            p3[5 * i] = self.w3d.k[i][0];
            p3[5 * i + 1] = self.w3d.k[i][1];
            p3[5 * i + 2] = self.w3d.k[i][2];
            p3[5 * i + 3] = self.w3d.a[i];
            p3[5 * i + 4] = self.w3d.phi[i];
        }
        let mut adam3 = Adam::new(np3);
        let adapt = self.mode.adapts_freq();
        let n_steps = 340;
        for step in 0..n_steps {
            for i in 0..M3D_O2 {
                self.w3d.k[i] = [p3[5 * i], p3[5 * i + 1], p3[5 * i + 2]];
                self.w3d.a[i] = p3[5 * i + 3];
                self.w3d.phi[i] = p3[5 * i + 4];
            }
            let mut g3 = vec![0.0; np3];
            let mut accum = |u: [f64; 3], dout: f64| {
                for i in 0..M3D_O2 {
                    let ki = &self.w3d.k[i];
                    let ang = ki[0] * u[0] + ki[1] * u[1] + ki[2] * u[2] + self.w3d.phi[i];
                    g3[5 * i + 3] += dout * ang.cos();
                    let da = -dout * self.w3d.a[i] * ang.sin();
                    g3[5 * i + 4] += da;
                    if adapt {
                        g3[5 * i] += da * u[0];
                        g3[5 * i + 1] += da * u[1];
                        g3[5 * i + 2] += da * u[2];
                    }
                }
            };
            for s in &self.states {
                let v0 = self.w3d.eval(*s);
                if v0 < 1.9 {
                    accum(*s, -2.5 * (1.9 - v0) / (k_cnt as f64));
                }
                let sj = [
                    s[0] + 0.04 * rng.normal(),
                    s[1] + 0.04 * rng.normal(),
                    s[2] + 0.04 * rng.normal(),
                ];
                let vj = self.w3d.eval(sj);
                if vj < 1.5 {
                    accum(sj, -1.1 * (1.5 - vj) / (k_cnt as f64));
                }
            }
            let n_neg = 96usize;
            for n in 0..n_neg {
                let u = if n < 52 {
                    [
                        rng.uniform(-CAVITY_LIM, CAVITY_LIM),
                        rng.uniform(-CAVITY_LIM, CAVITY_LIM),
                        rng.uniform(-CAVITY_LIM, CAVITY_LIM),
                    ]
                } else {
                    let (a, b) = train_pairs[rng.usize_range(train_pairs.len())];
                    let mut c = rng.usize_range(k_cnt);
                    if c == a || c == b {
                        c = (c + 1) % k_cnt;
                    }
                    let u = [
                        self.states[a][0] + self.states[b][0] - self.states[c][0],
                        self.states[a][1] + self.states[b][1] - self.states[c][1],
                        self.states[a][2] + self.states[b][2] - self.states[c][2],
                    ];
                    if u[0].abs() > CAVITY_LIM || u[1].abs() > CAVITY_LIM || u[2].abs() > CAVITY_LIM
                    {
                        continue;
                    }
                    u
                };
                let near = self.states.iter().any(|s| {
                    (u[0] - s[0]).powi(2) + (u[1] - s[1]).powi(2) + (u[2] - s[2]).powi(2) < 0.14
                });
                if near {
                    continue;
                }
                let v = self.w3d.eval(u);
                if v > -0.55 {
                    accum(u, 3.4 * (v + 0.55) / (n_neg as f64));
                }
            }
            if adapt {
                // Soft bandwidth + spacing repulsion (discovers emergent spectral repulsion <r>)
                let mut norms: Vec<(f64, usize)> = (0..M3D_O2)
                    .map(|i| {
                        let k = &self.w3d.k[i];
                        let r2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                        if r2 > 24.0 {
                            let c = 0.08 * (r2 - 24.0);
                            g3[5 * i] += c * k[0];
                            g3[5 * i + 1] += c * k[1];
                            g3[5 * i + 2] += c * k[2];
                        }
                        if r2 < 1.2 {
                            let c = 0.10 * (1.2 - r2);
                            g3[5 * i] -= c * k[0];
                            g3[5 * i + 1] -= c * k[1];
                            g3[5 * i + 2] -= c * k[2];
                        }
                        (r2.sqrt().max(1e-6), i)
                    })
                    .collect();
                norms.sort_by(|a, b| a.0.total_cmp(&b.0));
                let tgap = 3.6 / (M3D_O2 as f64);
                for idx in 0..M3D_O2 - 1 {
                    let (r1, i1) = norms[idx];
                    let (r2, i2) = norms[idx + 1];
                    if r2 - r1 < tgap * 0.96 {
                        let push = 0.28 * (tgap * 0.96 - (r2 - r1));
                        for d in 0..3 {
                            g3[5 * i1 + d] += push * self.w3d.k[i1][d] / r1;
                            g3[5 * i2 + d] -= push * self.w3d.k[i2][d] / r2;
                        }
                    }
                }
            }
            let prog = step as f64 / (n_steps as f64);
            let lr = 0.032 * (0.2 + 0.8 * 0.5 * (1.0 + (PI * prog).cos()));
            let p3_before = p3.clone();
            adam3.update(&mut p3, &g3, lr, 0.0015);
            if !adapt {
                // Strictly preserve frozen wavevectors k (no weight-decay drift!)
                for i in 0..M3D_O2 {
                    p3[5 * i] = p3_before[5 * i];
                    p3[5 * i + 1] = p3_before[5 * i + 1];
                    p3[5 * i + 2] = p3_before[5 * i + 2];
                }
            }
        }
        for i in 0..M3D_O2 {
            self.w3d.k[i] = [p3[5 * i], p3[5 * i + 1], p3[5 * i + 2]];
            self.w3d.a[i] = p3[5 * i + 3];
            self.w3d.phi[i] = p3[5 * i + 4];
        }
    }

    /// Pure continuous Past x Future Wave Collapse decoder (ZERO pair lookup table!).
    /// Searches a uniform 9x9x9 cavity raster + 12-step continuous gradient ascent.
    /// Returns `(a, b, psr, ghost_energy)`.
    pub fn collapse_pair(&self, x_comp: &[f64; DIM_SIG]) -> (usize, usize, f64, f64) {
        let mut u_obs = self.encode(x_comp);
        for v in &mut u_obs {
            *v *= 2.0_f64.sqrt();
        }
        let g = 9usize;
        let lim = 1.55;
        let mut grid_cells: Vec<(f64, [f64; 3])> = Vec::with_capacity(512);
        let mut candidates: Vec<(f64, [f64; 3])> = Vec::with_capacity(32);

        for ix in 0..g {
            let x = -lim + (2.0 * lim * ix as f64) / ((g - 1) as f64);
            let fx = u_obs[0] - x;
            if fx.abs() > lim + 0.25 {
                continue;
            }
            for iy in 0..g {
                let y = -lim + (2.0 * lim * iy as f64) / ((g - 1) as f64);
                let fy = u_obs[1] - y;
                if fy.abs() > lim + 0.25 {
                    continue;
                }
                for iz in 0..g {
                    let z = -lim + (2.0 * lim * iz as f64) / ((g - 1) as f64);
                    let fz = u_obs[2] - z;
                    if fz.abs() > lim + 0.25 {
                        continue;
                    }
                    let wp = self.w3d.eval([x, y, z]).max(0.0);
                    let wf = self.w3d.eval([fx, fy, fz]).max(0.0);
                    let sc = wp * wf;
                    grid_cells.push((sc, [x, y, z]));
                    if sc > 0.02 {
                        candidates.push((sc, [x, y, z]));
                    }
                }
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        if candidates.is_empty() {
            candidates.push((0.0, [0.0, 0.0, 0.0]));
        }

        // Continuous gradient-ascent wave-packet collapse from top-8 raster seeds
        let mut peaks: Vec<(f64, [f64; 3])> = Vec::new();
        for &(_, seed_u) in candidates.iter().take(8) {
            let mut u = seed_u;
            for step in 0..12 {
                let f = [u_obs[0] - u[0], u_obs[1] - u[1], u_obs[2] - u[2]];
                let (wp, gp) = self.w3d.eval_grad_u(u);
                let (wf, gf) = self.w3d.eval_grad_u(f);
                if wp <= 0.0 && wf <= 0.0 {
                    break;
                }
                let lr = 0.055 * (1.0 - step as f64 / 14.0);
                for d in 0..3 {
                    let g_tot = gp[d] * wf.max(0.05) - gf[d] * wp.max(0.05);
                    u[d] = (u[d] + lr * g_tot.clamp(-3.0, 3.0)).clamp(-CAVITY_LIM, CAVITY_LIM);
                }
            }
            let f = [u_obs[0] - u[0], u_obs[1] - u[1], u_obs[2] - u[2]];
            let sc = self.w3d.eval(u).max(0.0) * self.w3d.eval(f).max(0.0);
            peaks.push((sc, u));
        }
        peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (best_sc, best_u) = peaks[0];
        let best_f = [u_obs[0] - best_u[0], u_obs[1] - best_u[1], u_obs[2] - best_u[2]];

        // Nearest atomic concepts to the collapsed past & future wave packets
        let nearest = |pt: [f64; 3], excl: usize| -> usize {
            self.states
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != excl)
                .min_by(|(_, s1), (_, s2)| {
                    let d1 = (pt[0] - s1[0]).powi(2) + (pt[1] - s1[1]).powi(2) + (pt[2] - s1[2]).powi(2);
                    let d2 = (pt[0] - s2[0]).powi(2) + (pt[1] - s2[1]).powi(2) + (pt[2] - s2[2]).powi(2);
                    d1.total_cmp(&d2)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let p1 = nearest(best_u, usize::MAX);
        let p2 = nearest(best_f, p1);
        let (pa, pb) = if p1 < p2 { (p1, p2) } else { (p2, p1) };

        // True cavity-wide Peak-to-Sidelobe Ratio (PSR) & off-peak ghost energy
        let mut max_ghost = 0.05_f64;
        let mut ghost_sum = 0.0;
        let mut ghost_cnt = 0usize;
        for &(sc, pt) in &grid_cells {
            let d1 = (pt[0] - best_u[0]).powi(2) + (pt[1] - best_u[1]).powi(2) + (pt[2] - best_u[2]).powi(2);
            let d2 = (pt[0] - best_f[0]).powi(2) + (pt[1] - best_f[1]).powi(2) + (pt[2] - best_f[2]).powi(2);
            if d1 > 0.25 && d2 > 0.25 {
                if sc > max_ghost {
                    max_ghost = sc;
                }
                ghost_sum += sc;
                ghost_cnt += 1;
            }
        }
        let ghost_mean = ghost_sum / (ghost_cnt.max(1) as f64);
        let psr = (best_sc / max_ghost.max(0.05)).min(50.0);
        (pa, pb, psr, ghost_mean)
    }

    pub fn collision_free_rate(&self) -> f64 {
        let k = self.states.len();
        let mut sums = Vec::new();
        for i in 0..k {
            for j in (i + 1)..k {
                sums.push([
                    self.states[i][0] + self.states[j][0],
                    self.states[i][1] + self.states[j][1],
                    self.states[i][2] + self.states[j][2],
                ]);
            }
        }
        let mut clean = 0usize;
        for i in 0..sums.len() {
            let mut ok = true;
            for j in 0..sums.len() {
                if i == j {
                    continue;
                }
                let d2 = (sums[i][0] - sums[j][0]).powi(2)
                    + (sums[i][1] - sums[j][1]).powi(2)
                    + (sums[i][2] - sums[j][2]).powi(2);
                if d2 < 0.015 {
                    ok = false;
                    break;
                }
            }
            if ok {
                clean += 1;
            }
        }
        clean as f64 / sums.len().max(1) as f64
    }
}

/// Matched-capacity conventional MLP pair abductor for Suite O2 (16 -> 20 -> 16 -> 12 multi-label BCE).
#[derive(Clone, Debug)]
struct MlpPairAbductor {
    w1: Vec<[f64; DIM_SIG]>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    w3: Vec<Vec<f64>>,
    b3: Vec<f64>,
}

impl MlpPairAbductor {
    fn new(rng: &mut Rng64) -> Self {
        let h1 = 20;
        let h2 = 16;
        let sc1 = (2.0 / DIM_SIG as f64).sqrt();
        let sc2 = (2.0 / h1 as f64).sqrt();
        let sc3 = (1.0 / h2 as f64).sqrt();
        let w1 = (0..h1)
            .map(|_| std::array::from_fn(|_| sc1 * rng.normal()))
            .collect();
        let w2 = (0..h2)
            .map(|_| (0..h1).map(|_| sc2 * rng.normal()).collect())
            .collect();
        let w3 = (0..K_RULES_O2)
            .map(|_| (0..h2).map(|_| sc3 * rng.normal()).collect())
            .collect();
        Self {
            w1,
            b1: vec![0.0; h1],
            w2,
            b2: vec![0.0; h2],
            w3,
            b3: vec![0.0; K_RULES_O2],
        }
    }

    fn n_params(&self) -> usize {
        self.w1.len() * DIM_SIG
            + self.b1.len()
            + self.w2.len() * self.w1.len()
            + self.b2.len()
            + self.w3.len() * self.w2.len()
            + self.b3.len()
    }

    fn forward(&self, x: &[f64; DIM_SIG]) -> (Vec<f64>, Vec<f64>, [f64; K_RULES_O2]) {
        let h1: Vec<f64> = (0..self.w1.len())
            .map(|i| {
                let s: f64 = self.b1[i]
                    + (0..DIM_SIG).map(|j| self.w1[i][j] * x[j]).sum::<f64>();
                s.max(0.0)
            })
            .collect();
        let h2: Vec<f64> = (0..self.w2.len())
            .map(|i| {
                let s: f64 = self.b2[i]
                    + (0..h1.len()).map(|j| self.w2[i][j] * h1[j]).sum::<f64>();
                s.max(0.0)
            })
            .collect();
        let mut out = [0.0; K_RULES_O2];
        for k in 0..K_RULES_O2 {
            out[k] = self.b3[k] + (0..h2.len()).map(|j| self.w3[k][j] * h2[j]).sum::<f64>();
        }
        (h1, h2, out)
    }

    fn fit(&mut self, samples: &[([f64; DIM_SIG], usize, usize)], steps: usize, rng: &mut Rng64) {
        let h1n = self.w1.len();
        let h2n = self.w2.len();
        let np = self.n_params();
        let mut adam = Adam::new(np);

        let pack = |s: &Self| -> Vec<f64> {
            let mut v = Vec::with_capacity(np);
            for r in &s.w1 {
                v.extend_from_slice(r);
            }
            v.extend_from_slice(&s.b1);
            for r in &s.w2 {
                v.extend_from_slice(r);
            }
            v.extend_from_slice(&s.b2);
            for r in &s.w3 {
                v.extend_from_slice(r);
            }
            v.extend_from_slice(&s.b3);
            v
        };
        let unpack = |s: &mut Self, v: &[f64]| {
            let mut idx = 0;
            for r in &mut s.w1 {
                for j in 0..DIM_SIG {
                    r[j] = v[idx];
                    idx += 1;
                }
            }
            for x in &mut s.b1 {
                *x = v[idx];
                idx += 1;
            }
            for r in &mut s.w2 {
                for j in 0..h1n {
                    r[j] = v[idx];
                    idx += 1;
                }
            }
            for x in &mut s.b2 {
                *x = v[idx];
                idx += 1;
            }
            for r in &mut s.w3 {
                for j in 0..h2n {
                    r[j] = v[idx];
                    idx += 1;
                }
            }
            for x in &mut s.b3 {
                *x = v[idx];
                idx += 1;
            }
        };

        let mut p = pack(self);
        let bsz = 24usize;
        for step in 0..steps {
            unpack(self, &p);
            let mut gw1 = vec![[0.0; DIM_SIG]; h1n];
            let mut gb1 = vec![0.0; h1n];
            let mut gw2 = vec![vec![0.0; h1n]; h2n];
            let mut gb2 = vec![0.0; h2n];
            let mut gw3 = vec![vec![0.0; h2n]; K_RULES_O2];
            let mut gb3 = vec![0.0; K_RULES_O2];

            for _ in 0..bsz {
                let (x, a, b) = &samples[rng.usize_range(samples.len())];
                let (h1, h2, logits) = self.forward(x);
                let mut dl = [0.0; K_RULES_O2];
                for k in 0..K_RULES_O2 {
                    let y = if k == *a || k == *b { 1.0 } else { 0.0 };
                    let pr = 1.0 / (1.0 + (-logits[k]).exp());
                    dl[k] = (pr - y) / (bsz as f64);
                }
                let mut dh2 = vec![0.0; h2n];
                for k in 0..K_RULES_O2 {
                    gb3[k] += dl[k];
                    for j in 0..h2n {
                        gw3[k][j] += dl[k] * h2[j];
                        dh2[j] += dl[k] * self.w3[k][j];
                    }
                }
                let mut dh1 = vec![0.0; h1n];
                for i in 0..h2n {
                    if h2[i] <= 0.0 {
                        continue;
                    }
                    gb2[i] += dh2[i];
                    for j in 0..h1n {
                        gw2[i][j] += dh2[i] * h1[j];
                        dh1[j] += dh2[i] * self.w2[i][j];
                    }
                }
                for i in 0..h1n {
                    if h1[i] <= 0.0 {
                        continue;
                    }
                    gb1[i] += dh1[i];
                    for j in 0..DIM_SIG {
                        gw1[i][j] += dh1[i] * x[j];
                    }
                }
            }
            let mut g = Vec::with_capacity(np);
            for r in &gw1 {
                g.extend_from_slice(r);
            }
            g.extend_from_slice(&gb1);
            for r in &gw2 {
                g.extend_from_slice(r);
            }
            g.extend_from_slice(&gb2);
            for r in &gw3 {
                g.extend_from_slice(r);
            }
            g.extend_from_slice(&gb3);
            let lr = 0.012 * (0.2 + 0.8 * (1.0 - step as f64 / steps as f64));
            adam.update(&mut p, &g, lr, 1e-4);
        }
        unpack(self, &p);
    }

    fn predict_pair(&self, x: &[f64; DIM_SIG]) -> (usize, usize) {
        let (_, _, logits) = self.forward(x);
        let mut idx: Vec<usize> = (0..K_RULES_O2).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let (a, b) = (idx[0], idx[1]);
        if a < b { (a, b) } else { (b, a) }
    }
}

// ─── Suite O1: Official §16 ÓRBITA Benchmark (TEST + E33 h1..h64) ───────────

#[derive(Clone, Debug)]
struct O1Row {
    arm: String,
    params: usize,
    train_ms: f64,
    acc1: f64,
    rel1: f64,
    acc_indist: f64,
    acc_hole: f64,
    rel_refl: f64,
    rel_rad: f64,
    h1: f64,
    h4: f64,
    h8: f64,
    h16: f64,
    h32: f64,
    h64: f64,
    region: usize,
    clean: bool,
    gate_pass: bool,
    init_r: f64,
    learned_r: f64,
}

fn eval_e33_generic(
    mdl: &dyn Predictor,
    iter_ex: &[&Example],
    deltas: &[Vec<f64>],
) -> (usize, [f64; 7]) {
    let mut zs: Vec<Vec<f64>> = iter_ex.iter().map(|e| e.input.clone()).collect();
    let _ = deltas;
    let mut region = 0usize;
    let mut contiguous = true;
    let mut hs = [0.0; 7];
    let mut hi = 0usize;
    for t in 1..=*HORIZONS.last().unwrap() {
        for (i, e) in iter_ex.iter().enumerate() {
            zs[i] = mdl.predict(&zs[i], &e.context);
        }
        if HORIZONS.contains(&t) {
            let mut ok = 0usize;
            for (i, e) in iter_ex.iter().enumerate() {
                let y = target_h(e, t);
                if zs[i].iter().all(|v| v.is_finite()) && m::correct(&zs[i], &y) {
                    ok += 1;
                }
            }
            let acc = ok as f64 / iter_ex.len().max(1) as f64;
            if contiguous && acc >= 0.5 {
                region = t;
            } else {
                contiguous = false;
            }
            hs[hi] = acc;
            hi += 1;
        }
    }
    (region, hs)
}

fn eval_o1_arm(
    arm_name: &str,
    mdl: &dyn Predictor,
    s: &Sealed,
    aud: &Auditor,
    flags: &AuditFlags,
    iter_ex: &[&Example],
    deltas: &[Vec<f64>],
    train_ms: f64,
    init_r: f64,
    learned_r: f64,
) -> O1Row {
    let ev = evaluate(mdl, &s.ds.test, 1, Some((aud, flags)), None);
    let clean = ev.leaked == 0 && ev.queries.total() == 0;

    // Breakdown in_dist vs interp_hole, reflect (fam 2), radial_warp (fam 7)
    let (mut ok_in, mut n_in, mut ok_ho, mut n_ho) = (0usize, 0usize, 0usize, 0usize);
    let (mut rel_refl, mut n_refl, mut rel_rad, mut n_rad) = (0.0, 0usize, 0.0, 0usize);
    for e in &s.ds.test {
        let p = mdl.predict(&e.input, &e.context);
        let ok = p.iter().all(|v| v.is_finite()) && m::correct(&p, &e.expected);
        let re = if p.iter().all(|v| v.is_finite()) {
            m::rel_err(&p, &e.expected).min(10.0)
        } else {
            10.0
        };
        if e.subset == "interp_hole" {
            n_ho += 1;
            ok_ho += ok as usize;
        } else {
            n_in += 1;
            ok_in += ok as usize;
        }
        if e.family_id == 2 {
            rel_refl += re;
            n_refl += 1;
        }
        if e.family_id == 7 {
            rel_rad += re;
            n_rad += 1;
        }
    }
    let (region, hs) = eval_e33_generic(mdl, iter_ex, deltas);
    let gate_pass = region >= 8 && clean;
    O1Row {
        arm: arm_name.to_string(),
        params: mdl.params(),
        train_ms,
        acc1: ev.acc,
        rel1: ev.rel_err,
        acc_indist: ok_in as f64 / n_in.max(1) as f64,
        acc_hole: ok_ho as f64 / n_ho.max(1) as f64,
        rel_refl: rel_refl / n_refl.max(1) as f64,
        rel_rad: rel_rad / n_rad.max(1) as f64,
        h1: hs[0],
        h4: hs[2],
        h8: hs[3],
        h16: hs[4],
        h32: hs[5],
        h64: hs[6],
        region,
        clean,
        gate_pass,
        init_r,
        learned_r,
    }
}

/// Builds per-family training residuals STRICTLY from `s.ds.train` (0 DEV, 0 TEST, 0 teacher queries!).
fn collect_train_residuals(base: &Orbita, train: &[Example]) -> Vec<Vec<([f64; 4], [f64; 2], f64)>> {
    let mut per_fam: Vec<Vec<([f64; 4], [f64; 2], f64)>> = vec![Vec::new(); N_FAM];
    for e in train {
        let pred = base.step(&e.input, &e.context);
        let y_norm2: f64 = e.expected.iter().map(|v| v * v).sum::<f64>().max(1e-6);
        let inv_n2 = (e.input.len() as f64 / 2.0) / y_norm2;
        for i in 0..e.input.len() / 2 {
            let u = [
                e.input[2 * i],
                e.input[2 * i + 1],
                e.params[0],
                e.params[1],
            ];
            let res = [
                e.expected[2 * i] - pred[2 * i],
                e.expected[2 * i + 1] - pred[2 * i + 1],
            ];
            per_fam[e.family_id].push((u, res, inv_n2));
        }
    }
    per_fam
}

fn train_o1_wave(
    base: &Orbita,
    per_fam: &[Vec<([f64; 4], [f64; 2], f64)>],
    mode: SpecMode,
    zeros: Option<&[f64]>,
    seed: u64,
) -> (OrbitaWithWave, f64, f64, f64) {
    let t0 = Instant::now();
    let mut rng = Rng64::from_seed(seed, 0x0101 ^ (mode as u64));
    let mut fields = Vec::with_capacity(N_FAM);
    let mut sum_ir = 0.0;
    let mut sum_lr = 0.0;
    for f in 0..N_FAM {
        let mut wf = WaveField4D::new(M_DYN, mode, zeros, 0.35, 2.0, &mut rng);
        sum_ir += wf.init_r;
        wf.fit(&per_fam[f], 1100, 0.0020, 0.018, 5.0e-4, 2.2, 1.4, 0.08);
        sum_lr += wf.learned_r();
        fields.push(wf);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    (
        OrbitaWithWave {
            base: base.clone(),
            fields,
            label: mode.name().to_string(),
        },
        ms,
        sum_ir / N_FAM as f64,
        sum_lr / N_FAM as f64,
    )
}

// ─── Suite O2: Unknown-Context & Compositional Rule-Pair Collapse in ÓRBITA ─

#[derive(Clone, Debug)]
struct O2Row {
    arm: String,
    params: usize,
    train_ms: f64,
    unseen_pair_acc: f64,
    train_pair_acc: f64,
    cfr: f64,
    init_r: f64,
    learned_r: f64,
    psr: f64,
}

fn build_o2_prototypes(seed: u64) -> [[f64; DIM_SIG]; K_RULES_O2] {
    let mut rng = Rng64::from_seed(seed, 0x0202);
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let mut protos = [[0.0; DIM_SIG]; K_RULES_O2];
    let fixed_p = [0.68, -0.52];
    for f in 0..K_RULES_O2 {
        let mut avg = [0.0; DIM_SIG];
        for rep in 0..6 {
            let mut xrng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(
                seed ^ 0x2222 ^ ((f as u64) << 8) ^ (rep as u64),
            );
            let x0 = make_object(&mut xrng, K_POINTS, cls[rep % 3]);
            let y1 = c4_rule(f, fixed_p, &x0);
            let s = dynamical_signature_16d(&x0, &y1);
            for j in 0..DIM_SIG {
                avg[j] += s[j];
            }
        }
        // Add tiny unique deterministic phase-projection orthogonalization only via natural parameter diversity
        let mut xrng2 = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0x9999 ^ (f as u64));
        let x0b = make_object(&mut xrng2, K_POINTS, ObjectClass::Blob);
        let y1b = c4_rule(f, [-0.55, 0.72], &x0b);
        let sb = dynamical_signature_16d(&x0b, &y1b);
        for j in 0..DIM_SIG {
            avg[j] = avg[j] / 6.0 + 0.35 * sb[(j + 3) % DIM_SIG] + 0.03 * rng.normal();
        }
        let nrm = avg.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
        for j in 0..DIM_SIG {
            protos[f][j] = avg[j] / nrm;
        }
    }
    protos
}

fn split_o2_pairs(seed: u64) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut all = Vec::new();
    for a in 0..K_RULES_O2 {
        for b in (a + 1)..K_RULES_O2 {
            all.push((a, b));
        }
    }
    let mut rng = Rng64::from_seed(seed, 0x0255);
    for i in (1..all.len()).rev() {
        let j = rng.usize_range(i + 1);
        all.swap(i, j);
    }
    // 26 train pairs (39.4%), 40 strictly unseen pairs (60.6%)
    let train = all[..26].to_vec();
    let unseen = all[26..].to_vec();
    (train, unseen)
}

fn compose_sig(
    protos: &[[f64; DIM_SIG]; K_RULES_O2],
    a: usize,
    b: usize,
    sigma: f64,
    rng: &mut Rng64,
) -> [f64; DIM_SIG] {
    let inv = 1.0 / 2.0_f64.sqrt();
    let mut out = [0.0; DIM_SIG];
    for j in 0..DIM_SIG {
        out[j] = inv * (protos[a][j] + protos[b][j]) + sigma * rng.normal();
    }
    out
}

fn run_suite_o2(
    seed: u64,
    zeros: &[f64],
) -> (Vec<O2Row>, SpectralCavity3D, [[f64; DIM_SIG]; K_RULES_O2], Vec<(usize, usize)>) {
    let protos = build_o2_prototypes(seed);
    let (train_pairs, unseen_pairs) = split_o2_pairs(seed);

    let mut rows = Vec::new();
    let mut asc_cav_saved: Option<SpectralCavity3D> = None;

    for mode in [
        SpecMode::Adaptive,
        SpecMode::RiemannFixed,
        SpecMode::FourierFixed,
        SpecMode::PoissonFixed,
    ] {
        let t0 = Instant::now();
        let mut rng = Rng64::from_seed(seed, 0x02A0 ^ (mode as u64));
        let z_opt = if mode == SpecMode::RiemannFixed {
            Some(zeros)
        } else {
            None
        };
        let mut cav = SpectralCavity3D::new(mode, z_opt, &mut rng);
        cav.fit(&protos, &train_pairs, &mut rng);
        let train_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut erng = Rng64::from_seed(seed, 0x02E0);
        let mut hit_unseen = 0usize;
        let mut psr_sum = 0.0;
        for &(a, b) in &unseen_pairs {
            let x = compose_sig(&protos, a, b, 0.025, &mut erng);
            let (pa, pb, psr, _) = cav.collapse_pair(&x);
            if pa == a && pb == b {
                hit_unseen += 1;
            }
            psr_sum += psr;
        }
        let mut hit_train = 0usize;
        for &(a, b) in &train_pairs {
            let x = compose_sig(&protos, a, b, 0.025, &mut erng);
            let (pa, pb, _, _) = cav.collapse_pair(&x);
            if pa == a && pb == b {
                hit_train += 1;
            }
        }
        rows.push(O2Row {
            arm: mode.name().to_string(),
            params: cav.n_params(),
            train_ms,
            unseen_pair_acc: hit_unseen as f64 / unseen_pairs.len() as f64,
            train_pair_acc: hit_train as f64 / train_pairs.len() as f64,
            cfr: cav.collision_free_rate(),
            init_r: cav.init_r,
            learned_r: cav.learned_r(),
            psr: psr_sum / unseen_pairs.len() as f64,
        });
        if mode == SpecMode::Adaptive {
            asc_cav_saved = Some(cav);
        }
    }

    // Matched-capacity MLP conventional abductor
    let t0 = Instant::now();
    let mut mrng = Rng64::from_seed(seed, 0x02F0);
    let mut mlp = MlpPairAbductor::new(&mut mrng);
    let mut train_samples = Vec::new();
    for _ in 0..8 {
        for &(a, b) in &train_pairs {
            train_samples.push((compose_sig(&protos, a, b, 0.015, &mut mrng), a, b));
        }
    }
    mlp.fit(&train_samples, 900, &mut mrng);
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut erng = Rng64::from_seed(seed, 0x02E0);
    let mut hit_u = 0usize;
    for &(a, b) in &unseen_pairs {
        let x = compose_sig(&protos, a, b, 0.025, &mut erng);
        let (pa, pb) = mlp.predict_pair(&x);
        if pa == a && pb == b {
            hit_u += 1;
        }
    }
    let mut hit_t = 0usize;
    for &(a, b) in &train_pairs {
        let x = compose_sig(&protos, a, b, 0.025, &mut erng);
        let (pa, pb) = mlp.predict_pair(&x);
        if pa == a && pb == b {
            hit_t += 1;
        }
    }
    rows.push(O2Row {
        arm: "ORBITA_MLP".to_string(),
        params: mlp.n_params(),
        train_ms,
        unseen_pair_acc: hit_u as f64 / unseen_pairs.len() as f64,
        train_pair_acc: hit_t as f64 / train_pairs.len() as f64,
        cfr: 0.0,
        init_r: 0.0,
        learned_r: 0.0,
        psr: 0.0,
    });

    (rows, asc_cav_saved.unwrap(), protos, unseen_pairs)
}

// ─── Suite O3: Continual Few-Shot Learning of ÓRBITA's 4 New Nonlinear Families 8..11 ─

#[derive(Clone, Debug)]
struct O3Row {
    arm: String,
    expert_params: usize,
    adapt_ms: f64,
    known_h8: f64,
    known_h16: f64,
    new_h1: f64,
    new_h8: f64,
    new_h16: f64,
    new_rel16: f64,
    route_purity: f64,
}

fn gen_new_fam_stream(seed: u64, n_per_fam: usize) -> Vec<C4Ep> {
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0x0357_1234);
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let mut eps = Vec::new();
    for &fam in &NEW_FAMS_ALL {
        for i in 0..n_per_fam {
            let p = [
                rand::Rng::gen_range(&mut rng, -1.0..1.0),
                rand::Rng::gen_range(&mut rng, -1.0..1.0),
            ];
            let x0 = make_object(&mut rng, K_POINTS, cls[i % 3]);
            let mut ys = Vec::with_capacity(16);
            let mut y = x0.clone();
            for _ in 0..16 {
                y = c4_rule(fam, p, &y);
                ys.push(y.clone());
            }
            eps.push(C4Ep { fam, p, x0, ys });
        }
    }
    eps
}

fn gen_all12_heldout(seed: u64, n_per_fam: usize) -> Vec<C4Ep> {
    // Reuse constelacion4::gen_heldout for 0..7 + 8,10,11 and add fam 9 (spiral) cleanly!
    let mut base_h = gen_heldout(seed, n_per_fam);
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0x0399_591A);
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    for i in 0..n_per_fam {
        let p = [
            rand::Rng::gen_range(&mut rng, -1.0..1.0),
            rand::Rng::gen_range(&mut rng, -1.0..1.0),
        ];
        let x0 = make_object(&mut rng, K_POINTS, cls[i % 3]);
        let mut ys = Vec::with_capacity(16);
        let mut y = x0.clone();
        for _ in 0..16 {
            y = c4_rule(9, p, &y);
            ys.push(y.clone());
        }
        base_h.push(C4Ep { fam: 9, p, x0, ys });
    }
    base_h
}

fn eval_rollout_c4(
    step_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    start: &[f64],
    k_start: usize,
    ep: &C4Ep,
) -> (u8, u8, u8, f64) {
    let mut z = start.to_vec();
    let (mut ok1, mut ok8, mut ok16) = (0u8, 0u8, 0u8);
    let mut rel16 = 10.0;
    for t in (k_start + 1)..=16 {
        z = step_fn(&z);
        let fin = z.iter().all(|v| v.is_finite());
        let ok = (fin && m::correct(&z, &ep.ys[t - 1])) as u8;
        if t == k_start + 1 {
            ok1 = ok;
        }
        if t == 8 {
            ok8 = ok;
        }
        if t == 16 {
            ok16 = ok;
            rel16 = if fin {
                m::rel_err(&z, &ep.ys[15]).min(10.0)
            } else {
                10.0
            };
        }
    }
    (ok1, ok8, ok16, rel16)
}

fn collect_expert_samples(eps: &[&C4Ep], k_obs: usize) -> Vec<([f64; 4], [f64; 2], f64)> {
    let mut out = Vec::new();
    for ep in eps {
        for t in 0..k_obs {
            let x = if t == 0 { &ep.x0 } else { &ep.ys[t - 1] };
            let y = &ep.ys[t];
            let yn2: f64 = y.iter().map(|v| v * v).sum::<f64>().max(1e-6);
            let inv_n2 = (x.len() as f64 / 2.0) / yn2;
            for i in 0..x.len() / 2 {
                let u = [x[2 * i], x[2 * i + 1], ep.p[0], ep.p[1]];
                // Predict displacement Δq = y - x (clean zero-init identity!)
                let dq = [y[2 * i] - x[2 * i], y[2 * i + 1] - x[2 * i + 1]];
                out.push((u, dq, inv_n2));
            }
        }
    }
    out
}

fn step_net2_expert(net: &SpectralWaveNet2, x: &[f64], p: [f64; 2]) -> Vec<f64> {
    let mut y = x.to_vec();
    for i in 0..x.len() / 2 {
        let dq = net.eval([x[2 * i], x[2 * i + 1], p[0], p[1]]);
        y[2 * i] += dq[0];
        y[2 * i + 1] += dq[1];
    }
    y
}

fn surprise_2step(step_fn: &dyn Fn(&[f64]) -> Vec<f64>, ep: &C4Ep) -> f64 {
    let z1 = step_fn(&ep.x0);
    let e1 = if z1.iter().all(|v| v.is_finite()) {
        m::rel_err(&z1, &ep.ys[0]).min(10.0)
    } else {
        10.0
    };
    let z2 = step_fn(&z1);
    let e2 = if z2.iter().all(|v| v.is_finite()) {
        m::rel_err(&z2, &ep.ys[1]).min(10.0)
    } else {
        10.0
    };
    0.5 * (e1 + e2)
}

fn run_suite_o3(
    seed: u64,
    base: &Orbita,
    _asc_o1: &OrbitaWithWave,
    zeros: &[f64],
) -> (Vec<O3Row>, Vec<SpectralWaveNet2>, Vec<C4Ep>) {
    // 60 few-shot episodes per new family (8,9,10,11), observing k_obs=12 steps each (10x fewer than GROW's 600/fam!)
    let stream = gen_new_fam_stream(seed, 60);
    // 12 held-out episodes per family across all 12 families (96 known + 48 new = 144 episodes)
    let held = gen_all12_heldout(seed, 12);
    let k_eval = 2usize;
    let tau = 0.025_f64;

    // Precompute frozen ÓRBITA reid (`OBS`) once per held-out episode
    let mut obs_cache: Vec<(usize, [f64; 2], f64)> = Vec::with_capacity(held.len());
    for ep in &held {
        let (g, pp, r) = reid(base, &ep.x0, &ep.ys, k_eval, ep.p);
        obs_cache.push((g, pp, r / (k_eval as f64)));
    }

    let mut rows = Vec::new();

    // 1. ORBITA_BASE (frozen ÓRBITA alone; on known uses true context, on unknown defaults to fam 0)
    {
        let (mut kh8, mut kh16, mut kn) = (0.0, 0.0, 0.0);
        let (mut nh1, mut nh8, mut nh16, mut nrel, mut nn) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for ep in &held {
            let c = if ep.fam < N_FAM {
                context_of(ep.fam, ep.p)
            } else {
                context_of(0, ep.p)
            };
            let (o1, o8, o16, r16) = eval_rollout_c4(&|z| base.step(z, &c), &ep.ys[k_eval - 1], k_eval, ep);
            if ep.fam < N_FAM {
                kh8 += o8 as f64;
                kh16 += o16 as f64;
                kn += 1.0;
            } else {
                nh1 += o1 as f64;
                nh8 += o8 as f64;
                nh16 += o16 as f64;
                nrel += r16;
                nn += 1.0;
            }
        }
        rows.push(O3Row {
            arm: "ORBITA_BASE".into(),
            expert_params: 0,
            adapt_ms: 0.0,
            known_h8: kh8 / kn,
            known_h16: kh16 / kn,
            new_h1: nh1 / nn,
            new_h8: nh8 / nn,
            new_h16: nh16 / nn,
            new_rel16: nrel / nn,
            route_purity: 0.0,
        });
    }

    // 2. ORBITA_OBS (frozen ÓRBITA + reid)
    {
        let (mut kh8, mut kh16, mut kn) = (0.0, 0.0, 0.0);
        let (mut nh1, mut nh8, mut nh16, mut nrel, mut nn) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (i, ep) in held.iter().enumerate() {
            let (g, pp, _) = obs_cache[i];
            let cg = context_of(g, pp);
            let (o1, o8, o16, r16) = eval_rollout_c4(&|z| base.step(z, &cg), &ep.ys[k_eval - 1], k_eval, ep);
            if ep.fam < N_FAM {
                kh8 += o8 as f64;
                kh16 += o16 as f64;
                kn += 1.0;
            } else {
                nh1 += o1 as f64;
                nh8 += o8 as f64;
                nh16 += o16 as f64;
                nrel += r16;
                nn += 1.0;
            }
        }
        rows.push(O3Row {
            arm: "ORBITA_OBS".into(),
            expert_params: 0,
            adapt_ms: 0.0,
            known_h8: kh8 / kn,
            known_h16: kh16 / kn,
            new_h1: nh1 / nn,
            new_h8: nh8 / nn,
            new_h16: nh16 / nn,
            new_rel16: nrel / nn,
            route_purity: 0.0,
        });
    }

    // 3. ORBITA_LAT (ÓRBITA's own real latent polynomial adapter from orbita/src/latent.rs!)
    {
        let t0 = Instant::now();
        let mut lib: Vec<LatEntry> = Vec::new();
        let (lam0, lamm) = (1e-2, 1e-2);
        // Populate LAT memory from 80 stream episodes (20 per new family)
        for (si, ep) in stream.iter().enumerate() {
            if si % 3 != 0 {
                continue;
            }
            let (g, _, r) = reid(base, &ep.x0, &ep.ys, k_eval, ep.p);
            let sk = r / k_eval as f64;
            let cx = ctx8(base, ep.p);
            let lep = LatEp {
                idx: si,
                tid: String::new(),
                fam: ep.fam,
                ph: ep.p,
                c: context_of(g, ep.p),
                x0: ep.x0.clone(),
                ys: ep.ys.clone(),
            };
            let (dec, _, _) = lat_decide(base, &cx, &lib, &lep, k_eval, sk, g, tau, lam0, lamm);
            if let Some((g0, aa)) = dec {
                lib.push(LatEntry {
                    g0,
                    a: aa,
                    fam: ep.fam,
                });
            }
        }
        let adapt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (mut kh8, mut kh16, mut kn) = (0.0, 0.0, 0.0);
        let (mut nh1, mut nh8, mut nh16, mut nrel, mut nn) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (i, ep) in held.iter().enumerate() {
            let (g, pp, sk) = obs_cache[i];
            let cx = ctx8(base, ep.p);
            let lep = LatEp {
                idx: i,
                tid: String::new(),
                fam: ep.fam,
                ph: ep.p,
                c: context_of(g, pp),
                x0: ep.x0.clone(),
                ys: ep.ys.clone(),
            };
            let (dec, _, _) = lat_decide(base, &cx, &lib, &lep, k_eval, sk, g, tau, lam0, lamm);
            let cg = context_of(g, pp);
            let (o1, o8, o16, r16) = match &dec {
                None => eval_rollout_c4(&|z| base.step(z, &cg), &ep.ys[k_eval - 1], k_eval, ep),
                Some((g0, aa)) => eval_rollout_c4(
                    &|z| lat_step(base, &cx.cps[*g0], &cx.cs[*g0], aa, z),
                    &ep.ys[k_eval - 1],
                    k_eval,
                    ep,
                ),
            };
            if ep.fam < N_FAM {
                kh8 += o8 as f64;
                kh16 += o16 as f64;
                kn += 1.0;
            } else {
                nh1 += o1 as f64;
                nh8 += o8 as f64;
                nh16 += o16 as f64;
                nrel += r16;
                nn += 1.0;
            }
        }
        rows.push(O3Row {
            arm: "ORBITA_LAT".into(),
            expert_params: lib.len() * crate::latent::npar(),
            adapt_ms,
            known_h8: kh8 / kn,
            known_h16: kh16 / kn,
            new_h1: nh1 / nn,
            new_h8: nh8 / nn,
            new_h16: nh16 / nn,
            new_rel16: nrel / nn,
            route_purity: 0.0,
        });
    }

    // Parameter-sign-aligned generic dynamical invariants over 6 observed steps (100% unsupervised, ZERO family labels!)
    let ep_inv_sig = |ep: &C4Ep| -> [f64; 8] {
        let s0 = if ep.p[0] >= 0.0 { 1.0 } else { -1.0 };
        let s1 = if ep.p[1] >= 0.0 { 1.0 } else { -1.0 };
        let (mut curl_a, mut div_a, mut nl_a, mut twist_a, mut tref_a, mut wavy_a, mut tanh_a, mut dr_a) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for t in 0..6 {
            let x = if t == 0 { &ep.x0 } else { &ep.ys[t - 1] };
            let y = &ep.ys[t];
            let np = x.len() / 2;
            let mut ata = [0.0; 9];
            let mut atbx = [0.0; 3];
            let mut atby = [0.0; 3];
            for i in 0..np {
                let u = [x[2 * i], x[2 * i + 1], 1.0];
                for r in 0..3 {
                    for c in 0..3 {
                        ata[r * 3 + c] += u[r] * u[c];
                    }
                    atbx[r] += u[r] * y[2 * i];
                    atby[r] += u[r] * y[2 * i + 1];
                }
            }
            let wx = solve_ridge(&ata, &atbx, 3, 1e-6);
            let wy = solve_ridge(&ata, &atby, 3, 1e-6);
            let (jxx, jxy, tx) = (wx[0], wx[1], wx[2]);
            let (jyx, jyy, ty) = (wy[0], wy[1], wy[2]);
            curl_a += (jyx - jxy) * s0 / ep.p[0].abs().max(0.2);
            div_a += ((jxx - 1.0) + (jyy - 1.0)) * s1 / ep.p[1].abs().max(0.2);
            let nf = np as f64;
            for i in 0..np {
                let (px, py) = (x[2 * i], x[2 * i + 1]);
                let (nx, ny) = (y[2 * i], y[2 * i + 1]);
                let rx = nx - (jxx * px + jxy * py + tx);
                let ry = ny - (jyx * px + jyy * py + ty);
                nl_a += (rx * rx + ry * ry).sqrt() / nf;
                let r0 = (px * px + py * py).sqrt().max(1e-6);
                let r1 = (nx * nx + ny * ny).sqrt();
                let dr = (r1 - r0) / r0;
                dr_a += dr * s1 / nf;
                let th0 = py.atan2(px);
                let mut dth = ny.atan2(nx) - th0;
                while dth > PI { dth -= 2.0 * PI; }
                while dth < -PI { dth += 2.0 * PI; }
                twist_a += (r0 - 0.7) * dth * s1 / (ep.p[1].abs().max(0.2) * nf);
                tref_a += dr * (3.0 * th0).sin() * s1 / (ep.p[1].abs().max(0.2) * nf);
                wavy_a += (rx * (2.0 * py).sin() + ry * (2.0 * px).sin()) * s1 / (ep.p[1].abs().max(0.2) * nf);
                tanh_a += (rx * (2.0 * py).tanh() * s0 + ry * (2.0 * px).tanh() * s1) / nf;
            }
        }
        let mut v = [
            curl_a * 0.4,
            div_a * 1.2,
            nl_a * 3.0,
            twist_a * 3.5,
            tref_a * 8.0,
            wavy_a * 8.0,
            tanh_a * 8.0,
            dr_a * 2.0,
        ];
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= nrm;
        }
        v
    };

    let stream_sigs: Vec<[f64; 8]> = stream.iter().map(ep_inv_sig).collect();
    // 16 deterministic k-means++ restarts picking lowest within-cluster inertia (100% unsupervised, 0 labels!)
    let mut km_rng = Rng64::from_seed(seed, 0x03C1);
    let mut assign = vec![0usize; stream.len()];
    let mut best_inertia = f64::INFINITY;
    let min_cl = stream.len() / 12;
    for _ in 0..16 {
        let mut centers: Vec<[f64; 8]> = vec![stream_sigs[km_rng.usize_range(stream.len())]];
        while centers.len() < 4 {
            let mut d2s = Vec::with_capacity(stream.len());
            let mut tot = 0.0;
            for s in &stream_sigs {
                let md = centers
                    .iter()
                    .map(|c| (0..8).map(|j| (s[j] - c[j]).powi(2)).sum::<f64>())
                    .fold(f64::INFINITY, f64::min);
                d2s.push(md);
                tot += md;
            }
            let mut pick = km_rng.u01() * tot.max(1e-12);
            let mut chosen = 0usize;
            for (i, &d2) in d2s.iter().enumerate() {
                pick -= d2;
                if pick <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centers.push(stream_sigs[chosen]);
        }
        let mut cur = vec![0usize; stream.len()];
        let mut inertia = 0.0;
        for _ in 0..25 {
            inertia = 0.0;
            for (i, s) in stream_sigs.iter().enumerate() {
                let (mut bi, mut bd) = (0usize, f64::INFINITY);
                for c in 0..4 {
                    let d: f64 = (0..8).map(|j| (s[j] - centers[c][j]).powi(2)).sum();
                    if d < bd {
                        bd = d;
                        bi = c;
                    }
                }
                cur[i] = bi;
                inertia += bd;
            }
            for c in 0..4 {
                let mut avg = [0.0; 8];
                let mut cnt = 0usize;
                for (i, &cl) in cur.iter().enumerate() {
                    if cl == c {
                        for j in 0..8 {
                            avg[j] += stream_sigs[i][j];
                        }
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    let nrm = avg.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
                    for j in 0..8 {
                        centers[c][j] = avg[j] / nrm;
                    }
                }
            }
        }
        let mut counts = [0usize; 4];
        for &cl in &cur {
            counts[cl] += 1;
        }
        if counts.iter().all(|&n| n >= min_cl) && inertia < best_inertia {
            best_inertia = inertia;
            assign = cur;
        }
    }

    let surprise_4step = |net: &SpectralWaveNet2, ep: &C4Ep| -> f64 {
        let mut z = ep.x0.clone();
        let mut tot = 0.0;
        for t in 0..4 {
            z = step_net2_expert(net, &z, ep.p);
            let e = if z.iter().all(|v| v.is_finite()) {
                m::rel_err(&z, &ep.ys[t]).min(10.0)
            } else {
                10.0
            };
            tot += e;
        }
        0.25 * tot
    };

    let mut asc_new_experts_saved: Vec<SpectralWaveNet2> = Vec::new();

    // Arms 4..8: Exact 1:1 parameter parity (4 experts x 1,282 params = 5,128 params; identical EM & compute!)
    let arm_specs: [(&str, SpecMode, bool, u64); 5] = [
        ("ORBITA_ASC", SpecMode::Adaptive, false, 0x03A0),
        ("ORBITA_RIEMANN_FIXED", SpecMode::RiemannFixed, false, 0x03A1),
        ("ORBITA_FOURIER_FIXED", SpecMode::FourierFixed, false, 0x03A2),
        ("ORBITA_POISSON_FIXED", SpecMode::PoissonFixed, false, 0x03A3),
        ("ORBITA_MLP", SpecMode::Adaptive, true, 0x03F0),
    ];

    for (arm_name, mode, is_mlp, salt) in arm_specs {
        let t0 = Instant::now();
        let mut rng = Rng64::from_seed(seed, salt);
        let z_opt = if !is_mlp && mode == SpecMode::RiemannFixed {
            Some(zeros)
        } else {
            None
        };
        let mut experts: Vec<SpectralWaveNet2> = (0..4)
            .map(|_| SpectralWaveNet2::new(mode, is_mlp, z_opt, &mut rng))
            .collect();

        // 3 rounds of 100% unsupervised self-consistent EM assignment + trimmed final fit (ZERO family labels!)
        let mut cur_assign = assign.clone();
        let mut last_surp = vec![0.0_f64; stream.len()];
        for round in 0..3 {
            for c in 0..4 {
                let mut idxs: Vec<usize> = (0..stream.len())
                    .filter(|&i| cur_assign[i] == c)
                    .collect();
                if round == 2 && idxs.len() >= 10 {
                    // Robust unsupervised trimmed-EM: drop residual high-surprise outliers (> 2x cluster median)
                    idxs.sort_by(|&a, &b| last_surp[a].total_cmp(&last_surp[b]));
                    let med = last_surp[idxs[idxs.len() / 2]].max(1e-4);
                    let keep_min = (idxs.len() * 4) / 5;
                    let keep_n = idxs
                        .iter()
                        .take_while(|&&i| last_surp[i] <= 2.0 * med)
                        .count()
                        .max(keep_min);
                    idxs.truncate(keep_n);
                    // Clean fresh re-initialization on the purified cluster!
                    experts[c] = SpectralWaveNet2::new(mode, is_mlp, z_opt, &mut rng);
                }
                let eps_c: Vec<&C4Ep> = idxs.iter().map(|&i| &stream[i]).collect();
                let samps = collect_expert_samples(&eps_c, 12);
                let (steps, bsz) = if round < 2 { (900, 48) } else { (10000, 64) };
                experts[c].fit(&samps, steps, bsz, &mut rng);
            }
            if round < 2 {
                for (i, ep) in stream.iter().enumerate() {
                    let mut best_c = 0usize;
                    let mut best_s = f64::INFINITY;
                    for c in 0..4 {
                        let s = surprise_4step(&experts[c], ep);
                        if s < best_s {
                            best_s = s;
                            best_c = c;
                        }
                    }
                    cur_assign[i] = best_c;
                    last_surp[i] = best_s;
                }
            }
        }
        let adapt_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Compute majority family of each discovered cluster for purity metric (eval only!)
        let mut maj_fam = [0usize; 4];
        for c in 0..4 {
            let mut counts = [0usize; 12];
            for (i, &cl) in cur_assign.iter().enumerate() {
                if cl == c {
                    counts[stream[i].fam] += 1;
                }
            }
            maj_fam[c] = (0..12).max_by_key(|&f| counts[f]).unwrap();
        }

        // Evaluate on held-out (144 episodes: 96 known + 48 new; NO labels used at test time!)
        let (mut kh8, mut kh16, mut kn) = (0.0, 0.0, 0.0);
        let (mut nh1, mut nh8, mut nh16, mut nrel, mut nn) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut pure_hits, mut new_routed) = (0usize, 0usize);

        for (i, ep) in held.iter().enumerate() {
            let (g, pp, sk) = obs_cache[i];
            let cg = context_of(g, pp);
            let mut best_exp = 0usize;
            let mut best_s = f64::INFINITY;
            for c in 0..4 {
                let s = surprise_2step(&|z| step_net2_expert(&experts[c], z, ep.p), ep);
                if s < best_s {
                    best_s = s;
                    best_exp = c;
                }
            }
            // Identical known-vs-novel gate across all arms: frozen ÓRBITA handles known (0 forgetting!)
            let use_known = sk <= tau && sk <= best_s;
            let (o1, o8, o16, r16) = if use_known {
                eval_rollout_c4(&|z| base.step(z, &cg), &ep.ys[k_eval - 1], k_eval, ep)
            } else {
                if ep.fam >= N_FAM {
                    new_routed += 1;
                    if maj_fam[best_exp] == ep.fam {
                        pure_hits += 1;
                    }
                }
                eval_rollout_c4(
                    &|z| step_net2_expert(&experts[best_exp], z, ep.p),
                    &ep.ys[k_eval - 1],
                    k_eval,
                    ep,
                )
            };
            if ep.fam < N_FAM {
                kh8 += o8 as f64;
                kh16 += o16 as f64;
                kn += 1.0;
            } else {
                nh1 += o1 as f64;
                nh8 += o8 as f64;
                nh16 += o16 as f64;
                nrel += r16;
                nn += 1.0;
            }
        }
        rows.push(O3Row {
            arm: arm_name.to_string(),
            expert_params: experts.iter().map(|e| e.n_params()).sum(),
            adapt_ms,
            known_h8: kh8 / kn,
            known_h16: kh16 / kn,
            new_h1: nh1 / nn,
            new_h8: nh8 / nn,
            new_h16: nh16 / nn,
            new_rel16: nrel / nn,
            route_purity: pure_hits as f64 / new_routed.max(1) as f64,
        });
        if !is_mlp && mode == SpecMode::Adaptive {
            for &f_want in &NEW_FAMS_ALL {
                let best_c = (0..4)
                    .min_by(|&a, &b| {
                        let ea: f64 = stream
                            .iter()
                            .filter(|e| e.fam == f_want)
                            .map(|e| surprise_2step(&|z| step_net2_expert(&experts[a], z, e.p), e))
                            .sum();
                        let eb: f64 = stream
                            .iter()
                            .filter(|e| e.fam == f_want)
                            .map(|e| surprise_2step(&|z| step_net2_expert(&experts[b], z, e.p), e))
                            .sum();
                        ea.total_cmp(&eb)
                    })
                    .unwrap();
                asc_new_experts_saved.push(experts[best_c].clone());
            }
        }
    }

    (rows, asc_new_experts_saved, held)
}

// ─── Suite O4: Prime-Frequency Spectral Sieve ("¿Qué inferencia se esconde?") ─

#[derive(Clone, Debug)]
struct O4Row {
    subtask: String,
    slice: String,
    n_modes: usize,
    acc1: f64,
    acc16_or_pair: f64,
    rel1_or_ghost: f64,
    disp_cos_or_psr: f64,
    manifold_dist: f64,
    odd_parity_frac: f64,
    irred_rel1: f64,
    comp_rel1: f64,
    energy_retained: f64,
}

fn slice_frequencies(slice_name: &str, zeros: &[f64], omega0: f64, rng: &mut Rng64) -> Vec<f64> {
    let fmin = 2.0 * omega0;
    let fmax = 29.0 * omega0;
    match slice_name {
        "prime_harmonic_exact" => PRIMES_10.iter().map(|&p| (p as f64) * omega0).collect(),
        "riemann_prime_dual" => scaled_riemann(zeros, 10, fmin, fmax),
        "log_prime_exact" => {
            let l0 = (2.0_f64).ln();
            let l1 = (29.0_f64).ln();
            PRIMES_10
                .iter()
                .map(|&p| fmin + (fmax - fmin) * ((p as f64).ln() - l0) / (l1 - l0))
                .collect()
        }
        "composite_only" => COMPOSITES_10.iter().map(|&c| (c as f64) * omega0).collect(),
        "even_only" => EVENS_10.iter().map(|&e| (e as f64) * omega0).collect(),
        "random_matched" => {
            let mut v: Vec<f64> = (0..10).map(|_| rng.uniform(fmin, fmax)).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v
        }
        _ => unreachable!(),
    }
}

/// Projects an ALREADY-TRAINED ÓRBITA inference displacement field `f_full(u)` strictly onto
/// the frequency set `freqs` (exact L2 spectral projection on a deterministic phase grid —
/// ZERO test labels, ZERO retraining!), preserving ONLY those exact frequencies.
struct ProjectedDynField {
    dirs: Vec<[f64; 4]>,
    freqs: Vec<f64>,
    wc: Vec<[f64; 2]>, // cos weights
    ws: Vec<[f64; 2]>, // sin weights
}

impl ProjectedDynField {
    fn from_trained_field(
        f_full: &dyn Fn([f64; 4]) -> [f64; 2],
        learned_k: &[[f64; 4]],
        freqs: &[f64],
        include_cos: bool,
    ) -> Self {
        let m = freqs.len();
        // Unit directions extracted from the trained inference field's own top-energy wavevectors
        let mut dirs = Vec::with_capacity(m);
        for j in 0..m {
            let k = learned_k[j % learned_k.len()];
            let nrm = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3])
                .sqrt()
                .max(1e-9);
            dirs.push([k[0] / nrm, k[1] / nrm, k[2] / nrm, k[3] / nrm]);
        }
        // Deterministic quasi-uniform 4D probe grid in [-1.2, 1.2]^2 x [-0.9, 0.9]^2 (256 points, ZERO test data!)
        let n_grid = 256usize;
        let mut grid = Vec::with_capacity(n_grid);
        let mut targets = Vec::with_capacity(n_grid);
        let halton = |mut idx: usize, base: usize| -> f64 {
            let (mut f, mut r) = (1.0, 0.0);
            while idx > 0 {
                f /= base as f64;
                r += f * (idx % base) as f64;
                idx /= base;
            }
            r
        };
        for i in 1..=n_grid {
            let u = [
                -1.2 + 2.4 * halton(i, 2),
                -1.2 + 2.4 * halton(i, 3),
                -0.9 + 1.8 * halton(i, 5),
                -0.9 + 1.8 * halton(i, 7),
            ];
            targets.push(f_full(u));
            grid.push(u);
        }

        // Exact L2 orthogonal projection: solve (Phi^T Phi + 1e-4 I) w = Phi^T Y
        let nf = if include_cos { 2 * m } else { m };
        let mut ata = vec![0.0; nf * nf];
        let mut atb0 = vec![0.0; nf];
        let mut atb1 = vec![0.0; nf];
        let mut feat = vec![0.0; nf];
        for (u, y) in grid.iter().zip(&targets) {
            for j in 0..m {
                let d = &dirs[j];
                let ang = freqs[j] * (d[0] * u[0] + d[1] * u[1] + d[2] * u[2] + d[3] * u[3]);
                if include_cos {
                    feat[2 * j] = ang.cos();
                    feat[2 * j + 1] = ang.sin();
                } else {
                    feat[j] = ang.sin();
                }
            }
            for r in 0..nf {
                for c in 0..nf {
                    ata[r * nf + c] += feat[r] * feat[c];
                }
                atb0[r] += feat[r] * y[0];
                atb1[r] += feat[r] * y[1];
            }
        }
        let w0 = solve_ridge(&ata, &atb0, nf, 1e-3);
        let w1 = solve_ridge(&ata, &atb1, nf, 1e-3);
        let mut wc = vec![[0.0, 0.0]; m];
        let mut ws = vec![[0.0, 0.0]; m];
        for j in 0..m {
            if include_cos {
                wc[j] = [w0[2 * j], w1[2 * j]];
                ws[j] = [w0[2 * j + 1], w1[2 * j + 1]];
            } else {
                ws[j] = [w0[j], w1[j]];
            }
        }
        Self {
            dirs,
            freqs: freqs.to_vec(),
            wc,
            ws,
        }
    }

    #[inline]
    fn eval(&self, u: [f64; 4]) -> [f64; 2] {
        let (mut o0, mut o1) = (0.0, 0.0);
        for j in 0..self.freqs.len() {
            let d = &self.dirs[j];
            let ang = self.freqs[j] * (d[0] * u[0] + d[1] * u[1] + d[2] * u[2] + d[3] * u[3]);
            let c = ang.cos();
            let s = ang.sin();
            o0 += self.wc[j][0] * c + self.ws[j][0] * s;
            o1 += self.wc[j][1] * c + self.ws[j][1] * s;
        }
        [o0, o1]
    }
}

/// Computes exact parity decomposition and natural frequency energy of a trained wave field on
/// a periodic fundamental ω_0 = π / 3.0 (so half-period shift is L/2 = 3.0).
/// Under half-period shift u -> u + π/ω_0:
///   cos(p ω_0 (u + π/ω_0)) = (-1)^p cos(p ω_0 u)  => ODD primes p >= 3 flip sign (-1), EVEN harmonics stay (+1)!
fn half_period_odd_fraction(
    f_slice: &dyn Fn([f64; 4]) -> [f64; 2],
    dirs: &[[f64; 4]],
    omega0: f64,
) -> f64 {
    let shift = PI / omega0;
    let mut e_odd = 0.0;
    let mut e_even = 0.0;
    for j in 0..dirs.len().min(10) {
        let d = dirs[j];
        for s_idx in 0..24 {
            let t = -shift + (2.0 * shift * s_idx as f64) / 24.0;
            let u1 = [t * d[0], t * d[1], t * d[2], t * d[3]];
            let t2 = t + shift;
            let u2 = [t2 * d[0], t2 * d[1], t2 * d[2], t2 * d[3]];
            let f1 = f_slice(u1);
            let f2 = f_slice(u2);
            let odd0 = 0.5 * (f1[0] - f2[0]);
            let odd1 = 0.5 * (f1[1] - f2[1]);
            let ev0 = 0.5 * (f1[0] + f2[0]);
            let ev1 = 0.5 * (f1[1] + f2[1]);
            e_odd += odd0 * odd0 + odd1 * odd1;
            e_even += ev0 * ev0 + ev1 * ev1;
        }
    }
    e_odd / (e_odd + e_even).max(1e-12)
}

/// Projects the ALREADY-TRAINED 3D Past x Future Collapse Wave `cav.w3d` onto an exact
/// frequency set `freqs` (cardinality M_s = 10) via L2 spectral filtering along the learned
/// wave directions (ZERO test labels, ZERO retraining!), then evaluates Past x Future Collapse.
fn project_cavity_slice(
    cav: &SpectralCavity3D,
    freqs: &[f64],
    seed: u64,
) -> SpectralCavity3D {
    let m = freqs.len();
    // Pick top-M amplitude directions from the trained 3D cavity
    let mut order: Vec<usize> = (0..cav.w3d.m).collect();
    order.sort_by(|&a, &b| cav.w3d.a[b].abs().total_cmp(&cav.w3d.a[a].abs()));
    // Construct 30 directional wavevectors: 3 top learned orientations per exact frequency in `freqs`
    let n_dirs_per_freq = 3usize;
    let total_m = m * n_dirs_per_freq;
    let mut k_vec = Vec::with_capacity(total_m);
    for (fi, &freq) in freqs.iter().enumerate() {
        for di in 0..n_dirs_per_freq {
            let src = cav.w3d.k[order[(fi * n_dirs_per_freq + di) % cav.w3d.m]];
            let nrm = (src[0] * src[0] + src[1] * src[1] + src[2] * src[2])
                .sqrt()
                .max(1e-9);
            k_vec.push([freq * src[0] / nrm, freq * src[1] / nrm, freq * src[2] / nrm]);
        }
    }

    // Exact L2 spectral projection of trained W_3D(u) onto {cos(k_j·u), sin(k_j·u)} over cavity grid
    let mut rng = Rng64::from_seed(seed, 0x04B0);
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(420);
    for s in &cav.states {
        pts.push(*s);
        for _ in 0..4 {
            pts.push([
                s[0] + 0.05 * rng.normal(),
                s[1] + 0.05 * rng.normal(),
                s[2] + 0.05 * rng.normal(),
            ]);
        }
    }
    while pts.len() < 420 {
        pts.push([
            rng.uniform(-CAVITY_LIM, CAVITY_LIM),
            rng.uniform(-CAVITY_LIM, CAVITY_LIM),
            rng.uniform(-CAVITY_LIM, CAVITY_LIM),
        ]);
    }

    let nf = 2 * total_m;
    let mut ata = vec![0.0; nf * nf];
    let mut atb = vec![0.0; nf];
    let mut feat = vec![0.0; nf];
    for &u in &pts {
        let y = cav.w3d.eval(u) - DC_3D;
        for j in 0..total_m {
            let kj = &k_vec[j];
            let ang = kj[0] * u[0] + kj[1] * u[1] + kj[2] * u[2];
            feat[2 * j] = ang.cos();
            feat[2 * j + 1] = ang.sin();
        }
        for r in 0..nf {
            for c in 0..nf {
                ata[r * nf + c] += feat[r] * feat[c];
            }
            atb[r] += feat[r] * y;
        }
    }
    let w = solve_ridge(&ata, &atb, nf, 0.05);
    let mut a_out = Vec::with_capacity(total_m);
    let mut phi_out = Vec::with_capacity(total_m);
    for j in 0..total_m {
        let wc = w[2 * j];
        let ws = w[2 * j + 1];
        a_out.push((wc * wc + ws * ws).sqrt());
        phi_out.push((-ws).atan2(wc));
    }
    let mut out = cav.clone();
    out.w3d = CavWave3D {
        m: total_m,
        mode: cav.mode,
        k: k_vec,
        a: a_out,
        phi: phi_out,
    };
    out
}

fn run_suite_o4(
    seed: u64,
    asc_o1: &OrbitaWithWave,
    asc_new_experts: &[SpectralWaveNet2],
    held_all12: &[C4Ep],
    asc_cav: &SpectralCavity3D,
    protos: &[[f64; DIM_SIG]; K_RULES_O2],
    unseen_pairs: &[(usize, usize)],
    export_trace_dir: Option<&Path>,
) -> Vec<O4Row> {
    let zeros = compute_riemann_zeros(24);
    let mut rows = Vec::new();
    let omega0_dyn = PI / 16.0; // maps [2..29] onto smooth dynamical wavevectors [0.39..5.69]
    let omega0_cav = 2.0 * PI / 34.0; // maps [2..29] onto trained 3D cavity bandwidth [0.37..5.36]
    let slices = [
        "full_adaptive",
        "prime_harmonic_exact",
        "riemann_prime_dual",
        "log_prime_exact",
        "composite_only",
        "even_only",
        "random_matched",
    ];

    // Helper: full trained ÓRBITA_ASC 1-step displacement f_full(fam, u) = step(u) - q
    let eval_full_disp = |fam: usize, u: [f64; 4]| -> [f64; 2] {
        if fam < N_FAM {
            let c = context_of(fam, [u[2], u[3]]);
            let mut x12 = [0.0; 12];
            x12[0] = u[0];
            x12[1] = u[1];
            let y12 = asc_o1.step(&x12, &c);
            [y12[0] - u[0], y12[1] - u[1]]
        } else {
            asc_new_experts[fam - N_FAM].eval(u)
        }
    };

    // Baseline full energy per family on deterministic grid
    let mut full_energies = [0.0_f64; 12];
    for f in 0..12 {
        let mut e = 0.0;
        for i in 0..64 {
            let u = [
                -1.0 + 2.0 * ((i % 8) as f64) / 7.0,
                -1.0 + 2.0 * ((i / 8) as f64) / 7.0,
                0.5 * ((i as f64 * 0.7).sin()),
                0.5 * ((i as f64 * 1.1).cos()),
            ];
            let d = eval_full_disp(f, u);
            e += d[0] * d[0] + d[1] * d[1];
        }
        full_energies[f] = e.max(1e-9);
    }

    // Optional detailed trace on 0xB400 showing concrete hidden prime-filtered trajectories
    let mut trace_lines: Vec<String> = Vec::new();
    if export_trace_dir.is_some() {
        trace_lines.push("slice,fam,fam_type,p1,p2,x0,y0,true_x1,true_y1,pred_x1,pred_y1,rel_err,disp_cos,odd_parity".into());
    }

    // ── O4a: Prime-Frequency Extraction on ÓRBITA's Trained Dynamical Inference (all 12 families) ──
    for &sl in &slices {
        let mut srng = Rng64::from_seed(seed, 0x04A0);
        let projected: Option<Vec<ProjectedDynField>> = if sl == "full_adaptive" {
            None
        } else {
            let freqs = slice_frequencies(sl, &zeros, omega0_dyn, &mut srng);
            let mut v = Vec::with_capacity(12);
            for f in 0..12 {
                let lk: &[[f64; 4]] = if f < N_FAM {
                    &asc_o1.fields[f].k
                } else {
                    &asc_new_experts[f - N_FAM].w1
                };
                v.push(ProjectedDynField::from_trained_field(
                    &|u| eval_full_disp(f, u),
                    lk,
                    &freqs,
                    true,
                ));
            }
            Some(v)
        };

        let (mut ok1_s, mut ok16_s, mut rel1_s, mut cos_s, mut man_s, mut cnt) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut irred_r, mut irred_n, mut comp_r, mut comp_n) = (0.0, 0.0, 0.0, 0.0);

        for ep in held_all12 {
            let f = ep.fam;
            let step_ep = |x: &[f64]| -> Vec<f64> {
                match &projected {
                    None => {
                        if f < N_FAM {
                            let c = context_of(f, ep.p);
                            asc_o1.step(x, &c)
                        } else {
                            step_net2_expert(&asc_new_experts[f - N_FAM], x, ep.p)
                        }
                    }
                    Some(projs) => {
                        let pf = &projs[f];
                        let mut y = x.to_vec();
                        for i in 0..x.len() / 2 {
                            let dq = pf.eval([x[2 * i], x[2 * i + 1], ep.p[0], ep.p[1]]);
                            y[2 * i] += dq[0];
                            y[2 * i + 1] += dq[1];
                        }
                        y
                    }
                }
            };
            let y1_pred = step_ep(&ep.x0);
            let fin1 = y1_pred.iter().all(|v| v.is_finite());
            let r1 = if fin1 {
                m::rel_err(&y1_pred, &ep.ys[0]).min(10.0)
            } else {
                10.0
            };
            let c1 = if fin1 {
                m::disp_cosine(&ep.x0, &y1_pred, &ep.ys[0])
            } else {
                0.0
            };
            let md1 = if fin1 {
                m::manifold_distance(&ep.x0, &y1_pred)
            } else {
                10.0
            };
            let (_, _, ok16, _) = eval_rollout_c4(&step_ep, &ep.x0, 0, ep);
            ok1_s += (fin1 && m::correct(&y1_pred, &ep.ys[0])) as u8 as f64;
            ok16_s += ok16 as f64;
            rel1_s += r1;
            cos_s += c1;
            man_s += md1;
            cnt += 1.0;

            // Irreducible elementary generators (0:translate, 1:rotate, 2:reflect, 3:scale) vs composite/nonlinear (4..11)
            if f <= 3 {
                irred_r += r1;
                irred_n += 1.0;
            } else {
                comp_r += r1;
                comp_n += 1.0;
            }
        }

        // Measure exact half-period odd-parity antisymmetry & energy retention
        let mut odd_sum = 0.0;
        let mut eret_sum = 0.0;
        for f in 0..12 {
            let lk: &[[f64; 4]] = if f < N_FAM {
                &asc_o1.fields[f].k
            } else {
                &asc_new_experts[f - N_FAM].w1
            };
            let dirs: Vec<[f64; 4]> = lk
                .iter()
                .take(10)
                .map(|k| {
                    let n = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3])
                        .sqrt()
                        .max(1e-9);
                    [k[0] / n, k[1] / n, k[2] / n, k[3] / n]
                })
                .collect();
            let odd_f = match &projected {
                None => half_period_odd_fraction(&|u| eval_full_disp(f, u), &dirs, omega0_dyn),
                Some(projs) => {
                    // Exact spectral half-period odd-harmonic fraction per mode:
                    // sin^2(π * freq / (2 * omega0)) == 1 for odd integers (all primes p >= 3), 0 for even integers!
                    let pf = &projs[f];
                    let mut e_odd = 0.0;
                    let mut e_tot = 0.0;
                    for j in 0..pf.freqs.len() {
                        let pwr = pf.wc[j][0].powi(2)
                            + pf.wc[j][1].powi(2)
                            + pf.ws[j][0].powi(2)
                            + pf.ws[j][1].powi(2);
                        let s = (0.5 * PI * pf.freqs[j] / omega0_dyn).sin();
                        e_odd += pwr * s * s;
                        e_tot += pwr;
                    }
                    e_odd / e_tot.max(1e-12)
                }
            };
            odd_sum += odd_f;

            let mut e_sl = 0.0;
            for i in 0..64 {
                let u = [
                    -1.0 + 2.0 * ((i % 8) as f64) / 7.0,
                    -1.0 + 2.0 * ((i / 8) as f64) / 7.0,
                    0.5 * ((i as f64 * 0.7).sin()),
                    0.5 * ((i as f64 * 1.1).cos()),
                ];
                let d = match &projected {
                    None => eval_full_disp(f, u),
                    Some(projs) => projs[f].eval(u),
                };
                e_sl += d[0] * d[0] + d[1] * d[1];
            }
            eret_sum += (e_sl / full_energies[f]).min(2.0);

            if export_trace_dir.is_some() {
                if let Some(ep) = held_all12.iter().find(|e| e.fam == f) {
                    let y1 = match &projected {
                        None => {
                            if f < N_FAM {
                                asc_o1.step(&ep.x0, &context_of(f, ep.p))
                            } else {
                                step_net2_expert(&asc_new_experts[f - N_FAM], &ep.x0, ep.p)
                            }
                        }
                        Some(projs) => {
                            let dq = projs[f].eval([ep.x0[0], ep.x0[1], ep.p[0], ep.p[1]]);
                            let mut v = ep.x0.clone();
                            v[0] += dq[0];
                            v[1] += dq[1];
                            for i in 1..ep.x0.len() / 2 {
                                let dqi = projs[f].eval([ep.x0[2 * i], ep.x0[2 * i + 1], ep.p[0], ep.p[1]]);
                                v[2 * i] += dqi[0];
                                v[2 * i + 1] += dqi[1];
                            }
                            v
                        }
                    };
                    let re = m::rel_err(&y1, &ep.ys[0]).min(10.0);
                    let dc = m::disp_cosine(&ep.x0, &y1, &ep.ys[0]);
                    let ft = if f <= 3 { "irreducible" } else { "composite_nl" };
                    trace_lines.push(format!(
                        "{sl},{f},{ft},{:.4},{:.4},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{re:.6},{dc:.6},{odd_f:.6}",
                        ep.p[0], ep.p[1], ep.x0[0], ep.x0[1], ep.ys[0][0], ep.ys[0][1], y1[0], y1[1]
                    ));
                }
            }
        }

        rows.push(O4Row {
            subtask: "O4a_dynamics".into(),
            slice: sl.into(),
            n_modes: if sl == "full_adaptive" { M_DYN } else { 10 },
            acc1: ok1_s / cnt,
            acc16_or_pair: ok16_s / cnt,
            rel1_or_ghost: rel1_s / cnt,
            disp_cos_or_psr: cos_s / cnt,
            manifold_dist: man_s / cnt,
            odd_parity_frac: odd_sum / 12.0,
            irred_rel1: irred_r / irred_n,
            comp_rel1: comp_r / comp_n,
            energy_retained: eret_sum / 12.0,
        });
    }

    // ── O4b: Prime-Frequency Extraction on Trained Past x Future Wave Collapse Cavity ──
    for &sl in &slices {
        let mut srng = Rng64::from_seed(seed, 0x04B5);
        let cav_sl = if sl == "full_adaptive" {
            asc_cav.clone()
        } else {
            let freqs = slice_frequencies(sl, &zeros, omega0_cav, &mut srng);
            project_cavity_slice(asc_cav, &freqs, seed ^ 0x4444)
        };
        let mut erng = Rng64::from_seed(seed, 0x02E0);
        let mut hits = 0usize;
        let mut psr_s = 0.0;
        let mut ghost_s = 0.0;
        for &(a, b) in unseen_pairs {
            let x = compose_sig(protos, a, b, 0.025, &mut erng);
            let (pa, pb, psr, ghost) = cav_sl.collapse_pair(&x);
            if pa == a && pb == b {
                hits += 1;
            }
            psr_s += psr;
            ghost_s += ghost;
        }
        // Cavity wave energy retained vs full
        let mut e_sl = 0.0;
        let mut e_fu = 0.0;
        for s in &asc_cav.states {
            let v1 = cav_sl.w3d.eval(*s);
            let v0 = asc_cav.w3d.eval(*s);
            e_sl += v1 * v1;
            e_fu += v0 * v0;
        }
        rows.push(O4Row {
            subtask: "O4b_collapse".into(),
            slice: sl.into(),
            n_modes: if sl == "full_adaptive" { M3D_O2 } else { 10 },
            acc1: 0.0,
            acc16_or_pair: hits as f64 / unseen_pairs.len() as f64,
            rel1_or_ghost: ghost_s / unseen_pairs.len() as f64,
            disp_cos_or_psr: psr_s / unseen_pairs.len() as f64,
            manifold_dist: 0.0,
            odd_parity_frac: 0.0,
            irred_rel1: 0.0,
            comp_rel1: 0.0,
            energy_retained: (e_sl / e_fu.max(1e-9)).min(2.0),
        });
    }

    if let Some(dir) = export_trace_dir {
        fs::write(dir.join("trace_prime_sieve.csv"), trace_lines.join("\n") + "\n").unwrap();
    }

    rows
}

// ─── CLI Entrypoints (`orbita asc-run` & `orbita asc-fresh`) ────────────────

fn arg_val(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

fn parse_seed(s: &str) -> u64 {
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).unwrap()
    } else {
        s.parse().unwrap()
    }
}

/// Independent fresh-process verification for `ORBITA_ASC` on sealed §16 `TEST` (0 queries, exact acc parity).
pub fn fresh(a: &[String]) {
    let seed = parse_seed(&arg_val(a, "--seed").expect("--seed"));
    let wpath = arg_val(a, "--weights").expect("--weights");
    let test_sha = arg_val(a, "--test-sha").expect("--test-sha");
    let base = Orbita::from_json(&fs::read_to_string(&wpath).unwrap());
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
    if s.manifest.test_sha256 != test_sha {
        println!("{{\"error\":\"test_sha_mismatch\"}}");
        return;
    }
    let per_fam = collect_train_residuals(&base, &s.ds.train);
    let (asc, _, _, _) = train_o1_wave(&base, &per_fam, SpecMode::Adaptive, None, seed);
    let before = snapshot();
    let acc = evaluate(&asc, &s.ds.test, 1, None, None).acc;
    let q = snapshot().delta(&before).total();
    println!("{{\"acc\":{acc:.12},\"queries_total\":{q},\"pid\":{}}}", std::process::id());
}

/// Full 4-suite seed runner: `orbita asc-run --seed S --phase P --weights W --meta M --out DIR [--export-traces]`
pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_val(a, "--seed").expect("--seed"));
    let phase = arg_val(a, "--phase").unwrap_or_else(|| "DEV".into());
    let wpath = arg_val(a, "--weights").expect("--weights");
    let mpath = arg_val(a, "--meta").expect("--meta");
    let out_dir_str = arg_val(a, "--out").expect("--out");
    let export_traces = a.iter().any(|x| x == "--export-traces");
    let out_dir = Path::new(&out_dir_str);
    fs::create_dir_all(out_dir).unwrap();

    let wtxt = fs::read_to_string(&wpath).expect("read orbita weights");
    let base = Orbita::from_json(&wtxt);
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&mpath).expect("read orbita meta")).unwrap();

    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
    assert_eq!(
        meta["test_sha256"].as_str().unwrap(),
        s.manifest.test_sha256,
        "TEST SHA-256 mismatch vs published ÓRBITA meta!"
    );
    s.verify_test().expect("TEST sealed");
    let aud = Auditor::new(&s);
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    let mut flags = AuditFlags {
        cdt_canon: cdt.canonical_targets(),
        ..Default::default()
    };
    for h in meta["replay_equiv_hits"].as_array().unwrap() {
        flags.equiv_hits.insert(h.as_u64().unwrap());
    }
    flags.any_equiv_global = meta["replay_any_equiv_global"].as_bool().unwrap();
    drop(cdt);

    let iter_ex: Vec<&Example> = s
        .ds
        .test
        .iter()
        .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
        .collect();
    let mut drng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0xD33);
    let deltas: Vec<Vec<f64>> = iter_ex
        .iter()
        .map(|e| {
            let v: Vec<f64> = e
                .input
                .iter()
                .map(|_| rand::Rng::gen_range(&mut drng, -1.0..1.0))
                .collect();
            let nv = m::norm(&v).max(1e-12);
            v.iter()
                .map(|x| x / nv * 0.01 * m::norm(&e.input))
                .collect()
        })
        .collect();

    let zeros = compute_riemann_zeros(M3D_O2);

    // ── Suite O1: Official §16 TEST + E33 ──
    let per_fam = collect_train_residuals(&base, &s.ds.train);
    let mut o1_rows = Vec::new();
    o1_rows.push(eval_o1_arm(
        "ORBITA_BASE",
        &base,
        &s,
        &aud,
        &flags,
        &iter_ex,
        &deltas,
        0.0,
        0.0,
        0.0,
    ));

    let (asc_o1, ms_asc, ir_asc, lr_asc) =
        train_o1_wave(&base, &per_fam, SpecMode::Adaptive, None, seed);
    o1_rows.push(eval_o1_arm(
        "ORBITA_ASC",
        &asc_o1,
        &s,
        &aud,
        &flags,
        &iter_ex,
        &deltas,
        ms_asc,
        ir_asc,
        lr_asc,
    ));

    for mode in [
        SpecMode::RiemannFixed,
        SpecMode::FourierFixed,
        SpecMode::PoissonFixed,
    ] {
        let z_opt = if mode == SpecMode::RiemannFixed {
            Some(zeros.as_slice())
        } else {
            None
        };
        let (mdl, ms, ir, lr) = train_o1_wave(&base, &per_fam, mode, z_opt, seed);
        o1_rows.push(eval_o1_arm(
            mode.name(),
            &mdl,
            &s,
            &aud,
            &flags,
            &iter_ex,
            &deltas,
            ms,
            ir,
            lr,
        ));
    }

    // Matched-capacity MLP residual (exact 1:1 params = 3,514!)
    {
        let t0 = Instant::now();
        let mut mrng = Rng64::from_seed(seed, 0x01F0);
        let mut mlps = Vec::with_capacity(N_FAM);
        for f in 0..N_FAM {
            let mut ml = MlpRes4D::new(MLP_H_O1, &mut mrng);
            ml.fit(&per_fam[f], 1100, 0.018, 5.0e-4);
            mlps.push(ml);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let mdl = OrbitaWithMlp {
            base: base.clone(),
            mlps,
        };
        o1_rows.push(eval_o1_arm(
            "ORBITA_MLP_RES",
            &mdl,
            &s,
            &aud,
            &flags,
            &iter_ex,
            &deltas,
            ms,
            0.0,
            0.0,
        ));
    }

    // Fresh subprocess check for ORBITA_ASC on §16 TEST
    let exe = std::env::current_exe().unwrap();
    let fresh_out = std::process::Command::new(exe)
        .args([
            "asc-fresh",
            "--seed",
            &format!("0x{seed:X}"),
            "--weights",
            &wpath,
            "--test-sha",
            &s.manifest.test_sha256,
        ])
        .output()
        .expect("run asc-fresh");
    let fv: serde_json::Value = serde_json::from_str(
        String::from_utf8_lossy(&fresh_out.stdout)
            .lines()
            .last()
            .unwrap_or("{}"),
    )
    .unwrap_or_default();
    let fresh_acc = fv["acc"].as_f64().unwrap_or(f64::NAN);
    let fresh_q = fv["queries_total"].as_u64().unwrap_or(999);
    let asc_row_acc = o1_rows[1].acc1;
    let fresh_ok = (fresh_acc - asc_row_acc).abs() < 1e-12 && fresh_q == 0;
    assert!(
        fresh_ok,
        "Fresh process check failed: fresh_acc={fresh_acc} vs {asc_row_acc}, q={fresh_q}"
    );

    // ── Suite O2: Unknown-Context & Compositional Rule-Pair Collapse ──
    let (o2_rows, asc_cav, protos, unseen_pairs) = run_suite_o2(seed, &zeros);

    // ── Suite O3: Continual Few-Shot Learning of ÓRBITA's 4 New Nonlinear Families 8..11 ──
    let (o3_rows, asc_new_experts, held_all12) = run_suite_o3(seed, &base, &asc_o1, &zeros);

    // ── Suite O4: Prime-Frequency Spectral Sieve ("¿Qué inferencia se esconde?") ──
    let trace_dir = if export_traces { Some(out_dir) } else { None };
    let o4_rows = run_suite_o4(
        seed,
        &asc_o1,
        &asc_new_experts,
        &held_all12,
        &asc_cav,
        &protos,
        &unseen_pairs,
        trace_dir,
    );

    // Verify frozen ÓRBITA weights untouched!
    let base_after = Orbita::from_json(&wtxt);
    assert_eq!(base_after.raw, base.raw, "Frozen ÓRBITA weights mutated!");

    // Write deterministic CSV outputs
    {
        let mut f = File::create(out_dir.join("metrics_o1.csv")).unwrap();
        writeln!(f, "seed,phase,arm,params,train_ms,acc1,rel1,acc_indist,acc_hole,rel_refl,rel_rad,h1,h4,h8,h16,h32,h64,region,clean,gate_pass,fresh_ok,init_r,learned_r").unwrap();
        for r in &o1_rows {
            writeln!(
                f,
                "0x{seed:X},{phase},{},{},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.6},{:.6}",
                r.arm, r.params, r.train_ms, r.acc1, r.rel1, r.acc_indist, r.acc_hole, r.rel_refl, r.rel_rad,
                r.h1, r.h4, r.h8, r.h16, r.h32, r.h64, r.region, r.clean as u8, r.gate_pass as u8, fresh_ok as u8, r.init_r, r.learned_r
            ).unwrap();
        }
    }
    {
        let mut f = File::create(out_dir.join("metrics_o2.csv")).unwrap();
        writeln!(f, "seed,phase,arm,params,train_ms,unseen_pair_acc,train_pair_acc,cfr,init_r,learned_r,psr").unwrap();
        for r in &o2_rows {
            writeln!(
                f,
                "0x{seed:X},{phase},{},{},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                r.arm, r.params, r.train_ms, r.unseen_pair_acc, r.train_pair_acc, r.cfr, r.init_r, r.learned_r, r.psr
            ).unwrap();
        }
    }
    {
        let mut f = File::create(out_dir.join("metrics_o3.csv")).unwrap();
        writeln!(f, "seed,phase,arm,expert_params,adapt_ms,known_h8,known_h16,new_h1,new_h8,new_h16,new_rel16,route_purity").unwrap();
        for r in &o3_rows {
            writeln!(
                f,
                "0x{seed:X},{phase},{},{},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                r.arm, r.expert_params, r.adapt_ms, r.known_h8, r.known_h16, r.new_h1, r.new_h8, r.new_h16, r.new_rel16, r.route_purity
            ).unwrap();
        }
    }
    {
        let mut f = File::create(out_dir.join("metrics_o4.csv")).unwrap();
        writeln!(f, "seed,phase,subtask,slice,n_modes,acc1,acc16_or_pair,rel1_or_ghost,disp_cos_or_psr,manifold_dist,odd_parity_frac,irred_rel1,comp_rel1,energy_retained").unwrap();
        for r in &o4_rows {
            writeln!(
                f,
                "0x{seed:X},{phase},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                r.subtask, r.slice, r.n_modes, r.acc1, r.acc16_or_pair, r.rel1_or_ghost,
                r.disp_cos_or_psr, r.manifold_dist, r.odd_parity_frac, r.irred_rel1, r.comp_rel1, r.energy_retained
            ).unwrap();
        }
    }

    println!(
        "0x{seed:X} {phase} done | O1 base_h16={:.3} asc_h16={:.3} (rel1 {:.4}->{:.4}) | O2 asc={:.3} mlp={:.3} | O3 lat_new16={:.3} asc_new16={:.3} | fresh_ok={fresh_ok}",
        o1_rows[0].h16,
        o1_rows[1].h16,
        o1_rows[0].rel1,
        o1_rows[1].rel1,
        o2_rows[0].unseen_pair_acc,
        o2_rows[4].unseen_pair_acc,
        o3_rows[2].new_h16,
        o3_rows[3].new_h16,
    );
}

// ─── Unit Tests (Analytical Gradients < 1e-5, Riemann Zeros < 1e-9, Zero-Init Parity) ─

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riemann_zeros_accuracy() {
        let z = compute_riemann_zeros(5);
        let refs = [
            14.134725141734693,
            21.022039638771555,
            25.010857580145688,
            30.424876125859513,
            32.935061587739189,
        ];
        for i in 0..5 {
            assert!(
                (z[i] - refs[i]).abs() < 1e-9,
                "Zero #{i} mismatch: got {} vs {}",
                z[i],
                refs[i]
            );
        }
    }

    #[test]
    fn test_wave_field_4d_grad_finite_diff() {
        let mut rng = Rng64::from_seed(42, 1);
        let mut wf = WaveField4D::new(6, SpecMode::Adaptive, None, 0.4, 2.0, &mut rng);
        for m in 0..6 {
            wf.a[m] = [0.2 * rng.normal(), 0.2 * rng.normal()];
        }
        wf.bias = [0.1, -0.05];
        let u = [0.6, -0.4, 0.35, -0.7];
        let tgt = [0.25, -0.15];
        let loss = |w: &WaveField4D| {
            let p = w.eval(u);
            (p[0] - tgt[0]).powi(2) + (p[1] - tgt[1]).powi(2)
        };
        let p0 = wf.eval(u);
        let mut grads = vec![0.0; wf.n_params()];
        wf.accum_grad(u, [2.0 * (p0[0] - tgt[0]), 2.0 * (p0[1] - tgt[1])], &mut grads);

        let base_p = wf.pack();
        let eps = 1e-6;
        for i in 0..base_p.len() {
            let mut pp = base_p.clone();
            pp[i] += eps;
            let mut w1 = wf.clone();
            w1.unpack(&pp);
            let mut pm = base_p.clone();
            pm[i] -= eps;
            let mut w2 = wf.clone();
            w2.unpack(&pm);
            let fd = (loss(&w1) - loss(&w2)) / (2.0 * eps);
            assert!(
                (grads[i] - fd).abs() < 1e-5,
                "WaveField4D grad[{i}] mismatch: ana={} fd={}",
                grads[i],
                fd
            );
        }
    }

    #[test]
    fn test_mlp_res_4d_grad_finite_diff() {
        let mut rng = Rng64::from_seed(99, 2);
        let mut ml = MlpRes4D::new(6, &mut rng);
        for i in 0..6 {
            ml.w2[i] = [0.2 * rng.normal(), 0.2 * rng.normal()];
        }
        let u = [-0.5, 0.8, -0.2, 0.6];
        let tgt = [-0.3, 0.4];
        let loss = |m: &MlpRes4D| {
            let p = m.eval(u);
            (p[0] - tgt[0]).powi(2) + (p[1] - tgt[1]).powi(2)
        };
        let p0 = ml.eval(u);
        let mut grads = vec![0.0; ml.n_params()];
        ml.accum_grad(u, [2.0 * (p0[0] - tgt[0]), 2.0 * (p0[1] - tgt[1])], &mut grads);

        let base_p = ml.pack();
        let eps = 1e-6;
        for i in 0..base_p.len() {
            let mut pp = base_p.clone();
            pp[i] += eps;
            let mut m1 = ml.clone();
            m1.unpack(&pp);
            let mut pm = base_p.clone();
            pm[i] -= eps;
            let mut m2 = ml.clone();
            m2.unpack(&pm);
            let fd = (loss(&m1) - loss(&m2)) / (2.0 * eps);
            assert!(
                (grads[i] - fd).abs() < 1e-5,
                "MlpRes4D grad[{i}] mismatch: ana={} fd={}",
                grads[i],
                fd
            );
        }
    }

    #[test]
    fn test_spectral_wave_net2_grad_finite_diff() {
        for is_mlp in [false, true] {
            let mut rng = Rng64::from_seed(77, is_mlp as u64 + 1);
            let mut net = SpectralWaveNet2::new(SpecMode::Adaptive, is_mlp, None, &mut rng);
            for d in 0..2 {
                for i in 0..O3_H2 {
                    net.w3[d][i] = 0.15 * rng.normal();
                }
            }
            net.b3 = [0.05, -0.07];
            let u = [0.45, -0.62, 0.31, -0.58];
            let tgt = [0.18, -0.24];
            let loss = |n: &SpectralWaveNet2| {
                let p = n.eval(u);
                (p[0] - tgt[0]).powi(2) + (p[1] - tgt[1]).powi(2)
            };
            let p0 = net.eval(u);
            let mut grads = vec![0.0; net.n_params()];
            net.accum_grad(u, [2.0 * (p0[0] - tgt[0]), 2.0 * (p0[1] - tgt[1])], &mut grads);

            let base_p = net.pack();
            let eps = 1e-6;
            for i in 0..base_p.len() {
                let mut pp = base_p.clone();
                pp[i] += eps;
                let mut n1 = net.clone();
                n1.unpack(&pp);
                let mut pm = base_p.clone();
                pm[i] -= eps;
                let mut n2 = net.clone();
                n2.unpack(&pm);
                let fd = (loss(&n1) - loss(&n2)) / (2.0 * eps);
                assert!(
                    (grads[i] - fd).abs() < 1e-5,
                    "SpectralWaveNet2(mlp={is_mlp}) grad[{i}] mismatch: ana={} fd={}",
                    grads[i],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_zero_init_exact_parity_and_param_counts() {
        let mut rng = Rng64::from_seed(7, 3);
        let wf = WaveField4D::new(M_DYN, SpecMode::Adaptive, None, 0.4, 2.0, &mut rng);
        let ml = MlpRes4D::new(MLP_H_O1, &mut rng);
        assert_eq!(wf.n_params(), ml.n_params());
        assert_eq!(wf.n_params(), 114);
        assert_eq!(wf.eval([0.5, -0.7, 0.3, -0.2]), [0.0, 0.0]);
        assert_eq!(ml.eval([0.5, -0.7, 0.3, -0.2]), [0.0, 0.0]);

        let sn_w = SpectralWaveNet2::new(SpecMode::Adaptive, false, None, &mut rng);
        let sn_m = SpectralWaveNet2::new(SpecMode::Adaptive, true, None, &mut rng);
        assert_eq!(sn_w.n_params(), sn_m.n_params());
        assert_eq!(sn_w.n_params(), 1282);
        assert_eq!(sn_w.eval([0.5, -0.7, 0.3, -0.2]), [0.0, 0.0]);
        assert_eq!(sn_m.eval([0.5, -0.7, 0.3, -0.2]), [0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "AdaptiveSpectral must NEVER receive Riemann zeros")]
    fn test_adaptive_rejects_riemann_zeros() {
        let mut rng = Rng64::from_seed(1, 1);
        let zs = vec![14.1347, 21.0220];
        let _ = WaveField4D::new(2, SpecMode::Adaptive, Some(&zs), 0.4, 2.0, &mut rng);
    }
}
