//! Spectral frequency generators for all 6 spectral arms:
//! - Riemann zeta zeros (Euler-Maclaurin Hardy Z(t) + Brent root-finder)
//! - Fourier integer harmonics
//! - Dyson GUE (beta = 2) repelled levels
//! - Poisson i.i.d. uniform random frequencies
//! - QMC low-discrepancy Halton sequence (bases 2, 3, 5)
//! Plus emergent spectral spacing ratio `<r>` diagnostics.

use std::f64::consts::PI;

use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    #[inline]
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline]
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    #[inline]
    fn norm(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
    #[inline]
    fn ln(self) -> Self {
        Self::new(self.norm().ln(), self.im.atan2(self.re))
    }
    #[inline]
    fn exp_i(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }
}

const BERNOULLI_2K: [f64; 8] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
];

const FACT_2K: [f64; 8] = [
    2.0,
    24.0,
    720.0,
    40_320.0,
    3_628_800.0,
    479_001_600.0,
    87_178_291_200.0,
    20_922_789_888_000.0,
];

fn ln_gamma(mut z: C64) -> C64 {
    let mut shift = C64::new(0.0, 0.0);
    while z.re < 12.0 {
        shift = shift.add(z.ln());
        z.re += 1.0;
    }
    let ln_z = z.ln();
    let half_ln_2pi = 0.5 * (2.0 * PI).ln();
    let mut res = z
        .sub(C64::new(0.5, 0.0))
        .mul(ln_z)
        .sub(z)
        .add(C64::new(half_ln_2pi, 0.0));
    let inv_z = C64::new(1.0, 0.0).div(z);
    let inv_z2 = inv_z.mul(inv_z);
    let mut pow = inv_z;
    for k in 0..8 {
        let two_k = (2 * (k + 1)) as f64;
        let coeff = BERNOULLI_2K[k] / (two_k * (two_k - 1.0));
        res = res.add(pow.scale(coeff));
        pow = pow.mul(inv_z2);
    }
    res.sub(shift)
}

fn hardy_z(t: f64) -> f64 {
    let s = C64::new(0.5, t);
    let n_cut = ((1.35 * t.abs()).ceil() as usize).max(32);
    let mut sum = C64::new(0.0, 0.0);
    for n in 1..n_cut {
        let nf = n as f64;
        let mag = 1.0 / nf.sqrt();
        let angle = -t * nf.ln();
        sum.re += mag * angle.cos();
        sum.im += mag * angle.sin();
    }
    let nf = n_cut as f64;
    let angle = -t * nf.ln();
    let n_neg_s = C64::new(angle.cos() / nf.sqrt(), angle.sin() / nf.sqrt());
    let integral = n_neg_s.scale(nf).div(s.sub(C64::new(1.0, 0.0)));
    sum = sum.add(integral).add(n_neg_s.scale(0.5));
    let inv_n = 1.0 / nf;
    let inv_n2 = inv_n * inv_n;
    let mut n_pow = n_neg_s.scale(inv_n);
    let mut rising = s;
    for k in 0..8 {
        let coeff = BERNOULLI_2K[k] / FACT_2K[k];
        sum = sum.add(rising.mul(n_pow).scale(coeff));
        let j1 = (2 * k + 1) as f64;
        let j2 = (2 * k + 2) as f64;
        rising = rising
            .mul(C64::new(s.re + j1, s.im))
            .mul(C64::new(s.re + j2, s.im));
        n_pow = n_pow.scale(inv_n2);
    }
    let theta = ln_gamma(C64::new(0.25, 0.5 * t)).im - 0.5 * t * PI.ln();
    C64::exp_i(theta).mul(sum).re
}

fn brent_root<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let mut fa = f(a);
    let mut fb = f(b);
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    for _ in 0..100 {
        if fb == 0.0 || (b - a).abs() <= tol {
            return b;
        }
        let d = if fa != fc && fb != fc {
            let s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
            if (s - (3.0 * a + b) * 0.25) * (s - b) >= 0.0
                || (s - b).abs() >= 0.5 * (b - c).abs()
                || (b - c).abs() < tol
            {
                0.5 * (a - b)
            } else {
                s - b
            }
        } else {
            let s = b - fb * (b - a) / (fb - fa);
            if (s - (3.0 * a + b) * 0.25) * (s - b) >= 0.0 {
                0.5 * (a - b)
            } else {
                s - b
            }
        };
        c = b;
        fc = fb;
        b += if d.abs() > tol * 0.5 {
            d
        } else if a > b {
            tol * 0.5
        } else {
            -tol * 0.5
        };
        fb = f(b);
        if (fb > 0.0 && fa > 0.0) || (fb < 0.0 && fa < 0.0) {
            a = c;
            fa = fc;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    b
}

/// Computes the first `n_zeros` exact non-trivial zeros `gamma_n` of `zeta(1/2 + i*gamma)`.
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

/// Dyson GUE (beta = 2) repelled frequencies mapped onto `[f_min, f_max]`.
pub fn dyson_gue_frequencies(n_modes: usize, f_min: f64, f_max: f64, rng: &mut Rng) -> Vec<f64> {
    let radius = 2.0 * (n_modes as f64).sqrt();
    let mut w: Vec<f64> = (0..n_modes)
        .map(|_| rng.uniform(-0.85 * radius, 0.85 * radius))
        .collect();
    let dt: f64 = 0.002;
    let eps2: f64 = 0.03 * 0.03;
    let noise_s = dt.sqrt();
    let mut next_w = vec![0.0; n_modes];
    for _ in 0..900 {
        for i in 0..n_modes {
            let wi = w[i];
            let mut rep = 0.0;
            for j in 0..n_modes {
                if i == j {
                    continue;
                }
                let d = wi - w[j];
                rep += d / (d * d + eps2);
            }
            next_w[i] = wi + (-0.5 * wi + rep) * dt + noise_s * rng.normal();
        }
        std::mem::swap(&mut w, &mut next_w);
    }
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    w.into_iter()
        .map(|val| {
            let z = (val / radius).clamp(-0.9999, 0.9999);
            let cdf = 0.5 + (1.0 / PI) * (z * (1.0 - z * z).sqrt() + z.asin());
            f_min + (f_max - f_min) * cdf
        })
        .collect()
}

/// Radical inverse for Quasi-Monte Carlo Halton low-discrepancy sequence.
pub fn halton(mut index: usize, base: usize) -> f64 {
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    let bf = base as f64;
    while index > 0 {
        f /= bf;
        r += f * ((index % base) as f64);
        index /= base;
    }
    r
}

/// Generates QMC Halton frequencies mapped onto `[f_min, f_max]`.
pub fn qmc_halton_frequencies(n_modes: usize, base: usize, f_min: f64, f_max: f64) -> Vec<f64> {
    let mut out: Vec<f64> = (1..=n_modes)
        .map(|i| f_min + (f_max - f_min) * halton(i, base))
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

/// Generates regular integer Fourier harmonics mapped onto `[f_min, f_max]` on period `L = 4.2`.
pub fn fourier_harmonics(n_modes: usize, period: f64) -> Vec<f64> {
    (1..=n_modes)
        .map(|m| 2.0 * PI * (m as f64) / period)
        .collect()
}

/// Generates Poisson i.i.d. uniform random frequencies in `[f_min, f_max]`.
pub fn poisson_frequencies(n_modes: usize, f_min: f64, f_max: f64, rng: &mut Rng) -> Vec<f64> {
    let mut out: Vec<f64> = (0..n_modes).map(|_| rng.uniform(f_min, f_max)).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

/// Rescales the first `n_modes` Riemann zeros onto `[f_min, f_max]`.
pub fn scaled_riemann_frequencies(zeros: &[f64], n_modes: usize, f_min: f64, f_max: f64) -> Vec<f64> {
    assert!(zeros.len() >= n_modes);
    let z0 = zeros[0];
    let z1 = zeros[n_modes - 1];
    let span = (z1 - z0).max(1e-9);
    (0..n_modes)
        .map(|i| f_min + (f_max - f_min) * (zeros[i] - z0) / span)
        .collect()
}

/// Computes the nearest-neighbor level-spacing ratio `<r> = mean(min(s_i, s_{i+1}) / max(s_i, s_{i+1}))`.
/// Theoretical benchmarks: Poisson = 0.3863, GOE = 0.5307, GUE/Riemann = 0.5996.
pub fn mean_spacing_ratio(levels: &[f64]) -> f64 {
    if levels.len() < 3 {
        return 0.0;
    }
    let mut s: Vec<f64> = levels.iter().map(|v| v.abs()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut gaps = Vec::with_capacity(s.len() - 1);
    for w in s.windows(2) {
        gaps.push((w[1] - w[0]).max(1e-12));
    }
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
    if cnt > 0 {
        sum / (cnt as f64)
    } else {
        0.0
    }
}
