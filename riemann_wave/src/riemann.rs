//! Pure-Rust high-precision evaluation of Riemann zeta on the critical line `s = 1/2 + i*t`,
//! Hardy's real `Z(t)` function, exact non-trivial zeros `gamma_1..gamma_M` via Brent's method,
//! spectral unfolding, and explicit-formula prime-power wave interference.

use std::f64::consts::{E, PI};

#[derive(Clone, Copy, Debug)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }

    #[inline]
    pub fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }

    #[inline]
    pub fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    #[inline]
    pub fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn ln(self) -> Self {
        Self::new(self.norm().ln(), self.im.atan2(self.re))
    }

    #[inline]
    pub fn exp_i(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }
}

/// Bernoulli numbers B_{2k} for k = 1..=8.
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

/// Factorials (2k)! for k = 1..=8.
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

/// Principal branch of `ln(Gamma(z))` for `Re(z) > 0` via argument shift + 8-term Stirling series.
pub fn ln_gamma(mut z: C64) -> C64 {
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

/// Riemann-Siegel theta function `theta(t) = Im(ln Gamma(1/4 + i*t/2)) - (t/2) * ln(pi)`.
pub fn riemann_siegel_theta(t: f64) -> f64 {
    let lg = ln_gamma(C64::new(0.25, 0.5 * t));
    lg.im - 0.5 * t * PI.ln()
}

/// Evaluates `zeta(1/2 + i*t)` using adaptive Euler-Maclaurin summation (`N = max(32, ceil(1.35*t))`).
pub fn zeta_critical(t: f64) -> C64 {
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
    let ln_n = nf.ln();
    let angle = -t * ln_n;
    let n_neg_s = C64::new(angle.cos() / nf.sqrt(), angle.sin() / nf.sqrt());
    // Integral term: N^{1-s} / (s - 1) = (N * N^{-s}) / (s - 1)
    let integral = n_neg_s.scale(nf).div(s.sub(C64::new(1.0, 0.0)));
    sum = sum.add(integral).add(n_neg_s.scale(0.5));

    // Euler-Maclaurin Bernoulli correction terms
    let inv_n = 1.0 / nf;
    let inv_n2 = inv_n * inv_n;
    let mut n_pow = n_neg_s.scale(inv_n); // N^{-(s+1)}
    let mut rising = s; // s(s+1)...(s+2k-2), starts at s for k=1
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
    sum
}

/// Evaluates `e^{i*theta(t)} * zeta(1/2 + i*t)`. Its imaginary part is theoretically identically 0.
pub fn hardy_z_complex(t: f64) -> C64 {
    C64::exp_i(riemann_siegel_theta(t)).mul(zeta_critical(t))
}

/// Real-valued Hardy `Z(t)` whose real roots coincide with the non-trivial zeros on `Re(s) = 1/2`.
pub fn hardy_z(t: f64) -> f64 {
    hardy_z_complex(t).re
}

/// Brent's root-finding algorithm on bracket `[a, b]` with `f(a)*f(b) <= 0`.
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
    let mut d;

    for _ in 0..100 {
        if fb == 0.0 || (b - a).abs() <= tol {
            return b;
        }
        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            let s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
            let cond1 = (s - (3.0 * a + b) * 0.25) * (s - b) >= 0.0;
            let cond2 = (s - b).abs() >= 0.5 * (b - c).abs();
            let cond3 = (b - c).abs() < tol;
            if cond1 || cond2 || cond3 {
                d = 0.5 * (a - b);
            } else {
                d = s - b;
            }
        } else {
            // Secant / bisection
            let s = b - fb * (b - a) / (fb - fa);
            if (s - (3.0 * a + b) * 0.25) * (s - b) >= 0.0 {
                d = 0.5 * (a - b);
            } else {
                d = s - b;
            }
        }
        c = b;
        fc = fb;
        if d.abs() > tol * 0.5 {
            b += d;
        } else {
            b += if a > b { tol * 0.5 } else { -tol * 0.5 };
        }
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

/// Computes the first `n_zeros` imaginary parts `gamma_1 < gamma_2 < ... < gamma_n`
/// of the non-trivial zeros of `zeta(1/2 + i*t)` using adaptive bracket scanning + Brent.
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

/// Smooth Riemann-von Mangoldt counting function `N(T) = (T / 2pi) * ln(T / (2pi * e))`.
#[inline]
pub fn riemann_counting_smooth(t: f64) -> f64 {
    (t / (2.0 * PI)) * (t / (2.0 * PI * E)).ln()
}

/// Computes normalized nearest-neighbor spacings of Riemann zeros (mean = 1).
pub fn unfolded_riemann_spacings(zeros: &[f64]) -> Vec<f64> {
    let u: Vec<f64> = zeros.iter().map(|&g| riemann_counting_smooth(g)).collect();
    let mut diffs: Vec<f64> = u.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    for d in &mut diffs {
        *d /= mean;
    }
    diffs
}

/// Maps the first `m` Riemann zeros via smooth unfolding linearly onto `[1.0, m as f64]`.
pub fn riemann_zeros_as_frequencies(zeros: &[f64], m: usize) -> Vec<f64> {
    assert!(zeros.len() >= m && m >= 2);
    let u: Vec<f64> = zeros[..m].iter().map(|&g| riemann_counting_smooth(g)).collect();
    let u_min = u[0];
    let u_max = u[m - 1];
    let span = (u_max - u_min).max(1e-12);
    u.into_iter()
        .map(|val| 1.0 + ((m - 1) as f64) * (val - u_min) / span)
        .collect()
}

/// Maps the first `m` Riemann zeros via smooth unfolding to strictly open `(0, 1)` quantiles
/// suitable for inverse-CDF spectral transformation (Riemann-RFF).
pub fn riemann_zeros_as_quantiles(zeros: &[f64], m: usize) -> Vec<f64> {
    assert!(zeros.len() >= m && m >= 2);
    let u: Vec<f64> = zeros[..m].iter().map(|&g| riemann_counting_smooth(g)).collect();
    let u_min = u[0];
    let u_max = u[m - 1];
    let mean_gap = (u_max - u_min) / (m - 1) as f64;
    // Pad by half a mean gap on each end so quantiles mirror (k - 0.5)/m support
    let lo = u_min - 0.5 * mean_gap;
    let hi = u_max + 0.5 * mean_gap;
    u.into_iter()
        .map(|val| ((val - lo) / (hi - lo)).clamp(1e-6, 1.0 - 1e-6))
        .collect()
}

/// Evaluates Fejér-windowed explicit-formula wave superposition at `x`:
/// `W(x) = (-2 / sqrt(x)) * sum_k (1 - k/M) * cos(freqs[k] * ln(x))`.
pub fn prime_wave_amplitude(freqs: &[f64], x: f64) -> f64 {
    let m = freqs.len() as f64;
    let ln_x = x.ln();
    let mut acc = 0.0;
    for (k, &gamma) in freqs.iter().enumerate() {
        let weight = 1.0 - (k as f64) / m;
        acc += weight * (gamma * ln_x).cos();
    }
    (-2.0 / x.sqrt()) * acc
}

/// Returns true iff `n = p^k` for some prime `p` and integer `k >= 1`.
pub fn is_prime_power(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    let mut x = n;
    let mut p = 2;
    while p * p <= x {
        if x % p == 0 {
            while x % p == 0 {
                x /= p;
            }
            return x == 1;
        }
        p += 1;
    }
    true
}

/// Computes exact ROC-AUC and SNR separating prime powers (`2..=n_max`) from non-prime-power composites.
pub fn evaluate_prime_wave_discrimination(freqs: &[f64], n_max: usize) -> (f64, f64) {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for n in 2..=n_max {
        let amp = prime_wave_amplitude(freqs, n as f64);
        if is_prime_power(n) {
            pos.push(amp);
        } else {
            neg.push(amp);
        }
    }
    let mut wins = 0.0;
    for &p in &pos {
        for &c in &neg {
            if p > c {
                wins += 1.0;
            } else if (p - c).abs() < 1e-15 {
                wins += 0.5;
            }
        }
    }
    let auc = wins / ((pos.len() * neg.len()) as f64);
    let mu_pos = pos.iter().sum::<f64>() / pos.len() as f64;
    let mu_neg = neg.iter().sum::<f64>() / neg.len() as f64;
    let var_neg = neg
        .iter()
        .map(|&v| (v - mu_neg) * (v - mu_neg))
        .sum::<f64>()
        / neg.len().max(1) as f64;
    let snr = (mu_pos - mu_neg) / var_neg.sqrt().max(1e-12);
    (auc, snr)
}
