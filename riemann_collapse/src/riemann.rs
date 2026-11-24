//! Pure-Rust high-precision evaluation of Riemann zeta on `s = 1/2 + i*t`, Hardy's `Z(t)`,
//! exact zeros `gamma_1..gamma_M` via Brent's method, and elementary prime-number arithmetic
//! (Sieve of Eratosthenes, von Mangoldt Lambda(n), Chebyshev psi(x)).

use std::f64::consts::PI;

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

pub fn riemann_siegel_theta(t: f64) -> f64 {
    let lg = ln_gamma(C64::new(0.25, 0.5 * t));
    lg.im - 0.5 * t * PI.ln()
}

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
    sum
}

pub fn hardy_z_complex(t: f64) -> C64 {
    C64::exp_i(riemann_siegel_theta(t)).mul(zeta_critical(t))
}

pub fn hardy_z(t: f64) -> f64 {
    hardy_z_complex(t).re
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
    let mut d;

    for _ in 0..100 {
        if fb == 0.0 || (b - a).abs() <= tol {
            return b;
        }
        if fa != fc && fb != fc {
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

/// Computes the first `n_zeros` exact non-trivial zeros `gamma_1 < ... < gamma_n` of `zeta(1/2 + i*t)`.
pub fn compute_riemann_zeros(n_zeros: usize) -> Vec<f64> {
    let mut zeros: Vec<f64> = Vec::with_capacity(n_zeros);
    let mut t = 12.0;
    let mut z_prev = hardy_z(t);
    let step = 0.06;
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

/// Elementary Sieve of Eratosthenes + von Mangoldt function `Lambda(n)` up to `n_max` (0 zeta calls).
pub fn von_mangoldt_table(n_max: usize) -> Vec<f64> {
    let mut is_prime = vec![true; n_max + 1];
    if n_max >= 1 {
        is_prime[0] = false;
        is_prime[1] = false;
    }
    let mut p = 2;
    while p * p <= n_max {
        if is_prime[p] {
            let mut m = p * p;
            while m <= n_max {
                is_prime[m] = false;
                m += p;
            }
        }
        p += 1;
    }
    let mut lam = vec![0.0; n_max + 1];
    for p in 2..=n_max {
        if is_prime[p] {
            let lp = (p as f64).ln();
            let mut pk = p;
            while pk <= n_max {
                lam[pk] = lp;
                match pk.checked_mul(p) {
                    Some(v) => pk = v,
                    None => break,
                }
            }
        }
    }
    lam
}

pub fn is_prime_int(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

pub fn is_prime_power_int(n: usize) -> bool {
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

/// Exact Chebyshev `psi(n) = sum_{k <= n} Lambda(k)` for `n = 0..=n_max`.
pub fn chebyshev_psi_table(n_max: usize) -> Vec<f64> {
    let lam = von_mangoldt_table(n_max);
    let mut psi = vec![0.0; n_max + 1];
    let mut acc = 0.0;
    for n in 0..=n_max {
        acc += lam[n];
        psi[n] = acc;
    }
    psi
}
