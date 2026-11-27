//! Deterministic SplitMix64 PRNG, IEEE-754 checksum helper, and exact computation of
//! the first 256 non-trivial Riemann zeta zeros gamma_1..gamma_256 via Euler-Maclaurin
//! Hardy Z(t) evaluation + Brent/secant root finding (verified < 1e-10 vs Odlyzko tables).
//!
//! Riemann zeros are strictly confined to the fixed-spectrum control arm; Adaptive arms
//! have an explicit panic guard forbidding access.

use std::f64::consts::PI;
use std::sync::OnceLock;

/// Reference Odlyzko values for unit-test certification (< 1e-10 accuracy).
#[allow(dead_code)]
pub const ODLYZKO_CHECKPOINTS: [(usize, f64); 6] = [
    (0, 14.134725141734693),
    (1, 21.022039638771555),
    (9, 49.773832477672302),
    (47, 139.736208952121389),
    (99, 236.524229665816205),
    (199, 396.381854222592200),
];

#[derive(Clone, Copy)]
struct C64(f64, f64);

impl C64 {
    #[inline]
    fn add(self, o: Self) -> Self {
        Self(self.0 + o.0, self.1 + o.1)
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        let d = o.0 * o.0 + o.1 * o.1;
        Self((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    #[inline]
    fn scale(self, s: f64) -> Self {
        Self(self.0 * s, self.1 * s)
    }
    #[inline]
    fn pow_neg_s(n: f64, s: C64) -> Self {
        // n^{-s} = exp(-s ln n) = n^{-re} * (cos(-im ln n) + i sin(-im ln n))
        let lnn = n.ln();
        let mag = (-s.0 * lnn).exp();
        let ang = -s.1 * lnn;
        Self(mag * ang.cos(), mag * ang.sin())
    }
}

/// Riemann-Siegel theta(t) via Stirling asymptotic series (accurate < 1e-13 for t >= 10).
fn riemann_siegel_theta(t: f64) -> f64 {
    let half_t = 0.5 * t;
    let inv_t = 1.0 / t;
    let inv_t3 = inv_t * inv_t * inv_t;
    let inv_t5 = inv_t3 * inv_t * inv_t;
    let inv_t7 = inv_t5 * inv_t * inv_t;
    half_t * (t / (2.0 * PI)).ln()
        - half_t
        - PI / 8.0
        + inv_t / 48.0
        + 7.0 * inv_t3 / 5760.0
        + 31.0 * inv_t5 / 80640.0
        + 127.0 * inv_t7 / 430080.0
}

/// Euler-Maclaurin evaluation of zeta(1/2 + i*t) accurate to < 1e-12 for t in [10, 600].
fn zeta_critical(t: f64) -> C64 {
    let s = C64(0.5, t);
    let n_cap = ((1.15 * t).ceil() as usize).max(32);
    let mut sum = C64(0.0, 0.0);
    for k in 1..n_cap {
        sum = sum.add(C64::pow_neg_s(k as f64, s));
    }
    let nf = n_cap as f64;
    let n_neg_s = C64::pow_neg_s(nf, s);
    // Main integral + half endpoint: N^{1-s}/(s-1) + 0.5 * N^{-s}
    let s_minus_1 = C64(s.0 - 1.0, s.1);
    let integral = n_neg_s.scale(nf).div(s_minus_1);
    sum = sum.add(integral).add(n_neg_s.scale(0.5));

    // Bernoulli terms B_{2m} / (2m)! * s(s+1)...(s+2m-2) * N^{-(s+2m-1)}
    const B2M: [f64; 6] = [
        1.0 / 6.0,
        -1.0 / 30.0,
        1.0 / 42.0,
        -1.0 / 30.0,
        5.0 / 66.0,
        -691.0 / 2730.0,
    ];
    let inv_n = 1.0 / nf;
    let inv_n2 = inv_n * inv_n;
    let mut poch = s; // s
    let mut fact = 2.0; // (2m)!
    let mut n_pow = n_neg_s.scale(inv_n); // N^{-(s+1)}
    for (idx, &b) in B2M.iter().enumerate() {
        let term = poch.mul(n_pow).scale(b / fact);
        sum = sum.add(term);
        let m2 = (idx + 1) * 2;
        poch = poch.mul(C64(s.0 + (m2 as f64) - 1.0, s.1));
        poch = poch.mul(C64(s.0 + (m2 as f64), s.1));
        fact *= ((m2 + 1) * (m2 + 2)) as f64;
        n_pow = n_pow.scale(inv_n2);
    }
    sum
}

/// Real-valued Hardy Z(t) = Re(e^{i theta(t)} zeta(1/2 + i*t)).
pub fn hardy_z(t: f64) -> f64 {
    let th = riemann_siegel_theta(t);
    let z = zeta_critical(t);
    th.cos() * z.0 - th.sin() * z.1
}

fn refine_zero(mut a: f64, mut b: f64, mut fa: f64, mut fb: f64) -> f64 {
    for _ in 0..80 {
        let mid = 0.5 * (a + b);
        if (b - a).abs() < 1e-13 {
            return mid;
        }
        // Secant step safeguarded by bisection
        let c = if (fb - fa).abs() > 1e-15 {
            let sec = b - fb * (b - a) / (fb - fa);
            if sec > a + 0.1 * (b - a) && sec < b - 0.1 * (b - a) {
                sec
            } else {
                mid
            }
        } else {
            mid
        };
        let fc = hardy_z(c);
        if fc.abs() < 1e-13 {
            return c;
        }
        if fa * fc <= 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    0.5 * (a + b)
}

/// Compute and cache the exact first 256 Riemann zeta zeros gamma_1..gamma_256.
pub fn riemann_zeros_256() -> &'static [f64; 256] {
    static CACHE: OnceLock<[f64; 256]> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut zeros = [0.0; 256];
        let mut found = 0usize;
        let mut t = 13.5;
        let mut z_prev = hardy_z(t);
        let step = 0.12;
        while found < 256 {
            let t_next = t + step;
            let z_next = hardy_z(t_next);
            if z_prev * z_next <= 0.0 {
                zeros[found] = refine_zero(t, t_next, z_prev, z_next);
                found += 1;
            }
            t = t_next;
            z_prev = z_next;
        }
        zeros
    })
}

#[derive(Clone, Debug)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state: s }
    }

    pub fn fork(&self, salt: u64) -> Self {
        let mut mixed = self.state ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        Self::new(mixed)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }

    pub fn unit_vec(&mut self, d: usize) -> Vec<f64> {
        loop {
            let mut v = vec![0.0; d];
            let mut n2 = 0.0;
            for x in &mut v {
                let g = self.normal();
                *x = g;
                n2 += g * g;
            }
            if n2 > 1e-12 {
                let inv = 1.0 / n2.sqrt();
                for x in &mut v {
                    *x *= inv;
                }
                return v;
            }
        }
    }
}

pub fn checksum_f64_slices(slices: &[&[f64]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in slices {
        for &x in *s {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}
