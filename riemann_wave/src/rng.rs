//! Deterministic PRNG (SplitMix64 + Xoshiro256**), special functions (`erf`, inverse normal CDF),
//! and quasi-Monte Carlo Halton sequences with zero external dependencies.

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    spare_normal: Option<f64>,
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Creates an independent deterministic RNG stream from `(seed, stream_id)`.
    pub fn from_seed(seed: u64, stream_id: u64) -> Self {
        let mut sm = seed ^ stream_id.wrapping_mul(0xD1B5_4A32_D192_ED03);
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self {
            s,
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in strictly open `(0, 1)`.
    #[inline]
    pub fn uniform01(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64;
        (u + 0.5) / ((1u64 << 53) as f64)
    }

    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform01()
    }

    /// Standard Gaussian N(0, 1) via Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = self.uniform01();
        let u2 = self.uniform01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// High-precision error function `erf(x)` (~1e-15 accuracy across all real `x`).
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let val = if ax < 3.5 {
        // Series: erf(x) = (2/sqrt(pi)) * exp(-x^2) * sum_{k>=0} (2^k * x^{2k+1}) / (2k+1)!!
        let x2 = ax * ax;
        let mut term = ax;
        let mut sum = term;
        for k in 1..100 {
            term *= 2.0 * x2 / (2 * k + 1) as f64;
            sum += term;
            if term <= 1e-16 * sum {
                break;
            }
        }
        (2.0 / PI.sqrt()) * (-x2).exp() * sum
    } else if ax > 6.5 {
        1.0
    } else {
        // Continued fraction for erfc(x) = (exp(-x^2)/sqrt(pi)) * (1/(x + (1/2)/(x + 1/(x + (3/2)/...))))
        // Evaluated via Lentz's algorithm on erfc(x) = (exp(-x^2)/sqrt(pi)) / (x + (1/2)/(x + 1/(x + ...)))
        let tiny = 1e-30;
        let mut b = ax;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for n in 1..60 {
            let a = 0.5 * n as f64;
            b = ax;
            d = b + a * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + a / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let erfc_val = (-ax * ax).exp() / PI.sqrt() * h;
        1.0 - erfc_val
    };
    sign * val.clamp(0.0, 1.0)
}

/// Standard normal CDF `Phi(x)`.
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse standard normal CDF `Phi^{-1}(p)` via Peter Acklam's rational approximation
/// refined by one Newton step to machine precision (~1e-15).
pub fn inv_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // Single Newton polish step
    let err = normal_cdf(x) - p;
    let pdf = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
    if pdf > 1e-30 {
        x -= err / pdf;
    }
    x
}

/// Radical inverse in `base` for Halton quasi-Monte Carlo sequences (`index >= 1`).
pub fn halton(mut index: usize, base: usize) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    let b = base as f64;
    while index > 0 {
        f /= b;
        r += f * (index % base) as f64;
        index /= base;
    }
    r
}
