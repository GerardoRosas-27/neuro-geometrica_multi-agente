//! Deterministic PRNG (SplitMix64 + Xoshiro256**) with zero external dependencies.

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

    #[inline]
    pub fn uniform01(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64;
        (u + 0.5) / ((1u64 << 53) as f64)
    }

    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform01()
    }

    #[inline]
    pub fn usize_range(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }

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
