//! Deterministic PRNG (SplitMix64 + Xoshiro256**) and pure-Rust Adam optimizer.

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

/// Minimal deterministic Adam state over a flat parameter slice.
#[derive(Clone, Debug)]
pub struct AdamState {
    m: Vec<f64>,
    v: Vec<f64>,
    step: usize,
}

impl AdamState {
    pub fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
            step: 0,
        }
    }

    pub fn resize(&mut self, new_len: usize) {
        self.m.resize(new_len, 0.0);
        self.v.resize(new_len, 0.0);
    }

    pub fn update(&mut self, params: &mut [f64], grads: &[f64], lr: f64, weight_decay: f64) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.m.len());
        self.step += 1;
        let b1 = 0.9_f64;
        let b2 = 0.999_f64;
        let eps = 1e-8_f64;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);
        for i in 0..params.len() {
            let g = grads[i] + weight_decay * params[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            params[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
}
