//! Deterministic SplitMix64 PRNG and IEEE-754 bit-exact checksum helper.

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn fork(&self, salt: u64) -> Self {
        let mut s = self.state ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        s = (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self {
            state: s ^ (s >> 31),
        }
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
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
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
            let mut v: Vec<f64> = (0..d).map(|_| self.normal()).collect();
            let n2: f64 = v.iter().map(|x| x * x).sum();
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

/// Exact IEEE-754 FNV-1a 64-bit checksum over slices of f64.
pub fn checksum_f64_slices(slices: &[&[f64]]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for sl in slices {
        for &v in *sl {
            let bits = v.to_bits();
            for b in bits.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
    }
    h
}
