//! Deterministic SplitMix64 PRNG and exact reference Riemann zeros
//! (Riemann zeros are ONLY permitted in fixed-spectrum control arms;
//! adaptive arms have an explicit panic guard preventing access).

use std::f64::consts::PI;

/// First 48 non-trivial Riemann zeta zeros (imaginary parts gamma_1..gamma_48),
/// accurate to > 1e-10 (Odlyzko reference tables).
pub const RIEMANN_ZEROS_48: [f64; 48] = [
    14.134725141734693,
    21.022039638771555,
    25.010857580145689,
    30.424876125859513,
    32.935061587739190,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167160,
    49.773832477672302,
    52.970321477714461,
    56.446247697063395,
    59.347044002602353,
    60.831778524609810,
    65.112544048081607,
    67.079810529494174,
    69.546401711173980,
    72.067157674481908,
    75.704690699083933,
    77.144840068874805,
    79.337375020249368,
    82.910380854086030,
    84.735492980517050,
    87.425274613125229,
    88.809111207634465,
    92.491899270558484,
    94.651344040519887,
    95.870634228245310,
    98.831194218193692,
    101.317851005731391,
    103.725538040478339,
    105.446623052326094,
    107.168611184276408,
    111.029535543169675,
    111.874659176992637,
    114.320220915452713,
    116.226680320857554,
    118.790782865976217,
    121.370125002420646,
    122.946829293552588,
    124.256818554345767,
    127.516683879596495,
    129.578704199956051,
    131.087688530932657,
    133.497737202997586,
    134.756509753373871,
    138.116042054533443,
    139.736208952121389,
];

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

    /// Uniform in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard Gaussian N(0, 1) via Box-Muller.
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

    /// Random unit vector in R^d.
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

/// Deterministic FNV-1a 64-bit hash over f64 slice (exact IEEE-754 bit pattern)
/// used to certify bit-exact invariance of frozen cores during cross-domain transfer.
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
