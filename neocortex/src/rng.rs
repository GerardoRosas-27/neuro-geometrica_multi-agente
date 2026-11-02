//! Generador pseudoaleatorio determinista `Xoshiro256StarStar` (Blackman & Vigna, 2021)
//! inicializado mediante `SplitMix64`, libre de dependencias externas.

#[derive(Clone, Debug)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Inicializa el estado de 256 bits a partir de una semilla de 64 bits usando SplitMix64.
    pub fn seed_from_u64(mut state: u64) -> Self {
        let mut s = [0u64; 4];
        for slot in &mut s {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Devuelve un número uniforme en `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Devuelve un número uniforme en `[low, high)`.
    #[inline]
    pub fn gen_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// Muestra normal estándar $\mathcal{N}(0, 1)$ mediante transformación de Box-Muller.
    #[inline]
    pub fn randn(&mut self) -> f64 {
        let u1 = self.gen_range(1e-12, 1.0);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
