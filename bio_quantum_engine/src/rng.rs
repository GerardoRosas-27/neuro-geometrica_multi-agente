//! Generador pseudoaleatorio determinista y rápido xoshiro256**
//!
//! No requiere crates externos y garantiza reproducibilidad idéntica
//! en Rust y Python para todas las semillas del protocolo científico.

/// Estado interno del generador xoshiro256**
#[derive(Debug, Clone)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Inicializa a partir de una semilla numérica u64 mediante splitmix64
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_sm = || {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let s0 = next_sm();
        let s1 = next_sm();
        let s2 = next_sm();
        let s3 = next_sm();
        Self {
            s: [
                if s0 == 0 { 0xdeadbeef } else { s0 },
                if s1 == 0 { 0xcafebabe } else { s1 },
                if s2 == 0 { 0x12345678 } else { s2 },
                if s3 == 0 { 0x87654321 } else { s3 },
            ],
        }
    }

    /// Genera el siguiente entero de 64 bits
    pub fn next_u64(&mut self) -> u64 {
        let res = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        res
    }

    /// Genera un número de punto flotante en [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Muestra una variable normal estándar Gaussiana N(0, 1) vía Box-Muller
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Genera un entero uniforme en [0, n)
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}
