//! phi(hist) = [1 | z_t, z_{t-1}, ..., z_{t-C+1} | z_t[i]*z_t[j] (i<=j) | M1 biblioteca Koopman/SINDy (si bq10)]
//! Antes del inicio de la conversación se rellena repitiendo z_0.
//!
//! En modo BQ10 (`bq10 = true`, equivalente a `Mech::bq(10, false, true)` de `orbita/src/constelacion4/atlas.rs`):
//! * M1 añade generadores no lineales escalados por `LIB_S = 0.3` sobre `z_t` (con radio RMS `r = ||z_t||_2 / sqrt(d)`):
//!   - Par / radial (grado 2): `0.3 * (cos(2 z_t[i]) - 1)`, `0.3 * r * z_t[i]` (2*d cols, grado 2)
//!   - Impar no lineal (grado 3, pues la parte lineal z_t ya está en grado 1):
//!     `0.3 * sin(2 z_t[i])`, `0.3 * tanh(2 z_t[i])`, `0.3 * z_t[i]^3`, `0.3 * r^2 * z_t[i]` (4*d cols, grado 3)
//! * M7 DA-STR (`da = 10`): penalización ridge escalada por grado `lam_g = lam_1 * da^(g-1)` (×1, ×10, ×100).

pub const LIB_S: f32 = 0.3;
pub const BQ10_DA: f64 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FeatCfg {
    pub d: usize,
    pub ctx: usize,
    pub quad: bool,
    #[serde(default)]
    pub bq10: bool,
}

impl FeatCfg {
    pub fn dim(&self) -> usize {
        1 + self.ctx * self.d
            + if self.quad { self.d * (self.d + 1) / 2 } else { 0 }
            + if self.bq10 { 6 * self.d } else { 0 }
    }

    /// Bloque de grado de cada feature: 0 = sesgo, 1 = lineal, 2 = cuadrático/M1-g2, 3 = cúbico/M1-g3
    pub fn degree(&self) -> Vec<u8> {
        let mut g = vec![0u8];
        g.extend(std::iter::repeat(1u8).take(self.ctx * self.d));
        if self.quad {
            g.extend(std::iter::repeat(2u8).take(self.d * (self.d + 1) / 2));
        }
        if self.bq10 {
            g.extend(std::iter::repeat(1u8).take(2 * self.d)); // sin(2z), tanh(2z) (grado 1 en LIB_DEG de atlas.rs)
            g.extend(std::iter::repeat(2u8).take(2 * self.d)); // cos(2z)-1, r*z (grado 2)
            g.extend(std::iter::repeat(3u8).take(2 * self.d)); // z^3, r^2*z (grado 3)
        }
        g
    }

    /// Factor de escala de columna M7 DA-STR: s_j = da^(-(grado_j - 1)/2) para grado >= 1 (1.0 para sesgo y grado 1).
    pub fn colscale_m7(&self, da: f64) -> Vec<f64> {
        self.degree()
            .into_iter()
            .map(|g| if g <= 1 { 1.0 } else { da.powf(-((g as i32 - 1) as f64) / 2.0) })
            .collect()
    }

    /// `seq`: estados (t+1) x d disponibles (seq[t] es el actual). Escribe phi en `out`.
    pub fn build(&self, seq: &[f32], t: usize, out: &mut [f32]) {
        let d = self.d;
        out[0] = 1.0;
        for c in 0..self.ctx {
            let s = t.saturating_sub(c);
            out[1 + c * d..1 + (c + 1) * d].copy_from_slice(&seq[s * d..(s + 1) * d]);
        }
        let zt = &seq[t * d..(t + 1) * d];
        let mut k = 1 + self.ctx * d;
        if self.quad {
            for i in 0..d {
                for j in i..d {
                    out[k] = zt[i] * zt[j];
                    k += 1;
                }
            }
        }
        if self.bq10 {
            let r2: f32 = zt.iter().map(|&x| x * x).sum::<f32>() / (d.max(1) as f32);
            let r = r2.sqrt();
            // M1 impar: sin(2z), tanh(2z)
            for i in 0..d {
                out[k] = LIB_S * (2.0 * zt[i]).sin();
                k += 1;
            }
            for i in 0..d {
                out[k] = LIB_S * (2.0 * zt[i]).tanh();
                k += 1;
            }
            // M1 grado 2: cos(2z)-1, r*z
            for i in 0..d {
                out[k] = LIB_S * ((2.0 * zt[i]).cos() - 1.0);
                k += 1;
            }
            for i in 0..d {
                out[k] = LIB_S * r * zt[i];
                k += 1;
            }
            // M1 grado 3: z^3, r^2*z
            for i in 0..d {
                let zc = zt[i].clamp(-4.0, 4.0);
                out[k] = LIB_S * (zc * zc * zc);
                k += 1;
            }
            let r2c = r2.min(16.0);
            for i in 0..d {
                let zc = zt[i].clamp(-4.0, 4.0);
                out[k] = LIB_S * r2c * zc;
                k += 1;
            }
        }
    }
}
