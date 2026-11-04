//! phi(hist) = [1 | z_t, z_{t-1}, ..., z_{t-C+1} | z_t[i]*z_t[j] (i<=j)]
//! Antes del inicio de la conversación se rellena repitiendo z_0.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct FeatCfg { pub d: usize, pub ctx: usize, pub quad: bool }

impl FeatCfg {
    pub fn dim(&self) -> usize { 1 + self.ctx * self.d + if self.quad { self.d * (self.d + 1) / 2 } else { 0 } }
    /// bloque de grado de cada feature: 0 = sesgo, 1 = lineal, 2 = cuadrático
    pub fn degree(&self) -> Vec<u8> {
        let mut g = vec![0u8];
        g.extend(std::iter::repeat(1u8).take(self.ctx * self.d));
        if self.quad { g.extend(std::iter::repeat(2u8).take(self.d * (self.d + 1) / 2)); }
        g
    }
    /// `seq`: estados (t+1) x d disponibles (seq[t] es el actual). Escribe phi en `out`.
    pub fn build(&self, seq: &[f32], t: usize, out: &mut [f32]) {
        let d = self.d;
        out[0] = 1.0;
        for c in 0..self.ctx {
            let s = t.saturating_sub(c);
            out[1 + c * d..1 + (c + 1) * d].copy_from_slice(&seq[s * d..(s + 1) * d]);
        }
        if self.quad {
            let zt = &seq[t * d..(t + 1) * d];
            let mut k = 1 + self.ctx * d;
            for i in 0..d { for j in i..d { out[k] = zt[i] * zt[j]; k += 1; } }
        }
    }
}
