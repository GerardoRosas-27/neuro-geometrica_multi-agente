//! `aura-perception`: Codificador perceptual SIN TOKENS (Zero-Token Entropy Patcher)
//! y codificación delta-modulada sobre flujo crudo de bytes.
//!
//! Opera 100% con aritmética entera en el bucle caliente: un predictor causal ligero
//! (tabla de contextos con hash + `LOG2_LUT` entera en Q8) estima la sorpresa local
//! `H(b_t | b_<t)`. Mientras los bytes son predecibles se acumulan en un estado recurrente
//! entero sin multiplicaciones; cuando `H >= tau_corte` se cierra el parche y se emite
//! un `LatentVec` cuantizado `i8`.

use aura_math::{
    build_log2_lut, decay_sym, LatentVec, TernaryMatrix, Xorshift64, LATENT_DIM, WORDS_PER_ROW,
};

const CTX_BUCKETS: usize = 4096;
const SLOTS_PER_BUCKET: usize = 8;

/// Contrato del subsistema perceptual (fiel a la propuesta AURA §4).
pub trait PerceptionEncoder {
    /// Ingiere bytes crudos; devuelve `Some(LatentVec)` solo cuando detecta corte por entropía
    /// (o límite máximo de longitud de parche).
    fn push_byte(&mut self, byte: u8) -> Option<LatentVec>;
}

/// Diagnóstico detallado de cada parche emitido.
#[derive(Clone, Copy, Debug)]
pub struct PatchMeta {
    pub patch_len: usize,
    pub boundary_entropy_q8: u16,
    pub last_byte: u8,
}

#[derive(Clone, Copy)]
struct ContextBucket {
    bytes: [u8; SLOTS_PER_BUCKET],
    counts: [u8; SLOTS_PER_BUCKET],
    total: u16,
}

impl Default for ContextBucket {
    fn default() -> Self {
        Self {
            bytes: [0; SLOTS_PER_BUCKET],
            counts: [0; SLOTS_PER_BUCKET],
            total: 0,
        }
    }
}

impl ContextBucket {
    #[inline]
    fn count_of(&self, b: u8) -> u16 {
        for i in 0..SLOTS_PER_BUCKET {
            if self.counts[i] > 0 && self.bytes[i] == b {
                return self.counts[i] as u16;
            }
        }
        0
    }

    #[inline]
    fn observe(&mut self, b: u8) {
        // Halving preventivo para mantener conteos acotados (<= 255) y adaptables online.
        if self.total >= 240 {
            let mut new_total = 0u16;
            for c in &mut self.counts {
                *c >>= 1;
                new_total += *c as u16;
            }
            self.total = new_total;
        }

        // Buscar ranura existente o libre / mínima.
        let mut min_idx = 0usize;
        let mut min_val = u8::MAX;
        for i in 0..SLOTS_PER_BUCKET {
            if self.counts[i] > 0 && self.bytes[i] == b {
                self.counts[i] = self.counts[i].saturating_add(1);
                self.total = self.total.saturating_add(1);
                return;
            }
            if self.counts[i] < min_val {
                min_val = self.counts[i];
                min_idx = i;
            }
        }
        // Reemplazar ranura vacía o de menor frecuencia.
        self.total = self.total.saturating_sub(self.counts[min_idx] as u16);
        self.bytes[min_idx] = b;
        self.counts[min_idx] = 1;
        self.total = self.total.saturating_add(1);
    }
}

/// Tabla dispersa ternaria de proyección byte -> estado recurrente (0 multiplicaciones).
#[derive(Clone)]
struct ZeroMulByteEmbed {
    byte_pos: Vec<[u64; WORDS_PER_ROW]>,
    byte_neg: Vec<[u64; WORDS_PER_ROW]>,
    delta_pos: Vec<[u64; WORDS_PER_ROW]>,
    delta_neg: Vec<[u64; WORDS_PER_ROW]>,
}

impl ZeroMulByteEmbed {
    fn new(rng: &mut Xorshift64) -> Self {
        let make_table = |r: &mut Xorshift64| {
            let mut pos = vec![[0u64; WORDS_PER_ROW]; 256];
            let mut neg = vec![[0u64; WORDS_PER_ROW]; 256];
            for sym in 0..256 {
                for w in 0..WORDS_PER_ROW {
                    let mut pm = 0u64;
                    let mut nm = 0u64;
                    for bit in 0..64 {
                        // ~25% densidad dispersa por símbolo -> ~64 bits activos en 256D.
                        let d = r.next_u64() & 7;
                        if d == 0 {
                            pm |= 1u64 << bit;
                        } else if d == 1 {
                            nm |= 1u64 << bit;
                        }
                    }
                    pos[sym][w] = pm;
                    neg[sym][w] = nm;
                }
            }
            (pos, neg)
        };
        let (byte_pos, byte_neg) = make_table(rng);
        let (delta_pos, delta_neg) = make_table(rng);
        Self {
            byte_pos,
            byte_neg,
            delta_pos,
            delta_neg,
        }
    }

    #[inline]
    fn accumulate(&self, byte: u8, delta: u8, acc: &mut [i32; LATENT_DIM]) {
        let bp = &self.byte_pos[byte as usize];
        let bn = &self.byte_neg[byte as usize];
        let dp = &self.delta_pos[delta as usize];
        let dn = &self.delta_neg[delta as usize];
        for w in 0..WORDS_PER_ROW {
            let base = w << 6;
            let mut m = bp[w];
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                acc[base + b] += 48;
                m &= m - 1;
            }
            let mut m = bn[w];
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                acc[base + b] -= 48;
                m &= m - 1;
            }
            let mut m = dp[w];
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                acc[base + b] += 24;
                m &= m - 1;
            }
            let mut m = dn[w];
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                acc[base + b] -= 24;
                m &= m - 1;
            }
        }
    }
}

/// Codificador perceptual sin tokens basado en entropía causal local + proyección ternaria
/// con estabilización anti-colapso estilo VICReg (media y piso de varianza entera).
pub struct EntropyPatcher {
    buckets: Vec<ContextBucket>,
    unigram: [u16; 256],
    unigram_total: u32,
    log2_lut: [u16; 513],

    prev2: u8,
    prev1: u8,
    patch_len: usize,
    pub min_patch_len: usize,
    pub max_patch_len: usize,
    pub entropy_threshold_q8: u16,

    embed: ZeroMulByteEmbed,
    encoder_proj: TernaryMatrix,
    acc: [i32; LATENT_DIM],
    scratch_i8: [i8; LATENT_DIM],
    scratch_out: [i32; LATENT_DIM],

    // Regulador anti-colapso (VICReg entera): media móvil y desviación L1 por dimensión.
    pub use_vicreg: bool,
    running_mean_q4: [i32; LATENT_DIM],
    running_mad_q4: [i32; LATENT_DIM],
    collapse_gain_q8: [i32; LATENT_DIM],

    pub total_bytes: usize,
    pub total_patches: usize,
    pub last_meta: PatchMeta,
}

impl EntropyPatcher {
    pub fn new(
        seed: u64,
        min_patch_len: usize,
        max_patch_len: usize,
        entropy_threshold_q8: u16,
        use_vicreg: bool,
    ) -> Self {
        let mut rng = Xorshift64::from_seed(seed).split_stream(0x5045_5243);
        let embed = ZeroMulByteEmbed::new(&mut rng);
        let encoder_proj = TernaryMatrix::new_random_sparse(LATENT_DIM, LATENT_DIM, 20, &mut rng);
        Self {
            buckets: vec![ContextBucket::default(); CTX_BUCKETS],
            unigram: [0; 256],
            unigram_total: 0,
            log2_lut: build_log2_lut(),
            prev2: 0,
            prev1: 0,
            patch_len: 0,
            min_patch_len: min_patch_len.max(1),
            max_patch_len: max_patch_len.max(min_patch_len.max(1)),
            entropy_threshold_q8,
            embed,
            encoder_proj,
            acc: [0; LATENT_DIM],
            scratch_i8: [0; LATENT_DIM],
            scratch_out: [0; LATENT_DIM],
            use_vicreg,
            running_mean_q4: [0; LATENT_DIM],
            running_mad_q4: [32 << 4; LATENT_DIM],
            collapse_gain_q8: [256; LATENT_DIM],
            total_bytes: 0,
            total_patches: 0,
            last_meta: PatchMeta {
                patch_len: 0,
                boundary_entropy_q8: 0,
                last_byte: 0,
            },
        }
    }

    #[inline]
    fn context_index(&self) -> usize {
        let h = ((self.prev2 as usize) * 257) ^ ((self.prev1 as usize) * 31);
        h & (CTX_BUCKETS - 1)
    }

    #[inline]
    fn log2_q8(&self, x: usize) -> u16 {
        if x < self.log2_lut.len() {
            self.log2_lut[x]
        } else {
            let mut v = x;
            let mut shifts = 0u16;
            while v >= self.log2_lut.len() {
                v >>= 1;
                shifts += 1;
            }
            self.log2_lut[v].saturating_add(shifts << 8)
        }
    }

    /// Estima la sorpresa local `-log2 P(byte | b_{t-2}, b_{t-1})` en punto fijo Q8 (256 = 1 bit).
    #[inline]
    pub fn estimate_surprise_q8(&self, byte: u8) -> u16 {
        let bucket = &self.buckets[self.context_index()];
        if bucket.total >= 2 {
            let c = bucket.count_of(byte) as usize;
            let num = self.log2_q8((bucket.total as usize) + 4);
            let den = self.log2_q8(c + 1);
            num.saturating_sub(den)
        } else {
            let c = self.unigram[byte as usize] as usize;
            let tot = (self.unigram_total as usize).min(500);
            let num = self.log2_q8(tot + 8);
            let den = self.log2_q8(c + 1);
            num.saturating_sub(den)
        }
    }

    fn observe_predictor(&mut self, byte: u8) {
        let idx = self.context_index();
        self.buckets[idx].observe(byte);
        if self.unigram_total >= 4096 {
            let mut nt = 0u32;
            for u in &mut self.unigram {
                *u >>= 1;
                nt += *u as u32;
            }
            self.unigram_total = nt;
        }
        self.unigram[byte as usize] = self.unigram[byte as usize].saturating_add(1);
        self.unigram_total += 1;
    }

    fn finalize_patch(&mut self, entropy_q8: u16, last_byte: u8) -> LatentVec {
        for i in 0..LATENT_DIM {
            self.scratch_i8[i] = (self.acc[i] >> 1).clamp(-127, 127) as i8;
        }
        // Proyección lineal ternaria con CERO multiplicaciones.
        self.encoder_proj
            .project_zero_mul(&self.scratch_i8, &mut self.scratch_out);

        let mut data = [0i8; LATENT_DIM];
        for i in 0..LATENT_DIM {
            let raw = self.scratch_out[i] >> 3;
            let mean_cur = self.running_mean_q4[i] >> 4;
            let centered = raw - mean_cur;
            self.running_mean_q4[i] += ((raw << 4) - self.running_mean_q4[i]) >> 4;
            let abs_c = centered.abs();
            self.running_mad_q4[i] += ((abs_c << 4) - self.running_mad_q4[i]) >> 4;

            // Aplicar ganancia aprendida del codificador (en Q8, inicializada en 256 = 1.0).
            let gain = self.collapse_gain_q8[i];
            let scaled = (centered * gain) >> 8;
            data[i] = scaled.clamp(-127, 127) as i8;
        }

        // Conservar 1/4 del estado para continuidad temporal entre parches.
        for v in &mut self.acc {
            *v >>= 2;
        }
        self.last_meta = PatchMeta {
            patch_len: self.patch_len,
            boundary_entropy_q8: entropy_q8,
            last_byte,
        };
        self.patch_len = 0;
        self.total_patches += 1;
        LatentVec { data }
    }

    /// Actualización orgánica del codificador ante el error de predicción JEPA `|z_obs - z_pred|`.
    ///
    /// - Sin VICReg (`use_vicreg = false`): minimizar `|z_obs - z_pred|` encoge orgánicamente la
    ///   ganancia de las dimensiones difíciles de predecir (colapso de representación clásico §6).
    /// - Con VICReg (`use_vicreg = true`): añade término de piso de varianza (`MAD >= 24`) y
    ///   decorrelación local entre vecinos, evitando el colapso mientras conserva predictibilidad.
    pub fn adapt_encoder_from_prediction(&mut self, z_obs: &LatentVec, z_pred: &LatentVec) {
        const VAR_FLOOR: i32 = 24;
        for i in 0..LATENT_DIM {
            let err = (z_obs.data[i] as i32 - z_pred.data[i] as i32).abs();
            let obs_mag = (z_obs.data[i] as i32).abs();

            // Gradiente de contracción autopredictiva: ejes con alto error encogen su ganancia
            // para minimizar |z_obs - z_pred|; ejes altamente predecibles retienen ganancia.
            let mut delta_g: i32 = if err > 14 && obs_mag > 2 {
                -(err >> 3).clamp(2, 12)
            } else if err < 10 && obs_mag > 4 {
                3
            } else {
                0
            };

            if self.use_vicreg {
                let eff_mad = ((self.running_mad_q4[i] >> 4) * self.collapse_gain_q8[i]) >> 8;
                // Término de Varianza VICReg: si la dispersión efectiva cae bajo VAR_FLOOR, restaurar.
                if eff_mad < VAR_FLOOR {
                    delta_g += ((VAR_FLOOR - eff_mad) >> 1).max(3);
                } else if eff_mad > 64 {
                    delta_g -= 2;
                }
                // Término ligero de Covarianza: penalizar redundancia con eje adyacente.
                let j = (i + 1) & (LATENT_DIM - 1);
                let prod_sign = (z_obs.data[i] as i32).signum() * (z_obs.data[j] as i32).signum();
                if prod_sign > 0 && eff_mad > VAR_FLOOR + 4 {
                    delta_g -= 1;
                }
            }

            let min_gain = if self.use_vicreg { 128 } else { 2 };
            self.collapse_gain_q8[i] = (self.collapse_gain_q8[i] + delta_g).clamp(min_gain, 512);
        }
    }

    pub fn push_byte_with_meta(&mut self, byte: u8) -> Option<(LatentVec, PatchMeta)> {
        let surprise_q8 = self.estimate_surprise_q8(byte);
        let delta = self.prev1 ^ byte;

        for v in &mut self.acc {
            *v = decay_sym(*v, 2);
        }
        self.embed.accumulate(byte, delta, &mut self.acc);
        self.observe_predictor(byte);

        self.prev2 = self.prev1;
        self.prev1 = byte;
        self.patch_len += 1;
        self.total_bytes += 1;

        let should_cut = (self.patch_len >= self.min_patch_len
            && surprise_q8 >= self.entropy_threshold_q8)
            || self.patch_len >= self.max_patch_len;

        if should_cut {
            let z = self.finalize_patch(surprise_q8, byte);
            Some((z, self.last_meta))
        } else {
            None
        }
    }
}

impl PerceptionEncoder for EntropyPatcher {
    fn push_byte(&mut self, byte: u8) -> Option<LatentVec> {
        self.push_byte_with_meta(byte).map(|(z, _)| z)
    }
}

/// Control pareado de ventana fija (ignora la entropía; corta cada `window_len` bytes).
pub struct FixedPatcher {
    inner: EntropyPatcher,
    window_len: usize,
}

impl FixedPatcher {
    pub fn new(seed: u64, window_len: usize, use_vicreg: bool) -> Self {
        let w = window_len.max(1);
        // Umbral inalcanzable -> siempre corta exactamente en `max_patch_len = w`.
        Self {
            inner: EntropyPatcher::new(seed, w, w, u16::MAX, use_vicreg),
            window_len: w,
        }
    }

    pub fn push_byte_with_meta(&mut self, byte: u8) -> Option<(LatentVec, PatchMeta)> {
        self.inner.push_byte_with_meta(byte)
    }

    pub fn total_patches(&self) -> usize {
        self.inner.total_patches
    }

    pub fn window_len(&self) -> usize {
        self.window_len
    }

    pub fn adapt_encoder_from_prediction(&mut self, z_obs: &LatentVec, z_pred: &LatentVec) {
        self.inner.adapt_encoder_from_prediction(z_obs, z_pred);
    }
}

impl PerceptionEncoder for FixedPatcher {
    fn push_byte(&mut self, byte: u8) -> Option<LatentVec> {
        self.inner.push_byte(byte)
    }
}

/// Codificación delta-modulada sin tokens para `CorticalFabric` (Parte 1 de la propuesta).
/// Devuelve `(count, neuron_ids)` indicando qué neuronas sensoriales primarias (`0..16`)
/// disparan debido a bits que cambiaron entre `prev_byte` y `curr_byte`.
#[inline]
pub fn delta_modulate_byte(prev_byte: u8, curr_byte: u8) -> (usize, [u32; 8]) {
    let diff = prev_byte ^ curr_byte;
    let mut out = [0u32; 8];
    let mut n = 0usize;
    for bit in 0..8 {
        if ((diff >> bit) & 1) == 1 {
            let is_on = ((curr_byte >> bit) & 1) as u32;
            out[n] = (bit as u32) * 2 + is_on;
            n += 1;
        }
    }
    (n, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_patcher_compresses_predictable_more_than_random() {
        let mut patcher = EntropyPatcher::new(0xA400, 1, 8, 384, true);
        // Fase de calentamiento con patrón repetitivo predecible.
        let motif = b"ABCD1234";
        for _ in 0..30 {
            for &b in motif {
                let _ = patcher.push_byte(b);
            }
        }
        // Medir longitud media de parche sobre 240 bytes predecibles.
        let mut pred_patches = 0usize;
        for _ in 0..30 {
            for &b in motif {
                if patcher.push_byte(b).is_some() {
                    pred_patches += 1;
                }
            }
        }
        // Medir longitud media de parche sobre 240 bytes aleatorios sorprendentes.
        let mut rng = Xorshift64::from_seed(999);
        let mut rand_patches = 0usize;
        for _ in 0..240 {
            let b = (rng.next_u64() & 0xFF) as u8;
            if patcher.push_byte(b).is_some() {
                rand_patches += 1;
            }
        }
        assert!(
            rand_patches > pred_patches * 2,
            "expected entropy patcher to emit far more patches on random ({rand_patches}) than predictable ({pred_patches})"
        );
    }

    #[test]
    fn delta_modulation_detects_exact_bit_changes() {
        let (n, ids) = delta_modulate_byte(0b0000_0000, 0b0000_0101);
        assert_eq!(n, 2);
        assert_eq!(ids[0], 1); // bit 0 turned ON -> 0*2 + 1
        assert_eq!(ids[1], 5); // bit 2 turned ON -> 2*2 + 1
    }
}
