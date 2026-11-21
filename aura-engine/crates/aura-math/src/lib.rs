//! `aura-math`: Núcleo aritmético entero, pesos ternarios en planos de bits (`u64`),
//! Memoria de Pesos Rápidos (Regla Delta Normalizada en punto fijo Q12), tablas LUT
//! para galga $U(1)$ y entropía entera, y diagnóstico de rango efectivo (anti-colapso).
//!
//! Principio de diseño: cero dependencias externas, cero `f32`/`f64` y cero asignaciones
//! dinámicas en el bucle caliente de inferencia.

pub const LATENT_DIM: usize = 256;
pub const WORDS_PER_ROW: usize = LATENT_DIM / 64;
pub const ENSEMBLE_HEADS: usize = 4;
pub const PHASE_STEPS: usize = 256;

/// Umbrales por defecto de histéresis para consolidación de pesos ternarios en sueño.
pub const DEFAULT_HYSTERESIS_HI: i16 = 96;
pub const DEFAULT_HYSTERESIS_LO: i16 = 24;
pub const DEFAULT_SHADOW_CAP: i16 = 512;

/// Generador determinista Xorshift64 con mezcla SplitMix64 en la inicialización
/// para garantizar flujos independientes entre semillas hexadecimales contiguas.
#[derive(Clone, Copy, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn from_seed(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z == 0 {
            z = 0x9E37_79B9_7F4A_7C15;
        }
        Self { state: z }
    }

    /// Crea un sub-flujo determinista independiente etiquetado (para splits/componentes).
    pub fn split_stream(&self, tag: u64) -> Self {
        Self::from_seed(self.state ^ tag.wrapping_mul(0xD1B5_4A32_D192_ED03))
    }

    /// Paso Xorshift64 sin multiplicación.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline(always)]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        (self.next_u64() as usize) % bound
    }

    /// Entero centrado en `[-128, 127]` sin multiplicación.
    #[inline(always)]
    pub fn next_signed_byte(&mut self) -> i32 {
        ((self.next_u64() & 0xFF) as i32) - 128
    }
}

/// Decaimiento exponencial simétrico hacia cero usando únicamente desplazamientos de bits.
#[inline(always)]
pub fn decay_sym(v: i32, shift: u8) -> i32 {
    if shift == 0 {
        return 0;
    }
    if v >= 0 {
        v - (v >> shift)
    } else {
        v + ((-v) >> shift)
    }
}

/// Versión `i16` de decaimiento exponencial simétrico por desplazamiento de bits.
#[inline(always)]
pub fn decay_sym_i16(v: i16, shift: u8) -> i16 {
    if shift == 0 {
        return 0;
    }
    if v >= 0 {
        v - (v >> shift)
    } else {
        v + ((-v) >> shift)
    }
}

/// Tabla de cosenos precalculada en punto fijo Q8 (`-256..=256`).
/// Se construye una única vez al inicializar; el hot loop solo lee memoria L1.
pub fn build_cos_lut() -> [i16; PHASE_STEPS] {
    let mut lut = [0i16; PHASE_STEPS];
    for (i, slot) in lut.iter_mut().enumerate() {
        let angle = (i as f64) * std::f64::consts::TAU / (PHASE_STEPS as f64);
        *slot = (angle.cos() * 256.0).round() as i16;
    }
    lut
}

/// Tabla precalculada `round(256 * log2(x))` para `x in 0..=512` (punto fijo Q8).
/// Permite estimar entropía local en `aura-perception` sin punto flotante en el hot loop.
pub fn build_log2_lut() -> [u16; 513] {
    let mut lut = [0u16; 513];
    lut[0] = 0;
    for (i, slot) in lut.iter_mut().enumerate().skip(1) {
        *slot = ((i as f64).log2() * 256.0).round() as u16;
    }
    lut
}

/// Vector latente cuantizado en punto fijo Q7 (`-127..=127`), alineado a línea de caché (64 B).
#[repr(align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentVec {
    pub data: [i8; LATENT_DIM],
}

impl Default for LatentVec {
    fn default() -> Self {
        Self::zero()
    }
}

impl LatentVec {
    pub const fn zero() -> Self {
        Self {
            data: [0; LATENT_DIM],
        }
    }

    pub fn from_i32_shifted(src: &[i32], shift: u8) -> Self {
        debug_assert_eq!(src.len(), LATENT_DIM);
        let mut data = [0i8; LATENT_DIM];
        for (dst, &v) in data.iter_mut().zip(src.iter()) {
            *dst = (v >> shift).clamp(-127, 127) as i8;
        }
        Self { data }
    }

    /// Distancia L1 (Energía Libre de Laplace) sin multiplicaciones ni punto flotante.
    #[inline]
    pub fn l1_distance(&self, other: &Self) -> i32 {
        let mut acc = 0i32;
        for i in 0..LATENT_DIM {
            acc += (self.data[i] as i32 - other.data[i] as i32).abs();
        }
        acc
    }

    #[inline]
    pub fn l1_norm(&self) -> i32 {
        let mut acc = 0i32;
        for &v in &self.data {
            acc += (v as i32).abs();
        }
        acc
    }

    /// Consenso de media entera sobre las `K = 4` cabezas del ensamble (división por `>> 2`).
    #[inline]
    pub fn mean_of_four(preds: &[LatentVec; ENSEMBLE_HEADS]) -> Self {
        let mut data = [0i8; LATENT_DIM];
        for (i, dst) in data.iter_mut().enumerate() {
            let s = preds[0].data[i] as i32
                + preds[1].data[i] as i32
                + preds[2].data[i] as i32
                + preds[3].data[i] as i32;
            *dst = (s >> 2).clamp(-127, 127) as i8;
        }
        Self { data }
    }

    /// Consenso por mediana entera (óptimo bajo pérdida L1 de Laplace) sobre `K = 4` cabezas.
    #[inline]
    pub fn median_of_four(preds: &[LatentVec; ENSEMBLE_HEADS]) -> Self {
        let mut data = [0i8; LATENT_DIM];
        for (i, dst) in data.iter_mut().enumerate() {
            let mut vals = [
                preds[0].data[i] as i32,
                preds[1].data[i] as i32,
                preds[2].data[i] as i32,
                preds[3].data[i] as i32,
            ];
            vals.sort_unstable();
            *dst = ((vals[1] + vals[2]) >> 1).clamp(-127, 127) as i8;
        }
        Self { data }
    }

    /// Mezcla EMA entera: `self += (target - self) >> shift`.
    #[inline]
    pub fn ema_blend(&mut self, target: &Self, shift: u8) {
        for i in 0..LATENT_DIM {
            let cur = self.data[i] as i32;
            let tgt = target.data[i] as i32;
            self.data[i] = (cur + ((tgt - cur) >> shift)).clamp(-127, 127) as i8;
        }
    }
}

/// Matriz de pesos lentos ternarios `{-1, 0, +1}` empaquetada en planos de bits `u64`.
///
/// Cada peso ocupa exactamente 2 bits en inferencia (`pos_planes` y `neg_planes`),
/// reduciendo 16× el consumo de memoria frente a `f32` y eliminando por completo
/// las instrucciones de multiplicación (`IMUL`/`FMUL`) mediante iteración dispersa `O(nnz)`.
#[derive(Clone, Debug)]
pub struct TernaryMatrix {
    pub rows: usize,
    pub cols_words: usize,
    pub pos_planes: Vec<u64>,
    pub neg_planes: Vec<u64>,
    pub shadow_acc: Vec<i16>,
}

impl TernaryMatrix {
    pub fn new_zeros(rows: usize, cols: usize) -> Self {
        assert!(
            cols > 0 && cols % 64 == 0,
            "cols must be a positive multiple of 64"
        );
        let cols_words = cols / 64;
        Self {
            rows,
            cols_words,
            pos_planes: vec![0u64; rows * cols_words],
            neg_planes: vec![0u64; rows * cols_words],
            shadow_acc: vec![0i16; rows * cols],
        }
    }

    /// Inicialización dispersa determinista con densidad porcentual `density_pct` (0..=100).
    pub fn new_random_sparse(
        rows: usize,
        cols: usize,
        density_pct: u8,
        rng: &mut Xorshift64,
    ) -> Self {
        let mut m = Self::new_zeros(rows, cols);
        let thresh = density_pct.min(100) as u64;
        for r in 0..rows {
            for c in 0..cols {
                let draw = rng.next_u64() % 100;
                if draw < thresh {
                    let sign = if (rng.next_u64() & 1) == 0 { 1i8 } else { -1i8 };
                    let shadow = if sign > 0 { 128 } else { -128 };
                    m.set_weight_and_shadow(r, c, sign, shadow);
                }
            }
        }
        m
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols_words * 64
    }

    #[inline]
    pub fn weight_at(&self, row: usize, col: usize) -> i8 {
        let word_idx = row * self.cols_words + (col >> 6);
        let mask = 1u64 << (col & 63);
        if (self.pos_planes[word_idx] & mask) != 0 {
            1
        } else if (self.neg_planes[word_idx] & mask) != 0 {
            -1
        } else {
            0
        }
    }

    #[inline]
    pub fn set_weight_and_shadow(&mut self, row: usize, col: usize, w: i8, shadow: i16) {
        let word_idx = row * self.cols_words + (col >> 6);
        let mask = 1u64 << (col & 63);
        if w > 0 {
            self.pos_planes[word_idx] |= mask;
            self.neg_planes[word_idx] &= !mask;
        } else if w < 0 {
            self.neg_planes[word_idx] |= mask;
            self.pos_planes[word_idx] &= !mask;
        } else {
            self.pos_planes[word_idx] &= !mask;
            self.neg_planes[word_idx] &= !mask;
        }
        let cols = self.cols();
        self.shadow_acc[row * cols + col] = shadow;
    }

    /// Proyección lineal con CERO multiplicaciones: `y_r = sum_{w=+1} x - sum_{w=-1} x`.
    /// Recorre únicamente los bits no nulos (`O(nnz)`) usando `trailing_zeros` y `x &= x - 1`.
    pub fn project_zero_mul(&self, input: &[i8], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.cols());
        debug_assert_eq!(output.len(), self.rows);
        for (r, out_slot) in output.iter_mut().enumerate() {
            let mut sum = 0i32;
            let row_offset = r * self.cols_words;
            for w in 0..self.cols_words {
                let mut pos = self.pos_planes[row_offset + w];
                let mut neg = self.neg_planes[row_offset + w];
                let base_idx = w << 6;
                while pos != 0 {
                    let bit = pos.trailing_zeros() as usize;
                    sum += input[base_idx + bit] as i32;
                    pos &= pos - 1;
                }
                while neg != 0 {
                    let bit = neg.trailing_zeros() as usize;
                    sum -= input[base_idx + bit] as i32;
                    neg &= neg - 1;
                }
            }
            *out_slot = sum;
        }
    }

    /// Consolidación durante el Ciclo de Sueño con histéresis por defecto (`HI=96, LO=24`).
    #[inline]
    pub fn consolidate_sleep_weight(&mut self, row: usize, col: usize, delta: i16) {
        self.consolidate_with_thresholds(
            row,
            col,
            delta,
            DEFAULT_HYSTERESIS_HI,
            DEFAULT_HYSTERESIS_LO,
            DEFAULT_SHADOW_CAP,
        );
    }

    /// Consolidación con umbrales configurables (permite ablación exacta sin histéresis).
    #[inline]
    pub fn consolidate_with_thresholds(
        &mut self,
        row: usize,
        col: usize,
        delta: i16,
        hi: i16,
        lo: i16,
        shadow_cap: i16,
    ) {
        let cols = self.cols();
        let idx = row * cols + col;
        let acc = self.shadow_acc[idx]
            .saturating_add(delta)
            .clamp(-shadow_cap, shadow_cap);
        self.shadow_acc[idx] = acc;

        let word_idx = row * self.cols_words + (col >> 6);
        let bit_mask = 1u64 << (col & 63);

        if acc >= hi {
            self.pos_planes[word_idx] |= bit_mask;
            self.neg_planes[word_idx] &= !bit_mask;
        } else if acc <= -hi {
            self.neg_planes[word_idx] |= bit_mask;
            self.pos_planes[word_idx] &= !bit_mask;
        } else if acc.abs() < lo {
            self.pos_planes[word_idx] &= !bit_mask;
            self.neg_planes[word_idx] &= !bit_mask;
        }
    }

    pub fn nonzero_count(&self) -> usize {
        let pos: u32 = self.pos_planes.iter().map(|w| w.count_ones()).sum();
        let neg: u32 = self.neg_planes.iter().map(|w| w.count_ones()).sum();
        (pos + neg) as usize
    }

    /// Memoria ocupada por los planos de bits de inferencia (2 bits/peso).
    pub fn memory_bytes_packed(&self) -> usize {
        (self.pos_planes.len() + self.neg_planes.len()) * std::mem::size_of::<u64>()
    }

    /// Memoria equivalente de una matriz densa `f32` (32 bits/peso).
    pub fn memory_bytes_f32_equiv(&self) -> usize {
        self.rows * self.cols() * std::mem::size_of::<f32>()
    }
}

/// Memoria Asociativa de Pesos Rápidos (Regla Delta Normalizada en punto fijo Q12).
///
/// Actualización online: `M_t = gamma * M_{t-1} + eta_norm * (v_t - M_{t-1} k_t) k_t^T`.
/// La normalización por potencia de dos de `||k||_2^2` garantiza contractividad estricta
/// (`0 < eta_eff * ||k||_2^2 <= 2^{-lr_shift}`) para cualquier norma de clave, usando
/// exclusivamente desplazamientos de bits enteros y cero asignaciones en el hot loop.
#[derive(Clone, Debug)]
pub struct FastWeightMemory {
    pub dim: usize,
    pub weights: Vec<i32>,
    pub decay_shift: u8,
    scratch_pred: Vec<i32>,
}

impl FastWeightMemory {
    pub fn new(dim: usize, decay_shift: u8) -> Self {
        Self {
            dim,
            weights: vec![0i32; dim * dim],
            decay_shift,
            scratch_pred: vec![0i32; dim],
        }
    }

    pub fn reset(&mut self) {
        self.weights.fill(0);
        self.scratch_pred.fill(0);
    }

    /// Recuperación asociativa en punto fijo Q12: `v_hat = (M * k) >> 12`.
    pub fn query(&self, key: &[i8], out: &mut [i32]) {
        debug_assert_eq!(key.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        for (r, out_slot) in out.iter_mut().enumerate() {
            let row = &self.weights[r * self.dim..(r + 1) * self.dim];
            let mut acc = 0i64;
            for c in 0..self.dim {
                acc += (row[c] as i64) * (key[c] as i64);
            }
            *out_slot = (acc >> 12) as i32;
        }
    }

    /// Actualización online NLMS (Delta Rule normalizada por potencia de dos de `||k||_2^2`).
    /// `lr_shift >= 1` controla el paso efectivo (`1 => ~0.5`, `2 => ~0.25`, `3 => ~0.125`).
    pub fn update_online(&mut self, key: &[i8], target_val: &[i32], lr_shift: u8) {
        debug_assert_eq!(key.len(), self.dim);
        debug_assert_eq!(target_val.len(), self.dim);

        let mut norm_sq: i64 = 0;
        for &k in key {
            let kv = k as i64;
            norm_sq += kv * kv;
        }
        if norm_sq == 0 {
            return;
        }

        // Predecir en buffer interno (cero asignaciones dinámicas).
        for r in 0..self.dim {
            let row = &self.weights[r * self.dim..(r + 1) * self.dim];
            let mut acc = 0i64;
            for c in 0..self.dim {
                acc += (row[c] as i64) * (key[c] as i64);
            }
            self.scratch_pred[r] = (acc >> 12) as i32;
        }

        // Garantizar 0 < ||k||^2 * 2^(-net_shift) / 2^12 <= 2^(-lr_shift) < 1 para toda escala.
        let k_log2 = (63 - norm_sq.leading_zeros()) as i32;
        let net_shift = k_log2 - 12 + (lr_shift.max(1) as i32);
        let decay = self.decay_shift;

        for r in 0..self.dim {
            let err = target_val[r].saturating_sub(self.scratch_pred[r]) as i64;
            let row = &mut self.weights[r * self.dim..(r + 1) * self.dim];
            for c in 0..self.dim {
                let decayed = if decay > 0 {
                    decay_sym(row[c], decay)
                } else {
                    row[c]
                };
                let prod = err * (key[c] as i64);
                let shifted = if net_shift >= 0 {
                    prod >> (net_shift.min(48) as u32)
                } else {
                    prod.saturating_mul(1i64 << ((-net_shift).min(12) as u32))
                };
                let delta = shifted.clamp(-1_000_000, 1_000_000) as i32;
                row[c] = decayed.saturating_add(delta).clamp(-16_000_000, 16_000_000);
            }
        }
    }
}

/// Calcula el **rango efectivo** (effective rank = exp(H(lambda))) de una muestra de vectores
/// latentes `LatentVec`, siguiendo la advertencia de la §6 de la propuesta para vigilar
/// el colapso dimensional.
///
/// Usa tridiagonalización de Householder + iteración QL implícita con desplazamiento de Wilkinson
/// sobre la matriz de covarianza muestral (diagnóstico fuera del hot loop).
pub fn effective_rank(samples: &[LatentVec]) -> f64 {
    effective_rank_subspace(samples, LATENT_DIM)
}

pub fn effective_rank_subspace(samples: &[LatentVec], dim: usize) -> f64 {
    let n = samples.len();
    let d = dim.min(LATENT_DIM);
    if n < 2 || d == 0 {
        return 0.0;
    }

    let mut mean = vec![0.0f64; d];
    for s in samples {
        for i in 0..d {
            mean[i] += s.data[i] as f64;
        }
    }
    let inv_n = 1.0 / (n as f64);
    for m in &mut mean {
        *m *= inv_n;
    }

    let mut cov = vec![0.0f64; d * d];
    for s in samples {
        let mut centered = vec![0.0f64; d];
        for i in 0..d {
            centered[i] = (s.data[i] as f64) - mean[i];
        }
        for i in 0..d {
            let ci = centered[i];
            let row = i * d;
            for j in i..d {
                cov[row + j] += ci * centered[j];
            }
        }
    }
    let inv_nm1 = 1.0 / ((n - 1) as f64);
    for i in 0..d {
        for j in i..d {
            let v = cov[i * d + j] * inv_nm1;
            cov[i * d + j] = v;
            cov[j * d + i] = v;
        }
    }

    let eigs = symmetric_eigenvalues(&mut cov, d);
    let trace: f64 = eigs.iter().filter(|&&v| v > 1e-12).sum();
    if trace <= 1e-12 {
        return 0.0;
    }
    let mut entropy = 0.0f64;
    for &lam in &eigs {
        if lam > 1e-12 {
            let p = lam / trace;
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

/// Eigenvalores de una matriz simétrica densa `d x d` mediante Householder + QL implícito.
fn symmetric_eigenvalues(a: &mut [f64], d: usize) -> Vec<f64> {
    let mut diag = vec![0.0f64; d];
    let mut off = vec![0.0f64; d];

    // 1. Tridiagonalización de Householder (algoritmo clásico tred2 sin vectores propios).
    for i in (1..d).rev() {
        let l = i - 1;
        let mut h = 0.0f64;
        let mut scale = 0.0f64;
        if l > 0 {
            for k in 0..=l {
                scale += a[i * d + k].abs();
            }
            if scale == 0.0 {
                off[i] = a[i * d + l];
            } else {
                for k in 0..=l {
                    a[i * d + k] /= scale;
                    h += a[i * d + k] * a[i * d + k];
                }
                let f = a[i * d + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                off[i] = scale * g;
                h -= f * g;
                a[i * d + l] = f - g;
                let mut f_acc = 0.0f64;
                for j in 0..=l {
                    let mut g_acc = 0.0f64;
                    for k in 0..=j {
                        g_acc += a[j * d + k] * a[i * d + k];
                    }
                    for k in (j + 1)..=l {
                        g_acc += a[k * d + j] * a[i * d + k];
                    }
                    off[j] = g_acc / h;
                    f_acc += off[j] * a[i * d + j];
                }
                let hh = f_acc / (h + h);
                for j in 0..=l {
                    let f_j = a[i * d + j];
                    let g_j = off[j] - hh * f_j;
                    off[j] = g_j;
                    for k in 0..=j {
                        a[j * d + k] -= f_j * off[k] + g_j * a[i * d + k];
                    }
                }
            }
        } else {
            off[i] = a[i * d + l];
        }
        diag[i] = h;
    }
    for i in 0..d {
        diag[i] = a[i * d + i];
    }

    // Desplazar subdiagonal a off[0..d-1]
    for i in 1..d {
        off[i - 1] = off[i];
    }
    off[d - 1] = 0.0;

    // 2. Iteración QL implícita con desplazamientos de Wilkinson (tql1).
    for l in 0..d {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m < d - 1 {
                let dd = diag[m].abs() + diag[m + 1].abs();
                if off[m].abs() <= 1e-15 * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                break;
            }
            let mut g = (diag[l + 1] - diag[l]) / (2.0 * off[l]);
            let r = g.hypot(1.0);
            g = diag[m] - diag[l] + off[l] / (g + r.copysign(g));
            let mut s = 1.0f64;
            let mut c = 1.0f64;
            let mut p = 0.0f64;
            for i in (l..m).rev() {
                let f = s * off[i];
                let b = c * off[i];
                let r2 = f.hypot(g);
                off[i + 1] = r2;
                if r2 == 0.0 {
                    diag[i + 1] -= p;
                    off[m] = 0.0;
                    break;
                }
                s = f / r2;
                c = g / r2;
                g = diag[i + 1] - p;
                let r3 = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r3;
                diag[i + 1] = g + p;
                g = c * r3 - b;
            }
            diag[l] -= p;
            off[l] = g;
            off[m] = 0.0;
        }
    }
    diag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_project_zero_mul_exact_against_dense() {
        let mut rng = Xorshift64::from_seed(0xC0DE_0001);
        for &density in &[10u8, 40, 80, 100] {
            let mat = TernaryMatrix::new_random_sparse(128, 256, density, &mut rng);
            let mut input = [0i8; 256];
            for x in &mut input {
                *x = rng.next_signed_byte().clamp(-127, 127) as i8;
            }
            let mut out_zero_mul = vec![0i32; 128];
            mat.project_zero_mul(&input, &mut out_zero_mul);

            // Verificación exacta contra producto denso escalar tradicional.
            for (r, &got) in out_zero_mul.iter().enumerate() {
                let mut expected = 0i32;
                for (c, &x) in input.iter().enumerate() {
                    expected += (mat.weight_at(r, c) as i32) * (x as i32);
                }
                assert_eq!(got, expected, "mismatch at row {r}, density {density}");
            }
            // Comprobar densidad de memoria exactamente 16x menor que f32.
            assert_eq!(mat.memory_bytes_f32_equiv() / mat.memory_bytes_packed(), 16);
        }
    }

    #[test]
    fn fast_weight_nlms_contracts_monotonically_for_all_scales() {
        let mut rng = Xorshift64::from_seed(0xC0DE_0002);
        for &scale in &[4i32, 32, 120] {
            let mut fw = FastWeightMemory::new(64, 0);
            let mut key = [0i8; 64];
            let mut target = vec![0i32; 64];
            for i in 0..64 {
                key[i] = ((rng.next_signed_byte() * scale) / 128).clamp(-127, 127) as i8;
                target[i] = rng.next_signed_byte() * 4;
            }
            let mut pred = vec![0i32; 64];
            let mut prev_err = i64::MAX;
            for _ in 0..6 {
                fw.update_online(&key, &target, 1);
                fw.query(&key, &mut pred);
                let err: i64 = target
                    .iter()
                    .zip(pred.iter())
                    .map(|(t, p)| (*t as i64 - *p as i64).abs())
                    .sum();
                assert!(
                    err <= prev_err,
                    "NLMS failed monotonic contraction at scale {scale}: {err} > {prev_err}"
                );
                prev_err = err;
            }
            assert!(
                prev_err < 2500,
                "NLMS residual too high at scale {scale}: {prev_err}"
            );
        }
    }

    #[test]
    fn ternary_hysteresis_resists_small_perturbations() {
        let mut mat = TernaryMatrix::new_zeros(4, 64);
        mat.set_weight_and_shadow(0, 5, 1, 200);
        // Pequeñas perturbaciones negativas no deben voltear inmediatamente el peso +1.
        for _ in 0..5 {
            mat.consolidate_sleep_weight(0, 5, -16);
        }
        assert_eq!(mat.weight_at(0, 5), 1);
        // Empuje consistente sí logra neutralizar y luego invertir el peso.
        for _ in 0..20 {
            mat.consolidate_sleep_weight(0, 5, -16);
        }
        assert_eq!(mat.weight_at(0, 5), -1);
    }

    #[test]
    fn effective_rank_distinguishes_collapse_vs_spread() {
        // Caso 1: Colapso a rango 1 (todos los vectores son múltiplos de un eje).
        let mut collapsed = Vec::new();
        for i in 0..64 {
            let mut v = LatentVec::zero();
            let val = ((i as i32) - 32) * 3;
            v.data[0] = val.clamp(-127, 127) as i8;
            v.data[1] = val.clamp(-127, 127) as i8;
            collapsed.push(v);
        }
        let er_collapsed = effective_rank_subspace(&collapsed, 64);
        assert!(
            (er_collapsed - 1.0).abs() < 0.05,
            "expected erank ~ 1.0, got {er_collapsed}"
        );

        // Caso 2: Rango 8 ortogonal exacto.
        let mut rank8 = Vec::new();
        let mut rng = Xorshift64::from_seed(42);
        for _ in 0..128 {
            let mut v = LatentVec::zero();
            for axis in 0..8 {
                v.data[axis] = if (rng.next_u64() & 1) == 0 { 64 } else { -64 };
            }
            rank8.push(v);
        }
        let er_8 = effective_rank_subspace(&rank8, 64);
        assert!(
            er_8 > 6.5 && er_8 < 8.5,
            "expected erank ~ 8.0, got {er_8}"
        );
    }

    #[test]
    fn cos_and_log2_luts_valid() {
        let cos_lut = build_cos_lut();
        assert_eq!(cos_lut[0], 256);
        assert_eq!(cos_lut[64], 0);
        assert_eq!(cos_lut[128], -256);
        assert_eq!(cos_lut[192], 0);

        let log_lut = build_log2_lut();
        assert_eq!(log_lut[1], 0);
        assert_eq!(log_lut[2], 256);
        assert_eq!(log_lut[16], 1024);
        assert_eq!(log_lut[256], 2048);
    }
}
