//! `aura-math`: Zero-mul 2-bit bit-plane ternary arithmetic with AVX2 LUT-4 SIMD,
//! BitNet b1.58 scaled ternary training (AbsMean + STE + AdamW + importance hysteresis),
//! zero-allocation NLMS FastWeightMemory, fixed-point LUTs, and spectral effective rank.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const LATENT_DIM: usize = 256;
pub const WORDS_PER_ROW: usize = LATENT_DIM / 64;
pub const ENSEMBLE_HEADS: usize = 4;
pub const ROW_TILE: usize = 32;

/// Deterministic zero-multiplication Xorshift64 PRNG.
#[derive(Clone, Copy, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state: s }
    }

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
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline(always)]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    #[inline(always)]
    pub fn next_signed_f32(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Approximate standard normal using Irwin-Hall (sum of 12 uniforms - 6).
    #[inline]
    pub fn next_normal_f32(&mut self) -> f32 {
        let mut sum = 0.0f32;
        for _ in 0..12 {
            sum += self.next_f32();
        }
        sum - 6.0
    }
}

/// Precomputed cosine lookup table in Q8 (-256..=256) for U(1) phase-gauge attention.
pub fn build_cos_lut() -> [i16; 256] {
    let mut lut = [0i16; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let angle = (i as f64) * std::f64::consts::TAU / 256.0;
        *slot = (angle.cos() * 256.0).round() as i16;
    }
    lut
}

/// Precomputed -log2(p) table in Q8 for entropy calculation (index 0..=256).
pub fn build_neg_log2_lut() -> [u16; 257] {
    let mut lut = [0u16; 257];
    lut[0] = 8 * 256;
    for p in 1..=256usize {
        let prob = (p as f64) / 256.0;
        let bits = (-prob.log2()).clamp(0.0, 8.0);
        lut[p] = (bits * 256.0).round() as u16;
    }
    lut
}

/// Fixed-point quantized latent vector (Q7: -127..=127), 64-byte aligned.
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

    #[inline]
    pub fn l1_distance(&self, other: &Self) -> i32 {
        let mut acc = 0i32;
        for i in 0..LATENT_DIM {
            acc += (self.data[i] as i32 - other.data[i] as i32).abs();
        }
        acc
    }

    #[inline]
    pub fn norm_l1(&self) -> i32 {
        let mut acc = 0i32;
        for &v in &self.data {
            acc += (v as i32).abs();
        }
        acc
    }

    #[inline]
    pub fn mean_abs(&self) -> f64 {
        (self.norm_l1() as f64) / (LATENT_DIM as f64)
    }

    #[inline]
    pub fn from_i32_shift(vals: &[i32], shift: u8) -> Self {
        let mut out = Self::zero();
        let n = vals.len().min(LATENT_DIM);
        for i in 0..n {
            let v = vals[i] >> shift;
            out.data[i] = v.clamp(-127, 127) as i8;
        }
        out
    }

    #[inline]
    pub fn from_f32_clamped(vals: &[f32], scale: f32) -> Self {
        let mut out = Self::zero();
        let n = vals.len().min(LATENT_DIM);
        for i in 0..n {
            let v = (vals[i] * scale).round().clamp(-127.0, 127.0) as i8;
            out.data[i] = v;
        }
        out
    }

    #[inline]
    pub fn to_f32(&self) -> [f32; LATENT_DIM] {
        let mut out = [0.0f32; LATENT_DIM];
        for i in 0..LATENT_DIM {
            out[i] = self.data[i] as f32;
        }
        out
    }
}

/// BitNet b1.58 pre-quantization RMSNorm (prevents activation outliers from crushing i8 precision).
/// Returns `inv_rms` for exact backward differentiation via `rmsnorm_backward_inplace`.
#[inline]
pub fn rmsnorm_inplace(x: &mut [f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 1.0;
    }
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv_rms = 1.0 / (ss / (n as f32) + 1e-5).sqrt();
    for v in x.iter_mut() {
        *v *= inv_rms;
    }
    inv_rms
}

/// Exact backward Jacobian through RMSNorm: given normalized output `y = x * inv_rms`
/// and upstream gradient `dy`, overwrites `dy` with exact `dx`.
#[inline]
pub fn rmsnorm_backward_inplace(y_norm: &[f32], inv_rms: f32, dy: &mut [f32]) {
    let n = y_norm.len();
    if n == 0 {
        return;
    }
    let mut dot = 0.0f32;
    for i in 0..n {
        dot += dy[i] * y_norm[i];
    }
    let mean_dot = dot / (n as f32);
    for i in 0..n {
        dy[i] = inv_rms * (dy[i] - y_norm[i] * mean_dot);
    }
}

/// Quantize f32 vector into i8 using BitNet b1.58 per-token absmax quantization.
/// Returns `absmax` (minimum 1e-6 to avoid division by zero).
#[inline]
pub fn quantize_absmax_i8(input: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(input.len(), out.len());
    let mut max_abs = 0.0f32;
    for &v in input {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs < 1e-6 {
        out.fill(0);
        return 1e-6;
    }
    let inv = 127.0 / max_abs;
    for (dst, &v) in out.iter_mut().zip(input.iter()) {
        *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    max_abs
}

// ============================================================================
// BOTTLENECK 1 BASELINE: Proposal's Naive Scalar Bit-Scan Bit-Plane Matrix
// ============================================================================

/// Naive 2-bit bit-plane layout directly from the proposal (`while pos != 0 { trailing_zeros }`).
/// Used as the exact Baseline ("Before") in Bottleneck 1 benchmarks and parity verification.
#[derive(Clone, Debug)]
pub struct NaiveBitplaneMatrix {
    pub rows: usize,
    pub cols: usize,
    pub cols_words: usize,
    pub pos_planes: Vec<u64>,
    pub neg_planes: Vec<u64>,
}

impl NaiveBitplaneMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(cols.is_multiple_of(64), "cols must be a multiple of 64");
        let cols_words = cols / 64;
        Self {
            rows,
            cols,
            cols_words,
            pos_planes: vec![0u64; rows * cols_words],
            neg_planes: vec![0u64; rows * cols_words],
        }
    }

    pub fn from_random_density(rows: usize, cols: usize, density: f32, rng: &mut Xorshift64) -> Self {
        let mut m = Self::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let u = rng.next_f32();
                if u < density * 0.5 {
                    m.set_weight(r, c, 1);
                } else if u < density {
                    m.set_weight(r, c, -1);
                }
            }
        }
        m
    }

    #[inline]
    pub fn set_weight(&mut self, row: usize, col: usize, w: i8) {
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
    }

    #[inline]
    pub fn get_weight(&self, row: usize, col: usize) -> i8 {
        let word_idx = row * self.cols_words + (col >> 6);
        let mask = 1u64 << (col & 63);
        let p = ((self.pos_planes[word_idx] & mask) != 0) as i8;
        let n = ((self.neg_planes[word_idx] & mask) != 0) as i8;
        p - n
    }

    /// Exact proposal baseline: scalar bit-scan (`while pos != 0 { trailing_zeros }`).
    pub fn project_naive_bitscan(&self, input: &[i8], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), self.rows);
        for (r, out_r) in output.iter_mut().enumerate() {
            let mut sum = 0i32;
            let row_off = r * self.cols_words;
            for w in 0..self.cols_words {
                let mut pos = self.pos_planes[row_off + w];
                let mut neg = self.neg_planes[row_off + w];
                let base = w * 64;
                while pos != 0 {
                    let bit = pos.trailing_zeros() as usize;
                    sum += input[base + bit] as i32;
                    pos &= pos - 1;
                }
                while neg != 0 {
                    let bit = neg.trailing_zeros() as usize;
                    sum -= input[base + bit] as i32;
                    neg &= neg - 1;
                }
            }
            *out_r = sum;
        }
    }

    /// Reference exact i32 matvec (for verification only).
    pub fn project_reference_i32(&self, input: &[i8], output: &mut [i32]) {
        for (r, out_r) in output.iter_mut().enumerate() {
            let mut sum = 0i32;
            for (c, &x) in input.iter().enumerate() {
                sum += (self.get_weight(r, c) as i32) * (x as i32);
            }
            *out_r = sum;
        }
    }

    pub fn packed_bytes(&self) -> usize {
        (self.pos_planes.len() + self.neg_planes.len()) * std::mem::size_of::<u64>()
    }
}

// ============================================================================
// BOTTLENECK 1 FIX: 32-Row-Tiled 2-Bit Bit-Plane TernaryMatrix + AVX2 LUT-4
// ============================================================================

/// Canonical 2-bit bit-plane ternary matrix stored in 32-row L1-cache-friendly nibble tiles.
///
/// Memory footprint: exactly `rows_padded * cols / 4` bytes = **2 bits per weight**
/// (exact 16x reduction vs `f32` whenever `rows % 32 == 0`).
///
/// Layout:
/// - Rows are grouped into tiles of 32 (`tile = r / 32`, `r_in = r % 32`).
/// - Columns are grouped into bytes of 8 weights (`bc = c / 8`, low nibble = cols `0..4`,
///   high nibble = cols `4..8`).
/// - Index: `((tile * byte_cols) + bc) * 32 + r_in`.
/// Thus, loading 32 consecutive bytes gives 8 ternary weights across 32 rows simultaneously,
/// enabling branchless zero-multiplication AVX2 `vpshufb` LUT-4 evaluation.
#[derive(Clone, Debug)]
pub struct TernaryMatrix {
    pub rows: usize,
    pub rows_padded: usize,
    pub cols: usize,
    pub byte_cols: usize,
    pub pos_tiled: Vec<u8>,
    pub neg_tiled: Vec<u8>,
}

impl TernaryMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(cols.is_multiple_of(8), "cols must be a multiple of 8");
        let rows_padded = rows.div_ceil(ROW_TILE) * ROW_TILE;
        let byte_cols = cols / 8;
        let total = rows_padded * byte_cols;
        Self {
            rows,
            rows_padded,
            cols,
            byte_cols,
            pos_tiled: vec![0u8; total],
            neg_tiled: vec![0u8; total],
        }
    }

    pub fn from_naive(src: &NaiveBitplaneMatrix) -> Self {
        let mut m = Self::new(src.rows, src.cols);
        for r in 0..src.rows {
            for c in 0..src.cols {
                let w = src.get_weight(r, c);
                if w != 0 {
                    m.set_weight(r, c, w);
                }
            }
        }
        m
    }

    pub fn from_random_density(rows: usize, cols: usize, density: f32, rng: &mut Xorshift64) -> Self {
        let mut m = Self::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let u = rng.next_f32();
                if u < density * 0.5 {
                    m.set_weight(r, c, 1);
                } else if u < density {
                    m.set_weight(r, c, -1);
                }
            }
        }
        m
    }

    #[inline(always)]
    fn idx(&self, row: usize, col: usize) -> (usize, u8) {
        let tile = row / ROW_TILE;
        let r_in = row % ROW_TILE;
        let bc = col >> 3;
        let bit = 1u8 << (col & 7);
        (((tile * self.byte_cols) + bc) * ROW_TILE + r_in, bit)
    }

    #[inline(always)]
    pub fn set_weight(&mut self, row: usize, col: usize, w: i8) {
        debug_assert!(row < self.rows && col < self.cols);
        let (idx, mask) = self.idx(row, col);
        if w > 0 {
            self.pos_tiled[idx] |= mask;
            self.neg_tiled[idx] &= !mask;
        } else if w < 0 {
            self.neg_tiled[idx] |= mask;
            self.pos_tiled[idx] &= !mask;
        } else {
            self.pos_tiled[idx] &= !mask;
            self.neg_tiled[idx] &= !mask;
        }
    }

    #[inline(always)]
    pub fn get_weight(&self, row: usize, col: usize) -> i8 {
        debug_assert!(row < self.rows && col < self.cols);
        let (idx, mask) = self.idx(row, col);
        let p = ((self.pos_tiled[idx] & mask) != 0) as i8;
        let n = ((self.neg_tiled[idx] & mask) != 0) as i8;
        p - n
    }

    /// Packed 2-bit bit-plane memory footprint in bytes.
    pub fn packed_bytes(&self) -> usize {
        self.pos_tiled.len() + self.neg_tiled.len()
    }

    pub fn nonzero_fraction(&self) -> f64 {
        let mut nnz = 0usize;
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.get_weight(r, c) != 0 {
                    nnz += 1;
                }
            }
        }
        (nnz as f64) / ((self.rows * self.cols).max(1) as f64)
    }

    /// Build split lo/hi 16-entry LUT-4 tables for an i8 input vector using ONLY additions (zero mul).
    #[inline]
    pub fn build_lut4(input: &[i8], lut_lo: &mut [[u8; 16]], lut_hi: &mut [[u8; 16]]) {
        let groups = input.len() / 4;
        debug_assert!(lut_lo.len() >= groups && lut_hi.len() >= groups);
        for g in 0..groups {
            let b = g * 4;
            let x0 = input[b] as i16;
            let x1 = input[b + 1] as i16;
            let x2 = input[b + 2] as i16;
            let x3 = input[b + 3] as i16;

            let s0 = 0i16;
            let s1 = x0;
            let s2 = x1;
            let s3 = x0 + x1;
            let s4 = x2;
            let s5 = s1 + x2;
            let s6 = s2 + x2;
            let s7 = s3 + x2;
            let s8 = x3;
            let s9 = s1 + x3;
            let s10 = s2 + x3;
            let s11 = s3 + x3;
            let s12 = s4 + x3;
            let s13 = s5 + x3;
            let s14 = s6 + x3;
            let s15 = s7 + x3;

            let vals = [
                s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
            ];
            let lo = &mut lut_lo[g];
            let hi = &mut lut_hi[g];
            for k in 0..16 {
                let u = vals[k] as u16;
                lo[k] = (u & 0xFF) as u8;
                hi[k] = (u >> 8) as u8;
            }
        }
    }

    /// Scalar branchless LUT-4 zero-mul projection (portable fallback).
    pub fn project_lut4_scalar(&self, input: &[i8], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), self.rows);
        let groups = self.cols / 4;
        let mut lut_i16 = vec![[0i16; 16]; groups];
        for (g, tbl) in lut_i16.iter_mut().enumerate() {
            let b = g * 4;
            let x0 = input[b] as i16;
            let x1 = input[b + 1] as i16;
            let x2 = input[b + 2] as i16;
            let x3 = input[b + 3] as i16;
            tbl[0] = 0;
            tbl[1] = x0;
            tbl[2] = x1;
            tbl[3] = x0 + x1;
            tbl[4] = x2;
            tbl[5] = x0 + x2;
            tbl[6] = x1 + x2;
            tbl[7] = tbl[3] + x2;
            tbl[8] = x3;
            tbl[9] = x0 + x3;
            tbl[10] = x1 + x3;
            tbl[11] = tbl[3] + x3;
            tbl[12] = x2 + x3;
            tbl[13] = tbl[5] + x3;
            tbl[14] = tbl[6] + x3;
            tbl[15] = tbl[7] + x3;
        }

        let n_tiles = self.rows_padded / ROW_TILE;
        for tile in 0..n_tiles {
            let mut acc = [0i32; ROW_TILE];
            let tile_base = tile * self.byte_cols * ROW_TILE;
            for bc in 0..self.byte_cols {
                let off = tile_base + bc * ROW_TILE;
                let t0 = &lut_i16[bc * 2];
                let t1 = &lut_i16[bc * 2 + 1];
                let p_slice = &self.pos_tiled[off..off + ROW_TILE];
                let n_slice = &self.neg_tiled[off..off + ROW_TILE];
                for r in 0..ROW_TILE {
                    let pb = p_slice[r] as usize;
                    let nb = n_slice[r] as usize;
                    let d0 = (t0[pb & 0x0F] as i32) - (t0[nb & 0x0F] as i32);
                    let d1 = (t1[pb >> 4] as i32) - (t1[nb >> 4] as i32);
                    acc[r] += d0 + d1;
                }
            }
            let r_start = tile * ROW_TILE;
            let r_end = (r_start + ROW_TILE).min(self.rows);
            for r in r_start..r_end {
                output[r] = acc[r - r_start];
            }
        }
    }

    /// AVX2 zero-mul LUT-4 projection over 32 rows simultaneously using `vpshufb`.
    /// Zero heap allocations when `cols <= 1024`.
    pub fn project_lut4_simd(&self, input: &[i8], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), self.rows);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let groups = self.cols / 4;
                if groups <= 256 {
                    let mut lut_lo = [[0u8; 16]; 256];
                    let mut lut_hi = [[0u8; 16]; 256];
                    Self::build_lut4(input, &mut lut_lo[..groups], &mut lut_hi[..groups]);
                    unsafe {
                        self.project_lut4_avx2_with_luts(
                            &lut_lo[..groups],
                            &lut_hi[..groups],
                            output,
                        );
                    }
                    return;
                } else {
                    let mut lut_lo = vec![[0u8; 16]; groups];
                    let mut lut_hi = vec![[0u8; 16]; groups];
                    Self::build_lut4(input, &mut lut_lo, &mut lut_hi);
                    unsafe {
                        self.project_lut4_avx2_with_luts(&lut_lo, &lut_hi, output);
                    }
                    return;
                }
            }
        }
        self.project_lut4_scalar(input, output);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn project_lut4_avx2_with_luts(
        &self,
        lut_lo: &[[u8; 16]],
        lut_hi: &[[u8; 16]],
        output: &mut [i32],
    ) {
        let mask0f = _mm256_set1_epi8(0x0F);
        let n_tiles = self.rows_padded / ROW_TILE;

        for tile in 0..n_tiles {
            let mut out0 = _mm256_setzero_si256(); // rows 0..8
            let mut out1 = _mm256_setzero_si256(); // rows 8..16
            let mut out2 = _mm256_setzero_si256(); // rows 16..24
            let mut out3 = _mm256_setzero_si256(); // rows 24..32

            let tile_base = tile * self.byte_cols * ROW_TILE;

            // Process in chunks of 16 byte_cols (= 32 groups = 128 weights):
            // max |i16 acc| <= 128 * 128 = 16384 < 32767 -> exact, zero overflow.
            let mut bc = 0usize;
            while bc < self.byte_cols {
                let bc_end = (bc + 16).min(self.byte_cols);
                let mut acc_a = _mm256_setzero_si256(); // [0..8, 16..24] as i16
                let mut acc_b = _mm256_setzero_si256(); // [8..16, 24..32] as i16

                while bc < bc_end {
                    let off = tile_base + bc * ROW_TILE;
                    let p_bytes = _mm256_loadu_si256(self.pos_tiled.as_ptr().add(off) as *const __m256i);
                    let n_bytes = _mm256_loadu_si256(self.neg_tiled.as_ptr().add(off) as *const __m256i);

                    let p0 = _mm256_and_si256(p_bytes, mask0f);
                    let p1 = _mm256_and_si256(_mm256_srli_epi16(p_bytes, 4), mask0f);
                    let n0 = _mm256_and_si256(n_bytes, mask0f);
                    let n1 = _mm256_and_si256(_mm256_srli_epi16(n_bytes, 4), mask0f);

                    let g0 = bc * 2;
                    let g1 = g0 + 1;

                    // Group g0 (low nibble)
                    let t0_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        lut_lo.as_ptr().add(g0) as *const __m128i,
                    ));
                    let t0_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        lut_hi.as_ptr().add(g0) as *const __m128i,
                    ));
                    let vp0_lo = _mm256_shuffle_epi8(t0_lo, p0);
                    let vp0_hi = _mm256_shuffle_epi8(t0_hi, p0);
                    let vn0_lo = _mm256_shuffle_epi8(t0_lo, n0);
                    let vn0_hi = _mm256_shuffle_epi8(t0_hi, n0);

                    acc_a = _mm256_add_epi16(acc_a, _mm256_unpacklo_epi8(vp0_lo, vp0_hi));
                    acc_b = _mm256_add_epi16(acc_b, _mm256_unpackhi_epi8(vp0_lo, vp0_hi));
                    acc_a = _mm256_sub_epi16(acc_a, _mm256_unpacklo_epi8(vn0_lo, vn0_hi));
                    acc_b = _mm256_sub_epi16(acc_b, _mm256_unpackhi_epi8(vn0_lo, vn0_hi));

                    // Group g1 (high nibble)
                    let t1_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        lut_lo.as_ptr().add(g1) as *const __m128i,
                    ));
                    let t1_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        lut_hi.as_ptr().add(g1) as *const __m128i,
                    ));
                    let vp1_lo = _mm256_shuffle_epi8(t1_lo, p1);
                    let vp1_hi = _mm256_shuffle_epi8(t1_hi, p1);
                    let vn1_lo = _mm256_shuffle_epi8(t1_lo, n1);
                    let vn1_hi = _mm256_shuffle_epi8(t1_hi, n1);

                    acc_a = _mm256_add_epi16(acc_a, _mm256_unpacklo_epi8(vp1_lo, vp1_hi));
                    acc_b = _mm256_add_epi16(acc_b, _mm256_unpackhi_epi8(vp1_lo, vp1_hi));
                    acc_a = _mm256_sub_epi16(acc_a, _mm256_unpacklo_epi8(vn1_lo, vn1_hi));
                    acc_b = _mm256_sub_epi16(acc_b, _mm256_unpackhi_epi8(vn1_lo, vn1_hi));

                    bc += 1;
                }

                // Flush i16 accumulators into i32 accumulators
                out0 = _mm256_add_epi32(out0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_a)));
                out1 = _mm256_add_epi32(out1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_b)));
                out2 = _mm256_add_epi32(out2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_a, 1)));
                out3 = _mm256_add_epi32(out3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_b, 1)));
            }

            let r_start = tile * ROW_TILE;
            if r_start + ROW_TILE <= self.rows {
                let ptr = output.as_mut_ptr().add(r_start);
                _mm256_storeu_si256(ptr as *mut __m256i, out0);
                _mm256_storeu_si256(ptr.add(8) as *mut __m256i, out1);
                _mm256_storeu_si256(ptr.add(16) as *mut __m256i, out2);
                _mm256_storeu_si256(ptr.add(24) as *mut __m256i, out3);
            } else {
                let mut tmp = [0i32; ROW_TILE];
                _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out0);
                _mm256_storeu_si256(tmp.as_mut_ptr().add(8) as *mut __m256i, out1);
                _mm256_storeu_si256(tmp.as_mut_ptr().add(16) as *mut __m256i, out2);
                _mm256_storeu_si256(tmp.as_mut_ptr().add(24) as *mut __m256i, out3);
                let rem = self.rows - r_start;
                output[r_start..r_start + rem].copy_from_slice(&tmp[..rem]);
            }
        }
    }

    /// Cache-tiled batched zero-mul AVX2 projection for mini-batch training (`B` samples).
    /// Keeps each 32-row x 128-col weight tile resident in L1 cache across all `B` samples!
    pub fn project_batched_tiled(&self, inputs: &[&[i8]], outputs: &mut [&mut [i32]]) {
        let batch = inputs.len();
        debug_assert_eq!(batch, outputs.len());
        if batch == 0 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let groups = self.cols / 4;
                let mut luts_lo = vec![[0u8; 16]; batch * groups];
                let mut luts_hi = vec![[0u8; 16]; batch * groups];
                for (b, inp) in inputs.iter().enumerate() {
                    let s = b * groups;
                    Self::build_lut4(
                        inp,
                        &mut luts_lo[s..s + groups],
                        &mut luts_hi[s..s + groups],
                    );
                }
                unsafe {
                    self.project_batched_avx2_tiled(batch, groups, &luts_lo, &luts_hi, outputs);
                }
                return;
            }
        }
        for b in 0..batch {
            self.project_lut4_scalar(inputs[b], outputs[b]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn project_batched_avx2_tiled(
        &self,
        batch: usize,
        groups: usize,
        luts_lo: &[[u8; 16]],
        luts_hi: &[[u8; 16]],
        outputs: &mut [&mut [i32]],
    ) {
        let mask0f = _mm256_set1_epi8(0x0F);
        let n_tiles = self.rows_padded / ROW_TILE;

        for tile in 0..n_tiles {
            let tile_base = tile * self.byte_cols * ROW_TILE;
            let r_start = tile * ROW_TILE;
            let full_tile = r_start + ROW_TILE <= self.rows;

            // Initialize batch accumulators for this 32-row tile
            let mut tile_out = vec![[0i32; ROW_TILE]; batch];

            // Outer loop over 16-byte_col blocks (1 KiB weights: 512B pos + 512B neg -> L1 hot!)
            let mut bc_block = 0usize;
            while bc_block < self.byte_cols {
                let bc_end = (bc_block + 16).min(self.byte_cols);

                for b in 0..batch {
                    let lut_base = b * groups;
                    let mut acc_a = _mm256_setzero_si256();
                    let mut acc_b = _mm256_setzero_si256();

                    let mut bc = bc_block;
                    while bc < bc_end {
                        let off = tile_base + bc * ROW_TILE;
                        let p_bytes = _mm256_loadu_si256(
                            self.pos_tiled.as_ptr().add(off) as *const __m256i,
                        );
                        let n_bytes = _mm256_loadu_si256(
                            self.neg_tiled.as_ptr().add(off) as *const __m256i,
                        );

                        let p0 = _mm256_and_si256(p_bytes, mask0f);
                        let p1 = _mm256_and_si256(_mm256_srli_epi16(p_bytes, 4), mask0f);
                        let n0 = _mm256_and_si256(n_bytes, mask0f);
                        let n1 = _mm256_and_si256(_mm256_srli_epi16(n_bytes, 4), mask0f);

                        let g0 = lut_base + bc * 2;
                        let g1 = g0 + 1;

                        let t0_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            luts_lo.as_ptr().add(g0) as *const __m128i,
                        ));
                        let t0_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            luts_hi.as_ptr().add(g0) as *const __m128i,
                        ));
                        let vp0_lo = _mm256_shuffle_epi8(t0_lo, p0);
                        let vp0_hi = _mm256_shuffle_epi8(t0_hi, p0);
                        let vn0_lo = _mm256_shuffle_epi8(t0_lo, n0);
                        let vn0_hi = _mm256_shuffle_epi8(t0_hi, n0);

                        acc_a = _mm256_add_epi16(acc_a, _mm256_unpacklo_epi8(vp0_lo, vp0_hi));
                        acc_b = _mm256_add_epi16(acc_b, _mm256_unpackhi_epi8(vp0_lo, vp0_hi));
                        acc_a = _mm256_sub_epi16(acc_a, _mm256_unpacklo_epi8(vn0_lo, vn0_hi));
                        acc_b = _mm256_sub_epi16(acc_b, _mm256_unpackhi_epi8(vn0_lo, vn0_hi));

                        let t1_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            luts_lo.as_ptr().add(g1) as *const __m128i,
                        ));
                        let t1_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            luts_hi.as_ptr().add(g1) as *const __m128i,
                        ));
                        let vp1_lo = _mm256_shuffle_epi8(t1_lo, p1);
                        let vp1_hi = _mm256_shuffle_epi8(t1_hi, p1);
                        let vn1_lo = _mm256_shuffle_epi8(t1_lo, n1);
                        let vn1_hi = _mm256_shuffle_epi8(t1_hi, n1);

                        acc_a = _mm256_add_epi16(acc_a, _mm256_unpacklo_epi8(vp1_lo, vp1_hi));
                        acc_b = _mm256_add_epi16(acc_b, _mm256_unpackhi_epi8(vp1_lo, vp1_hi));
                        acc_a = _mm256_sub_epi16(acc_a, _mm256_unpacklo_epi8(vn1_lo, vn1_hi));
                        acc_b = _mm256_sub_epi16(acc_b, _mm256_unpackhi_epi8(vn1_lo, vn1_hi));

                        bc += 1;
                    }

                    let ptr = tile_out[b].as_mut_ptr();
                    let cur0 = _mm256_loadu_si256(ptr as *const __m256i);
                    let cur1 = _mm256_loadu_si256(ptr.add(8) as *const __m256i);
                    let cur2 = _mm256_loadu_si256(ptr.add(16) as *const __m256i);
                    let cur3 = _mm256_loadu_si256(ptr.add(24) as *const __m256i);

                    _mm256_storeu_si256(
                        ptr as *mut __m256i,
                        _mm256_add_epi32(cur0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_a))),
                    );
                    _mm256_storeu_si256(
                        ptr.add(8) as *mut __m256i,
                        _mm256_add_epi32(cur1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_b))),
                    );
                    _mm256_storeu_si256(
                        ptr.add(16) as *mut __m256i,
                        _mm256_add_epi32(
                            cur2,
                            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_a, 1)),
                        ),
                    );
                    _mm256_storeu_si256(
                        ptr.add(24) as *mut __m256i,
                        _mm256_add_epi32(
                            cur3,
                            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_b, 1)),
                        ),
                    );
                }

                bc_block = bc_end;
            }

            let rem = if full_tile {
                ROW_TILE
            } else {
                self.rows - r_start
            };
            for b in 0..batch {
                outputs[b][r_start..r_start + rem].copy_from_slice(&tile_out[b][..rem]);
            }
        }
    }

    /// Zero-mul STE backward pass: `dx[c] += sum_r w[r, c] * dy_scaled[r]`
    /// using only sign-selected additions/subtractions directly from the 2-bit bit-planes.
    pub fn project_transpose_f32_add(&self, dy_scaled: &[f32], dx: &mut [f32]) {
        debug_assert_eq!(dy_scaled.len(), self.rows);
        debug_assert_eq!(dx.len(), self.cols);
        let n_tiles = self.rows_padded / ROW_TILE;
        for tile in 0..n_tiles {
            let r_start = tile * ROW_TILE;
            let r_len = (self.rows - r_start).min(ROW_TILE);
            let tile_base = tile * self.byte_cols * ROW_TILE;
            for bc in 0..self.byte_cols {
                let off = tile_base + bc * ROW_TILE;
                let c_base = bc * 8;
                let mut acc = [0.0f32; 8];
                for r_in in 0..r_len {
                    let g = dy_scaled[r_start + r_in];
                    if g == 0.0 {
                        continue;
                    }
                    let mut p = self.pos_tiled[off + r_in];
                    let mut n = self.neg_tiled[off + r_in];
                    while p != 0 {
                        let bit = p.trailing_zeros() as usize;
                        acc[bit] += g;
                        p &= p - 1;
                    }
                    while n != 0 {
                        let bit = n.trailing_zeros() as usize;
                        acc[bit] -= g;
                        n &= n - 1;
                    }
                }
                for k in 0..8 {
                    dx[c_base + k] += acc[k];
                }
            }
        }
    }
}

/// Dense f32 control matrix (for Bottleneck 1 speedup benchmarking).
#[derive(Clone, Debug)]
pub struct DenseF32Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl DenseF32Matrix {
    pub fn from_ternary(t: &TernaryMatrix) -> Self {
        let mut data = vec![0.0f32; t.rows * t.cols];
        for r in 0..t.rows {
            for c in 0..t.cols {
                data[r * t.cols + c] = t.get_weight(r, c) as f32;
            }
        }
        Self {
            rows: t.rows,
            cols: t.cols,
            data,
        }
    }

    pub fn project_f32(&self, input: &[f32], output: &mut [f32]) {
        for (r, out_r) in output.iter_mut().enumerate() {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            let mut sum = 0.0f32;
            for (&w, &x) in row.iter().zip(input.iter()) {
                sum += w * x;
            }
            *out_r = sum;
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

// ============================================================================
// BOTTLENECK 3 FIX: BitNet b1.58 ScaledTernaryLayer (STE + AbsMean + AdamW + Hysteresis)
// ============================================================================

/// Configuration for BitNet b1.58 STE + Importance-Weighted Hysteresis training.
#[derive(Clone, Copy, Debug)]
pub struct TernaryTrainConfig {
    pub lr: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// AbsMean quantization threshold multiplier (`gamma_r = absmean_beta * mean(|W_r|)`).
    pub absmean_beta: f32,
    /// Hysteresis band width (0.0 = disabled; e.g., 0.25 creates deadband `[0.75*g, 1.25*g]`).
    pub hysteresis_band: f32,
    /// Elastic importance anchor penalty (EWC-style protection during sleep/Task B).
    pub importance_lambda: f32,
    /// Extra hysteresis boost scaled by weight importance.
    pub importance_hyst_gain: f32,
    /// Gradient clipping norm per element.
    pub grad_clip: f32,
}

impl Default for TernaryTrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.015,
            weight_decay: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            absmean_beta: 0.55,
            hysteresis_band: 0.20,
            importance_lambda: 0.0,
            importance_hyst_gain: 0.0,
            grad_clip: 2.0,
        }
    }
}

/// Trainable BitNet b1.58 Scaled Ternary Layer.
///
/// - **Inference:** 100% 2-bit bit-plane zero-mul (`TernaryMatrix::project_lut4_simd`)
///   + 1 scalar `row_scale[r]` per output row (`R` scalar multiplies vs `R * C`).
/// - **Training:** Straight-Through Estimator (STE) + AdamW on `shadow_w` + AbsMean
///   adaptive quantization + optional importance-weighted hysteresis for anti-forgetting.
#[derive(Clone, Debug)]
pub struct ScaledTernaryLayer {
    pub rows: usize,
    pub cols: usize,
    pub ternary: TernaryMatrix,
    pub row_scale: Vec<f32>,
    pub bias: Vec<f32>,

    // Training-only state (not required at deployment/inference)
    pub shadow_w: Vec<f32>,
    pub grad_w: Vec<f32>,
    pub grad_b: Vec<f32>,
    pub adam_m: Vec<f32>,
    pub adam_v: Vec<f32>,
    pub adam_mb: Vec<f32>,
    pub adam_vb: Vec<f32>,
    pub adam_t: u32,

    // Continual learning / Sleep consolidation protection
    pub importance: Vec<f32>,
    pub anchor_w: Vec<f32>,
    pub flip_count: u64,
    /// Optional microglia structural pruning mask (1 = pruned permanently to 0; empty = inactive)
    pub prune_mask: Vec<u8>,

    // Reusable buffers to avoid per-call allocation
    xq_buf: Vec<i8>,
    yint_buf: Vec<i32>,
}

impl ScaledTernaryLayer {
    pub fn new(rows: usize, cols: usize, rng: &mut Xorshift64) -> Self {
        let scale = (2.0 / (cols as f32)).sqrt();
        let mut shadow_w = vec![0.0f32; rows * cols];
        for w in &mut shadow_w {
            *w = rng.next_normal_f32() * scale * 0.5;
        }
        let mut layer = Self {
            rows,
            cols,
            ternary: TernaryMatrix::new(rows, cols),
            row_scale: vec![1.0; rows],
            bias: vec![0.0; rows],
            shadow_w,
            grad_w: vec![0.0; rows * cols],
            grad_b: vec![0.0; rows],
            adam_m: vec![0.0; rows * cols],
            adam_v: vec![0.0; rows * cols],
            adam_mb: vec![0.0; rows],
            adam_vb: vec![0.0; rows],
            adam_t: 0,
            importance: vec![0.0; rows * cols],
            anchor_w: vec![0.0; rows * cols],
            flip_count: 0,
            prune_mask: Vec::new(),
            xq_buf: vec![0i8; cols],
            yint_buf: vec![0i32; rows],
        };
        layer.quantize_from_shadow(0.55, 0.0, 0.0, false);
        layer
    }

    /// Packed inference memory in bytes: 2-bit bit-planes + per-row scale + bias.
    pub fn inference_bytes(&self) -> usize {
        self.ternary.packed_bytes() + (self.rows * 2) * std::mem::size_of::<f32>()
    }

    pub fn dense_f32_bytes(&self) -> usize {
        (self.rows * self.cols + self.rows) * std::mem::size_of::<f32>()
    }

    /// Zero-mul 2-bit forward pass (per-token absmax i8 activation quantization + AVX2 LUT-4).
    pub fn forward(&mut self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        let x_absmax = quantize_absmax_i8(x, &mut self.xq_buf);
        self.ternary.project_lut4_simd(&self.xq_buf, &mut self.yint_buf);
        let dequant = x_absmax / 127.0;
        for r in 0..self.rows {
            y[r] = (self.yint_buf[r] as f32) * dequant * self.row_scale[r] + self.bias[r];
        }
    }

    /// Immutable forward using caller-provided scratch buffers (zero-alloc).
    pub fn forward_immut(&self, x: &[f32], y: &mut [f32], xq: &mut [i8], yint: &mut [i32]) {
        let x_absmax = quantize_absmax_i8(x, xq);
        self.ternary.project_lut4_simd(xq, yint);
        let dequant = x_absmax / 127.0;
        for r in 0..self.rows {
            y[r] = (yint[r] as f32) * dequant * self.row_scale[r] + self.bias[r];
        }
    }

    /// Pure zero-mul inference with unscaled integer bit-planes (returns raw i32 logits).
    pub fn forward_i8_raw(&self, xq: &[i8], yint: &mut [i32]) {
        self.ternary.project_lut4_simd(xq, yint);
    }

    pub fn zero_grad(&mut self) {
        self.grad_w.fill(0.0);
        self.grad_b.fill(0.0);
    }

    /// Accumulate STE gradients for one sample (`dy`: upstream grad w.r.t. y).
    /// Optionally writes input gradient into `dx_out`.
    pub fn backward_accumulate(&mut self, x: &[f32], dy: &[f32], mut dx_out: Option<&mut [f32]>) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(dy.len(), self.rows);

        if let Some(ref mut dx) = dx_out {
            let mut dy_scaled = vec![0.0f32; self.rows];
            for r in 0..self.rows {
                dy_scaled[r] = dy[r] * self.row_scale[r];
            }
            self.ternary.project_transpose_f32_add(&dy_scaled, dx);
        }

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && self.cols.is_multiple_of(8);
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.backward_outer_avx2(x, dy);
            }
        } else {
            for r in 0..self.rows {
                let g = dy[r];
                self.grad_b[r] += g;
                if g.abs() < 1e-12 {
                    continue;
                }
                let row_off = r * self.cols;
                let gw = &mut self.grad_w[row_off..row_off + self.cols];
                for c in 0..self.cols {
                    gw[c] += g * x[c];
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn backward_outer_avx2(&mut self, x: &[f32], dy: &[f32]) {
        let cols = self.cols;
        let x_ptr = x.as_ptr();
        for r in 0..self.rows {
            let g = dy[r];
            self.grad_b[r] += g;
            if g.abs() < 1e-9 {
                continue;
            }
            let gv = _mm256_set1_ps(g);
            let gw_ptr = self.grad_w.as_mut_ptr().add(r * cols);
            let mut c = 0usize;
            while c < cols {
                let gw = _mm256_loadu_ps(gw_ptr.add(c));
                let xv = _mm256_loadu_ps(x_ptr.add(c));
                let upd = _mm256_add_ps(gw, _mm256_mul_ps(gv, xv));
                _mm256_storeu_ps(gw_ptr.add(c), upd);
                c += 8;
            }
        }
    }

    /// Accumulate diagonal Fisher/sensitivity importance from current gradients.
    pub fn accumulate_importance(&mut self, scale: f32) {
        for (imp, &g) in self.importance.iter_mut().zip(self.grad_w.iter()) {
            *imp += (g * g) * scale;
        }
    }

    /// Snapshot current `shadow_w` as continual-learning anchor and normalize importance to [0, 1].
    pub fn snapshot_anchor_and_normalize_importance(&mut self) {
        self.anchor_w.copy_from_slice(&self.shadow_w);
        let max_imp = self
            .importance
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        if max_imp > 1e-12 {
            let inv = 1.0 / max_imp;
            for imp in &mut self.importance {
                *imp = (*imp * inv).sqrt().clamp(0.0, 1.0);
            }
        }
    }

    /// Reset Adam moments (e.g., at phase transition) while keeping consolidated weights.
    pub fn reset_optimizer(&mut self) {
        self.adam_m.fill(0.0);
        self.adam_v.fill(0.0);
        self.adam_mb.fill(0.0);
        self.adam_vb.fill(0.0);
        self.adam_t = 0;
    }

    /// Quantize `shadow_w` into 2-bit `ternary` planes with BitNet AbsMean threshold,
    /// optional importance-weighted hysteresis, and per-row learned scale `alpha_r`.
    pub fn quantize_from_shadow(
        &mut self,
        absmean_beta: f32,
        hysteresis_band: f32,
        importance_hyst_gain: f32,
        count_flips: bool,
    ) {
        for r in 0..self.rows {
            let off = r * self.cols;
            let mut sum_abs = 0.0f32;
            for c in 0..self.cols {
                sum_abs += self.shadow_w[off + c].abs();
            }
            let mean_abs = (sum_abs / (self.cols as f32)).max(1e-5);
            let thresh = mean_abs * absmean_beta;

            let mut nz_sum = 0.0f32;
            let mut nz_cnt = 0usize;

            let has_mask = self.prune_mask.len() == self.shadow_w.len();
            for c in 0..self.cols {
                let idx = off + c;
                if has_mask && self.prune_mask[idx] != 0 {
                    if self.ternary.get_weight(r, c) != 0 {
                        self.ternary.set_weight(r, c, 0);
                        if count_flips {
                            self.flip_count += 1;
                        }
                    }
                    continue;
                }
                let w_lat = self.shadow_w[idx];
                let old_q = self.ternary.get_weight(r, c);
                let imp = self.importance[idx];
                let h = hysteresis_band + importance_hyst_gain * imp;

                let new_q = if h > 0.0 {
                    let hi = thresh * (1.0 + h);
                    let lo = thresh * (1.0 - h.min(0.85));
                    match old_q {
                        1 => {
                            if w_lat < -hi {
                                -1
                            } else if w_lat < lo {
                                0
                            } else {
                                1
                            }
                        }
                        -1 => {
                            if w_lat > hi {
                                1
                            } else if w_lat > -lo {
                                0
                            } else {
                                -1
                            }
                        }
                        _ => {
                            if w_lat > hi {
                                1
                            } else if w_lat < -hi {
                                -1
                            } else {
                                0
                            }
                        }
                    }
                } else if w_lat > thresh {
                    1
                } else if w_lat < -thresh {
                    -1
                } else {
                    0
                };

                if new_q != old_q {
                    self.ternary.set_weight(r, c, new_q);
                    if count_flips {
                        self.flip_count += 1;
                    }
                }
                if new_q != 0 {
                    nz_sum += w_lat.abs();
                    nz_cnt += 1;
                }
            }
            self.row_scale[r] = if nz_cnt > 0 {
                (nz_sum / (nz_cnt as f32)).max(1e-4)
            } else {
                mean_abs
            };
        }
    }

    /// AdamW + STE step on `shadow_w` & `bias`, followed by AbsMean/hysteresis requantization.
    pub fn step_adamw_ste(&mut self, batch_size: usize, cfg: &TernaryTrainConfig) {
        let inv_b = 1.0 / (batch_size.max(1) as f32);
        self.adam_t = self.adam_t.saturating_add(1);
        let t = self.adam_t as i32;
        let bc1 = 1.0 - cfg.beta1.powi(t);
        let bc2 = 1.0 - cfg.beta2.powi(t);
        let has_mask = self.prune_mask.len() == self.shadow_w.len();

        for i in 0..self.shadow_w.len() {
            if has_mask && self.prune_mask[i] != 0 {
                self.shadow_w[i] = 0.0;
                continue;
            }
            let mut g = (self.grad_w[i] * inv_b).clamp(-cfg.grad_clip, cfg.grad_clip);
            if cfg.importance_lambda > 0.0 && self.importance[i] > 0.0 {
                g += cfg.importance_lambda * self.importance[i] * (self.shadow_w[i] - self.anchor_w[i]);
            }
            let m = cfg.beta1 * self.adam_m[i] + (1.0 - cfg.beta1) * g;
            let v = cfg.beta2 * self.adam_v[i] + (1.0 - cfg.beta2) * g * g;
            self.adam_m[i] = m;
            self.adam_v[i] = v;
            let m_hat = m / bc1;
            let v_hat = v / bc2;
            self.shadow_w[i] *= 1.0 - cfg.lr * cfg.weight_decay;
            self.shadow_w[i] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
        }

        for r in 0..self.rows {
            let g = (self.grad_b[r] * inv_b).clamp(-cfg.grad_clip, cfg.grad_clip);
            let m = cfg.beta1 * self.adam_mb[r] + (1.0 - cfg.beta1) * g;
            let v = cfg.beta2 * self.adam_vb[r] + (1.0 - cfg.beta2) * g * g;
            self.adam_mb[r] = m;
            self.adam_vb[r] = v;
            let m_hat = m / bc1;
            let v_hat = v / bc2;
            self.bias[r] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
        }

        self.quantize_from_shadow(
            cfg.absmean_beta,
            cfg.hysteresis_band,
            cfg.importance_hyst_gain,
            true,
        );
        self.zero_grad();
    }

    /// Fraction of active non-zero 2-bit ternary weights in `{ -1, +1 }`.
    pub fn nonzero_fraction(&self) -> f64 {
        let mut nz = 0usize;
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.ternary.get_weight(r, c) != 0 {
                    nz += 1;
                }
            }
        }
        (nz as f64) / ((self.rows * self.cols).max(1) as f64)
    }

    /// Subroutine 3 (AURA v3 Genesis): Microglia structural pruning on 2-bit ternary weights.
    /// Permanently removes the lowest-salience `prune_frac` weights per row (combining shadow
    /// magnitude and Fisher/Adam second-moment utility) into exact `0` bit-planes and boosts
    /// surviving row scales cleanly. Returns number of pruned synapses.
    pub fn microglia_prune_ternary(&mut self, prune_frac: f32) -> usize {
        let frac = prune_frac.clamp(0.0, 0.90);
        if self.prune_mask.len() != self.shadow_w.len() {
            self.prune_mask = vec![0u8; self.shadow_w.len()];
        }
        let mut pruned_total = 0usize;
        let mut scores = vec![(0.0f32, 0usize); self.cols];

        for r in 0..self.rows {
            let off = r * self.cols;
            for c in 0..self.cols {
                let idx = off + c;
                // Biological salience: magnitude * (1 + sqrt(grad_energy))
                let sal = self.shadow_w[idx].abs() * (1.0 + self.adam_v[idx].sqrt() * 4.0);
                scores[c] = (sal, c);
            }
            scores.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let k_prune = ((self.cols as f32) * frac).floor() as usize;
            for &(_, c) in scores.iter().take(k_prune) {
                let idx = off + c;
                if self.prune_mask[idx] == 0 {
                    self.prune_mask[idx] = 1;
                    self.shadow_w[idx] = 0.0;
                    self.adam_m[idx] = 0.0;
                    self.adam_v[idx] = 0.0;
                    self.ternary.set_weight(r, c, 0);
                    pruned_total += 1;
                }
            }
        }
        self.quantize_from_shadow(0.45, 0.0, 0.0, false);
        pruned_total
    }
}

// ============================================================================
// BOTTLENECK 1.3 FIX: Zero-Alloc Normalized NLMS FastWeightMemory (AVX2)
// ============================================================================

/// Associative Fast-Weight Memory (`M_t` in Q12 fixed-point) implementing the online Delta Rule:
/// `M_t = gamma * M_{t-1} + eta_norm * (v_t - M_{t-1} k_t) k_t^T`.
#[derive(Clone, Debug)]
pub struct FastWeightMemory {
    pub dim: usize,
    pub weights: Vec<i32>, // Q12 fixed-point
    pub decay_shift: u8,
    pred_buf: Vec<i32>,    // Pre-allocated scratch -> ZERO heap allocs in hot loop
}

impl FastWeightMemory {
    pub fn new(dim: usize, decay_shift: u8) -> Self {
        Self {
            dim,
            weights: vec![0i32; dim * dim],
            decay_shift,
            pred_buf: vec![0i32; dim],
        }
    }

    pub fn reset(&mut self) {
        self.weights.fill(0);
    }

    /// Retrieve association `v_hat = (M * k) >> 12`.
    pub fn query(&self, key: &[i8], out: &mut [i32]) {
        debug_assert_eq!(key.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && self.dim.is_multiple_of(8) {
                unsafe {
                    self.query_avx2(key, out);
                }
                return;
            }
        }
        self.query_scalar(key, out);
    }

    pub fn query_scalar(&self, key: &[i8], out: &mut [i32]) {
        for (r, out_r) in out.iter_mut().enumerate() {
            let row = &self.weights[r * self.dim..(r + 1) * self.dim];
            let mut acc = 0i64;
            for c in 0..self.dim {
                acc += (row[c] as i64) * (key[c] as i64);
            }
            *out_r = (acc >> 12) as i32;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn query_avx2(&self, key: &[i8], out: &mut [i32]) {
        for (r, out_r) in out.iter_mut().enumerate() {
            let row_ptr = self.weights.as_ptr().add(r * self.dim);
            let mut acc_lo = _mm256_setzero_si256();
            let mut acc_hi = _mm256_setzero_si256();
            let mut c = 0usize;
            while c < self.dim {
                // Load 8 weights (i32) and 8 keys (i8 -> i32)
                let w = _mm256_loadu_si256(row_ptr.add(c) as *const __m256i);
                let k64 = _mm_loadl_epi64(key.as_ptr().add(c) as *const __m128i);
                let k32 = _mm256_cvtepi8_epi32(k64);

                // Exact i32 * i32 -> i64 using mul_epi32 on even & odd lanes
                let p_even = _mm256_mul_epi32(w, k32);
                let w_odd = _mm256_srli_epi64(w, 32);
                let k_odd = _mm256_srli_epi64(k32, 32);
                let p_odd = _mm256_mul_epi32(w_odd, k_odd);

                acc_lo = _mm256_add_epi64(acc_lo, p_even);
                acc_hi = _mm256_add_epi64(acc_hi, p_odd);
                c += 8;
            }
            let sum_v = _mm256_add_epi64(acc_lo, acc_hi);
            let mut lanes = [0i64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum_v);
            let total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            *out_r = (total >> 12) as i32;
        }
    }

    /// Exact proposal baseline: allocates `Vec` on every call + unnormalized `>> eta_shift`.
    /// Demonstrates both heap overhead and large-norm divergence / small-norm stall in B1.
    pub fn update_online_naive_proposal(&mut self, key: &[i8], target_val: &[i32], eta_shift: u8) {
        let mut pred = vec![0i32; self.dim];
        self.query_scalar(key, &mut pred);
        for r in 0..self.dim {
            let err = target_val[r].saturating_sub(pred[r]);
            let row = &mut self.weights[r * self.dim..(r + 1) * self.dim];
            for c in 0..self.dim {
                row[c] -= row[c] >> self.decay_shift;
                let delta = (((err as i64) * (key[c] as i64)) >> eta_shift) as i32;
                row[c] = row[c].saturating_add(delta);
            }
        }
    }

    /// Bottleneck 1.3 Fix: Zero-allocation, scale-invariant exact NLMS Delta Rule with AVX2.
    ///
    /// Uses exact Q8-scaled per-row step `scale_q8 = (err << 20) / ((||k||_2^2 + 16) << lr_shift)`
    /// so `||v - M_new k|| <= (1 - 2^{-lr_shift}) ||v - M_old k||` uniformly for ANY key scale
    /// without power-of-2 quantization slowdown or integer overflow.
    pub fn update_online_nlms(&mut self, key: &[i8], target_val: &[i32], lr_shift: u8) {
        debug_assert_eq!(key.len(), self.dim);
        debug_assert_eq!(target_val.len(), self.dim);

        let mut k_norm2: i64 = 16;
        for &k in key {
            let ki = k as i64;
            k_norm2 += ki * ki;
        }
        let denom: i64 = k_norm2 << (lr_shift.min(12) as u32);

        // Reuse pre-allocated pred_buf (zero heap allocation!)
        let mut pred = std::mem::take(&mut self.pred_buf);
        self.query(key, &mut pred);

        let clamp_lim = 1i32 << 22;
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && self.dim.is_multiple_of(8);
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.update_rows_avx2(key, target_val, &pred, denom, clamp_lim);
            }
        } else {
            for r in 0..self.dim {
                let err = target_val[r].saturating_sub(pred[r]) as i64;
                let scale_q8 = ((err << 20) / denom).clamp(-16_000_000, 16_000_000) as i32;
                let row = &mut self.weights[r * self.dim..(r + 1) * self.dim];
                for c in 0..self.dim {
                    if self.decay_shift < 31 {
                        row[c] -= row[c] >> self.decay_shift;
                    }
                    let delta = (scale_q8 * (key[c] as i32)) >> 8;
                    row[c] = row[c].saturating_add(delta).clamp(-clamp_lim, clamp_lim);
                }
            }
        }

        self.pred_buf = pred;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn update_rows_avx2(
        &mut self,
        key: &[i8],
        target_val: &[i32],
        pred: &[i32],
        denom: i64,
        clamp_lim: i32,
    ) {
        let min_v = _mm256_set1_epi32(-clamp_lim);
        let max_v = _mm256_set1_epi32(clamp_lim);
        let decay = self.decay_shift.min(31) as i32;
        let dec_v = _mm_cvtsi32_si128(decay);
        let apply_decay = self.decay_shift < 31;

        for r in 0..self.dim {
            let err = target_val[r].saturating_sub(pred[r]) as i64;
            let scale_q8 = ((err << 20) / denom).clamp(-16_000_000, 16_000_000) as i32;
            let scale_v = _mm256_set1_epi32(scale_q8);
            let row_ptr = self.weights.as_mut_ptr().add(r * self.dim);

            let mut c = 0usize;
            while c < self.dim {
                let w = _mm256_loadu_si256(row_ptr.add(c) as *const __m256i);
                let w_dec = if apply_decay {
                    _mm256_sub_epi32(w, _mm256_sra_epi32(w, dec_v))
                } else {
                    w
                };
                let k64 = _mm_loadl_epi64(key.as_ptr().add(c) as *const __m128i);
                let k32 = _mm256_cvtepi8_epi32(k64);
                let prod = _mm256_mullo_epi32(scale_v, k32);
                let delta = _mm256_srai_epi32(prod, 8);
                let upd = _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_add_epi32(w_dec, delta), min_v),
                    max_v,
                );
                _mm256_storeu_si256(row_ptr.add(c) as *mut __m256i, upd);
                c += 8;
            }
        }
    }
}

// ============================================================================
// Spectral Effective Rank (Roy & Vetterli 2007) via Symmetric Jacobi Solver
// ============================================================================

/// Computes the Roy-Vetterli Shannon effective rank `exp(H(p))` of a sample matrix
/// over a deterministic subspace of size `sub_dim` (<= 64). Zero external dependencies.
pub fn effective_rank_subspace(samples: &[&[f32]], sub_dim: usize) -> f64 {
    let n = samples.len();
    if n < 2 || samples[0].is_empty() {
        return 0.0;
    }
    let full_d = samples[0].len();
    let d = sub_dim.min(full_d).min(64);
    let stride = (full_d / d).max(1);

    let mut mean = vec![0.0f64; d];
    for s in samples {
        for j in 0..d {
            mean[j] += s[(j * stride) % full_d] as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    let mut cov = vec![0.0f64; d * d];
    let inv_n = 1.0 / ((n - 1) as f64);
    for s in samples {
        for i in 0..d {
            let xi = (s[(i * stride) % full_d] as f64) - mean[i];
            let row_off = i * d;
            for j in i..d {
                let xj = (s[(j * stride) % full_d] as f64) - mean[j];
                cov[row_off + j] += xi * xj * inv_n;
            }
        }
    }
    for i in 0..d {
        for j in 0..i {
            cov[i * d + j] = cov[j * d + i];
        }
    }

    // Cyclic Jacobi diagonalization of symmetric d x d covariance matrix
    for _sweep in 0..18 {
        let mut max_off = 0.0f64;
        for p in 0..d {
            for q in (p + 1)..d {
                let a_pq = cov[p * d + q];
                if a_pq.abs() > max_off {
                    max_off = a_pq.abs();
                }
                if a_pq.abs() < 1e-12 {
                    continue;
                }
                let a_pp = cov[p * d + p];
                let a_qq = cov[q * d + q];
                let tau = (a_qq - a_pp) / (2.0 * a_pq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                cov[p * d + p] = a_pp - t * a_pq;
                cov[q * d + q] = a_qq + t * a_pq;
                cov[p * d + q] = 0.0;
                cov[q * d + p] = 0.0;

                for r in 0..d {
                    if r != p && r != q {
                        let a_rp = cov[r * d + p];
                        let a_rq = cov[r * d + q];
                        let nrp = c * a_rp - s * a_rq;
                        let nrq = s * a_rp + c * a_rq;
                        cov[r * d + p] = nrp;
                        cov[p * d + r] = nrp;
                        cov[r * d + q] = nrq;
                        cov[q * d + r] = nrq;
                    }
                }
            }
        }
        if max_off < 1e-10 {
            break;
        }
    }

    let mut trace = 0.0f64;
    let mut eigs = Vec::with_capacity(d);
    for i in 0..d {
        let lam = cov[i * d + i].max(0.0);
        eigs.push(lam);
        trace += lam;
    }
    if trace <= 1e-12 {
        return 0.0;
    }
    let mut entropy = 0.0f64;
    for lam in eigs {
        if lam > 1e-15 {
            let p = lam / trace;
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

/// Convenience wrapper for `LatentVec` slices.
pub fn effective_rank_latents(latents: &[LatentVec], sub_dim: usize) -> f64 {
    let storage: Vec<[f32; LATENT_DIM]> = latents.iter().map(|v| v.to_f32()).collect();
    let refs: Vec<&[f32]> = storage.iter().map(|a| a.as_slice()).collect();
    effective_rank_subspace(&refs, sub_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_lut4_avx2_exact_parity_all_shapes_and_extremes() {
        let mut rng = Xorshift64::new(0x1234_5678_ABCD);
        for &(rows, cols, density) in &[
            (16, 64, 0.30f32),
            (48, 128, 0.67),
            (256, 256, 0.67),
            (256, 512, 1.00),
            (1024, 1024, 0.67),
        ] {
            let naive = NaiveBitplaneMatrix::from_random_density(rows, cols, density, &mut rng);
            let tiled = TernaryMatrix::from_naive(&naive);

            // Check exact 16x compression when rows % 32 == 0
            if rows.is_multiple_of(32) {
                let dense_bytes = rows * cols * 4;
                assert_eq!(dense_bytes / tiled.packed_bytes(), 16);
            }

            // Test both random i8 and extreme -128/+127 vectors
            for mode in 0..3 {
                let input: Vec<i8> = match mode {
                    0 => (0..cols).map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8).collect(),
                    1 => vec![-128i8; cols],
                    _ => vec![127i8; cols],
                };
                let mut ref_out = vec![0i32; rows];
                let mut naive_out = vec![0i32; rows];
                let mut scalar_out = vec![0i32; rows];
                let mut simd_out = vec![0i32; rows];

                naive.project_reference_i32(&input, &mut ref_out);
                naive.project_naive_bitscan(&input, &mut naive_out);
                tiled.project_lut4_scalar(&input, &mut scalar_out);
                tiled.project_lut4_simd(&input, &mut simd_out);

                assert_eq!(ref_out, naive_out, "naive mismatch ({rows}x{cols})");
                assert_eq!(ref_out, scalar_out, "lut4 scalar mismatch ({rows}x{cols})");
                assert_eq!(ref_out, simd_out, "lut4 simd mismatch ({rows}x{cols} mode {mode})");
            }

            // Batched tiled parity (B = 16)
            let batch_inputs: Vec<Vec<i8>> = (0..16)
                .map(|_| (0..cols).map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8).collect())
                .collect();
            let inp_refs: Vec<&[i8]> = batch_inputs.iter().map(|v| v.as_slice()).collect();
            let mut batch_outs: Vec<Vec<i32>> = (0..16).map(|_| vec![0i32; rows]).collect();
            let mut out_refs: Vec<&mut [i32]> = batch_outs.iter_mut().map(|v| v.as_mut_slice()).collect();
            tiled.project_batched_tiled(&inp_refs, &mut out_refs);

            for b in 0..16 {
                let mut expected = vec![0i32; rows];
                naive.project_naive_bitscan(&batch_inputs[b], &mut expected);
                assert_eq!(batch_outs[b], expected, "batched tiled mismatch b={b}");
            }
        }
    }

    #[test]
    fn fast_weight_nlms_contracts_across_extreme_key_scales() {
        for &scale in &[2i8, 25i8, 120i8] {
            let mut fw = FastWeightMemory::new(64, 31); // no decay for pure contraction test
            let mut key = [0i8; 64];
            for (i, k) in key.iter_mut().enumerate() {
                *k = if i % 2 == 0 { scale } else { -scale };
            }
            let target: Vec<i32> = (0..64).map(|i| ((i as i32) - 32) * 100).collect();

            let mut pred = vec![0i32; 64];
            fw.query(&key, &mut pred);
            let mut prev_err: i64 = target.iter().zip(&pred).map(|(t, p)| (t - p).abs() as i64).sum();

            for _step in 0..12 {
                fw.update_online_nlms(&key, &target, 1);
                fw.query(&key, &mut pred);
                let err: i64 = target.iter().zip(&pred).map(|(t, p)| (t - p).abs() as i64).sum();
                assert!(
                    err <= prev_err,
                    "NLMS failed to contract at scale={scale}: {err} > {prev_err}"
                );
                prev_err = err;
            }
            assert!(prev_err < 500, "NLMS did not converge at scale={scale}: err={prev_err}");
        }
    }

    #[test]
    fn scaled_ternary_ste_breaks_linear_separability_cleanly() {
        let mut rng = Xorshift64::new(0xBEEF);
        let mut layer = ScaledTernaryLayer::new(8, 64, &mut rng);
        let cfg = TernaryTrainConfig {
            lr: 0.04,
            hysteresis_band: 0.15,
            ..Default::default()
        };

        // Create 8 random orthogonal-ish binary sign targets
        let mut teacher = vec![0.0f32; 8 * 64];
        for w in &mut teacher {
            *w = rng.next_normal_f32();
        }
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..128 {
            let x: Vec<f32> = (0..64).map(|_| rng.next_normal_f32()).collect();
            let mut y = vec![0.0f32; 8];
            for r in 0..8 {
                let mut s = 0.0f32;
                for c in 0..64 {
                    s += teacher[r * 64 + c] * x[c];
                }
                y[r] = if s >= 0.0 { 1.0 } else { -1.0 };
            }
            xs.push(x);
            ys.push(y);
        }

        let mut pred = vec![0.0f32; 8];
        let mut dy = vec![0.0f32; 8];
        for _epoch in 0..35 {
            for b_start in (0..128).step_by(16) {
                for i in b_start..b_start + 16 {
                    layer.forward(&xs[i], &mut pred);
                    for r in 0..8 {
                        let margin = pred[r] * ys[i][r];
                        dy[r] = if margin < 1.5 { -ys[i][r] } else { 0.0 };
                    }
                    layer.backward_accumulate(&xs[i], &dy, None);
                }
                layer.step_adamw_ste(16, &cfg);
            }
        }

        let mut correct = 0usize;
        for i in 0..128 {
            layer.forward(&xs[i], &mut pred);
            for r in 0..8 {
                if (pred[r] >= 0.0) == (ys[i][r] >= 0.0) {
                    correct += 1;
                }
            }
        }
        let acc = (correct as f64) / (128.0 * 8.0);
        assert!(acc > 0.90, "ScaledTernaryLayer STE accuracy too low: {acc:.3}");
    }

    #[test]
    fn effective_rank_distinguishes_collapsed_from_isotropic() {
        let mut rng = Xorshift64::new(77);
        let collapsed: Vec<[f32; 32]> = (0..64)
            .map(|i| {
                let a = (i as f32) * 0.1;
                [a; 32]
            })
            .collect();
        let iso: Vec<[f32; 32]> = (0..128)
            .map(|_| std::array::from_fn(|_| rng.next_normal_f32()))
            .collect();
        let c_refs: Vec<&[f32]> = collapsed.iter().map(|a| a.as_slice()).collect();
        let i_refs: Vec<&[f32]> = iso.iter().map(|a| a.as_slice()).collect();

        let er_c = effective_rank_subspace(&c_refs, 32);
        let er_i = effective_rank_subspace(&i_refs, 32);
        assert!(er_c < 1.5, "collapsed erank should be ~1, got {er_c}");
        assert!(er_i > 20.0, "isotropic erank should be high, got {er_i}");
    }
}
