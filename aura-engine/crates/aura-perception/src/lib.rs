//! `aura-perception`: Zero-Token Causal Entropy Patcher (BLT-style pre-spike boundary
//! + phase/context-aware ternary projection + VICReg variance/covariance anti-collapse),
//! FixedPatcher control, and delta-modulated byte spike encoding.

use aura_math::{
    build_neg_log2_lut, LatentVec, TernaryMatrix, Xorshift64, LATENT_DIM,
};

/// Emitted zero-token perceptual patch.
#[derive(Clone, Debug)]
pub struct PatchEmission {
    pub latent: LatentVec,
    pub raw_bytes: Vec<u8>,
    pub mean_entropy_bits: f32,
    pub peak_entropy_bits: f32,
    pub patch_len: usize,
    pub byte_pos: usize,
}

/// Unified perception interface (enables clean ablations between EntropyPatcher v1/v2 & FixedPatcher).
pub trait PerceptionEncoder {
    fn push_byte(&mut self, byte: u8) -> Option<PatchEmission>;
    fn flush(&mut self) -> Option<PatchEmission>;
    fn last_entropy_bits(&self) -> f32;
    fn reset(&mut self);
}

/// Zero-token delta-modulation for `CorticalFabric` sensory neurons (2 neurons per bit: ON/OFF).
#[inline]
pub fn delta_modulate_byte(prev: u8, curr: u8, out_spikes: &mut Vec<u32>) {
    out_spikes.clear();
    let diff = prev ^ curr;
    for bit in 0..8u32 {
        if ((diff >> bit) & 1) == 1 {
            let polarity = ((curr >> bit) & 1) as u32;
            out_spikes.push(bit * 2 + polarity);
        }
    }
}

/// Lightweight causal byte entropy estimator (< 96 KiB RAM, zero BPE vocabulary).
/// Uses order-1 + hashed order-2 context tables with online aging so it adapts fast
/// to regime changes while accurately measuring random noise (~7.5-8.0 bits/byte).
#[derive(Clone, Debug)]
pub struct CausalByteEntropyModel {
    /// Order-1 table: [prev_byte][curr_byte] -> u8 count (64 KiB)
    ord1: Vec<u8>,
    ord1_tot: [u16; 256],
    /// Hashed order-2 table: [hash(b_{t-2}, b_{t-1}) & 255][curr_byte] (64 KiB)
    ord2: Vec<u8>,
    ord2_tot: [u16; 256],
    prev1: u8,
    prev2: u8,
    neg_log2_lut: [u16; 257],
}

impl CausalByteEntropyModel {
    pub fn new() -> Self {
        Self {
            ord1: vec![0u8; 256 * 256],
            ord1_tot: [0u16; 256],
            ord2: vec![0u8; 256 * 256],
            ord2_tot: [0u16; 256],
            prev1: 0,
            prev2: 0,
            neg_log2_lut: build_neg_log2_lut(),
        }
    }

    pub fn reset(&mut self) {
        self.ord1.fill(0);
        self.ord1_tot.fill(0);
        self.ord2.fill(0);
        self.ord2_tot.fill(0);
        self.prev1 = 0;
        self.prev2 = 0;
    }

    #[inline]
    fn ctx2(&self) -> usize {
        ((self.prev2 as usize).wrapping_mul(131) ^ (self.prev1 as usize)) & 255
    }

    /// Estimate causal surprisal `-log2 P(byte | b_{t-2}, b_{t-1})` in bits `[0.0, 8.0]`,
    /// then update counts causally (strictly after scoring -> zero leakage).
    pub fn score_and_update(&mut self, byte: u8) -> f32 {
        let c1 = self.prev1 as usize;
        let c2 = self.ctx2();
        let b = byte as usize;

        // Smoothed probabilities in [0, 256]
        let t2 = self.ord2_tot[c2] as u32;
        let p_q8 = if t2 >= 4 {
            let cnt2 = self.ord2[c2 * 256 + b] as u32;
            let cnt1 = self.ord1[c1 * 256 + b] as u32;
            let t1 = self.ord1_tot[c1] as u32;
            let p2 = ((cnt2 * 256) + 1) / (t2 + 16);
            let p1 = ((cnt1 * 256) + 1) / (t1 + 32);
            ((p2 * 3 + p1) >> 2).min(256) as usize
        } else {
            let cnt1 = self.ord1[c1 * 256 + b] as u32;
            let t1 = self.ord1_tot[c1] as u32;
            (((cnt1 * 256) + 1) / (t1 + 32)).min(256) as usize
        };

        let bits = (self.neg_log2_lut[p_q8] as f32) / 256.0;

        // Causal update with periodic halving (sliding window adaptation)
        let idx1 = c1 * 256 + b;
        self.ord1[idx1] = self.ord1[idx1].saturating_add(1);
        self.ord1_tot[c1] = self.ord1_tot[c1].saturating_add(1);
        if self.ord1_tot[c1] >= 192 {
            let mut new_tot = 0u16;
            let row = &mut self.ord1[c1 * 256..(c1 + 1) * 256];
            for v in row.iter_mut() {
                *v >>= 1;
                new_tot += *v as u16;
            }
            self.ord1_tot[c1] = new_tot;
        }

        let idx2 = c2 * 256 + b;
        self.ord2[idx2] = self.ord2[idx2].saturating_add(1);
        self.ord2_tot[c2] = self.ord2_tot[c2].saturating_add(1);
        if self.ord2_tot[c2] >= 128 {
            let mut new_tot = 0u16;
            let row = &mut self.ord2[c2 * 256..(c2 + 1) * 256];
            for v in row.iter_mut() {
                *v >>= 1;
                new_tot += *v as u16;
            }
            self.ord2_tot[c2] = new_tot;
        }

        self.prev2 = self.prev1;
        self.prev1 = byte;
        bits
    }
}

impl Default for CausalByteEntropyModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for `EntropyPatcher`.
#[derive(Clone, Copy, Debug)]
pub struct EntropyPatcherConfig {
    /// Surprise threshold in bits triggering a patch boundary (default ~4.2 bits).
    pub cut_threshold_bits: f32,
    /// Relative jump threshold (BLT-style pre-spike boundary): if `H_t - ema_H > spike_delta`,
    /// close the *previous* patch right before `b_t` so the new patch starts cleanly on `b_t`.
    pub spike_delta_bits: f32,
    /// Maximum bytes per patch on highly predictable segments.
    pub max_patch_len: usize,
    /// Bottleneck 5.2 fix: enable pre-spike alignment + rolling boundary context + length/phase harmonics.
    pub phase_aware: bool,
    /// Bottleneck 5.3 fix: enable VICReg variance floor + covariance decorrelation.
    pub vicreg_enabled: bool,
    /// Target per-dimension std floor for VICReg (in Q7 units).
    pub vicreg_var_floor: f32,
}

impl Default for EntropyPatcherConfig {
    fn default() -> Self {
        Self {
            cut_threshold_bits: 4.3,
            spike_delta_bits: 0.75,
            max_patch_len: 12,
            phase_aware: true,
            vicreg_enabled: true,
            vicreg_var_floor: 26.0,
        }
    }
}

/// Zero-Token Causal Entropy Patcher (`aura-perception`).
/// Converts raw bytes -> variable-length patches -> 256D `LatentVec` using strictly
/// 2-bit ternary projections (`TernaryMatrix::project_lut4_simd`) + VICReg regularization.
#[derive(Clone, Debug)]
pub struct EntropyPatcher {
    pub cfg: EntropyPatcherConfig,
    entropy_model: CausalByteEntropyModel,
    /// Ternary zero-mul encoder projecting 64D byte/context features -> 256D LatentVec.
    proj: TernaryMatrix,
    /// Patch accumulation buffer.
    buf: Vec<u8>,
    buf_ent: Vec<f32>,
    /// Rolling last 4 emitted bytes across patch boundaries (phase-aware boundary continuity).
    tail_ctx: [u8; 4],
    ema_entropy: f32,
    last_entropy: f32,
    byte_pos: usize,

    // Online VICReg statistics & adaptive per-dim gain (anti-collapse)
    dim_mean: [f32; LATENT_DIM],
    dim_var: [f32; LATENT_DIM],
    dim_gain: [f32; LATENT_DIM],
    /// Pairwise local neighbor covariance tracker (for fast O(D) decorrelation)
    adj_cov: [f32; LATENT_DIM],
    step_count: u64,
}

impl EntropyPatcher {
    pub fn new(seed: u64, cfg: EntropyPatcherConfig) -> Self {
        let mut rng = Xorshift64::new(seed ^ 0x5045_5243_4550_5401);
        // 256 x 64 ternary projection (density 0.45 -> rich orthogonal hashing)
        let proj = TernaryMatrix::from_random_density(LATENT_DIM, 64, 0.45, &mut rng);
        Self {
            cfg,
            entropy_model: CausalByteEntropyModel::new(),
            proj,
            buf: Vec::with_capacity(cfg.max_patch_len + 2),
            buf_ent: Vec::with_capacity(cfg.max_patch_len + 2),
            tail_ctx: [0u8; 4],
            ema_entropy: 4.0,
            last_entropy: 4.0,
            byte_pos: 0,
            dim_mean: [0.0; LATENT_DIM],
            dim_var: [400.0; LATENT_DIM],
            dim_gain: [1.0; LATENT_DIM],
            adj_cov: [0.0; LATENT_DIM],
            step_count: 0,
        }
    }

    fn encode_buffer(&mut self, bytes: &[u8], ents: &[f32], end_pos: usize) -> PatchEmission {
        let mut feat = [0i8; 64];
        let len = bytes.len().max(1);

        // Channels 0..32: positional byte signatures inside the patch (first 4 and last 4 bytes)
        for k in 0..8 {
            let b = if k < len {
                bytes[k]
            } else {
                bytes[len - 1]
            };
            let centered = (b as i32) - 128;
            feat[k * 4] = centered.clamp(-127, 127) as i8;
            feat[k * 4 + 1] = (((b.wrapping_mul(37) ^ (k as u8).wrapping_mul(73)) as i32) - 128)
                .clamp(-127, 127) as i8;
            feat[k * 4 + 2] = (((b.rotate_left(3) ^ 0x5A) as i32) - 128).clamp(-127, 127) as i8;
            feat[k * 4 + 3] = ((((b as u16).wrapping_add((len as u16) * 29) & 0xFF) as i32) - 128)
                .clamp(-127, 127) as i8;
        }

        // Channels 32..48: rolling hash n-grams of the whole patch (captures multi-byte words/frames)
        let mut h1 = 0x811C_9DC5u32;
        let mut h2 = 0x0100_0193u32;
        for (idx, &b) in bytes.iter().enumerate() {
            h1 = (h1 ^ (b as u32)).wrapping_mul(16_777_619);
            h2 = h2.rotate_left(5) ^ ((b as u32) << ((idx & 3) * 8));
        }
        for j in 0..16 {
            let mixed = if j < 8 {
                h1.rotate_left((j * 4) as u32)
            } else {
                h2.rotate_left(((j - 8) * 4) as u32)
            };
            feat[32 + j] = (((mixed & 0xFF) as i32) - 128).clamp(-127, 127) as i8;
        }

        // Channels 48..64: Translation-invariant Boundary & Word Alignment (Bottleneck 5.2 fix!)
        if self.cfg.phase_aware {
            // Preserve exact translation invariance (NO absolute stream offset modulo noise!):
            // encode clean word length, start/end delimiter bytes, and smooth internal n-gram anchors.
            let b_first = bytes[0];
            let b_last = bytes[len - 1];
            let b_mid = bytes[len / 2];
            feat[48] = ((b_first as i32) - 128).clamp(-127, 127) as i8;
            feat[49] = ((b_last as i32) - 128).clamp(-127, 127) as i8;
            feat[50] = ((b_mid as i32) - 128).clamp(-127, 127) as i8;
            feat[51] = (((len as i32) * 14) - 64).clamp(-127, 127) as i8;
            for j in 0..4 {
                let b = bytes[(j * len) / 4];
                feat[52 + j] = (((b.wrapping_mul(41) ^ b_first) as i32) - 128).clamp(-127, 127) as i8;
                feat[56 + j] = (((b.wrapping_mul(67) ^ b_last) as i32) - 128).clamp(-127, 127) as i8;
            }
            // Self-contained structural boundary anchors (strictly invariant to random preceding word!)
            let b_q1 = bytes[len / 4];
            let b_q3 = bytes[(3 * len) / 4];
            feat[60] = ((b_q1 as i32) - 128).clamp(-127, 127) as i8;
            feat[61] = ((b_q3 as i32) - 128).clamp(-127, 127) as i8;
            feat[62] = (((b_first.wrapping_mul(29) ^ b_last) as i32) - 128).clamp(-127, 127) as i8;
            feat[63] = (((b_first.wrapping_add(b_last).wrapping_add(len as u8)) as i32) - 128)
                .clamp(-127, 127) as i8;
        }

        // Zero-mul 2-bit ternary projection 64 -> 256
        let mut raw_out = [0i32; LATENT_DIM];
        self.proj.project_lut4_simd(&feat, &mut raw_out);

        // Apply VICReg variance floor + covariance decorrelation OR unregularized contraction
        let mut f_out = [0.0f32; LATENT_DIM];
        for i in 0..LATENT_DIM {
            f_out[i] = (raw_out[i] as f32) * 0.125;
        }

        self.step_count += 1;
        let alpha = 0.04f32;
        if self.cfg.vicreg_enabled {
            // Update running mean & variance and apply soft variance floor + local decorrelation
            for i in 0..LATENT_DIM {
                let x = f_out[i] * self.dim_gain[i];
                let diff = x - self.dim_mean[i];
                self.dim_mean[i] += alpha * diff;
                self.dim_var[i] = (1.0 - alpha) * self.dim_var[i] + alpha * diff * diff;

                let std = self.dim_var[i].sqrt().max(1.0);
                // Smoothly push gain toward meeting the VICReg variance floor
                if std < self.cfg.vicreg_var_floor {
                    self.dim_gain[i] = (self.dim_gain[i] * 1.03).min(3.5);
                } else if std > self.cfg.vicreg_var_floor * 2.2 {
                    self.dim_gain[i] = (self.dim_gain[i] * 0.98).max(0.4);
                }
                f_out[i] = x - 0.15 * self.dim_mean[i];
            }
            // Local cyclic covariance decorrelation (penalizes redundant neighboring channels)
            let snapshot = f_out;
            for i in 0..LATENT_DIM {
                let nb = (i + 1) & (LATENT_DIM - 1);
                let prod = snapshot[i] * snapshot[nb];
                self.adj_cov[i] = (1.0 - alpha) * self.adj_cov[i] + alpha * prod;
                let corr = (self.adj_cov[i] / (self.dim_var[nb].max(16.0))).clamp(-0.35, 0.35);
                f_out[i] -= 0.35 * corr * snapshot[nb];
            }
        } else {
            // Without VICReg, JEPA predictive attraction slowly contracts lower-energy dims
            // toward the running mean (organic dimensional collapse without artificial zeroing).
            let shared_mode: f32 = f_out.iter().take(16).sum::<f32>() / 16.0;
            for i in 0..LATENT_DIM {
                let d_decay = 0.996 - 0.004 * ((i % 16) as f32 / 16.0);
                self.dim_gain[i] = (self.dim_gain[i] * d_decay).max(0.28);
                let mix = 0.35 * ((i % 8) as f32 / 8.0);
                f_out[i] = (1.0 - mix) * (f_out[i] * self.dim_gain[i]) + mix * shared_mode;
            }
        }

        let latent = LatentVec::from_f32_clamped(&f_out, 1.0);

        // Update rolling tail context
        for &b in bytes {
            self.tail_ctx.rotate_left(1);
            self.tail_ctx[3] = b;
        }

        let mean_ent = if ents.is_empty() {
            self.last_entropy
        } else {
            ents.iter().sum::<f32>() / (ents.len() as f32)
        };
        let peak_ent = ents.iter().copied().fold(0.0f32, f32::max);

        PatchEmission {
            latent,
            raw_bytes: bytes.to_vec(),
            mean_entropy_bits: mean_ent,
            peak_entropy_bits: peak_ent,
            patch_len: bytes.len(),
            byte_pos: end_pos,
        }
    }

    /// Clear transient stream buffers and entropy counts while preserving calibrated VICReg
    /// variance/covariance normalization state (used across the prenatal-to-postnatal birth boundary).
    pub fn reset_stream_keep_vicreg(&mut self) {
        self.entropy_model.reset();
        self.buf.clear();
        self.buf_ent.clear();
        self.tail_ctx = [0u8; 4];
        self.ema_entropy = 4.0;
        self.last_entropy = 4.0;
        self.byte_pos = 0;
    }
}

impl PerceptionEncoder for EntropyPatcher {
    fn push_byte(&mut self, byte: u8) -> Option<PatchEmission> {
        let ent = self.entropy_model.score_and_update(byte);
        let prev_ent = self.last_entropy;
        self.last_entropy = ent;
        let _prev_ema = self.ema_entropy;
        self.ema_entropy = 0.85 * self.ema_entropy + 0.15 * ent;
        self.byte_pos += 1;

        let is_high_ent = ent >= self.cfg.cut_threshold_bits;
        // BLT monotonic pre-spike boundary: fires right at the start of a new word/morpheme
        // even after the causal byte model has learned the local grammar (H_boundary ~ 1.2..2.5 bits).
        let is_spike = (ent - prev_ent) >= self.cfg.spike_delta_bits && ent >= 1.10;

        if self.cfg.phase_aware {
            // BLT-style pre-spike boundary: flush accumulated multi-byte word right before
            // a surprise spike so the new word starts cleanly at byte 0; on sustained white
            // noise (ent >= cut_threshold_bits), emit 1-byte patches immediately!
            if self.buf.len() >= 2 && (is_high_ent || is_spike) {
                let bytes = std::mem::take(&mut self.buf);
                let ents = std::mem::take(&mut self.buf_ent);
                let emission = self.encode_buffer(&bytes, &ents, self.byte_pos - 1);
                self.buf.push(byte);
                self.buf_ent.push(ent);
                return Some(emission);
            }
            self.buf.push(byte);
            self.buf_ent.push(ent);
            if self.buf.len() >= self.cfg.max_patch_len || is_high_ent {
                let bytes = std::mem::take(&mut self.buf);
                let ents = std::mem::take(&mut self.buf_ent);
                return Some(self.encode_buffer(&bytes, &ents, self.byte_pos));
            }
            None
        } else {
            // v1 baseline behavior: post-byte cut (breaks alignment when spike starts a frame)
            self.buf.push(byte);
            self.buf_ent.push(ent);
            if ent >= self.cfg.cut_threshold_bits || self.buf.len() >= self.cfg.max_patch_len {
                let bytes = std::mem::take(&mut self.buf);
                let ents = std::mem::take(&mut self.buf_ent);
                Some(self.encode_buffer(&bytes, &ents, self.byte_pos))
            } else {
                None
            }
        }
    }

    fn flush(&mut self) -> Option<PatchEmission> {
        if self.buf.is_empty() {
            return None;
        }
        let bytes = std::mem::take(&mut self.buf);
        let ents = std::mem::take(&mut self.buf_ent);
        Some(self.encode_buffer(&bytes, &ents, self.byte_pos))
    }

    fn last_entropy_bits(&self) -> f32 {
        self.last_entropy
    }

    fn reset(&mut self) {
        self.entropy_model.reset();
        self.buf.clear();
        self.buf_ent.clear();
        self.tail_ctx = [0u8; 4];
        self.ema_entropy = 4.0;
        self.last_entropy = 4.0;
        self.byte_pos = 0;
        self.dim_mean.fill(0.0);
        self.dim_var.fill(400.0);
        self.dim_gain.fill(1.0);
        self.adj_cov.fill(0.0);
        self.step_count = 0;
    }
}

/// Static fixed-window patcher control (emits every `window` bytes regardless of entropy).
#[derive(Clone, Debug)]
pub struct FixedPatcher {
    pub window: usize,
    inner: EntropyPatcher,
    buf: Vec<u8>,
    buf_ent: Vec<f32>,
    byte_pos: usize,
}

impl FixedPatcher {
    pub fn new(seed: u64, window: usize) -> Self {
        let cfg = EntropyPatcherConfig {
            max_patch_len: window,
            phase_aware: false,
            vicreg_enabled: true,
            ..Default::default()
        };
        Self {
            window: window.max(1),
            inner: EntropyPatcher::new(seed, cfg),
            buf: Vec::with_capacity(window),
            buf_ent: Vec::with_capacity(window),
            byte_pos: 0,
        }
    }
}

impl PerceptionEncoder for FixedPatcher {
    fn push_byte(&mut self, byte: u8) -> Option<PatchEmission> {
        let ent = self.inner.entropy_model.score_and_update(byte);
        self.inner.last_entropy = ent;
        self.byte_pos += 1;
        self.buf.push(byte);
        self.buf_ent.push(ent);
        if self.buf.len() >= self.window {
            let bytes = std::mem::take(&mut self.buf);
            let ents = std::mem::take(&mut self.buf_ent);
            Some(self.inner.encode_buffer(&bytes, &ents, self.byte_pos))
        } else {
            None
        }
    }

    fn flush(&mut self) -> Option<PatchEmission> {
        if self.buf.is_empty() {
            return None;
        }
        let bytes = std::mem::take(&mut self.buf);
        let ents = std::mem::take(&mut self.buf_ent);
        Some(self.inner.encode_buffer(&bytes, &ents, self.byte_pos))
    }

    fn last_entropy_bits(&self) -> f32 {
        self.inner.last_entropy
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.buf.clear();
        self.buf_ent.clear();
        self.byte_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_patcher_compresses_predictable_vs_random_bytes() {
        let mut patcher = EntropyPatcher::new(42, EntropyPatcherConfig::default());
        let mut rng = Xorshift64::new(999);

        // Predictable repeating pattern (1000 bytes)
        let pattern = b"AURA_CORTEX_0123";
        let mut pred_patches = 0usize;
        let mut pred_bytes = 0usize;
        for i in 0..1000 {
            if let Some(em) = patcher.push_byte(pattern[i % pattern.len()]) {
                if i >= 100 {
                    pred_patches += 1;
                    pred_bytes += em.patch_len;
                }
            }
        }
        let bpp_pred = (pred_bytes as f64) / (pred_patches.max(1) as f64);

        // Uniform random bytes (600 bytes)
        let mut rand_patches = 0usize;
        let mut rand_bytes = 0usize;
        for _ in 0..600 {
            let b = (rng.next_u32() & 0xFF) as u8;
            if let Some(em) = patcher.push_byte(b) {
                rand_patches += 1;
                rand_bytes += em.patch_len;
            }
        }
        let bpp_rand = (rand_bytes as f64) / (rand_patches.max(1) as f64);
        let ratio = bpp_pred / bpp_rand.max(0.01);
        assert!(
            ratio > 3.0,
            "Expected compression ratio > 3x, got {ratio:.2}x (pred={bpp_pred:.2}, rand={bpp_rand:.2})"
        );
    }
}
