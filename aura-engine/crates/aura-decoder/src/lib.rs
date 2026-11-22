//! `aura-decoder`: Bottleneck 6 Fix — Zero-Token Latent-to-Byte Motor Decoder &
//! End-to-End Trainable AURA v2 Byte Engine.
//!
//! Combines:
//! - `EntropyPatcher` (zero-token dynamic patching -> compressed patch latent)
//! - 4-byte sub-patch learned embeddings (64D) + patch latent projection (64D) -> 128D
//! - 2-bit BitNet b1.58 `ScaledTernaryLayer(128 -> 128)` (zero-mul AVX2 LUT-4)
//! - Multi-timescale Selective SSM (128D) trained with Blelloch O(log L) parallel prefix scan
//! - 2-bit BitNet b1.58 Motor Byte Head `ScaledTernaryLayer(128 -> 256)` (0 BPE tokens!)
//! - Online domain-shift adaptation via Fast-Weight Delta-Rule residual + HippocampusGraph
//!   (leaves slow 2-bit weights 100% intact during unseen domain shift!).

use aura_cortex::{ssm_backward_parallel_scan, ssm_forward_parallel_scan};
use aura_genesis::PrenatalCurriculum;
use aura_hippocampus::{Episode, EpisodicHippocampus, HippocampusGraph};
use aura_math::{
    effective_rank_subspace, LatentVec, ScaledTernaryLayer, TernaryTrainConfig, Xorshift64,
};
use aura_perception::{EntropyPatcher, EntropyPatcherConfig, PerceptionEncoder};

pub const BYTE_EMB_DIM: usize = 12;
pub const CTX_BYTES: usize = 6;
pub const PATCH_CTX_DIM: usize = 56;
pub const MOTOR_IN_DIM: usize = CTX_BYTES * BYTE_EMB_DIM + PATCH_CTX_DIM; // 128
pub const MOTOR_HID_DIM: usize = 128;
pub const BYTE_VOCAB: usize = 256;

/// Training summary returned after end-to-end byte-level training.
#[derive(Clone, Debug)]
pub struct TrainStats {
    pub initial_train_bpb: f64,
    pub final_train_bpb: f64,
    pub eval_bpb: f64,
    pub compression_ratio: f64,
    pub effective_rank: f64,
    pub train_bytes_per_sec: f64,
    pub packed_ternary_bytes: usize,
    pub dense_f32_bytes: usize,
    pub total_flips: u64,
}

/// Extended AURA v3 Genesis training telemetry (adds early epoch-1 eval BPB & structural sparsity).
#[derive(Clone, Debug)]
pub struct GenesisByteStats {
    pub base: TrainStats,
    pub early_eval_bpb: f64,
    pub nonzero_peak_frac: f64,
    pub nonzero_final_frac: f64,
    pub pruned_synapses: usize,
}

/// Classical statistical baseline: smoothed Order-1 Bigram Byte Model (256 x 256).
#[derive(Clone, Debug)]
pub struct BigramByteBaseline {
    counts: Vec<u32>,
    totals: [u32; 256],
    alpha: f64,
}

impl BigramByteBaseline {
    pub fn new(alpha: f64) -> Self {
        Self {
            counts: vec![0u32; 256 * 256],
            totals: [0u32; 256],
            alpha,
        }
    }

    pub fn fit(&mut self, data: &[u8]) {
        let mut prev = 0usize;
        for &b in data {
            let bi = b as usize;
            self.counts[prev * 256 + bi] += 1;
            self.totals[prev] += 1;
            prev = bi;
        }
    }

    pub fn eval_bpb(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 8.0;
        }
        let mut prev = 0usize;
        let mut total_bits = 0.0f64;
        for &b in data {
            let bi = b as usize;
            let c = self.counts[prev * 256 + bi] as f64;
            let t = self.totals[prev] as f64;
            let p = (c + self.alpha) / (t + 256.0 * self.alpha);
            total_bits += -p.log2();
            prev = bi;
        }
        total_bits / (data.len() as f64)
    }
}

/// Softmax cross-entropy in nats + probability vector (numerically stable).
#[inline]
fn softmax_ce_probs(logits: &[f32; BYTE_VOCAB], target: u8, probs: &mut [f32; BYTE_VOCAB]) -> f32 {
    let mut max_l = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_l {
            max_l = l;
        }
    }
    let mut sum_exp = 0.0f32;
    for i in 0..BYTE_VOCAB {
        let e = (logits[i] - max_l).exp();
        probs[i] = e;
        sum_exp += e;
    }
    let inv = 1.0 / sum_exp.max(1e-12);
    for p in probs.iter_mut() {
        *p *= inv;
    }
    let pt = probs[target as usize].max(1e-9);
    -pt.ln()
}

/// Zero-Token End-to-End Trainable AURA v2 Byte Engine (`aura-decoder`).
#[derive(Clone, Debug)]
pub struct AuraByteModel {
    pub patcher: EntropyPatcher,
    /// Learned compact byte embeddings (256 x 16)
    pub byte_emb: Vec<f32>,
    emb_grad: Vec<f32>,
    emb_m: Vec<f32>,
    emb_v: Vec<f32>,
    emb_t: u32,

    /// 2-bit BitNet b1.58 cortical projection (128 -> 128)
    pub l1: ScaledTernaryLayer,
    /// Multi-timescale Selective SSM decay factors alpha[0..128]
    pub ssm_alpha: [f32; MOTOR_HID_DIM],
    /// 2-bit BitNet b1.58 motor byte head (128 -> 256)
    pub l2: ScaledTernaryLayer,

    // Online adaptation components (frozen during pre-training, active during domain shift!)
    pub fast_head: Vec<f32>,        // [256 * 128] fast-weight delta residual (0 at init)
    pub fast_bias: [f32; BYTE_VOCAB],
    pub hippocampus: HippocampusGraph,
}

impl AuraByteModel {
    pub fn new(seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed ^ 0xDEC0_DE00_0006);
        let patcher = EntropyPatcher::new(
            seed ^ 0xA001,
            EntropyPatcherConfig {
                cut_threshold_bits: 4.3,
                spike_delta_bits: 1.5,
                max_patch_len: 8,
                phase_aware: true,
                vicreg_enabled: true,
                vicreg_var_floor: 24.0,
            },
        );
        let mut byte_emb = vec![0.0f32; BYTE_VOCAB * BYTE_EMB_DIM];
        for v in &mut byte_emb {
            *v = rng.next_normal_f32() * 0.25;
        }
        let l1 = ScaledTernaryLayer::new(MOTOR_HID_DIM, MOTOR_IN_DIM, &mut rng);
        let l2 = ScaledTernaryLayer::new(BYTE_VOCAB, MOTOR_HID_DIM, &mut rng);

        // Log-spaced timescales across the 128 SSM channels (fast byte rhythm to phrase memory)
        let mut ssm_alpha = [0.0f32; MOTOR_HID_DIM];
        for d in 0..MOTOR_HID_DIM {
            let frac = (d as f32) / ((MOTOR_HID_DIM - 1) as f32);
            // Range [0.25, 0.92]
            ssm_alpha[d] = 0.25 + 0.67 * frac * frac;
        }

        Self {
            patcher,
            byte_emb,
            emb_grad: vec![0.0; BYTE_VOCAB * BYTE_EMB_DIM],
            emb_m: vec![0.0; BYTE_VOCAB * BYTE_EMB_DIM],
            emb_v: vec![0.0; BYTE_VOCAB * BYTE_EMB_DIM],
            emb_t: 0,
            l1,
            ssm_alpha,
            l2,
            fast_head: vec![0.0; BYTE_VOCAB * MOTOR_HID_DIM],
            fast_bias: [0.0; BYTE_VOCAB],
            hippocampus: HippocampusGraph::new(512, 0),
        }
    }

    pub fn packed_ternary_bytes(&self) -> usize {
        self.l1.ternary.packed_bytes() + self.l2.ternary.packed_bytes()
    }

    pub fn dense_f32_bytes(&self) -> usize {
        self.l1.dense_f32_bytes() + self.l2.dense_f32_bytes()
    }

    pub fn total_flips(&self) -> u64 {
        self.l1.flip_count + self.l2.flip_count
    }

    pub fn reset_online_memory(&mut self) {
        self.fast_head.fill(0.0);
        self.fast_bias.fill(0.0);
        self.hippocampus.clear();
    }

    /// Compress 256D patch LatentVec into 64D motor context (sum 4 contiguous channels).
    #[inline]
    fn compress_patch_latent(z: &LatentVec, out64: &mut [f32; PATCH_CTX_DIM]) {
        for j in 0..PATCH_CTX_DIM {
            let base = j * 4;
            let s = (z.data[base] as i32)
                + (z.data[base + 1] as i32)
                + (z.data[base + 2] as i32)
                + (z.data[base + 3] as i32);
            out64[j] = (s as f32) * (1.0 / 256.0);
        }
    }

    /// Build a 128D motor input from 4 previous raw bytes + current 64D patch latent.
    #[inline]
    fn build_input(&self, ctx: &[u8; CTX_BYTES], patch64: &[f32; PATCH_CTX_DIM], out: &mut [f32; MOTOR_IN_DIM]) {
        for slot in 0..CTX_BYTES {
            let b = ctx[slot] as usize;
            let emb = &self.byte_emb[b * BYTE_EMB_DIM..(b + 1) * BYTE_EMB_DIM];
            let dst = &mut out[slot * BYTE_EMB_DIM..(slot + 1) * BYTE_EMB_DIM];
            dst.copy_from_slice(emb);
        }
        out[CTX_BYTES * BYTE_EMB_DIM..].copy_from_slice(patch64);
    }

    /// Precompute causal patch latent stream over `data` (strictly causal: position `t` sees
    /// ONLY patches emitted from bytes `0..t`, NEVER `data[t]`!).
    fn extract_causal_patch_contexts(&mut self, data: &[u8]) -> (Vec<[f32; PATCH_CTX_DIM]>, f64) {
        self.patcher.reset();
        let mut patch_ctxs = Vec::with_capacity(data.len());
        let mut cur_patch = [0.0f32; PATCH_CTX_DIM];
        let mut n_patches = 0usize;

        for (i, &b) in data.iter().enumerate() {
            // Position `i` observes ONLY patches completed BEFORE byte `i`
            patch_ctxs.push(cur_patch);
            if let Some(em) = self.patcher.push_byte(b) {
                Self::compress_patch_latent(&em.latent, &mut cur_patch);
                if i >= 64 {
                    n_patches += 1;
                }
            }
        }
        let bpp = if n_patches > 0 {
            ((data.len().saturating_sub(64)) as f64) / (n_patches as f64)
        } else {
            1.0
        };
        (patch_ctxs, bpp)
    }

    /// Step byte embeddings with AdamW (or simple SGD in v1 unscaled baseline).
    fn step_embeddings(&mut self, batch_size: usize, lr: f32, use_adam: bool) {
        let inv_b = 1.0 / (batch_size.max(1) as f32);
        if use_adam {
            self.emb_t = self.emb_t.saturating_add(1);
            let t = self.emb_t as i32;
            let bc1 = 1.0 - 0.9f32.powi(t);
            let bc2 = 1.0 - 0.999f32.powi(t);
            for i in 0..self.byte_emb.len() {
                let g = (self.emb_grad[i] * inv_b).clamp(-2.0, 2.0);
                if g == 0.0 && self.emb_m[i] == 0.0 {
                    continue;
                }
                let m = 0.9 * self.emb_m[i] + 0.1 * g;
                let v = 0.999 * self.emb_v[i] + 0.001 * g * g;
                self.emb_m[i] = m;
                self.emb_v[i] = v;
                let mh = m / bc1;
                let vh = v / bc2;
                self.byte_emb[i] *= 1.0 - lr * 1e-4;
                self.byte_emb[i] -= lr * mh / (vh.sqrt() + 1e-8);
            }
        } else {
            for i in 0..self.byte_emb.len() {
                let g = (self.emb_grad[i] * inv_b).clamp(-2.0, 2.0);
                self.byte_emb[i] -= lr * g;
            }
        }
        self.emb_grad.fill(0.0);
    }

    /// End-to-end chunked training using **Blelloch Parallel Associative Prefix Scan**
    /// for the Selective SSM + BitNet b1.58 STE (`unscaled_v1_ablation = false`) OR
    /// naive unscaled ternary (`unscaled_v1_ablation = true`, isolating Bottleneck 3).
    pub fn train_corpus(
        &mut self,
        train_bytes: &[u8],
        eval_bytes: &[u8],
        epochs: usize,
        chunk_len: usize,
        unscaled_v1_ablation: bool,
        rng: &mut Xorshift64,
    ) -> TrainStats {
        let t_start = std::time::Instant::now();
        let (patch_ctxs, bpp_text) = self.extract_causal_patch_contexts(train_bytes);

        // Measure random-byte BPP for honest adaptive compression ratio on real text
        self.patcher.reset();
        let mut r_patches = 0usize;
        let mut r_rng = Xorshift64::new(rng.next_u64() ^ 0x5241_4E44);
        for _ in 0..512 {
            let rb = (r_rng.next_u32() & 0xFF) as u8;
            if self.patcher.push_byte(rb).is_some() {
                r_patches += 1;
            }
        }
        let bpp_rand = 512.0 / (r_patches.max(1) as f64);
        let compression_ratio = bpp_text / bpp_rand.max(0.1);

        let cfg = TernaryTrainConfig {
            lr: if unscaled_v1_ablation { 0.015 } else { 0.015 },
            weight_decay: 8e-4,
            absmean_beta: 0.55,
            hysteresis_band: 0.12,
            grad_clip: 2.0,
            ..Default::default()
        };

        let n = train_bytes.len();
        let chunk = chunk_len.max(8);
        let n_chunks = n / chunk;

        let mut initial_bpb = 8.0f64;
        let mut final_bpb = 8.0f64;
        let mut h_samples_for_erank: Vec<Vec<f32>> = Vec::with_capacity(128);

        let mut alphas_flat = vec![0.0f32; chunk * MOTOR_HID_DIM];
        for t in 0..chunk {
            alphas_flat[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM]
                .copy_from_slice(&self.ssm_alpha);
        }

        let mut chunk_order: Vec<usize> = (0..n_chunks).collect();

        for epoch in 0..epochs {
            for i in (1..n_chunks).rev() {
                let j = rng.next_usize(i + 1);
                chunk_order.swap(i, j);
            }
            let mut epoch_nats = 0.0f64;
            let mut epoch_count = 0usize;

            let mut ep_cfg = cfg;
            let progress = (epoch as f32) / (epochs.max(1) as f32);
            ep_cfg.lr =
                cfg.lr * (0.20 + 0.80 * (0.5 * (1.0 + (progress * std::f32::consts::PI).cos())));

            for &cid in &chunk_order {
                let start = cid * chunk;

                let mut xs = vec![[0.0f32; MOTOR_IN_DIM]; chunk];
                let mut ctxs = vec![[0u8; CTX_BYTES]; chunk];
                let mut u_pre = vec![[0.0f32; MOTOR_HID_DIM]; chunk];
                let mut u_act = vec![0.0f32; chunk * MOTOR_HID_DIM];

                for t in 0..chunk {
                    let pos = start + t;
                    let mut c6 = [0u8; CTX_BYTES];
                    for k in 0..CTX_BYTES {
                        c6[k] = if pos >= k + 1 { train_bytes[pos - k - 1] } else { 0 };
                    }
                    ctxs[t] = c6;
                    self.build_input(&c6, &patch_ctxs[pos], &mut xs[t]);
                    self.l1.forward(&xs[t], &mut u_pre[t]);
                    for d in 0..MOTOR_HID_DIM {
                        let v = u_pre[t][d];
                        u_act[t * MOTOR_HID_DIM + d] = if v > 0.0 { v } else { 0.1 * v };
                    }
                }

                // Parallel Associative Prefix Scan (Blelloch O(log L)) for Selective SSM!
                let h0 = [0.0f32; MOTOR_HID_DIM];
                let h_ssm =
                    ssm_forward_parallel_scan(&h0, &alphas_flat, &u_act, chunk, MOTOR_HID_DIM);

                let mut dh_ssm_out = vec![0.0f32; chunk * MOTOR_HID_DIM];
                let mut du_skip = vec![[0.0f32; MOTOR_HID_DIM]; chunk];
                let mut logits = [0.0f32; BYTE_VOCAB];
                let mut probs = [0.0f32; BYTE_VOCAB];
                let mut combined = [0.0f32; MOTOR_HID_DIM];

                for t in 0..chunk {
                    let u_sl = &u_act[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM];
                    let h_sl = &h_ssm[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM];
                    for d in 0..MOTOR_HID_DIM {
                        combined[d] = 0.65 * u_sl[d] + 0.35 * h_sl[d];
                    }
                    if epoch + 1 == epochs && t == chunk - 1 && h_samples_for_erank.len() < 128 {
                        h_samples_for_erank.push(combined.to_vec());
                    }

                    self.l2.forward(&combined, &mut logits);
                    let target = train_bytes[start + t];
                    let loss_nats = softmax_ce_probs(&logits, target, &mut probs);
                    epoch_nats += loss_nats as f64;
                    epoch_count += 1;

                    let eps_ls = if unscaled_v1_ablation { 0.0 } else { 0.01f32 };
                    let u_ls = eps_ls / (BYTE_VOCAB as f32);
                    let mut dy2 = probs;
                    for g in dy2.iter_mut() {
                        *g -= u_ls;
                    }
                    dy2[target as usize] -= 1.0 - eps_ls;

                    let mut dcomb = [0.0f32; MOTOR_HID_DIM];
                    self.l2.backward_accumulate(&combined, &dy2, Some(&mut dcomb));
                    for d in 0..MOTOR_HID_DIM {
                        du_skip[t][d] = 0.65 * dcomb[d];
                        dh_ssm_out[t * MOTOR_HID_DIM + d] = 0.35 * dcomb[d];
                    }
                }

                // Exact Parallel Associative Prefix Scan for SSM Backward Gradients!
                let dh_ssm =
                    ssm_backward_parallel_scan(&dh_ssm_out, &alphas_flat, chunk, MOTOR_HID_DIM);

                for t in 0..chunk {
                    let mut du = [0.0f32; MOTOR_HID_DIM];
                    for d in 0..MOTOR_HID_DIM {
                        let g_act = du_skip[t][d]
                            + dh_ssm[t * MOTOR_HID_DIM + d] * (1.0 - self.ssm_alpha[d]);
                        du[d] = if u_pre[t][d] > 0.0 { g_act } else { 0.1 * g_act };
                    }
                    let mut dx = [0.0f32; MOTOR_IN_DIM];
                    self.l1.backward_accumulate(&xs[t], &du, Some(&mut dx));
                    for slot in 0..CTX_BYTES {
                        let b = ctxs[t][slot] as usize;
                        let dst = &mut self.emb_grad[b * BYTE_EMB_DIM..(b + 1) * BYTE_EMB_DIM];
                        let src = &dx[slot * BYTE_EMB_DIM..(slot + 1) * BYTE_EMB_DIM];
                        for j in 0..BYTE_EMB_DIM {
                            dst[j] += src[j];
                        }
                    }
                }

                if unscaled_v1_ablation {
                    self.step_unscaled_v1(chunk, ep_cfg.lr);
                } else {
                    self.l1.step_adamw_ste(chunk, &ep_cfg);
                    self.l2.step_adamw_ste(chunk, &ep_cfg);
                    self.step_embeddings(chunk, ep_cfg.lr, true);
                }
            }

            let ep_bpb = (epoch_nats / (epoch_count.max(1) as f64)) / std::f64::consts::LN_2;
            if epoch == 0 {
                initial_bpb = ep_bpb;
            }
            final_bpb = ep_bpb;
        }

        let elapsed = t_start.elapsed().as_secs_f64().max(1e-6);
        let train_bps = ((n * epochs) as f64) / elapsed;
        let eval_bpb = self.eval_stream_bpb(eval_bytes, false);

        let erank = if h_samples_for_erank.len() >= 8 {
            let refs: Vec<&[f32]> = h_samples_for_erank.iter().map(|v| v.as_slice()).collect();
            effective_rank_subspace(&refs, 64)
        } else {
            0.0
        };

        TrainStats {
            initial_train_bpb: initial_bpb,
            final_train_bpb: final_bpb,
            eval_bpb,
            compression_ratio,
            effective_rank: erank,
            train_bytes_per_sec: train_bps,
            packed_ternary_bytes: self.packed_ternary_bytes(),
            dense_f32_bytes: self.dense_f32_bytes(),
            total_flips: self.total_flips(),
        }
    }

    /// Overall fraction of active non-zero 2-bit ternary weights across `l1` and `l2`.
    pub fn nonzero_fraction(&self) -> f64 {
        let n1 = (self.l1.rows * self.l1.cols) as f64;
        let n2 = (self.l2.rows * self.l2.cols) as f64;
        (self.l1.nonzero_fraction() * n1 + self.l2.nonzero_fraction() * n2) / (n1 + n2).max(1.0)
    }

    /// AURA v3 Genesis constructor applying:
    /// - Subroutine 1: Smooth 3D morphogenetic chemical gradient initialization for byte embeddings
    ///   (Wnt = byte order, BMP = universal ASCII class, Shh = harmonic phase — ZERO repo knowledge!)
    /// - Subroutine 2: Canonical 4-column laminar template instantiation across the 128 hidden units
    ///   (one 32-unit canonical template replicated 4x with topographic receptive-field alignment).
    pub fn new_genesis(seed: u64) -> Self {
        let mut model = Self::new(seed);
        let mut rng = Xorshift64::new(seed ^ 0x6E4E_5150_0003);

        // Subroutine 1: Smooth AND mutually separated morphogenetic byte coordinates
        // (Eph/ephrin orthogonal harmonics preserve ASCII neighborhood without colliding letters!)
        for b in 0..BYTE_VOCAB {
            let bu = b as u8;
            let wnt = (b as f32) / 255.0;
            let bmp = if bu == b' ' || bu == b'\n' || bu == b'\t' || bu == b'\r' {
                0.08
            } else if bu.is_ascii_digit() {
                0.24
            } else if bu.is_ascii_uppercase() {
                0.44
            } else if bu.is_ascii_lowercase() {
                0.64
            } else if matches!(bu, b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'<' | b'>' | b'`' | b'|') {
                0.80
            } else if bu.is_ascii_punctuation() {
                0.90
            } else {
                0.98
            };
            let local_phase = ((b as f32) * 0.618_034) * std::f32::consts::TAU;
            let off = b * BYTE_EMB_DIM;
            // Half bandwidth: morphogenetic chemical class; half bandwidth: orthogonal identity harmonics
            let coords: [f32; 12] = [
                (wnt - 0.5) * 0.28,
                (bmp - 0.5) * 0.34,
                (bmp * std::f32::consts::TAU).cos() * 0.18,
                (bmp * std::f32::consts::TAU).sin() * 0.18,
                local_phase.cos() * 0.24,
                local_phase.sin() * 0.24,
                (local_phase * 2.0).cos() * 0.22,
                (local_phase * 2.0).sin() * 0.22,
                (local_phase * 3.0).cos() * 0.20,
                (local_phase * 3.0).sin() * 0.20,
                if bu.is_ascii_whitespace() { 0.22 } else if bu.is_ascii_digit() { -0.22 } else { 0.0 },
                if bu.is_ascii_punctuation() { 0.22 } else if bu.is_ascii_uppercase() { -0.22 } else { 0.0 },
            ];
            for j in 0..BYTE_EMB_DIM {
                model.byte_emb[off + j] = coords[j] + rng.next_normal_f32() * 0.14;
            }
        }

        // Subroutine 2: Instantiate 4 Mountcastle canonical 32-unit minicolumns specialized ONLY
        // by topographic sensory cable wiring (exact variance-preserving gain: mean(g^2) = 1.0!)
        let col_rows = MOTOR_HID_DIM / 4; // 32 rows per canonical minicolumn
        let scale = (2.0 / (MOTOR_IN_DIM as f32)).sqrt() * 0.50;
        for col_id in 0..4 {
            let (band_lo, band_hi) = match col_id {
                0 => (0usize, 24usize),   // recent bytes t-1, t-2
                1 => (24usize, 48usize),  // medium bytes t-3, t-4
                2 => (48usize, 72usize),  // far bytes t-5, t-6
                _ => (72usize, 128usize), // EntropyPatcher compressed latent
            };
            let n_in = (band_hi - band_lo) as f32;
            let n_out = (MOTOR_IN_DIM as f32) - n_in;
            let g_in = 1.40f32;
            let g_out = (((MOTOR_IN_DIM as f32) - n_in * g_in * g_in) / n_out.max(1.0))
                .max(0.25)
                .sqrt();
            for r in 0..col_rows {
                let dst_r = col_id * col_rows + r;
                let dst_off = dst_r * MOTOR_IN_DIM;
                for c in 0..MOTOR_IN_DIM {
                    let gain = if c >= band_lo && c < band_hi { g_in } else { g_out };
                    model.l1.shadow_w[dst_off + c] = rng.next_normal_f32() * scale * gain;
                }
            }
        }
        // Start overproduced (Conel ChildhoodPeak: absmean_beta = 0.42 -> ~68% active density!)
        model.l1.quantize_from_shadow(0.42, 0.0, 0.0, false);
        model.l2.quantize_from_shadow(0.42, 0.0, 0.0, false);
        model
    }

    /// Execute one training epoch over chunked bytes using Blelloch Parallel Prefix Scan + BitNet STE.
    fn run_epoch_chunks(
        &mut self,
        data: &[u8],
        patch_ctxs: &[[f32; PATCH_CTX_DIM]],
        chunk: usize,
        ep_cfg: &TernaryTrainConfig,
        rng: &mut Xorshift64,
        mut collect_erank: Option<&mut Vec<Vec<f32>>>,
    ) -> f64 {
        let n_chunks = data.len() / chunk;
        let mut chunk_order: Vec<usize> = (0..n_chunks).collect();
        for i in (1..n_chunks).rev() {
            let j = rng.next_usize(i + 1);
            chunk_order.swap(i, j);
        }
        let mut alphas_flat = vec![0.0f32; chunk * MOTOR_HID_DIM];
        for t in 0..chunk {
            alphas_flat[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM]
                .copy_from_slice(&self.ssm_alpha);
        }

        let mut epoch_nats = 0.0f64;
        let mut epoch_count = 0usize;

        for &cid in &chunk_order {
            let start = cid * chunk;
            let mut xs = vec![[0.0f32; MOTOR_IN_DIM]; chunk];
            let mut ctxs = vec![[0u8; CTX_BYTES]; chunk];
            let mut u_pre = vec![[0.0f32; MOTOR_HID_DIM]; chunk];
            let mut u_act = vec![0.0f32; chunk * MOTOR_HID_DIM];

            for t in 0..chunk {
                let pos = start + t;
                let mut c6 = [0u8; CTX_BYTES];
                for k in 0..CTX_BYTES {
                    c6[k] = if pos >= k + 1 { data[pos - k - 1] } else { 0 };
                }
                ctxs[t] = c6;
                self.build_input(&c6, &patch_ctxs[pos], &mut xs[t]);
                self.l1.forward(&xs[t], &mut u_pre[t]);
                for d in 0..MOTOR_HID_DIM {
                    let v = u_pre[t][d];
                    u_act[t * MOTOR_HID_DIM + d] = if v > 0.0 { v } else { 0.1 * v };
                }
            }

            let h0 = [0.0f32; MOTOR_HID_DIM];
            let h_ssm = ssm_forward_parallel_scan(&h0, &alphas_flat, &u_act, chunk, MOTOR_HID_DIM);

            let mut dh_ssm_out = vec![0.0f32; chunk * MOTOR_HID_DIM];
            let mut du_skip = vec![[0.0f32; MOTOR_HID_DIM]; chunk];
            let mut logits = [0.0f32; BYTE_VOCAB];
            let mut probs = [0.0f32; BYTE_VOCAB];
            let mut combined = [0.0f32; MOTOR_HID_DIM];

            for t in 0..chunk {
                let u_sl = &u_act[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM];
                let h_sl = &h_ssm[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM];
                for d in 0..MOTOR_HID_DIM {
                    combined[d] = 0.65 * u_sl[d] + 0.35 * h_sl[d];
                }
                if let Some(ref mut buf) = collect_erank {
                    if t == chunk - 1 && buf.len() < 128 {
                        buf.push(combined.to_vec());
                    }
                }

                self.l2.forward(&combined, &mut logits);
                let target = data[start + t];
                let loss_nats = softmax_ce_probs(&logits, target, &mut probs);
                epoch_nats += loss_nats as f64;
                epoch_count += 1;

                let eps_ls = 0.01f32;
                let u_ls = eps_ls / (BYTE_VOCAB as f32);
                let mut dy2 = probs;
                for g in dy2.iter_mut() {
                    *g -= u_ls;
                }
                dy2[target as usize] -= 1.0 - eps_ls;

                let mut dcomb = [0.0f32; MOTOR_HID_DIM];
                self.l2.backward_accumulate(&combined, &dy2, Some(&mut dcomb));
                for d in 0..MOTOR_HID_DIM {
                    du_skip[t][d] = 0.65 * dcomb[d];
                    dh_ssm_out[t * MOTOR_HID_DIM + d] = 0.35 * dcomb[d];
                }
            }

            let dh_ssm = ssm_backward_parallel_scan(&dh_ssm_out, &alphas_flat, chunk, MOTOR_HID_DIM);

            for t in 0..chunk {
                let mut du = [0.0f32; MOTOR_HID_DIM];
                for d in 0..MOTOR_HID_DIM {
                    let g_act = du_skip[t][d]
                        + dh_ssm[t * MOTOR_HID_DIM + d] * (1.0 - self.ssm_alpha[d]);
                    du[d] = if u_pre[t][d] > 0.0 { g_act } else { 0.1 * g_act };
                }
                let mut dx = [0.0f32; MOTOR_IN_DIM];
                self.l1.backward_accumulate(&xs[t], &du, Some(&mut dx));
                for slot in 0..CTX_BYTES {
                    let b = ctxs[t][slot] as usize;
                    let dst = &mut self.emb_grad[b * BYTE_EMB_DIM..(b + 1) * BYTE_EMB_DIM];
                    let src = &dx[slot * BYTE_EMB_DIM..(slot + 1) * BYTE_EMB_DIM];
                    for j in 0..BYTE_EMB_DIM {
                        dst[j] += src[j];
                    }
                }
            }

            self.l1.step_adamw_ste(chunk, ep_cfg);
            self.l2.step_adamw_ste(chunk, ep_cfg);
            self.step_embeddings(chunk, ep_cfg.lr, true);
        }

        (epoch_nats / (epoch_count.max(1) as f64)) / std::f64::consts::LN_2
    }

    /// Standard AURA v2 training with identical early epoch-1 held-out checkpoint & sparsity tracking
    /// for strictly apples-to-apples comparison against `train_corpus_genesis`.
    pub fn train_corpus_v2_tracked(
        &mut self,
        train_bytes: &[u8],
        eval_bytes: &[u8],
        epochs: usize,
        chunk_len: usize,
        rng: &mut Xorshift64,
    ) -> GenesisByteStats {
        let t_start = std::time::Instant::now();
        let (patch_ctxs, bpp_text) = self.extract_causal_patch_contexts(train_bytes);
        self.patcher.reset();
        let mut r_patches = 0usize;
        let mut r_rng = Xorshift64::new(rng.next_u64() ^ 0x5241_4E44);
        for _ in 0..512 {
            if self.patcher.push_byte((r_rng.next_u32() & 0xFF) as u8).is_some() {
                r_patches += 1;
            }
        }
        let compression_ratio = bpp_text / (512.0 / (r_patches.max(1) as f64)).max(0.1);

        let cfg = TernaryTrainConfig {
            lr: 0.015,
            weight_decay: 8e-4,
            absmean_beta: 0.55,
            hysteresis_band: 0.12,
            grad_clip: 2.0,
            ..Default::default()
        };
        let chunk = chunk_len.max(8);
        let mut initial_bpb = 8.0f64;
        let mut final_bpb = 8.0f64;
        let mut early_eval_bpb = 8.0f64;
        let mut h_samples = Vec::with_capacity(128);
        let nonzero_peak = self.nonzero_fraction();

        for epoch in 0..epochs {
            let mut ep_cfg = cfg;
            let progress = (epoch as f32) / (epochs.max(1) as f32);
            ep_cfg.lr =
                cfg.lr * (0.20 + 0.80 * (0.5 * (1.0 + (progress * std::f32::consts::PI).cos())));
            let ep_bpb = self.run_epoch_chunks(
                train_bytes,
                &patch_ctxs,
                chunk,
                &ep_cfg,
                rng,
                if epoch + 1 == epochs { Some(&mut h_samples) } else { None },
            );
            if epoch == 0 {
                initial_bpb = ep_bpb;
                early_eval_bpb = self.eval_stream_bpb(eval_bytes, false);
            }
            final_bpb = ep_bpb;
        }

        let elapsed = t_start.elapsed().as_secs_f64().max(1e-6);
        let eval_bpb = self.eval_stream_bpb(eval_bytes, false);
        let erank = if h_samples.len() >= 8 {
            let refs: Vec<&[f32]> = h_samples.iter().map(|v| v.as_slice()).collect();
            effective_rank_subspace(&refs, 64)
        } else {
            0.0
        };
        GenesisByteStats {
            base: TrainStats {
                initial_train_bpb: initial_bpb,
                final_train_bpb: final_bpb,
                eval_bpb,
                compression_ratio,
                effective_rank: erank,
                train_bytes_per_sec: ((train_bytes.len() * epochs) as f64) / elapsed,
                packed_ternary_bytes: self.packed_ternary_bytes(),
                dense_f32_bytes: self.dense_f32_bytes(),
                total_flips: self.total_flips(),
            },
            early_eval_bpb,
            nonzero_peak_frac: nonzero_peak,
            nonzero_final_frac: self.nonzero_fraction(),
            pruned_synapses: 0,
        }
    }

    /// Universal structural ASCII byte class (8 language-agnostic syntactic categories:
    /// space, newline/tab, digit, uppercase, lowercase, bracket/delim, punctuation, other).
    /// Used exclusively for prenatal rhythm calibration (ZERO repository vocabulary!).
    #[inline]
    fn universal_byte_class(b: u8) -> usize {
        if b == b' ' {
            0
        } else if b == b'\n' || b == b'\t' || b == b'\r' {
            1
        } else if b.is_ascii_digit() {
            2
        } else if b.is_ascii_uppercase() {
            3
        } else if b.is_ascii_lowercase() {
            4
        } else if matches!(b, b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'<' | b'>' | b'`' | b'|') {
            5
        } else if b.is_ascii_punctuation() {
            6
        } else {
            7
        }
    }

    /// Run self-supervised prenatal rhythm passes predicting ONLY the 8 universal ASCII
    /// structural categories (space, newline/tab, digit, uppercase, lowercase, bracket,
    /// punctuation, other) on pure synthetic byte waves — calibrates `byte_emb`, `l1`,
    /// `EntropyPatcher` VICReg, and projects the language-agnostic 8-class motor rhythm
    /// uniformly per ASCII class onto `l2` (identical prior for all 26 letters / 10 digits,
    /// ZERO repository vocabulary or gibberish spelling bias!).
    fn run_prenatal_structural_epoch(
        &mut self,
        data: &[u8],
        chunk: usize,
        passes: usize,
        lr: f32,
        rng: &mut Xorshift64,
    ) {
        let (patch_ctxs, _) = self.extract_causal_patch_contexts(data);
        let n_chunks = data.len() / chunk;
        let mut head_w = vec![0.0f32; 8 * MOTOR_HID_DIM];
        let mut head_b = [0.0f32; 8];
        for v in &mut head_w {
            *v = rng.next_normal_f32() * 0.08;
        }
        let mut alphas_flat = vec![0.0f32; chunk * MOTOR_HID_DIM];
        for t in 0..chunk {
            alphas_flat[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM]
                .copy_from_slice(&self.ssm_alpha);
        }
        let pre_cfg = TernaryTrainConfig {
            lr,
            weight_decay: 4e-4,
            absmean_beta: 0.42, // Conel overproduction peak!
            hysteresis_band: 0.04,
            grad_clip: 2.0,
            ..Default::default()
        };

        for _pass in 0..passes.max(1) {
            for cid in 0..n_chunks {
                let start = cid * chunk;
                let mut xs = vec![[0.0f32; MOTOR_IN_DIM]; chunk];
                let mut ctxs = vec![[0u8; CTX_BYTES]; chunk];
                let mut u_pre = vec![[0.0f32; MOTOR_HID_DIM]; chunk];
                let mut u_act = vec![0.0f32; chunk * MOTOR_HID_DIM];

                for t in 0..chunk {
                    let pos = start + t;
                    let mut c6 = [0u8; CTX_BYTES];
                    for k in 0..CTX_BYTES {
                        c6[k] = if pos >= k + 1 { data[pos - k - 1] } else { 0 };
                    }
                    ctxs[t] = c6;
                    self.build_input(&c6, &patch_ctxs[pos], &mut xs[t]);
                    self.l1.forward(&xs[t], &mut u_pre[t]);
                    for d in 0..MOTOR_HID_DIM {
                        let v = u_pre[t][d];
                        u_act[t * MOTOR_HID_DIM + d] = if v > 0.0 { v } else { 0.1 * v };
                    }
                }

                let h0 = [0.0f32; MOTOR_HID_DIM];
                let h_ssm =
                    ssm_forward_parallel_scan(&h0, &alphas_flat, &u_act, chunk, MOTOR_HID_DIM);
                let mut dh_ssm_out = vec![0.0f32; chunk * MOTOR_HID_DIM];
                let mut du_skip = vec![[0.0f32; MOTOR_HID_DIM]; chunk];
                let mut combined = [0.0f32; MOTOR_HID_DIM];
                let mut head_grad = vec![0.0f32; 8 * MOTOR_HID_DIM];
                let mut hb_grad = [0.0f32; 8];

                for t in 0..chunk {
                    let u_sl = &u_act[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM];
                    let h_sl = &h_ssm[t * MOTOR_HID_DIM..(t + 1) * MOTOR_HID_DIM];
                    for d in 0..MOTOR_HID_DIM {
                        combined[d] = 0.65 * u_sl[d] + 0.35 * h_sl[d];
                    }
                    let mut logits = [0.0f32; 8];
                    let mut max_l = f32::NEG_INFINITY;
                    for c in 0..8 {
                        let mut acc = head_b[c];
                        let off = c * MOTOR_HID_DIM;
                        for d in 0..MOTOR_HID_DIM {
                            acc += head_w[off + d] * combined[d];
                        }
                        logits[c] = acc;
                        if acc > max_l {
                            max_l = acc;
                        }
                    }
                    let mut probs = [0.0f32; 8];
                    let mut sum_e = 0.0f32;
                    for c in 0..8 {
                        let e = (logits[c] - max_l).exp();
                        probs[c] = e;
                        sum_e += e;
                    }
                    let inv_e = 1.0 / sum_e.max(1e-12);
                    let target_cls = Self::universal_byte_class(data[start + t]);
                    let mut dcomb = [0.0f32; MOTOR_HID_DIM];
                    for c in 0..8 {
                        let p = probs[c] * inv_e;
                        let dy = p - if c == target_cls { 1.0 } else { 0.0 };
                        hb_grad[c] += dy;
                        let off = c * MOTOR_HID_DIM;
                        for d in 0..MOTOR_HID_DIM {
                            head_grad[off + d] += dy * combined[d];
                            dcomb[d] += dy * head_w[off + d];
                        }
                    }
                    for d in 0..MOTOR_HID_DIM {
                        du_skip[t][d] = 0.65 * dcomb[d];
                        dh_ssm_out[t * MOTOR_HID_DIM + d] = 0.35 * dcomb[d];
                    }
                }

                let inv_c = 1.0 / (chunk as f32);
                for i in 0..head_w.len() {
                    head_w[i] -= lr * 2.0 * (head_grad[i] * inv_c).clamp(-2.0, 2.0);
                }
                for c in 0..8 {
                    head_b[c] -= lr * 2.0 * (hb_grad[c] * inv_c).clamp(-2.0, 2.0);
                }

                let dh_ssm =
                    ssm_backward_parallel_scan(&dh_ssm_out, &alphas_flat, chunk, MOTOR_HID_DIM);
                for t in 0..chunk {
                    let mut du = [0.0f32; MOTOR_HID_DIM];
                    for d in 0..MOTOR_HID_DIM {
                        let g_act = du_skip[t][d]
                            + dh_ssm[t * MOTOR_HID_DIM + d] * (1.0 - self.ssm_alpha[d]);
                        du[d] = if u_pre[t][d] > 0.0 { g_act } else { 0.1 * g_act };
                    }
                    let mut dx = [0.0f32; MOTOR_IN_DIM];
                    self.l1.backward_accumulate(&xs[t], &du, Some(&mut dx));
                    for slot in 0..CTX_BYTES {
                        let b = ctxs[t][slot] as usize;
                        let dst = &mut self.emb_grad[b * BYTE_EMB_DIM..(b + 1) * BYTE_EMB_DIM];
                        let src = &dx[slot * BYTE_EMB_DIM..(slot + 1) * BYTE_EMB_DIM];
                        for j in 0..BYTE_EMB_DIM {
                            dst[j] += src[j];
                        }
                    }
                }
                self.l1.step_adamw_ste(chunk, &pre_cfg);
                self.step_embeddings(chunk, lr, true);
            }
        }

        // Exact maximum-entropy uniform-within-class per-byte log-prior derived 100% from the
        // synthetic prenatal byte wave `data`: `log((P_prenatal(cls) + eps) / N_cls)` centered.
        // Every byte inside a universal ASCII class (all 26 lowercase letters, all 26 uppercase
        // letters, all 10 digits, etc.) receives the EXACT SAME scalar prior (ZERO vocabulary,
        // ZERO spelling bias, and 100% preserves full-rank symmetry breaking of `l2.shadow_w`!).
        let mut cls_counts = [0usize; 8];
        for &b in data {
            cls_counts[Self::universal_byte_class(b)] += 1;
        }
        let mut cls_sizes = [0usize; 8];
        for b in 0..BYTE_VOCAB {
            cls_sizes[Self::universal_byte_class(b as u8)] += 1;
        }
        let total = data.len().max(1) as f32;
        let mut byte_prior = [0.0f32; BYTE_VOCAB];
        let mut prior_mean = 0.0f32;
        for b in 0..BYTE_VOCAB {
            let cls = Self::universal_byte_class(b as u8);
            let p_cls = ((cls_counts[cls] as f32) / total) + 0.002;
            let p_byte = p_cls / (cls_sizes[cls].max(1) as f32);
            let lp = p_byte.ln();
            byte_prior[b] = lp;
            prior_mean += lp;
        }
        prior_mean /= BYTE_VOCAB as f32;
        for b in 0..BYTE_VOCAB {
            self.l2.bias[b] += (0.65 * (byte_prior[b] - prior_mean)).clamp(-2.5, 2.5);
        }
    }

    /// Precompute causal patch latent stream while preserving prenatally calibrated VICReg statistics.
    fn extract_causal_patch_contexts_warm(
        &mut self,
        data: &[u8],
    ) -> (Vec<[f32; PATCH_CTX_DIM]>, f64) {
        self.patcher.reset_stream_keep_vicreg();
        let mut patch_ctxs = Vec::with_capacity(data.len());
        let mut cur_patch = [0.0f32; PATCH_CTX_DIM];
        let mut n_patches = 0usize;

        for (i, &b) in data.iter().enumerate() {
            patch_ctxs.push(cur_patch);
            if let Some(em) = self.patcher.push_byte(b) {
                Self::compress_patch_latent(&em.latent, &mut cur_patch);
                if i >= 64 {
                    n_patches += 1;
                }
            }
        }
        let bpp = if n_patches > 0 {
            ((data.len().saturating_sub(64)) as f64) / (n_patches as f64)
        } else {
            1.0
        };
        (patch_ctxs, bpp)
    }

    /// AURA v3 Genesis developmental training integrating ALL 4 DNA Subroutines:
    /// 1. Morphogenetic byte coordinates + 2. Canonical 4-column topographic wiring (`new_genesis`),
    /// 3. Prenatal synthetic structural-rhythm curriculum (8 universal ASCII classes, ZERO repo bytes!),
    /// 4. Overproduction -> Microglia structural pruning (permanent exact `0` bit-planes) +
    ///    SSM channel myelination!
    pub fn train_corpus_genesis(
        &mut self,
        train_bytes: &[u8],
        eval_bytes: &[u8],
        epochs: usize,
        chunk_len: usize,
        prenatal_seed: u64,
        rng: &mut Xorshift64,
    ) -> GenesisByteStats {
        let t_start = std::time::Instant::now();
        let chunk = chunk_len.max(8);

        // --- SUBROUTINE 4: Prenatal synthetic structural-rhythm curriculum (ZERO repo bytes!) ---
        // Pre-calibrates sensory byte embeddings, canonical `l1` filters, `EntropyPatcher`
        // VICReg, and exact uniform-within-class 8-category motor bias on pure synthetic waves.
        // Uses its own `prenatal_rng` so postnatal chunk shuffling stays paired with AURA v2!
        let mut prenatal_rng = Xorshift64::new(prenatal_seed ^ 0x9E37_79B9);
        let prenatal_bytes = PrenatalCurriculum::generate_prenatal_byte_waves(3072, prenatal_seed);
        self.run_prenatal_structural_epoch(&prenatal_bytes, chunk, 2, 0.008, &mut prenatal_rng);
        let nonzero_peak_frac = self.nonzero_fraction();

        // Birth transition: reset Adam moments cleanly while preserving calibrated `byte_emb`,
        // `l1`, `l2` 8-class uniform-within-class prior, and warm `EntropyPatcher` VICReg!
        self.l1.reset_optimizer();
        self.l2.reset_optimizer();
        self.emb_m.fill(0.0);
        self.emb_v.fill(0.0);
        self.emb_t = 0;

        // --- POSTNATAL REAL DATA TRAINING + CONEL OVERPRODUCTION -> MICROGLIA PRUNE + MYELIN ---
        let (patch_ctxs, bpp_text) = self.extract_causal_patch_contexts_warm(train_bytes);
        self.patcher.reset_stream_keep_vicreg();
        let mut r_patches = 0usize;
        let mut r_rng = Xorshift64::new(rng.next_u64() ^ 0x5241_4E44);
        for _ in 0..512 {
            if self.patcher.push_byte((r_rng.next_u32() & 0xFF) as u8).is_some() {
                r_patches += 1;
            }
        }
        let compression_ratio = bpp_text / (512.0 / (r_patches.max(1) as f64)).max(0.1);

        let base_lr = 0.0155f32;
        let mut initial_bpb = 8.0f64;
        let mut final_bpb = 8.0f64;
        let mut early_eval_bpb = 8.0f64;
        let mut pruned_synapses = 0usize;
        let mut h_samples = Vec::with_capacity(128);

        for epoch in 0..epochs {
            let progress = (epoch as f32) / (epochs.max(1) as f32);
            let lr =
                base_lr * (0.20 + 0.80 * (0.5 * (1.0 + (progress * std::f32::consts::PI).cos())));
            let ep_cfg = TernaryTrainConfig {
                lr,
                weight_decay: 8e-4,
                absmean_beta: if epoch < 2 { 0.50 } else { 0.56 },
                hysteresis_band: 0.12,
                grad_clip: 2.0,
                ..Default::default()
            };

            let ep_bpb = self.run_epoch_chunks(
                train_bytes,
                &patch_ctxs,
                chunk,
                &ep_cfg,
                rng,
                if epoch + 1 == epochs { Some(&mut h_samples) } else { None },
            );

            if epoch == 0 {
                initial_bpb = ep_bpb;
                early_eval_bpb = self.eval_stream_bpb(eval_bytes, false);
            }

            // SUBROUTINE 3: Microglia Structural Pruning + SSM Myelination after Childhood Peak!
            // Pruning after epoch 1 gives 2 full consolidation epochs (2 & 3) for surviving
            // myelinated pathways to recover and sharpen held-out generalization!
            if epoch == 1 {
                pruned_synapses += self.l1.microglia_prune_ternary(0.24);
                pruned_synapses += self.l2.microglia_prune_ternary(0.14);
                for d in 0..MOTOR_HID_DIM {
                    if self.l1.row_scale[d] > 0.12 {
                        self.ssm_alpha[d] = (self.ssm_alpha[d] * 0.90).max(0.18);
                    }
                }
            }

            final_bpb = ep_bpb;
        }

        let elapsed = t_start.elapsed().as_secs_f64().max(1e-6);
        let eval_bpb = self.eval_stream_bpb(eval_bytes, false);
        let erank = if h_samples.len() >= 8 {
            let refs: Vec<&[f32]> = h_samples.iter().map(|v| v.as_slice()).collect();
            effective_rank_subspace(&refs, 64)
        } else {
            0.0
        };

        GenesisByteStats {
            base: TrainStats {
                initial_train_bpb: initial_bpb,
                final_train_bpb: final_bpb,
                eval_bpb,
                compression_ratio,
                effective_rank: erank,
                train_bytes_per_sec: ((train_bytes.len() * epochs) as f64) / elapsed,
                packed_ternary_bytes: self.packed_ternary_bytes(),
                dense_f32_bytes: self.dense_f32_bytes(),
                total_flips: self.total_flips(),
            },
            early_eval_bpb,
            nonzero_peak_frac,
            nonzero_final_frac: self.nonzero_fraction(),
            pruned_synapses,
        }
    }

    /// Naive v1 unscaled ternary update (fixed row_scale = 1.0, fixed threshold 0.25, plain SGD):
    /// isolates why Bottleneck 3 (BitNet AbsMean + per-row scale + AdamW STE) is essential!
    fn step_unscaled_v1(&mut self, batch_size: usize, lr: f32) {
        let inv_b = 1.0 / (batch_size.max(1) as f32);
        for layer in [&mut self.l1, &mut self.l2] {
            for i in 0..layer.shadow_w.len() {
                let g = (layer.grad_w[i] * inv_b).clamp(-1.0, 1.0);
                layer.shadow_w[i] -= lr * g;
            }
            for r in 0..layer.rows {
                let g = (layer.grad_b[r] * inv_b).clamp(-1.0, 1.0);
                layer.bias[r] -= lr * g;
                let off = r * layer.cols;
                for c in 0..layer.cols {
                    let w = layer.shadow_w[off + c];
                    let old_q = layer.ternary.get_weight(r, c);
                    let new_q = if w > 0.25 {
                        1
                    } else if w < -0.25 {
                        -1
                    } else {
                        0
                    };
                    if new_q != old_q {
                        layer.ternary.set_weight(r, c, new_q);
                        layer.flip_count += 1;
                    }
                }
                layer.row_scale[r] = 1.0; // Unscaled!
            }
            layer.zero_grad();
        }
        self.step_embeddings(batch_size, lr, false);
    }

    /// Build a 256D episodic key for hippocampal lookup during online domain shift.
    #[inline]
    fn make_hipp_key(combined: &[f32; MOTOR_HID_DIM], ctx: &[u8; CTX_BYTES]) -> LatentVec {
        let mut out = LatentVec::zero();
        for d in 0..128 {
            out.data[d] = (combined[d] * 32.0).round().clamp(-127.0, 127.0) as i8;
        }
        for j in 0..128 {
            let b0 = ctx[j % CTX_BYTES];
            let b1 = ctx[(j + 1) % CTX_BYTES];
            let h = b0
                .wrapping_mul(31)
                .wrapping_add(b1.wrapping_mul(17))
                .wrapping_add((j as u8).wrapping_mul(13));
            out.data[128 + j] = ((h as i32) - 128).clamp(-127, 127) as i8;
        }
        out
    }

    /// Evaluate held-out or domain-shift byte stream in strictly causal Bits-Per-Byte (BPB).
    pub fn eval_stream_bpb(&mut self, data: &[u8], online_adapt: bool) -> f64 {
        self.eval_stream_bpb_inner(data, online_adapt, false)
    }

    pub fn eval_stream_bpb_unscaled(&mut self, data: &[u8]) -> f64 {
        self.eval_stream_bpb_inner(data, false, false)
    }

    fn eval_stream_bpb_inner(&mut self, data: &[u8], online_adapt: bool, _use_rmsnorm: bool) -> f64 {
        if data.is_empty() {
            return 8.0;
        }
        self.patcher.reset();
        if online_adapt {
            self.reset_online_memory();
        }

        let mut cur_patch = [0.0f32; PATCH_CTX_DIM];
        let mut h_ssm = [0.0f32; MOTOR_HID_DIM];
        let mut ctx = [0u8; CTX_BYTES];
        let mut x = [0.0f32; MOTOR_IN_DIM];
        let mut u_pre = [0.0f32; MOTOR_HID_DIM];
        let mut combined = [0.0f32; MOTOR_HID_DIM];
        let mut logits = [0.0f32; BYTE_VOCAB];
        let mut probs = [0.0f32; BYTE_VOCAB];
        let mut total_nats = 0.0f64;

        for (step, &target) in data.iter().enumerate() {
            self.build_input(&ctx, &cur_patch, &mut x);
            self.l1.forward(&x, &mut u_pre);
            for d in 0..MOTOR_HID_DIM {
                let u = if u_pre[d] > 0.0 { u_pre[d] } else { 0.1 * u_pre[d] };
                let a = self.ssm_alpha[d];
                h_ssm[d] = a * h_ssm[d] + (1.0 - a) * u;
                combined[d] = 0.65 * u + 0.35 * h_ssm[d];
            }
            self.l2.forward(&combined, &mut logits);

            // 2. If online adaptation is active, add Fast-Weight Delta residual + Hippocampal recall
            let mut hipp_key = LatentVec::zero();
            if online_adapt {
                for r in 0..BYTE_VOCAB {
                    let row = &self.fast_head[r * MOTOR_HID_DIM..(r + 1) * MOTOR_HID_DIM];
                    let mut acc = self.fast_bias[r];
                    for d in 0..MOTOR_HID_DIM {
                        acc += row[d] * combined[d];
                    }
                    logits[r] += acc;
                }

                hipp_key = Self::make_hipp_key(&combined, &ctx);
                if !self.hippocampus.is_empty() {
                    let matches = self.hippocampus.recall_with_indices(&hipp_key, 4);
                    for (_, dist, ep) in matches {
                        if dist < 4200 {
                            let sim = 1.0 - (dist as f32) / 4200.0;
                            let boost = 2.6 * sim * sim;
                            logits[ep.payload_id as usize] += boost;
                        }
                    }
                }
            }

            // 3. Score strictly BEFORE updating on `target` (zero data leakage!)
            let nats = softmax_ce_probs(&logits, target, &mut probs);
            total_nats += nats as f64;

            // 4. Online causal update (Fast Weights + Hippocampus ONLY; slow 2-bit weights frozen!)
            if online_adapt {
                let mut norm2 = 1.0f32;
                for &v in &combined {
                    norm2 += v * v;
                }
                let surprise_scale = (nats * 0.45).clamp(0.4, 2.5);
                let step_lr = (0.35 * surprise_scale) / norm2;
                let bias_lr = 0.06 * surprise_scale;
                let decay = 0.9992f32;

                for r in 0..BYTE_VOCAB {
                    let err = if r == target as usize {
                        1.0 - probs[r]
                    } else {
                        -probs[r]
                    };
                    self.fast_bias[r] = self.fast_bias[r] * decay + bias_lr * err;
                    if err.abs() < 0.008 {
                        continue;
                    }
                    let row = &mut self.fast_head[r * MOTOR_HID_DIM..(r + 1) * MOTOR_HID_DIM];
                    let factor = step_lr * err;
                    for d in 0..MOTOR_HID_DIM {
                        row[d] = (row[d] * decay + factor * combined[d]).clamp(-4.0, 4.0);
                    }
                }

                if nats > 2.0 {
                    self.hippocampus.insert(Episode {
                        key: hipp_key,
                        action_id: 0,
                        next_latent: LatentVec::zero(),
                        surprise: (nats * 1000.0) as i32,
                        reward_signal: 1,
                        timestamp: step as u64,
                        payload_id: target as u16,
                        salience: (nats * 100.0).min(60000.0) as u16,
                    });
                }
            }

            // 5. Advance causal byte context & zero-token patcher
            ctx.rotate_right(1);
            ctx[0] = target;
            if let Some(em) = self.patcher.push_byte(target) {
                Self::compress_patch_latent(&em.latent, &mut cur_patch);
            }
        }

        (total_nats / (data.len() as f64)) / std::f64::consts::LN_2
    }

    /// Generate raw bytes autoregressively from a prompt using strictly 2-bit ternary weights
    /// (zero BPE tokens!).
    pub fn generate_bytes(&mut self, prompt: &[u8], n_gen: usize, temperature: f32, rng: &mut Xorshift64) -> Vec<u8> {
        self.patcher.reset();
        let mut cur_patch = [0.0f32; PATCH_CTX_DIM];
        let mut h_ssm = [0.0f32; MOTOR_HID_DIM];
        let mut ctx = [0u8; CTX_BYTES];
        let mut x = [0.0f32; MOTOR_IN_DIM];
        let mut u_pre = [0.0f32; MOTOR_HID_DIM];
        let mut combined = [0.0f32; MOTOR_HID_DIM];
        let mut logits = [0.0f32; BYTE_VOCAB];
        let mut probs = [0.0f32; BYTE_VOCAB];

        for &b in prompt {
            self.build_input(&ctx, &cur_patch, &mut x);
            self.l1.forward(&x, &mut u_pre);
            for d in 0..MOTOR_HID_DIM {
                let u = if u_pre[d] > 0.0 { u_pre[d] } else { 0.1 * u_pre[d] };
                let a = self.ssm_alpha[d];
                h_ssm[d] = a * h_ssm[d] + (1.0 - a) * u;
            }
            ctx.rotate_right(1);
            ctx[0] = b;
            if let Some(em) = self.patcher.push_byte(b) {
                Self::compress_patch_latent(&em.latent, &mut cur_patch);
            }
        }

        let mut out = Vec::with_capacity(n_gen);
        let temp = temperature.max(0.05);
        for _ in 0..n_gen {
            self.build_input(&ctx, &cur_patch, &mut x);
            self.l1.forward(&x, &mut u_pre);
            for d in 0..MOTOR_HID_DIM {
                let u = if u_pre[d] > 0.0 { u_pre[d] } else { 0.1 * u_pre[d] };
                let a = self.ssm_alpha[d];
                h_ssm[d] = a * h_ssm[d] + (1.0 - a) * u;
                combined[d] = 0.65 * u + 0.35 * h_ssm[d];
            }
            self.l2.forward(&combined, &mut logits);
            for l in &mut logits {
                *l /= temp;
            }
            let _ = softmax_ce_probs(&logits, 0, &mut probs);

            // Top-k=16 sampling for coherent byte continuation
            let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            ranked.truncate(16);
            let sum_p: f32 = ranked.iter().map(|x| x.1).sum();
            let mut u = rng.next_f32() * sum_p;
            let mut chosen = ranked[0].0 as u8;
            for &(b, p) in &ranked {
                u -= p;
                if u <= 0.0 {
                    chosen = b as u8;
                    break;
                }
            }
            out.push(chosen);
            ctx.rotate_right(1);
            ctx[0] = chosen;
            if let Some(em) = self.patcher.push_byte(chosen) {
                Self::compress_patch_latent(&em.latent, &mut cur_patch);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aura_byte_model_trains_and_beats_uniform_and_unscaled() {
        let corpus = b"# Resultados de AURA v2\nEl modelo aprende patrones de texto sin tokens BPE.\n\
            ## Metodologia\nEl modelo aprende patrones de texto sin tokens BPE en Rust puro.\n"
            .repeat(60);
        let split = (corpus.len() * 4) / 5;
        let train = &corpus[..split];
        let eval = &corpus[split..];

        let mut rng1 = Xorshift64::new(42);
        let mut model_v2 = AuraByteModel::new(42);
        let stats_v2 = model_v2.train_corpus(train, eval, 4, 32, false, &mut rng1);

        let mut rng2 = Xorshift64::new(42);
        let mut model_v1 = AuraByteModel::new(42);
        let stats_v1 = model_v1.train_corpus(train, eval, 4, 32, true, &mut rng2);

        assert!(
            stats_v2.eval_bpb < 2.5,
            "v2 eval BPB should drop well below 2.5 on structured text, got {:.3}",
            stats_v2.eval_bpb
        );
        assert!(
            stats_v2.eval_bpb < stats_v1.eval_bpb - 0.5,
            "v2 ({:.3}) should decisively beat unscaled v1 ({:.3})",
            stats_v2.eval_bpb,
            stats_v1.eval_bpb
        );
    }
}
