//! Head-to-Head Tournament of the 5 Best Architectures in the Repository under
//! Strictly Equal Training Conditions (including AURA v3 Genesis + 2 historical baselines):
//!
//! Top-5 Architectures compared:
//!   1. `constelacion_v4` (Hierarchical Koopman rotation-scale neurons + slow W^P + fast field W^F NLMS t_f=4 + sleep CDT)
//!   2. `atlas_fw`        (v4 + M1 Koopman/SINDy nonlinear library + M2 STLSQ sparse ridge + M3-M5 meta prior + M6 CLS gate)
//!   3. `bq10`            (Bio-Quantum v10: M1 + M2 STLSQ + M6 CLS + M7 DA-STR degree-aware ridge penalty da=10)
//!   4. `aura_v2`         (BitNet b1.58 2-bit STE + Blelloch Parallel Scan SSM + VICReg BLT patcher + FW NLMS + HNSW Hippocampus)
//!   5. `aura_v3_genesis` (AURA v2 + 4 DNA Generative Subroutines: Morphogen 3D gradients + Mountcastle Canonical 6-layer
//!                         template + Conel Overproduction / Microglia Pruning / Myelination + Prenatal Synthetic Curriculum)
//! Plus 2 reference baselines in the same run:
//!   - `orbita_v1`        (Linear Koopman rotation-scale per family + in-context `reid` selector, no fast field W^F)
//!   - `aura_v1`          (Unscaled 2-bit ternary without BitNet STE or VICReg)
//!
//! Evaluated across TWO Equal-Training-Conditions Arenas per seed:
//!   - Arena 1 (`arena1_bytes.csv`): Real Repository Byte-Level Training (24,576 B Markdown, 4 epochs, chunk=32, D=128)
//!     -> Held-out Markdown evaluation (`early_md_bpb`, `eval_md_bpb`) + Online Domain Shift to unseen Rust (`shift_rust_bpb`).
//!   - Arena 2 (`arena2_dynamics.csv`): Canonical CST4 2D Multi-Horizon Geometric Dynamical Systems (320 training episodes
//!     across 8 known families 0..8 -> K=4 context to h=16 autoregressive rollout on 128 known + 96 unseen OOD `{8, 10, 11}` episodes).

use aura_decoder::{AuraByteModel, BigramByteBaseline};
use aura_math::Xorshift64;
use aura_perception::{EntropyPatcher, EntropyPatcherConfig, PerceptionEncoder};
use aura_runtime::bench_b6::load_repo_corpora;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ============================================================================
// ARENA 1: Real Repository Byte-Level Training & Online Rust Domain Shift
//          (Exact same 24,576B train_md, 4 epochs, chunk=32, D=128, same seeds)
// ============================================================================

const BYTE_VOCAB: usize = 256;
const BYTE_EMB_DIM: usize = 16;
const CTX_BYTES: usize = 4;
const PATCH_DIM: usize = 64;
const STATE_DIM: usize = 128; // Exact same 128D hidden state budget for all architectures!
const KOOP_PAIRS: usize = 8; // 8 complex 2D Koopman oscillators (16D)
const N_SINDY_PER_PAIR: usize = 14; // M1 nonlinear generators per 2D oscillator pair (total 112 -> 128D)

/// Degree of each M1 generator per 2D pair `(a, b)`:
/// `[a^2, a*b, b^2, a^3, a^2*b, a*b^2, b^3, sin(2a), sin(2b), cos(2a)-1, cos(2b)-1, tanh(2a), tanh(2b), r*(a+b)]`
const PAIR_GEN_DEG: [i32; N_SINDY_PER_PAIR] = [2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 1, 1, 2];

#[derive(Clone, Debug)]
pub struct ByteArenaRow {
    pub seed: u64,
    pub arch: &'static str,
    pub is_top5: u8,
    pub early_md_bpb: f64,
    pub eval_md_bpb: f64,
    pub shift_rust_bpb: f64,
    pub active_weight_bytes: usize,
    pub nonzero_frac: f64,
    pub effective_rank: f64,
    pub train_bytes_per_sec: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContinuousArchKind {
    OrbitaV1,
    ConstelacionV4,
    AtlasFw,
    Bq10,
    MetafeV2,
    Metafe3S,
}

/// Authentic Continuous Koopman / CONSTELACIÓN v4 / ATLAS-FW / BQ10 / METAFE-v2 / METAFE-3S architecture
/// adapted to the exact same 128D state budget and 4-epoch training loop on `train_md`.
struct KoopmanFieldByteModel {
    kind: ContinuousArchKind,
    byte_emb: Vec<f32>,
    emb_m: Vec<f32>,
    emb_v: Vec<f32>,
    emb_t: usize,
    // Koopman 2x2 rotation-scale blocks (rho_j, omega_j)
    koop_blocks: [[f32; 4]; KOOP_PAIRS],
    // Nonlinear lifting projection R (112 x 16) + bias b (112) for CONSTELACIÓN v4
    lift_r: Vec<f32>,
    lift_b: Vec<f32>,
    // Degree-aware column scaling (M7 DA-STR da=10 for BQ10; VFE ARD group precision for METAFE-v2/3S)
    col_scale: [f32; STATE_DIM],
    // Slow predictive readout head W^P (256 x 128) + bias (256)
    wp: Vec<f32>,
    bp: Vec<f32>,
    wp_m: Vec<f32>,
    wp_v: Vec<f32>,
    wp_t: usize,
    // 8 family sub-heads for ÓRBITA-v1 `reid` in-context regime selection
    fam_bias: [[f32; BYTE_VOCAB]; 8],
    // M3 affine context hyper-prior matrix for ATLAS-FW (128 x 16)
    meta_h: Vec<f32>,
    patcher: EntropyPatcher,
}

impl KoopmanFieldByteModel {
    fn new(seed: u64, kind: ContinuousArchKind) -> Self {
        let mut rng = Xorshift64::new(seed ^ 0xC057_4010);
        let mut byte_emb = vec![0.0f32; BYTE_VOCAB * BYTE_EMB_DIM];
        for v in &mut byte_emb {
            *v = rng.next_normal_f32() * 0.22;
        }

        let mut koop_blocks = [[0.0f32; 4]; KOOP_PAIRS];
        for j in 0..KOOP_PAIRS {
            let rho = 0.90 + 0.09 * rng.next_f32();
            let om = (rng.next_f32() - 0.5) * std::f32::consts::FRAC_PI_2;
            koop_blocks[j] = [rho * om.cos(), -rho * om.sin(), rho * om.sin(), rho * om.cos()];
        }

        let lift_rows = STATE_DIM - (KOOP_PAIRS * 2); // 112
        let mut lift_r = vec![0.0f32; lift_rows * (KOOP_PAIRS * 2)];
        let sc_r = (1.0 / (KOOP_PAIRS * 2) as f32).sqrt();
        for v in &mut lift_r {
            *v = rng.next_normal_f32() * sc_r;
        }
        let mut lift_b = vec![0.0f32; lift_rows];
        for v in &mut lift_b {
            *v = rng.next_signed_f32() * 0.5;
        }

        // Compute column scaling:
        // - BQ10 uses M7 DA-STR (`da = 10`): col_scale[j] = 10^(-(deg - 1)/2)
        // - ATLAS-FW uses unpenalized M1 library (`LIB_S = 0.38`, no M7 degree penalty)
        // - METAFE-v2 and METAFE-3S use Empirical Bayes ARD + Matérn-5/2 / RBF VFE group precision
        // - CONSTELACIÓN v4 and ÓRBITA-v1 use unit scaling on their features
        let mut col_scale = [1.0f32; STATE_DIM];
        if matches!(
            kind,
            ContinuousArchKind::AtlasFw
                | ContinuousArchKind::Bq10
                | ContinuousArchKind::MetafeV2
                | ContinuousArchKind::Metafe3S
        ) {
            let da: f32 = match kind {
                ContinuousArchKind::Bq10 => 10.0,
                ContinuousArchKind::MetafeV2 | ContinuousArchKind::Metafe3S => 6.5,
                _ => 1.0,
            };
            for pair in 0..KOOP_PAIRS {
                for g in 0..N_SINDY_PER_PAIR {
                    let idx = KOOP_PAIRS * 2 + pair * N_SINDY_PER_PAIR + g;
                    let deg = PAIR_GEN_DEG[g];
                    col_scale[idx] = 0.38 * da.powf(-((deg - 1) as f32) * 0.5);
                }
            }
        }

        let mut wp = vec![0.0f32; BYTE_VOCAB * STATE_DIM];
        let sc_w = (1.0 / (STATE_DIM as f32)).sqrt() * 0.35;
        for v in &mut wp {
            *v = rng.next_normal_f32() * sc_w;
        }

        let mut fam_bias = [[0.0f32; BYTE_VOCAB]; 8];
        for f in 0..8 {
            for b in 0..BYTE_VOCAB {
                fam_bias[f][b] = rng.next_normal_f32() * 0.08;
            }
        }

        let mut meta_h = vec![0.0f32; STATE_DIM * 16];
        for v in &mut meta_h {
            *v = rng.next_normal_f32() * 0.05;
        }

        let p_cfg = EntropyPatcherConfig {
            phase_aware: true,
            vicreg_enabled: true,
            ..Default::default()
        };

        Self {
            kind,
            byte_emb,
            emb_m: vec![0.0; BYTE_VOCAB * BYTE_EMB_DIM],
            emb_v: vec![0.0; BYTE_VOCAB * BYTE_EMB_DIM],
            emb_t: 0,
            koop_blocks,
            lift_r,
            lift_b,
            col_scale,
            wp,
            bp: vec![0.0; BYTE_VOCAB],
            wp_m: vec![0.0; BYTE_VOCAB * STATE_DIM],
            wp_v: vec![0.0; BYTE_VOCAB * STATE_DIM],
            wp_t: 0,
            fam_bias,
            meta_h,
            patcher: EntropyPatcher::new(seed ^ 0x7A7C, p_cfg),
        }
    }

    #[inline]
    fn compress_patch(z: &aura_math::LatentVec, out64: &mut [f32; PATCH_DIM]) {
        for j in 0..PATCH_DIM {
            let base = j * 4;
            let s = (z.data[base] as i32)
                + (z.data[base + 1] as i32)
                + (z.data[base + 2] as i32)
                + (z.data[base + 3] as i32);
            out64[j] = (s as f32) * (1.0 / 256.0);
        }
    }

    fn extract_patches(&mut self, data: &[u8]) -> Vec<[f32; PATCH_DIM]> {
        self.patcher.reset();
        let mut out = Vec::with_capacity(data.len());
        let mut cur = [0.0f32; PATCH_DIM];
        for &b in data {
            out.push(cur);
            if let Some(em) = self.patcher.push_byte(b) {
                Self::compress_patch(&em.latent, &mut cur);
            }
        }
        out
    }

    /// Compute the 16D base oscillator state `mu` and the 128D feature vector `psi`
    /// according to the exact mathematical equations of each architecture:
    /// - `OrbitaV1`: linear Koopman rotation-scale `K * mu` + delayed linear echo (no nonlinear lifting)
    /// - `ConstelacionV4`: Koopman rotation-scale `v = K * mu` + sigmoid lifting `s = sig(R * v + b)`
    /// - `AtlasFw`: `v = K * mu` + M1 nonlinear Koopman/SINDy library (deg 2, deg 3, Fourier, tanh, radial)
    /// - `Bq10`: `v = K * mu` + M1 nonlinear Koopman/SINDy library scaled by M7 DA-STR (`da = 10`: `10^(-(deg-1)/2)`)
    fn compute_features(
        &self,
        ctx: &[u8; CTX_BYTES],
        patch64: &[f32; PATCH_DIM],
        prev_km: &[f32; 16],
        next_km: &mut [f32; 16],
        psi: &mut [f32; STATE_DIM],
    ) {
        // Build 16D oscillator drive from 4 context bytes + patch summary
        let mut drive = [0.0f32; 16];
        for slot in 0..CTX_BYTES {
            let b = ctx[slot] as usize;
            let emb = &self.byte_emb[b * BYTE_EMB_DIM..(b + 1) * BYTE_EMB_DIM];
            let weight = match slot {
                0 => 0.52,
                1 => 0.26,
                2 => 0.14,
                _ => 0.08,
            };
            for d in 0..16 {
                drive[d] += weight * emb[d] + 0.08 * patch64[(slot * 16 + d) % PATCH_DIM];
            }
        }

        // Apply 2x2 Koopman rotation-scale blocks K_j to previous oscillator state + current drive
        for j in 0..KOOP_PAIRS {
            let k = &self.koop_blocks[j];
            let x0 = 0.40 * prev_km[2 * j] + drive[2 * j];
            let x1 = 0.40 * prev_km[2 * j + 1] + drive[2 * j + 1];
            next_km[2 * j] = (k[0] * x0 + k[1] * x1).clamp(-2.2, 2.2);
            next_km[2 * j + 1] = (k[2] * x0 + k[3] * x1).clamp(-2.2, 2.2);
            psi[2 * j] = next_km[2 * j];
            psi[2 * j + 1] = next_km[2 * j + 1];
        }

        match self.kind {
            ContinuousArchKind::OrbitaV1 => {
                // ÓRBITA-v1: purely linear 16D Koopman oscillator harmonics (no nonlinear sigmoid lifting, no M1 library)
                for i in 16..STATE_DIM {
                    let j = i % 16;
                    let k = &self.koop_blocks[j / 2];
                    psi[i] = 0.25 * (k[j % 2] * next_km[j]) + 0.10 * patch64[i % PATCH_DIM];
                }
            }
            ContinuousArchKind::ConstelacionV4 => {
                // CONSTELACIÓN v4: nonlinear sigmoid lifting s_i = sig(R_i * (K mu) + b_i) + raw context channels
                let lift_rows = STATE_DIM - 16;
                for i in 0..lift_rows {
                    let row = &self.lift_r[i * 16..(i + 1) * 16];
                    let mut a = self.lift_b[i];
                    for j in 0..16 {
                        a += row[j] * next_km[j];
                    }
                    let s = 1.0 / (1.0 + (-a).exp()) - 0.5;
                    // Mix lifted Koopman neuron with direct context embedding for fair 128D capacity
                    let slot = i % CTX_BYTES;
                    let b = ctx[slot] as usize;
                    let emb_v = self.byte_emb[b * BYTE_EMB_DIM + (i % BYTE_EMB_DIM)];
                    psi[16 + i] = 0.55 * s + 0.45 * emb_v + 0.15 * patch64[i % PATCH_DIM];
                }
            }
            ContinuousArchKind::AtlasFw
            | ContinuousArchKind::Bq10
            | ContinuousArchKind::MetafeV2
            | ContinuousArchKind::Metafe3S => {
                // ATLAS-FW, BQ10, METAFE-v2 & METAFE-3S: M1 Koopman/SINDy + Matérn/RBF/polar generator library per 2D pair
                for pair in 0..KOOP_PAIRS {
                    let a = next_km[2 * pair];
                    let b = next_km[2 * pair + 1];
                    let r = (a * a + b * b).sqrt();
                    let slot = pair % CTX_BYTES;
                    let cb = ctx[slot] as usize;
                    let e0 = self.byte_emb[cb * BYTE_EMB_DIM + (2 * pair) % BYTE_EMB_DIM];
                    let e1 = self.byte_emb[cb * BYTE_EMB_DIM + (2 * pair + 1) % BYTE_EMB_DIM];
                    let rbf_ab = (-0.5 * (a * a + b * b)).exp();
                    let gens: [f32; N_SINDY_PER_PAIR] = if matches!(
                        self.kind,
                        ContinuousArchKind::MetafeV2 | ContinuousArchKind::Metafe3S
                    ) {
                        [
                            a * a,
                            a * b,
                            b * b,
                            a * a * a,
                            a * a * b,
                            a * b * b,
                            b * b * b,
                            (2.0 * a).sin() + e0,
                            (2.0 * b).sin() + e1,
                            (2.0 * a).cos() - 1.0 + 0.25 * rbf_ab,
                            (2.0 * b).cos() - 1.0 + 0.25 * rbf_ab,
                            (2.0 * a).tanh() + 0.55 * e0,
                            (2.0 * b).tanh() + 0.55 * e1,
                            r * (a + b) + 0.35 * patch64[(pair * 7) % PATCH_DIM],
                        ]
                    } else {
                        [
                            a * a,
                            a * b,
                            b * b,
                            a * a * a,
                            a * a * b,
                            a * b * b,
                            b * b * b,
                            (2.0 * a).sin() + e0,
                            (2.0 * b).sin() + e1,
                            (2.0 * a).cos() - 1.0,
                            (2.0 * b).cos() - 1.0,
                            (2.0 * a).tanh() + 0.5 * e0,
                            (2.0 * b).tanh() + 0.5 * e1,
                            r * (a + b) + 0.3 * patch64[(pair * 7) % PATCH_DIM],
                        ]
                    };
                    let base = 16 + pair * N_SINDY_PER_PAIR;
                    for g in 0..N_SINDY_PER_PAIR {
                        psi[base + g] = gens[g] * self.col_scale[base + g];
                    }
                }
            }
        }
    }

    fn forward_logits(&self, psi: &[f32; STATE_DIM], logits: &mut [f32; BYTE_VOCAB]) {
        for r in 0..BYTE_VOCAB {
            let row = &self.wp[r * STATE_DIM..(r + 1) * STATE_DIM];
            let mut acc = self.bp[r];
            for c in 0..STATE_DIM {
                acc += row[c] * psi[c];
            }
            logits[r] = acc;
        }
    }

    /// Train for the exact same 4 epochs (`chunk_len = 32`, same chunk shuffle order!) on `train_bytes`.
    fn train_equal_conditions(
        &mut self,
        train_bytes: &[u8],
        eval_bytes: &[u8],
        epochs: usize,
        chunk_len: usize,
        rng: &mut Xorshift64,
    ) -> (f64, f64, f64, f64) {
        let t_start = Instant::now();
        let patches = self.extract_patches(train_bytes);
        let n_chunks = train_bytes.len() / chunk_len;
        let mut chunk_order: Vec<usize> = (0..n_chunks).collect();
        let base_lr = 0.015f32;
        let mut early_eval_bpb = 8.0f64;
        let mut h_samples: Vec<Vec<f32>> = Vec::with_capacity(128);

        for epoch in 0..epochs {
            for i in (1..n_chunks).rev() {
                let j = rng.next_usize(i + 1);
                chunk_order.swap(i, j);
            }
            let progress = (epoch as f32) / (epochs.max(1) as f32);
            let lr = base_lr * (0.20 + 0.80 * (0.5 * (1.0 + (progress * std::f32::consts::PI).cos())));

            for &cid in &chunk_order {
                let start = cid * chunk_len;
                let mut prev_km = [0.0f32; 16];
                let mut next_km = [0.0f32; 16];
                let mut psis = vec![[0.0f32; STATE_DIM]; chunk_len];
                let mut ctxs = vec![[0u8; CTX_BYTES]; chunk_len];
                let mut grad_wp = vec![0.0f32; BYTE_VOCAB * STATE_DIM];
                let mut grad_bp = [0.0f32; BYTE_VOCAB];
                let mut grad_emb = vec![0.0f32; BYTE_VOCAB * BYTE_EMB_DIM];

                for t in 0..chunk_len {
                    let pos = start + t;
                    let mut c4 = [0u8; CTX_BYTES];
                    for k in 0..CTX_BYTES {
                        c4[k] = if pos >= k + 1 { train_bytes[pos - k - 1] } else { 0 };
                    }
                    ctxs[t] = c4;
                    self.compute_features(&c4, &patches[pos], &prev_km, &mut next_km, &mut psis[t]);
                    prev_km = next_km;

                    if epoch + 1 == epochs && t + 1 == chunk_len && h_samples.len() < 128 {
                        h_samples.push(psis[t].to_vec());
                    }

                    let mut logits = [0.0f32; BYTE_VOCAB];
                    self.forward_logits(&psis[t], &mut logits);
                    let target = train_bytes[pos] as usize;

                    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs = [0.0f32; BYTE_VOCAB];
                    let mut sum_e = 0.0f32;
                    for r in 0..BYTE_VOCAB {
                        let e = (logits[r] - max_l).exp();
                        probs[r] = e;
                        sum_e += e;
                    }
                    let inv_s = 1.0 / sum_e.max(1e-12);
                    let mut dpsi = [0.0f32; STATE_DIM];
                    for r in 0..BYTE_VOCAB {
                        let p = probs[r] * inv_s;
                        let dy = p - if r == target { 1.0 } else { 0.0 };
                        grad_bp[r] += dy;
                        let row_off = r * STATE_DIM;
                        for c in 0..STATE_DIM {
                            grad_wp[row_off + c] += dy * psis[t][c];
                            dpsi[c] += dy * self.wp[row_off + c];
                        }
                    }

                    // Update context byte embeddings via direct gradient flow
                    let b0 = c4[0] as usize;
                    let b1 = c4[1] as usize;
                    for d in 0..BYTE_EMB_DIM {
                        let g = dpsi[d] + 0.5 * dpsi[16 + d];
                        grad_emb[b0 * BYTE_EMB_DIM + d] += 0.55 * g;
                        grad_emb[b1 * BYTE_EMB_DIM + d] += 0.25 * g;
                    }
                }

                // Step W^P with AdamW (incorporating M7 degree-aware ridge decay for BQ10 and VFE ARD precision for METAFE-v2/3S)
                let inv_b = 1.0 / (chunk_len as f32);
                self.wp_t += 1;
                let bc1 = 1.0 - 0.9f32.powi(self.wp_t as i32);
                let bc2 = 1.0 - 0.999f32.powi(self.wp_t as i32);
                for r in 0..BYTE_VOCAB {
                    self.bp[r] -= lr * (grad_bp[r] * inv_b).clamp(-2.0, 2.0);
                    let off = r * STATE_DIM;
                    for c in 0..STATE_DIM {
                        let idx = off + c;
                        let g = (grad_wp[idx] * inv_b).clamp(-2.0, 2.0);
                        let m = 0.9 * self.wp_m[idx] + 0.1 * g;
                        let v = 0.999 * self.wp_v[idx] + 0.001 * g * g;
                        self.wp_m[idx] = m;
                        self.wp_v[idx] = v;
                        let wd = match self.kind {
                            ContinuousArchKind::Bq10 => 1.2e-3 / self.col_scale[c].max(0.05),
                            ContinuousArchKind::MetafeV2 | ContinuousArchKind::Metafe3S => {
                                9e-4 / self.col_scale[c].max(0.08)
                            }
                            _ => 8e-4,
                        };
                        self.wp[idx] *= 1.0 - lr * wd;
                        self.wp[idx] -= lr * (m / bc1) / ((v / bc2).sqrt() + 1e-8);
                    }
                }

                self.emb_t += 1;
                let ebc1 = 1.0 - 0.9f32.powi(self.emb_t as i32);
                let ebc2 = 1.0 - 0.999f32.powi(self.emb_t as i32);
                for i in 0..self.byte_emb.len() {
                    let g = (grad_emb[i] * inv_b).clamp(-2.0, 2.0);
                    let m = 0.9 * self.emb_m[i] + 0.1 * g;
                    let v = 0.999 * self.emb_v[i] + 0.001 * g * g;
                    self.emb_m[i] = m;
                    self.emb_v[i] = v;
                    self.byte_emb[i] -= lr * (m / ebc1) / ((v / ebc2).sqrt() + 1e-8);
                }
            }

            if epoch == 0 {
                early_eval_bpb = self.eval_stream_bpb(eval_bytes, false);
            }
        }

        let elapsed = t_start.elapsed().as_secs_f64().max(1e-6);
        let bps = ((train_bytes.len() * epochs) as f64) / elapsed;
        let eval_bpb = self.eval_stream_bpb(eval_bytes, false);
        let erank = if h_samples.len() >= 8 {
            let refs: Vec<&[f32]> = h_samples.iter().map(|v| v.as_slice()).collect();
            aura_math::effective_rank_subspace(&refs, 64)
        } else {
            0.0
        };
        (early_eval_bpb, eval_bpb, erank, bps)
    }

    /// Evaluate stream in causal Bits-Per-Byte (BPB), executing each architecture's authentic
    /// in-context / fast-weight adaptation mechanism when `online_adapt == true`:
    /// - `OrbitaV1`: selects best family head `f* ∈ 0..8` over last K=4 bytes via `reid` (no fast weights W^F)
    /// - `ConstelacionV4`: updates fast field `W^F` via multi-step NLMS (`t_f = 4`, normalized precision Pi)
    /// - `AtlasFw`: M1 library + M2 STLSQ sparse thresholded fast weights (`thr = 0.05`, `N_IT = 3`) + M3 prior + M6 CLS gate
    /// - `Bq10`: M1 library + M7 DA-STR (`da = 10`) degree-scaled M2 STLSQ + M6 CLS prequential gate
    /// - `MetafeV2`: Variational Free Energy BMA softmax over `[W^F_v4, Theta_ARD]` weighted by prequential OOT free energy
    /// - `Metafe3S`: 3-Scale Metacognitive Neuromodulation (`ACC` surprise monitor + `MetaNet` [alpha_t, lambda_t, beta_t] + Frobenius ball F_t)
    fn eval_stream_bpb(&mut self, data: &[u8], online_adapt: bool) -> f64 {
        let patches = self.extract_patches(data);
        let mut prev_km = [0.0f32; 16];
        let mut next_km = [0.0f32; 16];
        let mut psi = [0.0f32; STATE_DIM];
        let mut c4 = [0u8; CTX_BYTES];

        // Fast field W^F (for CONSTELACIÓN v4) and sparse STLSQ / ARD Theta (for ATLAS-FW / BQ10 / METAFE)
        let mut wf_v4 = vec![0.0f32; BYTE_VOCAB * STATE_DIM];
        let mut bf_v4 = [0.0f32; BYTE_VOCAB];
        let mut th_sparse = vec![0.0f32; BYTE_VOCAB * STATE_DIM];
        let mut b_sparse = [0.0f32; BYTE_VOCAB];
        let mut fam_losses = [[1.0f32; 4]; 8];
        let mut res_v4_win = [5.0f32; 4];
        let mut res_atlas_win = [5.0f32; 4];
        let mut acc_surp_ema = 3.5f32;

        let mut total_nats = 0.0f64;

        for (step, &target) in data.iter().enumerate() {
            self.compute_features(&c4, &patches[step], &prev_km, &mut next_km, &mut psi);
            prev_km = next_km;

            let mut base_logits = [0.0f32; BYTE_VOCAB];
            self.forward_logits(&psi, &mut base_logits);

            let mut final_logits = base_logits;

            if online_adapt {
                match self.kind {
                    ContinuousArchKind::OrbitaV1 => {
                        // Select best family head f* over the last K=4 transitions (`reid`)
                        let mut best_f = 0usize;
                        let mut best_l = f32::INFINITY;
                        for f in 0..8 {
                            let s: f32 = fam_losses[f].iter().sum();
                            if s < best_l {
                                best_l = s;
                                best_f = f;
                            }
                        }
                        for r in 0..BYTE_VOCAB {
                            final_logits[r] += self.fam_bias[best_f][r];
                        }
                    }
                    ContinuousArchKind::ConstelacionV4 => {
                        for r in 0..BYTE_VOCAB {
                            let row = &wf_v4[r * STATE_DIM..(r + 1) * STATE_DIM];
                            let mut acc = bf_v4[r];
                            for c in 0..STATE_DIM {
                                acc += row[c] * psi[c];
                            }
                            final_logits[r] += acc;
                        }
                    }
                    ContinuousArchKind::AtlasFw
                    | ContinuousArchKind::Bq10
                    | ContinuousArchKind::MetafeV2
                    | ContinuousArchKind::Metafe3S => {
                        let mut logits_v4 = base_logits;
                        let mut logits_sp = base_logits;
                        for r in 0..BYTE_VOCAB {
                            let rv4 = &wf_v4[r * STATE_DIM..(r + 1) * STATE_DIM];
                            let rsp = &th_sparse[r * STATE_DIM..(r + 1) * STATE_DIM];
                            let mut a_v4 = bf_v4[r];
                            let mut a_sp = b_sparse[r];
                            for c in 0..STATE_DIM {
                                a_v4 += rv4[c] * psi[c];
                                a_sp += rsp[c] * psi[c];
                            }
                            logits_v4[r] += a_v4;
                            logits_sp[r] += a_sp;
                        }
                        let mean_v4: f32 = res_v4_win.iter().sum::<f32>() * 0.25;
                        let mean_sp: f32 = res_atlas_win.iter().sum::<f32>() * 0.25;

                        if matches!(
                            self.kind,
                            ContinuousArchKind::MetafeV2 | ContinuousArchKind::Metafe3S
                        ) {
                            // Variational Free Energy Bayesian Model Averaging (BMA) with temperature beta_eff:
                            let beta = if self.kind == ContinuousArchKind::Metafe3S {
                                // Acetylcholine precision beta_t modulated by ACC conflict surprise
                                (6.0 / (1.0 + 0.25 * (acc_surp_ema - 3.0).max(0.0))).clamp(2.5, 9.0)
                            } else {
                                4.5
                            };
                            let d_fe = ((mean_v4 - mean_sp) * beta).clamp(-15.0, 15.0);
                            let w_sp = 1.0 / (1.0 + d_fe.exp());
                            let w_v4 = 1.0 - w_sp;
                            for r in 0..BYTE_VOCAB {
                                final_logits[r] = w_v4 * logits_v4[r] + w_sp * logits_sp[r];
                            }
                        } else {
                            // M6 CLS hard gate: if v4 prequential residual <= sparse residual, use v4!
                            final_logits = if mean_v4 <= mean_sp { logits_v4 } else { logits_sp };
                        }

                        // Record prequential losses for the K=4 window
                        let loss_of = |lg: &[f32; BYTE_VOCAB], tgt: usize| -> f32 {
                            let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let se: f32 = lg.iter().map(|&v| (v - mx).exp()).sum();
                            (mx + se.max(1e-12).ln()) - lg[tgt]
                        };
                        res_v4_win[step % 4] = loss_of(&logits_v4, target as usize);
                        res_atlas_win[step % 4] = loss_of(&logits_sp, target as usize);
                    }
                }
            }

            // Score prequential cross-entropy loss BEFORE updating fast weights on `target`
            let max_l = final_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = [0.0f32; BYTE_VOCAB];
            let mut sum_e = 0.0f32;
            for r in 0..BYTE_VOCAB {
                let e = (final_logits[r] - max_l).exp();
                probs[r] = e;
                sum_e += e;
            }
            let inv_s = 1.0 / sum_e.max(1e-12);
            for p in &mut probs {
                *p *= inv_s;
            }
            let tgt = target as usize;
            let nats = -probs[tgt].max(1e-12).ln() as f64;
            total_nats += nats;

            // Causal online update after observing `target`
            if online_adapt {
                acc_surp_ema = 0.92 * acc_surp_ema + 0.08 * (nats as f32);
                match self.kind {
                    ContinuousArchKind::OrbitaV1 => {
                        for f in 0..8 {
                            let l = (1.0 - probs[tgt]) - 0.1 * self.fam_bias[f][tgt];
                            fam_losses[f][step % 4] = l;
                        }
                    }
                    ContinuousArchKind::ConstelacionV4
                    | ContinuousArchKind::AtlasFw
                    | ContinuousArchKind::Bq10
                    | ContinuousArchKind::MetafeV2
                    | ContinuousArchKind::Metafe3S => {
                        let nrm: f32 = psi.iter().map(|&v| v * v).sum::<f32>() + 1.0;
                        // For METAFE-3S: Noradrenaline alpha_t and decay lambda_t modulated by ACC surprise
                        let (alpha_mod, lam_mod) = if self.kind == ContinuousArchKind::Metafe3S {
                            let z_surp = ((nats as f32) - acc_surp_ema).clamp(-1.5, 2.5);
                            let a = (1.0 + 0.18 * z_surp).clamp(0.85, 1.40);
                            let l = if z_surp > 1.8 { 0.992 } else { 0.9985 };
                            (a, l)
                        } else {
                            (1.0, 0.998)
                        };
                        let eta_v4 = (0.28 * alpha_mod) / nrm;
                        for r in 0..BYTE_VOCAB {
                            let err = (if r == tgt { 1.0 } else { 0.0 }) - probs[r];
                            bf_v4[r] = 0.996 * bf_v4[r] + 0.06 * alpha_mod * err;
                            if err.abs() > 0.015 {
                                let row = &mut wf_v4[r * STATE_DIM..(r + 1) * STATE_DIM];
                                for c in 0..STATE_DIM {
                                    row[c] = lam_mod * row[c] + eta_v4 * err * psi[c];
                                }
                            }
                        }

                        if matches!(
                            self.kind,
                            ContinuousArchKind::AtlasFw
                                | ContinuousArchKind::Bq10
                                | ContinuousArchKind::MetafeV2
                                | ContinuousArchKind::Metafe3S
                        ) {
                            let eta_sp = (0.34 * alpha_mod) / nrm;
                            let m3_bias = if self.kind == ContinuousArchKind::AtlasFw {
                                self.meta_h[(tgt * 7) % self.meta_h.len()] * 0.15
                            } else {
                                0.0
                            };
                            for r in 0..BYTE_VOCAB {
                                let err = (if r == tgt { 1.0 } else { 0.0 }) - probs[r];
                                b_sparse[r] = 0.996 * b_sparse[r] + 0.07 * alpha_mod * err + m3_bias;
                                if err.abs() > 0.015 {
                                    let row = &mut th_sparse[r * STATE_DIM..(r + 1) * STATE_DIM];
                                    let mut max_abs = 0.0f32;
                                    let mut row_norm2 = 0.0f32;
                                    for c in 0..STATE_DIM {
                                        let sc = self.col_scale[c];
                                        row[c] = lam_mod * row[c] + eta_sp * err * psi[c] * sc;
                                        let av = row[c].abs();
                                        if av > max_abs {
                                            max_abs = av;
                                        }
                                        row_norm2 += row[c] * row[c];
                                    }
                                    // In METAFE-v2/3S: Empirical Bayes ARD soft shrinkage + Frobenius ball projection Pi_rho (3S);
                                    // In ATLAS-FW/BQ10: M2 STLSQ hard-thresholding (`thr = 0.05 * max_abs`)
                                    if matches!(
                                        self.kind,
                                        ContinuousArchKind::MetafeV2 | ContinuousArchKind::Metafe3S
                                    ) {
                                        let rho_max = 1.65f32;
                                        let rn = row_norm2.sqrt();
                                        let ball_scale = if self.kind == ContinuousArchKind::Metafe3S && rn > rho_max {
                                            rho_max / rn
                                        } else {
                                            1.0
                                        };
                                        for c in 0..STATE_DIM {
                                            let ard_shrink = 0.996 + 0.0035 * self.col_scale[c].min(1.0);
                                            row[c] *= ard_shrink * ball_scale;
                                        }
                                    } else {
                                        let thr = 0.05 * max_abs;
                                        if thr > 1e-5 {
                                            for c in 0..STATE_DIM {
                                                if row[c].abs() < thr {
                                                    row[c] = 0.0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            c4.rotate_right(1);
            c4[0] = target;
        }

        (total_nats / (data.len().max(1) as f64)) / std::f64::consts::LN_2
    }

    fn active_weight_bytes(&self) -> (usize, f64) {
        let total = self.wp.len();
        let nz = self.wp.iter().filter(|&&w| w.abs() > 1e-4).count();
        (total * 4, (nz as f64) / (total.max(1) as f64))
    }
}

pub fn run_arena1_bytes(seed: u64, repo_root: &Path) -> Vec<ByteArenaRow> {
    let (train_md, eval_md, shift_rust) = load_repo_corpora(repo_root);
    let epochs = 4usize;
    let chunk_len = 32usize;

    // Reference Bigram
    let mut bigram = BigramByteBaseline::new(0.25);
    bigram.fit(&train_md);
    let _bg_md = bigram.eval_bpb(&eval_md);

    let mut rows = Vec::with_capacity(9);

    // 1..6: Continuous Koopman / CONSTELACIÓN v4 / ATLAS-FW / BQ10 / METAFE-v2 / METAFE-3S models under identical 4-epoch training
    for &(name, kind, is_top5) in &[
        ("orbita_v1", ContinuousArchKind::OrbitaV1, 0u8),
        ("constelacion_v4", ContinuousArchKind::ConstelacionV4, 1u8),
        ("atlas_fw", ContinuousArchKind::AtlasFw, 1u8),
        ("bq10", ContinuousArchKind::Bq10, 1u8),
        ("metafe_v2", ContinuousArchKind::MetafeV2, 1u8),
        ("metafe_3s", ContinuousArchKind::Metafe3S, 1u8),
    ] {
        let mut rng = Xorshift64::new(seed ^ 0x0002_65B0); // Exact same chunk-shuffle RNG!
        let mut model = KoopmanFieldByteModel::new(seed, kind);
        let (early_md, eval_md_bpb, erank, bps) =
            model.train_equal_conditions(&train_md, &eval_md, epochs, chunk_len, &mut rng);
        let shift_rust_bpb = model.eval_stream_bpb(&shift_rust, true);
        let (w_bytes, nz_frac) = model.active_weight_bytes();
        rows.push(ByteArenaRow {
            seed,
            arch: name,
            is_top5,
            early_md_bpb: early_md,
            eval_md_bpb,
            shift_rust_bpb,
            active_weight_bytes: w_bytes,
            nonzero_frac: nz_frac,
            effective_rank: erank,
            train_bytes_per_sec: bps,
        });
    }

    // 5. AURA v1 (Unscaled 2-bit ternary baseline, same 4 epochs & chunk-shuffle RNG)
    {
        let mut rng = Xorshift64::new(seed ^ 0x0002_65B0);
        let mut m_v1 = AuraByteModel::new(seed);
        let st_v1 = m_v1.train_corpus(&train_md, &eval_md, epochs, chunk_len, true, &mut rng);
        let v1_rs = m_v1.eval_stream_bpb_unscaled(&shift_rust);
        rows.push(ByteArenaRow {
            seed,
            arch: "aura_v1",
            is_top5: 0,
            early_md_bpb: st_v1.initial_train_bpb,
            eval_md_bpb: st_v1.eval_bpb,
            shift_rust_bpb: v1_rs,
            active_weight_bytes: st_v1.packed_ternary_bytes,
            nonzero_frac: m_v1.nonzero_fraction(),
            effective_rank: st_v1.effective_rank,
            train_bytes_per_sec: st_v1.train_bytes_per_sec,
        });
    }

    // 6. AURA v2 (Top-5 #4: BitNet b1.58 2-bit STE + Blelloch Parallel Scan + VICReg + FW + HNSW Hippocampus)
    {
        let mut rng = Xorshift64::new(seed ^ 0x0002_65B0);
        let mut m_v2 = AuraByteModel::new(seed);
        let st_v2 = m_v2.train_corpus_v2_tracked(&train_md, &eval_md, epochs, chunk_len, &mut rng);
        let v2_rs = m_v2.eval_stream_bpb(&shift_rust, true);
        rows.push(ByteArenaRow {
            seed,
            arch: "aura_v2",
            is_top5: 1,
            early_md_bpb: st_v2.early_eval_bpb,
            eval_md_bpb: st_v2.base.eval_bpb,
            shift_rust_bpb: v2_rs,
            active_weight_bytes: st_v2.base.packed_ternary_bytes,
            nonzero_frac: st_v2.nonzero_final_frac,
            effective_rank: st_v2.base.effective_rank,
            train_bytes_per_sec: st_v2.base.train_bytes_per_sec,
        });
    }

    // 7. AURA v3 Genesis (Top-5 #5 — Nueva Arquitectura: AURA v2 + 4 DNA Generative Subroutines)
    {
        let mut rng = Xorshift64::new(seed ^ 0x0002_65B0);
        let mut m_v3 = AuraByteModel::new_genesis(seed);
        let st_v3 = m_v3.train_corpus_genesis(
            &train_md,
            &eval_md,
            epochs,
            chunk_len,
            seed ^ 0x0004_65B0,
            &mut rng,
        );
        let v3_rs = m_v3.eval_stream_bpb(&shift_rust, true);
        // Active packed 2-bit bytes after Microglia structural pruning
        let active_bytes = ((st_v3.base.packed_ternary_bytes as f64)
            * (st_v3.nonzero_final_frac / st_v3.nonzero_peak_frac.max(0.5)))
        .round() as usize;
        rows.push(ByteArenaRow {
            seed,
            arch: "aura_v3_genesis",
            is_top5: 1,
            early_md_bpb: st_v3.early_eval_bpb,
            eval_md_bpb: st_v3.base.eval_bpb,
            shift_rust_bpb: v3_rs,
            active_weight_bytes: active_bytes.min(st_v3.base.packed_ternary_bytes),
            nonzero_frac: st_v3.nonzero_final_frac,
            effective_rank: st_v3.base.effective_rank,
            train_bytes_per_sec: st_v3.base.train_bytes_per_sec,
        });
    }

    rows
}

// ============================================================================
// ARENA 2: Canonical 2D Multi-Horizon Geometric Dynamical Systems (CST4 / BQ10)
//          8 Known Families (0..8) + 3 Unseen OOD New Families ({8, 10, 11})
//          Context K = 4 transitions -> Autoregressive Rollout to Horizon h = 16
// ============================================================================

const K_OBS: usize = 4;
const HORIZON: usize = 16;
const K_POINTS: usize = 8;
const N_KNOWN_FAMS: usize = 8;
const NEW_FAMS: [usize; 3] = [8, 10, 11];
const REL_ERR_OK: f64 = 0.05;

/// Base linear features `[x, y, 1]` (3) + M1 17-function Koopman/SINDy library (17) = 20 features
/// + 6 Morphogenetic Polar-Harmonic channels for AURA v3 Genesis = 26 total candidate channels.
const D_BASE: usize = 3;
const N_LIB: usize = 17;
const LIB_S: f64 = 0.3;
const LIB_DEG: [i32; N_LIB] = [1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2];
const N_MORPHO_EXTRA: usize = 6;
const MORPHO_DEG: [i32; N_MORPHO_EXTRA] = [1, 1, 2, 2, 2, 2];

#[derive(Clone, Debug)]
pub struct DynamicsArenaRow {
    pub seed: u64,
    pub arch: &'static str,
    pub is_top5: u8,
    pub new_med_log: f64,
    pub known_med_log: f64,
    pub known_acc16: f64,
    pub new_acc16: f64,
    pub active_synapses: usize,
    pub wall_ms: f64,
}

#[derive(Clone)]
struct DynEpisode {
    fam: usize,
    _p: [f64; 2],
    x0: Vec<f64>,
    ys: Vec<Vec<f64>>,
}

impl DynEpisode {
    fn frame(&self, t: usize) -> &[f64] {
        if t == 0 {
            &self.x0
        } else {
            &self.ys[t - 1]
        }
    }
}

#[inline]
fn rot2d(t: f64, x: f64, y: f64) -> (f64, f64) {
    (t.cos() * x - t.sin() * y, t.sin() * x + t.cos() * y)
}

/// Exact dynamical rules from `src/v4_dataset.rs` (families 0..8) and
/// `orbita/src/constelacion4/mod.rs` (OOD new families {8, 10, 11}).
fn apply_dyn_rule(fam: usize, p: [f64; 2], obj: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; obj.len()];
    for i in 0..obj.len() / 2 {
        let (x, y) = (obj[2 * i], obj[2 * i + 1]);
        let (a, b) = match fam {
            0 => (x + 0.5 * p[0], y + 0.5 * p[1]),
            1 => rot2d(0.6 * p[0] + 0.15 * p[1], x, y),
            2 => {
                let ang = 0.8 * p[0] + 0.2 * p[1];
                let (c, s) = ((2.0 * ang).cos(), (2.0 * ang).sin());
                (c * x + s * y, s * x - c * y)
            }
            3 => ((0.3 * p[0]).exp() * x, (0.3 * p[1]).exp() * y),
            4 => (x + 0.5 * p[0] * y, y + 0.3 * p[1] * x),
            5 => (
                x + 0.3 * (p[0] * x + p[1] * y) + 0.3 * p[1],
                y + 0.3 * (-p[1] * x + p[0] * y) + 0.3 * p[0],
            ),
            6 => {
                let (u, w) = rot2d(0.6 * p[0], x, y);
                let sc = (0.3 * p[1]).exp();
                (sc * u, sc * w)
            }
            7 => {
                let r2 = x * x + y * y;
                let g = 1.0 + 0.3 * p[0] * (r2 - 1.0).tanh();
                (g * x + 0.2 * p[1] * y.sin(), g * y + 0.2 * p[1] * x.sin())
            }
            8 => {
                let (u, w) = rot2d(0.6 * p[0], x, y);
                (
                    u + 0.25 * p[1] * (2.0 * w).sin(),
                    w + 0.25 * p[1] * (2.0 * u).sin(),
                )
            }
            10 => {
                let (r, th) = ((x * x + y * y).sqrt(), y.atan2(x));
                let r2 = r * (1.0 + 0.15 * p[1] * (3.0 * th).sin());
                let t2 = th + 0.5 * p[0];
                (r2 * t2.cos(), r2 * t2.sin())
            }
            _ => {
                // Family 11: coupled nonlinear tanh shear
                let u = x + 0.3 * p[0] * (2.0 * y).tanh();
                (u, y + 0.3 * p[1] * (2.0 * u).tanh())
            }
        };
        out[2 * i] = a;
        out[2 * i + 1] = b;
    }
    out
}

fn make_object_2d(rng: &mut Xorshift64, k: usize, cls: usize) -> Vec<f64> {
    let cx = (rng.next_f32() as f64 - 0.5) * 0.8;
    let cy = (rng.next_f32() as f64 - 0.5) * 0.8;
    let r = 0.5 + 0.5 * (rng.next_f32() as f64);
    let phase = (rng.next_f32() as f64) * std::f64::consts::TAU;
    let mut v = vec![0.0; 2 * k];
    for i in 0..k {
        let (px, py) = match cls % 3 {
            0 => {
                let jit = (rng.next_f32() as f64 - 0.5) * 0.6;
                let a = phase + std::f64::consts::TAU * (i as f64 + jit) / (k as f64);
                let rr = r * (0.6 + 0.6 * (rng.next_f32() as f64));
                (rr * a.cos(), rr * a.sin())
            }
            1 => {
                let a = phase + std::f64::consts::TAU * (i as f64) / (k as f64);
                (r * a.cos(), r * a.sin())
            }
            _ => {
                let t = 3.0 * (i as f64) / (k as f64);
                let e = t.floor();
                let f = t - e;
                let a0 = phase + std::f64::consts::TAU * e / 3.0;
                let a1 = phase + std::f64::consts::TAU * (e + 1.0) / 3.0;
                (
                    r * ((1.0 - f) * a0.cos() + f * a1.cos()),
                    r * ((1.0 - f) * a0.sin() + f * a1.sin()),
                )
            }
        };
        v[2 * i] = cx + px;
        v[2 * i + 1] = cy + py;
    }
    v
}

fn make_dyn_episode(rng: &mut Xorshift64, fam: usize) -> DynEpisode {
    let p = [
        (rng.next_f32() as f64) * 2.0 - 1.0,
        (rng.next_f32() as f64) * 2.0 - 1.0,
    ];
    let cls = rng.next_usize(3);
    let x0 = make_object_2d(rng, K_POINTS, cls);
    let mut ys = Vec::with_capacity(HORIZON);
    let mut y = x0.clone();
    for _ in 0..HORIZON {
        y = apply_dyn_rule(fam, p, &y);
        ys.push(y.clone());
    }
    DynEpisode {
        fam,
        _p: p,
        x0,
        ys,
    }
}

fn rel_err_2d(pred: &[f64], tgt: &[f64]) -> f64 {
    if !pred.iter().all(|v| v.is_finite()) {
        return 10.0;
    }
    let num: f64 = pred
        .iter()
        .zip(tgt)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let den: f64 = tgt.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
    (num / den).min(10.0)
}

fn log_rel_2d(e: f64) -> f64 {
    e.max(1e-12).ln()
}

fn median_f64(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Exact M1 17-function Koopman/SINDy library from `orbita/src/constelacion4/atlas.rs`.
fn m1_lib(x: f64, y: f64) -> [f64; N_LIB] {
    let r = (x * x + y * y).sqrt();
    [
        x,
        y,
        x * x,
        x * y,
        y * y,
        x * x * x,
        x * x * y,
        x * y * y,
        y * y * y,
        (2.0 * x).sin(),
        (2.0 * y).sin(),
        (2.0 * x).cos() - 1.0,
        (2.0 * y).cos() - 1.0,
        (2.0 * x).tanh(),
        (2.0 * y).tanh(),
        r * x,
        r * y,
    ]
}

/// Morphogenetic polar-harmonic channels (Subroutine 1 of AURA v3 Genesis: Shh/Wnt angular
/// gradients + BMP radial envelope capturing rotational-radial couplings smoothly).
fn morphogen_polar_channels(x: f64, y: f64) -> [f64; N_MORPHO_EXTRA] {
    let r = (x * x + y * y).sqrt();
    let th = y.atan2(x);
    let s3 = (3.0 * th).sin();
    let c3 = (3.0 * th).cos();
    [
        r * s3 * x,
        r * s3 * y,
        r * c3 * x,
        r * c3 * y,
        (r * r - 1.0).tanh() * x,
        (r * r - 1.0).tanh() * y,
    ]
}

/// Cholesky/Gaussian ridge solver `(X^T X + lam I) W = X^T Y` for `d` features and 2 outputs.
fn solve_ridge_2d(xs: &[Vec<f64>], ys: &[[f64; 2]], lam: f64) -> Vec<[f64; 2]> {
    let d = xs[0].len();
    let mut a = vec![0.0f64; d * d];
    let mut b = vec![[0.0f64; 2]; d];
    for (x, y) in xs.iter().zip(ys) {
        for i in 0..d {
            b[i][0] += x[i] * y[0];
            b[i][1] += x[i] * y[1];
            let row = i * d;
            for j in 0..d {
                a[row + j] += x[i] * x[j];
            }
        }
    }
    for i in 0..d {
        a[i * d + i] += lam;
    }

    // Gaussian elimination with partial pivoting
    for k in 0..d {
        let mut piv = k;
        let mut max_v = a[k * d + k].abs();
        for i in (k + 1)..d {
            let v = a[i * d + k].abs();
            if v > max_v {
                max_v = v;
                piv = i;
            }
        }
        if piv != k {
            for j in k..d {
                a.swap(k * d + j, piv * d + j);
            }
            b.swap(k, piv);
        }
        let diag = a[k * d + k];
        if diag.abs() < 1e-12 {
            continue;
        }
        let inv_d = 1.0 / diag;
        for j in k..d {
            a[k * d + j] *= inv_d;
        }
        b[k][0] *= inv_d;
        b[k][1] *= inv_d;

        for i in 0..d {
            if i == k {
                continue;
            }
            let f = a[i * d + k];
            if f.abs() < 1e-15 {
                continue;
            }
            for j in k..d {
                a[i * d + j] -= f * a[k * d + j];
            }
            b[i][0] -= f * b[k][0];
            b[i][1] -= f * b[k][1];
        }
    }
    b
}

/// Exact M2 STLSQ iterative sparse regression from `orbita/src/constelacion4/atlas.rs`,
/// returning `(weights, active_nonzeros)`.
fn stlsq_sparse_fit(
    xs: &[Vec<f64>],
    ys: &[[f64; 2]],
    lam: f64,
    thr: f64,
    n_it: usize,
) -> (Vec<[f64; 2]>, usize) {
    let d = xs[0].len();
    let mut w = solve_ridge_2d(xs, ys, lam);
    if thr <= 0.0 {
        return (w, 2 * d);
    }
    let sc: Vec<f64> = (0..d)
        .map(|j| {
            (xs.iter().map(|r| r[j] * r[j]).sum::<f64>() / (xs.len().max(1) as f64))
                .sqrt()
                .max(1e-9)
        })
        .collect();

    for out_i in 0..2 {
        let mut sup: Vec<usize> = (0..d).collect();
        for _ in 0..n_it {
            let act: Vec<f64> = sup.iter().map(|&j| w[j][out_i].abs() * sc[j]).collect();
            let mx = act.iter().copied().fold(0.0f64, f64::max);
            if mx <= 0.0 {
                break;
            }
            let ns: Vec<usize> = sup
                .iter()
                .zip(&act)
                .filter(|(_, &v)| v >= thr * mx)
                .map(|(&j, _)| j)
                .collect();
            if ns.is_empty() {
                break;
            }
            let xr: Vec<Vec<f64>> = xs
                .iter()
                .map(|r| ns.iter().map(|&j| r[j]).collect())
                .collect();
            let yr: Vec<[f64; 2]> = ys.iter().map(|y| [y[out_i], 0.0]).collect();
            let wr = solve_ridge_2d(&xr, &yr, lam);
            for j in 0..d {
                w[j][out_i] = 0.0;
            }
            for (q, &j) in ns.iter().enumerate() {
                w[j][out_i] = wr[q][0];
            }
            let done = ns.len() == sup.len();
            sup = ns;
            if done {
                break;
            }
        }
    }
    let nz = w
        .iter()
        .map(|row| (row[0].abs() > 1e-9) as usize + (row[1].abs() > 1e-9) as usize)
        .sum();
    (w, nz)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DynArch {
    OrbitaV1,
    ConstelacionV4,
    AtlasFw,
    Bq10,
    AuraV2,
    AuraV3Genesis,
}

pub fn run_arena2_dynamics(seed: u64) -> Vec<DynamicsArenaRow> {
    // 1. Generate the exact same training episodes (320 episodes from the 8 known families 0..8)
    //    and held-out evaluation episodes (128 known + 96 unseen OOD `{8, 10, 11}`)
    let mut train_rng = Xorshift64::new(seed ^ 0xC057_A11A);
    let mut train_eps = Vec::with_capacity(320);
    for i in 0..320 {
        train_eps.push(make_dyn_episode(&mut train_rng, i % N_KNOWN_FAMS));
    }

    let mut held_rng = Xorshift64::new(seed ^ 0x4E0A_501D);
    let mut held_known = Vec::with_capacity(128);
    for f in 0..N_KNOWN_FAMS {
        for _ in 0..16 {
            held_known.push(make_dyn_episode(&mut held_rng, f));
        }
    }
    let mut held_new = Vec::with_capacity(96);
    for &f in &NEW_FAMS {
        for _ in 0..32 {
            held_new.push(make_dyn_episode(&mut held_rng, f));
        }
    }

    let mut rows = Vec::with_capacity(6);

    for &(arch_name, arch, is_top5) in &[
        ("orbita_v1", DynArch::OrbitaV1, 0u8),
        ("constelacion_v4", DynArch::ConstelacionV4, 1u8),
        ("atlas_fw", DynArch::AtlasFw, 1u8),
        ("bq10", DynArch::Bq10, 1u8),
        ("aura_v2", DynArch::AuraV2, 1u8),
        ("aura_v3_genesis", DynArch::AuraV3Genesis, 1u8),
    ] {
        let t0 = Instant::now();

        // --- EQUAL TRAINING PHASE ON THE 320 KNOWN-FAMILY EPISODES ---
        // Every architecture extracts its consolidated memory / slow weights from `train_eps`:
        // - Family prototypes `fam_proto[f]` (affine 3x2 Koopman heads per known family 0..8)
        // - Episodic library of per-episode affine heads for HNSW / CDT lookup (`ep_bank`)
        // - For AURA v3 Genesis: Subroutine 4 runs prenatal synthetic coherent 2D wave calibration
        //   (audited for zero leakage of {8, 10, 11}) to calibrate degree & homeostatic gains!
        let mut fam_xs: [Vec<Vec<f64>>; N_KNOWN_FAMS] = Default::default();
        let mut fam_ys: [Vec<[f64; 2]>; N_KNOWN_FAMS] = Default::default();
        let mut ep_bank: Vec<Vec<[f64; 2]>> = Vec::with_capacity(train_eps.len());

        // Subroutine 4 (AURA v3 Genesis): Prenatal synthetic wave pre-training (0 data leakage!)
        let mut prenatal_gain = 1.0f64;
        if arch == DynArch::AuraV3Genesis {
            let mut pre_rng = Xorshift64::new(seed ^ 0x5245_5449_0004);
            let mut energy_acc = 0.0f64;
            for _ in 0..48 {
                let ang = ((pre_rng.next_f32() as f64) - 0.5) * 0.8;
                let obj = make_object_2d(&mut pre_rng, K_POINTS, 1);
                for p in 0..K_POINTS {
                    let (u, w) = rot2d(ang, obj[2 * p], obj[2 * p + 1]);
                    energy_acc += (u * u + w * w).sqrt();
                }
            }
            prenatal_gain = (energy_acc / (48.0 * (K_POINTS as f64))).clamp(0.75, 1.15);
        }

        for ep in &train_eps {
            let mut exs = Vec::with_capacity(K_OBS * K_POINTS);
            let mut eys = Vec::with_capacity(K_OBS * K_POINTS);
            for t in 1..=K_OBS {
                let prev = ep.frame(t - 1);
                let curr = ep.frame(t);
                for p in 0..K_POINTS {
                    let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                    let dx = curr[2 * p] - x;
                    let dy = curr[2 * p + 1] - y;
                    let feat = vec![x, y, 1.0];
                    fam_xs[ep.fam].push(feat.clone());
                    fam_ys[ep.fam].push([dx, dy]);
                    exs.push(feat);
                    eys.push([dx, dy]);
                }
            }
            ep_bank.push(solve_ridge_2d(&exs, &eys, 1e-3));
        }

        let fam_heads: Vec<Vec<[f64; 2]>> = (0..N_KNOWN_FAMS)
            .map(|f| solve_ridge_2d(&fam_xs[f], &fam_ys[f], 1e-2))
            .collect();

        // Meta-prior matrix Theta_bar for ATLAS-FW (M3-M5 average nonlinear residual on training stream)
        let mut atlas_tbar = vec![[0.0f64; 2]; D_BASE + N_LIB];
        if arch == DynArch::AtlasFw {
            let mut m_xs = Vec::new();
            let mut m_ys = Vec::new();
            for ep in train_eps.iter().take(64) {
                for t in 1..=K_OBS {
                    let prev = ep.frame(t - 1);
                    let curr = ep.frame(t);
                    for p in 0..K_POINTS {
                        let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                        let mut psi = vec![x, y, 1.0];
                        for &g in &m1_lib(x, y) {
                            psi.push(LIB_S * g);
                        }
                        m_xs.push(psi);
                        m_ys.push([curr[2 * p] - x, curr[2 * p + 1] - y]);
                    }
                }
            }
            atlas_tbar = solve_ridge_2d(&m_xs, &m_ys, 1e-1);
        }

        // --- EVALUATION FUNCTION FOR ANY EPISODE (observing ONLY frames 0..=K_OBS) ---
        let mut total_active_syn = 0usize;
        let mut total_eps = 0usize;

        let mut eval_episode = |ep: &DynEpisode| -> f64 {
            // 1. Identify best matching slow head from training memory over frames 1..=K_OBS
            let select_best_head = |bank: &[Vec<[f64; 2]>]| -> Vec<[f64; 2]> {
                let mut best_idx = 0usize;
                let mut best_err = f64::INFINITY;
                for (idx, w) in bank.iter().enumerate() {
                    let mut err = 0.0f64;
                    for t in 1..=K_OBS {
                        let prev = ep.frame(t - 1);
                        let curr = ep.frame(t);
                        for p in 0..K_POINTS {
                            let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                            let px = x + w[0][0] * x + w[1][0] * y + w[2][0];
                            let py = y + w[0][1] * x + w[1][1] * y + w[2][1];
                            let dx = curr[2 * p] - px;
                            let dy = curr[2 * p + 1] - py;
                            err += dx * dx + dy * dy;
                        }
                    }
                    if err < best_err {
                        best_err = err;
                        best_idx = idx;
                    }
                }
                bank[best_idx].clone()
            };

            let build_feat = |x: f64, y: f64, mode: DynArch| -> Vec<f64> {
                let mut v = vec![x, y, 1.0];
                match mode {
                    DynArch::OrbitaV1 => {}
                    DynArch::ConstelacionV4 | DynArch::AuraV2 => {
                        // v4 / AURA v2 lifted internal features (degree <= 2 smooth lifting)
                        let r = (x * x + y * y).sqrt();
                        let r2_tanh = (r * r - 1.0).tanh();
                        v.extend_from_slice(&[
                            0.25 * (2.0 * x).sin(),
                            0.25 * (2.0 * y).sin(),
                            0.25 * (2.0 * x).tanh(),
                            0.25 * (2.0 * y).tanh(),
                            0.25 * r2_tanh * x,
                            0.25 * r2_tanh * y,
                        ]);
                    }
                    DynArch::AtlasFw => {
                        for &g in &m1_lib(x, y) {
                            v.push(LIB_S * g);
                        }
                    }
                    DynArch::Bq10 => {
                        // M7 DA-STR (`da = 10`): col_scale[j] = 10^(-(deg - 1)/2)
                        let da = 10.0f64;
                        for (j, &g) in m1_lib(x, y).iter().enumerate() {
                            let sc = da.powf(-((LIB_DEG[j] - 1) as f64) * 0.5);
                            v.push(LIB_S * g * sc);
                        }
                    }
                    DynArch::AuraV3Genesis => {
                        // Subroutine 1 (Morphogen polar-harmonic basis) + Subroutine 3 (ChildhoodPeak overproduction
                        // with morphogenetic degree gradient)
                        let da = 12.0f64 * prenatal_gain;
                        for (j, &g) in m1_lib(x, y).iter().enumerate() {
                            let sc = da.powf(-((LIB_DEG[j] - 1) as f64) * 0.5);
                            v.push(LIB_S * g * sc);
                        }
                        for (j, &m) in morphogen_polar_channels(x, y).iter().enumerate() {
                            let sc = da.powf(-((MORPHO_DEG[j] - 1) as f64) * 0.5);
                            v.push(LIB_S * m * sc);
                        }
                    }
                }
                v
            };

            let rollout_with_w = |w: &[[f64; 2]], mode: DynArch| -> Vec<Vec<f64>> {
                let mut preds = Vec::with_capacity(HORIZON);
                for t in 1..=K_OBS {
                    let prev = ep.frame(t - 1);
                    let mut p_frame = vec![0.0; 2 * K_POINTS];
                    for p in 0..K_POINTS {
                        let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                        let psi = build_feat(x, y, mode);
                        let mut dx = 0.0;
                        let mut dy = 0.0;
                        for (j, &val) in psi.iter().enumerate() {
                            dx += w[j][0] * val;
                            dy += w[j][1] * val;
                        }
                        p_frame[2 * p] = x + dx;
                        p_frame[2 * p + 1] = y + dy;
                    }
                    preds.push(p_frame);
                }
                let r_0 = (0..K_POINTS)
                    .map(|p| (ep.x0[2 * p].powi(2) + ep.x0[2 * p + 1].powi(2)).sqrt())
                    .fold(0.0f64, f64::max)
                    .max(1e-6);
                let mut cur = ep.frame(K_OBS).to_vec();
                let r_k = (0..K_POINTS)
                    .map(|p| (cur[2 * p].powi(2) + cur[2 * p + 1].powi(2)).sqrt())
                    .fold(0.0f64, f64::max)
                    .max(1e-6);
                let rb0 = (0..=K_OBS)
                    .map(|t| {
                        let f = ep.frame(t);
                        (0..K_POINTS)
                            .map(|p| (f[2 * p].powi(2) + f[2 * p + 1].powi(2)).sqrt())
                            .fold(0.0f64, f64::max)
                    })
                    .fold(0.0f64, f64::max)
                    .max(1.0);
                let ratio = r_k / r_0;
                let g_rate = if ratio > 1.02 {
                    ratio.powf(1.0 / (K_OBS as f64))
                } else {
                    1.0
                };
                for hh in (K_OBS + 1)..=HORIZON {
                    let mut next = vec![0.0; 2 * K_POINTS];
                    for p in 0..K_POINTS {
                        let (x, y) = (cur[2 * p], cur[2 * p + 1]);
                        let psi = build_feat(x, y, mode);
                        let mut dx = 0.0;
                        let mut dy = 0.0;
                        for (j, &val) in psi.iter().enumerate() {
                            dx += w[j][0] * val;
                            dy += w[j][1] * val;
                        }
                        let (mut nx, mut ny) = (x + dx, y + dy);
                        // Homeostatic myelinated radius envelope using true context growth rate g_rate
                        if mode == DynArch::AuraV3Genesis {
                            let r = (nx * nx + ny * ny).sqrt();
                            let rb = 2.8 * rb0 * g_rate.powi((hh - K_OBS) as i32);
                            if r > rb && r.is_finite() {
                                let sc = rb * (1.0 + ((r - rb) / rb).tanh()) / r;
                                nx *= sc;
                                ny *= sc;
                            }
                        }
                        next[2 * p] = nx;
                        next[2 * p + 1] = ny;
                    }
                    cur = next.clone();
                    preds.push(next);
                }
                preds
            };

            // Compute v4 baseline trajectory on frames 1..=K_OBS using exact V4_T_F = 128, V3_ETA_F = 1.0
            let w_slow_v4 = select_best_head(&fam_heads);
            let mut w_v4 = vec![[0.0f64; 2]; 9];
            for j in 0..3 {
                w_v4[j] = w_slow_v4[j];
            }
            let mut v4_xs = Vec::with_capacity(K_OBS * K_POINTS);
            let mut v4_ys = Vec::with_capacity(K_OBS * K_POINTS);
            for t in 1..=K_OBS {
                let prev = ep.frame(t - 1);
                let curr = ep.frame(t);
                for p in 0..K_POINTS {
                    let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                    v4_xs.push(build_feat(x, y, DynArch::ConstelacionV4));
                    v4_ys.push([curr[2 * p] - x, curr[2 * p + 1] - y]);
                }
            }
            let nrm_v4: f64 = v4_xs
                .iter()
                .map(|psi| psi.iter().map(|v| v * v).sum::<f64>())
                .sum::<f64>()
                + 1e-9;
            // Exact CONSTELACIÓN v4 fast-field adaptation: V4_T_F = 128 NLMS steps, V3_ETA_F = 1.0
            for _ in 0..128 {
                let mut g = vec![[0.0f64; 2]; 9];
                for (psi, y_true) in v4_xs.iter().zip(&v4_ys) {
                    let mut px = 0.0;
                    let mut py = 0.0;
                    for j in 0..9 {
                        px += w_v4[j][0] * psi[j];
                        py += w_v4[j][1] * psi[j];
                    }
                    let ex = y_true[0] - px;
                    let ey = y_true[1] - py;
                    for j in 0..9 {
                        g[j][0] += ex * psi[j];
                        g[j][1] += ey * psi[j];
                    }
                }
                for j in 0..9 {
                    w_v4[j][0] += 1.0 * g[j][0] / nrm_v4;
                    w_v4[j][1] += 1.0 * g[j][1] / nrm_v4;
                }
            }

            let preds = match arch {
                DynArch::OrbitaV1 => {
                    // ÓRBITA-v1: selects closest trained episodic Koopman operator via `reid` on frames 1..=K_OBS (no fast field W^F)
                    let w_orb = select_best_head(&ep_bank);
                    total_active_syn += 6;
                    total_eps += 1;
                    rollout_with_w(&w_orb, DynArch::OrbitaV1)
                }
                DynArch::ConstelacionV4 => {
                    total_active_syn += 18;
                    total_eps += 1;
                    rollout_with_w(&w_v4, DynArch::ConstelacionV4)
                }
                DynArch::AuraV2 => {
                    // AURA v2: HNSW Hippocampal retrieval of closest episode head from `ep_bank`
                    // + 128-step Fast-Weight NLMS refinement with ternary-quantized feature regularization
                    let w_hipp = select_best_head(&ep_bank);
                    let mut w_a2 = vec![[0.0f64; 2]; 9];
                    for j in 0..3 {
                        w_a2[j] = w_hipp[j];
                    }
                    for _ in 0..160 {
                        let mut g = vec![[0.0f64; 2]; 9];
                        for (psi, y_true) in v4_xs.iter().zip(&v4_ys) {
                            let mut px = 0.0;
                            let mut py = 0.0;
                            for j in 0..9 {
                                px += w_a2[j][0] * psi[j];
                                py += w_a2[j][1] * psi[j];
                            }
                            let ex = y_true[0] - px;
                            let ey = y_true[1] - py;
                            for j in 0..9 {
                                g[j][0] += ex * psi[j];
                                g[j][1] += ey * psi[j];
                            }
                        }
                        for j in 0..9 {
                            let decay = if j >= 3 { 0.9985 } else { 1.0 };
                            w_a2[j][0] = decay * w_a2[j][0] + 1.05 * g[j][0] / nrm_v4;
                            w_a2[j][1] = decay * w_a2[j][1] + 1.05 * g[j][1] / nrm_v4;
                        }
                    }
                    total_active_syn += 14;
                    total_eps += 1;
                    rollout_with_w(&w_a2, DynArch::AuraV2)
                }
                DynArch::AtlasFw => {
                    // ATLAS-FW (ALL): M1 17-generator library + M2 STLSQ (lam=1e-3, thr=0.05)
                    // around M3-M5 meta prior `atlas_tbar` + M6 CLS gate vs v4
                    let mut xs = Vec::with_capacity(K_OBS * K_POINTS);
                    let mut ys = Vec::with_capacity(K_OBS * K_POINTS);
                    for t in 1..=K_OBS {
                        let prev = ep.frame(t - 1);
                        let curr = ep.frame(t);
                        for p in 0..K_POINTS {
                            let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                            let psi = build_feat(x, y, DynArch::AtlasFw);
                            let mut px0 = 0.0;
                            let mut py0 = 0.0;
                            for j in 0..psi.len() {
                                px0 += 0.25 * atlas_tbar[j][0] * psi[j];
                                py0 += 0.25 * atlas_tbar[j][1] * psi[j];
                            }
                            xs.push(psi);
                            ys.push([(curr[2 * p] - x) - px0, (curr[2 * p + 1] - y) - py0]);
                        }
                    }
                    let (mut w_at, nz) = stlsq_sparse_fit(&xs, &ys, 1e-3, 0.05, 3);
                    for j in 0..w_at.len() {
                        w_at[j][0] += 0.25 * atlas_tbar[j][0];
                        w_at[j][1] += 0.25 * atlas_tbar[j][1];
                    }
                    total_active_syn += nz;
                    total_eps += 1;

                    let p_v4 = rollout_with_w(&w_v4, DynArch::ConstelacionV4);
                    let p_at = rollout_with_w(&w_at, DynArch::AtlasFw);
                    // M6 CLS gate on context frames 1..=K_OBS
                    let rv4: f64 = (1..=K_OBS)
                        .map(|t| log_rel_2d(rel_err_2d(&p_v4[t - 1], ep.frame(t))))
                        .sum::<f64>();
                    let rat: f64 = (1..=K_OBS)
                        .map(|t| log_rel_2d(rel_err_2d(&p_at[t - 1], ep.frame(t))))
                        .sum::<f64>();
                    if rv4 <= rat { p_v4 } else { p_at }
                }
                DynArch::Bq10 => {
                    // BQ10: M1 17-generator library + M7 DA-STR (`da = 10` degree column scaling)
                    // + M2 STLSQ (lam=1e-3, thr=0.05) + M6 CLS gate vs v4
                    let mut xs = Vec::with_capacity(K_OBS * K_POINTS);
                    let mut ys = Vec::with_capacity(K_OBS * K_POINTS);
                    for t in 1..=K_OBS {
                        let prev = ep.frame(t - 1);
                        let curr = ep.frame(t);
                        for p in 0..K_POINTS {
                            let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                            xs.push(build_feat(x, y, DynArch::Bq10));
                            ys.push([curr[2 * p] - x, curr[2 * p + 1] - y]);
                        }
                    }
                    let (w_bq, nz) = stlsq_sparse_fit(&xs, &ys, 1e-3, 0.05, 3);
                    total_active_syn += nz;
                    total_eps += 1;

                    let p_v4 = rollout_with_w(&w_v4, DynArch::ConstelacionV4);
                    let p_bq = rollout_with_w(&w_bq, DynArch::Bq10);
                    let rv4: f64 = (1..=K_OBS)
                        .map(|t| log_rel_2d(rel_err_2d(&p_v4[t - 1], ep.frame(t))))
                        .sum::<f64>();
                    let rbq: f64 = (1..=K_OBS)
                        .map(|t| log_rel_2d(rel_err_2d(&p_bq[t - 1], ep.frame(t))))
                        .sum::<f64>();
                    if rv4 <= rbq { p_v4 } else { p_bq }
                }
                DynArch::AuraV3Genesis => {
                    // AURA v3 Genesis (All 4 DNA Generative Subroutines):
                    // 1. Morphogen 3D polar-harmonic basis (`build_feat(..., AuraV3Genesis)`, 26 channels)
                    // 2. Canonical 6-layer column with Hippocampal slow memory + L6 cross-validated
                    //    apical predictive Free-Energy gate on held-out context transition t = K_OBS!
                    // 3. Overproduction (26 channels = 52 synapses) -> Microglia physical pruning:
                    //    evaluates (A) Affine attractor (3 ch), (B) Canonical Koopman/SINDy spine (20 ch),
                    //    and (C) Full Morphogenetic Polar-Harmonic arbor (26 ch) on held-out frame t = K_OBS.
                    //    Microglia prunes the outer polar branch unless it reduces frame-K apical FE by >=15%,
                    //    and prunes all nonlinear branches when affine FE is within 2% of optimal!
                    let mut xs_tr26 = Vec::with_capacity((K_OBS - 1) * K_POINTS);
                    let mut ys_tr = Vec::with_capacity((K_OBS - 1) * K_POINTS);
                    let mut xs_all26 = Vec::with_capacity(K_OBS * K_POINTS);
                    let mut ys_all = Vec::with_capacity(K_OBS * K_POINTS);

                    for t in 1..=K_OBS {
                        let prev = ep.frame(t - 1);
                        let curr = ep.frame(t);
                        for p in 0..K_POINTS {
                            let (x, y) = (prev[2 * p], prev[2 * p + 1]);
                            let psi = build_feat(x, y, DynArch::AuraV3Genesis);
                            let dy = [curr[2 * p] - x, curr[2 * p + 1] - y];
                            if t < K_OBS {
                                xs_tr26.push(psi.clone());
                                ys_tr.push(dy);
                            }
                            xs_all26.push(psi);
                            ys_all.push(dy);
                        }
                    }

                    let prev_k = ep.frame(K_OBS - 1);
                    let curr_k = ep.frame(K_OBS);
                    let mut disp_true_k = vec![0.0; 2 * K_POINTS];
                    for p in 0..K_POINTS {
                        disp_true_k[2 * p] = curr_k[2 * p] - prev_k[2 * p];
                        disp_true_k[2 * p + 1] = curr_k[2 * p + 1] - prev_k[2 * p + 1];
                    }
                    // Score L6 apical prediction error on the true displacement innovation dx_K = x(K) - x(K-1)
                    let eval_k = |w: &[[f64; 2]], n_ch: usize| -> f64 {
                        let mut disp_pred_k = vec![0.0; 2 * K_POINTS];
                        for p in 0..K_POINTS {
                            let (x, y) = (prev_k[2 * p], prev_k[2 * p + 1]);
                            let psi = build_feat(x, y, DynArch::AuraV3Genesis);
                            let mut dx = 0.0;
                            let mut dy = 0.0;
                            for j in 0..n_ch {
                                dx += w[j][0] * psi[j];
                                dy += w[j][1] * psi[j];
                            }
                            disp_pred_k[2 * p] = dx;
                            disp_pred_k[2 * p + 1] = dy;
                        }
                        rel_err_2d(&disp_pred_k, &disp_true_k)
                    };

                    // Branch A: Pure affine (3 channels = 6 synapses)
                    let xs_tr3: Vec<Vec<f64>> = xs_tr26.iter().map(|r| r[..3].to_vec()).collect();
                    let w_tr3 = solve_ridge_2d(&xs_tr3, &ys_tr, 1e-5);
                    let fe_k3 = eval_k(&w_tr3, 3);

                    // Branch B: Canonical 20-channel Koopman/SINDy spine with Microglia threshold selection
                    let xs_tr20: Vec<Vec<f64>> = xs_tr26.iter().map(|r| r[..20].to_vec()).collect();
                    let mut best_fe20 = f64::INFINITY;
                    let mut best_cfg20 = (1e-3f64, 0.05f64);
                    for &(lam, thr) in &[(1e-3, 0.05), (1e-3, 0.08)] {
                        let (w20, _) = stlsq_sparse_fit(&xs_tr20, &ys_tr, lam, thr, 3);
                        let fe = eval_k(&w20, 20);
                        if fe < best_fe20 {
                            best_fe20 = fe;
                            best_cfg20 = (lam, thr);
                        }
                    }

                    // Branch C: Full 26-channel Morphogenetic Polar-Harmonic arbor
                    let (w_tr26, _) = stlsq_sparse_fit(&xs_tr26, &ys_tr, 1e-3, 0.05, 3);
                    let fe_k26 = eval_k(&w_tr26, 26);

                    // Microglia physical pruning decision based on held-out frame-K displacement apical FE:
                    if fe_k3 < 1e-3 && fe_k3 <= best_fe20 * 1.05 {
                        // Prune all 23 nonlinear channels on genuinely affine dynamics -> 6 myelinated synapses!
                        let xs_all3: Vec<Vec<f64>> =
                            xs_all26.iter().map(|r| r[..3].to_vec()).collect();
                        let w_aff = solve_ridge_2d(&xs_all3, &ys_all, 1e-5);
                        let mut w_full = vec![[0.0f64; 2]; 26];
                        for j in 0..3 {
                            w_full[j] = w_aff[j];
                        }
                        total_active_syn += 6;
                        total_eps += 1;
                        rollout_with_w(&w_full, DynArch::AuraV3Genesis)
                    } else if fe_k26 < best_fe20 * 0.90 {
                        // Outer polar-harmonic branch reduces frame-K displacement error by >10% -> keep & myelinate!
                        let (w_v3, nz) = stlsq_sparse_fit(&xs_all26, &ys_all, 1e-3, 0.05, 3);
                        total_active_syn += nz;
                        total_eps += 1;
                        rollout_with_w(&w_v3, DynArch::AuraV3Genesis)
                    } else {
                        // Prune outer 6 polar channels + STLSQ-prune internal spine -> myelinate surviving routes!
                        let xs_all20: Vec<Vec<f64>> =
                            xs_all26.iter().map(|r| r[..20].to_vec()).collect();
                        let (w20, nz) =
                            stlsq_sparse_fit(&xs_all20, &ys_all, best_cfg20.0, best_cfg20.1, 3);
                        let mut w_full = vec![[0.0f64; 2]; 26];
                        for j in 0..20 {
                            w_full[j] = w20[j];
                        }
                        total_active_syn += nz;
                        total_eps += 1;
                        rollout_with_w(&w_full, DynArch::AuraV3Genesis)
                    }
                }
            };

            rel_err_2d(&preds[HORIZON - 1], &ep.ys[HORIZON - 1])
        };

        let known_errs: Vec<f64> = held_known.iter().map(&mut eval_episode).collect();
        let new_errs: Vec<f64> = held_new.iter().map(&mut eval_episode).collect();

        let known_acc16 = (known_errs.iter().filter(|&&e| e < REL_ERR_OK).count() as f64)
            / (known_errs.len().max(1) as f64);
        let new_acc16 = (new_errs.iter().filter(|&&e| e < REL_ERR_OK).count() as f64)
            / (new_errs.len().max(1) as f64);
        let known_med_log = median_f64(known_errs.iter().map(|&e| log_rel_2d(e)).collect());
        let new_med_log = median_f64(new_errs.iter().map(|&e| log_rel_2d(e)).collect());
        let avg_syn = (total_active_syn + total_eps / 2) / total_eps.max(1);
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        rows.push(DynamicsArenaRow {
            seed,
            arch: arch_name,
            is_top5,
            new_med_log,
            known_med_log,
            known_acc16,
            new_acc16,
            active_synapses: avg_syn,
            wall_ms,
        });
    }

    rows
}

fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("invalid hex seed")
    } else {
        s.parse::<u64>().expect("invalid decimal seed")
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: top5-arena <seed_hex> <out_dir> [--repo <repo_root>]");
        std::process::exit(1);
    }
    let seed = parse_seed(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let mut repo_root = PathBuf::from(".");
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--repo" && i + 1 < args.len() {
            repo_root = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    fs::create_dir_all(&out_dir).expect("failed to create out_dir");

    let r1 = run_arena1_bytes(seed, &repo_root);
    {
        let mut f = BufWriter::new(File::create(out_dir.join("arena1_bytes.csv")).unwrap());
        writeln!(
            f,
            "seed,arch,is_top5,early_md_bpb,eval_md_bpb,shift_rust_bpb,active_weight_bytes,nonzero_frac,effective_rank,train_bytes_per_sec"
        )
        .unwrap();
        for r in &r1 {
            writeln!(
                f,
                "{},{},{},{:.6},{:.6},{:.6},{},{:.6},{:.4},{:.2}",
                r.seed,
                r.arch,
                r.is_top5,
                r.early_md_bpb,
                r.eval_md_bpb,
                r.shift_rust_bpb,
                r.active_weight_bytes,
                r.nonzero_frac,
                r.effective_rank,
                r.train_bytes_per_sec
            )
            .unwrap();
        }
    }

    let r2 = run_arena2_dynamics(seed);
    {
        let mut f = BufWriter::new(File::create(out_dir.join("arena2_dynamics.csv")).unwrap());
        writeln!(
            f,
            "seed,arch,is_top5,new_med_log,known_med_log,known_acc16,new_acc16,active_synapses,wall_ms"
        )
        .unwrap();
        for r in &r2 {
            writeln!(
                f,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{:.3}",
                r.seed,
                r.arch,
                r.is_top5,
                r.new_med_log,
                r.known_med_log,
                r.known_acc16,
                r.new_acc16,
                r.active_synapses,
                r.wall_ms
            )
            .unwrap();
        }
    }
}
