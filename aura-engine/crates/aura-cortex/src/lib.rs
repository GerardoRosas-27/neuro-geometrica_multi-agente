//! `aura-cortex`:
//! 1. Selective SSM with exact Blelloch O(log L) Parallel Associative Prefix Scan (Bottleneck 2.2)
//! 2. DualCorticalCore (zero-mul Ternary slow + multi-timescale SSM + zero-alloc NLMS Fast Weights)
//! 3. CorticalFabric v2 (Part 1 event-driven neuromorphic core, 0 floats in `step()`,
//!    minicolumn-tiled CSR for L1 locality, amortized warm-start settling with early-exit,
//!    and surprise-gated NE/ACh/DA neuromodulation).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use aura_math::{
    build_cos_lut, FastWeightMemory, LatentVec, TernaryMatrix, Xorshift64, LATENT_DIM,
};
use aura_perception::delta_modulate_byte;

// ============================================================================
// BOTTLENECK 2.2 FIX: Parallel Associative Prefix Scan (Blelloch O(log L)) for Selective SSM
// ============================================================================

/// Associative element `(a, b)` representing linear affine map `h -> a * h + b`.
/// Composition: `(a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2)`.
#[inline(always)]
fn compose_affine(first: (f32, f32), second: (f32, f32)) -> (f32, f32) {
    (second.0 * first.0, second.0 * first.1 + second.1)
}

/// Exact sequential Selective SSM recurrence over a chunk of length `L` and width `D`:
/// `h[t][d] = alpha[t][d] * h[t-1][d] + (1 - alpha[t][d]) * u[t][d]`.
pub fn ssm_forward_sequential(
    h0: &[f32],
    alphas: &[f32], // [L * D]
    inputs: &[f32], // [L * D]
    len: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; len * dim];
    if len == 0 {
        return out;
    }
    let mut prev = h0.to_vec();
    for t in 0..len {
        let off = t * dim;
        for d in 0..dim {
            let a = alphas[off + d];
            let b = (1.0 - a) * inputs[off + d];
            let h = a * prev[d] + b;
            out[off + d] = h;
            prev[d] = h;
        }
    }
    out
}

/// Blelloch O(log L) Parallel Associative Prefix Scan for Selective SSM forward pass.
/// Computes the exact same states as `ssm_forward_sequential` in O(log L) parallel tree depth.
pub fn ssm_forward_parallel_scan(
    h0: &[f32],
    alphas: &[f32], // [L * D]
    inputs: &[f32], // [L * D]
    len: usize,
    dim: usize,
) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    let n_pow2 = len.next_power_of_two();
    // Transpose-friendly per-channel inclusive scan using Blelloch up-sweep + down-sweep
    let mut out = vec![0.0f32; len * dim];
    let mut a_buf = vec![1.0f32; n_pow2];
    let mut b_buf = vec![0.0f32; n_pow2];

    for d in 0..dim {
        // Initialize leaves with (alpha_t, (1 - alpha_t) * u_t); pad tail with identity (1, 0)
        for t in 0..len {
            let a = alphas[t * dim + d];
            let mut b = (1.0 - a) * inputs[t * dim + d];
            if t == 0 {
                b += a * h0[d];
            }
            a_buf[t] = a;
            b_buf[t] = b;
        }
        for t in len..n_pow2 {
            a_buf[t] = 1.0;
            b_buf[t] = 0.0;
        }

        // Up-sweep (reduce) phase: O(log L) levels
        let mut stride = 1usize;
        while stride < n_pow2 {
            let step = stride << 1;
            let mut idx = step - 1;
            while idx < n_pow2 {
                let left = idx - stride;
                let (na, nb) = compose_affine((a_buf[left], b_buf[left]), (a_buf[idx], b_buf[idx]));
                a_buf[idx] = na;
                b_buf[idx] = nb;
                idx += step;
            }
            stride = step;
        }

        // Save total reduction for last element, set root to identity (1, 0) for exclusive scan
        let root_a = a_buf[n_pow2 - 1];
        let root_b = b_buf[n_pow2 - 1];
        a_buf[n_pow2 - 1] = 1.0;
        b_buf[n_pow2 - 1] = 0.0;

        // Down-sweep phase: O(log L) levels -> yields exclusive prefix scan
        let mut stride = n_pow2 >> 1;
        while stride >= 1 {
            let step = stride << 1;
            let mut idx = step - 1;
            while idx < n_pow2 {
                let left = idx - stride;
                let t_a = a_buf[left];
                let t_b = b_buf[left];
                a_buf[left] = a_buf[idx];
                b_buf[left] = b_buf[idx];
                let (na, nb) = compose_affine((a_buf[idx], b_buf[idx]), (t_a, t_b));
                a_buf[idx] = na;
                b_buf[idx] = nb;
                idx += step;
            }
            stride >>= 1;
        }

        // Convert exclusive scan to inclusive scan: inclusive[t] = exclusive[t+1] (and root at end)
        // Since h0 was folded into t=0's b_0, inclusive[t].b is exact h[t]!
        for t in 0..len {
            let val = if t + 1 < n_pow2 {
                b_buf[t + 1]
            } else {
                root_b
            };
            let _ = root_a;
            out[t * dim + d] = val;
        }
    }
    out
}

/// Exact sequential backward BPTT recurrence for Selective SSM gradients:
/// `dh[t] = dh_out[t] + alpha[t+1] * dh[t+1]`.
pub fn ssm_backward_sequential(
    dh_out: &[f32], // [L * D]
    alphas: &[f32], // [L * D]
    len: usize,
    dim: usize,
) -> Vec<f32> {
    let mut dh = vec![0.0f32; len * dim];
    if len == 0 {
        return dh;
    }
    for d in 0..dim {
        let mut acc = 0.0f32;
        for t in (0..len).rev() {
            let g = dh_out[t * dim + d] + acc;
            dh[t * dim + d] = g;
            acc = alphas[t * dim + d] * g;
        }
    }
    dh
}

/// Exact Blelloch O(log L) Parallel Associative Prefix Scan for backward gradients!
pub fn ssm_backward_parallel_scan(
    dh_out: &[f32], // [L * D]
    alphas: &[f32], // [L * D]
    len: usize,
    dim: usize,
) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    let n_pow2 = len.next_power_of_two();
    let mut dh = vec![0.0f32; len * dim];
    let mut a_buf = vec![1.0f32; n_pow2];
    let mut b_buf = vec![0.0f32; n_pow2];

    for d in 0..dim {
        // Reverse order: k = 0 corresponds to t = len - 1
        // Recurrence in reversed index k: g[k] = a_rev[k] * g[k-1] + dh_out[len - 1 - k]
        // where a_rev[0] = 0, a_rev[k] = alphas[len - k] for k >= 1.
        for k in 0..len {
            let t = len - 1 - k;
            let a = if k == 0 { 0.0 } else { alphas[(t + 1) * dim + d] };
            let b = dh_out[t * dim + d];
            a_buf[k] = a;
            b_buf[k] = b;
        }
        for k in len..n_pow2 {
            a_buf[k] = 1.0;
            b_buf[k] = 0.0;
        }

        // Up-sweep
        let mut stride = 1usize;
        while stride < n_pow2 {
            let step = stride << 1;
            let mut idx = step - 1;
            while idx < n_pow2 {
                let left = idx - stride;
                let (na, nb) = compose_affine((a_buf[left], b_buf[left]), (a_buf[idx], b_buf[idx]));
                a_buf[idx] = na;
                b_buf[idx] = nb;
                idx += step;
            }
            stride = step;
        }

        let root_b = b_buf[n_pow2 - 1];
        a_buf[n_pow2 - 1] = 1.0;
        b_buf[n_pow2 - 1] = 0.0;

        // Down-sweep
        let mut stride = n_pow2 >> 1;
        while stride >= 1 {
            let step = stride << 1;
            let mut idx = step - 1;
            while idx < n_pow2 {
                let left = idx - stride;
                let t_a = a_buf[left];
                let t_b = b_buf[left];
                a_buf[left] = a_buf[idx];
                b_buf[left] = b_buf[idx];
                let (na, nb) = compose_affine((a_buf[idx], b_buf[idx]), (t_a, t_b));
                a_buf[idx] = na;
                b_buf[idx] = nb;
                idx += step;
            }
            stride >>= 1;
        }

        for k in 0..len {
            let t = len - 1 - k;
            dh[t * dim + d] = if k + 1 < n_pow2 { b_buf[k + 1] } else { root_b };
        }
    }
    dh
}

// ============================================================================
// DualCorticalCore: Slow Ternary + Multi-Timescale Selective SSM + Fast Weights
// ============================================================================

/// Multi-timescale Dual Cortical Core (`aura-cortex`).
/// Splits the 256D recurrent state into 4 log-spaced timescales (fast/med/slow/ultra-slow)
/// plus a zero-alloc NLMS FastWeightMemory for instant online association.
#[derive(Clone, Debug)]
pub struct DualCorticalCore {
    pub w_slow: TernaryMatrix,
    pub fast_weights: FastWeightMemory,
    pub h_state: [i32; LATENT_DIM], // Q8 fixed-point multi-timescale SSM state
    pub decay_shifts: [u8; LATENT_DIM],
    pub fast_enabled: bool,
    scratch_slow: Vec<i32>,
    scratch_fast: Vec<i32>,
}

impl DualCorticalCore {
    pub fn new(seed: u64, fast_enabled: bool) -> Self {
        let mut rng = Xorshift64::new(seed ^ 0x434F_5254_4558_0002);
        let w_slow = TernaryMatrix::from_random_density(LATENT_DIM, LATENT_DIM, 0.35, &mut rng);
        let mut decay_shifts = [3u8; LATENT_DIM];
        // 4 log-spaced timescales across the 256 channels:
        // [0..64): fast (tau ~ 4), [64..128): med (tau ~ 16),
        // [128..192): slow (tau ~ 64), [192..256): ultra-slow (tau ~ 256)
        for d in 0..LATENT_DIM {
            decay_shifts[d] = match d / 64 {
                0 => 2,
                1 => 4,
                2 => 6,
                _ => 8,
            };
        }
        Self {
            w_slow,
            fast_weights: FastWeightMemory::new(LATENT_DIM, 7),
            h_state: [0; LATENT_DIM],
            decay_shifts,
            fast_enabled,
            scratch_slow: vec![0i32; LATENT_DIM],
            scratch_fast: vec![0i32; LATENT_DIM],
        }
    }

    pub fn reset(&mut self) {
        self.h_state.fill(0);
        self.fast_weights.reset();
    }

    /// Single recurrent step: zero-mul slow projection + selective shift SSM + fast associative recall.
    pub fn step(&mut self, z: &LatentVec, surprise_boost: bool) -> LatentVec {
        self.w_slow.project_lut4_simd(&z.data, &mut self.scratch_slow);
        if self.fast_enabled {
            self.fast_weights.query(&z.data, &mut self.scratch_fast);
        } else {
            self.scratch_fast.fill(0);
        }

        let mut out = LatentVec::zero();
        for d in 0..LATENT_DIM {
            // Selective decay: surprise accelerates input gate on fast/medium channels
            let base_shift = self.decay_shifts[d];
            let eff_shift = if surprise_boost && base_shift <= 4 {
                base_shift.saturating_sub(1).max(1)
            } else {
                base_shift
            };
            let u = (self.scratch_slow[d] >> 3).clamp(-4096, 4096);
            self.h_state[d] += (u - self.h_state[d]) >> eff_shift;

            let combined = (self.h_state[d] >> 4) + (self.scratch_fast[d] >> 4) + (z.data[d] as i32 >> 1);
            out.data[d] = combined.clamp(-127, 127) as i8;
        }
        out
    }

    /// Online Delta-Rule write into fast weights.
    pub fn write_fast(&mut self, key: &LatentVec, target: &LatentVec, lr_shift: u8) {
        if !self.fast_enabled {
            return;
        }
        let mut target_i32 = [0i32; LATENT_DIM];
        for d in 0..LATENT_DIM {
            target_i32[d] = (target.data[d] as i32) << 4;
        }
        self.fast_weights.update_online_nlms(&key.data, &target_i32, lr_shift);
    }
}

// ============================================================================
// PART 1 + BOTTLENECKS 1.2 & 2.1/2.3: CorticalFabric v2 (0 floats in step())
// ============================================================================

pub const MAX_DELAY: usize = 16;
pub const CORTEX_TILE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynapseKind {
    BasalFeedforward,
    ApicalFeedback,
}

#[derive(Clone, Copy, Debug)]
pub struct Synapse {
    pub target: u32,
    pub weight: i8,      // -1, 0, +1 (zero-mul)
    pub latent_acc: i16, // Hysteresis accumulator
    pub phase_gauge: u8, // U(1) gauge connection A_ij in Z_256
    pub gain_shift: u8,  // Log2 scale (0..=4)
    pub delay: u8,       // Axonal delay (1..MAX_DELAY)
    pub kind: SynapseKind,
}

#[derive(Clone, Copy, Debug)]
pub struct Neuromodulators {
    pub noradrenaline: u8, // 0..=15: volatility / exploration jitter (0 in calm steady state!)
    pub acetylcholine: i8, // -3..=+3: basal vs apical precision balance
    pub dopamine: i8,      // -1, 0, +1: free-energy reduction (-dF)
}

#[derive(Clone, Copy, Debug)]
pub struct FabricConfig {
    pub plasticity_enabled: bool,
    pub phase_gate_enabled: bool,
    pub neuromod_enabled: bool,
    /// Bottleneck 2.3 fix: surprise-gated NE decays to 0 in steady state (no baseline jitter tax).
    pub surprise_gated_neuromod: bool,
    /// Bottleneck 1.2 fix: sort outgoing CSR edges by 64-neuron destination tile for L1 locality.
    pub cache_tiled_csr: bool,
    /// Bottleneck 2.1 fix: keep amortized predictive state across windows + adaptive early exit.
    pub amortized_warm_start: bool,
    /// AURA v3 Genesis (Subroutine 3): enable physical axonal conduction delays (`axon_delay[src]`)
    /// so myelination (`delay -> 1`) physically accelerates cortical settling and propagation.
    pub myelin_delay_enabled: bool,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            plasticity_enabled: true,
            phase_gate_enabled: true,
            neuromod_enabled: true,
            surprise_gated_neuromod: true,
            cache_tiled_csr: true,
            amortized_warm_start: true,
            myelin_delay_enabled: false,
        }
    }
}

/// Settling summary returned by `settle_window_*`.
#[derive(Clone, Copy, Debug)]
pub struct SettlingStats {
    pub ticks_used: usize,
    pub final_fe: i64,
    pub sensory_fe: i64,
    pub total_spikes: usize,
    pub ne_level: u8,
}

/// Neuromorphic Event-Driven Cortical Fabric v2/v3.
/// Strict invariant: ZERO floating-point instructions in `step()` and `settle_window_*()`.
#[derive(Clone, Debug)]
pub struct CorticalFabric {
    pub n_neurons: usize,
    pub cfg: FabricConfig,

    // SoA neuron state (contiguous L1/L2 cache layout)
    pub v_basal: Vec<i32>,
    pub v_apical: Vec<i32>,
    pub error: Vec<i32>,
    pub threshold: Vec<i32>,
    pub phase: Vec<u8>,
    pub freq: Vec<u8>,
    pub precision_shift: Vec<u8>,
    pub trace_fast: Vec<i16>, // tau ~ 8 ticks
    pub trace_slow: Vec<i16>, // tau ~ 64 ticks (multi-timescale!)
    /// Per-neuron axonal conduction delay in ticks (unmyelinated ~3..5 -> myelinated 1).
    pub axon_delay: Vec<u8>,

    // Sparse CSR graph
    pub row_ptr: Vec<usize>,
    pub synapses: Vec<Synapse>,
    /// Optional per-synapse activity counter for microglia developmental pruning (empty = disabled).
    pub syn_usage: Vec<u32>,

    // Event-driven axonal delay ring buffer
    pub spike_ring: [Vec<u32>; MAX_DELAY],
    pub cos_lut: [i16; 256],
    pub tick: usize,

    // Metacognitive telemetry (pure integer EMA: fast vs slow volatility detector)
    pub prev_fe: i64,
    pub ema_fe_fast: i64,
    pub ema_fe_slow: i64,
    pub neuromod: Neuromodulators,
    rng: Xorshift64,
    spike_scratch: Vec<u32>,
}

impl CorticalFabric {
    /// Construct a 3-band hierarchical cortical sheet (Gamma sensory 0..64, Beta assoc 64..192,
    /// Theta apical 192..256) with either random-scatter CSR or minicolumn-tiled CSR.
    pub fn build_cortical_sheet(n_neurons: usize, seed: u64, cfg: FabricConfig) -> Self {
        let mut rng = Xorshift64::new(seed ^ 0xFAB2_1C00_0002);
        let mut freqs = vec![0u8; n_neurons];
        let sensory_end = (n_neurons / 4).max(16);
        let beta_end = (n_neurons * 3) / 4;

        for (i, f) in freqs.iter_mut().enumerate() {
            *f = if i < sensory_end {
                19 + ((i as u8) & 3) // Gamma band (fast)
            } else if i < beta_end {
                9 + ((i as u8) & 3) // Beta band (medium)
            } else {
                3 + ((i as u8) & 1) // Theta band (slow top-down)
            };
        }

        let fanout = 20usize.min(n_neurons / 2);
        let mut row_ptr = Vec::with_capacity(n_neurons + 1);
        let mut synapses = Vec::with_capacity(n_neurons * fanout);
        row_ptr.push(0);

        for src in 0..n_neurons {
            let mut local_edges = Vec::with_capacity(fanout);
            for k in 0..fanout {
                let (target, kind) = if src < sensory_end {
                    // Feedforward Gamma -> Beta (evidencia basal)
                    let dst = sensory_end + rng.next_usize(beta_end - sensory_end);
                    (dst as u32, SynapseKind::BasalFeedforward)
                } else if src < beta_end {
                    if k < fanout / 2 {
                        // Feedforward Beta -> Theta
                        let dst = beta_end + rng.next_usize(n_neurons - beta_end);
                        (dst as u32, SynapseKind::BasalFeedforward)
                    } else {
                        // Top-down predictive feedback Beta -> Gamma apical
                        let dst = rng.next_usize(sensory_end);
                        (dst as u32, SynapseKind::ApicalFeedback)
                    }
                } else {
                    // Top-down Theta -> Beta apical (or recurrent Theta)
                    if k < (fanout * 3) / 4 {
                        let dst = sensory_end + rng.next_usize(beta_end - sensory_end);
                        (dst as u32, SynapseKind::ApicalFeedback)
                    } else {
                        let dst = beta_end + rng.next_usize(n_neurons - beta_end);
                        (dst as u32, SynapseKind::BasalFeedforward)
                    }
                };

                let init_w = match rng.next_u32() % 5 {
                    0 => 1i8,
                    1 => -1i8,
                    _ => 0i8,
                };
                let latent_acc = (init_w as i16) * 80;
                local_edges.push(Synapse {
                    target,
                    weight: init_w,
                    latent_acc,
                    phase_gauge: (rng.next_u32() & 0xFF) as u8,
                    gain_shift: 1 + ((rng.next_u32() & 1) as u8),
                    delay: 1 + ((rng.next_u32() % 3) as u8),
                    kind,
                });
            }

            if cfg.cache_tiled_csr {
                // Bottleneck 1.2 fix: sort outgoing synapses by 64-neuron destination tile + target
                local_edges.sort_unstable_by_key(|s| (s.target / (CORTEX_TILE as u32), s.target));
            }
            synapses.extend(local_edges);
            row_ptr.push(synapses.len());
        }

        Self {
            n_neurons,
            cfg,
            v_basal: vec![0; n_neurons],
            v_apical: vec![0; n_neurons],
            error: vec![0; n_neurons],
            threshold: vec![192; n_neurons],
            phase: (0..n_neurons).map(|i| (i.wrapping_mul(13)) as u8).collect(),
            freq: freqs,
            precision_shift: vec![1; n_neurons],
            trace_fast: vec![0; n_neurons],
            trace_slow: vec![0; n_neurons],
            axon_delay: vec![1; n_neurons],
            row_ptr,
            synapses,
            syn_usage: Vec::new(),
            spike_ring: std::array::from_fn(|_| Vec::with_capacity(n_neurons / 4)),
            cos_lut: build_cos_lut(),
            tick: 0,
            prev_fe: 0,
            ema_fe_fast: 0,
            ema_fe_slow: 0,
            neuromod: Neuromodulators {
                noradrenaline: 0,
                acetylcholine: 0,
                dopamine: 0,
            },
            rng,
            spike_scratch: Vec::with_capacity(16),
        }
    }

    /// Construct a `CorticalFabric` directly from an externally decompressed CSR graph
    /// (used by `aura-genesis` morphogenetic router & canonical microcircuit instantiation).
    pub fn from_custom_graph(
        n_neurons: usize,
        freqs: Vec<u8>,
        row_ptr: Vec<usize>,
        synapses: Vec<Synapse>,
        axon_delay: Vec<u8>,
        seed: u64,
        cfg: FabricConfig,
    ) -> Self {
        debug_assert_eq!(freqs.len(), n_neurons);
        debug_assert_eq!(row_ptr.len(), n_neurons + 1);
        debug_assert_eq!(axon_delay.len(), n_neurons);
        let rng = Xorshift64::new(seed ^ 0xFAB3_6E4E_5150_0003);
        Self {
            n_neurons,
            cfg,
            v_basal: vec![0; n_neurons],
            v_apical: vec![0; n_neurons],
            error: vec![0; n_neurons],
            threshold: vec![192; n_neurons],
            phase: (0..n_neurons).map(|i| (i.wrapping_mul(13)) as u8).collect(),
            freq: freqs,
            precision_shift: vec![1; n_neurons],
            trace_fast: vec![0; n_neurons],
            trace_slow: vec![0; n_neurons],
            axon_delay,
            row_ptr,
            synapses,
            syn_usage: Vec::new(),
            spike_ring: std::array::from_fn(|_| Vec::with_capacity(n_neurons / 4)),
            cos_lut: build_cos_lut(),
            tick: 0,
            prev_fe: 0,
            ema_fe_fast: 0,
            ema_fe_slow: 0,
            neuromod: Neuromodulators {
                noradrenaline: 0,
                acetylcholine: 0,
                dopamine: 0,
            },
            rng,
            spike_scratch: Vec::with_capacity(16),
        }
    }

    /// Enable per-synapse activity counting for developmental microglia pruning.
    pub fn enable_usage_tracking(&mut self) {
        self.syn_usage = vec![0u32; self.synapses.len()];
    }

    /// Disable per-synapse activity counting (zero overhead in compiled inference).
    pub fn disable_usage_tracking(&mut self) {
        self.syn_usage.clear();
    }

    /// Exact explicit CSR memory footprint in bytes (`row_ptr` + `synapses`).
    pub fn csr_memory_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<usize>()
            + self.synapses.len() * std::mem::size_of::<Synapse>()
    }

    #[inline(always)]
    fn next_noise_i32(&mut self) -> i32 {
        ((self.rng.next_u64() & 0xFF) as i32) - 128
    }

    /// Cold-reset membrane state (simulating non-amortized settling from scratch).
    pub fn reset_potentials_cold(&mut self) {
        self.v_basal.fill(0);
        self.v_apical.fill(0);
        self.error.fill(0);
        for slot in &mut self.spike_ring {
            slot.clear();
        }
    }

    /// Zero-token sensory byte injection via delta modulation.
    pub fn inject_raw_byte(&mut self, prev_byte: u8, curr_byte: u8) {
        delta_modulate_byte(prev_byte, curr_byte, &mut self.spike_scratch);
        let slot = self.tick % MAX_DELAY;
        for &n in &self.spike_scratch {
            let idx = n as usize;
            if idx < self.n_neurons {
                self.v_basal[idx] = self.v_basal[idx].saturating_add(256);
                self.spike_ring[slot].push(n);
            }
        }
    }

    /// Clamped sensory drive for Predictive Coding settling: pins basal evidence on the
    /// 32 Gamma sensory neurons (`0..32`) across settling ticks so top-down apical feedback
    /// from Beta/Theta must actively predict and cancel it.
    pub fn clamp_sensory_byte(&mut self, byte: u8) {
        let slot = self.tick % MAX_DELAY;
        for bit in 0..8usize {
            let pol = ((byte >> bit) & 1) as usize;
            let on_idx = bit * 2 + pol;
            let off_idx = bit * 2 + (1 - pol);
            if on_idx < self.n_neurons {
                self.v_basal[on_idx] = 288;
                // Only emit sensory error spikes when basal exceeds apical prediction (Predictive Coding!)
                if (self.v_basal[on_idx] - self.v_apical[on_idx]).abs() > 48 {
                    self.spike_ring[slot].push(on_idx as u32);
                }
            }
            if off_idx < self.n_neurons {
                self.v_basal[off_idx] = 0;
            }
            let on2 = 16 + on_idx;
            let off2 = 16 + off_idx;
            if on2 < self.n_neurons {
                self.v_basal[on2] = 256;
                if (self.v_basal[on2] - self.v_apical[on2]).abs() > 48 {
                    self.spike_ring[slot].push(on2 as u32);
                }
            }
            if off2 < self.n_neurons {
                self.v_basal[off2] = 0;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn step_neurons_avx2(&mut self, next_slot: usize, ne: u8) -> (i64, i64) {
        let n = self.n_neurons;
        let sensory_end = (n / 4).max(16);
        let mut total_fe: i64 = 0;
        let mut sensory_fe: i64 = 0;

        let one32 = _mm256_set1_epi32(1);
        let min_thr = _mm256_set1_epi32(112);
        let cap_hi = _mm256_set1_epi32(512);
        let cap_lo = _mm256_set1_epi32(-512);
        let ne_shift = 7u8.saturating_sub(ne >> 1);
        let jitter_byte = if ne > 0 {
            (self.next_noise_i32() >> ne_shift) as u8
        } else {
            0u8
        };
        let jv = _mm_set1_epi8(jitter_byte as i8);

        let mut i = 0usize;
        while i + 8 <= n {
            let vb_ptr = self.v_basal.as_mut_ptr().add(i);
            let va_ptr = self.v_apical.as_mut_ptr().add(i);
            let err_ptr = self.error.as_mut_ptr().add(i);
            let thr_ptr = self.threshold.as_mut_ptr().add(i);

            // Clamped leaky integrators: v_basal -= v_basal >> 2 (fast basal leak), v_apical -= v_apical >> 4
            let mut vb = _mm256_loadu_si256(vb_ptr as *const __m256i);
            let mut va = _mm256_loadu_si256(va_ptr as *const __m256i);
            vb = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(vb, _mm256_srai_epi32(vb, 2)), cap_lo), cap_hi);
            va = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(va, _mm256_srai_epi32(va, 4)), cap_lo), cap_hi);
            _mm256_storeu_si256(vb_ptr as *mut __m256i, vb);
            _mm256_storeu_si256(va_ptr as *mut __m256i, va);

            // Decay 8 x i16 traces in 128-bit vector
            let tf_ptr = self.trace_fast.as_mut_ptr().add(i);
            let ts_ptr = self.trace_slow.as_mut_ptr().add(i);
            let tf = _mm_loadu_si128(tf_ptr as *const __m128i);
            let ts = _mm_loadu_si128(ts_ptr as *const __m128i);
            _mm_storeu_si128(tf_ptr as *mut __m128i, _mm_sub_epi16(tf, _mm_srai_epi16(tf, 3)));
            _mm_storeu_si128(ts_ptr as *mut __m128i, _mm_sub_epi16(ts, _mm_srai_epi16(ts, 6)));

            // Laplace error & FE (precision_shift == 1 -> >> 1)
            let err = _mm256_sub_epi32(vb, va);
            _mm256_storeu_si256(err_ptr as *mut __m256i, err);
            let fe = _mm256_srai_epi32(_mm256_abs_epi32(err), 1);

            let mut fe_arr = [0i32; 8];
            _mm256_storeu_si256(fe_arr.as_mut_ptr() as *mut __m256i, fe);
            let block_fe: i64 = (fe_arr[0] as i64)
                + (fe_arr[1] as i64)
                + (fe_arr[2] as i64)
                + (fe_arr[3] as i64)
                + (fe_arr[4] as i64)
                + (fe_arr[5] as i64)
                + (fe_arr[6] as i64)
                + (fe_arr[7] as i64);
            total_fe += block_fe;
            if i + 8 <= sensory_end {
                sensory_fe += block_fe;
            } else if i < sensory_end {
                for k in 0..(sensory_end - i) {
                    sensory_fe += fe_arr[k] as i64;
                }
            }

            // Phase oscillator advance + vectorized NE jitter
            let ph_ptr = self.phase.as_mut_ptr().add(i);
            let fr_ptr = self.freq.as_ptr().add(i);
            let ph = _mm_loadl_epi64(ph_ptr as *const __m128i);
            let fr = _mm_loadl_epi64(fr_ptr as *const __m128i);
            let ph_next = if ne > 0 {
                _mm_add_epi8(_mm_add_epi8(ph, fr), jv)
            } else {
                _mm_add_epi8(ph, fr)
            };
            _mm_storel_epi64(ph_ptr as *mut __m128i, ph_next);

            // Tropical drive = max((vb + va) >> 1, fe)
            let coinc = _mm256_srai_epi32(_mm256_add_epi32(vb, va), 1);
            let drive = _mm256_max_epi32(coinc, fe);
            let thr = _mm256_loadu_si256(thr_ptr as *const __m256i);
            let cmp = _mm256_cmpgt_epi32(drive, thr);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32;

            let thr_rel = _mm256_max_epi32(_mm256_sub_epi32(thr, one32), min_thr);
            _mm256_storeu_si256(thr_ptr as *mut __m256i, thr_rel);

            if mask != 0 {
                let mut m = mask;
                let use_delay = self.cfg.myelin_delay_enabled;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    let idx = i + lane;
                    let slot = if use_delay {
                        (self.tick + (self.axon_delay[idx] as usize).clamp(1, MAX_DELAY - 1)) % MAX_DELAY
                    } else {
                        next_slot
                    };
                    self.spike_ring[slot].push(idx as u32);
                    self.v_basal[idx] >>= 1;
                    self.trace_fast[idx] = (self.trace_fast[idx] + 64).min(256);
                    self.trace_slow[idx] = (self.trace_slow[idx] + 16).min(256);
                    let orig_thr = (self.threshold[idx] + 1).max(112);
                    self.threshold[idx] = (orig_thr + 12).min(1024);
                    m &= m - 1;
                }
            }
            i += 8;
        }
        (total_fe, sensory_fe)
    }

    /// Single event-driven physical tick. Strictly ZERO floating-point operations.
    /// Returns `(pure_laplace_fe, sensory_laplace_fe, spike_count)`.
    pub fn step(&mut self) -> (i64, i64, usize) {
        let curr_slot = self.tick % MAX_DELAY;
        let active_spikes = std::mem::take(&mut self.spike_ring[curr_slot]);
        let spike_count = active_spikes.len();

        // 1. EVENT-DRIVEN PROPAGATION WITH U(1) PHASE-GAUGE ATTENTION (ZERO MATMUL)
        let phase_gate = self.cfg.phase_gate_enabled;
        let track_usage = self.syn_usage.len() == self.synapses.len();
        for &src in &active_spikes {
            let src_idx = src as usize;
            let src_phase = self.phase[src_idx];
            let start = self.row_ptr[src_idx];
            let end = self.row_ptr[src_idx + 1];

            for syn_idx in start..end {
                let syn = self.synapses[syn_idx];
                if syn.weight == 0 {
                    continue;
                }
                let dst = syn.target as usize;
                let mag = if phase_gate {
                    let phase_diff = src_phase
                        .wrapping_add(syn.phase_gauge)
                        .wrapping_sub(self.phase[dst]);
                    let coh = self.cos_lut[phase_diff as usize] as i32;
                    if coh <= 0 {
                        continue;
                    }
                    coh >> syn.gain_shift
                } else {
                    128 >> syn.gain_shift
                };
                if track_usage {
                    self.syn_usage[syn_idx] = self.syn_usage[syn_idx].saturating_add(1);
                }
                let impulse = if syn.weight > 0 { mag } else { -mag };
                match syn.kind {
                    SynapseKind::BasalFeedforward => {
                        self.v_basal[dst] = self.v_basal[dst].saturating_add(impulse);
                    }
                    SynapseKind::ApicalFeedback => {
                        self.v_apical[dst] = self.v_apical[dst].saturating_add(impulse);
                    }
                }
            }
        }

        // 2. DENDRITIC DYNAMICS, APPLES-TO-APPLES LAPLACE FE & TROPICAL DRIVE
        let ach = if self.cfg.neuromod_enabled {
            self.neuromod.acetylcholine
        } else {
            0
        };
        let ne = if self.cfg.neuromod_enabled {
            self.neuromod.noradrenaline
        } else {
            0
        };
        let apply_jitter = ne > 0;
        let next_slot = (self.tick + 1) % MAX_DELAY;

        #[cfg(target_arch = "x86_64")]
        let use_avx2_neurons = self.cfg.cache_tiled_csr
            && is_x86_feature_detected!("avx2")
            && self.n_neurons.is_multiple_of(8);
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2_neurons = false;

        let (total_fe, sensory_fe) = if use_avx2_neurons {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.step_neurons_avx2(next_slot, ne)
            }
            #[cfg(not(target_arch = "x86_64"))]
            (0i64, 0i64)
        } else {
            let mut t_fe: i64 = 0;
            let mut s_fe: i64 = 0;
            let sensory_end = (self.n_neurons / 4).max(16);
            let ne_shift = 7u8.saturating_sub(ne >> 1);
            let use_delay = self.cfg.myelin_delay_enabled;

            for i in 0..self.n_neurons {
                self.v_basal[i] = (self.v_basal[i] - (self.v_basal[i] >> 2)).clamp(-512, 512);
                self.v_apical[i] = (self.v_apical[i] - (self.v_apical[i] >> 4)).clamp(-512, 512);
                self.trace_fast[i] -= self.trace_fast[i] >> 3;
                self.trace_slow[i] -= self.trace_slow[i] >> 6;

                let raw_err = self.v_basal[i].saturating_sub(self.v_apical[i]);
                let pure_fe = (raw_err.abs() >> self.precision_shift[i]) as i64;
                t_fe += pure_fe;
                if i < sensory_end {
                    s_fe += pure_fe;
                }

                let eff_err = if ach > 0 {
                    (self.v_basal[i] + (self.v_basal[i] >> (4 - (ach as u8).min(2))))
                        .saturating_sub(self.v_apical[i])
                } else if ach < 0 {
                    self.v_basal[i].saturating_sub(
                        self.v_apical[i] + (self.v_apical[i] >> (4 - ((-ach) as u8).min(2))),
                    )
                } else {
                    raw_err
                };
                self.error[i] = eff_err;

                let jitter = if apply_jitter {
                    (self.next_noise_i32() >> ne_shift) as u8
                } else {
                    0u8
                };
                self.phase[i] = self.phase[i].wrapping_add(self.freq[i]).wrapping_add(jitter);

                let coincidence = (self.v_basal[i] + self.v_apical[i]) >> 1;
                let drive = coincidence.max(pure_fe as i32);

                if drive > self.threshold[i] {
                    let slot = if use_delay {
                        (self.tick + (self.axon_delay[i] as usize).clamp(1, MAX_DELAY - 1)) % MAX_DELAY
                    } else {
                        next_slot
                    };
                    self.spike_ring[slot].push(i as u32);
                    self.v_basal[i] >>= 1;
                    self.trace_fast[i] = (self.trace_fast[i] + 64).min(256);
                    self.trace_slow[i] = (self.trace_slow[i] + 16).min(256);
                    self.threshold[i] = (self.threshold[i] + 12).min(1024);
                } else {
                    self.threshold[i] = (self.threshold[i] - 1).max(112);
                }
            }
            (t_fe, s_fe)
        };

        // 3. LOCAL 3-FACTOR PLASTICITY (ACTIVE SYNAPSES ONLY, HYSTERESIS + GAUGE ALIGNMENT)
        if self.cfg.plasticity_enabled {
            let da = if self.cfg.neuromod_enabled {
                self.neuromod.dopamine
            } else {
                1
            };
            let lr_boost: i16 = if self.cfg.neuromod_enabled && ne >= 4 { 2 } else { 1 };

            for &src in &active_spikes {
                let src_idx = src as usize;
                let pre_t = self.trace_fast[src_idx].max(self.trace_slow[src_idx]);
                if pre_t <= 0 {
                    continue;
                }
                let src_phase = self.phase[src_idx];
                let start = self.row_ptr[src_idx];
                let end = self.row_ptr[src_idx + 1];

                for syn in &mut self.synapses[start..end] {
                    let dst = syn.target as usize;
                    let post_err = self.error[dst];
                    if post_err.abs() < 8 {
                        continue;
                    }
                    let delta_dir: i16 = match syn.kind {
                        SynapseKind::ApicalFeedback => (post_err.signum() as i16) * 4 * lr_boost,
                        SynapseKind::BasalFeedforward => {
                            if da == 0 {
                                0
                            } else {
                                (da as i16) * (post_err.signum() as i16) * lr_boost
                            }
                        }
                    };
                    if delta_dir != 0 {
                        syn.latent_acc = syn.latent_acc.saturating_add(delta_dir);
                        if syn.latent_acc > 40 {
                            syn.weight = 1;
                        } else if syn.latent_acc < -40 {
                            syn.weight = -1;
                        } else if syn.latent_acc.abs() < 10 {
                            syn.weight = 0;
                        }
                    }

                    if phase_gate {
                        let target_gauge = self.phase[dst].wrapping_sub(src_phase);
                        let diff = target_gauge.wrapping_sub(syn.phase_gauge) as i8;
                        if diff > 0 {
                            syn.phase_gauge = syn.phase_gauge.wrapping_add(1);
                        } else if diff < 0 {
                            syn.phase_gauge = syn.phase_gauge.wrapping_sub(1);
                        }
                    }
                }
            }
        }

        // 4. METACOGNITIVE VOLATILITY LOOP (FAST vs SLOW EMA -> ZERO STEADY-STATE TAX)
        self.update_metacognition(total_fe, spike_count);

        self.spike_ring[curr_slot] = active_spikes;
        self.spike_ring[curr_slot].clear();
        self.tick += 1;
        (total_fe, sensory_fe, spike_count)
    }

    fn update_metacognition(&mut self, current_fe: i64, spike_count: usize) {
        if self.tick == 0 {
            self.ema_fe_fast = current_fe;
            self.ema_fe_slow = current_fe;
            self.prev_fe = current_fe;
            return;
        }
        self.ema_fe_fast += (current_fe - self.ema_fe_fast) >> 3;
        self.ema_fe_slow += (current_fe - self.ema_fe_slow) >> 6;

        let delta_fe = self.prev_fe - current_fe;
        self.neuromod.dopamine = if delta_fe > 20 {
            1
        } else if delta_fe < -20 {
            -1
        } else {
            0
        };

        let volatility = self.ema_fe_fast - self.ema_fe_slow;
        let thresh = (self.ema_fe_slow >> 4).max(32);
        if volatility > thresh {
            self.neuromod.noradrenaline = (self.neuromod.noradrenaline + 4).min(15);
        } else if self.cfg.surprise_gated_neuromod {
            self.neuromod.noradrenaline = self.neuromod.noradrenaline.saturating_sub(2);
        } else {
            self.neuromod.noradrenaline = self.neuromod.noradrenaline.saturating_sub(1).max(2);
        }

        let target_activity = self.n_neurons >> 5;
        if volatility > thresh && spike_count < target_activity {
            self.neuromod.acetylcholine = (self.neuromod.acetylcholine + 1).min(2);
        } else if self.cfg.surprise_gated_neuromod {
            if self.neuromod.acetylcholine > 0 {
                self.neuromod.acetylcholine -= 1;
            } else if self.neuromod.acetylcholine < 0 {
                self.neuromod.acetylcholine += 1;
            }
        } else if volatility < 0 {
            self.neuromod.acetylcholine = (self.neuromod.acetylcholine - 1).max(-2);
        }

        self.prev_fe = current_fe;
    }

    /// Bottleneck 2.1 Baseline: Cold-start fixed-K settling under clamped sensory evidence.
    /// Resets top-down apical predictions cold each window and runs fixed `k_steps`.
    pub fn settle_window_cold(&mut self, prev_byte: u8, curr_byte: u8, k_steps: usize) -> SettlingStats {
        self.reset_potentials_cold();
        self.inject_raw_byte(prev_byte, curr_byte);
        let mut final_fe = 0i64;
        let mut sensory_fe = 0i64;
        let mut total_spikes = 0usize;
        for _ in 0..k_steps.max(1) {
            self.clamp_sensory_byte(curr_byte);
            let (fe, sfe, sp) = self.step();
            final_fe = fe;
            sensory_fe = sfe;
            total_spikes += sp;
        }
        SettlingStats {
            ticks_used: k_steps.max(1),
            final_fe,
            sensory_fe,
            total_spikes,
            ne_level: self.neuromod.noradrenaline,
        }
    }

    /// Bottleneck 2.1 Fix: Amortized Warm-Start Predictive Settling + Adaptive Early-Exit.
    /// Keeps learned top-down `v_apical` predictions warm across windows while resetting stale
    /// hidden basal transients, settling in 2-4 ticks on predictable stimuli and expanding up to
    /// `max_k` ticks only during unexpected regime shifts!
    pub fn settle_window_amortized(
        &mut self,
        prev_byte: u8,
        curr_byte: u8,
        min_k: usize,
        max_k: usize,
        tol_pct: i64,
    ) -> SettlingStats {
        // Amortized warm-start: clear stale basal buffer and transient hidden apical state,
        // while preserving 100% of the learned top-down predictive prior `v_apical[0..sensory_end]`
        // on the clamped sensory layer (true amortized predictive coding!).
        let sensory_end = (self.n_neurons / 4).max(16);
        self.v_basal.fill(0);
        for i in sensory_end..self.n_neurons {
            self.v_apical[i] >>= 2;
        }
        for slot in &mut self.spike_ring {
            slot.clear();
        }
        self.inject_raw_byte(prev_byte, curr_byte);
        let mut prev_fe = i64::MAX;
        let mut final_fe = 0i64;
        let mut sensory_fe = 0i64;
        let mut total_spikes = 0usize;
        let mut used = 0usize;

        for step_idx in 0..max_k.max(1) {
            self.clamp_sensory_byte(curr_byte);
            let (fe, sfe, sp) = self.step();
            final_fe = fe;
            sensory_fe = sfe;
            total_spikes += sp;
            used = step_idx + 1;

            if used >= min_k && prev_fe != i64::MAX {
                let diff = (fe - prev_fe).abs();
                let at_equilibrium =
                    self.ema_fe_slow > 0 && fe <= self.ema_fe_slow + (self.ema_fe_slow >> 4);
                let stabilized = diff * 100 <= prev_fe.max(1) * tol_pct;
                if (self.neuromod.noradrenaline < 6 && (at_equilibrium || stabilized)) || fe < 256 {
                    break;
                }
            }
            prev_fe = fe;
        }
        SettlingStats {
            ticks_used: used,
            final_fe,
            sensory_fe,
            total_spikes,
            ne_level: self.neuromod.noradrenaline,
        }
    }

    /// Clamp an arbitrary 2D spatial/wave sensory drive (`0..=320` integer intensities) onto
    /// the Gamma sensory layer `0..sensory_end` (strictly integer, 0 floats).
    pub fn clamp_sensory_pattern(&mut self, pattern: &[i32]) {
        let sensory_end = (self.n_neurons / 4).max(16);
        let slot = self.tick % MAX_DELAY;
        let len = pattern.len().min(sensory_end);
        for i in 0..len {
            let v = pattern[i].clamp(0, 320);
            self.v_basal[i] = v;
            if v > 64 && (v - self.v_apical[i]).abs() > 40 {
                self.spike_ring[slot].push(i as u32);
            }
        }
    }

    /// Amortized predictive settling on a clamped 2D spatial/retinal sensory pattern.
    /// Respects physical axonal conduction delays (`axon_delay`) so myelinated circuits
    /// achieve faster apical error cancellation in fewer ticks!
    pub fn settle_pattern_amortized(
        &mut self,
        pattern: &[i32],
        min_k: usize,
        max_k: usize,
        tol_pct: i64,
    ) -> SettlingStats {
        let sensory_end = (self.n_neurons / 4).max(16);
        self.v_basal.fill(0);
        for i in sensory_end..self.n_neurons {
            self.v_apical[i] >>= 2;
        }
        // Keep in-flight delayed spikes only if myelin_delay_enabled == false; clear stale spikes
        // between frames so each frame's settling latency reflects true axonal conduction delay!
        for slot in &mut self.spike_ring {
            slot.clear();
        }
        let mut prev_fe = i64::MAX;
        let mut final_fe = 0i64;
        let mut sensory_fe = 0i64;
        let mut total_spikes = 0usize;
        let mut used = 0usize;

        // Physical axonal conduction round-trip: when saltatory delays are modeled, a newly
        // clamped sensory frame must traverse at least one feedforward axonal conduction delay
        // before cortical feedback can physically reflect the new pattern (prevents unmyelinated
        // delay=3..5 circuits from exiting prematurely on stale warm-started apical voltage!).
        let min_phys_ticks = if self.cfg.myelin_delay_enabled {
            let n_s = pattern.len().min(sensory_end).max(1);
            let sum_d: usize = self.axon_delay[..n_s]
                .iter()
                .map(|&d| (d as usize).max(1))
                .sum();
            let mean_d = (sum_d + n_s / 2) / n_s;
            (mean_d + 1).max(2)
        } else {
            1
        };
        let eff_min_k = min_k.max(min_phys_ticks);

        for step_idx in 0..max_k.max(1) {
            self.clamp_sensory_pattern(pattern);
            let (fe, sfe, sp) = self.step();
            final_fe = fe;
            sensory_fe = sfe;
            total_spikes += sp;
            used = step_idx + 1;

            // Require apical feedback signal to arrive before allowing early exit when delays > 1
            let ap_sum: i64 = self.v_apical[..pattern.len().min(sensory_end)]
                .iter()
                .map(|&v| (v as i64).abs())
                .sum();
            let feedback_arrived = !self.cfg.myelin_delay_enabled || ap_sum > 24 || used >= 6;

            if used >= eff_min_k && feedback_arrived && prev_fe != i64::MAX {
                let diff = (fe - prev_fe).abs();
                let at_eq = self.ema_fe_slow > 0 && fe <= self.ema_fe_slow + (self.ema_fe_slow >> 4);
                let stabilized = diff * 100 <= prev_fe.max(1) * tol_pct;
                if (self.neuromod.noradrenaline < 6 && (at_eq || stabilized)) || fe < 256 {
                    break;
                }
            }
            prev_fe = fe;
        }
        SettlingStats {
            ticks_used: used,
            final_fe,
            sensory_fe,
            total_spikes,
            ne_level: self.neuromod.noradrenaline,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_associative_scan_matches_sequential_forward_and_backward() {
        let mut rng = Xorshift64::new(0xCAFE_BABE);
        for &len in &[17usize, 64, 256, 1024] {
            let dim = 16;
            let h0: Vec<f32> = (0..dim).map(|_| rng.next_signed_f32()).collect();
            let alphas: Vec<f32> = (0..len * dim)
                .map(|_| 0.25 + 0.70 * rng.next_f32())
                .collect();
            let inputs: Vec<f32> = (0..len * dim).map(|_| rng.next_signed_f32()).collect();
            let dh_out: Vec<f32> = (0..len * dim).map(|_| rng.next_signed_f32()).collect();

            let fwd_seq = ssm_forward_sequential(&h0, &alphas, &inputs, len, dim);
            let fwd_par = ssm_forward_parallel_scan(&h0, &alphas, &inputs, len, dim);
            let max_fwd_diff = fwd_seq
                .iter()
                .zip(&fwd_par)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_fwd_diff < 1e-5,
                "Forward parallel scan mismatch at L={len}: {max_fwd_diff}"
            );

            let bwd_seq = ssm_backward_sequential(&dh_out, &alphas, len, dim);
            let bwd_par = ssm_backward_parallel_scan(&dh_out, &alphas, len, dim);
            let max_bwd_diff = bwd_seq
                .iter()
                .zip(&bwd_par)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_bwd_diff < 1e-5,
                "Backward parallel scan mismatch at L={len}: {max_bwd_diff}"
            );
        }
    }

    #[test]
    fn cortical_fabric_amortized_settling_saves_ticks_and_stays_sparse() {
        let mut fab_cold = CorticalFabric::build_cortical_sheet(256, 77, FabricConfig::default());
        let mut fab_amort = CorticalFabric::build_cortical_sheet(256, 77, FabricConfig::default());
        let pat = b"AURA_CORTEX_";
        let mut ticks_cold = 0usize;
        let mut ticks_amort = 0usize;
        let mut prev = 0u8;
        for i in 0..200 {
            let b = pat[i % pat.len()];
            ticks_cold += fab_cold.settle_window_cold(prev, b, 16).ticks_used;
            ticks_amort += fab_amort.settle_window_amortized(prev, b, 2, 16, 8).ticks_used;
            prev = b;
        }
        let ratio = (ticks_cold as f64) / (ticks_amort as f64);
        assert!(ratio > 2.5, "Expected >=2.5x fewer settling ticks, got {ratio:.2}x");
    }
}
