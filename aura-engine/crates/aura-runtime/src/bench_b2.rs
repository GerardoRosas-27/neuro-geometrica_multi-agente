//! B2 — Bottleneck 2: Settling Time (K = 5..30 micro-steps) & Sequential SSM Scan Benchmark.
//! Compares:
//! - `naive_cold_k16`: Proposal's cold-start fixed K=16 settling + legacy baseline NE floor
//! - `cold_k4_truncated`: Naive truncation to K=4 without amortized warm-start
//! - `amortized_early_exit`: Bottleneck 2 Fix (amortized top-down warm-start + early-exit + surprise-gated NE)
//! - `frozen_control`: Amortized settling without 3-factor plasticity
//! - `no_neuromod_control`: Amortized settling with neuromodulation disabled
//! Plus exact Blelloch O(log L) Parallel Associative Prefix Scan parity & timing (L = 64, 256, 1024).

use aura_cortex::{
    ssm_backward_parallel_scan, ssm_backward_sequential, ssm_forward_parallel_scan,
    ssm_forward_sequential, CorticalFabric, FabricConfig,
};
use aura_math::Xorshift64;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct B2ArmRow {
    pub seed: u64,
    pub arm: &'static str,
    pub avg_ticks_per_window: f64,
    pub total_ticks: usize,
    pub wall_ms: f64,
    pub late_fe: f64,
    pub late_sensory_fe: f64,
    pub firing_rate_pct: f64,
    pub ne_pre_shift: f64,
    pub ne_post_shift: f64,
    pub ne_shift_delta: f64,
    pub scan_fwd_max_diff: f64,
    pub scan_bwd_max_diff: f64,
}

fn generate_stream(seed: u64) -> Vec<u8> {
    let mut rng = Xorshift64::new(seed ^ 0xB200_5712);
    // Regime A: stable low-bit structured pattern; Regime B: abrupt bit-polarity inversion at event 100
    let base_a = (0x15u8 ^ ((rng.next_u32() & 0x03) as u8)) & 0x3F;
    let base_b = !base_a; // Orthogonal/inverted bit polarities -> genuine distribution shift!
    let pat_a = [base_a, base_a ^ 0x01, base_a, base_a ^ 0x02];
    let pat_b = [base_b, base_b ^ 0x04, base_b ^ 0x08, base_b];

    let mut stream = Vec::with_capacity(240);
    for i in 0..100 {
        stream.push(pat_a[i % 4]);
    }
    for i in 0..100 {
        stream.push(pat_b[i % 4]);
    }
    for i in 0..40 {
        stream.push(pat_a[i % 4]);
    }
    stream
}

fn verify_parallel_scan(seed: u64) -> (f64, f64) {
    let mut rng = Xorshift64::new(seed ^ 0x5CA1_0002);
    let mut max_fwd = 0.0f64;
    let mut max_bwd = 0.0f64;
    let dim = 32usize;
    for &len in &[64usize, 256, 1024] {
        let h0: Vec<f32> = (0..dim).map(|_| rng.next_signed_f32()).collect();
        let alphas: Vec<f32> = (0..len * dim)
            .map(|_| 0.25 + 0.70 * rng.next_f32())
            .collect();
        let inputs: Vec<f32> = (0..len * dim).map(|_| rng.next_signed_f32()).collect();
        let dh_out: Vec<f32> = (0..len * dim).map(|_| rng.next_signed_f32()).collect();

        let fs = ssm_forward_sequential(&h0, &alphas, &inputs, len, dim);
        let fp = ssm_forward_parallel_scan(&h0, &alphas, &inputs, len, dim);
        for (a, b) in fs.iter().zip(&fp) {
            max_fwd = max_fwd.max((*a - *b).abs() as f64);
        }

        let bs = ssm_backward_sequential(&dh_out, &alphas, len, dim);
        let bp = ssm_backward_parallel_scan(&dh_out, &alphas, len, dim);
        for (a, b) in bs.iter().zip(&bp) {
            max_bwd = max_bwd.max((*a - *b).abs() as f64);
        }
    }
    (max_fwd, max_bwd)
}

pub fn run_b2(seed: u64) -> Vec<B2ArmRow> {
    let stream = generate_stream(seed);
    let (scan_fwd, scan_bwd) = verify_parallel_scan(seed);
    let n_neurons = 256usize;

    let arms: [(&'static str, FabricConfig, u8); 5] = [
        (
            "naive_cold_k16",
            FabricConfig {
                plasticity_enabled: true,
                phase_gate_enabled: true,
                neuromod_enabled: true,
                surprise_gated_neuromod: false,
                cache_tiled_csr: false,
                amortized_warm_start: false,
                myelin_delay_enabled: false,
            },
            0, // cold k=16
        ),
        (
            "cold_k4_truncated",
            FabricConfig {
                plasticity_enabled: true,
                phase_gate_enabled: true,
                neuromod_enabled: true,
                surprise_gated_neuromod: false,
                cache_tiled_csr: false,
                amortized_warm_start: false,
                myelin_delay_enabled: false,
            },
            1, // cold k=4
        ),
        (
            "amortized_early_exit",
            FabricConfig {
                plasticity_enabled: true,
                phase_gate_enabled: true,
                neuromod_enabled: true,
                surprise_gated_neuromod: true,
                cache_tiled_csr: true,
                amortized_warm_start: true,
                myelin_delay_enabled: false,
            },
            2, // amortized early-exit
        ),
        (
            "frozen_control",
            FabricConfig {
                plasticity_enabled: false,
                phase_gate_enabled: true,
                neuromod_enabled: true,
                surprise_gated_neuromod: true,
                cache_tiled_csr: true,
                amortized_warm_start: true,
                myelin_delay_enabled: false,
            },
            2,
        ),
        (
            "no_neuromod_control",
            FabricConfig {
                plasticity_enabled: true,
                phase_gate_enabled: true,
                neuromod_enabled: false,
                surprise_gated_neuromod: true,
                cache_tiled_csr: true,
                amortized_warm_start: true,
                myelin_delay_enabled: false,
            },
            2,
        ),
    ];

    let mut rows = Vec::with_capacity(arms.len());
    for (arm_name, cfg, mode) in arms {
        let mut fab = CorticalFabric::build_cortical_sheet(n_neurons, seed, cfg);
        let t0 = Instant::now();

        let mut prev = 0u8;
        let mut total_ticks = 0usize;
        let mut total_spikes = 0usize;
        let mut late_fe_sum = 0i64;
        let mut late_sfe_sum = 0i64;
        let mut late_cnt = 0usize;
        let mut ne_pre_sum = 0u64;
        let mut ne_pre_cnt = 0usize;
        let mut ne_post_sum = 0u64;
        let mut ne_post_cnt = 0usize;

        for (idx, &b) in stream.iter().enumerate() {
            let st = match mode {
                0 => fab.settle_window_cold(prev, b, 16),
                1 => fab.settle_window_cold(prev, b, 4),
                _ => fab.settle_window_amortized(prev, b, 3, 16, 8),
            };
            total_ticks += st.ticks_used;
            total_spikes += st.total_spikes;

            // Pre-shift calm window: events 60..100
            if (60..100).contains(&idx) {
                ne_pre_sum += st.ne_level as u64;
                ne_pre_cnt += 1;
            }
            // Immediate post-shift burst window: events 100..130
            if (100..130).contains(&idx) {
                ne_post_sum += st.ne_level as u64;
                ne_post_cnt += 1;
            }
            // Late settled window in Regime B: events 140..200
            if (140..200).contains(&idx) {
                late_fe_sum += st.final_fe;
                late_sfe_sum += st.sensory_fe;
                late_cnt += 1;
            }
            prev = b;
        }

        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let ne_pre = (ne_pre_sum as f64) / (ne_pre_cnt.max(1) as f64);
        let ne_post = (ne_post_sum as f64) / (ne_post_cnt.max(1) as f64);

        rows.push(B2ArmRow {
            seed,
            arm: arm_name,
            avg_ticks_per_window: (total_ticks as f64) / (stream.len() as f64),
            total_ticks,
            wall_ms,
            late_fe: (late_fe_sum as f64) / (late_cnt.max(1) as f64),
            late_sensory_fe: (late_sfe_sum as f64) / (late_cnt.max(1) as f64),
            firing_rate_pct: 100.0 * (total_spikes as f64) / ((total_ticks * n_neurons).max(1) as f64),
            ne_pre_shift: ne_pre,
            ne_post_shift: ne_post,
            ne_shift_delta: ne_post - ne_pre,
            scan_fwd_max_diff: scan_fwd,
            scan_bwd_max_diff: scan_bwd,
        });
    }

    rows
}
