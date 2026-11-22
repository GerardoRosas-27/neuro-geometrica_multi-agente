//! G3 — Subroutine 3: Synaptic Overproduction + Physical Microglia Pruning +
//! Saltatory Myelination (Jesse Conel Developmental Curve & Biological Lottery Ticket).
//!
//! Compares:
//! 1. `peak_unpruned` (Childhood Peak ~3.8x newborn synapses, unmyelinated delays 3..5)
//! 2. `sparse_from_scratch` (Lottery Ticket Control: EXACT same final synapse count as
//!    pruned, initialized directly sparse without overproduction)
//! 3. `compiled_myelinated` (Overproduce -> Experience -> Microglia Prune ~68% + Myelin delay->1)

use aura_cortex::{CorticalFabric, FabricConfig};
use aura_genesis::{DevelopmentalPruner, MorphogenGenome, MorphogenRouter, PrenatalCurriculum};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct G3ArmRow {
    pub seed: u64,
    pub arm: &'static str,
    pub synapses_newborn: usize,
    pub synapses_peak: usize,
    pub synapses_final: usize,
    pub csr_bytes_final: usize,
    pub prune_frac: f64,
    pub mean_axon_delay: f64,
    pub eval_fe: f64,
    pub eval_sensory_fe: f64,
    pub avg_ticks: f64,
    pub throughput_tps_2048: f64,
}

fn eval_compiled_fabric(fabric: &mut CorticalFabric, test_frames: &[[f32; 64]]) -> (f64, f64, f64) {
    let mut fe_sum = 0.0f64;
    let mut sfe_sum = 0.0f64;
    let mut ticks_sum = 0usize;
    let mut pat = [0i32; 64];

    for f in test_frames {
        for i in 0..64 {
            pat[i] = (f[i] * 300.0).round() as i32;
        }
        let st = fabric.settle_pattern_amortized(&pat, 2, 14, 10);
        fe_sum += st.final_fe as f64;
        sfe_sum += st.sensory_fe as f64;
        ticks_sum += st.ticks_used;
    }
    let n = test_frames.len().max(1) as f64;
    (fe_sum / n, sfe_sum / n, (ticks_sum as f64) / n)
}

fn bench_tps_2048(fabric: &mut CorticalFabric, seed: u64) -> f64 {
    let mut prev = 0x55u8;
    // Warmup
    for i in 0..40usize {
        let b = (seed as u8).wrapping_add((i * 13) as u8);
        fabric.inject_raw_byte(prev, b);
        let _ = fabric.step();
        prev = b;
    }
    let ticks = 350usize;
    let t0 = Instant::now();
    for i in 0..ticks {
        let b = (seed as u8).wrapping_add((i * 17) as u8);
        fabric.inject_raw_byte(prev, b);
        let _ = fabric.step();
        prev = b;
    }
    let dt = t0.elapsed().as_secs_f64().max(1e-9);
    (ticks as f64) / dt
}

pub fn run_g3(seed: u64) -> Vec<G3ArmRow> {
    // Immature developmental arborization has exuberant exploratory branching (`branch_jitter = 0.30`):
    // Overproduction explores 38 candidate branches per neuron so experience + Microglia can keep
    // the exact ~12 winning predictive routes, whereas SparseFromScratch only gets 12 blind draws!
    let mut genome = MorphogenGenome::default();
    genome.branch_jitter = 0.30;
    genome.capture_radius = 0.28;
    let cfg = FabricConfig {
        myelin_delay_enabled: true,
        ..Default::default()
    };

    let n_neurons = 256usize;
    let newborn_fanout = 10usize;
    let peak_fanout = 38usize;
    let syn_newborn = n_neurons * newborn_fanout;

    // 1. Childhood Peak (3.8x overproduction, unmyelinated delays ~3..5)
    let (mut peak_fab, _) =
        MorphogenRouter::decompress_fabric(n_neurons, &genome, peak_fanout, seed, cfg);
    let syn_peak = peak_fab.synapses.len();
    peak_fab.enable_usage_tracking();

    // Experience compilation stream (180 frames total = 150 childhood + 30 adolescent consolidation)
    // All 3 arms observe the EXACT same 180 experience frames!
    let exp_frames = PrenatalCurriculum::generate_retinal_waves(30, seed ^ 0x3E4E_0003);
    let test_frames = PrenatalCurriculum::generate_retinal_waves(18, seed ^ 0x7E57_0003);
    let prune_split = 150usize;
    let mut pat = [0i32; 64];
    for f in &exp_frames[..prune_split] {
        for i in 0..64 {
            pat[i] = (f[i] * 300.0).round() as i32;
        }
        let _ = peak_fab.settle_pattern_amortized(&pat, 2, 12, 10);
    }

    // Clone PeakUnpruned before microglia runs, then give it the remaining 30 frames (180 total)
    let mut arm_peak = peak_fab.clone();
    arm_peak.disable_usage_tracking();
    for f in &exp_frames[prune_split..] {
        for i in 0..64 {
            pat[i] = (f[i] * 300.0).round() as i32;
        }
        let _ = arm_peak.settle_pattern_amortized(&pat, 2, 12, 10);
    }

    // 2. CompiledMyelinated: Microglia physical pruning (~68%) + Saltatory Myelination (delay -> 1),
    //    followed by 30 adolescent consolidation frames (same 180 total experience frames!)
    let mut arm_compiled = peak_fab;
    let rep = DevelopmentalPruner::microglia_prune_and_myelinate(
        &mut arm_compiled,
        syn_newborn,
        0.32,
        0.80,
    );
    let final_fanout = (rep.synapses_compiled / n_neurons).max(4);
    for f in &exp_frames[prune_split..] {
        for i in 0..64 {
            pat[i] = (f[i] * 300.0).round() as i32;
        }
        let _ = arm_compiled.settle_pattern_amortized(&pat, 2, 12, 10);
    }

    // 3. Lottery Ticket Negative Control: SparseFromScratch (EXACT same final fanout = 12, NO overproduction!)
    let (mut arm_sparse, _) =
        MorphogenRouter::decompress_fabric(n_neurons, &genome, final_fanout, seed ^ 0x5A25, cfg);
    for f in &exp_frames {
        for i in 0..64 {
            pat[i] = (f[i] * 300.0).round() as i32;
        }
        let _ = arm_sparse.settle_pattern_amortized(&pat, 2, 12, 10);
    }

    // Evaluate all 3 arms on fresh held-out structured test frames (using clamped sensory_fe
    // so a silent/under-connected network cannot game total_fe by having zero hidden activity!)
    let (fe_peak, sfe_peak, ticks_peak) = eval_compiled_fabric(&mut arm_peak, &test_frames);
    let (fe_sparse, sfe_sparse, ticks_sparse) = eval_compiled_fabric(&mut arm_sparse, &test_frames);
    let (fe_comp, sfe_comp, ticks_comp) = eval_compiled_fabric(&mut arm_compiled, &test_frames);
    let _ = (fe_peak, fe_sparse, fe_comp);

    // Physical throughput benchmark at N = 2,048 neurons (PeakUnpruned vs CompiledMyelinated vs Sparse)
    let (mut big_peak, _) = MorphogenRouter::decompress_fabric(2048, &genome, peak_fanout, seed, cfg);
    big_peak.enable_usage_tracking();
    let mut prev = 0u8;
    for i in 0..60usize {
        let b = (i * 11) as u8;
        big_peak.inject_raw_byte(prev, b);
        let _ = big_peak.step();
        prev = b;
    }
    let mut big_peak_eval = big_peak.clone();
    big_peak_eval.disable_usage_tracking();
    let mut big_compiled = big_peak;
    let _ = DevelopmentalPruner::microglia_prune_and_myelinate(
        &mut big_compiled,
        2048 * newborn_fanout,
        0.32,
        0.75,
    );
    let (mut big_sparse, _) =
        MorphogenRouter::decompress_fabric(2048, &genome, final_fanout, seed ^ 0x5A25, cfg);

    let tps_peak = bench_tps_2048(&mut big_peak_eval, seed);
    let tps_comp = bench_tps_2048(&mut big_compiled, seed);
    let tps_sparse = bench_tps_2048(&mut big_sparse, seed);

    let delay_peak = arm_peak.axon_delay.iter().map(|&d| d as f64).sum::<f64>() / (n_neurons as f64);
    let delay_sparse =
        arm_sparse.axon_delay.iter().map(|&d| d as f64).sum::<f64>() / (n_neurons as f64);

    vec![
        G3ArmRow {
            seed,
            arm: "peak_unpruned",
            synapses_newborn: syn_newborn,
            synapses_peak: syn_peak,
            synapses_final: arm_peak.synapses.len(),
            csr_bytes_final: arm_peak.csr_memory_bytes(),
            prune_frac: 0.0,
            mean_axon_delay: delay_peak,
            eval_fe: sfe_peak,
            eval_sensory_fe: sfe_peak,
            avg_ticks: ticks_peak,
            throughput_tps_2048: tps_peak,
        },
        G3ArmRow {
            seed,
            arm: "sparse_from_scratch",
            synapses_newborn: syn_newborn,
            synapses_peak: syn_peak,
            synapses_final: arm_sparse.synapses.len(),
            csr_bytes_final: arm_sparse.csr_memory_bytes(),
            prune_frac: 1.0 - (arm_sparse.synapses.len() as f64) / (syn_peak as f64),
            mean_axon_delay: delay_sparse,
            eval_fe: sfe_sparse,
            eval_sensory_fe: sfe_sparse,
            avg_ticks: ticks_sparse,
            throughput_tps_2048: tps_sparse,
        },
        G3ArmRow {
            seed,
            arm: "compiled_myelinated",
            synapses_newborn: syn_newborn,
            synapses_peak: syn_peak,
            synapses_final: rep.synapses_compiled,
            csr_bytes_final: rep.csr_bytes_compiled,
            prune_frac: rep.prune_frac,
            mean_axon_delay: rep.mean_delay_after,
            eval_fe: sfe_comp,
            eval_sensory_fe: sfe_comp,
            avg_ticks: ticks_comp,
            throughput_tps_2048: tps_comp,
        },
    ]
}
