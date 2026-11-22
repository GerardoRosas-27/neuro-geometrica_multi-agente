//! G1 — Subroutine 1: Chemical Gradient Routing (MorphogenGenome + Autonomous Growth Cones).
//!
//! Tests:
//! - H_G1a: Genome <= 256 bytes achieving >1,000x compression vs explicit CSR (N=2,048).
//! - H_G1b: Retinotopic neighborhood preservation (`topo_corr > 0.85` vs ~0.0 random v2).
//! - H_G1c: Functional predictive advantage (lower Free Energy on smooth 2D spatial wave
//!          stimuli vs random-wired v2 with identical neuron & synapse budgets).

use aura_cortex::{CorticalFabric, FabricConfig};
use aura_genesis::{MorphogenGenome, MorphogenRouter, PrenatalCurriculum};

#[derive(Clone, Debug)]
pub struct G1ArmRow {
    pub seed: u64,
    pub arm: &'static str,
    pub genome_bytes: usize,
    pub csr_bytes_2048: usize,
    pub compression_ratio_2048: f64,
    pub topo_corr_2048: f64,
    pub spatial_late_fe: f64,
    pub spatial_sensory_fe: f64,
    pub avg_ticks: f64,
}

fn eval_spatial_fabric(fabric: &mut CorticalFabric, seed: u64) -> (f64, f64, f64) {
    // Smooth 2D moving spatial bars / waves across the 8x8 = 64 sensory sheet (160 frames)
    let frames = PrenatalCurriculum::generate_retinal_waves(28, seed ^ 0x6100_5BA7);
    let mut fe_sum = 0.0f64;
    let mut sfe_sum = 0.0f64;
    let mut tick_sum = 0usize;
    let mut cnt = 0usize;
    let mut pat = [0i32; 64];

    for (idx, f) in frames.iter().take(160).enumerate() {
        for i in 0..64 {
            pat[i] = (f[i] * 300.0).round() as i32;
        }
        let st = fabric.settle_pattern_amortized(&pat, 2, 12, 10);
        if idx >= 80 {
            fe_sum += st.final_fe as f64;
            sfe_sum += st.sensory_fe as f64;
            tick_sum += st.ticks_used;
            cnt += 1;
        }
    }
    let d = cnt.max(1) as f64;
    (fe_sum / d, sfe_sum / d, (tick_sum as f64) / d)
}

pub fn run_g1(seed: u64) -> Vec<G1ArmRow> {
    let genome = MorphogenGenome::default();
    let genome_bytes = genome.to_bytes().len();
    let cfg = FabricConfig::default();

    // 1. Scale compression & retinotopy at N = 2,048 neurons (fanout = 24)
    let (fab_morpho_2048, pos_2048) =
        MorphogenRouter::decompress_fabric(2048, &genome, 24, seed, cfg);
    let csr_m_2048 = fab_morpho_2048.csr_memory_bytes();
    let comp_m = (csr_m_2048 as f64) / (genome_bytes as f64);
    let topo_m = MorphogenRouter::measure_retinotopic_correlation(&fab_morpho_2048, &pos_2048);

    // Random v2 baseline at N = 2,048 (explicit CSR requires storing all edges -> comp = 1.0x)
    let fab_rand_2048 = CorticalFabric::build_cortical_sheet(2048, seed, cfg);
    let csr_r_2048 = fab_rand_2048.csr_memory_bytes();
    let topo_r = MorphogenRouter::measure_retinotopic_correlation(&fab_rand_2048, &pos_2048);

    // 2. Functional apples-to-apples spatial predictive test at N = 256 (EXACT same fanout = 20!)
    let mut fab_morpho_256 =
        MorphogenRouter::decompress_fabric(256, &genome, 20, seed, cfg).0;
    let mut fab_rand_256 = CorticalFabric::build_cortical_sheet(256, seed, cfg);

    let (fe_m, sfe_m, t_m) = eval_spatial_fabric(&mut fab_morpho_256, seed);
    let (fe_r, sfe_r, t_r) = eval_spatial_fabric(&mut fab_rand_256, seed);

    vec![
        G1ArmRow {
            seed,
            arm: "random_v2_wiring",
            genome_bytes: csr_r_2048,
            csr_bytes_2048: csr_r_2048,
            compression_ratio_2048: 1.0,
            topo_corr_2048: topo_r,
            spatial_late_fe: fe_r,
            spatial_sensory_fe: sfe_r,
            avg_ticks: t_r,
        },
        G1ArmRow {
            seed,
            arm: "morphogen_router_v3",
            genome_bytes,
            csr_bytes_2048: csr_m_2048,
            compression_ratio_2048: comp_m,
            topo_corr_2048: topo_m,
            spatial_late_fe: fe_m,
            spatial_sensory_fe: sfe_m,
            avg_ticks: t_m,
        },
    ]
}
