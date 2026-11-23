//! G5 — End-to-End Integration: AURA v3 Genesis (All 4 DNA Subroutines Integrated)
//! vs AURA v2 Congelado on BOTH `CorticalFabric` (G5a) and Real Byte-Level Language
//! Modeling + Online Rust Domain Shift (G5b).

use crate::bench_b6::load_repo_corpora;
use aura_cortex::{CorticalFabric, FabricConfig};
use aura_decoder::{AuraByteModel, BigramByteBaseline};
use aura_genesis::{DevelopmentalPruner, MorphogenGenome, MorphogenRouter, PrenatalCurriculum};
use aura_math::Xorshift64;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct G5FabricRow {
    pub seed: u64,
    pub arm: &'static str,
    pub genome_comp_2048: f64,
    pub synapses_final: usize,
    pub early_fe: f64,
    pub late_fe: f64,
    pub avg_ticks: f64,
}

#[derive(Clone, Debug)]
pub struct G5ByteRow {
    pub seed: u64,
    pub arm: &'static str,
    pub early_md_bpb: f64,
    pub eval_md_bpb: f64,
    pub shift_rust_bpb: f64,
    pub nonzero_peak_frac: f64,
    pub nonzero_final_frac: f64,
    pub pruned_synapses: usize,
    pub compression_ratio: f64,
    pub effective_rank: f64,
    pub train_bytes_per_sec: f64,
}

#[derive(Clone, Debug)]
pub struct G5Result {
    pub fabric_rows: Vec<G5FabricRow>,
    pub byte_rows: Vec<G5ByteRow>,
    pub sample_md_gen: String,
    pub sample_rust_gen: String,
}

fn sanitize_ascii(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| {
            if b == b'\n' {
                ' '
            } else if (32..=126).contains(&b) {
                b as char
            } else {
                '.'
            }
        })
        .collect()
}

pub fn run_g5(seed: u64, repo_root: &Path, emit_samples: bool) -> G5Result {
    // =========================================================================
    // G5a: CorticalFabric v3 Genesis (All 4 Subroutines) vs CorticalFabric v2
    // =========================================================================
    let genome = MorphogenGenome::default();
    let cfg_phys = FabricConfig {
        myelin_delay_enabled: true,
        ..Default::default()
    };

    // Genome compression reference at N = 2,048
    let (fab_2048, _) = MorphogenRouter::decompress_fabric(2048, &genome, 24, seed, cfg_phys);
    let comp_2048 = (fab_2048.csr_memory_bytes() as f64) / (MorphogenGenome::SERIALIZED_BYTES as f64);

    // Arm 1: AURA v2 baseline CorticalFabric (random wiring, unmyelinated delay=3, 0 prenatal waves, no pruning)
    let mut fab_v2 = CorticalFabric::build_cortical_sheet(256, seed, cfg_phys);
    fab_v2.axon_delay.fill(3);

    // Arm 2: AURA v3 Genesis CorticalFabric:
    //   1. Morphogen 3D decompression (overproduced ChildhoodPeak = 38 fanout)
    //   2. Prenatal coherent wave calibration (ZERO test frames!)
    //   3. Microglia structural pruning (~68%) + physical myelination (delay -> 1)
    let (mut fab_v3, _) = MorphogenRouter::decompress_fabric(256, &genome, 38, seed, cfg_phys);
    fab_v3.enable_usage_tracking();
    let pre_waves = PrenatalCurriculum::generate_retinal_waves(24, seed ^ 0x6500);
    let mut pat = [0i32; 64];
    for f in &pre_waves {
        for i in 0..64 {
            pat[i] = (f[i] * 300.0).round() as i32;
        }
        let _ = fab_v3.settle_pattern_amortized(&pat, 2, 8, 10);
    }
    let _ = DevelopmentalPruner::microglia_prune_and_myelinate(&mut fab_v3, 256 * 10, 0.32, 0.85);
    fab_v3.reset_potentials_cold();
    fab_v3.ema_fe_fast = 0;
    fab_v3.ema_fe_slow = 0;
    fab_v3.prev_fe = 0;
    fab_v3.tick = 0;

    // Postnatal real evaluation stream (160 structured 2D oriented-bar frames, disjoint seed)
    let mut rng_post = Xorshift64::new(seed ^ 0xB057_6500);
    let mut post_frames = Vec::with_capacity(160);
    for t in 0..160 {
        let ori = (t / 4) % 4;
        let sh = ((t % 4) as f32 - 1.5) * 0.22;
        post_frames.push(PrenatalCurriculum::visual_oriented_bar(ori, sh, 0.06, &mut rng_post));
    }

    let eval_f = |fab: &mut CorticalFabric| -> (f64, f64, f64) {
        let mut early_sum = 0.0f64;
        let mut late_sum = 0.0f64;
        let mut ticks_sum = 0usize;
        let mut p = [0i32; 64];
        for (idx, f) in post_frames.iter().enumerate() {
            for i in 0..64 {
                p[i] = (f[i] * 300.0).round() as i32;
            }
            let st = fab.settle_pattern_amortized(&p, 2, 14, 10);
            ticks_sum += st.ticks_used;
            if idx < 20 {
                early_sum += st.final_fe as f64;
            }
            if idx >= 100 {
                late_sum += st.final_fe as f64;
            }
        }
        (
            early_sum / 20.0,
            late_sum / 60.0,
            (ticks_sum as f64) / (post_frames.len() as f64),
        )
    };

    let (early_v2, late_v2, ticks_v2) = eval_f(&mut fab_v2);
    let (early_v3, late_v3, ticks_v3) = eval_f(&mut fab_v3);

    let fabric_rows = vec![
        G5FabricRow {
            seed,
            arm: "aura_v2_fabric",
            genome_comp_2048: 1.0,
            synapses_final: fab_v2.synapses.len(),
            early_fe: early_v2,
            late_fe: late_v2,
            avg_ticks: ticks_v2,
        },
        G5FabricRow {
            seed,
            arm: "aura_v3_genesis_fabric",
            genome_comp_2048: comp_2048,
            synapses_final: fab_v3.synapses.len(),
            early_fe: early_v3,
            late_fe: late_v3,
            avg_ticks: ticks_v3,
        },
    ];

    // =========================================================================
    // G5b: AuraByteModel v3 Genesis vs AURA v2 Congelado on Real Repo Bytes
    // =========================================================================
    let (train_md, eval_md, shift_rust) = load_repo_corpora(repo_root);

    let mut bigram = BigramByteBaseline::new(0.25);
    bigram.fit(&train_md);
    let bg_md = bigram.eval_bpb(&eval_md);
    let bg_rs = bigram.eval_bpb(&shift_rust);

    let epochs = 4usize;
    let chunk_len = 32usize;

    // AURA v2 Reference (exact v2 init & training)
    let mut rng_v2 = Xorshift64::new(seed ^ 0x0002_65B0);
    let mut m_v2 = AuraByteModel::new(seed);
    let st_v2 = m_v2.train_corpus_v2_tracked(&train_md, &eval_md, epochs, chunk_len, &mut rng_v2);
    let v2_rs_online = m_v2.eval_stream_bpb(&shift_rust, true);

    // AURA v3 Genesis (all 4 DNA subroutines: Morphogen init + Canonical 4-col + Prenatal waves + Microglia prune & SSM myelin)
    // Uses the EXACT same postnatal chunk-shuffle RNG as AURA v2 for strictly paired evaluation!
    let mut rng_v3 = Xorshift64::new(seed ^ 0x0002_65B0);
    let mut m_v3 = AuraByteModel::new_genesis(seed);
    let st_v3 = m_v3.train_corpus_genesis(
        &train_md,
        &eval_md,
        epochs,
        chunk_len,
        seed ^ 0x0004_65B0,
        &mut rng_v3,
    );
    let v3_rs_frozen = m_v3.eval_stream_bpb(&shift_rust, false);
    let v3_rs_online = m_v3.eval_stream_bpb(&shift_rust, true);

    let (sample_md_gen, sample_rust_gen) = if emit_samples {
        let mut s_rng = Xorshift64::new(seed ^ 0x5A0B_0003);
        let md_g = m_v3.generate_bytes(b"## Resultados de ", 96, 0.85, &mut s_rng);
        let rs_g = m_v3.generate_bytes(b"pub fn ", 96, 0.85, &mut s_rng);
        (sanitize_ascii(&md_g), sanitize_ascii(&rs_g))
    } else {
        (String::new(), String::new())
    };

    let byte_rows = vec![
        G5ByteRow {
            seed,
            arm: "bigram_byte_baseline",
            early_md_bpb: bg_md,
            eval_md_bpb: bg_md,
            shift_rust_bpb: bg_rs,
            nonzero_peak_frac: 1.0,
            nonzero_final_frac: 1.0,
            pruned_synapses: 0,
            compression_ratio: 1.0,
            effective_rank: 0.0,
            train_bytes_per_sec: 0.0,
        },
        G5ByteRow {
            seed,
            arm: "aura_v2_baseline",
            early_md_bpb: st_v2.early_eval_bpb,
            eval_md_bpb: st_v2.base.eval_bpb,
            shift_rust_bpb: v2_rs_online,
            nonzero_peak_frac: st_v2.nonzero_peak_frac,
            nonzero_final_frac: st_v2.nonzero_final_frac,
            pruned_synapses: 0,
            compression_ratio: st_v2.base.compression_ratio,
            effective_rank: st_v2.base.effective_rank,
            train_bytes_per_sec: st_v2.base.train_bytes_per_sec,
        },
        G5ByteRow {
            seed,
            arm: "aura_v3_genesis_frozen_infer",
            early_md_bpb: st_v3.early_eval_bpb,
            eval_md_bpb: st_v3.base.eval_bpb,
            shift_rust_bpb: v3_rs_frozen,
            nonzero_peak_frac: st_v3.nonzero_peak_frac,
            nonzero_final_frac: st_v3.nonzero_final_frac,
            pruned_synapses: st_v3.pruned_synapses,
            compression_ratio: st_v3.base.compression_ratio,
            effective_rank: st_v3.base.effective_rank,
            train_bytes_per_sec: st_v3.base.train_bytes_per_sec,
        },
        G5ByteRow {
            seed,
            arm: "aura_v3_genesis_full",
            early_md_bpb: st_v3.early_eval_bpb,
            eval_md_bpb: st_v3.base.eval_bpb,
            shift_rust_bpb: v3_rs_online,
            nonzero_peak_frac: st_v3.nonzero_peak_frac,
            nonzero_final_frac: st_v3.nonzero_final_frac,
            pruned_synapses: st_v3.pruned_synapses,
            compression_ratio: st_v3.base.compression_ratio,
            effective_rank: st_v3.base.effective_rank,
            train_bytes_per_sec: st_v3.base.train_bytes_per_sec,
        },
    ];

    G5Result {
        fabric_rows,
        byte_rows,
        sample_md_gen,
        sample_rust_gen,
    }
}
