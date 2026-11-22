//! B5 — Bottleneck 5: Latent Space Stability, Solving the TV-Static Epistemic Trap,
//! Phase-Aware Zero-Token Entropy Patching, and VICReg Anti-Collapse.
//!
//! Directly targets BOTH v1 E1 limitations (§7.1 fixed-4 phase alignment & §7.2 TV-static trap)
//! plus Part 2 §6.1 of the proposal.
//!
//! Stream (3,200 bytes per seed, 4 phases):
//! - Phase 1 (900 B): Familiar Regime A (mixed periodic 12B frames + variable-length structured words)
//! - Phase 2 (800 B): Abrupt Law Shift to structured Regime B (learnable)
//! - Phase 3 (600 B): TV Static (uniform white random noise -> unlearnable aleatoric!)
//! - Phase 4A (450 B): Return to Regime A (hippocampal one-shot recall test)
//! - Phase 4C (450 B): Unseen Structured OOD Regime C (learnable epistemic novelty!)

use crate::agent::AuraAgent;
use aura_math::{effective_rank_latents, LatentVec, Xorshift64};
use aura_metacog::MetacogConfig;
use aura_perception::{EntropyPatcher, EntropyPatcherConfig, FixedPatcher, PerceptionEncoder};

#[derive(Clone, Debug)]
pub struct B5ArmRow {
    pub seed: u64,
    pub arm: &'static str,
    pub total_patches: usize,
    pub compression_ratio: f64,
    pub bpp_predictable: f64,
    pub bpp_static: f64,
    pub p1_err: f64,
    pub p2_err: f64,
    pub p4a_err: f64,
    pub effective_rank: f64,
    pub familiar_reducible_epi: f64,
    pub static_reducible_epi: f64,
    pub ood_reducible_epi: f64,
    pub static_aleatoric: f64,
    pub ood_aleatoric: f64,
    pub static_hipp_writes: usize,
    pub ood_hipp_writes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    P1Familiar,
    P2Shift,
    P3TvStatic,
    P4AReturn,
    P4COod,
}

/// Generate a mixed periodic + variable-length aperiodic structured byte stream for a regime.
fn append_regime(
    stream: &mut Vec<(u8, Phase)>,
    n_bytes: usize,
    phase: Phase,
    vocab: &[&[u8]],
    rng: &mut Xorshift64,
) {
    let mut written = 0usize;
    let mut step = 0usize;
    while written < n_bytes {
        // Deterministic Markov-like transition over variable-length words (lengths 5..12, aperiodic!)
        let idx = (step + (rng.next_usize(2))) % vocab.len();
        let word = vocab[idx];
        for &b in word {
            if written >= n_bytes {
                break;
            }
            stream.push((b, phase));
            written += 1;
        }
        step = (step + 1) % vocab.len();
    }
}

fn build_stream(seed: u64) -> Vec<(u8, Phase)> {
    let mut rng = Xorshift64::new(seed ^ 0xB500_7E57);
    let mut s = Vec::with_capacity(3200);

    // Regime A: mixed variable-length structured tokens (lengths 5, 7, 9, 12 -> aperiodic!)
    let vocab_a: [&[u8]; 4] = [
        b"ALPHA|",
        b"BETA_01|",
        b"GAMMA_XYZ|",
        b"DELTA_FRAME|",
    ];
    // Regime B: abrupt structural law shift
    let vocab_b: [&[u8]; 4] = [
        b"[NODE:",
        b"EDGE=>",
        b"STATE#9]",
        b"{FLUX=42}",
    ];
    // Regime C: unseen structured OOD grammar (learnable, completely distinct alphabet!)
    let vocab_c: [&[u8]; 4] = [
        b"quantum;",
        b"lattice:",
        b"simplex/",
        b"tropical!",
    ];

    append_regime(&mut s, 900, Phase::P1Familiar, &vocab_a, &mut rng);
    append_regime(&mut s, 800, Phase::P2Shift, &vocab_b, &mut rng);
    // Phase 3: Pure uniform TV static (600 bytes)
    for _ in 0..600 {
        let rb = (rng.next_u32() & 0xFF) as u8;
        s.push((rb, Phase::P3TvStatic));
    }
    append_regime(&mut s, 450, Phase::P4AReturn, &vocab_a, &mut rng);
    append_regime(&mut s, 450, Phase::P4COod, &vocab_c, &mut rng);
    s
}

fn eval_arm<P: PerceptionEncoder>(
    seed: u64,
    arm: &'static str,
    stream: &[(u8, Phase)],
    mut agent: AuraAgent<P>,
) -> B5ArmRow {
    let mut total_patches = 0usize;
    let mut pred_bytes = 0usize;
    let mut pred_patches = 0usize;
    let mut static_bytes = 0usize;
    let mut static_patches = 0usize;

    let mut p1_err_sum = 0.0f64;
    let mut p1_cnt = 0usize;
    let mut p2_err_sum = 0.0f64;
    let mut p2_cnt = 0usize;
    let mut p4a_err_sum = 0.0f64;
    let mut p4a_cnt = 0usize;

    let mut fam_epi_sum = 0.0f64;
    let mut fam_epi_cnt = 0usize;
    let mut stat_epi_sum = 0.0f64;
    let mut stat_epi_cnt = 0usize;
    let mut ood_epi_sum = 0.0f64;
    let mut ood_epi_cnt = 0usize;

    let mut stat_alea_sum = 0.0f64;
    let mut ood_alea_sum = 0.0f64;

    let mut stat_hipp_writes = 0usize;
    let mut ood_hipp_writes = 0usize;

    let mut latent_log: Vec<LatentVec> = Vec::with_capacity(512);

    for (idx, &(b, phase)) in stream.iter().enumerate() {
        if let Some(em) = agent.perception.push_byte(b) {
            total_patches += 1;
            let (err, rep, wrote) = agent.step_patch(&em);
            if latent_log.len() < 512 && phase != Phase::P3TvStatic {
                latent_log.push(em.latent);
            }

            match phase {
                Phase::P1Familiar => {
                    if idx >= 180 {
                        pred_bytes += em.patch_len;
                        pred_patches += 1;
                        p1_err_sum += err;
                        p1_cnt += 1;
                    }
                    if idx >= 450 {
                        fam_epi_sum += rep.reducible_epistemic as f64;
                        fam_epi_cnt += 1;
                    }
                }
                Phase::P2Shift => {
                    p2_err_sum += err;
                    p2_cnt += 1;
                }
                Phase::P3TvStatic => {
                    static_bytes += em.patch_len;
                    static_patches += 1;
                    // Skip first 40 bytes of transition ramp into TV static
                    if idx >= 1740 {
                        stat_epi_sum += rep.reducible_epistemic as f64;
                        stat_alea_sum += rep.aleatoric_noise as f64;
                        stat_epi_cnt += 1;
                        if wrote {
                            stat_hipp_writes += 1;
                        }
                    }
                }
                Phase::P4AReturn => {
                    p4a_err_sum += err;
                    p4a_cnt += 1;
                }
                Phase::P4COod => {
                    // Skip first 30 bytes of transition from Phase 4A
                    if idx >= 2780 {
                        ood_epi_sum += rep.reducible_epistemic as f64;
                        ood_alea_sum += rep.aleatoric_noise as f64;
                        ood_epi_cnt += 1;
                        if wrote {
                            ood_hipp_writes += 1;
                        }
                    }
                }
            }
        }
    }

    let bpp_pred = (pred_bytes as f64) / (pred_patches.max(1) as f64);
    let bpp_stat = (static_bytes as f64) / (static_patches.max(1) as f64);
    let erank = effective_rank_latents(&latent_log, 64);

    B5ArmRow {
        seed,
        arm,
        total_patches,
        compression_ratio: bpp_pred / bpp_stat.max(0.01),
        bpp_predictable: bpp_pred,
        bpp_static: bpp_stat,
        p1_err: p1_err_sum / (p1_cnt.max(1) as f64),
        p2_err: p2_err_sum / (p2_cnt.max(1) as f64),
        p4a_err: p4a_err_sum / (p4a_cnt.max(1) as f64),
        effective_rank: erank,
        familiar_reducible_epi: fam_epi_sum / (fam_epi_cnt.max(1) as f64),
        static_reducible_epi: stat_epi_sum / (stat_epi_cnt.max(1) as f64),
        ood_reducible_epi: ood_epi_sum / (ood_epi_cnt.max(1) as f64),
        static_aleatoric: stat_alea_sum / (stat_epi_cnt.max(1) as f64),
        ood_aleatoric: ood_alea_sum / (ood_epi_cnt.max(1) as f64),
        static_hipp_writes: stat_hipp_writes,
        ood_hipp_writes: ood_hipp_writes,
    }
}

pub fn run_b5(seed: u64) -> Vec<B5ArmRow> {
    let stream = build_stream(seed);
    let mcfg = MetacogConfig {
        doubt_threshold: 380,
        abstain_threshold: 3800,
        hipp_write_threshold: 180,
        system2_enabled: true,
        ..Default::default()
    };

    // 1. v1 baseline: post-byte cut (phase_aware=false) + raw disagreement (reducible_gating=false)
    let p_v1 = EntropyPatcher::new(
        seed,
        EntropyPatcherConfig {
            phase_aware: false,
            vicreg_enabled: true,
            ..Default::default()
        },
    );
    let a_v1 = AuraAgent::new(p_v1, seed, true, true, false, mcfg);
    let r_v1 = eval_arm(seed, "aura_v1_baseline", &stream, a_v1);

    // 2. Fixed window 4 control
    let p_fix = FixedPatcher::new(seed, 4);
    let a_fix = AuraAgent::new(p_fix, seed, true, true, true, mcfg);
    let r_fix = eval_arm(seed, "fixed_patcher_4", &stream, a_fix);

    // 3. No-VICReg ablation (verifies dimensional collapse without VICReg!)
    let p_novic = EntropyPatcher::new(
        seed,
        EntropyPatcherConfig {
            phase_aware: true,
            vicreg_enabled: false,
            ..Default::default()
        },
    );
    let a_novic = AuraAgent::new(p_novic, seed, true, true, true, mcfg);
    let r_novic = eval_arm(seed, "no_vicreg", &stream, a_novic);

    // 4. AURA v2 Full (phase-aware BLT patching + VICReg + reducible learning-progress epistemic!)
    let p_v2 = EntropyPatcher::new(
        seed,
        EntropyPatcherConfig {
            phase_aware: true,
            vicreg_enabled: true,
            ..Default::default()
        },
    );
    let a_v2 = AuraAgent::new(p_v2, seed, true, true, true, mcfg);
    let r_v2 = eval_arm(seed, "aura_v2_full", &stream, a_v2);

    vec![r_v1, r_fix, r_novic, r_v2]
}
