//! G4 — Subroutine 4: Prenatal Synthetic Curriculum (Spontaneous Coherent Retinal Waves
//! & Fetal Motor Twitches in Utero — ZERO Data Leakage, with White-Noise Negative Control).
//!
//! Arms:
//! 1. `no_prenatal` (born blank, 0 prenatal steps)
//! 2. `white_noise_prenatal` (NEGATIVE CONTROL: exact same number of prenatal updates on
//!    unstructured i.i.d. white noise -> proves coherence, not step count, creates calibration!)
//! 3. `prenatal_waves_and_twitches` (coherent 2D traveling retinal waves + fetal motor twitches)

use aura_genesis::{CanonicalColumnTemplate, CanonicalCortex, PrenatalCurriculum};
use aura_math::{LatentVec, Xorshift64};
use aura_world::{EnsembleWorldModel, WorldModel};

#[derive(Clone, Debug)]
pub struct G4ArmRow {
    pub seed: u64,
    pub arm: &'static str,
    pub osi_at_birth: f64,
    pub early_visual_fe: f64,
    pub early_sm_err: f64,
}

fn eval_early_visual_fe(cortex: &mut CanonicalCortex, real_bars: &[[f32; 64]]) -> f64 {
    let mut fe_sum = 0.0f64;
    for bar in real_bars {
        // Strictly prequential: measure top-down L2/3->L4 predictive coding Free Energy
        // BEFORE any postnatal weight update on this frame!
        fe_sum += cortex.predictive_free_energy(bar);
        let _ = cortex.forward_sensory(bar, true);
    }
    fe_sum / (real_bars.len().max(1) as f64)
}

fn eval_early_sensorimotor_err(world: &mut EnsembleWorldModel, seed: u64) -> f64 {
    let mut rng = Xorshift64::new(seed ^ 0x5E45_0004);
    let mut state = LatentVec::zero();
    let mut err_sum = 0.0f64;
    let n_trials = 24usize;

    for step in 0..n_trials {
        let action = ((step * 3 + 1) % 8) as u16;
        let eff_key = PrenatalCurriculum::encode_efference_copy(&state, action);
        // Postnatal real noisy sensorimotor test trial (novel noise amplitude = 10)
        let target_raw = PrenatalCurriculum::simulate_fetal_body_echo(&state, action, 10, &mut rng);
        let preds = world.predict_ensemble(&eff_key, action);
        let consensus = world.consensus_prediction(&preds);
        let target_ema = world.ema_encoder.encode_target(&target_raw);
        let l1 = target_ema.l1_distance(&consensus) as f64;
        let denom = target_ema.norm_l1().max(256) as f64;
        err_sum += l1 / denom;

        // Online postnatal experience continues causally AFTER scoring
        let _ = world.observe_and_decompose(&eff_key, action, &target_raw, 2.5);
        state = target_raw;
    }
    err_sum / (n_trials as f64)
}

pub fn run_g4(seed: u64) -> Vec<G4ArmRow> {
    let tpl = CanonicalColumnTemplate::default();

    // Prenatal streams (100% synthetic, ZERO test bars!)
    // Negative control matches exact per-frame pixel brightness/energy histogram while destroying
    // 100% of spatial and temporal wave coherence!
    let waves = PrenatalCurriculum::generate_retinal_waves(80, seed ^ 0x1004);
    let noise_frames = PrenatalCurriculum::generate_shuffled_noise_frames(&waves, seed ^ 0x2004);

    // Real postnatal visual stimuli: first 20 real oriented bar windows + 4x12 OSI probes
    let mut rng_test = Xorshift64::new(seed ^ 0x9999_0004);
    let mut real_bars_20 = Vec::with_capacity(20);
    for t in 0..20 {
        let ori = t % 4;
        let sh = rng_test.next_signed_f32() * 0.35;
        let pol = if (t & 1) == 0 { 1.0 } else { -1.0 };
        real_bars_20.push(PrenatalCurriculum::visual_oriented_bar_polarity(
            ori,
            sh,
            pol,
            0.06,
            &mut rng_test,
        ));
    }
    let mut osi_probes: [Vec<[f32; 64]>; 4] = Default::default();
    for ori in 0..4 {
        for k in 0..12 {
            let sh = ((k as f32) / 11.0 - 0.5) * 0.6;
            osi_probes[ori].push(PrenatalCurriculum::visual_oriented_bar(ori, sh, 0.06, &mut rng_test));
        }
    }

    let mut rows = Vec::with_capacity(3);
    for &(arm, mode) in &[
        ("no_prenatal", 0u8),
        ("white_noise_prenatal", 1u8),
        ("prenatal_waves_and_twitches", 2u8),
    ] {
        // 1. Canonical Cortex for BOTH zero-shot OSI at birth AND prequential visual predictive FE
        let mut cortex = CanonicalCortex::instantiate(8, 4, tpl, seed ^ 0xC047, true);
        // 2. EnsembleWorldModel for fetal motor twitch sensorimotor calibration
        //    (reducible_gating=false during prenatal twitch updates so WhiteNoisePrenatal
        //     actually executes the EXACT same number of weight updates as the negative control!)
        let mut world = EnsembleWorldModel::new(seed ^ 0x1102, true, false);

        match mode {
            0 => {
                // Born completely blank (0 prenatal steps)
            }
            1 => {
                // Negative control: exact same step count & pixel energy histogram on spatially shuffled noise
                for f in &noise_frames {
                    let _ = cortex.forward_sensory(f, true);
                }
                PrenatalCurriculum::run_fetal_motor_twitches(&mut world, 96, true, seed);
            }
            _ => {
                // Coherent prenatal retinal waves + spontaneous fetal motor twitches
                for f in &waves {
                    let _ = cortex.forward_sensory(f, true);
                }
                PrenatalCurriculum::run_fetal_motor_twitches(&mut world, 96, false, seed);
            }
        }

        // Measure strictly AT BIRTH (step 0 real) before ANY postnatal updates,
        // then evaluate prequential predictive coding Free Energy over the first 20 real bars!
        let osi_birth = cortex.measure_osi(&osi_probes);
        let early_fe = eval_early_visual_fe(&mut cortex, &real_bars_20);
        let early_sm = eval_early_sensorimotor_err(&mut world, seed);

        rows.push(G4ArmRow {
            seed,
            arm,
            osi_at_birth: osi_birth,
            early_visual_fe: early_fe,
            early_sm_err: early_sm,
        });
    }
    rows
}
