//! Experimento controlado pre/post de deformación del paisaje energético.
//!
//! Una configuración binaria verificada se mantiene primero en memoria rápida.
//! El sueño la transfiere a las fases de arista CDT. Después se repiten
//! exactamente los mismos cues corrompidos sobre snapshots pre y post para
//! medir si cambió la cuenca del atractor, sin confundir el resultado con
//! energía física ni generalización fuera de distribución.

use crate::native_hybrid_phasor_cdt_engine::{
    NativeHybridConfig, NativeHybridError, NativeHybridPhasorCdtEngine, NativePhasorCue,
};
use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use crate::native_rng::{splitmix64, unit_from_u64};
use crate::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use rayon::prelude::*;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct ConsolidationBasinConfig {
    pub nodes: usize,
    pub trials_per_corruption: usize,
    pub corruption_fractions: Vec<f32>,
    pub phase_jitter: f32,
    pub success_accuracy: f32,
    pub basin_success_rate: f32,
    /// Ganancia media mínima de tasa de éxito post-sueño para que el gate
    /// declare expansión de cuenca. Versionado en config: un literal invisible
    /// en el código sería un umbral ajustable sin rastro.
    pub minimum_mean_success_gain: f32,
    pub seed: u64,
}

impl Default for ConsolidationBasinConfig {
    fn default() -> Self {
        Self {
            nodes: 32,
            trials_per_corruption: 24,
            corruption_fractions: vec![0.10, 0.20, 0.25, 0.30, 0.35, 0.40],
            phase_jitter: 0.10,
            success_accuracy: 0.90,
            basin_success_rate: 0.80,
            minimum_mean_success_gain: 0.10,
            seed: 0xBA51_CD72_2026,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BasinLevelMetrics {
    pub corruption_fraction: f32,
    pub trials: usize,
    /// Éxitos medidos con la exactitud directa (sin identificación de gauge).
    pub successes: usize,
    pub success_rate: f32,
    /// Exactitud directa: fracción de nodos cuyo signo coincide con el target.
    /// Es la métrica primaria; el flip global Z₂ cuenta como fallo.
    pub mean_accuracy: f32,
    /// Diagnóstico invariante ante el flip global Z₂ (un estado invertido
    /// completo cuenta como acierto). No alimenta el gate ni `successes`;
    /// sirve para distinguir recuperación real de un cambio de gauge global:
    /// si `mean_accuracy` cae pero esta métrica permanece alta, la dinámica
    /// está convergiendo al atractor con la convención de signo invertida.
    pub mean_gauge_invariant_accuracy: f32,
    pub mean_final_energy: f32,
    pub mean_iterations: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConsolidationBasinReport {
    pub nodes: usize,
    pub trials_per_corruption: usize,
    pub target_checksum: u64,
    pub sleep_accepted: usize,
    pub consolidated_edges: usize,
    pub pre: Vec<BasinLevelMetrics>,
    pub post: Vec<BasinLevelMetrics>,
    pub pre_critical_corruption: f32,
    pub post_critical_corruption: f32,
    pub mean_success_gain: f32,
    /// Tiempo de pared total del experimento (segundos). Permite detectar
    /// regresiones de coste, no sólo de tasas.
    pub wall_clock_seconds: f64,
    pub decision: &'static str,
}

/// Ejecuta un ensayo pareado: los targets, corrupciones, jitter y solver son
/// idénticos antes y después; la única variable es el commit de sueño a CDT.
pub fn run_consolidation_basin_experiment(
    config: ConsolidationBasinConfig,
) -> Result<ConsolidationBasinReport, NativeHybridError> {
    let started = Instant::now();
    let config = sanitize_config(config);
    let target = balanced_target(config.nodes, config.seed);
    let target_checksum = target_checksum(&target);
    let mut engine = training_engine(config.nodes, config.seed)?;
    let pre_core = engine.core.clone();

    let full_experience = target
        .iter()
        .enumerate()
        .map(|(node, bit)| NativePhasorCue {
            node,
            amplitude: 1.0,
            phase: bit_phase(*bit),
        })
        .collect::<Vec<_>>();
    let wake = engine.infer_and_stage(&full_experience)?;
    if !wake.gate.passed {
        return Ok(empty_failed_report(
            config.nodes,
            config.trials_per_corruption,
            target_checksum,
            "wake_gate_failed",
        ));
    }
    let sleep = engine.sleep_consolidate()?;
    if sleep.accepted != 1 {
        return Ok(empty_failed_report(
            config.nodes,
            config.trials_per_corruption,
            target_checksum,
            "sleep_gate_failed",
        ));
    }

    let pre = evaluate_basin(&pre_core, &target, &config);
    let post = evaluate_basin(&engine.core, &target, &config);
    let pre_critical_corruption = critical_corruption(&pre, config.basin_success_rate);
    let post_critical_corruption = critical_corruption(&post, config.basin_success_rate);
    let mean_success_gain = pre
        .iter()
        .zip(&post)
        .map(|(pre, post)| post.success_rate - pre.success_rate)
        .sum::<f32>()
        / pre.len().max(1) as f32;
    let decision = if post_critical_corruption > pre_critical_corruption
        && mean_success_gain >= config.minimum_mean_success_gain
        && post
            .iter()
            .zip(&pre)
            .all(|(post, pre)| post.mean_accuracy + 1.0e-6 >= pre.mean_accuracy)
    {
        "basin_expansion_pass"
    } else {
        "basin_expansion_not_demonstrated"
    };

    Ok(ConsolidationBasinReport {
        nodes: config.nodes,
        trials_per_corruption: config.trials_per_corruption,
        target_checksum,
        sleep_accepted: sleep.accepted,
        consolidated_edges: sleep.consolidated_edges,
        pre,
        post,
        pre_critical_corruption,
        post_critical_corruption,
        mean_success_gain,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        decision,
    })
}

fn training_engine(
    nodes: usize,
    seed: u64,
) -> Result<NativeHybridPhasorCdtEngine, NativeHybridError> {
    NativeHybridPhasorCdtEngine::new(
        NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: nodes,
            spatial_degree: 4,
            temporal_degree: 1,
            temperature: 0.0,
            diffusion: 0.0,
            pilot_gain: 0.0,
            amplitude_decay: 0.0,
            seed,
            ..NativeThermoCdtConfig::default()
        },
        NativePhasorConfig {
            coupling_strength: 0.0,
            radial_strength: 4.0,
            target_amplitude: 1.0,
            confinement: 0.0,
            stimulus_gain: 0.0,
            entropy_weight: 0.0,
            temperature_scale: 0.0,
            noise_scale: 0.0,
            ..NativePhasorConfig::default()
        },
        NativeHybridConfig {
            minimizer: NativePhasorMinimizerConfig {
                max_iterations: 40,
                residual_tolerance: 1.0e-5,
                topological_warm_start: false,
                ..NativePhasorMinimizerConfig::default()
            },
            consolidation_learning_rate: 1.0,
            minimum_relative_energy_drop: 0.0,
            maximum_residual: 1.0e-5,
            minimum_magnetic_coherence: -1.0,
            minimum_stability: 0.99,
            stability_probe_jitter: 0.01,
            stability_probe_iterations: 8,
            cdt_consolidation_steps: 0,
            ..NativeHybridConfig::default()
        },
    )
}

fn evaluate_basin(
    core: &NativeThermoCdtSubstrate,
    target: &[i8],
    config: &ConsolidationBasinConfig,
) -> Vec<BasinLevelMetrics> {
    let template = NativePhasorThermodynamicEngine::from_core(
        core,
        NativePhasorConfig {
            coupling_strength: 2.0,
            radial_strength: 4.0,
            target_amplitude: 1.0,
            confinement: 0.0,
            stimulus_gain: 0.0,
            entropy_weight: 0.0,
            temperature_scale: 0.0,
            noise_scale: 0.0,
            ..NativePhasorConfig::default()
        },
    )
    .expect("el snapshot CDT producido por el experimento debe ser válido");

    config
        .corruption_fractions
        .iter()
        .copied()
        .enumerate()
        .map(|(level, corruption_fraction)| {
            // `map_init` clona el template una vez por hilo trabajador en vez
            // de una por ensayo; cada trial sólo reescribe `phasors`. El
            // `collect` conserva el orden de índice, así que las sumas son
            // idénticas bit a bit al recorrido secuencial.
            let results = (0..config.trials_per_corruption)
                .into_par_iter()
                .map_init(
                    || template.clone(),
                    |inference, trial| {
                        let cue = corrupted_phases(
                            target,
                            corruption_fraction,
                            config.phase_jitter,
                            config.seed ^ (level as u64).rotate_left(17),
                            trial,
                        );
                        for (phasor, phase) in inference.phasors.iter_mut().zip(cue) {
                            *phasor = Complex32::from_polar(1.0, phase);
                        }
                        let result = inference.minimize_free_energy(NativePhasorMinimizerConfig {
                            max_iterations: 300,
                            residual_tolerance: 2.0e-3,
                            topological_warm_start: false,
                            ..NativePhasorMinimizerConfig::default()
                        });
                        let accuracy = direct_accuracy(&inference.phasors, target);
                        (
                            usize::from(accuracy >= config.success_accuracy),
                            accuracy,
                            gauge_invariant_accuracy(&inference.phasors, target),
                            result.final_report.free_energy,
                            result.iterations,
                        )
                    },
                )
                .collect::<Vec<_>>();
            let mut successes = 0;
            let mut accuracy_sum = 0.0;
            let mut gauge_accuracy_sum = 0.0;
            let mut energy_sum = 0.0;
            let mut iteration_sum = 0;
            for (success, accuracy, gauge_accuracy, energy, iterations) in results {
                successes += success;
                accuracy_sum += accuracy;
                gauge_accuracy_sum += gauge_accuracy;
                energy_sum += energy;
                iteration_sum += iterations;
            }
            BasinLevelMetrics {
                corruption_fraction,
                trials: config.trials_per_corruption,
                successes,
                success_rate: successes as f32 / config.trials_per_corruption as f32,
                mean_accuracy: accuracy_sum / config.trials_per_corruption as f32,
                mean_gauge_invariant_accuracy: gauge_accuracy_sum
                    / config.trials_per_corruption as f32,
                mean_final_energy: energy_sum / config.trials_per_corruption as f32,
                mean_iterations: iteration_sum as f32 / config.trials_per_corruption as f32,
            }
        })
        .collect()
}

fn critical_corruption(levels: &[BasinLevelMetrics], threshold: f32) -> f32 {
    levels
        .iter()
        .filter(|level| level.success_rate >= threshold)
        .map(|level| level.corruption_fraction)
        .fold(0.0, f32::max)
}

/// Exactitud directa contra el target. La energía del modelo es simétrica
/// ante un flip global Z₂, pero el cue corrompido conserva la convención de
/// signo mayoritaria, así que una recuperación genuina debe respetarla: un
/// estado completamente invertido se cuenta aquí como fallo.
fn direct_accuracy(state: &[Complex32], target: &[i8]) -> f32 {
    let matches = state
        .iter()
        .zip(target)
        .filter(|(value, expected)| {
            let observed = if value.re >= 0.0 { 1 } else { -1 };
            observed == **expected
        })
        .count();
    matches as f32 / target.len().max(1) as f32
}

/// Diagnóstico invariante ante el flip global Z₂. Nunca alimenta el gate:
/// sólo detecta convergencia al atractor con la convención invertida.
fn gauge_invariant_accuracy(state: &[Complex32], target: &[i8]) -> f32 {
    let direct = direct_accuracy(state, target);
    direct.max(1.0 - direct)
}

fn corrupted_phases(
    target: &[i8],
    corruption_fraction: f32,
    jitter: f32,
    seed: u64,
    trial: usize,
) -> Vec<f32> {
    let mut ranked = (0..target.len())
        .map(|node| {
            (
                splitmix64(seed ^ (trial as u64).rotate_left(29) ^ node as u64),
                node,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let corrupt = ((target.len() as f32 * corruption_fraction).round() as usize)
        .min(target.len().saturating_sub(1));
    let mut flipped = vec![false; target.len()];
    for (_, node) in ranked.into_iter().take(corrupt) {
        flipped[node] = true;
    }
    target
        .iter()
        .enumerate()
        .map(|(node, bit)| {
            let signed_bit = if flipped[node] { -*bit } else { *bit };
            let unit = unit_from_u64(splitmix64(
                seed.rotate_left(31) ^ trial as u64 ^ (node as u64).rotate_left(11),
            ));
            (bit_phase(signed_bit) + jitter * (2.0 * unit - 1.0)).rem_euclid(std::f32::consts::TAU)
        })
        .collect()
}

fn balanced_target(nodes: usize, seed: u64) -> Vec<i8> {
    (0..nodes)
        .map(|node| {
            if splitmix64(seed ^ node as u64) & 1 == 0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

fn bit_phase(bit: i8) -> f32 {
    if bit >= 0 {
        0.0
    } else {
        std::f32::consts::PI
    }
}

fn target_checksum(target: &[i8]) -> u64 {
    target
        .iter()
        .enumerate()
        .fold(0xCBF2_9CE4_8422_2325, |hash, (index, value)| {
            splitmix64(hash ^ index as u64 ^ (*value as i64 as u64))
        })
}

fn empty_failed_report(
    nodes: usize,
    trials_per_corruption: usize,
    target_checksum: u64,
    decision: &'static str,
) -> ConsolidationBasinReport {
    ConsolidationBasinReport {
        nodes,
        trials_per_corruption,
        target_checksum,
        sleep_accepted: 0,
        consolidated_edges: 0,
        pre: Vec::new(),
        post: Vec::new(),
        pre_critical_corruption: 0.0,
        post_critical_corruption: 0.0,
        mean_success_gain: 0.0,
        wall_clock_seconds: 0.0,
        decision,
    }
}

fn sanitize_config(mut config: ConsolidationBasinConfig) -> ConsolidationBasinConfig {
    config.nodes = config.nodes.max(8);
    config.trials_per_corruption = config.trials_per_corruption.max(1);
    config.phase_jitter = config.phase_jitter.clamp(0.0, 0.5);
    config.success_accuracy = config.success_accuracy.clamp(0.5, 1.0);
    config.basin_success_rate = config.basin_success_rate.clamp(0.0, 1.0);
    config.minimum_mean_success_gain = config.minimum_mean_success_gain.clamp(0.0, 1.0);
    config.corruption_fractions = config
        .corruption_fractions
        .into_iter()
        .map(|value| value.clamp(0.0, 0.49))
        .collect();
    config
        .corruption_fractions
        .sort_by(|left, right| left.total_cmp(right));
    config.corruption_fractions.dedup();
    if config.corruption_fractions.is_empty() {
        config.corruption_fractions.push(0.25);
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_accuracy_rejects_a_global_sign_flip() {
        let target = vec![1, -1, 1, -1, 1, -1];
        let flipped = target
            .iter()
            .map(|bit| Complex32::from_polar(1.0, bit_phase(-*bit)))
            .collect::<Vec<_>>();
        assert_eq!(direct_accuracy(&flipped, &target), 0.0);
        assert_eq!(gauge_invariant_accuracy(&flipped, &target), 1.0);
        let aligned = target
            .iter()
            .map(|bit| Complex32::from_polar(1.0, bit_phase(*bit)))
            .collect::<Vec<_>>();
        assert_eq!(direct_accuracy(&aligned, &target), 1.0);
    }

    #[test]
    fn consolidation_expands_the_measured_attractor_basin() {
        let report = run_consolidation_basin_experiment(ConsolidationBasinConfig {
            trials_per_corruption: 12,
            ..ConsolidationBasinConfig::default()
        })
        .unwrap();
        assert_eq!(report.sleep_accepted, 1, "{report:#?}");
        assert!(report.consolidated_edges > 0, "{report:#?}");
        assert_eq!(report.decision, "basin_expansion_pass", "{report:#?}");
    }

    #[test]
    fn basin_expansion_repeats_across_fixed_independent_seeds() {
        for seed in [
            0xA11C_E001,
            0xA11C_E002,
            0xA11C_E003,
            0xA11C_E004,
            0xA11C_E005,
            0xA11C_E006,
            0xA11C_E007,
            0xA11C_E008,
        ] {
            let report = run_consolidation_basin_experiment(ConsolidationBasinConfig {
                trials_per_corruption: 6,
                seed,
                ..ConsolidationBasinConfig::default()
            })
            .unwrap();
            assert_eq!(
                report.decision, "basin_expansion_pass",
                "seed={seed:#x} {report:#?}"
            );
        }
    }
}
