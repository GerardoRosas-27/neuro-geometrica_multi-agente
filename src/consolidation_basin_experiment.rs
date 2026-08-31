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
    /// Desviación típica muestral de la exactitud directa entre ensayos.
    pub std_accuracy: f32,
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

/// Holdout que no comparte semilla, patrón ni techo de corrupción con el
/// fixture de desarrollo. El 100 % post-sueño del patrón inyectado es el
/// techo esperado de escribir un atractor; esta tarea discrimina.
#[derive(Clone, Debug)]
pub struct BasinHoldoutConfig {
    pub nodes: usize,
    pub trials_per_corruption: usize,
    pub corruption_fractions: Vec<f32>,
    pub phase_jitter: f32,
    pub success_accuracy: f32,
    pub graph_noise_fraction: f32,
    pub graph_noise_radians: f32,
    pub seed: u64,
}

impl Default for BasinHoldoutConfig {
    fn default() -> Self {
        Self {
            nodes: 48,
            trials_per_corruption: 8,
            corruption_fractions: vec![0.20, 0.45, 0.55],
            phase_jitter: 0.10,
            success_accuracy: 0.90,
            graph_noise_fraction: 0.15,
            graph_noise_radians: 0.40,
            // Semilla ausente del fixture de desarrollo (0xBA51_CD72_2026 y
            // 0xA11C_E001..=0xA11C_E008).
            seed: 0xC0FF_EE42_2026,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BasinHoldoutReport {
    pub nodes: usize,
    pub seed: u64,
    pub injected: Vec<BasinLevelMetrics>,
    pub injected_mean_success: f32,
    pub injected_std_success: f32,
    pub non_injected: Vec<BasinLevelMetrics>,
    pub non_injected_mean_success: f32,
    pub non_injected_std_success: f32,
    pub wall_clock_seconds: f64,
    pub ceiling_note: &'static str,
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

pub(crate) fn training_engine(
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

pub(crate) fn evaluate_basin(
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
            let n = config.trials_per_corruption as f32;
            let mean_accuracy = accuracy_sum / n;
            let std_accuracy = sample_std(
                results.iter().map(|(_, accuracy, _, _, _)| *accuracy),
                mean_accuracy,
            );
            BasinLevelMetrics {
                corruption_fraction,
                trials: config.trials_per_corruption,
                successes,
                success_rate: successes as f32 / n,
                mean_accuracy,
                std_accuracy,
                mean_gauge_invariant_accuracy: gauge_accuracy_sum / n,
                mean_final_energy: energy_sum / n,
                mean_iterations: iteration_sum as f32 / n,
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
pub(crate) fn direct_accuracy(state: &[Complex32], target: &[i8]) -> f32 {
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

pub(crate) fn corrupted_phases(
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

pub(crate) fn balanced_target(nodes: usize, seed: u64) -> Vec<i8> {
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

pub(crate) fn bit_phase(bit: i8) -> f32 {
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

/// Holdout endurecido: más nodos, semilla nueva, ruido de grafo, corrupción
/// por encima del 40 % y un patrón que nunca se consolidó.
pub fn run_consolidation_basin_holdout(
    config: BasinHoldoutConfig,
) -> Result<BasinHoldoutReport, NativeHybridError> {
    let started = Instant::now();
    let config = sanitize_holdout_config(config);
    let injected = balanced_target(config.nodes, config.seed);
    let non_injected = balanced_target(config.nodes, config.seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut engine = training_engine(config.nodes, config.seed)?;
    let full_experience = injected
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
        return Ok(empty_holdout_report(
            config.nodes,
            config.seed,
            "wake_gate_failed",
        ));
    }
    let sleep = engine.sleep_consolidate()?;
    if sleep.accepted != 1 {
        return Ok(empty_holdout_report(
            config.nodes,
            config.seed,
            "sleep_gate_failed",
        ));
    }
    apply_graph_noise(
        &mut engine.core,
        config.graph_noise_fraction,
        config.graph_noise_radians,
        config.seed ^ 0xA11A_51E0,
    );
    let eval_config = ConsolidationBasinConfig {
        nodes: config.nodes,
        trials_per_corruption: config.trials_per_corruption,
        corruption_fractions: config.corruption_fractions.clone(),
        phase_jitter: config.phase_jitter,
        success_accuracy: config.success_accuracy,
        basin_success_rate: 0.80,
        minimum_mean_success_gain: 0.0,
        seed: config.seed,
    };
    let injected_levels = evaluate_basin(&engine.core, &injected, &eval_config);
    let non_injected_levels = evaluate_basin(&engine.core, &non_injected, &eval_config);
    let (injected_mean_success, injected_std_success) = mean_std_success(&injected_levels);
    let (non_injected_mean_success, non_injected_std_success) =
        mean_std_success(&non_injected_levels);
    let decision = if non_injected_mean_success < 1.0 - 1.0e-6 {
        "holdout_discriminates"
    } else {
        "holdout_still_saturated"
    };
    Ok(BasinHoldoutReport {
        nodes: config.nodes,
        seed: config.seed,
        injected: injected_levels,
        injected_mean_success,
        injected_std_success,
        non_injected: non_injected_levels,
        non_injected_mean_success,
        non_injected_std_success,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        ceiling_note: "techo esperado del atractor escrito; discrimina el patrón no inyectado",
        decision,
    })
}

fn apply_graph_noise(
    core: &mut NativeThermoCdtSubstrate,
    fraction: f32,
    radians: f32,
    seed: u64,
) {
    if core.edge_phase.is_empty() || fraction <= 0.0 || radians <= 0.0 {
        return;
    }
    let mut ranked = (0..core.edge_phase.len())
        .map(|edge| (splitmix64(seed ^ edge as u64), edge))
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let count = ((core.edge_phase.len() as f32 * fraction.clamp(0.0, 1.0)).round() as usize)
        .min(core.edge_phase.len());
    for (_, edge) in ranked.into_iter().take(count) {
        let unit = unit_from_u64(splitmix64(seed.rotate_left(13) ^ edge as u64));
        core.edge_phase[edge] = (core.edge_phase[edge]
            + radians * (2.0 * unit - 1.0))
            .rem_euclid(std::f32::consts::TAU);
    }
}

fn mean_std_success(levels: &[BasinLevelMetrics]) -> (f32, f32) {
    if levels.is_empty() {
        return (0.0, 0.0);
    }
    let mean = levels.iter().map(|level| level.success_rate).sum::<f32>() / levels.len() as f32;
    let std = sample_std(levels.iter().map(|level| level.success_rate), mean);
    (mean, std)
}

fn sample_std(values: impl Iterator<Item = f32> + Clone, mean: f32) -> f32 {
    let count = values.clone().count();
    if count < 2 {
        return 0.0;
    }
    let variance = values.map(|value| (value - mean).powi(2)).sum::<f32>() / (count - 1) as f32;
    variance.max(0.0).sqrt()
}

fn sanitize_holdout_config(mut config: BasinHoldoutConfig) -> BasinHoldoutConfig {
    config.nodes = config.nodes.max(8);
    config.trials_per_corruption = config.trials_per_corruption.max(1);
    config.phase_jitter = config.phase_jitter.clamp(0.0, 0.5);
    config.success_accuracy = config.success_accuracy.clamp(0.5, 1.0);
    config.graph_noise_fraction = config.graph_noise_fraction.clamp(0.0, 1.0);
    config.graph_noise_radians = config.graph_noise_radians.clamp(0.0, std::f32::consts::PI);
    config.corruption_fractions = config
        .corruption_fractions
        .into_iter()
        .map(|value| value.clamp(0.0, 0.70))
        .collect();
    config
        .corruption_fractions
        .sort_by(|left, right| left.total_cmp(right));
    config.corruption_fractions.dedup();
    if config.corruption_fractions.is_empty() {
        config.corruption_fractions.push(0.55);
    }
    config
}

fn empty_holdout_report(nodes: usize, seed: u64, decision: &'static str) -> BasinHoldoutReport {
    BasinHoldoutReport {
        nodes,
        seed,
        injected: Vec::new(),
        injected_mean_success: 0.0,
        injected_std_success: 0.0,
        non_injected: Vec::new(),
        non_injected_mean_success: 0.0,
        non_injected_std_success: 0.0,
        wall_clock_seconds: 0.0,
        ceiling_note: "el holdout no llegó a evaluar cuencas",
        decision,
    }
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


#[derive(Clone, Debug)]
pub struct BasinScaleConfig {
    pub node_counts: Vec<usize>,
    pub trials_per_corruption: usize,
    pub corruption_fractions: Vec<f32>,
    pub seed: u64,
}

impl Default for BasinScaleConfig {
    fn default() -> Self {
        Self {
            node_counts: vec![128, 512, 2048],
            trials_per_corruption: 4,
            corruption_fractions: vec![0.25],
            seed: 0x5CA1_E202_6u64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BasinScaleRow {
    pub nodes: usize,
    pub decision: &'static str,
    pub mean_success_gain: f32,
    pub wall_clock_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BasinScaleReport {
    pub rows: Vec<BasinScaleRow>,
    pub wall_clock_seconds: f64,
}

/// Escalado de cuenca: 128 / 512 / 2048 nodos, no un motor nuevo.
pub fn run_basin_scale_sweep(
    config: BasinScaleConfig,
) -> Result<BasinScaleReport, NativeHybridError> {
    let started = Instant::now();
    let mut rows = Vec::new();
    for nodes in config.node_counts {
        let report = run_consolidation_basin_experiment(ConsolidationBasinConfig {
            nodes,
            trials_per_corruption: config.trials_per_corruption,
            corruption_fractions: config.corruption_fractions.clone(),
            seed: config.seed ^ nodes as u64,
            ..ConsolidationBasinConfig::default()
        })?;
        rows.push(BasinScaleRow {
            nodes,
            decision: report.decision,
            mean_success_gain: report.mean_success_gain,
            wall_clock_seconds: report.wall_clock_seconds,
        });
    }
    Ok(BasinScaleReport {
        rows,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Debug)]
pub struct BoundedForgettingConfig {
    pub nodes: usize,
    pub trials_per_corruption: usize,
    pub corruption_fractions: Vec<f32>,
    pub epsilon: f32,
    pub seed: u64,
}

impl Default for BoundedForgettingConfig {
    fn default() -> Self {
        Self {
            nodes: 32,
            trials_per_corruption: 8,
            corruption_fractions: vec![0.20],
            epsilon: 0.10,
            seed: 0xDE1_7A_2026,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundedForgettingReport {
    pub nodes: usize,
    pub retention_before: f32,
    pub retention_after: f32,
    pub delta_r: f32,
    pub epsilon: f32,
    pub second_sleep_accepted: usize,
    pub wall_clock_seconds: f64,
    pub decision: &'static str,
}

/// ΔR del preprint §4: retención de A después de consolidar B.
pub fn run_bounded_forgetting(
    config: BoundedForgettingConfig,
) -> Result<BoundedForgettingReport, NativeHybridError> {
    let started = Instant::now();
    let first = balanced_target(config.nodes, config.seed);
    let second = balanced_target(config.nodes, config.seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut engine = training_engine(config.nodes, config.seed)?;
    if !consolidate_pattern(&mut engine, &first)? {
        return Ok(BoundedForgettingReport {
            nodes: config.nodes,
            retention_before: 0.0,
            retention_after: 0.0,
            delta_r: 0.0,
            epsilon: config.epsilon,
            second_sleep_accepted: 0,
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            decision: "first_sleep_gate_failed",
        });
    }
    let eval_config = ConsolidationBasinConfig {
        nodes: config.nodes,
        trials_per_corruption: config.trials_per_corruption,
        corruption_fractions: config.corruption_fractions.clone(),
        seed: config.seed,
        ..ConsolidationBasinConfig::default()
    };
    let before = evaluate_basin(&engine.core, &first, &eval_config);
    let retention_before = mean_success(&before);
    let second_sleep_accepted = usize::from(consolidate_pattern(&mut engine, &second)?);
    if second_sleep_accepted != 1 {
        return Ok(BoundedForgettingReport {
            nodes: config.nodes,
            retention_before,
            retention_after: retention_before,
            delta_r: 0.0,
            epsilon: config.epsilon,
            second_sleep_accepted,
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            decision: "second_sleep_gate_failed",
        });
    }
    let after = evaluate_basin(&engine.core, &first, &eval_config);
    let retention_after = mean_success(&after);
    let delta_r = retention_after - retention_before;
    let decision = if delta_r + 1.0e-6 >= -config.epsilon {
        "bounded_forgetting_pass"
    } else {
        "catastrophic_forgetting"
    };
    Ok(BoundedForgettingReport {
        nodes: config.nodes,
        retention_before,
        retention_after,
        delta_r,
        epsilon: config.epsilon,
        second_sleep_accepted,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        decision,
    })
}

fn consolidate_pattern(
    engine: &mut NativeHybridPhasorCdtEngine,
    target: &[i8],
) -> Result<bool, NativeHybridError> {
    let experience = target
        .iter()
        .enumerate()
        .map(|(node, bit)| NativePhasorCue {
            node,
            amplitude: 1.0,
            phase: bit_phase(*bit),
        })
        .collect::<Vec<_>>();
    let wake = engine.infer_and_stage(&experience)?;
    if !wake.gate.passed {
        return Ok(false);
    }
    let sleep = engine.sleep_consolidate()?;
    Ok(sleep.accepted == 1)
}

fn mean_success(levels: &[BasinLevelMetrics]) -> f32 {
    if levels.is_empty() {
        return 0.0;
    }
    levels.iter().map(|level| level.success_rate).sum::<f32>() / levels.len() as f32
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
    fn scientific_basin_expansion_repeats_across_fixed_independent_seeds() {
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

    #[test]
    fn scientific_holdout_non_injected_pattern_is_not_perfect() {
        let report = run_consolidation_basin_holdout(BasinHoldoutConfig::default()).unwrap();
        assert_eq!(report.decision, "holdout_discriminates", "{report:#?}");
        assert!(
            report.non_injected_mean_success < 1.0 - 1.0e-6,
            "el patrón no inyectado no debe saturar: {report:#?}"
        );
    }

    #[test]
    fn scientific_basin_scale_runs_128_nodes() {
        let report = run_basin_scale_sweep(BasinScaleConfig {
            node_counts: vec![128],
            trials_per_corruption: 2,
            corruption_fractions: vec![0.25],
            ..BasinScaleConfig::default()
        })
        .unwrap();
        assert_eq!(report.rows.len(), 1, "{report:#?}");
        assert_eq!(report.rows[0].nodes, 128, "{report:#?}");
    }

    #[test]
    fn scientific_bounded_forgetting_reports_delta_r() {
        let report = run_bounded_forgetting(BoundedForgettingConfig {
            trials_per_corruption: 4,
            ..BoundedForgettingConfig::default()
        })
        .unwrap();
        assert!(report.delta_r.is_finite(), "{report:#?}");
        assert!(
            report.decision == "bounded_forgetting_pass"
                || report.decision == "catastrophic_forgetting"
                || report.decision == "second_sleep_gate_failed"
                || report.decision == "first_sleep_gate_failed",
            "{report:#?}"
        );
    }
}
