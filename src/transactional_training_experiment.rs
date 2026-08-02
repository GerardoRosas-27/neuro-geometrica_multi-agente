//! Entrenamiento comparado según cómo se estructura el tiempo en la inferencia.
//!
//! La hipótesis que se pone a prueba es concreta y falsable: si el colapso es
//! la interferencia entre una evidencia que viene del pasado y una frontera que
//! viene del futuro, entonces entrenar con ambas condiciones de contorno debe
//! dejar en la geometría del CDT una memoria mejor que entrenar sólo con la
//! evidencia. No se afirma nada sobre retrocausalidad física: la "frontera
//! futura" es una post-selección algorítmica sobre el mismo funcional F.
//!
//! Los tres regímenes comparados son:
//!
//! - [`TemporalBoundaryMode::ForwardOnly`]: la cue sólo fija el estado inicial.
//!   No hay término de frontera, así que tampoco hay Handshake posible.
//! - [`TemporalBoundaryMode::PresentBoundary`]: la cue es además la frontera.
//!   Pasado y futuro coinciden; el "apretón de manos" es consigo mismo.
//! - [`TemporalBoundaryMode::TwoStateVector`]: la evidencia fija el pasado y un
//!   conjunto disjunto de nodos post-selecciona el futuro. Es la única
//!   estructura con dos vectores de estado genuinamente distintos.
//!
//! La evaluación es idéntica en los tres: cue parcial y corrompida, sin meta y
//! sin estímulo, sobre un motor reconstruido desde el CDT entrenado. Así se
//! mide lo que quedó grabado en la geometría, no el estado fasorial residual.

use crate::native_hybrid_phasor_cdt_engine::{
    NativeHybridConfig, NativeHybridError, NativeHybridPhasorCdtEngine, NativePhasorCue,
};
use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorInferencePolicy, NativePhasorMinimizerConfig,
    NativePhasorThermodynamicEngine,
};
use crate::native_rng::{splitmix64, unit_from_u64};
use crate::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum TemporalBoundaryMode {
    ForwardOnly,
    PresentBoundary,
    TwoStateVector,
}

impl TemporalBoundaryMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ForwardOnly => "solo_pasado",
            Self::PresentBoundary => "presente_como_frontera",
            Self::TwoStateVector => "dos_vectores_de_estado",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransactionalTrainingConfig {
    pub nodes: usize,
    pub patterns: usize,
    pub epochs: usize,
    /// Fracción de nodos revelados como evidencia hacia adelante.
    pub cue_fraction: f32,
    /// Fracción de nodos, disjunta de la anterior, usada como meta futura.
    pub goal_fraction: f32,
    /// Fracción de la evidencia que llega con el bit invertido.
    pub corruption: f32,
    pub phase_jitter: f32,
    pub evaluation_trials: usize,
    /// Exactitud directa a partir de la cual una recuperación cuenta.
    pub success_accuracy: f32,
    pub seed: u64,
}

impl Default for TransactionalTrainingConfig {
    fn default() -> Self {
        Self {
            nodes: 256,
            // Un solo patrón: las fases de arista son el único sustrato de
            // memoria y con grado 6 sobre 256 nodos no hay capacidad para
            // varios patrones balanceados. Lo que se compara aquí es la
            // estructura temporal de la inferencia, no la capacidad.
            patterns: 1,
            // Cada época presenta una máscara distinta. La geometría sólo
            // cristaliza donde el episodio estuvo anclado, así que hace falta
            // que la unión de máscaras llegue a cubrir el grafo.
            epochs: 12,
            cue_fraction: 0.35,
            goal_fraction: 0.25,
            corruption: 0.10,
            phase_jitter: 0.25,
            evaluation_trials: 12,
            success_accuracy: 0.95,
            seed: 0x7C5F_2026,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TransactionalTrainingReport {
    pub mode: TemporalBoundaryMode,
    pub modulators: &'static str,
    pub wake_cycles: usize,
    pub gate_passed: usize,
    pub sleep_accepted: usize,
    pub rejected_by_efficiency: usize,
    pub memory_size: usize,
    /// Φ medio durante el entrenamiento: interferencia pasado/futuro.
    pub mean_integrated_information: f32,
    /// Acuerdo medio entre evidencia hacia adelante y frontera hacia atrás.
    pub mean_handshake_coherence: f32,
    pub attention_ignitions: usize,
    pub train_seconds: f32,
    pub eval_accuracy: f32,
    pub eval_gauge_invariant_accuracy: f32,
    pub eval_success_rate: f32,
    pub eval_mean_iterations: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TransactionalComparisonReport {
    pub config_nodes: usize,
    pub patterns: usize,
    pub epochs: usize,
    pub runs: Vec<TransactionalTrainingReport>,
    pub decision: &'static str,
}

/// Ejecuta los tres regímenes temporales con y sin el ciclo Handshake+atención.
pub fn run_transactional_training_comparison(
    config: TransactionalTrainingConfig,
) -> Result<TransactionalComparisonReport, NativeHybridError> {
    let config = sanitize(config);
    let mut runs = Vec::new();
    for mode in [
        TemporalBoundaryMode::ForwardOnly,
        TemporalBoundaryMode::PresentBoundary,
        TemporalBoundaryMode::TwoStateVector,
    ] {
        for modulated in [false, true] {
            runs.push(train_and_evaluate(config, mode, modulated)?);
        }
    }

    let best = runs
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.eval_accuracy
                .partial_cmp(&right.eval_accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    let decision = match runs[best].mode {
        TemporalBoundaryMode::TwoStateVector => "dos_vectores_de_estado_gana",
        TemporalBoundaryMode::PresentBoundary => "frontera_presente_gana",
        TemporalBoundaryMode::ForwardOnly => "solo_pasado_gana",
    };

    Ok(TransactionalComparisonReport {
        config_nodes: config.nodes,
        patterns: config.patterns,
        epochs: config.epochs,
        runs,
        decision,
    })
}

fn train_and_evaluate(
    config: TransactionalTrainingConfig,
    mode: TemporalBoundaryMode,
    modulated: bool,
) -> Result<TransactionalTrainingReport, NativeHybridError> {
    let patterns = (0..config.patterns)
        .map(|index| balanced_pattern(config.nodes, config.seed ^ (index as u64).rotate_left(13)))
        .collect::<Vec<_>>();
    let mut engine = training_engine(config, mode, modulated)?;

    let started = Instant::now();
    let mut wake_cycles = 0;
    let mut gate_passed = 0;
    let mut sleep_accepted = 0;
    let mut rejected_by_efficiency = 0;
    let mut phi_sum = 0.0f64;
    let mut coherence_sum = 0.0f64;
    let mut attention_ignitions = 0;

    for epoch in 0..config.epochs {
        for (index, pattern) in patterns.iter().enumerate() {
            let episode = config
                .seed
                .rotate_left(7)
                ^ (epoch as u64).wrapping_mul(0x9E37_79B9)
                ^ (index as u64).rotate_left(41);
            let (cue, goal) = episode_boundaries(pattern, &config, mode, episode);
            let wake = engine.infer_and_stage_with_goal(&cue, &goal)?;
            wake_cycles += 1;
            gate_passed += usize::from(wake.gate.passed);
            phi_sum += f64::from(wake.minimization.mean_integrated_information);
            coherence_sum += f64::from(wake.minimization.mean_handshake_coherence);
            attention_ignitions += wake.minimization.attention_ignitions;
        }
        let sleep = engine.sleep_consolidate()?;
        sleep_accepted += sleep.accepted;
        rejected_by_efficiency += sleep.rejected_by_efficiency;
    }
    let train_seconds = started.elapsed().as_secs_f32();

    let evaluation = evaluate(&engine.core, &patterns, &config);

    Ok(TransactionalTrainingReport {
        mode,
        modulators: if modulated { "hibrido" } else { "armijo" },
        wake_cycles,
        gate_passed,
        sleep_accepted,
        rejected_by_efficiency,
        memory_size: engine.attractors().len(),
        mean_integrated_information: (phi_sum / wake_cycles.max(1) as f64) as f32,
        mean_handshake_coherence: (coherence_sum / wake_cycles.max(1) as f64) as f32,
        attention_ignitions,
        train_seconds,
        eval_accuracy: evaluation.accuracy,
        eval_gauge_invariant_accuracy: evaluation.gauge_invariant_accuracy,
        eval_success_rate: evaluation.success_rate,
        eval_mean_iterations: evaluation.mean_iterations,
    })
}

struct Evaluation {
    accuracy: f32,
    gauge_invariant_accuracy: f32,
    success_rate: f32,
    mean_iterations: f32,
}

/// Recuperación desde el CDT entrenado. Sin meta, sin estímulo y con la misma
/// configuración en los seis regímenes: lo único que cambia entre ellos es la
/// geometría que dejó el entrenamiento.
fn evaluate(
    core: &NativeThermoCdtSubstrate,
    patterns: &[Vec<i8>],
    config: &TransactionalTrainingConfig,
) -> Evaluation {
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
    .expect("el core entrenado por el experimento debe seguir siendo válido");

    let mut accuracy_sum = 0.0;
    let mut gauge_sum = 0.0;
    let mut successes = 0usize;
    let mut iteration_sum = 0usize;
    let mut trials = 0usize;

    for (index, pattern) in patterns.iter().enumerate() {
        for trial in 0..config.evaluation_trials {
            // Flujo de azar separado del entrenamiento: las máscaras de
            // evaluación nunca se vieron durante el aprendizaje.
            let probe = config.seed.rotate_left(29)
                ^ (index as u64).wrapping_mul(0xD1B5_4A32)
                ^ (trial as u64).rotate_left(19);
            let mut inference = template.clone();
            let revealed = mask(config.nodes, config.cue_fraction, probe);
            for (node, phasor) in inference.phasors.iter_mut().enumerate() {
                let phase = if revealed[node] {
                    let flipped = unit_from_u64(splitmix64(probe ^ (node as u64).rotate_left(3)))
                        < config.corruption;
                    let bit = if flipped { -pattern[node] } else { pattern[node] };
                    bit_phase(bit)
                        + config.phase_jitter
                            * (2.0 * unit_from_u64(splitmix64(probe ^ node as u64)) - 1.0)
                } else {
                    std::f32::consts::TAU
                        * unit_from_u64(splitmix64(probe.rotate_left(11) ^ node as u64))
                };
                *phasor = Complex32::from_polar(1.0, phase);
            }
            let result = inference.minimize_free_energy(NativePhasorMinimizerConfig {
                max_iterations: 400,
                residual_tolerance: 5.0e-3,
                topological_warm_start: false,
                ..NativePhasorMinimizerConfig::default()
            });
            let accuracy = direct_accuracy(&inference.phasors, pattern);
            accuracy_sum += accuracy;
            gauge_sum += gauge_invariant_accuracy(&inference.phasors, pattern);
            successes += usize::from(accuracy >= config.success_accuracy);
            iteration_sum += result.iterations;
            trials += 1;
        }
    }

    let trials = trials.max(1) as f32;
    Evaluation {
        accuracy: accuracy_sum / trials,
        gauge_invariant_accuracy: gauge_sum / trials,
        success_rate: successes as f32 / trials,
        mean_iterations: iteration_sum as f32 / trials,
    }
}

fn training_engine(
    config: TransactionalTrainingConfig,
    mode: TemporalBoundaryMode,
    modulated: bool,
) -> Result<NativeHybridPhasorCdtEngine, NativeHybridError> {
    let mut core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 1,
        nodes_per_slice: config.nodes,
        spatial_degree: 6,
        temporal_degree: 1,
        temperature: 0.0,
        diffusion: 0.0,
        pilot_gain: 0.0,
        amplitude_decay: 0.0,
        seed: config.seed,
        ..NativeThermoCdtConfig::default()
    });
    // Geometría de partida sin frustración: el prior es "todo en fase" y cada
    // patrón tiene que ganárselo doblando las fases de arista. Con las fases
    // aleatorias del sustrato el descenso arranca en un vidrio de espín y
    // ningún régimen llega a converger.
    core.edge_phase.fill(0.0);
    NativeHybridPhasorCdtEngine::from_core(
        core,
        // El mismo funcional F en los seis regímenes. Lo único que cambia es
        // qué nodos del estímulo se rellenan, nunca cuánto pesa el estímulo.
        // Acoplamiento real durante wake para que la inferencia propague a los
        // nodos que ninguna frontera cubre, pero por debajo del estímulo: con
        // el prior plano el acoplamiento empuja a fase uniforme, y si domina
        // borra la evidencia antes de que pueda consolidarse nada.
        NativePhasorConfig {
            coupling_strength: 0.5,
            radial_strength: 4.0,
            target_amplitude: 1.0,
            confinement: 0.0,
            stimulus_gain: 4.0,
            stimulus_decay: 1.0,
            entropy_weight: 0.0,
            temperature_scale: 0.0,
            noise_scale: 0.0,
            ..NativePhasorConfig::default()
        },
        NativeHybridConfig {
            minimizer: NativePhasorMinimizerConfig {
                max_iterations: 200,
                residual_tolerance: 5.0e-3,
                topological_warm_start: false,
                handshake_strength: if modulated { 0.65 } else { 0.0 },
                attention_strength: if modulated { 0.55 } else { 0.0 },
                attention_temperature: 0.75,
                attention_max_gain: 3.0,
                // La ignición tiene que ser un evento selectivo. Con la
                // saliencia medida contra la media, Φ vive en torno a 1e-2
                // durante el descenso: un umbral por debajo de eso enciende
                // el foco en casi toda iteración, y una atención permanente
                // amplifica correcciones justo en la región que la inferencia
                // no puede determinar.
                attention_ignition_threshold: 0.02,
                handshake_max_gain: 3.0,
                inference_policy: if modulated {
                    NativePhasorInferencePolicy::Adaptive
                } else {
                    NativePhasorInferencePolicy::Fixed
                },
                ..NativePhasorMinimizerConfig::default()
            },
            consolidation_learning_rate: 0.8,
            minimum_relative_energy_drop: 0.0,
            maximum_residual: 5.0e-3,
            minimum_magnetic_coherence: -1.0,
            minimum_stability: 0.90,
            stability_probe_jitter: 0.02,
            // Sleep revalida sin frontera, así que los regímenes con frontera
            // arrancan lejos de su mínimo y necesitan presupuesto real para
            // que la comparación no los descarte por falta de iteraciones.
            stability_probe_iterations: 150,
            // La dinámica propia del CDT reescribiría las fases recién
            // grabadas antes de poder evaluarlas.
            cdt_consolidation_steps: 0,
            // Escalera de tres peldaños: sin frontera, sólo la del pasado, y
            // pasado más futuro. Así el tercer régimen añade exactamente una
            // cosa sobre el segundo: la post-selección.
            cue_as_boundary: mode != TemporalBoundaryMode::ForwardOnly,
            // Eco débil del episodio durante el replay. Sin él, ningún patrón
            // que el prior no soporte ya puede consolidarse nunca y los tres
            // regímenes quedan empatados en cero memorias.
            sleep_replay_boundary_gain: 0.25,
            anchored_consolidation: true,
            ..NativeHybridConfig::default()
        },
    )
}

/// Evidencia y meta de un episodio. Los dos conjuntos son disjuntos: la meta
/// nunca revela un nodo que la evidencia ya haya fijado, así que la frontera
/// futura aporta información que el estado inicial no contiene.
fn episode_boundaries(
    pattern: &[i8],
    config: &TransactionalTrainingConfig,
    mode: TemporalBoundaryMode,
    episode: u64,
) -> (Vec<NativePhasorCue>, Vec<NativePhasorCue>) {
    let revealed = mask(config.nodes, config.cue_fraction, episode);
    let mut cue = Vec::new();
    let mut goal = Vec::new();
    for node in 0..config.nodes {
        if revealed[node] {
            let flipped =
                unit_from_u64(splitmix64(episode ^ (node as u64).rotate_left(3))) < config.corruption;
            let bit = if flipped { -pattern[node] } else { pattern[node] };
            let jitter = config.phase_jitter
                * (2.0 * unit_from_u64(splitmix64(episode ^ node as u64)) - 1.0);
            cue.push(NativePhasorCue {
                node,
                amplitude: 1.0,
                phase: bit_phase(bit) + jitter,
            });
        } else if mode == TemporalBoundaryMode::TwoStateVector
            && unit_from_u64(splitmix64(episode.rotate_left(37) ^ node as u64))
                < config.goal_fraction
        {
            goal.push(NativePhasorCue {
                node,
                amplitude: 1.0,
                phase: bit_phase(pattern[node]),
            });
        }
    }
    if cue.is_empty() {
        cue.push(NativePhasorCue {
            node: 0,
            amplitude: 1.0,
            phase: bit_phase(pattern[0]),
        });
    }
    (cue, goal)
}

fn mask(nodes: usize, fraction: f32, seed: u64) -> Vec<bool> {
    (0..nodes)
        .map(|node| unit_from_u64(splitmix64(seed.rotate_left(53) ^ node as u64)) < fraction)
        .collect()
}

/// Patrón balanceado: mitad de bits en cada signo, para que un estado trivial
/// uniforme no pueda puntuar alto por accidente.
fn balanced_pattern(nodes: usize, seed: u64) -> Vec<i8> {
    let mut pattern = (0..nodes)
        .map(|node| if node * 2 < nodes { 1i8 } else { -1 })
        .collect::<Vec<_>>();
    for index in (1..nodes).rev() {
        let swap = (splitmix64(seed ^ index as u64) % (index as u64 + 1)) as usize;
        pattern.swap(index, swap);
    }
    pattern
}

fn bit_phase(bit: i8) -> f32 {
    if bit >= 0 {
        0.0
    } else {
        std::f32::consts::PI
    }
}

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

fn gauge_invariant_accuracy(state: &[Complex32], target: &[i8]) -> f32 {
    let direct = direct_accuracy(state, target);
    direct.max(1.0 - direct)
}

fn sanitize(config: TransactionalTrainingConfig) -> TransactionalTrainingConfig {
    TransactionalTrainingConfig {
        nodes: config.nodes.max(8),
        patterns: config.patterns.max(1),
        epochs: config.epochs.max(1),
        cue_fraction: config.cue_fraction.clamp(0.05, 0.95),
        goal_fraction: config.goal_fraction.clamp(0.0, 0.95),
        corruption: config.corruption.clamp(0.0, 0.5),
        phase_jitter: config.phase_jitter.max(0.0),
        evaluation_trials: config.evaluation_trials.max(1),
        success_accuracy: config.success_accuracy.clamp(0.5, 1.0),
        ..config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_pattern_has_equal_sign_counts() {
        let pattern = balanced_pattern(128, 0x1234);
        let positive = pattern.iter().filter(|bit| **bit > 0).count();
        assert_eq!(positive, 64);
    }

    #[test]
    fn only_the_two_state_regime_supplies_a_disjoint_future_boundary() {
        let config = sanitize(TransactionalTrainingConfig::default());
        let pattern = balanced_pattern(config.nodes, 7);
        for mode in [
            TemporalBoundaryMode::ForwardOnly,
            TemporalBoundaryMode::PresentBoundary,
            TemporalBoundaryMode::TwoStateVector,
        ] {
            let (cue, goal) = episode_boundaries(&pattern, &config, mode, 42);
            assert!(!cue.is_empty());
            match mode {
                TemporalBoundaryMode::TwoStateVector => {
                    assert!(!goal.is_empty());
                    // La meta nunca repite un nodo que la evidencia ya fijó.
                    for item in &goal {
                        assert!(cue.iter().all(|other| other.node != item.node));
                    }
                }
                _ => assert!(goal.is_empty()),
            }
        }
    }

    #[test]
    fn recall_improves_monotonically_with_the_temporal_structure() {
        let report = run_transactional_training_comparison(TransactionalTrainingConfig {
            nodes: 128,
            epochs: 8,
            evaluation_trials: 6,
            ..TransactionalTrainingConfig::default()
        })
        .unwrap();
        assert_eq!(report.runs.len(), 6);

        let accuracy = |mode: TemporalBoundaryMode| {
            report
                .runs
                .iter()
                .find(|run| run.mode == mode && run.modulators == "armijo")
                .map(|run| run.eval_accuracy)
                .unwrap()
        };
        let forward = accuracy(TemporalBoundaryMode::ForwardOnly);
        let present = accuracy(TemporalBoundaryMode::PresentBoundary);
        let two_state = accuracy(TemporalBoundaryMode::TwoStateVector);

        // Cada condición de contorno adicional debe dejar mejor memoria: la
        // evidencia que sólo inicializa se borra, la que ancla persiste, y la
        // post-selección añade información que la evidencia no contiene.
        assert!(present > forward, "{report:#?}");
        assert!(two_state > present, "{report:#?}");
        assert_eq!(report.decision, "dos_vectores_de_estado_gana", "{report:#?}");
    }
}
