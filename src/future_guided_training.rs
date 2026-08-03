//! Entrenamiento por futuros propuestos y postselección variacional.
//!
//! Este módulo separa dos afirmaciones que suelen mezclarse:
//!
//! 1. un modelo generativo puede proponer futuros plausibles desde una entrada;
//! 2. una frontera futura útil mejora lo que la geometría CDT consolida.
//!
//! Para validar la segunda sin depender de que exista un GGUF durante los tests,
//! [`SyntheticGemmaPrior`] actúa como un prior preentrenado controlable. Conoce
//! un vocabulario de prototipos, pero durante inferencia sólo recibe la cue
//! parcial: identifica candidatos por compatibilidad con la evidencia y produce
//! futuros correctos y distractores. Nunca recibe el índice ni el target del
//! episodio. La misma interfaz puede ser implementada por Gemma 2.
//!
//! El protocolo compara, con mismo core, estado, semillas y presupuesto máximo:
//!
//! - [`FutureTrainingMode::Interference`]: cue como condición inicial y Armijo;
//! - [`FutureTrainingMode::PredictedFutureArmijo`]: futuros + postselección por F;
//! - [`FutureTrainingMode::PredictedFutureAdaptive`]: lo anterior más Handshake
//!   y atención adaptativa.
//!
//! Cada propuesta se evalúa en un clon del mismo estado. Sólo el clon con menor
//! energía libre normalizada sobrevive. Después, el gate de wake, la
//! revalidación de sleep y `ΔF_store` deciden si la experiencia se consolida.

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
pub enum FutureTrainingMode {
    Interference,
    PredictedFutureArmijo,
    PredictedFutureAdaptive,
}

impl FutureTrainingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Interference => "interferencia",
            Self::PredictedFutureArmijo => "futuro_armijo",
            Self::PredictedFutureAdaptive => "futuro_handshake_atencion",
        }
    }

    fn uses_future(self) -> bool {
        self != Self::Interference
    }

    fn uses_modulators(self) -> bool {
        self == Self::PredictedFutureAdaptive
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FutureProposal {
    pub goal: Vec<NativePhasorCue>,
    pub confidence: f32,
    /// Identificador opaco del candidato; sólo se usa para métricas sintéticas.
    pub latent_id: usize,
}

pub trait FutureProposalGenerator {
    fn propose(
        &mut self,
        cue: &[NativePhasorCue],
        node_count: usize,
        count: usize,
        seed: u64,
    ) -> Vec<FutureProposal>;
}

/// Entrenador operativo para una secuencia de entradas sin etiqueta.
///
/// El generador recibe únicamente la cue. El entrenador prueba sus futuros en
/// copias del mismo estado, conserva el mínimo de F y sólo entonces ejecuta
/// sleep. Por tanto, ni Gemma ni el selector escriben directamente en CDT.
#[derive(Clone, Debug)]
pub struct FutureGuidedTrainer<G> {
    pub engine: NativeHybridPhasorCdtEngine,
    pub generator: G,
    pub proposals_per_input: usize,
    episode: u64,
}

#[derive(Clone, Debug)]
pub struct FutureLearningEpisode {
    pub episode: u64,
    pub proposals_generated: usize,
    pub selected_latent_id: usize,
    pub selected_confidence: f32,
    pub selected_free_energy: f32,
    pub final_residual: f32,
    pub phase_coherence: f32,
    pub relative_energy_drop: f32,
    pub energy_evaluations: usize,
    pub gate_passed: bool,
    pub consolidated: usize,
    pub rejected_by_efficiency: usize,
    pub attention_ignitions: usize,
    pub handshake_coherence: f32,
    pub integrated_information: f32,
}

impl<G: FutureProposalGenerator> FutureGuidedTrainer<G> {
    pub fn new(
        engine: NativeHybridPhasorCdtEngine,
        generator: G,
        proposals_per_input: usize,
    ) -> Self {
        Self {
            engine,
            generator,
            proposals_per_input: proposals_per_input.max(1),
            episode: 0,
        }
    }

    pub fn learn_from_input(
        &mut self,
        cue: &[NativePhasorCue],
        seed: u64,
    ) -> Result<Option<FutureLearningEpisode>, NativeHybridError> {
        self.episode = self.episode.wrapping_add(1);
        let proposals = self.generator.propose(
            cue,
            self.engine.core.node_count(),
            self.proposals_per_input,
            seed,
        );
        let Some((selected, wake, latent_id, evaluations)) =
            select_future(&self.engine, cue, &proposals)?
        else {
            return Ok(None);
        };
        let selected_confidence = proposals
            .iter()
            .find(|item| item.latent_id == latent_id)
            .map_or(0.0, |item| item.confidence);
        let selected_free_energy = wake.minimization.final_report.free_energy;
        let final_residual = wake.minimization.final_report.gradient_residual;
        let phase_coherence = wake.minimization.final_report.phase_coherence;
        let relative_energy_drop = wake.relative_energy_drop;
        let gate_passed = wake.gate.passed;
        let attention_ignitions = wake.minimization.attention_ignitions;
        let handshake_coherence = wake.minimization.mean_handshake_coherence;
        let integrated_information = wake.minimization.mean_integrated_information;
        self.engine = selected;
        let sleep = self.engine.sleep_consolidate()?;
        Ok(Some(FutureLearningEpisode {
            episode: self.episode,
            proposals_generated: proposals.len(),
            selected_latent_id: latent_id,
            selected_confidence,
            selected_free_energy,
            final_residual,
            phase_coherence,
            relative_energy_drop,
            energy_evaluations: evaluations,
            gate_passed,
            consolidated: sleep.accepted,
            rejected_by_efficiency: sleep.rejected_by_efficiency,
            attention_ignitions,
            handshake_coherence,
            integrated_information,
        }))
    }
}

/// Sustituto controlado de un LLM preentrenado para validar el mecanismo.
///
/// El codebook representa conocimiento previo del generador. La consulta sólo
/// compara la cue observada con esos prototipos; no recibe el target activo.
#[derive(Clone, Debug)]
pub struct SyntheticGemmaPrior {
    prototypes: Vec<Vec<i8>>,
    goal_fraction: f32,
    proposal_noise: f32,
}

impl SyntheticGemmaPrior {
    pub fn new(prototypes: Vec<Vec<i8>>, goal_fraction: f32, proposal_noise: f32) -> Self {
        Self {
            prototypes,
            goal_fraction: goal_fraction.clamp(0.05, 0.95),
            proposal_noise: proposal_noise.clamp(0.0, 0.5),
        }
    }
}

impl FutureProposalGenerator for SyntheticGemmaPrior {
    fn propose(
        &mut self,
        cue: &[NativePhasorCue],
        node_count: usize,
        count: usize,
        seed: u64,
    ) -> Vec<FutureProposal> {
        let mut observed = vec![None; node_count];
        for item in cue {
            if item.node < node_count {
                observed[item.node] = Some(if item.phase.cos() >= 0.0 { 1i8 } else { -1 });
            }
        }

        let mut ranked = self
            .prototypes
            .iter()
            .enumerate()
            .filter(|(_, prototype)| prototype.len() == node_count)
            .map(|(index, prototype)| {
                let mut matches = 0usize;
                let mut compared = 0usize;
                for (expected, value) in prototype.iter().zip(&observed) {
                    if let Some(value) = value {
                        matches += usize::from(*expected == *value);
                        compared += 1;
                    }
                }
                let confidence = matches as f32 / compared.max(1) as f32;
                (index, confidence)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        ranked
            .into_iter()
            .take(count.max(1))
            .enumerate()
            .map(|(rank, (latent_id, confidence))| {
                let prototype = &self.prototypes[latent_id];
                let mut goal = Vec::new();
                for node in 0..node_count {
                    if observed[node].is_some() {
                        continue;
                    }
                    let node_seed = splitmix64(
                        seed ^ (rank as u64).rotate_left(11) ^ (node as u64).rotate_left(37),
                    );
                    if unit_from_u64(node_seed) >= self.goal_fraction {
                        continue;
                    }
                    let flipped =
                        unit_from_u64(splitmix64(node_seed.rotate_left(23))) < self.proposal_noise;
                    let bit = if flipped {
                        -prototype[node]
                    } else {
                        prototype[node]
                    };
                    goal.push(NativePhasorCue {
                        node,
                        amplitude: confidence.max(0.25),
                        phase: bit_phase(bit),
                    });
                }
                FutureProposal {
                    goal,
                    confidence,
                    latent_id,
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FutureGuidedTrainingConfig {
    pub nodes: usize,
    pub synthetic_tasks: usize,
    pub epochs: usize,
    pub proposals_per_input: usize,
    pub cue_fraction: f32,
    pub cue_corruption: f32,
    pub goal_fraction: f32,
    pub proposal_noise: f32,
    pub phase_jitter: f32,
    pub candidate_iterations: usize,
    pub evaluation_trials: usize,
    pub success_accuracy: f32,
    pub seed: u64,
}

impl Default for FutureGuidedTrainingConfig {
    fn default() -> Self {
        Self {
            nodes: 128,
            synthetic_tasks: 8,
            epochs: 16,
            proposals_per_input: 4,
            cue_fraction: 0.30,
            cue_corruption: 0.08,
            goal_fraction: 0.30,
            proposal_noise: 0.04,
            phase_jitter: 0.20,
            candidate_iterations: 64,
            evaluation_trials: 8,
            success_accuracy: 0.90,
            seed: 0xF077_2026,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FutureGuidedTrainingReport {
    pub mode: String,
    pub tasks: usize,
    pub wake_cycles: usize,
    pub generated_proposals: usize,
    pub correct_top1_futures: usize,
    pub selected_correct_futures: usize,
    pub gate_passed: usize,
    pub consolidated: usize,
    pub rejected_by_efficiency: usize,
    pub energy_evaluations: usize,
    pub attention_ignitions: usize,
    pub mean_handshake_coherence: f32,
    pub mean_phi: f32,
    pub train_seconds: f32,
    pub recall_accuracy: f32,
    pub recall_success_rate: f32,
    pub recall_iterations: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FutureGuidedComparisonReport {
    pub config: FutureGuidedTrainingConfigSnapshot,
    pub reports: Vec<FutureGuidedTrainingReport>,
    pub winner: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FutureGuidedTrainingConfigSnapshot {
    pub nodes: usize,
    pub tasks: usize,
    pub epochs: usize,
    pub proposals: usize,
}

pub fn run_synthetic_future_guided_comparison(
    config: FutureGuidedTrainingConfig,
) -> Result<FutureGuidedComparisonReport, NativeHybridError> {
    let config = sanitize(config);
    let prototypes = (0..config.synthetic_tasks)
        .map(|task| balanced_pattern(config.nodes, config.seed ^ (task as u64).rotate_left(17)))
        .collect::<Vec<_>>();
    let mut reports = Vec::new();
    for mode in [
        FutureTrainingMode::Interference,
        FutureTrainingMode::PredictedFutureArmijo,
        FutureTrainingMode::PredictedFutureAdaptive,
    ] {
        let mut generator = SyntheticGemmaPrior::new(
            prototypes.clone(),
            config.goal_fraction,
            config.proposal_noise,
        );
        reports.push(train_mode(config, &prototypes, mode, &mut generator)?);
    }
    let winner = reports
        .iter()
        .max_by(|left, right| {
            left.recall_accuracy
                .total_cmp(&right.recall_accuracy)
                .then_with(|| right.energy_evaluations.cmp(&left.energy_evaluations))
        })
        .map(|report| report.mode.clone())
        .unwrap_or_else(|| "sin_resultado".to_string());
    Ok(FutureGuidedComparisonReport {
        config: FutureGuidedTrainingConfigSnapshot {
            nodes: config.nodes,
            tasks: config.synthetic_tasks,
            epochs: config.epochs,
            proposals: config.proposals_per_input,
        },
        reports,
        winner,
    })
}

fn train_mode(
    config: FutureGuidedTrainingConfig,
    prototypes: &[Vec<i8>],
    mode: FutureTrainingMode,
    generator: &mut impl FutureProposalGenerator,
) -> Result<FutureGuidedTrainingReport, NativeHybridError> {
    let started = Instant::now();
    let mut report = FutureGuidedTrainingReport {
        mode: mode.label().to_string(),
        tasks: prototypes.len(),
        ..FutureGuidedTrainingReport::default()
    };
    let mut recall_accuracy = 0.0f64;
    let mut recall_successes = 0usize;
    let mut recall_iterations = 0usize;
    let mut recall_trials = 0usize;
    let mut handshake_sum = 0.0f64;
    let mut phi_sum = 0.0f64;

    for (task, target) in prototypes.iter().enumerate() {
        let task_seed = config.seed ^ (task as u64).rotate_left(29);
        let mut engine = future_training_engine(config, mode, task_seed)?;
        for epoch in 0..config.epochs {
            let episode_seed = task_seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9);
            let cue = partial_cue(target, &config, episode_seed);
            if mode.uses_future() {
                let proposals = generator.propose(
                    &cue,
                    config.nodes,
                    config.proposals_per_input,
                    episode_seed.rotate_left(7),
                );
                report.generated_proposals += proposals.len();
                report.correct_top1_futures +=
                    usize::from(proposals.first().is_some_and(|item| item.latent_id == task));
                if let Some((selected, wake, selected_id, evaluations)) =
                    select_future(&engine, &cue, &proposals)?
                {
                    engine = selected;
                    report.selected_correct_futures += usize::from(selected_id == task);
                    accumulate_wake(
                        &mut report,
                        &mut handshake_sum,
                        &mut phi_sum,
                        &wake,
                        evaluations,
                    );
                }
            } else {
                let wake = engine.infer_and_stage(&cue)?;
                let evaluations = wake.minimization.energy_evaluations;
                accumulate_wake(
                    &mut report,
                    &mut handshake_sum,
                    &mut phi_sum,
                    &wake,
                    evaluations,
                );
            }
            let sleep = engine.sleep_consolidate()?;
            report.consolidated += sleep.accepted;
            report.rejected_by_efficiency += sleep.rejected_by_efficiency;
        }

        let evaluation = evaluate_core(&engine.core, target, &config, task_seed.rotate_left(43));
        recall_accuracy += f64::from(evaluation.accuracy_sum);
        recall_successes += evaluation.successes;
        recall_iterations += evaluation.iterations;
        recall_trials += evaluation.trials;
    }

    let wake = report.wake_cycles.max(1) as f64;
    let recall = recall_trials.max(1) as f64;
    report.mean_handshake_coherence = (handshake_sum / wake) as f32;
    report.mean_phi = (phi_sum / wake) as f32;
    report.train_seconds = started.elapsed().as_secs_f32();
    report.recall_accuracy = (recall_accuracy / recall) as f32;
    report.recall_success_rate = recall_successes as f32 / recall as f32;
    report.recall_iterations = recall_iterations as f32 / recall as f32;
    Ok(report)
}

fn select_future(
    engine: &NativeHybridPhasorCdtEngine,
    cue: &[NativePhasorCue],
    proposals: &[FutureProposal],
) -> Result<
    Option<(
        NativeHybridPhasorCdtEngine,
        crate::native_hybrid_phasor_cdt_engine::NativeWakeInferenceReport,
        usize,
        usize,
    )>,
    NativeHybridError,
> {
    let mut best = None;
    let mut total_evaluations = 0usize;
    for proposal in proposals {
        if proposal.goal.is_empty() {
            continue;
        }
        let mut candidate = engine.clone();
        let wake = candidate.infer_and_stage_with_goal(cue, &proposal.goal)?;
        total_evaluations += wake.minimization.energy_evaluations;
        let score = wake.minimization.final_report.free_energy
            / wake.minimization.final_report.nodes.max(1) as f32;
        let replace = best
            .as_ref()
            .is_none_or(|(_, _, _, best_score): &(_, _, _, f32)| score < *best_score);
        if replace {
            best = Some((candidate, wake, proposal.latent_id, score));
        }
    }
    Ok(best.map(|(candidate, wake, id, _)| (candidate, wake, id, total_evaluations)))
}

fn accumulate_wake(
    report: &mut FutureGuidedTrainingReport,
    handshake_sum: &mut f64,
    phi_sum: &mut f64,
    wake: &crate::native_hybrid_phasor_cdt_engine::NativeWakeInferenceReport,
    evaluations: usize,
) {
    report.wake_cycles += 1;
    report.gate_passed += usize::from(wake.gate.passed);
    report.energy_evaluations += evaluations;
    report.attention_ignitions += wake.minimization.attention_ignitions;
    *handshake_sum += f64::from(wake.minimization.mean_handshake_coherence);
    *phi_sum += f64::from(wake.minimization.mean_integrated_information);
}

pub fn future_training_engine(
    config: FutureGuidedTrainingConfig,
    mode: FutureTrainingMode,
    seed: u64,
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
        seed,
        ..NativeThermoCdtConfig::default()
    });
    core.edge_phase.fill(0.0);
    let adaptive = mode.uses_modulators();
    NativeHybridPhasorCdtEngine::from_core(
        core,
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
                // Interferencia recibe el mismo presupuesto máximo total que
                // todos los candidatos futuros juntos.
                max_iterations: if mode.uses_future() {
                    config.candidate_iterations
                } else {
                    config.candidate_iterations * config.proposals_per_input
                },
                residual_tolerance: 5.0e-3,
                topological_warm_start: false,
                handshake_strength: if adaptive { 0.65 } else { 0.0 },
                attention_strength: if adaptive { 0.55 } else { 0.0 },
                attention_temperature: 0.75,
                attention_max_gain: 3.0,
                attention_ignition_threshold: 0.02,
                handshake_max_gain: 3.0,
                inference_policy: if adaptive {
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
            stability_probe_iterations: 150,
            cdt_consolidation_steps: 0,
            cue_as_boundary: mode.uses_future(),
            sleep_replay_boundary_gain: 0.25,
            anchored_consolidation: true,
            ..NativeHybridConfig::default()
        },
    )
}

struct Evaluation {
    accuracy_sum: f32,
    successes: usize,
    iterations: usize,
    trials: usize,
}

fn evaluate_core(
    core: &NativeThermoCdtSubstrate,
    target: &[i8],
    config: &FutureGuidedTrainingConfig,
    seed: u64,
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
    .expect("el core entrenado debe seguir siendo válido");
    let mut result = Evaluation {
        accuracy_sum: 0.0,
        successes: 0,
        iterations: 0,
        trials: config.evaluation_trials,
    };
    for trial in 0..config.evaluation_trials {
        let probe = seed ^ (trial as u64).rotate_left(19);
        let cue = partial_cue(target, config, probe);
        let mut inference = template.clone();
        for (node, phasor) in inference.phasors.iter_mut().enumerate() {
            let phase = cue.iter().find(|item| item.node == node).map_or_else(
                || std::f32::consts::TAU * unit_from_u64(splitmix64(probe ^ node as u64)),
                |item| item.phase,
            );
            *phasor = Complex32::from_polar(1.0, phase);
        }
        let minimization = inference.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 300,
            residual_tolerance: 5.0e-3,
            topological_warm_start: false,
            ..NativePhasorMinimizerConfig::default()
        });
        let accuracy = direct_accuracy(&inference.phasors, target);
        result.accuracy_sum += accuracy;
        result.successes += usize::from(accuracy >= config.success_accuracy);
        result.iterations += minimization.iterations;
    }
    result
}

fn partial_cue(
    target: &[i8],
    config: &FutureGuidedTrainingConfig,
    seed: u64,
) -> Vec<NativePhasorCue> {
    let mut cue = Vec::new();
    for (node, expected) in target.iter().copied().enumerate() {
        let node_seed = splitmix64(seed.rotate_left(53) ^ node as u64);
        if unit_from_u64(node_seed) >= config.cue_fraction {
            continue;
        }
        let flipped = unit_from_u64(splitmix64(node_seed.rotate_left(17))) < config.cue_corruption;
        let bit = if flipped { -expected } else { expected };
        let jitter = config.phase_jitter
            * (2.0 * unit_from_u64(splitmix64(node_seed.rotate_left(31))) - 1.0);
        cue.push(NativePhasorCue {
            node,
            amplitude: 1.0,
            phase: bit_phase(bit) + jitter,
        });
    }
    if cue.is_empty() {
        cue.push(NativePhasorCue {
            node: 0,
            amplitude: 1.0,
            phase: bit_phase(target[0]),
        });
    }
    cue
}

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
    state
        .iter()
        .zip(target)
        .filter(|(value, expected)| {
            let observed = if value.re >= 0.0 { 1 } else { -1 };
            observed == **expected
        })
        .count() as f32
        / target.len().max(1) as f32
}

fn sanitize(config: FutureGuidedTrainingConfig) -> FutureGuidedTrainingConfig {
    FutureGuidedTrainingConfig {
        nodes: config.nodes.max(16),
        synthetic_tasks: config.synthetic_tasks.clamp(2, 32),
        epochs: config.epochs.max(1),
        proposals_per_input: config
            .proposals_per_input
            .clamp(1, config.synthetic_tasks.max(1)),
        cue_fraction: config.cue_fraction.clamp(0.05, 0.90),
        cue_corruption: config.cue_corruption.clamp(0.0, 0.49),
        goal_fraction: config.goal_fraction.clamp(0.05, 0.90),
        proposal_noise: config.proposal_noise.clamp(0.0, 0.49),
        phase_jitter: config.phase_jitter.max(0.0),
        candidate_iterations: config.candidate_iterations.max(4),
        evaluation_trials: config.evaluation_trials.max(1),
        success_accuracy: config.success_accuracy.clamp(0.5, 1.0),
        ..config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_prior_uses_only_the_cue_to_rank_futures() {
        let config = sanitize(FutureGuidedTrainingConfig {
            nodes: 64,
            synthetic_tasks: 4,
            ..FutureGuidedTrainingConfig::default()
        });
        let patterns = (0..config.synthetic_tasks)
            .map(|task| balanced_pattern(config.nodes, config.seed ^ task as u64))
            .collect::<Vec<_>>();
        let cue = partial_cue(&patterns[2], &config, 17);
        let mut prior =
            SyntheticGemmaPrior::new(patterns, config.goal_fraction, config.proposal_noise);
        let proposals = prior.propose(&cue, config.nodes, 4, 91);
        assert_eq!(proposals.len(), 4);
        assert_eq!(proposals[0].latent_id, 2, "{proposals:?}");
        assert!(proposals.iter().all(|proposal| !proposal.goal.is_empty()));
    }

    #[test]
    fn predicted_futures_outperform_interference_on_paired_synthetic_tasks() {
        let report = run_synthetic_future_guided_comparison(FutureGuidedTrainingConfig {
            nodes: 64,
            synthetic_tasks: 4,
            epochs: 8,
            proposals_per_input: 3,
            evaluation_trials: 4,
            candidate_iterations: 96,
            ..FutureGuidedTrainingConfig::default()
        })
        .unwrap();
        let interference = report
            .reports
            .iter()
            .find(|item| item.mode == "interferencia")
            .unwrap();
        let future = report
            .reports
            .iter()
            .find(|item| item.mode == "futuro_handshake_atencion")
            .unwrap();
        assert!(
            future.recall_accuracy > interference.recall_accuracy + 0.08,
            "{report:#?}"
        );
        assert!(future.selected_correct_futures > 0, "{report:#?}");
        // Interferencia puede aceptar más episodios y aun así consolidar ruido.
        // El criterio operativo de aprendizaje es recuerdo fuera de muestra,
        // no el número bruto de escrituras en CDT.
        assert!(
            future.recall_iterations < interference.recall_iterations,
            "{report:#?}"
        );
    }
}
