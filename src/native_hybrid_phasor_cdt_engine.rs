//! Motor híbrido nativo sobre un único core CDT.
//!
//! La capa fasorial encuentra atractores rápidamente. La capa CDT valida,
//! consolida sus diferencias de fase en las aristas y conserva prototipos
//! persistentes sin depender de RQM ni EPR.

use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorError, NativePhasorMinimizationReport,
    NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use crate::native_rng::signed_unit;
use crate::native_thermodynamic_cdt::{
    NativeThermoCdtConfig, NativeThermoCdtReport, NativeThermoCdtSubstrate,
};
use num_complex::Complex32;
use std::fmt;

const EPSILON: f32 = 1.0e-7;

#[derive(Clone, Copy, Debug)]
pub struct NativeHybridConfig {
    pub minimizer: NativePhasorMinimizerConfig,
    pub consolidation_learning_rate: f32,
    pub minimum_relative_energy_drop: f32,
    pub maximum_residual: f32,
    pub minimum_magnetic_coherence: f32,
    pub minimum_stability: f32,
    pub stability_probe_jitter: f32,
    pub stability_probe_iterations: usize,
    pub attractor_merge_similarity: f32,
    pub max_attractors: usize,
    pub cdt_consolidation_steps: usize,
}

impl Default for NativeHybridConfig {
    fn default() -> Self {
        Self {
            minimizer: NativePhasorMinimizerConfig {
                max_iterations: 400,
                residual_tolerance: 5.0e-3,
                topological_warm_start: false,
                ..NativePhasorMinimizerConfig::default()
            },
            consolidation_learning_rate: 0.18,
            minimum_relative_energy_drop: 0.005,
            maximum_residual: 5.0e-3,
            minimum_magnetic_coherence: 0.80,
            minimum_stability: 0.97,
            stability_probe_jitter: 0.03,
            stability_probe_iterations: 80,
            attractor_merge_similarity: 0.985,
            max_attractors: 128,
            cdt_consolidation_steps: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePhasorCue {
    pub node: usize,
    pub amplitude: f32,
    pub phase: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConsolidatedCdtAttractor {
    pub id: usize,
    pub prototype: Vec<Complex32>,
    pub free_energy: f32,
    pub confidence: f32,
    pub consolidations: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NativeHybridConsolidationGate {
    pub energy_pass: bool,
    pub residual_pass: bool,
    pub coherence_pass: bool,
    pub stability_pass: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingPhasorAttractor {
    pub id: u64,
    pub prototype: Vec<Complex32>,
    pub free_energy: f32,
    pub confidence: f32,
    pub observations: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeWakeInferenceReport {
    pub cycle: u64,
    pub minimization: NativePhasorMinimizationReport,
    pub relative_energy_drop: f32,
    /// Se calcula durante sleep para mantener wake en la ruta rápida.
    pub stability: Option<f32>,
    pub confidence: f32,
    pub gate: NativeHybridConsolidationGate,
    pub pending_id: Option<u64>,
    pub pending_count: usize,
}

#[derive(Clone, Debug)]
pub struct NativeHybridSleepReport {
    pub sleep_cycle: u64,
    pub pending_before: usize,
    pub revalidated: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub mean_stability: f32,
    pub consolidated_edges: usize,
    pub memory_before: usize,
    pub memory_size: usize,
    pub cdt_before: NativeThermoCdtReport,
    pub cdt_after: NativeThermoCdtReport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeHybridError {
    Phasor(NativePhasorError),
    EmptyCue,
    InvalidCueNode { node: usize, nodes: usize },
}

impl fmt::Display for NativeHybridError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Phasor(error) => write!(formatter, "error fasorial: {error}"),
            Self::EmptyCue => write!(formatter, "el estímulo fasorial está vacío"),
            Self::InvalidCueNode { node, nodes } => {
                write!(
                    formatter,
                    "el cue usa el nodo {node}, pero sólo existen {nodes}"
                )
            }
        }
    }
}

impl std::error::Error for NativeHybridError {}

impl From<NativePhasorError> for NativeHybridError {
    fn from(error: NativePhasorError) -> Self {
        Self::Phasor(error)
    }
}

#[derive(Clone, Debug)]
pub struct NativeHybridPhasorCdtEngine {
    /// Core único y persistente.
    pub core: NativeThermoCdtSubstrate,
    /// Capa rápida de interferencia y búsqueda de atractores.
    pub phasor: NativePhasorThermodynamicEngine,
    pub config: NativeHybridConfig,
    attractors: Vec<ConsolidatedCdtAttractor>,
    pending: Vec<PendingPhasorAttractor>,
    cycle: u64,
    sleep_cycle: u64,
}

impl NativeHybridPhasorCdtEngine {
    pub fn new(
        cdt_config: NativeThermoCdtConfig,
        phasor_config: NativePhasorConfig,
        config: NativeHybridConfig,
    ) -> Result<Self, NativeHybridError> {
        let core = NativeThermoCdtSubstrate::new(cdt_config);
        Self::from_core(core, phasor_config, config)
    }

    pub fn from_core(
        core: NativeThermoCdtSubstrate,
        phasor_config: NativePhasorConfig,
        config: NativeHybridConfig,
    ) -> Result<Self, NativeHybridError> {
        let phasor = NativePhasorThermodynamicEngine::from_core(&core, phasor_config)?;
        Ok(Self {
            core,
            phasor,
            config: sanitize_config(config),
            attractors: Vec::new(),
            pending: Vec::new(),
            cycle: 0,
            sleep_cycle: 0,
        })
    }

    pub fn attractors(&self) -> &[ConsolidatedCdtAttractor] {
        &self.attractors
    }

    pub fn pending_attractors(&self) -> &[PendingPhasorAttractor] {
        &self.pending
    }

    /// Fase wake: infiere en fasores y deja el atractor en una cola volátil.
    /// No modifica ningún estado persistente del core CDT.
    pub fn infer_and_stage(
        &mut self,
        cue: &[NativePhasorCue],
    ) -> Result<NativeWakeInferenceReport, NativeHybridError> {
        if cue.is_empty() {
            return Err(NativeHybridError::EmptyCue);
        }
        for item in cue {
            if item.node >= self.core.node_count() {
                return Err(NativeHybridError::InvalidCueNode {
                    node: item.node,
                    nodes: self.core.node_count(),
                });
            }
        }

        self.cycle = self.cycle.wrapping_add(1);
        self.phasor.clear_stimulus();
        for item in cue {
            self.phasor.phasors[item.node] =
                Complex32::from_polar(item.amplitude.max(EPSILON), item.phase);
        }
        let minimization = self.phasor.minimize_free_energy(self.config.minimizer);
        let initial_energy = minimization.initial.free_energy;
        let final_energy = minimization.final_report.free_energy;
        let relative_energy_drop =
            ((initial_energy - final_energy) / initial_energy.abs().max(1.0)).max(0.0);
        let already_stable = minimization.initial.gradient_residual <= self.config.maximum_residual;
        let gate = NativeHybridConsolidationGate {
            energy_pass: final_energy <= initial_energy + 1.0e-6
                && (relative_energy_drop >= self.config.minimum_relative_energy_drop
                    || already_stable),
            residual_pass: minimization.final_report.gradient_residual
                <= self.config.maximum_residual,
            coherence_pass: minimization.final_report.phase_coherence
                >= self.config.minimum_magnetic_coherence,
            // La perturbación costosa queda diferida a sleep.
            stability_pass: true,
            passed: false,
        };
        let gate = NativeHybridConsolidationGate {
            passed: gate.energy_pass
                && gate.residual_pass
                && gate.coherence_pass
                && gate.stability_pass,
            ..gate
        };
        let confidence = ((minimization.final_report.phase_coherence + 1.0) * 0.5)
            / (1.0 + minimization.final_report.gradient_residual);
        let pending_id = if gate.passed {
            Some(self.stage_pending_attractor(confidence, minimization.final_report.free_energy))
        } else {
            None
        };

        Ok(NativeWakeInferenceReport {
            cycle: self.cycle,
            minimization,
            relative_energy_drop,
            stability: None,
            confidence,
            gate,
            pending_id,
            pending_count: self.pending.len(),
        })
    }

    /// Fase sleep: revalida la cola volátil y sólo aquí transfiere sus
    /// atractores al CDT. Ante un error, restaura core, memoria y cola.
    ///
    /// El rollback evita clonar el motor fasorial y las memorias completas:
    /// el core sí se clona entero (garantía transaccional), pero el estado
    /// fasorial se reconstruye desde el core restaurado (el operador es una
    /// función pura del core) y las ediciones de memoria se deshacen con un
    /// journal que sólo copia las entradas efectivamente tocadas.
    pub fn sleep_consolidate(&mut self) -> Result<NativeHybridSleepReport, NativeHybridError> {
        self.sleep_cycle = self.sleep_cycle.wrapping_add(1);
        let core_snapshot = self.core.clone();
        let phasor_state_snapshot = self.phasor.phasors.clone();
        let phasor_tick_snapshot = self.phasor.tick();
        let candidates = std::mem::take(&mut self.pending);
        let mut attractor_journal = Vec::new();
        let cdt_before = self.core.report();
        let memory_before = self.attractors.len();
        let pending_before = candidates.len();

        let result = (|| {
            let mut accepted = 0;
            let mut rejected = 0;
            let mut stability_sum = 0.0;
            let mut consolidated_edges = 0;
            for candidate in &candidates {
                self.phasor.recompile_from_core(&self.core)?;
                self.phasor.phasors.copy_from_slice(&candidate.prototype);
                let validation = self
                    .phasor
                    .minimize_free_energy(NativePhasorMinimizerConfig {
                        max_iterations: self.config.stability_probe_iterations,
                        topological_warm_start: false,
                        ..self.config.minimizer
                    });
                let stability = self.stability_probe();
                stability_sum += stability;
                let valid = validation.final_report.free_energy
                    <= validation.initial.free_energy + 1.0e-6
                    && validation.final_report.gradient_residual <= self.config.maximum_residual
                    && validation.final_report.phase_coherence
                        >= self.config.minimum_magnetic_coherence
                    && stability >= self.config.minimum_stability;
                if valid {
                    let confidence = candidate.confidence.min(
                        ((validation.final_report.phase_coherence + 1.0) * 0.5) * stability
                            / (1.0 + validation.final_report.gradient_residual),
                    );
                    let (_, edges) =
                        self.consolidate_attractor(confidence, &mut attractor_journal)?;
                    consolidated_edges += edges;
                    accepted += 1;
                } else {
                    rejected += 1;
                }
            }

            let mut cdt_after = self.core.report();
            if accepted > 0 {
                for _ in 0..self.config.cdt_consolidation_steps {
                    cdt_after = self.core.step();
                }
                self.phasor.synchronize_state_from_core(&self.core)?;
            }
            Ok(NativeHybridSleepReport {
                sleep_cycle: self.sleep_cycle,
                pending_before,
                revalidated: accepted + rejected,
                accepted,
                rejected,
                mean_stability: stability_sum / (accepted + rejected).max(1) as f32,
                consolidated_edges,
                memory_before,
                memory_size: self.attractors.len(),
                cdt_before,
                cdt_after,
            })
        })();

        if result.is_err() {
            self.core = core_snapshot;
            // El operador fasorial es determinista dado el core, así que se
            // reconstruye en vez de clonarse; luego se repone el campo tal
            // como lo dejó la última fase wake.
            if self.phasor.recompile_from_core(&self.core).is_ok() {
                self.phasor.phasors.copy_from_slice(&phasor_state_snapshot);
            }
            self.phasor.set_tick(phasor_tick_snapshot);
            rollback_attractor_journal(&mut self.attractors, attractor_journal);
            self.pending = candidates;
        }
        result
    }

    pub fn activate_attractor(&mut self, id: usize) -> bool {
        let Some(attractor) = self.attractors.iter().find(|attractor| attractor.id == id) else {
            return false;
        };
        self.phasor.phasors.copy_from_slice(&attractor.prototype);
        true
    }

    fn stage_pending_attractor(&mut self, confidence: f32, free_energy: f32) -> u64 {
        let observed = self.phasor.phasors.clone();
        if let Some(existing) = self.pending.iter_mut().find(|candidate| {
            attractor_similarity(&candidate.prototype, &observed)
                >= self.config.attractor_merge_similarity
        }) {
            let rate = self.config.consolidation_learning_rate;
            for (stored, value) in existing.prototype.iter_mut().zip(&observed) {
                *stored = *stored * (1.0 - rate) + *value * rate;
            }
            existing.free_energy = free_energy;
            existing.confidence += rate * (confidence - existing.confidence);
            existing.observations += 1;
            return existing.id;
        }
        let id = self.cycle;
        self.pending.push(PendingPhasorAttractor {
            id,
            prototype: observed,
            free_energy,
            confidence,
            observations: 1,
        });
        id
    }

    /// Sonda de estabilidad in-place: perturba, re-minimiza y mide la
    /// similitud, y después restaura fasores y tick. El motor queda bit a bit
    /// como antes de la sonda; evita clonar el Laplaciano magnético por
    /// candidato.
    fn stability_probe(&mut self) -> f32 {
        let reference = self.phasor.phasors.clone();
        let tick_before = self.phasor.tick();
        for (node, phasor) in self.phasor.phasors.iter_mut().enumerate() {
            let jitter = self.config.stability_probe_jitter
                * signed_unit(self.cycle ^ (node as u64).rotate_left(19));
            *phasor = Complex32::from_polar(phasor.norm(), phasor.arg() + jitter);
        }
        self.phasor
            .minimize_free_energy(NativePhasorMinimizerConfig {
                max_iterations: self.config.stability_probe_iterations,
                topological_warm_start: false,
                ..self.config.minimizer
            });
        let similarity = attractor_similarity(&reference, &self.phasor.phasors);
        self.phasor.phasors.copy_from_slice(&reference);
        self.phasor.set_tick(tick_before);
        similarity
    }

    fn consolidate_attractor(
        &mut self,
        confidence: f32,
        journal: &mut Vec<AttractorEdit>,
    ) -> Result<(usize, usize), NativeHybridError> {
        let learning_rate = (self.config.consolidation_learning_rate * confidence).clamp(0.0, 1.0);
        for node in 0..self.core.node_count() {
            let target = self.phasor.phasors[node];
            self.core.amplitude[node] +=
                learning_rate * (target.norm() - self.core.amplitude[node]);
            self.core.phase[node] = blend_phase(self.core.phase[node], target.arg(), learning_rate);
            self.core.thermal_state[node] +=
                learning_rate * (target.re - self.core.thermal_state[node]);
        }

        let edge_count = self
            .core
            .edge_a
            .len()
            .min(self.core.edge_b.len())
            .min(self.core.edge_weight.len())
            .min(self.core.edge_phase.len());
        for edge in 0..edge_count {
            let a = self.core.edge_a[edge];
            let b = self.core.edge_b[edge];
            let preferred_phase = (self.phasor.phasors[b].arg() - self.phasor.phasors[a].arg())
                .rem_euclid(std::f32::consts::TAU);
            let observed_strength =
                (self.phasor.phasors[a].norm() * self.phasor.phasors[b].norm()).sqrt();
            let old =
                Complex32::from_polar(self.core.edge_weight[edge], self.core.edge_phase[edge]);
            let observed = Complex32::from_polar(observed_strength, preferred_phase);
            let consolidated = old * (1.0 - learning_rate) + observed * learning_rate;
            self.core.edge_weight[edge] = consolidated.norm();
            self.core.edge_phase[edge] = consolidated.arg().rem_euclid(std::f32::consts::TAU);
            if let Some(stability) = self.core.edge_stability.get_mut(edge) {
                *stability += learning_rate * (confidence - *stability);
                *stability = (*stability).clamp(0.0, 1.0);
            }
        }
        self.phasor.recompile_from_core(&self.core)?;
        let id = self.store_or_merge_attractor(confidence, journal);
        Ok((id, edge_count))
    }

    fn store_or_merge_attractor(
        &mut self,
        confidence: f32,
        journal: &mut Vec<AttractorEdit>,
    ) -> usize {
        let merge_index = self.attractors.iter().position(|attractor| {
            attractor_similarity(&attractor.prototype, &self.phasor.phasors)
                >= self.config.attractor_merge_similarity
        });
        if let Some(index) = merge_index {
            journal.push(AttractorEdit::Merged {
                index,
                before: self.attractors[index].clone(),
            });
            let rate = self.config.consolidation_learning_rate;
            let existing = &mut self.attractors[index];
            for (stored, observed) in existing.prototype.iter_mut().zip(&self.phasor.phasors) {
                *stored = *stored * (1.0 - rate) + *observed * rate;
            }
            existing.free_energy = self.phasor.report().free_energy;
            existing.confidence += rate * (confidence - existing.confidence);
            existing.consolidations += 1;
            return existing.id;
        }

        if self.attractors.len() >= self.config.max_attractors {
            let remove = self
                .attractors
                .iter()
                .enumerate()
                .min_by(|left, right| left.1.confidence.total_cmp(&right.1.confidence))
                .map(|(index, _)| index)
                .unwrap_or(0);
            journal.push(AttractorEdit::Removed {
                index: remove,
                attractor: self.attractors.remove(remove),
            });
        }
        let id = self
            .attractors
            .iter()
            .map(|attractor| attractor.id)
            .max()
            .map_or(0, |id| id + 1);
        journal.push(AttractorEdit::Pushed);
        self.attractors.push(ConsolidatedCdtAttractor {
            id,
            prototype: self.phasor.phasors.clone(),
            free_energy: self.phasor.report().free_energy,
            confidence,
            consolidations: 1,
        });
        id
    }
}

/// Edición reversible sobre la memoria de atractores consolidados. Deshacer
/// en orden inverso restaura la memoria exacta previa a la transacción sin
/// clonarla entera por ciclo de sueño.
enum AttractorEdit {
    Merged {
        index: usize,
        before: ConsolidatedCdtAttractor,
    },
    Removed {
        index: usize,
        attractor: ConsolidatedCdtAttractor,
    },
    Pushed,
}

fn rollback_attractor_journal(
    attractors: &mut Vec<ConsolidatedCdtAttractor>,
    journal: Vec<AttractorEdit>,
) {
    for edit in journal.into_iter().rev() {
        match edit {
            AttractorEdit::Merged { index, before } => {
                if index < attractors.len() {
                    attractors[index] = before;
                }
            }
            AttractorEdit::Removed { index, attractor } => {
                attractors.insert(index.min(attractors.len()), attractor);
            }
            AttractorEdit::Pushed => {
                attractors.pop();
            }
        }
    }
}

fn attractor_similarity(left: &[Complex32], right: &[Complex32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let (inner, left_norm_sqr, right_norm_sqr) = left.iter().zip(right).fold(
        (Complex32::new(0.0, 0.0), 0.0_f32, 0.0_f32),
        |(inner, left_norm, right_norm), (left, right)| {
            (
                inner + left.conj() * right,
                left_norm + left.norm_sqr(),
                right_norm + right.norm_sqr(),
            )
        },
    );
    (inner.norm() / (left_norm_sqr * right_norm_sqr).sqrt().max(EPSILON)).clamp(0.0, 1.0)
}

fn blend_phase(current: f32, target: f32, amount: f32) -> f32 {
    let mixed =
        Complex32::from_polar(1.0 - amount, current) + Complex32::from_polar(amount, target);
    mixed.arg().rem_euclid(std::f32::consts::TAU)
}

fn sanitize_config(config: NativeHybridConfig) -> NativeHybridConfig {
    NativeHybridConfig {
        consolidation_learning_rate: config.consolidation_learning_rate.clamp(0.0, 1.0),
        minimum_relative_energy_drop: config.minimum_relative_energy_drop.max(0.0),
        maximum_residual: config.maximum_residual.max(EPSILON),
        minimum_magnetic_coherence: config.minimum_magnetic_coherence.clamp(-1.0, 1.0),
        minimum_stability: config.minimum_stability.clamp(0.0, 1.0),
        stability_probe_jitter: config.stability_probe_jitter.clamp(0.0, 0.5),
        stability_probe_iterations: config.stability_probe_iterations.max(1),
        attractor_merge_similarity: config.attractor_merge_similarity.clamp(0.0, 1.0),
        max_attractors: config.max_attractors.max(1),
        ..config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(config: NativeHybridConfig) -> NativeHybridPhasorCdtEngine {
        NativeHybridPhasorCdtEngine::new(
            NativeThermoCdtConfig {
                slices: 1,
                nodes_per_slice: 32,
                temperature: 0.0,
                diffusion: 0.0,
                pilot_gain: 0.0,
                amplitude_decay: 0.0,
                ..NativeThermoCdtConfig::default()
            },
            NativePhasorConfig {
                temperature_scale: 0.0,
                noise_scale: 0.0,
                entropy_weight: 0.0,
                ..NativePhasorConfig::default()
            },
            config,
        )
        .unwrap()
    }

    #[test]
    fn wake_inference_matches_the_standalone_phasor_solver() {
        let mut hybrid = engine(NativeHybridConfig {
            minimum_relative_energy_drop: 0.0,
            ..NativeHybridConfig::default()
        });
        let mut standalone =
            NativePhasorThermodynamicEngine::from_core(&hybrid.core, hybrid.phasor.config).unwrap();
        let cue = (0..hybrid.core.node_count())
            .map(|node| NativePhasorCue {
                node,
                amplitude: 1.0,
                phase: if node % 3 == 0 { 0.12 } else { 2.7 },
            })
            .collect::<Vec<_>>();
        for item in &cue {
            standalone.phasors[item.node] = Complex32::from_polar(item.amplitude, item.phase);
        }
        let standalone_report = standalone.minimize_free_energy(hybrid.config.minimizer);
        let wake_report = hybrid.infer_and_stage(&cue).unwrap();

        assert_eq!(
            wake_report.minimization.iterations,
            standalone_report.iterations
        );
        assert_eq!(
            wake_report.minimization.energy_evaluations,
            standalone_report.energy_evaluations
        );
        assert!(
            (wake_report.minimization.final_report.free_energy
                - standalone_report.final_report.free_energy)
                .abs()
                < 1.0e-6
        );
        assert!(
            (wake_report.minimization.final_report.gradient_residual
                - standalone_report.final_report.gradient_residual)
                .abs()
                < 1.0e-7
        );
        assert!(wake_report.stability.is_none());
    }

    #[test]
    fn wake_stages_and_sleep_consolidates_into_shared_cdt_core() {
        let mut engine = engine(NativeHybridConfig {
            minimum_relative_energy_drop: 0.0,
            minimum_stability: 0.90,
            ..NativeHybridConfig::default()
        });
        let phases_before = engine.core.phase.clone();
        let weights_before = engine.core.edge_weight.clone();
        let cue = (0..32)
            .map(|node| NativePhasorCue {
                node,
                amplitude: 1.0,
                phase: 0.0,
            })
            .collect::<Vec<_>>();
        let wake = engine.infer_and_stage(&cue).unwrap();
        assert!(wake.gate.passed, "{wake:?}");
        assert_eq!(wake.pending_count, 1);
        assert!(engine.attractors().is_empty());
        assert_eq!(engine.core.phase, phases_before);
        assert_eq!(engine.core.edge_weight, weights_before);

        let sleep = engine.sleep_consolidate().unwrap();
        assert_eq!(sleep.accepted, 1, "{sleep:?}");
        assert_eq!(sleep.memory_size, 1);
        assert_eq!(sleep.consolidated_edges, engine.core.edge_count());
        assert!(engine.pending_attractors().is_empty());
        assert_ne!(engine.core.phase, phases_before);
        assert_eq!(engine.attractors()[0].consolidations, 1);

        let replay = engine.infer_and_stage(&cue).unwrap();
        assert!(replay.gate.passed, "{replay:?}");
        assert_eq!(engine.attractors()[0].consolidations, 1);
        let replay_sleep = engine.sleep_consolidate().unwrap();
        assert_eq!(replay_sleep.memory_size, 1);
        assert_eq!(engine.attractors()[0].consolidations, 2);
    }

    #[test]
    fn rejected_attractor_leaves_cdt_memory_unchanged() {
        let mut engine = engine(NativeHybridConfig {
            maximum_residual: EPSILON,
            minimum_stability: 1.0,
            stability_probe_jitter: 0.5,
            stability_probe_iterations: 1,
            ..NativeHybridConfig::default()
        });
        let phases_before = engine.core.phase.clone();
        let weights_before = engine.core.edge_weight.clone();
        let report = engine
            .infer_and_stage(&[NativePhasorCue {
                node: 0,
                amplitude: 1.0,
                phase: 1.7,
            }])
            .unwrap();
        assert!(!report.gate.passed, "{report:?}");
        assert!(engine.attractors().is_empty());
        assert!(engine.pending_attractors().is_empty());
        assert_eq!(engine.core.phase, phases_before);
        assert_eq!(engine.core.edge_weight, weights_before);
        let sleep = engine.sleep_consolidate().unwrap();
        assert_eq!(sleep.pending_before, 0);
        assert_eq!(sleep.accepted, 0);
        assert_eq!(engine.core.phase, phases_before);
    }
}
