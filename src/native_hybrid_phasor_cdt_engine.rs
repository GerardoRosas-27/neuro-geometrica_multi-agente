//! Motor híbrido nativo sobre un único core CDT.
//!
//! La capa fasorial encuentra atractores rápidamente. La capa CDT valida,
//! consolida sus diferencias de fase en las aristas y conserva prototipos
//! persistentes sin depender de RQM ni EPR.

use crate::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorError, NativePhasorInferencePolicy,
    NativePhasorMinimizationReport, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
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
    /// Penalización MDL por escribir un atractor novedoso en memoria CDT.
    /// La topología es fija: representa costo de estado, no nuevos símplices.
    pub storage_complexity_weight: f32,
    /// Beneficio de precisión estable dentro del filtro variacional.
    pub storage_precision_weight: f32,
    /// Se consolida sólo cuando ΔF_store no supera este umbral.
    pub maximum_storage_delta_free_energy: f32,
    /// Escribe la cue también en el estímulo, de modo que entre en F como
    /// término de frontera `-g·Re(ψ*·s)` y no sólo como condición inicial.
    /// Es lo que da a Handshake una frontera que propagar. Sleep siempre
    /// revalida sin ella: un atractor debe sostenerse por la geometría, no
    /// por el estímulo que lo evocó.
    pub cue_as_boundary: bool,
    /// Peso con el que sleep vuelve a evocar la frontera del propio episodio
    /// durante la revalidación. Cero exige que el patrón ya sea atractor de la
    /// geometría actual, lo que impide aprender nada que el prior no soporte
    /// todavía. Un eco débil permite que la consolidación arranque sin que la
    /// frontera sostenga el recuerdo por sí sola.
    pub sleep_replay_boundary_gain: f32,
    /// Cristaliza cada arista sólo en la medida en que el episodio ancló sus
    /// dos extremos. Sin esto, la consolidación reescribe también la región
    /// que la inferencia tuvo que inventar y cada episodio borra lo que
    /// aprendió el anterior.
    pub anchored_consolidation: bool,
}

impl Default for NativeHybridConfig {
    fn default() -> Self {
        Self {
            // Con la cue como frontera, wake es el régimen para el que se
            // diseñó el ciclo: inferencia dirigida por una condición de
            // contorno desde un estado corrompido. Medido en
            // `native_hybrid_cue_boundary_benchmark`, es la única variante que
            // entra en el presupuesto de residuo de consolidación, y además
            // despierta más rápido que Armijo puro.
            minimizer: NativePhasorMinimizerConfig {
                max_iterations: 400,
                residual_tolerance: 5.0e-3,
                topological_warm_start: false,
                handshake_strength: 0.65,
                attention_strength: 0.55,
                attention_temperature: 0.75,
                attention_max_gain: 3.0,
                attention_ignition_threshold: 0.001,
                handshake_max_gain: 3.0,
                inference_policy: NativePhasorInferencePolicy::Adaptive,
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
            storage_complexity_weight: 0.005,
            storage_precision_weight: 0.01,
            maximum_storage_delta_free_energy: 0.0,
            cue_as_boundary: true,
            sleep_replay_boundary_gain: 0.0,
            anchored_consolidation: false,
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
    /// Reducción normalizada de F observada durante wake.
    pub relative_energy_drop: f32,
    /// Frontera que evocó el episodio, para poder replicarla atenuada durante
    /// el replay de sueño. Vacía cuando la inferencia no tuvo frontera.
    pub boundary: Vec<Complex32>,
    /// Anclaje por nodo del episodio: 1 donde la evidencia o la meta fijaron
    /// el estado, 0 donde la inferencia tuvo que inventarlo.
    pub anchors: Vec<f32>,
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
    pub rejected_by_efficiency: usize,
    /// Media de ΔF_store = complejidad - beneficio para candidatos revalidados.
    pub mean_storage_delta_free_energy: f32,
    /// Media sobre los candidatos que llegaron a la sonda de estabilidad.
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
    InvalidAttractorPrototype { length: usize, nodes: usize },
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
            Self::InvalidAttractorPrototype { length, nodes } => write!(
                formatter,
                "el prototipo restaurado tiene {length} nodos, pero el core tiene {nodes}"
            ),
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

    /// Restaura la memoria explícita después de reconstruir el core desde un
    /// checkpoint. La geometría CDT sigue siendo la fuente física del operador;
    /// esta lista recupera identidad, confianza y conteos para que los gates de
    /// novedad/merge no cambien tras reiniciar el proceso.
    pub fn restore_attractors(
        &mut self,
        attractors: Vec<ConsolidatedCdtAttractor>,
    ) -> Result<(), NativeHybridError> {
        let nodes = self.core.node_count();
        if let Some(attractor) = attractors
            .iter()
            .find(|attractor| attractor.prototype.len() != nodes)
        {
            return Err(NativeHybridError::InvalidAttractorPrototype {
                length: attractor.prototype.len(),
                nodes,
            });
        }
        self.attractors = attractors;
        if self.attractors.len() > self.config.max_attractors {
            let remove = self.attractors.len() - self.config.max_attractors;
            self.attractors.drain(0..remove);
        }
        self.pending.clear();
        Ok(())
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
        self.infer_and_stage_with_goal(cue, &[])
    }

    /// Wake con estructura de dos vectores de estado: `cue` fija la evidencia
    /// hacia adelante y `goal` fija la frontera hacia atrás por la que se
    /// post-selecciona. Ambas coexisten en F cuando `cue_as_boundary` está
    /// activo: el pasado pre-selecciona y el futuro post-selecciona.
    ///
    /// Los nodos de `goal` no se escriben en el estado: sólo entran en F como
    /// término de frontera, de modo que la inferencia tiene que reconciliar
    /// evidencia y meta en lugar de recibir la respuesta puesta.
    pub fn infer_and_stage_with_goal(
        &mut self,
        cue: &[NativePhasorCue],
        goal: &[NativePhasorCue],
    ) -> Result<NativeWakeInferenceReport, NativeHybridError> {
        if cue.is_empty() {
            return Err(NativeHybridError::EmptyCue);
        }
        for item in cue.iter().chain(goal) {
            if item.node >= self.core.node_count() {
                return Err(NativeHybridError::InvalidCueNode {
                    node: item.node,
                    nodes: self.core.node_count(),
                });
            }
        }

        self.cycle = self.cycle.wrapping_add(1);
        self.phasor.clear_stimulus();
        let cue_is_boundary = self.config.cue_as_boundary;
        let mut anchors = vec![0.0f32; self.core.node_count()];
        for item in cue {
            let field = Complex32::from_polar(item.amplitude.max(EPSILON), item.phase);
            self.phasor.phasors[item.node] = field;
            anchors[item.node] = 1.0;
            if cue_is_boundary {
                self.phasor.stimulus[item.node] = field;
            }
        }
        for item in goal {
            self.phasor.stimulus[item.node] =
                Complex32::from_polar(item.amplitude.max(EPSILON), item.phase);
            anchors[item.node] = 1.0;
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
            Some(self.stage_pending_attractor(
                confidence,
                minimization.final_report.free_energy,
                relative_energy_drop,
                anchors,
            ))
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
        // La consolidación mide si el patrón se sostiene por la geometría del
        // CDT, no por la frontera que lo evocó. Cada candidato repone la suya
        // atenuada por `sleep_replay_boundary_gain`; con ganancia cero la
        // revalidación es completamente libre de frontera.
        self.phasor.clear_stimulus();
        let replay_gain = self.config.sleep_replay_boundary_gain;
        let mut attractor_journal = Vec::new();
        let cdt_before = self.core.report();
        let memory_before = self.attractors.len();
        let pending_before = candidates.len();

        let result = (|| {
            let mut accepted = 0;
            let mut rejected = 0;
            let mut rejected_by_efficiency = 0;
            let mut stability_sum = 0.0;
            let mut stability_probes = 0usize;
            let mut storage_delta_sum = 0.0;
            let mut consolidated_edges = 0;
            for candidate in &candidates {
                self.phasor.recompile_from_core(&self.core)?;
                self.phasor.phasors.copy_from_slice(&candidate.prototype);
                if replay_gain > 0.0 && candidate.boundary.len() == self.phasor.stimulus.len() {
                    for (field, evoked) in
                        self.phasor.stimulus.iter_mut().zip(&candidate.boundary)
                    {
                        *field = *evoked * replay_gain;
                    }
                }
                let validation = self
                    .phasor
                    .minimize_free_energy(NativePhasorMinimizerConfig {
                        max_iterations: self.config.stability_probe_iterations,
                        topological_warm_start: false,
                        ..self.config.minimizer
                    });
                // La sonda de estabilidad es otra minimización completa, así
                // que sólo se paga cuando los criterios baratos ya pasaron.
                let quality = validation.final_report.free_energy
                    <= validation.initial.free_energy + 1.0e-6
                    && validation.final_report.gradient_residual <= self.config.maximum_residual
                    && validation.final_report.phase_coherence
                        >= self.config.minimum_magnetic_coherence;
                let stability = if quality {
                    let measured = self.stability_probe();
                    stability_sum += measured;
                    stability_probes += 1;
                    measured
                } else {
                    0.0
                };
                let valid = quality && stability >= self.config.minimum_stability;
                if valid {
                    let confidence = candidate.confidence.min(
                        ((validation.final_report.phase_coherence + 1.0) * 0.5) * stability
                            / (1.0 + validation.final_report.gradient_residual),
                    );
                    let revalidation_drop = ((validation.initial.free_energy
                        - validation.final_report.free_energy)
                        / validation.initial.free_energy.abs().max(1.0))
                    .max(0.0);
                    let storage_delta =
                        self.storage_delta_free_energy(candidate, stability, revalidation_drop);
                    storage_delta_sum += storage_delta;
                    if storage_delta <= self.config.maximum_storage_delta_free_energy {
                        let (_, edges) = self.consolidate_attractor(
                            confidence,
                            &candidate.anchors,
                            &mut attractor_journal,
                        )?;
                        consolidated_edges += edges;
                        accepted += 1;
                    } else {
                        rejected += 1;
                        rejected_by_efficiency += 1;
                    }
                } else {
                    rejected += 1;
                }
            }
            // El eco del replay no sobrevive al ciclo de sueño.
            self.phasor.clear_stimulus();

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
                rejected_by_efficiency,
                mean_storage_delta_free_energy: storage_delta_sum
                    / (accepted + rejected_by_efficiency).max(1) as f32,
                mean_stability: stability_sum / stability_probes.max(1) as f32,
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

    fn stage_pending_attractor(
        &mut self,
        confidence: f32,
        free_energy: f32,
        relative_energy_drop: f32,
        anchors: Vec<f32>,
    ) -> u64 {
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
            existing.relative_energy_drop +=
                rate * (relative_energy_drop - existing.relative_energy_drop);
            for (stored, value) in existing.boundary.iter_mut().zip(&self.phasor.stimulus) {
                *stored = *stored * (1.0 - rate) + *value * rate;
            }
            // El anclaje se acumula: dos episodios con máscaras distintas
            // cubren juntos más geometría que cualquiera por separado.
            for (stored, value) in existing.anchors.iter_mut().zip(&anchors) {
                *stored = stored.max(*value);
            }
            return existing.id;
        }
        let id = self.cycle;
        self.pending.push(PendingPhasorAttractor {
            id,
            prototype: observed,
            free_energy,
            confidence,
            observations: 1,
            relative_energy_drop,
            boundary: self.phasor.stimulus.clone(),
            anchors,
        });
        id
    }

    /// ΔF variacional del acto de almacenar. Un patrón nuevo cuesta más que
    /// fusionar uno conocido; la caída de F y su precisión estable pagan ese
    /// costo. Un valor positivo se considera compresión ineficiente.
    fn storage_delta_free_energy(
        &self,
        candidate: &PendingPhasorAttractor,
        stability: f32,
        revalidation_drop: f32,
    ) -> f32 {
        // El barrido es O(memoria × nodos). Al cruzar el umbral de fusión el
        // candidato ya no crea una entrada nueva, así que se corta: la
        // similitud hallada es una cota inferior de la real y por tanto
        // sobreestima el costo, nunca lo subestima.
        let mut maximum_similarity = 0.0f32;
        for attractor in &self.attractors {
            maximum_similarity = maximum_similarity
                .max(attractor_similarity(&attractor.prototype, &candidate.prototype));
            if maximum_similarity >= self.config.attractor_merge_similarity {
                break;
            }
        }
        let novelty = (1.0 - maximum_similarity).clamp(0.0, 1.0);
        let evidence_discount = (candidate.observations.max(1) as f32).sqrt().recip();
        let complexity = self.config.storage_complexity_weight * novelty * evidence_discount;
        let energy_benefit = candidate.relative_energy_drop + revalidation_drop;
        // La precisión que paga el almacenamiento es que el patrón vuelva a
        // emerger tras perturbarlo: eso es lo que reduce sorpresa futura. La
        // coherencia magnética ya se filtra en el gate de wake; volver a
        // multiplicar por ella aquí re-impondría en silencio un mínimo de
        // coherencia que el llamador pudo desactivar a propósito.
        let precision_benefit = self.config.storage_precision_weight * stability;
        complexity - energy_benefit - precision_benefit
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
        anchors: &[f32],
        journal: &mut Vec<AttractorEdit>,
    ) -> Result<(usize, usize), NativeHybridError> {
        let learning_rate = (self.config.consolidation_learning_rate * confidence).clamp(0.0, 1.0);
        // Sólo se graba donde el episodio estuvo anclado. La región que la
        // inferencia tuvo que inventar se deja como estaba en vez de escribir
        // ruido encima de lo que aprendieron episodios anteriores.
        let anchor_at = |node: usize| {
            if self.config.anchored_consolidation {
                anchors.get(node).copied().unwrap_or(0.0).clamp(0.0, 1.0)
            } else {
                1.0
            }
        };
        for node in 0..self.core.node_count() {
            let rate = learning_rate * anchor_at(node);
            if rate <= 0.0 {
                continue;
            }
            let target = self.phasor.phasors[node];
            self.core.amplitude[node] += rate * (target.norm() - self.core.amplitude[node]);
            self.core.phase[node] = blend_phase(self.core.phase[node], target.arg(), rate);
            self.core.thermal_state[node] += rate * (target.re - self.core.thermal_state[node]);
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
            // Una arista sólo cristaliza si el apretón de manos se cerró en
            // sus dos extremos; basta un extremo inventado para no grabarla.
            let rate = learning_rate * anchor_at(a).min(anchor_at(b));
            if rate <= 0.0 {
                continue;
            }
            let preferred_phase = (self.phasor.phasors[b].arg() - self.phasor.phasors[a].arg())
                .rem_euclid(std::f32::consts::TAU);
            let observed_strength =
                (self.phasor.phasors[a].norm() * self.phasor.phasors[b].norm()).sqrt();
            let old =
                Complex32::from_polar(self.core.edge_weight[edge], self.core.edge_phase[edge]);
            let observed = Complex32::from_polar(observed_strength, preferred_phase);
            let consolidated = old * (1.0 - rate) + observed * rate;
            self.core.edge_weight[edge] = consolidated.norm();
            self.core.edge_phase[edge] = consolidated.arg().rem_euclid(std::f32::consts::TAU);
            if let Some(stability) = self.core.edge_stability.get_mut(edge) {
                *stability += rate * (confidence - *stability);
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
            existing.free_energy = self.phasor.free_energy();
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
            free_energy: self.phasor.free_energy(),
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
        storage_complexity_weight: config.storage_complexity_weight.max(0.0),
        storage_precision_weight: config.storage_precision_weight.max(0.0),
        sleep_replay_boundary_gain: config.sleep_replay_boundary_gain.clamp(0.0, 1.0),
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
        // El aislado recibe la cue igual que wake: estado inicial y, si está
        // configurada como frontera, también estímulo. Así la comparación
        // sigue midiendo que wake no añade dinámica propia.
        for item in &cue {
            let field = Complex32::from_polar(item.amplitude, item.phase);
            standalone.phasors[item.node] = field;
            if hybrid.config.cue_as_boundary {
                standalone.stimulus[item.node] = field;
            }
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

    #[test]
    fn cue_enters_free_energy_only_when_configured_as_boundary() {
        for cue_as_boundary in [false, true] {
            let mut engine = engine(NativeHybridConfig {
                cue_as_boundary,
                ..NativeHybridConfig::default()
            });
            let cue = vec![NativePhasorCue {
                node: 0,
                amplitude: 1.0,
                phase: 0.4,
            }];
            engine.infer_and_stage(&cue).unwrap();
            let boundary_norm = engine
                .phasor
                .stimulus
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f32>();
            assert_eq!(boundary_norm > 0.0, cue_as_boundary);
        }
    }

    #[test]
    fn sleep_revalidates_without_the_cue_boundary() {
        let mut engine = engine(NativeHybridConfig {
            cue_as_boundary: true,
            minimum_relative_energy_drop: 0.0,
            ..NativeHybridConfig::default()
        });
        let cue = (0..engine.core.node_count())
            .map(|node| NativePhasorCue {
                node,
                amplitude: 1.0,
                phase: 0.15,
            })
            .collect::<Vec<_>>();
        engine.infer_and_stage(&cue).unwrap();
        assert!(engine.phasor.stimulus.iter().any(|value| value.norm() > 0.0));

        engine.sleep_consolidate().unwrap();
        // Un atractor debe sostenerse por la geometría del CDT, no por el
        // estímulo que lo evocó.
        assert!(engine
            .phasor
            .stimulus
            .iter()
            .all(|value| value.norm() == 0.0));
    }

    #[test]
    fn variational_storage_filter_rejects_an_inefficient_new_memory() {
        let mut engine = engine(NativeHybridConfig {
            minimum_relative_energy_drop: 0.0,
            minimum_stability: 0.90,
            storage_complexity_weight: 100.0,
            storage_precision_weight: 0.0,
            maximum_storage_delta_free_energy: 0.0,
            ..NativeHybridConfig::default()
        });
        let phases_before = engine.core.phase.clone();
        let cue = (0..engine.core.node_count())
            .map(|node| NativePhasorCue {
                node,
                amplitude: 1.0,
                phase: if node % 2 == 0 { 0.0 } else { 0.2 },
            })
            .collect::<Vec<_>>();
        let wake = engine.infer_and_stage(&cue).unwrap();
        assert!(wake.gate.passed, "{wake:?}");

        let sleep = engine.sleep_consolidate().unwrap();
        assert_eq!(sleep.accepted, 0, "{sleep:?}");
        assert_eq!(sleep.rejected_by_efficiency, 1, "{sleep:?}");
        assert!(sleep.mean_storage_delta_free_energy > 0.0, "{sleep:?}");
        assert!(engine.attractors().is_empty());
        assert_eq!(engine.core.phase, phases_before);
    }
}
