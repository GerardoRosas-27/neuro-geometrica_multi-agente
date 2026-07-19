use crate::entanglement::{EntanglementConfig, EntanglementField, EntanglementReport};
use crate::native_thermodynamic_cdt::{
    NativeSamplingConfig, NativeSamplingProgram, NativeThermoCdtConfig, NativeThermoCdtReport,
    NativeThermoCdtSubstrate,
};
use crate::relational_field::ObserverId;
use std::collections::HashMap;

const EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug)]
pub struct NativeThermoRqmConfig {
    pub amplitude_learning_rate: f32,
    pub coherence_learning_rate: f32,
    pub uncertainty_learning_rate: f32,
    pub phase_learning_rate: f32,
    pub amplitude_decay: f32,
    pub thermal_steps_per_train: usize,
    pub thermal_steps_per_query: usize,
    pub thermal_score_gain: f32,
    pub thermal_activation_margin: f32,
    pub collect_query_diagnostics: bool,
    pub max_candidates: usize,
    pub max_pilot_window_nodes: usize,
    pub sampling_block_size: usize,
    pub sampling_schedule_rounds: usize,
    pub max_sampling_blocks: usize,
}

impl Default for NativeThermoRqmConfig {
    fn default() -> Self {
        Self {
            amplitude_learning_rate: 0.12,
            coherence_learning_rate: 0.10,
            uncertainty_learning_rate: 0.12,
            phase_learning_rate: 0.18,
            amplitude_decay: 0.001,
            thermal_steps_per_train: 1,
            thermal_steps_per_query: 4,
            thermal_score_gain: 0.30,
            thermal_activation_margin: 0.02,
            collect_query_diagnostics: true,
            max_candidates: 32,
            max_pilot_window_nodes: 96,
            sampling_block_size: 16,
            sampling_schedule_rounds: 2,
            max_sampling_blocks: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct RelationKey {
    observer: ObserverId,
    source: usize,
    target: usize,
}

#[derive(Clone, Copy, Debug)]
struct NativeRelation {
    key: RelationKey,
    amplitude: f32,
    phase: f32,
    coherence: f32,
    uncertainty: f32,
    last_tick: u64,
}

#[derive(Clone, Debug)]
pub struct NativeCandidateScore {
    pub agent: usize,
    pub score: f32,
    pub relational_score: f32,
    pub thermal_multiplier: f32,
}

#[derive(Clone, Debug)]
pub struct NativeRqmQueryReport {
    pub observer: ObserverId,
    pub seeds: Vec<usize>,
    pub candidates: Vec<NativeCandidateScore>,
    pub thermal: NativeThermoCdtReport,
    pub entanglement: EntanglementReport,
}

#[derive(Clone, Copy, Debug)]
pub struct RealtimeUpdateConfig {
    pub max_relation_updates: usize,
    pub max_epr_observations: usize,
    pub max_epr_evictions: usize,
    pub epr_reserve_slots: usize,
    pub max_window_nodes: usize,
    pub thermal_microsteps: usize,
    pub min_success: f32,
}

impl Default for RealtimeUpdateConfig {
    fn default() -> Self {
        Self {
            max_relation_updates: 64,
            max_epr_observations: 32,
            max_epr_evictions: 2,
            epr_reserve_slots: 1,
            max_window_nodes: 96,
            thermal_microsteps: 1,
            min_success: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RealtimeUpdateReport {
    pub relation_updates: usize,
    pub epr_observations: usize,
    pub epr_created: usize,
    pub epr_evicted: usize,
    pub window_nodes: usize,
    pub thermal: NativeThermoCdtReport,
}

#[derive(Clone, Debug)]
pub struct RealtimeInteractionReport {
    pub query: NativeRqmQueryReport,
    pub update: RealtimeUpdateReport,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SymmetryGuidedRqmUpdateReport {
    pub observed_updates: usize,
    pub orbit_updates: usize,
    pub epr_predictive_accepts: usize,
    pub epr_conflicts: usize,
    pub symmetry_confidence: f32,
    pub prediction_error: f32,
}

#[derive(Clone, Debug)]
pub struct NativeThermoRqmEprSubstrate {
    pub thermal: NativeThermoCdtSubstrate,
    pub config: NativeThermoRqmConfig,
    pub entanglement: EntanglementField,
    sampling_program: NativeSamplingProgram,
    relations: Vec<NativeRelation>,
    relation_lookup: HashMap<RelationKey, usize>,
    neighbor_index: HashMap<(usize, usize), Vec<usize>>,
    candidate_accumulator: HashMap<usize, f32>,
    tick: u64,
}

impl NativeThermoRqmEprSubstrate {
    pub fn new(
        thermal_config: NativeThermoCdtConfig,
        rqm_config: NativeThermoRqmConfig,
        entanglement_config: EntanglementConfig,
    ) -> Self {
        let thermal = NativeThermoCdtSubstrate::new(thermal_config);
        let sampling_program = thermal.compile_sampling_program(NativeSamplingConfig {
            block_size: rqm_config.sampling_block_size,
            schedule_rounds: rqm_config.sampling_schedule_rounds,
            max_blocks_per_pulse: rqm_config.max_sampling_blocks,
        });
        Self {
            thermal,
            config: rqm_config,
            entanglement: EntanglementField::new(entanglement_config),
            sampling_program,
            relations: Vec::new(),
            relation_lookup: HashMap::new(),
            neighbor_index: HashMap::new(),
            candidate_accumulator: HashMap::new(),
            tick: 0,
        }
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn relation_entries(
        &self,
    ) -> impl Iterator<Item = (ObserverId, usize, usize, f32, f32, f32, f32, u64)> + '_ {
        self.relations.iter().map(|relation| {
            (
                relation.key.observer,
                relation.key.source,
                relation.key.target,
                relation.amplitude,
                relation.phase,
                relation.coherence,
                relation.uncertainty,
                relation.last_tick,
            )
        })
    }

    pub fn relation_count_for_observer(&self, observer: ObserverId) -> usize {
        self.relations
            .iter()
            .filter(|relation| relation.key.observer == observer)
            .count()
    }

    /// Conserva para un observador las relaciones de mayor utilidad
    /// (amplitud, coherencia, baja incertidumbre y recencia).
    pub fn prune_observer_relations_to_budget(
        &mut self,
        observer: ObserverId,
        max_relations: usize,
    ) -> usize {
        let mut candidates = self
            .relations
            .iter()
            .enumerate()
            .filter(|(_, relation)| relation.key.observer == observer)
            .map(|(index, relation)| {
                let utility = relation.amplitude
                    * (0.25 + 0.75 * relation.coherence)
                    * (1.0 - 0.8 * relation.uncertainty).max(0.05);
                (index, utility, relation.last_tick)
            })
            .collect::<Vec<_>>();
        let remove = candidates.len().saturating_sub(max_relations);
        if remove == 0 {
            return 0;
        }
        candidates.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.2.cmp(&right.2)));
        let mut keep = vec![true; self.relations.len()];
        for (index, _, _) in candidates.into_iter().take(remove) {
            keep[index] = false;
        }
        let mut cursor = 0usize;
        self.relations.retain(|_| {
            let retain = keep[cursor];
            cursor += 1;
            retain
        });
        self.rebuild_relation_indices();
        remove
    }

    /// Expande el hardware térmico y recompila el programa de muestreo.
    /// Relaciones y EPR existentes conservan IDs válidos porque los nodos
    /// anteriores mantienen sus posiciones.
    pub fn grow_thermal_slices(&mut self, additional_slices: usize) -> usize {
        let added = self.thermal.grow_slices(additional_slices);
        if added > 0 {
            self.sampling_program = self.thermal.compile_sampling_program(NativeSamplingConfig {
                block_size: self.config.sampling_block_size,
                schedule_rounds: self.config.sampling_schedule_rounds,
                max_blocks_per_pulse: self.config.max_sampling_blocks,
            });
        }
        added
    }

    fn rebuild_relation_indices(&mut self) {
        self.relation_lookup.clear();
        self.neighbor_index.clear();
        for (index, relation) in self.relations.iter().enumerate() {
            self.relation_lookup.insert(relation.key, index);
            self.neighbor_index
                .entry((relation.key.observer.0, relation.key.source))
                .or_default()
                .push(index);
        }
        self.candidate_accumulator.clear();
    }

    pub fn import_relation_state(
        &mut self,
        observer: ObserverId,
        source: usize,
        target: usize,
        amplitude: f32,
        phase: f32,
        coherence: f32,
        uncertainty: f32,
        last_tick: u64,
    ) {
        if source == target {
            return;
        }
        self.tick = self.tick.max(last_tick);
        let key = RelationKey {
            observer,
            source,
            target,
        };
        let relation = NativeRelation {
            key,
            amplitude: amplitude.clamp(0.0, 4.0),
            phase: phase.rem_euclid(std::f32::consts::TAU),
            coherence: coherence.clamp(0.0, 1.0),
            uncertainty: uncertainty.clamp(0.0, 1.0),
            last_tick,
        };
        if let Some(&idx) = self.relation_lookup.get(&key) {
            self.relations[idx] = relation;
            return;
        }

        let idx = self.relations.len();
        self.relations.push(relation);
        self.relation_lookup.insert(key, idx);
        self.neighbor_index
            .entry((observer.0, source))
            .or_default()
            .push(idx);
    }

    pub fn attenuate_relation(
        &mut self,
        observer: ObserverId,
        source: usize,
        target: usize,
        amount: f32,
    ) {
        let Some(&idx) = self.relation_lookup.get(&RelationKey {
            observer,
            source,
            target,
        }) else {
            return;
        };
        let amount = amount.clamp(0.0, 1.0);
        let relation = &mut self.relations[idx];
        relation.amplitude *= 1.0 - 0.35 * amount;
        relation.coherence *= 1.0 - 0.20 * amount;
        relation.uncertainty = (relation.uncertainty + 0.30 * amount).min(1.0);
        relation.last_tick = self.tick;
    }

    /// Mezcla de fase O(1) para plasticidad local.
    pub fn blend_relation_phase(
        &mut self,
        observer: ObserverId,
        source: usize,
        target: usize,
        target_phase: f32,
        amount: f32,
    ) -> bool {
        let Some(&idx) = self.relation_lookup.get(&RelationKey {
            observer,
            source,
            target,
        }) else {
            return false;
        };
        let relation = &mut self.relations[idx];
        relation.phase = blend_phase(relation.phase, target_phase, amount);
        relation.last_tick = self.tick;
        true
    }

    pub fn train_observed_transition(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        cause: &[usize],
        effect: &[usize],
        success: f32,
    ) {
        self.train_observed_transition_with_epr_benefit(
            observer,
            observer_phase,
            cause,
            effect,
            success,
            success,
        );
    }

    pub fn train_observed_transition_with_epr_benefit(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        cause: &[usize],
        effect: &[usize],
        success: f32,
        epr_benefit: f32,
    ) {
        self.tick = self.tick.wrapping_add(1);
        self.thermal
            .inject_pilot_pattern(cause, 2.0 * success.max(0.0), observer_phase);
        self.thermal
            .inject_pilot_pattern(effect, 1.4 * success.max(0.0), observer_phase);

        for &source in cause {
            for &target in effect {
                self.reinforce_relation(observer, source, target, observer_phase, success);
                self.entanglement
                    .observe_correlation(source, target, epr_benefit);
            }
        }
        for _ in 0..self.config.thermal_steps_per_train {
            self.thermal.step();
        }
    }

    /// Adaptador legacy: la simetría controla dónde se comparte aprendizaje,
    /// mientras RQM conserva amplitud/fase y EPR exige utilidad predictiva.
    pub fn train_symmetry_guided_edges(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        observed: (usize, usize),
        orbit: &[(usize, usize)],
        success: f32,
        symmetry_confidence: f32,
        prediction_error: f32,
    ) -> SymmetryGuidedRqmUpdateReport {
        let symmetry_confidence = symmetry_confidence.clamp(0.0, 1.0);
        let prediction_error = prediction_error.abs();
        let mut report = SymmetryGuidedRqmUpdateReport {
            symmetry_confidence,
            prediction_error,
            ..SymmetryGuidedRqmUpdateReport::default()
        };
        self.tick = self.tick.wrapping_add(1);
        self.thermal
            .inject_pilot_pattern(&[observed.0], 2.0 * success.max(0.0), observer_phase);
        self.reinforce_relation(observer, observed.0, observed.1, observer_phase, success);
        report.observed_updates = 1;
        let epr = self.entanglement.observe_predictive_correlation(
            observed.0,
            observed.1,
            success,
            prediction_error,
            0.25,
        );
        report.epr_predictive_accepts += usize::from(epr.accepted);
        report.epr_conflicts += usize::from(epr.conflict);

        if symmetry_confidence > 0.0 {
            let transfer = success * symmetry_confidence;
            for &(source, target) in orbit {
                if (source, target) == observed || source == target {
                    continue;
                }
                self.reinforce_relation(observer, source, target, observer_phase, transfer);
                report.orbit_updates += 1;
                let epr = self.entanglement.observe_predictive_correlation(
                    source,
                    target,
                    transfer,
                    prediction_error,
                    0.25,
                );
                report.epr_predictive_accepts += usize::from(epr.accepted);
                report.epr_conflicts += usize::from(epr.conflict);
            }
        }
        for _ in 0..self.config.thermal_steps_per_train {
            self.thermal.step();
        }
        report
    }

    /// Modificación online de latencia acotada. Actualiza solo un número máximo
    /// de pares y evoluciona una ventana térmica local; nunca ejecuta `step()`
    /// sobre todo el sustrato.
    pub fn train_observed_transition_realtime(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        cause: &[usize],
        effect: &[usize],
        success: f32,
        config: RealtimeUpdateConfig,
    ) -> RealtimeUpdateReport {
        let success = success.clamp(0.0, 1.0);
        if success < config.min_success || cause.is_empty() || effect.is_empty() {
            return RealtimeUpdateReport::default();
        }
        self.tick = self.tick.wrapping_add(1);
        let mut report = RealtimeUpdateReport::default();

        'pairs: for &source in cause {
            for &target in effect {
                if source == target {
                    continue;
                }
                if report.relation_updates >= config.max_relation_updates {
                    break 'pairs;
                }
                self.reinforce_relation(observer, source, target, observer_phase, success);
                report.relation_updates += 1;
                if report.epr_observations < config.max_epr_observations {
                    let (created, evicted) = if report.epr_evicted < config.max_epr_evictions {
                        self.entanglement.observe_correlation_with_reserve(
                            source,
                            target,
                            success,
                            config.epr_reserve_slots,
                        )
                    } else {
                        (
                            self.entanglement
                                .observe_correlation(source, target, success),
                            0,
                        )
                    };
                    report.epr_observations += 1;
                    report.epr_created += usize::from(created);
                    report.epr_evicted += evicted;
                }
            }
        }

        let window = self.thermal.local_neighborhood(
            cause,
            effect,
            config.max_window_nodes.max(cause.len() + effect.len()),
        );
        for &node in cause {
            self.thermal
                .inject_local_node(node, 1.2 * success, observer_phase, 0.55 * success);
        }
        for &node in effect {
            self.thermal
                .inject_local_node(node, 0.9 * success, observer_phase, 0.35 * success);
        }
        let mut thermal = self.thermal.report_local(&window);
        for _ in 0..config.thermal_microsteps {
            thermal = self.thermal.step_local(&window);
        }
        report.window_nodes = window.len();
        report.thermal = thermal;
        report
    }

    /// Consulta y aplica feedback en la misma transacción mutable.
    pub fn query_and_learn_realtime(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        seeds: &[usize],
        expected: &[usize],
        success: f32,
        config: RealtimeUpdateConfig,
    ) -> RealtimeInteractionReport {
        let query = self.query(observer, observer_phase, seeds);
        let update = self.train_observed_transition_realtime(
            observer,
            observer_phase,
            seeds,
            expected,
            success,
            config,
        );
        RealtimeInteractionReport { query, update }
    }

    pub fn query(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        seeds: &[usize],
    ) -> NativeRqmQueryReport {
        self.tick = self.tick.wrapping_add(1);

        let mut base_scores = self.relational_candidate_scores(observer, observer_phase, seeds);
        if base_scores.len() > self.config.max_candidates {
            base_scores.select_nth_unstable_by(self.config.max_candidates, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            base_scores.truncate(self.config.max_candidates);
        }
        base_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut candidate_ids = base_scores
            .iter()
            .map(|(agent, _)| *agent)
            .collect::<Vec<_>>();
        let relational_margin = score_margin(&base_scores);
        let ambiguous = base_scores.is_empty()
            || (self.config.thermal_activation_margin > 0.0
                && relational_margin <= self.config.thermal_activation_margin
                && base_scores.len() < self.config.max_candidates);
        let sync_epr = self.entanglement.config.max_syncs_per_step > 0 && ambiguous;
        let entanglement = if sync_epr {
            self.entanglement.synchronize_candidates_with_diagnostics(
                seeds,
                &mut candidate_ids,
                self.config.collect_query_diagnostics,
            )
        } else if self.config.collect_query_diagnostics {
            self.entanglement.summary()
        } else {
            EntanglementReport::default()
        };
        let use_thermal = self.config.thermal_steps_per_query > 0 && ambiguous;
        let thermal = if use_thermal {
            self.thermal.pulse_compiled_pilot(
                &self.sampling_program,
                seeds,
                &candidate_ids,
                observer_phase,
                adaptive_uncertainty(relational_margin, self.config.thermal_activation_margin),
            )
        } else if self.config.collect_query_diagnostics {
            self.thermal.report()
        } else {
            NativeThermoCdtReport {
                tick: self.thermal.tick(),
                nodes: self.thermal.node_count(),
                edges: self.thermal.edge_count(),
                ..NativeThermoCdtReport::default()
            }
        };

        self.candidate_accumulator.clear();
        self.candidate_accumulator
            .extend(base_scores.iter().copied());
        let mut candidates = Vec::with_capacity(candidate_ids.len());
        for agent in candidate_ids {
            let relational_score = match self.candidate_accumulator.get(&agent).copied() {
                Some(score) => score,
                None => self.relational_score(observer, observer_phase, seeds, agent),
            };
            let thermal_multiplier = if use_thermal {
                self.thermal_multiplier(agent)
            } else {
                1.0
            };
            let score = relational_score * thermal_multiplier;
            if score > EPSILON {
                candidates.push(NativeCandidateScore {
                    agent,
                    score,
                    relational_score,
                    thermal_multiplier,
                });
            }
        }
        self.candidate_accumulator.clear();

        let candidate_order = |a: &NativeCandidateScore, b: &NativeCandidateScore| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.agent.cmp(&b.agent))
        };
        if candidates.len() > self.config.max_candidates {
            candidates.select_nth_unstable_by(self.config.max_candidates, candidate_order);
            candidates.truncate(self.config.max_candidates);
        }
        candidates.sort_by(candidate_order);

        NativeRqmQueryReport {
            observer,
            seeds: seeds.to_vec(),
            candidates,
            thermal,
            entanglement,
        }
    }

    fn reinforce_relation(
        &mut self,
        observer: ObserverId,
        source: usize,
        target: usize,
        observer_phase: f32,
        success: f32,
    ) {
        if source == target {
            return;
        }
        let key = RelationKey {
            observer,
            source,
            target,
        };
        if let Some(&idx) = self.relation_lookup.get(&key) {
            let relation = &mut self.relations[idx];
            relation.amplitude =
                (relation.amplitude + self.config.amplitude_learning_rate * success).min(4.0);
            relation.coherence =
                (relation.coherence + self.config.coherence_learning_rate * success).min(1.0);
            relation.uncertainty =
                (relation.uncertainty - self.config.uncertainty_learning_rate * success).max(0.0);
            relation.phase = blend_phase(
                relation.phase,
                observer_phase,
                self.config.phase_learning_rate * success,
            );
            relation.last_tick = self.tick;
            return;
        }

        let idx = self.relations.len();
        self.relations.push(NativeRelation {
            key,
            amplitude: (0.05 + self.config.amplitude_learning_rate * success).max(0.0),
            phase: observer_phase,
            coherence: (0.20 + self.config.coherence_learning_rate * success).min(1.0),
            uncertainty: (0.80 - self.config.uncertainty_learning_rate * success).max(0.0),
            last_tick: self.tick,
        });
        self.relation_lookup.insert(key, idx);
        self.neighbor_index
            .entry((observer.0, source))
            .or_default()
            .push(idx);
    }

    fn relational_candidate_scores(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        seeds: &[usize],
    ) -> Vec<(usize, f32)> {
        self.candidate_accumulator.clear();
        for &seed in seeds {
            let Some(indices) = self.neighbor_index.get(&(observer.0, seed)) else {
                continue;
            };
            for &idx in indices {
                let relation = self.relations[idx];
                let age = self.tick.saturating_sub(relation.last_tick) as f32;
                let recency = 1.0 / (1.0 + self.config.amplitude_decay * age);
                let phase_alignment = (relation.phase - observer_phase).cos().max(0.0);
                let score = relation.amplitude
                    * relation.amplitude
                    * relation.coherence
                    * (1.0 - relation.uncertainty)
                    * phase_alignment
                    * recency;
                *self
                    .candidate_accumulator
                    .entry(relation.key.target)
                    .or_insert(0.0) += score;
            }
        }
        self.candidate_accumulator.drain().collect()
    }

    fn relational_score(
        &self,
        observer: ObserverId,
        observer_phase: f32,
        seeds: &[usize],
        agent: usize,
    ) -> f32 {
        seeds
            .iter()
            .filter_map(|&seed| {
                let idx = *self.relation_lookup.get(&RelationKey {
                    observer,
                    source: seed,
                    target: agent,
                })?;
                let relation = self.relations[idx];
                let age = self.tick.saturating_sub(relation.last_tick) as f32;
                let recency = 1.0 / (1.0 + self.config.amplitude_decay * age);
                let phase_alignment = (relation.phase - observer_phase).cos().max(0.0);
                Some(
                    relation.amplitude
                        * relation.amplitude
                        * relation.coherence
                        * (1.0 - relation.uncertainty)
                        * phase_alignment
                        * recency,
                )
            })
            .sum()
    }

    fn thermal_multiplier(&self, agent: usize) -> f32 {
        let Some(state) = self.thermal.thermal_state.get(agent).copied() else {
            return 1.0;
        };
        let amp = self.thermal.amplitude.get(agent).copied().unwrap_or(1.0);
        let energy = self.thermal.energy.get(agent).copied().unwrap_or(0.0);
        let temp = self.thermal.temperature.get(agent).copied().unwrap_or(1.0);
        let boltzmann = (-energy / temp.max(EPSILON)).exp().clamp(0.0, 4.0);
        (1.0 + self.config.thermal_score_gain * (state.tanh() + amp * 0.1 + boltzmann * 0.05))
            .clamp(0.25, 3.0)
    }
}

fn blend_phase(current: f32, target: f32, rate: f32) -> f32 {
    let delta = (target - current + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    (current + delta * rate.clamp(0.0, 1.0)).rem_euclid(std::f32::consts::TAU)
}

fn score_margin(scores: &[(usize, f32)]) -> f32 {
    let mut best = 0.0_f32;
    let mut second = 0.0_f32;
    for &(_, score) in scores {
        if score >= best {
            second = best;
            best = score;
        } else if score > second {
            second = score;
        }
    }
    best - second
}

fn adaptive_uncertainty(margin: f32, threshold: f32) -> f32 {
    if threshold <= EPSILON {
        return 0.0;
    }
    (1.0 - margin / threshold.max(EPSILON)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;

    #[test]
    fn realtime_update_obeys_budgets_and_stays_local() {
        let mut substrate = NativeThermoRqmEprSubstrate::new(
            NativeThermoCdtConfig {
                slices: 2,
                nodes_per_slice: 32,
                ..NativeThermoCdtConfig::default()
            },
            NativeThermoRqmConfig::default(),
            EntanglementConfig {
                create_threshold: 0.5,
                ..EntanglementConfig::default()
            },
        );
        let report = substrate.train_observed_transition_realtime(
            ObserverId(1),
            0.25,
            &[0, 1, 2, 3],
            &[32, 33, 34, 35],
            0.9,
            RealtimeUpdateConfig {
                max_relation_updates: 7,
                max_epr_observations: 3,
                max_window_nodes: 16,
                thermal_microsteps: 1,
                ..RealtimeUpdateConfig::default()
            },
        );
        assert_eq!(report.relation_updates, 7);
        assert_eq!(report.epr_observations, 3);
        assert!(report.window_nodes <= 16);
        assert!(report.thermal.mean_energy.is_finite());
    }

    #[test]
    fn structural_relation_prune_rebuilds_indices_and_growth_preserves_memory() {
        let mut substrate = NativeThermoRqmEprSubstrate::new(
            NativeThermoCdtConfig {
                slices: 2,
                nodes_per_slice: 16,
                ..NativeThermoCdtConfig::default()
            },
            NativeThermoRqmConfig::default(),
            EntanglementConfig::default(),
        );
        for target in 1..=10 {
            substrate.import_relation_state(
                ObserverId(7),
                0,
                target,
                target as f32 * 0.1,
                0.0,
                0.9,
                0.1,
                target as u64,
            );
        }
        substrate.import_relation_state(ObserverId(8), 0, 12, 1.0, 0.0, 1.0, 0.0, 20);
        assert_eq!(
            substrate.prune_observer_relations_to_budget(ObserverId(7), 3),
            7
        );
        assert_eq!(substrate.relation_count_for_observer(ObserverId(7)), 3);
        assert_eq!(substrate.relation_count(), 4);
        assert_eq!(substrate.grow_thermal_slices(2), 32);
        assert_eq!(substrate.thermal.node_count(), 64);
        let report = substrate.query(ObserverId(7), 0.0, &[0]);
        assert!(!report.candidates.is_empty());
    }

    #[test]
    fn symmetry_guided_legacy_rqm_transfers_orbit_and_gates_epr() {
        let observer = ObserverId(9);
        let mut substrate = NativeThermoRqmEprSubstrate::new(
            NativeThermoCdtConfig {
                slices: 1,
                nodes_per_slice: 8,
                temperature: 0.0,
                ..NativeThermoCdtConfig::default()
            },
            NativeThermoRqmConfig {
                thermal_steps_per_train: 0,
                thermal_steps_per_query: 0,
                ..NativeThermoRqmConfig::default()
            },
            EntanglementConfig {
                create_threshold: 0.5,
                ..EntanglementConfig::default()
            },
        );
        let orbit = [(0, 1), (2, 3), (4, 5)];
        for _ in 0..12 {
            let update = substrate.train_symmetry_guided_edges(
                observer,
                0.0,
                (0, 1),
                &orbit,
                1.0,
                1.0,
                0.01,
            );
            assert_eq!(update.orbit_updates, 2);
            assert!(update.epr_predictive_accepts > 0);
        }
        assert!(substrate
            .query(observer, 0.0, &[2])
            .candidates
            .iter()
            .any(|candidate| candidate.agent == 3));
        let conflict =
            substrate.train_symmetry_guided_edges(observer, 0.0, (0, 1), &orbit, 1.0, 1.0, 0.9);
        assert!(conflict.epr_conflicts > 0);
    }
}
