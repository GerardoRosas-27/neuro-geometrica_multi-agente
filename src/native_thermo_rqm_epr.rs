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

#[derive(Clone, Debug)]
pub struct NativeThermoRqmEprSubstrate {
    pub thermal: NativeThermoCdtSubstrate,
    pub config: NativeThermoRqmConfig,
    pub entanglement: EntanglementField,
    sampling_program: NativeSamplingProgram,
    relations: Vec<NativeRelation>,
    relation_lookup: HashMap<RelationKey, usize>,
    neighbor_index: HashMap<(usize, usize), Vec<usize>>,
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
            tick: 0,
        }
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn train_observed_transition(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        cause: &[usize],
        effect: &[usize],
        success: f32,
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
                    .observe_correlation(source, target, success);
            }
        }
        for _ in 0..self.config.thermal_steps_per_train {
            self.thermal.step();
        }
    }

    pub fn query(
        &mut self,
        observer: ObserverId,
        observer_phase: f32,
        seeds: &[usize],
    ) -> NativeRqmQueryReport {
        self.tick = self.tick.wrapping_add(1);

        let mut base_scores = self.relational_candidate_scores(observer, observer_phase, seeds);
        base_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        base_scores.truncate(self.config.max_candidates);
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
            self.entanglement
                .synchronize_candidates(seeds, &mut candidate_ids)
        } else if self.config.collect_query_diagnostics {
            self.entanglement.summary()
        } else {
            EntanglementReport::default()
        };
        candidate_ids.sort_unstable();
        candidate_ids.dedup();

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

        let mut candidates = candidate_ids
            .into_iter()
            .map(|agent| {
                let relational_score = base_scores
                    .iter()
                    .find(|(candidate, _)| *candidate == agent)
                    .map(|(_, score)| *score)
                    .unwrap_or_else(|| {
                        self.relational_score(observer, observer_phase, seeds, agent)
                    });
                let thermal_multiplier = if use_thermal {
                    self.thermal_multiplier(agent)
                } else {
                    1.0
                };
                NativeCandidateScore {
                    agent,
                    score: relational_score * thermal_multiplier,
                    relational_score,
                    thermal_multiplier,
                }
            })
            .filter(|candidate| candidate.score > EPSILON)
            .collect::<Vec<_>>();

        candidates.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.agent.cmp(&b.agent))
        });
        candidates.truncate(self.config.max_candidates);

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
        &self,
        observer: ObserverId,
        observer_phase: f32,
        seeds: &[usize],
    ) -> Vec<(usize, f32)> {
        let mut out = Vec::new();
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
                if let Some((_, existing)) = out
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == relation.key.target)
                {
                    *existing += score;
                } else {
                    out.push((relation.key.target, score));
                }
            }
        }
        out
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
