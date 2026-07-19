//! RQM clásico de amplitud/fase guiado por simetría y capa cognitiva separada.
//!
//! No simula un líquido de espines cuántico físico. Usa osciladores de fase y
//! asociaciones EPR clásicas como modelo computacional falsable.

use crate::entanglement::{EntanglementConfig, EntanglementField};
use crate::matrix_free_cognitive_substrate::LatentConceptId;
use crate::relational_field::ObserverId;
use std::collections::BTreeMap;

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
pub struct SymmetryGuidedRqmConfig {
    pub learning_rate: f64,
    pub phase_learning_rate: f64,
    pub eligibility_decay: f64,
    pub orbit_transfer_gain: f64,
    pub minimum_symmetry_confidence: f64,
    pub minimum_exposures: u64,
    pub max_prediction_error: f64,
    pub min_coherence: f64,
    pub abstention_score: f64,
    pub epr: EntanglementConfig,
}

impl Default for SymmetryGuidedRqmConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.35,
            phase_learning_rate: 0.20,
            eligibility_decay: 0.80,
            orbit_transfer_gain: 1.0,
            minimum_symmetry_confidence: 0.75,
            minimum_exposures: 12,
            max_prediction_error: 0.025,
            min_coherence: 0.90,
            abstention_score: 0.08,
            epr: EntanglementConfig {
                create_threshold: 0.75,
                ..EntanglementConfig::default()
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RqmRelationKey {
    pub observer: usize,
    pub source: LatentConceptId,
    pub target: LatentConceptId,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RqmPhaseRelationState {
    pub amplitude: f64,
    pub phase: f64,
    pub coherence: f64,
    pub uncertainty: f64,
    pub eligibility: f64,
    pub prediction_error: f64,
    pub exposures: u64,
    pub consolidated: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RqmLearningReport {
    pub observed_delta: f64,
    pub orbit_updates: usize,
    pub epr_accepts: usize,
    pub epr_conflicts: usize,
    pub observed: RqmPhaseRelationState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RqmCandidate {
    pub concept: LatentConceptId,
    pub score: f64,
    pub amplitude: f64,
    pub phase_alignment: f64,
    pub epr_bonus: f64,
}

#[derive(Clone, Debug)]
pub struct SymmetryGuidedRqmEprField {
    pub config: SymmetryGuidedRqmConfig,
    pub entanglement: EntanglementField,
    relations: BTreeMap<RqmRelationKey, RqmPhaseRelationState>,
}

impl SymmetryGuidedRqmEprField {
    pub fn new(config: SymmetryGuidedRqmConfig) -> Self {
        let config = sanitize_config(config);
        Self {
            entanglement: EntanglementField::new(config.epr),
            config,
            relations: BTreeMap::new(),
        }
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn relation_entries(
        &self,
    ) -> impl Iterator<Item = (RqmRelationKey, RqmPhaseRelationState)> + '_ {
        self.relations.iter().map(|(key, state)| (*key, *state))
    }

    pub fn import_relation(&mut self, key: RqmRelationKey, state: RqmPhaseRelationState) {
        self.relations.insert(key, state);
    }

    pub fn relation(
        &self,
        observer: ObserverId,
        source: LatentConceptId,
        target: LatentConceptId,
    ) -> Option<RqmPhaseRelationState> {
        self.relations
            .get(&RqmRelationKey {
                observer: observer.0,
                source,
                target,
            })
            .copied()
    }

    pub fn learn_transition(
        &mut self,
        observer: ObserverId,
        source: LatentConceptId,
        target: LatentConceptId,
        context_phase: f64,
        desired_strength: f64,
        symmetry_confidence: f64,
        orbit: &[(LatentConceptId, LatentConceptId)],
    ) -> RqmLearningReport {
        let desired_strength = desired_strength.clamp(0.0, 1.0);
        let observed_key = RqmRelationKey {
            observer: observer.0,
            source,
            target,
        };
        let (observed, observed_delta) =
            self.update_relation(observed_key, context_phase, desired_strength, 1.0);
        let mut report = RqmLearningReport {
            observed_delta,
            observed,
            ..RqmLearningReport::default()
        };
        self.update_predictive_epr(observed_key, observed, desired_strength, &mut report);

        let symmetry_confidence = symmetry_confidence.clamp(0.0, 1.0);
        if symmetry_confidence >= self.config.minimum_symmetry_confidence {
            let transfer_scale = self.config.orbit_transfer_gain * symmetry_confidence;
            for &(orbit_source, orbit_target) in orbit {
                let key = RqmRelationKey {
                    observer: observer.0,
                    source: orbit_source,
                    target: orbit_target,
                };
                if key == observed_key || orbit_source == orbit_target {
                    continue;
                }
                let (state, _) =
                    self.update_relation(key, context_phase, desired_strength, transfer_scale);
                self.update_predictive_epr(key, state, desired_strength, &mut report);
                report.orbit_updates += 1;
            }
        }
        report
    }

    pub fn query(
        &self,
        observer: ObserverId,
        source: LatentConceptId,
        context_phase: f64,
    ) -> Vec<RqmCandidate> {
        let mut candidates = self
            .relations
            .iter()
            .filter(|(key, _)| key.observer == observer.0 && key.source == source)
            .filter_map(|(key, state)| {
                let phase_alignment =
                    0.5 + 0.5 * circular_difference(state.phase, context_phase).cos();
                let epr_bonus = if self.entanglement.has_active_link(
                    epr_node(observer, key.source),
                    epr_node(observer, key.target),
                ) {
                    0.15
                } else {
                    0.0
                };
                let score = state.amplitude
                    * (0.25 + 0.75 * state.coherence)
                    * (1.0 - state.uncertainty).clamp(0.0, 1.0)
                    * (0.25 + 0.75 * phase_alignment)
                    + epr_bonus;
                (score >= self.config.abstention_score).then_some(RqmCandidate {
                    concept: key.target,
                    score,
                    amplitude: state.amplitude,
                    phase_alignment,
                    epr_bonus,
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.concept.cmp(&right.concept))
        });
        candidates
    }

    fn update_relation(
        &mut self,
        key: RqmRelationKey,
        context_phase: f64,
        desired_strength: f64,
        learning_scale: f64,
    ) -> (RqmPhaseRelationState, f64) {
        let relation = self.relations.entry(key).or_default();
        let error = desired_strength - relation.amplitude;
        relation.eligibility = self.config.eligibility_decay * relation.eligibility
            + (1.0 - self.config.eligibility_decay);
        let delta = self.config.learning_rate
            * learning_scale
            * desired_strength.max(0.25)
            * error
            * relation.eligibility;
        relation.amplitude = (relation.amplitude + delta).clamp(0.0, 1.0);
        relation.phase = blend_phase(
            relation.phase,
            context_phase,
            self.config.phase_learning_rate * learning_scale,
        );
        relation.prediction_error = (desired_strength - relation.amplitude).abs();
        let reliability = 1.0 - relation.prediction_error.clamp(0.0, 1.0);
        relation.coherence +=
            self.config.learning_rate * learning_scale * (reliability - relation.coherence);
        relation.coherence = relation.coherence.clamp(0.0, 1.0);
        relation.uncertainty += self.config.learning_rate
            * learning_scale
            * (relation.prediction_error - relation.uncertainty);
        relation.uncertainty = relation.uncertainty.clamp(0.0, 1.0);
        relation.exposures += 1;
        relation.consolidated = relation.exposures >= self.config.minimum_exposures
            && relation.prediction_error <= self.config.max_prediction_error
            && relation.coherence >= self.config.min_coherence;
        (*relation, delta)
    }

    fn update_predictive_epr(
        &mut self,
        key: RqmRelationKey,
        relation: RqmPhaseRelationState,
        desired_strength: f64,
        report: &mut RqmLearningReport,
    ) {
        let epr = self.entanglement.observe_predictive_correlation(
            epr_node(ObserverId(key.observer), key.source),
            epr_node(ObserverId(key.observer), key.target),
            desired_strength as f32,
            relation.prediction_error as f32,
            self.config.max_prediction_error.max(0.10) as f32,
        );
        report.epr_accepts += usize::from(epr.accepted);
        report.epr_conflicts += usize::from(epr.conflict);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RelationalCognitiveConfig {
    pub beam_width: usize,
    pub minimum_confidence: f64,
}

impl Default for RelationalCognitiveConfig {
    fn default() -> Self {
        Self {
            beam_width: 16,
            minimum_confidence: 0.08,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveInference {
    pub path: Vec<LatentConceptId>,
    pub confidence: f64,
    pub abstraction_depth: usize,
}

#[derive(Clone, Debug)]
pub struct RelationalCognitiveLayer {
    pub workspace: SymmetryGuidedRqmEprField,
    pub config: RelationalCognitiveConfig,
}

impl RelationalCognitiveLayer {
    pub fn new(workspace: SymmetryGuidedRqmEprField, config: RelationalCognitiveConfig) -> Self {
        Self { workspace, config }
    }

    /// La cognición operacional aparece aquí como composición y selección de
    /// relaciones; la simetría sólo controló cómo se aprendieron.
    pub fn infer(
        &self,
        observer: ObserverId,
        source: LatentConceptId,
        context_phase: f64,
        max_hops: usize,
    ) -> Option<CognitiveInference> {
        let mut beam = vec![CognitiveInference {
            path: vec![source],
            confidence: 1.0,
            abstraction_depth: 0,
        }];
        let mut best: Option<CognitiveInference> = None;
        for depth in 1..=max_hops {
            let mut next = Vec::new();
            for inference in &beam {
                let current = *inference.path.last()?;
                for candidate in self.workspace.query(observer, current, context_phase) {
                    if inference.path.contains(&candidate.concept) {
                        continue;
                    }
                    let confidence = inference.confidence * candidate.score.min(1.0);
                    if confidence < self.config.minimum_confidence {
                        continue;
                    }
                    let mut path = inference.path.clone();
                    path.push(candidate.concept);
                    next.push(CognitiveInference {
                        path,
                        confidence,
                        abstraction_depth: depth,
                    });
                }
            }
            next.sort_by(|left, right| {
                right
                    .confidence
                    .total_cmp(&left.confidence)
                    .then_with(|| left.path.cmp(&right.path))
            });
            next.truncate(self.config.beam_width.max(1));
            if let Some(candidate) = next.first() {
                best = Some(candidate.clone());
            }
            if next.is_empty() {
                break;
            }
            beam = next;
        }
        best
    }
}

fn sanitize_config(config: SymmetryGuidedRqmConfig) -> SymmetryGuidedRqmConfig {
    SymmetryGuidedRqmConfig {
        learning_rate: config.learning_rate.max(0.0),
        phase_learning_rate: config.phase_learning_rate.clamp(0.0, 1.0),
        eligibility_decay: config.eligibility_decay.clamp(0.0, 1.0),
        orbit_transfer_gain: config.orbit_transfer_gain.max(0.0),
        minimum_symmetry_confidence: config.minimum_symmetry_confidence.clamp(0.0, 1.0),
        minimum_exposures: config.minimum_exposures.max(1),
        max_prediction_error: config.max_prediction_error.max(EPSILON),
        min_coherence: config.min_coherence.clamp(0.0, 1.0),
        abstention_score: config.abstention_score.max(0.0),
        ..config
    }
}

fn circular_difference(left: f64, right: f64) -> f64 {
    (left - right + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

fn blend_phase(current: f64, target: f64, amount: f64) -> f64 {
    (current + amount.clamp(0.0, 1.0) * circular_difference(target, current))
        .rem_euclid(std::f64::consts::TAU)
}

fn epr_node(observer: ObserverId, concept: LatentConceptId) -> usize {
    observer.0.wrapping_mul(1_000_003).wrapping_add(concept.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train(
        field: &mut SymmetryGuidedRqmEprField,
        source: usize,
        target: usize,
        phase: f64,
        symmetry: f64,
        orbit: &[(LatentConceptId, LatentConceptId)],
    ) {
        for _ in 0..40 {
            field.learn_transition(
                ObserverId(1),
                LatentConceptId(source),
                LatentConceptId(target),
                phase,
                1.0,
                symmetry,
                orbit,
            );
        }
    }

    #[test]
    fn symmetry_alone_does_not_create_cognitive_content() {
        let layer = RelationalCognitiveLayer::new(
            SymmetryGuidedRqmEprField::new(SymmetryGuidedRqmConfig::default()),
            RelationalCognitiveConfig::default(),
        );
        assert!(layer
            .infer(ObserverId(1), LatentConceptId(0), 0.0, 3)
            .is_none());
    }

    #[test]
    fn symmetry_guides_orbit_transfer_but_local_learning_still_works() {
        let orbit = [
            (LatentConceptId(0), LatentConceptId(1)),
            (LatentConceptId(2), LatentConceptId(3)),
        ];
        let mut symmetric = SymmetryGuidedRqmEprField::new(SymmetryGuidedRqmConfig::default());
        train(&mut symmetric, 0, 1, 0.0, 1.0, &orbit);
        assert_eq!(
            symmetric.query(ObserverId(1), LatentConceptId(2), 0.0)[0].concept,
            LatentConceptId(3)
        );

        let mut local = SymmetryGuidedRqmEprField::new(SymmetryGuidedRqmConfig::default());
        train(&mut local, 0, 1, 0.0, 0.0, &orbit);
        assert!(local
            .query(ObserverId(1), LatentConceptId(2), 0.0)
            .is_empty());
        assert!(!local
            .query(ObserverId(1), LatentConceptId(0), 0.0)
            .is_empty());
    }

    #[test]
    fn phase_interference_changes_retrieval_without_changing_memory() {
        let mut field = SymmetryGuidedRqmEprField::new(SymmetryGuidedRqmConfig::default());
        train(&mut field, 0, 1, 0.0, 0.0, &[]);
        let aligned = field.query(ObserverId(1), LatentConceptId(0), 0.0)[0].score;
        let opposed = field.query(ObserverId(1), LatentConceptId(0), std::f64::consts::PI)[0].score;
        assert!(aligned > opposed);
    }

    #[test]
    fn signed_prediction_error_can_depress_a_relation() {
        let mut field = SymmetryGuidedRqmEprField::new(SymmetryGuidedRqmConfig::default());
        train(&mut field, 0, 1, 0.0, 0.0, &[]);
        let before = field
            .relation(ObserverId(1), LatentConceptId(0), LatentConceptId(1))
            .unwrap()
            .amplitude;
        for _ in 0..40 {
            field.learn_transition(
                ObserverId(1),
                LatentConceptId(0),
                LatentConceptId(1),
                0.0,
                0.0,
                0.0,
                &[],
            );
        }
        let after = field
            .relation(ObserverId(1), LatentConceptId(0), LatentConceptId(1))
            .unwrap()
            .amplitude;
        assert!(after < before * 0.1);
    }

    #[test]
    fn abstract_cognitive_layer_composes_rqm_relations() {
        let mut field = SymmetryGuidedRqmEprField::new(SymmetryGuidedRqmConfig::default());
        train(&mut field, 0, 1, 0.25, 0.0, &[]);
        train(&mut field, 1, 2, 0.25, 0.0, &[]);
        let layer = RelationalCognitiveLayer::new(field, RelationalCognitiveConfig::default());
        let inference = layer
            .infer(ObserverId(1), LatentConceptId(0), 0.25, 2)
            .unwrap();
        assert_eq!(
            inference.path,
            vec![LatentConceptId(0), LatentConceptId(1), LatentConceptId(2)]
        );
        assert_eq!(inference.abstraction_depth, 2);
    }
}
