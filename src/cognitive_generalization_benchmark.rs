//! Validación escalonada de memoria, variación, composición y transferencia.
//!
//! El protocolo usa relaciones abstractas y separa recuperación de
//! generalización estructural. La órbita isomórfica del nivel 4 se proporciona
//! al sistema; se valida la transferencia, no el descubrimiento autónomo de la
//! simetría.

use crate::matrix_free_cognitive_substrate::LatentConceptId;
use crate::relational_field::ObserverId;
use crate::unified_spin_cognitive_engine::{
    KnowledgeKey, UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};
use rayon::prelude::*;
use serde::Serialize;
use std::time::Instant;

const TRAINING_PAIRS: [(usize, usize); 4] = [(0, 1), (2, 3), (4, 5), (6, 7)];
const UNSEEN_PHASE_JITTERS: [f64; 4] = [-0.06, -0.02, 0.02, 0.06];

#[derive(Clone, Copy, Debug)]
pub struct CognitiveGeneralizationConfig {
    pub trials: usize,
    pub exposures: usize,
    pub minimum_rate: f64,
}

impl Default for CognitiveGeneralizationConfig {
    fn default() -> Self {
        Self {
            trials: 24,
            exposures: 24,
            minimum_rate: 0.95,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CognitiveGeneralizationReport {
    pub trials: usize,
    pub level1_exact_memory: f64,
    pub level2_unseen_variation: f64,
    pub level3_composition: f64,
    pub level3_direct_shortcut_absent: f64,
    pub level4_isomorphic_transfer: f64,
    pub level4_transfer_absent_without_symmetry: f64,
    pub ood_abstention: f64,
    /// Tiempo de pared total del benchmark (segundos).
    pub wall_clock_seconds: f64,
    pub decision: &'static str,
}

pub fn run_cognitive_generalization_benchmark(
    config: CognitiveGeneralizationConfig,
) -> CognitiveGeneralizationReport {
    let started = Instant::now();
    let trials = config.trials.max(1);
    let exposures = config.exposures.max(1);
    let minimum_rate = config.minimum_rate.clamp(0.0, 1.0);

    // Ensayos independientes cuya fase deriva del índice de trial: se
    // calculan en paralelo y se reducen en orden. Los contadores son enteros,
    // así que el resultado es idéntico bit a bit al recorrido secuencial.
    let total = (0..trials)
        .into_par_iter()
        .map(|trial| run_generalization_trial(trial, exposures))
        .collect::<Vec<_>>()
        .into_iter()
        .fold(
            GeneralizationTrialOutcome::default(),
            |mut total, outcome| {
                total.exact_successes += outcome.exact_successes;
                total.variation_successes += outcome.variation_successes;
                total.composition_successes += outcome.composition_successes;
                total.shortcut_absent += outcome.shortcut_absent;
                total.transfer_successes += outcome.transfer_successes;
                total.transfer_control_successes += outcome.transfer_control_successes;
                total.ood_successes += outcome.ood_successes;
                total
            },
        );

    let exact_denominator = (trials * TRAINING_PAIRS.len()) as f64;
    let variation_denominator = (trials * TRAINING_PAIRS.len() * UNSEEN_PHASE_JITTERS.len()) as f64;
    let trial_denominator = trials as f64;
    let mut report = CognitiveGeneralizationReport {
        trials,
        level1_exact_memory: total.exact_successes as f64 / exact_denominator,
        level2_unseen_variation: total.variation_successes as f64 / variation_denominator,
        level3_composition: total.composition_successes as f64 / trial_denominator,
        level3_direct_shortcut_absent: total.shortcut_absent as f64 / trial_denominator,
        level4_isomorphic_transfer: total.transfer_successes as f64 / trial_denominator,
        level4_transfer_absent_without_symmetry: total.transfer_control_successes as f64
            / trial_denominator,
        ood_abstention: total.ood_successes as f64 / trial_denominator,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        decision: "cognitive_generalization_not_demonstrated",
    };
    if report.level1_exact_memory >= minimum_rate
        && report.level2_unseen_variation >= minimum_rate
        && report.level3_composition >= minimum_rate
        && report.level3_direct_shortcut_absent >= minimum_rate
        && report.level4_isomorphic_transfer >= minimum_rate
        && report.level4_transfer_absent_without_symmetry >= minimum_rate
        && report.ood_abstention >= minimum_rate
    {
        report.decision = "limited_structural_generalization_pass";
    }
    report
}

#[derive(Clone, Copy, Debug, Default)]
struct GeneralizationTrialOutcome {
    exact_successes: usize,
    variation_successes: usize,
    composition_successes: usize,
    shortcut_absent: usize,
    transfer_successes: usize,
    transfer_control_successes: usize,
    ood_successes: usize,
}

fn run_generalization_trial(trial: usize, exposures: usize) -> GeneralizationTrialOutcome {
    let mut outcome = GeneralizationTrialOutcome::default();
    let observer = ObserverId(1_200_000 + trial);
    let phase = (0.17 + trial as f64 * 0.071).rem_euclid(std::f64::consts::TAU);
    let mut engine = fixture();

    for &(source, target) in &TRAINING_PAIRS {
        let learned = engine.train_relation(
            observer,
            LatentConceptId(source),
            LatentConceptId(target),
            phase,
            1.0,
            0.0,
            &[],
            exposures,
        );
        if learned.gate.passed && top_candidate(&engine, observer, source, phase) == Some(target) {
            outcome.exact_successes += 1;
        }
        for jitter in UNSEEN_PHASE_JITTERS {
            outcome.variation_successes += usize::from(
                top_candidate(&engine, observer, source, phase + jitter) == Some(target),
            );
        }
    }

    let composition_source = LatentConceptId(10);
    let composition_middle = LatentConceptId(11);
    let composition_target = LatentConceptId(12);
    engine.train_relation(
        observer,
        composition_source,
        composition_middle,
        phase,
        1.0,
        0.0,
        &[],
        exposures,
    );
    engine.train_relation(
        observer,
        composition_middle,
        composition_target,
        phase,
        1.0,
        0.0,
        &[],
        exposures,
    );
    outcome.shortcut_absent += usize::from(
        engine
            .cognition
            .workspace
            .relation(observer, composition_source, composition_target)
            .is_none(),
    );
    outcome.composition_successes += usize::from(
        engine
            .infer(observer, composition_source, phase, 2)
            .is_some_and(|inference| {
                inference.path == vec![composition_source, composition_middle, composition_target]
            }),
    );

    let isomorphic_orbit = [
        (LatentConceptId(22), LatentConceptId(23)),
        (LatentConceptId(24), LatentConceptId(25)),
        (LatentConceptId(26), LatentConceptId(27)),
    ];
    engine.train_relation(
        observer,
        LatentConceptId(20),
        LatentConceptId(21),
        phase,
        1.0,
        1.0,
        &isomorphic_orbit,
        exposures,
    );
    outcome.transfer_successes += usize::from(isomorphic_orbit.iter().all(|&(source, target)| {
        engine.knowledge.contains_key(&KnowledgeKey {
            observer: observer.0,
            source,
            target,
        }) && top_candidate(&engine, observer, source.0, phase) == Some(target.0)
    }));
    outcome.ood_successes += usize::from(
        engine
            .infer(observer, LatentConceptId(99), phase, 2)
            .is_none(),
    );

    let mut control = fixture();
    control.train_relation(
        observer,
        LatentConceptId(20),
        LatentConceptId(21),
        phase,
        1.0,
        0.0,
        &isomorphic_orbit,
        exposures,
    );
    outcome.transfer_control_successes +=
        usize::from(isomorphic_orbit.iter().all(|&(source, target)| {
            !control.knowledge.contains_key(&KnowledgeKey {
                observer: observer.0,
                source,
                target,
            })
        }));
    outcome
}

fn fixture() -> UnifiedSpinCognitiveEngine {
    UnifiedSpinCognitiveEngine::periodic_pyrochlore(
        2,
        1,
        1,
        UnifiedSpinCognitiveConfig {
            bootstrap_cooling_steps: 120,
            cooling_steps_per_observation: 1,
            real_steps_per_observation: 0,
            ..UnifiedSpinCognitiveConfig::default()
        },
    )
    .expect("el fixture cognitivo debe ser válido")
}

fn top_candidate(
    engine: &UnifiedSpinCognitiveEngine,
    observer: ObserverId,
    source: usize,
    phase: f64,
) -> Option<usize> {
    engine
        .cognition
        .workspace
        .query(observer, LatentConceptId(source), phase)
        .first()
        .map(|candidate| candidate.concept.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_level_protocol_demonstrates_limited_structural_generalization() {
        let report = run_cognitive_generalization_benchmark(CognitiveGeneralizationConfig {
            trials: 8,
            ..CognitiveGeneralizationConfig::default()
        });
        assert_eq!(
            report.decision, "limited_structural_generalization_pass",
            "{report:#?}"
        );
    }
}
