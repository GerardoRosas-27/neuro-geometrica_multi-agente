use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const FULL_SNGA_EPOCHS: usize = 12;
const CDT_RQM_FEWSHOT_EPOCHS: usize = 0;
const NODES_PER_SLICE: usize = 28;

#[derive(Clone, Copy)]
struct BinaryLesson {
    observer: ObserverId,
    phase: f32,
    cue: usize,
    effect: usize,
    competing_effect: usize,
}

#[derive(Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    purity_sum: f32,
    leakage_sum: f32,
    margin_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.purity_sum += if total > f32::EPSILON {
            expected / total
        } else {
            0.0
        };
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
    }

    fn accuracy(&self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    fn purity(&self) -> f32 {
        self.purity_sum / self.cases.max(1) as f32
    }

    fn leakage(&self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn margin(&self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let lessons = lessons();
    let mut snga = SimplicialNetwork::grid_3d(snga_config(), 2);
    train_snga(&mut snga, &lessons, FULL_SNGA_EPOCHS);
    let snga_metrics = evaluate_snga(&snga, &lessons);
    let causal_edges = snga.causal_edges_snapshot();

    let mut cdt_rqm_scratch = CdtRqmUniverseSubstrate::new(cdt_rqm_config());
    train_cdt_rqm(&mut cdt_rqm_scratch, &lessons, CDT_RQM_FEWSHOT_EPOCHS);

    let mut cdt_rqm_migrated = CdtRqmUniverseSubstrate::new(cdt_rqm_config());
    let migrated_edges = cdt_rqm_migrated.migrate_snga_causal_edges(
        ObserverId(900),
        0.0,
        causal_edges.iter().copied(),
        0.05,
    );
    let distilled_relations =
        distill_snga_predictions_into_cdt_rqm(&snga, &mut cdt_rqm_migrated, &lessons);
    train_cdt_rqm(&mut cdt_rqm_migrated, &lessons, CDT_RQM_FEWSHOT_EPOCHS);

    let scratch_metrics = evaluate_cdt_rqm(&cdt_rqm_scratch, &lessons);
    let migrated_metrics = evaluate_cdt_rqm(&cdt_rqm_migrated, &lessons);
    let scratch_hardware = cdt_rqm_scratch.hardware.step(&[]);
    let migrated_hardware = cdt_rqm_migrated.hardware.step(&[]);

    println!("CDT-RQM migration benchmark");
    println!(
        "lessons={} snga_epochs={} cdt_rqm_fewshot_epochs={} snga_causal_edges={} migrated_temporal_edges={} distilled_temporal_edges={} migrated_rqm_relations={}",
        lessons.len(),
        FULL_SNGA_EPOCHS,
        CDT_RQM_FEWSHOT_EPOCHS,
        causal_edges.len(),
        migrated_edges,
        distilled_relations,
        cdt_rqm_migrated.relation_count()
    );
    print_metrics("snga_previous_full", &snga_metrics);
    print_metrics("cdt_rqm_scratch_fewshot", &scratch_metrics);
    print_metrics("cdt_rqm_migrated_fewshot", &migrated_metrics);
    println!(
        "scratch_hardware: temp={:.3} active_edges={} regge={:.3} causality_violations={}",
        scratch_hardware.temperature,
        scratch_hardware.active_edges,
        scratch_hardware.regge_action,
        scratch_hardware.causality_violations
    );
    println!(
        "migrated_hardware: temp={:.3} active_edges={} regge={:.3} causality_violations={}",
        migrated_hardware.temperature,
        migrated_hardware.active_edges,
        migrated_hardware.regge_action,
        migrated_hardware.causality_violations
    );
    println!(
        "lectura: {}",
        if migrated_metrics.accuracy() >= scratch_metrics.accuracy()
            && migrated_metrics.leakage() <= scratch_metrics.leakage()
            && migrated_hardware.causality_violations == 0
        {
            "la migracion SNGA->CDT-RQM conserva conocimiento causal util y arranca sin entrenamiento adicional"
        } else {
            "la migracion funciona, pero no supera aun al entrenamiento CDT-RQM desde cero"
        }
    );
}

fn train_snga(network: &mut SimplicialNetwork, lessons: &[BinaryLesson], epochs: usize) {
    for _ in 0..epochs {
        for lesson in lessons {
            let cue = cue_pattern(lesson.cue);
            let effect = effect_pattern(lesson.effect);
            let mut fused = cue.clone();
            fused.extend(effect.iter().copied());
            fused.sort_unstable();
            fused.dedup();
            network.learn_transition(&cue, &effect);
            network.reinforce_coactivation_if_useful(&fused, 0.04, 0.92);
        }
    }
}

fn train_cdt_rqm(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[BinaryLesson], epochs: usize) {
    for _ in 0..epochs {
        for lesson in lessons {
            substrate.hardware.clear_activity();
            substrate.train_observed_transition(
                lesson.observer,
                lesson.phase,
                &cue_pattern(lesson.cue),
                &effect_pattern(lesson.effect),
                1.0,
            );
        }
    }
}

fn distill_snga_predictions_into_cdt_rqm(
    snga: &SimplicialNetwork,
    substrate: &mut CdtRqmUniverseSubstrate,
    lessons: &[BinaryLesson],
) -> usize {
    let mut distilled = 0;
    for lesson in lessons {
        let cue = cue_pattern(lesson.cue);
        let prediction = snga.predict_from(&cue, 16);
        for (target, score) in prediction {
            if score <= 0.0 || !effect_pattern(lesson.effect).contains(&target) {
                continue;
            }
            for &source in &cue {
                for _ in 0..4 {
                    substrate.software.reinforce_relation(
                        lesson.observer,
                        source,
                        target,
                        lesson.phase,
                        score.min(1.0),
                    );
                }
                if substrate
                    .hardware
                    .reinforce_temporal_link(source, target, score.min(1.0))
                {
                    distilled += 1;
                }
            }
        }
    }
    distilled
}

fn evaluate_snga(network: &SimplicialNetwork, lessons: &[BinaryLesson]) -> Metrics {
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let prediction = network.predict_from(&cue_pattern(lesson.cue), 64);
        let expected = score_prediction(&prediction, &effect_pattern(lesson.effect));
        let distractor = score_prediction(&prediction, &effect_pattern(lesson.competing_effect));
        metrics.record(expected, distractor);
    }
    metrics
}

fn evaluate_cdt_rqm(substrate: &CdtRqmUniverseSubstrate, lessons: &[BinaryLesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let cue = cue_pattern(lesson.cue);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(lesson.observer, lesson.phase, &cue);
        let expected = score_collapse(&report.collapse, &effect_pattern(lesson.effect));
        let distractor = score_collapse(&report.collapse, &effect_pattern(lesson.competing_effect));
        metrics.record(expected, distractor);
    }
    metrics
}

fn score_prediction(prediction: &[(usize, f32)], targets: &[usize]) -> f32 {
    prediction
        .iter()
        .filter(|(idx, _)| targets.contains(idx))
        .map(|(_, score)| *score)
        .sum()
}

fn score_collapse(report: &snga::relational_field::CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn print_metrics(label: &str, metrics: &Metrics) {
    println!(
        "{}: accuracy={:.1}% purity={:.1}% leakage={:.1}% margin={:.3}",
        label,
        metrics.accuracy() * 100.0,
        metrics.purity() * 100.0,
        metrics.leakage() * 100.0,
        metrics.margin()
    );
}

fn cue_pattern(ordinal: usize) -> Vec<usize> {
    pattern(0, ordinal)
}

fn effect_pattern(ordinal: usize) -> Vec<usize> {
    pattern(1, ordinal)
}

fn pattern(slice: usize, ordinal: usize) -> Vec<usize> {
    let base = slice * NODES_PER_SLICE + ordinal;
    vec![base, base + 1, base + 2]
}

fn lessons() -> Vec<BinaryLesson> {
    vec![
        lesson(ObserverId(1), 0.0, 0, 1, 5),
        lesson(ObserverId(1), 0.0, 4, 5, 1),
        lesson(ObserverId(2), std::f32::consts::FRAC_PI_2, 8, 9, 13),
        lesson(ObserverId(2), std::f32::consts::FRAC_PI_2, 10, 13, 9),
        lesson(ObserverId(3), std::f32::consts::PI, 14, 17, 21),
        lesson(ObserverId(3), std::f32::consts::PI, 18, 21, 17),
    ]
}

fn lesson(
    observer: ObserverId,
    phase: f32,
    cue: usize,
    effect: usize,
    competing_effect: usize,
) -> BinaryLesson {
    BinaryLesson {
        observer,
        phase,
        cue,
        effect,
        competing_effect,
    }
}

fn cdt_rqm_config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: cdt_config(),
        rqm: rqm_config(),
        max_quantum_candidates: 12,
        rqm_feedback_gain: 0.40,
    }
}

fn rqm_config() -> RelationalFieldConfig {
    RelationalFieldConfig {
        amplitude_learning_rate: 0.09,
        phase_learning_rate: 0.22,
        coherence_learning_rate: 0.12,
        uncertainty_learning_rate: 0.10,
        amplitude_decay: 0.001,
        coherence_decay: 0.0005,
        uncertainty_recovery: 0.002,
        activation_threshold: 0.025,
    }
}

fn cdt_config() -> CdtGraphityConfig {
    CdtGraphityConfig {
        slices: 4,
        nodes_per_slice: NODES_PER_SLICE,
        initial_spatial_connectivity: 0.30,
        initial_temporal_connectivity: 0.16,
        target_spatial_degree: 5,
        target_temporal_degree: 3,
        target_tetrahedra_per_edge: 4,
        cooling_rate: 0.055,
        heating_rate: 0.12,
        reinforcement_rate: 0.11,
        prune_threshold: 0.055,
        max_new_edges_per_step: 8,
        seed: 10_144,
    }
}

fn snga_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 32,
        height: 12,
        spacing: 8.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 40,
        inhibition_decay: 0.05,
        max_spikes_per_step: 96,
        local_inhibition_decay: 0.70,
        refractory_ticks: 1,
        rhythm_period: 16,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 128,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.075,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 10_145,
    }
}
