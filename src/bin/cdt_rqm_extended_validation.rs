use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const SNGA_EPOCHS: usize = 16;
const NODES_PER_SLICE: usize = 64;
const ANNEAL_ATTEMPTS: usize = 64;

#[derive(Clone, Copy)]
struct Lesson {
    observer: ObserverId,
    phase: f32,
    cue: usize,
    effect: usize,
    competing_effect: usize,
}

#[derive(Default, Clone, Copy)]
struct MemoryMetrics {
    cases: usize,
    correct: usize,
    purity_sum: f32,
    leakage_sum: f32,
    margin_sum: f32,
}

impl MemoryMetrics {
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

    fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    fn purity(self) -> f32 {
        self.purity_sum / self.cases.max(1) as f32
    }

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let lessons = lessons();
    let mut snga = SimplicialNetwork::grid_3d(snga_config(), 2);
    train_snga(&mut snga, &lessons);
    let snga_metrics = evaluate_snga(&snga, &lessons);
    let snga_stats = snga.plasticity_stats();

    let mut cdt_rqm = CdtRqmUniverseSubstrate::new(cdt_rqm_config());
    let migrated_temporal =
        cdt_rqm.migrate_snga_causal_edges(ObserverId(900), 0.0, snga.causal_edges_snapshot(), 0.05);
    let distilled_temporal = distill_snga_predictions(&snga, &mut cdt_rqm, &lessons);
    let before_metrics = evaluate_cdt_rqm(&cdt_rqm, &lessons);
    let before_regge = cdt_rqm.hardware.regge_action();
    let before_edges = cdt_rqm
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count();

    let validation = validation_set(&lessons);
    let anneal_report = cdt_rqm.anneal_after_migration(&validation, ANNEAL_ATTEMPTS);
    let after_metrics = evaluate_cdt_rqm(&cdt_rqm, &lessons);
    let after_regge = cdt_rqm.hardware.regge_action();
    let after_edges = cdt_rqm
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count();

    println!("CDT-RQM extended migrated validation");
    println!(
        "lessons={} observers={} snga_epochs={} anneal_attempts={} accepted={}",
        lessons.len(),
        4,
        SNGA_EPOCHS,
        anneal_report.attempts,
        anneal_report.accepted
    );
    println!(
        "migration: snga_causal_edges={} snga_associative_edges={} migrated_temporal_edges={} distilled_temporal_edges={} rqm_relations={}",
        snga_stats.causal_edges,
        snga_stats.associative_edges,
        migrated_temporal,
        distilled_temporal,
        cdt_rqm.relation_count()
    );
    print_metrics("snga_trained", snga_metrics);
    print_metrics("cdt_rqm_migrated_before_anneal", before_metrics);
    print_metrics("cdt_rqm_migrated_after_anneal", after_metrics);
    println!(
        "geometry: regge {:.3} -> {:.3} active_edges {} -> {} compression={:.1}% causality_violations={}",
        before_regge,
        after_regge,
        before_edges,
        after_edges,
        (1.0 - after_edges as f32 / before_edges.max(1) as f32) * 100.0,
        cdt_rqm.hardware.causality_violations()
    );
    println!(
        "memory_delta_vs_snga: accuracy={:.1}% purity={:.1}% leakage={:.1}%",
        (after_metrics.accuracy() - snga_metrics.accuracy()) * 100.0,
        (after_metrics.purity() - snga_metrics.purity()) * 100.0,
        (after_metrics.leakage() - snga_metrics.leakage()) * 100.0
    );
    println!(
        "lectura: {}",
        if after_metrics.accuracy() >= snga_metrics.accuracy()
            && after_metrics.leakage() <= snga_metrics.leakage() + 0.001
            && after_edges < before_edges
            && cdt_rqm.hardware.causality_violations() == 0
        {
            "CDT-RQM revalida la memoria SNGA y comprime la geometria causal sin degradar fuga"
        } else {
            "CDT-RQM migra y comprime parcialmente, pero aun requiere ajustar annealing para igualar SNGA"
        }
    );
}

fn train_snga(network: &mut SimplicialNetwork, lessons: &[Lesson]) {
    for _ in 0..SNGA_EPOCHS {
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

fn distill_snga_predictions(
    snga: &SimplicialNetwork,
    substrate: &mut CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
) -> usize {
    let mut distilled = 0;
    for lesson in lessons {
        let cue = cue_pattern(lesson.cue);
        for (target, score) in snga.predict_from(&cue, 24) {
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

fn evaluate_snga(network: &SimplicialNetwork, lessons: &[Lesson]) -> MemoryMetrics {
    let mut metrics = MemoryMetrics::default();
    for lesson in lessons {
        let prediction = network.predict_from(&cue_pattern(lesson.cue), 96);
        let expected = score_prediction(&prediction, &effect_pattern(lesson.effect));
        let distractor = score_prediction(&prediction, &effect_pattern(lesson.competing_effect));
        metrics.record(expected, distractor);
    }
    metrics
}

fn evaluate_cdt_rqm(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> MemoryMetrics {
    let mut trial = substrate.clone();
    let mut metrics = MemoryMetrics::default();
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

fn validation_set(
    lessons: &[Lesson],
) -> Vec<(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)> {
    lessons
        .iter()
        .map(|lesson| {
            (
                lesson.observer,
                lesson.phase,
                cue_pattern(lesson.cue),
                effect_pattern(lesson.effect),
                effect_pattern(lesson.competing_effect),
            )
        })
        .collect()
}

fn print_metrics(label: &str, metrics: MemoryMetrics) {
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

fn lessons() -> Vec<Lesson> {
    let mut out = Vec::new();
    let phases = [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        -std::f32::consts::FRAC_PI_2,
    ];
    for group in 0..4 {
        let observer = ObserverId(group + 1);
        let phase = phases[group];
        for offset in 0..5 {
            let cue = group * 12 + offset * 2;
            let effect = group * 12 + offset * 2 + 1;
            let competing = group * 12 + ((offset + 2) % 5) * 2 + 1;
            out.push(lesson(observer, phase, cue, effect, competing));
        }
    }
    out
}

fn lesson(
    observer: ObserverId,
    phase: f32,
    cue: usize,
    effect: usize,
    competing_effect: usize,
) -> Lesson {
    Lesson {
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
        max_quantum_candidates: 16,
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
        initial_spatial_connectivity: 0.22,
        initial_temporal_connectivity: 0.10,
        target_spatial_degree: 5,
        target_temporal_degree: 3,
        target_tetrahedra_per_edge: 4,
        cooling_rate: 0.055,
        heating_rate: 0.12,
        reinforcement_rate: 0.11,
        prune_threshold: 0.055,
        max_new_edges_per_step: 8,
        seed: 12_144,
    }
}

fn snga_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 32,
        height: 16,
        spacing: 8.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 64,
        inhibition_decay: 0.05,
        max_spikes_per_step: 128,
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
        seed: 12_145,
    }
}
