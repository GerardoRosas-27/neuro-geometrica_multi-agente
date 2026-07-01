use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const SNGA_EPOCHS: usize = 12;
const NODES_PER_SLICE: usize = 28;
const ANNEAL_ATTEMPTS: usize = 32;

#[derive(Clone, Copy)]
struct Lesson {
    observer: ObserverId,
    phase: f32,
    cue: usize,
    effect: usize,
    competing_effect: usize,
}

fn main() {
    let lessons = lessons();
    let mut snga = SimplicialNetwork::grid_3d(snga_config(), 2);
    train_snga(&mut snga, &lessons);

    let mut substrate = CdtRqmUniverseSubstrate::new(cdt_rqm_config());
    let migrated_temporal = substrate.migrate_snga_causal_edges(
        ObserverId(900),
        0.0,
        snga.causal_edges_snapshot(),
        0.05,
    );
    let distilled_relations = distill_predictions(&snga, &mut substrate, &lessons);
    let validation = validation_set(&lessons);
    let report = substrate.anneal_after_migration(&validation, ANNEAL_ATTEMPTS);

    println!("CDT-RQM post-migration Graphity annealing benchmark");
    println!(
        "lessons={} anneal_attempts={} accepted={} migrated_temporal_edges={} distilled_temporal_edges={} relations={}",
        lessons.len(),
        report.attempts,
        report.accepted,
        migrated_temporal,
        distilled_relations,
        substrate.relation_count()
    );
    println!(
        "memory: accuracy {:.1}% -> {:.1}% leakage {:.1}% -> {:.1}%",
        report.initial_accuracy * 100.0,
        report.final_accuracy * 100.0,
        report.initial_leakage * 100.0,
        report.final_leakage * 100.0
    );
    println!(
        "geometry: regge {:.3} -> {:.3} active_edges {} -> {} causality_violations={}",
        report.initial_regge,
        report.final_regge,
        report.initial_edges,
        report.final_edges,
        report.causality_violations
    );
    println!(
        "lectura: {}",
        if report.final_accuracy >= report.initial_accuracy
            && report.final_leakage <= report.initial_leakage
            && report.final_regge < report.initial_regge
            && report.causality_violations == 0
        {
            "annealing Graphity comprime la geometria migrada sin perder memoria causal"
        } else {
            "annealing conserva la causalidad, pero requiere ajustar criterios para mejorar memoria y geometria a la vez"
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

fn distill_predictions(
    snga: &SimplicialNetwork,
    substrate: &mut CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
) -> usize {
    let mut distilled = 0;
    for lesson in lessons {
        let cue = cue_pattern(lesson.cue);
        for (target, score) in snga.predict_from(&cue, 16) {
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
        seed: 11_144,
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
        seed: 11_145,
    }
}
