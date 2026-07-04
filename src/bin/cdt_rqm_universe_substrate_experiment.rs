use cdt_rqm_epr::cdt_graphity::{CdtGraphityConfig, CdtGraphitySubstrate};
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::relational_field::{ObserverId, RelationalFieldConfig};

#[derive(Clone)]
struct Lesson {
    observer: ObserverId,
    phase: f32,
    cue: Vec<usize>,
    effect: Vec<usize>,
}

fn main() {
    let lessons = lessons();
    let mut hardware_only = CdtGraphitySubstrate::graphity_hot_start(cdt_config());
    let mut universe = CdtRqmUniverseSubstrate::new(config());

    let hardware_initial = hardware_only.step(&[]);
    let universe_initial = universe.hardware.step(&[]);

    for epoch in 0..8 {
        for lesson in &lessons {
            hardware_only.inject_pattern(&lesson.cue, 1.0);
            hardware_only.step(&lesson.effect);

            universe.train_observed_transition(
                lesson.observer,
                lesson.phase,
                &lesson.cue,
                &lesson.effect,
                1.0,
            );
        }
        if epoch % 2 == 1 {
            let hw_score = evaluate_hardware(&hardware_only, &lessons);
            let rqm_score = evaluate_universe(&mut universe.clone(), &lessons);
            println!(
                "epoch={} hardware_score={:.1}% universe_score={:.1}% relations={} temp={:.3}",
                epoch + 1,
                hw_score * 100.0,
                rqm_score * 100.0,
                universe.relation_count(),
                universe.hardware.temperature
            );
        }
    }

    let hardware_score = evaluate_hardware(&hardware_only, &lessons);
    let universe_score = evaluate_universe(&mut universe.clone(), &lessons);
    let hardware_final = hardware_only.step(&[]);
    let universe_final = universe.hardware.step(&[]);

    println!("CDT hardware + RQM software universe substrate experiment");
    println!(
        "hardware_only: score={:.1}% temp_initial={:.3} temp_final={:.3} edges_initial={} edges_final={} regge_final={:.3} causality_violations={}",
        hardware_score * 100.0,
        hardware_initial.temperature,
        hardware_final.temperature,
        hardware_initial.active_edges,
        hardware_final.active_edges,
        hardware_final.regge_action,
        hardware_final.causality_violations
    );
    println!(
        "cdt_rqm_universe: score={:.1}% temp_initial={:.3} temp_final={:.3} edges_initial={} edges_final={} regge_final={:.3} relations={} causality_violations={}",
        universe_score * 100.0,
        universe_initial.temperature,
        universe_final.temperature,
        universe_initial.active_edges,
        universe_final.active_edges,
        universe_final.regge_action,
        universe.relation_count(),
        universe_final.causality_violations
    );
    println!(
        "lectura: {}",
        if universe_score >= hardware_score
            && universe_final.causality_violations == 0
            && universe.relation_count() > 0
        {
            "CDT funciona como hardware causal y RQM como software relacional que propone futuros observados sin romper la foliacion"
        } else {
            "la composicion CDT+RQM ejecuta, pero aun requiere calibrar el acoplamiento software-hardware"
        }
    );
}

fn evaluate_hardware(substrate: &CdtGraphitySubstrate, lessons: &[Lesson]) -> f32 {
    let mut trial = substrate.clone();
    let mut matched = 0;
    let mut total = 0;
    for lesson in lessons {
        trial.clear_activity();
        trial.inject_pattern(&lesson.cue, 1.0);
        let predicted = trial.predict_next(&lesson.cue, lesson.effect.len() * 3);
        for expected in &lesson.effect {
            total += 1;
            if predicted.iter().any(|(idx, _)| idx == expected) {
                matched += 1;
            }
        }
    }
    matched as f32 / total.max(1) as f32
}

fn evaluate_universe(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> f32 {
    let mut matched = 0;
    let mut total = 0;
    for lesson in lessons {
        substrate.hardware.clear_activity();
        substrate.hardware.inject_pattern(&lesson.cue, 1.0);
        let report = substrate.step_from_boundary(lesson.observer, lesson.phase, &lesson.cue);
        for expected in &lesson.effect {
            total += 1;
            if report.expected_from_rqm.iter().any(|idx| idx == expected) {
                matched += 1;
            }
        }
    }
    matched as f32 / total.max(1) as f32
}

fn lessons() -> Vec<Lesson> {
    vec![
        lesson(ObserverId(1), 0.0, 0, 1),
        lesson(ObserverId(1), 0.0, 4, 5),
        lesson(ObserverId(2), std::f32::consts::FRAC_PI_2, 8, 9),
    ]
}

fn lesson(
    observer: ObserverId,
    phase: f32,
    source_ordinal: usize,
    target_ordinal: usize,
) -> Lesson {
    Lesson {
        observer,
        phase,
        cue: pattern(0, source_ordinal),
        effect: pattern(1, target_ordinal),
    }
}

fn pattern(slice: usize, ordinal: usize) -> Vec<usize> {
    let base = slice * cdt_config().nodes_per_slice + ordinal;
    vec![base, base + 2, base + 4]
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: cdt_config(),
        rqm: rqm_config(),
        max_quantum_candidates: 10,
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
        slices: 5,
        nodes_per_slice: 12,
        initial_spatial_connectivity: 0.38,
        initial_temporal_connectivity: 0.24,
        target_spatial_degree: 5,
        target_temporal_degree: 3,
        target_tetrahedra_per_edge: 4,
        cooling_rate: 0.055,
        heating_rate: 0.12,
        reinforcement_rate: 0.11,
        prune_threshold: 0.055,
        max_new_edges_per_step: 6,
        seed: 8_144,
    }
}
