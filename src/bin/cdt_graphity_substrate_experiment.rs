use cdt_rqm_epr::cdt_graphity::{CdtGraphityConfig, CdtGraphitySubstrate};

#[derive(Clone)]
struct CausalLesson {
    cue: Vec<usize>,
    next: Vec<usize>,
}

fn main() {
    let mut substrate = CdtGraphitySubstrate::graphity_hot_start(config());
    let initial_report = substrate.step(&[]);
    let lessons = lessons();

    let mut reports = Vec::new();
    for epoch in 0..12 {
        for lesson in &lessons {
            substrate.inject_pattern(&lesson.cue, 1.0);
            let report = substrate.step(&lesson.next);
            reports.push(report);
        }
        if epoch % 4 == 3 {
            let latest = reports.last().expect("at least one report");
            println!(
                "epoch={} free_energy={:.3} regge={:.3} pred_error={:.3} temp={:.3} active_edges={} spatial={} temporal={} tetrahedra={} pruned={} proposed={} causality_violations={}",
                epoch + 1,
                latest.free_energy,
                latest.regge_action,
                latest.prediction_error,
                latest.temperature,
                latest.active_edges,
                latest.spatial_edges,
                latest.temporal_edges,
                latest.tetrahedra,
                latest.pruned_edges,
                latest.proposed_edges,
                latest.causality_violations
            );
        }
    }

    let final_report = reports.last().expect("training reports");
    println!("CDT + Graphity substrate experiment");
    println!(
        "initial: free_energy={:.3} regge={:.3} temp={:.3} active_edges={} spatial={} temporal={} tetrahedra={} causality_violations={}",
        initial_report.free_energy,
        initial_report.regge_action,
        initial_report.temperature,
        initial_report.active_edges,
        initial_report.spatial_edges,
        initial_report.temporal_edges,
        initial_report.tetrahedra,
        initial_report.causality_violations
    );
    println!(
        "final: free_energy={:.3} regge={:.3} pred_error={:.3} temp={:.3} active_nodes={} active_edges={} spatial={} temporal={} tetrahedra={} causality_violations={}",
        final_report.free_energy,
        final_report.regge_action,
        final_report.prediction_error,
        final_report.temperature,
        final_report.active_nodes,
        final_report.active_edges,
        final_report.spatial_edges,
        final_report.temporal_edges,
        final_report.tetrahedra,
        final_report.causality_violations
    );

    let prediction_score = evaluate_predictions(&substrate, &lessons);
    println!(
        "prediction_score={:.1}% edge_reduction={:.1}% temperature_drop={:.1}%",
        prediction_score * 100.0,
        edge_reduction(&initial_report, final_report) * 100.0,
        ((initial_report.temperature - final_report.temperature)
            / initial_report.temperature.max(0.001))
            * 100.0
    );
    println!(
        "lectura: {}",
        if final_report.causality_violations == 0
            && prediction_score > 0.75
            && final_report.active_edges < initial_report.active_edges
        {
            "la foliacion CDT se conserva y Graphity enfria el grafo hacia una geometria causal mas local"
        } else {
            "el sustrato ejecuta reglas CDT/Graphity, pero requiere ajustar poda, propuesta o curvatura"
        }
    );
}

fn evaluate_predictions(substrate: &CdtGraphitySubstrate, lessons: &[CausalLesson]) -> f32 {
    let mut matched = 0;
    let mut total = 0;
    let mut trial = substrate.clone();
    for lesson in lessons {
        trial.inject_pattern(&lesson.cue, 1.0);
        let predicted = trial.predict_next(&lesson.cue, lesson.next.len() * 3);
        for expected in &lesson.next {
            total += 1;
            if predicted.iter().any(|(idx, _)| idx == expected) {
                matched += 1;
            }
        }
    }
    matched as f32 / total.max(1) as f32
}

fn edge_reduction(
    initial: &cdt_rqm_epr::cdt_graphity::CdtGraphityStepReport,
    final_report: &cdt_rqm_epr::cdt_graphity::CdtGraphityStepReport,
) -> f32 {
    1.0 - final_report.active_edges as f32 / initial.active_edges.max(1) as f32
}

fn lessons() -> Vec<CausalLesson> {
    vec![lesson(0, 1), lesson(4, 5), lesson(8, 9), lesson(10, 11)]
}

fn lesson(source_ordinal: usize, target_ordinal: usize) -> CausalLesson {
    CausalLesson {
        cue: pattern(0, source_ordinal),
        next: pattern(1, target_ordinal),
    }
}

fn pattern(slice: usize, ordinal: usize) -> Vec<usize> {
    let base = slice * config().nodes_per_slice + ordinal;
    vec![base, base + 2, base + 4]
}

fn config() -> CdtGraphityConfig {
    CdtGraphityConfig {
        slices: 5,
        nodes_per_slice: 16,
        initial_spatial_connectivity: 0.42,
        initial_temporal_connectivity: 0.30,
        target_spatial_degree: 5,
        target_temporal_degree: 3,
        target_tetrahedra_per_edge: 4,
        cooling_rate: 0.055,
        heating_rate: 0.12,
        reinforcement_rate: 0.11,
        prune_threshold: 0.055,
        max_new_edges_per_step: 10,
        seed: 7_311,
    }
}
