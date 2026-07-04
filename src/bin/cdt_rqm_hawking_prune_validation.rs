use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::relational_field::{ObserverId, RelationalFieldConfig};

const NODES_PER_SLICE: usize = 128;
const ATTEMPTS: usize = 48;

#[derive(Clone, Copy)]
struct Lesson {
    observer: ObserverId,
    phase: f32,
    cue: usize,
    target: usize,
    distractor: usize,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
    margin_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
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

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let lessons = lessons();
    let validation = validation_set(&lessons);
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    substrate.enable_entanglement(epr_config());
    train(&mut substrate, &lessons);

    let before_metrics = evaluate(&substrate, &lessons);
    let before_edges = active_edges(&substrate);
    let before_regge = substrate.hardware.regge_action();

    let mut standard = substrate.clone();
    let standard_accepted = pure_standard_anneal(&mut standard, &validation, &lessons, ATTEMPTS);
    let standard_metrics = evaluate(&standard, &lessons);
    let standard_edges = active_edges(&standard);
    let standard_regge = standard.hardware.regge_action();

    let mut enhanced = substrate.clone();
    let enhanced_report = enhanced.anneal_after_migration(&validation, ATTEMPTS);
    let enhanced_metrics = evaluate(&enhanced, &lessons);
    let enhanced_edges = active_edges(&enhanced);
    let enhanced_regge = enhanced.hardware.regge_action();

    let mut hawking = substrate.clone();
    let hawking_report = hawking.hawking_radiation_after_migration(&validation, ATTEMPTS);
    let hawking_metrics = evaluate(&hawking, &lessons);
    let hawking_edges = active_edges(&hawking);
    let hawking_regge = hawking.hardware.regge_action();

    let enhanced_preserves = enhanced_metrics.accuracy() + 0.0001 >= standard_metrics.accuracy()
        && enhanced_metrics.leakage() <= standard_metrics.leakage() + 0.0001
        && enhanced.hardware.causality_violations() == 0;
    let enhanced_improves =
        enhanced_edges < standard_edges || enhanced_regge + 0.001 < standard_regge;
    let keep = enhanced_preserves && enhanced_improves;

    println!("CDT-RQM Hawking radiation prune validation");
    println!(
        "lessons={} attempts={} before_edges={} before_regge={:.3} relations={}",
        lessons.len(),
        ATTEMPTS,
        before_edges,
        before_regge,
        substrate.relation_count()
    );
    print_metrics("before_sleep", before_metrics);
    print_metrics("standard_anneal", standard_metrics);
    print_metrics("enhanced_anneal", enhanced_metrics);
    print_metrics("hawking_radiation", hawking_metrics);
    println!(
        "standard: accepted={} edges={} regge={:.3} compression={:.1}% causality_violations={}",
        standard_accepted,
        standard_edges,
        standard_regge,
        (1.0 - standard_edges as f32 / before_edges.max(1) as f32) * 100.0,
        standard.hardware.causality_violations()
    );
    println!(
        "enhanced: accepted={} edges={} regge={:.3} compression={:.1}% causality_violations={}",
        enhanced_report.accepted,
        enhanced_edges,
        enhanced_regge,
        (1.0 - enhanced_edges as f32 / before_edges.max(1) as f32) * 100.0,
        enhanced.hardware.causality_violations()
    );
    println!(
        "hawking: accepted={} edges={} regge={:.3} compression={:.1}% causality_violations={}",
        hawking_report.accepted,
        hawking_edges,
        hawking_regge,
        (1.0 - hawking_edges as f32 / before_edges.max(1) as f32) * 100.0,
        hawking.hardware.causality_violations()
    );
    println!(
        "decision: {}",
        if keep {
            "keep_hawking_radiation value=preserves_memory_and_improves_geometry"
        } else {
            "discard_hawking_radiation value=no_material_gain_over_standard_anneal"
        }
    );
}

fn pure_standard_anneal(
    substrate: &mut CdtRqmUniverseSubstrate,
    validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
    lessons: &[Lesson],
    attempts: usize,
) -> usize {
    let protected_edges = validation
        .iter()
        .flat_map(|(_, _, cue, expected, _)| {
            cue.iter()
                .flat_map(move |source| expected.iter().map(move |target| (*source, *target)))
        })
        .collect::<Vec<_>>();
    let mut best = substrate.clone();
    let mut best_metrics = evaluate(&best, lessons);
    let mut best_regge = best.hardware.regge_action();
    let mut best_edges = active_edges(&best);
    let mut accepted = 0;

    for _ in 0..attempts {
        let mut candidate = best.clone();
        candidate.hardware.anneal_geometry_step(&protected_edges);
        let metrics = evaluate(&candidate, lessons);
        let regge = candidate.hardware.regge_action();
        let edges = active_edges(&candidate);
        let preserves = metrics.accuracy() + 0.0001 >= best_metrics.accuracy()
            && metrics.leakage() <= best_metrics.leakage() + 0.0001;
        let improves = regge < best_regge || edges < best_edges;
        if preserves && improves {
            best = candidate;
            best_metrics = metrics;
            best_regge = regge;
            best_edges = edges;
            accepted += 1;
        }
    }

    *substrate = best;
    accepted
}

fn train(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson]) {
    for _ in 0..8 {
        for lesson in lessons {
            let cue = pattern(0, lesson.cue);
            let target = pattern(1, lesson.target);
            substrate.hardware.clear_activity();
            substrate.train_observed_transition(lesson.observer, lesson.phase, &cue, &target, 1.0);
            for (&a, &b) in cue.iter().zip(target.iter()) {
                substrate.observe_entanglement_correlation(a, b, 0.40);
            }
        }
    }
}

fn evaluate(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(lesson.observer, lesson.phase, &cue);
        let expected = score(&report.collapse, &pattern(1, lesson.target));
        let distractor = score(&report.collapse, &pattern(1, lesson.distractor));
        metrics.record(expected, distractor);
    }
    metrics
}

fn score(report: &cdt_rqm_epr::relational_field::CollapseReport, targets: &[usize]) -> f32 {
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
                pattern(0, lesson.cue),
                pattern(1, lesson.target),
                pattern(1, lesson.distractor),
            )
        })
        .collect()
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3}",
        label,
        metrics.accuracy() * 100.0,
        metrics.leakage() * 100.0,
        metrics.margin()
    );
}

fn active_edges(substrate: &CdtRqmUniverseSubstrate) -> usize {
    substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count()
}

fn pattern(slice: usize, ordinal: usize) -> Vec<usize> {
    let base = slice * NODES_PER_SLICE + ordinal * 3;
    vec![base, base + 1, base + 2]
}

fn lessons() -> Vec<Lesson> {
    let phases = [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        -std::f32::consts::FRAC_PI_2,
    ];
    let mut out = Vec::new();
    for group in 0..4 {
        for offset in 0..8 {
            out.push(Lesson {
                observer: ObserverId(77_000 + group),
                phase: phases[group],
                cue: group * 20 + offset * 2,
                target: group * 20 + offset * 2 + 1,
                distractor: group * 20 + ((offset + 3) % 8) * 2 + 1,
            });
        }
    }
    out
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.20,
            initial_temporal_connectivity: 0.08,
            target_spatial_degree: 5,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 51_991,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.09,
            phase_learning_rate: 0.22,
            coherence_learning_rate: 0.12,
            uncertainty_learning_rate: 0.10,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.025,
        },
        max_quantum_candidates: 24,
        rqm_feedback_gain: 0.40,
    }
}

fn epr_config() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 256,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}
