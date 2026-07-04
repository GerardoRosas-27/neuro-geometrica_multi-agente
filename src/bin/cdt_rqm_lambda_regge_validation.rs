use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::entanglement::EntanglementConfig;
use snga::relational_field::{ObserverId, RelationalFieldConfig};

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
    let lambda = auto_lambda(&substrate);
    let before_lambda_action = substrate.hardware.cosmological_regge_action(lambda);

    let mut current = substrate.clone();
    let current_report = current.anneal_after_migration(&validation, ATTEMPTS);
    let current_metrics = evaluate(&current, &lessons);
    let current_edges = active_edges(&current);
    let current_regge = current.hardware.regge_action();
    let current_lambda_action = current.hardware.cosmological_regge_action(lambda);

    let mut lambda_substrate = substrate.clone();
    let lambda_accepted = lambda_regge_anneal(
        &mut lambda_substrate,
        &validation,
        &lessons,
        ATTEMPTS,
        lambda,
    );
    let lambda_metrics = evaluate(&lambda_substrate, &lessons);
    let lambda_edges = active_edges(&lambda_substrate);
    let lambda_regge = lambda_substrate.hardware.regge_action();
    let lambda_action = lambda_substrate.hardware.cosmological_regge_action(lambda);

    let preserves = lambda_metrics.accuracy() + 0.0001 >= current_metrics.accuracy()
        && lambda_metrics.leakage() <= current_metrics.leakage() + 0.0001
        && lambda_substrate.hardware.causality_violations() == 0;
    let improves_effective_action = lambda_action + 0.001 < current_lambda_action;
    let does_not_bloat = lambda_edges <= current_edges + before_edges / 20;
    let keep = preserves && improves_effective_action && does_not_bloat;

    println!("CDT-RQM dynamic Lambda Regge validation");
    println!(
        "lessons={} attempts={} lambda={:.5} before_edges={} before_regge={:.3} before_lambda_action={:.3}",
        lessons.len(),
        ATTEMPTS,
        lambda,
        before_edges,
        before_regge,
        before_lambda_action
    );
    print_metrics("before_sleep", before_metrics);
    print_metrics("current_hawking_anneal", current_metrics);
    print_metrics("lambda_regge_anneal", lambda_metrics);
    println!(
        "current: accepted={} edges={} regge={:.3} lambda_action={:.3} tetrahedra={} compression={:.1}% causality_violations={}",
        current_report.accepted,
        current_edges,
        current_regge,
        current_lambda_action,
        current.hardware.tetrahedra.len(),
        (1.0 - current_edges as f32 / before_edges.max(1) as f32) * 100.0,
        current.hardware.causality_violations()
    );
    println!(
        "lambda: accepted={} edges={} regge={:.3} lambda_action={:.3} tetrahedra={} compression={:.1}% causality_violations={}",
        lambda_accepted,
        lambda_edges,
        lambda_regge,
        lambda_action,
        lambda_substrate.hardware.tetrahedra.len(),
        (1.0 - lambda_edges as f32 / before_edges.max(1) as f32) * 100.0,
        lambda_substrate.hardware.causality_violations()
    );
    println!(
        "decision: {}",
        if keep {
            "keep_lambda_regge value=preserves_memory_and_improves_effective_action"
        } else {
            "discard_lambda_regge value=no_material_gain_over_current_geometry"
        }
    );
}

fn lambda_regge_anneal(
    substrate: &mut CdtRqmUniverseSubstrate,
    validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
    lessons: &[Lesson],
    attempts: usize,
    lambda: f32,
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
    let mut best_action = best.hardware.cosmological_regge_action(lambda);
    let mut best_edges = active_edges(&best);
    let mut accepted = 0;

    for _ in 0..attempts {
        let mut candidate = best.clone();
        candidate
            .hardware
            .cosmological_constant_step(&protected_edges, lambda);
        let metrics = evaluate(&candidate, lessons);
        let action = candidate.hardware.cosmological_regge_action(lambda);
        let edges = active_edges(&candidate);
        let preserves = metrics.accuracy() + 0.0001 >= best_metrics.accuracy()
            && metrics.leakage() <= best_metrics.leakage() + 0.0001
            && candidate.hardware.causality_violations() == 0;
        let improves = action < best_action || edges < best_edges;
        if preserves && improves {
            best = candidate;
            best_metrics = metrics;
            best_action = action;
            best_edges = edges;
            accepted += 1;
        }
    }

    *substrate = best;
    accepted
}

fn auto_lambda(substrate: &CdtRqmUniverseSubstrate) -> f32 {
    let volume = substrate.hardware.tetrahedra.len().max(1) as f32;
    let curvature_density = substrate.hardware.regge_action() / volume;
    (0.05 / (1.0 + curvature_density / 16.0)).clamp(0.005, 0.05)
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

fn score(report: &snga::relational_field::CollapseReport, targets: &[usize]) -> f32 {
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
                observer: ObserverId(120_000 + group),
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
            seed: 55_991,
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
