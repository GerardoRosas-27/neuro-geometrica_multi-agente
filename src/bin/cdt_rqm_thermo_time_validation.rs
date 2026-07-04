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
    entropy_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        let p = if total > f32::EPSILON {
            (expected / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.entropy_sum += binary_entropy(p);
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

    fn entropy(self) -> f32 {
        self.entropy_sum / self.cases.max(1) as f32
    }
}

#[derive(Clone, Copy)]
struct Snapshot {
    metrics: Metrics,
    accepted: usize,
    edges: usize,
    regge: f32,
    lambda_action: f32,
    free_energy: f32,
    critical_distance: f32,
    temperature: f32,
    epr_entropy: f32,
    causality_violations: usize,
}

fn main() {
    let lessons = lessons();
    let validation = validation_set(&lessons);
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    substrate.enable_entanglement(epr_config());
    train(&mut substrate, &lessons);
    let lambda = auto_lambda(&substrate);

    let before = snapshot(&substrate, &lessons, lambda, 0);

    let mut current = substrate.clone();
    let current_report = current.anneal_after_migration(&validation, ATTEMPTS);
    let current = snapshot(&current, &lessons, lambda, current_report.accepted);

    let mut landauer = substrate.clone();
    let landauer_accepted = landauer_anneal(&mut landauer, &validation, &lessons, ATTEMPTS, lambda);
    let landauer = snapshot(&landauer, &lessons, lambda, landauer_accepted);

    let mut critical = substrate.clone();
    let critical_accepted = criticality_anneal(&mut critical, &validation, &lessons, ATTEMPTS);
    let critical = snapshot(&critical, &lessons, lambda, critical_accepted);

    let mut friston = substrate.clone();
    let friston_accepted =
        free_energy_anneal(&mut friston, &validation, &lessons, ATTEMPTS, lambda);
    let friston = snapshot(&friston, &lessons, lambda, friston_accepted);

    let arrow = thermodynamic_arrow_probe(&lessons);

    println!("CDT-RQM thermodynamic time validation");
    println!(
        "lessons={} attempts={} lambda={:.5} arrow_entropy_before={:.3} arrow_entropy_after={:.3} internal_time_delta={:.3}",
        lessons.len(),
        ATTEMPTS,
        lambda,
        arrow.0,
        arrow.1,
        arrow.1 - arrow.0
    );
    print_snapshot("before_sleep", before, before.edges);
    print_snapshot("current_combined", current, before.edges);
    print_snapshot("landauer_cost", landauer, before.edges);
    print_snapshot("criticality_gate", critical, before.edges);
    print_snapshot("friston_free_energy", friston, before.edges);

    println!(
        "decision_time_arrow: {}",
        if arrow.1 <= arrow.0 + 0.001 {
            "keep_as_metric value=subjective_time_entropy_does_not_drift_after_learning"
        } else {
            "discard value=entropy_gradient_not_stable"
        }
    );
    print_decision(
        "landauer",
        preserves(landauer, current) && landauer.free_energy + 0.001 < current.free_energy,
        "keep value=paid_for_forgetting_and_lowered_free_energy",
        "discard value=costly_forgetting_adds_no_gain",
    );
    print_decision(
        "criticality",
        preserves(critical, current)
            && critical.critical_distance + 0.001 < current.critical_distance
            && critical.free_energy <= current.free_energy + 0.001,
        "keep value=moves_substrate_toward_edge_of_chaos",
        "discard value=no_better_criticality_than_current",
    );
    print_decision(
        "friston_free_energy",
        preserves(friston, current) && friston.free_energy + 0.001 < current.free_energy,
        "keep value=unified_free_energy_improves_selection",
        "discard value=no_material_gain_over_current_combined_anneal",
    );
}

fn landauer_anneal(
    substrate: &mut CdtRqmUniverseSubstrate,
    validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
    lessons: &[Lesson],
    attempts: usize,
    lambda: f32,
) -> usize {
    let protected_edges = protected_edges(validation);
    let mut best = substrate.clone();
    let mut best_metrics = evaluate(&best, lessons);
    let mut best_objective = best.hardware.cosmological_regge_action(lambda);
    let mut best_edges = active_edges(&best);
    let mut accepted = 0;

    for _ in 0..attempts {
        for move_kind in [0_u8, 1, 2] {
            let mut candidate = best.clone();
            match move_kind {
                0 => {
                    candidate.hardware.anneal_geometry_step(&protected_edges);
                }
                1 => {
                    candidate.hardware.hawking_radiation_step(&protected_edges);
                }
                _ => {
                    candidate
                        .hardware
                        .cosmological_constant_step(&protected_edges, lambda);
                }
            }
            let metrics = evaluate(&candidate, lessons);
            let edges = active_edges(&candidate);
            let forgotten_bits = best_edges.saturating_sub(edges) as f32;
            let landauer_cost =
                forgotten_bits * candidate.hardware.temperature.max(0.001) * std::f32::consts::LN_2;
            let objective = candidate.hardware.cosmological_regge_action(lambda) + landauer_cost;
            if metrics.accuracy() + 0.0001 >= best_metrics.accuracy()
                && metrics.leakage() <= best_metrics.leakage() + 0.0001
                && candidate.hardware.causality_violations() == 0
                && objective < best_objective
            {
                best = candidate;
                best_metrics = metrics;
                best_objective = objective;
                best_edges = edges;
                accepted += 1;
            }
        }
    }

    *substrate = best;
    accepted
}

fn criticality_anneal(
    substrate: &mut CdtRqmUniverseSubstrate,
    validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
    lessons: &[Lesson],
    attempts: usize,
) -> usize {
    let protected_edges = protected_edges(validation);
    let mut best = substrate.clone();
    let mut best_metrics = evaluate(&best, lessons);
    let mut best_distance = critical_distance(&best);
    let mut accepted = 0;

    for _ in 0..attempts {
        let mut candidate = best.clone();
        if candidate.hardware.temperature > 1.0 {
            candidate.hardware.hawking_radiation_step(&protected_edges);
        } else {
            candidate.hardware.anneal_geometry_step(&protected_edges);
        }
        let metrics = evaluate(&candidate, lessons);
        let distance = critical_distance(&candidate);
        if metrics.accuracy() + 0.0001 >= best_metrics.accuracy()
            && metrics.leakage() <= best_metrics.leakage() + 0.0001
            && candidate.hardware.causality_violations() == 0
            && distance < best_distance
        {
            best = candidate;
            best_metrics = metrics;
            best_distance = distance;
            accepted += 1;
        }
    }

    *substrate = best;
    accepted
}

fn free_energy_anneal(
    substrate: &mut CdtRqmUniverseSubstrate,
    validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
    lessons: &[Lesson],
    attempts: usize,
    lambda: f32,
) -> usize {
    let protected_edges = protected_edges(validation);
    let mut best = substrate.clone();
    let mut best_snapshot = snapshot(&best, lessons, lambda, 0);
    let mut accepted = 0;

    for _ in 0..attempts {
        for move_kind in [0_u8, 1, 2] {
            let mut candidate = best.clone();
            match move_kind {
                0 => {
                    candidate.hardware.anneal_geometry_step(&protected_edges);
                }
                1 => {
                    candidate.hardware.hawking_radiation_step(&protected_edges);
                }
                _ => {
                    candidate
                        .hardware
                        .cosmological_constant_step(&protected_edges, lambda);
                }
            }
            let snap = snapshot(&candidate, lessons, lambda, 0);
            if preserves(snap, best_snapshot) && snap.free_energy < best_snapshot.free_energy {
                best = candidate;
                best_snapshot = snap;
                accepted += 1;
            }
        }
    }

    *substrate = best;
    accepted
}

fn thermodynamic_arrow_probe(lessons: &[Lesson]) -> (f32, f32) {
    let split = lessons.len() / 2;
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    substrate.enable_entanglement(epr_config());
    train(&mut substrate, &lessons[..split]);
    let before = evaluate(&substrate, &lessons[..split]).entropy();
    train(&mut substrate, &lessons[split..]);
    let after = evaluate(&substrate, &lessons[..split]).entropy();
    (before, after)
}

fn snapshot(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    lambda: f32,
    accepted: usize,
) -> Snapshot {
    let metrics = evaluate(substrate, lessons);
    let epr_entropy = substrate
        .entanglement_summary()
        .map(|report| report.mean_entropy)
        .unwrap_or(0.0);
    let edges = active_edges(substrate);
    let nodes = substrate.hardware.nodes.len().max(1) as f32;
    let complexity = edges as f32 / nodes;
    let lambda_action = substrate.hardware.cosmological_regge_action(lambda);
    let free_energy = (1.0 - metrics.accuracy())
        + metrics.leakage()
        + metrics.entropy()
        + 0.0001 * lambda_action
        + 0.03 * complexity
        + 0.10 * epr_entropy;
    Snapshot {
        metrics,
        accepted,
        edges,
        regge: substrate.hardware.regge_action(),
        lambda_action,
        free_energy,
        critical_distance: critical_distance(substrate),
        temperature: substrate.hardware.temperature,
        epr_entropy,
        causality_violations: substrate.hardware.causality_violations(),
    }
}

fn critical_distance(substrate: &CdtRqmUniverseSubstrate) -> f32 {
    let nodes = substrate.hardware.nodes.len().max(1) as f32;
    let edges = active_edges(substrate) as f32;
    let mean_degree = 2.0 * edges / nodes;
    let target_degree = (substrate.hardware.config.target_spatial_degree
        + substrate.hardware.config.target_temporal_degree) as f32;
    ((mean_degree - target_degree).abs() / target_degree.max(1.0))
        + (substrate.hardware.temperature - 1.0).abs()
}

fn preserves(candidate: Snapshot, baseline: Snapshot) -> bool {
    candidate.metrics.accuracy() + 0.0001 >= baseline.metrics.accuracy()
        && candidate.metrics.leakage() <= baseline.metrics.leakage() + 0.0001
        && candidate.causality_violations == 0
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

fn protected_edges(
    validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
) -> Vec<(usize, usize)> {
    validation
        .iter()
        .flat_map(|(_, _, cue, expected, _)| {
            cue.iter()
                .flat_map(move |source| expected.iter().map(move |target| (*source, *target)))
        })
        .collect()
}

fn print_snapshot(label: &str, snapshot: Snapshot, before_edges: usize) {
    println!(
        "{}: accepted={} accuracy={:.1}% leakage={:.1}% entropy={:.3} margin={:.3} edges={} regge={:.3} lambda_action={:.3} free_energy={:.3} critical_distance={:.3} temp={:.3} epr_entropy={:.3} compression={:.1}% causality_violations={}",
        label,
        snapshot.accepted,
        snapshot.metrics.accuracy() * 100.0,
        snapshot.metrics.leakage() * 100.0,
        snapshot.metrics.entropy(),
        snapshot.metrics.margin(),
        snapshot.edges,
        snapshot.regge,
        snapshot.lambda_action,
        snapshot.free_energy,
        snapshot.critical_distance,
        snapshot.temperature,
        snapshot.epr_entropy,
        (1.0 - snapshot.edges as f32 / before_edges.max(1) as f32) * 100.0,
        snapshot.causality_violations
    );
}

fn print_decision(name: &str, keep: bool, keep_reason: &str, discard_reason: &str) {
    println!(
        "decision_{}: {}",
        name,
        if keep { keep_reason } else { discard_reason }
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

fn auto_lambda(substrate: &CdtRqmUniverseSubstrate) -> f32 {
    let volume = substrate.hardware.tetrahedra.len().max(1) as f32;
    let curvature_density = substrate.hardware.regge_action() / volume;
    (0.05 / (1.0 + curvature_density / 16.0)).clamp(0.005, 0.05)
}

fn binary_entropy(p: f32) -> f32 {
    if p <= f32::EPSILON || 1.0 - p <= f32::EPSILON {
        return 0.0;
    }
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
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
                observer: ObserverId(170_000 + group),
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
            seed: 59_991,
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
