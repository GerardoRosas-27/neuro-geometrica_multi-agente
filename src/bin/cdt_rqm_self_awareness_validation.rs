use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::entanglement::EntanglementConfig;
use snga::relational_field::{CandidateScore, ObserverId, RelationalFieldConfig};

const NODES_PER_SLICE: usize = 128;
const OBSERVER: ObserverId = ObserverId(190_001);
const SELF_OBSERVER: ObserverId = ObserverId(190_999);
const ATTEMPTS: usize = 24;

#[derive(Clone, Copy)]
struct Lesson {
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

#[derive(Clone, Copy)]
struct Snapshot {
    metrics: Metrics,
    edges: usize,
    regge: f32,
    relations: usize,
    self_score: f32,
    phi: f32,
    orch_margin: f32,
    free_energy: f32,
    causality_violations: usize,
}

fn main() {
    let lessons = lessons();
    let validation = validation_set(&lessons);

    let mut baseline = substrate();
    train_standard(&mut baseline, &lessons, 6);
    baseline.anneal_after_migration(&validation, ATTEMPTS);
    let baseline_snapshot = snapshot(&baseline, &lessons, false);

    let mut self_observer = baseline.clone();
    train_self_observer(&mut self_observer);
    let self_snapshot = snapshot(&self_observer, &lessons, false);

    let mut phi_selected = baseline.clone();
    let phi_accepted = phi_guided_selection(&mut phi_selected, &validation, &lessons, ATTEMPTS);
    let phi_snapshot = snapshot(&phi_selected, &lessons, false);

    let orch_snapshot = snapshot(&baseline, &lessons, true);

    println!("CDT-RQM explicit self-awareness validation");
    println!("lessons={} attempts={}", lessons.len(), ATTEMPTS);
    print_snapshot(
        "baseline_current",
        baseline_snapshot,
        baseline_snapshot.edges,
    );
    print_snapshot(
        "self_observation_loop",
        self_snapshot,
        baseline_snapshot.edges,
    );
    print_snapshot_with_accepted(
        "phi_integrated_information",
        phi_snapshot,
        baseline_snapshot.edges,
        phi_accepted,
    );
    print_snapshot("orch_or_collapse", orch_snapshot, baseline_snapshot.edges);

    print_decision(
        "self_observation",
        preserves(self_snapshot, baseline_snapshot)
            && self_snapshot.self_score > baseline_snapshot.self_score + 0.001
            && self_snapshot.free_energy <= baseline_snapshot.free_energy + 0.005,
        "keep_as_metric value=self_model_emerges_without_task_degradation",
        "discard value=self_loop_adds_cost_or_no_metacognitive_signal",
    );
    print_decision(
        "integrated_information_phi",
        preserves(phi_snapshot, baseline_snapshot)
            && phi_snapshot.phi > baseline_snapshot.phi + 0.001
            && phi_snapshot.free_energy <= baseline_snapshot.free_energy + 0.001,
        "keep_as_metric value=whole_greater_than_parts_signal_improves_or_preserves_state",
        "discard value=phi_does_not_improve_selection_over_current_sleep",
    );
    print_decision(
        "orch_or",
        orch_snapshot.metrics.accuracy() >= baseline_snapshot.metrics.accuracy()
            && orch_snapshot.metrics.leakage() <= baseline_snapshot.metrics.leakage()
            && orch_snapshot.orch_margin > baseline_snapshot.metrics.margin() + 0.001,
        "keep value=gravity_triggered_collapse_improves_decision_margin",
        "discard value=geometric_collapse_does_not_improve_current_recall",
    );
}

fn train_standard(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson], epochs: usize) {
    for _ in 0..epochs {
        for lesson in lessons {
            train_lesson(substrate, lesson, OBSERVER);
        }
    }
}

fn train_lesson(substrate: &mut CdtRqmUniverseSubstrate, lesson: &Lesson, observer: ObserverId) {
    let cue = pattern(0, lesson.cue);
    let target = pattern(1, lesson.target);
    substrate.hardware.clear_activity();
    substrate.train_observed_transition(observer, lesson.phase, &cue, &target, 1.0);
    for (&a, &b) in cue.iter().zip(target.iter()) {
        substrate.observe_entanglement_correlation(a, b, 0.40);
    }
}

fn train_self_observer(substrate: &mut CdtRqmUniverseSubstrate) {
    let cue = self_cue(substrate);
    let target = self_target();
    for _ in 0..8 {
        for &source in &cue {
            for &target in &target {
                substrate
                    .software
                    .reinforce_relation(SELF_OBSERVER, source, target, 0.0, 1.0);
            }
        }
    }
}

fn phi_guided_selection(
    substrate: &mut CdtRqmUniverseSubstrate,
    validation: &[(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)],
    lessons: &[Lesson],
    attempts: usize,
) -> usize {
    let mut best = substrate.clone();
    let mut best_snapshot = snapshot(&best, lessons, false);
    let protected_edges = validation
        .iter()
        .flat_map(|(_, _, cue, expected, _)| {
            cue.iter()
                .flat_map(move |source| expected.iter().map(move |target| (*source, *target)))
        })
        .collect::<Vec<_>>();
    let mut accepted = 0;
    let lambda = auto_lambda(&best);

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
            let snap = snapshot(&candidate, lessons, false);
            let preserves = preserves(snap, best_snapshot);
            let improves_phi = snap.phi > best_snapshot.phi + 0.001;
            let improves_energy = snap.free_energy < best_snapshot.free_energy;
            if preserves && (improves_phi || improves_energy) {
                best = candidate;
                best_snapshot = snap;
                accepted += 1;
            }
        }
    }

    *substrate = best;
    accepted
}

fn snapshot(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson], orch_or: bool) -> Snapshot {
    let metrics = if orch_or {
        evaluate_orch_or(substrate, lessons)
    } else {
        evaluate(substrate, lessons)
    };
    let edges = active_edges(substrate);
    let regge = substrate.hardware.regge_action();
    let phi = integrated_information_phi(substrate, lessons);
    let self_score = self_score(substrate);
    let complexity = edges as f32 / substrate.hardware.nodes.len().max(1) as f32;
    let free_energy = (1.0 - metrics.accuracy()) + metrics.leakage() - metrics.margin() * 0.01
        + regge * 0.0001
        + complexity * 0.03
        - phi * 0.02
        - self_score * 0.01;
    Snapshot {
        metrics,
        edges,
        regge,
        relations: substrate.relation_count(),
        self_score,
        phi,
        orch_margin: evaluate_orch_or(substrate, lessons).margin(),
        free_energy,
        causality_violations: substrate.hardware.causality_violations(),
    }
}

fn evaluate(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(OBSERVER, lesson.phase, &cue);
        let expected = score(&report.collapse.candidates, &pattern(1, lesson.target));
        let distractor = score(&report.collapse.candidates, &pattern(1, lesson.distractor));
        metrics.record(expected, distractor);
    }
    metrics
}

fn evaluate_orch_or(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    let curvature_energy =
        substrate.hardware.regge_action() / active_edges(substrate).max(1) as f32;
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let mut report = trial.step_from_boundary(OBSERVER, lesson.phase, &cue);
        if curvature_energy > 4.0 {
            report
                .collapse
                .candidates
                .sort_by(|left, right| right.score.total_cmp(&left.score));
            report.collapse.candidates.truncate(1);
        }
        let expected = score(&report.collapse.candidates, &pattern(1, lesson.target));
        let distractor = score(&report.collapse.candidates, &pattern(1, lesson.distractor));
        metrics.record(expected, distractor);
    }
    metrics
}

fn integrated_information_phi(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> f32 {
    let mut trial = substrate.clone();
    let mut phi_sum = 0.0;
    let mut cases = 0;
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        let target = pattern(1, lesson.target);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let whole = trial.step_from_boundary(OBSERVER, lesson.phase, &cue);
        let whole_score = score(&whole.collapse.candidates, &target);

        let mut part_score_sum = 0.0;
        for &seed in &cue {
            trial.hardware.clear_activity();
            trial.hardware.inject_pattern(&[seed], 1.0);
            let part = trial.step_from_boundary(OBSERVER, lesson.phase, &[seed]);
            part_score_sum += score(&part.collapse.candidates, &target);
        }
        let part_mean = part_score_sum / cue.len().max(1) as f32;
        phi_sum += (whole_score - part_mean).max(0.0);
        cases += 1;
    }
    phi_sum / cases.max(1) as f32
}

fn self_score(substrate: &CdtRqmUniverseSubstrate) -> f32 {
    let cue = self_cue(substrate);
    let target = self_target();
    let mut trial = substrate.clone();
    trial.hardware.clear_activity();
    trial.hardware.inject_pattern(&cue, 1.0);
    let report = trial.step_from_boundary(SELF_OBSERVER, 0.0, &cue);
    score(&report.collapse.candidates, &target)
}

fn score(candidates: &[CandidateScore], targets: &[usize]) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn self_cue(substrate: &CdtRqmUniverseSubstrate) -> Vec<usize> {
    let edges = active_edges(substrate);
    let regge = (substrate.hardware.regge_action() as usize / 128).min(127);
    let epr = substrate
        .entanglement_summary()
        .map(|report| report.active_links)
        .unwrap_or(0);
    vec![
        2 * NODES_PER_SLICE + edges % NODES_PER_SLICE,
        2 * NODES_PER_SLICE + regge % NODES_PER_SLICE,
        2 * NODES_PER_SLICE + epr % NODES_PER_SLICE,
    ]
}

fn self_target() -> Vec<usize> {
    vec![
        3 * NODES_PER_SLICE + 1,
        3 * NODES_PER_SLICE + 2,
        3 * NODES_PER_SLICE + 3,
    ]
}

fn preserves(candidate: Snapshot, baseline: Snapshot) -> bool {
    candidate.metrics.accuracy() + 0.0001 >= baseline.metrics.accuracy()
        && candidate.metrics.leakage() <= baseline.metrics.leakage() + 0.0001
        && candidate.causality_violations == 0
}

fn print_snapshot(label: &str, snapshot: Snapshot, before_edges: usize) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} edges={} regge={:.3} relations={} self_score={:.3} phi={:.3} orch_margin={:.3} free_energy={:.3} edge_delta={} causality_violations={}",
        label,
        snapshot.metrics.accuracy() * 100.0,
        snapshot.metrics.leakage() * 100.0,
        snapshot.metrics.margin(),
        snapshot.edges,
        snapshot.regge,
        snapshot.relations,
        snapshot.self_score,
        snapshot.phi,
        snapshot.orch_margin,
        snapshot.free_energy,
        snapshot.edges as isize - before_edges as isize,
        snapshot.causality_violations
    );
}

fn print_snapshot_with_accepted(
    label: &str,
    snapshot: Snapshot,
    before_edges: usize,
    accepted: usize,
) {
    print_snapshot(label, snapshot, before_edges);
    println!("{}_accepted={}", label, accepted);
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

fn validation_set(
    lessons: &[Lesson],
) -> Vec<(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)> {
    lessons
        .iter()
        .map(|lesson| {
            (
                OBSERVER,
                lesson.phase,
                pattern(0, lesson.cue),
                pattern(1, lesson.target),
                pattern(1, lesson.distractor),
            )
        })
        .collect()
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
        for offset in 0..6 {
            out.push(Lesson {
                phase: phases[group],
                cue: group * 16 + offset * 2,
                target: group * 16 + offset * 2 + 1,
                distractor: group * 16 + ((offset + 2) % 6) * 2 + 1,
            });
        }
    }
    out
}

fn substrate() -> CdtRqmUniverseSubstrate {
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    substrate.enable_entanglement(epr_config());
    substrate
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.18,
            initial_temporal_connectivity: 0.07,
            target_spatial_degree: 5,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 61_991,
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
