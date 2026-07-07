use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::relational_field::{ObserverId, RelationalFieldConfig};
use std::hash::{Hash, Hasher};

const NODES_PER_SLICE: usize = 384;
const OBSERVER: ObserverId = ObserverId(160_001);

#[derive(Clone)]
struct Lesson {
    local: Vec<usize>,
    remote: Vec<usize>,
    distractor: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
    latency_sum: usize,
    margin_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32, latency: usize) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.latency_sum += latency;
        self.margin_sum += expected - distractor;
    }

    fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn latency(self) -> f32 {
        self.latency_sum as f32 / self.cases.max(1) as f32
    }

    fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }
}

#[derive(Clone, Copy)]
struct Snapshot {
    metrics: Metrics,
    active_edges: usize,
    regge: f32,
    relations: usize,
    epr_links: usize,
    epr_coherence: f32,
    epr_entropy: f32,
    area_law_ratio: f32,
    page_retention: f32,
    causality_violations: usize,
}

fn main() {
    let lessons = lessons();

    let mut baseline = CdtRqmUniverseSubstrate::new(config());
    baseline.enable_entanglement(epr_config(8));
    train_epr(&mut baseline, &lessons, false);
    let baseline_snapshot = snapshot(&baseline, &lessons, &lessons);

    let mut er_epr = CdtRqmUniverseSubstrate::new(config());
    er_epr.enable_entanglement(epr_config(8));
    train_epr(&mut er_epr, &lessons, true);
    let er_epr_snapshot = snapshot(&er_epr, &lessons, &lessons);

    let mut monogamy = CdtRqmUniverseSubstrate::new(config());
    monogamy.enable_entanglement(epr_config(4));
    train_epr(&mut monogamy, &lessons, false);
    let monogamy_snapshot = snapshot(&monogamy, &lessons, &lessons);

    let page_retention = page_curve_retention(&lessons);

    println!("CDT-RQM EPR information hypotheses validation");
    print_snapshot("baseline_epr", baseline_snapshot);
    print_snapshot("er_equals_epr", er_epr_snapshot);
    print_snapshot("monogamy_strict", monogamy_snapshot);
    println!("page_curve_global_retention={:.3}", page_retention);
    println!(
        "decision_area_law: {}",
        if baseline_snapshot.area_law_ratio <= 1.0 && baseline_snapshot.epr_entropy < 0.05 {
            "keep_as_metric value=epr_already_tracks_boundary_area_without_volume_noise"
        } else {
            "discard value=area_law_not_satisfied_or_too_noisy"
        }
    );
    println!(
        "decision_er_equals_epr: {}",
        if preserves(er_epr_snapshot, baseline_snapshot)
            && (er_epr_snapshot.latency_better_than(baseline_snapshot)
                || er_epr_snapshot.regge + 0.001 < baseline_snapshot.regge
                || er_epr_snapshot.active_edges < baseline_snapshot.active_edges)
        {
            "keep value=epr_links_improve_metric_geometry"
        } else {
            "discard value=no_material_gain_over_candidate_synchronization"
        }
    );
    println!(
        "decision_monogamy: {}",
        if preserves(monogamy_snapshot, baseline_snapshot)
            && monogamy_snapshot.epr_entropy <= baseline_snapshot.epr_entropy + 0.001
            && monogamy_snapshot.epr_coherence >= baseline_snapshot.epr_coherence - 0.001
            && monogamy_snapshot.epr_links < baseline_snapshot.epr_links
        {
            "keep value=lower_entanglement_degree_preserves_memory"
        } else {
            "discard value=current_monogamy_limit_is_better_or_equal"
        }
    );
    println!(
        "decision_page_curve_global: {}",
        if page_retention >= 0.999 {
            "keep_as_metric value=global_memory_retention_saturated"
        } else {
            "discard value=global_retention_not_stable_enough"
        }
    );
}

impl Snapshot {
    fn latency_better_than(self, other: Snapshot) -> bool {
        self.metrics.latency() + 0.001 < other.metrics.latency()
    }
}

fn preserves(candidate: Snapshot, baseline: Snapshot) -> bool {
    candidate.metrics.accuracy() + 0.0001 >= baseline.metrics.accuracy()
        && candidate.metrics.leakage() <= baseline.metrics.leakage() + 0.0001
        && candidate.causality_violations == 0
}

fn train_epr(substrate: &mut CdtRqmUniverseSubstrate, lessons: &[Lesson], er_equals_epr: bool) {
    for lesson in lessons {
        for (&a, &b) in lesson.local.iter().zip(lesson.remote.iter()) {
            if let Some(field) = substrate.entanglement.as_mut() {
                field.create_or_reinforce(a, b);
                field.create_or_reinforce(a, b);
                field.create_or_reinforce(a, b);
            }
            if er_equals_epr {
                substrate.hardware.reinforce_temporal_link(a, b, 0.70);
            }
        }
    }
}

fn snapshot(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    area_regions: &[Lesson],
) -> Snapshot {
    let metrics = evaluate(substrate, lessons);
    let epr = substrate.entanglement_summary().unwrap_or_default();
    Snapshot {
        metrics,
        active_edges: substrate
            .hardware
            .edges
            .iter()
            .filter(|edge| edge.active)
            .count(),
        regge: substrate.hardware.regge_action(),
        relations: substrate.relation_count(),
        epr_links: epr.active_links,
        epr_coherence: epr.mean_coherence,
        epr_entropy: epr.mean_entropy,
        area_law_ratio: area_law_ratio(epr.active_links, area_regions),
        page_retention: page_retention_for(substrate, lessons),
        causality_violations: substrate.hardware.causality_violations(),
    }
}

fn evaluate(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let latency = latency(&mut trial, &lesson.local, &lesson.remote, 8);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&lesson.local, 1.0);
        let report = trial.step_from_boundary(OBSERVER, 0.0, &lesson.local);
        let expected = report
            .expected_from_rqm
            .iter()
            .filter(|candidate| lesson.remote.contains(candidate))
            .count() as f32;
        let distractor = report
            .expected_from_rqm
            .iter()
            .filter(|candidate| lesson.distractor.contains(candidate))
            .count() as f32;
        metrics.record(expected, distractor, latency);
    }
    metrics
}

fn latency(
    substrate: &mut CdtRqmUniverseSubstrate,
    local: &[usize],
    remote: &[usize],
    max_steps: usize,
) -> usize {
    for step in 1..=max_steps {
        substrate.hardware.clear_activity();
        substrate.hardware.inject_pattern(local, 1.0);
        let report = substrate.step_from_boundary(OBSERVER, 0.0, local);
        let hits = report
            .expected_from_rqm
            .iter()
            .filter(|idx| remote.contains(idx))
            .count();
        if hits >= remote.len().min(4) {
            return step;
        }
    }
    max_steps + 1
}

fn area_law_ratio(active_links: usize, lessons: &[Lesson]) -> f32 {
    let boundary_area = lessons
        .iter()
        .map(|lesson| lesson.local.len().min(lesson.remote.len()))
        .sum::<usize>()
        .max(1);
    active_links as f32 / boundary_area as f32
}

fn page_curve_retention(lessons: &[Lesson]) -> f32 {
    let midpoint = lessons.len() / 2;
    let early = &lessons[..midpoint];
    let late = &lessons[midpoint..];
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    substrate.enable_entanglement(epr_config(8));
    train_epr(&mut substrate, early, false);
    let early_before = evaluate(&substrate, early);
    train_epr(&mut substrate, late, false);
    let early_after = evaluate(&substrate, early);
    if early_before.margin().abs() <= f32::EPSILON {
        return early_after.accuracy();
    }
    (early_after.margin() / early_before.margin()).clamp(0.0, 1.0)
}

fn page_retention_for(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> f32 {
    let metrics = evaluate(substrate, lessons);
    if metrics.accuracy() >= 1.0 && metrics.leakage() <= 0.001 {
        1.0
    } else {
        (metrics.accuracy() * (1.0 - metrics.leakage())).clamp(0.0, 1.0)
    }
}

fn print_snapshot(label: &str, snapshot: Snapshot) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% latency={:.2} margin={:.3} edges={} regge={:.3} relations={} epr_links={} epr_coherence={:.3} epr_entropy={:.3} area_law_ratio={:.3} page_retention={:.3} causality_violations={}",
        label,
        snapshot.metrics.accuracy() * 100.0,
        snapshot.metrics.leakage() * 100.0,
        snapshot.metrics.latency(),
        snapshot.metrics.margin(),
        snapshot.active_edges,
        snapshot.regge,
        snapshot.relations,
        snapshot.epr_links,
        snapshot.epr_coherence,
        snapshot.epr_entropy,
        snapshot.area_law_ratio,
        snapshot.page_retention,
        snapshot.causality_violations
    );
}

fn lessons() -> Vec<Lesson> {
    (0..8)
        .map(|idx| Lesson {
            local: pattern(&format!("local_{idx}"), 0),
            remote: pattern(&format!("remote_{idx}"), 1),
            distractor: pattern(&format!("distractor_{idx}"), 1),
        })
        .collect()
}

fn pattern(label: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..12)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * NODES_PER_SLICE + (hasher.finish() as usize % NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.0001,
            initial_temporal_connectivity: 0.00005,
            target_spatial_degree: 4,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 58_991,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.10,
            phase_learning_rate: 0.24,
            coherence_learning_rate: 0.13,
            uncertainty_learning_rate: 0.11,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.02,
        },
        max_quantum_candidates: 128,
        rqm_feedback_gain: 0.42,
    }
}

fn epr_config(max_links_per_node: usize) -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}
