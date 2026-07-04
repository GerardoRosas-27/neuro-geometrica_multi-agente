use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::cdt_rqm_experimental::{
    dvali_extended_report, mera_report, unified_free_energy, UnifiedFreeEnergyConfig,
};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::relational_field::{CollapseReport, ObserverId, RelationalFieldConfig};
use std::hash::{Hash, Hasher};

const STATE: &str = "data/cdt_rqm_evolutionary_kept.cdt_rqm";
const NODES_PER_SLICE: usize = 160;
const OBSERVER: ObserverId = ObserverId(260_001);

#[derive(Clone, Copy)]
enum LessonKind {
    Semantic,
    Episodic,
    Causal,
    Skill,
}

#[derive(Clone)]
struct Lesson {
    kind: LessonKind,
    local: Vec<usize>,
    action: Vec<usize>,
    remote: Vec<usize>,
    distractor: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
    margin_sum: f32,
    prediction_error_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32, prediction_error: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.prediction_error_sum += prediction_error;
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

    fn prediction_error(self) -> f32 {
        self.prediction_error_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let lessons = lessons();
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    let loaded = substrate.load_consolidated_state(STATE).is_ok();
    if !loaded {
        println!("CDT-RQM consolidated evaluation");
        println!("loaded=false state={STATE}");
        return;
    }

    let normal = evaluate(&substrate, &lessons, EvalMode::Normal);
    let action = evaluate(&substrate, &lessons, EvalMode::ActionConditioned);
    let typed = evaluate_typed(&substrate, &lessons);
    let contradiction = evaluate_contradiction(&substrate, &lessons);
    let dvali = dvali_extended_report(&substrate, normal.leakage(), normal.prediction_error());
    let free_energy = unified_free_energy(
        &substrate,
        normal.prediction_error(),
        normal.leakage(),
        UnifiedFreeEnergyConfig::default(),
    )
    .free_energy;
    let edges = substrate.hardware.active_edge_count();
    let relations = substrate.relation_count();
    let compute_cost =
        (edges as f32 + relations as f32 * 0.10) / substrate.hardware.nodes.len().max(1) as f32;
    let epr = substrate.entanglement_summary().unwrap_or_default();
    let all_pass = normal.accuracy() >= 1.0
        && action.accuracy() >= 1.0
        && typed.accuracy() >= 1.0
        && contradiction.leakage() <= normal.leakage() + 0.001
        && substrate.hardware.causality_violations() == 0;

    println!("CDT-RQM consolidated evaluation");
    println!("loaded=true state={STATE}");
    print_metrics("normal", normal);
    print_metrics("action_conditioned", action);
    print_metrics("typed_memory", typed);
    print_metrics("contradiction_probe", contradiction);
    println!(
        "geometry: edges={} relations={} regge={:.3} deficit_regge={:.3} free_energy={:.3} criticality_distance={:.3} mera_gain={:.3} compute_cost={:.3} causality_violations={}",
        edges,
        relations,
        substrate.hardware.regge_action(),
        substrate.hardware.discrete_regge_deficit_action(),
        free_energy,
        substrate.hardware.criticality_distance(),
        mera_report(&substrate, &[4, 8, 16, 32]).compression_gain,
        compute_cost,
        substrate.hardware.causality_violations()
    );
    println!(
        "dvali: N={:.1} alpha={:.6} alphaN={:.3} T_N={:.4} depletion={:.4} lifetime={:.1} break_time={:.1} species={:.1} cutoff={:.4} memory_burden={:.3} classical_radius={:.3}",
        dvali.occupation_number,
        dvali.alpha_eff,
        dvali.maximal_packing,
        dvali.n_portrait_temperature,
        dvali.depletion_rate,
        dvali.evaporation_lifetime,
        dvali.quantum_break_time,
        dvali.species_count,
        dvali.species_cutoff,
        dvali.memory_burden,
        dvali.classicalization_radius
    );
    println!(
        "epr: active_links={} mean_coherence={:.3} mean_entropy={:.3}",
        epr.active_links, epr.mean_coherence, epr.mean_entropy
    );
    println!("suite={}", if all_pass { "PASS" } else { "FAIL" });
}

#[derive(Clone, Copy)]
enum EvalMode {
    Normal,
    ActionConditioned,
}

fn evaluate(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson], mode: EvalMode) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let mut cue = lesson.local.clone();
        if matches!(mode, EvalMode::ActionConditioned) {
            cue.extend_from_slice(&lesson.action);
            cue.sort_unstable();
            cue.dedup();
        }
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(OBSERVER, 0.0, &cue);
        metrics.record(
            score(&report.collapse, &lesson.remote),
            score(&report.collapse, &lesson.distractor),
            report.cdt.prediction_error,
        );
    }
    metrics
}

fn evaluate_typed(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let observer = match lesson.kind {
            LessonKind::Semantic => ObserverId(261_001),
            LessonKind::Episodic => ObserverId(261_002),
            LessonKind::Causal => ObserverId(261_003),
            LessonKind::Skill => ObserverId(261_004),
        };
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&lesson.local, 1.0);
        let report = trial.step_from_boundary(observer, 0.0, &lesson.local);
        metrics.record(
            score(&report.collapse, &lesson.remote),
            score(&report.collapse, &lesson.distractor),
            report.cdt.prediction_error,
        );
    }
    metrics
}

fn evaluate_contradiction(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&lesson.local, 1.0);
        let report = trial.step_from_boundary(OBSERVER, 0.0, &lesson.local);
        let expected = score(&report.collapse, &lesson.remote);
        let distractor = score(&report.collapse, &lesson.distractor);
        metrics.record(expected, distractor, report.cdt.prediction_error);
    }
    metrics
}

fn score(report: &CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} prediction_error={:.3}",
        label,
        metrics.accuracy() * 100.0,
        metrics.leakage() * 100.0,
        metrics.margin(),
        metrics.prediction_error()
    );
}

fn lessons() -> Vec<Lesson> {
    [
        (
            LessonKind::Semantic,
            "vanchurin",
            "represent",
            "madelung",
            "noise",
        ),
        (
            LessonKind::Semantic,
            "mera",
            "compress",
            "holography",
            "flat",
        ),
        (
            LessonKind::Causal,
            "dvali",
            "stabilize",
            "criticality",
            "thermal",
        ),
        (
            LessonKind::Causal,
            "wolfram",
            "branch",
            "causal_invariance",
            "random",
        ),
        (
            LessonKind::Episodic,
            "graphity",
            "cool",
            "geometrogenesis",
            "complete",
        ),
        (
            LessonKind::Skill,
            "landauer",
            "forget",
            "dissipation",
            "free",
        ),
        (LessonKind::Episodic, "page", "retain", "retention", "loss"),
        (
            LessonKind::Skill,
            "markov",
            "separate",
            "blanket",
            "external",
        ),
    ]
    .into_iter()
    .map(|(kind, local, action, remote, distractor)| Lesson {
        kind,
        local: pattern(local, 0),
        action: pattern(action, 0),
        remote: pattern(remote, 1),
        distractor: pattern(distractor, 1),
    })
    .collect()
}

fn pattern(label: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..10)
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
            initial_spatial_connectivity: 0.0002,
            initial_temporal_connectivity: 0.0001,
            target_spatial_degree: 4,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 10,
            seed: 87_301,
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

#[allow(dead_code)]
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
