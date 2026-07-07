use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::relational_field::{CollapseReport, ObserverId, RelationalFieldConfig};
use cdt_rqm_epr::thermodynamic_substrate::{
    PilotForceField, ThermodynamicConfig, ThermodynamicStepReport, ThermodynamicSubstrate,
};
use std::hash::{Hash, Hasher};
use std::mem;
use std::time::{Duration, Instant};

const STATE: &str = "data/cdt_rqm_evolutionary_kept.cdt_rqm";
const NODES_PER_SLICE: usize = 160;
const OBSERVER: ObserverId = ObserverId(260_001);
const EVAL_REPEATS: usize = 24;
const THERMAL_MICROSTEPS: usize = 8;
const THERMAL_GAIN: f32 = 0.35;

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

#[derive(Clone, Copy, Default)]
struct Benchmark {
    metrics: Metrics,
    elapsed: Duration,
    thermal_report: Option<ThermodynamicStepReport>,
}

struct CandidatePilotField {
    forces: Vec<f32>,
}

impl CandidatePilotField {
    fn from_report(node_count: usize, report: &CollapseReport) -> Self {
        let mut forces = vec![0.0; node_count];
        let max_score = report
            .candidates
            .iter()
            .map(|candidate| candidate.score.abs())
            .fold(f32::EPSILON, f32::max);

        for candidate in &report.candidates {
            if candidate.agent < forces.len() {
                forces[candidate.agent] = candidate.score / max_score;
            }
        }

        for &seed in &report.seeds {
            if seed < forces.len() {
                forces[seed] = 1.0;
            }
        }

        Self { forces }
    }
}

impl PilotForceField for CandidatePilotField {
    fn write_forces(&self, forces: &mut [f32]) {
        for (slot, force) in forces.iter_mut().zip(self.forces.iter().copied()) {
            *slot = force;
        }
    }
}

fn main() {
    let lessons = lessons();
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    let loaded = substrate.load_consolidated_state(STATE).is_ok();
    if !loaded {
        println!("CDT-RQM native thermodynamic benchmark");
        println!("loaded=false state={STATE}");
        return;
    }

    let baseline = benchmark_classical(&substrate, &lessons, EVAL_REPEATS);
    let thermal = benchmark_thermal(&substrate, &lessons, EVAL_REPEATS);
    let knowledge_preserved = thermal.metrics.accuracy() + 0.0001 >= baseline.metrics.accuracy()
        && thermal.metrics.leakage() <= baseline.metrics.leakage() + 0.001;
    let stats_improved = thermal.metrics.margin() > baseline.metrics.margin()
        || thermal.metrics.prediction_error() < baseline.metrics.prediction_error()
        || thermal.metrics.leakage() < baseline.metrics.leakage();

    println!("CDT-RQM native thermodynamic benchmark");
    println!(
        "loaded=true state={} lessons={} repeats={} thermal_microsteps={} thermal_gain={:.2}",
        STATE,
        lessons.len(),
        EVAL_REPEATS,
        THERMAL_MICROSTEPS,
        THERMAL_GAIN
    );
    print_benchmark("classical_cdt_rqm", baseline);
    print_benchmark("native_thermodynamic_overlay", thermal);
    print_resource_model(&substrate, &thermal);
    println!(
        "decision: {}",
        if knowledge_preserved && stats_improved {
            "keep_thermal_overlay value=preserves_trained_knowledge_and_improves_statistics"
        } else if knowledge_preserved {
            "keep_for_further_tuning value=preserves_trained_knowledge_without_clear_stat_gain"
        } else {
            "discard_or_recalibrate value=thermal_overlay_degrades_trained_knowledge"
        }
    );
}

fn benchmark_classical(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    repeats: usize,
) -> Benchmark {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for _ in 0..repeats {
        metrics = merge_metrics(
            metrics,
            evaluate_classical(substrate, lessons, EvalMode::Normal),
        );
        metrics = merge_metrics(
            metrics,
            evaluate_classical(substrate, lessons, EvalMode::ActionConditioned),
        );
        metrics = merge_metrics(metrics, evaluate_typed_classical(substrate, lessons));
    }

    Benchmark {
        metrics,
        elapsed: start.elapsed(),
        thermal_report: None,
    }
}

fn benchmark_thermal(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    repeats: usize,
) -> Benchmark {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    let mut last_report = None;
    for _ in 0..repeats {
        let normal = evaluate_thermal(substrate, lessons, EvalMode::Normal);
        metrics = merge_metrics(metrics, normal.metrics);

        let action = evaluate_thermal(substrate, lessons, EvalMode::ActionConditioned);
        metrics = merge_metrics(metrics, action.metrics);

        let typed = evaluate_typed_thermal(substrate, lessons);
        last_report = typed.thermal_report;
        metrics = merge_metrics(metrics, typed.metrics);
    }

    Benchmark {
        metrics,
        elapsed: start.elapsed(),
        thermal_report: last_report,
    }
}

#[derive(Clone, Copy)]
enum EvalMode {
    Normal,
    ActionConditioned,
}

fn evaluate_classical(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    mode: EvalMode,
) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let cue = cue_for_mode(lesson, mode);
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

fn evaluate_typed_classical(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let observer = typed_observer(lesson.kind);
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

fn evaluate_thermal(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    mode: EvalMode,
) -> Benchmark {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    let mut thermal = thermal_substrate(substrate);
    let mut last_report = None;
    for lesson in lessons {
        let cue = cue_for_mode(lesson, mode);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(OBSERVER, 0.0, &cue);
        let thermal_report = relax_report(&mut thermal, &report.collapse);
        metrics.record(
            thermal_score(&thermal, &report.collapse, &lesson.remote),
            thermal_score(&thermal, &report.collapse, &lesson.distractor),
            report.cdt.prediction_error,
        );
        last_report = Some(thermal_report);
    }

    Benchmark {
        metrics,
        elapsed: Duration::ZERO,
        thermal_report: last_report,
    }
}

fn evaluate_typed_thermal(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Benchmark {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    let mut thermal = thermal_substrate(substrate);
    let mut last_report = None;
    for lesson in lessons {
        let observer = typed_observer(lesson.kind);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&lesson.local, 1.0);
        let report = trial.step_from_boundary(observer, 0.0, &lesson.local);
        let thermal_report = relax_report(&mut thermal, &report.collapse);
        metrics.record(
            thermal_score(&thermal, &report.collapse, &lesson.remote),
            thermal_score(&thermal, &report.collapse, &lesson.distractor),
            report.cdt.prediction_error,
        );
        last_report = Some(thermal_report);
    }

    Benchmark {
        metrics,
        elapsed: Duration::ZERO,
        thermal_report: last_report,
    }
}

fn relax_report(
    thermal: &mut ThermodynamicSubstrate,
    report: &CollapseReport,
) -> ThermodynamicStepReport {
    thermal.clear_pilot_forces();
    thermal.apply_pilot_field(&CandidatePilotField::from_report(thermal.len(), report));
    let mut thermal_report = thermal.report();
    for _ in 0..THERMAL_MICROSTEPS {
        thermal_report = thermal.step_langevin();
    }
    thermal_report
}

fn thermal_score(
    thermal: &ThermodynamicSubstrate,
    report: &CollapseReport,
    targets: &[usize],
) -> f32 {
    let probabilities = thermal.boltzmann_probabilities();
    let mean_probability = 1.0 / probabilities.len().max(1) as f32;
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| {
            let thermal_probability = probabilities
                .get(candidate.agent)
                .copied()
                .unwrap_or(mean_probability);
            let normalized = thermal_probability / mean_probability;
            let multiplier = (1.0 + THERMAL_GAIN * (normalized - 1.0)).clamp(0.25, 4.0);
            candidate.score * multiplier
        })
        .sum()
}

fn thermal_substrate(substrate: &CdtRqmUniverseSubstrate) -> ThermodynamicSubstrate {
    let node_count = substrate.hardware.nodes.len().max(1);
    let mut thermal = ThermodynamicSubstrate::new(ThermodynamicConfig {
        size: node_count,
        temperature: substrate.hardware.temperature.max(0.05),
        dt: 0.015,
        confinement: 0.06,
        initial_state_min: -0.05,
        initial_state_max: 0.05,
        state_clamp: 3.0,
        seed: 0x7A11_900D,
    });

    for node in &substrate.hardware.nodes {
        if node.id < thermal.states.len() {
            thermal.states[node.id] = node.surprise.clamp(-1.0, 1.0);
        }
    }

    thermal
}

fn score(report: &CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn cue_for_mode(lesson: &Lesson, mode: EvalMode) -> Vec<usize> {
    let mut cue = lesson.local.clone();
    if matches!(mode, EvalMode::ActionConditioned) {
        cue.extend_from_slice(&lesson.action);
        cue.sort_unstable();
        cue.dedup();
    }
    cue
}

fn typed_observer(kind: LessonKind) -> ObserverId {
    match kind {
        LessonKind::Semantic => ObserverId(261_001),
        LessonKind::Episodic => ObserverId(261_002),
        LessonKind::Causal => ObserverId(261_003),
        LessonKind::Skill => ObserverId(261_004),
    }
}

fn merge_metrics(left: Metrics, right: Metrics) -> Metrics {
    Metrics {
        cases: left.cases + right.cases,
        correct: left.correct + right.correct,
        leakage_sum: left.leakage_sum + right.leakage_sum,
        margin_sum: left.margin_sum + right.margin_sum,
        prediction_error_sum: left.prediction_error_sum + right.prediction_error_sum,
    }
}

fn print_benchmark(label: &str, benchmark: Benchmark) {
    let micros_per_case =
        benchmark.elapsed.as_secs_f64() * 1_000_000.0 / benchmark.metrics.cases.max(1) as f64;
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} prediction_error={:.3} cases={} elapsed_ms={:.3} us_per_case={:.3}",
        label,
        benchmark.metrics.accuracy() * 100.0,
        benchmark.metrics.leakage() * 100.0,
        benchmark.metrics.margin(),
        benchmark.metrics.prediction_error(),
        benchmark.metrics.cases,
        benchmark.elapsed.as_secs_f64() * 1_000.0,
        micros_per_case
    );

    if let Some(report) = benchmark.thermal_report {
        println!(
            "{}_thermal: tick={} mean_state={:.4} variance={:.4} mean_energy={:.4} free_energy_proxy={:.4} partition_proxy={:.4}",
            label,
            report.tick,
            report.mean_state,
            report.state_variance,
            report.mean_energy,
            report.free_energy_proxy,
            report.boltzmann_partition_proxy
        );
    }
}

fn print_resource_model(substrate: &CdtRqmUniverseSubstrate, thermal: &Benchmark) {
    let node_count = substrate.hardware.nodes.len();
    let active_edges = substrate.hardware.active_edge_count();
    let relations = substrate.relation_count();
    let node_bytes = substrate
        .hardware
        .nodes
        .first()
        .map(mem::size_of_val)
        .unwrap_or(0);
    let edge_bytes = substrate
        .hardware
        .edges
        .first()
        .map(mem::size_of_val)
        .unwrap_or(0);
    let tetrahedron_bytes = substrate
        .hardware
        .tetrahedra
        .first()
        .map(mem::size_of_val)
        .unwrap_or(0);
    let classical_bytes = node_count * node_bytes
        + substrate.hardware.edges.len() * edge_bytes
        + substrate.hardware.tetrahedra.len() * tetrahedron_bytes
        + relations * 64;
    let thermal_bytes = node_count * mem::size_of::<f32>() * 4;
    let speed_note = if thermal.elapsed.is_zero() {
        0.0
    } else {
        thermal.metrics.cases as f64 / thermal.elapsed.as_secs_f64()
    };

    println!(
        "resources: nodes={} active_edges={} relations={} classical_est_kib={:.1} thermal_overlay_kib={:.1} thermal_cases_per_sec={:.1} regge={:.3} causality_violations={}",
        node_count,
        active_edges,
        relations,
        classical_bytes as f32 / 1024.0,
        thermal_bytes as f32 / 1024.0,
        speed_note,
        substrate.hardware.regge_action(),
        substrate.hardware.causality_violations()
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
