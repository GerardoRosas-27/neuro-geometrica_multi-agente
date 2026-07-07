use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::relational_field::{CollapseReport, ObserverId, RelationalFieldConfig};
use std::time::{Duration, Instant};

const NODES_PER_SLICE: usize = 96;
const TRAIN_EPOCHS: usize = 5;
const EVAL_REPEATS: usize = 32;

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

#[derive(Clone, Copy, Default)]
struct BenchResult {
    metrics: Metrics,
    train_elapsed: Duration,
    eval_elapsed: Duration,
    relations: usize,
    epr_links: usize,
    thermal_energy: f32,
}

fn main() {
    let lessons = lessons();
    let previous = run_previous_substrate(&lessons);
    let native = run_native_substrate(&lessons);

    println!("Native thermodynamic RQM-EPR synthetic benchmark");
    println!(
        "lessons={} train_epochs={} eval_repeats={} nodes_per_slice={}",
        lessons.len(),
        TRAIN_EPOCHS,
        EVAL_REPEATS,
        NODES_PER_SLICE
    );
    print_result("previous_cdt_rqm_epr", previous);
    print_result("native_thermo_rqm_epr", native);
    println!(
        "decision: {}",
        if native.metrics.accuracy() + 0.0001 >= previous.metrics.accuracy()
            && native.metrics.leakage() <= previous.metrics.leakage() + 0.001
            && native.eval_elapsed <= previous.eval_elapsed
        {
            "keep_native value=preserves_accuracy_and_improves_runtime"
        } else if native.metrics.accuracy() + 0.0001 >= previous.metrics.accuracy() {
            "keep_for_tuning value=preserves_accuracy_but_needs_runtime_or_leakage_tuning"
        } else {
            "recalibrate_native value=synthetic_accuracy_regressed"
        }
    );
}

fn run_previous_substrate(lessons: &[Lesson]) -> BenchResult {
    let mut substrate = CdtRqmUniverseSubstrate::new(previous_config());
    substrate.enable_entanglement(epr_config());
    let train_start = Instant::now();
    for _ in 0..TRAIN_EPOCHS {
        for lesson in lessons {
            let cue = pattern(0, lesson.cue);
            let target = pattern(1, lesson.target);
            substrate.train_observed_transition(lesson.observer, lesson.phase, &cue, &target, 1.0);
            for (&a, &b) in cue.iter().zip(&target) {
                substrate.observe_entanglement_correlation(a, b, 0.5);
            }
        }
    }
    let train_elapsed = train_start.elapsed();

    let eval_start = Instant::now();
    let mut metrics = Metrics::default();
    for _ in 0..EVAL_REPEATS {
        metrics = merge_metrics(metrics, evaluate_previous(&substrate, lessons));
    }
    let eval_elapsed = eval_start.elapsed();
    let epr_links = substrate
        .entanglement_summary()
        .map(|report| report.active_links)
        .unwrap_or_default();

    BenchResult {
        metrics,
        train_elapsed,
        eval_elapsed,
        relations: substrate.relation_count(),
        epr_links,
        thermal_energy: substrate.hardware.regge_action(),
    }
}

fn run_native_substrate(lessons: &[Lesson]) -> BenchResult {
    let mut substrate = NativeThermoRqmEprSubstrate::new(
        native_config(),
        NativeThermoRqmConfig {
            thermal_steps_per_train: 0,
            thermal_steps_per_query: 2,
            max_candidates: 24,
            thermal_score_gain: 0.35,
            thermal_activation_margin: 0.0001,
            collect_query_diagnostics: false,
            max_pilot_window_nodes: 48,
            sampling_block_size: 12,
            sampling_schedule_rounds: 1,
            max_sampling_blocks: 4,
            ..NativeThermoRqmConfig::default()
        },
        epr_config(),
    );
    let train_start = Instant::now();
    for _ in 0..TRAIN_EPOCHS {
        for lesson in lessons {
            substrate.train_observed_transition(
                lesson.observer,
                lesson.phase,
                &pattern(0, lesson.cue),
                &pattern(1, lesson.target),
                1.0,
            );
        }
    }
    let train_elapsed = train_start.elapsed();

    let eval_start = Instant::now();
    let mut metrics = Metrics::default();
    for _ in 0..EVAL_REPEATS {
        metrics = merge_metrics(metrics, evaluate_native(&mut substrate, lessons));
    }
    let eval_elapsed = eval_start.elapsed();
    let epr_links = substrate.entanglement.summary().active_links;
    let thermal_energy = substrate.thermal.report().mean_energy;

    BenchResult {
        metrics,
        train_elapsed,
        eval_elapsed,
        relations: substrate.relation_count(),
        epr_links,
        thermal_energy,
    }
}

fn evaluate_previous(substrate: &CdtRqmUniverseSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let cue = pattern(0, lesson.cue);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(lesson.observer, lesson.phase, &cue);
        metrics.record(
            score_previous(&report.collapse, &pattern(1, lesson.target)),
            score_previous(&report.collapse, &pattern(1, lesson.distractor)),
        );
    }
    metrics
}

fn evaluate_native(substrate: &mut NativeThermoRqmEprSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let report = substrate.query(lesson.observer, lesson.phase, &pattern(0, lesson.cue));
        metrics.record(
            score_native(&report.candidates, &pattern(1, lesson.target)),
            score_native(&report.candidates, &pattern(1, lesson.distractor)),
        );
    }
    metrics
}

fn score_previous(report: &CollapseReport, targets: &[usize]) -> f32 {
    report
        .candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn score_native(
    candidates: &[cdt_rqm_epr::native_thermo_rqm_epr::NativeCandidateScore],
    targets: &[usize],
) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn print_result(label: &str, result: BenchResult) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.4} train_ms={:.3} eval_ms={:.3} us_per_case={:.3} relations={} epr_links={} energy={:.4}",
        label,
        result.metrics.accuracy() * 100.0,
        result.metrics.leakage() * 100.0,
        result.metrics.margin(),
        result.train_elapsed.as_secs_f64() * 1_000.0,
        result.eval_elapsed.as_secs_f64() * 1_000.0,
        result.eval_elapsed.as_secs_f64() * 1_000_000.0 / result.metrics.cases.max(1) as f64,
        result.relations,
        result.epr_links,
        result.thermal_energy
    );
}

fn merge_metrics(left: Metrics, right: Metrics) -> Metrics {
    Metrics {
        cases: left.cases + right.cases,
        correct: left.correct + right.correct,
        leakage_sum: left.leakage_sum + right.leakage_sum,
        margin_sum: left.margin_sum + right.margin_sum,
    }
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
                observer: ObserverId(220_000 + group),
                phase: phases[group],
                cue: group * 16 + offset * 2,
                target: group * 16 + offset * 2 + 1,
                distractor: group * 16 + ((offset + 2) % 6) * 2 + 1,
            });
        }
    }
    out
}

fn previous_config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 3,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.18,
            initial_temporal_connectivity: 0.07,
            target_spatial_degree: 4,
            target_temporal_degree: 2,
            target_tetrahedra_per_edge: 3,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.10,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 81_144,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.10,
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

fn native_config() -> NativeThermoCdtConfig {
    NativeThermoCdtConfig {
        slices: 3,
        nodes_per_slice: NODES_PER_SLICE,
        spatial_degree: 4,
        temporal_degree: 2,
        temperature: 0.24,
        dt: 0.01,
        diffusion: 0.20,
        confinement: 0.05,
        pilot_gain: 0.50,
        phase_coupling: 0.18,
        amplitude_decay: 0.003,
        state_clamp: 3.0,
        seed: 0xE9_24_51,
    }
}

fn epr_config() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 0,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}
