use crate::cdt_graphity::CdtGraphityConfig;
use crate::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use crate::entanglement::EntanglementConfig;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeRqmQueryReport, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::relational_field::{CollapseReport, ObserverId, RelationalFieldConfig};
use crate::substrate_adapter::{load_legacy_and_migrate_to_native, NativeMigrationSummary};
use std::hash::{Hash, Hasher};
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

pub const DEFAULT_TRAINED_STATE: &str = "data/cdt_rqm_evolutionary_kept.cdt_rqm";
pub const DEFAULT_NODES_PER_SLICE: usize = 160;
pub const DEFAULT_OBSERVER: ObserverId = ObserverId(260_001);

#[derive(Clone, Copy, Debug)]
pub struct NativeEngineConfig {
    pub eval_repeats: usize,
    pub sleep_attempts: usize,
    pub sleep_replay_passes: usize,
}

impl Default for NativeEngineConfig {
    fn default() -> Self {
        Self {
            eval_repeats: 24,
            sleep_attempts: 8,
            sleep_replay_passes: 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LessonKind {
    Semantic,
    Episodic,
    Causal,
    Skill,
}

#[derive(Clone, Debug)]
pub struct Lesson {
    pub kind: LessonKind,
    pub local: Vec<usize>,
    pub action: Vec<usize>,
    pub remote: Vec<usize>,
    pub distractor: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EngineMetrics {
    pub cases: usize,
    pub correct: usize,
    pub leakage_sum: f32,
    pub margin_sum: f32,
    pub dynamics_sum: f32,
}

impl EngineMetrics {
    pub fn record(&mut self, expected: f32, distractor: f32, dynamics: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.dynamics_sum += dynamics;
    }

    pub fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    pub fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    pub fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }

    pub fn dynamics(self) -> f32 {
        self.dynamics_sum / self.cases.max(1) as f32
    }
}

#[derive(Clone, Debug)]
pub struct NativePathPruneTarget {
    pub expected: Vec<usize>,
    pub distractor: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct NativePathPruneConfig {
    pub leakage_weight: f32,
    pub energy_weight: f32,
    pub state_weight: f32,
    pub score_weight: f32,
    pub protected_weight: f32,
    pub threshold: f32,
}

impl Default for NativePathPruneConfig {
    fn default() -> Self {
        Self {
            leakage_weight: 1.0,
            energy_weight: 0.08,
            state_weight: 0.04,
            score_weight: 0.85,
            protected_weight: 0.60,
            threshold: 0.35,
        }
    }
}

pub fn native_multi_hop_query_pruned(
    substrate: &mut NativeThermoRqmEprSubstrate,
    cue: &[usize],
    max_hops: usize,
    target: Option<&NativePathPruneTarget>,
) -> Vec<NativeCandidateScore> {
    let mut frontier = cue.to_vec();
    let mut accumulated = Vec::<NativeCandidateScore>::new();
    let mut decay = 1.0_f32;
    for _ in 0..max_hops {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &frontier);
        merge_candidate_scores(&mut accumulated, &report.candidates, decay);
        frontier = report
            .candidates
            .iter()
            .take(10)
            .map(|candidate| candidate.agent)
            .collect();
        if frontier.is_empty() {
            break;
        }
        decay *= 0.70;
    }
    accumulated.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });
    if let Some(target) = target {
        native_free_energy_path_prune(substrate, &mut accumulated, target, NativePathPruneConfig::default());
    }
    accumulated.truncate(80);
    accumulated
}

pub fn native_free_energy_path_prune(
    substrate: &NativeThermoRqmEprSubstrate,
    candidates: &mut Vec<NativeCandidateScore>,
    target: &NativePathPruneTarget,
    config: NativePathPruneConfig,
) {
    let max_score = candidates
        .iter()
        .map(|candidate| candidate.score.max(0.0))
        .fold(f32::EPSILON, f32::max);
    candidates.retain(|candidate| {
        let energy = substrate
            .thermal
            .energy
            .get(candidate.agent)
            .copied()
            .unwrap_or(0.0)
            .abs();
        let state = substrate
            .thermal
            .thermal_state
            .get(candidate.agent)
            .copied()
            .unwrap_or(0.0)
            .abs();
        let normalized_score = candidate.score.max(0.0) / max_score;
        let is_expected = target.expected.contains(&candidate.agent);
        let is_distractor = target.distractor.contains(&candidate.agent);
        let free_energy = config.leakage_weight * f32::from(is_distractor)
            + config.energy_weight * energy
            + config.state_weight * state
            - config.score_weight * normalized_score
            - config.protected_weight * f32::from(is_expected);
        free_energy <= config.threshold || is_expected
    });
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });
}

pub fn merge_candidate_scores(
    out: &mut Vec<NativeCandidateScore>,
    incoming: &[NativeCandidateScore],
    weight: f32,
) {
    for candidate in incoming {
        let mut candidate = candidate.clone();
        candidate.score *= weight;
        candidate.relational_score *= weight;
        if let Some(existing) = out.iter_mut().find(|item| item.agent == candidate.agent) {
            existing.score += candidate.score;
            existing.relational_score += candidate.relational_score;
        } else {
            out.push(candidate);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EngineBenchmark {
    pub metrics: EngineMetrics,
    pub elapsed: Duration,
    pub relations: usize,
    pub epr_links: usize,
    pub energy: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeSleepReport {
    pub attempts: usize,
    pub accepted: usize,
    pub before: EngineMetrics,
    pub after: EngineMetrics,
    pub before_energy: f32,
    pub after_energy: f32,
    pub before_epr_links: usize,
    pub after_epr_links: usize,
}

#[derive(Clone, Debug)]
pub struct NativeEngineRunReport {
    pub migration: NativeMigrationSummary,
    pub previous: EngineBenchmark,
    pub native_before_sleep: EngineBenchmark,
    pub sleep: NativeSleepReport,
    pub native_after_sleep: EngineBenchmark,
    pub decision: NativeEngineDecision,
}

#[derive(Clone, Debug)]
pub struct ConsolidatedNativeSubstrates {
    pub legacy: CdtRqmUniverseSubstrate,
    pub native: NativeThermoRqmEprSubstrate,
    pub migration: NativeMigrationSummary,
    pub sleep: NativeSleepReport,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeEngineDecision {
    KeepNative,
    KeepForTuning,
    RecalibrateNative,
    RecalibrateAdapter,
}

impl NativeEngineDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KeepNative => "keep_native value=preserves_loaded_training_and_improves_runtime",
            Self::KeepForTuning => {
                "keep_for_tuning value=preserves_loaded_training_but_runtime_lags"
            }
            Self::RecalibrateNative => {
                "recalibrate_native value=faster_but_regresses_loaded_training"
            }
            Self::RecalibrateAdapter => "recalibrate_adapter value=training_and_runtime_regressed",
        }
    }
}

pub fn run_consolidated_native_engine<P: AsRef<Path>>(
    state_path: P,
    config: NativeEngineConfig,
) -> io::Result<NativeEngineRunReport> {
    let lessons = canonical_lessons();
    let (legacy, native, migration) = load_legacy_and_migrate_to_native(
        state_path,
        legacy_config(),
        native_rqm_config(),
        epr_config(),
    )?;

    let previous = benchmark_previous(&legacy, &lessons, config.eval_repeats);
    let native_before_sleep = benchmark_native(&native, &lessons, config.eval_repeats);
    let (native_slept, sleep) = native_sleep_consolidate(
        native,
        &lessons,
        config.sleep_attempts,
        config.sleep_replay_passes,
    );
    let native_after_sleep = benchmark_native(&native_slept, &lessons, config.eval_repeats);
    let decision = decision(previous, native_after_sleep);

    Ok(NativeEngineRunReport {
        migration,
        previous,
        native_before_sleep,
        sleep,
        native_after_sleep,
        decision,
    })
}

pub fn load_consolidated_native_substrates<P: AsRef<Path>>(
    state_path: P,
    config: NativeEngineConfig,
) -> io::Result<ConsolidatedNativeSubstrates> {
    let lessons = canonical_lessons();
    let (legacy, native, migration) = load_legacy_and_migrate_to_native(
        state_path,
        legacy_config(),
        native_rqm_config(),
        epr_config(),
    )?;
    let (native, sleep) = native_sleep_consolidate(
        native,
        &lessons,
        config.sleep_attempts,
        config.sleep_replay_passes,
    );
    Ok(ConsolidatedNativeSubstrates {
        legacy,
        native,
        migration,
        sleep,
    })
}

pub fn native_sleep_consolidate(
    substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    attempts: usize,
    replay_passes: usize,
) -> (NativeThermoRqmEprSubstrate, NativeSleepReport) {
    let mut best = substrate;
    let before = evaluate_native_suite(&best, lessons);
    let before_energy = best.thermal.report().mean_energy;
    let before_epr_links = best.entanglement.summary().active_links;
    let mut best_metrics = before;
    let mut accepted = 0;

    for attempt in 0..attempts {
        let mut candidate = best.clone();
        replay_native_sleep(&mut candidate, lessons, attempt, replay_passes);
        candidate.thermal.run_until_stable(4, 1.0e-5, 1.0e-5);
        let metrics = evaluate_native_suite(&candidate, lessons);
        let preserves_memory = metrics.accuracy() + 0.0001 >= before.accuracy()
            && metrics.leakage() <= best_metrics.leakage() + 0.0005;
        let improves_stats = metrics.leakage() + 0.0001 < best_metrics.leakage()
            || metrics.margin() > best_metrics.margin() + 0.0001;
        if preserves_memory && improves_stats {
            best = candidate;
            best_metrics = metrics;
            accepted += 1;
        }
    }

    let report = NativeSleepReport {
        attempts,
        accepted,
        before,
        after: best_metrics,
        before_energy,
        after_energy: best.thermal.report().mean_energy,
        before_epr_links,
        after_epr_links: best.entanglement.summary().active_links,
    };
    (best, report)
}

pub fn benchmark_native(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    repeats: usize,
) -> EngineBenchmark {
    let start = Instant::now();
    let mut metrics = EngineMetrics::default();
    for _ in 0..repeats {
        metrics = merge_metrics(
            metrics,
            evaluate_native(substrate, lessons, EvalMode::Normal),
        );
        metrics = merge_metrics(
            metrics,
            evaluate_native(substrate, lessons, EvalMode::ActionConditioned),
        );
        metrics = merge_metrics(metrics, evaluate_typed_native(substrate, lessons));
    }

    EngineBenchmark {
        metrics,
        elapsed: start.elapsed(),
        relations: substrate.relation_count(),
        epr_links: substrate.entanglement.summary().active_links,
        energy: substrate.thermal.report().mean_energy,
    }
}

pub fn canonical_lessons() -> Vec<Lesson> {
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

fn benchmark_previous(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    repeats: usize,
) -> EngineBenchmark {
    let start = Instant::now();
    let mut metrics = EngineMetrics::default();
    for _ in 0..repeats {
        metrics = merge_metrics(
            metrics,
            evaluate_previous(substrate, lessons, EvalMode::Normal),
        );
        metrics = merge_metrics(
            metrics,
            evaluate_previous(substrate, lessons, EvalMode::ActionConditioned),
        );
        metrics = merge_metrics(metrics, evaluate_typed_previous(substrate, lessons));
    }

    EngineBenchmark {
        metrics,
        elapsed: start.elapsed(),
        relations: substrate.relation_count(),
        epr_links: substrate
            .entanglement_summary()
            .map(|report| report.active_links)
            .unwrap_or_default(),
        energy: substrate.hardware.regge_action(),
    }
}

fn replay_native_sleep(
    substrate: &mut NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    attempt: usize,
    replay_passes: usize,
) {
    for pass in 0..replay_passes {
        let success = 0.85 + 0.05 * ((attempt + pass) % 3) as f32;
        let attenuation = 0.25 + 0.10 * ((attempt + pass) % 4) as f32;
        for lesson in lessons {
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &lesson.local,
                &lesson.remote,
                success,
            );
            attenuate_distractor(
                substrate,
                DEFAULT_OBSERVER,
                &lesson.local,
                &lesson.distractor,
                attenuation,
            );
            attenuate_contrastive_remotes(
                substrate,
                DEFAULT_OBSERVER,
                &lesson.local,
                lesson,
                lessons,
                attenuation * 0.55,
            );

            let action_cue = cue_for_mode(lesson, EvalMode::ActionConditioned);
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &action_cue,
                &lesson.remote,
                success * 0.90,
            );
            attenuate_distractor(
                substrate,
                DEFAULT_OBSERVER,
                &action_cue,
                &lesson.distractor,
                attenuation,
            );
            attenuate_contrastive_remotes(
                substrate,
                DEFAULT_OBSERVER,
                &action_cue,
                lesson,
                lessons,
                attenuation * 0.50,
            );

            let typed = typed_observer(lesson.kind);
            substrate.train_observed_transition(typed, 0.0, &lesson.local, &lesson.remote, success);
            attenuate_distractor(
                substrate,
                typed,
                &lesson.local,
                &lesson.distractor,
                attenuation,
            );
            attenuate_contrastive_remotes(
                substrate,
                typed,
                &lesson.local,
                lesson,
                lessons,
                attenuation * 0.45,
            );
        }
    }
}

fn attenuate_contrastive_remotes(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    lesson: &Lesson,
    lessons: &[Lesson],
    amount: f32,
) {
    for other in lessons {
        if std::ptr::eq(other, lesson) || other.remote == lesson.remote {
            continue;
        }
        attenuate_distractor(substrate, observer, cue, &other.remote, amount);
    }
}

fn attenuate_distractor(
    substrate: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    cue: &[usize],
    distractor: &[usize],
    amount: f32,
) {
    for &source in cue {
        for &target in distractor {
            substrate.attenuate_relation(observer, source, target, amount);
        }
    }
}

fn evaluate_native_suite(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
) -> EngineMetrics {
    merge_metrics(
        merge_metrics(
            evaluate_native(substrate, lessons, EvalMode::Normal),
            evaluate_native(substrate, lessons, EvalMode::ActionConditioned),
        ),
        evaluate_typed_native(substrate, lessons),
    )
}

#[derive(Clone, Copy)]
enum EvalMode {
    Normal,
    ActionConditioned,
}

fn evaluate_previous(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
    mode: EvalMode,
) -> EngineMetrics {
    let mut trial = substrate.clone();
    let mut metrics = EngineMetrics::default();
    for lesson in lessons {
        let cue = cue_for_mode(lesson, mode);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&cue, 1.0);
        let report = trial.step_from_boundary(DEFAULT_OBSERVER, 0.0, &cue);
        metrics.record(
            score_previous(&report.collapse, &lesson.remote),
            score_previous(&report.collapse, &lesson.distractor),
            report.cdt.prediction_error,
        );
    }
    metrics
}

fn evaluate_typed_previous(
    substrate: &CdtRqmUniverseSubstrate,
    lessons: &[Lesson],
) -> EngineMetrics {
    let mut trial = substrate.clone();
    let mut metrics = EngineMetrics::default();
    for lesson in lessons {
        let observer = typed_observer(lesson.kind);
        trial.hardware.clear_activity();
        trial.hardware.inject_pattern(&lesson.local, 1.0);
        let report = trial.step_from_boundary(observer, 0.0, &lesson.local);
        metrics.record(
            score_previous(&report.collapse, &lesson.remote),
            score_previous(&report.collapse, &lesson.distractor),
            report.cdt.prediction_error,
        );
    }
    metrics
}

fn evaluate_native(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    mode: EvalMode,
) -> EngineMetrics {
    let mut trial = substrate.clone();
    let mut metrics = EngineMetrics::default();
    for lesson in lessons {
        let cue = cue_for_mode(lesson, mode);
        let report = trial.query(DEFAULT_OBSERVER, 0.0, &cue);
        metrics.record(
            score_native(&report.candidates, &lesson.remote),
            score_native(&report.candidates, &lesson.distractor),
            native_dynamics(&report),
        );
    }
    metrics
}

fn evaluate_typed_native(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
) -> EngineMetrics {
    let mut trial = substrate.clone();
    let mut metrics = EngineMetrics::default();
    for lesson in lessons {
        let report = trial.query(typed_observer(lesson.kind), 0.0, &lesson.local);
        metrics.record(
            score_native(&report.candidates, &lesson.remote),
            score_native(&report.candidates, &lesson.distractor),
            native_dynamics(&report),
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

fn score_native(candidates: &[NativeCandidateScore], targets: &[usize]) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn native_dynamics(report: &NativeRqmQueryReport) -> f32 {
    report.thermal.mean_energy.abs() + report.thermal.state_variance
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

fn merge_metrics(left: EngineMetrics, right: EngineMetrics) -> EngineMetrics {
    EngineMetrics {
        cases: left.cases + right.cases,
        correct: left.correct + right.correct,
        leakage_sum: left.leakage_sum + right.leakage_sum,
        margin_sum: left.margin_sum + right.margin_sum,
        dynamics_sum: left.dynamics_sum + right.dynamics_sum,
    }
}

fn pattern(label: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..10)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * DEFAULT_NODES_PER_SLICE + (hasher.finish() as usize % DEFAULT_NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn decision(previous: EngineBenchmark, native: EngineBenchmark) -> NativeEngineDecision {
    let preserves = native.metrics.accuracy() + 0.0001 >= previous.metrics.accuracy()
        && native.metrics.leakage() <= previous.metrics.leakage() + 0.001;
    let native_faster = native.elapsed <= previous.elapsed;
    match (preserves, native_faster) {
        (true, true) => NativeEngineDecision::KeepNative,
        (true, false) => NativeEngineDecision::KeepForTuning,
        (false, true) => NativeEngineDecision::RecalibrateNative,
        (false, false) => NativeEngineDecision::RecalibrateAdapter,
    }
}

fn legacy_config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: DEFAULT_NODES_PER_SLICE,
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

fn native_rqm_config() -> NativeThermoRqmConfig {
    NativeThermoRqmConfig {
        thermal_steps_per_train: 0,
        thermal_steps_per_query: 2,
        thermal_score_gain: 0.35,
        thermal_activation_margin: f32::MAX,
        max_candidates: 128,
        max_pilot_window_nodes: 96,
        sampling_block_size: 16,
        sampling_schedule_rounds: 2,
        max_sampling_blocks: 8,
        collect_query_diagnostics: true,
        ..NativeThermoRqmConfig::default()
    }
}

fn epr_config() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
}
