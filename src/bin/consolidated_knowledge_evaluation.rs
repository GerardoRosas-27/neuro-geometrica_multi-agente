use cdt_rqm_epr::cdt_rqm::CdtRqmUniverseSubstrate;
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeCandidateScore, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_engine::{
    canonical_lessons, load_consolidated_native_substrates, Lesson, LessonKind, NativeEngineConfig,
    DEFAULT_OBSERVER, DEFAULT_TRAINED_STATE,
};
use cdt_rqm_epr::relational_field::{CollapseReport, ObserverId};
use std::env;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct KnowledgeCase {
    category: &'static str,
    observer: ObserverId,
    cue: Vec<usize>,
    expected: Vec<usize>,
    distractor: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
struct KnowledgeMetrics {
    cases: usize,
    correct: usize,
    expected_sum: f32,
    distractor_sum: f32,
    leakage_sum: f32,
    margin_sum: f32,
    top_expected_sum: f32,
    top_distractor_sum: f32,
}

impl KnowledgeMetrics {
    fn record(&mut self, expected: f32, distractor: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.expected_sum += expected;
        self.distractor_sum += distractor;
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.top_expected_sum += expected.max(0.0);
        self.top_distractor_sum += distractor.max(0.0);
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

    fn expected(self) -> f32 {
        self.expected_sum / self.cases.max(1) as f32
    }

    fn distractor(self) -> f32 {
        self.distractor_sum / self.cases.max(1) as f32
    }

    fn signal_ratio(self) -> f32 {
        self.top_expected_sum / self.top_distractor_sum.max(f32::EPSILON)
    }
}

#[derive(Clone, Copy, Default)]
struct EvaluationResult {
    metrics: KnowledgeMetrics,
    elapsed: Duration,
}

fn main() {
    let state =
        env::var("NATIVE_THERMO_STATE").unwrap_or_else(|_| DEFAULT_TRAINED_STATE.to_string());
    let config = NativeEngineConfig {
        eval_repeats: env_usize("KNOWLEDGE_EVAL_REPEATS", 8),
        sleep_attempts: env_usize("NATIVE_THERMO_SLEEP_ATTEMPTS", 8),
        sleep_replay_passes: env_usize("NATIVE_THERMO_SLEEP_REPLAY_PASSES", 2),
    };

    let consolidated = match load_consolidated_native_substrates(&state, config) {
        Ok(consolidated) => consolidated,
        Err(err) => {
            println!("Consolidated knowledge evaluation");
            println!("loaded=false state={state} error={err}");
            return;
        }
    };

    let lessons = canonical_lessons();
    let cases = broad_knowledge_cases(&lessons);
    let legacy = evaluate_legacy(&consolidated.legacy, &cases, config.eval_repeats);
    let native = evaluate_native(&consolidated.native, &cases, config.eval_repeats);

    println!("Consolidated knowledge evaluation");
    println!(
        "loaded=true state={} lessons={} cases={} repeats={} sleep_attempts={} sleep_accepted={}",
        state,
        lessons.len(),
        cases.len(),
        config.eval_repeats,
        consolidated.sleep.attempts,
        consolidated.sleep.accepted
    );
    println!(
        "substrates: legacy_relations={} native_relations={} native_imported_edges={} legacy_epr_links={} native_epr_links={}",
        consolidated.migration.legacy_relations,
        consolidated.native.relation_count(),
        consolidated.migration.imported_edges,
        consolidated.migration.epr_links,
        consolidated.native.entanglement.summary().active_links
    );
    print_result("legacy_cdt_rqm_consolidated", legacy);
    print_result("native_thermodynamic_consolidated", native);
    print_category_breakdown("legacy_cdt_rqm_consolidated", |category| {
        evaluate_legacy_category(&consolidated.legacy, &cases, category, config.eval_repeats)
    });
    print_category_breakdown("native_thermodynamic_consolidated", |category| {
        evaluate_native_category(&consolidated.native, &cases, category, config.eval_repeats)
    });
    println!(
        "learning_delta: accuracy={:+.1}pp leakage={:+.1}pp margin={:+.3} signal_ratio={:.3}x",
        (native.metrics.accuracy() - legacy.metrics.accuracy()) * 100.0,
        (native.metrics.leakage() - legacy.metrics.leakage()) * 100.0,
        native.metrics.margin() - legacy.metrics.margin(),
        native.metrics.signal_ratio() / legacy.metrics.signal_ratio().max(f32::EPSILON)
    );
}

fn broad_knowledge_cases(lessons: &[Lesson]) -> Vec<KnowledgeCase> {
    let mut out = Vec::new();
    for (idx, lesson) in lessons.iter().enumerate() {
        let cross = &lessons[(idx + 1) % lessons.len()];
        out.push(KnowledgeCase {
            category: "direct_memory",
            observer: DEFAULT_OBSERVER,
            cue: lesson.local.clone(),
            expected: lesson.remote.clone(),
            distractor: lesson.distractor.clone(),
        });
        out.push(KnowledgeCase {
            category: "action_conditioned",
            observer: DEFAULT_OBSERVER,
            cue: action_cue(lesson),
            expected: lesson.remote.clone(),
            distractor: lesson.distractor.clone(),
        });
        out.push(KnowledgeCase {
            category: "typed_memory",
            observer: typed_observer(lesson.kind),
            cue: lesson.local.clone(),
            expected: lesson.remote.clone(),
            distractor: lesson.distractor.clone(),
        });
        out.push(KnowledgeCase {
            category: "partial_cue",
            observer: DEFAULT_OBSERVER,
            cue: prefix(&lesson.local, 5),
            expected: lesson.remote.clone(),
            distractor: lesson.distractor.clone(),
        });
        out.push(KnowledgeCase {
            category: "noisy_cue",
            observer: DEFAULT_OBSERVER,
            cue: with_noise(&lesson.local, &lesson.distractor),
            expected: lesson.remote.clone(),
            distractor: lesson.distractor.clone(),
        });
        out.push(KnowledgeCase {
            category: "cross_distractor",
            observer: DEFAULT_OBSERVER,
            cue: lesson.local.clone(),
            expected: lesson.remote.clone(),
            distractor: cross.remote.clone(),
        });
    }
    out
}

fn evaluate_legacy(
    substrate: &CdtRqmUniverseSubstrate,
    cases: &[KnowledgeCase],
    repeats: usize,
) -> EvaluationResult {
    let start = Instant::now();
    let mut metrics = KnowledgeMetrics::default();
    for _ in 0..repeats {
        let mut trial = substrate.clone();
        for case in cases {
            trial.hardware.clear_activity();
            trial.hardware.inject_pattern(&case.cue, 1.0);
            let report = trial.step_from_boundary(case.observer, 0.0, &case.cue);
            metrics.record(
                score_legacy(&report.collapse, &case.expected),
                score_legacy(&report.collapse, &case.distractor),
            );
        }
    }
    EvaluationResult {
        metrics,
        elapsed: start.elapsed(),
    }
}

fn evaluate_native(
    substrate: &NativeThermoRqmEprSubstrate,
    cases: &[KnowledgeCase],
    repeats: usize,
) -> EvaluationResult {
    let start = Instant::now();
    let mut metrics = KnowledgeMetrics::default();
    for _ in 0..repeats {
        let mut trial = substrate.clone();
        for case in cases {
            let report = trial.query(case.observer, 0.0, &case.cue);
            metrics.record(
                score_native(&report.candidates, &case.expected),
                score_native(&report.candidates, &case.distractor),
            );
        }
    }
    EvaluationResult {
        metrics,
        elapsed: start.elapsed(),
    }
}

fn evaluate_legacy_category(
    substrate: &CdtRqmUniverseSubstrate,
    cases: &[KnowledgeCase],
    category: &str,
    repeats: usize,
) -> EvaluationResult {
    let filtered = cases
        .iter()
        .filter(|case| case.category == category)
        .cloned()
        .collect::<Vec<_>>();
    evaluate_legacy(substrate, &filtered, repeats)
}

fn evaluate_native_category(
    substrate: &NativeThermoRqmEprSubstrate,
    cases: &[KnowledgeCase],
    category: &str,
    repeats: usize,
) -> EvaluationResult {
    let filtered = cases
        .iter()
        .filter(|case| case.category == category)
        .cloned()
        .collect::<Vec<_>>();
    evaluate_native(substrate, &filtered, repeats)
}

fn print_result(label: &str, result: EvaluationResult) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} expected_score={:.3} distractor_score={:.3} signal_ratio={:.3} cases={} elapsed_ms={:.3} us_per_case={:.3}",
        label,
        result.metrics.accuracy() * 100.0,
        result.metrics.leakage() * 100.0,
        result.metrics.margin(),
        result.metrics.expected(),
        result.metrics.distractor(),
        result.metrics.signal_ratio(),
        result.metrics.cases,
        result.elapsed.as_secs_f64() * 1_000.0,
        result.elapsed.as_secs_f64() * 1_000_000.0 / result.metrics.cases.max(1) as f64
    );
}

fn print_category_breakdown<F>(label: &str, mut eval: F)
where
    F: FnMut(&str) -> EvaluationResult,
{
    for category in [
        "direct_memory",
        "action_conditioned",
        "typed_memory",
        "partial_cue",
        "noisy_cue",
        "cross_distractor",
    ] {
        let result = eval(category);
        println!(
            "{} category={}: accuracy={:.1}% leakage={:.1}% margin={:.3} expected={:.3} distractor={:.3} signal_ratio={:.3}",
            label,
            category,
            result.metrics.accuracy() * 100.0,
            result.metrics.leakage() * 100.0,
            result.metrics.margin(),
            result.metrics.expected(),
            result.metrics.distractor(),
            result.metrics.signal_ratio()
        );
    }
}

fn score_legacy(report: &CollapseReport, targets: &[usize]) -> f32 {
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

fn action_cue(lesson: &Lesson) -> Vec<usize> {
    let mut cue = lesson.local.clone();
    cue.extend_from_slice(&lesson.action);
    cue.sort_unstable();
    cue.dedup();
    cue
}

fn prefix(values: &[usize], count: usize) -> Vec<usize> {
    values.iter().take(count).copied().collect()
}

fn with_noise(cue: &[usize], distractor: &[usize]) -> Vec<usize> {
    let mut noisy = cue.to_vec();
    noisy.extend(distractor.iter().take(3).copied());
    noisy.sort_unstable();
    noisy.dedup();
    noisy
}

fn typed_observer(kind: LessonKind) -> ObserverId {
    match kind {
        LessonKind::Semantic => ObserverId(261_001),
        LessonKind::Episodic => ObserverId(261_002),
        LessonKind::Causal => ObserverId(261_003),
        LessonKind::Skill => ObserverId(261_004),
    }
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}
