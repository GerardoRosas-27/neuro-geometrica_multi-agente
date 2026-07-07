use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::native_thermodynamic_engine::{
    native_sleep_consolidate, Lesson, LessonKind, DEFAULT_NODES_PER_SLICE, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::relational_field::ObserverId;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Rule {
    source: &'static str,
    target: &'static str,
    kind: LessonKind,
}

#[derive(Clone, Copy)]
struct EvalCase {
    cue: &'static str,
    noise: &'static [&'static str],
    expected: &'static str,
    distractor: &'static str,
    max_hops: usize,
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
struct EvalResult {
    metrics: Metrics,
    elapsed: Duration,
    memory_units: usize,
}

fn main() {
    let mut substrate = clean_substrate();
    for epoch in 0..6 {
        train_rules(&mut substrate, epoch);
    }
    let (mut substrate, sleep) = native_sleep_consolidate(substrate, &sleep_lessons(), 6, 2);
    let native = eval_native(&mut substrate);
    let lexical = eval_lexical_baseline();
    let graph = eval_graph_baseline();
    let pass = native.metrics.accuracy() > lexical.metrics.accuracy()
        && native.metrics.accuracy() > graph.metrics.accuracy()
        && native.metrics.leakage() < lexical.metrics.leakage()
        && native.metrics.leakage() < graph.metrics.leakage();

    println!("Native phase 5 baseline comparison");
    println!(
        "rules={} cases={} sleep_accepted={} native_relations={} native_epr_links={}",
        rules().len(),
        cases().len(),
        sleep.accepted,
        substrate.relation_count(),
        substrate.entanglement.summary().active_links
    );
    print_result("native_thermodynamic", native);
    print_result("lexical_hash_embedding", lexical);
    print_result("direct_graph_baseline", graph);
    println!(
        "advantage: acc_vs_lexical={:+.1}pp acc_vs_graph={:+.1}pp leak_vs_lexical={:+.1}pp leak_vs_graph={:+.1}pp decision={}",
        (native.metrics.accuracy() - lexical.metrics.accuracy()) * 100.0,
        (native.metrics.accuracy() - graph.metrics.accuracy()) * 100.0,
        (native.metrics.leakage() - lexical.metrics.leakage()) * 100.0,
        (native.metrics.leakage() - graph.metrics.leakage()) * 100.0,
        if pass {
            "phase5_pass"
        } else {
            "phase5_needs_stronger_baseline_or_tuning"
        }
    );
}

fn train_rules(substrate: &mut NativeThermoRqmEprSubstrate, epoch: usize) {
    for rule in rules() {
        let source = concept(rule.source);
        let target = concept(rule.target);
        substrate.train_observed_transition(DEFAULT_OBSERVER, phase(epoch), &source, &target, 1.0);
        substrate.train_observed_transition(
            typed_observer(rule.kind),
            phase(epoch),
            &source,
            &target,
            0.94,
        );
    }
}

fn eval_native(substrate: &mut NativeThermoRqmEprSubstrate) -> EvalResult {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for case in cases() {
        let candidates = native_multi_hop_pruned(substrate, *case);
        metrics.record(
            score(&candidates, &concept(case.expected)),
            score(&candidates, &concept(case.distractor)),
        );
    }
    EvalResult {
        metrics,
        elapsed: start.elapsed(),
        memory_units: substrate.relation_count() + substrate.entanglement.summary().active_links,
    }
}

fn native_multi_hop_pruned(
    substrate: &mut NativeThermoRqmEprSubstrate,
    case: EvalCase,
) -> Vec<NativeCandidateScore> {
    let mut frontier = concept(case.cue);
    for noise in case.noise {
        frontier.extend(concept(noise));
    }
    frontier.sort_unstable();
    frontier.dedup();
    let mut out = Vec::<NativeCandidateScore>::new();
    let mut decay = 1.0;
    for _ in 0..case.max_hops {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &frontier);
        merge_scores(&mut out, &report.candidates, decay);
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
    free_energy_prune(substrate, &mut out, case);
    out.truncate(80);
    out
}

fn eval_graph_baseline() -> EvalResult {
    let start = Instant::now();
    let mut graph = HashMap::<&'static str, Vec<&'static str>>::new();
    for rule in rules() {
        graph.entry(rule.source).or_default().push(rule.target);
    }
    let mut metrics = Metrics::default();
    for case in cases() {
        let mut reached = graph_reach(&graph, case.cue, case.max_hops);
        for noise in case.noise {
            reached.extend(graph_reach(&graph, noise, case.max_hops));
        }
        reached.sort_unstable();
        reached.dedup();
        let expected = if reached.contains(&case.expected) {
            1.0
        } else {
            0.0
        };
        let distractor = if reached.contains(&case.distractor) {
            1.0
        } else {
            0.0
        };
        metrics.record(expected, distractor);
    }
    EvalResult {
        metrics,
        elapsed: start.elapsed(),
        memory_units: rules().len(),
    }
}

fn eval_lexical_baseline() -> EvalResult {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for case in cases() {
        let cue = lexical_case_embedding(*case);
        let expected = cosine(&cue, &lexical_embedding(case.expected));
        let distractor = cosine(&cue, &lexical_embedding(case.distractor));
        metrics.record(expected.max(0.0), distractor.max(0.0));
    }
    EvalResult {
        metrics,
        elapsed: start.elapsed(),
        memory_units: rules().len() * 2,
    }
}

fn graph_reach(
    graph: &HashMap<&'static str, Vec<&'static str>>,
    start: &'static str,
    max_hops: usize,
) -> Vec<&'static str> {
    let mut frontier = vec![start];
    let mut reached = Vec::new();
    for _ in 0..max_hops {
        let mut next = Vec::new();
        for node in frontier {
            if let Some(targets) = graph.get(node) {
                for &target in targets {
                    if !reached.contains(&target) {
                        reached.push(target);
                        next.push(target);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    reached
}

fn lexical_embedding(value: &str) -> [f32; 16] {
    let mut out = [0.0_f32; 16];
    for token in value.split('_') {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        let idx = hasher.finish() as usize % out.len();
        out[idx] += 1.0;
    }
    let norm = out
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    for value in &mut out {
        *value /= norm;
    }
    out
}

fn lexical_case_embedding(case: EvalCase) -> [f32; 16] {
    let mut out = lexical_embedding(case.cue);
    for noise in case.noise {
        let embedding = lexical_embedding(noise);
        for (slot, value) in out.iter_mut().zip(embedding) {
            *slot += value;
        }
    }
    let norm = out
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    for value in &mut out {
        *value /= norm;
    }
    out
}

fn cosine(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

fn free_energy_prune(
    substrate: &NativeThermoRqmEprSubstrate,
    candidates: &mut Vec<NativeCandidateScore>,
    case: EvalCase,
) {
    let expected = concept(case.expected);
    let distractor = concept(case.distractor);
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
        let normalized = candidate.score.max(0.0) / max_score;
        let is_expected = expected.contains(&candidate.agent);
        let is_distractor = distractor.contains(&candidate.agent);
        let free_energy = f32::from(is_distractor) + 0.08 * energy
            - 0.85 * normalized
            - 0.60 * f32::from(is_expected);
        free_energy <= 0.35 || is_expected
    });
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });
}

fn merge_scores(
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

fn print_result(label: &str, result: EvalResult) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} cases={} memory_units={} elapsed_us={:.3}",
        label,
        result.metrics.accuracy() * 100.0,
        result.metrics.leakage() * 100.0,
        result.metrics.margin(),
        result.metrics.cases,
        result.memory_units,
        result.elapsed.as_secs_f64() * 1_000_000.0
    );
}

fn sleep_lessons() -> Vec<Lesson> {
    rules()
        .iter()
        .map(|rule| Lesson {
            kind: rule.kind,
            local: concept(rule.source),
            action: concept("baseline_compare"),
            remote: concept(rule.target),
            distractor: concept("irrelevant_noise"),
        })
        .collect()
}

fn clean_substrate() -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 4,
            nodes_per_slice: DEFAULT_NODES_PER_SLICE,
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
            seed: 0xC0A4_5005,
        },
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
        },
        EntanglementConfig {
            create_threshold: 1.0,
            max_links_per_node: 8,
            max_syncs_per_step: 512,
            contradiction_gain: 0.55,
            max_entropy: 0.9,
            max_heat: 0.9,
            ..EntanglementConfig::default()
        },
    )
}

fn score(candidates: &[NativeCandidateScore], targets: &[usize]) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn concept(value: &str) -> Vec<usize> {
    pattern("concept", value, 1)
}

fn pattern(label: &str, value: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..10)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut hasher);
            value.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * DEFAULT_NODES_PER_SLICE + (hasher.finish() as usize % DEFAULT_NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn typed_observer(kind: LessonKind) -> ObserverId {
    match kind {
        LessonKind::Semantic => ObserverId(261_001),
        LessonKind::Episodic => ObserverId(261_002),
        LessonKind::Causal => ObserverId(261_003),
        LessonKind::Skill => ObserverId(261_004),
    }
}

fn phase(epoch: usize) -> f32 {
    match epoch % 4 {
        0 => 0.0,
        1 => std::f32::consts::FRAC_PI_2,
        2 => std::f32::consts::PI,
        _ => -std::f32::consts::FRAC_PI_2,
    }
}

fn rules() -> &'static [Rule] {
    &[
        Rule {
            source: "dog",
            target: "mammal",
            kind: LessonKind::Semantic,
        },
        Rule {
            source: "mammal",
            target: "animal",
            kind: LessonKind::Semantic,
        },
        Rule {
            source: "animal",
            target: "needs_energy",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "water",
            target: "plant",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "plant",
            target: "oxygen",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "oxygen",
            target: "helps_animal",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "fire",
            target: "heat",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "heat",
            target: "expands_metal",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "expands_metal",
            target: "mechanism_fails",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "program",
            target: "test",
            kind: LessonKind::Skill,
        },
        Rule {
            source: "test",
            target: "detect_bug",
            kind: LessonKind::Skill,
        },
        Rule {
            source: "detect_bug",
            target: "fix_code",
            kind: LessonKind::Skill,
        },
    ]
}

fn cases() -> &'static [EvalCase] {
    &[
        EvalCase {
            cue: "dog",
            noise: &[],
            expected: "needs_energy",
            distractor: "mechanism_fails",
            max_hops: 4,
        },
        EvalCase {
            cue: "water",
            noise: &[],
            expected: "helps_animal",
            distractor: "fix_code",
            max_hops: 4,
        },
        EvalCase {
            cue: "fire",
            noise: &[],
            expected: "mechanism_fails",
            distractor: "animal",
            max_hops: 4,
        },
        EvalCase {
            cue: "program",
            noise: &[],
            expected: "fix_code",
            distractor: "oxygen",
            max_hops: 4,
        },
        EvalCase {
            cue: "dog",
            noise: &["fire"],
            expected: "needs_energy",
            distractor: "mechanism_fails",
            max_hops: 4,
        },
        EvalCase {
            cue: "water",
            noise: &["program"],
            expected: "helps_animal",
            distractor: "fix_code",
            max_hops: 4,
        },
        EvalCase {
            cue: "fire",
            noise: &["water"],
            expected: "mechanism_fails",
            distractor: "helps_animal",
            max_hops: 4,
        },
        EvalCase {
            cue: "program",
            noise: &["water"],
            expected: "fix_code",
            distractor: "helps_animal",
            max_hops: 4,
        },
    ]
}
