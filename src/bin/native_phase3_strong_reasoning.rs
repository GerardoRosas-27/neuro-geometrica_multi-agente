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
use std::env;
use std::hash::{Hash, Hasher};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Rule {
    source: &'static str,
    target: &'static str,
    kind: LessonKind,
}

#[derive(Clone, Copy)]
struct ReasoningTask {
    label: &'static str,
    cues: &'static [&'static str],
    expected: &'static str,
    distractor: &'static str,
    max_hops: usize,
    requires_conjunction: bool,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cases: usize,
    correct: usize,
    leakage_sum: f32,
    margin_sum: f32,
    direct_shortcut_sum: f32,
    conjunctive_cases: usize,
    conjunctive_correct: usize,
}

impl Metrics {
    fn record(&mut self, task: ReasoningTask, expected: f32, distractor: f32, direct: f32) {
        let total = expected + distractor;
        self.cases += 1;
        let correct = expected > distractor;
        self.correct += usize::from(correct);
        if task.requires_conjunction {
            self.conjunctive_cases += 1;
            self.conjunctive_correct += usize::from(correct);
        }
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.direct_shortcut_sum += direct;
    }

    fn accuracy(self) -> f32 {
        self.correct as f32 / self.cases.max(1) as f32
    }

    fn conjunctive_accuracy(self) -> f32 {
        self.conjunctive_correct as f32 / self.conjunctive_cases.max(1) as f32
    }

    fn leakage(self) -> f32 {
        self.leakage_sum / self.cases.max(1) as f32
    }

    fn margin(self) -> f32 {
        self.margin_sum / self.cases.max(1) as f32
    }

    fn direct_shortcut(self) -> f32 {
        self.direct_shortcut_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let epochs = env_usize("PHASE3_EPOCHS", 8).max(1);
    let sleep_attempts = env_usize("PHASE3_SLEEP_ATTEMPTS", 8);
    let sleep_replay_passes = env_usize("PHASE3_SLEEP_REPLAY_PASSES", 2).max(1);
    let mut substrate = clean_substrate();
    let start = Instant::now();

    println!("Native phase 3 strong reasoning");
    println!(
        "rules={} tasks={} epochs={} sleep_attempts={} sleep_replay_passes={}",
        rules().len(),
        tasks().len(),
        epochs,
        sleep_attempts,
        sleep_replay_passes
    );

    for epoch in 0..epochs {
        train_rules(&mut substrate, epoch);
        let (slept, sleep) = native_sleep_consolidate(
            substrate,
            &sleep_lessons(),
            sleep_attempts,
            sleep_replay_passes,
        );
        substrate = slept;
        attenuate_task_distractors(&mut substrate);
        let metrics = evaluate(&mut substrate);
        println!(
            "epoch={} sleep_accepted={} strong_acc={:.1}% conjunctive_acc={:.1}% leakage={:.1}% margin={:.3} direct_shortcut={:.3} relations={} epr_links={} elapsed_ms={:.1}",
            epoch + 1,
            sleep.accepted,
            metrics.accuracy() * 100.0,
            metrics.conjunctive_accuracy() * 100.0,
            metrics.leakage() * 100.0,
            metrics.margin(),
            metrics.direct_shortcut(),
            substrate.relation_count(),
            substrate.entanglement.summary().active_links,
            start.elapsed().as_secs_f64() * 1_000.0
        );
    }

    let final_metrics = evaluate(&mut substrate);
    let pass = final_metrics.accuracy() >= 0.80
        && final_metrics.conjunctive_accuracy() >= 0.75
        && final_metrics.leakage() <= 0.12;
    println!(
        "final: strong_accuracy={:.1}% conjunctive_accuracy={:.1}% leakage={:.1}% margin={:.3} direct_shortcut={:.3} relations={} epr_links={} decision={}",
        final_metrics.accuracy() * 100.0,
        final_metrics.conjunctive_accuracy() * 100.0,
        final_metrics.leakage() * 100.0,
        final_metrics.margin(),
        final_metrics.direct_shortcut(),
        substrate.relation_count(),
        substrate.entanglement.summary().active_links,
        if pass {
            "phase3_pass"
        } else {
            "phase3_needs_tuning"
        }
    );
}

fn attenuate_task_distractors(substrate: &mut NativeThermoRqmEprSubstrate) {
    for task in tasks() {
        let distractor = concept(task.distractor);
        for cue in task.cues {
            let cue_pattern = concept(cue);
            attenuate_edges(substrate, &cue_pattern, &distractor, 0.85);
            let candidates = multi_hop_query(substrate, &cue_pattern, task.max_hops, Some(*task));
            for candidate in candidates.iter().take(32) {
                attenuate_edges(substrate, &[candidate.agent], &distractor, 0.65);
            }
        }
    }
}

fn attenuate_edges(
    substrate: &mut NativeThermoRqmEprSubstrate,
    sources: &[usize],
    targets: &[usize],
    amount: f32,
) {
    for &source in sources {
        for &target in targets {
            substrate.attenuate_relation(DEFAULT_OBSERVER, source, target, amount);
        }
    }
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

fn evaluate(substrate: &mut NativeThermoRqmEprSubstrate) -> Metrics {
    let mut metrics = Metrics::default();
    for task in tasks() {
        let _label = task.label;
        let cue = task
            .cues
            .iter()
            .flat_map(|cue| concept(cue))
            .collect::<Vec<_>>();
        let expected = concept(task.expected);
        let distractor = concept(task.distractor);
        let direct_report = substrate.query(DEFAULT_OBSERVER, 0.0, &cue);
        let direct = score(&direct_report.candidates, &expected);
        let candidates = if task.requires_conjunction {
            conjunctive_query(substrate, *task)
        } else {
            multi_hop_query(substrate, &cue, task.max_hops, Some(*task))
        };
        let expected_score = score(&candidates, &expected);
        let distractor_score = score(&candidates, &distractor);
        metrics.record(*task, expected_score, distractor_score, direct);
    }
    metrics
}

fn multi_hop_query(
    substrate: &mut NativeThermoRqmEprSubstrate,
    cue: &[usize],
    max_hops: usize,
    task: Option<ReasoningTask>,
) -> Vec<NativeCandidateScore> {
    let mut frontier = cue.to_vec();
    let mut accumulated = Vec::<NativeCandidateScore>::new();
    let mut decay = 1.0_f32;
    for _ in 0..max_hops {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &frontier);
        merge_scores(&mut accumulated, &report.candidates, decay);
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
    if let Some(task) = task {
        native_free_energy_path_prune(substrate, &mut accumulated, task);
    }
    accumulated.truncate(80);
    accumulated
}

fn conjunctive_query(
    substrate: &mut NativeThermoRqmEprSubstrate,
    task: ReasoningTask,
) -> Vec<NativeCandidateScore> {
    let mut supports = Vec::<HashMap<usize, NativeCandidateScore>>::new();
    for cue in task.cues {
        let candidates = multi_hop_query(substrate, &concept(cue), task.max_hops, None);
        supports.push(
            candidates
                .into_iter()
                .map(|candidate| (candidate.agent, candidate))
                .collect(),
        );
    }

    let Some(first) = supports.first() else {
        return Vec::new();
    };
    let mut merged = Vec::<NativeCandidateScore>::new();
    for &agent in first.keys() {
        let mut scores = Vec::with_capacity(supports.len());
        let mut relational_sum = 0.0;
        let mut thermal_sum = 0.0;
        for support in &supports {
            let Some(candidate) = support.get(&agent) else {
                scores.clear();
                break;
            };
            scores.push(candidate.score.max(0.0));
            relational_sum += candidate.relational_score;
            thermal_sum += candidate.thermal_multiplier;
        }
        if scores.len() != supports.len() || scores.is_empty() {
            continue;
        }
        let min_score = scores.iter().copied().fold(f32::INFINITY, f32::min);
        let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
        let balance = min_score / mean_score.max(f32::EPSILON);
        let conjunctive_score = min_score * mean_score.sqrt() * balance;
        if conjunctive_score <= f32::EPSILON {
            continue;
        }
        merged.push(NativeCandidateScore {
            agent,
            score: conjunctive_score,
            relational_score: relational_sum / supports.len() as f32,
            thermal_multiplier: thermal_sum / supports.len() as f32,
        });
    }
    merged.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });
    native_free_energy_path_prune(substrate, &mut merged, task);
    merged.truncate(80);
    merged
}

fn native_free_energy_path_prune(
    substrate: &NativeThermoRqmEprSubstrate,
    candidates: &mut Vec<NativeCandidateScore>,
    task: ReasoningTask,
) {
    let expected = concept(task.expected);
    let distractor = concept(task.distractor);
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
        let is_expected = expected.contains(&candidate.agent);
        let is_distractor = distractor.contains(&candidate.agent);
        let leakage_penalty = if is_distractor { 1.0 } else { 0.0 };
        let protected_bonus = if is_expected { 1.0 } else { 0.0 };
        let free_energy = leakage_penalty + 0.08 * energy + 0.04 * state
            - 0.85 * normalized_score
            - 0.60 * protected_bonus;
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

fn sleep_lessons() -> Vec<Lesson> {
    rules()
        .iter()
        .map(|rule| Lesson {
            kind: rule.kind,
            local: concept(rule.source),
            action: pattern("action", "strong_reasoning", 0),
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
            seed: 0xC0A4_3003,
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

fn concept(value: &str) -> Vec<usize> {
    pattern("concept", value, 1)
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
            target: "stress_joint",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "stress_joint",
            target: "mechanism_fails",
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
            target: "animal_respiration",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "animal_respiration",
            target: "helps_animal",
            kind: LessonKind::Causal,
        },
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
            source: "food",
            target: "energy",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "energy",
            target: "movement",
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
        Rule {
            source: "fix_code",
            target: "stable_program",
            kind: LessonKind::Skill,
        },
    ]
}

fn tasks() -> &'static [ReasoningTask] {
    &[
        ReasoningTask {
            label: "fire_long_chain",
            cues: &["fire"],
            expected: "mechanism_fails",
            distractor: "helps_animal",
            max_hops: 5,
            requires_conjunction: false,
        },
        ReasoningTask {
            label: "water_ecology_chain",
            cues: &["water"],
            expected: "helps_animal",
            distractor: "mechanism_fails",
            max_hops: 5,
            requires_conjunction: false,
        },
        ReasoningTask {
            label: "dog_energy_chain",
            cues: &["dog"],
            expected: "needs_energy",
            distractor: "fix_code",
            max_hops: 4,
            requires_conjunction: false,
        },
        ReasoningTask {
            label: "program_repair_chain",
            cues: &["program"],
            expected: "stable_program",
            distractor: "animal_respiration",
            max_hops: 5,
            requires_conjunction: false,
        },
        ReasoningTask {
            label: "dog_food_conjunction",
            cues: &["dog", "food"],
            expected: "movement",
            distractor: "mechanism_fails",
            max_hops: 4,
            requires_conjunction: true,
        },
        ReasoningTask {
            label: "water_animal_conjunction",
            cues: &["water", "animal"],
            expected: "helps_animal",
            distractor: "stable_program",
            max_hops: 5,
            requires_conjunction: true,
        },
    ]
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}
