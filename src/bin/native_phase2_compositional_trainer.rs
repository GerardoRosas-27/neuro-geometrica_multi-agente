use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::native_thermodynamic_engine::{
    native_sleep_consolidate, Lesson, LessonKind, DEFAULT_NODES_PER_SLICE, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::relational_field::ObserverId;
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
struct Task {
    cue: &'static str,
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
    direct_leak_sum: f32,
}

impl Metrics {
    fn record(&mut self, expected: f32, distractor: f32, direct_expected: f32) {
        let total = expected + distractor;
        self.cases += 1;
        self.correct += usize::from(expected > distractor);
        self.leakage_sum += if total > f32::EPSILON {
            distractor / total
        } else {
            1.0
        };
        self.margin_sum += expected - distractor;
        self.direct_leak_sum += direct_expected;
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

    fn direct_shortcut(self) -> f32 {
        self.direct_leak_sum / self.cases.max(1) as f32
    }
}

fn main() {
    let epochs = env_usize("PHASE2_EPOCHS", 6).max(1);
    let sleep_attempts = env_usize("PHASE2_SLEEP_ATTEMPTS", 6);
    let sleep_replay_passes = env_usize("PHASE2_SLEEP_REPLAY_PASSES", 2).max(1);
    let mut substrate = clean_substrate();
    let start = Instant::now();

    println!("Native phase 2 compositional trainer");
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
        let lessons = sleep_lessons();
        let (slept, sleep) =
            native_sleep_consolidate(substrate, &lessons, sleep_attempts, sleep_replay_passes);
        substrate = slept;
        let metrics = evaluate_tasks(&mut substrate);
        println!(
            "epoch={} sleep_accepted={} compositional_acc={:.1}% leakage={:.1}% margin={:.3} direct_shortcut={:.3} relations={} epr_links={} elapsed_ms={:.1}",
            epoch + 1,
            sleep.accepted,
            metrics.accuracy() * 100.0,
            metrics.leakage() * 100.0,
            metrics.margin(),
            metrics.direct_shortcut(),
            substrate.relation_count(),
            substrate.entanglement.summary().active_links,
            start.elapsed().as_secs_f64() * 1_000.0
        );
    }

    let final_metrics = evaluate_tasks(&mut substrate);
    println!(
        "final: compositional_accuracy={:.1}% leakage={:.1}% margin={:.3} direct_shortcut={:.3} relations={} epr_links={} decision={}",
        final_metrics.accuracy() * 100.0,
        final_metrics.leakage() * 100.0,
        final_metrics.margin(),
        final_metrics.direct_shortcut(),
        substrate.relation_count(),
        substrate.entanglement.summary().active_links,
        if final_metrics.accuracy() >= 0.80 && final_metrics.leakage() <= 0.10 {
            "phase2_pass"
        } else {
            "phase2_needs_tuning"
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
            0.96,
        );
    }
}

fn evaluate_tasks(substrate: &mut NativeThermoRqmEprSubstrate) -> Metrics {
    let mut metrics = Metrics::default();
    for task in tasks() {
        let cue = concept(task.cue);
        let expected = concept(task.expected);
        let distractor = concept(task.distractor);
        let direct = substrate.query(DEFAULT_OBSERVER, 0.0, &cue);
        let direct_expected = score(&direct.candidates, &expected);
        let candidates = multi_hop_query(substrate, &cue, task.max_hops);
        let expected_score = score(&candidates, &expected);
        let distractor_score = score(&candidates, &distractor);
        metrics.record(expected_score, distractor_score, direct_expected);
    }
    metrics
}

fn multi_hop_query(
    substrate: &mut NativeThermoRqmEprSubstrate,
    cue: &[usize],
    max_hops: usize,
) -> Vec<NativeCandidateScore> {
    let mut frontier = cue.to_vec();
    let mut accumulated: Vec<NativeCandidateScore> = Vec::new();
    let mut decay = 1.0_f32;
    for _ in 0..max_hops {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &frontier);
        if report.candidates.is_empty() {
            break;
        }
        for mut candidate in report.candidates.iter().cloned() {
            candidate.score *= decay;
            candidate.relational_score *= decay;
            if let Some(existing) = accumulated
                .iter_mut()
                .find(|existing| existing.agent == candidate.agent)
            {
                existing.score += candidate.score;
                existing.relational_score += candidate.relational_score;
            } else {
                accumulated.push(candidate);
            }
        }
        frontier = report
            .candidates
            .iter()
            .take(8)
            .map(|candidate| candidate.agent)
            .collect();
        decay *= 0.72;
    }
    accumulated.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });
    accumulated.truncate(64);
    accumulated
}

fn sleep_lessons() -> Vec<Lesson> {
    rules()
        .iter()
        .map(|rule| Lesson {
            kind: rule.kind,
            local: concept(rule.source),
            action: pattern("action", "compose", 0),
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
            seed: 0xC0A4_2002,
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

fn tasks() -> &'static [Task] {
    &[
        Task {
            cue: "dog",
            expected: "needs_energy",
            distractor: "mechanism_fails",
            max_hops: 4,
        },
        Task {
            cue: "water",
            expected: "helps_animal",
            distractor: "fix_code",
            max_hops: 4,
        },
        Task {
            cue: "fire",
            expected: "mechanism_fails",
            distractor: "animal",
            max_hops: 4,
        },
        Task {
            cue: "program",
            expected: "fix_code",
            distractor: "oxygen",
            max_hops: 4,
        },
    ]
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}
