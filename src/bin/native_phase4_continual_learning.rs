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
struct MemoryItem {
    group: &'static str,
    kind: LessonKind,
    cue: &'static str,
    target: &'static str,
    distractor: &'static str,
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

fn main() {
    let base_epochs = env_usize("PHASE4_BASE_EPOCHS", 4).max(1);
    let stream_epochs = env_usize("PHASE4_STREAM_EPOCHS", 8).max(1);
    let sleep_attempts = env_usize("PHASE4_SLEEP_ATTEMPTS", 6);
    let sleep_replay_passes = env_usize("PHASE4_SLEEP_REPLAY_PASSES", 2).max(1);
    let mut substrate = clean_substrate();
    let start = Instant::now();

    println!("Native phase 4 continual learning");
    println!(
        "base_items={} stream_items={} base_epochs={} stream_epochs={} sleep_attempts={} sleep_replay_passes={}",
        base_items().len(),
        stream_items().len(),
        base_epochs,
        stream_epochs,
        sleep_attempts,
        sleep_replay_passes
    );

    for epoch in 0..base_epochs {
        train_items(&mut substrate, base_items(), epoch);
    }
    let (slept, base_sleep) = native_sleep_consolidate(
        substrate,
        &lessons(base_items()),
        sleep_attempts,
        sleep_replay_passes,
    );
    substrate = slept;
    let base_before = evaluate_group(&mut substrate, base_items());
    println!(
        "base_consolidated sleep_accepted={} acc={:.1}% leak={:.1}% margin={:.3} relations={} epr_links={}",
        base_sleep.accepted,
        base_before.accuracy() * 100.0,
        base_before.leakage() * 100.0,
        base_before.margin(),
        substrate.relation_count(),
        substrate.entanglement.summary().active_links
    );

    for epoch in 0..stream_epochs {
        let group = match epoch % 3 {
            0 => "tools",
            1 => "places",
            _ => "language",
        };
        let batch = stream_items()
            .iter()
            .copied()
            .filter(|item| item.group == group)
            .collect::<Vec<_>>();
        train_items(&mut substrate, &batch, epoch);
        let rehearsal = mixed_lessons(base_items(), stream_items());
        let (slept, sleep) =
            native_sleep_consolidate(substrate, &rehearsal, sleep_attempts, sleep_replay_passes);
        substrate = slept;
        let old_metrics = evaluate_group(&mut substrate, base_items());
        let new_metrics = evaluate_group(&mut substrate, stream_items());
        println!(
            "stream_epoch={} group={} sleep_accepted={} old_acc={:.1}% old_leak={:.1}% new_acc={:.1}% new_leak={:.1}% relations={} epr_links={} elapsed_ms={:.1}",
            epoch + 1,
            group,
            sleep.accepted,
            old_metrics.accuracy() * 100.0,
            old_metrics.leakage() * 100.0,
            new_metrics.accuracy() * 100.0,
            new_metrics.leakage() * 100.0,
            substrate.relation_count(),
            substrate.entanglement.summary().active_links,
            start.elapsed().as_secs_f64() * 1_000.0
        );
    }

    let base_after = evaluate_group(&mut substrate, base_items());
    let stream_after = evaluate_group(&mut substrate, stream_items());
    let all_after = evaluate_group(&mut substrate, &all_items());
    let forgetting_index = (base_before.accuracy() - base_after.accuracy()).max(0.0);
    let leakage_drift = base_after.leakage() - base_before.leakage();
    let margin_drift = base_after.margin() - base_before.margin();
    let pass = forgetting_index <= 0.03
        && base_after.accuracy() >= 0.97
        && stream_after.accuracy() >= 0.90
        && all_after.leakage() <= 0.08;

    println!(
        "final: base_before_acc={:.1}% base_after_acc={:.1}% stream_acc={:.1}% all_acc={:.1}% forgetting_index={:.3} leakage_drift={:+.3} margin_drift={:+.3} all_leakage={:.1}% relations={} epr_links={} decision={}",
        base_before.accuracy() * 100.0,
        base_after.accuracy() * 100.0,
        stream_after.accuracy() * 100.0,
        all_after.accuracy() * 100.0,
        forgetting_index,
        leakage_drift,
        margin_drift,
        all_after.leakage() * 100.0,
        substrate.relation_count(),
        substrate.entanglement.summary().active_links,
        if pass {
            "phase4_pass"
        } else {
            "phase4_needs_tuning"
        }
    );
}

fn train_items(substrate: &mut NativeThermoRqmEprSubstrate, items: &[MemoryItem], epoch: usize) {
    for item in items {
        let cue = concept(item.cue);
        let target = concept(item.target);
        let distractor = concept(item.distractor);
        substrate.train_observed_transition(DEFAULT_OBSERVER, phase(epoch), &cue, &target, 1.0);
        substrate.train_observed_transition(
            typed_observer(item.kind),
            phase(epoch),
            &cue,
            &target,
            0.94,
        );
        attenuate(substrate, DEFAULT_OBSERVER, &cue, &distractor, 0.35);
        attenuate(
            substrate,
            typed_observer(item.kind),
            &cue,
            &distractor,
            0.30,
        );
    }
}

fn evaluate_group(substrate: &mut NativeThermoRqmEprSubstrate, items: &[MemoryItem]) -> Metrics {
    let mut metrics = Metrics::default();
    for item in items {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &concept(item.cue));
        metrics.record(
            score(&report.candidates, &concept(item.target)),
            score(&report.candidates, &concept(item.distractor)),
        );
        let typed = substrate.query(typed_observer(item.kind), 0.0, &concept(item.cue));
        metrics.record(
            score(&typed.candidates, &concept(item.target)),
            score(&typed.candidates, &concept(item.distractor)),
        );
    }
    metrics
}

fn lessons(items: &[MemoryItem]) -> Vec<Lesson> {
    items
        .iter()
        .map(|item| Lesson {
            kind: item.kind,
            local: concept(item.cue),
            action: concept(item.group),
            remote: concept(item.target),
            distractor: concept(item.distractor),
        })
        .collect()
}

fn mixed_lessons(base: &[MemoryItem], stream: &[MemoryItem]) -> Vec<Lesson> {
    base.iter()
        .chain(stream.iter())
        .copied()
        .map(|item| Lesson {
            kind: item.kind,
            local: concept(item.cue),
            action: concept(item.group),
            remote: concept(item.target),
            distractor: concept(item.distractor),
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
            seed: 0xC0A4_4004,
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

fn all_items() -> Vec<MemoryItem> {
    base_items()
        .iter()
        .chain(stream_items().iter())
        .copied()
        .collect()
}

fn base_items() -> &'static [MemoryItem] {
    &[
        MemoryItem {
            group: "base",
            kind: LessonKind::Semantic,
            cue: "dog",
            target: "mammal",
            distractor: "metal",
        },
        MemoryItem {
            group: "base",
            kind: LessonKind::Semantic,
            cue: "water",
            target: "liquid",
            distractor: "fire",
        },
        MemoryItem {
            group: "base",
            kind: LessonKind::Causal,
            cue: "fire",
            target: "heat",
            distractor: "wet",
        },
        MemoryItem {
            group: "base",
            kind: LessonKind::Causal,
            cue: "plant",
            target: "oxygen",
            distractor: "machine",
        },
        MemoryItem {
            group: "base",
            kind: LessonKind::Skill,
            cue: "program",
            target: "logic",
            distractor: "animal",
        },
        MemoryItem {
            group: "base",
            kind: LessonKind::Episodic,
            cue: "rain_story",
            target: "wet_ground",
            distractor: "debug",
        },
    ]
}

fn stream_items() -> &'static [MemoryItem] {
    &[
        MemoryItem {
            group: "tools",
            kind: LessonKind::Skill,
            cue: "hammer",
            target: "build",
            distractor: "swim",
        },
        MemoryItem {
            group: "tools",
            kind: LessonKind::Skill,
            cue: "saw",
            target: "cut",
            distractor: "breathe",
        },
        MemoryItem {
            group: "tools",
            kind: LessonKind::Skill,
            cue: "debugger",
            target: "inspect_code",
            distractor: "photosynthesis",
        },
        MemoryItem {
            group: "places",
            kind: LessonKind::Episodic,
            cue: "forest",
            target: "trees",
            distractor: "compiler",
        },
        MemoryItem {
            group: "places",
            kind: LessonKind::Episodic,
            cue: "river",
            target: "flow",
            distractor: "keyboard",
        },
        MemoryItem {
            group: "places",
            kind: LessonKind::Episodic,
            cue: "kitchen",
            target: "cook",
            distractor: "oxygen",
        },
        MemoryItem {
            group: "language",
            kind: LessonKind::Semantic,
            cue: "chien",
            target: "dog",
            distractor: "fire",
        },
        MemoryItem {
            group: "language",
            kind: LessonKind::Semantic,
            cue: "eau",
            target: "water",
            distractor: "metal",
        },
        MemoryItem {
            group: "language",
            kind: LessonKind::Semantic,
            cue: "feu",
            target: "fire",
            distractor: "plant",
        },
    ]
}

fn attenuate(
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

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}
