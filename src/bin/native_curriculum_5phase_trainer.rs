use cdt_rqm_epr::entanglement::{EntanglementConfig, EntanglementField};
use cdt_rqm_epr::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::native_thermodynamic_engine::{
    native_multi_hop_query_pruned, native_sleep_consolidate, Lesson, LessonKind,
    NativePathPruneTarget, DEFAULT_NODES_PER_SLICE, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::relational_field::ObserverId;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const DEFAULT_OUTPUT: &str = "data/native_curriculum_5phase.cdt_native";
const DEFAULT_PROGRESS: &str = "data/native_curriculum_5phase.progress";
const GROW_SLICES: usize = 2;

#[derive(Clone, Copy)]
struct Item {
    kind: LessonKind,
    cue: &'static str,
    target: &'static str,
    distractor: &'static str,
}

#[derive(Clone, Copy)]
struct Rule {
    source: &'static str,
    target: &'static str,
    kind: LessonKind,
}

#[derive(Clone, Copy)]
struct Task {
    cues: &'static [&'static str],
    expected: &'static str,
    distractor: &'static str,
    max_hops: usize,
}

#[derive(Default, Clone, Copy)]
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
}

struct GemmaPeripheral {
    model: String,
    cache: HashMap<(String, String), String>,
}

fn main() {
    let max_cycles = arg_value("--cycles").and_then(|value| value.parse::<usize>().ok());
    let output = env::var("CURRICULUM_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT.to_string());
    let progress = env::var("CURRICULUM_PROGRESS").unwrap_or_else(|_| DEFAULT_PROGRESS.to_string());
    let save_every = env_usize("CURRICULUM_SAVE_EVERY_CYCLES", 1).max(1);
    let sleep_attempts = env_usize("CURRICULUM_SLEEP_ATTEMPTS", 4);
    let sleep_replay_passes = env_usize("CURRICULUM_SLEEP_REPLAY_PASSES", 2).max(1);
    let use_gemma = env_flag("CURRICULUM_USE_GEMMA", true);
    let grow_relation_density = env_f32("CURRICULUM_GROW_RELATIONS_PER_NODE", 18.0);
    let grow_epr_density = env_f32("CURRICULUM_GROW_EPR_PER_NODE", 1.25);
    let resume = has_flag("--resume") || env_flag("CURRICULUM_RESUME", false);
    let mut gemma =
        GemmaPeripheral::new(env::var("GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()));
    let mut cycle = if resume {
        load_cycle_from_progress(&progress).unwrap_or(0)
    } else {
        0
    };
    let mut growths = if resume {
        load_usize_from_progress(&progress, "growths").unwrap_or(0)
    } else {
        0
    };
    let mut substrate = if resume && Path::new(&output).exists() {
        match load_curriculum_state(&output) {
            Ok((loaded_cycle, loaded_growths, substrate)) => {
                cycle = cycle.max(loaded_cycle);
                growths = growths.max(loaded_growths);
                println!(
                    "resume=true output={} cycle_start={} growths={} nodes={} relations={} epr_links={}",
                    output,
                    cycle,
                    growths,
                    substrate.thermal.node_count(),
                    substrate.relation_count(),
                    substrate.entanglement.summary().active_links
                );
                substrate
            }
            Err(err) => {
                eprintln!(
                    "resume=false output={} error={} starting_clean=true",
                    output, err
                );
                cycle = 0;
                growths = 0;
                clean_substrate()
            }
        }
    } else {
        clean_substrate()
    };
    let start = Instant::now();

    println!("Native thermodynamic 5-phase infinite curriculum");
    println!(
        "output={} progress={} cycles={} save_every={} sleep_attempts={} sleep_replay_passes={} use_gemma={} model={} resume={}",
        output,
        progress,
        max_cycles
            .map(|value| value.to_string())
            .unwrap_or_else(|| "infinite".to_string()),
        save_every,
        sleep_attempts,
        sleep_replay_passes,
        use_gemma,
        gemma.model,
        resume
    );

    loop {
        cycle += 1;
        let phase1 = train_phase1(&mut substrate, &mut gemma, use_gemma);
        let phase2 = train_phase2(&mut substrate);
        let phase3 = train_phase3(&mut substrate);
        let phase4 = train_phase4(&mut substrate, sleep_attempts, sleep_replay_passes);
        let phase5 = eval_phase5(&mut substrate);
        let lessons = curriculum_lessons();
        let (slept, sleep) =
            native_sleep_consolidate(substrate, &lessons, sleep_attempts, sleep_replay_passes);
        substrate = slept;
        let epr = substrate.entanglement.summary();
        let report = substrate.thermal.report();
        let growth_summary = maybe_grow(
            &mut substrate,
            grow_relation_density,
            grow_epr_density,
            &mut growths,
        );

        let line = format!(
            "cycle={} phase1_acc={:.1}% phase2_acc={:.1}% phase3_acc={:.1}% phase4_forgetting={:.3} phase5_acc={:.1}% phase5_leak={:.1}% sleep_accepted={} nodes={} relations={} epr_links={} mean_energy={:.4} free_energy={:.4} growths={} {} output={} elapsed_ms={:.1}",
            cycle,
            phase1.accuracy() * 100.0,
            phase2.accuracy() * 100.0,
            phase3.accuracy() * 100.0,
            phase4,
            phase5.accuracy() * 100.0,
            phase5.leakage() * 100.0,
            sleep.accepted,
            substrate.thermal.node_count(),
            substrate.relation_count(),
            epr.active_links,
            report.mean_energy,
            report.free_energy_proxy,
            growths,
            growth_summary,
            output,
            start.elapsed().as_secs_f64() * 1_000.0
        );
        println!("{line}");
        write_progress(&progress, &line);
        if cycle % save_every == 0 {
            save_curriculum_state(&substrate, &output, cycle, growths);
        }

        if max_cycles.is_some_and(|limit| cycle >= limit) {
            save_curriculum_state(&substrate, &output, cycle, growths);
            break;
        }
    }
}

impl GemmaPeripheral {
    fn new(model: String) -> Self {
        Self {
            model,
            cache: HashMap::new(),
        }
    }

    fn map_alias(&mut self, language: &str, surface: &str) -> Option<String> {
        let key = (language.to_string(), surface.to_string());
        if let Some(value) = self.cache.get(&key) {
            return Some(value.clone());
        }
        let allowed = phase1_concepts()
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>()
            .join(", ");
        let prompt = format!(
            "Map this word to exactly one concept id from this list: {allowed}.\nLanguage: {language}\nText: {surface}\nReturn only the concept id."
        );
        let output = Command::new("ollama")
            .arg("run")
            .arg(&self.model)
            .arg(prompt)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout).to_uppercase();
        let mapped = phase1_concepts()
            .iter()
            .find(|(id, _)| text.contains(id))
            .map(|(id, _)| id.to_string())?;
        self.cache.insert(key, mapped.clone());
        Some(mapped)
    }
}

fn train_phase1(
    substrate: &mut NativeThermoRqmEprSubstrate,
    gemma: &mut GemmaPeripheral,
    use_gemma: bool,
) -> Metrics {
    let mut metrics = Metrics::default();
    for (concept, aliases) in phase1_concepts() {
        let concept_pattern = concept_node(concept);
        for (language, alias) in *aliases {
            let mapped = if use_gemma {
                gemma
                    .map_alias(language, alias)
                    .unwrap_or_else(|| concept.to_string())
            } else {
                concept.to_string()
            };
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &pattern("alias", alias, 0),
                &concept_node(&mapped),
                1.0,
            );
        }
        for attr in phase1_attributes(concept) {
            substrate.train_observed_transition(
                DEFAULT_OBSERVER,
                0.0,
                &concept_pattern,
                &concept_node(attr),
                1.0,
            );
        }
    }
    for (concept, aliases) in phase1_concepts() {
        for (language, alias) in *aliases {
            if *language == "es" {
                continue;
            }
            let mapped = if use_gemma {
                gemma
                    .map_alias(language, alias)
                    .unwrap_or_else(|| concept.to_string())
            } else {
                concept.to_string()
            };
            let report = substrate.query(DEFAULT_OBSERVER, 0.0, &concept_node(&mapped));
            metrics.record(
                phase1_attributes(concept)
                    .iter()
                    .map(|attr| score(&report.candidates, &concept_node(attr)))
                    .sum(),
                score(&report.candidates, &concept_node("irrelevant_noise")),
            );
        }
    }
    metrics
}

fn train_phase2(substrate: &mut NativeThermoRqmEprSubstrate) -> Metrics {
    for rule in phase2_rules() {
        substrate.train_observed_transition(
            DEFAULT_OBSERVER,
            0.0,
            &concept_node(rule.source),
            &concept_node(rule.target),
            1.0,
        );
    }
    evaluate_tasks(substrate, phase2_tasks())
}

fn train_phase3(substrate: &mut NativeThermoRqmEprSubstrate) -> Metrics {
    for rule in phase3_rules() {
        substrate.train_observed_transition(
            DEFAULT_OBSERVER,
            0.0,
            &concept_node(rule.source),
            &concept_node(rule.target),
            1.0,
        );
        substrate.train_observed_transition(
            typed_observer(rule.kind),
            0.0,
            &concept_node(rule.source),
            &concept_node(rule.target),
            0.94,
        );
    }
    evaluate_tasks(substrate, phase3_tasks())
}

fn train_phase4(
    substrate: &mut NativeThermoRqmEprSubstrate,
    sleep_attempts: usize,
    replay: usize,
) -> f32 {
    for item in phase4_base() {
        train_item(substrate, *item);
    }
    let before = evaluate_items(substrate, phase4_base());
    for item in phase4_stream() {
        train_item(substrate, *item);
    }
    let (slept, _) = native_sleep_consolidate(
        substrate.clone(),
        &lessons_from_items(phase4_all()),
        sleep_attempts,
        replay,
    );
    *substrate = slept;
    let after = evaluate_items(substrate, phase4_base());
    (before.accuracy() - after.accuracy()).max(0.0)
}

fn eval_phase5(substrate: &mut NativeThermoRqmEprSubstrate) -> Metrics {
    evaluate_tasks(substrate, phase5_cases())
}

fn evaluate_tasks(substrate: &mut NativeThermoRqmEprSubstrate, tasks: &[Task]) -> Metrics {
    let mut metrics = Metrics::default();
    for task in tasks {
        let mut cue = concept_node(task.cues[0]);
        for extra in &task.cues[1..] {
            cue.extend(concept_node(extra));
        }
        cue.sort_unstable();
        cue.dedup();
        let target = NativePathPruneTarget {
            expected: concept_node(task.expected),
            distractor: concept_node(task.distractor),
        };
        let candidates =
            native_multi_hop_query_pruned(substrate, &cue, task.max_hops, Some(&target));
        metrics.record(
            score(&candidates, &target.expected),
            score(&candidates, &target.distractor),
        );
    }
    metrics
}

fn evaluate_items(substrate: &mut NativeThermoRqmEprSubstrate, items: &[Item]) -> Metrics {
    let mut metrics = Metrics::default();
    for item in items {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &concept_node(item.cue));
        metrics.record(
            score(&report.candidates, &concept_node(item.target)),
            score(&report.candidates, &concept_node(item.distractor)),
        );
    }
    metrics
}

fn train_item(substrate: &mut NativeThermoRqmEprSubstrate, item: Item) {
    substrate.train_observed_transition(
        DEFAULT_OBSERVER,
        0.0,
        &concept_node(item.cue),
        &concept_node(item.target),
        1.0,
    );
    substrate.train_observed_transition(
        typed_observer(item.kind),
        0.0,
        &concept_node(item.cue),
        &concept_node(item.target),
        0.94,
    );
    attenuate(
        substrate,
        &concept_node(item.cue),
        &concept_node(item.distractor),
        0.35,
    );
}

fn curriculum_lessons() -> Vec<Lesson> {
    phase2_rules()
        .iter()
        .chain(phase3_rules().iter())
        .map(|rule| Lesson {
            kind: rule.kind,
            local: concept_node(rule.source),
            action: concept_node("curriculum"),
            remote: concept_node(rule.target),
            distractor: concept_node("irrelevant_noise"),
        })
        .chain(lessons_from_items(phase4_all()))
        .collect()
}

fn lessons_from_items(items: &[Item]) -> Vec<Lesson> {
    items
        .iter()
        .map(|item| Lesson {
            kind: item.kind,
            local: concept_node(item.cue),
            action: concept_node(item.group()),
            remote: concept_node(item.target),
            distractor: concept_node(item.distractor),
        })
        .collect()
}

impl Item {
    fn group(self) -> &'static str {
        match self.kind {
            LessonKind::Semantic => "semantic",
            LessonKind::Episodic => "episodic",
            LessonKind::Causal => "causal",
            LessonKind::Skill => "skill",
        }
    }
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
            seed: 0xC0A4_5151,
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

fn attenuate(
    substrate: &mut NativeThermoRqmEprSubstrate,
    cue: &[usize],
    distractor: &[usize],
    amount: f32,
) {
    for &source in cue {
        for &target in distractor {
            substrate.attenuate_relation(DEFAULT_OBSERVER, source, target, amount);
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

fn concept_node(value: &str) -> Vec<usize> {
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

fn phase1_concepts() -> &'static [(&'static str, &'static [(&'static str, &'static str)])] {
    &[
        ("DOG", &[("es", "perro"), ("en", "dog"), ("fr", "chien")]),
        ("WATER", &[("es", "agua"), ("en", "water"), ("fr", "eau")]),
        ("FIRE", &[("es", "fuego"), ("en", "fire"), ("fr", "feu")]),
    ]
}

fn phase1_attributes(concept: &str) -> &'static [&'static str] {
    match concept {
        "DOG" => &["mammal", "pet", "barks"],
        "WATER" => &["liquid", "life", "wet"],
        "FIRE" => &["heat", "burns", "light"],
        _ => &["unknown"],
    }
}

fn phase2_rules() -> &'static [Rule] {
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

fn phase2_tasks() -> &'static [Task] {
    &[
        Task {
            cues: &["dog"],
            expected: "needs_energy",
            distractor: "fix_code",
            max_hops: 4,
        },
        Task {
            cues: &["water"],
            expected: "helps_animal",
            distractor: "fix_code",
            max_hops: 4,
        },
        Task {
            cues: &["program"],
            expected: "fix_code",
            distractor: "oxygen",
            max_hops: 4,
        },
    ]
}

fn phase3_rules() -> &'static [Rule] {
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
            source: "food",
            target: "energy",
            kind: LessonKind::Causal,
        },
        Rule {
            source: "energy",
            target: "movement",
            kind: LessonKind::Causal,
        },
    ]
}

fn phase3_tasks() -> &'static [Task] {
    &[
        Task {
            cues: &["fire"],
            expected: "mechanism_fails",
            distractor: "helps_animal",
            max_hops: 5,
        },
        Task {
            cues: &["dog", "food"],
            expected: "movement",
            distractor: "mechanism_fails",
            max_hops: 4,
        },
        Task {
            cues: &["water", "animal"],
            expected: "helps_animal",
            distractor: "mechanism_fails",
            max_hops: 5,
        },
    ]
}

fn phase4_base() -> &'static [Item] {
    &[
        Item {
            kind: LessonKind::Semantic,
            cue: "dog",
            target: "mammal",
            distractor: "metal",
        },
        Item {
            kind: LessonKind::Causal,
            cue: "fire",
            target: "heat",
            distractor: "wet",
        },
        Item {
            kind: LessonKind::Skill,
            cue: "program",
            target: "logic",
            distractor: "animal",
        },
    ]
}

fn phase4_stream() -> &'static [Item] {
    &[
        Item {
            kind: LessonKind::Skill,
            cue: "hammer",
            target: "build",
            distractor: "swim",
        },
        Item {
            kind: LessonKind::Episodic,
            cue: "forest",
            target: "trees",
            distractor: "compiler",
        },
        Item {
            kind: LessonKind::Semantic,
            cue: "chien",
            target: "dog",
            distractor: "fire",
        },
        Item {
            kind: LessonKind::Semantic,
            cue: "eau",
            target: "water",
            distractor: "metal",
        },
        Item {
            kind: LessonKind::Semantic,
            cue: "feu",
            target: "fire",
            distractor: "plant",
        },
    ]
}

fn phase4_all() -> &'static [Item] {
    &[
        Item {
            kind: LessonKind::Semantic,
            cue: "dog",
            target: "mammal",
            distractor: "metal",
        },
        Item {
            kind: LessonKind::Causal,
            cue: "fire",
            target: "heat",
            distractor: "wet",
        },
        Item {
            kind: LessonKind::Skill,
            cue: "program",
            target: "logic",
            distractor: "animal",
        },
        Item {
            kind: LessonKind::Skill,
            cue: "hammer",
            target: "build",
            distractor: "swim",
        },
        Item {
            kind: LessonKind::Episodic,
            cue: "forest",
            target: "trees",
            distractor: "compiler",
        },
        Item {
            kind: LessonKind::Semantic,
            cue: "chien",
            target: "dog",
            distractor: "fire",
        },
        Item {
            kind: LessonKind::Semantic,
            cue: "eau",
            target: "water",
            distractor: "metal",
        },
        Item {
            kind: LessonKind::Semantic,
            cue: "feu",
            target: "fire",
            distractor: "plant",
        },
    ]
}

fn phase5_cases() -> &'static [Task] {
    &[
        Task {
            cues: &["dog", "fire"],
            expected: "needs_energy",
            distractor: "mechanism_fails",
            max_hops: 4,
        },
        Task {
            cues: &["water", "program"],
            expected: "helps_animal",
            distractor: "fix_code",
            max_hops: 4,
        },
        Task {
            cues: &["fire", "water"],
            expected: "mechanism_fails",
            distractor: "helps_animal",
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

fn env_flag(name: &str, fallback: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(fallback)
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}

fn maybe_grow(
    substrate: &mut NativeThermoRqmEprSubstrate,
    relation_density_limit: f32,
    epr_density_limit: f32,
    growths: &mut usize,
) -> String {
    let nodes = substrate.thermal.node_count().max(1) as f32;
    let relation_density = substrate.relation_count() as f32 / nodes;
    let epr_density = substrate.entanglement.summary().active_links as f32 / nodes;
    if relation_density < relation_density_limit && epr_density < epr_density_limit {
        return format!(
            "growth=checked relation_density={:.3} epr_density={:.3}",
            relation_density, epr_density
        );
    }

    let old_nodes = substrate.thermal.node_count();
    let mut config = substrate.thermal.config;
    config.slices += GROW_SLICES;
    config.seed ^= (*growths as u64 + 1).wrapping_mul(0xA24B_AED4);
    let entries = substrate.relation_entries().collect::<Vec<_>>();
    let entanglement = substrate.entanglement.clone();
    let old = substrate.clone();
    let mut grown = NativeThermoRqmEprSubstrate::new(config, substrate.config, entanglement.config);
    copy_prefix(&old.thermal.thermal_state, &mut grown.thermal.thermal_state);
    copy_prefix(&old.thermal.amplitude, &mut grown.thermal.amplitude);
    copy_prefix(&old.thermal.phase, &mut grown.thermal.phase);
    copy_prefix(&old.thermal.temperature, &mut grown.thermal.temperature);
    copy_prefix(&old.thermal.energy, &mut grown.thermal.energy);
    grown.entanglement = entanglement;
    for (observer, source, target, amplitude, phase, coherence, uncertainty, last_tick) in entries {
        grown.import_relation_state(
            observer,
            source,
            target,
            amplitude,
            phase,
            coherence,
            uncertainty,
            last_tick,
        );
    }
    *substrate = grown;
    *growths += 1;
    format!(
        "growth=added old_nodes={} new_nodes={} slices={} relation_density={:.3} epr_density={:.3}",
        old_nodes,
        substrate.thermal.node_count(),
        substrate.thermal.config.slices,
        relation_density,
        epr_density
    )
}

fn save_curriculum_state(
    substrate: &NativeThermoRqmEprSubstrate,
    output: &str,
    cycle: usize,
    growths: usize,
) {
    if let Some(parent) = Path::new(output).parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::write(
        output,
        serialize_curriculum_state(substrate, cycle, growths),
    ) {
        Ok(()) => println!("saved=true cycle={} output={}", cycle, output),
        Err(err) => eprintln!(
            "saved=false cycle={} output={} error={}",
            cycle, output, err
        ),
    }
}

fn serialize_curriculum_state(
    substrate: &NativeThermoRqmEprSubstrate,
    cycle: usize,
    growths: usize,
) -> String {
    let mut out = String::new();
    out.push_str("NATIVE_THERMO_RQM_EPR_CURRICULUM_STATE_V1\n");
    out.push_str(&format!("curriculum_stats {} {}\n", cycle, growths));
    let cdt = substrate.thermal.config;
    out.push_str(&format!(
        "thermal_config {} {} {} {} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {}\n",
        cdt.slices,
        cdt.nodes_per_slice,
        cdt.spatial_degree,
        cdt.temporal_degree,
        cdt.temperature,
        cdt.dt,
        cdt.diffusion,
        cdt.confinement,
        cdt.pilot_gain,
        cdt.phase_coupling,
        cdt.amplitude_decay,
        cdt.state_clamp,
        cdt.seed
    ));
    let rqm = substrate.config;
    out.push_str(&format!(
        "rqm_config {:.7} {:.7} {:.7} {:.7} {:.7} {} {} {:.7} {:.7} {} {} {} {} {} {}\n",
        rqm.amplitude_learning_rate,
        rqm.coherence_learning_rate,
        rqm.uncertainty_learning_rate,
        rqm.phase_learning_rate,
        rqm.amplitude_decay,
        rqm.thermal_steps_per_train,
        rqm.thermal_steps_per_query,
        rqm.thermal_score_gain,
        rqm.thermal_activation_margin,
        usize::from(rqm.collect_query_diagnostics),
        rqm.max_candidates,
        rqm.max_pilot_window_nodes,
        rqm.sampling_block_size,
        rqm.sampling_schedule_rounds,
        rqm.max_sampling_blocks
    ));
    out.push_str(&format!("nodes {}\n", substrate.thermal.node_count()));
    for idx in 0..substrate.thermal.node_count() {
        out.push_str(&format!(
            "n {} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7}\n",
            idx,
            substrate.thermal.thermal_state[idx],
            substrate.thermal.amplitude[idx],
            substrate.thermal.phase[idx],
            substrate.thermal.temperature[idx],
            substrate.thermal.energy[idx],
            substrate.thermal.activation[idx]
        ));
    }
    let entries = substrate.relation_entries().collect::<Vec<_>>();
    out.push_str(&format!("relations {}\n", entries.len()));
    for (observer, source, target, amplitude, phase, coherence, uncertainty, last_tick) in entries {
        out.push_str(&format!(
            "r {} {} {} {:.7} {:.7} {:.7} {:.7} {}\n",
            observer.0, source, target, amplitude, phase, coherence, uncertainty, last_tick
        ));
    }
    out.push_str("entanglement_begin\n");
    out.push_str(&substrate.entanglement.serialize_persistent_state());
    out.push_str("entanglement_end\n");
    out.push_str("end\n");
    out
}

fn load_curriculum_state(
    path: &str,
) -> Result<(usize, usize, NativeThermoRqmEprSubstrate), String> {
    let contents = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let mut lines = contents.lines();
    if lines.next() != Some("NATIVE_THERMO_RQM_EPR_CURRICULUM_STATE_V1") {
        return Err("version curriculum nativa invalida".to_string());
    }
    let stats_line = lines.next().ok_or("falta curriculum_stats")?;
    let stats = stats_line.split_whitespace().collect::<Vec<_>>();
    if stats.len() != 3 || stats[0] != "curriculum_stats" {
        return Err(format!("curriculum_stats invalido: {stats_line}"));
    }
    let cycle = parse_usize(stats[1], "cycle")?;
    let growths = parse_usize(stats[2], "growths")?;
    let thermal_config = parse_thermal_config(lines.next().ok_or("falta thermal_config")?)?;
    let rqm_config = parse_rqm_config(lines.next().ok_or("falta rqm_config")?)?;
    let mut substrate =
        NativeThermoRqmEprSubstrate::new(thermal_config, rqm_config, EntanglementConfig::default());
    let node_count = parse_count_header(lines.next().ok_or("faltan nodes")?, "nodes")?;
    for _ in 0..node_count {
        let line = lines.next().ok_or("faltan nodos")?;
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 8 || parts[0] != "n" {
            return Err(format!("nodo curriculum invalido: {line}"));
        }
        let idx = parse_usize(parts[1], "idx")?;
        if idx < substrate.thermal.node_count() {
            substrate.thermal.thermal_state[idx] = parse_f32(parts[2], "state")?;
            substrate.thermal.amplitude[idx] = parse_f32(parts[3], "amplitude")?;
            substrate.thermal.phase[idx] = parse_f32(parts[4], "phase")?;
            substrate.thermal.temperature[idx] = parse_f32(parts[5], "temperature")?;
            substrate.thermal.energy[idx] = parse_f32(parts[6], "energy")?;
            substrate.thermal.activation[idx] = parse_f32(parts[7], "activation")?;
        }
    }
    let relation_count = parse_count_header(lines.next().ok_or("faltan relations")?, "relations")?;
    for _ in 0..relation_count {
        let line = lines.next().ok_or("faltan relaciones")?;
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 9 || parts[0] != "r" {
            return Err(format!("relacion curriculum invalida: {line}"));
        }
        substrate.import_relation_state(
            ObserverId(parse_usize(parts[1], "observer")?),
            parse_usize(parts[2], "source")?,
            parse_usize(parts[3], "target")?,
            parse_f32(parts[4], "amplitude")?,
            parse_f32(parts[5], "phase")?,
            parse_f32(parts[6], "coherence")?,
            parse_f32(parts[7], "uncertainty")?,
            parse_u64(parts[8], "last_tick")?,
        );
    }
    let rest = lines.collect::<Vec<_>>().join("\n");
    if let Some(entanglement) = section(&rest, "entanglement_begin", "entanglement_end") {
        let mut field = EntanglementField::new(EntanglementConfig::default());
        field.apply_persistent_state(&entanglement)?;
        substrate.entanglement = field;
    }
    Ok((cycle, growths, substrate))
}

fn write_progress(path: &str, line: &str) {
    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, line);
}

fn load_cycle_from_progress(path: &str) -> Option<usize> {
    load_usize_from_progress(path, "cycle")
}

fn load_usize_from_progress(path: &str, key: &str) -> Option<usize> {
    let contents = fs::read_to_string(path).ok()?;
    contents
        .split_whitespace()
        .find_map(|part| part.strip_prefix(&format!("{key}="))?.parse().ok())
}

fn copy_prefix(from: &[f32], to: &mut [f32]) {
    for (dst, src) in to.iter_mut().zip(from.iter().copied()) {
        *dst = src;
    }
}

fn parse_thermal_config(line: &str) -> Result<NativeThermoCdtConfig, String> {
    let p = line.split_whitespace().collect::<Vec<_>>();
    if p.len() != 14 || p[0] != "thermal_config" {
        return Err(format!("thermal_config invalida: {line}"));
    }
    Ok(NativeThermoCdtConfig {
        slices: parse_usize(p[1], "slices")?,
        nodes_per_slice: parse_usize(p[2], "nodes_per_slice")?,
        spatial_degree: parse_usize(p[3], "spatial_degree")?,
        temporal_degree: parse_usize(p[4], "temporal_degree")?,
        temperature: parse_f32(p[5], "temperature")?,
        dt: parse_f32(p[6], "dt")?,
        diffusion: parse_f32(p[7], "diffusion")?,
        confinement: parse_f32(p[8], "confinement")?,
        pilot_gain: parse_f32(p[9], "pilot_gain")?,
        phase_coupling: parse_f32(p[10], "phase_coupling")?,
        amplitude_decay: parse_f32(p[11], "amplitude_decay")?,
        state_clamp: parse_f32(p[12], "state_clamp")?,
        seed: parse_u64(p[13], "seed")?,
    })
}

fn parse_rqm_config(line: &str) -> Result<NativeThermoRqmConfig, String> {
    let p = line.split_whitespace().collect::<Vec<_>>();
    if p.len() != 16 || p[0] != "rqm_config" {
        return Err(format!("rqm_config invalida: {line}"));
    }
    Ok(NativeThermoRqmConfig {
        amplitude_learning_rate: parse_f32(p[1], "amplitude_lr")?,
        coherence_learning_rate: parse_f32(p[2], "coherence_lr")?,
        uncertainty_learning_rate: parse_f32(p[3], "uncertainty_lr")?,
        phase_learning_rate: parse_f32(p[4], "phase_lr")?,
        amplitude_decay: parse_f32(p[5], "amplitude_decay")?,
        thermal_steps_per_train: parse_usize(p[6], "thermal_steps_per_train")?,
        thermal_steps_per_query: parse_usize(p[7], "thermal_steps_per_query")?,
        thermal_score_gain: parse_f32(p[8], "thermal_score_gain")?,
        thermal_activation_margin: parse_f32(p[9], "thermal_activation_margin")?,
        collect_query_diagnostics: parse_usize(p[10], "diagnostics")? != 0,
        max_candidates: parse_usize(p[11], "max_candidates")?,
        max_pilot_window_nodes: parse_usize(p[12], "max_pilot_window_nodes")?,
        sampling_block_size: parse_usize(p[13], "sampling_block_size")?,
        sampling_schedule_rounds: parse_usize(p[14], "sampling_schedule_rounds")?,
        max_sampling_blocks: parse_usize(p[15], "max_sampling_blocks")?,
    })
}

fn parse_count_header(line: &str, label: &str) -> Result<usize, String> {
    let p = line.split_whitespace().collect::<Vec<_>>();
    if p.len() != 2 || p[0] != label {
        return Err(format!("cabecera {label} invalida: {line}"));
    }
    parse_usize(p[1], label)
}

fn section(contents: &str, begin: &str, end: &str) -> Option<String> {
    let start = contents.find(begin)? + begin.len();
    let tail = &contents[start..];
    let stop = tail.find(end)?;
    Some(tail[..stop].trim_matches('\n').to_string())
}

fn env_f32(name: &str, fallback: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn has_flag(name: &str) -> bool {
    env::args().any(|arg| arg == name)
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("{label} invalido: {err}"))
}

fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("{label} invalido: {err}"))
}

fn parse_f32(value: &str, label: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|err| format!("{label} invalido: {err}"))
}
