use cdt_rqm_epr::entanglement::EntanglementConfig;
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
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::time::Instant;

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
    let sleep_attempts = env_usize("CURRICULUM_SLEEP_ATTEMPTS", 4);
    let sleep_replay_passes = env_usize("CURRICULUM_SLEEP_REPLAY_PASSES", 2).max(1);
    let use_gemma = env_flag("CURRICULUM_USE_GEMMA", true);
    let mut gemma = GemmaPeripheral::new(env::var("GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()));
    let mut substrate = clean_substrate();
    let mut cycle = 0_usize;
    let start = Instant::now();

    println!("Native thermodynamic 5-phase infinite curriculum");
    println!(
        "cycles={} sleep_attempts={} sleep_replay_passes={} use_gemma={} model={}",
        max_cycles
            .map(|value| value.to_string())
            .unwrap_or_else(|| "infinite".to_string()),
        sleep_attempts,
        sleep_replay_passes,
        use_gemma,
        gemma.model
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

        println!(
            "cycle={} phase1_acc={:.1}% phase2_acc={:.1}% phase3_acc={:.1}% phase4_forgetting={:.3} phase5_acc={:.1}% phase5_leak={:.1}% sleep_accepted={} relations={} epr_links={} mean_energy={:.4} free_energy={:.4} elapsed_ms={:.1}",
            cycle,
            phase1.accuracy() * 100.0,
            phase2.accuracy() * 100.0,
            phase3.accuracy() * 100.0,
            phase4,
            phase5.accuracy() * 100.0,
            phase5.leakage() * 100.0,
            sleep.accepted,
            substrate.relation_count(),
            epr.active_links,
            report.mean_energy,
            report.free_energy_proxy,
            start.elapsed().as_secs_f64() * 1_000.0
        );

        if max_cycles.is_some_and(|limit| cycle >= limit) {
            break;
        }
    }
}

impl GemmaPeripheral {
    fn new(model: String) -> Self {
        Self { model, cache: HashMap::new() }
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
                gemma.map_alias(language, alias).unwrap_or_else(|| concept.to_string())
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
                gemma.map_alias(language, alias).unwrap_or_else(|| concept.to_string())
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

fn train_phase4(substrate: &mut NativeThermoRqmEprSubstrate, sleep_attempts: usize, replay: usize) -> f32 {
    for item in phase4_base() {
        train_item(substrate, *item);
    }
    let before = evaluate_items(substrate, phase4_base());
    for item in phase4_stream() {
        train_item(substrate, *item);
    }
    let (slept, _) = native_sleep_consolidate(substrate.clone(), &lessons_from_items(phase4_all()), sleep_attempts, replay);
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
        let candidates = native_multi_hop_query_pruned(substrate, &cue, task.max_hops, Some(&target));
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
    attenuate(substrate, &concept_node(item.cue), &concept_node(item.distractor), 0.35);
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

fn attenuate(substrate: &mut NativeThermoRqmEprSubstrate, cue: &[usize], distractor: &[usize], amount: f32) {
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
        Rule { source: "dog", target: "mammal", kind: LessonKind::Semantic },
        Rule { source: "mammal", target: "animal", kind: LessonKind::Semantic },
        Rule { source: "animal", target: "needs_energy", kind: LessonKind::Causal },
        Rule { source: "water", target: "plant", kind: LessonKind::Causal },
        Rule { source: "plant", target: "oxygen", kind: LessonKind::Causal },
        Rule { source: "oxygen", target: "helps_animal", kind: LessonKind::Causal },
        Rule { source: "program", target: "test", kind: LessonKind::Skill },
        Rule { source: "test", target: "detect_bug", kind: LessonKind::Skill },
        Rule { source: "detect_bug", target: "fix_code", kind: LessonKind::Skill },
    ]
}

fn phase2_tasks() -> &'static [Task] {
    &[
        Task { cues: &["dog"], expected: "needs_energy", distractor: "fix_code", max_hops: 4 },
        Task { cues: &["water"], expected: "helps_animal", distractor: "fix_code", max_hops: 4 },
        Task { cues: &["program"], expected: "fix_code", distractor: "oxygen", max_hops: 4 },
    ]
}

fn phase3_rules() -> &'static [Rule] {
    &[
        Rule { source: "fire", target: "heat", kind: LessonKind::Causal },
        Rule { source: "heat", target: "expands_metal", kind: LessonKind::Causal },
        Rule { source: "expands_metal", target: "stress_joint", kind: LessonKind::Causal },
        Rule { source: "stress_joint", target: "mechanism_fails", kind: LessonKind::Causal },
        Rule { source: "food", target: "energy", kind: LessonKind::Causal },
        Rule { source: "energy", target: "movement", kind: LessonKind::Causal },
    ]
}

fn phase3_tasks() -> &'static [Task] {
    &[
        Task { cues: &["fire"], expected: "mechanism_fails", distractor: "helps_animal", max_hops: 5 },
        Task { cues: &["dog", "food"], expected: "movement", distractor: "mechanism_fails", max_hops: 4 },
        Task { cues: &["water", "animal"], expected: "helps_animal", distractor: "mechanism_fails", max_hops: 5 },
    ]
}

fn phase4_base() -> &'static [Item] {
    &[
        Item { kind: LessonKind::Semantic, cue: "dog", target: "mammal", distractor: "metal" },
        Item { kind: LessonKind::Causal, cue: "fire", target: "heat", distractor: "wet" },
        Item { kind: LessonKind::Skill, cue: "program", target: "logic", distractor: "animal" },
    ]
}

fn phase4_stream() -> &'static [Item] {
    &[
        Item { kind: LessonKind::Skill, cue: "hammer", target: "build", distractor: "swim" },
        Item { kind: LessonKind::Episodic, cue: "forest", target: "trees", distractor: "compiler" },
        Item { kind: LessonKind::Semantic, cue: "chien", target: "dog", distractor: "fire" },
    ]
}

fn phase4_all() -> &'static [Item] {
    &[
        Item { kind: LessonKind::Semantic, cue: "dog", target: "mammal", distractor: "metal" },
        Item { kind: LessonKind::Causal, cue: "fire", target: "heat", distractor: "wet" },
        Item { kind: LessonKind::Skill, cue: "program", target: "logic", distractor: "animal" },
        Item { kind: LessonKind::Skill, cue: "hammer", target: "build", distractor: "swim" },
        Item { kind: LessonKind::Episodic, cue: "forest", target: "trees", distractor: "compiler" },
        Item { kind: LessonKind::Semantic, cue: "chien", target: "dog", distractor: "fire" },
    ]
}

fn phase5_cases() -> &'static [Task] {
    &[
        Task { cues: &["dog", "fire"], expected: "needs_energy", distractor: "mechanism_fails", max_hops: 4 },
        Task { cues: &["water", "program"], expected: "helps_animal", distractor: "fix_code", max_hops: 4 },
        Task { cues: &["fire", "water"], expected: "mechanism_fails", distractor: "helps_animal", max_hops: 4 },
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
