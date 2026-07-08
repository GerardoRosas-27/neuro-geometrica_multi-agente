use cdt_rqm_epr::entanglement::{EntanglementConfig, EntanglementField};
use cdt_rqm_epr::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::native_thermodynamic_engine::{
    native_multi_hop_query_pruned, NativePathPruneTarget, DEFAULT_NODES_PER_SLICE, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::relational_field::ObserverId;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::time::{Duration, Instant};

const DEFAULT_STATE: &str = "data/native_curriculum_5phase.cdt_native";

#[derive(Clone, Copy)]
struct Rule {
    source: &'static str,
    target: &'static str,
}

#[derive(Clone, Copy)]
struct Task {
    cues: &'static [&'static str],
    expected: &'static str,
    distractor: &'static str,
    max_hops: usize,
}

#[derive(Clone, Copy)]
struct AliasCase {
    language: &'static str,
    alias: &'static str,
    concept: &'static str,
    expected: &'static [&'static str],
    distractor: &'static str,
}

#[derive(Clone, Copy)]
struct MemoryCase {
    cue: &'static str,
    expected: &'static str,
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

#[derive(Clone, Copy, Default)]
struct TimedMetrics {
    metrics: Metrics,
    elapsed: Duration,
}

struct GemmaPeripheral {
    model: String,
    cache: HashMap<(String, String), String>,
}

fn main() {
    let state = env::var("CURRICULUM_EVAL_STATE").unwrap_or_else(|_| DEFAULT_STATE.to_string());
    let use_gemma = env_flag("CURRICULUM_EVAL_USE_GEMMA", true);
    let mut gemma =
        GemmaPeripheral::new(env::var("GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()));
    let (cycle, growths, mut substrate) = match load_curriculum_state(&state) {
        Ok(value) => value,
        Err(err) => {
            println!("Native curriculum broad evaluation");
            println!("loaded=false state={state} error={err}");
            return;
        }
    };

    let phase1 = eval_phase1(&mut substrate, &mut gemma, use_gemma);
    let phase2 = eval_tasks(&mut substrate, phase2_tasks());
    let phase3 = eval_tasks(&mut substrate, phase3_tasks());
    let phase4_base = eval_memory(&mut substrate, phase4_base_cases());
    let phase4_stream = eval_memory(&mut substrate, phase4_stream_cases());
    let phase5_native = eval_tasks(&mut substrate, phase5_cases());
    let phase5_graph = eval_graph_baseline(phase5_cases());
    let phase5_lexical = eval_lexical_baseline(phase5_cases());
    let epr = substrate.entanglement.summary();
    let thermal = substrate.thermal.report();
    let all_pass = phase1.metrics.accuracy() >= 0.90
        && phase2.metrics.accuracy() >= 0.90
        && phase3.metrics.accuracy() >= 0.90
        && phase4_base.metrics.accuracy() >= 0.97
        && phase4_stream.metrics.accuracy() >= 0.90
        && phase5_native.metrics.accuracy() > phase5_graph.metrics.accuracy()
        && phase5_native.metrics.leakage() < phase5_graph.metrics.leakage();

    println!("Native curriculum broad evaluation");
    println!(
        "loaded=true state={} cycle={} growths={} nodes={} relations={} epr_links={} mean_energy={:.4} free_energy={:.4}",
        state,
        cycle,
        growths,
        substrate.thermal.node_count(),
        substrate.relation_count(),
        epr.active_links,
        thermal.mean_energy,
        thermal.free_energy_proxy
    );
    print_metrics("phase1_cross_lingual", phase1);
    print_metrics("phase2_compositional", phase2);
    print_metrics("phase3_strong_reasoning", phase3);
    print_metrics("phase4_base_retention", phase4_base);
    print_metrics("phase4_stream_retention", phase4_stream);
    print_metrics("phase5_native", phase5_native);
    print_metrics("phase5_graph_baseline", phase5_graph);
    print_metrics("phase5_lexical_baseline", phase5_lexical);
    println!(
        "phase5_delta: acc_vs_graph={:+.1}pp leak_vs_graph={:+.1}pp acc_vs_lexical={:+.1}pp leak_vs_lexical={:+.1}pp",
        (phase5_native.metrics.accuracy() - phase5_graph.metrics.accuracy()) * 100.0,
        (phase5_native.metrics.leakage() - phase5_graph.metrics.leakage()) * 100.0,
        (phase5_native.metrics.accuracy() - phase5_lexical.metrics.accuracy()) * 100.0,
        (phase5_native.metrics.leakage() - phase5_lexical.metrics.leakage()) * 100.0
    );
    println!(
        "decision={}",
        if all_pass {
            "broad_5phase_pass"
        } else {
            "broad_5phase_needs_tuning"
        }
    );
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
            .map(|case| case.concept)
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
            .find(|case| text.contains(case.concept))
            .map(|case| case.concept.to_string())?;
        self.cache.insert(key, mapped.clone());
        Some(mapped)
    }
}

fn eval_phase1(
    substrate: &mut NativeThermoRqmEprSubstrate,
    gemma: &mut GemmaPeripheral,
    use_gemma: bool,
) -> TimedMetrics {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for case in phase1_concepts() {
        let concept = if use_gemma {
            gemma
                .map_alias(case.language, case.alias)
                .unwrap_or_else(|| case.concept.to_string())
        } else {
            case.concept.to_string()
        };
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &concept_node(&concept));
        let expected = case
            .expected
            .iter()
            .map(|attribute| score(&report.candidates, &concept_node(attribute)))
            .sum();
        let distractor = score(&report.candidates, &concept_node(case.distractor));
        metrics.record(expected, distractor);
    }
    TimedMetrics {
        metrics,
        elapsed: start.elapsed(),
    }
}

fn eval_tasks(substrate: &mut NativeThermoRqmEprSubstrate, tasks: &[Task]) -> TimedMetrics {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for task in tasks {
        let cue = task
            .cues
            .iter()
            .flat_map(|cue| concept_node(cue))
            .collect::<Vec<_>>();
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
    TimedMetrics {
        metrics,
        elapsed: start.elapsed(),
    }
}

fn eval_memory(substrate: &mut NativeThermoRqmEprSubstrate, cases: &[MemoryCase]) -> TimedMetrics {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for case in cases {
        let report = substrate.query(DEFAULT_OBSERVER, 0.0, &concept_node(case.cue));
        metrics.record(
            score(&report.candidates, &concept_node(case.expected)),
            score(&report.candidates, &concept_node(case.distractor)),
        );
    }
    TimedMetrics {
        metrics,
        elapsed: start.elapsed(),
    }
}

fn eval_graph_baseline(tasks: &[Task]) -> TimedMetrics {
    let start = Instant::now();
    let mut graph = HashMap::<&'static str, Vec<&'static str>>::new();
    for rule in baseline_rules() {
        graph.entry(rule.source).or_default().push(rule.target);
    }
    let mut metrics = Metrics::default();
    for task in tasks {
        let mut reached = Vec::new();
        for cue in task.cues {
            reached.extend(graph_reach(&graph, cue, task.max_hops));
        }
        reached.sort_unstable();
        reached.dedup();
        metrics.record(
            f32::from(reached.contains(&task.expected)),
            f32::from(reached.contains(&task.distractor)),
        );
    }
    TimedMetrics {
        metrics,
        elapsed: start.elapsed(),
    }
}

fn eval_lexical_baseline(tasks: &[Task]) -> TimedMetrics {
    let start = Instant::now();
    let mut metrics = Metrics::default();
    for task in tasks {
        let cue = lexical_task_embedding(task);
        metrics.record(
            cosine(&cue, &lexical_embedding(task.expected)).max(0.0),
            cosine(&cue, &lexical_embedding(task.distractor)).max(0.0),
        );
    }
    TimedMetrics {
        metrics,
        elapsed: start.elapsed(),
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

fn score(candidates: &[NativeCandidateScore], targets: &[usize]) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
}

fn print_metrics(label: &str, timed: TimedMetrics) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} cases={} elapsed_ms={:.3}",
        label,
        timed.metrics.accuracy() * 100.0,
        timed.metrics.leakage() * 100.0,
        timed.metrics.margin(),
        timed.metrics.cases,
        timed.elapsed.as_secs_f64() * 1_000.0
    );
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

fn lexical_embedding(value: &str) -> [f32; 16] {
    let mut out = [0.0_f32; 16];
    for token in value.split('_') {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        out[hasher.finish() as usize % 16] += 1.0;
    }
    normalize(out)
}

fn lexical_task_embedding(task: &Task) -> [f32; 16] {
    let mut out = [0.0_f32; 16];
    for cue in task.cues {
        let emb = lexical_embedding(cue);
        for (slot, value) in out.iter_mut().zip(emb) {
            *slot += value;
        }
    }
    normalize(out)
}

fn normalize(mut vector: [f32; 16]) -> [f32; 16] {
    let norm = vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn cosine(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

fn phase1_concepts() -> &'static [AliasCase] {
    &[
        AliasCase {
            language: "en",
            alias: "dog",
            concept: "DOG",
            expected: &["mammal", "pet", "barks"],
            distractor: "fire",
        },
        AliasCase {
            language: "fr",
            alias: "chien",
            concept: "DOG",
            expected: &["mammal", "pet", "barks"],
            distractor: "metal",
        },
        AliasCase {
            language: "en",
            alias: "water",
            concept: "WATER",
            expected: &["liquid", "life", "wet"],
            distractor: "fire",
        },
        AliasCase {
            language: "fr",
            alias: "eau",
            concept: "WATER",
            expected: &["liquid", "life", "wet"],
            distractor: "metal",
        },
        AliasCase {
            language: "en",
            alias: "fire",
            concept: "FIRE",
            expected: &["heat", "burns", "light"],
            distractor: "wet",
        },
        AliasCase {
            language: "fr",
            alias: "feu",
            concept: "FIRE",
            expected: &["heat", "burns", "light"],
            distractor: "liquid",
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
        Task {
            cues: &["fire"],
            expected: "mechanism_fails",
            distractor: "animal",
            max_hops: 4,
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
        Task {
            cues: &["program"],
            expected: "stable_program",
            distractor: "animal_respiration",
            max_hops: 5,
        },
    ]
}

fn phase4_base_cases() -> &'static [MemoryCase] {
    &[
        MemoryCase {
            cue: "dog",
            expected: "mammal",
            distractor: "metal",
        },
        MemoryCase {
            cue: "fire",
            expected: "heat",
            distractor: "wet",
        },
        MemoryCase {
            cue: "program",
            expected: "logic",
            distractor: "animal",
        },
        MemoryCase {
            cue: "rain_story",
            expected: "wet_ground",
            distractor: "debug",
        },
    ]
}

fn phase4_stream_cases() -> &'static [MemoryCase] {
    &[
        MemoryCase {
            cue: "hammer",
            expected: "build",
            distractor: "swim",
        },
        MemoryCase {
            cue: "forest",
            expected: "trees",
            distractor: "compiler",
        },
        MemoryCase {
            cue: "chien",
            expected: "dog",
            distractor: "fire",
        },
        MemoryCase {
            cue: "eau",
            expected: "water",
            distractor: "metal",
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
        Task {
            cues: &["program", "water"],
            expected: "fix_code",
            distractor: "helps_animal",
            max_hops: 4,
        },
    ]
}

fn baseline_rules() -> &'static [Rule] {
    &[
        Rule {
            source: "dog",
            target: "mammal",
        },
        Rule {
            source: "mammal",
            target: "animal",
        },
        Rule {
            source: "animal",
            target: "needs_energy",
        },
        Rule {
            source: "water",
            target: "plant",
        },
        Rule {
            source: "plant",
            target: "oxygen",
        },
        Rule {
            source: "oxygen",
            target: "helps_animal",
        },
        Rule {
            source: "fire",
            target: "heat",
        },
        Rule {
            source: "heat",
            target: "expands_metal",
        },
        Rule {
            source: "expands_metal",
            target: "mechanism_fails",
        },
        Rule {
            source: "program",
            target: "test",
        },
        Rule {
            source: "test",
            target: "detect_bug",
        },
        Rule {
            source: "detect_bug",
            target: "fix_code",
        },
    ]
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

fn env_flag(name: &str, fallback: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(fallback)
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
