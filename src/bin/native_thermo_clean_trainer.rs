use cdt_rqm_epr::entanglement::{EntanglementConfig, EntanglementField};
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use cdt_rqm_epr::native_thermodynamic_engine::{
    native_sleep_consolidate, Lesson, LessonKind, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::relational_field::ObserverId;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

const DEFAULT_OUTPUT: &str = "data/native_thermo_clean.cdt_native";
const DEFAULT_PROGRESS: &str = "data/native_thermo_clean.progress";
const DEFAULT_NODES_PER_SLICE: usize = 160;
const DEFAULT_SLICES: usize = 4;
const GROW_SLICES: usize = 2;

#[derive(Clone, Copy)]
struct Sample {
    kind: LessonKind,
    input: &'static str,
    action: &'static str,
    target: &'static str,
    distractor: &'static str,
}

#[derive(Default)]
struct Stats {
    batches: usize,
    samples: usize,
    semantic: usize,
    causal: usize,
    skill: usize,
    episodic: usize,
    sleep_runs: usize,
    growths: usize,
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
    let output =
        env::var("NATIVE_THERMO_TRAIN_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT.to_string());
    let progress =
        env::var("NATIVE_THERMO_TRAIN_PROGRESS").unwrap_or_else(|_| DEFAULT_PROGRESS.to_string());
    let batch_size = env_usize("NATIVE_THERMO_BATCH_SIZE", 16).max(1);
    let save_every = env_usize("NATIVE_THERMO_SAVE_EVERY_BATCHES", 1).max(1);
    let sleep_every = env_usize("NATIVE_THERMO_SLEEP_EVERY_BATCHES", 4).max(1);
    let sleep_attempts = env_usize("NATIVE_THERMO_SLEEP_ATTEMPTS", 6);
    let sleep_replay_passes = env_usize("NATIVE_THERMO_SLEEP_REPLAY_PASSES", 2).max(1);
    let relation_density_limit = env_f32("NATIVE_THERMO_GROW_RELATIONS_PER_NODE", 18.0);
    let epr_density_limit = env_f32("NATIVE_THERMO_GROW_EPR_PER_NODE", 1.25);
    let max_batches = arg_value("--batches").and_then(|value| value.parse::<usize>().ok());
    let resume = has_flag("--resume") || env_flag("NATIVE_THERMO_RESUME");

    let mut stats = if resume {
        load_stats_from_progress(&progress).unwrap_or_default()
    } else {
        Stats::default()
    };
    let mut substrate = if resume && Path::new(&output).exists() {
        match load_native_state(&output) {
            Ok(substrate) => {
                println!(
                    "resume=true output={} batch_start={} slices={} nodes={} relations={} epr_links={}",
                    output,
                    stats.batches,
                    substrate.thermal.config.slices,
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
                stats = Stats::default();
                clean_substrate(DEFAULT_SLICES, DEFAULT_NODES_PER_SLICE)
            }
        }
    } else {
        clean_substrate(DEFAULT_SLICES, DEFAULT_NODES_PER_SLICE)
    };

    println!("Native thermodynamic clean infinite trainer");
    println!(
        "output={} progress={} batch_size={} sleep_every={} save_every={} max_batches={} resume={}",
        output,
        progress,
        batch_size,
        sleep_every,
        save_every,
        max_batches
            .map(|value| value.to_string())
            .unwrap_or_else(|| "infinite".to_string()),
        resume
    );

    loop {
        stats.batches += 1;
        for i in 0..batch_size {
            let sample = dataset()[(stats.samples + i) % dataset().len()];
            train_sample(&mut substrate, sample, &mut stats);
        }

        let mut sleep_summary = "sleep=skipped".to_string();
        if stats.batches % sleep_every == 0 {
            stats.sleep_runs += 1;
            let lessons = validation_lessons();
            let before = evaluate(&substrate, &lessons);
            let (slept, report) =
                native_sleep_consolidate(substrate, &lessons, sleep_attempts, sleep_replay_passes);
            substrate = slept;
            let after = evaluate(&substrate, &lessons);
            sleep_summary = format!(
                "sleep=contrastive accepted={} acc={:.1}%->{:.1}% leak={:.1}%->{:.1}% margin={:.3}->{:.3}",
                report.accepted,
                before.accuracy() * 100.0,
                after.accuracy() * 100.0,
                before.leakage() * 100.0,
                after.leakage() * 100.0,
                before.margin(),
                after.margin()
            );
        }

        let growth_summary = maybe_grow(
            &mut substrate,
            relation_density_limit,
            epr_density_limit,
            &mut stats,
        );
        let report = substrate.thermal.report();
        let epr = substrate.entanglement.summary();
        let validation = evaluate(&substrate, &validation_lessons());
        let line = format!(
            "batch={} samples={} semantic={} causal={} skill={} episodic={} slices={} nodes={} relations={} epr_links={} acc={:.1}% leak={:.1}% margin={:.3} mean_energy={:.4} free_energy={:.4} active_nodes={} sleep_runs={} growths={} {} {} output={}\n",
            stats.batches,
            stats.samples,
            stats.semantic,
            stats.causal,
            stats.skill,
            stats.episodic,
            substrate.thermal.config.slices,
            substrate.thermal.node_count(),
            substrate.relation_count(),
            epr.active_links,
            validation.accuracy() * 100.0,
            validation.leakage() * 100.0,
            validation.margin(),
            report.mean_energy,
            report.free_energy_proxy,
            report.active_nodes,
            stats.sleep_runs,
            stats.growths,
            sleep_summary,
            growth_summary,
            output
        );
        print!("{line}");
        write_progress(&progress, &line);

        if stats.batches % save_every == 0 {
            save_native_state(&substrate, &output, &stats);
        }
        if max_batches.is_some_and(|limit| stats.batches >= limit) {
            save_native_state(&substrate, &output, &stats);
            break;
        }
    }
}

fn clean_substrate(slices: usize, nodes_per_slice: usize) -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices,
            nodes_per_slice,
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
            seed: 0xC1EA_0001,
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

fn train_sample(substrate: &mut NativeThermoRqmEprSubstrate, sample: Sample, stats: &mut Stats) {
    let observer = observer(sample.kind, sample.input);
    let typed = typed_observer(sample.kind);
    let phase = phase_for_kind(sample.kind);
    let input = pattern("input", sample.input, 0);
    let action = pattern("action", sample.action, 0);
    let target = pattern("target", sample.target, 1);
    let distractor = pattern("distractor", sample.distractor, 1);
    let action_cue = merge(&input, &action);

    substrate.train_observed_transition(DEFAULT_OBSERVER, 0.0, &input, &target, 1.0);
    substrate.train_observed_transition(DEFAULT_OBSERVER, 0.0, &action_cue, &target, 0.92);
    substrate.train_observed_transition(typed, phase, &input, &target, 1.0);
    substrate.train_observed_transition(observer, phase, &input, &target, 1.0);
    attenuate(substrate, DEFAULT_OBSERVER, &input, &distractor, 0.30);
    attenuate(substrate, typed, &input, &distractor, 0.25);

    if matches!(sample.kind, LessonKind::Skill) {
        let output = pattern("skill_output", sample.target, 2);
        substrate.train_observed_transition(typed, std::f32::consts::PI, &target, &output, 0.85);
    }

    stats.samples += 1;
    match sample.kind {
        LessonKind::Semantic => stats.semantic += 1,
        LessonKind::Causal => stats.causal += 1,
        LessonKind::Skill => stats.skill += 1,
        LessonKind::Episodic => stats.episodic += 1,
    }
}

fn maybe_grow(
    substrate: &mut NativeThermoRqmEprSubstrate,
    relation_density_limit: f32,
    epr_density_limit: f32,
    stats: &mut Stats,
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
    config.seed ^= (stats.growths as u64 + 1).wrapping_mul(0x9E37_79B9);
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
    stats.growths += 1;
    format!(
        "growth=added old_nodes={} new_nodes={} slices={} relation_density={:.3} epr_density={:.3}",
        old_nodes,
        substrate.thermal.node_count(),
        substrate.thermal.config.slices,
        relation_density,
        epr_density
    )
}

fn evaluate(substrate: &NativeThermoRqmEprSubstrate, lessons: &[Lesson]) -> Metrics {
    let mut trial = substrate.clone();
    let mut metrics = Metrics::default();
    for lesson in lessons {
        let report = trial.query(DEFAULT_OBSERVER, 0.0, &lesson.local);
        metrics.record(
            score(&report.candidates, &lesson.remote),
            score(&report.candidates, &lesson.distractor),
        );
        let action_cue = merge(&lesson.local, &lesson.action);
        let action = trial.query(DEFAULT_OBSERVER, 0.0, &action_cue);
        metrics.record(
            score(&action.candidates, &lesson.remote),
            score(&action.candidates, &lesson.distractor),
        );
        let typed = trial.query(typed_observer(lesson.kind), 0.0, &lesson.local);
        metrics.record(
            score(&typed.candidates, &lesson.remote),
            score(&typed.candidates, &lesson.distractor),
        );
    }
    metrics
}

fn validation_lessons() -> Vec<Lesson> {
    dataset()
        .iter()
        .take(12)
        .map(|sample| Lesson {
            kind: sample.kind,
            local: pattern("input", sample.input, 0),
            action: pattern("action", sample.action, 0),
            remote: pattern("target", sample.target, 1),
            distractor: pattern("distractor", sample.distractor, 1),
        })
        .collect()
}

fn save_native_state(substrate: &NativeThermoRqmEprSubstrate, output: &str, stats: &Stats) {
    if let Some(parent) = Path::new(output).parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::write(output, serialize_native_state(substrate, stats)) {
        Ok(()) => println!("saved=true batch={} output={}", stats.batches, output),
        Err(err) => eprintln!(
            "saved=false batch={} output={} error={}",
            stats.batches, output, err
        ),
    }
}

fn serialize_native_state(substrate: &NativeThermoRqmEprSubstrate, stats: &Stats) -> String {
    let mut out = String::new();
    out.push_str("NATIVE_THERMO_RQM_EPR_CLEAN_STATE_V1\n");
    out.push_str(&format!(
        "stats {} {} {} {} {} {} {} {}\n",
        stats.batches,
        stats.samples,
        stats.semantic,
        stats.causal,
        stats.skill,
        stats.episodic,
        stats.sleep_runs,
        stats.growths
    ));
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

fn load_native_state(path: &str) -> Result<NativeThermoRqmEprSubstrate, String> {
    let contents = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let mut lines = contents.lines();
    if lines.next() != Some("NATIVE_THERMO_RQM_EPR_CLEAN_STATE_V1") {
        return Err("version nativa invalida".to_string());
    }
    let _stats_line = lines.next().ok_or("falta stats")?;
    let thermal_config = parse_thermal_config(lines.next().ok_or("falta thermal_config")?)?;
    let rqm_config = parse_rqm_config(lines.next().ok_or("falta rqm_config")?)?;
    let mut substrate =
        NativeThermoRqmEprSubstrate::new(thermal_config, rqm_config, EntanglementConfig::default());
    let node_count = parse_count_header(lines.next().ok_or("faltan nodes")?, "nodes")?;
    for _ in 0..node_count {
        let line = lines.next().ok_or("faltan nodos")?;
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 8 || parts[0] != "n" {
            return Err(format!("nodo nativo invalido: {line}"));
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
            return Err(format!("relacion nativa invalida: {line}"));
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
    Ok(substrate)
}

fn write_progress(path: &str, line: &str) {
    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, line);
}

fn load_stats_from_progress(path: &str) -> Option<Stats> {
    let contents = fs::read_to_string(path).ok()?;
    Some(Stats {
        batches: progress_usize(&contents, "batch")?,
        samples: progress_usize(&contents, "samples").unwrap_or(0),
        semantic: progress_usize(&contents, "semantic").unwrap_or(0),
        causal: progress_usize(&contents, "causal").unwrap_or(0),
        skill: progress_usize(&contents, "skill").unwrap_or(0),
        episodic: progress_usize(&contents, "episodic").unwrap_or(0),
        sleep_runs: progress_usize(&contents, "sleep_runs").unwrap_or(0),
        growths: progress_usize(&contents, "growths").unwrap_or(0),
    })
}

fn dataset() -> &'static [Sample] {
    &[
        Sample {
            kind: LessonKind::Semantic,
            input: "vanchurin",
            action: "represent",
            target: "madelung",
            distractor: "noise",
        },
        Sample {
            kind: LessonKind::Semantic,
            input: "mera",
            action: "compress",
            target: "holography",
            distractor: "flat",
        },
        Sample {
            kind: LessonKind::Causal,
            input: "dvali",
            action: "stabilize",
            target: "criticality",
            distractor: "thermal",
        },
        Sample {
            kind: LessonKind::Causal,
            input: "wolfram",
            action: "branch",
            target: "causal_invariance",
            distractor: "random",
        },
        Sample {
            kind: LessonKind::Episodic,
            input: "graphity",
            action: "cool",
            target: "geometrogenesis",
            distractor: "complete",
        },
        Sample {
            kind: LessonKind::Skill,
            input: "landauer",
            action: "forget",
            target: "dissipation",
            distractor: "free",
        },
        Sample {
            kind: LessonKind::Episodic,
            input: "page",
            action: "retain",
            target: "retention",
            distractor: "loss",
        },
        Sample {
            kind: LessonKind::Skill,
            input: "markov",
            action: "separate",
            target: "blanket",
            distractor: "external",
        },
        Sample {
            kind: LessonKind::Semantic,
            input: "perro",
            action: "clasificar",
            target: "mamifero_domestico",
            distractor: "fuego_calor",
        },
        Sample {
            kind: LessonKind::Causal,
            input: "lluvia",
            action: "mojar",
            target: "suelo_mojado",
            distractor: "hambre_saciada",
        },
        Sample {
            kind: LessonKind::Skill,
            input: "programar",
            action: "probar",
            target: "codigo_corregido",
            distractor: "caminar",
        },
        Sample {
            kind: LessonKind::Episodic,
            input: "gato_lluvia",
            action: "recordar",
            target: "gato_negro_entorno",
            distractor: "vidrio_roto",
        },
    ]
}

fn observer(kind: LessonKind, value: &str) -> ObserverId {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    kind_label(kind).hash(&mut hasher);
    value.hash(&mut hasher);
    ObserverId(700_000 + (hasher.finish() as usize % 200_000))
}

fn typed_observer(kind: LessonKind) -> ObserverId {
    match kind {
        LessonKind::Semantic => ObserverId(261_001),
        LessonKind::Episodic => ObserverId(261_002),
        LessonKind::Causal => ObserverId(261_003),
        LessonKind::Skill => ObserverId(261_004),
    }
}

fn phase_for_kind(kind: LessonKind) -> f32 {
    match kind {
        LessonKind::Semantic => 0.0,
        LessonKind::Episodic => -std::f32::consts::FRAC_PI_2,
        LessonKind::Causal => std::f32::consts::FRAC_PI_2,
        LessonKind::Skill => std::f32::consts::PI,
    }
}

fn kind_label(kind: LessonKind) -> &'static str {
    match kind {
        LessonKind::Semantic => "semantic",
        LessonKind::Episodic => "episodic",
        LessonKind::Causal => "causal",
        LessonKind::Skill => "skill",
    }
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

fn merge(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = left.to_vec();
    out.extend_from_slice(right);
    out.sort_unstable();
    out.dedup();
    out
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

fn score(
    candidates: &[cdt_rqm_epr::native_thermo_rqm_epr::NativeCandidateScore],
    targets: &[usize],
) -> f32 {
    candidates
        .iter()
        .filter(|candidate| targets.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum()
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

fn progress_usize(contents: &str, key: &str) -> Option<usize> {
    contents
        .split_whitespace()
        .find_map(|part| part.strip_prefix(&format!("{key}="))?.parse().ok())
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn env_f32(name: &str, fallback: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false)
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
