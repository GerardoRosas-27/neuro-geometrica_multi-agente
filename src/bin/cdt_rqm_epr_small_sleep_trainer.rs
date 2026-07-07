use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::relational_field::{ObserverId, RelationalFieldConfig};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_OUTPUT: &str = "data/cdt_rqm_epr_small_sleep.cdt_rqm";
const DEFAULT_PROGRESS: &str = "data/cdt_rqm_epr_small_sleep.progress";
const DEFAULT_NODES_PER_SLICE: usize = 1024;
const BLOCK_SLICES: usize = 4;
static ACTIVE_BLOCK_START: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct Sample {
    kind: SampleKind,
    input: &'static str,
    target: &'static str,
    distractor: &'static str,
}

#[derive(Clone, Copy)]
enum SampleKind {
    Concept,
    Causal,
    Skill,
    Episode,
    Correlation,
}

#[derive(Default)]
struct Stats {
    batches: usize,
    samples: usize,
    concept: usize,
    causal: usize,
    skill: usize,
    episode: usize,
    correlation: usize,
    sleep_runs: usize,
    growths: usize,
}

fn main() {
    let output =
        env::var("CDT_RQM_EPR_SMALL_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT.to_string());
    let progress =
        env::var("CDT_RQM_EPR_SMALL_PROGRESS").unwrap_or_else(|_| DEFAULT_PROGRESS.to_string());
    let nodes_per_slice =
        env_usize("CDT_RQM_EPR_SMALL_NODES_PER_SLICE", DEFAULT_NODES_PER_SLICE).max(256);
    let batch_size = env_usize("CDT_RQM_EPR_SMALL_BATCH_SIZE", 8).max(1);
    let save_every = env_usize("CDT_RQM_EPR_SMALL_SAVE_EVERY_BATCHES", 1).max(1);
    let sleep_every = env_usize("CDT_RQM_EPR_SMALL_SLEEP_EVERY_BATCHES", 3).max(1);
    let sleep_attempts = env_usize("CDT_RQM_EPR_SMALL_SLEEP_ATTEMPTS", 8);
    let grow_edge_ratio = env_f32("CDT_RQM_EPR_SMALL_GROW_EDGE_RATIO", 0.72);
    let grow_regge_per_edge = env_f32("CDT_RQM_EPR_SMALL_GROW_REGGE_PER_EDGE", 10.0);
    let max_batches = arg_value("--batches").and_then(|value| value.parse::<usize>().ok());

    let mut substrate = CdtRqmUniverseSubstrate::new(config(nodes_per_slice));
    let epr_config = EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    };
    substrate.enable_entanglement(epr_config);

    let mut stats = load_stats_from_progress(&progress).unwrap_or_default();
    if Path::new(&output).exists() {
        match substrate.load_consolidated_state(&output) {
            Ok(()) => {
                if substrate.entanglement.is_none() {
                    substrate.enable_entanglement(epr_config);
                }
                ACTIVE_BLOCK_START.store(
                    stats_block_start(&progress).unwrap_or_else(|| {
                        substrate
                            .hardware
                            .config
                            .slices
                            .saturating_sub(BLOCK_SLICES)
                    }),
                    Ordering::Relaxed,
                );
                println!(
                    "resume=true output={} batch_start={} slices={} relations={} epr_links={}",
                    output,
                    stats.batches,
                    substrate.hardware.config.slices,
                    substrate.relation_count(),
                    substrate
                        .entanglement_summary()
                        .map(|r| r.active_links)
                        .unwrap_or(0)
                );
            }
            Err(err) => {
                eprintln!(
                    "resume=false output={} error={} starting_fresh=true",
                    output, err
                );
                stats = Stats::default();
            }
        }
    }
    println!("CDT-RQM+EPR small sleep trainer");
    println!(
        "output={} nodes_per_slice={} batch_size={} sleep_every={} max_batches={}",
        output,
        nodes_per_slice,
        batch_size,
        sleep_every,
        max_batches
            .map(|value| value.to_string())
            .unwrap_or_else(|| "infinite".to_string())
    );

    loop {
        stats.batches += 1;
        for i in 0..batch_size {
            let sample = dataset()[(stats.batches + i) % dataset().len()];
            train_sample(&mut substrate, sample, &mut stats);
        }

        let mut sleep_summary = "sleep=skipped".to_string();
        let mut growth_summary = "growth=none".to_string();
        if stats.batches % sleep_every == 0 {
            stats.sleep_runs += 1;
            let validation = validation_set();
            let report = substrate.anneal_after_migration(&validation, sleep_attempts);
            sleep_summary = format!(
                "sleep=graphity accepted={} acc={:.1}%->{:.1}% leak={:.1}%->{:.1}% regge={:.1}->{:.1}",
                report.accepted,
                report.initial_accuracy * 100.0,
                report.final_accuracy * 100.0,
                report.initial_leakage * 100.0,
                report.final_leakage * 100.0,
                report.initial_regge,
                report.final_regge
            );
            growth_summary = maybe_grow(
                &mut substrate,
                grow_edge_ratio,
                grow_regge_per_edge,
                &mut stats,
            );
        }

        let epr = substrate.entanglement_summary().unwrap();
        let active_edges = active_edges(&substrate);
        let line = format!(
            "batch={} block_start={} slices={} samples={} concept={} causal={} skill={} episode={} corr={} relations={} epr_links={} active_edges={} regge={:.1} temp={:.3} {} {} output={}\n",
            stats.batches,
            ACTIVE_BLOCK_START.load(Ordering::Relaxed),
            substrate.hardware.config.slices,
            stats.samples,
            stats.concept,
            stats.causal,
            stats.skill,
            stats.episode,
            stats.correlation,
            substrate.relation_count(),
            epr.active_links,
            active_edges,
            substrate.hardware.regge_action(),
            substrate.hardware.temperature,
            sleep_summary,
            growth_summary,
            output
        );
        print!("{line}");
        write_progress(&progress, &line);

        if stats.batches % save_every == 0 {
            save(&substrate, &output, stats.batches);
        }
        if max_batches.is_some_and(|limit| stats.batches >= limit) {
            save(&substrate, &output, stats.batches);
            break;
        }
    }
}

fn train_sample(substrate: &mut CdtRqmUniverseSubstrate, sample: Sample, stats: &mut Stats) {
    let phase = phase_for_kind(sample.kind);
    let observer = observer(sample.kind, sample.input);
    let input = pattern("input", sample.input, 0);
    let target = pattern("target", sample.target, 1);
    substrate.hardware.clear_activity();
    substrate.train_observed_transition(observer, phase, &input, &target, 1.0);

    // EPR is a logical shortcut between boundary input and concept/target attractor.
    for (&a, &b) in input.iter().zip(target.iter()) {
        substrate.observe_entanglement_correlation(a, b, 0.40);
    }
    if matches!(sample.kind, SampleKind::Skill) {
        let output = pattern("skill_output", sample.target, 2);
        substrate.hardware.clear_activity();
        substrate.train_observed_transition(observer, std::f32::consts::PI, &target, &output, 0.90);
    }

    stats.samples += 1;
    match sample.kind {
        SampleKind::Concept => stats.concept += 1,
        SampleKind::Causal => stats.causal += 1,
        SampleKind::Skill => stats.skill += 1,
        SampleKind::Episode => stats.episode += 1,
        SampleKind::Correlation => stats.correlation += 1,
    }
}

fn maybe_grow(
    substrate: &mut CdtRqmUniverseSubstrate,
    edge_ratio_threshold: f32,
    regge_per_edge_threshold: f32,
    stats: &mut Stats,
) -> String {
    let active = active_edges(substrate);
    let total = substrate.hardware.edges.len().max(1);
    let edge_ratio = active as f32 / total as f32;
    let regge_per_edge = substrate.hardware.regge_action() / active.max(1) as f32;
    if edge_ratio < edge_ratio_threshold && regge_per_edge < regge_per_edge_threshold {
        return format!(
            "growth=checked edge_ratio={:.3} regge_per_edge={:.3}",
            edge_ratio, regge_per_edge
        );
    }
    let start = substrate.grow_foliated_block(BLOCK_SLICES);
    ACTIVE_BLOCK_START.store(start, Ordering::Relaxed);
    stats.growths += 1;
    format!(
        "growth=added start_slice={} total_slices={} edge_ratio={:.3} regge_per_edge={:.3}",
        start, substrate.hardware.config.slices, edge_ratio, regge_per_edge
    )
}

fn validation_set() -> Vec<(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)> {
    dataset()
        .iter()
        .take(8)
        .map(|sample| {
            (
                observer(sample.kind, sample.input),
                phase_for_kind(sample.kind),
                pattern("input", sample.input, 0),
                pattern("target", sample.target, 1),
                pattern("distractor", sample.distractor, 1),
            )
        })
        .collect()
}

fn active_edges(substrate: &CdtRqmUniverseSubstrate) -> usize {
    substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count()
}

fn save(substrate: &CdtRqmUniverseSubstrate, output: &str, batch: usize) {
    match substrate.save_consolidated_state(output) {
        Ok(()) => println!("saved=true batch={} output={}", batch, output),
        Err(err) => eprintln!(
            "saved=false batch={} output={} error={}",
            batch, output, err
        ),
    }
}

fn write_progress(path: &str, line: &str) {
    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, line);
}

fn load_stats_from_progress(path: &str) -> Option<Stats> {
    let contents = fs::read_to_string(path).ok()?;
    let mut stats = Stats::default();
    stats.batches = progress_usize(&contents, "batch")?;
    stats.samples = progress_usize(&contents, "samples").unwrap_or(0);
    stats.concept = progress_usize(&contents, "concept").unwrap_or(0);
    stats.causal = progress_usize(&contents, "causal").unwrap_or(0);
    stats.skill = progress_usize(&contents, "skill").unwrap_or(0);
    stats.episode = progress_usize(&contents, "episode").unwrap_or(0);
    stats.correlation = progress_usize(&contents, "corr").unwrap_or(0);
    Some(stats)
}

fn stats_block_start(path: &str) -> Option<usize> {
    let contents = fs::read_to_string(path).ok()?;
    progress_usize(&contents, "block_start")
}

fn progress_usize(contents: &str, key: &str) -> Option<usize> {
    contents
        .split_whitespace()
        .find_map(|part| part.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse::<usize>().ok())
}

fn pattern(prefix: &str, value: &str, slice: usize) -> Vec<usize> {
    let nodes_per_slice =
        env_usize("CDT_RQM_EPR_SMALL_NODES_PER_SLICE", DEFAULT_NODES_PER_SLICE).max(256);
    let block = ACTIVE_BLOCK_START.load(Ordering::Relaxed);
    let mut out = (0..16)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalize(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            (block + slice) * nodes_per_slice + (hasher.finish() as usize % nodes_per_slice)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn observer(kind: SampleKind, value: &str) -> ObserverId {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sample_kind_label(kind).hash(&mut hasher);
    normalize(value).hash(&mut hasher);
    ObserverId(500_000 + (hasher.finish() as usize % 400_000))
}

fn phase_for_kind(kind: SampleKind) -> f32 {
    match kind {
        SampleKind::Concept => 0.0,
        SampleKind::Causal => std::f32::consts::FRAC_PI_2,
        SampleKind::Skill => std::f32::consts::PI,
        SampleKind::Episode => -std::f32::consts::FRAC_PI_2,
        SampleKind::Correlation => 0.25,
    }
}

fn sample_kind_label(kind: SampleKind) -> &'static str {
    match kind {
        SampleKind::Concept => "concept",
        SampleKind::Causal => "causal",
        SampleKind::Skill => "skill",
        SampleKind::Episode => "episode",
        SampleKind::Correlation => "correlation",
    }
}

fn normalize(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| match ch {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            other => other,
        })
        .filter(|ch| ch.is_ascii_alphanumeric() || ch.is_whitespace())
        .collect()
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn env_f32(name: &str, fallback: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(fallback)
}

fn config(nodes_per_slice: usize) -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: BLOCK_SLICES,
            nodes_per_slice,
            initial_spatial_connectivity: 0.00004,
            initial_temporal_connectivity: 0.00002,
            target_spatial_degree: 5,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 12,
            seed: 92_001,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.09,
            phase_learning_rate: 0.22,
            coherence_learning_rate: 0.12,
            uncertainty_learning_rate: 0.10,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.025,
        },
        max_quantum_candidates: 96,
        rqm_feedback_gain: 0.40,
    }
}

fn dataset() -> &'static [Sample] {
    &[
        Sample {
            kind: SampleKind::Concept,
            input: "perro",
            target: "mamifero domestico que ladra y puede ser mascota",
            distractor: "fuego produce calor",
        },
        Sample {
            kind: SampleKind::Concept,
            input: "agua",
            target: "liquido que moja y sostiene vida",
            distractor: "vidrio fragil",
        },
        Sample {
            kind: SampleKind::Causal,
            input: "fuego",
            target: "calor",
            distractor: "suelo mojado",
        },
        Sample {
            kind: SampleKind::Causal,
            input: "lluvia",
            target: "suelo mojado",
            distractor: "hambre saciada",
        },
        Sample {
            kind: SampleKind::Causal,
            input: "golpear vidrio",
            target: "vidrio roto",
            distractor: "planta crece",
        },
        Sample {
            kind: SampleKind::Skill,
            input: "sumar",
            target: "leer numeros alinear cantidades combinar resultado",
            distractor: "dibujar contorno",
        },
        Sample {
            kind: SampleKind::Skill,
            input: "programar",
            target: "definir objetivo diseñar pasos escribir codigo probar corregir",
            distractor: "caminar equilibrar pie",
        },
        Sample {
            kind: SampleKind::Episode,
            input: "vi gato negro bajo lluvia",
            target: "gato negro existe y lluvia cambia entorno",
            distractor: "fuego calienta",
        },
        Sample {
            kind: SampleKind::Episode,
            input: "comi y dejo de doler hambre",
            target: "comer sacia hambre",
            distractor: "lluvia moja suelo",
        },
        Sample {
            kind: SampleKind::Correlation,
            input: "perro humano",
            target: "mascota relacion social",
            distractor: "vidrio roto",
        },
        Sample {
            kind: SampleKind::Correlation,
            input: "agua planta",
            target: "agua ayuda crecimiento planta",
            distractor: "programar codigo",
        },
    ]
}
