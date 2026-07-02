use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_OUTPUT: &str = "data/cdt_rqm_infinite_concepts.cdt_rqm";
const DEFAULT_PROGRESS: &str = "data/cdt_rqm_infinite_concepts.progress";
const BLOCK_SLICES: usize = 4;
static ACTIVE_BLOCK_START: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct ConceptSeed {
    name: &'static str,
    attributes: &'static [&'static str],
}

#[derive(Clone, Copy)]
struct CausalSeed {
    cause: &'static str,
    effect: &'static str,
}

#[derive(Clone, Copy)]
struct SkillSeed {
    name: &'static str,
    steps: &'static [&'static str],
}

#[derive(Clone, Copy)]
struct EpisodeSeed {
    event: &'static str,
    facts: &'static [&'static str],
    consolidated: &'static [&'static str],
}

fn main() {
    let output = env::var("CDT_RQM_INFINITE_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT.to_string());
    let progress_path =
        env::var("CDT_RQM_INFINITE_PROGRESS").unwrap_or_else(|_| DEFAULT_PROGRESS.to_string());
    let nodes_per_slice = env_usize("CDT_RQM_INFINITE_NODES_PER_SLICE", 65_536).max(1024);
    let batch_size = env_usize("CDT_RQM_INFINITE_BATCH_SIZE", 32).max(1);
    let save_every_batches = env_usize("CDT_RQM_INFINITE_SAVE_EVERY_BATCHES", 5).max(1);
    let anneal_every_batches = env_usize("CDT_RQM_INFINITE_ANNEAL_EVERY_BATCHES", 5).max(1);
    let anneal_attempts = env_usize("CDT_RQM_INFINITE_ANNEAL_ATTEMPTS", 8);
    let grow_every_batches = env_usize("CDT_RQM_INFINITE_GROW_CHECK_EVERY_BATCHES", 5).max(1);
    let grow_edge_ratio = env_f32("CDT_RQM_INFINITE_GROW_EDGE_RATIO", 0.65);
    let grow_regge_per_edge = env_f32("CDT_RQM_INFINITE_GROW_REGGE_PER_EDGE", 8.0);
    let max_batches = arg_value("--batches").and_then(|value| value.parse::<usize>().ok());

    let mut substrate = CdtRqmUniverseSubstrate::new(config(nodes_per_slice));
    let mut stats = TrainerStats::default();
    let mut batch = 0_usize;

    println!("CDT-RQM infinite concept trainer");
    println!(
        "output={} nodes_per_slice={} batch_size={} save_every_batches={} anneal_every_batches={} grow_every_batches={} max_batches={}",
        output,
        nodes_per_slice,
        batch_size,
        save_every_batches,
        anneal_every_batches,
        grow_every_batches,
        max_batches
            .map(|value| value.to_string())
            .unwrap_or_else(|| "infinite".to_string())
    );

    loop {
        batch += 1;
        for item in 0..batch_size {
            train_curriculum_item(&mut substrate, batch, item, &mut stats);
        }

        let mut anneal_summary = "anneal=skipped".to_string();
        if batch % anneal_every_batches == 0 {
            let validation = validation_set();
            let report = substrate.anneal_after_migration(&validation, anneal_attempts);
            stats.anneal_attempts += report.attempts;
            stats.anneal_accepted += report.accepted;
            anneal_summary = format!(
                "anneal=run accepted={} regge={:.3}->{:.3} leakage={:.1}%->{:.1}%",
                report.accepted,
                report.initial_regge,
                report.final_regge,
                report.initial_leakage * 100.0,
                report.final_leakage * 100.0
            );
        }

        let active_edges = substrate
            .hardware
            .edges
            .iter()
            .filter(|edge| edge.active)
            .count();
        let mut growth_summary = "growth=none".to_string();
        if batch % grow_every_batches == 0 {
            growth_summary = maybe_grow_substrate(
                &mut substrate,
                active_edges,
                grow_edge_ratio,
                grow_regge_per_edge,
            );
        }

        let line = format!(
            "batch={} block_start={} slices={} concepts={} causal={} skills={} episodes={} correlations={} relations={} active_edges={} regge={:.3} temp={:.3} {} {} output={}\n",
            batch,
            ACTIVE_BLOCK_START.load(Ordering::Relaxed),
            substrate.hardware.config.slices,
            stats.concepts,
            stats.causal,
            stats.skills,
            stats.episodes,
            stats.correlations,
            substrate.relation_count(),
            active_edges,
            substrate.hardware.regge_action(),
            substrate.hardware.temperature,
            anneal_summary,
            growth_summary,
            output
        );
        print!("{line}");
        write_progress(&progress_path, &line);

        if batch % save_every_batches == 0 {
            save_state(&substrate, &output, batch);
        }

        if max_batches.is_some_and(|limit| batch >= limit) {
            save_state(&substrate, &output, batch);
            break;
        }
    }
}

#[derive(Default)]
struct TrainerStats {
    concepts: usize,
    causal: usize,
    skills: usize,
    episodes: usize,
    correlations: usize,
    anneal_attempts: usize,
    anneal_accepted: usize,
}

fn train_curriculum_item(
    substrate: &mut CdtRqmUniverseSubstrate,
    batch: usize,
    item: usize,
    stats: &mut TrainerStats,
) {
    match (batch + item) % 4 {
        0 => {
            let concept = concept_seeds()[(batch + item) % concept_seeds().len()];
            train_concept(substrate, concept);
            stats.concepts += 1;
        }
        1 => {
            let causal = causal_seeds()[(batch + item) % causal_seeds().len()];
            train_causal(substrate, causal);
            stats.causal += 1;
        }
        2 => {
            let skill = skill_seeds()[(batch + item) % skill_seeds().len()];
            train_skill(substrate, skill);
            stats.skills += 1;
        }
        _ => {
            let episode = episode_seeds()[(batch + item) % episode_seeds().len()];
            train_episode(substrate, episode);
            stats.episodes += 1;
        }
    }
    stats.correlations += train_correlations(substrate, batch, item);
}

fn train_concept(substrate: &mut CdtRqmUniverseSubstrate, concept: ConceptSeed) {
    let observer = observer("concept", concept.name);
    let concept_knot = concept_pattern(concept.name, 1);
    for attribute in concept.attributes {
        let attr = concept_pattern(attribute, 1);
        substrate.train_observed_transition(observer, 0.0, &concept_knot, &attr, 1.0);
        reinforce_bidirectional(substrate, observer, 0.0, &concept_knot, &attr, 0.92);
    }
    reinforce_internal(substrate, observer, 0.0, &concept_knot, 0.88);
}

fn train_causal(substrate: &mut CdtRqmUniverseSubstrate, causal: CausalSeed) {
    let cause = concept_pattern(causal.cause, 0);
    let effect = concept_pattern(causal.effect, 1);
    let observer = observer("causal", causal.cause);
    substrate.train_observed_transition(
        observer,
        std::f32::consts::FRAC_PI_2,
        &cause,
        &effect,
        1.0,
    );
}

fn train_skill(substrate: &mut CdtRqmUniverseSubstrate, skill: SkillSeed) {
    let observer = observer("skill", skill.name);
    let sequence = skill
        .steps
        .iter()
        .enumerate()
        .map(|(idx, step)| concept_pattern(step, idx.min(2)))
        .collect::<Vec<_>>();
    substrate.train_binary_sequence(observer, std::f32::consts::PI, &sequence, 1.0);
}

fn train_episode(substrate: &mut CdtRqmUniverseSubstrate, episode: EpisodeSeed) {
    let observer = observer("episode", episode.event);
    let event = concept_pattern(episode.event, 0);
    for fact in episode.facts {
        let fact_pattern = concept_pattern(fact, 1);
        substrate.train_observed_transition(
            observer,
            -std::f32::consts::FRAC_PI_2,
            &event,
            &fact_pattern,
            0.95,
        );
    }
    for consolidated in episode.consolidated {
        let consolidated_pattern = concept_pattern(consolidated, 2);
        substrate.train_observed_transition(observer, 0.0, &event, &consolidated_pattern, 0.85);
    }
}

fn train_correlations(substrate: &mut CdtRqmUniverseSubstrate, batch: usize, item: usize) -> usize {
    let left = concept_seeds()[(batch + item) % concept_seeds().len()];
    let right = concept_seeds()[(batch + item + 3) % concept_seeds().len()];
    let observer = observer("correlation", left.name);
    let left_pattern = concept_pattern(left.name, 1);
    let right_pattern = concept_pattern(right.name, 1);
    reinforce_bidirectional(
        substrate,
        observer,
        0.0,
        &left_pattern,
        &right_pattern,
        0.35,
    );
    1
}

fn reinforce_bidirectional(
    substrate: &mut CdtRqmUniverseSubstrate,
    observer: ObserverId,
    phase: f32,
    left: &[usize],
    right: &[usize],
    strength: f32,
) {
    for &a in left.iter().take(8) {
        for &b in right.iter().take(8) {
            substrate
                .software
                .reinforce_relation(observer, a, b, phase, strength);
        }
    }
}

fn reinforce_internal(
    substrate: &mut CdtRqmUniverseSubstrate,
    observer: ObserverId,
    phase: f32,
    pattern: &[usize],
    strength: f32,
) {
    for window in pattern.windows(2) {
        substrate
            .software
            .reinforce_relation(observer, window[0], window[1], phase, strength);
    }
}

fn validation_set() -> Vec<(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)> {
    causal_seeds()
        .iter()
        .take(8)
        .map(|seed| {
            (
                observer("causal", seed.cause),
                std::f32::consts::FRAC_PI_2,
                concept_pattern(seed.cause, 0),
                concept_pattern(seed.effect, 1),
                concept_pattern("distractor_validation", 1),
            )
        })
        .collect()
}

fn concept_pattern(value: &str, slice: usize) -> Vec<usize> {
    let nodes_per_slice = env_usize("CDT_RQM_INFINITE_NODES_PER_SLICE", 65_536).max(1024);
    let block_start = ACTIVE_BLOCK_START.load(Ordering::Relaxed);
    let mut out = (0..16)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            normalize(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            (block_start + slice) * nodes_per_slice + (hasher.finish() as usize % nodes_per_slice)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn observer(kind: &str, value: &str) -> ObserverId {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    kind.hash(&mut hasher);
    normalize(value).hash(&mut hasher);
    ObserverId(100_000 + (hasher.finish() as usize % 900_000))
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

fn save_state(substrate: &CdtRqmUniverseSubstrate, output: &str, batch: usize) {
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

fn maybe_grow_substrate(
    substrate: &mut CdtRqmUniverseSubstrate,
    active_edges: usize,
    edge_ratio_threshold: f32,
    regge_per_edge_threshold: f32,
) -> String {
    let total_edges = substrate.hardware.edges.len().max(1);
    let edge_ratio = active_edges as f32 / total_edges as f32;
    let regge_per_edge = substrate.hardware.regge_action() / active_edges.max(1) as f32;
    if edge_ratio < edge_ratio_threshold && regge_per_edge < regge_per_edge_threshold {
        return format!(
            "growth=checked edge_ratio={:.3} regge_per_edge={:.3}",
            edge_ratio, regge_per_edge
        );
    }
    let start_slice = substrate.grow_foliated_block(BLOCK_SLICES);
    ACTIVE_BLOCK_START.store(start_slice, Ordering::Relaxed);
    format!(
        "growth=added start_slice={} total_slices={} edge_ratio={:.3} regge_per_edge={:.3}",
        start_slice, substrate.hardware.config.slices, edge_ratio, regge_per_edge
    )
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
            slices: 4,
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
            max_new_edges_per_step: 16,
            seed: 91_001,
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
        max_quantum_candidates: 64,
        rqm_feedback_gain: 0.40,
    }
}

fn concept_seeds() -> &'static [ConceptSeed] {
    &[
        ConceptSeed {
            name: "perro",
            attributes: &[
                "tiene cuatro patas",
                "es mamifero",
                "puede ladrar",
                "animal domestico",
                "relacionado con humanos",
                "parecido a lobos",
                "puede ser mascota",
            ],
        },
        ConceptSeed {
            name: "agua",
            attributes: &[
                "liquido",
                "moja superficies",
                "necesaria para vida",
                "fluye",
                "puede evaporarse",
            ],
        },
        ConceptSeed {
            name: "fuego",
            attributes: &[
                "produce calor",
                "produce luz",
                "consume combustible",
                "puede quemar",
            ],
        },
        ConceptSeed {
            name: "vidrio",
            attributes: &[
                "fragil",
                "transparente",
                "puede romperse",
                "material solido",
            ],
        },
        ConceptSeed {
            name: "hambre",
            attributes: &["necesidad corporal", "se reduce al comer", "señal interna"],
        },
    ]
}

fn causal_seeds() -> &'static [CausalSeed] {
    &[
        CausalSeed {
            cause: "fuego",
            effect: "calor",
        },
        CausalSeed {
            cause: "golpear vidrio",
            effect: "romper vidrio",
        },
        CausalSeed {
            cause: "comer",
            effect: "saciar hambre",
        },
        CausalSeed {
            cause: "lluvia",
            effect: "suelo mojado",
        },
        CausalSeed {
            cause: "estudiar",
            effect: "aprender",
        },
        CausalSeed {
            cause: "practicar",
            effect: "mejorar habilidad",
        },
        CausalSeed {
            cause: "abrir puerta",
            effect: "permitir paso",
        },
        CausalSeed {
            cause: "sembrar semilla",
            effect: "crecer planta",
        },
    ]
}

fn skill_seeds() -> &'static [SkillSeed] {
    &[
        SkillSeed {
            name: "sumar",
            steps: &[
                "leer numeros",
                "alinear cantidades",
                "combinar",
                "producir resultado",
            ],
        },
        SkillSeed {
            name: "programar",
            steps: &[
                "definir objetivo",
                "diseñar pasos",
                "escribir codigo",
                "probar",
                "corregir",
            ],
        },
        SkillSeed {
            name: "caminar",
            steps: &["equilibrio", "levantar pie", "avanzar", "apoyar pie"],
        },
        SkillSeed {
            name: "dibujar",
            steps: &[
                "observar forma",
                "trazar contorno",
                "agregar detalle",
                "ajustar proporcion",
            ],
        },
        SkillSeed {
            name: "conducir",
            steps: &["percibir camino", "controlar velocidad", "girar", "frenar"],
        },
    ]
}

fn episode_seeds() -> &'static [EpisodeSeed] {
    &[
        EpisodeSeed {
            event: "vi un gato negro bajo lluvia ayer",
            facts: &["gato negro", "lluvia", "ayer", "observacion visual"],
            consolidated: &["gatos negros existen", "llover cambia el entorno"],
        },
        EpisodeSeed {
            event: "toque vidrio y se rompio",
            facts: &["contacto con vidrio", "vidrio fragil", "ruptura"],
            consolidated: &["golpear vidrio puede romperlo"],
        },
        EpisodeSeed {
            event: "comi y dejo de doler hambre",
            facts: &["comer", "hambre previa", "saciedad posterior"],
            consolidated: &["comer sacia hambre"],
        },
    ]
}
