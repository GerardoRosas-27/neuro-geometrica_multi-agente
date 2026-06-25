use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::hash::{Hash, Hasher};

const BASE_STATE_PATH: &str = "data/snga_scaled_gemma_language_fractal_compressed.snga";
const TRAINED_STATE_PATH: &str = "data/snga_fractal_gemma_spanish_curriculum.snga";
const DEFAULT_AGENT_COUNT: usize = 5_760;
const DEFAULT_PATTERN_NODES: usize = 5_760;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;
const TOP_K: usize = 32;

#[derive(Clone, Copy)]
enum Region {
    FineLetters,
    LocalSyllables,
    MediumWords,
    UpperSentences,
    AssociativeMeaning,
}

#[derive(Clone, Copy)]
struct Case {
    stage: &'static str,
    unit: &'static str,
    input: &'static str,
    target: &'static str,
}

#[derive(Default)]
struct StageStats {
    total: usize,
    nonzero: usize,
    target_hits: usize,
    confidence: f32,
    overlap: f32,
}

fn main() {
    println!("SNGA fractal Gemma curriculum probe");
    let trained_state_path =
        env::var("SNGA_CURRICULUM_STATE_PATH").unwrap_or_else(|_| TRAINED_STATE_PATH.to_string());

    let mut trained = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    match trained.load_persistent_state(&trained_state_path) {
        Ok(report) => println!(
            "trained_loaded=true path={} agents={} edges={} causal_edges={}",
            trained_state_path, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("trained_loaded=false error={err}");
            return;
        }
    }

    let mut baseline = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    match baseline.load_persistent_state(BASE_STATE_PATH) {
        Ok(report) => println!(
            "baseline_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("baseline_loaded=false error={err}");
            println!("baseline_fallback=empty_fractal_same_size");
        }
    };

    let cases = validation_cases();
    println!("cases={}", cases.len());
    for stage in [
        "letras",
        "silabas",
        "palabras",
        "uniones_de_palabras",
        "oraciones",
        "gramatica_basica",
        "espanol_medio",
    ] {
        let trained_stats = eval_stage(&trained, &cases, stage);
        let baseline_stats = eval_stage(&baseline, &cases, stage);
        let trained_legacy_stats = eval_stage_legacy(&trained, &cases, stage);
        print_stage("trained", stage, trained_stats);
        print_stage("trained_legacy", stage, trained_legacy_stats);
        print_stage("baseline", stage, baseline_stats);
    }

    print_network("trained", &trained);
    print_network("baseline", &baseline);
}

fn eval_stage_legacy(network: &SimplicialNetwork, cases: &[Case], stage: &str) -> StageStats {
    let mut stats = StageStats::default();
    for case in cases.iter().filter(|case| case.stage == stage) {
        stats.total += 1;
        let input = legacy_hierarchical_text_pattern("input", case.input, network.agents.len());
        let target = legacy_hierarchical_text_pattern("target", case.target, network.agents.len());
        let predicted = network.predict_next_pattern(&input, 1, TOP_K);
        let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let overlap = overlap_ratio(&predicted_ids, &target);
        if !predicted.is_empty() {
            stats.nonzero += 1;
        }
        if overlap > 0.0 {
            stats.target_hits += 1;
        }
        stats.confidence += predicted.iter().map(|(_, score)| *score).sum::<f32>() / TOP_K as f32;
        stats.overlap += overlap;
    }
    if stats.total > 0 {
        stats.confidence /= stats.total as f32;
        stats.overlap /= stats.total as f32;
    }
    stats
}

fn eval_stage(network: &SimplicialNetwork, cases: &[Case], stage: &str) -> StageStats {
    let mut stats = StageStats::default();
    for case in cases.iter().filter(|case| case.stage == stage) {
        stats.total += 1;
        let input = hierarchical_text_pattern("input", case.input, network.agents.len());
        let target = hierarchical_text_pattern("target", case.target, network.agents.len());
        let predicted = network.predict_next_pattern(&input, 1, TOP_K);
        let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let overlap = overlap_ratio(&predicted_ids, &target);
        if !predicted.is_empty() {
            stats.nonzero += 1;
        }
        if overlap > 0.0 {
            stats.target_hits += 1;
        }
        stats.confidence += predicted.iter().map(|(_, score)| *score).sum::<f32>() / TOP_K as f32;
        stats.overlap += overlap;
        println!(
            "case stage={} unit={:?} input={:?} target={:?} predicted={} overlap={:.1}%",
            stage,
            case.unit,
            case.input,
            case.target,
            predicted.len(),
            overlap * 100.0
        );
    }
    if stats.total > 0 {
        stats.confidence /= stats.total as f32;
        stats.overlap /= stats.total as f32;
    }
    stats
}

fn print_stage(label: &str, stage: &str, stats: StageStats) {
    println!(
        "{label}_{stage}: nonzero={}/{} target_hits={}/{} conf={:.3} target_overlap={:.1}%",
        stats.nonzero,
        stats.total,
        stats.target_hits,
        stats.total,
        stats.confidence,
        stats.overlap * 100.0
    );
}

fn print_network(label: &str, network: &SimplicialNetwork) {
    let stats = network.plasticity_stats();
    println!(
        "{label}_network: nodes={} edges={} associative={} consolidated={} causal={} energy={:.1}",
        network.agents.len(),
        stats.active_edges,
        stats.associative_edges,
        stats.consolidated_edges,
        stats.causal_edges,
        network.total_free_energy()
    );
}

fn validation_cases() -> Vec<Case> {
    vec![
        Case {
            stage: "letras",
            unit: "vocal a",
            input: "a",
            target: "vocal abierta",
        },
        Case {
            stage: "letras",
            unit: "consonante m",
            input: "m",
            target: "consonante nasal",
        },
        Case {
            stage: "silabas",
            unit: "ma",
            input: "m a",
            target: "ma",
        },
        Case {
            stage: "silabas",
            unit: "pa",
            input: "p a",
            target: "pa",
        },
        Case {
            stage: "palabras",
            unit: "casa",
            input: "c a s a",
            target: "casa lugar para vivir",
        },
        Case {
            stage: "palabras",
            unit: "nino",
            input: "n i n o",
            target: "nino persona pequena",
        },
        Case {
            stage: "uniones_de_palabras",
            unit: "nino corre",
            input: "nino corre",
            target: "sujeto y verbo",
        },
        Case {
            stage: "uniones_de_palabras",
            unit: "come pan",
            input: "come pan",
            target: "verbo y objeto",
        },
        Case {
            stage: "oraciones",
            unit: "oracion simple",
            input: "el nino come pan",
            target: "sujeto verbo objeto",
        },
        Case {
            stage: "oraciones",
            unit: "pregunta",
            input: "que es una palabra",
            target: "pregunta por definicion",
        },
        Case {
            stage: "gramatica_basica",
            unit: "plural",
            input: "los ninos comen",
            target: "varios sujetos",
        },
        Case {
            stage: "espanol_medio",
            unit: "causa",
            input: "llueve entonces el suelo se moja",
            target: "causa y efecto",
        },
    ]
}

fn hierarchical_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    let mut out = pattern(prefix, text, nodes);
    let normalized = normalize_text(text);
    out.extend(regional_pattern(
        prefix,
        &normalized,
        PATTERN_SIZE,
        nodes,
        text_region(&normalized),
    ));
    for (pos, ch) in normalized.chars().enumerate().take(24) {
        out.extend(letter_pattern(ch, pos, nodes));
    }
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    for word in words.iter().take(12) {
        out.extend(pattern("word", word, nodes));
        out.extend(regional_pattern(
            "word",
            word,
            PATTERN_SIZE,
            nodes,
            Region::MediumWords,
        ));
    }
    for pair in words.windows(2) {
        out.extend(pattern(
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
            nodes,
        ));
        out.extend(regional_pattern(
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
            PATTERN_SIZE,
            nodes,
            Region::UpperSentences,
        ));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn legacy_hierarchical_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    let mut out = pattern(prefix, text, nodes);
    let normalized = normalize_text(text);
    for (pos, ch) in normalized.chars().enumerate().take(24) {
        out.extend(legacy_letter_pattern(ch, pos, nodes));
    }
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    for word in words.iter().take(12) {
        out.extend(pattern("word", word, nodes));
    }
    for pair in words.windows(2) {
        out.extend(pattern(
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
            nodes,
        ));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn legacy_letter_pattern(ch: char, pos: usize, nodes: usize) -> Vec<usize> {
    let nodes = pattern_nodes(nodes);
    (0..LETTER_PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "letter".hash(&mut hasher);
            ch.hash(&mut hasher);
            pos.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn letter_pattern(ch: char, pos: usize, nodes: usize) -> Vec<usize> {
    let mut pattern = (0..LETTER_PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "letter".hash(&mut hasher);
            ch.hash(&mut hasher);
            pos.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect::<Vec<_>>();
    pattern.extend(regional_pattern(
        "letter",
        &format!("{ch}_{pos}"),
        LETTER_PATTERN_SIZE,
        nodes,
        Region::FineLetters,
    ));
    pattern.sort_unstable();
    pattern.dedup();
    pattern
}

fn pattern(prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    let nodes = pattern_nodes(nodes);
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalize_text(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn regional_pattern(
    prefix: &str,
    value: &str,
    size: usize,
    nodes: usize,
    region: Region,
) -> Vec<usize> {
    let (start, len) = region_bounds(region, nodes);
    (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "regional".hash(&mut hasher);
            prefix.hash(&mut hasher);
            normalize_text(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            start + (hasher.finish() as usize % len.max(1))
        })
        .collect()
}

fn text_region(normalized: &str) -> Region {
    let words = normalized.split_whitespace().count();
    let chars = normalized.chars().filter(|ch| !ch.is_whitespace()).count();
    if words <= 1 && chars <= 2 {
        Region::LocalSyllables
    } else if words <= 1 {
        Region::MediumWords
    } else if words <= 4 {
        Region::UpperSentences
    } else {
        Region::AssociativeMeaning
    }
}

fn region_bounds(region: Region, nodes: usize) -> (usize, usize) {
    let (start, end) = match region {
        Region::FineLetters => (0.00_f32, 0.20_f32),
        Region::LocalSyllables => (0.20, 0.40),
        Region::MediumWords => (0.40, 0.65),
        Region::UpperSentences => (0.65, 0.85),
        Region::AssociativeMeaning => (0.85, 1.00),
    };
    let start_idx = (nodes as f32 * start).floor() as usize;
    let end_idx = (nodes as f32 * end).ceil() as usize;
    (
        start_idx.min(nodes.saturating_sub(1)),
        end_idx.saturating_sub(start_idx).max(1),
    )
}

fn normalize_text(text: &str) -> String {
    text.to_lowercase()
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

fn overlap_ratio(left: &[usize], right: &[usize]) -> f32 {
    let hits = left.iter().filter(|idx| right.contains(idx)).count();
    hits as f32 / right.len().max(1) as f32
}

fn fractal_mesh_config() -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: agent_count(),
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    }
}

fn agent_count() -> usize {
    env::var("SNGA_AGENT_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_AGENT_COUNT)
        .max(DEFAULT_AGENT_COUNT)
}

fn pattern_nodes(nodes: usize) -> usize {
    env::var("SNGA_PATTERN_NODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PATTERN_NODES)
        .min(nodes)
        .max(1)
}

fn config() -> SimplicialConfig {
    SimplicialConfig {
        width: 72,
        height: 40,
        spacing: 6.5,
        elasticity: 0.005,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00012,
        max_active_agents: 384,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 2048,
        replay_interval: 8,
        replay_batch: 12,
        replay_learning_rate: 0.05,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.00008,
        hyperbolic_curvature: 0.0,
        seed: 401,
    }
}
