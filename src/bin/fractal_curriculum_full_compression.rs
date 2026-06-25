use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};

const TRAINED_STATE_PATH: &str = "data/snga_fractal_gemma_spanish_curriculum.snga";
const COMPRESSED_STATE_PATH: &str = "data/snga_fractal_gemma_spanish_curriculum_compressed.snga";
const AGENT_COUNT: usize = 5_760;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;
const TOP_K: usize = 32;
const TARGET_ASSOCIATIVE: usize = 500_000;
const CAUSAL_ATTEMPTS: usize = 8;

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
    input: &'static str,
}

fn main() {
    println!("SNGA fractal curriculum full compression");
    let input_state =
        env::var("SNGA_COMPRESS_INPUT").unwrap_or_else(|_| TRAINED_STATE_PATH.to_string());
    let output_state =
        env::var("SNGA_COMPRESS_OUTPUT").unwrap_or_else(|_| COMPRESSED_STATE_PATH.to_string());
    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    match network.load_persistent_state(&input_state) {
        Ok(report) => println!(
            "loaded=true path={} agents={} edges={} causal_edges={}",
            input_state, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("loaded=false error={err}");
            return;
        }
    }

    let reference = validation_signature(&network);
    let start = network.plasticity_stats();
    println!(
        "start: edges={} associative={} causal={} energy={:.1} bytes={}",
        start.active_edges,
        start.associative_edges,
        start.causal_edges,
        network.total_free_energy(),
        file_size(&input_state)
    );

    let removed_associative = prune_associative(&mut network, &reference);
    let removed_causal = prune_causal_conservative(&mut network, &reference);
    network.anneal_active_edge_rest_lengths(1.0, 0.0);

    if validation_signature(&network) != reference {
        println!("saved=false error=validation signature changed after compression");
        return;
    }

    match network.save_persistent_state(&output_state) {
        Ok(report) => println!(
            "saved=true path={} agents={} edges={} causal_edges={}",
            output_state, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("saved=false error={err}");
            return;
        }
    }

    let final_stats = network.plasticity_stats();
    println!(
        "final: knowledge_exact=true removed_associative={} removed_causal={} edges={} associative={} causal={} energy={:.1}",
        removed_associative,
        removed_causal,
        final_stats.active_edges,
        final_stats.associative_edges,
        final_stats.causal_edges,
        network.total_free_energy()
    );
    println!(
        "files: original_bytes={} compressed_bytes={} ratio={:.3}",
        file_size(&input_state),
        file_size(&output_state),
        file_size(&output_state) as f64 / file_size(&input_state).max(1) as f64
    );
}

fn prune_associative(network: &mut SimplicialNetwork, reference: &[Vec<usize>]) -> usize {
    let mut removed_total = 0;
    let mut chunk = 200_000;
    while network.plasticity_stats().associative_edges > TARGET_ASSOCIATIVE && chunk > 0 {
        let before = network.clone();
        let removed = network.prune_low_value_associative_edges(chunk);
        if removed == 0 {
            break;
        }
        if validation_signature(network) == reference {
            removed_total += removed;
            println!(
                "assoc_accepted: removed={} total={} associative={}",
                removed,
                removed_total,
                network.plasticity_stats().associative_edges
            );
        } else {
            *network = before;
            chunk /= 2;
            println!("assoc_rejected: next_chunk={chunk}");
        }
    }
    removed_total
}

fn prune_causal_conservative(network: &mut SimplicialNetwork, reference: &[Vec<usize>]) -> usize {
    let mut removed_total = 0;
    let mut chunk = 80_000;
    let mut attempts = 0;
    while chunk > 0 && attempts < CAUSAL_ATTEMPTS {
        attempts += 1;
        let before = network.clone();
        let removed = network.prune_low_value_causal_edges(chunk);
        if removed == 0 {
            break;
        }
        if validation_signature(network) == reference {
            removed_total += removed;
            println!(
                "causal_accepted: removed={} total={} causal={}",
                removed,
                removed_total,
                network.plasticity_stats().causal_edges
            );
        } else {
            *network = before;
            chunk /= 2;
            println!("causal_rejected: next_chunk={chunk}");
        }
    }
    removed_total
}

fn validation_signature(network: &SimplicialNetwork) -> Vec<Vec<usize>> {
    let mut signatures = Vec::new();
    for case in validation_cases() {
        for regional in [true, false] {
            let input = if regional {
                hierarchical_text_pattern("input", case.input, network.agents.len())
            } else {
                legacy_hierarchical_text_pattern("input", case.input, network.agents.len())
            };
            signatures.push(
                network
                    .predict_next_pattern(&input, 1, TOP_K)
                    .into_iter()
                    .map(|(idx, _)| idx)
                    .collect::<Vec<_>>(),
            );
        }
    }
    signatures
}

fn validation_cases() -> Vec<Case> {
    vec![
        Case { input: "a" },
        Case { input: "m" },
        Case { input: "m a" },
        Case { input: "p a" },
        Case { input: "c a s a" },
        Case { input: "n i n o" },
        Case {
            input: "nino corre",
        },
        Case { input: "come pan" },
        Case {
            input: "el nino come pan",
        },
        Case {
            input: "que es una palabra",
        },
        Case {
            input: "los ninos comen",
        },
        Case {
            input: "llueve entonces el suelo se moja",
        },
    ]
}

fn hierarchical_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    let mut out = legacy_hierarchical_text_pattern(prefix, text, nodes);
    let normalized = normalize_text(text);
    for (pos, ch) in normalized.chars().enumerate().take(24) {
        out.extend(regional_pattern(
            "letter",
            &format!("{ch}_{pos}"),
            LETTER_PATTERN_SIZE,
            nodes,
            Region::FineLetters,
        ));
    }
    out.extend(regional_pattern(
        prefix,
        &normalized,
        PATTERN_SIZE,
        nodes,
        text_region(&normalized),
    ));
    for word in normalized.split_whitespace().take(12) {
        out.extend(regional_pattern(
            "word",
            word,
            PATTERN_SIZE,
            nodes,
            Region::MediumWords,
        ));
    }
    for pair in normalized.split_whitespace().collect::<Vec<_>>().windows(2) {
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

fn pattern(prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
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

fn file_size(path: &str) -> u64 {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

fn fractal_mesh_config() -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: AGENT_COUNT,
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    }
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
