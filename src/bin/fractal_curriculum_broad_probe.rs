use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::hash::{Hash, Hasher};

const STATE_PATH: &str = "data/snga_fractal_gemma_spanish_curriculum_max_compressed.snga";
const BASELINE_PATH: &str = "data/snga_scaled_gemma_language_fractal_compressed.snga";
const AGENT_COUNT: usize = 5_760;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;
const TOP_K: usize = 64;

#[derive(Clone, Copy)]
enum Region {
    FineLetters,
    LocalSyllables,
    MediumWords,
    UpperSentences,
    AssociativeMeaning,
}

#[derive(Clone, Copy)]
struct ProbeCase {
    group: &'static str,
    input: &'static str,
    expected: &'static str,
}

#[derive(Default, Clone, Copy)]
struct GroupStats {
    cases: usize,
    nonzero: usize,
    hits: usize,
    confidence: f32,
    overlap: f32,
}

fn main() {
    println!("SNGA broad linguistic and reasoning probe");

    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    match network.load_persistent_state(STATE_PATH) {
        Ok(report) => println!(
            "trained_loaded=true agents={} edges={} causal_edges={} state={}",
            report.agents, report.edges, report.causal_edges, STATE_PATH
        ),
        Err(err) => {
            println!("trained_loaded=false error={err}");
            return;
        }
    }

    let mut baseline = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    let _ = baseline.load_persistent_state(BASELINE_PATH);

    let cases = probe_cases();
    for group in [
        "letras",
        "silabas",
        "palabras_significado",
        "frases",
        "oraciones",
        "gramatica",
        "semantica_media",
    ] {
        let trained = eval_group(&network, &cases, group);
        let base = eval_group(&baseline, &cases, group);
        print_group("trained", group, trained);
        print_group("baseline", group, base);
    }

    controlled_family_reasoning(&network);
    print_network("trained", &network);
}

fn eval_group(network: &SimplicialNetwork, cases: &[ProbeCase], group: &str) -> GroupStats {
    let mut stats = GroupStats::default();
    for case in cases.iter().filter(|case| case.group == group) {
        let input = hierarchical_text_pattern("input", case.input, network.agents.len());
        let expected = hierarchical_text_pattern("target", case.expected, network.agents.len());
        let predicted = network.predict_next_pattern(&input, 1, TOP_K);
        let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let overlap = overlap_ratio(&predicted_ids, &expected);
        let confidence = predicted.iter().map(|(_, score)| *score).sum::<f32>() / TOP_K as f32;

        stats.cases += 1;
        stats.nonzero += usize::from(!predicted.is_empty());
        stats.hits += usize::from(overlap > 0.0);
        stats.confidence += confidence;
        stats.overlap += overlap;

        println!(
            "case group={} input={:?} expected={:?} predicted={} conf={:.3} overlap={:.1}%",
            group,
            case.input,
            case.expected,
            predicted.len(),
            confidence,
            overlap * 100.0
        );
    }

    if stats.cases > 0 {
        stats.confidence /= stats.cases as f32;
        stats.overlap /= stats.cases as f32;
    }
    stats
}

fn controlled_family_reasoning(base: &SimplicialNetwork) {
    let mut net = base.clone();
    let juan = concept_pattern("persona", "juan", net.agents.len());
    let ana = concept_pattern("persona", "ana", net.agents.len());
    let luis = concept_pattern("persona", "luis", net.agents.len());
    let padre = concept_pattern("relacion", "padre", net.agents.len());
    let madre = concept_pattern("relacion", "madre", net.agents.len());
    let abuelo = concept_pattern("relacion", "abuelo", net.agents.len());

    let juan_padre_ana = fact_pattern("juan", "padre", "ana", net.agents.len());
    let ana_madre_luis = fact_pattern("ana", "madre", "luis", net.agents.len());
    let juan_abuelo_luis = fact_pattern("juan", "abuelo", "luis", net.agents.len());

    net.learn_transition(&juan, &juan_padre_ana);
    net.learn_transition(&juan_padre_ana, &ana);
    net.learn_transition(&ana, &ana_madre_luis);
    net.learn_transition(&ana_madre_luis, &luis);
    net.learn_transition(&padre, &abuelo);
    net.learn_transition(&madre, &abuelo);
    net.learn_transition(&juan_padre_ana, &juan_abuelo_luis);
    net.learn_transition(&ana_madre_luis, &juan_abuelo_luis);

    let path_prediction = net.infer_transitive_from(&juan, 4, TOP_K);
    let conclusion_prediction = net.predict_next_pattern(&juan_padre_ana, 1, TOP_K);
    let path_ids = path_prediction
        .iter()
        .map(|(idx, _)| *idx)
        .collect::<Vec<_>>();
    let conclusion_ids = conclusion_prediction
        .iter()
        .map(|(idx, _)| *idx)
        .collect::<Vec<_>>();

    println!(
        "reasoning_family_path: juan -> ana -> luis overlap_luis={:.1}% predicted={}",
        overlap_ratio(&path_ids, &luis) * 100.0,
        path_prediction.len()
    );
    println!(
        "reasoning_family_conclusion: juan_padre_ana + ana_madre_luis => juan_abuelo_luis overlap={:.1}% predicted={}",
        overlap_ratio(&conclusion_ids, &juan_abuelo_luis) * 100.0,
        conclusion_prediction.len()
    );
}

fn print_group(label: &str, group: &str, stats: GroupStats) {
    println!(
        "{label}_{group}: nonzero={}/{} hits={}/{} conf={:.3} overlap={:.1}%",
        stats.nonzero,
        stats.cases,
        stats.hits,
        stats.cases,
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

fn probe_cases() -> Vec<ProbeCase> {
    vec![
        ProbeCase {
            group: "letras",
            input: "a",
            expected: "vocal abierta",
        },
        ProbeCase {
            group: "letras",
            input: "m",
            expected: "consonante nasal",
        },
        ProbeCase {
            group: "silabas",
            input: "m a",
            expected: "ma",
        },
        ProbeCase {
            group: "silabas",
            input: "p a",
            expected: "pa",
        },
        ProbeCase {
            group: "palabras_significado",
            input: "c a s a",
            expected: "casa lugar para vivir",
        },
        ProbeCase {
            group: "palabras_significado",
            input: "n i n o",
            expected: "nino persona pequena",
        },
        ProbeCase {
            group: "palabras_significado",
            input: "p a n",
            expected: "pan alimento",
        },
        ProbeCase {
            group: "frases",
            input: "nino corre",
            expected: "sujeto y verbo",
        },
        ProbeCase {
            group: "frases",
            input: "come pan",
            expected: "verbo y objeto",
        },
        ProbeCase {
            group: "oraciones",
            input: "el nino come pan",
            expected: "sujeto verbo objeto",
        },
        ProbeCase {
            group: "oraciones",
            input: "que es una palabra",
            expected: "pregunta por definicion",
        },
        ProbeCase {
            group: "gramatica",
            input: "los ninos comen",
            expected: "varios sujetos",
        },
        ProbeCase {
            group: "gramatica",
            input: "el perro no ladra",
            expected: "accion negada",
        },
        ProbeCase {
            group: "semantica_media",
            input: "llueve entonces el suelo se moja",
            expected: "causa y efecto",
        },
        ProbeCase {
            group: "semantica_media",
            input: "si estudias aprendes",
            expected: "condicion y resultado",
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

fn concept_pattern(kind: &str, value: &str, nodes: usize) -> Vec<usize> {
    regional_pattern(kind, value, PATTERN_SIZE, nodes, Region::AssociativeMeaning)
}

fn fact_pattern(left: &str, relation: &str, right: &str, nodes: usize) -> Vec<usize> {
    regional_pattern(
        "fact",
        &format!("{left}_{relation}_{right}"),
        PATTERN_SIZE,
        nodes,
        Region::AssociativeMeaning,
    )
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
