use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::hash::{Hash, Hasher};

const DEFAULT_INPUT_STATE: &str =
    "data/snga_fractal_semantic_executive_gemma_adapter_repaired.snga";
const DEFAULT_OUTPUT_STATE: &str = "data/snga_fractal_semantic_executive_gemma_adapter_pruned.snga";
const DEFAULT_REGION_SIZE: usize = 8_192;
const REGION_COUNT: usize = 12;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Region {
    SemanticHubAtl = 0,
    ConceptBinder = 1,
    SemanticControl = 2,
    ExecutiveLogicDlpfc = 3,
    WorkingMemory = 4,
    Planner = 5,
    ControlGate = 6,
    VisualSlot = 7,
    AuditorySlot = 8,
    SomatosensorySlot = 9,
    LinguisticSlot = 10,
    EpisodicSlot = 11,
}

#[derive(Clone, Debug)]
struct AdapterLesson {
    user_input: &'static str,
    internal_concepts: &'static str,
    response_frame: &'static str,
    verbal_intent: &'static str,
    ideal_response: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct EvalReport {
    input_to_concept_hits: usize,
    input_to_concept_wide_hits: usize,
    frame_to_verbal_hits: usize,
    output_to_verification_hits: usize,
    total: usize,
}

fn main() {
    let input = arg_value("--input").unwrap_or_else(|| DEFAULT_INPUT_STATE.to_string());
    let output = arg_value("--output").unwrap_or_else(|| DEFAULT_OUTPUT_STATE.to_string());
    let rounds = arg_value("--rounds")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6);
    let causal_chunk = arg_value("--causal-chunk")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(120_000);
    let assoc_chunk = arg_value("--assoc-chunk")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(80_000);

    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes()));
    match network.load_persistent_state(&input) {
        Ok(report) => println!(
            "loaded=true path={} agents={} edges={} causal_edges={}",
            input, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("loaded=false error={err}");
            return;
        }
    }

    let mut best = evaluate(&network);
    println!("before {}", eval_line(best));

    let (ling_start, ling_len) = region_range(Region::LinguisticSlot, network.agents.len());
    let ling_end = ling_start + ling_len;
    let protected = protected_core_ranges(network.agents.len());
    let mut removed_causal = 0;
    let mut removed_assoc = 0;

    for round in 1..=rounds {
        let before = network.clone();
        let causal_removed = network.prune_low_value_causal_edges_from_range_except_targets(
            causal_chunk,
            ling_start,
            ling_end,
            &protected,
        );
        let assoc_removed =
            network.prune_low_value_associative_edges_in_range(assoc_chunk, ling_start, ling_end);
        let current = evaluate(&network);

        if current.input_to_concept_wide_hits < best.input_to_concept_wide_hits
            || current.output_to_verification_hits < best.output_to_verification_hits
            || current.frame_to_verbal_hits < best.frame_to_verbal_hits
        {
            network = before;
            println!(
                "round={} rejected causal_removed={} assoc_removed={} {}",
                round,
                causal_removed,
                assoc_removed,
                eval_line(current)
            );
            break;
        }

        removed_causal += causal_removed;
        removed_assoc += assoc_removed;
        best = current;
        println!(
            "round={} accepted causal_removed={} assoc_removed={} total_causal_removed={} total_assoc_removed={} {}",
            round,
            causal_removed,
            assoc_removed,
            removed_causal,
            removed_assoc,
            eval_line(best)
        );

        if causal_removed == 0 && assoc_removed == 0 {
            break;
        }
    }

    let adjusted = network.anneal_active_edge_rest_lengths(0.4, 1.05);
    match network.save_persistent_state(&output) {
        Ok(report) => println!(
            "saved=true path={} agents={} edges={} causal_edges={} adjusted={} {}",
            output,
            report.agents,
            report.edges,
            report.causal_edges,
            adjusted,
            eval_line(evaluate(&network))
        ),
        Err(err) => println!("saved=false error={err}"),
    }
}

fn evaluate(network: &SimplicialNetwork) -> EvalReport {
    let lessons = validation_lessons();
    let mut report = EvalReport {
        total: lessons.len(),
        ..EvalReport::default()
    };

    for lesson in lessons {
        let llm_input =
            linguistic_text_pattern("gemma_input", lesson.user_input, network.agents.len());
        let concept = regional_pattern(
            Region::SemanticHubAtl,
            "internal_concepts",
            lesson.internal_concepts,
            PATTERN_SIZE,
            network.agents.len(),
        );
        let planner = regional_pattern(
            Region::Planner,
            "response_frame",
            lesson.response_frame,
            PATTERN_SIZE,
            network.agents.len(),
        );
        let verbal =
            linguistic_text_pattern("verbal_intent", lesson.verbal_intent, network.agents.len());
        let llm_output =
            linguistic_text_pattern("gemma_output", lesson.ideal_response, network.agents.len());
        let verification = regional_pattern(
            Region::SemanticControl,
            "verification_same_state",
            &format!("{} {}", lesson.internal_concepts, lesson.response_frame),
            PATTERN_SIZE,
            network.agents.len(),
        );

        report.input_to_concept_hits += usize::from(
            overlap_ratio(
                &ids(&network.infer_transitive_from(&llm_input, 3, 128)),
                &concept,
            ) > 0.0,
        );
        report.input_to_concept_wide_hits += usize::from(
            overlap_ratio(
                &ids(&network.infer_transitive_from(&llm_input, 3, 4096)),
                &concept,
            ) > 0.0,
        );
        report.frame_to_verbal_hits += usize::from(
            overlap_ratio(
                &ids(&network.infer_transitive_from(&planner, 2, 128)),
                &verbal,
            ) > 0.0,
        );
        report.output_to_verification_hits += usize::from(
            overlap_ratio(
                &ids(&network.predict_next_pattern(&llm_output, 1, 128)),
                &verification,
            ) > 0.0,
        );
    }

    report
}

fn eval_line(report: EvalReport) -> String {
    format!(
        "input_to_concept={}/{} input_to_concept_wide={}/{} frame_to_verbal={}/{} output_to_verify={}/{}",
        report.input_to_concept_hits,
        report.total,
        report.input_to_concept_wide_hits,
        report.total,
        report.frame_to_verbal_hits,
        report.total,
        report.output_to_verification_hits,
        report.total
    )
}

fn protected_core_ranges(nodes: usize) -> Vec<(usize, usize)> {
    [
        Region::SemanticHubAtl,
        Region::ConceptBinder,
        Region::SemanticControl,
        Region::ExecutiveLogicDlpfc,
        Region::WorkingMemory,
        Region::Planner,
        Region::ControlGate,
    ]
    .iter()
    .map(|region| {
        let (start, len) = region_range(*region, nodes);
        (start, start + len)
    })
    .collect()
}

fn validation_lessons() -> Vec<AdapterLesson> {
    vec![
        AdapterLesson {
            user_input: "que es una manzana roja",
            internal_concepts: "manzana fruta roja redonda comestible",
            response_frame: "definir manzana con rasgos visuales y categoria",
            verbal_intent: "expresar definicion breve de manzana",
            ideal_response: "Una manzana roja es una fruta comestible, redonda y dulce.",
        },
        AdapterLesson {
            user_input: "planea cena vegetariana sin carne",
            internal_concepts: "vegetariano lechuga lentejas arroz excluir carne",
            response_frame: "proponer plato permitido y explicar exclusion",
            verbal_intent: "expresar plan vegetariano simple",
            ideal_response: "Puedo combinar lentejas con arroz y dejar fuera la carne.",
        },
        AdapterLesson {
            user_input: "banco en el parque",
            internal_concepts: "banco asiento parque no financiero",
            response_frame: "explicar que banco significa asiento",
            verbal_intent: "expresar desambiguacion contextual",
            ideal_response: "En ese contexto, banco significa un asiento para sentarse.",
        },
        AdapterLesson {
            user_input: "si llueve que pasa con el suelo",
            internal_concepts: "lluvia agua suelo mojado",
            response_frame: "responder efecto probable de lluvia",
            verbal_intent: "expresar consecuencia causal",
            ideal_response: "Si llueve, el agua cae y el suelo probablemente se moja.",
        },
        AdapterLesson {
            user_input: "explica tu plan antes de responder",
            internal_concepts: "meta pasos memoria trabajo respuesta",
            response_frame: "mencionar objetivo restriccion y accion",
            verbal_intent: "expresar marco de plan",
            ideal_response:
                "Primero fijo el objetivo, luego reviso restricciones y finalmente doy la accion.",
        },
    ]
}

fn linguistic_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    let mut out = regional_pattern(Region::LinguisticSlot, prefix, text, PATTERN_SIZE, nodes);
    let normalized = normalize_text(text);
    for (pos, ch) in normalized.chars().enumerate().take(32) {
        out.extend(regional_pattern(
            Region::LinguisticSlot,
            "letter",
            &format!("{ch}_{pos}"),
            LETTER_PATTERN_SIZE,
            nodes,
        ));
    }
    for word in normalized.split_whitespace().take(16) {
        out.extend(regional_pattern(
            Region::LinguisticSlot,
            "word",
            word,
            PATTERN_SIZE,
            nodes,
        ));
    }
    for pair in normalized.split_whitespace().collect::<Vec<_>>().windows(2) {
        out.extend(regional_pattern(
            Region::LinguisticSlot,
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
            PATTERN_SIZE,
            nodes,
        ));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn regional_pattern(
    region: Region,
    prefix: &str,
    value: &str,
    size: usize,
    nodes: usize,
) -> Vec<usize> {
    let (start, len) = region_range(region, nodes);
    let normalized = normalize_text(value);
    (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (region as usize).hash(&mut hasher);
            prefix.hash(&mut hasher);
            normalized.hash(&mut hasher);
            offset.hash(&mut hasher);
            start + (hasher.finish() as usize % len.max(1))
        })
        .collect()
}

fn region_range(region: Region, nodes: usize) -> (usize, usize) {
    let region_size = inferred_region_size(nodes);
    let start = region as usize * region_size;
    let end = (start + region_size).min(nodes);
    (start, end.saturating_sub(start).max(1))
}

fn inferred_region_size(nodes: usize) -> usize {
    (nodes / REGION_COUNT).max(DEFAULT_REGION_SIZE)
}

fn total_nodes() -> usize {
    DEFAULT_REGION_SIZE * REGION_COUNT
}

fn ids(predicted: &[(usize, f32)]) -> Vec<usize> {
    predicted.iter().map(|(idx, _)| *idx).collect()
}

fn overlap_ratio(left: &[usize], right: &[usize]) -> f32 {
    let hits = left.iter().filter(|idx| right.contains(idx)).count();
    hits as f32 / right.len().max(1) as f32
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

fn fractal_mesh_config(target_nodes: usize) -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 6,
        target_dimension: 2.72,
        target_nodes,
        base_radius: 0.0,
        lateral_link_weight: 0.32,
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
        activation_threshold: 0.63,
        simplex_area_weight: 0.00012,
        max_active_agents: 448,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 14,
        rhythm_amplitude: 0.045,
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
        seed: 727,
    }
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
