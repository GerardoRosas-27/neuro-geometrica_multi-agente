use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::hash::{Hash, Hasher};

const DEFAULT_STATE_PATH: &str = "data/snga_fractal_semantic_executive_gemma_adapter.snga";
const DEFAULT_BASE_STATE: &str = "data/snga_fractal_semantic_executive_substrate.snga";
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

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct AdapterLesson {
    user_input: &'static str,
    linguistic_intent: &'static str,
    internal_concepts: &'static str,
    control_task: &'static str,
    response_frame: &'static str,
    verbal_intent: &'static str,
    ideal_response: &'static str,
}

#[derive(Default, Clone, Copy)]
struct EvalReport {
    total: usize,
    input_to_frame_hits: usize,
    frame_to_verbal_hits: usize,
    output_to_verification_hits: usize,
    input_to_concept_hits: usize,
    input_to_concept_wide_hits: usize,
    mean_input_frame_overlap: f32,
    mean_frame_verbal_overlap: f32,
    mean_output_verify_overlap: f32,
    mean_input_concept_overlap: f32,
    confidence: f32,
}

fn main() {
    let state_path =
        env::var("SNGA_SEMEXEC_ADAPTER_STATE").unwrap_or_else(|_| DEFAULT_STATE_PATH.to_string());
    let base_path =
        env::var("SNGA_SEMEXEC_BASE_STATE").unwrap_or_else(|_| DEFAULT_BASE_STATE.to_string());

    println!("SNGA semantic-executive Gemma adapter probe");
    println!("state={state_path}");
    println!("base={base_path}");

    let mut trained = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes()));
    match trained.load_persistent_state(&state_path) {
        Ok(report) => println!(
            "trained_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("trained_loaded=false error={err}");
            return;
        }
    }

    let mut base = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes()));
    match base.load_persistent_state(&base_path) {
        Ok(report) => println!(
            "base_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => println!("base_loaded=false error={err}"),
    }

    let trained_eval = evaluate("trained", &trained);
    let base_eval = evaluate("base", &base);
    print_summary("trained", trained_eval);
    print_summary("base", base_eval);
    print_network("trained", &trained);
    print_network("base", &base);
}

fn evaluate(label: &str, network: &SimplicialNetwork) -> EvalReport {
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

        let input_to_frame = network.infer_transitive_from(&llm_input, 7, 256);
        let input_to_concept = network.infer_transitive_from(&llm_input, 3, 128);
        let input_to_concept_wide = network.infer_transitive_from(&llm_input, 3, 4096);
        let frame_to_verbal = network.infer_transitive_from(&planner, 2, 128);
        let output_to_verification = network.predict_next_pattern(&llm_output, 1, 128);

        let input_frame_overlap = overlap_ratio(&ids(&input_to_frame), &planner);
        let input_concept_overlap = overlap_ratio(&ids(&input_to_concept), &concept);
        let input_concept_wide_overlap = overlap_ratio(&ids(&input_to_concept_wide), &concept);
        let frame_verbal_overlap = overlap_ratio(&ids(&frame_to_verbal), &verbal);
        let output_verify_overlap = overlap_ratio(&ids(&output_to_verification), &verification);

        report.input_to_frame_hits += usize::from(input_frame_overlap > 0.0);
        report.input_to_concept_hits += usize::from(input_concept_overlap > 0.0);
        report.input_to_concept_wide_hits += usize::from(input_concept_wide_overlap > 0.0);
        report.frame_to_verbal_hits += usize::from(frame_verbal_overlap > 0.0);
        report.output_to_verification_hits += usize::from(output_verify_overlap > 0.0);
        report.mean_input_frame_overlap += input_frame_overlap;
        report.mean_input_concept_overlap += input_concept_overlap;
        report.mean_frame_verbal_overlap += frame_verbal_overlap;
        report.mean_output_verify_overlap += output_verify_overlap;
        report.confidence += confidence(&input_to_frame)
            + confidence(&input_to_concept)
            + confidence(&frame_to_verbal)
            + confidence(&output_to_verification);

        println!(
            "case {label} input={:?} input_to_concept={:.1}% input_to_concept_wide={:.1}% input_to_frame={:.1}% frame_to_verbal={:.1}% output_to_verify={:.1}%",
            lesson.user_input,
            input_concept_overlap * 100.0,
            input_concept_wide_overlap * 100.0,
            input_frame_overlap * 100.0,
            frame_verbal_overlap * 100.0,
            output_verify_overlap * 100.0
        );
    }

    let total = report.total.max(1) as f32;
    report.mean_input_frame_overlap /= total;
    report.mean_input_concept_overlap /= total;
    report.mean_frame_verbal_overlap /= total;
    report.mean_output_verify_overlap /= total;
    report.confidence /= total * 4.0;
    report
}

fn print_summary(label: &str, report: EvalReport) {
    println!(
        "{label}_summary: input_to_concept_hits={}/{} input_to_concept_wide_hits={}/{} input_to_frame_hits={}/{} frame_to_verbal_hits={}/{} output_to_verification_hits={}/{} input_to_concept_overlap={:.1}% input_to_frame_overlap={:.1}% frame_to_verbal_overlap={:.1}% output_to_verify_overlap={:.1}% confidence={:.3}",
        report.input_to_concept_hits,
        report.total,
        report.input_to_concept_wide_hits,
        report.total,
        report.input_to_frame_hits,
        report.total,
        report.frame_to_verbal_hits,
        report.total,
        report.output_to_verification_hits,
        report.total,
        report.mean_input_concept_overlap * 100.0,
        report.mean_input_frame_overlap * 100.0,
        report.mean_frame_verbal_overlap * 100.0,
        report.mean_output_verify_overlap * 100.0,
        report.confidence
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

fn validation_lessons() -> Vec<AdapterLesson> {
    vec![
        AdapterLesson {
            user_input: "que es una manzana roja",
            linguistic_intent: "pregunta definicion objeto",
            internal_concepts: "manzana fruta roja redonda comestible",
            control_task: "activar concepto desde palabra y rasgos",
            response_frame: "definir manzana con rasgos visuales y categoria",
            verbal_intent: "expresar definicion breve de manzana",
            ideal_response: "Una manzana roja es una fruta comestible, redonda y dulce.",
        },
        AdapterLesson {
            user_input: "planea cena vegetariana sin carne",
            linguistic_intent: "instruccion plan con restriccion",
            internal_concepts: "vegetariano lechuga lentejas arroz excluir carne",
            control_task: "filtrar conceptos no permitidos por restriccion",
            response_frame: "proponer plato permitido y explicar exclusion",
            verbal_intent: "expresar plan vegetariano simple",
            ideal_response: "Puedo combinar lentejas con arroz y dejar fuera la carne.",
        },
        AdapterLesson {
            user_input: "banco en el parque",
            linguistic_intent: "desambiguar palabra por contexto",
            internal_concepts: "banco asiento parque no financiero",
            control_task: "seleccionar significado secundario por contexto",
            response_frame: "explicar que banco significa asiento",
            verbal_intent: "expresar desambiguacion contextual",
            ideal_response: "En ese contexto, banco significa un asiento para sentarse.",
        },
        AdapterLesson {
            user_input: "si llueve que pasa con el suelo",
            linguistic_intent: "pregunta causa efecto",
            internal_concepts: "lluvia agua suelo mojado",
            control_task: "seguir cadena causal simple",
            response_frame: "responder efecto probable de lluvia",
            verbal_intent: "expresar consecuencia causal",
            ideal_response: "Si llueve, el agua cae y el suelo probablemente se moja.",
        },
        AdapterLesson {
            user_input: "explica tu plan antes de responder",
            linguistic_intent: "pedir planificacion",
            internal_concepts: "meta pasos memoria trabajo respuesta",
            control_task: "ordenar pasos antes de verbalizar",
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
        out.extend(letter_pattern(ch, pos, nodes));
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

fn letter_pattern(ch: char, pos: usize, nodes: usize) -> Vec<usize> {
    regional_pattern(
        Region::LinguisticSlot,
        "letter",
        &format!("{ch}_{pos}"),
        LETTER_PATTERN_SIZE,
        nodes,
    )
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

fn confidence(predicted: &[(usize, f32)]) -> f32 {
    if predicted.is_empty() {
        return 0.0;
    }
    predicted.iter().map(|(_, score)| *score).sum::<f32>() / predicted.len() as f32
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
