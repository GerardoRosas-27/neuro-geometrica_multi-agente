use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::ConceptProjection;
use std::env;
use std::hash::{Hash, Hasher};

const DEFAULT_OUTPUT: &str = "data/cdt_rqm_blank_gemma_boundary_concept.cdt_rqm";
const DEFAULT_NODES_PER_SLICE: usize = 512;
const OBSERVER_BOUNDARY: ObserverId = ObserverId(42_001);
const OBSERVER_CONCEPT: ObserverId = ObserverId(42_002);

const DEFAULT_QUESTION: &str = "¿Cómo codificas un concepto dentro del CDT-RQM para que sea independiente del lenguaje y, al mismo tiempo, pueda ser consultado eficientemente por el periférico lingüístico?";
const FALLBACK_TEACHING: &str = "Un concepto en CDT-RQM se codifica como una variedad simplicial interna estable, no como una palabra. El lenguaje solo opera en la frontera: una sonda lingüística activa nodos de borde, RQM colapsa el atractor geométrico relativo al observador, y el patrón resultante se proyecta de regreso a la frontera para que Gemma lo verbalice. Así el conocimiento queda en el sustrato y el idioma queda como periférico.";

fn main() {
    let output =
        env::var("CDT_RQM_BLANK_GEMMA_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT.to_string());
    let offline = env::args().any(|arg| arg == "--offline-fallback");
    let epochs = env_usize("CDT_RQM_BLANK_EPOCHS", 24).max(1);
    let anneal_attempts = env_usize("CDT_RQM_BLANK_ANNEAL_ATTEMPTS", 24);
    let nodes_per_slice =
        env_usize("CDT_RQM_BLANK_NODES_PER_SLICE", DEFAULT_NODES_PER_SLICE).max(128);
    let question =
        env::var("CDT_RQM_TRAIN_QUESTION").unwrap_or_else(|_| DEFAULT_QUESTION.to_string());
    let teacher = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_TEACHER_MODEL")
            .or_else(|_| env::var("SNGA_GEMMA_MODEL"))
            .unwrap_or_else(|_| "gemma2:2b".to_string()),
    };
    let peripheral = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_PERIPHERAL_MODEL")
            .or_else(|_| env::var("SNGA_GEMMA_MODEL"))
            .unwrap_or_else(|_| "gemma2:2b".to_string()),
    };

    let teaching = if offline {
        FALLBACK_TEACHING.to_string()
    } else {
        ask_gemma_teacher(&teacher, &question).unwrap_or_else(|| FALLBACK_TEACHING.to_string())
    };

    let mut substrate = CdtRqmUniverseSubstrate::new(cdt_rqm_config(nodes_per_slice));
    let boundary_query = boundary_pattern("query_boundary", &question, 0);
    let concept_knot = concept_pattern("language_independent_concept", &teaching, 1);
    let response_boundary = boundary_pattern("response_boundary", &teaching, 2);

    println!("CDT-RQM blank Gemma boundary concept trainer");
    println!("output={output} epochs={epochs} anneal_attempts={anneal_attempts} nodes_per_slice={nodes_per_slice} offline={offline}");
    println!("question={question}");
    println!("teaching={teaching}");

    for epoch in 0..epochs {
        train_boundary_concept(
            &mut substrate,
            &boundary_query,
            &concept_knot,
            &response_boundary,
        );
        if epoch % 4 == 3 || epoch + 1 == epochs {
            let report = substrate.step_from_boundary(OBSERVER_BOUNDARY, 0.0, &boundary_query);
            println!(
                "epoch={} candidates={} hardware_score={:.3} active_edges={} regge={:.3}",
                epoch + 1,
                report.expected_from_rqm.len(),
                report.hardware_prediction_score,
                substrate
                    .hardware
                    .edges
                    .iter()
                    .filter(|edge| edge.active)
                    .count(),
                substrate.hardware.regge_action()
            );
        }
    }

    let validation = vec![(
        OBSERVER_BOUNDARY,
        0.0,
        boundary_query.clone(),
        concept_knot.clone(),
        response_boundary.clone(),
    )];
    let anneal = substrate.anneal_after_migration(&validation, anneal_attempts);
    let query_report = substrate.step_from_boundary(OBSERVER_BOUNDARY, 0.0, &boundary_query);
    let concept_hits = query_report
        .expected_from_rqm
        .iter()
        .filter(|idx| concept_knot.contains(idx))
        .count();
    let response = verbalize_with_peripheral(
        &peripheral,
        &question,
        &teaching,
        concept_hits,
        query_report.expected_from_rqm.len(),
        offline,
    );

    match substrate.save_consolidated_state(&output) {
        Ok(()) => {
            println!("saved=true output={output}");
            println!(
                "result: concept_hits={} concept_size={} rqm_relations={} active_edges={} regge={:.3} causality_violations={}",
                concept_hits,
                concept_knot.len(),
                substrate.relation_count(),
                substrate.hardware.edges.iter().filter(|edge| edge.active).count(),
                substrate.hardware.regge_action(),
                substrate.hardware.causality_violations()
            );
            println!(
                "anneal: accepted={} accuracy={:.1}%->{:.1}% leakage={:.1}%->{:.1}% regge={:.3}->{:.3}",
                anneal.accepted,
                anneal.initial_accuracy * 100.0,
                anneal.final_accuracy * 100.0,
                anneal.initial_leakage * 100.0,
                anneal.final_leakage * 100.0,
                anneal.initial_regge,
                anneal.final_regge
            );
            println!("periferico_gemma> {response}");
        }
        Err(err) => println!("saved=false output={} error={err}", output),
    }
}

fn train_boundary_concept(
    substrate: &mut CdtRqmUniverseSubstrate,
    boundary_query: &[usize],
    concept_knot: &[usize],
    response_boundary: &[usize],
) {
    substrate.hardware.clear_activity();
    substrate.train_observed_transition(OBSERVER_BOUNDARY, 0.0, boundary_query, concept_knot, 1.0);
    substrate.hardware.clear_activity();
    substrate.train_observed_transition(
        OBSERVER_CONCEPT,
        std::f32::consts::FRAC_PI_2,
        concept_knot,
        response_boundary,
        1.0,
    );
    reinforce_internal_knot(substrate, concept_knot);
}

fn reinforce_internal_knot(substrate: &mut CdtRqmUniverseSubstrate, concept_knot: &[usize]) {
    for window in concept_knot.windows(2) {
        substrate.software.reinforce_relation(
            OBSERVER_CONCEPT,
            window[0],
            window[1],
            std::f32::consts::FRAC_PI_2,
            1.0,
        );
    }
    for &a in concept_knot.iter().take(12) {
        for &b in concept_knot.iter().skip(1).take(12) {
            if a != b {
                substrate.software.reinforce_relation(
                    OBSERVER_CONCEPT,
                    a,
                    b,
                    std::f32::consts::FRAC_PI_2,
                    0.85,
                );
            }
        }
    }
}

fn ask_gemma_teacher(engine: &OllamaGemmaEngine, question: &str) -> Option<String> {
    let context = LinguisticContext {
        user_prompt: format!(
            "Enseña a CDT-RQM esta idea en una respuesta breve y técnica: {question}"
        ),
        inferred_intent: "maestro_gemma_para_entrenar_concepto_cdt_rqm".to_string(),
        geometric_projection: ConceptProjection {
            top_agents: Vec::new(),
        },
        memory_summary: "El sustrato CDT-RQM esta en blanco. Debes enseñar el concepto como geometria independiente del lenguaje y frontera lingüistica.".to_string(),
    };
    engine
        .generate(&context)
        .ok()
        .map(|response| clean(&response.text))
}

fn verbalize_with_peripheral(
    engine: &OllamaGemmaEngine,
    question: &str,
    teaching: &str,
    concept_hits: usize,
    candidates: usize,
    offline: bool,
) -> String {
    if offline {
        return teaching.to_string();
    }
    let context = LinguisticContext {
        user_prompt: question.to_string(),
        inferred_intent: "responder_desde_concepto_cdt_rqm_consolidado".to_string(),
        geometric_projection: ConceptProjection {
            top_agents: vec![(concept_hits, 1.0), (candidates, 0.5)],
        },
        memory_summary: format!(
            "CDT-RQM recupero {concept_hits} nodos del nudo conceptual y {candidates} candidatos de frontera. Conocimiento: {teaching}"
        ),
    };
    engine
        .generate(&context)
        .map(|response| clean(&response.text))
        .unwrap_or_else(|_| teaching.to_string())
}

fn boundary_pattern(prefix: &str, text: &str, slice: usize) -> Vec<usize> {
    pattern(prefix, text, slice, 24)
}

fn concept_pattern(prefix: &str, text: &str, slice: usize) -> Vec<usize> {
    let mut out = pattern(prefix, text, slice, 32);
    for word in normalize_text(text).split_whitespace().take(24) {
        out.extend(pattern("concept_word", word, slice, 4));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn pattern(prefix: &str, value: &str, slice: usize, size: usize) -> Vec<usize> {
    let normalized = normalize_text(value);
    let nodes_per_slice =
        env_usize("CDT_RQM_BLANK_NODES_PER_SLICE", DEFAULT_NODES_PER_SLICE).max(128);
    let mut out = (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalized.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * nodes_per_slice + (hasher.finish() as usize % nodes_per_slice)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn clean(text: &str) -> String {
    text.replace('\r', " ")
        .replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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

fn cdt_rqm_config(nodes_per_slice: usize) -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice,
            initial_spatial_connectivity: 0.006,
            initial_temporal_connectivity: 0.003,
            target_spatial_degree: 4,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 3,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 77_001,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.10,
            phase_learning_rate: 0.24,
            coherence_learning_rate: 0.13,
            uncertainty_learning_rate: 0.11,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.02,
        },
        max_quantum_candidates: 96,
        rqm_feedback_gain: 0.42,
    }
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}
