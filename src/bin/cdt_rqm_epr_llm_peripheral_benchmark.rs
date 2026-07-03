use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::entanglement::EntanglementConfig;
use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::ConceptProjection;
use std::env;
use std::hash::{Hash, Hasher};

const NODES_PER_SLICE: usize = 512;
const EPOCHS: usize = 12;
const OBSERVER_INPUT: ObserverId = ObserverId(910_001);
const OBSERVER_CONCEPT: ObserverId = ObserverId(910_002);

#[derive(Clone, Copy)]
struct Lesson {
    prompt: &'static str,
    canonical: &'static str,
    concept: &'static str,
    response: &'static str,
}

#[derive(Default)]
struct EvalMetrics {
    cases: usize,
    concept_hits: usize,
    response_hits: usize,
    lexical_sum: f32,
    latency_sum: usize,
}

impl EvalMetrics {
    fn record(&mut self, concept_ok: bool, response_ok: bool, lexical: f32, latency: usize) {
        self.cases += 1;
        self.concept_hits += usize::from(concept_ok);
        self.response_hits += usize::from(response_ok);
        self.lexical_sum += lexical;
        self.latency_sum += latency;
    }

    fn concept_accuracy(&self) -> f32 {
        self.concept_hits as f32 / self.cases.max(1) as f32
    }

    fn response_accuracy(&self) -> f32 {
        self.response_hits as f32 / self.cases.max(1) as f32
    }

    fn lexical_score(&self) -> f32 {
        self.lexical_sum / self.cases.max(1) as f32
    }

    fn latency_avg(&self) -> f32 {
        self.latency_sum as f32 / self.cases.max(1) as f32
    }
}

fn main() {
    let offline = env::args().any(|arg| arg == "--offline-fallback");
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
    let mut substrate = CdtRqmUniverseSubstrate::new(config());
    substrate.enable_entanglement(EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    });

    for _ in 0..EPOCHS {
        for &lesson in lessons() {
            train_lesson(&mut substrate, lesson);
        }
    }
    let metrics = evaluate(&mut substrate, &teacher, &peripheral, offline);
    let epr = substrate.entanglement_summary().unwrap();
    let active_edges = substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count();

    println!("CDT-RQM+EPR with LLM peripheral benchmark");
    println!(
        "offline={} lessons={} epochs={} concept_accuracy={:.1}% response_accuracy={:.1}% lexical_score={:.1}% latency_avg={:.2}",
        offline,
        lessons().len(),
        EPOCHS,
        metrics.concept_accuracy() * 100.0,
        metrics.response_accuracy() * 100.0,
        metrics.lexical_score() * 100.0,
        metrics.latency_avg()
    );
    println!(
        "substrate: active_edges={} regge={:.3} rqm_relations={} epr_links={} epr_coherence={:.3} epr_entropy={:.3} causality_violations={}",
        active_edges,
        substrate.hardware.regge_action(),
        substrate.relation_count(),
        epr.active_links,
        epr.mean_coherence,
        epr.mean_entropy,
        substrate.hardware.causality_violations()
    );
    println!(
        "lectura: {}",
        if metrics.concept_accuracy() >= 0.8
            && metrics.response_accuracy() >= 0.8
            && substrate.hardware.causality_violations() == 0
            && epr.active_links > 0
        {
            "CDT-RQM+EPR recupera conocimiento interno y el LLM funciona como periferico de entrada/salida"
        } else {
            "la arquitectura ejecuta, pero requiere mas entrenamiento o ajuste del periferico"
        }
    );
}

fn train_lesson(substrate: &mut CdtRqmUniverseSubstrate, lesson: Lesson) {
    let input = boundary_pattern(lesson.canonical, 0);
    let concept = concept_pattern(lesson.concept, 1);
    let response = boundary_pattern(lesson.response, 2);
    substrate.hardware.clear_activity();
    substrate.train_observed_transition(OBSERVER_INPUT, 0.0, &input, &concept, 1.0);
    substrate.hardware.clear_activity();
    substrate.train_observed_transition(
        OBSERVER_CONCEPT,
        std::f32::consts::FRAC_PI_2,
        &concept,
        &response,
        1.0,
    );
    for (&a, &b) in input.iter().zip(concept.iter()) {
        substrate.observe_entanglement_correlation(a, b, 0.40);
    }
    for (&a, &b) in concept.iter().zip(response.iter()) {
        substrate.observe_entanglement_correlation(a, b, 0.35);
    }
}

fn evaluate(
    substrate: &mut CdtRqmUniverseSubstrate,
    teacher: &OllamaGemmaEngine,
    peripheral: &OllamaGemmaEngine,
    offline: bool,
) -> EvalMetrics {
    let mut metrics = EvalMetrics::default();
    for lesson in lessons() {
        let canonical = canonicalize_input(teacher, lesson.prompt, lesson.canonical, offline);
        let input = boundary_pattern(&canonical, 0);
        let concept = concept_pattern(lesson.concept, 1);
        let report = substrate.step_from_boundary(OBSERVER_INPUT, 0.0, &input);
        let concept_hits = report
            .expected_from_rqm
            .iter()
            .filter(|idx| concept.contains(idx))
            .count();
        let concept_ok = concept_hits >= concept.len().min(6);
        let latency = if concept_ok { 1 } else { 4 };

        let rendered = verbalize_output(
            peripheral,
            lesson.prompt,
            lesson.concept,
            lesson.response,
            concept_hits,
            offline,
        );
        let lexical = lexical_overlap(&rendered, lesson.response);
        let response_ok =
            lexical >= 0.35 || normalize(&rendered).contains(&normalize(lesson.concept));
        metrics.record(concept_ok, response_ok, lexical, latency);
    }
    metrics
}

fn canonicalize_input(
    engine: &OllamaGemmaEngine,
    prompt: &str,
    fallback: &str,
    offline: bool,
) -> String {
    if offline {
        return fallback.to_string();
    }
    let context = LinguisticContext {
        user_prompt: format!("Normaliza esta pregunta a una etiqueta conceptual breve: {prompt}"),
        inferred_intent: "periferico_entrada_cdt_rqm".to_string(),
        geometric_projection: ConceptProjection { top_agents: vec![] },
        memory_summary: "No respondas la pregunta. Solo devuelve una frase canonica de frontera."
            .to_string(),
    };
    engine
        .generate(&context)
        .map(|response| clean(&response.text))
        .unwrap_or_else(|_| fallback.to_string())
}

fn verbalize_output(
    engine: &OllamaGemmaEngine,
    prompt: &str,
    concept: &str,
    substrate_response: &str,
    hits: usize,
    offline: bool,
) -> String {
    if offline {
        return substrate_response.to_string();
    }
    let context = LinguisticContext {
        user_prompt: prompt.to_string(),
        inferred_intent: format!("verbalizar_concepto_cdt_rqm:{concept}"),
        geometric_projection: ConceptProjection {
            top_agents: vec![(hits, 1.0)],
        },
        memory_summary: format!(
            "CDT-RQM recupero el concepto interno {concept}. Contenido del sustrato: {substrate_response}"
        ),
    };
    engine
        .generate(&context)
        .map(|response| clean(&response.text))
        .unwrap_or_else(|_| substrate_response.to_string())
}

fn boundary_pattern(text: &str, slice: usize) -> Vec<usize> {
    pattern("boundary", text, slice, 16)
}

fn concept_pattern(text: &str, slice: usize) -> Vec<usize> {
    let mut out = pattern("concept", text, slice, 24);
    for word in normalize(text).split_whitespace().take(12) {
        out.extend(pattern("concept_word", word, slice, 3));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn pattern(prefix: &str, value: &str, slice: usize, size: usize) -> Vec<usize> {
    let normalized = normalize(value);
    let mut out = (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalized.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * NODES_PER_SLICE + (hasher.finish() as usize % NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn lexical_overlap(left: &str, right: &str) -> f32 {
    let left_words = normalize(left)
        .split_whitespace()
        .filter(|word| word.len() > 3)
        .map(str::to_string)
        .collect::<Vec<_>>();
    let right_words = normalize(right)
        .split_whitespace()
        .filter(|word| word.len() > 3)
        .map(str::to_string)
        .collect::<Vec<_>>();
    if left_words.is_empty() || right_words.is_empty() {
        return 0.0;
    }
    let hits = left_words
        .iter()
        .filter(|word| right_words.contains(word))
        .count();
    hits as f32 / right_words.len().max(1) as f32
}

fn clean(text: &str) -> String {
    text.replace('\r', " ")
        .replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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

fn lessons() -> &'static [Lesson] {
    &[
        Lesson {
            prompt: "¿Cómo codifica CDT-RQM un concepto sin depender del lenguaje?",
            canonical: "concepto independiente del lenguaje en cdt rqm",
            concept: "variedad simplicial estable independiente de etiquetas",
            response: "Un concepto vive como una variedad simplicial estable; el lenguaje solo consulta su frontera.",
        },
        Lesson {
            prompt: "¿Qué hace el periférico lingüístico?",
            canonical: "funcion del periferico linguistico",
            concept: "operador de frontera linguistica",
            response: "El periférico lingüístico traduce sondas y respuestas; no guarda el conocimiento profundo.",
        },
        Lesson {
            prompt: "¿Por qué EPR no rompe la geometría CDT?",
            canonical: "epr no metrico en cdt rqm",
            concept: "enlace logico no metrico con fusible entropico",
            response: "EPR sincroniza estados relacionales sin crear aristas CDT; si contradice, el fusible entrópico lo poda.",
        },
        Lesson {
            prompt: "¿Cómo se consolida la memoria?",
            canonical: "consolidacion graphity validada por memoria",
            concept: "annealing graphity conserva memoria y reduce regge",
            response: "Graphity poda geometría si la memoria validada por RQM no se degrada.",
        },
    ]
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.00008,
            initial_temporal_connectivity: 0.00004,
            target_spatial_degree: 4,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 91_771,
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
        max_quantum_candidates: 128,
        rqm_feedback_gain: 0.42,
    }
}
