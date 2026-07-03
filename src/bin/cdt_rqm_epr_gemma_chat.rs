use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::entanglement::EntanglementConfig;
use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::ConceptProjection;
use std::env;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};

const DEFAULT_STATE: &str = "data/cdt_rqm_epr_small_sleep.cdt_rqm";
const DEFAULT_NODES_PER_SLICE: usize = 2048;
const MIN_CONFIDENCE: f32 = 0.08;

#[derive(Clone, Copy)]
struct Sample {
    kind: SampleKind,
    input: &'static str,
    target: &'static str,
    response: &'static str,
}

#[derive(Clone, Copy)]
enum SampleKind {
    Concept,
    Causal,
    Skill,
    Episode,
    Correlation,
}

struct Retrieval {
    sample: Sample,
    score: f32,
    hits: usize,
    candidates: usize,
}

fn main() {
    let state = env::var("CDT_RQM_EPR_CHAT_STATE").unwrap_or_else(|_| DEFAULT_STATE.to_string());
    let offline = env::args().any(|arg| arg == "--offline-fallback");
    let peripheral = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_PERIPHERAL_MODEL")
            .or_else(|_| env::var("SNGA_GEMMA_MODEL"))
            .unwrap_or_else(|_| "gemma2:2b".to_string()),
    };

    let mut substrate = CdtRqmUniverseSubstrate::new(config(DEFAULT_NODES_PER_SLICE));
    match substrate.load_consolidated_state(&state) {
        Ok(()) => println!("CDT-RQM+EPR cargado: {state}"),
        Err(err) => {
            println!("No pude cargar {state}: {err}");
            return;
        }
    }
    if substrate.entanglement.is_none() {
        substrate.enable_entanglement(epr_config());
    }
    println!(
        "modo: peripheral={} offline={} relations={} epr_links={} causality_violations={}",
        peripheral.model,
        offline,
        substrate.relation_count(),
        substrate
            .entanglement_summary()
            .map(|r| r.active_links)
            .unwrap_or(0),
        substrate.hardware.causality_violations()
    );
    println!("Escribe una pregunta. Comandos: /salir, /estado");

    loop {
        print!("\nusuario> ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let prompt = input.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "/salir" {
            break;
        }
        if prompt == "/estado" {
            print_status(&substrate);
            continue;
        }

        let canonical = canonicalize_prompt(prompt, offline);
        let retrieval = retrieve(&mut substrate, &canonical);
        match retrieval {
            Some(retrieval) if retrieval.score >= MIN_CONFIDENCE => {
                let answer = verbalize(&peripheral, prompt, &canonical, &retrieval, offline);
                println!(
                    "cdt-rqm> concepto={} score={:.3} hits={} candidates={}",
                    retrieval.sample.target, retrieval.score, retrieval.hits, retrieval.candidates
                );
                println!("gemma-periferico> {answer}");
            }
            _ => {
                println!(
                    "cdt-rqm> no_conozco: no hay ruta confiable para {:?}.",
                    prompt
                );
                println!("gemma-periferico> No invento memoria nueva: necesito que el sustrato aprenda esa relación.");
            }
        }
    }
}

fn retrieve(substrate: &mut CdtRqmUniverseSubstrate, canonical: &str) -> Option<Retrieval> {
    let mut best = None::<Retrieval>;
    for sample in samples() {
        let lexical = lexical_overlap(canonical, sample.input);
        if lexical <= 0.0 {
            continue;
        }
        let input = pattern("input", sample.input, 0);
        let target = pattern("target", sample.target, 1);
        substrate.hardware.clear_activity();
        substrate.hardware.inject_pattern(&input, 1.0);
        let report = substrate.step_from_boundary(
            observer(sample.kind, sample.input),
            phase_for_kind(sample.kind),
            &input,
        );
        let hits = report
            .expected_from_rqm
            .iter()
            .filter(|idx| target.contains(idx))
            .count();
        let score = lexical * 0.45 + hits as f32 / target.len().max(1) as f32 * 0.55;
        let candidate = Retrieval {
            sample: *sample,
            score,
            hits,
            candidates: report.expected_from_rqm.len(),
        };
        match &best {
            Some(current) if current.score >= candidate.score => {}
            _ => best = Some(candidate),
        }
    }
    best
}

fn canonicalize_prompt(prompt: &str, offline: bool) -> String {
    if offline {
        return normalize(prompt);
    }
    normalize(prompt)
}

fn verbalize(
    engine: &OllamaGemmaEngine,
    user_prompt: &str,
    canonical: &str,
    retrieval: &Retrieval,
    offline: bool,
) -> String {
    if offline {
        return retrieval.sample.response.to_string();
    }
    let context = LinguisticContext {
        user_prompt: user_prompt.to_string(),
        inferred_intent: format!("consulta_cdt_rqm_epr:{}", retrieval.sample.target),
        geometric_projection: ConceptProjection {
            top_agents: vec![(retrieval.hits, retrieval.score)],
        },
        memory_summary: format!(
            "Entrada canonica: {canonical}. El sustrato recupero: {}. Respuesta base: {}. No agregues hechos nuevos.",
            retrieval.sample.target, retrieval.sample.response
        ),
    };
    engine
        .generate(&context)
        .map(|response| clean(&response.text))
        .unwrap_or_else(|_| retrieval.sample.response.to_string())
}

fn print_status(substrate: &CdtRqmUniverseSubstrate) {
    let epr = substrate.entanglement_summary();
    println!(
        "estado> nodes={} slices={} relations={} active_edges={} regge={:.1} temp={:.3} epr_links={} causality_violations={}",
        substrate.hardware.nodes.len(),
        substrate.hardware.config.slices,
        substrate.relation_count(),
        substrate.hardware.edges.iter().filter(|edge| edge.active).count(),
        substrate.hardware.regge_action(),
        substrate.hardware.temperature,
        epr.map(|r| r.active_links).unwrap_or(0),
        substrate.hardware.causality_violations()
    );
}

fn pattern(prefix: &str, value: &str, slice: usize) -> Vec<usize> {
    let nodes_per_slice =
        env_usize("CDT_RQM_EPR_SMALL_NODES_PER_SLICE", DEFAULT_NODES_PER_SLICE).max(256);
    let mut out = (0..16)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalize(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * nodes_per_slice + (hasher.finish() as usize % nodes_per_slice)
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

fn lexical_overlap(left: &str, right: &str) -> f32 {
    let left_words = normalize(left)
        .split_whitespace()
        .filter(|word| word.len() > 2)
        .map(str::to_string)
        .collect::<Vec<_>>();
    let right_words = normalize(right)
        .split_whitespace()
        .filter(|word| word.len() > 2)
        .map(str::to_string)
        .collect::<Vec<_>>();
    if left_words.is_empty() || right_words.is_empty() {
        return 0.0;
    }
    let hits = left_words
        .iter()
        .filter(|word| right_words.contains(word))
        .count();
    hits as f32 / left_words.len().max(right_words.len()).max(1) as f32
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

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn epr_config() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    }
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

fn samples() -> &'static [Sample] {
    &[
        Sample { kind: SampleKind::Concept, input: "perro", target: "mamifero domestico que ladra y puede ser mascota", response: "Un perro es un mamifero domestico, puede ladrar y puede ser mascota." },
        Sample { kind: SampleKind::Concept, input: "agua", target: "liquido que moja y sostiene vida", response: "El agua es un liquido que moja y sostiene la vida." },
        Sample { kind: SampleKind::Causal, input: "fuego", target: "calor", response: "En el sustrato, fuego activa una relacion causal hacia calor." },
        Sample { kind: SampleKind::Causal, input: "lluvia", target: "suelo mojado", response: "Lluvia causa suelo mojado." },
        Sample { kind: SampleKind::Causal, input: "golpear vidrio", target: "vidrio roto", response: "Golpear vidrio puede producir vidrio roto." },
        Sample { kind: SampleKind::Skill, input: "sumar", target: "leer numeros alinear cantidades combinar resultado", response: "Sumar es una habilidad secuencial: leer numeros, alinear cantidades, combinarlas y producir resultado." },
        Sample { kind: SampleKind::Skill, input: "programar", target: "definir objetivo diseñar pasos escribir codigo probar corregir", response: "Programar implica definir objetivo, diseñar pasos, escribir codigo, probar y corregir." },
        Sample { kind: SampleKind::Episode, input: "vi gato negro bajo lluvia", target: "gato negro existe y lluvia cambia entorno", response: "El episodio consolida que gatos negros existen y que la lluvia cambia el entorno." },
        Sample { kind: SampleKind::Episode, input: "comi y dejo de doler hambre", target: "comer sacia hambre", response: "El episodio consolida la relacion comer -> saciar hambre." },
        Sample { kind: SampleKind::Correlation, input: "perro humano", target: "mascota relacion social", response: "Perro y humano se correlacionan como mascota y relacion social." },
        Sample { kind: SampleKind::Correlation, input: "agua planta", target: "agua ayuda crecimiento planta", response: "Agua y planta se correlacionan porque el agua ayuda al crecimiento." },
    ]
}
