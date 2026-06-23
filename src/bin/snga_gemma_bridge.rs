use snga::linguistic_engine::{
    fallback_response, LinguisticContext, LinguisticEngine, OllamaGemmaEngine,
};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;

fn main() {
    let prompt = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    let prompt = if prompt.is_empty() {
        "explica como aprende la malla geometrica".to_string()
    } else {
        prompt
    };

    let mut network = SimplicialNetwork::grid_3d(language_core_config(), 2);
    train_geometric_language_core(&mut network);
    network.inject_text_pattern(&prompt);
    for _ in 0..6 {
        network.step();
    }

    let context = LinguisticContext {
        user_prompt: prompt.clone(),
        inferred_intent: infer_intent(&prompt),
        geometric_projection: network.project_active_state(12),
        memory_summary: format!(
            "energia_libre={:.3}; nodos={}; aristas={}; el nucleo SNGA conserva memoria en la geometria y usa el LLM solo para verbalizar",
            network.total_free_energy(),
            network.agents.len(),
            network.edges.len()
        ),
    };

    let model = env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string());
    let host = env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string());
    let engine = OllamaGemmaEngine { host, model };
    let response = engine.generate(&context).unwrap_or_else(|err| {
        eprintln!("Gemma/Ollama no disponible: {err}");
        eprintln!("Sugerencia: instala Ollama y ejecuta `ollama pull gemma2:2b`.");
        fallback_response(&context)
    });

    println!("SNGA + Gemma linguistic bridge");
    println!("prompt: {}", prompt);
    println!("engine: {}", response.engine);
    println!("respuesta:\n{}", response.text);
}

fn train_geometric_language_core(network: &mut SimplicialNetwork) {
    let memories = [
        "malla geometrica aprende deformando distancias",
        "memoria episodica refuerza rutas utiles",
        "oscilaciones delta consolidan episodios",
        "planificacion busca rutas causales",
        "lenguaje es periferico y traduce estados internos",
    ];

    for sentence in memories {
        let pattern = text_pattern(sentence, network.agents.len());
        network.inject_pattern(&pattern, 1.0, 1);
        network.reinforce_coactivation_if_useful(&pattern, 0.08, 0.95);
        for _ in 0..3 {
            network.step();
        }
        network.clear_activity();
    }
}

fn infer_intent(prompt: &str) -> String {
    let lower = prompt.to_lowercase();
    if lower.contains("aprende") || lower.contains("entrena") {
        "aprendizaje_geometrico".to_string()
    } else if lower.contains("memoria") || lower.contains("recuerda") {
        "memoria_episodica".to_string()
    } else if lower.contains("lenguaje") || lower.contains("palabra") {
        "renderizado_linguistico".to_string()
    } else if lower.contains("plan") || lower.contains("razona") {
        "planificacion_causal".to_string()
    } else {
        "consulta_general_snga".to_string()
    }
}

fn text_pattern(text: &str, nodes: usize) -> Vec<usize> {
    text.bytes()
        .enumerate()
        .map(|(i, byte)| ((byte as usize * 37) + i * 53 + text.len() * 11) % nodes)
        .collect()
}

fn language_core_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 36,
        height: 24,
        spacing: 10.0,
        elasticity: 0.008,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00025,
        max_active_agents: 96,
        inhibition_decay: 0.06,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.76,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 128,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.04,
        causal_learning_rate: 0.16,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 223,
    }
}
