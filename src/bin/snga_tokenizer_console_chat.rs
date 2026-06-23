use snga::simplicial::{ConceptProjection, SimplicialConfig, SimplicialNetwork};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

const SCALED_STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const DISTILLED_STATE_PATH: &str = "data/snga_gemma_distilled_language.snga";
const SAVE_EVERY_INTERACTIONS: usize = 8;
const SAVE_EVERY_SECONDS: u64 = 180;

#[derive(Clone)]
struct Tokenizer {
    token_to_id: HashMap<String, usize>,
}

impl Tokenizer {
    fn from_sentences(sentences: &[&str]) -> Self {
        let mut vocab = BTreeSet::new();
        vocab.insert("<unk>".to_string());
        for sentence in sentences {
            for token in tokenize(sentence) {
                vocab.insert(token);
            }
        }
        let id_to_token = vocab.into_iter().collect::<Vec<_>>();
        let token_to_id = id_to_token
            .iter()
            .enumerate()
            .map(|(idx, token)| (token.clone(), idx))
            .collect();
        Self { token_to_id }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        tokenize(text)
            .into_iter()
            .map(|token| self.token_to_id.get(&token).copied().unwrap_or(0))
            .collect()
    }
}

struct ResponseCandidate {
    topic: &'static str,
    response: &'static str,
}

fn main() {
    let candidates = response_candidates();
    let tokenizer = Tokenizer::from_sentences(
        &candidates
            .iter()
            .flat_map(|candidate| [candidate.topic, candidate.response])
            .collect::<Vec<_>>(),
    );

    let (state_path, mut network) = loadable_network();
    match network.load_persistent_state(state_path) {
        Ok(report) => {
            println!(
                "SNGA console chat cargado: agentes={} aristas={} causales={}",
                report.agents, report.edges, report.causal_edges
            );
        }
        Err(err) => {
            println!("No pude cargar {state_path}: {err}");
            println!("Ejecuta primero la destilacion o revisa que el archivo exista.");
            return;
        }
    }
    let mut dirty_interactions = 0_usize;
    let mut last_save = Instant::now();

    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("--once") {
        let prompt = args.iter().skip(1).cloned().collect::<Vec<_>>().join(" ");
        println!("usuario> {prompt}");
        println!(
            "snga> {}",
            answer(&mut network, &tokenizer, &candidates, &prompt)
        );
        save_state(&network, "guardado tras --once");
        return;
    }

    println!("Modo interactivo SNGA-only. Tokenizador + sustrato aprendido. Escribe 'salir' para guardar y terminar.");
    loop {
        print!("usuario> ");
        io::stdout().flush().ok();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            save_state(&network, "guardado al salir");
            break;
        }
        if input.is_empty() {
            continue;
        }
        println!(
            "snga> {}",
            answer(&mut network, &tokenizer, &candidates, input)
        );
        dirty_interactions += 1;
        if dirty_interactions >= SAVE_EVERY_INTERACTIONS
            || last_save.elapsed() >= Duration::from_secs(SAVE_EVERY_SECONDS)
        {
            save_state(&network, "checkpoint");
            dirty_interactions = 0;
            last_save = Instant::now();
        }
    }
}

fn answer(
    network: &mut SimplicialNetwork,
    tokenizer: &Tokenizer,
    candidates: &[ResponseCandidate],
    prompt: &str,
) -> String {
    let topic = infer_topic(prompt);
    let prompt_pattern = prompt_pattern(prompt, tokenizer, network.agents.len());
    let topic_pattern = text_pattern(topic, network.agents.len());
    let mut query = prompt_pattern.clone();
    query.extend(topic_pattern.iter().copied());
    query.sort_unstable();
    query.dedup();

    network.clear_activity();
    network.inject_pattern(&query, 1.2, 2);
    for _ in 0..6 {
        network.step();
    }

    let predicted = network.predict_next_pattern(&topic_pattern, 1, 96);
    let projection = network.project_active_state(8);
    let best = score_candidates(&predicted, candidates, network.agents.len(), topic)
        .first()
        .copied()
        .unwrap_or(0);

    let symbolic_response = candidates[best].response;
    learn_interaction(network, prompt, topic, symbolic_response);

    format!(
        "{}\n  [motor=snga-tokenizer, tema={}, confianza={:.3}, activacion={}]",
        symbolic_response,
        candidates[best].topic,
        predicted.iter().map(|(_, score)| *score).sum::<f32>() / 96.0,
        compact_projection(&projection)
    )
}

fn learn_interaction(network: &mut SimplicialNetwork, prompt: &str, topic: &str, response: &str) {
    let prompt_pattern = text_pattern(prompt, network.agents.len());
    let topic_pattern = text_pattern(topic, network.agents.len());
    let response_pattern = text_pattern(response, network.agents.len());
    let mut fused = prompt_pattern.clone();
    fused.extend(topic_pattern.iter().copied());
    fused.extend(response_pattern.iter().copied());
    fused.sort_unstable();
    fused.dedup();

    network.set_attention_goal(&response_pattern);
    network.learn_transition(&prompt_pattern, &topic_pattern);
    network.learn_transition(&topic_pattern, &response_pattern);
    network.reinforce_coactivation_if_useful(&fused, 0.055, 0.9);
    for _ in 0..4 {
        network.step();
    }
    network.clear_attention_goal();
}

fn save_state(network: &SimplicialNetwork, label: &str) {
    let (state_path, _) = state_path_and_config();
    match network.save_persistent_state(state_path) {
        Ok(report) => println!(
            "[{label}] agentes={} aristas={} causales={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => eprintln!("[{label}] fallo guardando: {err}"),
    }
}

fn score_candidates(
    predicted: &[(usize, f32)],
    candidates: &[ResponseCandidate],
    nodes: usize,
    inferred_topic: &str,
) -> Vec<usize> {
    let scores = predicted.iter().copied().collect::<HashMap<_, _>>();
    let mut ranked = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let topic = text_pattern(candidate.topic, nodes);
            let response = text_pattern(candidate.response, nodes);
            let score = topic
                .iter()
                .chain(response.iter())
                .map(|agent| scores.get(agent).copied().unwrap_or(0.0))
                .sum::<f32>();
            let topic_bonus = if candidate.topic == inferred_topic {
                10_000.0
            } else {
                0.0
            };
            (idx, score + topic_bonus)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().map(|(idx, _)| idx).collect()
}

fn response_candidates() -> Vec<ResponseCandidate> {
    vec![
        ResponseCandidate {
            topic: "palabra como simbolo estable",
            response: "Una palabra activa una region estable de la malla y puede recuperar significado aprendido.",
        },
        ResponseCandidate {
            topic: "frase como secuencia de intencion",
            response: "Una frase organiza una intencion en secuencia: contexto, accion y objeto.",
        },
        ResponseCandidate {
            topic: "concepto como region geometrica",
            response: "Un concepto es una region compacta del sustrato geometrico con rutas y sinapsis reforzadas.",
        },
        ResponseCandidate {
            topic: "causa y efecto local",
            response: "Una causa predice un efecto cuando la ruta causal fue aprendida en la malla.",
        },
        ResponseCandidate {
            topic: "modelo interno del mundo",
            response: "El mundo interno simula futuros cortos usando patrones activos y predicciones de la red.",
        },
        ResponseCandidate {
            topic: "plan a varios pasos",
            response: "Un plan selecciona rutas causales hacia un objetivo y evita contradicciones energeticas.",
        },
        ResponseCandidate {
            topic: "memoria episodica de evento",
            response: "La memoria episodica guarda contexto, evento y consecuencia para reforzarlos por replay.",
        },
        ResponseCandidate {
            topic: "aprendizaje continuo sin olvidar",
            response: "El aprendizaje continuo consolida lo util, poda ruido y conserva el sustrato aprendido.",
        },
    ]
}

fn infer_topic(prompt: &str) -> &'static str {
    let lower = prompt.to_lowercase();
    if lower.contains("palabra") || lower.contains("simbolo") {
        "palabra como simbolo estable"
    } else if lower.contains("frase") || lower.contains("oracion") {
        "frase como secuencia de intencion"
    } else if lower.contains("concepto") || lower.contains("idea") {
        "concepto como region geometrica"
    } else if lower.contains("causa") || lower.contains("efecto") {
        "causa y efecto local"
    } else if lower.contains("mundo") || lower.contains("simula") {
        "modelo interno del mundo"
    } else if lower.contains("plan") || lower.contains("ruta") {
        "plan a varios pasos"
    } else if lower.contains("memoria") || lower.contains("evento") {
        "memoria episodica de evento"
    } else {
        "aprendizaje continuo sin olvidar"
    }
}

fn prompt_pattern(prompt: &str, tokenizer: &Tokenizer, nodes: usize) -> Vec<usize> {
    tokenizer
        .encode(prompt)
        .into_iter()
        .enumerate()
        .map(|(i, token_id)| (token_id * 97 + i * 53 + prompt.len() * 11) % nodes)
        .collect()
}

fn text_pattern(text: &str, nodes: usize) -> Vec<usize> {
    text.bytes()
        .enumerate()
        .map(|(i, byte)| ((byte as usize * 41) + i * 67 + text.len() * 13) % nodes)
        .collect()
}

fn tokenize(sentence: &str) -> Vec<String> {
    sentence
        .to_lowercase()
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn compact_projection(projection: &ConceptProjection) -> String {
    projection
        .top_agents
        .iter()
        .map(|(idx, value)| format!("{idx}:{value:.2}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn loadable_network() -> (&'static str, SimplicialNetwork) {
    let (state_path, config) = state_path_and_config();
    (state_path, SimplicialNetwork::grid_3d(config, 2))
}

fn state_path_and_config() -> (&'static str, SimplicialConfig) {
    if Path::new(SCALED_STATE_PATH).exists() {
        (SCALED_STATE_PATH, scaled_config())
    } else {
        (DISTILLED_STATE_PATH, distilled_config())
    }
}

fn distilled_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 40,
        height: 24,
        spacing: 9.0,
        elasticity: 0.007,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 128,
        inhibition_decay: 0.05,
        max_spikes_per_step: 384,
        local_inhibition_decay: 0.76,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 512,
        replay_interval: 8,
        replay_batch: 8,
        replay_learning_rate: 0.06,
        causal_learning_rate: 0.20,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 313,
    }
}

fn scaled_config() -> SimplicialConfig {
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
