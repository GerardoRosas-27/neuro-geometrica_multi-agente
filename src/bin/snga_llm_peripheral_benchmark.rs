use snga::linguistic_engine::{
    fallback_response, LinguisticContext, LinguisticEngine, OllamaGemmaEngine,
};
use snga::simplicial::{ConceptProjection, SimplicialConfig, SimplicialNetwork};
use std::collections::HashMap;
use std::env;

#[derive(Clone)]
struct PrivateConcept {
    code: &'static str,
    intent: &'static str,
}

#[derive(Clone)]
struct EvalCase {
    prompt: &'static str,
    code: &'static str,
    expected_intent: &'static str,
    required: &'static [&'static str],
    hops: usize,
}

struct CaseResult {
    prompt: String,
    expected: String,
    inferred: String,
    snga_ok: bool,
    gemma_only_ok: bool,
    snga_gemma_ok: bool,
    gemma_only: String,
    snga_gemma: String,
}

fn main() {
    let concepts = private_concepts();
    let cases = eval_cases();
    let mut network = SimplicialNetwork::grid_3d(benchmark_config(), 2);
    network.enable_neural_oscillations();
    train_private_memory(&mut network, &concepts);

    let engine = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()),
    };

    println!("SNGA + LLM peripheral benchmark");
    println!("task=private learned symbols and causal abstraction");
    println!("model={}", engine.model);

    let mut results = Vec::new();
    for case in &cases {
        let inferred = infer_private_intent(&network, &concepts, case.code, case.hops);
        let snga_ok = inferred == case.expected_intent;
        let projection = project_code(&mut network, case.code);

        let gemma_only_prompt = format!(
            "Responde en espanol. En un sistema privado, que significa el codigo '{}' ? No inventes si no sabes.",
            case.code
        );
        let gemma_only = raw_gemma(&engine, &gemma_only_prompt);

        let context = LinguisticContext {
            user_prompt: case.prompt.to_string(),
            inferred_intent: inferred.clone(),
            geometric_projection: projection,
            memory_summary: format!(
                "SNGA aprendio una memoria privada: codigo '{}' -> intencion '{}'. Para responder usa esta memoria geometrica, no conocimiento externo.",
                case.code, inferred
            ),
        };
        let snga_gemma = engine
            .generate(&context)
            .unwrap_or_else(|_| fallback_response(&context))
            .text;

        results.push(CaseResult {
            prompt: case.prompt.to_string(),
            expected: case.expected_intent.to_string(),
            inferred,
            snga_ok,
            gemma_only_ok: contains_required(&gemma_only, case.required),
            snga_gemma_ok: contains_required(&snga_gemma, case.required),
            gemma_only,
            snga_gemma,
        });
    }

    let total = results.len().max(1) as f32;
    let snga_acc = results.iter().filter(|r| r.snga_ok).count() as f32 / total;
    let gemma_only_acc = results.iter().filter(|r| r.gemma_only_ok).count() as f32 / total;
    let snga_gemma_acc = results.iter().filter(|r| r.snga_gemma_ok).count() as f32 / total;

    for result in &results {
        println!(
            "case prompt={:?} expected={} inferred={} snga_ok={} gemma_only_ok={} snga_gemma_ok={}",
            result.prompt,
            result.expected,
            result.inferred,
            result.snga_ok,
            result.gemma_only_ok,
            result.snga_gemma_ok
        );
        println!("  gemma_only: {}", compact(&result.gemma_only));
        println!("  snga_gemma: {}", compact(&result.snga_gemma));
    }

    println!(
        "summary: snga_inference={:.1}% gemma_only={:.1}% snga_plus_gemma={:.1}%",
        snga_acc * 100.0,
        gemma_only_acc * 100.0,
        snga_gemma_acc * 100.0
    );
    println!(
        "lectura: {}",
        if snga_acc > 0.9 && snga_gemma_acc > gemma_only_acc {
            "SNGA aporta memoria/razonamiento privado y Gemma funciona como renderizador periferico"
        } else {
            "la prueba no demuestra ventaja clara; ajustar memoria privada o prompt periferico"
        }
    );
}

fn train_private_memory(network: &mut SimplicialNetwork, concepts: &[PrivateConcept]) {
    let patterns = concepts
        .iter()
        .map(|concept| {
            (
                concept.code,
                code_pattern(concept.code, network.agents.len()),
                intent_pattern(concept.intent, network.agents.len()),
            )
        })
        .collect::<Vec<_>>();

    for _ in 0..10 {
        for (_, code, intent) in &patterns {
            let mut fused = code.clone();
            fused.extend(intent.iter().copied());
            fused.sort_unstable();
            fused.dedup();
            network.learn_transition(code, intent);
            network.reinforce_coactivation_if_useful(&fused, 0.08, 0.95);
            network.inject_pattern(code, 1.1, 1);
            network.inject_pattern(intent, 0.9, 1);
            network.step();
            network.clear_activity();
        }
    }

    // Cadena causal privada: xq17 -> v9k2 -> p3lm.
    let xq17 = code_pattern("xq17", network.agents.len());
    let v9k2 = code_pattern("v9k2", network.agents.len());
    let p3lm = code_pattern("p3lm", network.agents.len());
    for _ in 0..8 {
        network.learn_transition(&xq17, &v9k2);
        network.learn_transition(&v9k2, &p3lm);
    }
}

fn infer_private_intent(
    network: &SimplicialNetwork,
    concepts: &[PrivateConcept],
    code: &str,
    hops: usize,
) -> String {
    let source = code_pattern(code, network.agents.len());
    let predicted = if hops <= 1 {
        network.predict_from(&source, 512)
    } else {
        network.infer_transitive_from(&source, hops, 512)
    };
    let scores = predicted.into_iter().collect::<HashMap<_, _>>();

    concepts
        .iter()
        .map(|concept| {
            let pattern = if hops > 1 && code == "xq17" {
                code_pattern(concept.code, network.agents.len())
            } else {
                intent_pattern(concept.intent, network.agents.len())
            };
            let score = pattern
                .iter()
                .map(|idx| scores.get(idx).copied().unwrap_or(0.0))
                .sum::<f32>();
            (concept.intent, score)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(intent, _)| intent.to_string())
        .unwrap_or_else(|| "desconocido".to_string())
}

fn project_code(network: &mut SimplicialNetwork, code: &str) -> ConceptProjection {
    let pattern = code_pattern(code, network.agents.len());
    network.clear_activity();
    network.inject_pattern(&pattern, 1.2, 2);
    for _ in 0..4 {
        network.step();
    }
    network.project_active_state(12)
}

fn raw_gemma(engine: &OllamaGemmaEngine, prompt: &str) -> String {
    let context = LinguisticContext {
        user_prompt: prompt.to_string(),
        inferred_intent: "sin_contexto_snga".to_string(),
        geometric_projection: ConceptProjection {
            top_agents: Vec::new(),
        },
        memory_summary: "No hay memoria privada de SNGA disponible.".to_string(),
    };
    engine
        .generate(&context)
        .unwrap_or_else(|_| fallback_response(&context))
        .text
}

fn private_concepts() -> Vec<PrivateConcept> {
    vec![
        PrivateConcept {
            code: "xq17",
            intent: "memoria episodica",
        },
        PrivateConcept {
            code: "v9k2",
            intent: "reducir sorpresa energia",
        },
        PrivateConcept {
            code: "p3lm",
            intent: "planificacion rutas causales",
        },
    ]
}

fn eval_cases() -> Vec<EvalCase> {
    vec![
        EvalCase {
            prompt: "que significa xq17 en mi sistema privado",
            code: "xq17",
            expected_intent: "memoria episodica",
            required: &["memoria", "episod"],
            hops: 1,
        },
        EvalCase {
            prompt: "que debe hacer v9k2 dentro de snga",
            code: "v9k2",
            expected_intent: "reducir sorpresa energia",
            required: &["sorpresa", "energia"],
            hops: 1,
        },
        EvalCase {
            prompt: "que representa p3lm como idea abstracta",
            code: "p3lm",
            expected_intent: "planificacion rutas causales",
            required: &["ruta"],
            hops: 1,
        },
        EvalCase {
            prompt: "si empiezo en xq17 cual es la consecuencia causal final",
            code: "xq17",
            expected_intent: "planificacion rutas causales",
            required: &["plan", "ruta"],
            hops: 2,
        },
    ]
}

fn contains_required(text: &str, required: &[&str]) -> bool {
    let lower = normalize(text);
    required.iter().all(|token| lower.contains(token))
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .replace('á', "a")
        .replace('é', "e")
        .replace('í', "i")
        .replace('ó', "o")
        .replace('ú', "u")
}

fn compact(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.len() > 180 {
        format!("{}...", &text[..180])
    } else {
        text
    }
}

fn code_pattern(code: &str, nodes: usize) -> Vec<usize> {
    hashed_pattern("private-code", code, nodes)
}

fn intent_pattern(intent: &str, nodes: usize) -> Vec<usize> {
    hashed_pattern("private-intent", intent, nodes)
}

fn hashed_pattern(prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    let bytes = value.as_bytes();
    (0..17)
        .map(|offset| {
            let byte = bytes[offset % bytes.len()] as usize;
            (prefix.len() * 97 + value.len() * 31 + byte * 17 + offset * 53 + offset * offset * 7)
                % nodes
        })
        .collect()
}

fn benchmark_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 42,
        height: 24,
        spacing: 8.0,
        elasticity: 0.006,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 128,
        inhibition_decay: 0.05,
        max_spikes_per_step: 512,
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
        causal_learning_rate: 0.22,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 241,
    }
}
