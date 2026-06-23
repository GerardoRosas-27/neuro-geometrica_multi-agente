use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};

const STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const PROGRESS_PATH: &str = "data/snga_scaled_gemma_language.progress";
const PATTERN_SIZE: usize = 12;

fn main() {
    let mut learned = SimplicialNetwork::grid_3d(config(), 2);
    let baseline = SimplicialNetwork::grid_3d(config(), 2);

    println!("SNGA scaled Gemma probe");
    match learned.load_persistent_state(STATE_PATH) {
        Ok(report) => println!(
            "loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("loaded=false error={err}");
            println!("lectura: no hay entrenamiento escalado guardado");
            return;
        }
    }
    if let Ok(progress) = fs::read_to_string(PROGRESS_PATH) {
        println!("progress:\n{}", progress.trim());
    }

    let topic_prompts = [
        "lenguaje",
        "concepto",
        "causalidad",
        "memoria",
        "planificacion",
        "mundo",
        "herramienta",
        "emocion",
        "sociedad",
        "fisica simple",
        "objeto",
        "categoria",
    ];
    let question_prompts = [
        "que es una palabra",
        "como se agrupan ideas",
        "que hace una causa",
        "para que sirve un plan",
        "como cambia un objeto",
        "que es una emocion",
    ];
    let relation_prompts = [
        "simbolo-significado",
        "rasgo-categoria",
        "causa-efecto",
        "objetivo-ruta",
        "pregunta-respuesta",
    ];

    let learned_topics = eval(&learned, "topic", &topic_prompts);
    let baseline_topics = eval(&baseline, "topic", &topic_prompts);
    let learned_questions = eval(&learned, "question", &question_prompts);
    let baseline_questions = eval(&baseline, "question", &question_prompts);
    let learned_relations = eval(&learned, "relation", &relation_prompts);
    let baseline_relations = eval(&baseline, "relation", &relation_prompts);

    let stats = learned.plasticity_stats();
    println!(
        "topics: learned={}/{} conf={:.3} baseline={}/{} conf={:.3}",
        learned_topics.nonzero,
        topic_prompts.len(),
        learned_topics.confidence,
        baseline_topics.nonzero,
        topic_prompts.len(),
        baseline_topics.confidence
    );
    println!(
        "questions: learned={}/{} conf={:.3} baseline={}/{} conf={:.3}",
        learned_questions.nonzero,
        question_prompts.len(),
        learned_questions.confidence,
        baseline_questions.nonzero,
        question_prompts.len(),
        baseline_questions.confidence
    );
    println!(
        "relations: learned={}/{} conf={:.3} baseline={}/{} conf={:.3}",
        learned_relations.nonzero,
        relation_prompts.len(),
        learned_relations.confidence,
        baseline_relations.nonzero,
        relation_prompts.len(),
        baseline_relations.confidence
    );
    println!(
        "network: edges={} associative={} consolidated={} causal={} energy={:.1}",
        stats.active_edges,
        stats.associative_edges,
        stats.consolidated_edges,
        stats.causal_edges,
        learned.total_free_energy()
    );
    println!(
        "lectura: {}",
        if learned_topics.nonzero > baseline_topics.nonzero
            && learned_questions.nonzero > baseline_questions.nonzero
            && stats.causal_edges > 0
        {
            "hay aprendizaje del entrenamiento escalado por lotes en SNGA"
        } else {
            "el entrenamiento escalado aun necesita mas lotes o mejor generacion"
        }
    );
}

struct EvalStats {
    nonzero: usize,
    confidence: f32,
}

fn eval(network: &SimplicialNetwork, prefix: &str, prompts: &[&str]) -> EvalStats {
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for prompt in prompts {
        let p = pattern(prefix, prompt, network.agents.len());
        let predicted = network.predict_next_pattern(&p, 1, 32);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        let conf = predicted.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;
        confidence += conf;
        println!(
            "probe {}={:?} predicted={} confidence={:.3}",
            prefix,
            prompt,
            predicted.len(),
            conf
        );
    }
    EvalStats {
        nonzero,
        confidence: confidence / prompts.len().max(1) as f32,
    }
}

fn pattern(prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = DefaultHasher::new();
            prefix.hash(&mut hasher);
            value.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
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
