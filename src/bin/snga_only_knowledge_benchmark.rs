use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};

const STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const PROGRESS_PATH: &str = "data/snga_scaled_gemma_language.progress";
const PATTERN_SIZE: usize = 12;

struct EvalStats {
    total: usize,
    nonzero: usize,
    mean_confidence: f32,
}

fn main() {
    let mut learned = SimplicialNetwork::grid_3d(config(), 2);
    let baseline = SimplicialNetwork::grid_3d(config(), 2);

    println!("SNGA-only knowledge benchmark");
    match learned.load_persistent_state(STATE_PATH) {
        Ok(report) => println!(
            "loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("loaded=false error={err}");
            println!("lectura: no hay sustrato escalado guardado");
            return;
        }
    }
    if let Ok(progress) = fs::read_to_string(PROGRESS_PATH) {
        println!("progress:\n{}", progress.trim());
    }

    let topics = [
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
        "biologia",
        "salud",
        "matematicas",
        "historia",
        "geografia",
        "tecnologia",
        "vida cotidiana",
    ];
    let questions = [
        "que es una palabra",
        "como se agrupan ideas",
        "que hace una causa",
        "para que sirve un plan",
        "como cambia un objeto",
        "que es una emocion",
        "que es la gravedad",
        "que es una celula",
        "por que beber agua",
        "que es una proporcion",
        "por que fue importante la agricultura",
        "que hace un rio",
        "para que sirve un sensor",
        "por que cooperan las personas",
        "para que sirve el miedo",
        "que es una herramienta",
    ];
    let relations = [
        "simbolo-significado",
        "rasgo-categoria",
        "causa-efecto",
        "objetivo-ruta",
        "pregunta-respuesta",
        "masa-atraccion",
        "vida-estructura",
        "hidratacion-cuerpo",
        "cantidad-relacion",
        "alimento-sociedad",
        "agua-territorio",
        "entorno-medicion",
        "grupo-objetivo",
        "amenaza-respuesta",
        "accion-meta",
    ];
    let paraphrases = [
        ("para que sirve una palabra", "palabra"),
        ("como se forma una idea", "concepto"),
        ("que pasa despues de una causa", "causalidad"),
        ("como se guarda un recuerdo", "memoria"),
        ("como organizar pasos", "planificacion"),
        ("como interpretar una escena", "mundo"),
        ("como medir el entorno", "tecnologia"),
        ("como evitar peligro", "emocion"),
    ];

    let learned_topics = eval_prefix(&learned, "topic", &topics);
    let baseline_topics = eval_prefix(&baseline, "topic", &topics);
    let learned_questions = eval_prefix(&learned, "question", &questions);
    let baseline_questions = eval_prefix(&baseline, "question", &questions);
    let learned_relations = eval_prefix(&learned, "relation", &relations);
    let baseline_relations = eval_prefix(&baseline, "relation", &relations);
    let transitive = eval_transitive(&learned, &questions, &relations);
    let intuitive = eval_paraphrase_intuition(&learned, &paraphrases);

    print_stats("topics", &learned_topics, &baseline_topics);
    print_stats("questions", &learned_questions, &baseline_questions);
    print_stats("relations", &learned_relations, &baseline_relations);
    println!(
        "transitive_reasoning: hits={}/{} mean_conf={:.3}",
        transitive.nonzero, transitive.total, transitive.mean_confidence
    );
    println!(
        "paraphrase_intuition: hits={}/{} mean_conf={:.3}",
        intuitive.nonzero, intuitive.total, intuitive.mean_confidence
    );

    let stats = learned.plasticity_stats();
    println!(
        "network: active_edges={} associative_edges={} consolidated_edges={} causal_edges={} energy={:.1}",
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
            && learned_relations.nonzero > baseline_relations.nonzero
            && transitive.nonzero > 0
        {
            "SNGA conserva conocimiento linguistico y muestra razonamiento transitivo local sin LLM"
        } else {
            "SNGA aprendio señales, pero faltan mas lotes para razonamiento robusto"
        }
    );
}

fn eval_prefix(network: &SimplicialNetwork, prefix: &str, prompts: &[&str]) -> EvalStats {
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for prompt in prompts {
        let p = pattern(prefix, prompt, network.agents.len());
        let predicted = network.predict_next_pattern(&p, 1, 32);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        confidence += predicted.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;
    }
    EvalStats {
        total: prompts.len(),
        nonzero,
        mean_confidence: confidence / prompts.len().max(1) as f32,
    }
}

fn eval_transitive(
    network: &SimplicialNetwork,
    questions: &[&str],
    relations: &[&str],
) -> EvalStats {
    let mut hits = 0;
    let mut confidence = 0.0;
    let total = questions.len().min(relations.len());
    for idx in 0..total {
        let q = pattern("question", questions[idx], network.agents.len());
        let rel = pattern("relation", relations[idx], network.agents.len());
        let predicted = network.infer_transitive_from(&q, 2, 96);
        let predicted_ids = predicted.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let overlap = overlap_count(&predicted_ids, &rel);
        if overlap > 0 {
            hits += 1;
        }
        confidence += overlap as f32 / rel.len().max(1) as f32;
    }
    EvalStats {
        total,
        nonzero: hits,
        mean_confidence: confidence / total.max(1) as f32,
    }
}

fn eval_paraphrase_intuition(
    network: &SimplicialNetwork,
    paraphrases: &[(&str, &str)],
) -> EvalStats {
    let mut hits = 0;
    let mut confidence = 0.0;
    for (prompt, expected_topic) in paraphrases {
        let q = pattern("question", prompt, network.agents.len());
        let expected = pattern("topic", expected_topic, network.agents.len());
        let predicted = network.infer_transitive_from(&q, 2, 96);
        let predicted_ids = predicted.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let overlap = overlap_count(&predicted_ids, &expected);
        if overlap > 0 {
            hits += 1;
        }
        confidence += overlap as f32 / expected.len().max(1) as f32;
    }
    EvalStats {
        total: paraphrases.len(),
        nonzero: hits,
        mean_confidence: confidence / paraphrases.len().max(1) as f32,
    }
}

fn print_stats(label: &str, learned: &EvalStats, baseline: &EvalStats) {
    println!(
        "{}: learned={}/{} conf={:.3} baseline={}/{} conf={:.3}",
        label,
        learned.nonzero,
        learned.total,
        learned.mean_confidence,
        baseline.nonzero,
        baseline.total,
        baseline.mean_confidence
    );
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

fn overlap_count(left: &[usize], right: &[usize]) -> usize {
    let right = right
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    left.iter().filter(|idx| right.contains(idx)).count()
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
