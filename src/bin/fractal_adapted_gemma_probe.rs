use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const ADAPTED_STATE_PATH: &str = "data/snga_scaled_gemma_language_fractal_adapted.snga";
const AGENT_COUNT: usize = 5_760;
const PATTERN_SIZE: usize = 12;

fn main() {
    println!("SNGA fractal adapted Gemma probe");
    println!("state={ADAPTED_STATE_PATH}");

    let fractal_config = FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: AGENT_COUNT,
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    };
    let mut adapted = SimplicialNetwork::fractal_3d(config(), fractal_config);
    let baseline = SimplicialNetwork::fractal_3d(config(), fractal_config);

    match adapted.load_persistent_state(ADAPTED_STATE_PATH) {
        Ok(report) => println!(
            "loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("loaded=false error={err}");
            return;
        }
    }

    let adapted_eval = evaluate(&adapted, "adapted");
    let baseline_eval = evaluate(&baseline, "baseline");
    print_summary("adapted", &adapted_eval, &adapted);
    print_summary("baseline", &baseline_eval, &baseline);

    println!(
        "lectura: {}",
        if adapted_eval.all_nonzero()
            && adapted_eval.average_confidence() > baseline_eval.average_confidence()
        {
            "el sustrato fractal adaptado preserva la memoria linguistica guardada"
        } else {
            "el sustrato fractal adaptado carga, pero la senal linguistica no supera al baseline"
        }
    );
}

struct EvalGroup {
    topics: EvalStats,
    questions: EvalStats,
    relations: EvalStats,
}

impl EvalGroup {
    fn all_nonzero(&self) -> bool {
        self.topics.nonzero == self.topics.total
            && self.questions.nonzero == self.questions.total
            && self.relations.nonzero == self.relations.total
    }

    fn average_confidence(&self) -> f32 {
        (self.topics.confidence + self.questions.confidence + self.relations.confidence) / 3.0
    }
}

struct EvalStats {
    total: usize,
    nonzero: usize,
    confidence: f32,
}

fn evaluate(network: &SimplicialNetwork, label: &str) -> EvalGroup {
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

    EvalGroup {
        topics: eval_prompts(network, label, "topic", &topic_prompts),
        questions: eval_prompts(network, label, "question", &question_prompts),
        relations: eval_prompts(network, label, "relation", &relation_prompts),
    }
}

fn eval_prompts(
    network: &SimplicialNetwork,
    label: &str,
    prefix: &str,
    prompts: &[&str],
) -> EvalStats {
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for prompt in prompts {
        let cue = pattern(prefix, prompt, network.agents.len());
        let predicted = network.predict_next_pattern(&cue, 1, 32);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        let conf = predicted.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;
        confidence += conf;
        println!(
            "probe {label} {prefix}={prompt:?} predicted={} confidence={:.3}",
            predicted.len(),
            conf
        );
    }

    EvalStats {
        total: prompts.len(),
        nonzero,
        confidence: confidence / prompts.len().max(1) as f32,
    }
}

fn print_summary(label: &str, eval: &EvalGroup, network: &SimplicialNetwork) {
    let stats = network.plasticity_stats();
    println!(
        "{label}: topics={}/{} conf={:.3} questions={}/{} conf={:.3} relations={}/{} conf={:.3}",
        eval.topics.nonzero,
        eval.topics.total,
        eval.topics.confidence,
        eval.questions.nonzero,
        eval.questions.total,
        eval.questions.confidence,
        eval.relations.nonzero,
        eval.relations.total,
        eval.relations.confidence
    );
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
