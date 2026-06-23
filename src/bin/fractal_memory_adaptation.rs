use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const ADAPTED_STATE_PATH: &str = "data/snga_scaled_gemma_language_fractal_adapted.snga";
const PATTERN_SIZE: usize = 12;
const EPOCHS: usize = 6;

fn main() {
    println!("SNGA fractal memory adaptation");

    let mut source = SimplicialNetwork::grid_3d(config(), 2);
    let source_report = match source.load_persistent_state(STATE_PATH) {
        Ok(report) => report,
        Err(err) => {
            println!("loaded=false error={err}");
            return;
        }
    };
    println!(
        "source_loaded=true agents={} edges={} causal_edges={}",
        source_report.agents, source_report.edges, source_report.causal_edges
    );

    let fractal_config = FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: source.agents.len(),
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    };
    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_config);
    match network.load_persistent_memory_state(STATE_PATH) {
        Ok(report) => println!(
            "fractal_memory_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("fractal_memory_loaded=false error={err}");
            return;
        }
    }
    network.enable_neural_oscillations();

    let before = evaluate(&network);
    println!(
        "before: coverage={}/{} confidence={:.3} energy={:.1}",
        before.nonzero,
        before.total,
        before.confidence,
        network.total_free_energy()
    );

    let prompts = adaptation_prompts();
    for epoch in 0..EPOCHS {
        let mut replayed = 0;
        for (prefix, prompt) in &prompts {
            let cue = pattern(prefix, prompt, network.agents.len());
            let predicted = network
                .predict_next_pattern(&cue, 1, PATTERN_SIZE)
                .into_iter()
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>();

            if !predicted.is_empty() {
                network.set_attention_goal(&predicted);
            }
            network.inject_pattern(&cue, 1.05, 2);
            for _ in 0..2 {
                network.step();
            }
            network.clear_attention_goal();
            network.clear_activity();
            replayed += 1;
        }

        let anneal_rate = 0.18 / (epoch as f32 + 1.0).sqrt();
        let adjusted = network.anneal_active_edge_rest_lengths(anneal_rate, 1.05);
        for _ in 0..4 {
            network.step();
        }
        let eval = evaluate(&network);
        println!(
            "epoch={} replayed={} adjusted_edges={} anneal_rate={:.3} coverage={}/{} confidence={:.3} energy={:.1}",
            epoch + 1,
            replayed,
            adjusted,
            anneal_rate,
            eval.nonzero,
            eval.total,
            eval.confidence,
            network.total_free_energy()
        );
    }

    let after = evaluate(&network);
    let stats = network.plasticity_stats();
    println!(
        "after: coverage={}/{} confidence={:.3} energy={:.1} active_edges={} associative={} consolidated={} causal={}",
        after.nonzero,
        after.total,
        after.confidence,
        network.total_free_energy(),
        stats.active_edges,
        stats.associative_edges,
        stats.consolidated_edges,
        stats.causal_edges
    );

    match network.save_persistent_state(ADAPTED_STATE_PATH) {
        Ok(report) => println!(
            "saved_adapted=true path={} agents={} edges={} causal_edges={}",
            ADAPTED_STATE_PATH, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => println!("saved_adapted=false error={err}"),
    }
}

struct EvalStats {
    total: usize,
    nonzero: usize,
    confidence: f32,
}

fn evaluate(network: &SimplicialNetwork) -> EvalStats {
    let prompts = adaptation_prompts();
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for (prefix, prompt) in &prompts {
        let cue = pattern(prefix, prompt, network.agents.len());
        let predicted = network.predict_next_pattern(&cue, 1, 32);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        confidence += predicted.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;
    }
    EvalStats {
        total: prompts.len(),
        nonzero,
        confidence: confidence / prompts.len().max(1) as f32,
    }
}

fn adaptation_prompts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("topic", "lenguaje"),
        ("topic", "concepto"),
        ("topic", "causalidad"),
        ("topic", "memoria"),
        ("topic", "planificacion"),
        ("topic", "mundo"),
        ("topic", "herramienta"),
        ("topic", "emocion"),
        ("topic", "sociedad"),
        ("topic", "fisica simple"),
        ("topic", "objeto"),
        ("topic", "categoria"),
        ("question", "que es una palabra"),
        ("question", "como se agrupan ideas"),
        ("question", "que hace una causa"),
        ("question", "para que sirve un plan"),
        ("question", "como cambia un objeto"),
        ("question", "que es una emocion"),
        ("relation", "simbolo-significado"),
        ("relation", "rasgo-categoria"),
        ("relation", "causa-efecto"),
        ("relation", "objetivo-ruta"),
        ("relation", "pregunta-respuesta"),
    ]
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
