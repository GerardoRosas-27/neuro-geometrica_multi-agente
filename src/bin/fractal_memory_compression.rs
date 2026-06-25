use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};

const GRID_STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const ADAPTED_STATE_PATH: &str = "data/snga_scaled_gemma_language_fractal_adapted.snga";
const COMPRESSED_STATE_PATH: &str = "data/snga_scaled_gemma_language_fractal_compressed.snga";
const AGENT_COUNT: usize = 5_760;
const PATTERN_SIZE: usize = 12;
const TOP_K: usize = 32;
const TARGET_EDGES: usize = 360_000;

fn main() {
    println!("SNGA fractal memory compression");

    let mut grid = SimplicialNetwork::grid_3d(config(), 2);
    if let Err(err) = grid.load_persistent_state(GRID_STATE_PATH) {
        println!("grid_loaded=false error={err}");
        return;
    }
    let reference = prediction_signature(&grid);

    let fractal_config = FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: AGENT_COUNT,
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    };
    let mut fractal = SimplicialNetwork::fractal_3d(config(), fractal_config);
    if let Err(err) = fractal.load_persistent_state(ADAPTED_STATE_PATH) {
        println!("fractal_loaded=false error={err}");
        return;
    }

    println!(
        "start: grid_edges={} fractal_edges={} grid_energy={:.1} fractal_energy={:.1}",
        grid.plasticity_stats().active_edges,
        fractal.plasticity_stats().active_edges,
        grid.total_free_energy(),
        fractal.total_free_energy()
    );

    let mut chunk = 80_000;
    let mut accepted_pruned = 0;
    while fractal.plasticity_stats().active_edges > TARGET_EDGES && chunk > 0 {
        let before = fractal.clone();
        let removed = fractal.prune_low_value_associative_edges(chunk);
        if removed == 0 {
            break;
        }

        let signature = prediction_signature(&fractal);
        if signature == reference {
            accepted_pruned += removed;
            println!(
                "accepted: removed={} total_pruned={} edges={} energy={:.1}",
                removed,
                accepted_pruned,
                fractal.plasticity_stats().active_edges,
                fractal.total_free_energy()
            );
        } else {
            fractal = before;
            chunk /= 2;
            println!("rejected: next_chunk={chunk}");
        }
    }

    let final_signature = prediction_signature(&fractal);
    let mut knowledge_exact = final_signature == reference;
    if knowledge_exact {
        for phase in 0..6 {
            let rate = 0.16 / (phase as f32 + 1.0).sqrt();
            let adjusted = fractal.anneal_active_edge_rest_lengths(rate, 1.05);
            for _ in 0..4 {
                fractal.step();
            }
            knowledge_exact = prediction_signature(&fractal) == reference;
            println!(
                "relax: phase={} adjusted_edges={} rate={:.3} knowledge_exact={} energy={:.1}",
                phase + 1,
                adjusted,
                rate,
                knowledge_exact,
                fractal.total_free_energy()
            );
            if !knowledge_exact {
                break;
            }
        }
    }

    if !knowledge_exact {
        println!("saved=false error=knowledge changed after compression");
        return;
    }

    match fractal.save_persistent_state(COMPRESSED_STATE_PATH) {
        Ok(report) => println!(
            "saved=true path={} agents={} edges={} causal_edges={}",
            COMPRESSED_STATE_PATH, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("saved=false error={err}");
            return;
        }
    }

    let mut reloaded = SimplicialNetwork::fractal_3d(
        config(),
        FractalMeshConfig {
            levels: 7,
            branches_per_region: 5,
            target_dimension: 2.65,
            target_nodes: AGENT_COUNT,
            base_radius: 0.0,
            lateral_link_weight: 0.35,
            parent_link_weight: 1.0,
        },
    );
    if let Err(err) = reloaded.load_persistent_state(COMPRESSED_STATE_PATH) {
        println!("reload=false error={err}");
        return;
    }
    reloaded.anneal_active_edge_rest_lengths(1.0, 0.0);
    knowledge_exact = prediction_signature(&reloaded) == reference;
    if !knowledge_exact {
        println!("saved_stabilized=false error=knowledge changed after reload stabilization");
        return;
    }
    match reloaded.save_persistent_state(COMPRESSED_STATE_PATH) {
        Ok(report) => println!(
            "saved_stabilized=true path={} agents={} edges={} causal_edges={} energy={:.1}",
            COMPRESSED_STATE_PATH,
            report.agents,
            report.edges,
            report.causal_edges,
            reloaded.total_free_energy()
        ),
        Err(err) => {
            println!("saved_stabilized=false error={err}");
            return;
        }
    }
    fractal = reloaded;

    let grid_size = file_size(GRID_STATE_PATH);
    let adapted_size = file_size(ADAPTED_STATE_PATH);
    let compressed_size = file_size(COMPRESSED_STATE_PATH);
    println!(
        "files: grid_bytes={} adapted_bytes={} compressed_bytes={} compressed_vs_grid={:.3}",
        grid_size,
        adapted_size,
        compressed_size,
        compressed_size as f64 / grid_size.max(1) as f64
    );
    println!(
        "final: knowledge_exact={} pruned={} edges={} associative={} consolidated={} causal={} energy={:.1}",
        knowledge_exact,
        accepted_pruned,
        fractal.plasticity_stats().active_edges,
        fractal.plasticity_stats().associative_edges,
        fractal.plasticity_stats().consolidated_edges,
        fractal.plasticity_stats().causal_edges,
        fractal.total_free_energy()
    );
}

fn prediction_signature(network: &SimplicialNetwork) -> Vec<Vec<usize>> {
    comparison_prompts()
        .into_iter()
        .map(|(prefix, prompt)| {
            let cue = pattern(prefix, prompt, network.agents.len());
            network
                .predict_next_pattern(&cue, 1, TOP_K)
                .into_iter()
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn comparison_prompts() -> Vec<(&'static str, &'static str)> {
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

fn file_size(path: &str) -> u64 {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
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
