use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;

const DEFAULT_INPUT_STATE: &str =
    "data/snga_fractal_semantic_executive_gemma_adapter_compressed.snga";
const DEFAULT_OUTPUT_STATE: &str =
    "data/snga_fractal_semantic_executive_gemma_adapter_expanded.snga";
const DEFAULT_REGION_SIZE: usize = 16_384;
const REGION_COUNT: usize = 12;

fn main() {
    let input = arg_value("--input").unwrap_or_else(|| DEFAULT_INPUT_STATE.to_string());
    let output = arg_value("--output").unwrap_or_else(|| DEFAULT_OUTPUT_STATE.to_string());
    let region_size = arg_value("--region-size")
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            env::var("SNGA_SEMEXEC_REGION_SIZE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(DEFAULT_REGION_SIZE)
        .max(8_192);
    let target_nodes = region_size * REGION_COUNT;

    println!("SNGA semantic-executive adapter expansion");
    println!("input={input} output={output} region_size={region_size} target_nodes={target_nodes}");

    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(target_nodes));
    match network.load_persistent_memory_state(&input) {
        Ok(report) => println!(
            "memory_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("memory_loaded=false error={err}");
            return;
        }
    }

    let adjusted = network.anneal_active_edge_rest_lengths(1.0, 0.0);
    match network.save_persistent_state(&output) {
        Ok(report) => println!(
            "saved=true path={} agents={} edges={} causal_edges={} adjusted={} energy={:.1}",
            output,
            report.agents,
            report.edges,
            report.causal_edges,
            adjusted,
            network.total_free_energy()
        ),
        Err(err) => {
            println!("saved=false error={err}");
            return;
        }
    }

    println!(
        "files: input_bytes={} output_bytes={} ratio={:.3}",
        file_size(&input),
        file_size(&output),
        file_size(&output) as f64 / file_size(&input).max(1) as f64
    );
}

fn fractal_mesh_config(target_nodes: usize) -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 6,
        target_dimension: 2.72,
        target_nodes,
        base_radius: 0.0,
        lateral_link_weight: 0.32,
        parent_link_weight: 1.0,
    }
}

fn config() -> SimplicialConfig {
    SimplicialConfig {
        width: 72,
        height: 40,
        spacing: 6.5,
        elasticity: 0.005,
        damping: 0.86,
        activation_threshold: 0.63,
        simplex_area_weight: 0.00012,
        max_active_agents: 448,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 14,
        rhythm_amplitude: 0.045,
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
        seed: 727,
    }
}

fn file_size(path: &str) -> u64 {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}
