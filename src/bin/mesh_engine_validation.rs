use snga::mesh_engine::{MeshConfig, SimplicialMeshEngine};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

fn main() {
    let mesh_config = MeshConfig {
        width: 16,
        height: 12,
        spacing: 10.0,
        seed: 211,
    };
    let topology = SimplicialMeshEngine::grid_3d(mesh_config, 3);
    let stats = topology.stats(3);

    let mut network = SimplicialNetwork::grid_3d(neural_config(), 3);
    network.enable_neural_oscillations();

    let cue = pattern(20);
    let target = pattern(180);
    let mut episode = cue.clone();
    episode.extend(target.iter().copied());

    for _ in 0..5 {
        network.inject_pattern(&episode, 1.2, 1);
        network.step();
        network.clear_activity();
    }
    for _ in 0..160 {
        network.step();
        network.clear_activity();
    }
    network.set_attention_goal(&target);
    network.inject_pattern(&cue, 1.2, 2);
    for _ in 0..5 {
        network.step();
    }

    let recall = active_recall(&network, &target);
    let osc = network.oscillation_stats();

    println!("SNGA mesh engine validation");
    println!(
        "mesh: nodes={} edges={} triangles={} tetrahedra={} depth_layers={}",
        stats.nodes, stats.edges, stats.triangles, stats.tetrahedra, stats.depth_layers
    );
    println!(
        "neural: recall={:.1}% oscillations={} mode={:?} regions={} beta_regions={} gamma_regions={}",
        recall * 100.0,
        osc.enabled,
        osc.mode,
        osc.regions,
        osc.beta_regions,
        osc.gamma_regions
    );
    println!(
        "lectura: {}",
        if stats.tetrahedra > 0 && recall > 0.8 && osc.enabled {
            "motor 3D separado y capa neuronal oscilatoria trabajan correctamente"
        } else {
            "la separacion compila, pero requiere ajuste en topologia u oscilaciones"
        }
    );
}

fn active_recall(network: &SimplicialNetwork, pattern: &[usize]) -> f32 {
    let active = pattern
        .iter()
        .filter(|&&idx| network.agents[idx].surprise > 0.08)
        .count();
    active as f32 / pattern.len().max(1) as f32
}

fn pattern(start: usize) -> Vec<usize> {
    vec![
        start,
        start + 3,
        start + 7,
        start + 11,
        start + 17,
        start + 23,
    ]
}

fn neural_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 16,
        height: 12,
        spacing: 10.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.66,
        simplex_area_weight: 0.0002,
        max_active_agents: 16,
        inhibition_decay: 0.04,
        max_spikes_per_step: 8,
        local_inhibition_decay: 0.72,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.001,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 128,
        replay_interval: 0,
        replay_batch: 6,
        replay_learning_rate: 0.12,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 211,
    }
}
