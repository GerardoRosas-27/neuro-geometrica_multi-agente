use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork, GOLDEN_UTILITY_THRESHOLD};

#[derive(Clone, Copy, Debug)]
enum Topology {
    Grid3d,
    Fractal3d,
}

#[derive(Default)]
struct TrialReport {
    nodes: usize,
    edges: usize,
    tetrahedra: usize,
    recall: f32,
    precision: f32,
    leakage: f32,
    mean_energy: f32,
    mean_active_agents: f32,
    active_edges: usize,
}

fn main() {
    println!("SNGA fractal mesh experiment");
    println!(
        "growth_gate=1/phi={:.6} target_dimension=2.65",
        GOLDEN_UTILITY_THRESHOLD
    );

    for topology in [Topology::Grid3d, Topology::Fractal3d] {
        let report = run_trial(topology);
        println!(
            "{:?}: nodes={} edges={} tetrahedra={} recall={:.1}% precision={:.1}% leakage={:.1}% mean_energy={:.3} mean_active={:.1} active_edges={}",
            topology,
            report.nodes,
            report.edges,
            report.tetrahedra,
            report.recall * 100.0,
            report.precision * 100.0,
            report.leakage * 100.0,
            report.mean_energy,
            report.mean_active_agents,
            report.active_edges
        );
    }
}

fn run_trial(topology: Topology) -> TrialReport {
    let mut network = match topology {
        Topology::Grid3d => SimplicialNetwork::grid_3d(test_config(), 2),
        Topology::Fractal3d => {
            SimplicialNetwork::fractal_3d(test_config(), FractalMeshConfig::default())
        }
    };

    let nodes = network.agents.len();
    let language = pattern(11, nodes);
    let target = pattern(181, nodes);
    let distractor = pattern(337, nodes);
    let mut concept = language.clone();
    concept.extend(target.iter().copied());
    let mut noisy = language.clone();
    noisy.extend(distractor.iter().copied());

    for epoch in 0..10 {
        let utility = 0.72 + epoch as f32 * 0.02;
        network.reinforce_coactivation_if_useful(&concept, 0.10, utility);
        network.reinforce_coactivation_if_useful(&noisy, 0.035, 0.42);
        network.clear_activity();
    }

    network.inject_pattern(&language, 1.2, 3);
    let mut total_energy = 0.0;
    let mut total_active = 0_usize;
    let steps = 6;
    for _ in 0..steps {
        let stats = network.step();
        total_energy += stats.total_free_energy;
        total_active += stats.active_agents;
    }

    let target_active = count_active(&network, &target);
    let distractor_active = count_active(&network, &distractor);
    TrialReport {
        nodes,
        edges: network.edges.len(),
        tetrahedra: network.tetrahedra.len(),
        recall: target_active as f32 / target.len() as f32,
        precision: target_active as f32 / (target_active + distractor_active).max(1) as f32,
        leakage: distractor_active as f32 / distractor.len() as f32,
        mean_energy: total_energy / steps as f32,
        mean_active_agents: total_active as f32 / steps as f32,
        active_edges: network.edges.iter().filter(|edge| edge.active).count(),
    }
}

fn test_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 30,
        height: 18,
        spacing: 10.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.62,
        simplex_area_weight: 0.0002,
        max_active_agents: 36,
        inhibition_decay: 0.04,
        max_spikes_per_step: 96,
        local_inhibition_decay: 0.75,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.001,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 128,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 313,
    }
}

fn pattern(seed: usize, nodes: usize) -> Vec<usize> {
    (0..8)
        .map(|offset| (seed * 97 + offset * 31 + seed * offset * 7) % nodes)
        .collect()
}

fn count_active(network: &SimplicialNetwork, pattern: &[usize]) -> usize {
    pattern
        .iter()
        .filter(|&&idx| network.agents[idx].surprise > 0.08)
        .count()
}
