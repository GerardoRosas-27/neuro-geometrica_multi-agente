use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const STATE_PATH: &str = "data/persistent_substrate_test.snga";

fn main() {
    let concept = pattern(24);
    let mut trained = SimplicialNetwork::grid_3d(experiment_config(), 2);

    for _ in 0..14 {
        trained.inject_pattern(&concept, 1.2, 2);
        trained.reinforce_coactivation_if_useful(&concept, 0.10, 0.95);
        for _ in 0..4 {
            trained.step();
        }
        trained.clear_activity();
    }
    for _ in 0..80 {
        trained.step();
    }

    let trained_distance = mean_pair_distance(&trained, &concept);
    let save_report = trained
        .save_persistent_state(STATE_PATH)
        .expect("guardar sustrato persistente");

    let mut loaded = SimplicialNetwork::grid_3d(experiment_config(), 2);
    let load_report = loaded
        .load_persistent_state(STATE_PATH)
        .expect("cargar sustrato persistente");
    let loaded_distance = mean_pair_distance(&loaded, &concept);

    loaded.inject_pattern(&concept, 1.0, 2);
    for _ in 0..3 {
        loaded.step();
    }
    let recall = active_recall(&loaded, &concept);

    println!("SNGA persistent substrate experiment");
    println!(
        "save: agents={} edges={} causal={}",
        save_report.agents, save_report.edges, save_report.causal_edges
    );
    println!(
        "load: agents={} edges={} causal={}",
        load_report.agents, load_report.edges, load_report.causal_edges
    );
    println!(
        "geometry: trained_distance={:.3} loaded_distance={:.3} delta={:.6}",
        trained_distance,
        loaded_distance,
        (trained_distance - loaded_distance).abs()
    );
    println!("recall_after_load={:.1}%", recall * 100.0);
    println!(
        "lectura: {}",
        if (trained_distance - loaded_distance).abs() < 0.001 && recall > 0.8 {
            "el sustrato geometrico y los pesos se conservan al reiniciar"
        } else {
            "la persistencia guarda datos, pero la recuperacion requiere ajuste"
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

fn mean_pair_distance(network: &SimplicialNetwork, pattern: &[usize]) -> f32 {
    let mut total = 0.0;
    let mut count = 0;
    for i in 0..pattern.len() {
        for j in (i + 1)..pattern.len() {
            let pa = network.agents[pattern[i]].position;
            let pb = network.agents[pattern[j]].position;
            let dz = network.agents[pattern[i]].depth - network.agents[pattern[j]].depth;
            total += ((pa - pb).length_squared() + dz * dz).sqrt();
            count += 1;
        }
    }
    total / count.max(1) as f32
}

fn pattern(start: usize) -> Vec<usize> {
    vec![
        start,
        start + 5,
        start + 11,
        start + 19,
        start + 29,
        start + 41,
        start + 55,
    ]
}

fn experiment_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 30,
        height: 20,
        spacing: 12.0,
        elasticity: 0.018,
        damping: 0.84,
        activation_threshold: 0.66,
        simplex_area_weight: 0.00025,
        max_active_agents: 64,
        inhibition_decay: 0.08,
        max_spikes_per_step: 128,
        local_inhibition_decay: 0.75,
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
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.16,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 193,
    }
}
