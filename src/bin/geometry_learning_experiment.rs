use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

#[derive(Clone, Copy)]
struct GeometryMetrics {
    intra_distance: f32,
    distractor_distance: f32,
    compactness_ratio: f32,
    total_free_energy: f32,
    associative_edges: usize,
    mean_associative_weight: f32,
    mean_associative_rest_length: f32,
}

fn main() {
    let concept = pattern(24);
    let distractor = pattern(360);
    let mut network = SimplicialNetwork::grid_3d(experiment_config(), 2);

    let before = measure(&network, &concept, &distractor);

    for _ in 0..18 {
        network.inject_pattern(&concept, 1.2, 2);
        network.reinforce_coactivation_if_useful(&concept, 0.11, 0.95);
        for _ in 0..4 {
            network.step();
        }
        network.clear_activity();
    }

    for _ in 0..120 {
        network.step();
    }

    let after = measure(&network, &concept, &distractor);

    println!("SNGA geometry learning experiment");
    print_metrics("before", before);
    print_metrics("after", after);
    println!(
        "delta: intra_distance={:.2}% distractor_distance={:.2}% compactness={:.2}% free_energy={:.2}% associative_edges={} mean_weight={:.2}%",
        pct_delta(after.intra_distance, before.intra_distance),
        pct_delta(after.distractor_distance, before.distractor_distance),
        pct_delta(after.compactness_ratio, before.compactness_ratio),
        pct_delta(after.total_free_energy, before.total_free_energy),
        after.associative_edges as isize - before.associative_edges as isize,
        pct_delta(after.mean_associative_weight, before.mean_associative_weight.max(0.001)),
    );
    println!(
        "lectura: {}",
        if after.intra_distance < before.intra_distance
            && after.compactness_ratio < before.compactness_ratio
            && after.associative_edges > before.associative_edges
            && after.mean_associative_weight > before.mean_associative_weight
        {
            "el aprendizaje deforma la geometria: compacta el concepto y crea sinapsis mas fuertes"
        } else {
            "hay cambios de pesos, pero la deformacion geometrica requiere ajuste"
        }
    );
}

fn measure(
    network: &SimplicialNetwork,
    concept: &[usize],
    distractor: &[usize],
) -> GeometryMetrics {
    let intra_distance = mean_pair_distance(network, concept);
    let distractor_distance = mean_cross_distance(network, concept, distractor);
    let (associative_edges, mean_associative_weight, mean_associative_rest_length) =
        associative_edge_stats(network, concept);

    GeometryMetrics {
        intra_distance,
        distractor_distance,
        compactness_ratio: intra_distance / distractor_distance.max(0.001),
        total_free_energy: network.total_free_energy(),
        associative_edges,
        mean_associative_weight,
        mean_associative_rest_length,
    }
}

fn print_metrics(label: &str, metrics: GeometryMetrics) {
    println!(
        "{}: intra_distance={:.3} distractor_distance={:.3} compactness={:.3} free_energy={:.3} associative_edges={} mean_weight={:.3} mean_rest={:.3}",
        label,
        metrics.intra_distance,
        metrics.distractor_distance,
        metrics.compactness_ratio,
        metrics.total_free_energy,
        metrics.associative_edges,
        metrics.mean_associative_weight,
        metrics.mean_associative_rest_length,
    );
}

fn mean_pair_distance(network: &SimplicialNetwork, pattern: &[usize]) -> f32 {
    let mut total = 0.0;
    let mut count = 0;
    for i in 0..pattern.len() {
        for j in (i + 1)..pattern.len() {
            total += agent_distance(network, pattern[i], pattern[j]);
            count += 1;
        }
    }
    total / count.max(1) as f32
}

fn mean_cross_distance(network: &SimplicialNetwork, left: &[usize], right: &[usize]) -> f32 {
    let mut total = 0.0;
    let mut count = 0;
    for &a in left {
        for &b in right {
            total += agent_distance(network, a, b);
            count += 1;
        }
    }
    total / count.max(1) as f32
}

fn associative_edge_stats(network: &SimplicialNetwork, pattern: &[usize]) -> (usize, f32, f32) {
    let mut count = 0;
    let mut weight_sum = 0.0;
    let mut rest_sum = 0.0;
    for edge in &network.edges {
        if !edge.active || edge.weight <= 1.05 {
            continue;
        }
        if pattern.contains(&edge.a) && pattern.contains(&edge.b) {
            count += 1;
            weight_sum += edge.weight;
            rest_sum += edge.rest_length;
        }
    }
    (
        count,
        weight_sum / count.max(1) as f32,
        rest_sum / count.max(1) as f32,
    )
}

fn agent_distance(network: &SimplicialNetwork, a: usize, b: usize) -> f32 {
    let pa = network.agents[a].position;
    let pb = network.agents[b].position;
    let dz = network.agents[a].depth - network.agents[b].depth;
    ((pa - pb).length_squared() + dz * dz).sqrt()
}

fn pct_delta(after: f32, before: f32) -> f32 {
    (after - before) / before.abs().max(0.001) * 100.0
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
