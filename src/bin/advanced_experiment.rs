use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

fn main() {
    let mut network = SimplicialNetwork::grid_3d(advanced_config(), 3);

    let concept_a = vec![3, 17, 29, 41, 53];
    let concept_b = vec![211, 223, 239, 251, 263];
    let concept_c = vec![419, 431, 443, 457, 467];
    let transient = vec![601, 613, 617, 619, 631];

    for _ in 0..4 {
        network.inject_pattern(&concept_a, 1.2, 2);
        network.reinforce_coactivation(&concept_a, 0.18);
        network.learn_transition(&concept_a, &concept_b);
        network.step();
        network.clear_activity();

        network.inject_pattern(&concept_b, 1.2, 2);
        network.reinforce_coactivation(&concept_b, 0.18);
        network.learn_transition(&concept_b, &concept_c);
        network.step();
        network.clear_activity();
    }

    // Huella transitoria no consolidada: debe poder olvidarse/podarse.
    network.reinforce_coactivation(&transient, 0.06);

    let before = network.plasticity_stats();

    for _ in 0..96 {
        network.step();
        network.clear_activity();
    }

    let after = network.plasticity_stats();
    let prediction_ab = network.evaluate_prediction(&concept_a, &concept_b, concept_b.len());
    let prediction_bc = network.evaluate_prediction(&concept_b, &concept_c, concept_c.len());

    println!("SNGA advanced mechanisms validation");
    println!("nodos={}", network.agents.len());
    println!("tetrahedra={}", after.tetrahedra);
    println!(
        "antes: tick={} aristas_activas={} asociativas={} consolidadas={} episodios={} causal={}",
        before.tick,
        before.active_edges,
        before.associative_edges,
        before.consolidated_edges,
        before.episodes,
        before.causal_edges
    );
    println!(
        "despues: tick={} aristas_activas={} asociativas={} consolidadas={} episodios={} causal={}",
        after.tick,
        after.active_edges,
        after.associative_edges,
        after.consolidated_edges,
        after.episodes,
        after.causal_edges
    );
    println!(
        "prediccion A->B: precision={:.1}% recall={:.1}% matches={}/{}",
        prediction_ab.precision * 100.0,
        prediction_ab.recall * 100.0,
        prediction_ab.matched_agents,
        prediction_ab.expected_agents
    );
    println!(
        "prediccion B->C: precision={:.1}% recall={:.1}% matches={}/{}",
        prediction_bc.precision * 100.0,
        prediction_bc.recall * 100.0,
        prediction_bc.matched_agents,
        prediction_bc.expected_agents
    );
    println!(
        "lectura: {}",
        if after.tetrahedra > 0
            && after.consolidated_edges > 0
            && after.active_edges <= before.active_edges
            && prediction_ab.recall > 0.8
            && prediction_bc.recall > 0.8
        {
            "mecanismos avanzados estables: consolidacion, olvido/poda, replay, causalidad y geometria 3D activos"
        } else {
            "hay actividad avanzada, pero algun mecanismo requiere ajuste"
        }
    );
}

fn advanced_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 18,
        height: 12,
        spacing: 18.0,
        elasticity: 0.01,
        damping: 0.86,
        activation_threshold: 0.66,
        simplex_area_weight: 0.0005,
        max_active_agents: 48,
        inhibition_decay: 0.08,
        max_spikes_per_step: 128,
        local_inhibition_decay: 0.65,
        refractory_ticks: 1,
        rhythm_period: 16,
        rhythm_amplitude: 0.10,
        forgetting_rate: 0.004,
        prune_below_weight: 1.08,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.05,
        max_episodes: 32,
        replay_interval: 8,
        replay_batch: 4,
        replay_learning_rate: 0.04,
        causal_learning_rate: 0.22,
        simplex3_weight: 0.0002,
        hyperbolic_curvature: 0.00001,
        seed: 31,
    }
}
