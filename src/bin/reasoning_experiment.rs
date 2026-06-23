use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

fn main() {
    let mut network = SimplicialNetwork::grid_3d(reasoning_config(), 2);

    let fire = pattern(10);
    let heat = pattern(70);
    let expansion = pattern(130);
    let rupture = pattern(190);

    let dog = pattern(260);
    let mammal = pattern(320);
    let animal = pattern(380);

    let ice = pattern(460);
    let cold = pattern(520);
    let hot = pattern(580);

    for _ in 0..6 {
        // Cadenas causales sin entrenar los atajos fire->rupture ni dog->animal.
        network.learn_transition(&fire, &heat);
        network.learn_transition(&heat, &expansion);
        network.learn_transition(&expansion, &rupture);

        network.learn_transition(&dog, &mammal);
        network.learn_transition(&mammal, &animal);

        network.learn_transition(&ice, &cold);
        network.learn_transition(&fire, &hot);
        network.learn_contradiction(&cold, &hot);

        network.reinforce_coactivation(&fire, 0.12);
        network.reinforce_coactivation(&heat, 0.12);
        network.reinforce_coactivation(&expansion, 0.12);
        network.reinforce_coactivation(&rupture, 0.12);
        network.reinforce_coactivation(&dog, 0.12);
        network.reinforce_coactivation(&mammal, 0.12);
        network.reinforce_coactivation(&animal, 0.12);
        network.reinforce_coactivation(&ice, 0.12);
        network.reinforce_coactivation(&cold, 0.12);
        network.reinforce_coactivation(&hot, 0.12);
    }

    let direct_fire_rupture = network.evaluate_prediction(&fire, &rupture, rupture.len());
    let transitive_fire_rupture =
        network.evaluate_transitive_prediction(&fire, &rupture, 3, rupture.len());
    let direct_dog_animal = network.evaluate_prediction(&dog, &animal, animal.len());
    let transitive_dog_animal =
        network.evaluate_transitive_prediction(&dog, &animal, 2, animal.len());
    let direct_ice_cold = network.evaluate_prediction(&ice, &cold, cold.len());
    let direct_fire_hot = network.evaluate_prediction(&fire, &hot, 10);

    let contradiction_before = network.contradiction_tension(&cold, &hot);
    network.clear_activity();
    network.inject_pattern(&cold, 1.2, 1);
    network.inject_pattern(&hot, 1.2, 1);
    let energy_with_contradiction = network.total_free_energy();
    network.clear_activity();
    network.inject_pattern(&cold, 1.2, 1);
    let energy_without_contradiction = network.total_free_energy();

    let stats = network.plasticity_stats();

    println!("SNGA reasoning validation");
    println!("nodos={}", network.agents.len());
    println!(
        "estructura: causal_edges={} contradiction_edges={} tetrahedra={}",
        stats.causal_edges, stats.contradiction_edges, stats.tetrahedra
    );
    println!(
        "directo fuego->ruptura: precision={:.1}% recall={:.1}% matches={}/{}",
        direct_fire_rupture.precision * 100.0,
        direct_fire_rupture.recall * 100.0,
        direct_fire_rupture.matched_agents,
        direct_fire_rupture.expected_agents
    );
    println!(
        "transitivo fuego->ruptura: precision={:.1}% recall={:.1}% matches={}/{}",
        transitive_fire_rupture.precision * 100.0,
        transitive_fire_rupture.recall * 100.0,
        transitive_fire_rupture.matched_agents,
        transitive_fire_rupture.expected_agents
    );
    println!(
        "directo perro->animal: precision={:.1}% recall={:.1}% matches={}/{}",
        direct_dog_animal.precision * 100.0,
        direct_dog_animal.recall * 100.0,
        direct_dog_animal.matched_agents,
        direct_dog_animal.expected_agents
    );
    println!(
        "transitivo perro->animal: precision={:.1}% recall={:.1}% matches={}/{}",
        transitive_dog_animal.precision * 100.0,
        transitive_dog_animal.recall * 100.0,
        transitive_dog_animal.matched_agents,
        transitive_dog_animal.expected_agents
    );
    println!(
        "directo hielo->frio: precision={:.1}% recall={:.1}% matches={}/{}",
        direct_ice_cold.precision * 100.0,
        direct_ice_cold.recall * 100.0,
        direct_ice_cold.matched_agents,
        direct_ice_cold.expected_agents
    );
    println!(
        "directo fuego->caliente: precision={:.1}% recall={:.1}% matches={}/{}",
        direct_fire_hot.precision * 100.0,
        direct_fire_hot.recall * 100.0,
        direct_fire_hot.matched_agents,
        direct_fire_hot.expected_agents
    );
    println!(
        "contradiccion frio/caliente: tension={:.3} energia_sin={:.3} energia_con={:.3} delta={:.3}",
        contradiction_before,
        energy_without_contradiction,
        energy_with_contradiction,
        energy_with_contradiction - energy_without_contradiction
    );
    println!(
        "lectura: {}",
        if direct_fire_rupture.recall == 0.0
            && transitive_fire_rupture.recall > 0.8
            && direct_dog_animal.recall == 0.0
            && transitive_dog_animal.recall > 0.8
            && direct_ice_cold.recall > 0.8
            && direct_fire_hot.recall > 0.8
            && energy_with_contradiction > energy_without_contradiction
        {
            "hay razonamiento topologico inicial: infiere rutas no entrenadas y detecta contradiccion energetica"
        } else {
            "la red aprendio asociaciones, pero la inferencia general aun requiere ajuste"
        }
    );
}

fn pattern(start: usize) -> Vec<usize> {
    vec![start, start + 3, start + 7, start + 11, start + 17]
}

fn reasoning_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 32,
        height: 18,
        spacing: 12.0,
        elasticity: 0.008,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0004,
        max_active_agents: 64,
        inhibition_decay: 0.08,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.70,
        refractory_ticks: 1,
        rhythm_period: 16,
        rhythm_amplitude: 0.08,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 64,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.22,
        contradiction_learning_rate: 0.30,
        contradiction_energy_weight: 4.0,
        simplex3_weight: 0.0002,
        hyperbolic_curvature: 0.00001,
        seed: 43,
    }
}
