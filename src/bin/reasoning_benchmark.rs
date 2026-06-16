use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const CAUSAL_CHAINS: usize = 5_000;
const HIERARCHY_CHAINS: usize = 3_000;
const CONTRADICTIONS: usize = 3_000;
const EVAL_CAUSAL: usize = 600;
const EVAL_HIERARCHY: usize = 400;
const EVAL_CONTRADICTIONS: usize = 400;
const PATTERN_SIZE: usize = 5;
const TRANSITIVE_LIMIT: usize = 512;

#[derive(Default)]
struct ReasoningAggregate {
    direct_recall: f32,
    broad_recall: f32,
    broad_precision: f32,
    optimized_recall: f32,
    optimized_precision: f32,
    rewarded_paths: usize,
    evaporated_paths: usize,
    samples: usize,
}

#[derive(Default)]
struct ContradictionAggregate {
    tension: f32,
    energy_delta: f32,
    samples: usize,
}

fn main() {
    let mut network = SimplicialNetwork::grid(benchmark_config());

    println!("SNGA large reasoning benchmark");
    println!("causal_chains={CAUSAL_CHAINS}");
    println!("hierarchy_chains={HIERARCHY_CHAINS}");
    println!("contradictions={CONTRADICTIONS}");
    println!("nodos={}", network.agents.len());
    println!();

    train_causal_chains(&mut network);
    train_hierarchies(&mut network);
    train_contradictions(&mut network);

    let stats = network.plasticity_stats();
    println!(
        "post_entrenamiento: aristas_activas={} asociativas={} consolidadas={} episodios={} causal={} contradicciones={}",
        stats.active_edges,
        stats.associative_edges,
        stats.consolidated_edges,
        stats.episodes,
        stats.causal_edges,
        stats.contradiction_edges
    );
    println!();

    let causal = evaluate_causal_chains(&mut network);
    let hierarchy = evaluate_hierarchies(&mut network);
    let contradiction = evaluate_contradictions(&mut network);

    print_reasoning_summary("causal", &causal);
    print_reasoning_summary("jerarquia", &hierarchy);
    print_contradiction_summary(&contradiction);

    let causal_direct = causal.direct_recall / causal.samples as f32;
    let causal_transitive = causal.optimized_recall / causal.samples as f32;
    let hierarchy_direct = hierarchy.direct_recall / hierarchy.samples as f32;
    let hierarchy_transitive = hierarchy.optimized_recall / hierarchy.samples as f32;
    let contradiction_delta = contradiction.energy_delta / contradiction.samples as f32;
    let reasoning_ok = causal_direct < 0.05
        && causal_transitive > 0.90
        && hierarchy_direct < 0.05
        && hierarchy_transitive > 0.90
        && contradiction_delta > 0.1;

    println!();
    println!(
        "lectura: {}",
        if reasoning_ok {
            "surge razonamiento topologico medible: infiere relaciones no entrenadas y detecta incompatibilidades"
        } else {
            "hay inferencia parcial, pero el razonamiento aun requiere mejor separacion o reglas energeticas"
        }
    );
}

fn train_causal_chains(network: &mut SimplicialNetwork) {
    for chain_id in 0..CAUSAL_CHAINS {
        let a = pattern("causal", chain_id, 0, network.agents.len());
        let b = pattern("causal", chain_id, 1, network.agents.len());
        let c = pattern("causal", chain_id, 2, network.agents.len());
        let d = pattern("causal", chain_id, 3, network.agents.len());

        network.learn_transition(&a, &b);
        network.learn_transition(&b, &c);
        network.learn_transition(&c, &d);
        reinforce_states(network, [&a, &b, &c, &d]);
    }
}

fn train_hierarchies(network: &mut SimplicialNetwork) {
    for chain_id in 0..HIERARCHY_CHAINS {
        let leaf = pattern("hierarchy", chain_id, 0, network.agents.len());
        let parent = pattern("hierarchy", chain_id, 1, network.agents.len());
        let root = pattern("hierarchy", chain_id, 2, network.agents.len());

        network.learn_transition(&leaf, &parent);
        network.learn_transition(&parent, &root);
        reinforce_states(network, [&leaf, &parent, &root]);
    }
}

fn train_contradictions(network: &mut SimplicialNetwork) {
    for pair_id in 0..CONTRADICTIONS {
        let left = pattern("contradiction", pair_id, 0, network.agents.len());
        let right = pattern("contradiction", pair_id, 1, network.agents.len());

        network.learn_contradiction(&left, &right);
        reinforce_states(network, [&left, &right]);
    }
}

fn reinforce_states<const N: usize>(network: &mut SimplicialNetwork, states: [&Vec<usize>; N]) {
    for state in states {
        network.reinforce_coactivation(state, 0.08);
    }
}

fn evaluate_causal_chains(network: &mut SimplicialNetwork) -> ReasoningAggregate {
    let mut aggregate = ReasoningAggregate::default();
    for chain_id in 0..EVAL_CAUSAL {
        let a = pattern("causal", chain_id, 0, network.agents.len());
        let d = pattern("causal", chain_id, 3, network.agents.len());
        let direct = network.evaluate_prediction(&a, &d, d.len());
        let broad = network.evaluate_transitive_prediction(&a, &d, 3, TRANSITIVE_LIMIT);
        let optimized =
            network.optimize_routes_to_expected(&a, &d, 3, TRANSITIVE_LIMIT, 0.08, 0.04);
        aggregate.direct_recall += direct.recall;
        aggregate.broad_recall += broad.recall;
        aggregate.broad_precision += broad.precision;
        aggregate.optimized_recall += optimized.prediction.recall;
        aggregate.optimized_precision += optimized.prediction.precision;
        aggregate.rewarded_paths += optimized.rewarded_paths;
        aggregate.evaporated_paths += optimized.evaporated_paths;
        aggregate.samples += 1;
    }
    aggregate
}

fn evaluate_hierarchies(network: &mut SimplicialNetwork) -> ReasoningAggregate {
    let mut aggregate = ReasoningAggregate::default();
    for chain_id in 0..EVAL_HIERARCHY {
        let leaf = pattern("hierarchy", chain_id, 0, network.agents.len());
        let root = pattern("hierarchy", chain_id, 2, network.agents.len());
        let direct = network.evaluate_prediction(&leaf, &root, root.len());
        let broad = network.evaluate_transitive_prediction(&leaf, &root, 2, TRANSITIVE_LIMIT);
        let optimized =
            network.optimize_routes_to_expected(&leaf, &root, 2, TRANSITIVE_LIMIT, 0.08, 0.04);
        aggregate.direct_recall += direct.recall;
        aggregate.broad_recall += broad.recall;
        aggregate.broad_precision += broad.precision;
        aggregate.optimized_recall += optimized.prediction.recall;
        aggregate.optimized_precision += optimized.prediction.precision;
        aggregate.rewarded_paths += optimized.rewarded_paths;
        aggregate.evaporated_paths += optimized.evaporated_paths;
        aggregate.samples += 1;
    }
    aggregate
}

fn evaluate_contradictions(network: &mut SimplicialNetwork) -> ContradictionAggregate {
    let mut aggregate = ContradictionAggregate::default();
    for pair_id in 0..EVAL_CONTRADICTIONS {
        let left = pattern("contradiction", pair_id, 0, network.agents.len());
        let right = pattern("contradiction", pair_id, 1, network.agents.len());

        let tension = network.contradiction_tension(&left, &right);
        aggregate.tension += tension;
        aggregate.energy_delta += tension * network.config.contradiction_energy_weight;
        aggregate.samples += 1;
    }
    aggregate
}

fn print_reasoning_summary(label: &str, aggregate: &ReasoningAggregate) {
    let n = aggregate.samples.max(1) as f32;
    println!(
        "resumen_{label}: direct_recall={:.1}% broad_recall={:.1}% broad_precision={:.1}% optimized_recall={:.1}% optimized_precision={:.1}% rutas_reforzadas={} rutas_evaporadas={} muestras={}",
        aggregate.direct_recall / n * 100.0,
        aggregate.broad_recall / n * 100.0,
        aggregate.broad_precision / n * 100.0,
        aggregate.optimized_recall / n * 100.0,
        aggregate.optimized_precision / n * 100.0,
        aggregate.rewarded_paths,
        aggregate.evaporated_paths,
        aggregate.samples
    );
}

fn print_contradiction_summary(aggregate: &ContradictionAggregate) {
    let n = aggregate.samples.max(1) as f32;
    println!(
        "resumen_contradiccion: tension_media={:.3} delta_energia_medio={:.3} muestras={}",
        aggregate.tension / n,
        aggregate.energy_delta / n,
        aggregate.samples
    );
}

fn pattern(domain: &str, id: usize, role: usize, nodes: usize) -> Vec<usize> {
    (0..PATTERN_SIZE)
        .map(|term| {
            let mut hasher = DefaultHasher::new();
            domain.hash(&mut hasher);
            id.hash(&mut hasher);
            role.hash(&mut hasher);
            term.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn benchmark_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 720,
        height: 360,
        spacing: 3.0,
        elasticity: 0.0025,
        damping: 0.88,
        activation_threshold: 0.66,
        simplex_area_weight: 0.00008,
        max_active_agents: 64,
        inhibition_decay: 0.02,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.80,
        refractory_ticks: 0,
        rhythm_period: 32,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 4,
        consolidated_forgetting_scale: 0.2,
        max_episodes: 256,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.25,
        contradiction_energy_weight: 4.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 53,
    }
}
