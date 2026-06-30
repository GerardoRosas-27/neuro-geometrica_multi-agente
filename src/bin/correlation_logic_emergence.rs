use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::hash::{Hash, Hasher};

const PATTERN_SIZE: usize = 12;
const TOP_K: usize = 12;

#[derive(Clone, Copy)]
struct LogicCase {
    label: &'static str,
    a: &'static str,
    b: &'static str,
    c: &'static str,
    distractor: &'static str,
}

#[derive(Default)]
struct EvalStats {
    cases: usize,
    hits: usize,
    distractor_hits: usize,
    mean_overlap: f32,
    mean_distractor_overlap: f32,
    mean_confidence: f32,
}

fn main() {
    println!("SNGA correlation -> logic emergence probe");
    println!(
        "criterion: conclusion C must appear from A without direct A->C training; distractor should stay low"
    );

    let pure = run_condition("correlation_only", train_correlation_only);
    let directed = run_condition("directed_chain", train_directed_chain);
    let corrected = run_condition("prediction_error_focus", train_prediction_error_focus);

    print_summary("correlation_only", pure);
    print_summary("directed_chain", directed);
    print_summary("prediction_error_focus", corrected);
}

fn run_condition(label: &str, train: fn(&mut SimplicialNetwork, &LogicCase)) -> EvalStats {
    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    for case in logic_cases() {
        train(&mut network, case);
    }

    let mut stats = EvalStats::default();
    for case in logic_cases() {
        let a = pattern("premise_a", case.a, network.agents.len());
        let c = pattern("conclusion_c", case.c, network.agents.len());
        let distractor = pattern("distractor", case.distractor, network.agents.len());
        let predicted = network.infer_transitive_from(&a, 2, TOP_K);
        let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let overlap = overlap_ratio(&predicted_ids, &c);
        let distractor_overlap = overlap_ratio(&predicted_ids, &distractor);
        let confidence = confidence(&predicted);

        stats.cases += 1;
        stats.hits += usize::from(overlap > 0.0);
        stats.distractor_hits += usize::from(distractor_overlap > 0.0);
        stats.mean_overlap += overlap;
        stats.mean_distractor_overlap += distractor_overlap;
        stats.mean_confidence += confidence;

        println!(
            "case {label} {:?}: conclusion_overlap={:.1}% distractor_overlap={:.1}% confidence={:.3}",
            case.label,
            overlap * 100.0,
            distractor_overlap * 100.0,
            confidence
        );
    }

    if stats.cases > 0 {
        let cases = stats.cases as f32;
        stats.mean_overlap /= cases;
        stats.mean_distractor_overlap /= cases;
        stats.mean_confidence /= cases;
    }

    let plasticity = network.plasticity_stats();
    println!(
        "network {label}: edges={} associative={} cells={} causal={} energy={:.1}",
        plasticity.active_edges,
        plasticity.associative_edges,
        plasticity.semantic_cells,
        plasticity.causal_edges,
        network.total_free_energy()
    );
    stats
}

fn train_correlation_only(network: &mut SimplicialNetwork, case: &LogicCase) {
    let a = pattern("premise_a", case.a, network.agents.len());
    let b = pattern("bridge_b", case.b, network.agents.len());
    let c = pattern("conclusion_c", case.c, network.agents.len());
    let distractor = pattern("distractor", case.distractor, network.agents.len());

    for _ in 0..8 {
        reinforce_group(network, &[&a, &b], 0.10);
        reinforce_group(network, &[&b, &c], 0.10);
        reinforce_group(network, &[&a, &distractor], 0.075);
    }
}

fn train_directed_chain(network: &mut SimplicialNetwork, case: &LogicCase) {
    let a = pattern("premise_a", case.a, network.agents.len());
    let b = pattern("bridge_b", case.b, network.agents.len());
    let c = pattern("conclusion_c", case.c, network.agents.len());

    for _ in 0..4 {
        network.learn_transition(&a, &b);
        network.learn_transition(&b, &c);
        reinforce_group(network, &[&a, &b], 0.08);
        reinforce_group(network, &[&b, &c], 0.08);
    }
}

fn train_prediction_error_focus(network: &mut SimplicialNetwork, case: &LogicCase) {
    train_directed_chain(network, case);

    let a = pattern("premise_a", case.a, network.agents.len());
    let c = pattern("conclusion_c", case.c, network.agents.len());
    for _ in 0..3 {
        network.learn_from_prediction_error(&a, &c, 2, TOP_K, 0.12);
    }
}

fn reinforce_group(network: &mut SimplicialNetwork, parts: &[&Vec<usize>], learning_rate: f32) {
    let mut fused = Vec::new();
    for part in parts {
        fused.extend(part.iter().copied());
    }
    fused.sort_unstable();
    fused.dedup();
    network.reinforce_coactivation_if_useful(&fused, learning_rate, 0.92);
}

fn print_summary(label: &str, stats: EvalStats) {
    println!(
        "{label}_summary: hits={}/{} distractor_hits={}/{} mean_overlap={:.1}% mean_distractor_overlap={:.1}% mean_confidence={:.3}",
        stats.hits,
        stats.cases,
        stats.distractor_hits,
        stats.cases,
        stats.mean_overlap * 100.0,
        stats.mean_distractor_overlap * 100.0,
        stats.mean_confidence
    );
}

fn logic_cases() -> &'static [LogicCase] {
    &[
        LogicCase {
            label: "lluvia moja suelo",
            a: "nubes densas anuncian lluvia",
            b: "lluvia cae sobre la calle",
            c: "suelo queda mojado",
            distractor: "suelo queda seco",
        },
        LogicCase {
            label: "fuego produce vapor",
            a: "fuego calienta recipiente",
            b: "agua alcanza hervor",
            c: "vapor aparece",
            distractor: "hielo aparece",
        },
        LogicCase {
            label: "semilla crece",
            a: "semilla recibe agua",
            b: "semilla germina",
            c: "planta empieza a crecer",
            distractor: "semilla se vuelve piedra",
        },
        LogicCase {
            label: "objeto cae",
            a: "objeto queda sin soporte",
            b: "objeto desciende",
            c: "objeto toca el suelo",
            distractor: "objeto sube al cielo",
        },
        LogicCase {
            label: "alarma dirige atencion",
            a: "alarma empieza a sonar",
            b: "evento se vuelve relevante",
            c: "atencion se dirige a la alarma",
            distractor: "atencion ignora la alarma",
        },
    ]
}

fn pattern(role: &str, text: &str, nodes: usize) -> Vec<usize> {
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            role.hash(&mut hasher);
            text.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn overlap_ratio(left: &[usize], right: &[usize]) -> f32 {
    let hits = right.iter().filter(|idx| left.contains(idx)).count();
    hits as f32 / right.len().max(1) as f32
}

fn confidence(predicted: &[(usize, f32)]) -> f32 {
    if predicted.is_empty() {
        return 0.0;
    }
    predicted.iter().map(|(_, score)| *score).sum::<f32>() / predicted.len() as f32
}

fn fractal_mesh_config() -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 5,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: 4096,
        base_radius: 220.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    }
}

fn config() -> SimplicialConfig {
    SimplicialConfig {
        width: 32,
        height: 20,
        spacing: 10.0,
        elasticity: 0.006,
        damping: 0.90,
        activation_threshold: 0.62,
        simplex_area_weight: 0.0002,
        max_active_agents: 96,
        inhibition_decay: 0.08,
        max_spikes_per_step: 256,
        local_inhibition_decay: 0.85,
        refractory_ticks: 0,
        rhythm_period: 24,
        rhythm_amplitude: 0.0,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.2,
        max_episodes: 128,
        replay_interval: 0,
        replay_batch: 4,
        replay_learning_rate: 0.03,
        causal_learning_rate: 0.14,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.00008,
        hyperbolic_curvature: 0.0,
        seed: 919,
    }
}
