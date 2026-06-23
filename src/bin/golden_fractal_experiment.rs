use snga::geometry::Vec2;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork, GOLDEN_UTILITY_THRESHOLD};
use std::collections::HashMap;

const PHI: f32 = 1.618_034;
const PHI_INV: f32 = GOLDEN_UTILITY_THRESHOLD;

#[derive(Clone, Copy, Debug)]
enum Variant {
    Baseline,
    FibonacciLayout,
    GoldenLearning,
    GoldenPruning,
}

#[derive(Default, Clone)]
struct TrialScores {
    associative_recall: f32,
    associative_precision: f32,
    associative_leakage: f32,
    causal_recall: f32,
    causal_precision: f32,
    language_top1: f32,
    language_top3: f32,
}

impl TrialScores {
    fn score(&self) -> f32 {
        self.associative_recall * 0.20
            + self.associative_precision * 0.15
            + (1.0 - self.associative_leakage).max(0.0) * 0.15
            + self.causal_recall * 0.20
            + self.causal_precision * 0.10
            + self.language_top1 * 0.15
            + self.language_top3 * 0.05
    }
}

fn main() {
    let variants = [
        Variant::Baseline,
        Variant::FibonacciLayout,
        Variant::GoldenLearning,
        Variant::GoldenPruning,
    ];

    println!("SNGA golden/fractal variants experiment");
    println!("phi={PHI:.6} phi_inv={PHI_INV:.6}");

    let mut best = (Variant::Baseline, TrialScores::default());
    for variant in variants {
        let scores = evaluate_variant(variant);
        println!(
            "{:?}: assoc_recall={:.1}% assoc_precision={:.1}% leakage={:.1}% causal_recall={:.1}% causal_precision={:.1}% lang_top1={:.1}% lang_top3={:.1}% score={:.3}",
            variant,
            scores.associative_recall * 100.0,
            scores.associative_precision * 100.0,
            scores.associative_leakage * 100.0,
            scores.causal_recall * 100.0,
            scores.causal_precision * 100.0,
            scores.language_top1 * 100.0,
            scores.language_top3 * 100.0,
            scores.score(),
        );
        if scores.score() > best.1.score() {
            best = (variant, scores);
        }
    }

    println!(
        "winner={:?} score={:.3} lectura={}",
        best.0,
        best.1.score(),
        match best.0 {
            Variant::Baseline =>
                "ninguna variante aurea supera al baseline; no conviene integrarlas",
            Variant::FibonacciLayout =>
                "la distribucion espacial Fibonacci/aurea aporta mas que pesos o poda",
            Variant::GoldenLearning =>
                "el escalado aureo de aprendizaje aporta mas y conviene integrarlo",
            Variant::GoldenPruning => "la poda/gating aurea aporta mas y conviene integrarla",
        }
    );
}

fn evaluate_variant(variant: Variant) -> TrialScores {
    TrialScores {
        associative_recall: associative_trial(variant).0,
        associative_precision: associative_trial(variant).1,
        associative_leakage: associative_trial(variant).2,
        causal_recall: causal_trial(variant).0,
        causal_precision: causal_trial(variant).1,
        language_top1: language_trial(variant).0,
        language_top3: language_trial(variant).1,
    }
}

fn associative_trial(variant: Variant) -> (f32, f32, f32) {
    let mut network = SimplicialNetwork::grid_3d(test_config(), 2);
    apply_variant_layout(&mut network, variant);

    let language = pattern(10);
    let target = pattern(180);
    let distractor = pattern(330);
    let mut concept = language.clone();
    concept.extend(target.iter().copied());
    let mut noisy = language.clone();
    noisy.extend(distractor.iter().copied());

    for _ in 0..8 {
        reinforce_variant(&mut network, &concept, variant, 0.11, 0.85);
        reinforce_variant(&mut network, &noisy, variant, 0.035, 0.25);
        network.clear_activity();
    }

    network.inject_pattern(&language, 1.2, 3);
    for _ in 0..5 {
        network.step();
    }

    let target_active = count_active(&network, &target);
    let distractor_active = count_active(&network, &distractor);
    let recall = target_active as f32 / target.len() as f32;
    let precision = target_active as f32 / (target_active + distractor_active).max(1) as f32;
    let leakage = distractor_active as f32 / distractor.len() as f32;
    (recall, precision, leakage)
}

fn causal_trial(variant: Variant) -> (f32, f32) {
    let mut network = SimplicialNetwork::grid_3d(test_config(), 2);
    apply_variant_layout(&mut network, variant);

    let a = pattern(20);
    let b = pattern(210);
    let c = pattern(390);
    let distractor = pattern(550);

    for _ in 0..8 {
        network.learn_transition(&a, &b);
        network.learn_transition(&b, &c);
        if matches!(variant, Variant::GoldenLearning) {
            network.reinforce_coactivation(&b, 0.08 * PHI_INV);
            network.reinforce_coactivation(&c, 0.08 * PHI_INV);
        } else {
            network.reinforce_coactivation(&b, 0.08);
            network.reinforce_coactivation(&c, 0.08);
        }
        if !matches!(variant, Variant::GoldenPruning) {
            network.learn_transition(&a, &distractor);
        }
    }

    let predicted = network.infer_transitive_from(&a, 2, c.len() + distractor.len());
    let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
    let hits = overlap_count(&predicted_ids, &c);
    let false_hits = overlap_count(&predicted_ids, &distractor);
    let recall = hits as f32 / c.len() as f32;
    let precision = hits as f32 / (hits + false_hits).max(1) as f32;
    (recall, precision)
}

fn language_trial(variant: Variant) -> (f32, f32) {
    let corpus = [
        vec!["sistema", "explica", "energia", "libre"],
        vec!["sistema", "explica", "memoria", "episodica"],
        vec!["red", "predice", "ruta", "causal"],
        vec!["malla", "reduce", "sorpresa", "local"],
    ];
    let eval = [
        vec!["sistema", "explica", "energia", "libre"],
        vec!["red", "predice", "ruta", "causal"],
        vec!["malla", "reduce", "sorpresa", "local"],
    ];
    let vocab = vocabulary(&corpus);
    let mut network = SimplicialNetwork::grid_3d(test_config(), 2);
    apply_variant_layout(&mut network, variant);

    for _ in 0..10 {
        for sentence in &corpus {
            for pos in 1..sentence.len() {
                let context = token_pattern(vocab[sentence[pos - 1]], network.agents.len());
                let next = token_pattern(vocab[sentence[pos]], network.agents.len());
                network.learn_transition(&context, &next);
                reinforce_variant(&mut network, &next, variant, 0.05, 0.8);
            }
        }
    }

    let mut total = 0;
    let mut top1 = 0;
    let mut top3 = 0;
    for sentence in &eval {
        for pos in 1..sentence.len() {
            let context = token_pattern(vocab[sentence[pos - 1]], network.agents.len());
            let target_id = vocab[sentence[pos]];
            let predictions = network.predict_from(&context, 128);
            let ranked = score_tokens(&predictions, vocab.len(), network.agents.len());
            total += 1;
            if ranked.first().copied() == Some(target_id) {
                top1 += 1;
            }
            if ranked.iter().take(3).any(|&idx| idx == target_id) {
                top3 += 1;
            }
        }
    }

    (
        top1 as f32 / total.max(1) as f32,
        top3 as f32 / total.max(1) as f32,
    )
}

fn reinforce_variant(
    network: &mut SimplicialNetwork,
    pattern: &[usize],
    variant: Variant,
    learning_rate: f32,
    utility: f32,
) {
    match variant {
        Variant::GoldenLearning => network.reinforce_coactivation(pattern, learning_rate * PHI_INV),
        Variant::GoldenPruning => {
            network.reinforce_coactivation_if_useful(pattern, learning_rate, utility);
        }
        _ => network.reinforce_coactivation(pattern, learning_rate),
    }
}

fn apply_variant_layout(network: &mut SimplicialNetwork, variant: Variant) {
    if !matches!(variant, Variant::FibonacciLayout) {
        return;
    }
    let n = network.agents.len().max(1);
    let radius =
        (network.config.width.max(network.config.height) as f32 * network.config.spacing) * 0.45;
    let center = Vec2::new(
        network.config.width as f32 * network.config.spacing * 0.5,
        network.config.height as f32 * network.config.spacing * 0.5,
    );
    for (i, agent) in network.agents.iter_mut().enumerate() {
        let t = (i as f32 + 0.5) / n as f32;
        let z = 1.0 - 2.0 * t;
        let r = (1.0 - z * z).sqrt();
        let theta = std::f32::consts::TAU * i as f32 * PHI_INV;
        agent.position = center + Vec2::new(theta.cos() * r * radius, theta.sin() * r * radius);
        agent.depth = z * radius;
        agent.velocity = Vec2::ZERO;
        agent.depth_velocity = 0.0;
    }
}

fn test_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 28,
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
        seed: 131,
    }
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

fn count_active(network: &SimplicialNetwork, pattern: &[usize]) -> usize {
    pattern
        .iter()
        .filter(|&&idx| network.agents[idx].surprise > 0.08)
        .count()
}

fn overlap_count(left: &[usize], right: &[usize]) -> usize {
    let right = right
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    left.iter().filter(|idx| right.contains(idx)).count()
}

fn vocabulary(corpus: &[Vec<&str>]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for token in corpus.iter().flat_map(|sentence| sentence.iter()) {
        let next_id = map.len();
        map.entry((*token).to_string()).or_insert(next_id);
    }
    map
}

fn token_pattern(token_id: usize, nodes: usize) -> Vec<usize> {
    (0..9)
        .map(|offset| (token_id * 97 + offset * 31 + token_id * offset * 7) % nodes)
        .collect()
}

fn score_tokens(predicted: &[(usize, f32)], vocab_size: usize, nodes: usize) -> Vec<usize> {
    let scores = predicted.iter().copied().collect::<HashMap<_, _>>();
    let mut ranked = (0..vocab_size)
        .map(|token_id| {
            let score = token_pattern(token_id, nodes)
                .iter()
                .map(|idx| scores.get(idx).copied().unwrap_or(0.0))
                .sum::<f32>();
            (token_id, score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().map(|(token_id, _)| token_id).collect()
}
