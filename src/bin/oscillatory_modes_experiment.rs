use snga::simplicial::{BrainMode, SimplicialConfig, SimplicialNetwork};

#[derive(Default)]
struct Scores {
    target_recall: f32,
    leakage: f32,
    sequence_recall: f32,
    replay_edges: usize,
    focused_regions: usize,
    sleep_regions: usize,
}

impl Scores {
    fn score(&self) -> f32 {
        self.target_recall * 0.35
            + (1.0 - self.leakage).max(0.0) * 0.20
            + self.sequence_recall * 0.30
            + (self.replay_edges.min(64) as f32 / 64.0) * 0.10
            + ((self.focused_regions + self.sleep_regions).min(16) as f32 / 16.0) * 0.05
    }
}

fn main() {
    let baseline = run_trial(false);
    let oscillatory = run_trial(true);

    println!("SNGA oscillatory modes experiment");
    print_scores("baseline", &baseline);
    print_scores("oscillatory", &oscillatory);
    println!(
        "comparison: score_gain={:.3} recall_gain={:.1}% leakage_reduction={:.1}% sequence_gain={:.1}%",
        oscillatory.score() - baseline.score(),
        (oscillatory.target_recall - baseline.target_recall) * 100.0,
        (baseline.leakage - oscillatory.leakage) * 100.0,
        (oscillatory.sequence_recall - baseline.sequence_recall) * 100.0
    );
    println!(
        "lectura: {}",
        if oscillatory.score() > baseline.score() {
            "las ondas funcionales mejoran la coordinacion global sin campo fisico"
        } else {
            "las ondas ejecutan, pero requieren ajustar acoplamientos antes de integrarse"
        }
    );
}

fn run_trial(enable_oscillations: bool) -> Scores {
    let mut network = SimplicialNetwork::grid_3d(test_config(), 2);
    if enable_oscillations {
        network.enable_neural_oscillations();
    }

    let cue = pattern(20);
    let target = pattern(180);
    let distractor = pattern(330);
    let seq_a = pattern(430);
    let seq_b = pattern(520);
    let seq_c = pattern(610);

    let mut target_memory = cue.clone();
    target_memory.extend(target.iter().copied());
    for _ in 0..6 {
        network.inject_pattern(&target_memory, 1.2, 1);
        network.step();
        network.clear_activity();
        network.learn_transition(&seq_a, &seq_b);
        network.learn_transition(&seq_b, &seq_c);
        network.inject_pattern(&seq_a, 1.0, 1);
        network.inject_pattern(&seq_b, 1.0, 1);
        network.inject_pattern(&seq_c, 1.0, 1);
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
    let target_recall = active_recall(&network, &target);
    let leakage = active_recall(&network, &distractor);
    network.clear_attention_goal();
    network.clear_activity();

    let sequence = network.evaluate_transitive_prediction(&seq_a, &seq_c, 2, seq_c.len());
    let plasticity = network.plasticity_stats();
    let osc = network.oscillation_stats();

    Scores {
        target_recall,
        leakage,
        sequence_recall: sequence.recall,
        replay_edges: plasticity.consolidated_edges,
        focused_regions: osc.beta_regions + osc.gamma_regions,
        sleep_regions: usize::from(osc.mode == BrainMode::SleepReplay) + osc.delta_regions,
    }
}

fn print_scores(label: &str, scores: &Scores) {
    println!(
        "{}: target_recall={:.1}% leakage={:.1}% sequence_recall={:.1}% replay_edges={} focused_regions={} sleep_regions={} score={:.3}",
        label,
        scores.target_recall * 100.0,
        scores.leakage * 100.0,
        scores.sequence_recall * 100.0,
        scores.replay_edges,
        scores.focused_regions,
        scores.sleep_regions,
        scores.score()
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

fn test_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 32,
        height: 22,
        spacing: 10.0,
        elasticity: 0.006,
        damping: 0.88,
        activation_threshold: 0.66,
        simplex_area_weight: 0.0002,
        max_active_agents: 14,
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
        seed: 151,
    }
}
