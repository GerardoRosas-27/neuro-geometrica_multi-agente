use snga::simplicial::{
    EpisodicRecall, PatternPredictionReport, SimplicialConfig, SimplicialNetwork,
};
use std::collections::HashSet;

#[derive(Clone)]
struct Event {
    label: &'static str,
    pattern: Vec<usize>,
}

#[derive(Clone)]
struct Story {
    label: &'static str,
    events: Vec<Event>,
}

fn main() {
    let dataset = synthetic_dataset();
    let mut network = SimplicialNetwork::grid_3d(experiment_config(), 2);

    train_dataset(&mut network, &dataset, 10);
    consolidate_replay(&mut network, 72);

    let smoke = event(&dataset, "smoke");
    let fire = event(&dataset, "fire");
    let alarm = event(&dataset, "alarm");
    let evacuate = event(&dataset, "evacuate");
    let clouds = event(&dataset, "clouds");
    let rain = event(&dataset, "rain");

    let episodic = validate_episodic_memory(&network, &alarm.pattern);
    let prediction =
        network.evaluate_pattern_prediction(&smoke.pattern, &fire.pattern, 1, fire.pattern.len());
    let prediction_ok = prediction.recall >= 0.80 && prediction.prediction_error <= 0.35;

    network.clear_activity();
    network.set_attention_goal(&evacuate.pattern);
    network.inject_pattern(&alarm.pattern, 1.15, 2);
    network.step();
    let attention = network.attention_report(24);
    let attention_goal_hits = overlap_count(
        &attention
            .context_agents
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>(),
        &evacuate.pattern,
    );
    let attention_ok = !attention.goal_agents.is_empty() && attention_goal_hits > 0;

    let rollout = network.internal_rollout(&clouds.pattern, 3, 1, rain.pattern.len());
    let rollout_first = rollout
        .steps
        .first()
        .map(|step| recall(&step.predicted_pattern, &rain.pattern))
        .unwrap_or(0.0);
    let rollout_ok = rollout.steps.len() == 3 && rollout_first >= 0.80;

    let plan = network.plan_to_goal(&smoke.pattern, &evacuate.pattern, 3, 24);
    let plan_ok = plan.reached_goal && plan.path.len() >= 2;

    let stats = network.plasticity_stats();

    println!("SNGA internal improvements experiment");
    println!(
        "dataset: stories={} [{}] events={} nodes={}",
        dataset.len(),
        dataset
            .iter()
            .map(|story| story.label)
            .collect::<Vec<_>>()
            .join(","),
        dataset
            .iter()
            .map(|story| story.events.len())
            .sum::<usize>(),
        network.agents.len()
    );
    println!(
        "training: episodes={} causal_edges={} active_edges={} consolidated_edges={}",
        stats.episodes, stats.causal_edges, stats.active_edges, stats.consolidated_edges
    );
    print_episodic_result(&episodic, &alarm.pattern);
    print_prediction_result("pattern_prediction smoke->fire", &prediction);
    println!(
        "attention alarm->evacuate: goal_agents={} context_goal_hits={} boosted={} suppressed={} verdict={}",
        attention.goal_agents.len(),
        attention_goal_hits,
        attention.boosted_agents,
        attention.suppressed_agents,
        verdict(attention_ok)
    );
    println!(
        "world_rollout clouds->rain: steps={} first_step_recall={:.1}% energy_delta={:.3} verdict={}",
        rollout.steps.len(),
        rollout_first * 100.0,
        rollout.energy_delta,
        verdict(rollout_ok)
    );
    println!(
        "planner smoke->evacuate: reached={} path_len={} score={:.4} verdict={}",
        plan.reached_goal,
        plan.path.len(),
        plan.score,
        verdict(plan_ok)
    );
    println!(
        "lectura: {}",
        if episodic.matches.first().is_some()
            && episodic
                .matches
                .first()
                .map(|item| pattern_recall(&item.pattern, &alarm.pattern) >= 0.80)
                .unwrap_or(false)
            && prediction_ok
            && attention_ok
            && rollout_ok
            && plan_ok
        {
            "las 5 mejoras internas estan activas y cooperan sin usar el tokenizador"
        } else {
            "alguna mejora requiere ajuste antes de considerarse estable"
        }
    );
}

fn synthetic_dataset() -> Vec<Story> {
    vec![
        Story {
            label: "hazard_response",
            events: vec![
                Event {
                    label: "smoke",
                    pattern: pattern(20),
                },
                Event {
                    label: "fire",
                    pattern: pattern(80),
                },
                Event {
                    label: "alarm",
                    pattern: pattern(140),
                },
                Event {
                    label: "evacuate",
                    pattern: pattern(200),
                },
            ],
        },
        Story {
            label: "cooking_sequence",
            events: vec![
                Event {
                    label: "kitchen",
                    pattern: pattern(260),
                },
                Event {
                    label: "hunger",
                    pattern: pattern(320),
                },
                Event {
                    label: "cook",
                    pattern: pattern(380),
                },
                Event {
                    label: "eat",
                    pattern: pattern(440),
                },
                Event {
                    label: "satisfied",
                    pattern: pattern(500),
                },
            ],
        },
        Story {
            label: "weather_response",
            events: vec![
                Event {
                    label: "clouds",
                    pattern: pattern(560),
                },
                Event {
                    label: "rain",
                    pattern: pattern(620),
                },
                Event {
                    label: "wet_ground",
                    pattern: pattern(680),
                },
                Event {
                    label: "shelter",
                    pattern: pattern(740),
                },
            ],
        },
    ]
}

fn train_dataset(network: &mut SimplicialNetwork, dataset: &[Story], epochs: usize) {
    for _ in 0..epochs {
        for story in dataset {
            network.clear_activity();
            for event in &story.events {
                network.inject_pattern(&event.pattern, 1.20, 2);
                network.reinforce_coactivation(&event.pattern, 0.14);
                network.step();
            }
            network.clear_activity();
        }
    }
}

fn consolidate_replay(network: &mut SimplicialNetwork, steps: usize) {
    for _ in 0..steps {
        network.step();
        network.clear_activity();
    }
}

fn validate_episodic_memory(network: &SimplicialNetwork, alarm: &[usize]) -> EpisodicRecall {
    let noisy_alarm = vec![alarm[0], alarm[2], alarm[4], 500];
    network.retrieve_episodes(&noisy_alarm, 3)
}

fn print_episodic_result(recall_report: &EpisodicRecall, expected: &[usize]) {
    let top_recall = recall_report
        .matches
        .first()
        .map(|item| pattern_recall(&item.pattern, expected))
        .unwrap_or(0.0);
    let top_similarity = recall_report
        .matches
        .first()
        .map(|item| item.similarity)
        .unwrap_or(0.0);
    let ok = top_recall >= 0.80 && top_similarity >= 0.30;

    println!(
        "episodic_recall noisy_alarm: matches={} top_recall={:.1}% top_similarity={:.3} merged={} verdict={}",
        recall_report.matches.len(),
        top_recall * 100.0,
        top_similarity,
        recall_report.merged_pattern.len(),
        verdict(ok)
    );
}

fn print_prediction_result(label: &str, report: &PatternPredictionReport) {
    let ok = report.recall >= 0.80 && report.prediction_error <= 0.35;
    println!(
        "{}: precision={:.1}% recall={:.1}% error={:.3} matches={} verdict={}",
        label,
        report.precision * 100.0,
        report.recall * 100.0,
        report.prediction_error,
        report.matched_agents,
        verdict(ok)
    );
}

fn event<'a>(dataset: &'a [Story], label: &str) -> &'a Event {
    dataset
        .iter()
        .flat_map(|story| story.events.iter())
        .find(|event| event.label == label)
        .unwrap_or_else(|| panic!("missing event: {label}"))
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

fn recall(predicted: &[(usize, f32)], expected: &[usize]) -> f32 {
    let predicted_agents = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
    pattern_recall(&predicted_agents, expected)
}

fn pattern_recall(predicted: &[usize], expected: &[usize]) -> f32 {
    let expected_set = expected.iter().copied().collect::<HashSet<_>>();
    overlap_count(predicted, expected) as f32 / expected_set.len().max(1) as f32
}

fn overlap_count(left: &[usize], right: &[usize]) -> usize {
    let right_set = right.iter().copied().collect::<HashSet<_>>();
    left.iter().filter(|idx| right_set.contains(idx)).count()
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "ok"
    } else {
        "fail"
    }
}

fn experiment_config() -> SimplicialConfig {
    let mut config = SimplicialConfig::default();
    config.width = 36;
    config.height = 24;
    config.spacing = 12.0;
    config.elasticity = 0.008;
    config.damping = 0.86;
    config.activation_threshold = 0.64;
    config.simplex_area_weight = 0.0004;
    config.max_active_agents = 72;
    config.inhibition_decay = 0.08;
    config.max_spikes_per_step = 256;
    config.local_inhibition_decay = 0.70;
    config.refractory_ticks = 1;
    config.rhythm_period = 16;
    config.rhythm_amplitude = 0.08;
    config.forgetting_rate = 0.001;
    config.prune_below_weight = 0.02;
    config.consolidate_after = 3;
    config.consolidated_forgetting_scale = 0.05;
    config.max_episodes = 128;
    config.replay_interval = 6;
    config.replay_batch = 8;
    config.replay_learning_rate = 0.04;
    config.causal_learning_rate = 0.24;
    config.contradiction_learning_rate = 0.30;
    config.contradiction_energy_weight = 3.0;
    config.simplex3_weight = 0.0002;
    config.hyperbolic_curvature = 0.00001;
    config.seed = 71;
    config
}
