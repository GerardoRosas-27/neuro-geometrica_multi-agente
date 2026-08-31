//! Benchmark multiseed de planificación, abstención y sueño cognitivo.

use cdt_rqm_epr::cognitive_logistics::{
    Location, LogisticsAction, LogisticsController, LogisticsGoal, LogisticsPlannerConfig,
    LogisticsState, LogisticsTask, Package, PrimitiveEpisode,
};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use serde_json::json;
use std::fs;
use std::time::Instant;

const DEFAULT_SEEDS: usize = 20;
const REPORT_PATH: &str = "data/native_cognitive_logistics_report.json";

#[derive(Clone)]
struct PrivateCase {
    task: LogisticsTask,
    possible: bool,
    optimal_steps: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct EvalMetrics {
    cases: usize,
    possible: usize,
    solved: usize,
    valid_plans: usize,
    emitted_plans: usize,
    impossible: usize,
    correct_abstentions: usize,
    spl_sum: f32,
    deep_successes: usize,
}

impl EvalMetrics {
    fn success_rate(self) -> f32 {
        self.solved as f32 / self.possible.max(1) as f32
    }

    fn validity(self) -> f32 {
        self.valid_plans as f32 / self.emitted_plans.max(1) as f32
    }

    fn abstention_accuracy(self) -> f32 {
        self.correct_abstentions as f32 / self.impossible.max(1) as f32
    }

    fn spl(self) -> f32 {
        self.spl_sum / self.possible.max(1) as f32
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seed_count = std::env::var("COGNITIVE_LOGISTICS_SEEDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SEEDS)
        .max(1);
    let cases = benchmark_cases();
    let dev_cases = cases[..2].to_vec();
    let started = Instant::now();
    let mut latencies_us = Vec::<f64>::new();
    let mut full_runs = Vec::new();
    let mut greedy_runs = Vec::new();
    let mut one_hop_runs = Vec::new();
    let mut no_prior_runs = Vec::new();
    let mut sleep_commits = 0usize;

    println!(
        "Native Cognitive Logistics Benchmark seeds={} cases={} possible={} impossible={}",
        seed_count,
        cases.len(),
        cases.iter().filter(|case| case.possible).count(),
        cases.iter().filter(|case| !case.possible).count(),
    );

    for seed_index in 0..seed_count {
        let seed = 0xC061_571C_u64
            .wrapping_add((seed_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut controller = trained_controller(seed);
        let before_dev = evaluate(&controller, &dev_cases, None);
        let mut sleep_candidate = controller.clone();
        sleep_candidate.dream_replay(3);
        let after_dev = evaluate(&sleep_candidate, &dev_cases, None);
        let relation_budget = sleep_candidate.substrate.relation_count()
            <= controller.substrate.relation_count() * 5 / 4;
        let sleep_accept = after_dev.success_rate() + 1.0e-6 >= before_dev.success_rate()
            && after_dev.validity() + 1.0e-6 >= before_dev.validity()
            && relation_budget
            && sleep_candidate
                .substrate
                .thermal
                .report()
                .mean_energy
                .is_finite();
        if sleep_accept {
            controller = sleep_candidate;
            sleep_commits += 1;
        }

        let full = evaluate(&controller, &cases, Some(&mut latencies_us));
        let mut greedy = controller.clone();
        greedy.config.beam_width = 1;
        let greedy_metrics = evaluate(&greedy, &cases, None);
        let one_hop_cases = cases
            .iter()
            .cloned()
            .map(|mut case| {
                case.task.max_steps = 1;
                case
            })
            .collect::<Vec<_>>();
        let one_hop = evaluate(&controller, &one_hop_cases, None);
        let mut no_prior = controller.clone();
        no_prior.config.procedural_gain = 0.0;
        let no_prior_metrics = evaluate(&no_prior, &cases, None);
        println!(
            "seed={} sleep_commit={} success={:.1}% validity={:.1}% spl={:.3} abstention={:.1}% deep={} greedy={:.1}% one_hop={:.1}% no_prior={:.1}%",
            seed_index + 1,
            sleep_accept,
            full.success_rate() * 100.0,
            full.validity() * 100.0,
            full.spl(),
            full.abstention_accuracy() * 100.0,
            full.deep_successes,
            greedy_metrics.success_rate() * 100.0,
            one_hop.success_rate() * 100.0,
            no_prior_metrics.success_rate() * 100.0,
        );
        full_runs.push(full);
        greedy_runs.push(greedy_metrics);
        one_hop_runs.push(one_hop);
        no_prior_runs.push(no_prior_metrics);
    }

    latencies_us.sort_by(|left, right| left.total_cmp(right));
    let p95_index = ((latencies_us.len() as f32 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(latencies_us.len().saturating_sub(1));
    let p95_us = latencies_us.get(p95_index).copied().unwrap_or_default();
    let full_mean = mean_metrics(&full_runs);
    let greedy_mean = mean_metrics(&greedy_runs);
    let one_hop_mean = mean_metrics(&one_hop_runs);
    let no_prior_mean = mean_metrics(&no_prior_runs);
    let worst_success = full_runs
        .iter()
        .map(|metrics| metrics.success_rate())
        .fold(1.0f32, f32::min);
    let worst_validity = full_runs
        .iter()
        .map(|metrics| metrics.validity())
        .fold(1.0f32, f32::min);
    let worst_abstention = full_runs
        .iter()
        .map(|metrics| metrics.abstention_accuracy())
        .fold(1.0f32, f32::min);
    let passed = full_mean.success_rate() >= 0.75
        && worst_success >= 0.75
        && worst_validity >= 0.999
        && worst_abstention >= 0.999
        && full_mean.spl() >= 0.75
        && full_mean.deep_successes >= 2
        && greedy_mean.success_rate() + 0.10 <= full_mean.success_rate()
        && one_hop_mean.success_rate() + 0.25 <= full_mean.success_rate()
        && sleep_commits == seed_count
        && p95_us < 50_000.0;
    let decision = if passed {
        "cognitive_logistics_multiseed_pass"
    } else {
        "cognitive_logistics_multiseed_needs_tuning"
    };
    println!(
        "summary success_mean={:.1}% success_worst={:.1}% validity_worst={:.1}% spl={:.3} abstention_worst={:.1}% p95_us={:.1} sleep_commits={}/{} decision={}",
        full_mean.success_rate() * 100.0,
        worst_success * 100.0,
        worst_validity * 100.0,
        full_mean.spl(),
        worst_abstention * 100.0,
        p95_us,
        sleep_commits,
        seed_count,
        decision,
    );

    let report = json!({
        "schema": "native_cognitive_logistics_report_v1",
        "decision": decision,
        "seeds": seed_count,
        "cases_per_seed": cases.len(),
        "possible_cases": cases.iter().filter(|case| case.possible).count(),
        "impossible_cases": cases.iter().filter(|case| !case.possible).count(),
        "inference_receives_oracle": false,
        "test_cases_used_in_sleep": false,
        "full": {
            "success_mean": full_mean.success_rate(),
            "success_worst": worst_success,
            "validity_worst": worst_validity,
            "spl_mean": full_mean.spl(),
            "abstention_worst": worst_abstention,
            "deep_successes_mean": full_mean.deep_successes,
            "latency_p95_us": p95_us,
        },
        "ablations": {
            "greedy_success_mean": greedy_mean.success_rate(),
            "one_hop_success_mean": one_hop_mean.success_rate(),
            "no_procedural_prior_success_mean": no_prior_mean.success_rate(),
        },
        "sleep": {
            "accepted": sleep_commits,
            "attempted": seed_count,
            "replay_source": "primitive_train_episodes_only",
        },
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    if let Some(parent) = std::path::Path::new(REPORT_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(REPORT_PATH, serde_json::to_vec_pretty(&report)?)?;
    println!("report={REPORT_PATH}");
    if !passed {
        return Err("benchmark logístico no alcanzó el gate".into());
    }
    Ok(())
}

fn trained_controller(seed: u64) -> LogisticsController {
    let substrate = NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 6,
            nodes_per_slice: 512,
            temperature: 0.18,
            dt: 0.01,
            diffusion: 0.14,
            confinement: 0.055,
            pilot_gain: 0.35,
            phase_coupling: 0.10,
            amplitude_decay: 0.003,
            seed,
            ..NativeThermoCdtConfig::default()
        },
        NativeThermoRqmConfig {
            thermal_steps_per_train: 1,
            thermal_steps_per_query: 1,
            thermal_activation_margin: 0.001,
            thermal_score_gain: 0.10,
            collect_query_diagnostics: false,
            max_candidates: 128,
            ..NativeThermoRqmConfig::default()
        },
        EntanglementConfig {
            max_links_per_node: 8,
            max_syncs_per_step: 0,
            create_threshold: 1.5,
            ..EntanglementConfig::default()
        },
    );
    let mut controller = LogisticsController::new(substrate, LogisticsPlannerConfig::default());
    for episode in primitive_curriculum() {
        controller.observe(episode);
    }
    controller
}

fn primitive_curriculum() -> Vec<PrimitiveEpisode> {
    let edges = line_connections();
    let mut episodes = Vec::new();
    for &(a, b) in &edges {
        for (from, to) in [(a, b), (b, a)] {
            let before = state(from, Some(Location(3)), false, false, Vec::new());
            let after = before.apply(LogisticsAction::Move(to)).unwrap();
            episodes.push(PrimitiveEpisode {
                before,
                action: LogisticsAction::Move(to),
                after,
                reward: 1.0,
            });
        }
    }
    for location in [Location(0), Location(1), Location(2), Location(3)] {
        let before = state(location, Some(location), false, false, Vec::new());
        let after = before.apply(LogisticsAction::Pickup(Package(0))).unwrap();
        episodes.push(PrimitiveEpisode {
            before,
            action: LogisticsAction::Pickup(Package(0)),
            after,
            reward: 1.0,
        });
        let before = state(location, None, true, false, Vec::new());
        let after = before.apply(LogisticsAction::Drop(Package(0))).unwrap();
        episodes.push(PrimitiveEpisode {
            before,
            action: LogisticsAction::Drop(Package(0)),
            after,
            reward: 1.0,
        });
    }
    let before = state(
        Location(2),
        Some(Location(1)),
        false,
        true,
        vec![(Location(2), Location(3))],
    );
    let after = before.apply(LogisticsAction::Unlock(Location(3))).unwrap();
    episodes.push(PrimitiveEpisode {
        before,
        action: LogisticsAction::Unlock(Location(3)),
        after,
        reward: 1.0,
    });
    episodes
}

fn benchmark_cases() -> Vec<PrivateCase> {
    vec![
        case(
            "deliver-a-c",
            Location(0),
            Location(0),
            Location(2),
            false,
            true,
            4,
        ),
        case(
            "deliver-c-a",
            Location(2),
            Location(2),
            Location(0),
            false,
            true,
            4,
        ),
        case(
            "collect-b-d",
            Location(0),
            Location(1),
            Location(3),
            false,
            true,
            5,
        ),
        case(
            "collect-c-a",
            Location(3),
            Location(2),
            Location(0),
            false,
            true,
            5,
        ),
        case(
            "deliver-d-a",
            Location(3),
            Location(3),
            Location(0),
            false,
            true,
            5,
        ),
        PrivateCase {
            task: LogisticsTask {
                id: "unlock-and-deliver".into(),
                initial: state(
                    Location(0),
                    Some(Location(1)),
                    false,
                    true,
                    vec![(Location(2), Location(3))],
                ),
                goal: LogisticsGoal {
                    package: Package(0),
                    destination: Location(3),
                },
                max_steps: 6,
            },
            possible: true,
            optimal_steps: 6,
        },
        PrivateCase {
            task: LogisticsTask {
                id: "beam-avoids-locked-dead-end".into(),
                initial: branching_dead_end_state(),
                goal: LogisticsGoal {
                    package: Package(0),
                    destination: Location(3),
                },
                max_steps: 4,
            },
            possible: true,
            optimal_steps: 4,
        },
        PrivateCase {
            task: LogisticsTask {
                id: "disconnected-goal".into(),
                initial: state(Location(0), Some(Location(0)), false, false, Vec::new()),
                goal: LogisticsGoal {
                    package: Package(0),
                    destination: Location(9),
                },
                max_steps: 6,
            },
            possible: false,
            optimal_steps: 0,
        },
        PrivateCase {
            task: LogisticsTask {
                id: "locked-without-key".into(),
                initial: state(
                    Location(0),
                    Some(Location(1)),
                    false,
                    false,
                    vec![(Location(2), Location(3))],
                ),
                goal: LogisticsGoal {
                    package: Package(0),
                    destination: Location(3),
                },
                max_steps: 6,
            },
            possible: false,
            optimal_steps: 0,
        },
    ]
}

fn case(
    id: &str,
    robot: Location,
    package: Location,
    destination: Location,
    carrying: bool,
    possible: bool,
    optimal_steps: usize,
) -> PrivateCase {
    PrivateCase {
        task: LogisticsTask {
            id: id.into(),
            initial: state(
                robot,
                (!carrying).then_some(package),
                carrying,
                false,
                Vec::new(),
            ),
            goal: LogisticsGoal {
                package: Package(0),
                destination,
            },
            max_steps: optimal_steps,
        },
        possible,
        optimal_steps,
    }
}

fn state(
    robot_at: Location,
    package_at: Option<Location>,
    carrying: bool,
    has_key: bool,
    locked_edges: Vec<(Location, Location)>,
) -> LogisticsState {
    let mut state = LogisticsState {
        robot_at,
        package_at: vec![package_at],
        carrying: carrying.then_some(Package(0)),
        has_key,
        connections: line_connections(),
        locked_edges,
    };
    state.canonicalize();
    state
}

fn line_connections() -> Vec<(Location, Location)> {
    vec![
        (Location(0), Location(1)),
        (Location(1), Location(2)),
        (Location(2), Location(3)),
    ]
}

fn branching_dead_end_state() -> LogisticsState {
    let mut state = LogisticsState {
        robot_at: Location(0),
        package_at: vec![None],
        carrying: Some(Package(0)),
        has_key: false,
        connections: vec![
            (Location(0), Location(1)),
            (Location(1), Location(3)),
            (Location(0), Location(4)),
            (Location(4), Location(5)),
            (Location(5), Location(3)),
        ],
        locked_edges: vec![(Location(1), Location(3))],
    };
    state.canonicalize();
    state
}

fn evaluate(
    controller: &LogisticsController,
    cases: &[PrivateCase],
    mut latencies_us: Option<&mut Vec<f64>>,
) -> EvalMetrics {
    let mut metrics = EvalMetrics {
        cases: cases.len(),
        possible: cases.iter().filter(|case| case.possible).count(),
        impossible: cases.iter().filter(|case| !case.possible).count(),
        ..EvalMetrics::default()
    };
    for case in cases {
        let mut trial = controller.clone();
        let started = Instant::now();
        let decision = trial.plan(&case.task);
        if let Some(latencies) = latencies_us.as_deref_mut() {
            latencies.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        }
        if let Some(plan) = decision.plan {
            metrics.emitted_plans += 1;
            let verification = LogisticsController::verify(&case.task, &plan);
            metrics.valid_plans += usize::from(verification.actions_valid);
            let solved = case.possible && verification.actions_valid && verification.goal_reached;
            metrics.solved += usize::from(solved);
            if solved {
                metrics.spl_sum +=
                    case.optimal_steps as f32 / plan.len().max(case.optimal_steps) as f32;
                metrics.deep_successes += usize::from(plan.len() >= 4);
            }
        } else if !case.possible && decision.abstained {
            metrics.correct_abstentions += 1;
        }
    }
    metrics
}

fn mean_metrics(runs: &[EvalMetrics]) -> EvalMetrics {
    if runs.is_empty() {
        return EvalMetrics::default();
    }
    let n = runs.len();
    EvalMetrics {
        cases: runs.iter().map(|metrics| metrics.cases).sum::<usize>() / n,
        possible: runs.iter().map(|metrics| metrics.possible).sum::<usize>() / n,
        solved: runs.iter().map(|metrics| metrics.solved).sum::<usize>() / n,
        valid_plans: runs
            .iter()
            .map(|metrics| metrics.valid_plans)
            .sum::<usize>()
            / n,
        emitted_plans: runs
            .iter()
            .map(|metrics| metrics.emitted_plans)
            .sum::<usize>()
            / n,
        impossible: runs.iter().map(|metrics| metrics.impossible).sum::<usize>() / n,
        correct_abstentions: runs
            .iter()
            .map(|metrics| metrics.correct_abstentions)
            .sum::<usize>()
            / n,
        spl_sum: runs.iter().map(|metrics| metrics.spl_sum).sum::<f32>() / n as f32,
        deep_successes: runs
            .iter()
            .map(|metrics| metrics.deep_successes)
            .sum::<usize>()
            / n,
    }
}
