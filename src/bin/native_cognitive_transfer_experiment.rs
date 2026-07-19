//! Transferencia zero-shot de memoria procedural térmica a IDs y topologías
//! no observados durante entrenamiento.

use cdt_rqm_epr::cognitive_logistics::{
    Location, LogisticsAction, LogisticsController, LogisticsGoal, LogisticsPlannerConfig,
    LogisticsState, LogisticsTask, Package, PrimitiveEpisode,
};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use serde_json::json;
use std::fs;

const DEFAULT_SEEDS: usize = 30;
const THERMAL_SLEEP_PULSES: usize = 72;
const REPORT_PATH: &str = "data/native_cognitive_transfer_report.json";

#[derive(Clone, Copy, Default)]
struct Arm {
    successes: usize,
    valid: usize,
    emitted: usize,
}

impl Arm {
    fn success_rate(self, cases: usize) -> f32 {
        self.successes as f32 / cases.max(1) as f32
    }

    fn validity(self) -> f32 {
        self.valid as f32 / self.emitted.max(1) as f32
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seeds = std::env::var("COGNITIVE_TRANSFER_SEEDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SEEDS)
        .max(1);
    let dev = branch_task(0, Package(0), false);
    let tests = [
        branch_task(10, Package(1), false),
        branch_task(20, Package(2), true),
    ];
    let mut full = Arm::default();
    let mut rqm_only = Arm::default();
    let mut no_prior = Arm::default();
    let mut inverted = Arm::default();
    let mut before_sleep = Arm::default();
    let mut signal_gain_sum = 0.0f32;
    let mut relation_mismatches = 0usize;
    let total_cases = seeds * tests.len();

    println!(
        "Cognitive schema transfer seeds={} test_topologies={} train_ids=0..5 test_ids=10..26 packages=1,2",
        seeds,
        tests.len(),
    );
    for seed_index in 0..seeds {
        let seed = 0x7A4A_5FE2_u64
            .wrapping_add((seed_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let base = equal_schema_controller(seed, &dev);
        for test in &tests {
            score(base.clone(), test, &mut before_sleep);
        }

        let dev_correct = correct_action(&dev);
        let dev_decoy = decoy_action(&dev);
        let mut full_controller = base.clone();
        let before_signal =
            full_controller.action_schema_thermal_signal(&dev.initial, dev.goal, dev_correct)
                - full_controller.action_schema_thermal_signal(&dev.initial, dev.goal, dev_decoy);
        full_controller.incubate_action_schema_thermal(
            &dev.initial,
            dev.goal,
            dev_correct,
            1.0,
            THERMAL_SLEEP_PULSES,
        );
        let after_signal =
            full_controller.action_schema_thermal_signal(&dev.initial, dev.goal, dev_correct)
                - full_controller.action_schema_thermal_signal(&dev.initial, dev.goal, dev_decoy);
        signal_gain_sum += after_signal - before_signal;

        let mut rqm_controller = full_controller.clone();
        rqm_controller.substrate.config.thermal_score_gain = 0.0;
        let mut no_prior_controller = full_controller.clone();
        no_prior_controller.config.procedural_gain = 0.0;
        let mut inverted_controller = base.clone();
        inverted_controller.incubate_action_schema_thermal(
            &dev.initial,
            dev.goal,
            dev_decoy,
            1.0,
            THERMAL_SLEEP_PULSES,
        );
        let relations = full_controller.substrate.relation_count();
        relation_mismatches += usize::from(
            rqm_controller.substrate.relation_count() != relations
                || no_prior_controller.substrate.relation_count() != relations
                || inverted_controller.substrate.relation_count() != relations,
        );

        let mut seed_full = 0usize;
        for test in &tests {
            seed_full += usize::from(score(full_controller.clone(), test, &mut full));
            score(rqm_controller.clone(), test, &mut rqm_only);
            score(no_prior_controller.clone(), test, &mut no_prior);
            score(inverted_controller.clone(), test, &mut inverted);
        }
        println!(
            "seed={} transferred={}/{} schema_signal_gain={:.4}",
            seed_index + 1,
            seed_full,
            tests.len(),
            after_signal - before_signal,
        );
    }

    let full_rate = full.success_rate(total_cases);
    let rqm_rate = rqm_only.success_rate(total_cases);
    let no_prior_rate = no_prior.success_rate(total_cases);
    let inverted_rate = inverted.success_rate(total_cases);
    let before_rate = before_sleep.success_rate(total_cases);
    let advantage = full_rate - rqm_rate.max(no_prior_rate);
    let signal_gain = signal_gain_sum / seeds as f32;
    let passed = full_rate >= 0.80
        && full.validity() >= 0.999
        && advantage >= 0.60
        && before_rate <= 0.20
        && inverted_rate <= 0.20
        && signal_gain > 0.10
        && relation_mismatches == 0;
    let decision = if passed {
        "cognitive_schema_transfer_pass"
    } else {
        "cognitive_schema_transfer_needs_tuning"
    };
    println!(
        "summary before={:.1}% transfer={:.1}% rqm_only={:.1}% no_prior={:.1}% inverted={:.1}% advantage={:.1}pp validity={:.1}% signal_gain={:.4} decision={}",
        before_rate * 100.0,
        full_rate * 100.0,
        rqm_rate * 100.0,
        no_prior_rate * 100.0,
        inverted_rate * 100.0,
        advantage * 100.0,
        full.validity() * 100.0,
        signal_gain,
        decision,
    );

    let report = json!({
        "schema": "native_cognitive_schema_transfer_v1",
        "decision": decision,
        "seeds": seeds,
        "test_topologies": tests.len(),
        "test_cases": total_cases,
        "disjoint_grounded_ids": true,
        "train_package_id": 0,
        "test_package_ids": [1, 2],
        "complete_test_plans_seen_in_training": false,
        "verified_dev_primitive_used_in_sleep": true,
        "same_rqm_relations_between_arms": relation_mismatches == 0,
        "oracle_passed_to_inference": false,
        "arms": {
            "before_sleep": before_rate,
            "schema_rqm_plus_cdt": full_rate,
            "schema_rqm_without_cdt": rqm_rate,
            "without_procedural_prior": no_prior_rate,
            "inverted_schema_thermal_memory": inverted_rate,
        },
        "paired_advantage": advantage,
        "validity": full.validity(),
        "mean_schema_thermal_signal_gain": signal_gain,
        "thermal_sleep_pulses": THERMAL_SLEEP_PULSES,
    });
    if let Some(parent) = std::path::Path::new(REPORT_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(REPORT_PATH, serde_json::to_vec_pretty(&report)?)?;
    println!("report={REPORT_PATH}");
    if !passed {
        return Err("la transferencia de esquemas no alcanzó el gate".into());
    }
    Ok(())
}

fn equal_schema_controller(seed: u64, dev: &LogisticsTask) -> LogisticsController {
    let substrate = NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 6,
            nodes_per_slice: 384,
            temperature: 0.08,
            dt: 0.008,
            diffusion: 0.10,
            confinement: 0.06,
            pilot_gain: 0.45,
            phase_coupling: 0.10,
            amplitude_decay: 0.001,
            seed,
            ..NativeThermoCdtConfig::default()
        },
        NativeThermoRqmConfig {
            thermal_steps_per_train: 1,
            thermal_steps_per_query: 2,
            thermal_score_gain: 1.50,
            thermal_activation_margin: f32::MAX,
            collect_query_diagnostics: false,
            max_candidates: 96,
            ..NativeThermoRqmConfig::default()
        },
        EntanglementConfig {
            max_syncs_per_step: 0,
            create_threshold: 2.0,
            ..EntanglementConfig::default()
        },
    );
    let mut controller = LogisticsController::new(
        substrate,
        LogisticsPlannerConfig {
            beam_width: 1,
            procedural_gain: 12.0,
            max_expansions: 128,
            ..LogisticsPlannerConfig::default()
        },
    );
    for round in 0..5 {
        let actions = if round % 2 == 0 {
            [decoy_action(dev), correct_action(dev)]
        } else {
            [correct_action(dev), decoy_action(dev)]
        };
        for action in actions {
            let before = dev.initial.clone();
            let after = before.apply(action).unwrap();
            controller.observe_for_goal(
                PrimitiveEpisode {
                    before,
                    action,
                    after,
                    reward: 1.0,
                },
                dev.goal,
            );
        }
    }
    controller
}

fn score(mut controller: LogisticsController, task: &LogisticsTask, arm: &mut Arm) -> bool {
    let decision = controller.plan(task);
    let Some(plan) = decision.plan else {
        return false;
    };
    arm.emitted += 1;
    let verification = LogisticsController::verify(task, &plan);
    arm.valid += usize::from(verification.actions_valid);
    let success = verification.actions_valid && verification.goal_reached;
    arm.successes += usize::from(success);
    success
}

fn correct_action(task: &LogisticsTask) -> LogisticsAction {
    LogisticsAction::Move(Location(task.initial.robot_at.0 + 4))
}

fn decoy_action(task: &LogisticsTask) -> LogisticsAction {
    LogisticsAction::Move(Location(task.initial.robot_at.0 + 1))
}

fn branch_task(offset: u8, package: Package, longer_detour: bool) -> LogisticsTask {
    let start = Location(offset);
    let decoy = Location(offset + 1);
    let goal_location = Location(offset + 3);
    let detour_a = Location(offset + 4);
    let detour_b = Location(offset + 5);
    let mut connections = vec![
        (start, decoy),
        (decoy, goal_location),
        (start, detour_a),
        (detour_a, detour_b),
        (detour_b, goal_location),
    ];
    let max_steps = if longer_detour {
        let detour_c = Location(offset + 6);
        connections.retain(|edge| *edge != (detour_b, goal_location));
        connections.push((detour_b, detour_c));
        connections.push((detour_c, goal_location));
        5
    } else {
        4
    };
    let mut package_at = vec![None; package.0 as usize + 1];
    package_at[package.0 as usize] = None;
    let mut initial = LogisticsState {
        robot_at: start,
        package_at,
        carrying: Some(package),
        has_key: false,
        connections,
        locked_edges: vec![(decoy, goal_location)],
    };
    initial.canonicalize();
    LogisticsTask {
        id: format!("transfer-{}-{}", offset, package.0),
        initial,
        goal: LogisticsGoal {
            package,
            destination: goal_location,
        },
        max_steps,
    }
}
