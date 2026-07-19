//! Intervención causal pareada: ¿la memoria térmica CDT cambia un plan cuando
//! las relaciones RQM y el conocimiento del mundo son idénticos?

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
const REPORT_PATH: &str = "data/native_cognitive_rqm_cdt_report.json";
const THERMAL_SLEEP_PULSES: usize = 72;
const CORRECT_ACTION: LogisticsAction = LogisticsAction::Move(Location(4));
const DECOY_ACTION: LogisticsAction = LogisticsAction::Move(Location(1));

#[derive(Clone, Copy, Debug, Default)]
struct ArmMetrics {
    successes: usize,
    valid: usize,
    emitted: usize,
}

impl ArmMetrics {
    fn success_rate(self, seeds: usize) -> f32 {
        self.successes as f32 / seeds.max(1) as f32
    }

    fn validity(self) -> f32 {
        self.valid as f32 / self.emitted.max(1) as f32
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seeds = std::env::var("COGNITIVE_CAUSAL_SEEDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SEEDS)
        .max(1);
    let task = branch_task();
    let mut before_sleep = ArmMetrics::default();
    let mut full = ArmMetrics::default();
    let mut rqm_only = ArmMetrics::default();
    let mut no_prior = ArmMetrics::default();
    let mut inverted = ArmMetrics::default();
    let mut correct_signal_gain = 0.0f32;
    let mut relation_mismatches = 0usize;

    println!(
        "RQM/CDT paired causal experiment seeds={} task_steps={} intervention=verified_primitive_action",
        seeds, task.max_steps
    );
    for index in 0..seeds {
        let seed =
            0xCD70_CA05_u64.wrapping_add((index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let base = equal_rqm_controller(seed, &task);
        score_arm(base.clone(), &task, &mut before_sleep);

        let mut full_controller = base.clone();
        let signal_before =
            full_controller.action_schema_thermal_signal(&task.initial, task.goal, CORRECT_ACTION)
                - full_controller.action_schema_thermal_signal(
                    &task.initial,
                    task.goal,
                    DECOY_ACTION,
                );
        full_controller.incubate_action_schema_thermal(
            &task.initial,
            task.goal,
            CORRECT_ACTION,
            1.0,
            THERMAL_SLEEP_PULSES,
        );
        let signal_after =
            full_controller.action_schema_thermal_signal(&task.initial, task.goal, CORRECT_ACTION)
                - full_controller.action_schema_thermal_signal(
                    &task.initial,
                    task.goal,
                    DECOY_ACTION,
                );
        correct_signal_gain += signal_after - signal_before;

        let mut rqm_controller = full_controller.clone();
        rqm_controller.substrate.config.thermal_score_gain = 0.0;
        let mut no_prior_controller = full_controller.clone();
        no_prior_controller.config.procedural_gain = 0.0;
        let mut inverted_controller = base.clone();
        inverted_controller.incubate_action_schema_thermal(
            &task.initial,
            task.goal,
            DECOY_ACTION,
            1.0,
            THERMAL_SLEEP_PULSES,
        );

        let relation_count = full_controller.substrate.relation_count();
        relation_mismatches += usize::from(
            rqm_controller.substrate.relation_count() != relation_count
                || no_prior_controller.substrate.relation_count() != relation_count
                || inverted_controller.substrate.relation_count() != relation_count,
        );
        let full_success = score_arm(full_controller, &task, &mut full);
        let rqm_success = score_arm(rqm_controller, &task, &mut rqm_only);
        let no_prior_success = score_arm(no_prior_controller, &task, &mut no_prior);
        let inverted_success = score_arm(inverted_controller, &task, &mut inverted);
        println!(
            "seed={} full={} rqm_only={} no_prior={} inverted={} thermal_delta={:.4}",
            index + 1,
            full_success,
            rqm_success,
            no_prior_success,
            inverted_success,
            signal_after - signal_before,
        );
    }

    let full_rate = full.success_rate(seeds);
    let before_rate = before_sleep.success_rate(seeds);
    let rqm_rate = rqm_only.success_rate(seeds);
    let no_prior_rate = no_prior.success_rate(seeds);
    let inverted_rate = inverted.success_rate(seeds);
    let paired_advantage = full_rate - rqm_rate.max(no_prior_rate);
    let mean_signal_gain = correct_signal_gain / seeds as f32;
    let passed = full_rate >= 0.80
        && full.validity() >= 0.999
        && paired_advantage >= 0.60
        && before_rate <= 0.20
        && inverted_rate <= 0.20
        && mean_signal_gain > 0.10
        && relation_mismatches == 0;
    let decision = if passed {
        "rqm_cdt_causal_planning_pass"
    } else {
        "rqm_cdt_causal_planning_needs_tuning"
    };
    println!(
        "summary before={:.1}% full={:.1}% rqm_only={:.1}% no_prior={:.1}% inverted={:.1}% advantage={:.1}pp validity={:.1}% signal_gain={:.4} relation_mismatches={} decision={}",
        before_rate * 100.0,
        full_rate * 100.0,
        rqm_rate * 100.0,
        no_prior_rate * 100.0,
        inverted_rate * 100.0,
        paired_advantage * 100.0,
        full.validity() * 100.0,
        mean_signal_gain,
        relation_mismatches,
        decision,
    );
    let report = json!({
        "schema": "native_cognitive_rqm_cdt_causal_v1",
        "decision": decision,
        "seeds": seeds,
        "task": {
            "steps": task.max_steps,
            "complete_plan_seen_during_training": false,
            "verified_primitive_action_used_for_thermal_sleep": true,
            "thermal_sleep_pulses": THERMAL_SLEEP_PULSES,
        },
        "controls": {
            "same_rqm_relations": relation_mismatches == 0,
            "same_world_model": true,
            "same_planner": true,
            "oracle_passed_to_inference": false,
        },
        "arms": {
            "before_sleep_success": before_rate,
            "rqm_plus_cdt_success": full_rate,
            "rqm_without_cdt_success": rqm_rate,
            "without_procedural_prior_success": no_prior_rate,
            "inverted_thermal_memory_success": inverted_rate,
        },
        "paired_advantage": paired_advantage,
        "mean_correct_action_thermal_signal_gain": mean_signal_gain,
        "validity": full.validity(),
    });
    if let Some(parent) = std::path::Path::new(REPORT_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(REPORT_PATH, serde_json::to_vec_pretty(&report)?)?;
    println!("report={REPORT_PATH}");
    if !passed {
        return Err("la intervención causal RQM/CDT no alcanzó el gate".into());
    }
    Ok(())
}

fn equal_rqm_controller(seed: u64, task: &LogisticsTask) -> LogisticsController {
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
            max_candidates: 64,
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
    // Ambas alternativas reciben exactamente la misma evidencia RQM. El plan
    // completo nunca se entrena; solo dos decisiones primitivas competidoras.
    for round in 0..4 {
        let actions = if round % 2 == 0 {
            [DECOY_ACTION, CORRECT_ACTION]
        } else {
            [CORRECT_ACTION, DECOY_ACTION]
        };
        for action in actions {
            let before = task.initial.clone();
            let after = before.apply(action).unwrap();
            controller.observe_for_goal(
                PrimitiveEpisode {
                    before,
                    action,
                    after,
                    reward: 1.0,
                },
                task.goal,
            );
        }
    }
    controller
}

fn score_arm(
    mut controller: LogisticsController,
    task: &LogisticsTask,
    metrics: &mut ArmMetrics,
) -> bool {
    let decision = controller.plan(task);
    let Some(plan) = decision.plan else {
        return false;
    };
    metrics.emitted += 1;
    let verification = LogisticsController::verify(task, &plan);
    metrics.valid += usize::from(verification.actions_valid);
    let success = verification.actions_valid && verification.goal_reached;
    metrics.successes += usize::from(success);
    success
}

fn branch_task() -> LogisticsTask {
    let mut initial = LogisticsState {
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
    initial.canonicalize();
    LogisticsTask {
        id: "causal-thermal-branch".into(),
        initial,
        goal: LogisticsGoal {
            package: Package(0),
            destination: Location(3),
        },
        max_steps: 4,
    }
}
