//! Inducción automática de esquemas de efecto y transferencia OOD a grafos
//! no isomorfos con entidades completamente disjuntas.

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
const REPORT_PATH: &str = "data/native_cognitive_ood_schema_report.json";

#[derive(Clone, Copy, Default)]
struct Arm {
    success: usize,
    valid: usize,
    emitted: usize,
}

impl Arm {
    fn rate(self, total: usize) -> f32 {
        self.success as f32 / total.max(1) as f32
    }

    fn validity(self) -> f32 {
        self.valid as f32 / self.emitted.max(1) as f32
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seeds = std::env::var("COGNITIVE_OOD_SEEDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SEEDS)
        .max(1);
    let train = training_topology();
    let tests = [tree_topology(), ring_topology(), mesh_topology()];
    let total = seeds * tests.len();
    let mut full = Arm::default();
    let mut before = Arm::default();
    let mut rqm_only = Arm::default();
    let mut schemas_off = Arm::default();
    let mut inverted = Arm::default();
    let mut learned_schema_min = usize::MAX;
    let mut learned_schema_max = 0usize;
    let mut relation_mismatches = 0usize;
    let mut signal_gain_sum = 0.0f32;
    let mut topology_successes = [0usize; 3];

    println!(
        "Automatic OOD schema experiment seeds={} train_graph=branch test_graphs=tree,ring,mesh grounded_ids_disjoint=true",
        seeds
    );
    for seed_index in 0..seeds {
        let seed = 0xA070_5C4E_u64
            .wrapping_add((seed_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let base = learned_controller(seed, &train);
        learned_schema_min = learned_schema_min.min(base.learned_schema_count());
        learned_schema_max = learned_schema_max.max(base.learned_schema_count());
        for test in &tests {
            score(base.clone(), test, &mut before);
        }

        let correct = correct_action(&train);
        let decoy = decoy_action(&train);
        let mut full_controller = base.clone();
        let signal_before = full_controller
            .learned_schema_thermal_signal(&train.initial, train.goal, correct)
            .unwrap_or_default()
            - full_controller
                .learned_schema_thermal_signal(&train.initial, train.goal, decoy)
                .unwrap_or_default();
        let incubated = full_controller.incubate_learned_schema_thermal(
            &train.initial,
            train.goal,
            correct,
            1.0,
            THERMAL_SLEEP_PULSES,
        );
        let signal_after = full_controller
            .learned_schema_thermal_signal(&train.initial, train.goal, correct)
            .unwrap_or_default()
            - full_controller
                .learned_schema_thermal_signal(&train.initial, train.goal, decoy)
                .unwrap_or_default();
        signal_gain_sum += signal_after - signal_before;

        let mut rqm_controller = full_controller.clone();
        rqm_controller.substrate.config.thermal_score_gain = 0.0;
        let mut schemas_off_controller = full_controller.clone();
        schemas_off_controller.config.use_learned_schemas = false;
        let mut inverted_controller = base.clone();
        let inverted_incubated = inverted_controller.incubate_learned_schema_thermal(
            &train.initial,
            train.goal,
            decoy,
            1.0,
            THERMAL_SLEEP_PULSES,
        );
        let relations = full_controller.substrate.relation_count();
        relation_mismatches += usize::from(
            rqm_controller.substrate.relation_count() != relations
                || schemas_off_controller.substrate.relation_count() != relations
                || inverted_controller.substrate.relation_count() != relations,
        );

        let mut transferred = 0usize;
        for (topology_index, test) in tests.iter().enumerate() {
            let success = score(full_controller.clone(), test, &mut full);
            transferred += usize::from(success);
            topology_successes[topology_index] += usize::from(success);
            score(rqm_controller.clone(), test, &mut rqm_only);
            score(schemas_off_controller.clone(), test, &mut schemas_off);
            score(inverted_controller.clone(), test, &mut inverted);
        }
        println!(
            "seed={} transferred={}/{} schemas={} incubated={} inverted_incubated={} signal_gain={:.4}",
            seed_index + 1,
            transferred,
            tests.len(),
            base.learned_schema_count(),
            incubated,
            inverted_incubated,
            signal_after - signal_before,
        );
    }

    let full_rate = full.rate(total);
    let before_rate = before.rate(total);
    let rqm_rate = rqm_only.rate(total);
    let schemas_off_rate = schemas_off.rate(total);
    let inverted_rate = inverted.rate(total);
    let advantage = full_rate - rqm_rate.max(schemas_off_rate);
    let signal_gain = signal_gain_sum / seeds as f32;
    let passed = full_rate >= 0.80
        && full.validity() >= 0.999
        && advantage >= 0.60
        && rqm_rate <= 0.20
        && schemas_off_rate <= 0.20
        && inverted_rate <= 0.20
        && learned_schema_min >= 2
        && learned_schema_max <= 4
        && signal_gain > 0.10
        && relation_mismatches == 0;
    let decision = if passed {
        "automatic_ood_schema_transfer_pass"
    } else {
        "automatic_ood_schema_transfer_needs_tuning"
    };
    println!(
        "summary before={:.1}% learned_rqm_cdt={:.1}% rqm_only={:.1}% schemas_off={:.1}% inverted={:.1}% advantage={:.1}pp validity={:.1}% topologies={:?} schemas={}..{} signal_gain={:.4} decision={}",
        before_rate * 100.0,
        full_rate * 100.0,
        rqm_rate * 100.0,
        schemas_off_rate * 100.0,
        inverted_rate * 100.0,
        advantage * 100.0,
        full.validity() * 100.0,
        topology_successes,
        learned_schema_min,
        learned_schema_max,
        signal_gain,
        decision,
    );

    let report = json!({
        "schema": "native_cognitive_automatic_ood_schema_v1",
        "decision": decision,
        "seeds": seeds,
        "train_distribution": "five-node locked branch",
        "test_distributions": ["deep tree", "cycle/ring", "meshed graph with side branches"],
        "test_cases": total,
        "grounded_ids_disjoint": true,
        "train_package": 0,
        "test_packages": [1, 2, 3],
        "handcrafted_action_schemas_enabled": false,
        "signature_induction": {
            "source": "observed before/after deltas",
            "dimensions": ["action_kind", "open_distance_delta", "carrying_delta", "goal_delta", "locked_edge_delta"],
            "learned_schema_min": learned_schema_min,
            "learned_schema_max": learned_schema_max,
        },
        "complete_test_plans_seen_in_training": false,
        "oracle_passed_to_inference": false,
        "same_rqm_relations_between_arms": relation_mismatches == 0,
        "arms": {
            "before_sleep": before_rate,
            "learned_schema_rqm_plus_cdt": full_rate,
            "learned_schema_rqm_without_cdt": rqm_rate,
            "learned_schemas_disabled": schemas_off_rate,
            "inverted_learned_thermal_memory": inverted_rate,
        },
        "topology_success": {
            "deep_tree": topology_successes[0] as f32 / seeds as f32,
            "cycle_ring": topology_successes[1] as f32 / seeds as f32,
            "meshed_graph": topology_successes[2] as f32 / seeds as f32,
        },
        "paired_advantage": advantage,
        "validity": full.validity(),
        "mean_thermal_signal_gain": signal_gain,
        "thermal_sleep_pulses": THERMAL_SLEEP_PULSES,
    });
    if let Some(parent) = std::path::Path::new(REPORT_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(REPORT_PATH, serde_json::to_vec_pretty(&report)?)?;
    println!("report={REPORT_PATH}");
    if !passed {
        return Err("la transferencia OOD automática no alcanzó el gate".into());
    }
    Ok(())
}

fn learned_controller(seed: u64, train: &LogisticsTask) -> LogisticsController {
    let substrate = NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 6,
            nodes_per_slice: 384,
            temperature: 0.02,
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
            procedural_gain: 14.0,
            max_expansions: 192,
            use_handcrafted_schemas: false,
            use_learned_schemas: true,
            ..LogisticsPlannerConfig::default()
        },
    );
    for round in 0..6 {
        let actions = if round % 2 == 0 {
            [decoy_action(train), correct_action(train)]
        } else {
            [correct_action(train), decoy_action(train)]
        };
        for action in actions {
            let before = train.initial.clone();
            let after = before.apply(action).unwrap();
            controller.observe_for_goal(
                PrimitiveEpisode {
                    before,
                    action,
                    after,
                    reward: 1.0,
                },
                train.goal,
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
    arm.success += usize::from(success);
    success
}

fn training_topology() -> LogisticsTask {
    task_from_graph(
        "train-branch",
        Package(0),
        Location(0),
        Location(1),
        Location(3),
        Location(4),
        vec![
            (Location(0), Location(1)),
            (Location(1), Location(3)),
            (Location(0), Location(4)),
            (Location(4), Location(5)),
            (Location(5), Location(3)),
        ],
        4,
    )
}

fn tree_topology() -> LogisticsTask {
    task_from_graph(
        "ood-tree",
        Package(1),
        Location(10),
        Location(11),
        Location(13),
        Location(14),
        vec![
            (Location(10), Location(11)),
            (Location(11), Location(13)),
            (Location(10), Location(14)),
            (Location(14), Location(15)),
            (Location(15), Location(16)),
            (Location(16), Location(13)),
        ],
        5,
    )
}

fn ring_topology() -> LogisticsTask {
    task_from_graph(
        "ood-ring",
        Package(2),
        Location(20),
        Location(21),
        Location(23),
        Location(24),
        vec![
            (Location(20), Location(21)),
            (Location(21), Location(23)),
            (Location(20), Location(24)),
            (Location(24), Location(25)),
            (Location(25), Location(26)),
            (Location(26), Location(23)),
            (Location(25), Location(27)),
            (Location(27), Location(24)),
        ],
        5,
    )
}

fn mesh_topology() -> LogisticsTask {
    task_from_graph(
        "ood-mesh",
        Package(3),
        Location(30),
        Location(31),
        Location(33),
        Location(34),
        vec![
            (Location(30), Location(31)),
            (Location(31), Location(33)),
            (Location(30), Location(34)),
            (Location(34), Location(35)),
            (Location(35), Location(36)),
            (Location(36), Location(37)),
            (Location(37), Location(33)),
            (Location(34), Location(38)),
            (Location(38), Location(39)),
            (Location(39), Location(35)),
            (Location(36), Location(40)),
            (Location(40), Location(38)),
        ],
        6,
    )
}

#[allow(clippy::too_many_arguments)]
fn task_from_graph(
    id: &str,
    package: Package,
    start: Location,
    decoy: Location,
    destination: Location,
    _correct_first: Location,
    connections: Vec<(Location, Location)>,
    max_steps: usize,
) -> LogisticsTask {
    let mut package_at = vec![None; package.0 as usize + 1];
    package_at[package.0 as usize] = None;
    let mut initial = LogisticsState {
        robot_at: start,
        package_at,
        carrying: Some(package),
        has_key: false,
        connections,
        locked_edges: vec![(decoy, destination)],
    };
    initial.canonicalize();
    LogisticsTask {
        id: id.into(),
        initial,
        goal: LogisticsGoal {
            package,
            destination,
        },
        max_steps,
    }
}

fn correct_action(task: &LogisticsTask) -> LogisticsAction {
    let start = task.initial.robot_at;
    let mut destinations = task
        .initial
        .connections
        .iter()
        .filter_map(|&(a, b)| {
            if a == start {
                Some(b)
            } else if b == start {
                Some(a)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    destinations.sort_unstable();
    LogisticsAction::Move(*destinations.last().unwrap())
}

fn decoy_action(task: &LogisticsTask) -> LogisticsAction {
    let start = task.initial.robot_at;
    let mut destinations = task
        .initial
        .connections
        .iter()
        .filter_map(|&(a, b)| {
            if a == start {
                Some(b)
            } else if b == start {
                Some(a)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    destinations.sort_unstable();
    LogisticsAction::Move(destinations[0])
}
