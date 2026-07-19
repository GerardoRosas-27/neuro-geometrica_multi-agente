//! Entrenamiento desde cero y validación del sistema operativo cognitivo.
//!
//! No usa Transformer: prueba primero que CDT-RQM-EPR pueda recordar, explorar,
//! decidir y corregirse. Un Transformer puede conectarse después como traductor
//! de texto <-> `CognitiveTask` / `CognitiveEpisode`.

use cdt_rqm_epr::cognitive_os::{
    CognitiveMetrics, CognitiveOperatingSystem, CognitiveOsConfig, CognitiveRelation,
    CognitiveTask, CognitiveVerifier,
};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const REPORT_PATH: &str = "data/native_cognitive_os_report.json";
const MAX_SLEEP_CYCLES: usize = 8;
const REPLAY_PASSES: usize = 3;

#[derive(Default)]
struct WorldVerifier {
    traces: HashMap<String, Vec<String>>,
}

impl WorldVerifier {
    fn insert(&mut self, task_id: &str, trace: [&str; 4]) {
        self.traces.insert(
            task_id.to_string(),
            trace.into_iter().map(str::to_string).collect(),
        );
    }
}

impl CognitiveVerifier for WorldVerifier {
    fn expected_trace(&self, task: &CognitiveTask) -> Option<Vec<String>> {
        self.traces.get(&task.id).cloned()
    }
}

#[derive(Clone, Copy)]
struct ScoreBoard {
    adaptation: CognitiveMetrics,
    retention: CognitiveMetrics,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let (mut os, verifier, adaptation_tasks, retention_tasks) = build_from_scratch();
    let initial = evaluate(&os, &verifier, &adaptation_tasks, &retention_tasks);

    println!("Native Cognitive OS — entrenamiento desde cero");
    print_score("wake_baseline", initial, &os);

    // Control de destrucción: relajación térmica sin feedback semántico.
    let mut relaxation_ablation = os.clone();
    relaxation_ablation.relax_without_feedback(MAX_SLEEP_CYCLES * 8);
    let relaxation_score = evaluate(
        &relaxation_ablation,
        &verifier,
        &adaptation_tasks,
        &retention_tasks,
    );
    print_score(
        "relaxation_only_ablation",
        relaxation_score,
        &relaxation_ablation,
    );

    let mut accepted_cycles = 0usize;
    for cycle in 1..=MAX_SLEEP_CYCLES {
        let (_, feedback_episodes) = os.evaluate(&adaptation_tasks, &verifier);
        let before = evaluate(&os, &verifier, &adaptation_tasks, &retention_tasks);
        let mut candidate = os.clone();
        candidate.dream_with_feedback(&feedback_episodes, &verifier, REPLAY_PASSES);
        candidate.restore_thermal_homeostasis_from(&os);
        let after = evaluate(&candidate, &verifier, &adaptation_tasks, &retention_tasks);
        let preserves_retention =
            after.retention.accuracy() + 1.0e-6 >= before.retention.accuracy();
        let improves_adaptation =
            after.adaptation.accuracy() > before.adaptation.accuracy() + 1.0e-6;
        let preserves_adaptation =
            after.adaptation.accuracy() + 1.0e-6 >= before.adaptation.accuracy();
        let converged = after.adaptation.accuracy() >= 0.999;
        let accept =
            preserves_retention && preserves_adaptation && (improves_adaptation || converged);
        println!(
            "cycle={} accept={} improved={} adaptation={:.1}%->{:.1}% retention={:.1}%->{:.1}% episodes={} relations={} epr={}",
            cycle,
            accept,
            improves_adaptation,
            before.adaptation.accuracy() * 100.0,
            after.adaptation.accuracy() * 100.0,
            before.retention.accuracy() * 100.0,
            after.retention.accuracy() * 100.0,
            feedback_episodes.len(),
            candidate.substrate.relation_count(),
            candidate.substrate.entanglement.active_count(),
        );
        if accept {
            os = candidate;
            accepted_cycles += 1;
        }
        if converged && preserves_retention {
            break;
        }
    }

    let final_score = evaluate(&os, &verifier, &adaptation_tasks, &retention_tasks);
    print_score("verified_sleep_final", final_score, &os);
    print_examples(&os, &verifier, &adaptation_tasks);

    let sleep_gain = final_score.adaptation.accuracy() - initial.adaptation.accuracy();
    let ablation_gain = relaxation_score.adaptation.accuracy() - initial.adaptation.accuracy();
    let passed = final_score.adaptation.accuracy() >= 0.90
        && final_score.retention.accuracy() >= 0.99
        && sleep_gain >= ablation_gain + 0.25
        && accepted_cycles > 0;
    let decision = if passed {
        "cognitive_os_closed_loop_pass"
    } else {
        "cognitive_os_closed_loop_needs_tuning"
    };

    let report = json!({
        "schema": "native_cognitive_os_report_v1",
        "decision": decision,
        "trained_from_scratch": true,
        "transformer_role": "linguistic_peripheral_not_used_in_core_validation",
        "inference_receives_ground_truth": false,
        "accepted_sleep_cycles": accepted_cycles,
        "replay_passes_per_cycle": REPLAY_PASSES,
        "adaptation": {
            "cases": adaptation_tasks.len(),
            "initial_accuracy": initial.adaptation.accuracy(),
            "relaxation_only_accuracy": relaxation_score.adaptation.accuracy(),
            "verified_sleep_accuracy": final_score.adaptation.accuracy(),
            "sleep_gain": sleep_gain,
            "relaxation_gain": ablation_gain,
        },
        "retention": {
            "cases": retention_tasks.len(),
            "initial_accuracy": initial.retention.accuracy(),
            "verified_sleep_accuracy": final_score.retention.accuracy(),
        },
        "substrate": {
            "nodes": os.substrate.thermal.node_count(),
            "relations": os.substrate.relation_count(),
            "epr_links": os.substrate.entanglement.active_count(),
            "semantic_facts": os.facts().len(),
            "episodic_memories": os.episodic_memory().len(),
        },
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    if let Some(parent) = std::path::Path::new(REPORT_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(REPORT_PATH, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "decision={} report={} elapsed_ms={:.1}",
        decision,
        REPORT_PATH,
        started.elapsed().as_secs_f64() * 1_000.0
    );
    if !passed {
        return Err("la validación cognitiva no alcanzó el criterio".into());
    }
    Ok(())
}

fn build_from_scratch() -> (
    CognitiveOperatingSystem,
    WorldVerifier,
    Vec<CognitiveTask>,
    Vec<CognitiveTask>,
) {
    let thermal = NativeThermoCdtConfig {
        slices: 6,
        // Capacidad holgada para que las asambleas semánticas no compartan
        // nodos por accidente en este test de lógica.
        nodes_per_slice: 768,
        spatial_degree: 4,
        temporal_degree: 2,
        temperature: 0.20,
        dt: 0.010,
        diffusion: 0.14,
        confinement: 0.055,
        pilot_gain: 0.38,
        phase_coupling: 0.12,
        amplitude_decay: 0.003,
        seed: 0xC09A_171E_05,
        ..NativeThermoCdtConfig::default()
    };
    let rqm = NativeThermoRqmConfig {
        amplitude_learning_rate: 0.12,
        coherence_learning_rate: 0.10,
        uncertainty_learning_rate: 0.12,
        phase_learning_rate: 0.18,
        amplitude_decay: 0.0005,
        thermal_steps_per_train: 1,
        thermal_steps_per_query: 2,
        thermal_score_gain: 0.15,
        // El CDT interviene solo en empates reales. Un margen amplio convertía
        // consultas ya resueltas en muestreo ruidoso y degradaba retención.
        thermal_activation_margin: 0.001,
        collect_query_diagnostics: false,
        max_candidates: 128,
        max_pilot_window_nodes: 128,
        ..NativeThermoRqmConfig::default()
    };
    let epr = EntanglementConfig {
        max_links_per_node: 8,
        // Los enlaces EPR se consolidan como memoria, pero no expanden el pool
        // durante esta prueba lógica tipada. La expansión no respeta ObserverId
        // y mezclaba predicados no relacionados durante sueño.
        max_syncs_per_step: 0,
        create_threshold: 1.5,
        ..EntanglementConfig::default()
    };
    let substrate = NativeThermoRqmEprSubstrate::new(thermal, rqm, epr);
    let mut os = CognitiveOperatingSystem::new(
        substrate,
        CognitiveOsConfig {
            beam_width: 5,
            alternatives_per_step: 5,
            sleep_replay_strength: 1.0,
            failed_path_attenuation: 0.90,
            ..CognitiveOsConfig::default()
        },
    );

    let worlds = [
        ("ada", "redwood", "france", "paris"),
        ("bruno", "azure", "mexico", "mexico-city"),
        ("carla", "amber", "japan", "tokyo"),
        ("diego", "jade", "canada", "ottawa"),
        ("elena", "silver", "italy", "rome"),
        ("farah", "violet", "egypt", "cairo"),
        ("gita", "copper", "india", "new-delhi"),
        ("hugo", "onyx", "peru", "lima"),
        ("iris", "pearl", "spain", "madrid"),
        ("jon", "coral", "chile", "santiago"),
    ];
    let false_worlds = [
        ("ada", "false-a", "norway", "oslo"),
        ("bruno", "false-b", "greece", "athens"),
        ("carla", "false-c", "austria", "vienna"),
        ("diego", "false-d", "ireland", "dublin"),
        ("elena", "false-e", "portugal", "lisbon"),
        ("farah", "false-f", "finland", "helsinki"),
    ];

    let mut verifier = WorldVerifier::default();
    let mut adaptation_tasks = Vec::new();
    let mut retention_tasks = Vec::new();
    for (index, &(person, team, country, capital)) in worlds.iter().enumerate() {
        // Se memorizan hechos atómicos, nunca la ruta compuesta completa.
        let member_strength = if index < false_worlds.len() {
            0.35
        } else {
            1.0
        };
        os.remember_fact(person, CognitiveRelation::MemberOf, team, member_strength);
        os.remember_fact(team, CognitiveRelation::BasedIn, country, 1.0);
        os.remember_fact(country, CognitiveRelation::CapitalOf, capital, 1.0);
        let id = format!("{person}-capital");
        verifier.insert(&id, [person, team, country, capital]);
        let task = CognitiveTask {
            id,
            start: person.to_string(),
            program: vec![
                CognitiveRelation::MemberOf,
                CognitiveRelation::BasedIn,
                CognitiveRelation::CapitalOf,
            ],
        };
        if index < false_worlds.len() {
            adaptation_tasks.push(task);
        } else {
            retention_tasks.push(task);
        }
    }
    // Experiencias ruidosas plausibles dominan inicialmente seis decisiones.
    // El verificador externo las corregirá durante sueño; inferencia no las conoce.
    for &(person, false_team, false_country, false_capital) in &false_worlds {
        os.remember_fact(person, CognitiveRelation::MemberOf, false_team, 1.0);
        os.remember_fact(false_team, CognitiveRelation::BasedIn, false_country, 1.0);
        os.remember_fact(
            false_country,
            CognitiveRelation::CapitalOf,
            false_capital,
            1.0,
        );
    }
    (os, verifier, adaptation_tasks, retention_tasks)
}

fn evaluate(
    os: &CognitiveOperatingSystem,
    verifier: &WorldVerifier,
    adaptation_tasks: &[CognitiveTask],
    retention_tasks: &[CognitiveTask],
) -> ScoreBoard {
    ScoreBoard {
        adaptation: os.evaluate(adaptation_tasks, verifier).0,
        retention: os.evaluate(retention_tasks, verifier).0,
    }
}

fn print_score(label: &str, score: ScoreBoard, os: &CognitiveOperatingSystem) {
    println!(
        "{} adaptation_acc={:.1}% coverage={:.1}% confidence={:.3} retention_acc={:.1}% relations={} epr={} energy={:.5}",
        label,
        score.adaptation.accuracy() * 100.0,
        score.adaptation.coverage() * 100.0,
        score.adaptation.mean_confidence(),
        score.retention.accuracy() * 100.0,
        os.substrate.relation_count(),
        os.substrate.entanglement.active_count(),
        os.substrate.thermal.report().mean_energy,
    );
}

fn print_examples(
    os: &CognitiveOperatingSystem,
    verifier: &WorldVerifier,
    tasks: &[CognitiveTask],
) {
    let (_, episodes) = os.evaluate(tasks, verifier);
    for episode in &episodes {
        let route = episode
            .working_memory
            .steps
            .iter()
            .map(|step| step.chosen.as_str())
            .collect::<Vec<_>>()
            .join(" -> ");
        println!(
            "trace task={} start={} route={} answer={:?} verified={:?}",
            episode.task.id, episode.task.start, route, episode.answer, episode.verified,
        );
        if episode.verified == Some(false) {
            for (index, step) in episode.working_memory.steps.iter().enumerate() {
                let alternatives = step
                    .alternatives
                    .iter()
                    .map(|item| format!("{}:{:.5}", item.entity, item.score))
                    .collect::<Vec<_>>()
                    .join(",");
                println!(
                    "  step={} relation={:?} from={} chosen={} alternatives=[{}]",
                    index + 1,
                    step.relation,
                    step.from,
                    step.chosen,
                    alternatives,
                );
            }
        }
    }
}

#[allow(dead_code)]
fn _assert_substrate_is_native(_: &NativeThermoCdtSubstrate) {}
