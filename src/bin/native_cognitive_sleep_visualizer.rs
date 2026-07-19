//! Escritorio del sistema operativo cognitivo.
//!
//! Ejecuta sueño infinito por fases, guarda un checkpoint al terminar cada
//! fase y representa CDT/RQM/EPR/esquemas aprendidos en 2D y 3D.

use cdt_rqm_epr::cognitive_logistics::{
    Location, LogisticsAction, LogisticsController, LogisticsControllerSnapshot, LogisticsGoal,
    LogisticsPlannerConfig, LogisticsState, LogisticsTask, Package, PrimitiveEpisode,
};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::native_checkpoint::{atomic_write, save_native_checkpoint_transactional};
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use cdt_rqm_epr::native_thermodynamic_cdt::{
    NativeCdtEdgeKind, NativeThermoCdtConfig, NativeThermoCdtSubstrate,
};
use cdt_rqm_epr::native_thermodynamic_engine::load_native_checkpoint;
use macroquad::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

const STATE_ROOT: &str = "data/native_cognitive_desktop";
const LATEST_MANIFEST: &str = "data/native_cognitive_desktop/latest.json";
const DEFAULT_PHASE_FRAMES: u64 = 240;
const DEFAULT_SNAPSHOT_RETENTION: usize = 60;
const MAX_2D_RELATIONS: usize = 1_600;
const MAX_3D_RELATIONS: usize = 1_200;
const MAX_3D_EPR: usize = 1_200;
const MAX_3D_CDT_EDGES: usize = 8_000;
const MAX_COGNITIVE_MACRONODES: usize = 96;
const FREE_ENERGY_HISTORY: usize = 240;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
enum CognitiveSleepPhase {
    Observe,
    SchemaInduction,
    ThermalConsolidation,
    OodExploration,
    Validation,
}

impl CognitiveSleepPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Observe => "OBSERVACIÓN WAKE",
            Self::SchemaInduction => "INDUCCIÓN DE ESQUEMAS",
            Self::ThermalConsolidation => "CONSOLIDACIÓN TÉRMICA",
            Self::OodExploration => "EXPLORACIÓN OOD",
            Self::Validation => "VALIDACIÓN Y GATE",
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::SchemaInduction => "schema",
            Self::ThermalConsolidation => "thermal",
            Self::OodExploration => "explore",
            Self::Validation => "validate",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Observe => Self::SchemaInduction,
            Self::SchemaInduction => Self::ThermalConsolidation,
            Self::ThermalConsolidation => Self::OodExploration,
            Self::OodExploration => Self::Validation,
            Self::Validation => Self::Observe,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DesktopManifest {
    schema_version: u32,
    cycle: u64,
    next_phase: CognitiveSleepPhase,
    checkpoint: String,
    cognitive_snapshot: String,
    completed_phase: CognitiveSleepPhase,
    accepted_cycles: u64,
    rejected_cycles: u64,
    total_phase_saves: u64,
    summary: String,
}

struct DesktopState {
    controller: LogisticsController,
    baseline: LogisticsController,
    cycle: u64,
    phase: CognitiveSleepPhase,
    phase_frame: u64,
    phase_frames: u64,
    accepted_cycles: u64,
    rejected_cycles: u64,
    total_phase_saves: u64,
    last_summary: String,
    last_save: String,
    last_error: Option<String>,
    current_plans: Vec<String>,
    ood_success: f32,
    ood_validity: f32,
    retention: usize,
    free_energy_history: VecDeque<f32>,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "CDT-RQM-EPR · Sistema Operativo Cognitivo".to_string(),
        window_width: 1440,
        window_height: 900,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut desktop = DesktopState::load_or_new();
    let mut paused = false;
    let mut view_3d = env_flag("COGNITIVE_DESKTOP_START_3D", true);
    let mut show_edges = true;
    let mut completed_this_run = 0u64;
    let exit_after = env_u64("COGNITIVE_DESKTOP_EXIT_AFTER_PHASES", 0);

    loop {
        if is_key_pressed(KeyCode::Escape) {
            let _ = desktop.force_save();
            break;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::Tab) {
            view_3d = !view_3d;
        }
        if is_key_pressed(KeyCode::E) {
            show_edges = !show_edges;
        }
        if is_key_pressed(KeyCode::S) {
            let _ = desktop.force_save();
        }

        if !paused {
            desktop.controller.substrate.thermal.step();
            desktop.record_free_energy();
            desktop.phase_frame = desktop.phase_frame.saturating_add(1);
            if desktop.phase_frame >= desktop.phase_frames {
                if desktop.complete_phase().is_ok() {
                    completed_this_run = completed_this_run.saturating_add(1);
                }
            }
        }

        clear_background(Color::from_rgba(6, 10, 20, 255));
        if view_3d {
            draw_cognitive_3d(&desktop, show_edges);
            set_default_camera();
        } else {
            draw_cognitive_2d(&desktop, show_edges);
        }
        draw_dashboard(&desktop, paused, view_3d, show_edges);
        draw_future_energy_monitor(&desktop);
        next_frame().await;

        if exit_after > 0 && completed_this_run >= exit_after {
            break;
        }
    }
}

impl DesktopState {
    fn load_or_new() -> Self {
        let phase_frames = env_u64("COGNITIVE_DESKTOP_PHASE_FRAMES", DEFAULT_PHASE_FRAMES).max(1);
        let retention = env_usize(
            "COGNITIVE_DESKTOP_SNAPSHOT_RETENTION",
            DEFAULT_SNAPSHOT_RETENTION,
        )
        .max(5);
        if let Ok(body) = fs::read(LATEST_MANIFEST) {
            if let Ok(manifest) = serde_json::from_slice::<DesktopManifest>(&body) {
                let restored = load_native_checkpoint(&manifest.checkpoint).and_then(|substrate| {
                    let body = fs::read(&manifest.cognitive_snapshot)
                        .map_err(|error| error.to_string())?;
                    let snapshot = serde_json::from_slice::<LogisticsControllerSnapshot>(&body)
                        .map_err(|error| error.to_string())?;
                    LogisticsController::from_snapshot(substrate, snapshot)
                });
                if let Ok(controller) = restored {
                    let initial_free_energy =
                        controller.substrate.thermal.report().free_energy_proxy;
                    return Self {
                        baseline: controller.clone(),
                        controller,
                        cycle: manifest.cycle,
                        phase: manifest.next_phase,
                        phase_frame: 0,
                        phase_frames,
                        accepted_cycles: manifest.accepted_cycles,
                        rejected_cycles: manifest.rejected_cycles,
                        total_phase_saves: manifest.total_phase_saves,
                        last_summary: format!("restaurado: {}", manifest.summary),
                        last_save: manifest.checkpoint,
                        last_error: None,
                        current_plans: Vec::new(),
                        ood_success: 0.0,
                        ood_validity: 0.0,
                        retention,
                        free_energy_history: VecDeque::from([initial_free_energy]),
                    };
                }
            }
        }
        let controller = fresh_controller();
        let initial_free_energy = controller.substrate.thermal.report().free_energy_proxy;
        Self {
            baseline: controller.clone(),
            controller,
            cycle: 1,
            phase: CognitiveSleepPhase::Observe,
            phase_frame: 0,
            phase_frames,
            accepted_cycles: 0,
            rejected_cycles: 0,
            total_phase_saves: 0,
            last_summary: "sustrato cognitivo nuevo".to_string(),
            last_save: "sin checkpoint".to_string(),
            last_error: None,
            current_plans: Vec::new(),
            ood_success: 0.0,
            ood_validity: 0.0,
            retention,
            free_energy_history: VecDeque::from([initial_free_energy]),
        }
    }

    fn record_free_energy(&mut self) {
        self.free_energy_history
            .push_back(self.controller.substrate.thermal.report().free_energy_proxy);
        while self.free_energy_history.len() > FREE_ENERGY_HISTORY {
            self.free_energy_history.pop_front();
        }
    }

    fn complete_phase(&mut self) -> Result<(), String> {
        let completed = self.phase;
        let before_phase = self.controller.clone();
        let mut next_cycle = self.cycle;
        let mut next_phase = completed.next();
        let summary = match completed {
            CognitiveSleepPhase::Observe => {
                observe_balanced_primitives(&mut self.controller, 2);
                format!(
                    "episodios={} relaciones={}",
                    self.controller.episodes().len(),
                    self.controller.substrate.relation_count()
                )
            }
            CognitiveSleepPhase::SchemaInduction => {
                observe_balanced_primitives(&mut self.controller, 1);
                format!(
                    "prototipos inducidos={} firmas manuales=false",
                    self.controller.learned_schema_count()
                )
            }
            CognitiveSleepPhase::ThermalConsolidation => {
                let train = training_task();
                let action = correct_action(&train);
                let accepted = self.controller.incubate_learned_schema_thermal(
                    &train.initial,
                    train.goal,
                    action,
                    1.0,
                    72,
                );
                format!(
                    "esquema térmico consolidado={} señal={:.4}",
                    accepted,
                    self.controller
                        .learned_schema_thermal_signal(&train.initial, train.goal, action)
                        .unwrap_or_default()
                )
            }
            CognitiveSleepPhase::OodExploration => {
                let (success, validity, plans) = evaluate_ood(&self.controller);
                self.ood_success = success;
                self.ood_validity = validity;
                self.current_plans = plans;
                format!(
                    "OOD success={:.1}% validez={:.1}% planes={}",
                    success * 100.0,
                    validity * 100.0,
                    self.current_plans.len()
                )
            }
            CognitiveSleepPhase::Validation => {
                let (success, validity, plans) = evaluate_ood(&self.controller);
                self.ood_success = success;
                self.ood_validity = validity;
                self.current_plans = plans;
                let accepted = success >= 0.66
                    && validity >= 0.999
                    && self
                        .controller
                        .substrate
                        .thermal
                        .report()
                        .mean_energy
                        .is_finite();
                if accepted {
                    self.accepted_cycles = self.accepted_cycles.saturating_add(1);
                } else {
                    self.controller = self.baseline.clone();
                    self.rejected_cycles = self.rejected_cycles.saturating_add(1);
                }
                next_cycle = self.cycle.saturating_add(1);
                next_phase = CognitiveSleepPhase::Observe;
                format!(
                    "gate={} success={:.1}% validez={:.1}%",
                    if accepted { "accept" } else { "rollback" },
                    success * 100.0,
                    validity * 100.0
                )
            }
        };

        match self.persist_phase(completed, next_cycle, next_phase, &summary) {
            Ok(path) => {
                self.last_summary = summary;
                self.last_save = path;
                self.last_error = None;
                self.phase = next_phase;
                self.cycle = next_cycle;
                self.phase_frame = 0;
                self.total_phase_saves = self.total_phase_saves.saturating_add(1);
                if completed == CognitiveSleepPhase::Validation {
                    self.baseline = self.controller.clone();
                }
                self.cleanup_old_snapshots();
                Ok(())
            }
            Err(error) => {
                self.controller = before_phase;
                self.phase_frame = 0;
                self.last_error = Some(error.clone());
                Err(error)
            }
        }
    }

    fn force_save(&mut self) -> Result<(), String> {
        let summary = format!("guardado manual durante {}", self.phase.label());
        let path = self.persist_phase(self.phase, self.cycle, self.phase, &summary)?;
        self.last_save = path;
        self.last_summary = summary;
        self.total_phase_saves = self.total_phase_saves.saturating_add(1);
        Ok(())
    }

    fn persist_phase(
        &self,
        completed: CognitiveSleepPhase,
        next_cycle: u64,
        next_phase: CognitiveSleepPhase,
        summary: &str,
    ) -> Result<String, String> {
        let directory = Path::new(STATE_ROOT).join("checkpoints");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let generation = self.total_phase_saves.saturating_add(1);
        let stem = format!(
            "cycle-{:06}-phase-{}-{:09}",
            self.cycle,
            completed.slug(),
            generation
        );
        let checkpoint = directory.join(format!("{stem}.cdt_native"));
        let cognitive = directory.join(format!("{stem}.cognitive.json"));
        save_native_checkpoint_transactional(&self.controller.substrate, &checkpoint)?;
        let snapshot =
            serde_json::to_vec_pretty(&self.controller.snapshot()).map_err(|e| e.to_string())?;
        atomic_write(&cognitive, &snapshot)?;
        let manifest = DesktopManifest {
            schema_version: 1,
            cycle: next_cycle,
            next_phase,
            checkpoint: path_string(&checkpoint),
            cognitive_snapshot: path_string(&cognitive),
            completed_phase: completed,
            accepted_cycles: self.accepted_cycles,
            rejected_cycles: self.rejected_cycles,
            total_phase_saves: generation,
            summary: summary.to_string(),
        };
        let body = serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
        atomic_write(Path::new(LATEST_MANIFEST), &body)?;
        Ok(path_string(&checkpoint))
    }

    fn cleanup_old_snapshots(&self) {
        let directory = Path::new(STATE_ROOT).join("checkpoints");
        let Ok(entries) = fs::read_dir(directory) else {
            return;
        };
        let mut files = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension == "cdt_native")
            })
            .collect::<Vec<_>>();
        files.sort();
        let remove = files.len().saturating_sub(self.retention);
        for checkpoint in files.into_iter().take(remove) {
            let cognitive = PathBuf::from(
                checkpoint
                    .to_string_lossy()
                    .replace(".cdt_native", ".cognitive.json"),
            );
            let _ = fs::remove_file(checkpoint);
            let _ = fs::remove_file(cognitive);
        }
    }
}

fn fresh_controller() -> LogisticsController {
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
            seed: 0xC09A_DE57,
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
    LogisticsController::new(
        substrate,
        LogisticsPlannerConfig {
            beam_width: 2,
            procedural_gain: 14.0,
            max_expansions: 256,
            use_handcrafted_schemas: false,
            use_learned_schemas: true,
            ..LogisticsPlannerConfig::default()
        },
    )
}

fn observe_balanced_primitives(controller: &mut LogisticsController, rounds: usize) {
    let task = training_task();
    for round in 0..rounds {
        let actions = if round % 2 == 0 {
            [decoy_action(&task), correct_action(&task)]
        } else {
            [correct_action(&task), decoy_action(&task)]
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
}

fn evaluate_ood(controller: &LogisticsController) -> (f32, f32, Vec<String>) {
    let tasks = [tree_task(), ring_task(), mesh_task()];
    let mut success = 0usize;
    let mut valid = 0usize;
    let mut emitted = 0usize;
    let mut plans = Vec::new();
    for task in tasks {
        let mut trial = controller.clone();
        let decision = trial.plan(&task);
        if let Some(plan) = decision.plan {
            emitted += 1;
            let verification = LogisticsController::verify(&task, &plan);
            valid += usize::from(verification.actions_valid);
            success += usize::from(verification.actions_valid && verification.goal_reached);
            plans.push(format!(
                "{}: {}",
                task.id,
                plan.iter()
                    .map(|action| format!("{action:?}"))
                    .collect::<Vec<_>>()
                    .join(" → ")
            ));
        } else {
            plans.push(format!("{}: abstención", task.id));
        }
    }
    (
        success as f32 / 3.0,
        valid as f32 / emitted.max(1) as f32,
        plans,
    )
}

fn training_task() -> LogisticsTask {
    graph_task(
        "train",
        Package(0),
        0,
        vec![(0, 1), (1, 3), (0, 4), (4, 5), (5, 3)],
        4,
    )
}

fn tree_task() -> LogisticsTask {
    graph_task(
        "árbol",
        Package(1),
        10,
        vec![(10, 11), (11, 13), (10, 14), (14, 15), (15, 16), (16, 13)],
        5,
    )
}

fn ring_task() -> LogisticsTask {
    graph_task(
        "anillo",
        Package(2),
        20,
        vec![
            (20, 21),
            (21, 23),
            (20, 24),
            (24, 25),
            (25, 26),
            (26, 23),
            (25, 27),
            (27, 24),
        ],
        5,
    )
}

fn mesh_task() -> LogisticsTask {
    graph_task(
        "malla",
        Package(3),
        30,
        vec![
            (30, 31),
            (31, 33),
            (30, 34),
            (34, 35),
            (35, 36),
            (36, 37),
            (37, 33),
            (34, 38),
            (38, 39),
            (39, 35),
            (36, 40),
            (40, 38),
        ],
        6,
    )
}

fn graph_task(
    id: &str,
    package: Package,
    offset: u8,
    edges: Vec<(u8, u8)>,
    max_steps: usize,
) -> LogisticsTask {
    let mut package_at = vec![None; package.0 as usize + 1];
    package_at[package.0 as usize] = None;
    let mut initial = LogisticsState {
        robot_at: Location(offset),
        package_at,
        carrying: Some(package),
        has_key: false,
        connections: edges
            .into_iter()
            .map(|(a, b)| (Location(a), Location(b)))
            .collect(),
        locked_edges: vec![(Location(offset + 1), Location(offset + 3))],
    };
    initial.canonicalize();
    LogisticsTask {
        id: id.to_string(),
        initial,
        goal: LogisticsGoal {
            package,
            destination: Location(offset + 3),
        },
        max_steps,
    }
}

fn correct_action(task: &LogisticsTask) -> LogisticsAction {
    first_actions(task).1
}

fn decoy_action(task: &LogisticsTask) -> LogisticsAction {
    first_actions(task).0
}

fn first_actions(task: &LogisticsTask) -> (LogisticsAction, LogisticsAction) {
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
    (
        LogisticsAction::Move(destinations[0]),
        LogisticsAction::Move(*destinations.last().unwrap()),
    )
}

fn draw_dashboard(desktop: &DesktopState, paused: bool, view_3d: bool, show_edges: bool) {
    let report = desktop.controller.substrate.thermal.report();
    draw_rectangle(
        0.0,
        0.0,
        screen_width(),
        218.0,
        Color::from_rgba(4, 8, 18, 235),
    );
    draw_text(
        "SISTEMA OPERATIVO COGNITIVO · SUEÑO INFINITO",
        24.0,
        34.0,
        27.0,
        SKYBLUE,
    );
    draw_text(
        &format!(
            "{} | ciclo={} fase={} progreso={:.0}% | guardados={} +{} -{}",
            if paused { "PAUSADO" } else { "ACTIVO" },
            desktop.cycle,
            desktop.phase.label(),
            desktop.phase_frame as f32 / desktop.phase_frames as f32 * 100.0,
            desktop.total_phase_saves,
            desktop.accepted_cycles,
            desktop.rejected_cycles,
        ),
        24.0,
        62.0,
        18.0,
        LIGHTGRAY,
    );
    draw_text(
        &format!(
            "CDT nodos={} E={:.4} F={:.4} | RQM={} EPR={} | esquemas={} episodios={} | OOD={:.1}% validez={:.1}%",
            report.nodes,
            report.mean_energy,
            report.free_energy_proxy,
            desktop.controller.substrate.relation_count(),
            desktop.controller.substrate.entanglement.active_count(),
            desktop.controller.learned_schema_count(),
            desktop.controller.episodes().len(),
            desktop.ood_success * 100.0,
            desktop.ood_validity * 100.0,
        ),
        24.0,
        88.0,
        17.0,
        Color::from_rgba(195, 220, 245, 255),
    );
    draw_text(&desktop.last_summary, 24.0, 114.0, 16.0, VIOLET);
    draw_text(
        &format!("checkpoint: {}", desktop.last_save),
        24.0,
        138.0,
        14.0,
        GRAY,
    );
    if let Some(error) = &desktop.last_error {
        draw_text(&format!("ERROR: {error}"), 24.0, 160.0, 15.0, RED);
    }
    let width = screen_width() - 48.0;
    draw_rectangle(24.0, 174.0, width, 8.0, DARKGRAY);
    draw_rectangle(
        24.0,
        174.0,
        width * (desktop.phase_frame as f32 / desktop.phase_frames as f32).clamp(0.0, 1.0),
        8.0,
        SKYBLUE,
    );
    draw_text(
        &format!(
            "vista={} [Tab]  pausa=[Espacio]  relaciones=[E:{}]  guardar=[S]  salir+guardar=[Esc]",
            if view_3d { "3D" } else { "2D" },
            if show_edges { "on" } else { "off" }
        ),
        24.0,
        205.0,
        15.0,
        GRAY,
    );
    let mut y = 235.0;
    for plan in desktop.current_plans.iter().take(3) {
        draw_text(plan, 24.0, y, 14.0, Color::from_rgba(120, 245, 190, 220));
        y += 18.0;
    }
}

fn draw_cognitive_2d(desktop: &DesktopState, show_edges: bool) {
    let thermal = &desktop.controller.substrate.thermal;
    let top = 300.0;
    let left = 36.0;
    let width = screen_width() - 72.0;
    let height = screen_height() - top - 35.0;
    let slices = thermal.config.slices;
    let nodes_per_slice = thermal.config.nodes_per_slice;
    let schema_nodes = cognitive_schema_nodes(&desktop.controller);
    let x_step = width / slices.saturating_sub(1).max(1) as f32;
    let y_step = height / nodes_per_slice.saturating_sub(1).max(1) as f32;
    if show_edges {
        for (index, (_, source, target, _, _, coherence, _, _)) in
            desktop.controller.substrate.relation_entries().enumerate()
        {
            if index >= MAX_2D_RELATIONS {
                break;
            }
            let a = node_position_2d(source, nodes_per_slice, left, top, x_step, y_step);
            let b = node_position_2d(target, nodes_per_slice, left, top, x_step, y_step);
            draw_line(
                a.x,
                a.y,
                b.x,
                b.y,
                0.7,
                Color::new(0.55, 0.3, 1.0, coherence * 0.16),
            );
        }
    }
    for node in 0..thermal.node_count() {
        let position = node_position_2d(node, nodes_per_slice, left, top, x_step, y_step);
        let schema = schema_nodes.contains(&node);
        let activation = thermal.activation[node].clamp(0.0, 1.0);
        let energy = thermal.energy[node].abs().min(2.0) / 2.0;
        let color = if schema {
            Color::new(0.1, 1.0, 0.72, 0.95)
        } else {
            Color::new(0.18 + energy * 0.75, 0.35, 0.85 - energy * 0.4, 0.55)
        };
        draw_circle(
            position.x,
            position.y,
            if schema { 5.5 } else { 1.7 + activation * 2.8 },
            color,
        );
    }
}

fn draw_cognitive_3d(desktop: &DesktopState, show_edges: bool) {
    let thermal = &desktop.controller.substrate.thermal;
    let time = get_time() as f32;
    let inspection_yaw = (time * 0.08).sin() * 1.8;
    let camera = Camera3D {
        position: vec3(inspection_yaw, 1.8, 16.5),
        up: Vec3::Y,
        target: Vec3::ZERO,
        fovy: 39.0,
        ..Default::default()
    };
    set_camera(&camera);
    let schema_nodes = cognitive_schema_nodes(&desktop.controller);
    let attractors = attractor_nodes(thermal, 8);
    let mut rqm_degree = vec![0usize; thermal.node_count()];
    let mut epr_degree = vec![0usize; thermal.node_count()];
    for (_, source, target, _, _, _, _, _) in desktop.controller.substrate.relation_entries() {
        if source < rqm_degree.len() {
            rqm_degree[source] += 1;
        }
        if target < rqm_degree.len() {
            rqm_degree[target] += 1;
        }
    }
    for link in desktop
        .controller
        .substrate
        .entanglement
        .active_link_entries()
    {
        if link.a < epr_degree.len() {
            epr_degree[link.a] += 1;
        }
        if link.b < epr_degree.len() {
            epr_degree[link.b] += 1;
        }
    }
    let max_rqm_degree = rqm_degree.iter().copied().max().unwrap_or(1).max(1);
    let max_epr_degree = epr_degree.iter().copied().max().unwrap_or(1).max(1);

    draw_brain_envelope(time);
    draw_cognitive_macronodes(desktop, show_edges, time);
    if show_edges {
        let edge_stride = thermal.edge_count().div_ceil(MAX_3D_CDT_EDGES).max(1);
        for edge in (0..thermal.edge_count())
            .step_by(edge_stride)
            .take(MAX_3D_CDT_EDGES)
        {
            let color = match thermal.edge_kind[edge] {
                NativeCdtEdgeKind::Spatial => Color::from_rgba(45, 105, 155, 22),
                NativeCdtEdgeKind::Temporal => Color::from_rgba(40, 205, 235, 70),
            };
            draw_line_3d(
                node_position_3d(thermal, thermal.edge_a[edge]),
                node_position_3d(thermal, thermal.edge_b[edge]),
                color,
            );
        }
        let relation_stride = desktop
            .controller
            .substrate
            .relation_count()
            .div_ceil(MAX_3D_RELATIONS)
            .max(1);
        for (_, source, target, _, _, coherence, _, _) in desktop
            .controller
            .substrate
            .relation_entries()
            .step_by(relation_stride)
            .take(MAX_3D_RELATIONS)
        {
            draw_line_3d(
                node_position_3d(thermal, source),
                node_position_3d(thermal, target),
                Color::new(0.55, 0.25, 1.0, 0.08 + coherence * 0.16),
            );
        }
        let epr_count = desktop.controller.substrate.entanglement.active_count();
        let stride = epr_count.div_ceil(MAX_3D_EPR).max(1);
        for link in desktop
            .controller
            .substrate
            .entanglement
            .active_link_entries()
            .step_by(stride)
            .take(MAX_3D_EPR)
        {
            draw_epr_arc(
                node_position_3d(thermal, link.a),
                node_position_3d(thermal, link.b),
                0.45 + link.coherence,
                Color::new(1.0, 0.12, 0.82, 0.16 + link.coherence * 0.28),
            );
        }
    }
    for node in 0..thermal.node_count() {
        let position = node_position_3d(thermal, node);
        let schema = schema_nodes.contains(&node);
        let activation = thermal.activation[node].clamp(0.0, 1.0);
        let energy = thermal.energy[node].abs().min(2.0) / 2.0;
        let color = if schema {
            Color::new(0.05, 1.0, 0.7, 1.0)
        } else {
            let phase = thermal.phase[node].sin() * 0.5 + 0.5;
            Color::new(
                0.12 + energy * 0.78,
                0.2 + phase * 0.62,
                0.9 - energy * 0.48,
                0.88,
            )
        };
        let node_size = if schema {
            0.18
        } else {
            0.045 + thermal.amplitude[node].clamp(0.0, 1.0) * 0.065
        };
        draw_cube(position, vec3(node_size, node_size, node_size), None, color);
        if activation > 0.58 {
            draw_sphere(
                position,
                0.12 + activation * 0.1,
                None,
                Color::new(0.2, 0.9, 1.0, 0.22),
            );
        }
        if schema {
            draw_sphere_wires(position, 0.22, None, Color::new(0.2, 1.0, 0.85, 0.7));
        }
        let rqm_load = rqm_degree[node] as f32 / max_rqm_degree as f32;
        if rqm_load > 0.12 {
            draw_sphere_wires(
                position,
                0.13 + rqm_load.sqrt() * 0.17,
                None,
                Color::new(0.55, 0.3, 1.0, 0.2 + rqm_load * 0.55),
            );
        }
        let epr_load = epr_degree[node] as f32 / max_epr_degree as f32;
        if epr_load > 0.0 {
            draw_sphere_wires(
                position,
                0.17 + epr_load.sqrt() * 0.2,
                None,
                Color::new(1.0, 0.15, 0.82, 0.25 + epr_load * 0.65),
            );
        }
        if attractors.contains(&node) {
            draw_sphere_wires(position, 0.34, None, Color::from_rgba(255, 215, 55, 205));
        }
    }
}

fn draw_brain_envelope(time: f32) {
    let pulse = 1.0 + (time * 1.7).sin() * 0.018;
    draw_sphere_wires(
        vec3(-2.55, 0.0, 0.0),
        3.25 * pulse,
        None,
        Color::from_rgba(45, 150, 195, 42),
    );
    draw_sphere_wires(
        vec3(2.55, 0.0, 0.0),
        3.25 * pulse,
        None,
        Color::from_rgba(45, 150, 195, 42),
    );
    draw_line_3d(
        vec3(0.0, -3.4, 0.0),
        vec3(0.0, 3.4, 0.0),
        Color::from_rgba(80, 220, 245, 45),
    );
}

fn draw_cognitive_macronodes(desktop: &DesktopState, show_edges: bool, time: f32) {
    let thermal = &desktop.controller.substrate.thermal;
    let patterns = desktop
        .controller
        .pattern_entries()
        .filter(|(key, nodes)| {
            !nodes.is_empty()
                && (key.starts_with("learned_action_schema:")
                    || key.starts_with("abstract:")
                    || key.starts_with("goal:")
                    || key.starts_with("action:"))
        })
        .take(MAX_COGNITIVE_MACRONODES)
        .collect::<Vec<_>>();
    let count = patterns.len().max(1);
    for (index, (key, nodes)) in patterns.into_iter().enumerate() {
        let angle = index as f32 * 2.399_963_1 + time * 0.025;
        let y = 1.0 - 2.0 * (index as f32 + 0.5) / count as f32;
        let shell = (1.0 - y * y).sqrt();
        let position = vec3(
            angle.cos() * shell * 7.1,
            y * 4.8,
            angle.sin() * shell * 5.5,
        );
        let schema = key.starts_with("learned_action_schema:");
        let pulse = (time * 2.2 + index as f32 * 0.7).sin() * 0.5 + 0.5;
        let radius = if schema {
            0.2 + pulse * 0.1
        } else {
            0.1 + pulse * 0.04
        };
        let color = if schema {
            Color::new(0.1, 1.0, 0.65, 0.85)
        } else {
            Color::new(1.0, 0.42, 0.12, 0.7)
        };
        draw_sphere(position, radius, None, color);
        draw_sphere_wires(
            position,
            radius + 0.11,
            None,
            Color::new(color.r, color.g, color.b, 0.4),
        );
        if show_edges {
            let mut center = Vec3::ZERO;
            let mut valid = 0.0;
            for &node in nodes.iter().take(12) {
                if node < thermal.node_count() {
                    center += node_position_3d(thermal, node);
                    valid += 1.0;
                }
            }
            if valid > 0.0 {
                center /= valid;
                draw_line_3d(
                    position,
                    center,
                    Color::new(color.r, color.g, color.b, 0.22),
                );
            }
        }
    }
}

fn draw_epr_arc(a: Vec3, b: Vec3, lift: f32, color: Color) {
    let midpoint = (a + b) * 0.5 + vec3(0.0, lift, 0.0);
    draw_line_3d(a, midpoint, color);
    draw_line_3d(midpoint, b, color);
}

fn attractor_nodes(substrate: &NativeThermoCdtSubstrate, count: usize) -> HashSet<usize> {
    let mut candidates = (0..substrate.node_count())
        .map(|node| {
            let score = substrate.amplitude[node] * (1.0 + substrate.activation[node])
                / (1.0 + substrate.energy[node].abs());
            (node, score)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    candidates
        .into_iter()
        .take(count)
        .map(|(node, _)| node)
        .collect()
}

fn cognitive_schema_nodes(controller: &LogisticsController) -> HashSet<usize> {
    controller
        .pattern_entries()
        .filter(|(key, _)| key.starts_with("learned_action_schema:"))
        .flat_map(|(_, nodes)| nodes.iter().copied())
        .collect()
}

fn node_position_2d(
    node: usize,
    nodes_per_slice: usize,
    left: f32,
    top: f32,
    x_step: f32,
    y_step: f32,
) -> Vec2 {
    let slice = node / nodes_per_slice;
    let offset = node % nodes_per_slice;
    vec2(left + slice as f32 * x_step, top + offset as f32 * y_step)
}

fn node_position_3d(thermal: &NativeThermoCdtSubstrate, node: usize) -> Vec3 {
    let slices = thermal.config.slices.max(1);
    let nodes_per_slice = thermal.config.nodes_per_slice.max(1);
    let slice = node / nodes_per_slice;
    let offset = node % nodes_per_slice;
    let slice_x = ((slice as f32 + 0.5) / slices as f32) * 2.0 - 1.0;
    let angle = offset as f32 * 2.399_963_1;
    let disk_radius = ((offset as f32 + 0.5) / nodes_per_slice as f32).sqrt();
    let cross_section = (1.0 - 0.78 * slice_x * slice_x).max(0.12).sqrt();
    let deformation = 1.0
        + thermal.thermal_state[node].tanh() * 0.16
        + thermal.phase[node].sin() * thermal.amplitude[node].min(2.0) * 0.035;
    vec3(
        slice_x * 5.4 + thermal.thermal_state[node] * 0.18,
        angle.cos() * disk_radius * cross_section * 3.8 * deformation,
        angle.sin() * disk_radius * cross_section * 3.2 * deformation,
    )
}

fn draw_future_energy_monitor(desktop: &DesktopState) {
    let history = &desktop.free_energy_history;
    if history.is_empty() {
        return;
    }
    let width = 350.0;
    let height = 105.0;
    let left = screen_width() - width - 24.0;
    let top = 228.0;
    draw_rectangle(left, top, width, height, Color::from_rgba(3, 12, 22, 225));
    draw_rectangle_lines(
        left,
        top,
        width,
        height,
        1.0,
        Color::from_rgba(60, 205, 235, 110),
    );

    let values = history.iter().copied().collect::<Vec<_>>();
    let lookback = values.len().min(36);
    let recent = &values[values.len() - lookback..];
    let slope = linear_slope(recent);
    let current = *values.last().unwrap_or(&0.0);
    let future = current + slope * 30.0;
    let improving = future <= current;
    draw_text(
        "MONITOR PREDICTIVO · ENERGÍA LIBRE",
        left + 12.0,
        top + 20.0,
        15.0,
        SKYBLUE,
    );
    draw_text(
        &format!(
            "F ahora {current:.4}  →  F(+30) {future:.4}  {}",
            if improving { "↓" } else { "↑" }
        ),
        left + 12.0,
        top + 40.0,
        15.0,
        if improving { LIME } else { ORANGE },
    );

    let chart_left = left + 12.0;
    let chart_top = top + 51.0;
    let chart_width = width - 24.0;
    let chart_height = height - 62.0;
    let min = values
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .min(future);
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(future);
    let range = (max - min).abs().max(1e-5);
    for index in 1..values.len() {
        let x0 = chart_left + (index - 1) as f32 / FREE_ENERGY_HISTORY as f32 * chart_width;
        let x1 = chart_left + index as f32 / FREE_ENERGY_HISTORY as f32 * chart_width;
        let y0 = chart_top + chart_height - (values[index - 1] - min) / range * chart_height;
        let y1 = chart_top + chart_height - (values[index] - min) / range * chart_height;
        draw_line(x0, y0, x1, y1, 1.5, LIME);
    }
    let x_current = chart_left
        + values.len().saturating_sub(1) as f32 / FREE_ENERGY_HISTORY as f32 * chart_width;
    let y_current = chart_top + chart_height - (current - min) / range * chart_height;
    let y_future = chart_top + chart_height - (future - min) / range * chart_height;
    draw_line(
        x_current,
        y_current,
        chart_left + chart_width,
        y_future,
        1.5,
        if improving { SKYBLUE } else { ORANGE },
    );
}

fn linear_slope(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean_x = (n - 1.0) * 0.5;
    let mean_y = values.iter().sum::<f32>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (index, &value) in values.iter().enumerate() {
        let dx = index as f32 - mean_x;
        numerator += dx * (value - mean_y);
        denominator += dx * dx;
    }
    numerator / denominator.max(f32::EPSILON)
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(default)
}
