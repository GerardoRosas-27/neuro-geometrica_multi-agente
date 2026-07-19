//! Visualizador nativo, en tiempo real, del sustrato térmico CDT.
//!
//! Controles: Espacio pausa/reanuda, E alterna enlaces, R reinicia, Esc cierra.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use cdt_rqm_epr::native_thermo_rqm_epr::{NativeThermoRqmEprSubstrate, RealtimeUpdateConfig};
use cdt_rqm_epr::native_thermodynamic_cdt::{
    NativeCdtEdgeKind, NativeThermoCdtReport, NativeThermoCdtSubstrate,
};
use cdt_rqm_epr::native_thermodynamic_engine::{
    canonical_lessons, evaluate_native_suite, load_native_checkpoint, native_sleep_consolidate,
    native_sleep_prospective, EngineMetrics, Lesson, ProspectiveSleepConfig, DEFAULT_OBSERVER,
};
use cdt_rqm_epr::plasticity_controller::{run_plasticity_cycle, PlasticityConfig};
use cdt_rqm_epr::relational_field::ObserverId;
use cdt_rqm_epr::thermo_router::{
    ActivationFingerprint, ContextInjection, RouteId, RouterConfig, ThermoAssociativeRouter,
    TransformerActivationAdapter,
};
use macroquad::prelude::*;
use ::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread;

const STEPS_PER_FRAME: usize = 2;
const TRAINED_CHECKPOINT: &str = "data/native_curriculum_5phase.cdt_native";
const SLEEP_CHECKPOINT: &str =
    "data/native_tinyllama_paged_thermo/native_thermo_visualizer_sleep.cdt_native";
const WAVE_LIFETIME_TICKS: u64 = 110;
const EPR_EVENT_LIFETIME_TICKS: u64 = 90;
const PAGED_MODEL_ROOT: &str = "data/native_tinyllama_paged_thermo";
const MACRO_NODE_TARGET: usize = 1_000;
const MAX_ACTIVE_MACROS: usize = 64;
const DREAM_MEMORY_LOG: &str =
    "data/native_tinyllama_paged_thermo/transformer_dream_memory.tsv";
const ROUTER_MEMORY_LOG: &str =
    "data/native_tinyllama_paged_thermo/thermo_route_memory.json";
const TRANSFORMER_OBSERVER: ObserverId = ObserverId(778_002);
const DREAM_CONTEXT_LIMIT: usize = 256;
const DREAM_HISTORY_LIMIT: usize = 16;
const DREAM_CONSOLIDATION_INTERVAL: u64 = 64;
const MAX_DREAM_IDS_PER_FRAME: usize = 2;
const MAX_DREAM_TRANSITIONS: usize = 4_096;
const LUCID_IDS_PER_CYCLE: u64 = 256;
const MAX_CYCLE_LEAK_DRIFT: f32 = 0.002;
const MAX_VISUAL_RQM_EDGES: usize = 6_000;
const MAX_VISUAL_EPR_LINKS: usize = 6_000;
const STRUCTURAL_PRUNE_ROUNDS: usize = 8;
const RELATION_PRESSURE_PER_NODE: usize = 16;
const TRANSFORMER_RELATIONS_PER_NODE: usize = 6;
const EPR_LINKS_PER_NODE: usize = 8;
const EPR_MAX_LINKS_PER_NODE: usize = 24;
const MAX_GROW_SLICES_PER_CYCLE: usize = 16;
const MACRO_FORWARD_TOP_K: usize = 64;
const MACRO_FORWARD_GAIN: f32 = 0.55;

#[derive(Clone)]
struct PilotWave {
    seeds: Vec<usize>,
    born_tick: u64,
}

#[derive(Clone, Copy)]
struct EprEvent {
    a: usize,
    b: usize,
    born_tick: u64,
    created: bool,
}

struct NetworkOverlay {
    rqm_degree: Vec<usize>,
    epr_degree: Vec<usize>,
    rqm_nodes: usize,
    epr_nodes: usize,
    max_rqm_degree: usize,
    max_epr_degree: usize,
    mean_rqm_coherence: f32,
    mean_rqm_uncertainty: f32,
    mean_epr_coherence: f32,
    mean_epr_entropy: f32,
    mean_epr_heat: f32,
}

impl NetworkOverlay {
    fn empty() -> Self {
        Self {
            rqm_degree: Vec::new(),
            epr_degree: Vec::new(),
            rqm_nodes: 0,
            epr_nodes: 0,
            max_rqm_degree: 0,
            max_epr_degree: 0,
            mean_rqm_coherence: 0.0,
            mean_rqm_uncertainty: 0.0,
            mean_epr_coherence: 0.0,
            mean_epr_entropy: 0.0,
            mean_epr_heat: 0.0,
        }
    }
}

#[derive(Clone)]
struct MacroNodeVisual {
    shard: String,
    start: u64,
    values: u64,
    amplitude: f32,
    phase: f32,
    energy: f32,
    activation: f32,
}

#[derive(Clone)]
struct PagedModelVisual {
    root: String,
    model: String,
    logical_edges: u64,
    nodes: Vec<MacroNodeVisual>,
}

#[derive(Clone)]
struct RawDreamId {
    id: u32,
    confidence: f32,
    entropy: f32,
    generation: u64,
    macro_injected: bool,
    macro_changed_top: bool,
    fingerprint: ActivationFingerprint,
    context_tail: Vec<u32>,
    feedback_route: Option<RouteId>,
    route_context_injected: bool,
}

struct MacroForwardInjection {
    weights: Vec<f32>,
    generation: u64,
}

#[derive(Default)]
struct MacroForwardReport {
    applied: bool,
    changed_top: bool,
}

struct MacroForwardAdapter {
    receiver: Receiver<MacroForwardInjection>,
}

struct TransformerDream {
    receiver: Receiver<Result<RawDreamId, String>>,
    macro_sender: mpsc::Sender<MacroForwardInjection>,
    route_sender: mpsc::Sender<ContextInjection>,
    model_name: String,
    layers: usize,
    history: VecDeque<RawDreamId>,
    transitions: HashMap<(u32, u32), u32>,
    previous_id: Option<u32>,
    generated: u64,
    consolidated: u64,
    layer_pulse: f32,
    last_error: Option<String>,
    macro_injections: u64,
    macro_top_changes: u64,
    route_context_injections: u64,
    last_route_sent: Option<RouteId>,
    last_route_sent_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InfiniteSleepPhase {
    LucidTransformer,
    PruneRelax,
    Prospective,
    Plasticity,
    Validation,
}

struct PhaseOutcome {
    substrate: NativeThermoRqmEprSubstrate,
    summary: String,
}

#[derive(Default)]
struct StructuralPruneReport {
    rounds_accepted: usize,
    relations_pruned: usize,
    epr_pruned: usize,
    slices_added: usize,
    nodes_added: usize,
}

struct InfiniteSleepCycle {
    phase: InfiniteSleepPhase,
    cycle: u64,
    lucid_start_generation: u64,
    baseline_substrate: NativeThermoRqmEprSubstrate,
    baseline_metrics: EngineMetrics,
    baseline_transitions: HashMap<(u32, u32), u32>,
    baseline_paged_model: PagedModelVisual,
    baseline_router: ThermoAssociativeRouter,
    worker: Option<Receiver<Result<PhaseOutcome, String>>>,
    accepted_cycles: u64,
    rejected_cycles: u64,
    last_summary: String,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "CDT-RQM-EPR · Motor termodinámico".to_owned(),
        window_width: 1440,
        window_height: 900,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut substrate = new_substrate();
    let mut paged_model = PagedModelVisual::load(PAGED_MODEL_ROOT).unwrap_or_else(|error| {
        eprintln!("paged_model_loaded=false error={error}");
        PagedModelVisual::empty()
    });
    let mut transformer = TransformerDream::spawn(PAGED_MODEL_ROOT);
    if let Err(error) = transformer.replay_memory(&mut substrate) {
        eprintln!("transformer_dream_memory_replayed=false error={error}");
    }
    let mut router = ThermoAssociativeRouter::load_or_new(
        ROUTER_MEMORY_LOG,
        transformer.model_name.clone(),
        RouterConfig::for_substrate(substrate.thermal.node_count()),
    );
    let lessons = canonical_lessons();
    let mut sleep_cycle = InfiniteSleepCycle::new(
        &substrate,
        &transformer,
        &paged_model,
        &router,
        &lessons,
    );
    let mut paused = false;
    let mut show_edges = true;
    let mut view_3d = false;
    let mut pulse = 0usize;
    let mut energy_history = Vec::with_capacity(240);
    let mut variance_history = Vec::with_capacity(240);
    let mut coherence_history = Vec::with_capacity(240);
    let mut previous_epr = substrate
        .entanglement
        .active_link_entries()
        .map(|link| ordered_pair(link.a, link.b))
        .collect::<HashSet<_>>();
    let mut pilot_waves = Vec::<PilotWave>::new();
    let mut epr_events = Vec::<EprEvent>::new();
    let mut updates = RealtimeUpdateConfig::default();
    updates.thermal_microsteps = 1;

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::E) {
            show_edges = !show_edges;
        }
        if is_key_pressed(KeyCode::Tab) {
            view_3d = !view_3d;
            if view_3d {
                previous_epr = substrate
                    .entanglement
                    .active_link_entries()
                    .take(MAX_VISUAL_EPR_LINKS)
                    .map(|link| ordered_pair(link.a, link.b))
                    .collect();
                epr_events.clear();
            }
        }
        if is_key_pressed(KeyCode::R) {
            substrate = new_substrate();
            paged_model.clear_activation();
            energy_history.clear();
            variance_history.clear();
            coherence_history.clear();
            previous_epr = substrate
                .entanglement
                .active_link_entries()
                .map(|link| ordered_pair(link.a, link.b))
                .collect();
            pilot_waves.clear();
            epr_events.clear();
            transformer.reset_session();
            if let Err(error) = transformer.replay_memory(&mut substrate) {
                eprintln!("transformer_dream_memory_replayed=false error={error}");
            }
            router = ThermoAssociativeRouter::load_or_new(
                ROUTER_MEMORY_LOG,
                transformer.model_name.clone(),
                RouterConfig::for_substrate(substrate.thermal.node_count()),
            );
            sleep_cycle = InfiniteSleepCycle::new(
                &substrate,
                &transformer,
                &paged_model,
                &router,
                &lessons,
            );
            view_3d = false;
            pulse = 0;
        }

        if !paused {
            if sleep_cycle.phase == InfiniteSleepPhase::LucidTransformer {
                for seeds in transformer.poll_and_couple(
                    &mut substrate,
                    &mut paged_model,
                    &mut router,
                    updates,
                    MAX_DREAM_IDS_PER_FRAME,
                ) {
                    pilot_waves.push(PilotWave {
                        seeds,
                        born_tick: substrate.thermal.tick(),
                    });
                }
            }
            for _ in 0..STEPS_PER_FRAME {
                if sleep_cycle.phase == InfiniteSleepPhase::LucidTransformer
                    && substrate.thermal.tick() % 96 == 0
                {
                    let seeds = inject_dream_pulse(&mut substrate, &lessons, pulse, updates);
                    pilot_waves.push(PilotWave {
                        seeds,
                        born_tick: substrate.thermal.tick(),
                    });
                    pulse = pulse.wrapping_add(1);
                }
                substrate.thermal.step();
                if let Some(wave) = pilot_waves.last() {
                    paged_model.feedback_from_core(&substrate.thermal, &wave.seeds);
                }
            }
            paged_model.decay_activation();
            transformer.decay_visual();
            sleep_cycle.update(
                &mut substrate,
                &mut transformer,
                &mut paged_model,
                &mut router,
                &lessons,
            );
        }

        let report = substrate.thermal.report();
        let wave = wave_metrics(&substrate.thermal);
        let epr_total = substrate.entanglement.active_count();
        let relation_total = substrate.relation_count();
        let epr_stride = epr_total.div_ceil(MAX_VISUAL_EPR_LINKS).max(1);
        let relation_stride = relation_total.div_ceil(MAX_VISUAL_RQM_EDGES).max(1);
        let epr_links = if view_3d {
            substrate
                .entanglement
                .active_link_entries()
                .step_by(epr_stride)
                .take(MAX_VISUAL_EPR_LINKS)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let relations = if view_3d {
            substrate
                .relation_entries()
                .step_by(relation_stride)
                .take(MAX_VISUAL_RQM_EDGES)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let network = if view_3d {
            build_network_overlay(substrate.thermal.node_count(), &relations, &epr_links)
        } else {
            NetworkOverlay::empty()
        };
        let (epr_created, epr_destroyed) = if view_3d && epr_total <= MAX_VISUAL_EPR_LINKS {
            let current_epr = epr_links
                .iter()
                .map(|link| ordered_pair(link.a, link.b))
                .collect::<HashSet<_>>();
            let created = current_epr.difference(&previous_epr).count();
            let destroyed = previous_epr.difference(&current_epr).count();
            for &(a, b) in current_epr.difference(&previous_epr) {
                epr_events.push(EprEvent {
                    a,
                    b,
                    born_tick: report.tick,
                    created: true,
                });
            }
            for &(a, b) in previous_epr.difference(&current_epr) {
                epr_events.push(EprEvent {
                    a,
                    b,
                    born_tick: report.tick,
                    created: false,
                });
            }
            previous_epr = current_epr;
            (created, destroyed)
        } else {
            epr_events.clear();
            (0, 0)
        };
        pilot_waves.retain(|wave| report.tick.saturating_sub(wave.born_tick) < WAVE_LIFETIME_TICKS);
        epr_events
            .retain(|event| report.tick.saturating_sub(event.born_tick) < EPR_EVENT_LIFETIME_TICKS);
        push_history(&mut energy_history, report.free_energy_proxy);
        push_history(&mut variance_history, report.state_variance);
        push_history(&mut coherence_history, wave.phase_coherence);

        clear_background(Color::from_rgba(7, 12, 24, 255));
        if view_3d {
            draw_substrate_3d(
                &substrate.thermal,
                &epr_links,
                &relations,
                &network,
                &paged_model,
                &transformer,
                &sleep_cycle,
                &pilot_waves,
                &epr_events,
                show_edges,
            );
            set_default_camera();
        }
        draw_header(
            &report,
            &substrate,
            &wave,
            &network,
            &paged_model,
            &transformer,
            &router,
            &sleep_cycle,
            paused,
            show_edges,
            view_3d,
            epr_created,
            epr_destroyed,
        );
        if !view_3d {
            draw_substrate(&substrate.thermal, show_edges);
        }
        draw_charts(&energy_history, &variance_history, &coherence_history);
        draw_legend();

        next_frame().await;
    }

    sleep_cycle.finish_on_exit(
        &mut substrate,
        &mut transformer,
        &mut paged_model,
        &mut router,
        &lessons,
    );
}

fn new_substrate() -> NativeThermoRqmEprSubstrate {
    let checkpoint = env::var("NATIVE_THERMO_VISUALIZER_STATE").unwrap_or_else(|_| {
        if Path::new(SLEEP_CHECKPOINT).exists() {
            SLEEP_CHECKPOINT.to_string()
        } else {
            TRAINED_CHECKPOINT.to_string()
        }
    });
    println!("visualizer_checkpoint={checkpoint}");
    load_native_checkpoint(&checkpoint)
        .unwrap_or_else(|error| panic!("no se pudo cargar el sustrato {checkpoint}: {error}"))
}

impl PagedModelVisual {
    fn load(root: &str) -> Result<Self, String> {
        let root = Path::new(root);
        let manifest = fs::read_to_string(root.join("manifest.txt"))
            .map_err(|error| format!("manifest paginado: {error}"))?;
        let model = manifest_value(&manifest, "model")
            .unwrap_or("unknown")
            .to_string();
        let logical_edges = manifest_value(&manifest, "logical_edges")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let index = fs::read_to_string(root.join("macro_thermodynamic_index.tsv"))
            .map_err(|error| format!("índice térmico paginado: {error}"))?;
        let base_nodes = index
            .lines()
            .skip(1)
            .filter_map(|line| {
                let parts = line.split('\t').collect::<Vec<_>>();
                if parts.len() != 5 {
                    return None;
                }
                Some(MacroNodeVisual {
                    shard: root.join(parts[0]).to_string_lossy().to_string(),
                    start: 0,
                    values: parts[1].parse().ok()?,
                    amplitude: parts[2].parse().ok()?,
                    phase: parts[3].parse().ok()?,
                    energy: parts[4].parse().ok()?,
                    activation: 0.0,
                })
            })
            .collect::<Vec<_>>();
        let memory_path = root.join("macro_thermodynamic_memory_v2.tsv");
        let nodes = load_macro_memory(&memory_path).unwrap_or_else(|| {
            let split = split_macro_nodes(&base_nodes);
            let _ = save_macro_memory(&memory_path, &split);
            split
        });
        if nodes.is_empty() {
            return Err("índice térmico paginado vacío".to_string());
        }
        println!(
            "paged_model_loaded=true model={} logical_edges={} macro_nodes={}",
            model,
            logical_edges,
            nodes.len()
        );
        Ok(Self {
            root: root.to_string_lossy().to_string(),
            model,
            logical_edges,
            nodes,
        })
    }

    fn empty() -> Self {
        Self {
            root: String::new(),
            model: "not_loaded".to_string(),
            logical_edges: 0,
            nodes: Vec::new(),
        }
    }

    fn activate_and_sample(&mut self, key: &str) -> Result<Vec<f32>, String> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let query_phase = stable_hash(&key) as f64 / u64::MAX as f64 * std::f64::consts::TAU;
        let free_energy = self
            .nodes
            .iter()
            .map(|node| {
                node.energy as f64
                    - 2.0 * node.amplitude as f64 * (node.phase as f64 - query_phase).cos()
            })
            .collect::<Vec<_>>();
        let mean = free_energy.iter().sum::<f64>() / free_energy.len() as f64;
        let variance = free_energy
            .iter()
            .map(|energy| (energy - mean).powi(2))
            .sum::<f64>()
            / free_energy.len() as f64;
        let temperature = variance.sqrt().max(1.0e-6);
        let mut ranked = free_energy
            .iter()
            .enumerate()
            .map(|(index, energy)| {
                let activation = (-(energy - mean) / temperature).exp().clamp(0.0, 1.0);
                (index, activation)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        ranked.truncate(MAX_ACTIVE_MACROS);
        for node in &mut self.nodes {
            node.activation *= 0.15;
        }
        let mut weights = Vec::new();
        for (index, activation) in ranked {
            let node = &mut self.nodes[index];
            node.activation = activation.max(0.25) as f32;
            weights.push(read_macro_weight(node, stable_hash(&(key, index)))?);
        }
        Ok(weights)
    }

    fn feedback_from_core(&mut self, core: &NativeThermoCdtSubstrate, seeds: &[usize]) {
        let core_signal = seeds
            .iter()
            .filter(|&&node| node < core.node_count())
            .map(|&node| core.energy[node].abs() + core.activation[node])
            .sum::<f32>()
            / seeds.len().max(1) as f32;
        let gain = (core_signal / (1.0 + core_signal)).clamp(0.0, 1.0);
        for node in &mut self.nodes {
            if node.activation > 0.01 {
                node.activation = (node.activation + 0.18 * gain).min(1.0);
                node.energy = (node.energy * 0.985 + 0.015 * core_signal).max(0.0);
            }
        }
    }

    fn decay_activation(&mut self) {
        for node in &mut self.nodes {
            node.activation *= 0.965;
        }
    }

    fn clear_activation(&mut self) {
        for node in &mut self.nodes {
            node.activation = 0.0;
        }
    }

    fn active_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.activation > 0.10)
            .count()
    }

    fn consolidate_after_sleep(&mut self, core_energy: f32) {
        for node in &mut self.nodes {
            let retained = node.activation * 0.35;
            node.activation = retained;
            node.energy = (node.energy * 0.98 + core_energy.abs() * 0.02).max(0.0);
            node.phase = (node.phase + retained * 0.03).rem_euclid(std::f32::consts::TAU);
        }
    }

    fn save_memory(&self) -> Result<(), String> {
        if self.root.is_empty() {
            return Ok(());
        }
        save_macro_memory(
            &Path::new(&self.root).join("macro_thermodynamic_memory_v2.tsv"),
            &self.nodes,
        )
    }
}

impl TransformerDream {
    fn spawn(root: &str) -> Self {
        let root = PathBuf::from(root);
        let manifest = fs::read_to_string(root.join("manifest.txt")).unwrap_or_default();
        let model_name = manifest_value(&manifest, "model")
            .unwrap_or("not_loaded")
            .to_string();
        let source = manifest_value(&manifest, "source").map(PathBuf::from);
        let layers = transformer_layer_count(&root).unwrap_or(0);
        let (sender, receiver) = mpsc::sync_channel(4);
        let (macro_sender, macro_receiver) = mpsc::channel();
        let (route_sender, route_receiver) = mpsc::channel();
        thread::Builder::new()
            .name("transformer-dream".to_string())
            .spawn(move || {
                let result = source
                    .ok_or_else(|| "manifest sin source GGUF".to_string())
                    .and_then(|path| {
                        run_raw_transformer_dream(
                            &path,
                            sender.clone(),
                            macro_receiver,
                            route_receiver,
                        )
                    });
                if let Err(error) = result {
                    let _ = sender.send(Err(error));
                }
            })
            .unwrap_or_else(|error| panic!("no se pudo iniciar el sueño Transformer: {error}"));
        Self {
            receiver,
            macro_sender,
            route_sender,
            model_name,
            layers,
            history: VecDeque::with_capacity(DREAM_HISTORY_LIMIT),
            transitions: HashMap::new(),
            previous_id: None,
            generated: 0,
            consolidated: 0,
            layer_pulse: 0.0,
            last_error: None,
            macro_injections: 0,
            macro_top_changes: 0,
            route_context_injections: 0,
            last_route_sent: None,
            last_route_sent_generation: 0,
        }
    }

    fn reset_session(&mut self) {
        self.history.clear();
        self.previous_id = None;
        self.layer_pulse = 0.0;
        self.last_error = None;
        self.macro_injections = 0;
        self.macro_top_changes = 0;
        self.route_context_injections = 0;
        self.last_route_sent = None;
        self.last_route_sent_generation = 0;
    }

    fn poll_and_couple(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        paged_model: &mut PagedModelVisual,
        router: &mut ThermoAssociativeRouter,
        config: RealtimeUpdateConfig,
        limit: usize,
    ) -> Vec<Vec<usize>> {
        let mut waves = Vec::new();
        for _ in 0..limit {
            let Ok(message) = self.receiver.try_recv() else {
                break;
            };
            let dream = match message {
                Ok(dream) => dream,
                Err(error) => {
                    self.last_error = Some(error);
                    break;
                }
            };
            self.macro_injections += u64::from(dream.macro_injected);
            self.macro_top_changes += u64::from(dream.macro_changed_top);
            self.route_context_injections += u64::from(dream.route_context_injected);
            let route_outcome = router.process(
                substrate,
                &dream.fingerprint,
                &dream.context_tail,
                dream.generation,
                dream.feedback_route,
            );
            if let Some(injection) = route_outcome.recalled {
                let should_send = self.last_route_sent.is_none()
                    || dream
                        .generation
                        .saturating_sub(self.last_route_sent_generation)
                        >= 16;
                if should_send {
                    self.last_route_sent = Some(injection.route_id);
                    self.last_route_sent_generation = dream.generation;
                    if self.route_sender.send(injection).is_err() {
                        self.last_error =
                            Some("canal de contexto de ruta desconectado".to_string());
                    }
                }
            }
            let targets = raw_id_nodes(dream.id, substrate.thermal.node_count());
            let contact_nodes =
                raw_id_contact_nodes(dream.id, self.layers, substrate.thermal.node_count());
            if let Some(previous) = self.previous_id {
                let sources = raw_id_nodes(previous, substrate.thermal.node_count());
                let strength = (0.35 + 0.60 * dream.confidence).clamp(0.35, 0.95);
                substrate.train_observed_transition_realtime(
                    TRANSFORMER_OBSERVER,
                    (dream.generation % 628) as f32 * 0.01,
                    &sources,
                    &targets,
                    strength,
                    config,
                );
                let observations = self.transitions.entry((previous, dream.id)).or_insert(0);
                *observations = observations.saturating_add(1);
                self.prune_transitions();
            }
            for (contact, &node) in contact_nodes.iter().enumerate() {
                let layer = contact / 2;
                let depth = (layer + 1) as f32 / self.layers.max(1) as f32;
                substrate.thermal.inject_local_node(
                    node,
                    (0.18 + 0.62 * dream.confidence) * (0.75 + 0.25 * depth),
                    (dream.id as f32 * 0.017 + layer as f32 * 0.29)
                        .rem_euclid(std::f32::consts::TAU),
                    (0.12 + 0.48 * dream.entropy) * (0.8 + 0.2 * depth),
                );
            }
            let weights = paged_model
                .activate_and_sample(&format!("raw_transformer_id_{}", dream.id))
                .unwrap_or_else(|error| {
                    self.last_error = Some(error);
                    Vec::new()
                });
            couple_paged_model_to_core(substrate, &contact_nodes, &weights);
            if !weights.is_empty()
                && self
                    .macro_sender
                    .send(MacroForwardInjection {
                        weights,
                        generation: dream.generation,
                    })
                    .is_err()
            {
                self.last_error = Some("canal de inyección macro desconectado".to_string());
            }
            self.previous_id = Some(dream.id);
            self.generated = self.generated.max(dream.generation);
            if self.generated > 0 && self.generated % DREAM_CONSOLIDATION_INTERVAL == 0 {
                println!(
                    "macro_forward_adapter applied={} changed_top={} change_rate={:.2}% router_routes={} router_recalls={} context_injections={}",
                    self.macro_injections,
                    self.macro_top_changes,
                    self.macro_top_changes as f64 / self.macro_injections.max(1) as f64 * 100.0,
                    router.registry.routes().len(),
                    router.recalls,
                    self.route_context_injections,
                );
            }
            self.history.push_back(dream);
            while self.history.len() > DREAM_HISTORY_LIMIT {
                self.history.pop_front();
            }
            self.layer_pulse = 1.0;
            waves.push(contact_nodes);
            if self.generated > 0
                && self.generated % DREAM_CONSOLIDATION_INTERVAL == 0
                && self.generated != self.consolidated
            {
                self.consolidate(substrate, config);
                self.consolidated = self.generated;
            }
        }
        waves
    }

    fn consolidate(
        &self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        config: RealtimeUpdateConfig,
    ) {
        let mut strongest = self
            .transitions
            .iter()
            .map(|(&(source, target), &count)| (source, target, count))
            .collect::<Vec<_>>();
        strongest.sort_by(|left, right| right.2.cmp(&left.2));
        for (source, target, count) in strongest.into_iter().take(64) {
            substrate.train_observed_transition_realtime(
                TRANSFORMER_OBSERVER,
                count as f32 * 0.01,
                &raw_id_nodes(source, substrate.thermal.node_count()),
                &raw_id_nodes(target, substrate.thermal.node_count()),
                (0.55 + (count as f32).ln_1p() * 0.08).min(0.95),
                config,
            );
        }
        substrate.thermal.run_until_stable(2, 1.0e-4, 1.0e-4);
    }

    fn prune_transitions(&mut self) {
        if self.transitions.len() <= MAX_DREAM_TRANSITIONS {
            return;
        }
        let mut ranked = self
            .transitions
            .iter()
            .map(|(&transition, &count)| (transition, count))
            .collect::<Vec<_>>();
        ranked.sort_by_key(|(_, count)| *count);
        for (transition, _) in ranked
            .into_iter()
            .take(self.transitions.len() - MAX_DREAM_TRANSITIONS)
        {
            self.transitions.remove(&transition);
        }
    }

    fn replay_memory(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
    ) -> Result<(), String> {
        let Ok(file) = File::open(DREAM_MEMORY_LOG) else {
            return Ok(());
        };
        let config = RealtimeUpdateConfig {
            thermal_microsteps: 0,
            ..RealtimeUpdateConfig::default()
        };
        let rebuild_substrate =
            substrate.relation_count_for_observer(TRANSFORMER_OBSERVER) == 0;
        for line in BufReader::new(file).lines().skip(1) {
            let line = line.map_err(|error| error.to_string())?;
            let mut parts = line.split('\t');
            let Some(source) = parts.next().and_then(|value| value.parse::<u32>().ok()) else {
                continue;
            };
            let Some(target) = parts.next().and_then(|value| value.parse::<u32>().ok()) else {
                continue;
            };
            let count = parts
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(1);
            self.transitions.insert((source, target), count);
            if rebuild_substrate {
                substrate.train_observed_transition_realtime(
                    TRANSFORMER_OBSERVER,
                    count as f32 * 0.01,
                    &raw_id_nodes(source, substrate.thermal.node_count()),
                    &raw_id_nodes(target, substrate.thermal.node_count()),
                    (0.50 + (count as f32).ln_1p() * 0.08).min(0.92),
                    config,
                );
            }
        }
        self.prune_transitions();
        Ok(())
    }

    fn save_memory(&self) -> Result<(), String> {
        if let Some(parent) = Path::new(DREAM_MEMORY_LOG).parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temporary = Path::new(DREAM_MEMORY_LOG).with_extension("tmp");
        let mut file = File::create(&temporary).map_err(|error| error.to_string())?;
        writeln!(file, "source_id\ttarget_id\tobservations").map_err(|error| error.to_string())?;
        let mut transitions = self.transitions.iter().collect::<Vec<_>>();
        transitions.sort_by_key(|(&(source, target), _)| (source, target));
        for (&(source, target), &count) in transitions {
            writeln!(file, "{source}\t{target}\t{count}").map_err(|error| error.to_string())?;
        }
        file.flush().map_err(|error| error.to_string())?;
        if Path::new(DREAM_MEMORY_LOG).exists() {
            fs::remove_file(DREAM_MEMORY_LOG).map_err(|error| error.to_string())?;
        }
        fs::rename(temporary, DREAM_MEMORY_LOG).map_err(|error| error.to_string())
    }

    fn decay_visual(&mut self) {
        self.layer_pulse *= 0.94;
    }
}

impl InfiniteSleepPhase {
    fn label(self) -> &'static str {
        match self {
            Self::LucidTransformer => "SUEÑO LÚCIDO",
            Self::PruneRelax => "PODA Y RELAJACIÓN",
            Self::Prospective => "SUEÑO PROSPECTIVO",
            Self::Plasticity => "PLASTICIDAD R/S/Z + EPR",
            Self::Validation => "VALIDACIÓN Y GATE",
        }
    }

}

impl InfiniteSleepCycle {
    fn new(
        substrate: &NativeThermoRqmEprSubstrate,
        transformer: &TransformerDream,
        paged_model: &PagedModelVisual,
        router: &ThermoAssociativeRouter,
        lessons: &[Lesson],
    ) -> Self {
        Self {
            phase: InfiniteSleepPhase::LucidTransformer,
            cycle: 1,
            lucid_start_generation: transformer.generated,
            baseline_substrate: substrate.clone(),
            baseline_metrics: evaluate_native_suite(substrate, lessons),
            baseline_transitions: transformer.transitions.clone(),
            baseline_paged_model: paged_model.clone(),
            baseline_router: router.clone(),
            worker: None,
            accepted_cycles: 0,
            rejected_cycles: 0,
            last_summary: "inicio del ciclo transaccional".to_string(),
        }
    }

    fn update(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        transformer: &mut TransformerDream,
        paged_model: &mut PagedModelVisual,
        router: &mut ThermoAssociativeRouter,
        lessons: &[Lesson],
    ) {
        if let Some(result) = self.poll_worker() {
            match result {
                Ok(outcome) => {
                    *substrate = outcome.substrate;
                    self.last_summary = outcome.summary;
                    println!(
                        "sleep_cycle={} phase={} completed=true summary={}",
                        self.cycle,
                        self.phase.label(),
                        self.last_summary,
                    );
                    self.phase = match self.phase {
                        InfiniteSleepPhase::PruneRelax => InfiniteSleepPhase::Prospective,
                        InfiniteSleepPhase::Prospective => InfiniteSleepPhase::Plasticity,
                        InfiniteSleepPhase::Plasticity => InfiniteSleepPhase::Validation,
                        phase => phase,
                    };
                }
                Err(error) => {
                    self.last_summary = format!("rollback por error: {error}");
                    self.reject_cycle(substrate, transformer, paged_model, router, lessons);
                    return;
                }
            }
        }
        if self.worker.is_some() {
            return;
        }

        match self.phase {
            InfiniteSleepPhase::LucidTransformer => {
                if transformer
                    .generated
                    .saturating_sub(self.lucid_start_generation)
                    >= LUCID_IDS_PER_CYCLE
                {
                    self.phase = InfiniteSleepPhase::PruneRelax;
                    self.launch_worker(substrate.clone(), lessons.to_vec());
                }
            }
            InfiniteSleepPhase::PruneRelax
            | InfiniteSleepPhase::Prospective
            | InfiniteSleepPhase::Plasticity => {
                self.launch_worker(substrate.clone(), lessons.to_vec());
            }
            InfiniteSleepPhase::Validation => {
                self.validate_cycle(substrate, transformer, paged_model, router, lessons);
            }
        }
    }

    fn launch_worker(&mut self, substrate: NativeThermoRqmEprSubstrate, lessons: Vec<Lesson>) {
        let phase = self.phase;
        let (sender, receiver) = mpsc::channel();
        self.worker = Some(receiver);
        self.last_summary = format!("{} ejecutándose en segundo plano", phase.label());
        println!(
            "sleep_cycle={} phase={} started=true",
            self.cycle,
            phase.label(),
        );
        if let Err(error) = thread::Builder::new()
            .name(format!("sleep-phase-{phase:?}"))
            .spawn(move || {
                let outcome = match phase {
                    InfiniteSleepPhase::PruneRelax => {
                        let (pruned, structural) =
                            long_structural_prune(substrate, &lessons);
                        let (next, report) =
                            native_sleep_consolidate(pruned, &lessons, 8, 3);
                        PhaseOutcome {
                            substrate: next,
                            summary: format!(
                                "poda_larga={}/{} RQM-{} EPR-{} slices+{} nodos+{} | consolidación={}/{} E={:.4}→{:.4}",
                                structural.rounds_accepted,
                                STRUCTURAL_PRUNE_ROUNDS,
                                structural.relations_pruned,
                                structural.epr_pruned,
                                structural.slices_added,
                                structural.nodes_added,
                                report.accepted,
                                report.attempts,
                                report.before_energy,
                                report.after_energy,
                            ),
                        }
                    }
                    InfiniteSleepPhase::Prospective => {
                        let config = ProspectiveSleepConfig {
                            attempts: 2,
                            futures_per_attempt: 8,
                            ..ProspectiveSleepConfig::default()
                        };
                        let (next, report) =
                            native_sleep_prospective(substrate, &lessons, config);
                        PhaseOutcome {
                            substrate: next,
                            summary: format!(
                                "{} futuros={} entrenados={} epr={}->{}",
                                report.decision,
                                report.futures_generated,
                                report.futures_trained,
                                report.epr_before,
                                report.epr_after,
                            ),
                        }
                    }
                    InfiniteSleepPhase::Plasticity => {
                        let (next, report) =
                            run_plasticity_cycle(substrate, &lessons, PlasticityConfig::default());
                        PhaseOutcome {
                            substrate: next,
                            summary: format!(
                                "{} consolidaciones_finales={}",
                                report.decision, report.final_consolidation_accepted,
                            ),
                        }
                    }
                    _ => unreachable!("fase sin worker"),
                };
                let _ = sender.send(Ok(outcome));
            })
        {
            self.worker = None;
            self.last_summary = format!("no se pudo iniciar fase: {error}");
        }
    }

    fn poll_worker(&mut self) -> Option<Result<PhaseOutcome, String>> {
        let result = match self.worker.as_ref()?.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err("worker de fase desconectado".to_string()))
            }
        };
        if result.is_some() {
            self.worker = None;
        }
        result
    }

    fn validate_cycle(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        transformer: &mut TransformerDream,
        paged_model: &mut PagedModelVisual,
        router: &mut ThermoAssociativeRouter,
        lessons: &[Lesson],
    ) {
        let after = evaluate_native_suite(substrate, lessons);
        let preserves_accuracy = after.accuracy() + 1.0e-6 >= self.baseline_metrics.accuracy();
        let preserves_leakage =
            after.leakage() <= self.baseline_metrics.leakage() + MAX_CYCLE_LEAK_DRIFT;
        if preserves_accuracy && preserves_leakage {
            let energy = substrate.thermal.report().mean_energy;
            paged_model.consolidate_after_sleep(energy);
            let persisted = save_sleep_checkpoint(substrate, SLEEP_CHECKPOINT)
                .and_then(|_| paged_model.save_memory())
                .and_then(|_| transformer.save_memory())
                .and_then(|_| router.save(ROUTER_MEMORY_LOG));
            match persisted {
                Ok(()) => {
                    self.accepted_cycles = self.accepted_cycles.saturating_add(1);
                    self.last_summary = format!(
                        "ciclo guardado: accuracy={:.1}% fuga={:.4}",
                        after.accuracy() * 100.0,
                        after.leakage(),
                    );
                    println!(
                        "sleep_cycle={} decision=accept saved=true accuracy={:.4} leakage={:.6}",
                        self.cycle,
                        after.accuracy(),
                        after.leakage(),
                    );
                    self.begin_next_cycle(substrate, transformer, paged_model, router, lessons);
                }
                Err(error) => {
                    self.last_summary = format!("rollback por fallo de guardado: {error}");
                    self.reject_cycle(substrate, transformer, paged_model, router, lessons);
                }
            }
        } else {
            self.last_summary = format!(
                "rechazado: accuracy {:.1}%→{:.1}% fuga {:.4}→{:.4}",
                self.baseline_metrics.accuracy() * 100.0,
                after.accuracy() * 100.0,
                self.baseline_metrics.leakage(),
                after.leakage(),
            );
            println!(
                "sleep_cycle={} decision=reject saved=false summary={}",
                self.cycle,
                self.last_summary,
            );
            self.reject_cycle(substrate, transformer, paged_model, router, lessons);
        }
    }

    fn reject_cycle(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        transformer: &mut TransformerDream,
        paged_model: &mut PagedModelVisual,
        router: &mut ThermoAssociativeRouter,
        lessons: &[Lesson],
    ) {
        *substrate = self.baseline_substrate.clone();
        transformer.transitions = self.baseline_transitions.clone();
        transformer.reset_session();
        *paged_model = self.baseline_paged_model.clone();
        *router = self.baseline_router.clone();
        self.rejected_cycles = self.rejected_cycles.saturating_add(1);
        self.begin_next_cycle(substrate, transformer, paged_model, router, lessons);
    }

    fn begin_next_cycle(
        &mut self,
        substrate: &NativeThermoRqmEprSubstrate,
        transformer: &TransformerDream,
        paged_model: &PagedModelVisual,
        router: &ThermoAssociativeRouter,
        lessons: &[Lesson],
    ) {
        self.cycle = self.cycle.saturating_add(1);
        self.phase = InfiniteSleepPhase::LucidTransformer;
        self.lucid_start_generation = transformer.generated;
        self.baseline_substrate = substrate.clone();
        self.baseline_metrics = evaluate_native_suite(substrate, lessons);
        self.baseline_transitions = transformer.transitions.clone();
        self.baseline_paged_model = paged_model.clone();
        self.baseline_router = router.clone();
    }

    fn finish_on_exit(
        &mut self,
        substrate: &mut NativeThermoRqmEprSubstrate,
        transformer: &mut TransformerDream,
        paged_model: &mut PagedModelVisual,
        router: &mut ThermoAssociativeRouter,
        lessons: &[Lesson],
    ) {
        let (pruned, structural) = long_structural_prune(substrate.clone(), lessons);
        let (candidate, sleep) = native_sleep_consolidate(pruned, lessons, 4, 2);
        let after = evaluate_native_suite(&candidate, lessons);
        let valid = after.accuracy() + 1.0e-6 >= self.baseline_metrics.accuracy()
            && after.leakage() <= self.baseline_metrics.leakage() + MAX_CYCLE_LEAK_DRIFT;
        if !valid {
            println!(
                "sleep_saved=false reason=knowledge_gate rollback=true accuracy={:.4}->{:.4} leakage={:.4}->{:.4}",
                self.baseline_metrics.accuracy(),
                after.accuracy(),
                self.baseline_metrics.leakage(),
                after.leakage(),
            );
            return;
        }
        *substrate = candidate;
        paged_model.consolidate_after_sleep(sleep.after_energy);
        let result = save_sleep_checkpoint(substrate, SLEEP_CHECKPOINT)
            .and_then(|_| paged_model.save_memory())
            .and_then(|_| transformer.save_memory())
            .and_then(|_| router.save(ROUTER_MEMORY_LOG));
        match result {
            Ok(()) => println!(
                "sleep_saved=true output={} accepted={} energy={:.4}->{:.4} epr={}->{} accuracy={:.1}% exit_prune_rqm={} exit_prune_epr={}",
                SLEEP_CHECKPOINT,
                sleep.accepted,
                sleep.before_energy,
                sleep.after_energy,
                sleep.before_epr_links,
                sleep.after_epr_links,
                after.accuracy() * 100.0,
                structural.relations_pruned,
                structural.epr_pruned,
            ),
            Err(error) => eprintln!("sleep_saved=false output={} error={error}", SLEEP_CHECKPOINT),
        }
    }
}

fn long_structural_prune(
    substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
) -> (NativeThermoRqmEprSubstrate, StructuralPruneReport) {
    let baseline = evaluate_native_suite(&substrate, lessons);
    let mut best = substrate;
    let mut report = StructuralPruneReport::default();

    let nodes = best.thermal.node_count().max(1);
    let pressure = best
        .relation_count()
        .max(best.entanglement.active_count());
    let required_nodes = pressure.div_ceil(RELATION_PRESSURE_PER_NODE);
    if required_nodes > nodes {
        let nodes_per_slice = best.thermal.config.nodes_per_slice.max(1);
        let slices = (required_nodes - nodes)
            .div_ceil(nodes_per_slice)
            .min(MAX_GROW_SLICES_PER_CYCLE);
        let mut grown = best.clone();
        let added = grown.grow_thermal_slices(slices);
        let metrics = evaluate_native_suite(&grown, lessons);
        if metrics.accuracy() + 1.0e-6 >= baseline.accuracy()
            && metrics.leakage() <= baseline.leakage() + 0.0005
        {
            best = grown;
            report.slices_added = slices;
            report.nodes_added = added;
        }
    }

    let nodes = best.thermal.node_count().max(1);
    let transformer_start = best.relation_count_for_observer(TRANSFORMER_OBSERVER);
    let transformer_final = nodes.saturating_mul(TRANSFORMER_RELATIONS_PER_NODE);
    let epr_start = best.entanglement.active_count();
    let epr_final = nodes.saturating_mul(EPR_LINKS_PER_NODE);

    for round in 1..=STRUCTURAL_PRUNE_ROUNDS {
        let mut candidate = best.clone();
        let transformer_target = transformer_start.saturating_sub(
            transformer_start
                .saturating_sub(transformer_final)
                .saturating_mul(round)
                / STRUCTURAL_PRUNE_ROUNDS,
        );
        let epr_target = epr_start.saturating_sub(
            epr_start
                .saturating_sub(epr_final)
                .saturating_mul(round)
                / STRUCTURAL_PRUNE_ROUNDS,
        );
        let relations_pruned = candidate
            .prune_observer_relations_to_budget(TRANSFORMER_OBSERVER, transformer_target);
        let epr_pruned = candidate
            .entanglement
            .prune_to_budget(epr_target, EPR_MAX_LINKS_PER_NODE);
        candidate
            .thermal
            .run_until_stable(12, 1.0e-5, 1.0e-5);
        let metrics = evaluate_native_suite(&candidate, lessons);
        let preserves_knowledge = metrics.accuracy() + 1.0e-6 >= baseline.accuracy()
            && metrics.leakage() <= baseline.leakage() + 0.0005;
        if !preserves_knowledge {
            break;
        }
        best = candidate;
        report.rounds_accepted += 1;
        report.relations_pruned += relations_pruned;
        report.epr_pruned += epr_pruned;
    }

    (best, report)
}

impl MacroForwardAdapter {
    fn new(receiver: Receiver<MacroForwardInjection>) -> Self {
        Self { receiver }
    }

    fn apply(&self, logits: &mut [f32]) -> MacroForwardReport {
        let Some(injection) = self.receiver.try_iter().max_by_key(|item| item.generation) else {
            return MacroForwardReport::default();
        };
        if logits.is_empty() || injection.weights.is_empty() {
            return MacroForwardReport::default();
        }
        let mut ranked = logits.iter().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(left.1));
        ranked.truncate(MACRO_FORWARD_TOP_K.min(injection.weights.len()));
        let baseline_top = ranked.first().map(|(id, _)| *id);
        let candidate_ids = ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let max_abs = injection
            .weights
            .iter()
            .filter(|value| value.is_finite())
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
            .max(f32::EPSILON);
        for (rank, token_id) in candidate_ids.iter().enumerate() {
            let signal = injection.weights[rank] / max_abs;
            logits[*token_id] += signal.clamp(-1.0, 1.0) * MACRO_FORWARD_GAIN;
        }
        let modulated_top = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(id, _)| id);
        MacroForwardReport {
            applied: true,
            changed_top: baseline_top != modulated_top,
        }
    }
}

fn run_raw_transformer_dream(
    model_path: &Path,
    sender: mpsc::SyncSender<Result<RawDreamId, String>>,
    macro_receiver: Receiver<MacroForwardInjection>,
    route_receiver: Receiver<ContextInjection>,
) -> Result<(), String> {
    let device = Device::Cpu;
    let mut file = File::open(model_path).map_err(|error| format!("abrir GGUF: {error}"))?;
    let content =
        gguf_file::Content::read(&mut file).map_err(|error| format!("leer GGUF: {error}"))?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)
        .map_err(|error| format!("cargar Transformer: {error}"))?;
    model.clear_kv_cache();
    let mut recent = VecDeque::<u32>::with_capacity(32);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD4EA_5EED);
    let mut input_ids = vec![1u32];
    let mut position = 0usize;
    let mut generation = 0u64;
    let macro_adapter = MacroForwardAdapter::new(macro_receiver);
    let activation_adapter = TransformerActivationAdapter::new(16);
    loop {
        let route_injection = route_receiver.try_iter().last();
        let mut feedback_route = None;
        let mut route_context_injected = false;
        if let Some(injection) = route_injection {
            if !injection.context_ids.is_empty() {
                model.clear_kv_cache();
                let tail = &injection.context_ids
                    [injection.context_ids.len().saturating_sub(DREAM_CONTEXT_LIMIT - 1)..];
                input_ids.clear();
                input_ids.push(1);
                input_ids.extend_from_slice(tail);
                position = 0;
                recent.clear();
                recent.extend(tail.iter().copied());
                feedback_route = Some(injection.route_id);
                route_context_injected = true;
            }
        }
        let input = Tensor::new(input_ids.as_slice(), &device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| format!("tensor de sueño: {error}"))?;
        let mut logits = model
            .forward(&input, position)
            .and_then(|tensor| tensor.squeeze(0))
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| format!("inferencia de sueño: {error}"))?;
        for &id in recent.iter().rev().take(16) {
            if let Some(logit) = logits.get_mut(id as usize) {
                *logit -= 0.85;
            }
        }
        let fingerprint_context = recent.iter().copied().collect::<Vec<_>>();
        let fingerprint =
            activation_adapter.capture_with_context(&logits, &fingerprint_context);
        let macro_report = macro_adapter.apply(&mut logits);
        let (next, confidence, entropy) = sample_raw_logits(&logits, &mut rng)?;
        generation = generation.wrapping_add(1);
        let mut context_tail = recent.iter().copied().collect::<Vec<_>>();
        context_tail.push(next);
        if context_tail.len() > 32 {
            context_tail = context_tail.split_off(context_tail.len() - 32);
        }
        sender
            .send(Ok(RawDreamId {
                id: next,
                confidence,
                entropy,
                generation,
                macro_injected: macro_report.applied,
                macro_changed_top: macro_report.changed_top,
                fingerprint,
                context_tail,
                feedback_route,
                route_context_injected,
            }))
            .map_err(|_| "visualizador cerrado".to_string())?;
        recent.push_back(next);
        while recent.len() > 32 {
            recent.pop_front();
        }
        if next == 2 || position + input_ids.len() >= DREAM_CONTEXT_LIMIT {
            model.clear_kv_cache();
            input_ids.clear();
            input_ids.push(1);
            if next != 2 {
                input_ids.push(next);
            }
            position = 0;
        } else {
            position += input_ids.len();
            input_ids.clear();
            input_ids.push(next);
        }
    }
}

fn sample_raw_logits(
    logits: &[f32],
    rng: &mut Xoshiro256PlusPlus,
) -> Result<(u32, f32, f32), String> {
    let Some((_, &maximum)) = logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
    else {
        return Err("logits vacíos".to_string());
    };
    let mut sum = 0.0f64;
    let mut weighted = 0.0f64;
    for &logit in logits {
        let weight = ((logit - maximum) as f64).exp();
        sum += weight;
        if weight > 0.0 {
            weighted += weight * weight.ln();
        }
    }
    let mut ranked = logits.iter().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(left.1));
    ranked.truncate(40);
    let temperature = 0.85f64;
    let top_weights = ranked
        .iter()
        .map(|(_, logit)| (((**logit - maximum) as f64) / temperature).exp())
        .collect::<Vec<_>>();
    let top_sum = top_weights.iter().sum::<f64>();
    if top_sum <= f64::EPSILON {
        return Err("distribución de sueño degenerada".to_string());
    }
    let mut draw = rng.gen_range(0.0..top_sum);
    let mut selected = 0usize;
    for (index, weight) in top_weights.iter().enumerate() {
        draw -= weight;
        if draw <= 0.0 {
            selected = index;
            break;
        }
    }
    let next = ranked[selected].0;
    let confidence = if sum > 0.0 {
        (((logits[next] - maximum) as f64).exp() / sum).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let entropy = if sum > 0.0 && logits.len() > 1 {
        ((sum.ln() - weighted / sum) / (logits.len() as f64).ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Ok((next as u32, confidence as f32, entropy as f32))
}

fn raw_id_nodes(id: u32, node_count: usize) -> Vec<usize> {
    if node_count == 0 {
        return Vec::new();
    }
    (0..4)
        .map(|projection| stable_hash(&("raw_transformer_id", id, projection)) as usize % node_count)
        .collect()
}

fn raw_id_contact_nodes(id: u32, layers: usize, node_count: usize) -> Vec<usize> {
    if node_count == 0 {
        return Vec::new();
    }
    (0..layers.max(1))
        .flat_map(|layer| {
            (0..2).map(move |projection| {
                stable_hash(&("transformer_layer_contact", id, layer, projection)) as usize
                    % node_count
            })
        })
        .collect()
}

fn transformer_layer_count(root: &Path) -> Result<usize, String> {
    let catalog = fs::read_to_string(root.join("tensor_catalog.tsv"))
        .map_err(|error| format!("catálogo Transformer: {error}"))?;
    let layers = catalog
        .lines()
        .filter_map(|line| line.split('\t').nth(1))
        .filter_map(|name| name.strip_prefix("blk."))
        .filter_map(|tail| tail.split('.').next())
        .filter_map(|value| value.parse::<usize>().ok())
        .collect::<HashSet<_>>();
    Ok(layers.len())
}

fn read_macro_weight(node: &MacroNodeVisual, hash: u64) -> Result<f32, String> {
    let mut file = File::open(&node.shard).map_err(|error| format!("abrir macro peso: {error}"))?;
    let index = node.start + hash % node.values.max(1);
    file.seek(SeekFrom::Start(index * 4))
        .map_err(|error| format!("seek macro peso: {error}"))?;
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("leer macro peso: {error}"))?;
    Ok(f32::from_le_bytes(bytes))
}

fn split_macro_nodes(base_nodes: &[MacroNodeVisual]) -> Vec<MacroNodeVisual> {
    let total_values = base_nodes.iter().map(|node| node.values).sum::<u64>();
    let chunk = total_values.div_ceil(MACRO_NODE_TARGET as u64).max(1);
    let mut out = Vec::with_capacity(MACRO_NODE_TARGET);
    for base in base_nodes {
        let mut start = 0u64;
        while start < base.values {
            let values = (base.values - start).min(chunk);
            out.push(MacroNodeVisual {
                shard: base.shard.clone(),
                start,
                values,
                amplitude: base.amplitude,
                phase: base.phase,
                energy: base.energy,
                activation: 0.0,
            });
            start += values;
        }
    }
    out
}

fn load_macro_memory(path: &Path) -> Option<Vec<MacroNodeVisual>> {
    let contents = fs::read_to_string(path).ok()?;
    let nodes = contents
        .lines()
        .skip(1)
        .filter_map(|line| {
            let parts = line.split('\t').collect::<Vec<_>>();
            if parts.len() != 7 {
                return None;
            }
            Some(MacroNodeVisual {
                shard: parts[0].to_string(),
                start: parts[1].parse().ok()?,
                values: parts[2].parse().ok()?,
                amplitude: parts[3].parse().ok()?,
                phase: parts[4].parse().ok()?,
                energy: parts[5].parse().ok()?,
                activation: parts[6].parse().ok()?,
            })
        })
        .collect::<Vec<_>>();
    (nodes.len() >= MACRO_NODE_TARGET).then_some(nodes)
}

fn save_macro_memory(path: &Path, nodes: &[MacroNodeVisual]) -> Result<(), String> {
    let mut out = String::from("shard\tstart\tvalues\tamplitude\tphase\tenergy\tactivation\n");
    for node in nodes {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\n",
            node.shard,
            node.start,
            node.values,
            node.amplitude,
            node.phase,
            node.energy,
            node.activation,
        ));
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, out).map_err(|error| format!("guardar memoria macro: {error}"))?;
    if path.exists() {
        fs::remove_file(path).map_err(|error| format!("reemplazar memoria macro: {error}"))?;
    }
    fs::rename(temporary, path).map_err(|error| format!("confirmar memoria macro: {error}"))
}

fn couple_paged_model_to_core(
    substrate: &mut NativeThermoRqmEprSubstrate,
    seeds: &[usize],
    weights: &[f32],
) {
    if weights.is_empty() {
        return;
    }
    let max_abs = weights
        .iter()
        .map(|weight| weight.abs())
        .fold(0.0f32, f32::max)
        .max(f32::EPSILON);
    for (index, &node) in seeds.iter().enumerate() {
        if node >= substrate.thermal.node_count() {
            continue;
        }
        let normalized = weights[index % weights.len()] / max_abs;
        substrate.thermal.inject_local_node(
            node,
            0.35 + normalized.abs() * 0.75,
            if normalized < 0.0 {
                std::f32::consts::PI
            } else {
                0.0
            },
            0.20 + normalized.abs() * 0.45,
        );
    }
}

fn manifest_value<'a>(manifest: &'a str, key: &str) -> Option<&'a str> {
    manifest
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
}

fn stable_hash(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn inject_dream_pulse(
    substrate: &mut NativeThermoRqmEprSubstrate,
    lessons: &[cdt_rqm_epr::native_thermodynamic_engine::Lesson],
    pulse: usize,
    config: RealtimeUpdateConfig,
) -> Vec<usize> {
    let lesson = &lessons[pulse % lessons.len()];
    substrate.train_observed_transition_realtime(
        DEFAULT_OBSERVER,
        pulse as f32 * 0.01,
        &lesson.local,
        &lesson.remote,
        0.9,
        config,
    );
    lesson.local.clone()
}

fn draw_header(
    report: &NativeThermoCdtReport,
    substrate: &NativeThermoRqmEprSubstrate,
    wave: &WaveMetrics,
    network: &NetworkOverlay,
    paged_model: &PagedModelVisual,
    transformer: &TransformerDream,
    router: &ThermoAssociativeRouter,
    sleep_cycle: &InfiniteSleepCycle,
    paused: bool,
    show_edges: bool,
    view_3d: bool,
    epr_created: usize,
    epr_destroyed: usize,
) {
    draw_text(
        "MOTOR TERMODINÁMICO CDT · DINÁMICA DE SUEÑO",
        28.0,
        38.0,
        27.0,
        SKYBLUE,
    );
    let state = if paused { "PAUSADO" } else { "EVOLUCIONANDO" };
    let details = format!(
        "{state}  | tick {} | nodos {} | CDT {} | RQM {} | EPR {} | activos {} ({:.1}%)",
        report.tick,
        report.nodes,
        report.edges,
        substrate.relation_count(),
        substrate.entanglement.active_count(),
        report.active_nodes,
        wave.active_ratio * 100.0,
    );
    draw_text(&details, 28.0, 65.0, 18.0, LIGHTGRAY);
    draw_text(
        &format!(
            "PESOS PAGINADOS={} aristas={} macronodos={} activos={}",
            paged_model.model,
            paged_model.logical_edges,
            paged_model.nodes.len(),
            paged_model.active_count(),
        ),
        28.0,
        146.0,
        15.0,
        Color::from_rgba(95, 235, 255, 255),
    );
    let raw_ids = transformer
        .history
        .iter()
        .rev()
        .take(8)
        .rev()
        .map(|dream| dream.id.to_string())
        .collect::<Vec<_>>()
        .join(" → ");
    let transformer_state = transformer
        .last_error
        .as_deref()
        .map(|error| format!("ERROR {error}"))
        .unwrap_or_else(|| "GENERANDO".to_string());
    draw_text(
        &format!(
            "CICLO {} · {} | TRANSFORMER {} | IDs=[{}] gen={} | macro={} cambios={} | rutas={} recalls={} ctx_inyectado={} | ciclos +{} -{} | {}",
            sleep_cycle.cycle,
            sleep_cycle.phase.label(),
            transformer.model_name,
            raw_ids,
            transformer.generated,
            transformer.macro_injections,
            transformer.macro_top_changes,
            router.registry.routes().len(),
            router.recalls,
            transformer.route_context_injections,
            sleep_cycle.accepted_cycles,
            sleep_cycle.rejected_cycles,
            transformer_state,
        ),
        28.0,
        166.0,
        14.0,
        Color::from_rgba(255, 110, 235, 255),
    );
    draw_text(
        &format!(
            "E={:.3}  F={:.3}  σ²={:.4}  A={:.3}  |ψ|={:.3}  coherencia={:.3}  fuerza={:.3}  T={:.3}",
            report.mean_energy,
            report.free_energy_proxy,
            report.state_variance,
            report.mean_amplitude,
            wave.rms_state,
            wave.phase_coherence,
            wave.mean_force,
            wave.mean_temperature,
        ),
        28.0,
        87.0,
        17.0,
        Color::from_rgba(190, 220, 245, 255),
    );
    draw_text(
        &format!("EPR: +{epr_created} / -{epr_destroyed} en el último frame de sueño"),
        28.0,
        106.0,
        16.0,
        YELLOW,
    );
    let overlay_details = if view_3d {
        format!(
            "RQM nodos={} grado_max={} coh={:.3} incertidumbre={:.3}  |  EPR nodos={} grado_max={} coh={:.3} entropía={:.3} calor={:.3}",
            network.rqm_nodes,
            network.max_rqm_degree,
            network.mean_rqm_coherence,
            network.mean_rqm_uncertainty,
            network.epr_nodes,
            network.max_epr_degree,
            network.mean_epr_coherence,
            network.mean_epr_entropy,
            network.mean_epr_heat,
        )
    } else {
        "OVERLAY RQM/EPR omitido en 2D para rendimiento; conteos totales visibles arriba"
            .to_string()
    };
    draw_text(
        &overlay_details,
        28.0,
        126.0,
        15.0,
        Color::from_rgba(200, 170, 255, 255),
    );
    draw_text(
        &sleep_cycle.last_summary,
        28.0,
        185.0,
        14.0,
        Color::from_rgba(210, 185, 255, 255),
    );
    draw_text(
        &format!(
            "vista={} (2D por defecto)  [Tab] cambio manual 2D/3D  [Espacio] pausa  [E] enlaces: {}  [R] reiniciar  [Esc] gate y guardar",
            if view_3d { "3D" } else { "2D" },
            if show_edges { "visibles" } else { "ocultos" }
        ),
        28.0,
        204.0,
        16.0,
        GRAY,
    );
}

fn draw_substrate(substrate: &NativeThermoCdtSubstrate, show_edges: bool) {
    let left = 42.0;
    let top = 230.0;
    let width = screen_width() - 84.0;
    let height = screen_height() - 355.0;
    let slices = substrate.config.slices;
    let nodes_per_slice = substrate.config.nodes_per_slice;
    let x_step = width / (slices.saturating_sub(1).max(1) as f32);
    let y_step = height / (nodes_per_slice.saturating_sub(1).max(1) as f32);

    draw_rectangle_lines(
        left - 14.0,
        top - 14.0,
        width + 28.0,
        height + 28.0,
        1.0,
        DARKGRAY,
    );
    for slice in 0..slices {
        let x = left + slice as f32 * x_step;
        draw_line(
            x,
            top - 8.0,
            x,
            top + height + 8.0,
            1.0,
            Color::from_rgba(28, 48, 79, 150),
        );
        draw_text(&format!("t{slice}"), x - 10.0, top - 18.0, 15.0, GRAY);
    }

    if show_edges {
        for edge in 0..substrate.edge_count() {
            let a = substrate.edge_a[edge];
            let b = substrate.edge_b[edge];
            let (ax, ay) = node_position(a, nodes_per_slice, left, top, x_step, y_step);
            let (bx, by) = node_position(b, nodes_per_slice, left, top, x_step, y_step);
            let color = match substrate.edge_kind[edge] {
                NativeCdtEdgeKind::Spatial => Color::from_rgba(56, 82, 120, 32),
                NativeCdtEdgeKind::Temporal => Color::from_rgba(80, 180, 220, 58),
            };
            draw_line(ax, ay, bx, by, 1.0, color);
        }
    }

    for node in 0..substrate.node_count() {
        let (x, y) = node_position(node, nodes_per_slice, left, top, x_step, y_step);
        let activation = substrate.activation[node].abs().clamp(0.0, 1.0);
        let energy = substrate.energy[node].abs().clamp(0.0, 3.0) / 3.0;
        let phase = substrate.phase[node].sin() * 0.5 + 0.5;
        let color = Color::new(
            (0.12 + 0.88 * energy).min(1.0),
            (0.18 + 0.72 * phase).min(1.0),
            (0.35 + 0.65 * (1.0 - energy)).min(1.0),
            0.65 + 0.35 * activation,
        );
        let radius = 2.2 + 3.8 * substrate.amplitude[node].clamp(0.0, 1.0);
        draw_circle(x, y, radius, color);
        if activation > 0.55 {
            draw_circle_lines(
                x,
                y,
                radius + 2.0,
                1.0,
                Color::from_rgba(245, 250, 255, 140),
            );
        }
    }
}

fn draw_substrate_3d(
    substrate: &NativeThermoCdtSubstrate,
    epr_links: &[cdt_rqm_epr::entanglement::EntanglementLink],
    relations: &[(ObserverId, usize, usize, f32, f32, f32, f32, u64)],
    network: &NetworkOverlay,
    paged_model: &PagedModelVisual,
    transformer: &TransformerDream,
    sleep_cycle: &InfiniteSleepCycle,
    pilot_waves: &[PilotWave],
    epr_events: &[EprEvent],
    show_edges: bool,
) {
    let time = get_time() as f32;
    let yaw = time * 0.12;
    let camera = Camera3D {
        position: vec3(17.0 * yaw.cos(), 7.5, 17.0 * yaw.sin()),
        up: vec3(0.0, 1.0, 0.0),
        target: vec3(0.0, 0.0, 0.0),
        fovy: 39.0,
        ..Default::default()
    };
    set_camera(&camera);

    let slices = substrate.config.slices;
    let nodes_per_slice = substrate.config.nodes_per_slice;
    draw_paged_model_3d(paged_model, show_edges);
    draw_transformer_layers_3d(transformer, sleep_cycle, substrate, show_edges);

    if show_edges {
        for edge in 0..substrate.edge_count() {
            let a = node_position_3d(substrate, substrate.edge_a[edge], nodes_per_slice, slices);
            let b = node_position_3d(substrate, substrate.edge_b[edge], nodes_per_slice, slices);
            let color = match substrate.edge_kind[edge] {
                NativeCdtEdgeKind::Spatial => Color::from_rgba(55, 95, 145, 22),
                NativeCdtEdgeKind::Temporal => Color::from_rgba(60, 195, 240, 62),
            };
            draw_line_3d(a, b, color);
        }

        // Todas las relaciones RQM entrenadas, codificadas por coherencia.
        for &(_, source, target, _, _, coherence, uncertainty, _) in relations {
            if source >= substrate.node_count() || target >= substrate.node_count() {
                continue;
            }
            let a = node_position_3d(substrate, source, nodes_per_slice, slices);
            let b = node_position_3d(substrate, target, nodes_per_slice, slices);
            let alpha =
                (0.08 + coherence.max(0.0) * 0.22) * (1.0 - uncertainty.clamp(0.0, 1.0) * 0.55);
            draw_line_3d(a, b, Color::new(0.48, 0.3, 1.0, alpha));
        }
    }

    // EPR activo: puentes magenta externos a las conexiones CDT.
    for link in epr_links {
        let a = node_position_3d(substrate, link.a, nodes_per_slice, slices);
        let b = node_position_3d(substrate, link.b, nodes_per_slice, slices);
        let intensity = (0.35 + link.coherence * 0.65).clamp(0.0, 1.0);
        let color = Color::new(1.0, 0.1 + 0.55 * intensity, 0.95, intensity);
        draw_epr_arc(a, b, 0.55 + link.coherence, color);
    }

    // Eventos EPR: verde al crearse, rojo al destruirse; se desvanecen.
    for event in epr_events {
        let age = substrate.tick().saturating_sub(event.born_tick) as f32
            / EPR_EVENT_LIFETIME_TICKS as f32;
        let alpha = (1.0 - age).clamp(0.0, 1.0);
        let a = node_position_3d(substrate, event.a, nodes_per_slice, slices);
        let b = node_position_3d(substrate, event.b, nodes_per_slice, slices);
        let color = if event.created {
            Color::new(0.1, 1.0, 0.45, alpha)
        } else {
            Color::new(1.0, 0.12, 0.08, alpha)
        };
        draw_epr_arc(a, b, 1.4 + alpha, color);
        draw_sphere(a, 0.14 + alpha * 0.12, None, color);
        draw_sphere(b, 0.14 + alpha * 0.12, None, color);
    }

    // Frentes de la onda piloto: cascarones cian que se expanden desde la activación.
    for wave in pilot_waves {
        let age =
            substrate.tick().saturating_sub(wave.born_tick) as f32 / WAVE_LIFETIME_TICKS as f32;
        let mut center = Vec3::ZERO;
        let mut valid = 0.0;
        for &node in &wave.seeds {
            if node < substrate.node_count() {
                center += node_position_3d(substrate, node, nodes_per_slice, slices);
                valid += 1.0;
            }
        }
        if valid > 0.0 {
            center /= valid;
            let alpha = (1.0 - age).clamp(0.0, 1.0);
            draw_sphere_wires(
                center,
                0.25 + age * 5.2,
                None,
                Color::new(0.1, 0.85, 1.0, alpha * 0.75),
            );
        }
    }

    let attractors = attractor_nodes(substrate, 8);
    for node in 0..substrate.node_count() {
        let position = node_position_3d(substrate, node, nodes_per_slice, slices);
        let activation = substrate.activation[node].abs().clamp(0.0, 1.0);
        let energy = substrate.energy[node].abs().clamp(0.0, 3.0) / 3.0;
        let phase = substrate.phase[node].sin() * 0.5 + 0.5;
        let color = Color::new(
            (0.12 + 0.88 * energy).min(1.0),
            (0.18 + 0.72 * phase).min(1.0),
            (0.35 + 0.65 * (1.0 - energy)).min(1.0),
            1.0,
        );
        draw_cube(
            position,
            vec3(
                0.055 + 0.055 * substrate.amplitude[node].clamp(0.0, 1.0),
                0.055 + 0.055 * substrate.amplitude[node].clamp(0.0, 1.0),
                0.055 + 0.055 * substrate.amplitude[node].clamp(0.0, 1.0),
            ),
            None,
            color,
        );
        if activation > 0.55 {
            draw_sphere(position, 0.22, None, Color::from_rgba(130, 240, 255, 105));
        }
        let rqm_load = network.rqm_degree[node] as f32 / network.max_rqm_degree.max(1) as f32;
        if rqm_load > 0.0 {
            draw_sphere_wires(
                position,
                0.13 + rqm_load.sqrt() * 0.20,
                None,
                Color::new(0.52, 0.28, 1.0, 0.25 + rqm_load * 0.65),
            );
        }
        let epr_load = network.epr_degree[node] as f32 / network.max_epr_degree.max(1) as f32;
        if epr_load > 0.0 {
            draw_sphere_wires(
                position,
                0.17 + epr_load.sqrt() * 0.25,
                None,
                Color::new(1.0, 0.12, 0.82, 0.32 + epr_load * 0.68),
            );
        }
        if attractors.contains(&node) {
            draw_sphere_wires(position, 0.34, None, Color::from_rgba(255, 210, 45, 190));
        }
    }
}

fn draw_paged_model_3d(model: &PagedModelVisual, show_edges: bool) {
    let count = model.nodes.len().max(1);
    for (index, node) in model.nodes.iter().enumerate() {
        let position = macro_node_position(index, count);
        if show_edges && index > 0 {
            draw_line_3d(
                macro_node_position(index - 1, count),
                position,
                Color::from_rgba(40, 130, 180, 45),
            );
        }
        let active = node.activation.clamp(0.0, 1.0);
        let energy = (node.energy * 200.0).clamp(0.0, 1.0);
        let phase = node.phase.sin() * 0.5 + 0.5;
        let color = if active > 0.02 {
            Color::new(1.0, 0.35 + active * 0.65, 0.08, 0.75 + active * 0.25)
        } else {
            Color::new(0.08 + energy * 0.22, 0.25 + phase * 0.35, 0.55, 0.55)
        };
        let size_scale = (node.values.max(1) as f32).log10() / 8.0;
        let radius =
            0.05 + size_scale * 0.10 + node.amplitude.clamp(0.0, 0.2) * 1.4 + active * 0.16;
        draw_sphere(position, radius, None, color);
        if active > 0.1 {
            draw_sphere_wires(
                position,
                radius + 0.13 + active * 0.16,
                None,
                Color::new(0.2, 0.95, 1.0, active),
            );
        }
    }
}

fn draw_transformer_layers_3d(
    transformer: &TransformerDream,
    sleep_cycle: &InfiniteSleepCycle,
    substrate: &NativeThermoCdtSubstrate,
    show_edges: bool,
) {
    let layers = transformer.layers.max(1);
    for layer in 0..layers {
        let position = transformer_layer_position(layer, layers);
        if show_edges && layer > 0 {
            draw_line_3d(
                transformer_layer_position(layer - 1, layers),
                position,
                Color::from_rgba(255, 70, 220, 125),
            );
        }
        let travel = if transformer.generated == 0 {
            0.0
        } else {
            let active_layer = transformer.generated as usize % layers;
            let distance = active_layer.abs_diff(layer) as f32;
            (-distance * 0.42).exp() * transformer.layer_pulse
        };
        let radius = 0.13 + travel * 0.24;
        draw_sphere(
            position,
            radius,
            None,
            Color::new(0.65 + travel * 0.35, 0.12, 0.72 + travel * 0.28, 0.82),
        );
        draw_sphere_wires(
            position,
            radius + 0.08,
            None,
            Color::new(1.0, 0.35 + travel * 0.45, 0.95, 0.45 + travel * 0.55),
        );
    }
    let Some(last) = transformer.history.back() else {
        return;
    };
    if sleep_cycle.phase != InfiniteSleepPhase::LucidTransformer {
        return;
    }
    for layer in 0..layers {
        let source = transformer_layer_position(layer, layers);
        for projection in 0..2 {
            let node = stable_hash(&("transformer_layer_contact", last.id, layer, projection))
                as usize
                % substrate.node_count().max(1);
            let target = node_position_3d(
                substrate,
                node,
                substrate.config.nodes_per_slice,
                substrate.config.slices,
            );
            let depth = (layer + 1) as f32 / layers as f32;
            draw_line_3d(
                source,
                target,
                Color::new(
                    0.72 + depth * 0.28,
                    0.12 + depth * 0.22,
                    0.92,
                    0.18 + last.confidence * 0.62,
                ),
            );
            draw_sphere(
                target,
                0.08 + last.confidence * 0.08,
                None,
                Color::new(1.0, 0.2, 0.82, 0.7),
            );
        }
    }
}

fn transformer_layer_position(layer: usize, layers: usize) -> Vec3 {
    let progress = (layer as f32 + 0.5) / layers.max(1) as f32;
    let angle = progress * std::f32::consts::TAU * 1.35;
    vec3(
        angle.cos() * 9.0,
        (progress - 0.5) * 8.2,
        angle.sin() * 9.0,
    )
}

fn macro_node_position(index: usize, count: usize) -> Vec3 {
    let y = 1.0 - 2.0 * (index as f32 + 0.5) / count as f32;
    let radius = (1.0 - y * y).sqrt();
    let angle = index as f32 * 2.399_963_1;
    vec3(
        angle.cos() * radius * 7.2,
        y * 5.4,
        angle.sin() * radius * 6.3,
    )
}

fn node_position_3d(
    substrate: &NativeThermoCdtSubstrate,
    node: usize,
    nodes_per_slice: usize,
    slices: usize,
) -> Vec3 {
    let slice = node / nodes_per_slice;
    let offset = node % nodes_per_slice;
    let slice_x = ((slice as f32 + 0.5) / slices.max(1) as f32) * 2.0 - 1.0;
    let angle = offset as f32 * 2.399_963_1;
    let disk_radius = ((offset as f32 + 0.5) / nodes_per_slice.max(1) as f32).sqrt();
    let cross_section = (1.0 - 0.78 * slice_x * slice_x).max(0.12).sqrt();
    // La red base no cambia de topología. Este desplazamiento visualiza su
    // relajación/contracción térmica según el estado físico de cada nodo.
    let deformation = 1.0
        + substrate.thermal_state[node].tanh() * 0.16
        + substrate.phase[node].sin() * substrate.amplitude[node].min(2.0) * 0.035;
    vec3(
        slice_x * 5.4 + substrate.thermal_state[node] * 0.18,
        angle.cos() * disk_radius * cross_section * 3.8 * deformation,
        angle.sin() * disk_radius * cross_section * 3.2 * deformation,
    )
}

fn draw_epr_arc(a: Vec3, b: Vec3, lift: f32, color: Color) {
    let midpoint = (a + b) * 0.5 + vec3(0.0, lift, 0.0);
    draw_line_3d(a, midpoint, color);
    draw_line_3d(midpoint, b, color);
}

fn build_network_overlay(
    nodes: usize,
    relations: &[(ObserverId, usize, usize, f32, f32, f32, f32, u64)],
    epr_links: &[cdt_rqm_epr::entanglement::EntanglementLink],
) -> NetworkOverlay {
    let mut rqm_degree = vec![0usize; nodes];
    let mut epr_degree = vec![0usize; nodes];
    let mut rqm_coherence = 0.0;
    let mut rqm_uncertainty = 0.0;
    for &(_, source, target, _, _, coherence, uncertainty, _) in relations {
        if source < nodes {
            rqm_degree[source] += 1;
        }
        if target < nodes {
            rqm_degree[target] += 1;
        }
        rqm_coherence += coherence;
        rqm_uncertainty += uncertainty;
    }
    let mut epr_coherence = 0.0;
    let mut epr_entropy = 0.0;
    let mut epr_heat = 0.0;
    for link in epr_links {
        if link.a < nodes {
            epr_degree[link.a] += 1;
        }
        if link.b < nodes {
            epr_degree[link.b] += 1;
        }
        epr_coherence += link.coherence;
        epr_entropy += link.entropy;
        epr_heat += link.heat;
    }
    let relation_count = relations.len().max(1) as f32;
    let epr_count = epr_links.len().max(1) as f32;
    NetworkOverlay {
        rqm_nodes: rqm_degree.iter().filter(|&&degree| degree > 0).count(),
        epr_nodes: epr_degree.iter().filter(|&&degree| degree > 0).count(),
        max_rqm_degree: rqm_degree.iter().copied().max().unwrap_or(0),
        max_epr_degree: epr_degree.iter().copied().max().unwrap_or(0),
        mean_rqm_coherence: rqm_coherence / relation_count,
        mean_rqm_uncertainty: rqm_uncertainty / relation_count,
        mean_epr_coherence: epr_coherence / epr_count,
        mean_epr_entropy: epr_entropy / epr_count,
        mean_epr_heat: epr_heat / epr_count,
        rqm_degree,
        epr_degree,
    }
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

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn node_position(
    node: usize,
    nodes_per_slice: usize,
    left: f32,
    top: f32,
    x_step: f32,
    y_step: f32,
) -> (f32, f32) {
    let slice = node / nodes_per_slice;
    let offset = node % nodes_per_slice;
    (left + slice as f32 * x_step, top + offset as f32 * y_step)
}

fn draw_charts(energy: &[f32], variance: &[f32], coherence: &[f32]) {
    let left = 42.0;
    let top = screen_height() - 190.0;
    let width = screen_width() - 84.0;
    let height = 42.0;
    draw_chart(energy, left, top, width, height, "ENERGÍA LIBRE", LIME);
    draw_chart(
        variance,
        left,
        top + 56.0,
        width,
        height,
        "VARIANZA DEL ESTADO",
        ORANGE,
    );
    draw_chart(
        coherence,
        left,
        top + 112.0,
        width,
        height,
        "COHERENCIA DE FASE",
        SKYBLUE,
    );
}

fn draw_chart(
    history: &[f32],
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    label: &str,
    color: Color,
) {
    draw_text(label, left, top - 7.0, 14.0, LIGHTGRAY);
    draw_rectangle_lines(left, top, width, height, 1.0, DARKGRAY);
    if history.len() < 2 {
        return;
    }
    let min = history.iter().copied().fold(f32::INFINITY, f32::min);
    let max = history.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(0.001);
    for index in 1..history.len() {
        let x0 = left + (index - 1) as f32 / (history.len() - 1) as f32 * width;
        let x1 = left + index as f32 / (history.len() - 1) as f32 * width;
        let y0 = top + height - (history[index - 1] - min) / range * height;
        let y1 = top + height - (history[index] - min) / range * height;
        draw_line(x0, y0, x1, y1, 1.5, color);
    }
    draw_text(&format!("{max:.3}"), left + 5.0, top + 16.0, 14.0, GRAY);
    draw_text(
        &format!("{min:.3}"),
        left + 5.0,
        top + height - 5.0,
        14.0,
        GRAY,
    );
}

#[derive(Clone, Copy)]
struct WaveMetrics {
    rms_state: f32,
    phase_coherence: f32,
    mean_force: f32,
    mean_temperature: f32,
    active_ratio: f32,
}

fn wave_metrics(substrate: &NativeThermoCdtSubstrate) -> WaveMetrics {
    let nodes = substrate.node_count().max(1) as f32;
    let rms_state = (substrate
        .thermal_state
        .iter()
        .map(|state| state * state)
        .sum::<f32>()
        / nodes)
        .sqrt();
    let (phase_x, phase_y) = substrate.phase.iter().fold((0.0, 0.0), |(x, y), phase| {
        (x + phase.cos(), y + phase.sin())
    });
    WaveMetrics {
        rms_state,
        phase_coherence: (phase_x.hypot(phase_y) / nodes).clamp(0.0, 1.0),
        mean_force: substrate
            .pilot_force
            .iter()
            .map(|force| force.abs())
            .sum::<f32>()
            / nodes,
        mean_temperature: substrate.temperature.iter().sum::<f32>() / nodes,
        active_ratio: substrate
            .activation
            .iter()
            .filter(|&&value| value > 0.05)
            .count() as f32
            / nodes,
    }
}

fn push_history(history: &mut Vec<f32>, value: f32) {
    history.push(value);
    if history.len() > 240 {
        history.remove(0);
    }
}

fn save_sleep_checkpoint(
    substrate: &NativeThermoRqmEprSubstrate,
    output: &str,
) -> Result<(), String> {
    if let Some(parent) = Path::new(output).parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut out = String::from("NATIVE_THERMO_RQM_EPR_CLEAN_STATE_V1\n");
    out.push_str("stats 0 0 0 0 0 0 1 0\n");
    let cdt = substrate.thermal.config;
    out.push_str(&format!(
        "thermal_config {} {} {} {} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {}\n",
        cdt.slices,
        cdt.nodes_per_slice,
        cdt.spatial_degree,
        cdt.temporal_degree,
        cdt.temperature,
        cdt.dt,
        cdt.diffusion,
        cdt.confinement,
        cdt.pilot_gain,
        cdt.phase_coupling,
        cdt.amplitude_decay,
        cdt.state_clamp,
        cdt.seed
    ));
    let rqm = substrate.config;
    out.push_str(&format!(
        "rqm_config {:.7} {:.7} {:.7} {:.7} {:.7} {} {} {:.7} {:.7} {} {} {} {} {} {}\n",
        rqm.amplitude_learning_rate,
        rqm.coherence_learning_rate,
        rqm.uncertainty_learning_rate,
        rqm.phase_learning_rate,
        rqm.amplitude_decay,
        rqm.thermal_steps_per_train,
        rqm.thermal_steps_per_query,
        rqm.thermal_score_gain,
        rqm.thermal_activation_margin,
        usize::from(rqm.collect_query_diagnostics),
        rqm.max_candidates,
        rqm.max_pilot_window_nodes,
        rqm.sampling_block_size,
        rqm.sampling_schedule_rounds,
        rqm.max_sampling_blocks
    ));
    out.push_str(&format!("nodes {}\n", substrate.thermal.node_count()));
    for node in 0..substrate.thermal.node_count() {
        out.push_str(&format!(
            "n {} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7}\n",
            node,
            substrate.thermal.thermal_state[node],
            substrate.thermal.amplitude[node],
            substrate.thermal.phase[node],
            substrate.thermal.temperature[node],
            substrate.thermal.energy[node],
            substrate.thermal.activation[node]
        ));
    }
    let relations = substrate.relation_entries().collect::<Vec<_>>();
    out.push_str(&format!("relations {}\n", relations.len()));
    for (observer, source, target, amplitude, phase, coherence, uncertainty, last_tick) in relations
    {
        out.push_str(&format!(
            "r {} {} {} {:.7} {:.7} {:.7} {:.7} {}\n",
            observer.0, source, target, amplitude, phase, coherence, uncertainty, last_tick
        ));
    }
    out.push_str("entanglement_begin\n");
    out.push_str(&substrate.entanglement.serialize_persistent_state());
    out.push_str("entanglement_end\nend\n");
    fs::write(output, out).map_err(|error| error.to_string())
}

fn draw_legend() {
    let y = screen_height() - 23.0;
    draw_circle(36.0, y - 5.0, 5.0, ORANGE);
    draw_text("energía", 47.0, y, 15.0, GRAY);
    draw_circle(130.0, y - 5.0, 5.0, SKYBLUE);
    draw_text("fase", 141.0, y, 15.0, GRAY);
    draw_line(
        205.0,
        y - 5.0,
        227.0,
        y - 5.0,
        2.0,
        Color::from_rgba(80, 180, 220, 150),
    );
    draw_text("enlace temporal", 235.0, y, 15.0, GRAY);
    draw_line(375.0, y - 5.0, 397.0, y - 5.0, 2.0, VIOLET);
    draw_text("RQM entrenado", 405.0, y, 15.0, GRAY);
    draw_line(520.0, y - 5.0, 542.0, y - 5.0, 2.0, MAGENTA);
    draw_text("EPR activo", 550.0, y, 15.0, GRAY);
    draw_circle(645.0, y - 5.0, 5.0, GREEN);
    draw_text("EPR creado", 656.0, y, 15.0, GRAY);
    draw_circle(751.0, y - 5.0, 5.0, RED);
    draw_text("EPR destruido", 762.0, y, 15.0, GRAY);
    draw_circle_lines(887.0, y - 5.0, 6.0, 1.0, YELLOW);
    draw_text("atractor", 899.0, y, 15.0, GRAY);
    draw_circle(980.0, y - 5.0, 5.0, BLUE);
    draw_text("modelo inactivo", 991.0, y, 15.0, GRAY);
    draw_circle(1115.0, y - 5.0, 6.0, ORANGE);
    draw_text("modelo activado", 1127.0, y, 15.0, GRAY);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_forward_adapter_modulates_real_logits() {
        let (sender, receiver) = mpsc::channel();
        sender
            .send(MacroForwardInjection {
                weights: vec![-1.0, 1.0],
                generation: 1,
            })
            .unwrap();
        let adapter = MacroForwardAdapter::new(receiver);
        let mut logits = vec![1.0, 0.9, 0.2];
        let report = adapter.apply(&mut logits);
        assert!(report.applied);
        assert!(report.changed_top);
        assert!(logits[1] > logits[0]);
    }
}
