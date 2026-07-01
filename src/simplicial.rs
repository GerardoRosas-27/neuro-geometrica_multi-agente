use crate::geometry::Vec2;
use crate::mesh_engine::{FractalMeshConfig, MeshConfig, MeshTopology, SimplicialMeshEngine};
use crate::relational_field::{
    CollapseReport, ObserverId, RelationalFieldConfig, RelationalFieldSubstrate, SimplexPhaseReport,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::Path;

const ASSOCIATIVE_EDGE_THRESHOLD: f32 = 1.05;
const ASSOCIATIVE_CELL_THRESHOLD: f32 = 1.10;
const FOCUS_EDGE_LEARNING_SCALE: f32 = 250.0;
const FOCUS_EDGE_MAX: f32 = 10_000.0;
const FOCUS_PROMOTION_TOP_TARGETS: usize = 512;
const MAX_FOCUS_SOURCE_VERTICES: usize = 16;
const MAX_FOCUS_TARGET_VERTICES: usize = 16;
const MAX_ASSOCIATIVE_CELL_VERTICES: usize = 96;
pub const GOLDEN_UTILITY_THRESHOLD: f32 = 0.618_034;
const OSCILLATORY_REGION_SIZE: usize = 64;

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: usize,
    pub position: Vec2,
    pub depth: f32,
    pub velocity: Vec2,
    pub depth_velocity: f32,
    pub activation: bool,
    pub surprise: f32,
    pub refractory: u8,
}

impl Agent {
    fn new(id: usize, position: Vec2) -> Self {
        Self {
            id,
            position,
            depth: 0.0,
            velocity: Vec2::ZERO,
            depth_velocity: 0.0,
            activation: false,
            surprise: 0.0,
            refractory: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub rest_length: f32,
    pub weight: f32,
    pub age: u32,
    pub last_active_tick: u64,
    pub consolidated: bool,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub struct Simplex2 {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub target_area: f32,
}

#[derive(Clone, Debug)]
pub struct Simplex3 {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
    pub target_volume: f32,
}

#[derive(Clone, Debug)]
pub struct SemanticCell {
    pub id: usize,
    pub vertices: Vec<usize>,
    pub edges: Vec<usize>,
    pub weight: f32,
    pub age: u32,
    pub last_active_tick: u64,
    pub active: bool,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Spike {
    pub source: usize,
    pub target: usize,
    pub ttl: u8,
}

#[derive(Clone, Debug)]
pub struct SimplicialConfig {
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub elasticity: f32,
    pub damping: f32,
    pub activation_threshold: f32,
    pub simplex_area_weight: f32,
    pub max_active_agents: usize,
    pub inhibition_decay: f32,
    pub max_spikes_per_step: usize,
    pub local_inhibition_decay: f32,
    pub refractory_ticks: u8,
    pub rhythm_period: u64,
    pub rhythm_amplitude: f32,
    pub forgetting_rate: f32,
    pub prune_below_weight: f32,
    pub consolidate_after: u32,
    pub consolidated_forgetting_scale: f32,
    pub max_episodes: usize,
    pub replay_interval: u64,
    pub replay_batch: usize,
    pub replay_learning_rate: f32,
    pub causal_learning_rate: f32,
    pub contradiction_learning_rate: f32,
    pub contradiction_energy_weight: f32,
    pub simplex3_weight: f32,
    pub hyperbolic_curvature: f32,
    pub seed: u64,
}

impl Default for SimplicialConfig {
    fn default() -> Self {
        Self {
            width: 18,
            height: 12,
            spacing: 38.0,
            elasticity: 0.015,
            damping: 0.88,
            activation_threshold: 0.72,
            simplex_area_weight: 0.0008,
            max_active_agents: 96,
            inhibition_decay: 0.18,
            max_spikes_per_step: 512,
            local_inhibition_decay: 1.0,
            refractory_ticks: 0,
            rhythm_period: 32,
            rhythm_amplitude: 0.0,
            forgetting_rate: 0.0,
            prune_below_weight: 0.02,
            consolidate_after: 4,
            consolidated_forgetting_scale: 0.2,
            max_episodes: 256,
            replay_interval: 0,
            replay_batch: 4,
            replay_learning_rate: 0.03,
            causal_learning_rate: 0.08,
            contradiction_learning_rate: 0.2,
            contradiction_energy_weight: 1.0,
            simplex3_weight: 0.0001,
            hyperbolic_curvature: 0.0,
            seed: 7,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Episode {
    pub pattern: Vec<usize>,
    pub strength: f32,
    pub created_tick: u64,
    pub context: Vec<usize>,
    pub predicted_next: Vec<usize>,
    pub prediction_error: f32,
    pub novelty: f32,
}

#[derive(Clone, Debug)]
pub struct PredictionReport {
    pub predicted_agents: Vec<(usize, f32)>,
    pub matched_agents: usize,
    pub expected_agents: usize,
    pub precision: f32,
    pub recall: f32,
}

#[derive(Clone, Debug)]
pub struct PatternPredictionReport {
    pub predicted_pattern: Vec<(usize, f32)>,
    pub observed_pattern: Vec<usize>,
    pub matched_agents: usize,
    pub precision: f32,
    pub recall: f32,
    pub prediction_error: f32,
}

#[derive(Clone, Debug)]
pub struct EpisodicMatch {
    pub pattern: Vec<usize>,
    pub context: Vec<usize>,
    pub similarity: f32,
    pub strength: f32,
    pub age_ticks: u64,
    pub prediction_error: f32,
}

#[derive(Clone, Debug)]
pub struct EpisodicRecall {
    pub matches: Vec<EpisodicMatch>,
    pub merged_pattern: Vec<(usize, f32)>,
}

#[derive(Clone, Debug)]
pub struct AttentionReport {
    pub goal_agents: Vec<usize>,
    pub context_agents: Vec<(usize, f32)>,
    pub boosted_agents: usize,
    pub suppressed_agents: usize,
}

#[derive(Clone, Debug)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub active_pattern: Vec<usize>,
    pub projection: ConceptProjection,
    pub free_energy: f32,
}

#[derive(Clone, Debug)]
pub struct RolloutStep {
    pub step: usize,
    pub predicted_pattern: Vec<(usize, f32)>,
    pub snapshot: WorldSnapshot,
}

#[derive(Clone, Debug)]
pub struct RolloutReport {
    pub initial_pattern: Vec<usize>,
    pub terminal_pattern: Vec<usize>,
    pub energy_delta: f32,
    pub steps: Vec<RolloutStep>,
}

#[derive(Clone, Debug)]
pub struct PlanStep {
    pub agent: usize,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct PlanReport {
    pub start: Vec<usize>,
    pub goal: Vec<usize>,
    pub horizon: usize,
    pub reached_goal: bool,
    pub path: Vec<PlanStep>,
    pub score: f32,
    pub terminal_prediction: Vec<(usize, f32)>,
}

#[derive(Clone, Debug)]
pub struct RouteOptimizationReport {
    pub candidates: usize,
    pub rewarded_paths: usize,
    pub evaporated_paths: usize,
    pub prediction: PredictionReport,
}

#[derive(Clone, Debug)]
pub struct PlasticityStats {
    pub tick: u64,
    pub active_edges: usize,
    pub associative_edges: usize,
    pub consolidated_edges: usize,
    pub semantic_cells: usize,
    pub episodes: usize,
    pub causal_edges: usize,
    pub contradiction_edges: usize,
    pub tetrahedra: usize,
}

#[derive(Clone, Debug)]
pub struct ConceptProjection {
    pub top_agents: Vec<(usize, f32)>,
}

#[derive(Clone, Debug)]
pub struct EnergyStats {
    pub total_free_energy: f32,
    pub active_agents: usize,
    pub active_spikes: usize,
}

#[derive(Clone, Debug)]
pub struct PersistentStateReport {
    pub agents: usize,
    pub edges: usize,
    pub causal_edges: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaveBand {
    Delta,
    Theta,
    Alpha,
    Beta,
    Gamma,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BrainMode {
    Exploration,
    Focus,
    SleepReplay,
}

#[derive(Clone, Copy, Debug)]
pub struct OscillationGains {
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub excitability: f32,
    pub inhibition: f32,
    pub replay: f32,
    pub plasticity: f32,
    pub prediction: f32,
}

#[derive(Clone, Debug)]
pub struct OscillationStats {
    pub enabled: bool,
    pub mode: BrainMode,
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub regions: usize,
    pub delta_regions: usize,
    pub theta_regions: usize,
    pub alpha_regions: usize,
    pub beta_regions: usize,
    pub gamma_regions: usize,
}

#[derive(Clone, Debug)]
pub struct SimplicialNetwork {
    pub agents: Vec<Agent>,
    pub edges: Vec<Edge>,
    pub simplices: Vec<Simplex2>,
    pub tetrahedra: Vec<Simplex3>,
    pub semantic_cells: Vec<SemanticCell>,
    pub spikes: VecDeque<Spike>,
    pub config: SimplicialConfig,
    adjacency: Vec<Vec<usize>>,
    agent_to_cells: Vec<Vec<usize>>,
    edge_lookup: HashMap<(usize, usize), usize>,
    cell_lookup: HashMap<Vec<usize>, usize>,
    causal_edges: HashMap<(usize, usize), f32>,
    causal_adjacency: HashMap<usize, Vec<(usize, f32)>>,
    focus_edges: HashMap<(usize, usize), f32>,
    focus_adjacency: HashMap<usize, Vec<(usize, f32)>>,
    contradiction_edges: HashMap<(usize, usize), f32>,
    episodes: VecDeque<Episode>,
    last_episode_pattern: Vec<usize>,
    attention_goal: Vec<usize>,
    attention_context: HashMap<usize, f32>,
    world_snapshots: VecDeque<WorldSnapshot>,
    relational_field: Option<RelationalFieldSubstrate>,
    relational_observer: Option<ObserverId>,
    relational_observer_phase: f32,
    oscillations_enabled: bool,
    brain_mode: BrainMode,
    agent_regions: Vec<usize>,
    region_bands: Vec<WaveBand>,
    tick: u64,
    // Scratch buffers reutilizados entre pasos para evitar reasignaciones por frame.
    forces_buffer: Vec<Vec2>,
    inhibition_scratch: Vec<(usize, f32)>,
}

impl SimplicialNetwork {
    pub fn grid(config: SimplicialConfig) -> Self {
        let topology = SimplicialMeshEngine::grid(mesh_config_from(&config));
        Self::from_mesh_topology(config, topology)
    }

    pub fn grid_3d(config: SimplicialConfig, depth_layers: usize) -> Self {
        let topology = SimplicialMeshEngine::grid_3d(mesh_config_from(&config), depth_layers);
        Self::from_mesh_topology(config, topology)
    }

    pub fn fractal_3d(config: SimplicialConfig, fractal: FractalMeshConfig) -> Self {
        let topology = SimplicialMeshEngine::fractal_3d(mesh_config_from(&config), fractal);
        Self::from_mesh_topology(config, topology)
    }

    fn from_mesh_topology(config: SimplicialConfig, topology: MeshTopology) -> Self {
        let mut agents = topology
            .nodes
            .iter()
            .map(|node| {
                let mut agent = Agent::new(node.id, node.position);
                agent.depth = node.depth;
                agent
            })
            .collect::<Vec<_>>();
        agents.sort_by_key(|agent| agent.id);

        let mut network = Self {
            adjacency: vec![Vec::new(); agents.len()],
            agent_to_cells: vec![Vec::new(); agents.len()],
            agents,
            edges: Vec::new(),
            simplices: Vec::new(),
            tetrahedra: Vec::new(),
            semantic_cells: Vec::new(),
            spikes: VecDeque::new(),
            edge_lookup: HashMap::new(),
            cell_lookup: HashMap::new(),
            causal_edges: HashMap::new(),
            causal_adjacency: HashMap::new(),
            focus_edges: HashMap::new(),
            focus_adjacency: HashMap::new(),
            contradiction_edges: HashMap::new(),
            episodes: VecDeque::new(),
            last_episode_pattern: Vec::new(),
            attention_goal: Vec::new(),
            attention_context: HashMap::new(),
            world_snapshots: VecDeque::new(),
            relational_field: None,
            relational_observer: None,
            relational_observer_phase: 0.0,
            oscillations_enabled: false,
            brain_mode: BrainMode::Exploration,
            agent_regions: Vec::new(),
            region_bands: Vec::new(),
            tick: 0,
            forces_buffer: Vec::new(),
            inhibition_scratch: Vec::new(),
            config,
        };

        for edge in topology.edges {
            network.add_edge(edge.a, edge.b, edge.rest_length, edge.weight);
        }
        for simplex in topology.simplices {
            network.add_simplex(simplex.a, simplex.b, simplex.c);
        }
        for simplex in topology.tetrahedra {
            network.add_simplex3(simplex.a, simplex.b, simplex.c, simplex.d);
        }
        network.initialize_oscillatory_regions();
        network
    }

    pub fn inject_text_pattern(&mut self, text: &str) {
        let bytes = text.as_bytes();
        if bytes.is_empty() || self.agents.is_empty() {
            return;
        }

        let pattern = bytes
            .iter()
            .enumerate()
            .map(|(i, byte)| ((*byte as usize * 31) + i * 17) % self.agents.len())
            .collect::<Vec<_>>();
        self.inject_pattern(&pattern, 1.0, 3);
    }

    pub fn inject_pattern(&mut self, pattern: &[usize], surprise: f32, ttl: u8) {
        self.record_episode(pattern, surprise);
        for &idx in pattern {
            if idx >= self.agents.len() {
                continue;
            }
            self.agents[idx].activation = true;
            self.agents[idx].surprise = self.agents[idx].surprise.max(surprise);
            self.agents[idx].refractory = self.config.refractory_ticks;
            self.spikes.push_back(Spike {
                source: idx,
                target: idx,
                ttl,
            });
        }
    }

    pub fn reinforce_coactivation(&mut self, pattern: &[usize], learning_rate: f32) {
        for i in 0..pattern.len() {
            for j in (i + 1)..pattern.len() {
                let a = pattern[i];
                let b = pattern[j];
                if a == b || a >= self.agents.len() || b >= self.agents.len() {
                    continue;
                }
                self.reinforce_pair(a, b, learning_rate);
            }
        }
        self.reinforce_semantic_cell(pattern, learning_rate);
    }

    pub fn reinforce_coactivation_if_useful(
        &mut self,
        pattern: &[usize],
        learning_rate: f32,
        utility: f32,
    ) -> bool {
        if utility < GOLDEN_UTILITY_THRESHOLD {
            return false;
        }
        self.reinforce_coactivation(pattern, learning_rate);
        true
    }

    pub fn learn_transition(&mut self, cause: &[usize], effect: &[usize]) {
        let lr = self.config.causal_learning_rate;
        for &a in cause {
            if a >= self.agents.len() {
                continue;
            }
            for &b in effect {
                if b >= self.agents.len() || a == b {
                    continue;
                }
                let weight = self.causal_edges.entry((a, b)).or_insert(0.0);
                *weight = (*weight + lr).min(1.0);
                upsert_weighted_neighbor(&mut self.causal_adjacency, a, b, *weight);
            }
        }
    }

    fn set_causal_weight(&mut self, source: usize, target: usize, weight: f32) {
        let weight = weight.clamp(0.0, 1.0);
        if weight <= f32::EPSILON {
            self.causal_edges.remove(&(source, target));
            if let Some(neighbors) = self.causal_adjacency.get_mut(&source) {
                neighbors.retain(|(idx, _)| *idx != target);
            }
            return;
        }

        self.causal_edges.insert((source, target), weight);
        upsert_weighted_neighbor(&mut self.causal_adjacency, source, target, weight);
    }

    fn reinforce_focus_transition(&mut self, source: &[usize], target: &[usize], gain: f32) {
        if gain <= 0.0 {
            return;
        }
        let source = sample_vertices(
            compact_pattern(source, self.agents.len()),
            MAX_FOCUS_SOURCE_VERTICES,
        );
        let target = sample_vertices(
            compact_pattern(target, self.agents.len()),
            MAX_FOCUS_TARGET_VERTICES,
        );
        if source.is_empty() || target.is_empty() {
            return;
        }

        let delta = (gain * FOCUS_EDGE_LEARNING_SCALE).max(1.0);
        for from in source {
            for &to in &target {
                if from == to {
                    continue;
                }
                let weight = self.focus_edges.entry((from, to)).or_insert(0.0);
                *weight = (*weight + delta).min(FOCUS_EDGE_MAX);
                upsert_weighted_neighbor(&mut self.focus_adjacency, from, to, *weight);
            }
        }
    }

    fn rebuild_focus_adjacency(&mut self) {
        self.focus_adjacency.clear();
        for (&(source, target), &weight) in &self.focus_edges {
            upsert_weighted_neighbor(&mut self.focus_adjacency, source, target, weight);
        }
    }

    pub fn learn_contradiction(&mut self, left: &[usize], right: &[usize]) {
        let lr = self.config.contradiction_learning_rate;
        for &a in left {
            if a >= self.agents.len() {
                continue;
            }
            for &b in right {
                if b >= self.agents.len() || a == b {
                    continue;
                }
                let weight = self
                    .contradiction_edges
                    .entry(edge_key(a, b))
                    .or_insert(0.0);
                *weight = (*weight + lr).min(1.0);
            }
        }
    }

    pub fn predict_from(&self, cause: &[usize], limit: usize) -> Vec<(usize, f32)> {
        let mut scores = HashMap::<usize, f32>::new();
        for &a in cause {
            if let Some(neighbors) = self.causal_adjacency.get(&a) {
                for &(target, weight) in neighbors {
                    *scores.entry(target).or_insert(0.0) += weight;
                }
            }
        }
        let mut predicted = scores.into_iter().collect::<Vec<_>>();
        predicted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        predicted.truncate(limit);
        predicted
    }

    pub fn causal_edges_snapshot(&self) -> Vec<(usize, usize, f32)> {
        let mut edges = self
            .causal_edges
            .iter()
            .map(|(&(source, target), &weight)| (source, target, weight))
            .collect::<Vec<_>>();
        edges.sort_by(|a, b| {
            (a.0, a.1)
                .cmp(&(b.0, b.1))
                .then_with(|| b.2.total_cmp(&a.2))
        });
        edges
    }

    pub fn infer_transitive_from(
        &self,
        cause: &[usize],
        max_hops: usize,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let mut all_scores = HashMap::<usize, f32>::new();
        let mut frontier = HashMap::<usize, f32>::new();
        let source_set = cause
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();

        for &idx in cause {
            if idx < self.agents.len() {
                frontier.insert(idx, 1.0);
            }
        }

        for hop in 1..=max_hops {
            let mut next = HashMap::<usize, f32>::new();
            let hop_discount = 1.0 / hop as f32;
            for (&source, &source_score) in &frontier {
                let Some(neighbors) = self.causal_adjacency.get(&source) else {
                    continue;
                };
                for &(target, weight) in neighbors {
                    if source_set.contains(&target) {
                        continue;
                    }
                    let score = source_score * weight * hop_discount;
                    *all_scores.entry(target).or_insert(0.0) += score;
                    *next.entry(target).or_insert(0.0) += score;
                }
                if let Some(focus_neighbors) = self.focus_adjacency.get(&source) {
                    for &(target, weight) in focus_neighbors {
                        if source_set.contains(&target) {
                            continue;
                        }
                        let score = source_score * weight;
                        *all_scores.entry(target).or_insert(0.0) += score;
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        self.promote_focus_representatives(cause, cause, &mut all_scores);
        let mut predicted = all_scores.into_iter().collect::<Vec<_>>();
        predicted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        predicted.truncate(limit);
        predicted
    }

    pub fn evaluate_transitive_prediction(
        &self,
        cause: &[usize],
        expected: &[usize],
        max_hops: usize,
        limit: usize,
    ) -> PredictionReport {
        let predicted_agents = self.infer_transitive_from(cause, max_hops, limit);
        prediction_report(predicted_agents, expected)
    }

    pub fn infer_exact_hop_from(
        &self,
        cause: &[usize],
        hops: usize,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let mut frontier = HashMap::<usize, f32>::new();
        let source_set = cause
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();

        for &idx in cause {
            if idx < self.agents.len() {
                frontier.insert(idx, 1.0);
            }
        }

        for _ in 0..hops {
            let mut next = HashMap::<usize, f32>::new();
            for (&source, &source_score) in &frontier {
                let Some(neighbors) = self.causal_adjacency.get(&source) else {
                    continue;
                };
                for &(target, weight) in neighbors {
                    if source_set.contains(&target) {
                        continue;
                    }
                    *next.entry(target).or_insert(0.0) += source_score * weight;
                }
            }
            if next.is_empty() {
                return Vec::new();
            }
            frontier = next;
        }

        self.promote_focus_representatives(cause, cause, &mut frontier);
        let mut predicted = frontier.into_iter().collect::<Vec<_>>();
        predicted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        predicted.truncate(limit);
        predicted
    }

    pub fn evaluate_exact_hop_prediction(
        &self,
        cause: &[usize],
        expected: &[usize],
        hops: usize,
        limit: usize,
    ) -> PredictionReport {
        prediction_report(self.infer_exact_hop_from(cause, hops, limit), expected)
    }

    pub fn optimize_routes_to_expected(
        &mut self,
        cause: &[usize],
        expected: &[usize],
        hops: usize,
        beam_width: usize,
        evaporation: f32,
        deposit: f32,
    ) -> RouteOptimizationReport {
        let expected_set = expected
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let source_set = cause
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut frontier = cause
            .iter()
            .copied()
            .filter(|&idx| idx < self.agents.len())
            .map(|idx| (idx, 1.0_f32, Vec::<(usize, usize)>::new()))
            .collect::<Vec<_>>();

        for _ in 0..hops {
            let mut next = Vec::new();
            for (source, score, path) in frontier {
                let Some(neighbors) = self.causal_adjacency.get(&source) else {
                    continue;
                };
                for &(target, weight) in neighbors {
                    if source_set.contains(&target) {
                        continue;
                    }
                    let mut next_path = path.clone();
                    next_path.push((source, target));
                    next.push((target, score * weight, next_path));
                }
            }

            if next.is_empty() {
                return RouteOptimizationReport {
                    candidates: 0,
                    rewarded_paths: 0,
                    evaporated_paths: 0,
                    prediction: prediction_report(Vec::new(), expected),
                };
            }

            next.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            next.truncate(beam_width.max(expected.len()).max(1));
            frontier = next;
        }

        let candidates = frontier.len();
        let mut rewarded_edges = std::collections::HashSet::<(usize, usize)>::new();
        let mut evaporated_edges = std::collections::HashSet::<(usize, usize)>::new();
        let mut prediction = Vec::new();

        for (terminal, score, path) in &frontier {
            if expected_set.contains(terminal) {
                prediction.push((*terminal, *score));
                for &(source, target) in path {
                    rewarded_edges.insert((source, target));
                }
            } else {
                for &(source, target) in path {
                    evaporated_edges.insert((source, target));
                }
            }
        }

        for &(source, target) in &rewarded_edges {
            let current = *self.causal_edges.get(&(source, target)).unwrap_or(&0.0);
            self.set_causal_weight(source, target, (current + deposit).min(1.0));
        }

        let evaporation = evaporation.clamp(0.0, 1.0);
        for &(source, target) in &evaporated_edges {
            if rewarded_edges.contains(&(source, target)) {
                continue;
            }
            let current = *self.causal_edges.get(&(source, target)).unwrap_or(&0.0);
            self.set_causal_weight(source, target, current * (1.0 - evaporation));
        }

        prediction.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        prediction.truncate(expected.len());

        RouteOptimizationReport {
            candidates,
            rewarded_paths: rewarded_edges.len(),
            evaporated_paths: evaporated_edges.len(),
            prediction: prediction_report(prediction, expected),
        }
    }

    pub fn contradiction_tension(&self, left: &[usize], right: &[usize]) -> f32 {
        let mut tension = 0.0;
        for &a in left {
            for &b in right {
                if let Some(weight) = self.contradiction_edges.get(&edge_key(a, b)) {
                    tension += weight;
                }
            }
        }
        tension
    }

    pub fn evaluate_prediction(
        &self,
        cause: &[usize],
        expected: &[usize],
        limit: usize,
    ) -> PredictionReport {
        prediction_report(self.predict_from(cause, limit), expected)
    }

    pub fn predict_next_pattern(
        &self,
        current: &[usize],
        horizon: usize,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let mut scores = HashMap::<usize, f32>::new();
        let current_set = current.iter().copied().collect::<HashSet<_>>();
        let causal = if horizon <= 1 {
            self.predict_from(current, limit.max(current.len()).max(1) * 2)
        } else {
            self.infer_transitive_from(current, horizon, limit.max(current.len()).max(1) * 2)
        };

        for (idx, score) in causal {
            if !current_set.contains(&idx) {
                *scores.entry(idx).or_insert(0.0) += score;
            }
        }
        for episode in &self.episodes {
            let context_similarity = pattern_similarity(current, &episode.context);
            if context_similarity <= 0.0 {
                continue;
            }
            let age = self.tick.saturating_sub(episode.created_tick).max(1) as f32;
            let recency = 1.0 / age.sqrt();
            let gain = context_similarity
                * episode.strength
                * (1.0 + episode.novelty)
                * (1.0 + episode.prediction_error)
                * recency;
            for &idx in &episode.pattern {
                if !current_set.contains(&idx) {
                    *scores.entry(idx).or_insert(0.0) += gain;
                }
            }
        }

        let mut predicted = scores.into_iter().collect::<Vec<_>>();
        predicted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        predicted.truncate(limit);
        predicted
    }

    pub fn evaluate_pattern_prediction(
        &self,
        current: &[usize],
        observed: &[usize],
        horizon: usize,
        limit: usize,
    ) -> PatternPredictionReport {
        let predicted_pattern = self.predict_next_pattern(current, horizon, limit);
        let observed_set = observed.iter().copied().collect::<HashSet<_>>();
        let matched_agents = predicted_pattern
            .iter()
            .filter(|(idx, _)| observed_set.contains(idx))
            .count();
        let precision = matched_agents as f32 / predicted_pattern.len().max(1) as f32;
        let recall = matched_agents as f32 / observed_set.len().max(1) as f32;

        PatternPredictionReport {
            predicted_pattern,
            observed_pattern: observed_set.into_iter().collect(),
            matched_agents,
            precision,
            recall,
            prediction_error: 1.0 - ((precision + recall) * 0.5),
        }
    }

    pub fn learn_from_prediction_error(
        &mut self,
        current: &[usize],
        expected: &[usize],
        horizon: usize,
        limit: usize,
        learning_rate: f32,
    ) -> PatternPredictionReport {
        let report = self.evaluate_pattern_prediction(current, expected, horizon, limit);
        if report.prediction_error <= f32::EPSILON {
            return report;
        }

        let predicted = report
            .predicted_pattern
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<HashSet<_>>();
        let missed = expected
            .iter()
            .copied()
            .filter(|idx| !predicted.contains(idx))
            .collect::<Vec<_>>();
        if missed.is_empty() {
            return report;
        }

        let repeats = 1 + (report.prediction_error * 4.0).ceil() as usize;
        for _ in 0..repeats {
            self.learn_transition(current, &missed);
        }
        let mut corrective_cell = Vec::with_capacity(current.len() + expected.len());
        corrective_cell.extend_from_slice(current);
        corrective_cell.extend_from_slice(expected);
        let corrective_cell = compact_pattern(&corrective_cell, self.agents.len());
        let gain = learning_rate * report.prediction_error.max(0.05);
        self.reinforce_focus_transition(current, &missed, gain);
        self.reinforce_focus_transition(current, expected, gain * 0.5);
        self.reinforce_coactivation(&missed, gain * 0.75);
        self.reinforce_coactivation(&corrective_cell, gain);
        report
    }

    pub fn retrieve_episodes(&self, query: &[usize], limit: usize) -> EpisodicRecall {
        let mut matches = self
            .episodes
            .iter()
            .filter_map(|episode| {
                let pattern_score = pattern_similarity(query, &episode.pattern);
                let context_score = pattern_similarity(query, &episode.context) * 0.5;
                let similarity = pattern_score.max(context_score);
                if similarity <= 0.0 {
                    return None;
                }
                Some(EpisodicMatch {
                    pattern: episode.pattern.clone(),
                    context: episode.context.clone(),
                    similarity,
                    strength: episode.strength,
                    age_ticks: self.tick.saturating_sub(episode.created_tick),
                    prediction_error: episode.prediction_error,
                })
            })
            .collect::<Vec<_>>();

        matches.sort_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| b.strength.total_cmp(&a.strength))
        });
        matches.truncate(limit);

        let mut merged = HashMap::<usize, f32>::new();
        for item in &matches {
            let age = item.age_ticks.max(1) as f32;
            let gain = item.similarity * item.strength / age.sqrt();
            for &idx in &item.pattern {
                *merged.entry(idx).or_insert(0.0) += gain;
            }
        }

        let mut merged_pattern = merged.into_iter().collect::<Vec<_>>();
        merged_pattern.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        merged_pattern.truncate(limit.max(1) * 4);

        EpisodicRecall {
            matches,
            merged_pattern,
        }
    }

    pub fn set_attention_goal(&mut self, goal: &[usize]) {
        self.attention_goal = compact_pattern(goal, self.agents.len());
    }

    pub fn clear_attention_goal(&mut self) {
        self.attention_goal.clear();
    }

    pub fn enable_relational_field(&mut self, config: RelationalFieldConfig) {
        self.relational_field = Some(RelationalFieldSubstrate::new(config));
    }

    pub fn relational_field(&self) -> Option<&RelationalFieldSubstrate> {
        self.relational_field.as_ref()
    }

    pub fn relational_field_mut(&mut self) -> Option<&mut RelationalFieldSubstrate> {
        self.relational_field.as_mut()
    }

    pub fn relational_relation_count(&self) -> usize {
        self.relational_field
            .as_ref()
            .map(RelationalFieldSubstrate::relation_count)
            .unwrap_or(0)
    }

    pub fn set_relational_observer(&mut self, observer: ObserverId, phase: f32) {
        self.relational_observer = Some(observer);
        self.relational_observer_phase = phase;
    }

    pub fn clear_relational_observer(&mut self) {
        self.relational_observer = None;
        self.relational_observer_phase = 0.0;
    }

    pub fn reinforce_relational_relation(
        &mut self,
        observer: ObserverId,
        a: usize,
        b: usize,
        phase: f32,
        prediction_success: f32,
    ) -> bool {
        let Some(field) = self.relational_field.as_mut() else {
            return false;
        };
        if a >= self.agents.len() || b >= self.agents.len() || a == b {
            return false;
        }
        field.reinforce_relation(observer, a, b, phase, prediction_success);
        true
    }

    pub fn reinforce_relational_pattern(
        &mut self,
        observer: ObserverId,
        pattern: &[usize],
        phase: f32,
        prediction_success: f32,
    ) -> usize {
        let pattern = compact_pattern(pattern, self.agents.len());
        let mut reinforced = 0;
        for i in 0..pattern.len() {
            for j in (i + 1)..pattern.len() {
                if self.reinforce_relational_relation(
                    observer,
                    pattern[i],
                    pattern[j],
                    phase,
                    prediction_success,
                ) {
                    reinforced += 1;
                }
            }
        }
        reinforced
    }

    pub fn reinforce_relational_links(
        &mut self,
        observer: ObserverId,
        sources: &[usize],
        targets: &[usize],
        phase: f32,
        prediction_success: f32,
    ) -> usize {
        let sources = compact_pattern(sources, self.agents.len());
        let targets = compact_pattern(targets, self.agents.len());
        let mut reinforced = 0;
        for source in sources {
            for &target in &targets {
                if self.reinforce_relational_relation(
                    observer,
                    source,
                    target,
                    phase,
                    prediction_success,
                ) {
                    reinforced += 1;
                }
            }
        }
        reinforced
    }

    pub fn observe_relational_pattern(
        &mut self,
        pattern: &[usize],
        limit: usize,
    ) -> Option<CollapseReport> {
        let observer = self.relational_observer?;
        let field = self.relational_field.as_mut()?;
        Some(field.observe_pattern(observer, pattern, self.relational_observer_phase, limit))
    }

    pub fn relational_simplex_phase_report(
        &self,
        observer: ObserverId,
        a: usize,
        b: usize,
        c: usize,
    ) -> Option<SimplexPhaseReport> {
        self.relational_field
            .as_ref()?
            .simplex_phase_report(observer, a, b, c)
    }

    pub fn attention_report(&self, limit: usize) -> AttentionReport {
        let mut context_agents = self
            .attention_context
            .iter()
            .map(|(&idx, &weight)| (idx, weight))
            .collect::<Vec<_>>();
        context_agents.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        context_agents.truncate(limit);

        AttentionReport {
            goal_agents: self.attention_goal.clone(),
            context_agents,
            boosted_agents: self
                .agents
                .iter()
                .filter(|agent| {
                    agent.surprise > 0.0
                        && (self.attention_goal.contains(&agent.id)
                            || self.attention_context.contains_key(&agent.id))
                })
                .count(),
            suppressed_agents: self
                .agents
                .iter()
                .filter(|agent| agent.activation == false && agent.surprise == 0.0)
                .count(),
        }
    }

    pub fn capture_world_snapshot(&self, limit: usize) -> WorldSnapshot {
        WorldSnapshot {
            tick: self.tick,
            active_pattern: self.active_pattern(limit),
            projection: self.project_active_state(limit),
            free_energy: self.total_free_energy(),
        }
    }

    pub fn remember_world_snapshot(&mut self, limit: usize) -> WorldSnapshot {
        let snapshot = self.capture_world_snapshot(limit);
        self.world_snapshots.push_back(snapshot.clone());
        while self.world_snapshots.len() > self.config.max_episodes.max(1) {
            self.world_snapshots.pop_front();
        }
        snapshot
    }

    pub fn internal_rollout(
        &self,
        seed_pattern: &[usize],
        steps: usize,
        horizon: usize,
        limit: usize,
    ) -> RolloutReport {
        let mut imagined = self.clone();
        imagined.clear_activity();
        imagined.inject_pattern(seed_pattern, 1.0, 2);
        let initial_energy = imagined.total_free_energy();
        let mut rollout_steps = Vec::new();
        let mut active = compact_pattern(seed_pattern, imagined.agents.len());

        for step in 0..steps {
            let predicted = imagined.predict_next_pattern(&active, horizon.max(1), limit);
            let predicted_pattern = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
            if !predicted_pattern.is_empty() {
                imagined.inject_pattern(&predicted_pattern, 0.7, 1);
            }
            imagined.step();
            let snapshot = imagined.capture_world_snapshot(limit);
            active = snapshot.active_pattern.clone();
            rollout_steps.push(RolloutStep {
                step,
                predicted_pattern: predicted,
                snapshot,
            });
        }

        let terminal_pattern = active;
        let terminal_energy = rollout_steps
            .last()
            .map(|step| step.snapshot.free_energy)
            .unwrap_or(initial_energy);

        RolloutReport {
            initial_pattern: compact_pattern(seed_pattern, self.agents.len()),
            terminal_pattern,
            energy_delta: terminal_energy - initial_energy,
            steps: rollout_steps,
        }
    }

    pub fn plan_to_goal(
        &self,
        start: &[usize],
        goal: &[usize],
        horizon: usize,
        beam_width: usize,
    ) -> PlanReport {
        let start_pattern = compact_pattern(start, self.agents.len());
        let goal_pattern = compact_pattern(goal, self.agents.len());
        let goal_set = goal_pattern.iter().copied().collect::<HashSet<_>>();
        let mut frontier = start_pattern
            .iter()
            .copied()
            .map(|idx| {
                (
                    idx,
                    1.0_f32,
                    vec![PlanStep {
                        agent: idx,
                        score: 1.0,
                    }],
                )
            })
            .collect::<Vec<_>>();

        let mut best_path = frontier
            .first()
            .map(|(_, score, path)| (*score, path.clone()))
            .unwrap_or((0.0, Vec::new()));
        let mut best_goal_path = None::<(f32, Vec<PlanStep>)>;

        for depth in 0..horizon {
            let mut next = Vec::new();
            for (source, score, path) in frontier {
                let Some(neighbors) = self.causal_adjacency.get(&source) else {
                    continue;
                };
                for &(target, weight) in neighbors {
                    if path.iter().any(|step| step.agent == target) {
                        continue;
                    }
                    let goal_bonus = if goal_set.contains(&target) { 1.5 } else { 1.0 };
                    let contradiction_penalty =
                        1.0 / (1.0 + self.contradiction_tension(&[target], &goal_pattern));
                    let depth_discount = 1.0 / (depth as f32 + 1.0).sqrt();
                    let next_score =
                        score * weight * goal_bonus * contradiction_penalty * depth_discount;
                    let mut next_path = path.clone();
                    next_path.push(PlanStep {
                        agent: target,
                        score: next_score,
                    });
                    if next_score > best_path.0 {
                        best_path = (next_score, next_path.clone());
                    }
                    if goal_set.contains(&target) {
                        let should_replace = best_goal_path
                            .as_ref()
                            .map(|(goal_score, _)| next_score > *goal_score)
                            .unwrap_or(true);
                        if should_replace {
                            best_goal_path = Some((next_score, next_path.clone()));
                        }
                    }
                    next.push((target, next_score, next_path));
                }
            }

            if next.is_empty() {
                break;
            }

            next.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            next.truncate(beam_width.max(goal_pattern.len()).max(1));
            frontier = next;
        }

        let selected_path = best_goal_path.unwrap_or(best_path);
        let terminal = selected_path
            .1
            .last()
            .map(|step| vec![step.agent])
            .unwrap_or_default();

        PlanReport {
            start: start_pattern,
            goal: goal_pattern.clone(),
            horizon,
            reached_goal: selected_path
                .1
                .last()
                .map(|step| goal_pattern.contains(&step.agent))
                .unwrap_or(false),
            path: selected_path.1,
            score: selected_path.0,
            terminal_prediction: self.predict_next_pattern(&terminal, 1, goal_pattern.len().max(1)),
        }
    }

    pub fn plasticity_stats(&self) -> PlasticityStats {
        PlasticityStats {
            tick: self.tick,
            active_edges: self.edges.iter().filter(|edge| edge.active).count(),
            associative_edges: self
                .edges
                .iter()
                .filter(|edge| edge.active && edge.weight >= ASSOCIATIVE_EDGE_THRESHOLD)
                .count(),
            consolidated_edges: self
                .edges
                .iter()
                .filter(|edge| edge.active && edge.consolidated)
                .count(),
            semantic_cells: self
                .semantic_cells
                .iter()
                .filter(|cell| cell.active && cell.weight >= ASSOCIATIVE_CELL_THRESHOLD)
                .count(),
            episodes: self.episodes.len(),
            causal_edges: self.causal_edges.len(),
            contradiction_edges: self.contradiction_edges.len(),
            tetrahedra: self.tetrahedra.len(),
        }
    }

    pub fn save_persistent_state<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> io::Result<PersistentStateReport> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.serialize_persistent_state())?;
        Ok(PersistentStateReport {
            agents: self.agents.len(),
            edges: self.edges.len(),
            causal_edges: self.causal_edges.len(),
        })
    }

    pub fn load_persistent_state<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> io::Result<PersistentStateReport> {
        let contents = fs::read_to_string(path)?;
        self.apply_persistent_state(&contents)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
    }

    pub fn load_persistent_memory_state<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> io::Result<PersistentStateReport> {
        let contents = fs::read_to_string(path)?;
        self.apply_persistent_memory_state(&contents)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
    }

    pub fn serialize_persistent_state(&self) -> String {
        let mut out = String::new();
        out.push_str("SNGA_PERSISTENT_STATE_V1\n");
        out.push_str(&format!("agents {}\n", self.agents.len()));
        for agent in &self.agents {
            out.push_str(&format!(
                "a {} {:.7} {:.7} {:.7}\n",
                agent.id, agent.position.x, agent.position.y, agent.depth
            ));
        }

        out.push_str(&format!("edges {}\n", self.edges.len()));
        for (idx, edge) in self.edges.iter().enumerate() {
            out.push_str(&format!(
                "e {} {} {} {:.7} {:.7} {} {} {} {}\n",
                idx,
                edge.a,
                edge.b,
                edge.rest_length,
                edge.weight,
                edge.age,
                edge.last_active_tick,
                u8::from(edge.consolidated),
                u8::from(edge.active)
            ));
        }

        out.push_str(&format!("causal {}\n", self.causal_edges.len()));
        let mut causal = self
            .causal_edges
            .iter()
            .map(|(&(source, target), &weight)| (source, target, weight))
            .collect::<Vec<_>>();
        causal.sort_by_key(|(source, target, _)| (*source, *target));
        for (source, target, weight) in causal {
            out.push_str(&format!("c {} {} {:.7}\n", source, target, weight));
        }
        out.push_str(&format!("cells {}\n", self.semantic_cells.len()));
        for cell in &self.semantic_cells {
            out.push_str(&format!(
                "s {} {:.7} {} {} {} {}",
                cell.id,
                cell.weight,
                cell.age,
                cell.last_active_tick,
                u8::from(cell.active),
                cell.vertices.len()
            ));
            for vertex in &cell.vertices {
                out.push_str(&format!(" {vertex}"));
            }
            out.push_str(&format!(" {}\n", bytes_to_hex(&cell.payload)));
        }
        out.push_str(&format!("focus {}\n", self.focus_edges.len()));
        let mut focus = self
            .focus_edges
            .iter()
            .map(|(&(source, target), &weight)| (source, target, weight))
            .collect::<Vec<_>>();
        focus.sort_by_key(|(source, target, _)| (*source, *target));
        for (source, target, weight) in focus {
            out.push_str(&format!("f {} {} {:.7}\n", source, target, weight));
        }
        out.push_str("end\n");
        out
    }

    pub fn apply_persistent_state(
        &mut self,
        contents: &str,
    ) -> Result<PersistentStateReport, String> {
        let mut lines = contents.lines();
        if lines.next() != Some("SNGA_PERSISTENT_STATE_V1") {
            return Err("version de estado SNGA invalida".to_string());
        }

        let agent_header = lines.next().ok_or("faltan agentes")?;
        let agent_count = parse_count_header(agent_header, "agents")?;
        if agent_count != self.agents.len() {
            return Err(format!(
                "conteo de agentes incompatible: estado={} red={}",
                agent_count,
                self.agents.len()
            ));
        }

        for _ in 0..agent_count {
            let line = lines.next().ok_or("faltan lineas de agentes")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 5 || parts[0] != "a" {
                return Err(format!("linea de agente invalida: {line}"));
            }
            let id = parse_usize(parts[1], "agent id")?;
            if id >= self.agents.len() {
                return Err(format!("agent id fuera de rango: {id}"));
            }
            self.agents[id].position = Vec2::new(
                parse_f32(parts[2], "agent x")?,
                parse_f32(parts[3], "agent y")?,
            );
            self.agents[id].depth = parse_f32(parts[4], "agent depth")?;
        }
        self.recalibrate_simplex_targets();

        let edge_header = lines.next().ok_or("faltan aristas")?;
        let edge_count = parse_count_header(edge_header, "edges")?;
        for _ in 0..edge_count {
            let line = lines.next().ok_or("faltan lineas de aristas")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts[0] != "e" {
                return Err(format!("linea de arista invalida: {line}"));
            }
            let (idx, a, b, rest_idx, weight_idx, age_idx, tick_idx, consolidated_idx, active_idx) =
                if parts.len() == 10 {
                    (
                        parse_usize(parts[1], "edge idx")?,
                        parse_usize(parts[2], "edge a")?,
                        parse_usize(parts[3], "edge b")?,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                    )
                } else if parts.len() == 8 {
                    let idx = parse_usize(parts[1], "edge idx")?;
                    let Some(edge) = self.edges.get(idx) else {
                        return Err(format!("edge idx fuera de rango: {idx}"));
                    };
                    (idx, edge.a, edge.b, 2, 3, 4, 5, 6, 7)
                } else {
                    return Err(format!("linea de arista invalida: {line}"));
                };
            if a >= self.agents.len() || b >= self.agents.len() {
                return Err(format!("arista fuera de rango: {a}->{b}"));
            }
            while idx >= self.edges.len() {
                self.add_edge(a, b, parse_f32(parts[rest_idx], "edge rest_length")?, 1.0);
            }
            let edge = &mut self.edges[idx];
            edge.rest_length = parse_f32(parts[rest_idx], "edge rest_length")?;
            edge.weight = parse_f32(parts[weight_idx], "edge weight")?;
            edge.age = parse_u32(parts[age_idx], "edge age")?;
            edge.last_active_tick = parse_u64(parts[tick_idx], "edge tick")?;
            edge.consolidated = parse_bool_flag(parts[consolidated_idx], "edge consolidated")?;
            edge.active = parse_bool_flag(parts[active_idx], "edge active")?;
        }

        let causal_header = lines.next().ok_or("faltan causales")?;
        let causal_count = parse_count_header(causal_header, "causal")?;
        self.causal_edges.clear();
        self.causal_adjacency.clear();
        for _ in 0..causal_count {
            let line = lines.next().ok_or("faltan lineas causales")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 4 || parts[0] != "c" {
                return Err(format!("linea causal invalida: {line}"));
            }
            let source = parse_usize(parts[1], "causal source")?;
            let target = parse_usize(parts[2], "causal target")?;
            let weight = parse_f32(parts[3], "causal weight")?;
            if source < self.agents.len() && target < self.agents.len() {
                self.set_causal_weight(source, target, weight);
            }
        }
        self.apply_optional_semantic_cells(&mut lines)?;

        Ok(PersistentStateReport {
            agents: self.agents.len(),
            edges: self.edges.len(),
            causal_edges: self.causal_edges.len(),
        })
    }

    pub fn apply_persistent_memory_state(
        &mut self,
        contents: &str,
    ) -> Result<PersistentStateReport, String> {
        let mut lines = contents.lines();
        if lines.next() != Some("SNGA_PERSISTENT_STATE_V1") {
            return Err("version de estado SNGA invalida".to_string());
        }

        let agent_header = lines.next().ok_or("faltan agentes")?;
        let agent_count = parse_count_header(agent_header, "agents")?;
        if agent_count > self.agents.len() {
            return Err(format!(
                "conteo de agentes incompatible para memoria: estado={} red={}",
                agent_count,
                self.agents.len()
            ));
        }

        for _ in 0..agent_count {
            let line = lines.next().ok_or("faltan lineas de agentes")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 5 || parts[0] != "a" {
                return Err(format!("linea de agente invalida: {line}"));
            }
        }

        let edge_header = lines.next().ok_or("faltan aristas")?;
        let edge_count = parse_count_header(edge_header, "edges")?;
        for _ in 0..edge_count {
            let line = lines.next().ok_or("faltan lineas de aristas")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts[0] != "e" {
                return Err(format!("linea de arista invalida: {line}"));
            }
            let (a, b, weight_idx, age_idx, tick_idx, consolidated_idx, active_idx) =
                if parts.len() == 10 {
                    (
                        parse_usize(parts[2], "edge a")?,
                        parse_usize(parts[3], "edge b")?,
                        5,
                        6,
                        7,
                        8,
                        9,
                    )
                } else {
                    return Err(format!("linea de arista invalida para memoria: {line}"));
                };
            if a >= self.agents.len() || b >= self.agents.len() || a == b {
                continue;
            }

            let active = parse_bool_flag(parts[active_idx], "edge active")?;
            if !active {
                continue;
            }

            let weight = parse_f32(parts[weight_idx], "edge weight")?;
            let age = parse_u32(parts[age_idx], "edge age")?;
            let last_active_tick = parse_u64(parts[tick_idx], "edge tick")?;
            let consolidated = parse_bool_flag(parts[consolidated_idx], "edge consolidated")?;
            self.import_memory_edge(a, b, weight, age, last_active_tick, consolidated);
        }

        let causal_header = lines.next().ok_or("faltan causales")?;
        let causal_count = parse_count_header(causal_header, "causal")?;
        self.causal_edges.clear();
        self.causal_adjacency.clear();
        for _ in 0..causal_count {
            let line = lines.next().ok_or("faltan lineas causales")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 4 || parts[0] != "c" {
                return Err(format!("linea causal invalida: {line}"));
            }
            let source = parse_usize(parts[1], "causal source")?;
            let target = parse_usize(parts[2], "causal target")?;
            let weight = parse_f32(parts[3], "causal weight")?;
            if source < self.agents.len() && target < self.agents.len() {
                self.set_causal_weight(source, target, weight);
            }
        }
        self.apply_optional_semantic_cells(&mut lines)?;

        Ok(PersistentStateReport {
            agents: self.agents.len(),
            edges: self.edges.len(),
            causal_edges: self.causal_edges.len(),
        })
    }

    pub fn enable_neural_oscillations(&mut self) {
        self.oscillations_enabled = true;
        self.initialize_oscillatory_regions();
        self.update_oscillatory_state();
    }

    pub fn disable_neural_oscillations(&mut self) {
        self.oscillations_enabled = false;
        self.brain_mode = BrainMode::Exploration;
        for band in &mut self.region_bands {
            *band = WaveBand::Theta;
        }
    }

    pub fn oscillation_stats(&self) -> OscillationStats {
        let gains = self.oscillation_gains();
        let mut delta_regions = 0;
        let mut theta_regions = 0;
        let mut alpha_regions = 0;
        let mut beta_regions = 0;
        let mut gamma_regions = 0;

        for band in &self.region_bands {
            match band {
                WaveBand::Delta => delta_regions += 1,
                WaveBand::Theta => theta_regions += 1,
                WaveBand::Alpha => alpha_regions += 1,
                WaveBand::Beta => beta_regions += 1,
                WaveBand::Gamma => gamma_regions += 1,
            }
        }

        OscillationStats {
            enabled: self.oscillations_enabled,
            mode: self.brain_mode,
            delta: gains.delta,
            theta: gains.theta,
            alpha: gains.alpha,
            beta: gains.beta,
            gamma: gains.gamma,
            regions: self.region_bands.len(),
            delta_regions,
            theta_regions,
            alpha_regions,
            beta_regions,
            gamma_regions,
        }
    }

    pub fn project_active_state(&self, limit: usize) -> ConceptProjection {
        let mut top_agents = self
            .agents
            .iter()
            .filter(|agent| agent.surprise > 0.0)
            .map(|agent| (agent.id, agent.surprise))
            .collect::<Vec<_>>();
        top_agents.sort_by(|a, b| b.1.total_cmp(&a.1));
        top_agents.truncate(limit);
        ConceptProjection { top_agents }
    }

    pub fn active_pattern(&self, limit: usize) -> Vec<usize> {
        let mut active = self
            .agents
            .iter()
            .filter(|agent| agent.surprise > 0.0)
            .map(|agent| (agent.id, agent.surprise))
            .collect::<Vec<_>>();
        active.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        active.truncate(limit);
        active.into_iter().map(|(idx, _)| idx).collect()
    }

    pub fn clear_activity(&mut self) {
        self.spikes.clear();
        self.last_episode_pattern.clear();
        for agent in &mut self.agents {
            agent.activation = false;
            agent.surprise = 0.0;
            agent.velocity = Vec2::ZERO;
            agent.depth_velocity = 0.0;
            agent.refractory = 0;
        }
    }

    pub fn excite_center(&mut self) {
        let cx = self.config.width / 2;
        let cy = self.config.height / 2;
        let center = cy * self.config.width + cx;
        self.agents[center].activation = true;
        self.agents[center].surprise = 1.0;

        for &edge_idx in &self.adjacency[center] {
            let edge = &self.edges[edge_idx];
            let target = if edge.a == center { edge.b } else { edge.a };
            self.spikes.push_back(Spike {
                source: center,
                target,
                ttl: 4,
            });
        }
    }

    pub fn step(&mut self) -> EnergyStats {
        self.tick = self.tick.wrapping_add(1);
        self.update_oscillatory_state();
        self.maybe_replay();
        self.update_attention_context();
        self.propagate_spikes();
        self.relax_geometry();
        self.maintain_plasticity();
        if let Some(field) = self.relational_field.as_mut() {
            field.step_decay();
        }
        self.decay_activation();
        self.stats()
    }

    pub fn stats(&self) -> EnergyStats {
        EnergyStats {
            total_free_energy: self.total_free_energy(),
            active_agents: self.agents.iter().filter(|agent| agent.activation).count(),
            active_spikes: self.spikes.len(),
        }
    }

    pub fn neighbor_ids(&self, agent_id: usize) -> Vec<usize> {
        self.adjacency[agent_id]
            .iter()
            .map(|&edge_idx| {
                let edge = &self.edges[edge_idx];
                if edge.a == agent_id {
                    edge.b
                } else {
                    edge.a
                }
            })
            .collect()
    }

    pub fn total_free_energy(&self) -> f32 {
        let edge_energy = self.edges.iter().fold(0.0, |acc, edge| {
            if !edge.active {
                return acc;
            }
            let a = self.agents[edge.a].position;
            let b = self.agents[edge.b].position;
            let stretch = self.agent_distance(edge.a, edge.b, a, b) - edge.rest_length;
            acc + edge.weight * stretch * stretch
        });

        let simplex_energy = self.simplices.iter().fold(0.0, |acc, simplex| {
            let area = self.simplex_area(simplex);
            let delta = area - simplex.target_area;
            acc + self.config.simplex_area_weight * delta * delta
        });

        let simplex3_energy = self.tetrahedra.iter().fold(0.0, |acc, simplex| {
            let volume = self.simplex3_volume(simplex);
            let delta = volume - simplex.target_volume;
            acc + self.config.simplex3_weight * delta * delta
        });

        let contradiction_energy = self.active_contradiction_energy();

        edge_energy + simplex_energy + simplex3_energy + contradiction_energy
    }

    pub fn anneal_active_edge_rest_lengths(&mut self, rate: f32, min_weight: f32) -> usize {
        let rate = rate.clamp(0.0, 1.0);
        if rate <= 0.0 {
            return 0;
        }

        let mut adjusted = 0;
        for idx in 0..self.edges.len() {
            if !self.edges[idx].active || self.edges[idx].weight < min_weight {
                continue;
            }

            let a = self.edges[idx].a;
            let b = self.edges[idx].b;
            let pa = self.agents[a].position;
            let pb = self.agents[b].position;
            let current_distance = self.agent_distance(a, b, pa, pb).max(1.0);
            let edge = &mut self.edges[idx];
            edge.rest_length += (current_distance - edge.rest_length) * rate;
            adjusted += 1;
        }
        adjusted
    }

    pub fn prune_low_value_associative_edges(&mut self, limit: usize) -> usize {
        if limit == 0 {
            return 0;
        }

        let mut candidates = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.active && edge.weight >= ASSOCIATIVE_EDGE_THRESHOLD)
            .map(|(idx, edge)| {
                let consolidation_bonus = if edge.consolidated { 10_000.0 } else { 0.0 };
                let score = edge.weight + edge.age as f32 * 0.001 + consolidation_bonus;
                (idx, score)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| self.edges[a.0].age.cmp(&self.edges[b.0].age))
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut remove = vec![false; self.edges.len()];
        let mut removed = 0;
        for (idx, _) in candidates.into_iter().take(limit) {
            remove[idx] = true;
            removed += 1;
        }

        if removed == 0 {
            return 0;
        }

        let mut next_edges = Vec::with_capacity(self.edges.len() - removed);
        for (idx, edge) in self.edges.drain(..).enumerate() {
            if !remove[idx] {
                next_edges.push(edge);
            }
        }
        self.edges = next_edges;
        self.rebuild_edge_indices();
        removed
    }

    pub fn prune_low_value_associative_edges_in_range(
        &mut self,
        limit: usize,
        start: usize,
        end: usize,
    ) -> usize {
        if limit == 0 || start >= end {
            return 0;
        }

        let mut candidates = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                edge.active
                    && edge.weight >= ASSOCIATIVE_EDGE_THRESHOLD
                    && edge.a >= start
                    && edge.a < end
                    && edge.b >= start
                    && edge.b < end
            })
            .map(|(idx, edge)| {
                let consolidation_bonus = if edge.consolidated { 10_000.0 } else { 0.0 };
                let score = edge.weight + edge.age as f32 * 0.001 + consolidation_bonus;
                (idx, score)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| self.edges[a.0].age.cmp(&self.edges[b.0].age))
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut remove = vec![false; self.edges.len()];
        let mut removed = 0;
        for (idx, _) in candidates.into_iter().take(limit) {
            remove[idx] = true;
            removed += 1;
        }

        if removed == 0 {
            return 0;
        }

        let mut next_edges = Vec::with_capacity(self.edges.len() - removed);
        for (idx, edge) in self.edges.drain(..).enumerate() {
            if !remove[idx] {
                next_edges.push(edge);
            }
        }
        self.edges = next_edges;
        self.rebuild_edge_indices();
        removed
    }

    pub fn prune_low_value_causal_edges(&mut self, limit: usize) -> usize {
        if limit == 0 {
            return 0;
        }

        let mut candidates = self
            .causal_edges
            .iter()
            .map(|(&(source, target), &weight)| ((source, target), weight))
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| (a.0 .0, a.0 .1).cmp(&(b.0 .0, b.0 .1)))
        });

        let mut removed = 0;
        for (key, _) in candidates.into_iter().take(limit) {
            if self.causal_edges.remove(&key).is_some() {
                removed += 1;
            }
        }

        if removed > 0 {
            self.rebuild_causal_adjacency();
        }
        removed
    }

    pub fn prune_low_value_causal_edges_in_range(
        &mut self,
        limit: usize,
        start: usize,
        end: usize,
    ) -> usize {
        if limit == 0 || start >= end {
            return 0;
        }

        let mut candidates = self
            .causal_edges
            .iter()
            .filter(|(&(source, target), _)| {
                source >= start && source < end && target >= start && target < end
            })
            .map(|(&(source, target), &weight)| ((source, target), weight))
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| (a.0 .0, a.0 .1).cmp(&(b.0 .0, b.0 .1)))
        });

        let mut removed = 0;
        for (key, _) in candidates.into_iter().take(limit) {
            if self.causal_edges.remove(&key).is_some() {
                removed += 1;
            }
        }

        if removed > 0 {
            self.rebuild_causal_adjacency();
        }
        removed
    }

    pub fn prune_low_value_causal_edges_from_range_except_targets(
        &mut self,
        limit: usize,
        source_start: usize,
        source_end: usize,
        protected_targets: &[(usize, usize)],
    ) -> usize {
        if limit == 0 || source_start >= source_end {
            return 0;
        }

        let mut candidates = self
            .causal_edges
            .iter()
            .filter(|(&(source, target), _)| {
                source >= source_start
                    && source < source_end
                    && !protected_targets
                        .iter()
                        .any(|(start, end)| target >= *start && target < *end)
            })
            .map(|(&(source, target), &weight)| ((source, target), weight))
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| (a.0 .0, a.0 .1).cmp(&(b.0 .0, b.0 .1)))
        });

        let mut removed = 0;
        for (key, _) in candidates.into_iter().take(limit) {
            if self.causal_edges.remove(&key).is_some() {
                removed += 1;
            }
        }

        if removed > 0 {
            self.rebuild_causal_adjacency();
        }
        removed
    }

    fn add_edge(&mut self, a: usize, b: usize, rest_length: f32, weight: f32) {
        let edge_idx = self.edges.len();
        self.edges.push(Edge {
            a,
            b,
            rest_length,
            weight,
            age: 0,
            last_active_tick: self.tick,
            consolidated: false,
            active: true,
        });
        self.adjacency[a].push(edge_idx);
        self.adjacency[b].push(edge_idx);
        self.edge_lookup.insert(edge_key(a, b), edge_idx);
    }

    fn rebuild_edge_indices(&mut self) {
        self.adjacency = vec![Vec::new(); self.agents.len()];
        self.edge_lookup.clear();
        for (idx, edge) in self.edges.iter().enumerate() {
            if edge.a >= self.agents.len() || edge.b >= self.agents.len() {
                continue;
            }
            self.adjacency[edge.a].push(idx);
            self.adjacency[edge.b].push(idx);
            self.edge_lookup.insert(edge_key(edge.a, edge.b), idx);
        }
        self.rebuild_cell_indices();
    }

    fn rebuild_causal_adjacency(&mut self) {
        self.causal_adjacency.clear();
        for (&(source, target), &weight) in &self.causal_edges {
            upsert_weighted_neighbor(&mut self.causal_adjacency, source, target, weight);
        }
    }

    fn import_memory_edge(
        &mut self,
        a: usize,
        b: usize,
        weight: f32,
        age: u32,
        last_active_tick: u64,
        consolidated: bool,
    ) {
        let pa = self.agents[a].position;
        let pb = self.agents[b].position;
        let rest_length = self.agent_distance(a, b, pa, pb).max(1.0) * 0.92;
        if let Some(&edge_idx) = self.edge_lookup.get(&edge_key(a, b)) {
            let edge = &mut self.edges[edge_idx];
            edge.rest_length = rest_length;
            edge.weight = edge.weight.max(weight);
            edge.age = edge.age.max(age);
            edge.last_active_tick = edge.last_active_tick.max(last_active_tick);
            edge.consolidated |= consolidated;
            edge.active = true;
            return;
        }

        self.add_edge(a, b, rest_length, weight);
        if let Some(&edge_idx) = self.edge_lookup.get(&edge_key(a, b)) {
            let edge = &mut self.edges[edge_idx];
            edge.age = age;
            edge.last_active_tick = last_active_tick;
            edge.consolidated = consolidated;
        }
    }

    fn initialize_oscillatory_regions(&mut self) {
        let region_count =
            (self.agents.len() + OSCILLATORY_REGION_SIZE - 1) / OSCILLATORY_REGION_SIZE;
        self.agent_regions = (0..self.agents.len())
            .map(|idx| idx / OSCILLATORY_REGION_SIZE)
            .collect();
        self.region_bands = vec![WaveBand::Theta; region_count.max(1)];
    }

    fn update_oscillatory_state(&mut self) {
        if !self.oscillations_enabled {
            return;
        }
        if self.region_bands.is_empty() || self.agent_regions.len() != self.agents.len() {
            self.initialize_oscillatory_regions();
        }

        let mut surprise_sum = vec![0.0_f32; self.region_bands.len()];
        let mut active_count = vec![0_usize; self.region_bands.len()];
        let mut goal_count = vec![0_usize; self.region_bands.len()];
        let mut total_surprise = 0.0;
        let mut total_active = 0_usize;

        for agent in &self.agents {
            if agent.surprise > 0.0 {
                let region = self.agent_regions[agent.id];
                surprise_sum[region] += agent.surprise;
                active_count[region] += 1;
                total_surprise += agent.surprise;
                total_active += 1;
            }
        }

        for &idx in &self.attention_goal {
            if idx < self.agent_regions.len() {
                goal_count[self.agent_regions[idx]] += 1;
            }
        }

        let mean_surprise = total_surprise / total_active.max(1) as f32;
        self.brain_mode =
            if total_active == 0 && !self.episodes.is_empty() && self.delta_wave() > 0.65 {
                BrainMode::SleepReplay
            } else if !self.attention_goal.is_empty() || (total_active > 0 && mean_surprise < 0.75)
            {
                BrainMode::Focus
            } else {
                BrainMode::Exploration
            };

        for region in 0..self.region_bands.len() {
            let avg = surprise_sum[region] / active_count[region].max(1) as f32;
            self.region_bands[region] = match self.brain_mode {
                BrainMode::SleepReplay => WaveBand::Delta,
                BrainMode::Focus if goal_count[region] > 0 => WaveBand::Beta,
                _ if avg > 0.95 => WaveBand::Gamma,
                _ if avg > 0.20 => WaveBand::Theta,
                _ => WaveBand::Alpha,
            };
        }
    }

    fn oscillation_gains(&self) -> OscillationGains {
        if !self.oscillations_enabled {
            return OscillationGains {
                delta: 0.0,
                theta: 0.0,
                alpha: 0.0,
                beta: 0.0,
                gamma: 0.0,
                excitability: 1.0,
                inhibition: 1.0,
                replay: 1.0,
                plasticity: 1.0,
                prediction: 1.0,
            };
        }

        let delta = self.delta_wave();
        let theta = oscillation(self.tick, 48);
        let alpha = oscillation(self.tick, 24);
        let beta = oscillation(self.tick, 12);
        let gamma = oscillation(self.tick, 4);

        let mode_boost = match self.brain_mode {
            BrainMode::Exploration => (1.10, 0.95, 0.90, 1.05),
            BrainMode::Focus => (0.98, 1.15, 0.95, 1.25),
            BrainMode::SleepReplay => (0.80, 1.05, 1.80, 0.85),
        };

        OscillationGains {
            delta,
            theta,
            alpha,
            beta,
            gamma,
            excitability: mode_boost.0 * (1.0 + gamma * 0.18 + theta * 0.08 - alpha * 0.10),
            inhibition: mode_boost.1 * (1.0 + alpha * 0.35),
            replay: mode_boost.2 * (1.0 + delta * 0.80 + theta * 0.25),
            plasticity: 1.0 + theta * 0.20 + delta * 0.10,
            prediction: mode_boost.3 * (1.0 + beta * 0.30 + theta * 0.10),
        }
    }

    fn delta_wave(&self) -> f32 {
        oscillation(self.tick, 128)
    }

    fn oscillatory_weight(&self, agent_id: usize) -> f32 {
        if !self.oscillations_enabled || agent_id >= self.agent_regions.len() {
            return 1.0;
        }
        let gains = self.oscillation_gains();
        match self.region_bands[self.agent_regions[agent_id]] {
            WaveBand::Delta => 0.85 + gains.delta * 0.25,
            WaveBand::Theta => 1.00 + gains.theta * 0.20,
            WaveBand::Alpha => (1.0 - gains.alpha * 0.25).max(0.60),
            WaveBand::Beta => 1.00 + gains.beta * 0.30,
            WaveBand::Gamma => 1.00 + gains.gamma * 0.35,
        }
    }

    fn reinforce_pair(&mut self, a: usize, b: usize, learning_rate: f32) {
        if let Some(&edge_idx) = self.edge_lookup.get(&edge_key(a, b)) {
            let edge = &mut self.edges[edge_idx];
            edge.active = true;
            edge.weight = (edge.weight + learning_rate).min(5.0);
            edge.rest_length *= 1.0 - learning_rate.min(0.08) * 0.12;
            edge.age = edge.age.saturating_add(1);
            edge.last_active_tick = self.tick;
            if edge.age >= self.config.consolidate_after {
                edge.consolidated = true;
            }
            return;
        }

        let distance = self.agents[a].position.distance(self.agents[b].position);
        self.add_edge(
            a,
            b,
            distance.max(1.0) * 0.92,
            ASSOCIATIVE_EDGE_THRESHOLD + learning_rate.max(0.05),
        );
    }

    fn reinforce_semantic_cell(&mut self, pattern: &[usize], learning_rate: f32) {
        let vertices = bounded_cell_vertices(compact_pattern(pattern, self.agents.len()));
        if vertices.len() < 3 {
            return;
        }

        if let Some(&cell_idx) = self.cell_lookup.get(&vertices) {
            let cell = &mut self.semantic_cells[cell_idx];
            cell.active = true;
            cell.weight = (cell.weight + learning_rate * vertices.len() as f32 * 0.05).min(5.0);
            cell.age = cell.age.saturating_add(1);
            cell.last_active_tick = self.tick;
            return;
        }

        let id = self.semantic_cells.len();
        let edges = self.collect_cell_edges(&vertices);
        let payload = semantic_cell_payload(&vertices);
        self.semantic_cells.push(SemanticCell {
            id,
            vertices: vertices.clone(),
            edges,
            weight: ASSOCIATIVE_CELL_THRESHOLD + learning_rate.max(0.03),
            age: 1,
            last_active_tick: self.tick,
            active: true,
            payload,
        });
        self.cell_lookup.insert(vertices.clone(), id);
        for vertex in vertices {
            if vertex < self.agent_to_cells.len() {
                self.agent_to_cells[vertex].push(id);
            }
        }
    }

    fn collect_cell_edges(&self, vertices: &[usize]) -> Vec<usize> {
        let mut edges = Vec::new();
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                if let Some(&edge_idx) = self.edge_lookup.get(&edge_key(vertices[i], vertices[j])) {
                    edges.push(edge_idx);
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    fn rebuild_cell_indices(&mut self) {
        self.agent_to_cells = vec![Vec::new(); self.agents.len()];
        self.cell_lookup.clear();
        for idx in 0..self.semantic_cells.len() {
            let vertices = bounded_cell_vertices(compact_pattern(
                &self.semantic_cells[idx].vertices,
                self.agents.len(),
            ));
            let edges = self.collect_cell_edges(&vertices);
            self.semantic_cells[idx].id = idx;
            self.semantic_cells[idx].vertices = vertices.clone();
            self.semantic_cells[idx].edges = edges;
            self.cell_lookup.insert(vertices.clone(), idx);
            for vertex in vertices {
                if vertex < self.agent_to_cells.len() {
                    self.agent_to_cells[vertex].push(idx);
                }
            }
        }
    }

    fn import_semantic_cell(
        &mut self,
        vertices: Vec<usize>,
        weight: f32,
        age: u32,
        last_active_tick: u64,
        active: bool,
        payload: Vec<u8>,
    ) {
        let vertices = bounded_cell_vertices(compact_pattern(&vertices, self.agents.len()));
        if vertices.len() < 3 {
            return;
        }
        if let Some(&cell_idx) = self.cell_lookup.get(&vertices) {
            let cell = &mut self.semantic_cells[cell_idx];
            cell.weight = cell.weight.max(weight);
            cell.age = cell.age.max(age);
            cell.last_active_tick = cell.last_active_tick.max(last_active_tick);
            cell.active |= active;
            if cell.payload.is_empty() && !payload.is_empty() {
                cell.payload = payload;
            }
            return;
        }

        let id = self.semantic_cells.len();
        let edges = self.collect_cell_edges(&vertices);
        let payload = if payload.is_empty() {
            semantic_cell_payload(&vertices)
        } else {
            payload
        };
        self.semantic_cells.push(SemanticCell {
            id,
            vertices: vertices.clone(),
            edges,
            weight,
            age,
            last_active_tick,
            active,
            payload,
        });
        self.cell_lookup.insert(vertices.clone(), id);
        for vertex in vertices {
            if vertex < self.agent_to_cells.len() {
                self.agent_to_cells[vertex].push(id);
            }
        }
    }

    fn apply_optional_semantic_cells<'a, I>(&mut self, lines: &mut I) -> Result<(), String>
    where
        I: Iterator<Item = &'a str>,
    {
        self.semantic_cells.clear();
        self.agent_to_cells = vec![Vec::new(); self.agents.len()];
        self.cell_lookup.clear();
        self.focus_edges.clear();
        self.focus_adjacency.clear();

        let Some(header) = lines.next() else {
            return Ok(());
        };
        if header == "end" {
            return Ok(());
        }

        let cell_count = parse_count_header(header, "cells")?;
        for _ in 0..cell_count {
            let line = lines.next().ok_or("faltan lineas de celdas")?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 8 || parts[0] != "s" {
                return Err(format!("linea de celda invalida: {line}"));
            }

            let weight = parse_f32(parts[2], "cell weight")?;
            let age = parse_u32(parts[3], "cell age")?;
            let last_active_tick = parse_u64(parts[4], "cell tick")?;
            let active = parse_bool_flag(parts[5], "cell active")?;
            let vertex_count = parse_usize(parts[6], "cell vertices")?;
            let expected_len = 8 + vertex_count;
            if parts.len() != expected_len {
                return Err(format!("linea de celda invalida: {line}"));
            }

            let mut vertices = Vec::with_capacity(vertex_count);
            for idx in 0..vertex_count {
                vertices.push(parse_usize(parts[7 + idx], "cell vertex")?);
            }
            let payload = hex_to_bytes(parts[7 + vertex_count])?;
            self.import_semantic_cell(vertices, weight, age, last_active_tick, active, payload);
        }

        match lines.next() {
            Some("end") | None => Ok(()),
            Some(header) if header.starts_with("focus ") => {
                let focus_count = parse_count_header(header, "focus")?;
                for _ in 0..focus_count {
                    let line = lines.next().ok_or("faltan lineas de foco")?;
                    let parts = line.split_whitespace().collect::<Vec<_>>();
                    if parts.len() != 4 || parts[0] != "f" {
                        return Err(format!("linea de foco invalida: {line}"));
                    }
                    let source = parse_usize(parts[1], "focus source")?;
                    let target = parse_usize(parts[2], "focus target")?;
                    let weight = parse_f32(parts[3], "focus weight")?.clamp(0.0, FOCUS_EDGE_MAX);
                    if source < self.agents.len() && target < self.agents.len() && weight > 0.0 {
                        self.focus_edges.insert((source, target), weight);
                    }
                }
                self.rebuild_focus_adjacency();
                match lines.next() {
                    Some("end") | None => Ok(()),
                    Some(line) => Err(format!("seccion final invalida: {line}")),
                }
            }
            Some(line) => Err(format!("seccion final invalida: {line}")),
        }
    }

    fn add_semantic_cell_scores(
        &self,
        sources: &[usize],
        excluded: &[usize],
        scale: f32,
        scores: &mut HashMap<usize, f32>,
    ) {
        if scale <= 0.0 || self.semantic_cells.is_empty() {
            return;
        }

        let source_set = sources.iter().copied().collect::<HashSet<_>>();
        let excluded = excluded.iter().copied().collect::<HashSet<_>>();
        let mut visited_cells = HashSet::<usize>::new();

        for &source in sources {
            if source >= self.agent_to_cells.len() {
                continue;
            }
            for &cell_idx in &self.agent_to_cells[source] {
                if !visited_cells.insert(cell_idx) {
                    continue;
                }
                let Some(cell) = self.semantic_cells.get(cell_idx) else {
                    continue;
                };
                if !cell.active || cell.weight < ASSOCIATIVE_CELL_THRESHOLD {
                    continue;
                }

                let overlap = cell
                    .vertices
                    .iter()
                    .filter(|idx| source_set.contains(idx))
                    .count();
                if overlap == 0 {
                    continue;
                }

                let gain = cell.weight * scale * (overlap as f32).sqrt();
                for &target in &cell.vertices {
                    if !excluded.contains(&target) {
                        *scores.entry(target).or_insert(0.0) += gain;
                    }
                }
            }
        }
    }

    fn promote_focus_representatives(
        &self,
        seeds: &[usize],
        excluded: &[usize],
        scores: &mut HashMap<usize, f32>,
    ) {
        if self.focus_adjacency.is_empty() {
            return;
        }

        let excluded = excluded.iter().copied().collect::<HashSet<_>>();
        let max_score = scores
            .values()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.0)
            .max(1.0);
        let mut representatives = HashMap::<usize, f32>::new();
        for &seed in seeds {
            let Some(neighbors) = self.focus_adjacency.get(&seed) else {
                continue;
            };
            for &(target, weight) in neighbors {
                if excluded.contains(&target) {
                    continue;
                }
                let normalized = (weight / FOCUS_EDGE_MAX).clamp(0.0, 1.0);
                let promoted = max_score * (1.5 + normalized);
                representatives
                    .entry(target)
                    .and_modify(|score| *score = (*score).max(promoted))
                    .or_insert(promoted);
            }
        }

        let mut representatives = representatives.into_iter().collect::<Vec<_>>();
        representatives.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        representatives.truncate(FOCUS_PROMOTION_TOP_TARGETS);

        for (target, promoted) in representatives {
            let score = scores.entry(target).or_insert(0.0);
            *score = (*score).max(promoted);
        }
    }

    fn record_episode(&mut self, pattern: &[usize], strength: f32) {
        if pattern.len() < 2 || self.config.max_episodes == 0 {
            return;
        }

        let compact = compact_pattern(pattern, self.agents.len());
        if compact.len() < 2 {
            return;
        }

        let context = self.last_episode_pattern.clone();
        let predicted_next = if context.is_empty() {
            Vec::new()
        } else {
            self.predict_next_pattern(&context, 1, compact.len())
                .into_iter()
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>()
        };
        let prediction_error = if predicted_next.is_empty() {
            1.0
        } else {
            1.0 - pattern_similarity(&predicted_next, &compact)
        };
        let novelty = 1.0
            - self
                .retrieve_episodes(&compact, 1)
                .matches
                .first()
                .map(|item| item.similarity)
                .unwrap_or(0.0);

        if !context.is_empty() {
            self.learn_transition(&context, &compact);
        }

        self.episodes.push_back(Episode {
            pattern: compact.clone(),
            strength,
            created_tick: self.tick,
            context,
            predicted_next,
            prediction_error,
            novelty,
        });
        self.last_episode_pattern = compact;

        while self.episodes.len() > self.config.max_episodes {
            self.episodes.pop_front();
        }
    }

    fn maybe_replay(&mut self) {
        let scheduled_replay =
            self.config.replay_interval != 0 && self.tick % self.config.replay_interval == 0;
        let oscillatory_replay =
            self.oscillations_enabled && self.delta_wave() > 0.85 && self.tick % 8 == 0;
        if self.config.replay_batch == 0
            || self.episodes.is_empty()
            || (!scheduled_replay && !oscillatory_replay)
        {
            return;
        }

        let batch = self.config.replay_batch.min(self.episodes.len());
        let mut episodes = self.episodes.iter().cloned().collect::<Vec<_>>();
        episodes.sort_by(|a, b| {
            let age_a = self.tick.saturating_sub(a.created_tick).max(1) as f32;
            let age_b = self.tick.saturating_sub(b.created_tick).max(1) as f32;
            let score_a = (a.strength + a.novelty + a.prediction_error) / age_a.sqrt();
            let score_b = (b.strength + b.novelty + b.prediction_error) / age_b.sqrt();
            score_b.total_cmp(&score_a)
        });
        episodes.truncate(batch);

        for episode in episodes {
            let age = self.tick.saturating_sub(episode.created_tick).max(1) as f32;
            let replay_gain = self.config.replay_learning_rate
                * episode.strength
                * (1.0 + episode.novelty)
                * (1.0 + episode.prediction_error)
                * self.oscillation_gains().replay
                / age.sqrt();
            self.reinforce_coactivation(&episode.pattern, replay_gain);
            if !episode.context.is_empty() {
                self.learn_transition(&episode.context, &episode.pattern);
            }
        }
    }

    fn maintain_plasticity(&mut self) {
        if self.config.forgetting_rate <= 0.0 {
            return;
        }

        let prune_below = self.config.prune_below_weight;
        let forgetting_rate = self.config.forgetting_rate;
        let consolidated_scale = self.config.consolidated_forgetting_scale;
        let mut inactive_edges = Vec::new();

        for (idx, edge) in self.edges.iter_mut().enumerate() {
            if !edge.active || edge.weight < ASSOCIATIVE_EDGE_THRESHOLD {
                continue;
            }

            let idle_ticks = self.tick.saturating_sub(edge.last_active_tick) as f32;
            if idle_ticks < 1.0 {
                continue;
            }

            let scale = if edge.consolidated {
                consolidated_scale
            } else {
                1.0
            };
            edge.weight = (edge.weight - forgetting_rate * scale * idle_ticks.sqrt()).max(0.0);

            if !edge.consolidated && edge.weight < prune_below {
                edge.active = false;
                inactive_edges.push(idx);
            }
        }

        for edge_idx in inactive_edges {
            let edge = &self.edges[edge_idx];
            self.edge_lookup.remove(&edge_key(edge.a, edge.b));
        }
    }

    fn add_simplex(&mut self, a: usize, b: usize, c: usize) {
        let pa = self.agents[a].position;
        let pb = self.agents[b].position;
        let pc = self.agents[c].position;
        let target_area = triangle_area(pa, pb, pc);
        self.simplices.push(Simplex2 {
            a,
            b,
            c,
            target_area,
        });
    }

    fn add_simplex3(&mut self, a: usize, b: usize, c: usize, d: usize) {
        let target_volume = tetra_volume(
            self.agent_point3(a),
            self.agent_point3(b),
            self.agent_point3(c),
            self.agent_point3(d),
        );
        self.tetrahedra.push(Simplex3 {
            a,
            b,
            c,
            d,
            target_volume,
        });
    }

    fn recalibrate_simplex_targets(&mut self) {
        for idx in 0..self.simplices.len() {
            let a = self.simplices[idx].a;
            let b = self.simplices[idx].b;
            let c = self.simplices[idx].c;
            let target_area = triangle_area(
                self.agents[a].position,
                self.agents[b].position,
                self.agents[c].position,
            );
            self.simplices[idx].target_area = target_area;
        }

        for idx in 0..self.tetrahedra.len() {
            let a = self.tetrahedra[idx].a;
            let b = self.tetrahedra[idx].b;
            let c = self.tetrahedra[idx].c;
            let d = self.tetrahedra[idx].d;
            let target_volume = tetra_volume(
                self.agent_point3(a),
                self.agent_point3(b),
                self.agent_point3(c),
                self.agent_point3(d),
            );
            self.tetrahedra[idx].target_volume = target_volume;
        }
    }

    fn propagate_spikes(&mut self) {
        let mut next = Vec::new();

        while let Some(spike) = self.spikes.pop_front() {
            if spike.ttl == 0 {
                continue;
            }

            if spike.source != spike.target && self.agents[spike.target].refractory > 0 {
                continue;
            }

            self.agents[spike.target].activation = true;
            self.agents[spike.target].surprise =
                (self.agents[spike.target].surprise + 0.35).min(1.5);
            self.agents[spike.target].refractory = self.config.refractory_ticks;

            for &edge_idx in &self.adjacency[spike.target] {
                let edge = &self.edges[edge_idx];
                if !edge.active || edge.weight < ASSOCIATIVE_EDGE_THRESHOLD {
                    continue;
                }
                let neighbor = if edge.a == spike.target {
                    edge.b
                } else {
                    edge.a
                };
                if neighbor != spike.source
                    && self.agents[spike.target].surprise > self.rhythmic_threshold()
                {
                    next.push((
                        edge.weight
                            * self.attention_weight(neighbor)
                            * self.oscillatory_weight(neighbor)
                            * self.relational_spike_weight(spike.target, neighbor),
                        Spike {
                            source: spike.target,
                            target: neighbor,
                            ttl: spike.ttl - 1,
                        },
                    ));
                }
            }

            if spike.ttl > 1 {
                let mut cell_scores = HashMap::new();
                self.add_semantic_cell_scores(
                    &[spike.target],
                    &[spike.source, spike.target],
                    0.30,
                    &mut cell_scores,
                );
                for (neighbor, score) in cell_scores {
                    next.push((
                        score
                            * self.attention_weight(neighbor)
                            * self.oscillatory_weight(neighbor)
                            * self.relational_spike_weight(spike.target, neighbor),
                        Spike {
                            source: spike.target,
                            target: neighbor,
                            ttl: spike.ttl - 1,
                        },
                    ));
                }
            }
        }

        next.sort_by(|a, b| b.0.total_cmp(&a.0));
        next.truncate(self.config.max_spikes_per_step);
        self.spikes = next.into_iter().map(|(_, spike)| spike).collect();
        self.apply_lateral_inhibition();
    }

    fn update_attention_context(&mut self) {
        self.attention_context.clear();
        if self.agents.is_empty() {
            return;
        }

        let active = self.active_pattern(self.config.max_active_agents.max(1));
        for idx in active.iter().copied() {
            self.attention_context.insert(idx, 1.0);
        }

        for (idx, score) in
            self.predict_next_pattern(&active, 1, self.config.max_active_agents.max(1))
        {
            let entry = self.attention_context.entry(idx).or_insert(0.0);
            *entry = (*entry).max((score * 0.7).min(1.2));
        }

        if !self.attention_goal.is_empty() {
            let goal = self.attention_goal.clone();
            for idx in goal {
                if idx < self.agents.len() {
                    self.attention_context.insert(idx, 1.5);
                }
            }
            for (idx, score) in self.infer_transitive_from(
                &self.active_pattern(self.config.max_active_agents.max(1)),
                3,
                self.config.max_active_agents.max(1),
            ) {
                if self.attention_goal.contains(&idx) {
                    let entry = self.attention_context.entry(idx).or_insert(0.0);
                    *entry = (*entry).max((score * 1.25).min(1.5));
                }
            }
        }
    }

    fn attention_weight(&self, agent_id: usize) -> f32 {
        let goal_gain = if self.attention_goal.contains(&agent_id) {
            1.35
        } else {
            1.0
        };
        let context_gain = self
            .attention_context
            .get(&agent_id)
            .copied()
            .unwrap_or(0.0)
            .min(1.5);
        goal_gain * (1.0 + context_gain * 0.35) * self.oscillatory_weight(agent_id)
    }

    fn relational_spike_weight(&self, source: usize, target: usize) -> f32 {
        let Some(observer) = self.relational_observer else {
            return 1.0;
        };
        let Some(field) = &self.relational_field else {
            return 1.0;
        };
        let Some(modulation) =
            field.modulation(observer, source, target, self.relational_observer_phase)
        else {
            return 1.0;
        };
        // Unknown relations remain neutral. Known relations become contextual gates:
        // aligned high-probability relations pass; incompatible phases are strongly damped.
        (0.05 + modulation * 1.35).clamp(0.02, 1.25)
    }

    fn apply_lateral_inhibition(&mut self) {
        let mut active = std::mem::take(&mut self.inhibition_scratch);
        active.clear();
        for agent in &self.agents {
            if agent.surprise > 0.0 {
                active.push((agent.id, agent.surprise * self.attention_weight(agent.id)));
            }
        }

        let keep = self.config.max_active_agents;
        if active.len() > keep {
            active.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let decay = self.config.inhibition_decay;
            // Solo decae la cola fuera del top-k; los agentes en reposo (sorpresa 0)
            // quedarian inalterados, asi que no hace falta recorrerlos.
            for &(agent_id, _) in &active[keep..] {
                let agent = &mut self.agents[agent_id];
                let attention_decay = if self.attention_context.contains_key(&agent_id)
                    || self.attention_goal.contains(&agent_id)
                {
                    decay.max(0.5)
                } else {
                    decay
                };
                agent.surprise *= attention_decay;
                if agent.surprise < 0.08 {
                    agent.activation = false;
                    agent.surprise = 0.0;
                }
            }
        }

        self.apply_local_inhibition(&active);
        self.inhibition_scratch = active;
    }

    fn apply_local_inhibition(&mut self, active: &[(usize, f32)]) {
        let local_decay = self.config.local_inhibition_decay;
        if local_decay >= 1.0 || active.is_empty() {
            return;
        }

        for &(agent_id, surprise) in active.iter().take(self.config.max_active_agents) {
            for &edge_idx in &self.adjacency[agent_id] {
                let edge = &self.edges[edge_idx];
                if !edge.active || edge.weight >= ASSOCIATIVE_EDGE_THRESHOLD {
                    continue;
                }
                let neighbor = if edge.a == agent_id { edge.b } else { edge.a };
                if self.agents[neighbor].surprise > 0.0 && self.agents[neighbor].surprise < surprise
                {
                    self.agents[neighbor].surprise *= local_decay;
                    if self.agents[neighbor].surprise < 0.08 {
                        self.agents[neighbor].activation = false;
                        self.agents[neighbor].surprise = 0.0;
                    }
                }
            }
        }
    }

    fn relax_geometry(&mut self) {
        let mut forces = std::mem::take(&mut self.forces_buffer);
        forces.clear();
        forces.resize(self.agents.len(), Vec2::ZERO);

        for edge in &self.edges {
            if !edge.active {
                continue;
            }
            let pa = self.agents[edge.a].position;
            let pb = self.agents[edge.b].position;
            let delta = pb - pa;
            // Una sola raiz cuadrada: longitud real para distancia y direccion.
            let depth_delta = self.agents[edge.b].depth - self.agents[edge.a].depth;
            let len = (delta.length_squared() + depth_delta * depth_delta).sqrt();
            let distance = len.max(1.0);
            let stretch = distance - edge.rest_length;
            let activation_gain =
                if self.agents[edge.a].activation || self.agents[edge.b].activation {
                    1.85
                } else {
                    1.0
                };
            let direction = if len <= f32::EPSILON {
                Vec2::ZERO
            } else {
                delta / len
            };
            let force =
                direction * (stretch * edge.weight * self.config.elasticity * activation_gain);
            forces[edge.a] += force;
            forces[edge.b] += force * -1.0;
            let depth_force =
                depth_delta / distance * (stretch * edge.weight * self.config.elasticity);
            self.agents[edge.a].depth_velocity += depth_force;
            self.agents[edge.b].depth_velocity -= depth_force;
        }

        for simplex in &self.simplices {
            let area = self.simplex_area(simplex);
            let area_error = area - simplex.target_area;
            let centroid = (self.agents[simplex.a].position
                + self.agents[simplex.b].position
                + self.agents[simplex.c].position)
                / 3.0;

            for idx in [simplex.a, simplex.b, simplex.c] {
                let outward = (self.agents[idx].position - centroid).normalized_or_zero();
                forces[idx] += outward * (-area_error * self.config.simplex_area_weight);
            }
        }

        for (agent, force) in self.agents.iter_mut().zip(forces.iter().copied()) {
            let local_force = force.clamp_length(4.0);
            agent.velocity = (agent.velocity + local_force) * self.config.damping;
            agent.position += agent.velocity;
            agent.depth_velocity *= self.config.damping;
            agent.depth += agent.depth_velocity.clamp(-4.0, 4.0);
        }

        self.forces_buffer = forces;
    }

    fn decay_activation(&mut self) {
        for agent in &mut self.agents {
            if agent.refractory > 0 {
                agent.refractory -= 1;
            }
            agent.surprise *= 0.94;
            if agent.surprise < 0.08 {
                agent.activation = false;
                agent.surprise = 0.0;
            }
        }
    }

    fn simplex_area(&self, simplex: &Simplex2) -> f32 {
        triangle_area(
            self.agents[simplex.a].position,
            self.agents[simplex.b].position,
            self.agents[simplex.c].position,
        )
    }

    fn simplex3_volume(&self, simplex: &Simplex3) -> f32 {
        tetra_volume(
            self.agent_point3(simplex.a),
            self.agent_point3(simplex.b),
            self.agent_point3(simplex.c),
            self.agent_point3(simplex.d),
        )
    }

    fn agent_point3(&self, idx: usize) -> [f32; 3] {
        [
            self.agents[idx].position.x,
            self.agents[idx].position.y,
            self.agents[idx].depth,
        ]
    }

    fn agent_distance(&self, a: usize, b: usize, pa: Vec2, pb: Vec2) -> f32 {
        let dz = self.agents[b].depth - self.agents[a].depth;
        let euclidean = ((pb - pa).length_squared() + dz * dz).sqrt();
        let curvature = self.config.hyperbolic_curvature.max(0.0);
        if curvature <= f32::EPSILON {
            euclidean
        } else {
            (euclidean * curvature.sqrt()).asinh() / curvature.sqrt()
        }
    }

    fn rhythmic_threshold(&self) -> f32 {
        let oscillatory_gain = if self.oscillations_enabled {
            let gains = self.oscillation_gains();
            (1.0 + gains.alpha * 0.18 + gains.beta * 0.05 - gains.gamma * 0.12 - gains.theta * 0.06)
                .clamp(0.75, 1.35)
        } else {
            1.0
        };
        if self.config.rhythm_period == 0 || self.config.rhythm_amplitude <= 0.0 {
            return self.config.activation_threshold * oscillatory_gain;
        }

        let phase = (self.tick % self.config.rhythm_period) as f32
            / self.config.rhythm_period as f32
            * std::f32::consts::TAU;
        let gain = 1.0 + self.config.rhythm_amplitude * phase.sin();
        self.config.activation_threshold * gain * oscillatory_gain
    }

    fn active_contradiction_energy(&self) -> f32 {
        if self.contradiction_edges.is_empty() {
            return 0.0;
        }

        let active = self
            .agents
            .iter()
            .filter(|agent| agent.surprise > 0.08)
            .map(|agent| agent.id)
            .collect::<Vec<_>>();

        let mut energy = 0.0;
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                if let Some(weight) = self
                    .contradiction_edges
                    .get(&edge_key(active[i], active[j]))
                {
                    energy += weight * self.config.contradiction_energy_weight;
                }
            }
        }
        energy
    }
}

fn triangle_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() * 0.5
}

fn tetra_volume(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f32 {
    let ax = b[0] - a[0];
    let ay = b[1] - a[1];
    let az = b[2] - a[2];
    let bx = c[0] - a[0];
    let by = c[1] - a[1];
    let bz = c[2] - a[2];
    let cx = d[0] - a[0];
    let cy = d[1] - a[1];
    let cz = d[2] - a[2];

    let cross_x = by * cz - bz * cy;
    let cross_y = bz * cx - bx * cz;
    let cross_z = bx * cy - by * cx;
    (ax * cross_x + ay * cross_y + az * cross_z).abs() / 6.0
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn upsert_weighted_neighbor(
    adjacency: &mut HashMap<usize, Vec<(usize, f32)>>,
    source: usize,
    target: usize,
    weight: f32,
) {
    let neighbors = adjacency.entry(source).or_default();
    if let Some((_, existing_weight)) = neighbors.iter_mut().find(|(idx, _)| *idx == target) {
        *existing_weight = weight;
    } else {
        neighbors.push((target, weight));
    }
}

fn compact_pattern(pattern: &[usize], agent_count: usize) -> Vec<usize> {
    let mut compact = pattern
        .iter()
        .copied()
        .filter(|&idx| idx < agent_count)
        .collect::<Vec<_>>();
    compact.sort_unstable();
    compact.dedup();
    compact
}

fn bounded_cell_vertices(vertices: Vec<usize>) -> Vec<usize> {
    if vertices.len() <= MAX_ASSOCIATIVE_CELL_VERTICES {
        return vertices;
    }

    let last = vertices.len() - 1;
    let max_last = MAX_ASSOCIATIVE_CELL_VERTICES - 1;
    let mut bounded = (0..MAX_ASSOCIATIVE_CELL_VERTICES)
        .map(|idx| vertices[idx * last / max_last])
        .collect::<Vec<_>>();
    bounded.sort_unstable();
    bounded.dedup();
    bounded
}

fn sample_vertices(vertices: Vec<usize>, limit: usize) -> Vec<usize> {
    if limit == 0 || vertices.len() <= limit {
        return vertices;
    }

    let last = vertices.len() - 1;
    let max_last = limit - 1;
    let mut sampled = (0..limit)
        .map(|idx| vertices[idx * last / max_last])
        .collect::<Vec<_>>();
    sampled.sort_unstable();
    sampled.dedup();
    sampled
}

fn semantic_cell_payload(vertices: &[usize]) -> Vec<u8> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    vertices.hash(&mut hasher);
    hasher.finish().to_le_bytes().to_vec()
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn hex_to_bytes(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 {
        return Err(format!("payload hex invalido: {value}"));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for idx in (0..value.len()).step_by(2) {
        let byte = u8::from_str_radix(&value[idx..idx + 2], 16)
            .map_err(|err| format!("payload hex invalido '{value}': {err}"))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

fn pattern_similarity(left: &[usize], right: &[usize]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    let left_set = left.iter().copied().collect::<HashSet<_>>();
    let right_set = right.iter().copied().collect::<HashSet<_>>();
    let intersection = left_set.intersection(&right_set).count() as f32;
    let union = left_set.union(&right_set).count().max(1) as f32;
    intersection / union
}

fn oscillation(tick: u64, period: u64) -> f32 {
    if period == 0 {
        return 0.0;
    }
    let phase = (tick % period) as f32 / period as f32 * std::f32::consts::TAU;
    (phase.sin() + 1.0) * 0.5
}

fn parse_count_header(line: &str, expected: &str) -> Result<usize, String> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.len() != 2 || parts[0] != expected {
        return Err(format!("cabecera invalida para {expected}: {line}"));
    }
    parse_usize(parts[1], expected)
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("{label} invalido '{value}': {err}"))
}

fn parse_u32(value: &str, label: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|err| format!("{label} invalido '{value}': {err}"))
}

fn parse_u64(value: &str, label: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("{label} invalido '{value}': {err}"))
}

fn parse_f32(value: &str, label: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|err| format!("{label} invalido '{value}': {err}"))
}

fn parse_bool_flag(value: &str, label: &str) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{label} invalido '{value}', esperado 0 o 1")),
    }
}

fn mesh_config_from(config: &SimplicialConfig) -> MeshConfig {
    MeshConfig {
        width: config.width,
        height: config.height,
        spacing: config.spacing,
        seed: config.seed,
    }
}

fn prediction_report(predicted_agents: Vec<(usize, f32)>, expected: &[usize]) -> PredictionReport {
    let expected_set = expected
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let matched_agents = predicted_agents
        .iter()
        .filter(|(idx, _)| expected_set.contains(idx))
        .count();
    let precision = matched_agents as f32 / predicted_agents.len().max(1) as f32;
    let recall = matched_agents as f32 / expected_set.len().max(1) as f32;

    PredictionReport {
        predicted_agents,
        matched_agents,
        expected_agents: expected_set.len(),
        precision,
        recall,
    }
}
