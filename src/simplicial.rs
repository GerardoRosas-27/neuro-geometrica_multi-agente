use crate::geometry::Vec2;
use crate::mesh_engine::{MeshConfig, MeshTopology, SimplicialMeshEngine};
use std::collections::{HashMap, HashSet, VecDeque};

const ASSOCIATIVE_EDGE_THRESHOLD: f32 = 1.05;
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
    pub spikes: VecDeque<Spike>,
    pub config: SimplicialConfig,
    adjacency: Vec<Vec<usize>>,
    edge_lookup: HashMap<(usize, usize), usize>,
    causal_edges: HashMap<(usize, usize), f32>,
    causal_adjacency: HashMap<usize, Vec<(usize, f32)>>,
    contradiction_edges: HashMap<(usize, usize), f32>,
    episodes: VecDeque<Episode>,
    last_episode_pattern: Vec<usize>,
    attention_goal: Vec<usize>,
    attention_context: HashMap<usize, f32>,
    world_snapshots: VecDeque<WorldSnapshot>,
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
            agents,
            edges: Vec::new(),
            simplices: Vec::new(),
            tetrahedra: Vec::new(),
            spikes: VecDeque::new(),
            edge_lookup: HashMap::new(),
            causal_edges: HashMap::new(),
            causal_adjacency: HashMap::new(),
            contradiction_edges: HashMap::new(),
            episodes: VecDeque::new(),
            last_episode_pattern: Vec::new(),
            attention_goal: Vec::new(),
            attention_context: HashMap::new(),
            world_snapshots: VecDeque::new(),
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
            }

            if next.is_empty() {
                break;
            }
            frontier = next;
        }

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
            episodes: self.episodes.len(),
            causal_edges: self.causal_edges.len(),
            contradiction_edges: self.contradiction_edges.len(),
            tetrahedra: self.tetrahedra.len(),
        }
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
                            * self.oscillatory_weight(neighbor),
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
