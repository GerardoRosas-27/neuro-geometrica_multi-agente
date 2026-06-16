use crate::geometry::Vec2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, VecDeque};

const ASSOCIATIVE_EDGE_THRESHOLD: f32 = 1.05;

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
    tick: u64,
    // Scratch buffers reutilizados entre pasos para evitar reasignaciones por frame.
    forces_buffer: Vec<Vec2>,
    inhibition_scratch: Vec<(usize, f32)>,
}

impl SimplicialNetwork {
    pub fn grid(config: SimplicialConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut agents = Vec::with_capacity(config.width * config.height);

        for y in 0..config.height {
            for x in 0..config.width {
                let jitter = Vec2::new(rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0));
                let position =
                    Vec2::new(x as f32 * config.spacing, y as f32 * config.spacing) + jitter;
                agents.push(Agent::new(y * config.width + x, position));
            }
        }

        let mut network = Self {
            agents,
            edges: Vec::new(),
            simplices: Vec::new(),
            tetrahedra: Vec::new(),
            spikes: VecDeque::new(),
            adjacency: vec![Vec::new(); config.width * config.height],
            edge_lookup: HashMap::new(),
            causal_edges: HashMap::new(),
            causal_adjacency: HashMap::new(),
            contradiction_edges: HashMap::new(),
            episodes: VecDeque::new(),
            tick: 0,
            forces_buffer: Vec::new(),
            inhibition_scratch: Vec::new(),
            config,
        };

        network.build_grid_topology();
        network
    }

    pub fn grid_3d(config: SimplicialConfig, depth_layers: usize) -> Self {
        let layers = depth_layers.max(1);
        let mut rng = StdRng::seed_from_u64(config.seed);
        let layer_size = config.width * config.height;
        let mut agents = Vec::with_capacity(layer_size * layers);

        for z in 0..layers {
            for y in 0..config.height {
                for x in 0..config.width {
                    let jitter = Vec2::new(rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0));
                    let position =
                        Vec2::new(x as f32 * config.spacing, y as f32 * config.spacing) + jitter;
                    let mut agent = Agent::new(z * layer_size + y * config.width + x, position);
                    agent.depth = z as f32 * config.spacing;
                    agents.push(agent);
                }
            }
        }

        let mut network = Self {
            agents,
            edges: Vec::new(),
            simplices: Vec::new(),
            tetrahedra: Vec::new(),
            spikes: VecDeque::new(),
            adjacency: vec![Vec::new(); layer_size * layers],
            edge_lookup: HashMap::new(),
            causal_edges: HashMap::new(),
            causal_adjacency: HashMap::new(),
            contradiction_edges: HashMap::new(),
            episodes: VecDeque::new(),
            tick: 0,
            forces_buffer: Vec::new(),
            inhibition_scratch: Vec::new(),
            config,
        };

        network.build_grid_topology_3d(layers);
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

    pub fn clear_activity(&mut self) {
        self.spikes.clear();
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
        self.maybe_replay();
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

    fn build_grid_topology(&mut self) {
        for y in 0..self.config.height {
            for x in 0..self.config.width {
                let id = y * self.config.width + x;
                if x + 1 < self.config.width {
                    self.add_edge(
                        id,
                        y * self.config.width + (x + 1),
                        self.config.spacing,
                        1.0,
                    );
                }
                if y + 1 < self.config.height {
                    self.add_edge(
                        id,
                        (y + 1) * self.config.width + x,
                        self.config.spacing,
                        1.0,
                    );
                }
                if x + 1 < self.config.width && y + 1 < self.config.height {
                    self.add_edge(
                        id,
                        (y + 1) * self.config.width + (x + 1),
                        self.config.spacing * 2.0_f32.sqrt(),
                        0.45,
                    );
                    self.add_simplex(
                        id,
                        y * self.config.width + (x + 1),
                        (y + 1) * self.config.width + (x + 1),
                    );
                    self.add_simplex(
                        id,
                        (y + 1) * self.config.width + x,
                        (y + 1) * self.config.width + (x + 1),
                    );
                }
            }
        }
    }

    fn build_grid_topology_3d(&mut self, depth_layers: usize) {
        let layer_size = self.config.width * self.config.height;
        for z in 0..depth_layers {
            for y in 0..self.config.height {
                for x in 0..self.config.width {
                    let id = z * layer_size + y * self.config.width + x;
                    if x + 1 < self.config.width {
                        self.add_edge(
                            id,
                            z * layer_size + y * self.config.width + (x + 1),
                            self.config.spacing,
                            1.0,
                        );
                    }
                    if y + 1 < self.config.height {
                        self.add_edge(
                            id,
                            z * layer_size + (y + 1) * self.config.width + x,
                            self.config.spacing,
                            1.0,
                        );
                    }
                    if z + 1 < depth_layers {
                        self.add_edge(
                            id,
                            (z + 1) * layer_size + y * self.config.width + x,
                            self.config.spacing,
                            1.0,
                        );
                    }
                    if x + 1 < self.config.width && y + 1 < self.config.height {
                        let bx = z * layer_size + y * self.config.width + (x + 1);
                        let cy = z * layer_size + (y + 1) * self.config.width + x;
                        let dxy = z * layer_size + (y + 1) * self.config.width + (x + 1);
                        self.add_edge(id, dxy, self.config.spacing * 2.0_f32.sqrt(), 0.45);
                        self.add_simplex(id, bx, dxy);
                        self.add_simplex(id, cy, dxy);

                        if z + 1 < depth_layers {
                            let up = (z + 1) * layer_size + y * self.config.width + x;
                            self.add_simplex3(id, bx, cy, up);
                        }
                    }
                }
            }
        }
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

        let mut compact = pattern
            .iter()
            .copied()
            .filter(|&idx| idx < self.agents.len())
            .collect::<Vec<_>>();
        compact.sort_unstable();
        compact.dedup();
        if compact.len() < 2 {
            return;
        }

        self.episodes.push_back(Episode {
            pattern: compact,
            strength,
            created_tick: self.tick,
        });

        while self.episodes.len() > self.config.max_episodes {
            self.episodes.pop_front();
        }
    }

    fn maybe_replay(&mut self) {
        if self.config.replay_interval == 0
            || self.config.replay_batch == 0
            || self.episodes.is_empty()
            || self.tick % self.config.replay_interval != 0
        {
            return;
        }

        let batch = self.config.replay_batch.min(self.episodes.len());
        let start = self.episodes.len() - batch;
        let episodes = self
            .episodes
            .iter()
            .skip(start)
            .cloned()
            .collect::<Vec<_>>();

        for episode in episodes {
            let age = self.tick.saturating_sub(episode.created_tick).max(1) as f32;
            let replay_gain = self.config.replay_learning_rate * episode.strength / age.sqrt();
            self.reinforce_coactivation(&episode.pattern, replay_gain);
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
                        edge.weight,
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

    fn apply_lateral_inhibition(&mut self) {
        let mut active = std::mem::take(&mut self.inhibition_scratch);
        active.clear();
        for agent in &self.agents {
            if agent.surprise > 0.0 {
                active.push((agent.id, agent.surprise));
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
                agent.surprise *= decay;
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
        if self.config.rhythm_period == 0 || self.config.rhythm_amplitude <= 0.0 {
            return self.config.activation_threshold;
        }

        let phase = (self.tick % self.config.rhythm_period) as f32
            / self.config.rhythm_period as f32
            * std::f32::consts::TAU;
        let gain = 1.0 + self.config.rhythm_amplitude * phase.sin();
        self.config.activation_threshold * gain
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
