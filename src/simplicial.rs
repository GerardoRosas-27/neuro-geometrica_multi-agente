use crate::geometry::Vec2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub activation: bool,
    pub surprise: f32,
}

impl Agent {
    fn new(id: usize, position: Vec2) -> Self {
        Self {
            id,
            position,
            velocity: Vec2::ZERO,
            activation: false,
            surprise: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub rest_length: f32,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct Simplex2 {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub target_area: f32,
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
            seed: 7,
        }
    }
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
    pub spikes: VecDeque<Spike>,
    pub config: SimplicialConfig,
    adjacency: Vec<Vec<usize>>,
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
            spikes: VecDeque::new(),
            adjacency: vec![Vec::new(); config.width * config.height],
            config,
        };

        network.build_grid_topology();
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
        for &idx in pattern {
            if idx >= self.agents.len() {
                continue;
            }
            self.agents[idx].activation = true;
            self.agents[idx].surprise = self.agents[idx].surprise.max(surprise);
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
        self.propagate_spikes();
        self.relax_geometry();
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
            let a = self.agents[edge.a].position;
            let b = self.agents[edge.b].position;
            let stretch = a.distance(b) - edge.rest_length;
            acc + edge.weight * stretch * stretch
        });

        let simplex_energy = self.simplices.iter().fold(0.0, |acc, simplex| {
            let area = self.simplex_area(simplex);
            let delta = area - simplex.target_area;
            acc + self.config.simplex_area_weight * delta * delta
        });

        edge_energy + simplex_energy
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

    fn add_edge(&mut self, a: usize, b: usize, rest_length: f32, weight: f32) {
        let edge_idx = self.edges.len();
        self.edges.push(Edge {
            a,
            b,
            rest_length,
            weight,
        });
        self.adjacency[a].push(edge_idx);
        self.adjacency[b].push(edge_idx);
    }

    fn reinforce_pair(&mut self, a: usize, b: usize, learning_rate: f32) {
        if let Some(edge) = self
            .edges
            .iter_mut()
            .find(|edge| (edge.a == a && edge.b == b) || (edge.a == b && edge.b == a))
        {
            edge.weight = (edge.weight + learning_rate).min(5.0);
            edge.rest_length *= 1.0 - learning_rate.min(0.08) * 0.12;
            return;
        }

        let distance = self.agents[a].position.distance(self.agents[b].position);
        self.add_edge(a, b, distance.max(1.0) * 0.92, learning_rate.max(0.05));
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

    fn propagate_spikes(&mut self) {
        let mut next = VecDeque::new();

        while let Some(spike) = self.spikes.pop_front() {
            if spike.ttl == 0 {
                continue;
            }

            self.agents[spike.target].activation = true;
            self.agents[spike.target].surprise =
                (self.agents[spike.target].surprise + 0.35).min(1.5);

            for &edge_idx in &self.adjacency[spike.target] {
                let edge = &self.edges[edge_idx];
                let neighbor = if edge.a == spike.target {
                    edge.b
                } else {
                    edge.a
                };
                if neighbor != spike.source
                    && self.agents[spike.target].surprise > self.config.activation_threshold
                {
                    next.push_back(Spike {
                        source: spike.target,
                        target: neighbor,
                        ttl: spike.ttl - 1,
                    });
                }
            }
        }

        self.spikes = next;
    }

    fn relax_geometry(&mut self) {
        let mut forces = vec![Vec2::ZERO; self.agents.len()];

        for edge in &self.edges {
            let pa = self.agents[edge.a].position;
            let pb = self.agents[edge.b].position;
            let delta = pb - pa;
            let distance = delta.length().max(1.0);
            let stretch = distance - edge.rest_length;
            let activation_gain =
                if self.agents[edge.a].activation || self.agents[edge.b].activation {
                    1.85
                } else {
                    1.0
                };
            let force = delta.normalized_or_zero()
                * (stretch * edge.weight * self.config.elasticity * activation_gain);
            forces[edge.a] += force;
            forces[edge.b] += force * -1.0;
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

        for (agent, force) in self.agents.iter_mut().zip(forces.into_iter()) {
            let local_force = force.clamp_length(4.0);
            agent.velocity = (agent.velocity + local_force) * self.config.damping;
            agent.position += agent.velocity;
        }
    }

    fn decay_activation(&mut self) {
        for agent in &mut self.agents {
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
}

fn triangle_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() * 0.5
}
