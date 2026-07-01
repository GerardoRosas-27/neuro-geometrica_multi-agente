use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CdtGraphityEdgeKind {
    Spatial,
    Temporal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CdtSimplexKind {
    T31,
    T22,
}

#[derive(Clone, Debug)]
pub struct CdtGraphityNode {
    pub id: usize,
    pub slice: usize,
    pub activation: bool,
    pub surprise: f32,
}

#[derive(Clone, Debug)]
pub struct CdtGraphityEdge {
    pub a: usize,
    pub b: usize,
    pub kind: CdtGraphityEdgeKind,
    pub stability: f32,
    pub prediction_error: f32,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub struct CdtGraphitySimplex3 {
    pub vertices: [usize; 4],
    pub kind: CdtSimplexKind,
}

#[derive(Clone, Copy, Debug)]
pub struct CdtGraphityConfig {
    pub slices: usize,
    pub nodes_per_slice: usize,
    pub initial_spatial_connectivity: f32,
    pub initial_temporal_connectivity: f32,
    pub target_spatial_degree: usize,
    pub target_temporal_degree: usize,
    pub target_tetrahedra_per_edge: usize,
    pub cooling_rate: f32,
    pub heating_rate: f32,
    pub reinforcement_rate: f32,
    pub prune_threshold: f32,
    pub max_new_edges_per_step: usize,
    pub seed: u64,
}

impl Default for CdtGraphityConfig {
    fn default() -> Self {
        Self {
            slices: 5,
            nodes_per_slice: 24,
            initial_spatial_connectivity: 0.40,
            initial_temporal_connectivity: 0.35,
            target_spatial_degree: 5,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.05,
            heating_rate: 0.16,
            reinforcement_rate: 0.08,
            prune_threshold: 0.06,
            max_new_edges_per_step: 24,
            seed: 1_337,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CdtGraphityStepReport {
    pub tick: u64,
    pub free_energy: f32,
    pub regge_action: f32,
    pub prediction_error: f32,
    pub temperature: f32,
    pub active_nodes: usize,
    pub active_edges: usize,
    pub spatial_edges: usize,
    pub temporal_edges: usize,
    pub tetrahedra: usize,
    pub pruned_edges: usize,
    pub proposed_edges: usize,
    pub causality_violations: usize,
}

#[derive(Clone, Debug)]
pub struct CdtGraphitySubstrate {
    pub config: CdtGraphityConfig,
    pub nodes: Vec<CdtGraphityNode>,
    pub edges: Vec<CdtGraphityEdge>,
    pub tetrahedra: Vec<CdtGraphitySimplex3>,
    pub temperature: f32,
    tick: u64,
    edge_lookup: HashMap<(usize, usize), usize>,
    adjacency: Vec<Vec<usize>>,
    rng: StdRng,
}

impl CdtGraphitySubstrate {
    pub fn graphity_hot_start(config: CdtGraphityConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let node_count = config.slices.max(1) * config.nodes_per_slice.max(1);
        let mut nodes = Vec::with_capacity(node_count);
        for id in 0..node_count {
            nodes.push(CdtGraphityNode {
                id,
                slice: id / config.nodes_per_slice.max(1),
                activation: false,
                surprise: 0.0,
            });
        }

        let mut substrate = Self {
            config,
            nodes,
            edges: Vec::new(),
            tetrahedra: Vec::new(),
            temperature: 1.0,
            tick: 0,
            edge_lookup: HashMap::new(),
            adjacency: vec![Vec::new(); node_count],
            rng: StdRng::seed_from_u64(config.seed ^ 0xA53A_9EED),
        };

        for slice in 0..substrate.config.slices {
            let ids = substrate.slice_nodes(slice);
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    if rng.gen::<f32>() <= substrate.config.initial_spatial_connectivity {
                        substrate.add_edge(ids[i], ids[j], CdtGraphityEdgeKind::Spatial, 0.35);
                    }
                }
            }
        }

        for slice in 0..substrate.config.slices.saturating_sub(1) {
            let current = substrate.slice_nodes(slice);
            let next = substrate.slice_nodes(slice + 1);
            for &a in &current {
                for &b in &next {
                    if rng.gen::<f32>() <= substrate.config.initial_temporal_connectivity {
                        substrate.add_edge(a, b, CdtGraphityEdgeKind::Temporal, 0.35);
                    }
                }
            }
        }

        substrate.rebuild_cdt_tetrahedra();
        substrate
    }

    pub fn inject_pattern(&mut self, pattern: &[usize], surprise: f32) {
        for &idx in pattern {
            if let Some(node) = self.nodes.get_mut(idx) {
                node.activation = true;
                node.surprise = node.surprise.max(surprise);
            }
        }
    }

    pub fn clear_activity(&mut self) {
        for node in &mut self.nodes {
            node.activation = false;
            node.surprise = 0.0;
        }
    }

    pub fn active_pattern(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|node| node.activation || node.surprise > 0.05)
            .map(|node| node.id)
            .collect()
    }

    pub fn predict_next(&self, active: &[usize], limit: usize) -> Vec<(usize, f32)> {
        let active = active.iter().copied().collect::<HashSet<_>>();
        let mut scores = HashMap::<usize, f32>::new();
        for &source in &active {
            if source >= self.nodes.len() {
                continue;
            }
            for &edge_idx in &self.adjacency[source] {
                let edge = &self.edges[edge_idx];
                if !edge.active || edge.kind != CdtGraphityEdgeKind::Temporal {
                    continue;
                }
                let Some(target) = self.temporal_target_from(edge, source) else {
                    continue;
                };
                *scores.entry(target).or_insert(0.0) +=
                    edge.stability * (1.0 - edge.prediction_error).max(0.0);
            }
        }
        let mut predicted = scores.into_iter().collect::<Vec<_>>();
        predicted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        predicted.truncate(limit);
        predicted
    }

    pub fn reinforce_temporal_link(
        &mut self,
        source: usize,
        target: usize,
        stability: f32,
    ) -> bool {
        if source >= self.nodes.len()
            || target >= self.nodes.len()
            || self.nodes[source].slice + 1 != self.nodes[target].slice
        {
            return false;
        }
        let inserted = self.add_edge(
            source,
            target,
            CdtGraphityEdgeKind::Temporal,
            stability.clamp(0.0, 1.0),
        );
        if let Some(&edge_idx) = self.edge_lookup.get(&edge_key(source, target)) {
            let edge = &mut self.edges[edge_idx];
            edge.active = true;
            edge.stability = edge.stability.max(stability.clamp(0.0, 1.0));
            edge.prediction_error *= 0.5;
        }
        inserted
    }

    pub fn anneal_geometry_step(
        &mut self,
        protected_edges: &[(usize, usize)],
    ) -> CdtGraphityStepReport {
        self.tick = self.tick.wrapping_add(1);
        let protected = protected_edges
            .iter()
            .map(|&(a, b)| edge_key(a, b))
            .collect::<HashSet<_>>();
        let mut pruned_edges = 0;
        let mut incident = vec![0_usize; self.edges.len()];

        for tetra in &self.tetrahedra {
            for i in 0..tetra.vertices.len() {
                for j in (i + 1)..tetra.vertices.len() {
                    if let Some(&edge_idx) = self
                        .edge_lookup
                        .get(&edge_key(tetra.vertices[i], tetra.vertices[j]))
                    {
                        incident[edge_idx] += 1;
                    }
                }
            }
        }

        for (idx, edge) in self.edges.iter_mut().enumerate() {
            if !edge.active || protected.contains(&edge_key(edge.a, edge.b)) {
                continue;
            }
            let target = match edge.kind {
                CdtGraphityEdgeKind::Spatial => self.config.target_tetrahedra_per_edge,
                CdtGraphityEdgeKind::Temporal => self.config.target_tetrahedra_per_edge + 1,
            };
            let deficit = target.abs_diff(incident[idx]) as f32 / target.max(1) as f32;
            let degree_pressure = match edge.kind {
                CdtGraphityEdgeKind::Spatial => 0.12,
                CdtGraphityEdgeKind::Temporal => 0.04,
            };
            let instability = edge.prediction_error
                + deficit * 0.50
                + (1.0 - edge.stability) * 0.30
                + degree_pressure;
            if instability > 0.55 || edge.stability < self.config.prune_threshold {
                edge.active = false;
                pruned_edges += 1;
            } else {
                edge.stability = (edge.stability + self.config.cooling_rate * 0.25).min(1.0);
                edge.prediction_error *= 1.0 - self.config.cooling_rate;
            }
        }

        self.temperature *= 1.0 - self.config.cooling_rate;
        self.rebuild_cdt_tetrahedra();
        let regge_action = self.regge_action();
        CdtGraphityStepReport {
            tick: self.tick,
            free_energy: regge_action * 0.015 + self.temperature * 0.05,
            regge_action,
            prediction_error: 0.0,
            temperature: self.temperature,
            active_nodes: self
                .nodes
                .iter()
                .filter(|node| node.surprise > 0.05)
                .count(),
            active_edges: self.edges.iter().filter(|edge| edge.active).count(),
            spatial_edges: self
                .edges
                .iter()
                .filter(|edge| edge.active && edge.kind == CdtGraphityEdgeKind::Spatial)
                .count(),
            temporal_edges: self
                .edges
                .iter()
                .filter(|edge| edge.active && edge.kind == CdtGraphityEdgeKind::Temporal)
                .count(),
            tetrahedra: self.tetrahedra.len(),
            pruned_edges,
            proposed_edges: 0,
            causality_violations: self.causality_violations(),
        }
    }

    pub fn step(&mut self, expected_next: &[usize]) -> CdtGraphityStepReport {
        self.tick = self.tick.wrapping_add(1);
        let active = self.active_pattern();
        let predicted = self.predict_next(&active, expected_next.len().max(1) * 3);
        let prediction_error = prediction_error(&predicted, expected_next);

        self.update_temporal_edges(&active, expected_next, prediction_error);
        self.update_spatial_edges_from_curvature();
        let regge_action = self.regge_action();
        let free_energy = prediction_error + regge_action * 0.015 + self.temperature * 0.05;

        if prediction_error > 0.35 {
            self.temperature =
                (self.temperature + self.config.heating_rate * prediction_error).min(1.5);
        } else {
            self.temperature *= 1.0 - self.config.cooling_rate * (1.0 - prediction_error).max(0.0);
        }

        let pruned_edges = self.graphity_prune(free_energy);
        let proposed_edges = self.cdt_propose_edges(&active, expected_next, free_energy);
        self.rebuild_cdt_tetrahedra();
        self.propagate_activation();

        CdtGraphityStepReport {
            tick: self.tick,
            free_energy,
            regge_action,
            prediction_error,
            temperature: self.temperature,
            active_nodes: self
                .nodes
                .iter()
                .filter(|node| node.surprise > 0.05)
                .count(),
            active_edges: self.edges.iter().filter(|edge| edge.active).count(),
            spatial_edges: self
                .edges
                .iter()
                .filter(|edge| edge.active && edge.kind == CdtGraphityEdgeKind::Spatial)
                .count(),
            temporal_edges: self
                .edges
                .iter()
                .filter(|edge| edge.active && edge.kind == CdtGraphityEdgeKind::Temporal)
                .count(),
            tetrahedra: self.tetrahedra.len(),
            pruned_edges,
            proposed_edges,
            causality_violations: self.causality_violations(),
        }
    }

    pub fn regge_action(&self) -> f32 {
        let mut incident = vec![0_usize; self.edges.len()];
        for tetra in &self.tetrahedra {
            for i in 0..tetra.vertices.len() {
                for j in (i + 1)..tetra.vertices.len() {
                    if let Some(&edge_idx) = self
                        .edge_lookup
                        .get(&edge_key(tetra.vertices[i], tetra.vertices[j]))
                    {
                        if self.edges[edge_idx].active {
                            incident[edge_idx] += 1;
                        }
                    }
                }
            }
        }

        self.edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.active)
            .map(|(idx, edge)| {
                let target = match edge.kind {
                    CdtGraphityEdgeKind::Spatial => self.config.target_tetrahedra_per_edge,
                    CdtGraphityEdgeKind::Temporal => self.config.target_tetrahedra_per_edge + 1,
                };
                let deficit = target.abs_diff(incident[idx]) as f32;
                let length = match edge.kind {
                    CdtGraphityEdgeKind::Spatial => 1.0,
                    CdtGraphityEdgeKind::Temporal => 1.25,
                };
                length * deficit
            })
            .sum()
    }

    pub fn causality_violations(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| {
                edge.active
                    && match edge.kind {
                        CdtGraphityEdgeKind::Spatial => {
                            self.nodes[edge.a].slice != self.nodes[edge.b].slice
                        }
                        CdtGraphityEdgeKind::Temporal => {
                            self.nodes[edge.a].slice + 1 != self.nodes[edge.b].slice
                        }
                    }
            })
            .count()
    }

    pub fn save_persistent_state<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.serialize_persistent_state())
    }

    pub fn serialize_persistent_state(&self) -> String {
        let mut out = String::new();
        out.push_str("SNGA_CDT_GRAPHITY_STATE_V1\n");
        out.push_str(&format!(
            "config {} {} {:.7} {:.7} {} {} {} {:.7} {:.7} {:.7} {:.7} {} {}\n",
            self.config.slices,
            self.config.nodes_per_slice,
            self.config.initial_spatial_connectivity,
            self.config.initial_temporal_connectivity,
            self.config.target_spatial_degree,
            self.config.target_temporal_degree,
            self.config.target_tetrahedra_per_edge,
            self.config.cooling_rate,
            self.config.heating_rate,
            self.config.reinforcement_rate,
            self.config.prune_threshold,
            self.config.max_new_edges_per_step,
            self.config.seed
        ));
        out.push_str(&format!("tick {}\n", self.tick));
        out.push_str(&format!("temperature {:.7}\n", self.temperature));
        out.push_str(&format!("nodes {}\n", self.nodes.len()));
        for node in &self.nodes {
            out.push_str(&format!(
                "n {} {} {} {:.7}\n",
                node.id,
                node.slice,
                if node.activation { 1 } else { 0 },
                node.surprise
            ));
        }
        out.push_str(&format!("edges {}\n", self.edges.len()));
        for (idx, edge) in self.edges.iter().enumerate() {
            let kind = match edge.kind {
                CdtGraphityEdgeKind::Spatial => "spatial",
                CdtGraphityEdgeKind::Temporal => "temporal",
            };
            out.push_str(&format!(
                "e {} {} {} {} {:.7} {:.7} {}\n",
                idx,
                edge.a,
                edge.b,
                kind,
                edge.stability,
                edge.prediction_error,
                if edge.active { 1 } else { 0 }
            ));
        }
        out.push_str(&format!("tetrahedra {}\n", self.tetrahedra.len()));
        for (idx, tetra) in self.tetrahedra.iter().enumerate() {
            let kind = match tetra.kind {
                CdtSimplexKind::T31 => "t31",
                CdtSimplexKind::T22 => "t22",
            };
            out.push_str(&format!(
                "t {} {} {} {} {} {}\n",
                idx,
                tetra.vertices[0],
                tetra.vertices[1],
                tetra.vertices[2],
                tetra.vertices[3],
                kind
            ));
        }
        out.push_str("end\n");
        out
    }

    fn add_edge(&mut self, a: usize, b: usize, kind: CdtGraphityEdgeKind, stability: f32) -> bool {
        if !self.is_valid_edge(a, b, kind) {
            return false;
        }
        let (a, b) = orient_edge(&self.nodes, a, b, kind);
        let key = edge_key(a, b);
        if let Some(&idx) = self.edge_lookup.get(&key) {
            let edge = &mut self.edges[idx];
            edge.active = true;
            edge.stability = edge.stability.max(stability);
            return false;
        }
        let idx = self.edges.len();
        self.edges.push(CdtGraphityEdge {
            a,
            b,
            kind,
            stability: stability.clamp(0.0, 1.0),
            prediction_error: 0.0,
            active: true,
        });
        self.edge_lookup.insert(key, idx);
        self.adjacency[a].push(idx);
        self.adjacency[b].push(idx);
        true
    }

    fn is_valid_edge(&self, a: usize, b: usize, kind: CdtGraphityEdgeKind) -> bool {
        if a == b || a >= self.nodes.len() || b >= self.nodes.len() {
            return false;
        }
        let sa = self.nodes[a].slice;
        let sb = self.nodes[b].slice;
        match kind {
            CdtGraphityEdgeKind::Spatial => sa == sb,
            CdtGraphityEdgeKind::Temporal => sa.abs_diff(sb) == 1,
        }
    }

    fn slice_nodes(&self, slice: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|node| node.slice == slice)
            .map(|node| node.id)
            .collect()
    }

    fn temporal_target_from(&self, edge: &CdtGraphityEdge, source: usize) -> Option<usize> {
        if edge.kind != CdtGraphityEdgeKind::Temporal {
            return None;
        }
        if edge.a == source {
            Some(edge.b)
        } else {
            None
        }
    }

    fn update_temporal_edges(&mut self, active: &[usize], expected_next: &[usize], error: f32) {
        let active = active.iter().copied().collect::<HashSet<_>>();
        let expected = expected_next.iter().copied().collect::<HashSet<_>>();
        for edge in &mut self.edges {
            if !edge.active
                || edge.kind != CdtGraphityEdgeKind::Temporal
                || !active.contains(&edge.a)
            {
                continue;
            }
            let success = expected.contains(&edge.b);
            if success {
                edge.stability += self.config.reinforcement_rate * (1.0 - edge.stability);
                edge.prediction_error *= 0.65;
            } else {
                edge.prediction_error = (edge.prediction_error + error * 0.35).min(1.0);
                edge.stability *= 1.0 - self.temperature.min(1.0) * 0.05;
            }
            edge.stability = edge.stability.clamp(0.0, 1.0);
        }
    }

    fn update_spatial_edges_from_curvature(&mut self) {
        let mut incident = vec![0_usize; self.edges.len()];
        for tetra in &self.tetrahedra {
            for i in 0..tetra.vertices.len() {
                for j in (i + 1)..tetra.vertices.len() {
                    if let Some(&edge_idx) = self
                        .edge_lookup
                        .get(&edge_key(tetra.vertices[i], tetra.vertices[j]))
                    {
                        incident[edge_idx] += 1;
                    }
                }
            }
        }
        for (idx, edge) in self.edges.iter_mut().enumerate() {
            if !edge.active || edge.kind != CdtGraphityEdgeKind::Spatial {
                continue;
            }
            let deficit = self
                .config
                .target_tetrahedra_per_edge
                .abs_diff(incident[idx]) as f32;
            let curvature_penalty =
                (deficit / self.config.target_tetrahedra_per_edge.max(1) as f32).min(1.0)
                    * self.temperature
                    * 0.05;
            edge.prediction_error = (edge.prediction_error + curvature_penalty).min(1.0);
            edge.stability *= 1.0 - curvature_penalty * 0.5;
        }
    }

    fn graphity_prune(&mut self, free_energy: f32) -> usize {
        let mut pruned = 0;
        let heat = (self.temperature + free_energy * 0.05).clamp(0.0, 1.5);
        for edge in &mut self.edges {
            if !edge.active {
                continue;
            }
            let instability = edge.prediction_error * heat;
            if edge.stability < self.config.prune_threshold || instability > 0.85 {
                edge.active = false;
                pruned += 1;
            }
        }
        pruned
    }

    fn cdt_propose_edges(
        &mut self,
        active: &[usize],
        expected_next: &[usize],
        free_energy: f32,
    ) -> usize {
        if free_energy < 0.20 {
            return 0;
        }
        let mut proposed = 0;
        let active = active
            .iter()
            .copied()
            .filter(|idx| *idx < self.nodes.len())
            .collect::<Vec<_>>();
        let expected = expected_next
            .iter()
            .copied()
            .filter(|idx| *idx < self.nodes.len())
            .collect::<Vec<_>>();

        for &source in &active {
            for &target in &expected {
                if proposed >= self.config.max_new_edges_per_step {
                    return proposed;
                }
                if self.nodes[source].slice + 1 == self.nodes[target].slice
                    && self.add_edge(source, target, CdtGraphityEdgeKind::Temporal, 0.40)
                {
                    proposed += 1;
                }
            }
        }

        while proposed < self.config.max_new_edges_per_step {
            let slice = self.rng.gen_range(0..self.config.slices);
            let ids = self.slice_nodes(slice);
            if ids.len() < 2 {
                break;
            }
            let a = ids[self.rng.gen_range(0..ids.len())];
            let b = ids[self.rng.gen_range(0..ids.len())];
            if self.add_edge(a, b, CdtGraphityEdgeKind::Spatial, 0.25) {
                proposed += 1;
            } else if self.rng.gen::<f32>() > self.temperature.min(1.0) {
                break;
            }
        }
        proposed
    }

    fn rebuild_cdt_tetrahedra(&mut self) {
        self.tetrahedra.clear();
        let tetra_limit = self.nodes.len().saturating_mul(8).max(1);
        for slice in 0..self.config.slices.saturating_sub(1) {
            let current = self.slice_nodes(slice);
            let next = self.slice_nodes(slice + 1);
            for triple in current.windows(3) {
                if self.tetrahedra.len() >= tetra_limit {
                    return;
                }
                if let Some(&future) = next.iter().find(|&&candidate| {
                    triple
                        .iter()
                        .all(|&source| self.has_active_edge(source, candidate))
                }) {
                    self.tetrahedra.push(CdtGraphitySimplex3 {
                        vertices: [triple[0], triple[1], triple[2], future],
                        kind: CdtSimplexKind::T31,
                    });
                }
            }
            for pair_current in current.windows(2) {
                for pair_next in next.windows(2) {
                    if self.tetrahedra.len() >= tetra_limit {
                        return;
                    }
                    if self.has_active_edge(pair_current[0], pair_next[0])
                        && self.has_active_edge(pair_current[1], pair_next[1])
                    {
                        self.tetrahedra.push(CdtGraphitySimplex3 {
                            vertices: [
                                pair_current[0],
                                pair_current[1],
                                pair_next[0],
                                pair_next[1],
                            ],
                            kind: CdtSimplexKind::T22,
                        });
                        break;
                    }
                }
            }
        }
    }

    fn has_active_edge(&self, a: usize, b: usize) -> bool {
        self.edge_lookup
            .get(&edge_key(a, b))
            .map(|&idx| self.edges[idx].active)
            .unwrap_or(false)
    }

    fn propagate_activation(&mut self) {
        let active = self.active_pattern();
        let mut next_surprise = vec![0.0_f32; self.nodes.len()];
        for source in active {
            for &edge_idx in &self.adjacency[source] {
                let edge = &self.edges[edge_idx];
                if !edge.active {
                    continue;
                }
                let target = match edge.kind {
                    CdtGraphityEdgeKind::Spatial => {
                        if edge.a == source {
                            edge.b
                        } else {
                            edge.a
                        }
                    }
                    CdtGraphityEdgeKind::Temporal => {
                        let Some(target) = self.temporal_target_from(edge, source) else {
                            continue;
                        };
                        target
                    }
                };
                next_surprise[target] =
                    next_surprise[target].max(self.nodes[source].surprise * edge.stability * 0.55);
            }
        }
        for (idx, node) in self.nodes.iter_mut().enumerate() {
            node.surprise = (node.surprise * 0.45).max(next_surprise[idx]);
            node.activation = node.surprise > 0.08;
        }
    }
}

fn prediction_error(predicted: &[(usize, f32)], expected: &[usize]) -> f32 {
    if expected.is_empty() {
        return 0.0;
    }
    let predicted = predicted
        .iter()
        .map(|(idx, _)| *idx)
        .collect::<HashSet<_>>();
    let expected = expected.iter().copied().collect::<HashSet<_>>();
    let matched = expected
        .iter()
        .filter(|idx| predicted.contains(idx))
        .count();
    1.0 - matched as f32 / expected.len().max(1) as f32
}

fn orient_edge(
    nodes: &[CdtGraphityNode],
    a: usize,
    b: usize,
    kind: CdtGraphityEdgeKind,
) -> (usize, usize) {
    match kind {
        CdtGraphityEdgeKind::Spatial => {
            if a <= b {
                (a, b)
            } else {
                (b, a)
            }
        }
        CdtGraphityEdgeKind::Temporal => {
            if nodes[a].slice <= nodes[b].slice {
                (a, b)
            } else {
                (b, a)
            }
        }
    }
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}
