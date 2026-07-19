//! Red simplicial vacía que crece por estímulos y se repara mediante ondas piloto.
//!
//! La consolidación está separada del aprendizaje provisional: un rótulo sólo
//! pasa a memoria consolidada cuando su tetraedro alcanza el umbral de simetría.

use crate::simplicial_thermodynamic_engine::{
    SimplicialEngineConfig, SimplicialHodgeOperators, SimplicialThermodynamicEngine,
};
use crate::symmetry_thermodynamic_substrate::{
    SimplicialEdge, SymmetryMetrics, SymmetrySubstrateError, SymmetryThermodynamicConfig,
    SymmetryThermodynamicSubstrate, Vec3,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
pub struct PlasticSymmetryConfig {
    pub initial_wave_amplitude: f64,
    pub wave_decay: f64,
    pub minimum_wave_amplitude: f64,
    pub collision_symmetry_threshold: f64,
    pub consolidation_symmetry_threshold: f64,
    pub edge_rewrite_strain: f64,
    pub base_relaxation_steps: usize,
    pub relaxation_tolerance: f64,
    pub target_edge_length: f64,
    pub substrate: SymmetryThermodynamicConfig,
}

impl Default for PlasticSymmetryConfig {
    fn default() -> Self {
        Self {
            initial_wave_amplitude: 1.0,
            wave_decay: 0.45,
            minimum_wave_amplitude: 0.10,
            collision_symmetry_threshold: 0.995,
            // "Perfecta" dentro de tolerancia numérica; 1.0 exacto no es robusto
            // con fronteras compartidas y aritmética flotante.
            consolidation_symmetry_threshold: 0.999,
            edge_rewrite_strain: 0.025,
            base_relaxation_steps: 600,
            relaxation_tolerance: 1.0e-14,
            target_edge_length: 1.0,
            substrate: SymmetryThermodynamicConfig {
                temperature: 0.0,
                dt: 0.12,
                ..SymmetryThermodynamicConfig::default()
            },
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlasticRegionState {
    pub pending_label: Option<String>,
    pub consolidated_label: Option<String>,
    pub consolidation_count: u64,
    pub last_wave: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PilotWaveFrame {
    pub hop: usize,
    pub amplitude: f64,
    pub visited_tetrahedra: usize,
    pub broken_edges: usize,
    pub reformed_edges: usize,
    pub newly_consolidated: usize,
    pub reconsolidated: usize,
    pub mean_local_symmetry: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PilotWaveReport {
    pub wave_id: u64,
    pub origin_tetrahedron: usize,
    pub frames: Vec<PilotWaveFrame>,
    pub reached_tetrahedra: usize,
    pub total_broken_edges: usize,
    pub total_reformed_edges: usize,
    pub total_newly_consolidated: usize,
    pub total_reconsolidated: usize,
}

#[derive(Clone, Debug)]
pub struct NativePlasticSymmetryNetwork {
    pub config: PlasticSymmetryConfig,
    pub engine: Option<SimplicialThermodynamicEngine>,
    pub regions: Vec<PlasticRegionState>,
    wave_id: u64,
}

impl NativePlasticSymmetryNetwork {
    pub fn new(config: PlasticSymmetryConfig) -> Self {
        Self {
            config: sanitized_config(config),
            engine: None,
            regions: Vec::new(),
            wave_id: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.engine.is_none()
    }

    pub fn tetrahedron_count(&self) -> usize {
        self.engine
            .as_ref()
            .map_or(0, |engine| engine.substrate.tetrahedra.len())
    }

    pub fn vertex_count(&self) -> usize {
        self.engine
            .as_ref()
            .map_or(0, |engine| engine.substrate.vertices.len())
    }

    /// Entrena en el frente de crecimiento. El primer estímulo nuclea la red.
    pub fn train_new_region(
        &mut self,
        label: impl Into<String>,
    ) -> Result<PilotWaveReport, SymmetrySubstrateError> {
        let label = label.into();
        let origin = if self.engine.is_none() {
            self.nucleate()?;
            0
        } else {
            self.grow_neighbor(self.tetrahedron_count() - 1)?
        };
        Ok(self.stimulate_region(origin, label))
    }

    pub fn stimulate_region(
        &mut self,
        tetrahedron: usize,
        label: impl Into<String>,
    ) -> PilotWaveReport {
        if tetrahedron >= self.regions.len() {
            return PilotWaveReport::default();
        }
        self.regions[tetrahedron].pending_label = Some(label.into());
        self.propagate_pilot_wave(tetrahedron)
    }

    pub fn recall(&self, tetrahedron: usize) -> Option<&str> {
        let engine = self.engine.as_ref()?;
        let region = self.regions.get(tetrahedron)?;
        if local_symmetry(engine, tetrahedron) + EPSILON
            < self.config.consolidation_symmetry_threshold
        {
            return None;
        }
        region.consolidated_label.as_deref()
    }

    pub fn local_symmetry(&self, tetrahedron: usize) -> Option<f64> {
        self.engine
            .as_ref()
            .map(|engine| local_symmetry(engine, tetrahedron))
    }

    pub fn global_metrics(&self) -> Option<SymmetryMetrics> {
        self.engine
            .as_ref()
            .map(|engine| engine.substrate.metrics())
    }

    pub fn lesion_region(
        &mut self,
        tetrahedron: usize,
        displacement: Vec3,
        remove_edge: bool,
    ) -> bool {
        let Some(engine) = self.engine.as_mut() else {
            return false;
        };
        let Some(simplex) = engine.substrate.tetrahedra.get(tetrahedron).copied() else {
            return false;
        };
        let moved = *simplex.0.iter().max().expect("tetrahedron is non-empty");
        engine.substrate.break_symmetry(moved, displacement, 0.0);
        if remove_edge {
            let partner = *simplex
                .0
                .iter()
                .filter(|vertex| **vertex != moved)
                .min()
                .expect("tetrahedron has another vertex");
            engine.substrate.remove_edge(moved, partner);
        }
        true
    }

    fn nucleate(&mut self) -> Result<usize, SymmetrySubstrateError> {
        let substrate = SymmetryThermodynamicSubstrate::regular_tetrahedron(
            self.config.target_edge_length,
            self.config.substrate,
        )?;
        self.engine = Some(SimplicialThermodynamicEngine::new(
            substrate,
            SimplicialEngineConfig::default(),
        ));
        self.regions.push(PlasticRegionState::default());
        Ok(0)
    }

    fn grow_neighbor(&mut self, parent: usize) -> Result<usize, SymmetrySubstrateError> {
        let engine = self.engine.as_mut().expect("network was nucleated");
        let parent_tetrahedron = engine.substrate.tetrahedra[parent].0;
        let incidence = face_incidence(&engine.substrate);
        let mut candidates = tetrahedron_faces(parent_tetrahedron)
            .into_iter()
            .filter(|face| incidence.get(face).copied().unwrap_or(0) == 1)
            .collect::<Vec<_>>();
        candidates.sort_unstable();

        let mut selected = None;
        for face in candidates {
            let opposite = parent_tetrahedron
                .into_iter()
                .find(|vertex| !face.contains(vertex))
                .expect("a tetrahedron has one opposite vertex");
            let position = reflect_across_face(
                engine.substrate.vertices[opposite].position,
                engine.substrate.vertices[face[0]].position,
                engine.substrate.vertices[face[1]].position,
                engine.substrate.vertices[face[2]].position,
            );
            let duplicates_existing = engine
                .substrate
                .vertices
                .iter()
                .any(|vertex| (vertex.position - position).norm() < 1.0e-8);
            if !duplicates_existing {
                selected = Some((face, position));
                break;
            }
        }
        let (face, position) = selected.expect("no available boundary face for growth");
        let tetrahedron =
            engine
                .substrate
                .append_tetrahedron(face, position, self.config.target_edge_length)?;
        engine.hodge = SimplicialHodgeOperators::from_substrate(&engine.substrate);
        engine.edge_flow.resize(engine.hodge.edge_count(), 0.0);
        engine.face_flux.resize(engine.hodge.triangle_count(), 0.0);
        self.regions.push(PlasticRegionState::default());
        Ok(tetrahedron)
    }

    fn propagate_pilot_wave(&mut self, origin: usize) -> PilotWaveReport {
        let Some(engine) = self.engine.as_mut() else {
            return PilotWaveReport::default();
        };
        self.wave_id += 1;
        let wave_id = self.wave_id;
        let distances = tetrahedral_distances(&engine.substrate, origin);
        let max_hop = distances
            .iter()
            .copied()
            .filter(|distance| *distance != usize::MAX)
            .max()
            .unwrap_or(0);
        let mut report = PilotWaveReport {
            wave_id,
            origin_tetrahedron: origin,
            ..PilotWaveReport::default()
        };

        for hop in 0..=max_hop {
            let amplitude =
                self.config.initial_wave_amplitude * self.config.wave_decay.powi(hop as i32);
            if amplitude + EPSILON < self.config.minimum_wave_amplitude {
                break;
            }
            let layer = distances
                .iter()
                .enumerate()
                .filter(|(_, distance)| **distance == hop)
                .map(|(tetrahedron, _)| tetrahedron)
                .collect::<Vec<_>>();
            if layer.is_empty() {
                continue;
            }

            let mut broken_edges = 0;
            let mut reformed_edges = 0;
            for &tetrahedron in &layer {
                if local_symmetry(engine, tetrahedron) < self.config.collision_symmetry_threshold {
                    let rewrite = recrystallize_edges(
                        engine,
                        tetrahedron,
                        self.config.target_edge_length,
                        self.config.edge_rewrite_strain,
                    );
                    broken_edges += rewrite.0;
                    reformed_edges += rewrite.1;
                }
            }
            let steps =
                ((self.config.base_relaxation_steps as f64 * amplitude).ceil() as usize).max(1);
            let reached = distances
                .iter()
                .enumerate()
                .filter(|(_, distance)| **distance <= hop)
                .map(|(tetrahedron, _)| tetrahedron)
                .collect::<Vec<_>>();
            let mut frozen_vertices = distances
                .iter()
                .enumerate()
                .filter(|(_, distance)| **distance < hop)
                .flat_map(|(tetrahedron, _)| engine.substrate.tetrahedra[tetrahedron].0)
                .collect::<BTreeSet<_>>();
            for (tetrahedron, region) in self.regions.iter().enumerate() {
                if region.consolidated_label.is_some()
                    && local_symmetry(engine, tetrahedron) + EPSILON
                        >= self.config.consolidation_symmetry_threshold
                {
                    frozen_vertices.extend(engine.substrate.tetrahedra[tetrahedron].0);
                }
            }
            relax_tetrahedra(
                engine,
                &layer,
                &frozen_vertices,
                steps,
                amplitude,
                self.config.relaxation_tolerance,
            );
            engine.hodge = SimplicialHodgeOperators::from_substrate(&engine.substrate);
            engine.edge_flow.resize(engine.hodge.edge_count(), 0.0);
            engine.face_flux.resize(engine.hodge.triangle_count(), 0.0);

            let mut newly_consolidated = 0;
            let mut reconsolidated = 0;
            let mut symmetry_sum = 0.0;
            for &tetrahedron in &layer {
                let symmetry = local_symmetry(engine, tetrahedron);
                symmetry_sum += symmetry;
                self.regions[tetrahedron].last_wave = wave_id;
            }
            for &tetrahedron in &reached {
                let symmetry = local_symmetry(engine, tetrahedron);
                if symmetry + EPSILON >= self.config.consolidation_symmetry_threshold {
                    if let Some(label) = self.regions[tetrahedron].pending_label.take() {
                        if self.regions[tetrahedron].consolidated_label.is_some() {
                            reconsolidated += 1;
                        } else {
                            newly_consolidated += 1;
                        }
                        self.regions[tetrahedron].consolidated_label = Some(label);
                        self.regions[tetrahedron].consolidation_count += 1;
                    }
                }
            }
            report.frames.push(PilotWaveFrame {
                hop,
                amplitude,
                visited_tetrahedra: layer.len(),
                broken_edges,
                reformed_edges,
                newly_consolidated,
                reconsolidated,
                mean_local_symmetry: symmetry_sum / layer.len() as f64,
            });
            report.reached_tetrahedra += layer.len();
            report.total_broken_edges += broken_edges;
            report.total_reformed_edges += reformed_edges;
            report.total_newly_consolidated += newly_consolidated;
            report.total_reconsolidated += reconsolidated;
        }
        report
    }
}

fn sanitized_config(config: PlasticSymmetryConfig) -> PlasticSymmetryConfig {
    PlasticSymmetryConfig {
        initial_wave_amplitude: config.initial_wave_amplitude.max(0.0),
        wave_decay: config.wave_decay.clamp(0.0, 1.0),
        minimum_wave_amplitude: config.minimum_wave_amplitude.max(0.0),
        collision_symmetry_threshold: config.collision_symmetry_threshold.clamp(0.0, 1.0),
        consolidation_symmetry_threshold: config.consolidation_symmetry_threshold.clamp(0.0, 1.0),
        edge_rewrite_strain: config.edge_rewrite_strain.max(0.0),
        base_relaxation_steps: config.base_relaxation_steps.max(1),
        relaxation_tolerance: config.relaxation_tolerance.max(0.0),
        target_edge_length: config.target_edge_length.abs().max(EPSILON),
        ..config
    }
}

fn relax_tetrahedra(
    engine: &mut SimplicialThermodynamicEngine,
    tetrahedra: &[usize],
    frozen_vertices: &BTreeSet<usize>,
    steps: usize,
    amplitude: f64,
    tolerance: f64,
) {
    let geometry_weight = engine.substrate.config.geometry_weight;
    if geometry_weight <= EPSILON {
        return;
    }
    let edge_keys = tetrahedra
        .iter()
        .flat_map(|tetrahedron| {
            let simplex = engine.substrate.tetrahedra[*tetrahedron];
            let mut edges = Vec::with_capacity(6);
            for i in 0..4 {
                for j in (i + 1)..4 {
                    edges.push(ordered_pair(simplex.0[i], simplex.0[j]));
                }
            }
            edges
        })
        .collect::<BTreeSet<_>>();
    let edge_count = edge_keys.len().max(1) as f64;
    let dt = engine.substrate.config.dt * amplitude.max(0.0);
    let mut previous_energy = f64::INFINITY;
    for _ in 0..steps {
        let mut gradients = vec![Vec3::default(); engine.substrate.vertices.len()];
        let mut energy = 0.0;
        for &key in &edge_keys {
            let Some(edge) = engine
                .substrate
                .edges
                .iter()
                .find(|edge| ordered_pair(edge.a, edge.b) == key)
            else {
                continue;
            };
            let delta = engine.substrate.vertices[edge.a].position
                - engine.substrate.vertices[edge.b].position;
            let length = delta.norm().max(EPSILON);
            let target = edge.target_length.max(EPSILON);
            let strain = (length - target) / target;
            energy += strain * strain / edge_count;
            let factor =
                geometry_weight * 2.0 * (length - target) / (edge_count * target * target * length);
            let gradient = delta * factor;
            gradients[edge.a] = gradients[edge.a] + gradient;
            gradients[edge.b] = gradients[edge.b] - gradient;
        }
        for (index, (vertex, gradient)) in engine
            .substrate
            .vertices
            .iter_mut()
            .zip(gradients)
            .enumerate()
        {
            if !frozen_vertices.contains(&index) {
                vertex.position = vertex.position - gradient * dt;
            }
        }
        if (previous_energy - energy).abs() <= tolerance {
            break;
        }
        previous_energy = energy;
    }
}

fn local_symmetry(engine: &SimplicialThermodynamicEngine, tetrahedron: usize) -> f64 {
    let Some(simplex) = engine.substrate.tetrahedra.get(tetrahedron) else {
        return 0.0;
    };
    let edges = engine
        .substrate
        .edges
        .iter()
        .map(|edge| (ordered_pair(edge.a, edge.b), edge))
        .collect::<BTreeMap<_, _>>();
    let mut strain = 0.0;
    let mut missing = 0;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let key = ordered_pair(simplex.0[i], simplex.0[j]);
            if let Some(edge) = edges.get(&key) {
                let length = (engine.substrate.vertices[edge.a].position
                    - engine.substrate.vertices[edge.b].position)
                    .norm();
                strain += ((length - edge.target_length) / edge.target_length.max(EPSILON)).powi(2);
            } else {
                missing += 1;
            }
        }
    }
    let defect = missing as f64 / 6.0;
    (-(24.0 * strain / 6.0 + 4.0 * defect)).exp()
}

fn recrystallize_edges(
    engine: &mut SimplicialThermodynamicEngine,
    tetrahedron: usize,
    target_length: f64,
    rewrite_strain: f64,
) -> (usize, usize) {
    let simplex = engine.substrate.tetrahedra[tetrahedron];
    let mut required = Vec::with_capacity(6);
    for i in 0..4 {
        for j in (i + 1)..4 {
            required.push(ordered_pair(simplex.0[i], simplex.0[j]));
        }
    }
    let mut existing = engine
        .substrate
        .edges
        .iter()
        .map(|edge| ordered_pair(edge.a, edge.b))
        .collect::<BTreeSet<_>>();
    let mut broken = 0;
    let mut reformed = 0;

    let most_strained = required
        .iter()
        .filter_map(|key| {
            let edge = engine
                .substrate
                .edges
                .iter()
                .find(|edge| ordered_pair(edge.a, edge.b) == *key)?;
            let length = (engine.substrate.vertices[edge.a].position
                - engine.substrate.vertices[edge.b].position)
                .norm();
            Some((
                *key,
                ((length - edge.target_length) / edge.target_length).abs(),
            ))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1));
    if let Some((edge, strain)) = most_strained {
        if strain > rewrite_strain && engine.substrate.remove_edge(edge.0, edge.1) {
            existing.remove(&edge);
            broken += 1;
        }
    }
    for edge in required {
        if existing.insert(edge) {
            engine.substrate.edges.push(SimplicialEdge {
                a: edge.0,
                b: edge.1,
                target_length,
                weight: 1.0,
            });
            reformed += 1;
        }
    }
    (broken, reformed)
}

fn tetrahedral_distances(substrate: &SymmetryThermodynamicSubstrate, origin: usize) -> Vec<usize> {
    let count = substrate.tetrahedra.len();
    let mut distances = vec![usize::MAX; count];
    if origin >= count {
        return distances;
    }
    distances[origin] = 0;
    let mut queue = VecDeque::from([origin]);
    while let Some(current) = queue.pop_front() {
        for candidate in 0..count {
            if distances[candidate] != usize::MAX {
                continue;
            }
            let shared = substrate.tetrahedra[current]
                .0
                .iter()
                .filter(|vertex| substrate.tetrahedra[candidate].0.contains(vertex))
                .count();
            if shared >= 3 {
                distances[candidate] = distances[current] + 1;
                queue.push_back(candidate);
            }
        }
    }
    distances
}

fn face_incidence(substrate: &SymmetryThermodynamicSubstrate) -> BTreeMap<[usize; 3], usize> {
    let mut incidence = BTreeMap::new();
    for tetrahedron in &substrate.tetrahedra {
        for face in tetrahedron_faces(tetrahedron.0) {
            *incidence.entry(face).or_default() += 1;
        }
    }
    incidence
}

fn tetrahedron_faces(tetrahedron: [usize; 4]) -> [[usize; 3]; 4] {
    let mut faces = [
        [tetrahedron[1], tetrahedron[2], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[2], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[1], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[1], tetrahedron[2]],
    ];
    for face in &mut faces {
        face.sort_unstable();
    }
    faces
}

fn reflect_across_face(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let normal = (b - a).cross(c - a);
    let unit = normal / normal.norm();
    point - unit * (2.0 * (point - a).dot(unit))
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_network_nucleates_and_grows_only_when_trained() {
        let mut network = NativePlasticSymmetryNetwork::new(PlasticSymmetryConfig::default());
        assert!(network.is_empty());
        let first = network.train_new_region("entrada_a").unwrap();
        assert_eq!(network.tetrahedron_count(), 1);
        assert_eq!(network.recall(0), Some("entrada_a"));
        assert_eq!(first.total_newly_consolidated, 1);

        network.train_new_region("entrada_b").unwrap();
        assert_eq!(network.tetrahedron_count(), 2);
        assert_eq!(network.recall(1), Some("entrada_b"));
    }

    #[test]
    fn decaying_wave_does_not_crystallize_the_entire_network() {
        let mut network = NativePlasticSymmetryNetwork::new(PlasticSymmetryConfig::default());
        for index in 0..6 {
            network.train_new_region(format!("m{index}")).unwrap();
        }
        for tetrahedron in 0..network.tetrahedron_count() {
            network.lesion_region(tetrahedron, Vec3::new(0.18, -0.09, 0.12), true);
        }
        let report = network.stimulate_region(0, "nuevo");
        assert!(report.reached_tetrahedra < network.tetrahedron_count());
        let unreached = network
            .regions
            .iter()
            .filter(|region| region.last_wave != report.wave_id)
            .count();
        assert!(unreached > 0);
        assert!(network.recall(network.tetrahedron_count() - 1).is_none());
    }

    #[test]
    fn collision_rewrites_edges_and_recovers_consolidated_memory() {
        let mut network = NativePlasticSymmetryNetwork::new(PlasticSymmetryConfig::default());
        network.train_new_region("memoria").unwrap();
        network.lesion_region(0, Vec3::new(0.42, -0.18, 0.27), true);
        assert!(network.recall(0).is_none());
        let report = network.stimulate_region(0, "memoria");
        assert!(report.total_reformed_edges >= 1);
        assert_eq!(network.recall(0), Some("memoria"));
        assert!(
            network.local_symmetry(0).unwrap() >= network.config.consolidation_symmetry_threshold
        );
    }

    #[test]
    fn consolidation_requires_symmetry_and_supports_reconsolidation() {
        let mut network = NativePlasticSymmetryNetwork::new(PlasticSymmetryConfig::default());
        network.train_new_region("antigua").unwrap();
        network.lesion_region(0, Vec3::new(0.35, 0.12, -0.21), false);
        let report = network.stimulate_region(0, "nueva");
        assert_eq!(report.total_reconsolidated, 1);
        assert_eq!(network.recall(0), Some("nueva"));

        let mut ablated_config = PlasticSymmetryConfig::default();
        ablated_config.substrate.geometry_weight = 0.0;
        let mut ablated = NativePlasticSymmetryNetwork::new(ablated_config);
        ablated.train_new_region("antigua").unwrap();
        ablated.lesion_region(0, Vec3::new(0.35, 0.12, -0.21), false);
        let failed = ablated.stimulate_region(0, "nueva");
        assert_eq!(failed.total_reconsolidated, 0);
        assert!(ablated.recall(0).is_none());
        assert_eq!(
            ablated.regions[0].consolidated_label.as_deref(),
            Some("antigua")
        );
    }
}
