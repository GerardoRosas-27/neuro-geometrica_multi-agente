//! Sustrato termodinámico independiente sobre complejos simpliciales 3D.
//!
//! El motor no afirma que la simetría sea cognición. Implementa una hipótesis
//! falsable más limitada: un patrón se representa como un campo escalar sobre
//! vértices, su irregularidad se mide con el Laplaciano de Hodge de grado cero
//! y la geometría se relaja hacia tetraedros localmente simétricos.
//!
//! Fundamentos:
//! - `L0 = B1 * B1^T` y `x^T L0 x = sum_edges w_ij (x_i-x_j)^2`.
//! - La simetría tetraédrica completa es el grupo `S4` de 24 permutaciones.
//! - La energía elástica de arista es `w ((l-l0)/l0)^2`.
//! - La energía libre usada por el integrador es `F = U - T S`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimplicialVertex {
    pub position: Vec3,
    pub field: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimplicialEdge {
    pub a: usize,
    pub b: usize,
    pub target_length: f64,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tetrahedron(pub [usize; 4]);

#[derive(Clone, Copy, Debug)]
pub struct SymmetryThermodynamicConfig {
    pub dt: f64,
    pub min_dt: f64,
    pub temperature: f64,
    pub cooling_rate: f64,
    pub geometry_weight: f64,
    pub hodge_weight: f64,
    pub prediction_weight: f64,
    pub topology_weight: f64,
    pub max_line_search_steps: usize,
}

impl Default for SymmetryThermodynamicConfig {
    fn default() -> Self {
        Self {
            dt: 0.08,
            min_dt: 1.0e-8,
            temperature: 0.05,
            cooling_rate: 0.995,
            geometry_weight: 1.0,
            hodge_weight: 0.35,
            prediction_weight: 1.0,
            topology_weight: 2.0,
            max_line_search_steps: 24,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SymmetryMetrics {
    /// Uno significa invariancia local exacta bajo las 24 acciones de S4.
    pub symmetry_score: f64,
    pub automorphism_residual: f64,
    pub edge_length_cv: f64,
    pub topology_defect: f64,
    pub hodge_energy: f64,
    pub prediction_error: f64,
    pub internal_energy: f64,
    pub entropy: f64,
    pub free_energy: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SymmetryStepReport {
    pub tick: u64,
    pub accepted: bool,
    pub accepted_dt: f64,
    pub before: SymmetryMetrics,
    pub after: SymmetryMetrics,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TopologyRepairReport {
    pub inserted_edges: usize,
    pub removed_duplicates: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SymmetrySubstrateError {
    EmptyVertices,
    EmptyTetrahedra,
    InvalidTargetLength,
    VertexOutOfBounds { tetrahedron: usize, vertex: usize },
    RepeatedVertex { tetrahedron: usize, vertex: usize },
    DegenerateTetrahedron { tetrahedron: usize },
}

impl fmt::Display for SymmetrySubstrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVertices => write!(f, "el complejo no contiene vértices"),
            Self::EmptyTetrahedra => write!(f, "el complejo no contiene tetraedros"),
            Self::InvalidTargetLength => {
                write!(f, "la longitud objetivo debe ser finita y positiva")
            }
            Self::VertexOutOfBounds {
                tetrahedron,
                vertex,
            } => write!(
                f,
                "tetraedro {tetrahedron} referencia el vértice inexistente {vertex}"
            ),
            Self::RepeatedVertex {
                tetrahedron,
                vertex,
            } => write!(f, "tetraedro {tetrahedron} repite el vértice {vertex}"),
            Self::DegenerateTetrahedron { tetrahedron } => {
                write!(f, "tetraedro {tetrahedron} tiene volumen nulo")
            }
        }
    }
}

impl std::error::Error for SymmetrySubstrateError {}

#[derive(Clone, Debug)]
pub struct SymmetryThermodynamicSubstrate {
    pub config: SymmetryThermodynamicConfig,
    pub vertices: Vec<SimplicialVertex>,
    pub edges: Vec<SimplicialEdge>,
    pub tetrahedra: Vec<Tetrahedron>,
    observations: Vec<f64>,
    observation_mask: Vec<bool>,
    tick: u64,
}

impl SymmetryThermodynamicSubstrate {
    pub fn new(
        positions: Vec<Vec3>,
        tetrahedra: Vec<Tetrahedron>,
        target_edge_length: f64,
        config: SymmetryThermodynamicConfig,
    ) -> Result<Self, SymmetrySubstrateError> {
        if positions.is_empty() {
            return Err(SymmetrySubstrateError::EmptyVertices);
        }
        if tetrahedra.is_empty() {
            return Err(SymmetrySubstrateError::EmptyTetrahedra);
        }
        if !target_edge_length.is_finite() || target_edge_length <= EPSILON {
            return Err(SymmetrySubstrateError::InvalidTargetLength);
        }
        validate_tetrahedra(&positions, &tetrahedra)?;

        let vertices = positions
            .into_iter()
            .map(|position| SimplicialVertex {
                position,
                field: 0.0,
            })
            .collect::<Vec<_>>();
        let edges = expected_edge_keys(&tetrahedra)
            .into_iter()
            .map(|(a, b)| SimplicialEdge {
                a,
                b,
                target_length: target_edge_length,
                weight: 1.0,
            })
            .collect();
        let node_count = vertices.len();
        Ok(Self {
            config: sanitized_config(config),
            vertices,
            edges,
            tetrahedra,
            observations: vec![0.0; node_count],
            observation_mask: vec![false; node_count],
            tick: 0,
        })
    }

    pub fn regular_tetrahedron(
        edge_length: f64,
        config: SymmetryThermodynamicConfig,
    ) -> Result<Self, SymmetrySubstrateError> {
        let h2 = (3.0_f64).sqrt() * edge_length / 2.0;
        let h3 = (2.0_f64 / 3.0).sqrt() * edge_length;
        Self::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(edge_length, 0.0, 0.0),
                Vec3::new(edge_length / 2.0, h2, 0.0),
                Vec3::new(edge_length / 2.0, h2 / 3.0, h3),
            ],
            vec![Tetrahedron([0, 1, 2, 3])],
            edge_length,
            config,
        )
    }

    /// Añade un tetraedro pegado a una cara existente sin reconstruir el motor.
    pub fn append_tetrahedron(
        &mut self,
        face: [usize; 3],
        new_position: Vec3,
        target_edge_length: f64,
    ) -> Result<usize, SymmetrySubstrateError> {
        let tetrahedron_index = self.tetrahedra.len();
        let mut seen = BTreeSet::new();
        for vertex in face {
            if vertex >= self.vertices.len() {
                return Err(SymmetrySubstrateError::VertexOutOfBounds {
                    tetrahedron: tetrahedron_index,
                    vertex,
                });
            }
            if !seen.insert(vertex) {
                return Err(SymmetrySubstrateError::RepeatedVertex {
                    tetrahedron: tetrahedron_index,
                    vertex,
                });
            }
        }
        if !target_edge_length.is_finite() || target_edge_length <= EPSILON {
            return Err(SymmetrySubstrateError::InvalidTargetLength);
        }
        let [a, b, c] = face.map(|index| self.vertices[index].position);
        if (b - a).cross(c - a).dot(new_position - a).abs() <= EPSILON {
            return Err(SymmetrySubstrateError::DegenerateTetrahedron {
                tetrahedron: tetrahedron_index,
            });
        }

        let new_vertex = self.vertices.len();
        self.vertices.push(SimplicialVertex {
            position: new_position,
            field: 0.0,
        });
        self.observations.push(0.0);
        self.observation_mask.push(false);
        let tetrahedron = Tetrahedron([face[0], face[1], face[2], new_vertex]);
        self.tetrahedra.push(tetrahedron);

        let mut existing = self
            .edges
            .iter()
            .map(|edge| ordered_pair(edge.a, edge.b))
            .collect::<BTreeSet<_>>();
        for i in 0..4 {
            for j in (i + 1)..4 {
                let edge = ordered_pair(tetrahedron.0[i], tetrahedron.0[j]);
                if existing.insert(edge) {
                    self.edges.push(SimplicialEdge {
                        a: edge.0,
                        b: edge.1,
                        target_length: target_edge_length,
                        weight: 1.0,
                    });
                }
            }
        }
        Ok(tetrahedron_index)
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn apply_stimulus(&mut self, vertex: usize, expected_field: f64) -> bool {
        if vertex >= self.vertices.len() || !expected_field.is_finite() {
            return false;
        }
        self.observations[vertex] = expected_field;
        self.observation_mask[vertex] = true;
        true
    }

    pub fn clear_stimuli(&mut self) {
        self.observation_mask.fill(false);
    }

    /// Proyección de Reynolds local para 0-cochains bajo la acción de S4.
    ///
    /// Promediar las 24 permutaciones equivale a asignar a los cuatro vértices
    /// la media del campo. La operación es exacta e idempotente. Para cochains
    /// orientadas de grado mayor se requiere una acción con signo, no este método.
    pub fn project_tetrahedral_field(&mut self, tetrahedron_index: usize) -> bool {
        let Some(tetrahedron) = self.tetrahedra.get(tetrahedron_index).copied() else {
            return false;
        };
        let mean = tetrahedron
            .0
            .iter()
            .map(|&vertex| self.vertices[vertex].field)
            .sum::<f64>()
            / 4.0;
        for vertex in tetrahedron.0 {
            self.vertices[vertex].field = mean;
        }
        true
    }

    /// Perturbación explícita para experimentos causales de ruptura/restauración.
    pub fn break_symmetry(&mut self, vertex: usize, displacement: Vec3, field_delta: f64) -> bool {
        let Some(target) = self.vertices.get_mut(vertex) else {
            return false;
        };
        if !field_delta.is_finite()
            || !displacement.x.is_finite()
            || !displacement.y.is_finite()
            || !displacement.z.is_finite()
        {
            return false;
        }
        target.position = target.position + displacement;
        target.field += field_delta;
        true
    }

    pub fn remove_edge(&mut self, a: usize, b: usize) -> bool {
        let key = ordered_pair(a, b);
        let before = self.edges.len();
        self.edges
            .retain(|edge| ordered_pair(edge.a, edge.b) != key);
        self.edges.len() != before
    }

    /// Restaura el 1-esqueleto requerido por todos los tetraedros.
    pub fn repair_simplicial_edges(&mut self) -> TopologyRepairReport {
        let expected = expected_edge_keys(&self.tetrahedra);
        let mut unique = BTreeMap::<(usize, usize), SimplicialEdge>::new();
        let mut removed_duplicates = 0;
        for edge in self.edges.drain(..) {
            let key = ordered_pair(edge.a, edge.b);
            if !expected.contains(&key) {
                continue;
            }
            if unique.insert(key, edge).is_some() {
                removed_duplicates += 1;
            }
        }
        let default_target = unique
            .values()
            .map(|edge| edge.target_length)
            .find(|length| *length > EPSILON)
            .unwrap_or(1.0);
        let mut inserted_edges = 0;
        for &(a, b) in &expected {
            unique.entry((a, b)).or_insert_with(|| {
                inserted_edges += 1;
                SimplicialEdge {
                    a,
                    b,
                    target_length: default_target,
                    weight: 1.0,
                }
            });
        }
        self.edges = unique.into_values().collect();
        TopologyRepairReport {
            inserted_edges,
            removed_duplicates,
        }
    }

    pub fn metrics(&self) -> SymmetryMetrics {
        metrics_for(
            &self.vertices,
            &self.edges,
            &self.tetrahedra,
            &self.observations,
            &self.observation_mask,
            self.config,
        )
    }

    /// Relajación adaptativa. Un paso sólo se acepta si no aumenta F.
    pub fn step(&mut self) -> SymmetryStepReport {
        let before = self.metrics();
        let (position_gradient, field_gradient) = self.gradients();
        let old_positions = self
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();
        let old_fields = self
            .vertices
            .iter()
            .map(|vertex| vertex.field)
            .collect::<Vec<_>>();

        let thermal_scale = 1.0 / (1.0 + self.config.temperature.max(0.0));
        let mut dt = self.config.dt * thermal_scale;
        let mut accepted = false;
        let mut after = before;
        for _ in 0..self.config.max_line_search_steps {
            for (index, vertex) in self.vertices.iter_mut().enumerate() {
                vertex.position = old_positions[index] - position_gradient[index] * dt;
                vertex.field = old_fields[index] - field_gradient[index] * dt;
            }
            let candidate = self.metrics();
            if candidate.free_energy <= before.free_energy + 1.0e-12 {
                accepted = true;
                after = candidate;
                break;
            }
            dt *= 0.5;
            if dt < self.config.min_dt {
                break;
            }
        }
        if !accepted {
            for (index, vertex) in self.vertices.iter_mut().enumerate() {
                vertex.position = old_positions[index];
                vertex.field = old_fields[index];
            }
            dt = 0.0;
        }
        self.tick += 1;
        self.config.temperature *= self.config.cooling_rate;
        SymmetryStepReport {
            tick: self.tick,
            accepted,
            accepted_dt: dt,
            before,
            after,
        }
    }

    pub fn equilibrate(&mut self, max_steps: usize, tolerance: f64) -> Vec<SymmetryStepReport> {
        let mut reports = Vec::with_capacity(max_steps);
        for _ in 0..max_steps {
            let report = self.step();
            let delta = (report.before.free_energy - report.after.free_energy).abs();
            reports.push(report);
            if delta <= tolerance.max(0.0) {
                break;
            }
        }
        reports
    }

    /// Relaja únicamente un frente de vértices; el resto actúa como frontera fija.
    pub fn equilibrate_vertices(
        &mut self,
        active_vertices: &[usize],
        max_steps: usize,
        tolerance: f64,
    ) -> Vec<SymmetryStepReport> {
        let mut active = vec![false; self.vertices.len()];
        for &vertex in active_vertices {
            if vertex < active.len() {
                active[vertex] = true;
            }
        }
        let mut reports = Vec::with_capacity(max_steps);
        for _ in 0..max_steps {
            let report = self.step_selected(&active);
            let delta = (report.before.free_energy - report.after.free_energy).abs();
            reports.push(report);
            if delta <= tolerance.max(0.0) {
                break;
            }
        }
        reports
    }

    fn step_selected(&mut self, active: &[bool]) -> SymmetryStepReport {
        let before = self.metrics();
        let (position_gradient, field_gradient) = self.gradients();
        let old_positions = self
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();
        let old_fields = self
            .vertices
            .iter()
            .map(|vertex| vertex.field)
            .collect::<Vec<_>>();
        let thermal_scale = 1.0 / (1.0 + self.config.temperature.max(0.0));
        let mut dt = self.config.dt * thermal_scale;
        let mut accepted = false;
        let mut after = before;
        for _ in 0..self.config.max_line_search_steps {
            for (index, vertex) in self.vertices.iter_mut().enumerate() {
                if active.get(index).copied().unwrap_or(false) {
                    vertex.position = old_positions[index] - position_gradient[index] * dt;
                    vertex.field = old_fields[index] - field_gradient[index] * dt;
                }
            }
            let candidate = self.metrics();
            if candidate.free_energy <= before.free_energy + 1.0e-12 {
                accepted = true;
                after = candidate;
                break;
            }
            dt *= 0.5;
            if dt < self.config.min_dt {
                break;
            }
        }
        if !accepted {
            for (index, vertex) in self.vertices.iter_mut().enumerate() {
                vertex.position = old_positions[index];
                vertex.field = old_fields[index];
            }
            dt = 0.0;
        }
        self.tick += 1;
        self.config.temperature *= self.config.cooling_rate;
        SymmetryStepReport {
            tick: self.tick,
            accepted,
            accepted_dt: dt,
            before,
            after,
        }
    }

    fn gradients(&self) -> (Vec<Vec3>, Vec<f64>) {
        let mut position_gradient = vec![Vec3::default(); self.vertices.len()];
        let mut field_gradient = vec![0.0; self.vertices.len()];
        let edge_norm = self.edges.len().max(1) as f64;

        for edge in &self.edges {
            let delta = self.vertices[edge.a].position - self.vertices[edge.b].position;
            let length = delta.norm().max(EPSILON);
            let target = edge.target_length.max(EPSILON);
            let geometry_factor =
                self.config.geometry_weight * 2.0 * edge.weight * (length - target)
                    / (edge_norm * target * target * length);
            let gradient = delta * geometry_factor;
            position_gradient[edge.a] = position_gradient[edge.a] + gradient;
            position_gradient[edge.b] = position_gradient[edge.b] - gradient;

            let field_delta = self.vertices[edge.a].field - self.vertices[edge.b].field;
            let hodge_gradient =
                self.config.hodge_weight * 2.0 * edge.weight * field_delta / edge_norm;
            field_gradient[edge.a] += hodge_gradient;
            field_gradient[edge.b] -= hodge_gradient;
        }

        let active_observations = self
            .observation_mask
            .iter()
            .filter(|active| **active)
            .count()
            .max(1) as f64;
        for (index, active) in self.observation_mask.iter().copied().enumerate() {
            if active {
                field_gradient[index] += self.config.prediction_weight
                    * 2.0
                    * (self.vertices[index].field - self.observations[index])
                    / active_observations;
            }
        }
        (position_gradient, field_gradient)
    }
}

fn sanitized_config(config: SymmetryThermodynamicConfig) -> SymmetryThermodynamicConfig {
    SymmetryThermodynamicConfig {
        dt: config.dt.abs().max(EPSILON),
        min_dt: config.min_dt.abs().max(EPSILON),
        temperature: config.temperature.max(0.0),
        cooling_rate: config.cooling_rate.clamp(0.0, 1.0),
        geometry_weight: config.geometry_weight.max(0.0),
        hodge_weight: config.hodge_weight.max(0.0),
        prediction_weight: config.prediction_weight.max(0.0),
        topology_weight: config.topology_weight.max(0.0),
        max_line_search_steps: config.max_line_search_steps.max(1),
    }
}

fn validate_tetrahedra(
    positions: &[Vec3],
    tetrahedra: &[Tetrahedron],
) -> Result<(), SymmetrySubstrateError> {
    for (tetrahedron_index, tetrahedron) in tetrahedra.iter().enumerate() {
        let mut seen = BTreeSet::new();
        for &vertex in &tetrahedron.0 {
            if vertex >= positions.len() {
                return Err(SymmetrySubstrateError::VertexOutOfBounds {
                    tetrahedron: tetrahedron_index,
                    vertex,
                });
            }
            if !seen.insert(vertex) {
                return Err(SymmetrySubstrateError::RepeatedVertex {
                    tetrahedron: tetrahedron_index,
                    vertex,
                });
            }
        }
        let [a, b, c, d] = tetrahedron.0.map(|index| positions[index]);
        let signed_six_volume = (b - a).cross(c - a).dot(d - a);
        if signed_six_volume.abs() <= EPSILON {
            return Err(SymmetrySubstrateError::DegenerateTetrahedron {
                tetrahedron: tetrahedron_index,
            });
        }
    }
    Ok(())
}

fn metrics_for(
    vertices: &[SimplicialVertex],
    edges: &[SimplicialEdge],
    tetrahedra: &[Tetrahedron],
    observations: &[f64],
    observation_mask: &[bool],
    config: SymmetryThermodynamicConfig,
) -> SymmetryMetrics {
    let expected = expected_edge_keys(tetrahedra);
    let actual = edges
        .iter()
        .map(|edge| ordered_pair(edge.a, edge.b))
        .collect::<BTreeSet<_>>();
    let topology_defect =
        expected.difference(&actual).count() as f64 / expected.len().max(1) as f64;

    let mut geometry_energy = 0.0;
    let mut hodge_energy = 0.0;
    let mut lengths = Vec::with_capacity(edges.len());
    for edge in edges {
        let length = (vertices[edge.a].position - vertices[edge.b].position).norm();
        lengths.push(length);
        geometry_energy +=
            edge.weight * ((length - edge.target_length) / edge.target_length.max(EPSILON)).powi(2);
        hodge_energy += edge.weight * (vertices[edge.a].field - vertices[edge.b].field).powi(2);
    }
    geometry_energy /= edges.len().max(1) as f64;
    hodge_energy /= edges.len().max(1) as f64;

    let active_count = observation_mask.iter().filter(|active| **active).count();
    let prediction_error = if active_count == 0 {
        0.0
    } else {
        observation_mask
            .iter()
            .enumerate()
            .filter(|(_, active)| **active)
            .map(|(index, _)| (vertices[index].field - observations[index]).powi(2))
            .sum::<f64>()
            / active_count as f64
    };

    let edge_length_cv = coefficient_of_variation(&lengths);
    let automorphism_residual = tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedral_automorphism_residual(vertices, *tetrahedron))
        .sum::<f64>()
        / tetrahedra.len().max(1) as f64;
    let internal_energy = config.geometry_weight * geometry_energy
        + config.hodge_weight * hodge_energy
        + config.prediction_weight * prediction_error
        + config.topology_weight * topology_defect;
    let entropy = normalized_gibbs_entropy(vertices, config.temperature);
    let free_energy = internal_energy - config.temperature * entropy;
    let symmetry_score = (-(automorphism_residual + topology_defect)).exp();

    SymmetryMetrics {
        symmetry_score,
        automorphism_residual,
        edge_length_cv,
        topology_defect,
        hodge_energy,
        prediction_error,
        internal_energy,
        entropy,
        free_energy,
    }
}

fn tetrahedral_automorphism_residual(
    vertices: &[SimplicialVertex],
    tetrahedron: Tetrahedron,
) -> f64 {
    let ids = tetrahedron.0;
    let mut distance = [[0.0; 4]; 4];
    let mut geometry_scale = 0.0;
    let mut field_scale = 0.0;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let squared = (vertices[ids[i]].position - vertices[ids[j]].position)
                .dot(vertices[ids[i]].position - vertices[ids[j]].position);
            distance[i][j] = squared;
            distance[j][i] = squared;
            geometry_scale += squared * squared;
            field_scale += (vertices[ids[i]].field - vertices[ids[j]].field).powi(2);
        }
    }
    geometry_scale = (geometry_scale / 6.0).max(EPSILON);
    field_scale = (field_scale / 6.0).max(EPSILON);

    let mut geometry_residual = 0.0;
    let mut field_residual = 0.0;
    for permutation in TETRAHEDRAL_PERMUTATIONS {
        for i in 0..4 {
            for j in (i + 1)..4 {
                geometry_residual += (distance[i][j] - distance[permutation[i]][permutation[j]])
                    .powi(2)
                    / geometry_scale;
                let original_field_delta = vertices[ids[i]].field - vertices[ids[j]].field;
                let transformed_field_delta =
                    vertices[ids[permutation[i]]].field - vertices[ids[permutation[j]]].field;
                field_residual +=
                    (original_field_delta - transformed_field_delta).powi(2) / field_scale;
            }
        }
    }
    let normalization = (TETRAHEDRAL_PERMUTATIONS.len() * 6) as f64;
    let geometry_residual = geometry_residual / normalization;
    let field_residual = if field_scale <= EPSILON {
        0.0
    } else {
        field_residual / normalization
    };
    0.75 * geometry_residual + 0.25 * field_residual
}

fn normalized_gibbs_entropy(vertices: &[SimplicialVertex], temperature: f64) -> f64 {
    if vertices.len() <= 1 {
        return 0.0;
    }
    let mean = vertices.iter().map(|vertex| vertex.field).sum::<f64>() / vertices.len() as f64;
    let scale = temperature.max(1.0e-6);
    let logits = vertices
        .iter()
        .map(|vertex| -(vertex.field - mean).powi(2) / scale)
        .collect::<Vec<_>>();
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let partition = logits
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .sum::<f64>()
        .max(EPSILON);
    let entropy = logits
        .iter()
        .map(|logit| (logit - max_logit).exp() / partition)
        .filter(|probability| *probability > EPSILON)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();
    entropy / (vertices.len() as f64).ln()
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean.abs().max(EPSILON)
}

fn expected_edge_keys(tetrahedra: &[Tetrahedron]) -> BTreeSet<(usize, usize)> {
    let mut edges = BTreeSet::new();
    for tetrahedron in tetrahedra {
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.insert(ordered_pair(tetrahedron.0[i], tetrahedron.0[j]));
            }
        }
    }
    edges
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

const TETRAHEDRAL_PERMUTATIONS: [[usize; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn cold_config() -> SymmetryThermodynamicConfig {
        SymmetryThermodynamicConfig {
            temperature: 0.0,
            dt: 0.12,
            ..SymmetryThermodynamicConfig::default()
        }
    }

    #[test]
    fn regular_tetrahedron_has_pure_local_symmetry() {
        let substrate =
            SymmetryThermodynamicSubstrate::regular_tetrahedron(1.0, cold_config()).unwrap();
        let metrics = substrate.metrics();
        assert!((metrics.symmetry_score - 1.0).abs() < 1.0e-12);
        assert!(metrics.automorphism_residual < 1.0e-24);
        assert!(metrics.edge_length_cv < 1.0e-12);
        assert!(metrics.internal_energy < 1.0e-24);
    }

    #[test]
    fn symmetry_measure_is_invariant_to_vertex_relabeling() {
        let mut original =
            SymmetryThermodynamicSubstrate::regular_tetrahedron(1.0, cold_config()).unwrap();
        original.break_symmetry(3, Vec3::new(0.23, -0.08, 0.14), 0.7);
        let permutation = [2, 0, 3, 1];
        let positions = permutation
            .iter()
            .map(|&old| original.vertices[old].position)
            .collect::<Vec<_>>();
        let mut relabeled = SymmetryThermodynamicSubstrate::new(
            positions,
            vec![Tetrahedron([0, 1, 2, 3])],
            1.0,
            cold_config(),
        )
        .unwrap();
        for (new, &old) in permutation.iter().enumerate() {
            relabeled.vertices[new].field = original.vertices[old].field;
        }
        let a = original.metrics();
        let b = relabeled.metrics();
        assert!((a.symmetry_score - b.symmetry_score).abs() < 1.0e-12);
        assert!((a.automorphism_residual - b.automorphism_residual).abs() < 1.0e-12);
    }

    #[test]
    fn perturbation_breaks_and_relaxation_restores_symmetry() {
        let mut substrate =
            SymmetryThermodynamicSubstrate::regular_tetrahedron(1.0, cold_config()).unwrap();
        substrate.break_symmetry(3, Vec3::new(0.45, -0.20, 0.30), 1.5);
        let broken = substrate.metrics();
        assert!(broken.symmetry_score < 0.9);
        assert!(broken.internal_energy > 0.1);

        let reports = substrate.equilibrate(600, 1.0e-14);
        assert!(reports
            .iter()
            .all(|report| { report.after.free_energy <= report.before.free_energy + 1.0e-10 }));
        let restored = substrate.metrics();
        assert!(restored.internal_energy < broken.internal_energy * 1.0e-4);
        assert!(restored.symmetry_score > broken.symmetry_score);
        assert!(restored.symmetry_score > 0.999);
    }

    #[test]
    fn local_stimulus_creates_prediction_then_clearing_it_diffuses_field() {
        let mut substrate =
            SymmetryThermodynamicSubstrate::regular_tetrahedron(1.0, cold_config()).unwrap();
        assert!(substrate.apply_stimulus(0, 2.0));
        substrate.equilibrate(300, 1.0e-12);
        let stimulated = substrate.metrics();
        assert!(substrate.vertices[0].field > substrate.vertices[1].field);
        assert!(stimulated.hodge_energy > 0.0);

        substrate.clear_stimuli();
        substrate.equilibrate(600, 1.0e-14);
        let consolidated = substrate.metrics();
        assert!(consolidated.hodge_energy < stimulated.hodge_energy * 1.0e-4);
        assert!(consolidated.symmetry_score > stimulated.symmetry_score);
    }

    #[test]
    fn topology_repair_restores_missing_simplicial_edge() {
        let mut substrate =
            SymmetryThermodynamicSubstrate::regular_tetrahedron(1.0, cold_config()).unwrap();
        assert!(substrate.remove_edge(0, 1));
        assert!(substrate.metrics().topology_defect > 0.0);
        let report = substrate.repair_simplicial_edges();
        assert_eq!(report.inserted_edges, 1);
        assert_eq!(substrate.edges.len(), 6);
        assert_eq!(substrate.metrics().topology_defect, 0.0);
    }

    #[test]
    fn reynolds_projection_is_exact_and_idempotent_for_vertex_fields() {
        let mut substrate =
            SymmetryThermodynamicSubstrate::regular_tetrahedron(1.0, cold_config()).unwrap();
        for (vertex, field) in substrate.vertices.iter_mut().zip([1.0, -2.0, 4.0, 5.0]) {
            vertex.field = field;
        }
        assert!(substrate.project_tetrahedral_field(0));
        let once = substrate
            .vertices
            .iter()
            .map(|vertex| vertex.field)
            .collect::<Vec<_>>();
        assert!(once.iter().all(|field| (*field - 2.0).abs() < EPSILON));
        assert!(substrate.project_tetrahedral_field(0));
        let twice = substrate
            .vertices
            .iter()
            .map(|vertex| vertex.field)
            .collect::<Vec<_>>();
        assert_eq!(once, twice);
        assert!(substrate.metrics().hodge_energy < EPSILON);
    }

    #[test]
    fn rejects_degenerate_tetrahedron() {
        let result = SymmetryThermodynamicSubstrate::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            vec![Tetrahedron([0, 1, 2, 3])],
            1.0,
            cold_config(),
        );
        assert!(matches!(
            result,
            Err(SymmetrySubstrateError::DegenerateTetrahedron { .. })
        ));
    }
}
