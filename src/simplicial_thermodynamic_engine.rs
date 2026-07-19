//! Motor consolidado: complejo simplicial 3D → Hodge/simetría → RQM → EPR → conceptos.
//!
//! Esta arquitectura es paralela al motor CDT legacy. La geometría 3D no es una
//! visualización: aristas, caras y tetraedros son entidades topológicas reales.

use crate::entanglement::{EntanglementConfig, EntanglementField};
use crate::symmetry_thermodynamic_substrate::{
    SimplicialEdge, SymmetryMetrics, SymmetryStepReport, SymmetryThermodynamicSubstrate, Vec3,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OrientedTriangle(pub [usize; 3]);

#[derive(Clone, Debug)]
pub struct SimplicialHodgeOperators {
    pub triangles: Vec<OrientedTriangle>,
    edge_index: BTreeMap<(usize, usize), usize>,
    triangle_index: BTreeMap<[usize; 3], usize>,
    tetrahedra: Vec<[usize; 4]>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HodgeReport {
    pub l0_energy: f64,
    pub l1_energy: f64,
    pub l1_divergence_energy: f64,
    pub l1_curl_energy: f64,
    pub l2_energy: f64,
    pub l2_boundary_energy: f64,
    pub l2_volume_energy: f64,
    pub chain_complex_valid: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReggeReport {
    pub interior_edges: usize,
    pub mean_abs_deficit: f64,
    pub rms_deficit: f64,
    pub action: f64,
}

impl SimplicialHodgeOperators {
    pub fn from_substrate(substrate: &SymmetryThermodynamicSubstrate) -> Self {
        let edge_index = substrate
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| (ordered_pair(edge.a, edge.b), index))
            .collect::<BTreeMap<_, _>>();
        let mut triangle_keys = BTreeSet::new();
        let mut tetrahedra = Vec::with_capacity(substrate.tetrahedra.len());
        for tetrahedron in &substrate.tetrahedra {
            let mut vertices = tetrahedron.0;
            vertices.sort_unstable();
            tetrahedra.push(vertices);
            for omitted in 0..4 {
                let mut face = Vec::with_capacity(3);
                for (index, vertex) in vertices.iter().copied().enumerate() {
                    if index != omitted {
                        face.push(vertex);
                    }
                }
                triangle_keys.insert([face[0], face[1], face[2]]);
            }
        }
        let triangles = triangle_keys
            .iter()
            .copied()
            .map(OrientedTriangle)
            .collect::<Vec<_>>();
        let triangle_index = triangle_keys
            .into_iter()
            .enumerate()
            .map(|(index, triangle)| (triangle, index))
            .collect();
        Self {
            triangles,
            edge_index,
            triangle_index,
            tetrahedra,
        }
    }

    pub fn edge_count(&self) -> usize {
        self.edge_index.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn tetrahedron_count(&self) -> usize {
        self.tetrahedra.len()
    }

    /// Verifica exactamente B1·B2=0 y B2·B3=0 con coeficientes enteros.
    pub fn chain_complex_valid(&self) -> bool {
        for triangle in &self.triangles {
            let mut vertex_boundary = BTreeMap::<usize, i32>::new();
            for (edge, sign) in triangle_boundary(triangle.0) {
                *vertex_boundary.entry(edge.0).or_default() -= sign;
                *vertex_boundary.entry(edge.1).or_default() += sign;
            }
            if vertex_boundary
                .values()
                .any(|coefficient| *coefficient != 0)
            {
                return false;
            }
        }
        for tetrahedron in &self.tetrahedra {
            let mut edge_boundary = BTreeMap::<(usize, usize), i32>::new();
            for (face, face_sign) in tetrahedron_boundary(*tetrahedron) {
                for (edge, edge_sign) in triangle_boundary(face) {
                    *edge_boundary.entry(edge).or_default() += face_sign * edge_sign;
                }
            }
            if edge_boundary.values().any(|coefficient| *coefficient != 0) {
                return false;
            }
        }
        true
    }

    pub fn vertex_gradient(&self, vertex_field: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; self.edge_count()];
        for (&(a, b), &edge) in &self.edge_index {
            if a < vertex_field.len() && b < vertex_field.len() {
                gradient[edge] = vertex_field[b] - vertex_field[a];
            }
        }
        gradient
    }

    pub fn edge_curl(&self, edge_flow: &[f64]) -> Vec<f64> {
        self.triangles
            .iter()
            .map(|triangle| {
                triangle_boundary(triangle.0)
                    .into_iter()
                    .map(|(edge, sign)| {
                        self.edge_index
                            .get(&edge)
                            .and_then(|index| edge_flow.get(*index))
                            .copied()
                            .unwrap_or(0.0)
                            * sign as f64
                    })
                    .sum()
            })
            .collect()
    }

    pub fn edge_divergence(&self, edge_flow: &[f64], vertex_count: usize) -> Vec<f64> {
        let mut divergence = vec![0.0; vertex_count];
        for (&(a, b), &edge) in &self.edge_index {
            let value = edge_flow.get(edge).copied().unwrap_or(0.0);
            divergence[a] -= value;
            divergence[b] += value;
        }
        divergence
    }

    pub fn face_boundary(&self, face_flux: &[f64]) -> Vec<f64> {
        let mut boundary = vec![0.0; self.edge_count()];
        for (face_index, triangle) in self.triangles.iter().enumerate() {
            let value = face_flux.get(face_index).copied().unwrap_or(0.0);
            for (edge, sign) in triangle_boundary(triangle.0) {
                if let Some(&edge_index) = self.edge_index.get(&edge) {
                    boundary[edge_index] += sign as f64 * value;
                }
            }
        }
        boundary
    }

    pub fn volume_coboundary(&self, face_flux: &[f64]) -> Vec<f64> {
        self.tetrahedra
            .iter()
            .map(|tetrahedron| {
                tetrahedron_boundary(*tetrahedron)
                    .into_iter()
                    .map(|(face, sign)| {
                        self.triangle_index
                            .get(&face)
                            .and_then(|index| face_flux.get(*index))
                            .copied()
                            .unwrap_or(0.0)
                            * sign as f64
                    })
                    .sum()
            })
            .collect()
    }

    pub fn hodge_report(
        &self,
        vertex_field: &[f64],
        edge_flow: &[f64],
        face_flux: &[f64],
    ) -> HodgeReport {
        let l0_energy = squared_norm(&self.vertex_gradient(vertex_field));
        let divergence_energy = squared_norm(&self.edge_divergence(edge_flow, vertex_field.len()));
        let curl_energy = squared_norm(&self.edge_curl(edge_flow));
        let boundary_energy = squared_norm(&self.face_boundary(face_flux));
        let volume_energy = squared_norm(&self.volume_coboundary(face_flux));
        HodgeReport {
            l0_energy,
            l1_energy: divergence_energy + curl_energy,
            l1_divergence_energy: divergence_energy,
            l1_curl_energy: curl_energy,
            l2_energy: boundary_energy + volume_energy,
            l2_boundary_energy: boundary_energy,
            l2_volume_energy: volume_energy,
            chain_complex_valid: self.chain_complex_valid(),
        }
    }

    pub fn regge_report(&self, substrate: &SymmetryThermodynamicSubstrate) -> ReggeReport {
        let mut face_incidence = BTreeMap::<[usize; 3], usize>::new();
        let mut edge_tetrahedra = BTreeMap::<(usize, usize), Vec<[usize; 4]>>::new();
        for &tetrahedron in &self.tetrahedra {
            for (face, _) in tetrahedron_boundary(tetrahedron) {
                *face_incidence.entry(face).or_default() += 1;
            }
            for i in 0..4 {
                for j in (i + 1)..4 {
                    edge_tetrahedra
                        .entry(ordered_pair(tetrahedron[i], tetrahedron[j]))
                        .or_default()
                        .push(tetrahedron);
                }
            }
        }

        let mut deficits = Vec::new();
        let mut action = 0.0;
        for (edge, incident_tetrahedra) in edge_tetrahedra {
            let touches_boundary = face_incidence.iter().any(|(face, count)| {
                *count == 1 && face.contains(&edge.0) && face.contains(&edge.1)
            });
            if touches_boundary {
                continue;
            }
            let mut angle_sum = 0.0;
            let mut valid = true;
            for tetrahedron in incident_tetrahedra {
                let opposite = tetrahedron
                    .into_iter()
                    .filter(|vertex| *vertex != edge.0 && *vertex != edge.1)
                    .collect::<Vec<_>>();
                let Some(angle) = dihedral_angle(
                    substrate.vertices[edge.0].position,
                    substrate.vertices[edge.1].position,
                    substrate.vertices[opposite[0]].position,
                    substrate.vertices[opposite[1]].position,
                ) else {
                    valid = false;
                    break;
                };
                angle_sum += angle;
            }
            if !valid {
                continue;
            }
            let deficit = std::f64::consts::TAU - angle_sum;
            let length =
                (substrate.vertices[edge.0].position - substrate.vertices[edge.1].position).norm();
            action += length * deficit;
            deficits.push(deficit);
        }
        let count = deficits.len();
        ReggeReport {
            interior_edges: count,
            mean_abs_deficit: deficits.iter().map(|value| value.abs()).sum::<f64>()
                / count.max(1) as f64,
            rms_deficit: (deficits.iter().map(|value| value * value).sum::<f64>()
                / count.max(1) as f64)
                .sqrt(),
            action,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SimplicialEngineConfig {
    pub relation_learning_rate: f64,
    pub concept_symmetry_threshold: f64,
    pub temporal_history: usize,
    pub entanglement: EntanglementConfig,
}

impl Default for SimplicialEngineConfig {
    fn default() -> Self {
        Self {
            relation_learning_rate: 0.25,
            concept_symmetry_threshold: 0.95,
            temporal_history: 128,
            entanglement: EntanglementConfig {
                create_threshold: 0.75,
                ..EntanglementConfig::default()
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimplicialRelation {
    pub weight: f64,
    pub coherence: f64,
    pub observations: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimplicialCandidate {
    pub vertex: usize,
    pub score: f64,
    pub relational_score: f64,
    pub epr_bonus: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymmetryConcept {
    pub id: usize,
    pub label: String,
    pub vertices: Vec<usize>,
    pub invariant_field: f64,
    pub symmetry_score: f64,
    pub regge_action: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometricConceptPrototype {
    pub id: usize,
    pub label: String,
    /// Firma canónica de las seis longitudes, invariante a escala y re-etiquetado.
    pub signature: [f64; 6],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShapeRecognition {
    pub concept_id: usize,
    pub label: String,
    pub distance: f64,
    pub confidence: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SymmetryRepairReport {
    pub before: SymmetryMetrics,
    pub after: SymmetryMetrics,
    pub inserted_edges: usize,
    pub removed_duplicates: usize,
    pub relaxation_steps: usize,
    pub chain_complex_valid: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SymmetryPropagationFrame {
    pub hop: usize,
    pub visited_tetrahedra: usize,
    pub active_vertices: usize,
    pub inserted_edges: usize,
    pub recognized_tetrahedra: usize,
    pub symmetry: SymmetryMetrics,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymmetryPropagationReport {
    pub baseline_recognized: usize,
    pub final_recognized: usize,
    pub total_tetrahedra: usize,
    pub frames: Vec<SymmetryPropagationFrame>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimplicialTemporalFrame {
    pub tick: u64,
    pub symmetry: SymmetryMetrics,
    pub hodge: HodgeReport,
    pub regge: ReggeReport,
    pub relations: usize,
    pub epr_links: usize,
    pub concepts: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimplicialCognitiveCycleReport {
    pub frame: SimplicialTemporalFrame,
    pub relation: SimplicialRelation,
    pub epr_created: bool,
    pub relaxation_steps: usize,
}

#[derive(Clone, Debug)]
pub struct SimplicialThermodynamicEngine {
    pub substrate: SymmetryThermodynamicSubstrate,
    pub hodge: SimplicialHodgeOperators,
    pub edge_flow: Vec<f64>,
    pub face_flux: Vec<f64>,
    pub entanglement: EntanglementField,
    pub concepts: Vec<SymmetryConcept>,
    pub geometric_concepts: Vec<GeometricConceptPrototype>,
    pub config: SimplicialEngineConfig,
    relations: BTreeMap<(usize, usize), SimplicialRelation>,
    temporal_frames: VecDeque<SimplicialTemporalFrame>,
    tick: u64,
}

impl SimplicialThermodynamicEngine {
    pub fn new(substrate: SymmetryThermodynamicSubstrate, config: SimplicialEngineConfig) -> Self {
        let hodge = SimplicialHodgeOperators::from_substrate(&substrate);
        Self {
            edge_flow: vec![0.0; hodge.edge_count()],
            face_flux: vec![0.0; hodge.triangle_count()],
            entanglement: EntanglementField::new(config.entanglement),
            substrate,
            hodge,
            concepts: Vec::new(),
            geometric_concepts: Vec::new(),
            config,
            relations: BTreeMap::new(),
            temporal_frames: VecDeque::new(),
            tick: 0,
        }
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn temporal_frames(&self) -> &VecDeque<SimplicialTemporalFrame> {
        &self.temporal_frames
    }

    pub fn observe_transition(
        &mut self,
        source: usize,
        target: usize,
        evidence: f64,
    ) -> Option<(SimplicialRelation, bool)> {
        if source >= self.substrate.vertices.len()
            || target >= self.substrate.vertices.len()
            || source == target
            || !evidence.is_finite()
        {
            return None;
        }
        let evidence = evidence.clamp(0.0, 1.0);
        let relation = self.relations.entry((source, target)).or_default();
        relation.weight += self.config.relation_learning_rate * evidence * (1.0 - relation.weight);
        relation.coherence += self.config.relation_learning_rate
            * self.substrate.metrics().symmetry_score
            * (1.0 - relation.coherence);
        relation.observations += 1;

        if let Some(&edge) = self.hodge.edge_index.get(&ordered_pair(source, target)) {
            let direction = if source < target { 1.0 } else { -1.0 };
            self.edge_flow[edge] = 0.8 * self.edge_flow[edge] + 0.2 * direction * relation.weight;
        }
        let epr_created = self
            .entanglement
            .observe_correlation(source, target, evidence as f32);
        Some((*relation, epr_created))
    }

    pub fn query(&self, cue: usize) -> Vec<SimplicialCandidate> {
        let mut candidates = self
            .relations
            .iter()
            .filter(|((source, _), _)| *source == cue)
            .map(|((_, target), relation)| {
                let relational_score = relation.weight * relation.coherence;
                let epr_bonus = if self.entanglement.has_active_link(cue, *target) {
                    0.25
                } else {
                    0.0
                };
                let field_bonus = self.substrate.vertices[*target].field.abs().min(1.0) * 0.05;
                SimplicialCandidate {
                    vertex: *target,
                    score: relational_score + epr_bonus + field_bonus,
                    relational_score,
                    epr_bonus,
                }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.vertex.cmp(&right.vertex))
        });
        candidates
    }

    pub fn form_concept(
        &mut self,
        label: impl Into<String>,
        vertices: &[usize],
    ) -> Option<SymmetryConcept> {
        if vertices.is_empty()
            || vertices
                .iter()
                .any(|vertex| *vertex >= self.substrate.vertices.len())
        {
            return None;
        }
        let symmetry = self.substrate.metrics().symmetry_score;
        if symmetry < self.config.concept_symmetry_threshold {
            return None;
        }
        let invariant_field = vertices
            .iter()
            .map(|vertex| self.substrate.vertices[*vertex].field)
            .sum::<f64>()
            / vertices.len() as f64;
        let concept = SymmetryConcept {
            id: self.concepts.len(),
            label: label.into(),
            vertices: vertices.to_vec(),
            invariant_field,
            symmetry_score: symmetry,
            regge_action: self.hodge.regge_report(&self.substrate).action,
        };
        self.concepts.push(concept.clone());
        Some(concept)
    }

    /// Aprende una forma sin usar `symmetry_score` como etiqueta.
    ///
    /// La salida cognitiva del experimento es el reconocimiento posterior de
    /// esta firma; así se evita definir circularmente cognición como simetría.
    pub fn learn_geometric_concept(
        &mut self,
        label: impl Into<String>,
        tetrahedron_index: usize,
    ) -> Option<GeometricConceptPrototype> {
        let signature = geometric_signature(&self.substrate, tetrahedron_index)?;
        let concept = GeometricConceptPrototype {
            id: self.geometric_concepts.len(),
            label: label.into(),
            signature,
        };
        self.geometric_concepts.push(concept.clone());
        Some(concept)
    }

    pub fn recognize_geometric_concept(
        &self,
        tetrahedron_index: usize,
        max_distance: f64,
    ) -> Option<ShapeRecognition> {
        let signature = geometric_signature(&self.substrate, tetrahedron_index)?;
        let (concept, distance) = self
            .geometric_concepts
            .iter()
            .map(|concept| {
                let squared_distance = concept
                    .signature
                    .iter()
                    .zip(signature)
                    .map(|(expected, observed)| (expected - observed).powi(2))
                    .sum::<f64>();
                let distance = (squared_distance / 6.0).sqrt();
                (concept, distance)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))?;
        if distance > max_distance.max(0.0) {
            return None;
        }
        Some(ShapeRecognition {
            concept_id: concept.id,
            label: concept.label.clone(),
            distance,
            confidence: (-distance).exp(),
        })
    }

    /// Repara cierre simplicial y relaja geometría/campo con gate energético.
    pub fn homeostatic_symmetry_repair(
        &mut self,
        max_steps: usize,
        tolerance: f64,
    ) -> SymmetryRepairReport {
        let before = self.substrate.metrics();
        let topology = self.substrate.repair_simplicial_edges();
        let reports = self.substrate.equilibrate(max_steps, tolerance);
        self.hodge = SimplicialHodgeOperators::from_substrate(&self.substrate);
        self.edge_flow.resize(self.hodge.edge_count(), 0.0);
        self.face_flux.resize(self.hodge.triangle_count(), 0.0);
        self.tick += reports.len() as u64;
        let after = self.substrate.metrics();
        self.capture_frame();
        SymmetryRepairReport {
            before,
            after,
            inserted_edges: topology.inserted_edges,
            removed_duplicates: topology.removed_duplicates,
            relaxation_steps: reports.len(),
            chain_complex_valid: self.hodge.chain_complex_valid(),
        }
    }

    /// Propaga un frente de restauración por tetraedros que comparten una cara.
    pub fn propagate_symmetry_from(
        &mut self,
        seed_vertices: &[usize],
        concept_id: usize,
        max_hops: usize,
        steps_per_hop: usize,
        tolerance: f64,
        recognition_tolerance: f64,
    ) -> SymmetryPropagationReport {
        let tetrahedron_count = self.substrate.tetrahedra.len();
        let mut distance = vec![usize::MAX; tetrahedron_count];
        let mut queue = VecDeque::new();
        for (index, tetrahedron) in self.substrate.tetrahedra.iter().enumerate() {
            if seed_vertices
                .iter()
                .any(|seed| tetrahedron.0.contains(seed))
            {
                distance[index] = 0;
                queue.push_back(index);
            }
        }
        while let Some(current) = queue.pop_front() {
            if distance[current] >= max_hops {
                continue;
            }
            for candidate in 0..tetrahedron_count {
                if distance[candidate] != usize::MAX || candidate == current {
                    continue;
                }
                let shared = self.substrate.tetrahedra[current]
                    .0
                    .iter()
                    .filter(|vertex| self.substrate.tetrahedra[candidate].0.contains(vertex))
                    .count();
                if shared >= 3 {
                    distance[candidate] = distance[current] + 1;
                    queue.push_back(candidate);
                }
            }
        }

        let baseline_recognized = self.recognized_tetrahedra(concept_id, recognition_tolerance);
        let mut frames = Vec::new();
        for hop in 0..=max_hops {
            let layer = distance
                .iter()
                .enumerate()
                .filter(|(_, distance)| **distance == hop)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if layer.is_empty() {
                continue;
            }
            let inserted_edges = layer
                .iter()
                .map(|tetrahedron| self.repair_tetrahedron_edges(*tetrahedron))
                .sum();
            let active_tetrahedra = distance
                .iter()
                .enumerate()
                .filter(|(_, distance)| **distance <= hop)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let active_vertices = active_tetrahedra
                .iter()
                .flat_map(|tetrahedron| self.substrate.tetrahedra[*tetrahedron].0)
                .collect::<BTreeSet<_>>();
            self.substrate.equilibrate_vertices(
                &active_vertices.iter().copied().collect::<Vec<_>>(),
                steps_per_hop,
                tolerance,
            );
            self.hodge = SimplicialHodgeOperators::from_substrate(&self.substrate);
            self.edge_flow.resize(self.hodge.edge_count(), 0.0);
            self.face_flux.resize(self.hodge.triangle_count(), 0.0);
            frames.push(SymmetryPropagationFrame {
                hop,
                visited_tetrahedra: active_tetrahedra.len(),
                active_vertices: active_vertices.len(),
                inserted_edges,
                recognized_tetrahedra: self
                    .recognized_tetrahedra(concept_id, recognition_tolerance),
                symmetry: self.substrate.metrics(),
            });
        }
        self.tick += frames.len() as u64;
        self.capture_frame();
        SymmetryPropagationReport {
            baseline_recognized,
            final_recognized: self.recognized_tetrahedra(concept_id, recognition_tolerance),
            total_tetrahedra: tetrahedron_count,
            frames,
        }
    }

    fn recognized_tetrahedra(&self, concept_id: usize, tolerance: f64) -> usize {
        (0..self.substrate.tetrahedra.len())
            .filter(|tetrahedron| {
                self.recognize_geometric_concept(*tetrahedron, tolerance)
                    .is_some_and(|recognition| recognition.concept_id == concept_id)
            })
            .count()
    }

    fn repair_tetrahedron_edges(&mut self, tetrahedron_index: usize) -> usize {
        let Some(tetrahedron) = self.substrate.tetrahedra.get(tetrahedron_index).copied() else {
            return 0;
        };
        let target_length = self
            .substrate
            .edges
            .first()
            .map_or(1.0, |edge| edge.target_length);
        let mut existing = self
            .substrate
            .edges
            .iter()
            .map(|edge| ordered_pair(edge.a, edge.b))
            .collect::<BTreeSet<_>>();
        let mut inserted = 0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let edge = ordered_pair(tetrahedron.0[i], tetrahedron.0[j]);
                if existing.insert(edge) {
                    self.substrate.edges.push(SimplicialEdge {
                        a: edge.0,
                        b: edge.1,
                        target_length,
                        weight: 1.0,
                    });
                    inserted += 1;
                }
            }
        }
        inserted
    }

    pub fn cognitive_cycle(
        &mut self,
        cue: usize,
        target: usize,
        evidence: f64,
        relaxation_steps: usize,
    ) -> Option<SimplicialCognitiveCycleReport> {
        self.substrate.apply_stimulus(cue, evidence);
        let (relation, epr_created) = self.observe_transition(cue, target, evidence)?;
        let reports = self.substrate.equilibrate(relaxation_steps, 1.0e-12);
        self.substrate.clear_stimuli();
        self.tick += 1;
        let frame = self.capture_frame();
        Some(SimplicialCognitiveCycleReport {
            frame,
            relation,
            epr_created,
            relaxation_steps: reports.len(),
        })
    }

    pub fn relax(&mut self, max_steps: usize, tolerance: f64) -> Vec<SymmetryStepReport> {
        let reports = self.substrate.equilibrate(max_steps, tolerance);
        self.tick += reports.len() as u64;
        self.capture_frame();
        reports
    }

    pub fn capture_frame(&mut self) -> SimplicialTemporalFrame {
        let vertex_field = self
            .substrate
            .vertices
            .iter()
            .map(|vertex| vertex.field)
            .collect::<Vec<_>>();
        let frame = SimplicialTemporalFrame {
            tick: self.tick,
            symmetry: self.substrate.metrics(),
            hodge: self
                .hodge
                .hodge_report(&vertex_field, &self.edge_flow, &self.face_flux),
            regge: self.hodge.regge_report(&self.substrate),
            relations: self.relations.len(),
            epr_links: self.entanglement.active_count(),
            concepts: self.concepts.len(),
        };
        self.temporal_frames.push_back(frame);
        while self.temporal_frames.len() > self.config.temporal_history.max(1) {
            self.temporal_frames.pop_front();
        }
        frame
    }
}

fn geometric_signature(
    substrate: &SymmetryThermodynamicSubstrate,
    tetrahedron_index: usize,
) -> Option<[f64; 6]> {
    let tetrahedron = substrate.tetrahedra.get(tetrahedron_index)?.0;
    let actual_edges = substrate
        .edges
        .iter()
        .map(|edge| ordered_pair(edge.a, edge.b))
        .collect::<BTreeSet<_>>();
    for i in 0..4 {
        for j in (i + 1)..4 {
            if !actual_edges.contains(&ordered_pair(tetrahedron[i], tetrahedron[j])) {
                return None;
            }
        }
    }

    let mut signature = [0.0; 6];
    let mut cursor = 0;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let distance = (substrate.vertices[tetrahedron[i]].position
                - substrate.vertices[tetrahedron[j]].position)
                .norm();
            signature[cursor] = distance;
            cursor += 1;
        }
    }
    let mean = signature.iter().sum::<f64>() / 6.0;
    if mean <= EPSILON {
        return None;
    }
    for length in &mut signature {
        *length /= mean;
    }
    signature.sort_by(f64::total_cmp);
    Some(signature)
}

fn triangle_boundary([a, b, c]: [usize; 3]) -> [((usize, usize), i32); 3] {
    [
        (ordered_pair(b, c), 1),
        (ordered_pair(a, c), -1),
        (ordered_pair(a, b), 1),
    ]
}

fn tetrahedron_boundary([a, b, c, d]: [usize; 4]) -> [([usize; 3], i32); 4] {
    [
        ([b, c, d], 1),
        ([a, c, d], -1),
        ([a, b, d], 1),
        ([a, b, c], -1),
    ]
}

fn dihedral_angle(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Option<f64> {
    let edge = b - a;
    let edge_norm = edge.norm();
    if edge_norm <= EPSILON {
        return None;
    }
    let axis = edge / edge_norm;
    let c_perpendicular = (c - a) - axis * (c - a).dot(axis);
    let d_perpendicular = (d - a) - axis * (d - a).dot(axis);
    let denominator = c_perpendicular.norm() * d_perpendicular.norm();
    if denominator <= EPSILON {
        return None;
    }
    Some(
        (c_perpendicular.dot(d_perpendicular) / denominator)
            .clamp(-1.0, 1.0)
            .acos(),
    )
}

fn squared_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
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
    use crate::symmetry_thermodynamic_substrate::{
        SymmetryThermodynamicConfig, SymmetryThermodynamicSubstrate, Tetrahedron,
    };

    fn engine() -> SimplicialThermodynamicEngine {
        let substrate = SymmetryThermodynamicSubstrate::regular_tetrahedron(
            1.0,
            SymmetryThermodynamicConfig {
                temperature: 0.0,
                ..SymmetryThermodynamicConfig::default()
            },
        )
        .unwrap();
        SimplicialThermodynamicEngine::new(substrate, SimplicialEngineConfig::default())
    }

    fn tetrahedral_chain() -> SimplicialThermodynamicEngine {
        let base = SymmetryThermodynamicSubstrate::regular_tetrahedron(
            1.0,
            SymmetryThermodynamicConfig {
                temperature: 0.0,
                dt: 0.12,
                ..SymmetryThermodynamicConfig::default()
            },
        )
        .unwrap();
        let mut positions = base
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();
        let p4 = reflect_across_face(positions[0], positions[1], positions[2], positions[3]);
        positions.push(p4);
        let p5 = reflect_across_face(positions[1], positions[4], positions[2], positions[3]);
        positions.push(p5);
        let substrate = SymmetryThermodynamicSubstrate::new(
            positions,
            vec![
                Tetrahedron([0, 1, 2, 3]),
                Tetrahedron([4, 1, 2, 3]),
                Tetrahedron([4, 5, 2, 3]),
            ],
            1.0,
            SymmetryThermodynamicConfig {
                temperature: 0.0,
                dt: 0.12,
                ..SymmetryThermodynamicConfig::default()
            },
        )
        .unwrap();
        SimplicialThermodynamicEngine::new(substrate, SimplicialEngineConfig::default())
    }

    fn reflect_across_face(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        let normal = (b - a).cross(c - a);
        let unit = normal / normal.norm();
        point - unit * (2.0 * (point - a).dot(unit))
    }

    #[test]
    fn boundary_operators_form_an_exact_chain_complex() {
        let engine = engine();
        assert_eq!(engine.hodge.edge_count(), 6);
        assert_eq!(engine.hodge.triangle_count(), 4);
        assert_eq!(engine.hodge.tetrahedron_count(), 1);
        assert!(engine.hodge.chain_complex_valid());
    }

    #[test]
    fn gradient_fields_have_zero_discrete_curl() {
        let engine = engine();
        let gradient = engine.hodge.vertex_gradient(&[0.3, -1.2, 2.0, 0.7]);
        let curl = engine.hodge.edge_curl(&gradient);
        assert!(squared_norm(&curl) < 1.0e-24);
    }

    #[test]
    fn boundary_faces_have_zero_volume_coboundary() {
        let engine = engine();
        let edge_flow = [0.2, -0.7, 0.5, 1.1, -0.3, 0.9];
        let exact_face_flux = engine.hodge.edge_curl(&edge_flow);
        let volume = engine.hodge.volume_coboundary(&exact_face_flux);
        assert!(squared_norm(&volume) < 1.0e-24);
    }

    #[test]
    fn regge_excludes_boundary_edges_of_single_tetrahedron() {
        let engine = engine();
        let report = engine.hodge.regge_report(&engine.substrate);
        assert_eq!(report.interior_edges, 0);
        assert_eq!(report.action, 0.0);
    }

    #[test]
    fn rqm_epr_and_concept_layers_are_operational() {
        let mut engine = engine();
        let first = engine.observe_transition(0, 1, 1.0).unwrap();
        let second = engine.observe_transition(0, 1, 1.0).unwrap();
        let third = engine.observe_transition(0, 1, 1.0).unwrap();
        assert!(second.0.weight > first.0.weight);
        assert!(first.1 || second.1 || third.1);
        let candidates = engine.query(0);
        assert_eq!(candidates[0].vertex, 1);
        assert!(candidates[0].epr_bonus > 0.0);
        let concept = engine.form_concept("tetraedro_regular", &[0, 1, 2, 3]);
        assert!(concept.is_some());
    }

    #[test]
    fn temporal_layer_keeps_a_bounded_history() {
        let mut engine = engine();
        engine.config.temporal_history = 2;
        engine.capture_frame();
        engine.capture_frame();
        engine.capture_frame();
        assert_eq!(engine.temporal_frames().len(), 2);
    }

    #[test]
    fn geometric_recognition_is_independent_of_rigid_motion_and_relabeling() {
        let mut engine = engine();
        engine
            .substrate
            .break_symmetry(3, Vec3::new(0.12, -0.04, 0.08), 0.0);
        let concept = engine.learn_geometric_concept("forma_a", 0).unwrap();
        let original = engine
            .substrate
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();
        let permutation = [2, 0, 3, 1];
        for (vertex, old) in engine.substrate.vertices.iter_mut().zip(permutation) {
            let position = original[old];
            vertex.position = Vec3::new(-position.y + 10.0, position.x - 3.0, position.z + 2.0);
        }
        let recognition = engine.recognize_geometric_concept(0, 1.0e-10).unwrap();
        assert_eq!(recognition.concept_id, concept.id);
        assert!(recognition.distance < 1.0e-12);
    }

    #[test]
    fn lesion_blocks_recognition_and_symmetry_repair_recovers_it() {
        let mut engine = engine();
        engine
            .learn_geometric_concept("tetraedro_regular", 0)
            .unwrap();
        engine
            .substrate
            .break_symmetry(3, Vec3::new(0.45, -0.20, 0.30), 1.5);
        assert!(engine.substrate.remove_edge(0, 1));
        assert!(engine.recognize_geometric_concept(0, 1.0e-3).is_none());

        let repair = engine.homeostatic_symmetry_repair(600, 1.0e-14);
        let recognition = engine.recognize_geometric_concept(0, 1.0e-3).unwrap();
        assert_eq!(repair.inserted_edges, 1);
        assert!(repair.chain_complex_valid);
        assert!(repair.after.symmetry_score > 0.999);
        assert!(recognition.confidence > 0.999);
    }

    #[test]
    fn geometry_ablation_prevents_cognitive_recovery() {
        let substrate = SymmetryThermodynamicSubstrate::regular_tetrahedron(
            1.0,
            SymmetryThermodynamicConfig {
                temperature: 0.0,
                geometry_weight: 0.0,
                ..SymmetryThermodynamicConfig::default()
            },
        )
        .unwrap();
        let mut engine =
            SimplicialThermodynamicEngine::new(substrate, SimplicialEngineConfig::default());
        engine
            .learn_geometric_concept("tetraedro_regular", 0)
            .unwrap();
        engine
            .substrate
            .break_symmetry(3, Vec3::new(0.45, -0.20, 0.30), 0.0);
        engine.homeostatic_symmetry_repair(600, 1.0e-14);
        assert!(engine.recognize_geometric_concept(0, 1.0e-3).is_none());
        assert!(engine.substrate.metrics().symmetry_score < 0.9);
    }

    #[test]
    fn input_seed_propagates_symmetry_and_recovers_trained_recognition() {
        let mut engine = tetrahedral_chain();
        let concept = engine
            .learn_geometric_concept("tetraedro_regular", 0)
            .unwrap();
        assert_eq!(engine.recognized_tetrahedra(concept.id, 1.0e-3), 3);

        engine
            .substrate
            .break_symmetry(0, Vec3::new(0.30, -0.12, 0.18), 0.0);
        engine
            .substrate
            .break_symmetry(1, Vec3::new(-0.22, 0.16, -0.08), 0.0);
        engine
            .substrate
            .break_symmetry(5, Vec3::new(0.25, 0.11, -0.17), 0.0);
        assert!(engine.substrate.remove_edge(0, 1));
        assert!(engine.substrate.remove_edge(1, 4));
        assert!(engine.substrate.remove_edge(4, 5));
        assert_eq!(engine.recognized_tetrahedra(concept.id, 1.0e-3), 0);

        let report = engine.propagate_symmetry_from(&[0], concept.id, 2, 600, 1.0e-14, 1.0e-3);
        let recovered = report
            .frames
            .iter()
            .map(|frame| frame.recognized_tetrahedra)
            .collect::<Vec<_>>();
        assert_eq!(report.baseline_recognized, 0);
        assert_eq!(report.final_recognized, 3);
        assert_eq!(recovered.last(), Some(&3));
        assert!(recovered.iter().any(|count| *count > 0 && *count < 3));
        assert!(report
            .frames
            .windows(2)
            .all(|pair| pair[0].recognized_tetrahedra <= pair[1].recognized_tetrahedra));
    }

    #[test]
    fn subdivided_tetrahedron_has_near_flat_interior_regge_edges() {
        let outer = SymmetryThermodynamicSubstrate::regular_tetrahedron(
            1.0,
            SymmetryThermodynamicConfig::default(),
        )
        .unwrap();
        let mut positions = outer
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();
        let center = positions
            .iter()
            .copied()
            .fold(Vec3::default(), |sum, position| sum + position)
            / 4.0;
        positions.push(center);
        let substrate = SymmetryThermodynamicSubstrate::new(
            positions,
            vec![
                Tetrahedron([4, 1, 2, 3]),
                Tetrahedron([0, 4, 2, 3]),
                Tetrahedron([0, 1, 4, 3]),
                Tetrahedron([0, 1, 2, 4]),
            ],
            1.0,
            SymmetryThermodynamicConfig::default(),
        )
        .unwrap();
        let engine =
            SimplicialThermodynamicEngine::new(substrate, SimplicialEngineConfig::default());
        let report = engine.hodge.regge_report(&engine.substrate);
        assert_eq!(report.interior_edges, 4);
        assert!(report.rms_deficit < 1.0e-10, "{report:?}");
    }
}
