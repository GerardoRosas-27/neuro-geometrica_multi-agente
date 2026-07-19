//! Sustrato cognitivo multicanal sin matrices densas.
//!
//! La dinámica usa eventos sobre aristas y sinapsis dispersas. Las matrices de
//! atención/pesos no se materializan: sólo existen interacciones locales activas.

use crate::entanglement::{EntanglementConfig, EntanglementField};
use crate::simplicial_thermodynamic_engine::SimplicialHodgeOperators;
use crate::symmetry_thermodynamic_substrate::SymmetryThermodynamicSubstrate;
use std::collections::{BTreeMap, VecDeque};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LatentConceptId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SparsePattern {
    pub components: Vec<(usize, f64)>,
}

impl SparsePattern {
    pub fn one_hot(channel: usize) -> Self {
        Self {
            components: vec![(channel, 1.0)],
        }
    }

    pub fn new(components: impl IntoIterator<Item = (usize, f64)>) -> Self {
        let mut merged = BTreeMap::<usize, f64>::new();
        for (channel, value) in components {
            if value.is_finite() && value.abs() > EPSILON {
                *merged.entry(channel).or_default() += value;
            }
        }
        Self {
            components: merged
                .into_iter()
                .filter(|(_, value)| value.abs() > EPSILON)
                .collect(),
        }
    }

    fn dense(&self, channels: usize) -> Option<Vec<f64>> {
        let mut dense = vec![0.0; channels];
        for &(channel, value) in &self.components {
            if channel >= channels {
                return None;
            }
            dense[channel] = value;
        }
        Some(dense)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MatrixFreeCognitiveConfig {
    pub channels: usize,
    pub learning_rate: f64,
    pub eligibility_decay: f64,
    pub event_attenuation: f64,
    pub max_synapse_weight: f64,
    pub symmetry_tying: bool,
    /// Control estrictamente local: sólo actualiza la arista observada.
    pub local_updates_only: bool,
    pub minimum_exposures: u64,
    pub max_prediction_error: f64,
    pub max_invariance_error: f64,
    pub max_equivariance_error: f64,
    pub max_stability_drift: f64,
    pub recognition_tolerance: f64,
    pub entanglement: EntanglementConfig,
}

impl Default for MatrixFreeCognitiveConfig {
    fn default() -> Self {
        Self {
            channels: 8,
            learning_rate: 0.35,
            eligibility_decay: 0.80,
            event_attenuation: 0.95,
            max_synapse_weight: 2.0,
            symmetry_tying: true,
            local_updates_only: false,
            minimum_exposures: 12,
            max_prediction_error: 2.0e-3,
            max_invariance_error: 2.0e-3,
            max_equivariance_error: 2.0e-3,
            max_stability_drift: 1.0e-2,
            recognition_tolerance: 0.08,
            entanglement: EntanglementConfig {
                create_threshold: 0.75,
                ..EntanglementConfig::default()
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiCochainState {
    pub channels: usize,
    pub vertex: Vec<f64>,
    pub edge: Vec<f64>,
    pub face: Vec<f64>,
    pub tetrahedron: Vec<f64>,
    pub prediction: Vec<f64>,
    pub error: Vec<f64>,
    vertex_generation: Vec<usize>,
}

impl MultiCochainState {
    fn new(
        vertices: usize,
        edges: usize,
        faces: usize,
        tetrahedra: usize,
        channels: usize,
    ) -> Self {
        Self {
            channels,
            vertex: vec![0.0; vertices * channels],
            edge: vec![0.0; edges * channels],
            face: vec![0.0; faces * channels],
            tetrahedron: vec![0.0; tetrahedra * channels],
            prediction: vec![0.0; vertices * channels],
            error: vec![0.0; vertices * channels],
            vertex_generation: vec![0; vertices * channels],
        }
    }

    fn clear_dynamics(&mut self) {
        self.vertex.fill(0.0);
        self.edge.fill(0.0);
        self.face.fill(0.0);
        self.tetrahedron.fill(0.0);
        self.prediction.fill(0.0);
        self.error.fill(0.0);
        self.vertex_generation.fill(0);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SynapseKey {
    source: usize,
    target: usize,
    pre_channel: usize,
    post_channel: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SparseLocalSynapse {
    pub weight: f64,
    pub eligibility: f64,
    pub updates: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LatentConcept {
    pub id: LatentConceptId,
    pub prototype: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConsolidationGateReport {
    pub prediction_error: f64,
    pub invariance_error: f64,
    pub equivariance_error: f64,
    pub stability_drift: f64,
    pub exposures: u64,
    pub consolidated: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransitionMemory {
    pub source: LatentConceptId,
    pub target: LatentConceptId,
    pub gate: ConsolidationGateReport,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EventRunReport {
    pub processed_events: usize,
    pub emitted_events: usize,
    pub dropped_events: usize,
    pub max_generation: usize,
    pub active_synapses: usize,
}

#[derive(Clone, Copy, Debug)]
struct LocalEvent {
    vertex: usize,
    channel: usize,
    value: f64,
    generation: usize,
    remaining_hops: usize,
}

#[derive(Clone, Debug)]
pub struct MatrixFreeCognitiveSubstrate {
    pub geometry: SymmetryThermodynamicSubstrate,
    pub hodge: SimplicialHodgeOperators,
    pub state: MultiCochainState,
    pub entanglement: EntanglementField,
    pub concepts: Vec<LatentConcept>,
    pub transitions: BTreeMap<(LatentConceptId, LatentConceptId), TransitionMemory>,
    pub config: MatrixFreeCognitiveConfig,
    synapses: BTreeMap<SynapseKey, SparseLocalSynapse>,
    queue: VecDeque<LocalEvent>,
}

impl MatrixFreeCognitiveSubstrate {
    pub fn new(
        geometry: SymmetryThermodynamicSubstrate,
        config: MatrixFreeCognitiveConfig,
    ) -> Self {
        let config = sanitized_config(config);
        let hodge = SimplicialHodgeOperators::from_substrate(&geometry);
        let state = MultiCochainState::new(
            geometry.vertices.len(),
            geometry.edges.len(),
            hodge.triangle_count(),
            hodge.tetrahedron_count(),
            config.channels,
        );
        Self {
            geometry,
            hodge,
            state,
            entanglement: EntanglementField::new(config.entanglement),
            concepts: Vec::new(),
            transitions: BTreeMap::new(),
            config,
            synapses: BTreeMap::new(),
            queue: VecDeque::new(),
        }
    }

    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    pub fn register_concept(&mut self, pattern: &SparsePattern) -> Option<LatentConceptId> {
        let prototype = pattern.dense(self.config.channels)?;
        if prototype.iter().all(|value| value.abs() <= EPSILON) {
            return None;
        }
        let id = LatentConceptId(self.concepts.len());
        self.concepts.push(LatentConcept { id, prototype });
        Some(id)
    }

    pub fn recognize_pattern(&self, pattern: &SparsePattern) -> Option<LatentConceptId> {
        let vector = pattern.dense(self.config.channels)?;
        self.recognize_vector(&vector)
    }

    pub fn recognize_vertex(&self, vertex: usize) -> Option<LatentConceptId> {
        let vector = self.frontier_vector(vertex)?;
        self.recognize_vector(&vector)
    }

    pub fn train_transition(
        &mut self,
        source_concept: LatentConceptId,
        target_concept: LatentConceptId,
        source_vertex: usize,
        target_vertex: usize,
    ) -> Option<ConsolidationGateReport> {
        self.train_transition_sample(
            source_concept,
            target_concept,
            source_vertex,
            target_vertex,
            1.0,
            1.0,
        )
    }

    pub fn train_transition_sample(
        &mut self,
        source_concept: LatentConceptId,
        target_concept: LatentConceptId,
        source_vertex: usize,
        target_vertex: usize,
        source_scale: f64,
        target_scale: f64,
    ) -> Option<ConsolidationGateReport> {
        if !self.has_directed_edge(source_vertex, target_vertex) {
            return None;
        }
        if !source_scale.is_finite() || !target_scale.is_finite() {
            return None;
        }
        let source = self
            .concepts
            .get(source_concept.0)?
            .prototype
            .iter()
            .map(|value| value * source_scale)
            .collect::<Vec<_>>();
        let target = self
            .concepts
            .get(target_concept.0)?
            .prototype
            .iter()
            .map(|value| value * target_scale)
            .collect::<Vec<_>>();
        let prediction = self.direct_prediction(source_vertex, target_vertex, &source);
        let error = target
            .iter()
            .zip(&prediction)
            .map(|(expected, predicted)| expected - predicted)
            .collect::<Vec<_>>();
        let previous = self
            .transitions
            .get(&(source_concept, target_concept))
            .copied()
            .unwrap_or(TransitionMemory {
                source: source_concept,
                target: target_concept,
                gate: ConsolidationGateReport::default(),
            });
        // Ambos brazos de la ablación actualizan el mismo número de sinapsis.
        // El control usa una acción de canales determinista pero incorrecta en
        // las aristas no observadas, en vez de recibir menos parámetros/cómputo.
        let update_edges = if self.config.local_updates_only {
            vec![(source_vertex, target_vertex)]
        } else {
            self.all_directed_edges()
        };
        let mut max_drift: f64 = 0.0;
        for (orbit_rank, (from, to)) in update_edges.into_iter().enumerate() {
            for (pre_channel, pre) in source.iter().copied().enumerate() {
                if pre.abs() <= EPSILON {
                    continue;
                }
                for (post_channel, post) in target.iter().copied().enumerate() {
                    if post.abs() <= EPSILON {
                        continue;
                    }
                    let effective_post_channel = if self.config.symmetry_tying
                        || self.config.local_updates_only
                        || (from == source_vertex && to == target_vertex)
                    {
                        post_channel
                    } else {
                        (post_channel + orbit_rank + 1) % self.config.channels
                    };
                    let key = SynapseKey {
                        source: from,
                        target: to,
                        pre_channel,
                        post_channel: effective_post_channel,
                    };
                    let synapse = self.synapses.entry(key).or_default();
                    synapse.eligibility = self.config.eligibility_decay * synapse.eligibility
                        + (1.0 - self.config.eligibility_decay) * (pre * post).abs();
                    let delta = self.config.learning_rate
                        * pre
                        * post
                        * error[post_channel]
                        * synapse.eligibility;
                    synapse.weight = (synapse.weight + delta).clamp(
                        -self.config.max_synapse_weight,
                        self.config.max_synapse_weight,
                    );
                    synapse.updates += 1;
                    max_drift = max_drift.max(delta.abs());
                    if let Some(edge) = self.edge_index(from, to) {
                        self.state.edge[edge * self.config.channels + effective_post_channel] +=
                            delta;
                    }
                }
            }
        }
        let exposures = previous.gate.exposures + 1;
        let gate = self.evaluate_gate(
            &source,
            &target,
            source_vertex,
            target_vertex,
            exposures,
            max_drift,
        );
        self.transitions.insert(
            (source_concept, target_concept),
            TransitionMemory {
                source: source_concept,
                target: target_concept,
                gate,
            },
        );
        if gate.consolidated {
            self.entanglement
                .observe_correlation(source_vertex, target_vertex, 1.0);
        }
        Some(gate)
    }

    pub fn predict_transition(
        &self,
        source_concept: LatentConceptId,
        source_vertex: usize,
        target_vertex: usize,
    ) -> Option<Vec<f64>> {
        let source = &self.concepts.get(source_concept.0)?.prototype;
        Some(self.direct_prediction(source_vertex, target_vertex, source))
    }

    pub fn transition_memory(
        &self,
        source: LatentConceptId,
        target: LatentConceptId,
    ) -> Option<TransitionMemory> {
        self.transitions.get(&(source, target)).copied()
    }

    pub fn propagate_concept(
        &mut self,
        concept: LatentConceptId,
        seed_vertex: usize,
        hops: usize,
        event_budget: usize,
    ) -> Option<EventRunReport> {
        let prototype = self.concepts.get(concept.0)?.prototype.clone();
        if seed_vertex >= self.geometry.vertices.len() {
            return None;
        }
        self.state.clear_dynamics();
        self.queue.clear();
        for (channel, value) in prototype.into_iter().enumerate() {
            if value.abs() > EPSILON {
                self.queue.push_back(LocalEvent {
                    vertex: seed_vertex,
                    channel,
                    value,
                    generation: 0,
                    remaining_hops: hops,
                });
            }
        }
        Some(self.run_events(event_budget))
    }

    pub fn cochain_hodge_energy(&self) -> (f64, f64, f64) {
        let mut l0 = 0.0;
        let mut l1 = 0.0;
        let mut l2 = 0.0;
        for channel in 0..self.config.channels {
            let vertices = gather_channel(
                &self.state.vertex,
                self.geometry.vertices.len(),
                self.config.channels,
                channel,
            );
            let edges = gather_channel(
                &self.state.edge,
                self.geometry.edges.len(),
                self.config.channels,
                channel,
            );
            let faces = gather_channel(
                &self.state.face,
                self.hodge.triangle_count(),
                self.config.channels,
                channel,
            );
            let report = self.hodge.hodge_report(&vertices, &edges, &faces);
            l0 += report.l0_energy;
            l1 += report.l1_energy;
            l2 += report.l2_energy;
        }
        (l0, l1, l2)
    }

    fn run_events(&mut self, event_budget: usize) -> EventRunReport {
        let mut report = EventRunReport {
            active_synapses: self.synapses.len(),
            ..EventRunReport::default()
        };
        while let Some(event) = self.queue.pop_front() {
            if report.processed_events >= event_budget {
                report.dropped_events += 1 + self.queue.len();
                self.queue.clear();
                break;
            }
            report.processed_events += 1;
            report.max_generation = report.max_generation.max(event.generation);
            let index = event.vertex * self.config.channels + event.channel;
            if event.generation > self.state.vertex_generation[index] {
                self.state.vertex[index] = 0.0;
                self.state.vertex_generation[index] = event.generation;
            }
            if event.generation == self.state.vertex_generation[index] {
                self.state.vertex[index] =
                    (self.state.vertex[index] + event.value).clamp(-4.0, 4.0);
            }
            if event.remaining_hops == 0 {
                continue;
            }

            let outgoing = self
                .synapses
                .iter()
                .filter(|(key, synapse)| {
                    key.source == event.vertex
                        && key.pre_channel == event.channel
                        && synapse.weight.abs() > EPSILON
                })
                .map(|(key, synapse)| (*key, *synapse))
                .collect::<Vec<_>>();
            for (key, synapse) in outgoing {
                let epr_gain = if self.entanglement.has_active_link(key.source, key.target) {
                    1.02
                } else {
                    1.0
                };
                let transmitted =
                    event.value * synapse.weight * self.config.event_attenuation * epr_gain;
                if transmitted.abs() <= EPSILON {
                    continue;
                }
                if let Some(edge) = self.edge_index(key.source, key.target) {
                    self.state.edge[edge * self.config.channels + key.post_channel] += transmitted;
                }
                self.queue.push_back(LocalEvent {
                    vertex: key.target,
                    channel: key.post_channel,
                    value: transmitted,
                    generation: event.generation + 1,
                    remaining_hops: event.remaining_hops - 1,
                });
                report.emitted_events += 1;
            }
        }
        self.refresh_higher_cochains();
        report
    }

    fn refresh_higher_cochains(&mut self) {
        self.state.face.fill(0.0);
        self.state.tetrahedron.fill(0.0);
        for channel in 0..self.config.channels {
            let edge = gather_channel(
                &self.state.edge,
                self.geometry.edges.len(),
                self.config.channels,
                channel,
            );
            let curl = self.hodge.edge_curl(&edge);
            for (face, value) in curl.iter().copied().enumerate() {
                self.state.face[face * self.config.channels + channel] = value;
            }
            let volume = self.hodge.volume_coboundary(&curl);
            for (tetrahedron, value) in volume.into_iter().enumerate() {
                self.state.tetrahedron[tetrahedron * self.config.channels + channel] = value;
            }
        }
    }

    fn frontier_vector(&self, vertex: usize) -> Option<Vec<f64>> {
        if vertex >= self.geometry.vertices.len() {
            return None;
        }
        let start = vertex * self.config.channels;
        let generations = &self.state.vertex_generation[start..start + self.config.channels];
        let max_generation = generations.iter().copied().max().unwrap_or(0);
        Some(
            (0..self.config.channels)
                .map(|channel| {
                    let index = start + channel;
                    if self.state.vertex_generation[index] == max_generation {
                        self.state.vertex[index]
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
    }

    fn recognize_vector(&self, vector: &[f64]) -> Option<LatentConceptId> {
        let observed_norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        if observed_norm <= EPSILON {
            return None;
        }
        let (concept, distance) = self
            .concepts
            .iter()
            .map(|concept| {
                let prototype_norm = concept
                    .prototype
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
                    .max(EPSILON);
                let cosine = concept
                    .prototype
                    .iter()
                    .zip(vector)
                    .map(|(expected, observed)| expected * observed)
                    .sum::<f64>()
                    / (prototype_norm * observed_norm);
                let distance = 1.0 - cosine.clamp(-1.0, 1.0);
                (concept.id, distance)
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))?;
        (distance <= self.config.recognition_tolerance).then_some(concept)
    }

    fn evaluate_gate(
        &self,
        source: &[f64],
        target: &[f64],
        source_vertex: usize,
        target_vertex: usize,
        exposures: u64,
        stability_drift: f64,
    ) -> ConsolidationGateReport {
        let prediction = self.direct_prediction(source_vertex, target_vertex, source);
        let prediction_error = mean_squared_error(target, &prediction);
        let orbit_predictions = self
            .all_directed_edges()
            .into_iter()
            .map(|(from, to)| self.direct_prediction(from, to, source))
            .collect::<Vec<_>>();
        let invariance_error = orbit_predictions
            .iter()
            .map(|prediction| mean_squared_error(target, prediction))
            .sum::<f64>()
            / orbit_predictions.len().max(1) as f64;
        let mut equivariance_error = 0.0;
        for channel in 0..self.config.channels {
            let mean = orbit_predictions
                .iter()
                .map(|prediction| prediction[channel])
                .sum::<f64>()
                / orbit_predictions.len().max(1) as f64;
            equivariance_error += orbit_predictions
                .iter()
                .map(|prediction| (prediction[channel] - mean).powi(2))
                .sum::<f64>()
                / orbit_predictions.len().max(1) as f64;
        }
        equivariance_error /= self.config.channels as f64;
        let consolidated = exposures >= self.config.minimum_exposures
            && prediction_error <= self.config.max_prediction_error
            && invariance_error <= self.config.max_invariance_error
            && equivariance_error <= self.config.max_equivariance_error
            && stability_drift <= self.config.max_stability_drift;
        ConsolidationGateReport {
            prediction_error,
            invariance_error,
            equivariance_error,
            stability_drift,
            exposures,
            consolidated,
        }
    }

    fn direct_prediction(&self, source: usize, target: usize, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.config.channels];
        for (key, synapse) in &self.synapses {
            if key.source == source && key.target == target && key.pre_channel < input.len() {
                output[key.post_channel] +=
                    input[key.pre_channel] * synapse.weight * self.config.event_attenuation;
            }
        }
        output
    }

    fn has_directed_edge(&self, source: usize, target: usize) -> bool {
        self.geometry.edges.iter().any(|edge| {
            (edge.a == source && edge.b == target) || (edge.a == target && edge.b == source)
        })
    }

    fn all_directed_edges(&self) -> Vec<(usize, usize)> {
        self.geometry
            .edges
            .iter()
            .flat_map(|edge| [(edge.a, edge.b), (edge.b, edge.a)])
            .collect()
    }

    fn edge_index(&self, source: usize, target: usize) -> Option<usize> {
        self.geometry.edges.iter().position(|edge| {
            (edge.a == source && edge.b == target) || (edge.a == target && edge.b == source)
        })
    }
}

fn sanitized_config(config: MatrixFreeCognitiveConfig) -> MatrixFreeCognitiveConfig {
    MatrixFreeCognitiveConfig {
        channels: config.channels.max(1),
        learning_rate: config.learning_rate.max(0.0),
        eligibility_decay: config.eligibility_decay.clamp(0.0, 1.0),
        event_attenuation: config.event_attenuation.clamp(0.0, 1.0),
        max_synapse_weight: config.max_synapse_weight.abs().max(EPSILON),
        minimum_exposures: config.minimum_exposures.max(1),
        max_prediction_error: config.max_prediction_error.max(0.0),
        max_invariance_error: config.max_invariance_error.max(0.0),
        max_equivariance_error: config.max_equivariance_error.max(0.0),
        max_stability_drift: config.max_stability_drift.max(0.0),
        recognition_tolerance: config.recognition_tolerance.max(0.0),
        ..config
    }
}

fn gather_channel(values: &[f64], entities: usize, channels: usize, channel: usize) -> Vec<f64> {
    (0..entities)
        .map(|entity| values[entity * channels + channel])
        .collect()
}

fn mean_squared_error(expected: &[f64], observed: &[f64]) -> f64 {
    expected
        .iter()
        .zip(observed)
        .map(|(expected, observed)| (expected - observed).powi(2))
        .sum::<f64>()
        / expected.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry_thermodynamic_substrate::{
        SymmetryThermodynamicConfig, SymmetryThermodynamicSubstrate,
    };

    fn substrate(symmetry_tying: bool) -> MatrixFreeCognitiveSubstrate {
        let geometry = SymmetryThermodynamicSubstrate::regular_tetrahedron(
            1.0,
            SymmetryThermodynamicConfig::default(),
        )
        .unwrap();
        MatrixFreeCognitiveSubstrate::new(
            geometry,
            MatrixFreeCognitiveConfig {
                symmetry_tying,
                ..MatrixFreeCognitiveConfig::default()
            },
        )
    }

    fn train(
        substrate: &mut MatrixFreeCognitiveSubstrate,
        source: LatentConceptId,
        target: LatentConceptId,
        edge: (usize, usize),
    ) -> ConsolidationGateReport {
        let mut report = ConsolidationGateReport::default();
        for _ in 0..40 {
            report = substrate
                .train_transition(source, target, edge.0, edge.1)
                .unwrap();
        }
        report
    }

    #[test]
    fn multichannel_cochains_are_materialized_without_dense_weights() {
        let substrate = substrate(true);
        assert_eq!(substrate.state.vertex.len(), 4 * 8);
        assert_eq!(substrate.state.edge.len(), 6 * 8);
        assert_eq!(substrate.state.face.len(), 4 * 8);
        assert_eq!(substrate.state.tetrahedron.len(), 8);
        assert_eq!(substrate.synapse_count(), 0);
    }

    #[test]
    fn three_factor_rule_reduces_prediction_error_and_consolidates() {
        let mut substrate = substrate(true);
        let a = substrate
            .register_concept(&SparsePattern::one_hot(0))
            .unwrap();
        let b = substrate
            .register_concept(&SparsePattern::one_hot(1))
            .unwrap();
        let before = substrate.predict_transition(a, 0, 1).unwrap();
        let before_error = mean_squared_error(&substrate.concepts[b.0].prototype, &before);
        let early = substrate.train_transition(a, b, 0, 1).unwrap();
        assert!(!early.consolidated);
        let gate = train(&mut substrate, a, b, (0, 1));
        let after = substrate.predict_transition(a, 0, 1).unwrap();
        let after_error = mean_squared_error(&substrate.concepts[b.0].prototype, &after);
        assert!(after_error < before_error * 1.0e-3);
        assert!(gate.consolidated, "{gate:?}");
        assert!(substrate.synapse_count() > 0);
        assert!(substrate.entanglement.active_count() > 0);
    }

    #[test]
    fn functional_invariance_can_consolidate_without_geometric_crystallization() {
        let mut substrate = substrate(true);
        substrate.geometry.break_symmetry(
            3,
            crate::symmetry_thermodynamic_substrate::Vec3::new(0.45, -0.2, 0.3),
            0.0,
        );
        let geometric_symmetry = substrate.geometry.metrics().symmetry_score;
        let a = substrate
            .register_concept(&SparsePattern::one_hot(0))
            .unwrap();
        let b = substrate
            .register_concept(&SparsePattern::one_hot(1))
            .unwrap();
        let gate = train(&mut substrate, a, b, (0, 1));
        assert!(geometric_symmetry < 0.9);
        assert!(gate.consolidated, "{gate:?}");
        assert!(gate.invariance_error <= substrate.config.max_invariance_error);
    }

    #[test]
    fn symmetry_tying_generalizes_to_untrained_edges() {
        let mut full = substrate(true);
        let a = full.register_concept(&SparsePattern::one_hot(0)).unwrap();
        let b = full.register_concept(&SparsePattern::one_hot(1)).unwrap();
        let full_gate = train(&mut full, a, b, (0, 1));
        let held_out = full.predict_transition(a, 2, 3).unwrap();
        assert_eq!(
            full.recognize_vector(&held_out),
            Some(b),
            "full must generalize over the S4 edge orbit"
        );
        assert!(full_gate.consolidated);

        let mut ablated = substrate(false);
        let a0 = ablated
            .register_concept(&SparsePattern::one_hot(0))
            .unwrap();
        let b0 = ablated
            .register_concept(&SparsePattern::one_hot(1))
            .unwrap();
        let ablated_gate = train(&mut ablated, a0, b0, (0, 1));
        let held_out = ablated.predict_transition(a0, 2, 3).unwrap();
        assert_eq!(ablated.recognize_vector(&held_out), None);
        assert!(!ablated_gate.consolidated);
        assert!(ablated_gate.invariance_error > full_gate.invariance_error);
        assert_eq!(full.synapse_count(), ablated.synapse_count());
    }

    #[test]
    fn event_driven_two_hop_composition_recovers_unseen_transition() {
        let mut substrate = substrate(true);
        let a = substrate
            .register_concept(&SparsePattern::one_hot(0))
            .unwrap();
        let b = substrate
            .register_concept(&SparsePattern::one_hot(1))
            .unwrap();
        let c = substrate
            .register_concept(&SparsePattern::one_hot(2))
            .unwrap();
        train(&mut substrate, a, b, (0, 1));
        train(&mut substrate, b, c, (1, 2));
        assert!(substrate.transition_memory(a, c).is_none());

        let report = substrate.propagate_concept(a, 0, 2, 10_000).unwrap();
        let frontier = (0..4)
            .map(|vertex| substrate.frontier_vector(vertex).unwrap())
            .collect::<Vec<_>>();
        let recovered = (0..4)
            .filter(|vertex| substrate.recognize_vertex(*vertex) == Some(c))
            .count();
        assert!(report.processed_events > 0);
        assert!(recovered > 0, "A→C must emerge through A→B→C: {frontier:?}");
        let (_, l1, l2) = substrate.cochain_hodge_energy();
        assert!(l1.is_finite() && l2.is_finite());
    }

    #[test]
    fn independent_sparse_rules_resist_interference() {
        let mut substrate = substrate(true);
        let a = substrate
            .register_concept(&SparsePattern::one_hot(0))
            .unwrap();
        let b = substrate
            .register_concept(&SparsePattern::one_hot(1))
            .unwrap();
        let d = substrate
            .register_concept(&SparsePattern::one_hot(3))
            .unwrap();
        let e = substrate
            .register_concept(&SparsePattern::one_hot(4))
            .unwrap();
        train(&mut substrate, a, b, (0, 1));
        let before = substrate.predict_transition(a, 2, 3).unwrap();
        train(&mut substrate, d, e, (0, 1));
        let after = substrate.predict_transition(a, 2, 3).unwrap();
        assert!(mean_squared_error(&before, &after) < 1.0e-12);
        assert_eq!(substrate.recognize_vector(&after), Some(b));
    }

    #[test]
    fn unknown_channel_pattern_is_rejected_as_ood() {
        let mut substrate = substrate(true);
        substrate
            .register_concept(&SparsePattern::one_hot(0))
            .unwrap();
        substrate
            .register_concept(&SparsePattern::one_hot(1))
            .unwrap();
        assert_eq!(
            substrate.recognize_pattern(&SparsePattern::one_hot(7)),
            None
        );
    }
}
