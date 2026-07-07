use crate::cdt_graphity::{CdtGraphityEdgeKind, CdtGraphitySubstrate};
use crate::relational_field::{CandidateScore, CollapseReport};
use std::collections::HashMap;

const EPSILON: f32 = 1.0e-5;

#[derive(Clone, Copy, Debug)]
pub struct RelationalGuidanceConfig {
    pub surface_gain: f32,
    pub pressure_gain: f32,
    pub wave_gain: f32,
    pub capillary_gain: f32,
    pub relational_pilot_gain: f32,
    pub cdt_pilot_gain: f32,
    pub regge_gain: f32,
    pub microticks: usize,
    pub dt: f32,
}

impl Default for RelationalGuidanceConfig {
    fn default() -> Self {
        Self {
            surface_gain: 0.30,
            pressure_gain: 0.42,
            wave_gain: 0.28,
            capillary_gain: 0.36,
            relational_pilot_gain: 0.22,
            cdt_pilot_gain: 0.30,
            regge_gain: 0.18,
            microticks: 5,
            dt: 0.18,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RelationalGuidanceReport {
    pub candidates: usize,
    pub mean_multiplier: f32,
    pub mean_quantum_potential: f32,
    pub mean_guidance_flow: f32,
    pub mean_regge_cost: f32,
}

#[derive(Clone, Debug, Default)]
pub struct RelationalGuidanceEngine {
    cache: GuidanceGeometryCache,
}

impl RelationalGuidanceEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(
        &mut self,
        hardware: &CdtGraphitySubstrate,
        report: &mut CollapseReport,
    ) -> RelationalGuidanceReport {
        if report.candidates.is_empty() {
            return RelationalGuidanceReport::default();
        }

        let config = RelationalGuidanceConfig::default();
        self.cache.refresh(hardware);
        let context = GuidanceContext::new(hardware, report, &self.cache);
        apply_with_context(config, hardware, report, &context)
    }
}

fn apply_with_context(
    config: RelationalGuidanceConfig,
    hardware: &CdtGraphitySubstrate,
    report: &mut CollapseReport,
    context: &GuidanceContext<'_>,
) -> RelationalGuidanceReport {
    let mut summary = RelationalGuidanceReport::default();
    for candidate in &mut report.candidates {
        let terms = context.terms(hardware, candidate);
        let multiplier = guidance_multiplier(config, terms);
        candidate.score *= multiplier;

        summary.candidates += 1;
        summary.mean_multiplier += multiplier;
        summary.mean_quantum_potential += terms.quantum_potential;
        summary.mean_guidance_flow += terms.guidance_flow;
        summary.mean_regge_cost += terms.regge_cost;
    }

    report.candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.agent.cmp(&b.agent))
    });

    let n = summary.candidates.max(1) as f32;
    summary.mean_multiplier /= n;
    summary.mean_quantum_potential /= n;
    summary.mean_guidance_flow /= n;
    summary.mean_regge_cost /= n;
    summary
}

#[derive(Clone, Debug, Default)]
struct GuidanceGeometryCache {
    signature: GuidanceGeometrySignature,
    incident_edges: Vec<Vec<usize>>,
    edge_incidents: Vec<usize>,
}

impl GuidanceGeometryCache {
    fn refresh(&mut self, hardware: &CdtGraphitySubstrate) {
        let signature = GuidanceGeometrySignature::from_hardware(hardware);
        if self.signature == signature {
            return;
        }
        self.signature = signature;
        self.incident_edges = incident_edges(hardware);
        self.edge_incidents = edge_tetrahedra_incidents(hardware);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GuidanceGeometrySignature {
    nodes: usize,
    edges: usize,
    tetrahedra: usize,
    active_hash: u64,
}

impl GuidanceGeometrySignature {
    fn from_hardware(hardware: &CdtGraphitySubstrate) -> Self {
        let mut active_hash = 0xcbf2_9ce4_8422_2325_u64;
        for (idx, edge) in hardware.edges.iter().enumerate() {
            if !edge.active {
                continue;
            }
            active_hash ^= idx as u64
                ^ ((edge.a as u64) << 17)
                ^ ((edge.b as u64) << 33)
                ^ match edge.kind {
                    CdtGraphityEdgeKind::Spatial => 0x51,
                    CdtGraphityEdgeKind::Temporal => 0xA7,
                };
            active_hash = active_hash.wrapping_mul(0x100_0000_01B3);
        }
        Self {
            nodes: hardware.nodes.len(),
            edges: hardware.edges.len(),
            tetrahedra: hardware.tetrahedra.len(),
            active_hash,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct GuidanceTerms {
    surface_tension: f32,
    pressure_flow: f32,
    wave_interference: f32,
    capillary_memory: f32,
    relational_pilot: f32,
    quantum_potential: f32,
    guidance_flow: f32,
    regge_cost: f32,
    causal_gate: f32,
    micro_evolution: f32,
}

struct GuidanceContext<'a> {
    amplitudes: Vec<f32>,
    seeds: Vec<bool>,
    seed_slices: Vec<usize>,
    incident_edges: &'a [Vec<usize>],
    edge_incidents: &'a [usize],
}

impl<'a> GuidanceContext<'a> {
    fn new(
        hardware: &CdtGraphitySubstrate,
        report: &CollapseReport,
        cache: &'a GuidanceGeometryCache,
    ) -> Self {
        let mut amplitudes = vec![0.0; hardware.nodes.len()];
        for candidate in &report.candidates {
            if candidate.agent < amplitudes.len() {
                amplitudes[candidate.agent] = (candidate.probability.max(0.0) + EPSILON).sqrt();
            }
        }

        let mut seeds = vec![false; hardware.nodes.len()];
        let mut seed_slices = Vec::new();
        for &seed in &report.seeds {
            if seed < seeds.len() {
                seeds[seed] = true;
                amplitudes[seed] = 1.0;
                if let Some(node) = hardware.nodes.get(seed) {
                    seed_slices.push(node.slice);
                }
            }
        }

        Self {
            amplitudes,
            seeds,
            seed_slices,
            incident_edges: &cache.incident_edges,
            edge_incidents: &cache.edge_incidents,
        }
    }

    fn terms(&self, hardware: &CdtGraphitySubstrate, candidate: &CandidateScore) -> GuidanceTerms {
        let amplitude = self
            .amplitudes
            .get(candidate.agent)
            .copied()
            .unwrap_or(EPSILON)
            .max(EPSILON);
        let phase = phase_drive(candidate);
        let cdt_laplacian = self.cdt_laplacian(hardware, candidate.agent, amplitude);
        let quantum_potential = quantum_potential(cdt_laplacian, amplitude);
        let guidance_flow = self.guidance_flow(hardware, candidate.agent, amplitude, phase);
        let regge_cost = self.local_regge_cost(hardware, candidate.agent);
        let causal_gate = self.causal_gate(hardware, candidate.agent);
        let pressure_flow = self.pressure_flow(hardware, candidate.agent);
        let capillary_memory = self.capillary_memory(hardware, candidate, quantum_potential);

        GuidanceTerms {
            surface_tension: surface_tension(candidate),
            pressure_flow,
            wave_interference: phase,
            capillary_memory,
            relational_pilot: relational_pilot(candidate, quantum_potential, guidance_flow),
            quantum_potential,
            guidance_flow,
            regge_cost,
            causal_gate,
            micro_evolution: micro_evolution(
                guidance_flow,
                quantum_potential,
                regge_cost,
                causal_gate,
            ),
        }
    }

    fn cdt_laplacian(&self, hardware: &CdtGraphitySubstrate, agent: usize, amplitude: f32) -> f32 {
        let Some(edges) = self.incident_edges.get(agent) else {
            return 0.0;
        };
        let mut laplacian = 0.0_f32;
        let mut norm = 0.0_f32;
        for &edge_idx in edges {
            let edge = &hardware.edges[edge_idx];
            let neighbor = if edge.a == agent { edge.b } else { edge.a };
            let neighbor_amplitude = self.amplitudes.get(neighbor).copied().unwrap_or(0.0);
            let conductance = edge_conductance(edge.kind, edge.stability, edge.prediction_error);
            laplacian += conductance * (neighbor_amplitude - amplitude);
            norm += conductance.abs();
        }
        if norm <= EPSILON {
            0.0
        } else {
            laplacian / norm
        }
    }

    fn guidance_flow(
        &self,
        hardware: &CdtGraphitySubstrate,
        agent: usize,
        amplitude: f32,
        phase: f32,
    ) -> f32 {
        let Some(edges) = self.incident_edges.get(agent) else {
            return 0.0;
        };
        let mut flow = 0.0_f32;
        let mut channels = 0_usize;
        for &edge_idx in edges {
            let edge = &hardware.edges[edge_idx];
            if edge.kind != CdtGraphityEdgeKind::Temporal || edge.b != agent {
                continue;
            }
            if self.seeds.get(edge.a).copied().unwrap_or(false) {
                flow += edge_conductance(edge.kind, edge.stability, edge.prediction_error)
                    * amplitude
                    * phase;
                channels += 1;
            }
        }
        if channels == 0 {
            0.0
        } else {
            (flow / channels as f32).clamp(0.0, 1.0)
        }
    }

    fn pressure_flow(&self, hardware: &CdtGraphitySubstrate, agent: usize) -> f32 {
        let Some(edges) = self.incident_edges.get(agent) else {
            return 0.0;
        };
        let mut flow = 0.0_f32;
        let mut channels = 0_usize;
        for &edge_idx in edges {
            let edge = &hardware.edges[edge_idx];
            if edge.kind != CdtGraphityEdgeKind::Temporal || edge.b != agent {
                continue;
            }
            if self.seeds.get(edge.a).copied().unwrap_or(false) {
                let pressure_seed = hardware
                    .nodes
                    .get(edge.a)
                    .map_or(1.0, |node| 1.0 + node.surprise);
                let pressure_candidate = self.amplitudes.get(agent).copied().unwrap_or(0.0) * 0.35;
                flow += edge_conductance(edge.kind, edge.stability, edge.prediction_error)
                    * (pressure_seed - pressure_candidate).max(0.0);
                channels += 1;
            }
        }
        if channels == 0 {
            0.0
        } else {
            (flow / channels as f32).clamp(0.0, 1.0)
        }
    }

    fn capillary_memory(
        &self,
        hardware: &CdtGraphitySubstrate,
        candidate: &CandidateScore,
        quantum_potential: f32,
    ) -> f32 {
        let support = self
            .incident_edges
            .get(candidate.agent)
            .into_iter()
            .flatten()
            .map(|&idx| {
                let edge = &hardware.edges[idx];
                edge_conductance(edge.kind, edge.stability, edge.prediction_error)
            })
            .sum::<f32>()
            / self
                .incident_edges
                .get(candidate.agent)
                .map(|edges| edges.len().max(1) as f32)
                .unwrap_or(1.0);
        let memory_well = candidate.mean_coherence * (1.0 - candidate.mean_uncertainty);
        (memory_well + support * 0.35 - quantum_potential * 0.20).clamp(0.0, 1.0)
    }

    fn local_regge_cost(&self, hardware: &CdtGraphitySubstrate, agent: usize) -> f32 {
        let Some(edges) = self.incident_edges.get(agent) else {
            return 0.5;
        };
        let mut cost = 0.0_f32;
        let mut count = 0_usize;
        for &edge_idx in edges {
            let edge = &hardware.edges[edge_idx];
            let target = match edge.kind {
                CdtGraphityEdgeKind::Spatial => hardware.config.target_tetrahedra_per_edge,
                CdtGraphityEdgeKind::Temporal => hardware.config.target_tetrahedra_per_edge + 1,
            }
            .max(1);
            let incident = self.edge_incidents.get(edge_idx).copied().unwrap_or(0);
            cost += target.abs_diff(incident) as f32 / target as f32;
            count += 1;
        }
        if count == 0 {
            0.5
        } else {
            (cost / count as f32).clamp(0.0, 2.0) * 0.5
        }
    }

    fn causal_gate(&self, hardware: &CdtGraphitySubstrate, candidate: usize) -> f32 {
        let Some(candidate_node) = hardware.nodes.get(candidate) else {
            return 0.0;
        };
        for seed_slice in &self.seed_slices {
            if candidate_node.slice == seed_slice + 1 {
                return 1.0;
            }
            if candidate_node.slice == *seed_slice {
                return 0.35;
            }
        }
        0.0
    }
}

fn guidance_multiplier(config: RelationalGuidanceConfig, terms: GuidanceTerms) -> f32 {
    let fluidic = (1.0 + config.surface_gain * terms.surface_tension)
        * (1.0 + config.pressure_gain * terms.pressure_flow)
        * (1.0 + config.wave_gain * terms.wave_interference.max(0.0))
        * (1.0 + config.capillary_gain * terms.capillary_memory);
    let relational_pilot = 1.0 + config.relational_pilot_gain * terms.relational_pilot;
    let cdt_quality = (terms.guidance_flow + terms.causal_gate * 0.35)
        / (1.0 + terms.quantum_potential + terms.regge_cost);
    let cdt_pilot = (1.0 + config.cdt_pilot_gain * cdt_quality)
        * terms.micro_evolution.max(1.0 - config.regge_gain * 0.25);
    (fluidic * relational_pilot * cdt_pilot).clamp(0.05, 8.0)
}

fn surface_tension(candidate: &CandidateScore) -> f32 {
    let gamma = 0.50 + 0.35 * candidate.mean_uncertainty - 0.30 * candidate.mean_coherence;
    (0.72 - gamma).clamp(-0.5, 0.8)
}

fn relational_pilot(candidate: &CandidateScore, quantum_potential: f32, guidance_flow: f32) -> f32 {
    let phase = phase_drive(candidate).max(0.0);
    (phase + guidance_flow - quantum_potential * 0.35).clamp(0.0, 1.0)
}

fn quantum_potential(laplacian: f32, amplitude: f32) -> f32 {
    let q = -laplacian / (amplitude + EPSILON);
    (q.abs() / (1.0 + q.abs())).clamp(0.0, 1.0)
}

fn micro_evolution(
    guidance_flow: f32,
    quantum_potential: f32,
    regge_cost: f32,
    causal_gate: f32,
) -> f32 {
    let config = RelationalGuidanceConfig::default();
    let mut gain = 1.0_f32;
    let ticks = config.microticks.max(1);
    for _ in 0..ticks {
        let guide = guidance_flow * (0.5 + causal_gate);
        let drag = quantum_potential * 0.35 + regge_cost * 0.25;
        gain += config.dt * (guide - drag) / ticks as f32;
    }
    gain.clamp(0.70, 1.60)
}

fn edge_tetrahedra_incidents(hardware: &CdtGraphitySubstrate) -> Vec<usize> {
    let mut incidents = vec![0_usize; hardware.edges.len()];
    let edge_lookup = hardware
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| edge.active)
        .map(|(idx, edge)| (ordered_pair(edge.a, edge.b), idx))
        .collect::<HashMap<_, _>>();
    for tetra in &hardware.tetrahedra {
        for i in 0..tetra.vertices.len() {
            for j in (i + 1)..tetra.vertices.len() {
                if let Some(&idx) =
                    edge_lookup.get(&ordered_pair(tetra.vertices[i], tetra.vertices[j]))
                {
                    incidents[idx] += 1;
                }
            }
        }
    }
    incidents
}

fn incident_edges(hardware: &CdtGraphitySubstrate) -> Vec<Vec<usize>> {
    let mut incident_edges = vec![Vec::new(); hardware.nodes.len()];
    for (idx, edge) in hardware.edges.iter().enumerate() {
        if !edge.active {
            continue;
        }
        if edge.a < incident_edges.len() {
            incident_edges[edge.a].push(idx);
        }
        if edge.b < incident_edges.len() {
            incident_edges[edge.b].push(idx);
        }
    }
    incident_edges
}

fn edge_conductance(kind: CdtGraphityEdgeKind, stability: f32, prediction_error: f32) -> f32 {
    let kind_factor = match kind {
        CdtGraphityEdgeKind::Spatial => 0.45,
        CdtGraphityEdgeKind::Temporal => 1.0,
    };
    kind_factor * stability * (1.0 - prediction_error).max(0.0)
}

fn phase_drive(candidate: &CandidateScore) -> f32 {
    let scale = candidate.interference.abs() + candidate.probability + EPSILON;
    (candidate.interference / scale).clamp(-1.0, 1.0)
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}
