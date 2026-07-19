//! Motor unificado: CDT simplicial → líquido de espines → RQM/EPR → cognición.
//!
//! La simetría guía la transferencia y el gate de consolidación, pero el
//! contenido cognitivo reside en relaciones aprendidas y su composición.

use crate::matrix_free_cognitive_substrate::LatentConceptId;
use crate::oxicuda_peps3d_backend::{PyrochlorePeps3dAdapter, PyrochlorePeps3dReport};
use crate::oxicuda_pyrochlore_backend::{
    solve_pyrochlore_dmrg, PyrochloreDmrgReport, PyrochloreMpoConfig,
};
use crate::pyrochlore_graph_tensor_network::{
    GraphTensorNetworkReport, PyrochloreGraphTensorNetwork,
};
use crate::quantum_spin_thermodynamic_engine::{
    periodic_pyrochlore_model, QuantumSpinConfig, QuantumSpinError, QuantumSpinReport,
    QuantumSpinThermodynamicEngine,
};
use crate::relational_field::ObserverId;
use crate::symmetry_guided_rqm_epr::{
    CognitiveInference, RelationalCognitiveConfig, RelationalCognitiveLayer, RqmLearningReport,
    SymmetryGuidedRqmConfig, SymmetryGuidedRqmEprField,
};
use crate::symmetry_thermodynamic_substrate::{
    SymmetrySubstrateError, SymmetryThermodynamicConfig,
};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct UnifiedSpinCognitiveConfig {
    pub quantum: QuantumSpinConfig,
    pub rqm: SymmetryGuidedRqmConfig,
    pub cognitive: RelationalCognitiveConfig,
    pub bootstrap_cooling_steps: usize,
    pub cooling_steps_per_observation: usize,
    pub real_steps_per_observation: usize,
    pub backreaction_rate: f64,
    pub minimum_topological_symmetry: f64,
    pub minimum_spin_entropy: f64,
    pub require_entanglement_witness: bool,
}

impl Default for UnifiedSpinCognitiveConfig {
    fn default() -> Self {
        Self {
            quantum: QuantumSpinConfig {
                spin_lattice_alpha: 0.0,
                max_spins: 16,
                ..QuantumSpinConfig::default()
            },
            rqm: SymmetryGuidedRqmConfig::default(),
            cognitive: RelationalCognitiveConfig::default(),
            bootstrap_cooling_steps: 300,
            cooling_steps_per_observation: 2,
            real_steps_per_observation: 1,
            backreaction_rate: 0.0,
            minimum_topological_symmetry: 0.99,
            minimum_spin_entropy: 0.10,
            require_entanglement_witness: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpinLiquidBackendKind {
    #[default]
    ExactStateVector,
    ExactWithOxiCudaMpo,
    ExactWithOxiCudaMpoAndPeps3d,
    ExactWithAllTensorBackends,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct KnowledgeKey {
    pub observer: usize,
    pub source: LatentConceptId,
    pub target: LatentConceptId,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConsolidatedKnowledge {
    pub key: KnowledgeKey,
    pub confidence: f64,
    pub topological_symmetry: f64,
    pub spin_entropy: f64,
    pub prediction_error: f64,
    pub consolidations: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UnifiedConsolidationGate {
    pub topology_pass: bool,
    pub spin_coherence_pass: bool,
    pub entanglement_pass: bool,
    pub relational_pass: bool,
    pub passed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UnifiedObservationReport {
    pub learning: RqmLearningReport,
    pub spin: QuantumSpinReport,
    pub topological_symmetry: f64,
    pub gate: UnifiedConsolidationGate,
    pub knowledge_count: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UnifiedEngineReport {
    pub backend: SpinLiquidBackendKind,
    pub spins: usize,
    pub tetrahedra: usize,
    pub physical_bonds: usize,
    pub topological_symmetry: f64,
    pub quantum: QuantumSpinReport,
    pub rqm_relations: usize,
    pub epr_links: usize,
    pub consolidated_knowledge: usize,
    pub tensor_network_energy: Option<f64>,
    pub tensor_network_mpo_bond_dimension: Option<usize>,
    pub peps3d_norm: Option<f64>,
    pub peps3d_entropy: Option<f64>,
    pub peps3d_nonlocal_bonds: Option<usize>,
    pub graph_tensor_bonds: Option<usize>,
    pub graph_tensor_entropy: Option<f64>,
    pub graph_tensor_contraction_cost: Option<usize>,
}

#[derive(Debug)]
pub enum UnifiedEngineError {
    Geometry(SymmetrySubstrateError),
    Quantum(QuantumSpinError),
}

impl fmt::Display for UnifiedEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(error) => write!(f, "error CDT: {error}"),
            Self::Quantum(error) => write!(f, "error spin: {error}"),
        }
    }
}

impl std::error::Error for UnifiedEngineError {}

impl From<SymmetrySubstrateError> for UnifiedEngineError {
    fn from(error: SymmetrySubstrateError) -> Self {
        Self::Geometry(error)
    }
}

impl From<QuantumSpinError> for UnifiedEngineError {
    fn from(error: QuantumSpinError) -> Self {
        Self::Quantum(error)
    }
}

#[derive(Clone, Debug)]
pub struct UnifiedSpinCognitiveEngine {
    pub spin_liquid: QuantumSpinThermodynamicEngine,
    pub cognition: RelationalCognitiveLayer,
    pub config: UnifiedSpinCognitiveConfig,
    pub knowledge: BTreeMap<KnowledgeKey, ConsolidatedKnowledge>,
    pub tensor_network: Option<PyrochloreDmrgReport>,
    pub peps3d: Option<PyrochlorePeps3dReport>,
    pub graph_tensor: Option<GraphTensorNetworkReport>,
}

impl UnifiedSpinCognitiveEngine {
    pub fn periodic_pyrochlore(
        nx: usize,
        ny: usize,
        nz: usize,
        config: UnifiedSpinCognitiveConfig,
    ) -> Result<Self, UnifiedEngineError> {
        let (geometry, bonds) = periodic_pyrochlore_model(
            nx,
            ny,
            nz,
            SymmetryThermodynamicConfig {
                temperature: 0.0,
                ..SymmetryThermodynamicConfig::default()
            },
        )?;
        let mut spin_liquid =
            QuantumSpinThermodynamicEngine::new_neel_with_bonds(geometry, bonds, config.quantum)?;
        spin_liquid.cool(config.bootstrap_cooling_steps);
        let workspace = SymmetryGuidedRqmEprField::new(config.rqm);
        Ok(Self {
            spin_liquid,
            cognition: RelationalCognitiveLayer::new(workspace, config.cognitive),
            config,
            knowledge: BTreeMap::new(),
            tensor_network: None,
            peps3d: None,
            graph_tensor: None,
        })
    }

    pub fn observe_relation(
        &mut self,
        observer: ObserverId,
        source: LatentConceptId,
        target: LatentConceptId,
        context_phase: f64,
        desired_strength: f64,
        orbit_confidence: f64,
        orbit: &[(LatentConceptId, LatentConceptId)],
    ) -> UnifiedObservationReport {
        let source_spin = source.0 % self.spin_liquid.spin_count();
        self.spin_liquid
            .apply_local_phase_pulse(source_spin, context_phase);
        if !self.spin_liquid.bonds.is_empty() {
            let bond = source.0 % self.spin_liquid.bonds.len();
            self.spin_liquid
                .apply_exchange_pulse(bond, self.spin_liquid.config.real_time_step);
        }
        for _ in 0..self.config.real_steps_per_observation {
            self.spin_liquid.real_time_step();
        }
        for _ in 0..self.config.cooling_steps_per_observation {
            self.spin_liquid.imaginary_time_step();
        }
        if self.config.backreaction_rate > 0.0 {
            self.spin_liquid
                .spin_lattice_backreaction(self.config.backreaction_rate);
        }

        let topological_symmetry = self.topological_symmetry();
        let effective_symmetry = orbit_confidence.clamp(0.0, 1.0) * topological_symmetry;
        let learning = self.cognition.workspace.learn_transition(
            observer,
            source,
            target,
            context_phase,
            desired_strength,
            effective_symmetry,
            orbit,
        );
        let spin = self.spin_liquid.report();
        let gate = self.consolidation_gate(learning.observed, spin, topological_symmetry);
        if gate.passed {
            self.store_knowledge(
                observer,
                source,
                target,
                learning.observed,
                spin,
                topological_symmetry,
            );
            for &(orbit_source, orbit_target) in orbit {
                if let Some(relation) =
                    self.cognition
                        .workspace
                        .relation(observer, orbit_source, orbit_target)
                {
                    let orbit_gate = self.consolidation_gate(relation, spin, topological_symmetry);
                    if orbit_gate.passed {
                        self.store_knowledge(
                            observer,
                            orbit_source,
                            orbit_target,
                            relation,
                            spin,
                            topological_symmetry,
                        );
                    }
                }
            }
        }
        UnifiedObservationReport {
            learning,
            spin,
            topological_symmetry,
            gate,
            knowledge_count: self.knowledge.len(),
        }
    }

    pub fn train_relation(
        &mut self,
        observer: ObserverId,
        source: LatentConceptId,
        target: LatentConceptId,
        context_phase: f64,
        desired_strength: f64,
        orbit_confidence: f64,
        orbit: &[(LatentConceptId, LatentConceptId)],
        exposures: usize,
    ) -> UnifiedObservationReport {
        let mut report = UnifiedObservationReport::default();
        for _ in 0..exposures.max(1) {
            report = self.observe_relation(
                observer,
                source,
                target,
                context_phase,
                desired_strength,
                orbit_confidence,
                orbit,
            );
        }
        report
    }

    pub fn infer(
        &self,
        observer: ObserverId,
        source: LatentConceptId,
        context_phase: f64,
        max_hops: usize,
    ) -> Option<CognitiveInference> {
        let inference = self
            .cognition
            .infer(observer, source, context_phase, max_hops)?;
        let all_consolidated = inference.path.windows(2).all(|pair| {
            self.knowledge.contains_key(&KnowledgeKey {
                observer: observer.0,
                source: pair[0],
                target: pair[1],
            })
        });
        all_consolidated.then_some(inference)
    }

    /// Calcula un snapshot DMRG del mismo Hamiltoniano pyrochlore mediante
    /// OxiCUDA. El estado exacto sigue activo para dinámica local; este backend
    /// sirve como sustituto escalable y referencia de energía.
    pub fn refresh_tensor_network(
        &mut self,
        config: PyrochloreMpoConfig,
    ) -> oxicuda_tn::error::TnResult<PyrochloreDmrgReport> {
        let (_, report) = solve_pyrochlore_dmrg(
            self.spin_liquid.spin_count(),
            &self.spin_liquid.bonds,
            config,
        )?;
        self.tensor_network = Some(report);
        Ok(report)
    }

    pub fn refresh_peps3d_product_scaffold(
        &mut self,
        nx: usize,
        ny: usize,
        nz: usize,
        boundary_bond: usize,
    ) -> oxicuda_tn::error::TnResult<PyrochlorePeps3dReport> {
        let state = (0..self.spin_liquid.spin_count())
            .map(|spin| spin % 2)
            .collect::<Vec<_>>();
        let adapter =
            PyrochlorePeps3dAdapter::product_state(nx, ny, nz, &state, &self.spin_liquid.bonds)?;
        let report = adapter.report(boundary_bond)?;
        self.peps3d = Some(report);
        Ok(report)
    }

    pub fn refresh_graph_tensor_network(
        &mut self,
    ) -> oxicuda_tn::error::TnResult<GraphTensorNetworkReport> {
        let network = PyrochloreGraphTensorNetwork::ghz(
            self.spin_liquid.spin_count(),
            &self.spin_liquid.bonds,
        )?;
        let report = network.report(self.spin_liquid.config.exchange)?;
        self.graph_tensor = Some(report);
        Ok(report)
    }

    pub fn report(&self) -> UnifiedEngineReport {
        UnifiedEngineReport {
            backend: if self.tensor_network.is_some()
                && self.peps3d.is_some()
                && self.graph_tensor.is_some()
            {
                SpinLiquidBackendKind::ExactWithAllTensorBackends
            } else if self.tensor_network.is_some() && self.peps3d.is_some() {
                SpinLiquidBackendKind::ExactWithOxiCudaMpoAndPeps3d
            } else if self.tensor_network.is_some() {
                SpinLiquidBackendKind::ExactWithOxiCudaMpo
            } else {
                SpinLiquidBackendKind::ExactStateVector
            },
            spins: self.spin_liquid.spin_count(),
            tetrahedra: self.spin_liquid.geometry.tetrahedra.len(),
            physical_bonds: self
                .spin_liquid
                .bonds
                .iter()
                .map(|bond| bond.multiplicity as usize)
                .sum(),
            topological_symmetry: self.topological_symmetry(),
            quantum: self.spin_liquid.report(),
            rqm_relations: self.cognition.workspace.relation_count(),
            epr_links: self.cognition.workspace.entanglement.active_count(),
            consolidated_knowledge: self.knowledge.len(),
            tensor_network_energy: self.tensor_network.map(|report| report.energy),
            tensor_network_mpo_bond_dimension: self
                .tensor_network
                .map(|report| report.mpo_bond_dimension),
            peps3d_norm: self.peps3d.map(|report| report.approximate_norm),
            peps3d_entropy: self.peps3d.map(|report| report.z_entanglement_entropy),
            peps3d_nonlocal_bonds: self.peps3d.map(|report| report.nonlocal_bonds),
            graph_tensor_bonds: self.graph_tensor.map(|report| report.represented_bonds),
            graph_tensor_entropy: self.graph_tensor.map(|report| report.single_spin_entropy),
            graph_tensor_contraction_cost: self.graph_tensor.map(|report| report.contraction_cost),
        }
    }

    pub fn topological_symmetry(&self) -> f64 {
        let spins = self.spin_liquid.spin_count();
        if spins == 0 {
            return 0.0;
        }
        let mut degree = vec![0.0; spins];
        for bond in &self.spin_liquid.bonds {
            degree[bond.a] += bond.multiplicity as f64;
            degree[bond.b] += bond.multiplicity as f64;
        }
        let target_degree = 6.0;
        let defect = degree
            .iter()
            .map(|degree| ((degree - target_degree) / target_degree).powi(2))
            .sum::<f64>()
            / spins as f64;
        (-defect).exp()
    }

    fn consolidation_gate(
        &self,
        relation: crate::symmetry_guided_rqm_epr::RqmPhaseRelationState,
        spin: QuantumSpinReport,
        topological_symmetry: f64,
    ) -> UnifiedConsolidationGate {
        let topology_pass = topological_symmetry >= self.config.minimum_topological_symmetry;
        let spin_coherence_pass = spin.mean_single_spin_entropy >= self.config.minimum_spin_entropy;
        let entanglement_pass =
            !self.config.require_entanglement_witness || spin.entangled_edges > 0;
        let relational_pass = relation.consolidated;
        UnifiedConsolidationGate {
            topology_pass,
            spin_coherence_pass,
            entanglement_pass,
            relational_pass,
            passed: topology_pass && spin_coherence_pass && entanglement_pass && relational_pass,
        }
    }

    fn store_knowledge(
        &mut self,
        observer: ObserverId,
        source: LatentConceptId,
        target: LatentConceptId,
        relation: crate::symmetry_guided_rqm_epr::RqmPhaseRelationState,
        spin: QuantumSpinReport,
        topological_symmetry: f64,
    ) {
        let key = KnowledgeKey {
            observer: observer.0,
            source,
            target,
        };
        let existing = self.knowledge.get(&key).copied().unwrap_or_default();
        self.knowledge.insert(
            key,
            ConsolidatedKnowledge {
                key,
                confidence: relation.amplitude * relation.coherence,
                topological_symmetry,
                spin_entropy: spin.mean_single_spin_entropy,
                prediction_error: relation.prediction_error,
                consolidations: existing.consolidations + 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> UnifiedSpinCognitiveEngine {
        UnifiedSpinCognitiveEngine::periodic_pyrochlore(
            2,
            1,
            1,
            UnifiedSpinCognitiveConfig {
                bootstrap_cooling_steps: 120,
                cooling_steps_per_observation: 1,
                ..UnifiedSpinCognitiveConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn unified_engine_has_regular_cdt_and_entangled_spin_backend() {
        let engine = engine();
        let report = engine.report();
        assert_eq!(report.spins, 8);
        assert_eq!(report.physical_bonds, 24);
        assert!((report.topological_symmetry - 1.0).abs() < 1.0e-12);
        assert!(report.quantum.mean_single_spin_entropy > 0.1);
        assert!(report.quantum.entangled_edges > 0);
    }

    #[test]
    fn symmetry_without_relations_does_not_create_knowledge() {
        let engine = engine();
        assert!(engine.knowledge.is_empty());
        assert!(engine
            .infer(ObserverId(1), LatentConceptId(0), 0.0, 3)
            .is_none());
    }

    #[test]
    fn abstract_relations_consolidate_and_compose_over_spin_liquid() {
        let mut engine = engine();
        let observer = ObserverId(2);
        let first = engine.train_relation(
            observer,
            LatentConceptId(0),
            LatentConceptId(1),
            0.2,
            1.0,
            0.0,
            &[],
            40,
        );
        assert!(first.gate.passed, "{first:?}");
        let second = engine.train_relation(
            observer,
            LatentConceptId(1),
            LatentConceptId(2),
            0.2,
            1.0,
            0.0,
            &[],
            40,
        );
        assert!(second.gate.passed, "{second:?}");
        let inference = engine.infer(observer, LatentConceptId(0), 0.2, 2).unwrap();
        assert_eq!(
            inference.path,
            vec![LatentConceptId(0), LatentConceptId(1), LatentConceptId(2)]
        );
    }

    #[test]
    fn broken_cdt_symmetry_blocks_knowledge_consolidation() {
        let mut engine = engine();
        engine.spin_liquid.bonds.pop();
        let report = engine.train_relation(
            ObserverId(3),
            LatentConceptId(0),
            LatentConceptId(1),
            0.0,
            1.0,
            1.0,
            &[],
            40,
        );
        assert!(!report.gate.topology_pass);
        assert!(!report.gate.passed);
        assert!(engine.knowledge.is_empty());
    }

    #[test]
    fn oxicuda_mpo_snapshot_attaches_to_unified_engine() {
        let mut engine = UnifiedSpinCognitiveEngine::periodic_pyrochlore(
            1,
            1,
            1,
            UnifiedSpinCognitiveConfig {
                bootstrap_cooling_steps: 60,
                ..UnifiedSpinCognitiveConfig::default()
            },
        )
        .unwrap();
        let tensor = engine
            .refresh_tensor_network(PyrochloreMpoConfig {
                initial_bond_dimension: 4,
                dmrg: oxicuda_tn::dmrg::dmrg::DmrgConfig {
                    max_sweeps: 8,
                    chi_max: 16,
                    trunc_tol: 1.0e-10,
                    energy_tol: 1.0e-9,
                    lanczos_iter: 32,
                    lanczos_tol: 1.0e-10,
                },
                ..PyrochloreMpoConfig::default()
            })
            .unwrap();
        assert!((tensor.energy + 3.0).abs() < 1.0e-6);
        let report = engine.report();
        assert_eq!(report.backend, SpinLiquidBackendKind::ExactWithOxiCudaMpo);
        assert_eq!(report.tensor_network_energy, Some(tensor.energy));
        let peps = engine.refresh_peps3d_product_scaffold(1, 1, 1, 4).unwrap();
        assert_eq!(peps.pyrochlore_sites, 4);
        let report = engine.report();
        assert_eq!(
            report.backend,
            SpinLiquidBackendKind::ExactWithOxiCudaMpoAndPeps3d
        );
        assert_eq!(report.peps3d_norm, Some(1.0));
        let graph = engine.refresh_graph_tensor_network().unwrap();
        assert_eq!(graph.represented_bonds, 12);
        let report = engine.report();
        assert_eq!(
            report.backend,
            SpinLiquidBackendKind::ExactWithAllTensorBackends
        );
        assert_eq!(report.graph_tensor_bonds, Some(12));
    }
}
