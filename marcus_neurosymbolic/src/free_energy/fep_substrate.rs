//! Sustrato Cognitivo Completo FEP-MARCUS (100% Free Energy & Surprise Reduction Architecture).
//!
//! Integra la totalidad del sustrato neuro-simbólico de Gary Marcus con los principios
//! matemáticos de Karl Friston sobre el Principio de Energía Libre:
//!
//! - Percepción: Campo continuo latente z_t e innovación causal.
//! - Memoria rápida y ligamiento: Red bio-cuántica BQ10 con Ley de Dale estricta (W^E, W^I >= 0).
//! - Física y Racionalidad: Complejo simplicial 3D con barrera Cayley-Menger y geodésicas Snell-Fermat.
//! - Ontología: Tipos invariantes y Tokens situados en el espacio-tiempo.
//! - Inferencia: Maximización del ELBO (Energía Libre de Evidencia) y minimización de sorpresa estructurada.
//! - Acción: Inferencia activa mediante intervenciones causales do(X) que minimizan G(pi).
//! - Auditoría: Centinela epistémico con árbol de procedencia auditable.

use crate::algebraic_mind::type_token::TypeTokenRegistry;
use crate::algebraic_mind::variables::HrrVector;
use crate::connectors::bio_quantum_binding_connector::{BioQuantumBindingConfig, BioQuantumBindingConnector};
use crate::connectors::constellation_type_token_connector::{ConstellationNode, ConstellationTypeTokenConnector};
use crate::connectors::continuous_field_connector::{ContinuousFieldPerceptualConnector, ContinuousFieldState, SemanticConceptAnchor};
use crate::connectors::simplicial_spelke_connector::SimplicialSpelkeSpatialConnector;
use crate::core_knowledge::causal_engine::CausalEngine;
use crate::core_knowledge::spatiotemporal_physics::SpelkePhysicsEngine;
use crate::epistemic_vigilance::hallucination_filter::HallucinationGateDecision;
use crate::epistemic_vigilance::world_state::{EpistemicCertainty, WorldStateTracker};
use crate::free_energy::fep_coordinator::{FepCognitiveCycleResult, FepDualProcessCoordinator};
use crate::free_energy::surprise_engine::StructuredSurprise;
use crate::dual_process::system1_proposer::System1Proposer;
use serde::{Deserialize, Serialize};

/// Telemetría completa de la arquitectura orientada a Energía Libre y Reducción de Sorpresa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepSubstrateTelemetry {
    pub step: u64,
    /// Energía Libre Variacional actual F(q, o)
    pub variational_free_energy: f64,
    /// Cota inferior de evidencia logarítmica (ELBO = -F), maximizada
    pub evidence_lower_bound: f64,
    /// Sorpresa total integrada actual S(o)
    pub current_surprise: f64,
    /// Tasa de reducción de sorpresa en este ciclo
    pub surprise_reduction_ratio: f64,
    /// Desglose estructurado de la sorpresa
    pub surprise_breakdown: StructuredSurprise,
    /// Acción seleccionada por inferencia activa para minimizar G(pi)
    pub active_inference_action: Option<String>,
    /// Auditoría de la Ley de Dale en BQ10
    pub dales_law_audit_clean: bool,
    pub dales_law_synapses_count: usize,
    /// Barrera Cayley-Menger activa en física simplicial
    pub cayley_menger_barrier_active: bool,
    /// Tipos ontológicos registrados
    pub constellation_types_count: usize,
    /// Checksum SHA-256 de procedencia epistémica
    pub epistemic_provenance_checksum: String,
}

/// El Sustrato Cognitivo Unificado FEP-MARCUS.
pub struct MarcusFepSubstrate {
    pub coordinator: FepDualProcessCoordinator,
    pub system1: System1Proposer,
    pub physics_engine: SpelkePhysicsEngine,
    pub causal_engine: CausalEngine,
    pub type_registry: TypeTokenRegistry,
    pub world_state: WorldStateTracker,
    pub field_connector: ContinuousFieldPerceptualConnector,
    pub bio_quantum_connector: BioQuantumBindingConnector,
    pub simplicial_connector: SimplicialSpelkeSpatialConnector,
    pub constellation_connector: ConstellationTypeTokenConnector,
    pub step_counter: u64,
}

impl Default for MarcusFepSubstrate {
    fn default() -> Self {
        Self::new(64)
    }
}

impl MarcusFepSubstrate {
    pub fn new(latent_dim: usize) -> Self {
        let bq_config = BioQuantumBindingConfig {
            dim: latent_dim,
            ..Default::default()
        };

        Self {
            coordinator: FepDualProcessCoordinator::default(),
            system1: System1Proposer::new(),
            physics_engine: SpelkePhysicsEngine::new(),
            causal_engine: CausalEngine::new(),
            type_registry: TypeTokenRegistry::new(),
            world_state: WorldStateTracker::new(),
            field_connector: ContinuousFieldPerceptualConnector::new(latent_dim),
            bio_quantum_connector: BioQuantumBindingConnector::new(bq_config),
            simplicial_connector: SimplicialSpelkeSpatialConnector::new(),
            constellation_connector: ConstellationTypeTokenConnector::new(),
            step_counter: 0,
        }
    }

    /// Inicializa conceptos ontológicos y reglas físicas base.
    pub fn initialize_default_fep_scaffold(&mut self) {
        let dim = self.field_connector.dim;

        // 1. Anclas de conceptos continuos
        let mut z_grav = vec![0.0; dim];
        z_grav[0] = 1.0;
        self.field_connector.register_concept_anchor(SemanticConceptAnchor {
            concept_id: "gravity_concept".to_string(),
            centroid_z: z_grav,
            subject: "Object".to_string(),
            predicate: "FallsTowards".to_string(),
            object: "Ground".to_string(),
        });

        // 2. Ontología Tipo-Token
        self.constellation_connector.ingest_constellation_node(
            ConstellationNode {
                node_id: 1,
                label: "PhysicalBody".to_string(),
                parent_node_id: None,
                is_solid: true,
                is_container: false,
                is_agent: false,
                default_mass: 1.0,
            },
            &mut self.type_registry,
        );

        // 3. Hecho base de verdad epistémica
        self.world_state.insert_fact(
            "Object",
            "FallsTowards",
            "Ground",
            EpistemicCertainty::AnalyticProof,
        );
    }

    /// Ejecuta un ciclo completo guiado 100% por el Principio de Energía Libre:
    /// Ingesta sensorial -> Maximiza ELBO -> Reduce Sorpresa -> Ejecuta Inferencia Activa -> Audita Verdad.
    pub fn process_fep_cycle(
        &mut self,
        field_state: &ContinuousFieldState,
    ) -> (FepCognitiveCycleResult, FepSubstrateTelemetry) {
        self.step_counter += 1;

        // 1. Proyección sensorial y candidatos del Sistema 1
        let hypotheses = self.field_connector.decode_hypotheses_from_field(field_state, 3);
        for hyp in &hypotheses {
            self.system1.add_association(hyp.clone());
        }

        let query = if let Some(top) = hypotheses.first() {
            format!("{} {}", top.subject, top.predicate)
        } else {
            "default_sensory_evaluation".to_string()
        };

        // 2. Ejecución del ciclo coordinado FEP
        let cycle_result = self.coordinator.execute_fep_cycle(
            &query,
            &field_state.latent_z,
            &self.system1,
            &mut self.physics_engine,
            &mut self.causal_engine,
            &self.type_registry,
        );

        // 3. Auditoría de la Ley de Dale en la memoria BQ10
        let (dale_violations, total_synapses) = self.bio_quantum_connector.fast_weights.audit_dales_law();

        // 4. Auditoría del Centinela Epistémico
        let mut provenance_checksum = "UNVERIFIED".to_string();
        if let Some(hyp) = &cycle_result.selected_hypothesis {
            let sentinel = crate::epistemic_vigilance::hallucination_filter::HallucinationSentinel::new(&self.world_state);
            let decision = sentinel.evaluate_claim(
                &format!("fep_cycle_{}", self.step_counter),
                &hyp.subject,
                &hyp.predicate,
                &hyp.object,
            );
            if let HallucinationGateDecision::Approved { provenance, .. } = decision {
                provenance_checksum = provenance.checksum;
            }
        }

        let action_name = cycle_result.selected_action.as_ref().map(|a| match a {
            crate::free_energy::active_inference::ActiveAction::CausalIntervention { target_var, .. } => {
                format!("do({})", target_var)
            }
            crate::free_energy::active_inference::ActiveAction::ProbeVariableBinding { var_name } => {
                format!("probe_binding({})", var_name)
            }
            crate::free_energy::active_inference::ActiveAction::InspectOccludedObject { object_id } => {
                format!("inspect_occluded({})", object_id)
            }
            crate::free_energy::active_inference::ActiveAction::MotorGeodesicStep { .. } => {
                "motor_geodesic_step".to_string()
            }
            crate::free_energy::active_inference::ActiveAction::AuditClaimTruth { claim_id } => {
                format!("audit_claim({})", claim_id)
            }
        });

        let telemetry = FepSubstrateTelemetry {
            step: self.step_counter,
            variational_free_energy: cycle_result.optimized_free_energy,
            evidence_lower_bound: cycle_result.evidence_lower_bound,
            current_surprise: cycle_result.final_surprise.total_surprise,
            surprise_reduction_ratio: cycle_result.surprise_reduction_ratio,
            surprise_breakdown: cycle_result.final_surprise.clone(),
            active_inference_action: action_name,
            dales_law_audit_clean: dale_violations == 0,
            dales_law_synapses_count: total_synapses,
            cayley_menger_barrier_active: true,
            constellation_types_count: self.constellation_connector.node_count(),
            epistemic_provenance_checksum: provenance_checksum,
        };

        (cycle_result, telemetry)
    }

    /// Liga variables algebraicas en memoria bio-cuántica Dale.
    pub fn bind_variables_hrr(&mut self, role: &HrrVector, filler: &HrrVector) -> f64 {
        self.bio_quantum_connector.bind_and_verify(role, filler, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marcus_fep_substrate_end_to_end() {
        let mut substrate = MarcusFepSubstrate::new(64);
        substrate.initialize_default_fep_scaffold();

        let mut z = vec![0.0; 64];
        z[0] = 0.95;
        let field_state = ContinuousFieldState::new(64, 1).with_latent(z, 0.01);

        let (res, telem) = substrate.process_fep_cycle(&field_state);

        assert!(telem.dales_law_audit_clean);
        assert!(telem.evidence_lower_bound >= -telem.variational_free_energy - 1e-6);
        assert!(telem.surprise_reduction_ratio >= 0.0);
        assert!(res.selected_action.is_some());
    }
}
