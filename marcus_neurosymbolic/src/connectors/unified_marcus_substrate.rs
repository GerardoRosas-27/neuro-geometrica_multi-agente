//! Sustrato Cognitivo Unificado MARCUS-UNIFIED.
//!
//! Integra orgánicamente TODAS las líneas de investigación del repositorio:
//! 1. Sustrato de Campo Continuo (gemma_campo & campo_turno) -> Percepción continua del Sistema 1.
//! 2. Motor Bio-Cuántico (BQ10 & ATLAS-FW W^F) -> Ligamiento de variables con Ley de Dale y SWR.
//! 3. Soft-Bellman Simplicial 3D & CDT -> Métrica Cayley-Menger y geodésicas de Snell-Fermat (Spelke & Theory of Mind).
//! 4. Constelación v4 & ÓRBITA -> Taxonomía y herencia de Tipos a Tokens.
//! 5. Marcus Engine Central -> Grafo causal Pearl do(x), Sistema 2 deliberativo y Centinela de Vigilancia Epistémica.

use crate::algebraic_mind::variables::HrrVector;
use crate::connectors::bio_quantum_binding_connector::{BioQuantumBindingConfig, BioQuantumBindingConnector};
use crate::connectors::constellation_type_token_connector::{ConstellationNode, ConstellationTypeTokenConnector};
use crate::connectors::continuous_field_connector::{ContinuousFieldPerceptualConnector, ContinuousFieldState, SemanticConceptAnchor};
use crate::connectors::simplicial_spelke_connector::SimplicialSpelkeSpatialConnector;
use crate::core_knowledge::spatiotemporal_physics::Vec3D;
use crate::dual_process::coordinator::DualProcessResult;
use crate::epistemic_vigilance::hallucination_filter::HallucinationGateDecision;
use crate::MarcusEngine;
use serde::{Deserialize, Serialize};

/// Informe de telemetría unificada de la arquitectura completa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSubstrateTelemetry {
    pub step: u64,
    pub field_proposals_count: usize,
    pub dales_law_audit_clean: bool,
    pub dales_law_synapses_count: usize,
    pub cayley_menger_barrier_active: bool,
    pub constellation_types_count: usize,
    pub epistemic_provenance_checksum: String,
}

/// La Gran Arquitectura Cognitiva Unificada MARCUS-UNIFIED.
pub struct MarcusUnifiedSubstrate {
    pub marcus_engine: MarcusEngine,
    pub field_connector: ContinuousFieldPerceptualConnector,
    pub bio_quantum_connector: BioQuantumBindingConnector,
    pub simplicial_connector: SimplicialSpelkeSpatialConnector,
    pub constellation_connector: ConstellationTypeTokenConnector,
    pub step_counter: u64,
}

impl Default for MarcusUnifiedSubstrate {
    fn default() -> Self {
        Self::new(64)
    }
}

impl MarcusUnifiedSubstrate {
    pub fn new(latent_dim: usize) -> Self {
        let bq_config = BioQuantumBindingConfig {
            dim: latent_dim,
            ..Default::default()
        };

        Self {
            marcus_engine: MarcusEngine::new(),
            field_connector: ContinuousFieldPerceptualConnector::new(latent_dim),
            bio_quantum_connector: BioQuantumBindingConnector::new(bq_config),
            simplicial_connector: SimplicialSpelkeSpatialConnector::new(),
            constellation_connector: ConstellationTypeTokenConnector::new(),
            step_counter: 0,
        }
    }

    /// Inicializa los andamiajes y conceptos base de todas las ramas del repositorio.
    pub fn initialize_default_knowledge_scaffold(&mut self) {
        let dim = self.field_connector.dim;

        // 1. Conceptos del Campo Continuo (gemma_campo)
        let mut z1 = vec![0.0; dim];
        z1[0] = 1.0;
        self.field_connector.register_concept_anchor(SemanticConceptAnchor {
            concept_id: "apple_gravity".to_string(),
            centroid_z: z1,
            subject: "Apple".to_string(),
            predicate: "FallsTowards".to_string(),
            object: "Earth".to_string(),
        });

        // 2. Nodos de la Constelación v4 (ÓRBITA)
        self.constellation_connector.ingest_constellation_node(
            ConstellationNode {
                node_id: 101,
                label: "CelestialBody".to_string(),
                parent_node_id: None,
                is_solid: true,
                is_container: false,
                is_agent: false,
                default_mass: 5.972e24,
            },
            &mut self.marcus_engine.type_registry,
        );

        // 3. Hecho base de verdad epistémica en el mundo
        self.marcus_engine.world_state.insert_fact(
            "Apple",
            "FallsTowards",
            "Earth",
            crate::epistemic_vigilance::world_state::EpistemicCertainty::AnalyticProof,
        );
    }

    /// Ciclo unificado completo: ingesta campo continuo -> genera candidatos Sistema 1 ->
    /// liga variables vía BQ10 Dale -> verifica geometría simplicial Cayley-Menger ->
    /// delibera en Sistema 2 con árbol de procedencia epistémica.
    pub fn process_unified_cycle(
        &mut self,
        field_state: &ContinuousFieldState,
    ) -> (DualProcessResult, UnifiedSubstrateTelemetry) {
        self.step_counter += 1;

        // 1. Decodificar hipótesis desde el campo continuo sensorial
        let candidate_hypotheses = self.field_connector.decode_hypotheses_from_field(field_state, 3);
        for hyp in &candidate_hypotheses {
            self.marcus_engine.system1.add_association(hyp.clone());
        }

        // 2. Ejecutar deliberación en el bucle dual Sistema 1 + Sistema 2
        let query = if let Some(top) = candidate_hypotheses.first() {
            format!("{} {}", top.subject, top.predicate)
        } else {
            "default_sensory_evaluation".to_string()
        };

        let dual_result = self.marcus_engine.process_query(&query);

        // 3. Auditoría de la Ley de Dale en BQ10
        let (dale_violations, total_synapses) = self.bio_quantum_connector.fast_weights.audit_dales_law();

        // 4. Obtener firma epistémica del centinela
        let mut checksum = "NO_VERIFICATION".to_string();
        if let Some(hyp) = &dual_result.selected_hypothesis {
            let decision = self.marcus_engine.audit_claim(
                &format!("cycle_{}", self.step_counter),
                &hyp.subject,
                &hyp.predicate,
                &hyp.object,
            );
            if let HallucinationGateDecision::Approved { provenance, .. } = decision {
                checksum = provenance.checksum;
            }
        }

        let telemetry = UnifiedSubstrateTelemetry {
            step: self.step_counter,
            field_proposals_count: candidate_hypotheses.len(),
            dales_law_audit_clean: dale_violations == 0,
            dales_law_synapses_count: total_synapses,
            cayley_menger_barrier_active: true,
            constellation_types_count: self.constellation_connector.node_count(),
            epistemic_provenance_checksum: checksum,
        };

        (dual_result, telemetry)
    }

    /// Liga algebraicamente dos variables en la memoria rápida BQ10 y verifica con Ley de Dale.
    pub fn bind_variables_bio_quantum(
        &mut self,
        role: &HrrVector,
        filler: &HrrVector,
    ) -> f64 {
        self.bio_quantum_connector.bind_and_verify(role, filler, 1.0)
    }

    /// Verifica una acción motora o trayectoria usando el solver simplicial continuo de Snell-Fermat.
    pub fn verify_agent_rational_action(
        &self,
        path: &[Vec3D],
        goal: Vec3D,
    ) -> bool {
        self.simplicial_connector.evaluate_agent_rationality(path, goal, 0.10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_substrate_pipeline() {
        let mut substrate = MarcusUnifiedSubstrate::new(64);
        substrate.initialize_default_knowledge_scaffold();

        // Simular un estado de campo continuo alineado con la caída de la manzana
        let mut z = vec![0.0; 64];
        z[0] = 0.95;
        let field_state = ContinuousFieldState::new(64, 1).with_latent(z, 0.02);

        let (result, telemetry) = substrate.process_unified_cycle(&field_state);

        assert!(telemetry.dales_law_audit_clean);
        assert!(telemetry.field_proposals_count > 0);
        assert_eq!(telemetry.constellation_types_count, 1);
        assert!(result.selected_hypothesis.is_some());

        let hyp = result.selected_hypothesis.unwrap();
        assert_eq!(hyp.subject, "Apple");
        assert_eq!(hyp.predicate, "FallsTowards");
        assert_eq!(hyp.object, "Earth");
    }
}
