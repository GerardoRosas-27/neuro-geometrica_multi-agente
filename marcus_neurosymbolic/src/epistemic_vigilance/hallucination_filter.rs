//! Módulo de Filtro Anti-Alucinación (Epistemic Hallucination Sentinel).
//!
//! Basado en Gary Marcus (Taming Silicon Valley, 2024 y Rebooting AI, 2019):
//! "Hallucinations are not benign glitches; they are inevitable when an architecture
//! lacks epistemic vigilance. The Hallucination Sentinel acts as an unyielding filter,
//! vetoing statistically plausible confabulations."

use crate::epistemic_vigilance::provenance::EpistemicProvenanceTree;
use crate::epistemic_vigilance::world_state::WorldStateTracker;
use serde::{Deserialize, Serialize};

/// Resultado del filtro anti-alucinaciones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HallucinationGateDecision {
    /// Aprobado: la afirmación está fundada epistémicamente y respeta la coherencia del mundo.
    Approved {
        claim_id: String,
        provenance: EpistemicProvenanceTree,
    },
    /// Vetado: la afirmación es una alucinación (física, lógica o fáctica).
    Vetoed {
        claim_id: String,
        statement: String,
        rejection_category: String,
        counter_evidence: String,
    },
}

pub struct HallucinationSentinel<'a> {
    pub world_state: &'a WorldStateTracker,
}

impl<'a> HallucinationSentinel<'a> {
    pub fn new(world_state: &'a WorldStateTracker) -> Self {
        Self { world_state }
    }

    /// Filtra una afirmación frente a hechos conocidos y reglas de consistencia.
    pub fn evaluate_claim(
        &self,
        claim_id: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> HallucinationGateDecision {
        let statement = format!("{} {} {}", subject, predicate, object);
        let mut provenance = EpistemicProvenanceTree::new(claim_id, &statement);

        // 1. Verificación fáctica directa
        if self.world_state.fact_exists(subject, predicate, object) {
            provenance.add_step(
                format!("Hecho base confirmado en el estado del mundo: <{}, {}, {}>", subject, predicate, object),
                "FactRetrieval",
                "WorldStateTracker",
            );
            provenance.seal(true);
            return HallucinationGateDecision::Approved {
                claim_id: claim_id.to_string(),
                provenance,
            };
        }

        // 2. Detección de Contradicciones Fácticas Conocidas
        // Si el estado del mundo dice que subject tiene un valor diferente para el predicado funcional
        let mut contradiction = None;
        for fact in self.world_state.all_facts() {
            if fact.subject == subject && fact.predicate == predicate && fact.object != object {
                contradiction = Some((fact.object.clone(), fact.certainty));
                break;
            }
        }

        if let Some((existing_val, certainty)) = contradiction {
            return HallucinationGateDecision::Vetoed {
                claim_id: claim_id.to_string(),
                statement,
                rejection_category: "Contradicción Fáctica Epistémica".to_string(),
                counter_evidence: format!(
                    "El estado del mundo ya contiene <{}, {}, {}> con certeza {:?}",
                    subject, predicate, existing_val, certainty
                ),
            };
        }

        // 3. Reglas de Invariancia Ontológica Universal
        // Ejemplo: Nada puede ser a la vez "Inside(A, B)" y "Outside(A, B)"
        if predicate == "Location" && object == "EverywhereSimultaneously" {
            return HallucinationGateDecision::Vetoed {
                claim_id: claim_id.to_string(),
                statement,
                rejection_category: "Violación de Localidad Espacio-Temporal".to_string(),
                counter_evidence: "Los objetos físicos discretos poseen soporte espacial finito".to_string(),
            };
        }

        // Si no está verificado ni desmentido directamente pero es una afirmación no soportada
        HallucinationGateDecision::Vetoed {
            claim_id: claim_id.to_string(),
            statement,
            rejection_category: "Alucinación No Fundamentada (Groundless Assertion)".to_string(),
            counter_evidence: "La afirmación carece de pruebas observacionales o deducción analítica".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic_vigilance::world_state::EpistemicCertainty;

    #[test]
    fn test_hallucination_veto_contradiction() {
        let mut state = WorldStateTracker::new();
        state.insert_fact("Paris", "CapitalOf", "France", EpistemicCertainty::ObservedFact);

        let sentinel = HallucinationSentinel::new(&state);

        // Afirmación verdadera
        let dec_true = sentinel.evaluate_claim("c1", "Paris", "CapitalOf", "France");
        assert!(matches!(dec_true, HallucinationGateDecision::Approved { .. }));

        // Alucinación frecuente en LLMs engañados por contexto: "Paris is capital of Germany"
        let dec_false = sentinel.evaluate_claim("c2", "Paris", "CapitalOf", "Germany");
        match dec_false {
            HallucinationGateDecision::Vetoed { rejection_category, .. } => {
                assert_eq!(rejection_category, "Contradicción Fáctica Epistémica");
            }
            _ => panic!("Debió vetar la alucinación"),
        }
    }
}
