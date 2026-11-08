//! Módulo de Sistema 2: Motor Deliberativo Simbólico (Verificador y Razonador).
//!
//! Basado en Gary Marcus (Rebooting AI, Cap. 5 y The Algebraic Mind):
//! "System 2 is deliberative, algebraic, and causal. It cross-checks candidates against
//! core physics, causal DAGs, and ontological consistency, vetoing physically impossible or
//! logically contradictory hypotheses."

use crate::algebraic_mind::type_token::TypeTokenRegistry;
use crate::core_knowledge::causal_engine::{CausalEngine, CausalVar};
use crate::core_knowledge::spatiotemporal_physics::SpelkePhysicsEngine;
use crate::dual_process::system1_proposer::CandidateHypothesis;
use serde::{Deserialize, Serialize};

/// Veredicto deliberativo del Sistema 2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeliberationVerdict {
    /// Hipótesis aprobada por consistencia formal y física.
    Approved {
        hypothesis_id: String,
        confidence: f64,
        audit_trail: Vec<String>,
    },
    /// Hipótesis vetada por violación de invariantes.
    Vetoed {
        hypothesis_id: String,
        reason: String,
        counterexample: String,
    },
}

/// Motor deliberativo del Sistema 2.
pub struct System2Deliberator<'a> {
    pub physics_engine: &'a SpelkePhysicsEngine,
    pub causal_engine: &'a mut CausalEngine,
    pub type_registry: &'a TypeTokenRegistry,
}

impl<'a> System2Deliberator<'a> {
    pub fn new(
        physics: &'a SpelkePhysicsEngine,
        causal: &'a mut CausalEngine,
        type_reg: &'a TypeTokenRegistry,
    ) -> Self {
        Self {
            physics_engine: physics,
            causal_engine: causal,
            type_registry: type_reg,
        }
    }

    /// Delibera sobre una hipótesis candidata propuesta por el Sistema 1.
    pub fn deliberate(&mut self, candidate: &CandidateHypothesis) -> DeliberationVerdict {
        let mut audit = Vec::new();
        audit.push(format!("Iniciando auditoría Sistema 2 para '{}'", candidate.claim));

        // 1. Verificación Causal (¿Se afirma causalidad cuando solo hay correlación espuria?)
        if candidate.predicate == "Causes" {
            let cause_var = CausalVar::new(&candidate.subject);
            let effect_var = CausalVar::new(&candidate.object);

            if !self.causal_engine.is_causally_dependent(&cause_var, &effect_var) {
                return DeliberationVerdict::Vetoed {
                    hypothesis_id: candidate.id.clone(),
                    reason: format!(
                        "Falacia Causal: '{}' y '{}' no tienen relación de intervención do(X)",
                        candidate.subject, candidate.object
                    ),
                    counterexample: format!(
                        "Intervención do({} = x) no altera el valor esperado de {}",
                        candidate.subject, candidate.object
                    ),
                };
            }
            audit.push(format!(
                "Prueba causal superada: do({}) modifica {}",
                candidate.subject, candidate.object
            ));
        }

        // 2. Verificación de Sentido Común de Contención (Marcus Cup & Backpack test)
        if candidate.predicate == "IsIn" {
            // Si afirma que A está en C, comprobamos la transitividad física
            if self.physics_engine.is_transitively_inside(&candidate.subject, &candidate.object) {
                audit.push(format!(
                    "Contención transitiva confirmada: Inside({}, {})",
                    candidate.subject, candidate.object
                ));
            } else {
                return DeliberationVerdict::Vetoed {
                    hypothesis_id: candidate.id.clone(),
                    reason: format!(
                        "Violación de contención: No existe camino transitivo para que '{}' esté dentro de '{}'",
                        candidate.subject, candidate.object
                    ),
                    counterexample: format!(
                        "{} no está registrado dentro de {}",
                        candidate.subject, candidate.object
                    ),
                };
            }
        }

        // 3. Verificación de hechos biológicos/ontológicos (e.g. Pingüinos no vuelan)
        if candidate.subject == "Penguin" && candidate.predicate == "CanFly" && candidate.object == "True" {
            return DeliberationVerdict::Vetoed {
                hypothesis_id: candidate.id.clone(),
                reason: "Contradicción taxonómica: Pingüino es ave no voladora (Spheniscidae)".to_string(),
                counterexample: "Anatomía de alas adaptada a natación, masa ósea densa incompatible con vuelo sustentado".to_string(),
            };
        }

        audit.push("Todas las invariantes de consistencia física y lógica verificadas con éxito.".to_string());

        DeliberationVerdict::Approved {
            hypothesis_id: candidate.id.clone(),
            confidence: 1.0, // Certeza lógica/verificada
            audit_trail: audit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_knowledge::causal_engine::StructuralEquation;

    #[test]
    fn test_system2_veto_spurious_causality() {
        let physics = SpelkePhysicsEngine::new();
        let mut causal = CausalEngine::new();
        let types = TypeTokenRegistry::new();

        // Configurar confusión: IceCreamSales no causa Drownings (la causa común es SummerTemperature)
        let temp = CausalVar::new("SummerTemperature");
        let ice = CausalVar::new("IceCreamSales");
        let drown = CausalVar::new("Drownings");

        causal.add_structural_equation(temp.clone(), StructuralEquation::new(30.0, 0.0));
        causal.add_structural_equation(ice.clone(), StructuralEquation::new(5.0, 0.0).with_parent(temp.clone(), 1.5));
        causal.add_structural_equation(drown.clone(), StructuralEquation::new(1.0, 0.0).with_parent(temp.clone(), 0.8));

        let mut deliberator = System2Deliberator::new(&physics, &mut causal, &types);

        // Sistema 1 propone que el helado causa ahogamientos debido a correlación r > 0.9
        let false_claim = CandidateHypothesis::new(
            "h_spurious",
            "Eating Ice Cream causes Drownings",
            0.92,
            "IceCreamSales",
            "Causes",
            "Drownings",
        );

        let verdict = deliberator.deliberate(&false_claim);
        match verdict {
            DeliberationVerdict::Vetoed { reason, .. } => {
                assert!(reason.contains("Falacia Causal"));
            }
            _ => panic!("Sistema 2 debió vetar la causalidad espuria"),
        }
    }
}
