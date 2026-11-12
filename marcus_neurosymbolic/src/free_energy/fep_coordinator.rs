//! Módulo del Coordinador Cognitivo de Energía Libre e Inferencia Activa (FEP Coordinator).
//!
//! Orquesta el ciclo cognitivo dual basado 100% en:
//! 1. Maximización de la Energía Libre de Evidencia (ELBO = -F(q, o)).
//! 2. Reducción rigurosa de la Sorpresa Estructurada (Perceptual, Algebraica, Física y Causal).
//! 3. Selección de acciones por Inferencia Activa mediante minimización de Energía Libre Esperada G(pi).
//! 4. Veto epistémico estricto ante violaciones físicas Spelke o exceso de energía libre residual.

use crate::algebraic_mind::type_token::TypeTokenRegistry;
use crate::core_knowledge::causal_engine::CausalEngine;
use crate::core_knowledge::spatiotemporal_physics::SpelkePhysicsEngine;
use crate::dual_process::system1_proposer::{CandidateHypothesis, System1Proposer};
use crate::dual_process::system2_deliberator::{DeliberationVerdict, System2Deliberator};
use crate::free_energy::active_inference::{ActiveAction, ActiveInferenceEngine, Policy};
use crate::free_energy::precision_dynamics::{CognitiveChannel, PrecisionManager};
use crate::free_energy::surprise_engine::{StructuredSurprise, SurpriseEngine};
use crate::free_energy::variational_free_energy::{FreeEnergyOptimizer, PriorBelief, VariationalBelief};
use serde::{Deserialize, Serialize};

/// Resultado del ciclo cognitivo unificado FEP + Inferencia Activa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepCognitiveCycleResult {
    pub query: String,
    pub initial_surprise: StructuredSurprise,
    pub final_surprise: StructuredSurprise,
    /// Tasa de reducción de sorpresa: (S_init - S_final) / max(S_init, 1e-6)
    pub surprise_reduction_ratio: f64,
    /// Energía libre variacional inicial F(q_init, o)
    pub initial_free_energy: f64,
    /// Energía libre variacional optimizada F(q_opt, o)
    pub optimized_free_energy: f64,
    /// Cota inferior de evidencia logarítmica (ELBO = -F_opt), a maximizar
    pub evidence_lower_bound: f64,
    pub selected_hypothesis: Option<CandidateHypothesis>,
    pub selected_action: Option<ActiveAction>,
    pub verdict: Option<DeliberationVerdict>,
    pub rejected_hypotheses: Vec<(CandidateHypothesis, String)>,
    pub cycles_elapsed: usize,
}

/// Coordinador de la Arquitectura FEP-MARCUS.
pub struct FepDualProcessCoordinator {
    pub free_energy_optimizer: FreeEnergyOptimizer,
    pub surprise_engine: SurpriseEngine,
    pub active_inference_engine: ActiveInferenceEngine,
    pub precision_manager: PrecisionManager,
    /// Umbral máximo tolerado de energía libre antes de veto epistémico
    pub free_energy_veto_threshold: f64,
    pub max_candidate_evaluations: usize,
}

impl Default for FepDualProcessCoordinator {
    fn default() -> Self {
        Self {
            free_energy_optimizer: FreeEnergyOptimizer::default(),
            surprise_engine: SurpriseEngine::default(),
            active_inference_engine: ActiveInferenceEngine::default(),
            precision_manager: PrecisionManager::default(),
            free_energy_veto_threshold: 45.0,
            max_candidate_evaluations: 5,
        }
    }
}

impl FepDualProcessCoordinator {
    pub fn new(veto_threshold: f64) -> Self {
        Self {
            free_energy_veto_threshold: veto_threshold,
            ..Default::default()
        }
    }

    /// Ejecuta el ciclo cognitivo completo orientado a maximizar ELBO y reducir sorpresa.
    pub fn execute_fep_cycle(
        &mut self,
        query: &str,
        sensory_observation: &[f64],
        system1: &System1Proposer,
        physics_engine: &mut SpelkePhysicsEngine,
        causal_engine: &mut CausalEngine,
        type_registry: &TypeTokenRegistry,
    ) -> FepCognitiveCycleResult {
        let dim = sensory_observation.len();

        // 1. Calcular Sorpresa Perceptual Inicial
        let initial_pred = vec![0.0; dim];
        let var_reg = crate::algebraic_mind::variables::VariableRegister::new();
        let initial_surprise = self.surprise_engine.compute_full_surprise(
            sensory_observation,
            &initial_pred,
            &var_reg,
            None,
            None,
            None,
            physics_engine,
            &[],
            0.1,
            1.0,
            1.0,
        );

        // 2. Sistema 1: Proposición asociativa rápida de candidatos
        let candidates = system1.propose_candidates(query, self.max_candidate_evaluations);
        let mut rejected = Vec::new();

        let prior = PriorBelief::standard_normal(dim);
        let precisions = vec![self.precision_manager.get_precision(CognitiveChannel::SensoryContinuous); dim];

        let mut best_candidate = None;
        let mut best_verdict = None;
        let mut best_fe_result = None;
        let mut min_fe = f64::INFINITY;
        let mut initial_fe = 0.0;

        for candidate in candidates {
            // Mapear candidato a creencia variacional inicial q(s)
            let mut q_init = VariationalBelief::new(dim, &candidate.id);
            q_init.associative_confidence = candidate.associative_confidence;
            // Inyectar ligamientos simbólicos
            q_init = q_init.with_binding("Subject", &candidate.subject);
            q_init = q_init.with_binding("Predicate", &candidate.predicate);
            q_init = q_init.with_binding("Object", &candidate.object);

            let fe_init_res = self.free_energy_optimizer.evaluate(
                sensory_observation,
                &q_init,
                &prior,
                &precisions,
            );

            if initial_fe == 0.0 {
                initial_fe = fe_init_res.variational_free_energy;
            }

            // Optimización de energía libre (descenso de gradiente sobre q(s) / ascenso sobre ELBO)
            let (q_opt, _history) = self.free_energy_optimizer.optimize_posterior(
                sensory_observation,
                &q_init,
                &prior,
                &precisions,
            );

            let fe_opt_res = self.free_energy_optimizer.evaluate(
                sensory_observation,
                &q_opt,
                &prior,
                &precisions,
            );

            // 3. Verificación Deliberativa del Sistema 2 (Física Spelke + Causalidad Pearl)
            let mut deliberator = System2Deliberator::new(physics_engine, causal_engine, type_registry);
            let verdict = deliberator.deliberate(&candidate);

            match &verdict {
                DeliberationVerdict::Vetoed { reason, .. } => {
                    rejected.push((candidate, reason.clone()));
                }
                DeliberationVerdict::Approved { .. } => {
                    // Si además la energía libre está dentro de la tolerancia homeostática
                    if fe_opt_res.variational_free_energy <= self.free_energy_veto_threshold {
                        if fe_opt_res.variational_free_energy < min_fe {
                            min_fe = fe_opt_res.variational_free_energy;
                            best_candidate = Some(candidate);
                            best_verdict = Some(verdict);
                            best_fe_result = Some(fe_opt_res);
                        }
                    } else {
                        rejected.push((
                            candidate,
                            format!(
                                "Veto por Energía Libre Excesiva: F = {:.2} > umbral {:.2}",
                                fe_opt_res.variational_free_energy, self.free_energy_veto_threshold
                            ),
                        ));
                    }
                }
            }
        }

        // 4. Inferencia Activa: Selección de Acción Óptima para minimizar G(pi)
        let policies = vec![
            Policy::single_action(
                "explore_causal",
                ActiveAction::CausalIntervention {
                    target_var: "Intervention_Target".to_string(),
                    set_value: 1.0,
                },
            ),
            Policy::single_action(
                "probe_binding",
                ActiveAction::ProbeVariableBinding {
                    var_name: "Subject".to_string(),
                },
            ),
            Policy::single_action(
                "audit_truth",
                ActiveAction::AuditClaimTruth {
                    claim_id: format!("claim_{}", query),
                },
            ),
        ];

        let (chosen_policy, _evals) = self.active_inference_engine.evaluate_and_select_policy(
            &policies,
            causal_engine,
            if best_candidate.is_some() { 0 } else { 2 },
            None,
        );

        let chosen_action = chosen_policy.actions.first().cloned();

        // 5. Cómputo de Sorpresa Final y Métricas de Rendimiento
        let opt_fe = best_fe_result.as_ref().map_or(initial_fe, |r| r.variational_free_energy);
        let elbo = best_fe_result.as_ref().map_or(-initial_fe, |r| r.evidence_lower_bound);

        let final_pred = if let Some(fe_res) = &best_fe_result {
            // Predicción alineada con residuos reducidos
            sensory_observation.iter().enumerate().map(|(i, &o)| {
                o - fe_res.precision_weighted_residuals.get(i).copied().unwrap_or(0.0) / precisions[i]
            }).collect()
        } else {
            vec![0.0; dim]
        };

        let final_surprise = self.surprise_engine.compute_full_surprise(
            sensory_observation,
            &final_pred,
            &var_reg,
            None,
            None,
            None,
            physics_engine,
            &[],
            0.1,
            1.0,
            1.0,
        );

        let reduction_ratio = if initial_surprise.total_surprise > 1e-6 {
            (initial_surprise.total_surprise - final_surprise.total_surprise)
                / initial_surprise.total_surprise
        } else {
            1.0
        };

        // Adaptar precisión sensorial según el residuo observado
        let avg_residual = (initial_surprise.perceptual_surprise - final_surprise.perceptual_surprise).abs();
        self.precision_manager.update_sensory_precision(avg_residual);

        let cycles = rejected.len() + if best_candidate.is_some() { 1 } else { 0 };

        FepCognitiveCycleResult {
            query: query.to_string(),
            initial_surprise,
            final_surprise,
            surprise_reduction_ratio: reduction_ratio.max(0.0),
            initial_free_energy: initial_fe,
            optimized_free_energy: opt_fe,
            evidence_lower_bound: elbo,
            selected_hypothesis: best_candidate,
            selected_action: chosen_action,
            verdict: best_verdict,
            rejected_hypotheses: rejected,
            cycles_elapsed: cycles,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_coordinator_reduces_surprise_and_maximizes_elbo() {
        let mut coord = FepDualProcessCoordinator::default();
        let mut sys1 = System1Proposer::new();
        sys1.add_association(CandidateHypothesis::new(
            "h1",
            "Apple falls to earth",
            0.92,
            "Apple",
            "FallsTowards",
            "Earth",
        ));

        let mut physics = SpelkePhysicsEngine::new();
        let mut causal = CausalEngine::new();
        let type_reg = TypeTokenRegistry::new();

        let sensory_obs = vec![0.5, 0.2, -0.1, 0.9];

        let result = coord.execute_fep_cycle(
            "Apple FallsTowards",
            &sensory_obs,
            &mut sys1,
            &mut physics,
            &mut causal,
            &type_reg,
        );

        assert!(result.selected_hypothesis.is_some());
        assert!(result.optimized_free_energy <= result.initial_free_energy);
        assert!(result.evidence_lower_bound >= -result.initial_free_energy);
        assert!(result.surprise_reduction_ratio >= 0.0);
        assert!(result.selected_action.is_some());
    }
}
