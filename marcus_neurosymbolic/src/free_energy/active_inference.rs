//! Módulo de Inferencia Activa y Energía Libre Esperada (Active Inference & Expected Free Energy).
//!
//! Basado en la teoría de Inferencia Activa (Karl Friston, Pezzulo, Parr):
//! Los agentes no sólo actualizan sus creencias pasivas; seleccionan políticas y acciones pi
//! para minimizar la Energía Libre Esperada G(pi):
//!
//! G(pi) = - Valor_Epistémico (Ganancia de Información / Curiosidad / Reducción de Sorpresa Futura)
//!        - Valor_Pragmático (Satisfacción de Metas y Preferencias a Priori C)
//!
//! En la síntesis con Gary Marcus:
//! Las acciones pi incluyen Intervenciones Causales do(X = x) de Judea Pearl, sondeos de
//! ligamiento de variables y exploraciones de permanencia física de objetos ocluidos.

use crate::core_knowledge::causal_engine::{CausalEngine, CausalVar};
use serde::{Deserialize, Serialize};

/// Acción causal o exploratoria disponible para el agente en el bucle de Inferencia Activa.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActiveAction {
    /// Intervención causal estricta de Pearl: do(Var = Value)
    CausalIntervention {
        target_var: String,
        set_value: f64,
    },
    /// Sondeo epistémico de ligamiento de variable formal
    ProbeVariableBinding {
        var_name: String,
    },
    /// Sondeo de permanencia de objeto ocluido en el espacio 3D
    InspectOccludedObject {
        object_id: String,
    },
    /// Acción de navegación o ajuste de trayectoria geodésica
    MotorGeodesicStep {
        target_pos: (f64, f64, f64),
    },
    /// Verificación epistémica de una afirmación contra el estado del mundo
    AuditClaimTruth {
        claim_id: String,
    },
}

/// Política: secuencia de acciones a ejecutar en el horizonte de planificación.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub policy_id: String,
    pub actions: Vec<ActiveAction>,
}

impl Policy {
    pub fn single_action(id: impl Into<String>, action: ActiveAction) -> Self {
        Self {
            policy_id: id.into(),
            actions: vec![action],
        }
    }
}

/// Evaluación de una política bajo el Principio de Energía Libre Esperada.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFreeEnergyEvaluation {
    pub policy_id: String,
    /// G(pi) = - EpistemicValue - PragmaticValue (a minimizar)
    pub expected_free_energy: f64,
    /// Ganancia de información / Resolución de incertidumbre esperada
    pub epistemic_value: f64,
    /// Utilidad / Alineación con metas y preferencias del agente ln p(o | C)
    pub pragmatic_value: f64,
    /// Probabilidad de selección bajo softmax Boltzmann: p(pi) propto exp(-gamma * G)
    pub selection_probability: f64,
}

/// Motor de Inferencia Activa.
#[derive(Debug, Clone)]
pub struct ActiveInferenceEngine {
    /// Precisión de la política gamma (inversa de temperatura)
    pub policy_precision_gamma: f64,
    /// Ponderación de curiosidad epistémica vs recompensa pragmática
    pub epistemic_weight: f64,
    pub pragmatic_weight: f64,
}

impl Default for ActiveInferenceEngine {
    fn default() -> Self {
        Self {
            policy_precision_gamma: 1.5,
            epistemic_weight: 1.0,
            pragmatic_weight: 1.0,
        }
    }
}

impl ActiveInferenceEngine {
    pub fn new(gamma: f64, ep_weight: f64, prag_weight: f64) -> Self {
        Self {
            policy_precision_gamma: gamma,
            epistemic_weight: ep_weight,
            pragmatic_weight: prag_weight,
        }
    }

    /// Evalúa el Valor Epistémico (ganancia de información / resolución de sorpresa)
    /// de una acción dada sobre el modelo causal y el estado del mundo.
    pub fn compute_epistemic_value(
        &self,
        action: &ActiveAction,
        causal_engine: &CausalEngine,
        unbound_variables_count: usize,
    ) -> f64 {
        match action {
            ActiveAction::CausalIntervention { target_var, .. } => {
                let var = CausalVar::new(target_var);
                // Si la variable tiene dependencias no resueltas o correlaciones dudosas,
                // intervenir sobre ella produce alta ganancia epistémica
                if causal_engine.contains_var(&var) {
                    let parents_count = causal_engine.parent_count(&var);
                    // Más padres = mayor confusión previa = mayor ganancia al hacer do(X)
                    2.0 + (parents_count as f64) * 1.5
                } else {
                    1.0
                }
            }
            ActiveAction::ProbeVariableBinding { .. } => {
                // Sondear variables no ligadas reduce la incertidumbre sintáctica
                if unbound_variables_count > 0 {
                    3.0 * (unbound_variables_count as f64)
                } else {
                    0.5
                }
            }
            ActiveAction::InspectOccludedObject { .. } => {
                // Comprobar la permanencia reduce la incertidumbre sobre el estado oculto
                4.0
            }
            ActiveAction::MotorGeodesicStep { .. } => 0.5,
            ActiveAction::AuditClaimTruth { .. } => 2.5,
        }
    }

    /// Evalúa el Valor Pragmático (satisfacción de preferencias a priori C).
    pub fn compute_pragmatic_value(
        &self,
        action: &ActiveAction,
        preferred_goal: Option<&(f64, f64, f64)>,
    ) -> f64 {
        match action {
            ActiveAction::MotorGeodesicStep { target_pos } => {
                if let Some(goal) = preferred_goal {
                    let dist = ((target_pos.0 - goal.0).powi(2)
                        + (target_pos.1 - goal.1).powi(2)
                        + (target_pos.2 - goal.2).powi(2))
                    .sqrt();
                    // Preferencia: menor distancia al objetivo
                    5.0 / (1.0 + dist)
                } else {
                    1.0
                }
            }
            ActiveAction::CausalIntervention { .. } => 1.5,
            ActiveAction::ProbeVariableBinding { .. } => 1.0,
            ActiveAction::InspectOccludedObject { .. } => 2.0,
            ActiveAction::AuditClaimTruth { .. } => 2.0,
        }
    }

    /// Evalúa una lista de políticas candidatas y selecciona la óptima que minimiza G(pi).
    pub fn evaluate_and_select_policy(
        &self,
        policies: &[Policy],
        causal_engine: &CausalEngine,
        unbound_vars: usize,
        preferred_goal: Option<&(f64, f64, f64)>,
    ) -> (Policy, Vec<ExpectedFreeEnergyEvaluation>) {
        assert!(!policies.is_empty(), "La lista de políticas no puede estar vacía");

        let mut evaluations = Vec::with_capacity(policies.len());

        for policy in policies {
            let mut total_epistemic = 0.0;
            let mut total_pragmatic = 0.0;

            for action in &policy.actions {
                total_epistemic += self.compute_epistemic_value(action, causal_engine, unbound_vars);
                total_pragmatic += self.compute_pragmatic_value(action, preferred_goal);
            }

            // G(pi) = - (w_e * Epistemic + w_p * Pragmatic)
            let efe = -(self.epistemic_weight * total_epistemic + self.pragmatic_weight * total_pragmatic);

            evaluations.push(ExpectedFreeEnergyEvaluation {
                policy_id: policy.policy_id.clone(),
                expected_free_energy: efe,
                epistemic_value: total_epistemic,
                pragmatic_value: total_pragmatic,
                selection_probability: 0.0,
            });
        }

        // Cómputo de probabilidades Softmax: p(pi) = exp(-gamma * G) / sum [ exp(-gamma * G) ]
        // Dado que G ya es negativo cuando la política es buena, -gamma * G es positivo para buenas políticas
        let min_efe = evaluations.iter().map(|e| e.expected_free_energy).fold(f64::INFINITY, f64::min);
        let max_val = evaluations.iter().map(|e| -self.policy_precision_gamma * (e.expected_free_energy - min_efe)).fold(f64::NEG_INFINITY, f64::max);

        let mut sum_exp = 0.0;
        let mut unnormalized_weights = Vec::with_capacity(evaluations.len());

        for eval in &evaluations {
            let logit = -self.policy_precision_gamma * (eval.expected_free_energy - min_efe) - max_val;
            let w = logit.exp();
            unnormalized_weights.push(w);
            sum_exp += w;
        }

        let mut best_idx = 0;
        let mut best_prob = -1.0;

        for (i, eval) in evaluations.iter_mut().enumerate() {
            eval.selection_probability = if sum_exp > 0.0 {
                unnormalized_weights[i] / sum_exp
            } else {
                1.0 / policies.len() as f64
            };

            if eval.selection_probability > best_prob {
                best_prob = eval.selection_probability;
                best_idx = i;
            }
        }

        (policies[best_idx].clone(), evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_exploration_chosen_under_uncertainty() {
        let engine = ActiveInferenceEngine::default();
        let causal = CausalEngine::new();

        // Política A: Acción pasiva
        let pol_passive = Policy::single_action(
            "passive",
            ActiveAction::MotorGeodesicStep { target_pos: (0.0, 0.0, 0.0) },
        );
        // Política B: Intervención causal exploratoria en presencia de variables no ligadas
        let pol_active = Policy::single_action(
            "causal_intervention",
            ActiveAction::ProbeVariableBinding { var_name: "x".to_string() },
        );

        let policies = vec![pol_passive, pol_active];
        // Con 4 variables no ligadas, la política epistémica debe ganar claramente
        let (selected, evals) = engine.evaluate_and_select_policy(&policies, &causal, 4, None);

        assert_eq!(selected.policy_id, "causal_intervention");
        assert!(evals[1].epistemic_value > evals[0].epistemic_value);
        assert!(evals[1].selection_probability > evals[0].selection_probability);
    }
}
