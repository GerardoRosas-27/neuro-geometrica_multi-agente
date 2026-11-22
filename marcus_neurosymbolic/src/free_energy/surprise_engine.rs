//! Módulo del Motor de Reducción de Sorpresa Estructurada (Multi-Modal Structured Surprise).
//!
//! En el Principio de Energía Libre, la Sorpresa es el negativo del logaritmo de verosimilitud:
//! S(o) = -ln p(o)
//!
//! En la cognición neuro-simbólica de Gary Marcus, la sorpresa no es un escalar opaco;
//! se descompone de forma explicable en cuatro canales fundamentales:
//! 1. Sorpresa Perceptual (S_perc): Desajuste sensorial continuo ||o - o_hat||.
//! 2. Sorpresa Algebraica (S_alg): Violación de ligamiento de variables, crosstalk o gramática OOD.
//! 3. Sorpresa Física Spelke (S_phys): Teletransportación, penetración de sólidos o flotación sin soporte.
//! 4. Sorpresa Causal Pearl (S_causal): Divergencia entre correlación P(Y|X) e intervención real do(X).

use crate::algebraic_mind::variables::{HrrVector, VariableRegister};
use crate::core_knowledge::spatiotemporal_physics::{PhysicalViolation, SpatiotemporalObject, SpelkePhysicsEngine};
use serde::{Deserialize, Serialize};

/// Desglose estructurado de la sorpresa del sistema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredSurprise {
    /// Sorpresa sensorial continua (desajuste perceptivo)
    pub perceptual_surprise: f64,
    /// Sorpresa sintáctica y de ligamiento de variables
    pub algebraic_surprise: f64,
    /// Sorpresa por violación de física Spelke (invariantes innatos)
    pub physical_surprise: f64,
    /// Sorpresa causal por disparidad intervencionista do(X)
    pub causal_surprise: f64,
    /// Sorpresa global integrada ponderada por precisión
    pub total_surprise: f64,
    /// Diagnóstico explicable de la causa primaria de sorpresa
    pub dominant_source: String,
    pub diagnostic_details: Vec<String>,
}

/// Motor de evaluación y reducción de sorpresa.
#[derive(Debug, Clone)]
pub struct SurpriseEngine {
    pub weight_perceptual: f64,
    pub weight_algebraic: f64,
    pub weight_physical: f64,
    pub weight_causal: f64,
    pub physical_violation_penalty: f64,
}

impl Default for SurpriseEngine {
    fn default() -> Self {
        Self {
            weight_perceptual: 1.0,
            weight_algebraic: 2.0,
            weight_physical: 10.0, // Violaciones físicas son altamente informativas/sorprendentes
            weight_causal: 3.0,
            physical_violation_penalty: 50.0,
        }
    }
}

impl SurpriseEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evalúa la sorpresa perceptual continua dado un vector observado y uno predicho:
    /// S_perc = 0.5 * sum [ (o_i - pred_i)^2 ]
    pub fn evaluate_perceptual_surprise(observed: &[f64], predicted: &[f64]) -> f64 {
        let n = observed.len().min(predicted.len());
        if n == 0 {
            return 0.0;
        }
        let mut err = 0.0;
        for i in 0..n {
            let d = observed[i] - predicted[i];
            err += d * d;
        }
        0.5 * err
    }

    /// Evalúa la sorpresa algebraica:
    /// - Conflictos en el registro de ligamiento de variables.
    /// - Error de reconstrucción HRR rol-rellenador (diafonía/crosstalk).
    pub fn evaluate_algebraic_surprise(
        var_register: &VariableRegister,
        target_role: Option<&HrrVector>,
        target_filler: Option<&HrrVector>,
        bound_representation: Option<&HrrVector>,
    ) -> (f64, Vec<String>) {
        let mut surprise = 0.0;
        let mut diagnostics = Vec::new();

        // 1. Conflictos de ligamiento (si hay variables no asignadas o incoherentes)
        let bindings_count = var_register.all_bindings().len();
        if bindings_count == 0 {
            surprise += 1.0;
            diagnostics.push("Registro de variables vacío (sorpresa por indeterminación sintáctica)".to_string());
        }

        // 2. Fidelidad de desemparejamiento HRR (The Algebraic Mind Cap 3)
        if let (Some(role), Some(expected_filler), Some(bound)) = (target_role, target_filler, bound_representation) {
            let unbind = role.circular_correlation(bound);
            let sim = unbind.cosine_similarity(expected_filler);
            // Si la similitud es baja, la sorpresa por crosstalk es alta
            let hrr_surprise = (1.0f64 - sim).max(0.0f64) * 10.0;
            surprise += hrr_surprise;
            if hrr_surprise > 1.0 {
                diagnostics.push(format!("Diafonía HRR detectada en rol-rellenador: similitud {:.3}, sorpresa {:.3}", sim, hrr_surprise));
            }
        }

        (surprise, diagnostics)
    }

    /// Evalúa la sorpresa física de Spelke a partir de violaciones espacio-temporales.
    pub fn evaluate_physical_surprise(
        violations: &[PhysicalViolation],
        penalty_per_violation: f64,
    ) -> (f64, Vec<String>) {
        let mut surprise = 0.0;
        let mut diagnostics = Vec::new();

        for v in violations {
            match v {
                PhysicalViolation::Teleportation { object_id, observed_speed, max_speed, .. } => {
                    let s = penalty_per_violation * 2.0;
                    surprise += s;
                    diagnostics.push(format!("Teletransportación anómala en {}: velocidad {:.2} excede límite {:.2} (sorpresa {:.1})", object_id, observed_speed, max_speed, s));
                }
                PhysicalViolation::SolidInterpenetration { object_a, object_b, distance, min_allowed_distance } => {
                    let s = penalty_per_violation * 1.5;
                    surprise += s;
                    diagnostics.push(format!("Penetración de sólidos entre {} y {}: distancia {:.2} < mínimo {:.2} (sorpresa {:.1})", object_a, object_b, distance, min_allowed_distance, s));
                }
                PhysicalViolation::ObjectPermanenceFailure { object_id, last_position } => {
                    let s = penalty_per_violation * 2.5;
                    surprise += s;
                    diagnostics.push(format!("Violación de permanencia de objeto {}: desaparecido mágicamente desde {:?}", object_id, last_position));
                }
                PhysicalViolation::GravityDefiance { object_id, altitude_z } => {
                    let s = penalty_per_violation;
                    surprise += s;
                    diagnostics.push(format!("Desafío a la gravedad en {}: flotando en altitud z={:.2} sin soporte (sorpresa {:.1})", object_id, altitude_z, s));
                }
                PhysicalViolation::ContainmentTransitivityFailure { object_a, container_b, container_c } => {
                    let s = penalty_per_violation * 1.8;
                    surprise += s;
                    diagnostics.push(format!("Fallo de transitividad de contención: {} en {} y {} en {}, pero no transitivo (sorpresa {:.1})", object_a, container_b, container_b, container_c, s));
                }
            }
        }

        (surprise, diagnostics)
    }

    /// Evalúa la sorpresa causal: disparidad entre correlación esperada e intervención real.
    pub fn evaluate_causal_surprise(
        expected_correlational: f64,
        interventional_actual: f64,
    ) -> (f64, Option<String>) {
        let diff = (expected_correlational - interventional_actual).abs();
        let surprise = diff * 5.0;
        let diag = if surprise > 1.0 {
            Some(format!(
                "Disparidad causal Pearl: expectativa correlacional {:.3} vs intervención real do(X) {:.3} (sorpresa {:.3})",
                expected_correlational, interventional_actual, surprise
            ))
        } else {
            None
        };
        (surprise, diag)
    }

    /// Integra todas las fuentes de sorpresa en un informe multi-modal.
    pub fn compute_full_surprise(
        &self,
        observed_vec: &[f64],
        predicted_vec: &[f64],
        var_reg: &VariableRegister,
        target_role: Option<&HrrVector>,
        target_filler: Option<&HrrVector>,
        bound_rep: Option<&HrrVector>,
        physics_engine: &mut SpelkePhysicsEngine,
        objects: &[SpatiotemporalObject],
        dt: f64,
        expected_corr: f64,
        intervened_act: f64,
    ) -> StructuredSurprise {
        // 1. Perceptual
        let s_perc = Self::evaluate_perceptual_surprise(observed_vec, predicted_vec);

        // 2. Algebraica
        let (s_alg, mut diag_alg) = Self::evaluate_algebraic_surprise(
            var_reg,
            target_role,
            target_filler,
            bound_rep,
        );

        // 3. Física Spelke
        let violations = physics_engine.step_and_verify(objects, dt);
        let (s_phys, mut diag_phys) = Self::evaluate_physical_surprise(&violations, self.physical_violation_penalty);

        // 4. Causal
        let (s_causal, diag_causal) = Self::evaluate_causal_surprise(expected_corr, intervened_act);

        // Ponderación integrada
        let total = self.weight_perceptual * s_perc
            + self.weight_algebraic * s_alg
            + self.weight_physical * s_phys
            + self.weight_causal * s_causal;

        // Identificar fuente dominante
        let mut sources = [
            (self.weight_perceptual * s_perc, "Perceptual"),
            (self.weight_algebraic * s_alg, "AlgebraicBinding"),
            (self.weight_physical * s_phys, "SpelkePhysicalViolation"),
            (self.weight_causal * s_causal, "CausalFallacy"),
        ];
        sources.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let dominant = sources[0].1.to_string();

        let mut all_diags = Vec::new();
        all_diags.append(&mut diag_alg);
        all_diags.append(&mut diag_phys);
        if let Some(dc) = diag_causal {
            all_diags.push(dc);
        }

        StructuredSurprise {
            perceptual_surprise: s_perc,
            algebraic_surprise: s_alg,
            physical_surprise: s_phys,
            causal_surprise: s_causal,
            total_surprise: total,
            dominant_source: dominant,
            diagnostic_details: all_diags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_surprise_calculation() {
        let obs = vec![1.0, 2.0, 3.0];
        let pred = vec![1.0, 2.0, 3.0];
        assert_eq!(SurpriseEngine::evaluate_perceptual_surprise(&obs, &pred), 0.0);

        let pred_noisy = vec![1.0, 2.0, 5.0]; // diff = 2, sq = 4, 0.5 * 4 = 2.0
        assert_eq!(SurpriseEngine::evaluate_perceptual_surprise(&obs, &pred_noisy), 2.0);
    }

    #[test]
    fn test_physical_surprise_on_teleportation() {
        let mut physics = SpelkePhysicsEngine::new();
        physics.max_physical_speed = 10.0;

        let obj1 = SpatiotemporalObject::new("ball_1", (0.0, 0.0, 0.0), 1.0);
        physics.step_and_verify(&[obj1], 1.0);

        // En el siguiente paso, se mueve a (100, 0, 0) en dt=1.0 -> velocidad 100 > 10
        let obj1_teleported = SpatiotemporalObject::new("ball_1", (100.0, 0.0, 0.0), 1.0);
        let violations = physics.step_and_verify(&[obj1_teleported], 1.0);

        let (surprise, diags) = SurpriseEngine::evaluate_physical_surprise(&violations, 50.0);
        assert!(surprise > 0.0);
        assert!(!diags.is_empty());
        assert!(diags[0].contains("Teletransportación"));
    }

    #[test]
    fn test_causal_surprise_on_spurious_correlation() {
        let (surprise, diag) = SurpriseEngine::evaluate_causal_surprise(1.0, 0.0);
        assert_eq!(surprise, 5.0);
        assert!(diag.is_some());
    }
}
