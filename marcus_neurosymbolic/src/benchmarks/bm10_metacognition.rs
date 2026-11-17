//! Benchmark 10: Calibración Metacognitiva, Veto Anti-Sobreconfianza y Esfuerzo Racional.
//!
//! Evalúa la hipótesis H-FEP5:
//! "Un controlador de Inferencia Activa de segundo orden (Metacognición) logra menor
//! error de calibración probabilística (Brier score), detecta conflictos ocultos mediante
//! veto de duda epistémica (0% falsas certezas sobre ilusiones físicas/causales) y reduce
//! el costo computacional promedio asignando salidas rápidas Sistema-1 en estímulos dominados."

use crate::free_energy::metacognition::{CognitiveEffortTier, MetacognitiveController};
use crate::free_energy::precision_dynamics::PrecisionProfile;
use crate::free_energy::surprise_engine::StructuredSurprise;
use crate::free_energy::variational_free_energy::{FreeEnergyOptimizer, PriorBelief, VariationalBelief};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm10Results {
    pub name: String,
    pub total_trials: usize,
    pub baseline_uncalibrated_brier: f64,
    pub metacognitive_brier_score: f64,
    pub brier_improvement_pct: f64,
    pub illusion_veto_recall: f64,
    pub compute_savings_pct: f64,
    pub passed_h_fep5: bool,
}

pub fn run_bm10_metacognition() -> Bm10Results {
    let mut meta = MetacognitiveController::new();
    let opt = FreeEnergyOptimizer::new(0.08, 12, 1e-4);
    let profile = PrecisionProfile::default();
    let dim = 8;
    let prior = PriorBelief::standard_normal(dim);
    let precisions = vec![2.0; dim];

    let total_trials = 60;
    let mut baseline_brier_sum = 0.0;
    let mut total_baseline_steps = 0usize;
    let mut total_meta_steps = 0usize;
    let mut illusion_cases = 0usize;
    let mut illusion_vetoed = 0usize;

    for t in 0..total_trials {
        let is_illusion = t % 4 == 3; // 25% trampas OOD / ilusiones sobreconfiadas del Sistema-1
        let is_known_familiar = t % 4 == 0 || t % 4 == 1;

        let obs = if is_known_familiar {
            vec![0.25; dim]
        } else {
            vec![0.95; dim]
        };

        let mut belief = VariationalBelief::new(dim, format!("trial_{}", t));
        belief.associative_confidence = if is_illusion { 0.94 } else if is_known_familiar { 0.90 } else { 0.45 };
        belief.physical_validity = !is_illusion;

        let init_fe = opt.evaluate(&obs, &belief, &prior, &precisions).variational_free_energy;
        let (mut q_opt, _) = opt.optimize_posterior(&obs, &belief, &prior, &precisions);
        q_opt.associative_confidence = belief.associative_confidence;
        q_opt.physical_validity = belief.physical_validity;
        if is_known_familiar {
            q_opt.log_variance = vec![-2.4; dim];
        }
        let fe_res = opt.evaluate(&obs, &q_opt, &prior, &precisions);

        let surprise = if is_illusion {
            StructuredSurprise {
                perceptual_surprise: 0.45,
                algebraic_surprise: 0.35,
                physical_surprise: 40.0,
                causal_surprise: 1.5,
                total_surprise: 42.3,
                dominant_source: "PhysicalSpelke".to_string(),
                diagnostic_details: vec![],
            }
        } else if is_known_familiar {
            StructuredSurprise {
                perceptual_surprise: 0.04,
                algebraic_surprise: 0.0,
                physical_surprise: 0.0,
                causal_surprise: 0.0,
                total_surprise: 0.04,
                dominant_source: "None".to_string(),
                diagnostic_details: vec![],
            }
        } else {
            StructuredSurprise {
                perceptual_surprise: 0.65,
                algebraic_surprise: 0.80,
                physical_surprise: 0.0,
                causal_surprise: 0.90,
                total_surprise: 2.35,
                dominant_source: "Causal".to_string(),
                diagnostic_details: vec![],
            }
        };

        // Verdad base: los casos familiares se resuelven; las ilusiones no vetadas fallarían
        let actual_success = is_known_familiar;
        let y = if actual_success { 1.0 } else { 0.0 };

        // Línea base ingenua: confía ciegamente en `associative_confidence`
        baseline_brier_sum += (y - belief.associative_confidence).powi(2);
        total_baseline_steps += 15;

        let assess = meta.assess_and_control(&q_opt, init_fe, &fe_res, &surprise, &profile);
        total_meta_steps += assess.recommended_e_steps;

        if is_illusion {
            illusion_cases += 1;
            if assess.self_doubt_veto && assess.recommended_effort == CognitiveEffortTier::ActiveMetaInquiry {
                illusion_vetoed += 1;
            }
        }

        meta.record_outcome(&assess, &q_opt, &surprise, actual_success);
    }

    let stats = meta.calibration_stats();
    let baseline_brier = baseline_brier_sum / total_trials as f64;
    let brier_imp = ((baseline_brier - stats.brier_score) / baseline_brier.max(1e-8)) * 100.0;
    let veto_recall = illusion_vetoed as f64 / illusion_cases.max(1) as f64;
    let compute_savings =
        ((total_baseline_steps as f64 - total_meta_steps as f64) / total_baseline_steps as f64) * 100.0;

    let passed = stats.brier_score < baseline_brier * 0.5 && veto_recall == 1.0 && compute_savings > 15.0;

    Bm10Results {
        name: "BM10: Metacognitive Calibration & Rational Effort Control".to_string(),
        total_trials,
        baseline_uncalibrated_brier: baseline_brier,
        metacognitive_brier_score: stats.brier_score,
        brier_improvement_pct: brier_imp,
        illusion_veto_recall: veto_recall,
        compute_savings_pct: compute_savings,
        passed_h_fep5: passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm10_execution() {
        let res = run_bm10_metacognition();
        assert!(res.passed_h_fep5);
        assert_eq!(res.illusion_veto_recall, 1.0);
        assert!(res.brier_improvement_pct > 50.0);
        assert!(res.compute_savings_pct > 15.0);
    }
}
