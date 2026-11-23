//! Módulo de Arquitectura de Metacognición Variacional (Second-Order Active Inference).
//!
//! Implementa el bucle jerárquico Monitoreo-Control (Nelson & Narens + Friston & Fleming):
//! - Nivel 1 (Objeto): creencia variacional q(s) = (mu, log_var), F(q, o), S(o).
//! - Nivel 2 (Meta): introspección sobre q(s), entropía diferencial H[q],
//!   calibración probabilística (Brier / EMA ECE), conflicto inter-canal y
//!   asignación racional de esfuerzo computacional (System1Fast vs System2Deliberative
//!   vs ActiveMetaInquiry) con veto anti-sobreconfianza (Self-Doubt Epistemic Vigilance).

use crate::free_energy::precision_dynamics::PrecisionProfile;
use crate::free_energy::surprise_engine::StructuredSurprise;
use crate::free_energy::variational_free_energy::{VariationalBelief, VariationalFreeEnergyResult};
use serde::{Deserialize, Serialize};

/// Nivel de esfuerzo computacional asignado por el controlador metacognitivo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveEffortTier {
    /// Salida rápida Sistema-1 cuando la confianza calibrada es alta y no hay conflicto.
    System1Fast,
    /// Deliberación estándar Sistema-2 ante novedad conceptual moderada.
    System2Deliberative,
    /// Profundización epistémica máxima ante conflicto inter-canal o anomalía OOD.
    ActiveMetaInquiry,
}

/// Estado metacognitivo de segundo orden producido tras monitorear un ciclo cognitivo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveAssessment {
    /// Entropía diferencial normalizada del posterior q(s)
    pub posterior_entropy: f64,
    /// Probabilidad epistémica calibrada de éxito/comprensión c_meta in [0, 1]
    pub calibrated_confidence: f64,
    /// Índice de conflicto inter-canal (desacuerdo entre Sistema-1 asociativo y verificadores)
    pub conflict_index: f64,
    /// Ganancia relativa de Energía Libre lograda: (F_init - F_opt) / max(|F_init|, 1)
    pub free_energy_compression_ratio: f64,
    /// Si se activa veto metacognitivo por sobreconfianza ante contradicción física/causal
    pub self_doubt_veto: bool,
    /// Si se recomienda escalar deliberación activa o consulta a especialistas
    pub escalate_to_active_inquiry: bool,
    /// Nivel de esfuerzo recomendado para el ciclo actual/siguiente
    pub recommended_effort: CognitiveEffortTier,
    /// Presupuesto recomendado de pasos de descenso variacional
    pub recommended_e_steps: usize,
    /// Precisión de política modulada gamma_meta
    pub modulated_policy_gamma: f64,
    /// Pesos epistémico y pragmático modulados
    pub modulated_epistemic_weight: f64,
    pub modulated_pragmatic_weight: f64,
}

/// Estadísticas agregadas de calibración metacognitiva.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveCalibrationStats {
    pub total_observations: usize,
    pub brier_score: f64,
    pub ema_calibration_error: f64,
    pub mean_confidence: f64,
    pub empirical_accuracy: f64,
    pub self_doubt_vetoes: usize,
    pub fast_exit_rate: f64,
}

/// Controlador Metacognitivo de Segundo Orden FEP-MARCUS.
#[derive(Debug, Clone)]
pub struct MetacognitiveController {
    pub base_policy_gamma: f64,
    pub base_epistemic_weight: f64,
    pub base_pragmatic_weight: f64,
    pub fast_confidence_threshold: f64,
    pub doubt_conflict_threshold: f64,
    /// Activa las 4 subrutinas de ADN de AURA v3 Genesis (Mielinización Saltatoria + Pre-calibración Prenatal In-Utero)
    pub aura_v3_genesis: bool,
    pub myelinated_pathways: usize,
    /// Pesos aprendidos en línea para calibración logística de confianza (bias, w_comp, w_ent, w_surp, w_prior)
    calib_weights: [f64; 5],
    total_observations: usize,
    brier_sum: f64,
    ema_calib_err: f64,
    confidence_sum: f64,
    success_sum: f64,
    self_doubt_vetoes: usize,
    fast_exits: usize,
}

impl Default for MetacognitiveController {
    fn default() -> Self {
        Self {
            base_policy_gamma: 1.5,
            base_epistemic_weight: 1.0,
            base_pragmatic_weight: 1.0,
            fast_confidence_threshold: 0.78,
            doubt_conflict_threshold: 0.42,
            aura_v3_genesis: false,
            myelinated_pathways: 0,
            calib_weights: [0.35, 1.40, -0.35, -1.65, 1.15],
            total_observations: 0,
            brier_sum: 0.0,
            ema_calib_err: 0.15,
            confidence_sum: 0.0,
            success_sum: 0.0,
            self_doubt_vetoes: 0,
            fast_exits: 0,
        }
    }
}

impl MetacognitiveController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructor con las 4 subrutinas generativas de ADN de AURA v3 Genesis:
    /// Ejecuta el currículo sintético prenatal in-utero (ondas espontáneas coherentes,
    /// 0 fuga de datos de evaluación) para pre-calibrar los pesos metacognitivos antes del Episodio 0.
    pub fn with_aura_v3_genesis() -> Self {
        let mut ctrl = Self {
            aura_v3_genesis: true,
            fast_confidence_threshold: 0.74,
            ..Self::default()
        };
        ctrl.run_prenatal_in_utero_curriculum();
        ctrl
    }

    /// Subrutina 4 (AURA v3 Genesis): Pre-entrenamiento sintético en el útero (32 ondas espontáneas)
    /// Calibra el mapa topográfico entre compresión de energía libre, entropía y sorpresa sin datos reales.
    pub fn run_prenatal_in_utero_curriculum(&mut self) {
        let lr = 0.14;
        for wave in 0..32 {
            let is_coherent_retinal_wave = wave % 2 == 0;
            let (y, feats) = if is_coherent_retinal_wave {
                // Onda prenatal coherente de baja entropía y alta compresión
                (1.0, [1.0, 0.52, -0.85, 0.01, 0.88])
            } else {
                // Espasmo motor prenatal discordante (conflicto sintético)
                (0.0, [1.0, 0.08, 0.35, 0.65, -0.15])
            };
            let logit = self.calib_weights[0] * feats[0]
                + self.calib_weights[1] * feats[1]
                + self.calib_weights[2] * feats[2]
                + self.calib_weights[3] * feats[3]
                + self.calib_weights[4] * feats[4];
            let p = Self::sigmoid(logit).clamp(0.001, 0.999);
            let err = y - p;
            for i in 0..5 {
                self.calib_weights[i] =
                    (self.calib_weights[i] * 0.999 + lr * err * feats[i]).clamp(-4.5, 4.5);
            }
            self.ema_calib_err = 0.90 * self.ema_calib_err + 0.10 * err.abs();
        }
    }

    #[inline]
    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    /// Calcula la entropía diferencial media por dimensión de q(s) = N(mu, exp(log_var)):
    /// H[q] / d = 0.5 * ln(2 * pi * e) + 0.5 * mean(log_var)
    pub fn compute_normalized_entropy(belief: &VariationalBelief) -> f64 {
        let d = belief.log_variance.len().max(1) as f64;
        let const_term = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln();
        let mean_lv: f64 = belief.log_variance.iter().sum::<f64>() / d;
        const_term + 0.5 * mean_lv
    }

    /// Monitorea un estado de Nivel 1 y genera control metacognitivo de Nivel 2.
    pub fn assess_and_control(
        &mut self,
        belief: &VariationalBelief,
        initial_fe: f64,
        fe_result: &VariationalFreeEnergyResult,
        surprise: &StructuredSurprise,
        precision_profile: &PrecisionProfile,
    ) -> MetacognitiveAssessment {
        let entropy = Self::compute_normalized_entropy(belief);
        let fe_denom = initial_fe.abs().max(1.0);
        let compression = ((initial_fe - fe_result.variational_free_energy) / fe_denom).clamp(-1.0, 1.0);

        // Conflicto inter-canal: confianza asociativa alta combinada con sorpresa estructural alta o invalidez física
        let structural_risk = (surprise.physical_surprise * 0.08
            + surprise.causal_surprise * 0.20
            + surprise.algebraic_surprise * 0.15)
            .clamp(0.0, 2.0);
        let assoc = belief.associative_confidence.clamp(0.0, 1.0);
        let conflict_raw = if !belief.physical_validity {
            1.0
        } else {
            assoc * structural_risk.min(1.0)
        };
        let conflict_index = conflict_raw.clamp(0.0, 1.0);

        // Vector de características metacognitivas para estimar probabilidad de dominio real
        let norm_surp = (surprise.total_surprise / 10.0).clamp(0.0, 2.0);
        let norm_ent = (entropy - 1.0).clamp(-2.0, 2.0);
        let logit = self.calib_weights[0]
            + self.calib_weights[1] * compression.max(0.0)
            + self.calib_weights[2] * norm_ent
            + self.calib_weights[3] * norm_surp
            + self.calib_weights[4] * (assoc - conflict_index);

        let mut confidence = Self::sigmoid(logit).clamp(0.01, 0.99);

        // Veto metacognitivo anti-sobreconfianza (Marcus Vigilance):
        // Si Sistema-1 está seguro pero hay contradicción física, causal o alto conflicto, veta y baja confianza.
        let self_doubt_veto = !belief.physical_validity
            || (assoc >= 0.70 && conflict_index >= self.doubt_conflict_threshold)
            || surprise.physical_surprise > 1.0;

        if self_doubt_veto {
            confidence = confidence.min(if self.aura_v3_genesis { 0.14 } else { 0.24 });
            self.self_doubt_vetoes += 1;
        }

        // Selección racional de esfuerzo computacional + Subrutina 3 (Mielinización Saltatoria en AURA v3 Genesis)
        let (effort_tier, e_steps, escalate) =
            if self_doubt_veto || confidence < 0.55 || surprise.total_surprise > 2.0 {
                (CognitiveEffortTier::ActiveMetaInquiry, 15, true)
            } else if confidence >= self.fast_confidence_threshold
                && conflict_index < 0.20
                && precision_profile.sensory_precision >= 0.8
            {
                self.fast_exits += 1;
                if self.aura_v3_genesis {
                    self.myelinated_pathways += 1;
                    // Ruta mielinizada (axon_delay -> 1): convergencia en 2 pasos saltatorios
                    (CognitiveEffortTier::System1Fast, 2, false)
                } else {
                    (CognitiveEffortTier::System1Fast, 4, false)
                }
            } else {
                let s2_steps = if self.aura_v3_genesis { 7 } else { 9 };
                (CognitiveEffortTier::System2Deliberative, s2_steps, false)
            };

        // Control metacognitivo de política G(pi): sube exploración epistémica cuando la incertidumbre o el error de calibración crecen
        let uncertainty = 1.0 - confidence;
        let modulated_epistemic_weight =
            self.base_epistemic_weight * (0.70 + 1.15 * uncertainty + 0.50 * self.ema_calib_err);
        let modulated_pragmatic_weight =
            self.base_pragmatic_weight * (0.55 + 0.75 * confidence);
        let modulated_policy_gamma =
            (self.base_policy_gamma * (0.65 + 0.90 * confidence)).clamp(0.5, 4.0);

        MetacognitiveAssessment {
            posterior_entropy: entropy,
            calibrated_confidence: confidence,
            conflict_index,
            free_energy_compression_ratio: compression,
            self_doubt_veto,
            escalate_to_active_inquiry: escalate,
            recommended_effort: effort_tier,
            recommended_e_steps: e_steps,
            modulated_policy_gamma,
            modulated_epistemic_weight,
            modulated_pragmatic_weight,
        }
    }

    /// Retroalimenta el resultado real (éxito = true/false) para calibrar en línea la confianza metacognitiva.
    pub fn record_outcome(
        &mut self,
        assessment: &MetacognitiveAssessment,
        belief: &VariationalBelief,
        surprise: &StructuredSurprise,
        actual_success: bool,
    ) {
        let y = if actual_success { 1.0 } else { 0.0 };
        let p = assessment.calibrated_confidence.clamp(0.001, 0.999);
        let err = y - p;
        let brier = err * err;

        self.total_observations += 1;
        self.brier_sum += brier;
        self.confidence_sum += p;
        self.success_sum += y;
        self.ema_calib_err = 0.90 * self.ema_calib_err + 0.10 * err.abs();

        // Actualización SGD regularizada sobre pesos de calibración logística
        let lr = 0.12;
        let norm_surp = (surprise.total_surprise / 10.0).clamp(0.0, 2.0);
        let norm_ent = (assessment.posterior_entropy - 1.0).clamp(-2.0, 2.0);
        let feats = [
            1.0,
            assessment.free_energy_compression_ratio.max(0.0),
            norm_ent,
            norm_surp,
            belief.associative_confidence.clamp(0.0, 1.0) - assessment.conflict_index,
        ];
        for i in 0..5 {
            self.calib_weights[i] =
                (self.calib_weights[i] * 0.998 + lr * err * feats[i]).clamp(-4.0, 4.0);
        }
    }

    pub fn calibration_stats(&self) -> MetacognitiveCalibrationStats {
        let n = self.total_observations.max(1) as f64;
        MetacognitiveCalibrationStats {
            total_observations: self.total_observations,
            brier_score: self.brier_sum / n,
            ema_calibration_error: self.ema_calib_err,
            mean_confidence: self.confidence_sum / n,
            empirical_accuracy: self.success_sum / n,
            self_doubt_vetoes: self.self_doubt_vetoes,
            fast_exit_rate: self.fast_exits as f64 / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metacognition_detects_overconfident_physical_conflict() {
        let mut meta = MetacognitiveController::new();
        let mut belief = VariationalBelief::new(8, "illusion");
        belief.associative_confidence = 0.95; // Sistema-1 muy seguro
        belief.physical_validity = false;     // Violación Spelke

        let fe_res = VariationalFreeEnergyResult {
            variational_free_energy: 20.0,
            evidence_lower_bound: -20.0,
            accuracy: -18.0,
            complexity: 2.0,
            surprise_upper_bound: 20.0,
            precision_weighted_residuals: vec![0.2; 8],
        };
        let surprise = StructuredSurprise {
            perceptual_surprise: 0.4,
            algebraic_surprise: 0.2,
            physical_surprise: 50.0,
            causal_surprise: 0.0,
            total_surprise: 50.6,
            dominant_source: "PhysicalSpelke".to_string(),
            diagnostic_details: vec![],
        };

        let assess = meta.assess_and_control(
            &belief,
            24.0,
            &fe_res,
            &surprise,
            &PrecisionProfile::default(),
        );

        assert!(assess.self_doubt_veto);
        assert!(assess.escalate_to_active_inquiry);
        assert_eq!(assess.recommended_effort, CognitiveEffortTier::ActiveMetaInquiry);
        assert!(assess.calibrated_confidence <= 0.25);
        assert!(assess.modulated_epistemic_weight > meta.base_epistemic_weight);
    }

    #[test]
    fn test_metacognition_calibration_improves_brier_and_routes_fast_when_confident() {
        let mut meta = MetacognitiveController::new();
        let mut belief = VariationalBelief::new(8, "known");
        belief.associative_confidence = 0.92;
        belief.log_variance = vec![-2.5; 8];

        let fe_res = VariationalFreeEnergyResult {
            variational_free_energy: 6.0,
            evidence_lower_bound: -6.0,
            accuracy: -5.5,
            complexity: 0.5,
            surprise_upper_bound: 6.0,
            precision_weighted_residuals: vec![0.02; 8],
        };
        let surprise = StructuredSurprise {
            perceptual_surprise: 0.02,
            algebraic_surprise: 0.0,
            physical_surprise: 0.0,
            causal_surprise: 0.0,
            total_surprise: 0.02,
            dominant_source: "None".to_string(),
            diagnostic_details: vec![],
        };

        let profile = PrecisionProfile::default();
        let mut first_brier = 0.0;
        let mut last_brier = 0.0;
        for step in 0..40 {
            let assess = meta.assess_and_control(&belief, 14.0, &fe_res, &surprise, &profile);
            let b = (1.0 - assess.calibrated_confidence).powi(2);
            if step == 0 {
                first_brier = b;
            }
            last_brier = b;
            meta.record_outcome(&assess, &belief, &surprise, true);
        }
        let final_assess = meta.assess_and_control(&belief, 14.0, &fe_res, &surprise, &profile);
        assert!(last_brier < first_brier);
        assert_eq!(final_assess.recommended_effort, CognitiveEffortTier::System1Fast);
        assert_eq!(final_assess.recommended_e_steps, 4);
    }
}
