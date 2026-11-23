//! Módulo de Dinámica de Precisión Sináptica y Modulación de Ganancia (Precision Dynamics).
//!
//! En Inferencia Activa y Codificación Predictiva (Friston & Feldman, 2012), la precisión (pi)
//! es la inversa de la varianza condicional (pi = 1 / sigma^2), y actúa como un control de ganancia
//! neuromoduladora sobre los errores de predicción:
//!
//! epsilon_ponderado = pi * (observacion - prediccion)
//!
//! En la síntesis con Gary Marcus:
//! 1. Los invariantes del Andamiaje Innato de Spelke poseen precisión infinita/máxima (innegociables).
//! 2. Las proposiciones causales do(X) poseen precisión superior a correlaciones observacionales pasivas.
//! 3. Las matrices de sinapsis siguen la Ley de Dale (W^E, W^I >= 0), garantizando que la modulación
//!    de precisión sea numéricamente estable y libre de oscilaciones caóticas.

use serde::{Deserialize, Serialize};

/// Canal cognitivo para el control de precisión.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveChannel {
    SensoryContinuous,
    AlgebraicBinding,
    SpelkePhysicalInvariants,
    PearlCausalInterventions,
    EpistemicProvenance,
}

/// Estado dinámico del controlador de precisión.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionProfile {
    pub sensory_precision: f64,
    pub algebraic_precision: f64,
    pub physical_precision: f64,
    pub causal_precision: f64,
    pub epistemic_precision: f64,
}

impl Default for PrecisionProfile {
    fn default() -> Self {
        Self {
            sensory_precision: 1.0,
            algebraic_precision: 4.0,
            // Spelke physics tiene la máxima prioridad ontológica
            physical_precision: 50.0,
            // Causalidad Pearl supera a correlación
            causal_precision: 8.0,
            epistemic_precision: 10.0,
        }
    }
}

/// Gestor de dinámica de precisión con adaptación homeostática basada en errores pasados.
#[derive(Debug, Clone)]
pub struct PrecisionManager {
    pub current_profile: PrecisionProfile,
    pub learning_rate: f64,
    pub min_precision: f64,
    pub max_precision: f64,
    /// Historial de volatilidad de errores recientes
    error_volatility_history: Vec<f64>,
}

impl Default for PrecisionManager {
    fn default() -> Self {
        Self {
            current_profile: PrecisionProfile::default(),
            learning_rate: 0.05,
            min_precision: 0.1,
            max_precision: 100.0,
            error_volatility_history: Vec::new(),
        }
    }
}

impl PrecisionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Obtiene la precisión para un canal específico.
    pub fn get_precision(&self, channel: CognitiveChannel) -> f64 {
        match channel {
            CognitiveChannel::SensoryContinuous => self.current_profile.sensory_precision,
            CognitiveChannel::AlgebraicBinding => self.current_profile.algebraic_precision,
            CognitiveChannel::SpelkePhysicalInvariants => self.current_profile.physical_precision,
            CognitiveChannel::PearlCausalInterventions => self.current_profile.causal_precision,
            CognitiveChannel::EpistemicProvenance => self.current_profile.epistemic_precision,
        }
    }

    /// Pondera un error de predicción por su precisión:
    /// e_tilde = pi * e
    pub fn weight_error(&self, error: f64, channel: CognitiveChannel) -> f64 {
        self.get_precision(channel) * error
    }

    /// Actualiza la precisión sensorial continua en función de la volatilidad observada.
    /// Si los errores sensoriales son estables y pequeños, aumenta la precisión (mayor confianza).
    /// Si los errores son ruidosos y altamente variables, reduce la precisión (menor confianza sensorial).
    pub fn update_sensory_precision(&mut self, current_residual_norm: f64) {
        self.error_volatility_history.push(current_residual_norm);
        if self.error_volatility_history.len() > 20 {
            self.error_volatility_history.remove(0);
        }

        if self.error_volatility_history.len() >= 3 {
            let mean = self.error_volatility_history.iter().sum::<f64>()
                / self.error_volatility_history.len() as f64;
            let variance = self.error_volatility_history
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / self.error_volatility_history.len() as f64;

            // La precisión es inversamente proporcional a la varianza del error + regularizador
            let target_precision = 1.0 / (variance + 0.05);
            let updated = (1.0 - self.learning_rate) * self.current_profile.sensory_precision
                + self.learning_rate * target_precision;

            self.current_profile.sensory_precision = updated
                .clamp(self.min_precision, self.max_precision);
        }
    }

    /// Aumenta temporalmente la precisión causal durante un experimento activo do(X).
    pub fn boost_interventional_precision(&mut self, factor: f64) {
        self.current_profile.causal_precision = (self.current_profile.causal_precision * factor)
            .min(self.max_precision);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_profile_invariants() {
        let mgr = PrecisionManager::default();
        // La física de Spelke debe tener mayor precisión que la percepción continua
        assert!(
            mgr.get_precision(CognitiveChannel::SpelkePhysicalInvariants)
                > mgr.get_precision(CognitiveChannel::SensoryContinuous)
        );
        // Causalidad debe tener mayor precisión que percepción pasiva
        assert!(
            mgr.get_precision(CognitiveChannel::PearlCausalInterventions)
                > mgr.get_precision(CognitiveChannel::SensoryContinuous)
        );
    }

    #[test]
    fn test_precision_adaptation_to_noise() {
        let mut mgr = PrecisionManager::default();
        let initial_p = mgr.get_precision(CognitiveChannel::SensoryContinuous);

        // Inyectar errores altamente ruidosos y variables
        for i in 0..15 {
            let noise = if i % 2 == 0 { 10.0 } else { 0.1 };
            mgr.update_sensory_precision(noise);
        }

        let adapted_p = mgr.get_precision(CognitiveChannel::SensoryContinuous);
        // Ante alta varianza, la precisión debe descender
        assert!(adapted_p < initial_p);
    }
}
