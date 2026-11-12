//! Módulo de Energía Libre Variacional (Variational Free Energy & ELBO Maximization).
//!
//! Basado en el Principio de Energía Libre (Karl Friston) acoplado a la cognición
//! neuro-simbólica estructurada de Gary Marcus:
//!
//! 1. Sorpresa (Surprisal): S(o) = -ln p(o)
//! 2. Energía Libre Variacional: F(q, o) = D_KL(q(s) || p(s)) - E_q[ln p(o | s)] >= S(o)
//! 3. Maximización de Evidencia (ELBO): F_evidence = -F(q, o) = E_q[ln p(o | s)] - D_KL(q(s) || p(s)) <= ln p(o)
//!
//! Maximizar la energía libre de evidencia (ELBO) minimiza directamente la sorpresa sensorial y
//! estructural del agente, garantizando auto-organización, homeostasis y calibración epistémica.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Densidad de reconocimiento / Creencia variacional q(s) sobre el estado latente.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariationalBelief {
    pub state_id: String,
    /// Media del estado latente continuo mu in R^d
    pub mean_latent: Vec<f64>,
    /// Log-varianza del estado latente ln(sigma^2)
    pub log_variance: Vec<f64>,
    /// Ligamientos simbólicos activos (Variables formales -> Valores/Tokens)
    pub variable_bindings: HashMap<String, String>,
    /// Si el estado satisface los axiomas físicos de Spelke
    pub physical_validity: bool,
    /// Confianza asociativa proveniente del Sistema 1
    pub associative_confidence: f64,
}

impl VariationalBelief {
    pub fn new(dim: usize, state_id: impl Into<String>) -> Self {
        Self {
            state_id: state_id.into(),
            mean_latent: vec![0.0; dim],
            log_variance: vec![0.0; dim], // sigma^2 = 1.0 por defecto
            variable_bindings: HashMap::new(),
            physical_validity: true,
            associative_confidence: 0.5,
        }
    }

    pub fn with_mean_and_variance(
        mut self,
        mean: Vec<f64>,
        log_var: Vec<f64>,
    ) -> Self {
        self.mean_latent = mean;
        self.log_variance = log_var;
        self
    }

    pub fn with_binding(mut self, var: impl Into<String>, val: impl Into<String>) -> Self {
        self.variable_bindings.insert(var.into(), val.into());
        self
    }
}

/// Distribución a priori generativa p(s).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorBelief {
    pub prior_mean: Vec<f64>,
    pub prior_log_var: Vec<f64>,
    pub expected_bindings: HashMap<String, String>,
}

impl PriorBelief {
    pub fn standard_normal(dim: usize) -> Self {
        Self {
            prior_mean: vec![0.0; dim],
            prior_log_var: vec![0.0; dim],
            expected_bindings: HashMap::new(),
        }
    }

    pub fn with_informative_prior(
        prior_mean: Vec<f64>,
        prior_log_var: Vec<f64>,
    ) -> Self {
        Self {
            prior_mean,
            prior_log_var,
            expected_bindings: HashMap::new(),
        }
    }
}

/// Resultado formal del cómputo de Energía Libre Variacional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalFreeEnergyResult {
    /// F(q, o) = Complejidad - Precisión (Friston Free Energy, a minimizar)
    pub variational_free_energy: f64,
    /// F_evidence = -F(q, o) = Precisión - Complejidad (ELBO, a maximizar)
    pub evidence_lower_bound: f64,
    /// Precisión observacional esperada: E_q[ln p(o | s)]
    pub accuracy: f64,
    /// Complejidad / Divergencia KL: D_KL(q(s) || p(s))
    pub complexity: f64,
    /// Cota superior sobre la sorpresa sensorial: S(o) <= F(q, o)
    pub surprise_upper_bound: f64,
    /// Residuos ponderados por precisión por dimensión: pi_i * (o_i - g(mu_i))
    pub precision_weighted_residuals: Vec<f64>,
}

/// Calculador y Optimizador de Energía Libre Variacional.
#[derive(Debug, Clone)]
pub struct FreeEnergyOptimizer {
    pub learning_rate: f64,
    pub max_optimization_steps: usize,
    pub convergence_epsilon: f64,
}

impl Default for FreeEnergyOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            max_optimization_steps: 50,
            convergence_epsilon: 1e-4,
        }
    }
}

impl FreeEnergyOptimizer {
    pub fn new(lr: f64, steps: usize, eps: f64) -> Self {
        Self {
            learning_rate: lr,
            max_optimization_steps: steps,
            convergence_epsilon: eps,
        }
    }

    /// Calcula la divergencia KL analítica entre dos densidades gaussianas multivariadas diagonales:
    /// D_KL(q || p) = 0.5 * sum [ (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2 - 1 - ln(sigma_q^2 / sigma_p^2) ]
    pub fn compute_kl_divergence(
        q_mean: &[f64],
        q_log_var: &[f64],
        p_mean: &[f64],
        p_log_var: &[f64],
    ) -> f64 {
        assert_eq!(q_mean.len(), q_log_var.len());
        assert_eq!(q_mean.len(), p_mean.len());
        assert_eq!(q_mean.len(), p_log_var.len());

        let mut kl = 0.0;
        for i in 0..q_mean.len() {
            let var_q = q_log_var[i].exp();
            let var_p = p_log_var[i].exp();
            let diff = q_mean[i] - p_mean[i];

            let term = (var_q + diff * diff) / var_p - 1.0 - (q_log_var[i] - p_log_var[i]);
            kl += 0.5 * term;
        }
        kl.max(0.0)
    }

    /// Calcula la precisión log-verosimilitud observacional:
    /// E_q[ln p(o | s)] = -0.5 * sum [ pi_i * (o_i - mu_i)^2 + pi_i * sigma_q,i^2 - ln(pi_i / (2*pi)) ]
    pub fn compute_expected_log_likelihood(
        observation: &[f64],
        q_mean: &[f64],
        q_log_var: &[f64],
        precisions: &[f64],
    ) -> (f64, Vec<f64>) {
        assert_eq!(observation.len(), q_mean.len());
        assert_eq!(q_mean.len(), q_log_var.len());
        assert_eq!(observation.len(), precisions.len());

        let mut accuracy = 0.0;
        let mut residuals = Vec::with_capacity(observation.len());
        let log_2pi = (2.0 * std::f64::consts::PI).ln();

        for i in 0..observation.len() {
            let pi_i = precisions[i].max(1e-6);
            let diff = observation[i] - q_mean[i];
            let var_q = q_log_var[i].exp();

            let weighted_err = pi_i * (diff * diff + var_q);
            let log_norm = pi_i.ln() - log_2pi;
            let log_p = 0.5 * (log_norm - weighted_err);

            accuracy += log_p;
            residuals.push(pi_i * diff);
        }

        (accuracy, residuals)
    }

    /// Evalúa la Energía Libre Variacional F(q, o) y el ELBO para un estado variacional dado.
    pub fn evaluate(
        &self,
        observation: &[f64],
        q: &VariationalBelief,
        prior: &PriorBelief,
        precisions: &[f64],
    ) -> VariationalFreeEnergyResult {
        let complexity = Self::compute_kl_divergence(
            &q.mean_latent,
            &q.log_variance,
            &prior.prior_mean,
            &prior.prior_log_var,
        );

        let (accuracy, residuals) = Self::compute_expected_log_likelihood(
            observation,
            &q.mean_latent,
            &q.log_variance,
            precisions,
        );

        // F(q, o) = Complejidad - Accuracy
        let free_energy = complexity - accuracy;
        // ELBO = Accuracy - Complejidad = -F(q, o)
        let elbo = accuracy - complexity;

        VariationalFreeEnergyResult {
            variational_free_energy: free_energy,
            evidence_lower_bound: elbo,
            accuracy,
            complexity,
            surprise_upper_bound: free_energy,
            precision_weighted_residuals: residuals,
        }
    }

    /// Optimiza iterativamente la creencia variacional q(s) para minimizar F(q, o)
    /// (equivalentemente, maximizar ELBO) mediante descenso de gradiente sobre mu y log_var.
    pub fn optimize_posterior(
        &self,
        observation: &[f64],
        initial_q: &VariationalBelief,
        prior: &PriorBelief,
        precisions: &[f64],
    ) -> (VariationalBelief, Vec<f64>) {
        let mut current_q = initial_q.clone();
        let dim = observation.len();
        let mut free_energy_history = Vec::new();

        for _step in 0..self.max_optimization_steps {
            let res = self.evaluate(observation, &current_q, prior, precisions);
            free_energy_history.push(res.variational_free_energy);

            // Gradiente respecto a mu_i:
            // dF/dmu_i = dComplexity/dmu_i - dAccuracy/dmu_i
            // dComplexity/dmu_i = (mu_i - p_mean_i) / p_var_i
            // dAccuracy/dmu_i = pi_i * (observation_i - mu_i)
            // Por tanto: grad_mu = (mu_i - p_mean_i) / p_var_i - pi_i * (observation_i - mu_i)
            let mut max_grad_norm = 0.0f64;

            for i in 0..dim {
                let p_var = prior.prior_log_var[i].exp();
                let pi_i = precisions[i].max(1e-6);

                let d_comp_dmu = (current_q.mean_latent[i] - prior.prior_mean[i]) / p_var;
                let d_acc_dmu = pi_i * (observation[i] - current_q.mean_latent[i]);
                let grad_mu = d_comp_dmu - d_acc_dmu;

                // Gradiente respecto a log_variance_i:
                // Sea s_i = log_var_i, var_q = exp(s_i)
                // dComplexity/ds_i = 0.5 * (var_q / p_var - 1)
                // dAccuracy/ds_i = -0.5 * pi_i * var_q
                // grad_s = 0.5 * (var_q / p_var - 1 + pi_i * var_q)
                let var_q = current_q.log_variance[i].exp();
                let grad_s = 0.5 * (var_q / p_var - 1.0 + pi_i * var_q);

                current_q.mean_latent[i] -= self.learning_rate * grad_mu;
                current_q.log_variance[i] -= self.learning_rate * grad_s;

                max_grad_norm = max_grad_norm.max(grad_mu.abs()).max(grad_s.abs());
            }

            if max_grad_norm < self.convergence_epsilon {
                break;
            }
        }

        (current_q, free_energy_history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_energy_and_elbo_symmetry() {
        let dim = 4;
        let opt = FreeEnergyOptimizer::default();
        let obs = vec![1.0, 0.5, -0.5, 2.0];
        let precisions = vec![1.0; dim];
        let q = VariationalBelief::new(dim, "q0");
        let prior = PriorBelief::standard_normal(dim);

        let res = opt.evaluate(&obs, &q, &prior, &precisions);

        // ELBO debe ser exactamente el opuesto de Free Energy
        assert!((res.evidence_lower_bound + res.variational_free_energy).abs() < 1e-9);
        assert!(res.complexity >= 0.0);
    }

    #[test]
    fn test_free_energy_minimization_reduces_surprise() {
        let dim = 8;
        let opt = FreeEnergyOptimizer::new(0.1, 40, 1e-5);
        let obs = vec![0.8, -1.2, 0.4, 0.0, 1.5, -0.7, 0.3, -0.1];
        let precisions = vec![2.0; dim];
        let q_init = VariationalBelief::new(dim, "q_init");
        let prior = PriorBelief::standard_normal(dim);

        let initial_fe = opt.evaluate(&obs, &q_init, &prior, &precisions).variational_free_energy;
        let (q_opt, history) = opt.optimize_posterior(&obs, &q_init, &prior, &precisions);
        let final_fe = opt.evaluate(&obs, &q_opt, &prior, &precisions).variational_free_energy;

        // La energía libre debe decrecer estrictamente
        assert!(final_fe < initial_fe, "Final FE ({}) no es menor que inicial ({})", final_fe, initial_fe);
        assert!(!history.is_empty());
        assert!(history.last().unwrap() <= history.first().unwrap());
    }
}
