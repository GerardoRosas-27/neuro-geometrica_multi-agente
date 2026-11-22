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
    /// Activa las 4 subrutinas generativas de ADN de AURA v3 Genesis
    /// (Gradientes Morfogenéticos 3D, Microcircuito Canónico de 6 Capas Mountcastle,
    /// Sobreproducción + Poda por Microglía + Mielinización, y Ondas Prenatales In-Utero).
    pub aura_v3_genesis: bool,
}

impl Default for FreeEnergyOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            max_optimization_steps: 50,
            convergence_epsilon: 1e-4,
            aura_v3_genesis: false,
        }
    }
}

impl FreeEnergyOptimizer {
    pub fn new(lr: f64, steps: usize, eps: f64) -> Self {
        Self {
            learning_rate: lr,
            max_optimization_steps: steps,
            convergence_epsilon: eps,
            aura_v3_genesis: false,
        }
    }

    /// Constructor con las 4 subrutinas generativas de ADN de AURA v3 Genesis activadas.
    pub fn with_aura_v3_genesis(mut self) -> Self {
        self.aura_v3_genesis = true;
        self
    }

    /// Subrutina 1 (AURA v3 Genesis): Factor de afinidad quimiotáctica por gradientes
    /// morfogenéticos 3D [SHH, Wnt, BMP] sin direcciones explícitas.
    #[inline]
    pub fn morphogen_gradient_gain(dim_idx: usize, total_dim: usize) -> f64 {
        let u = (dim_idx as f64) / (total_dim.max(1) as f64);
        let c_shh = (-2.2 * u).exp();
        let c_wnt = (-2.0 * (u - 0.5).powi(2)).exp();
        let c_bmp = (-2.2 * (1.0 - u)).exp();
        // Ganancia homeostática balanceada por receptores EphA/EphB/Robo
        1.0 + 0.18 * (0.35 * c_shh + 0.40 * c_wnt + 0.25 * c_bmp)
    }

    /// Subrutina 4 (AURA v3 Genesis): Pre-calibración prenatal in-utero mediante
    /// ondas espontáneas coherentes (retinal waves & motor twitches) sin fuga de datos.
    pub fn calibrate_prenatal_prior(&self, prior: &PriorBelief, precisions: &[f64]) -> PriorBelief {
        if !self.aura_v3_genesis {
            return prior.clone();
        }
        let dim = prior.prior_mean.len();
        let mut cal_mean = prior.prior_mean.clone();
        let mut cal_lv = prior.prior_log_var.clone();
        // 16 ondas armónicas prenatales espontáneas (0 datos externos)
        for wave in 0..16 {
            let phase = (wave as f64) * std::f64::consts::PI / 8.0;
            for i in 0..dim {
                let morph = Self::morphogen_gradient_gain(i, dim);
                let pi_eff = (precisions[i].max(1e-6)) * morph;
                let retinal_wave = (((i as f64) * 0.2).sin() * phase.cos()) * 0.04;
                cal_mean[i] = 0.92 * cal_mean[i] + 0.08 * retinal_wave;
                // Pre-ajuste homeostático suave hacia precisión sensorial esperada
                let target_lv = -(1.0 + 0.12 * pi_eff).ln();
                cal_lv[i] = 0.90 * cal_lv[i] + 0.10 * target_lv;
            }
        }
        PriorBelief {
            prior_mean: cal_mean,
            prior_log_var: cal_lv,
            expected_bindings: prior.expected_bindings.clone(),
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

            // Amortiguamiento de gradiente natural por curvatura diagonal (Levenberg-Marquardt):
            // Garantiza convergencia monótona incluso en canales de alta precisión (e.g. Spelke pi=50).
            let mut max_grad_norm = 0.0f64;
            let mut grad_mu_vec = Vec::with_capacity(dim);
            let mut grad_s_vec = Vec::with_capacity(dim);
            let mut step_mu_vec = Vec::with_capacity(dim);
            let mut step_s_vec = Vec::with_capacity(dim);

            for i in 0..dim {
                let p_var = prior.prior_log_var[i].exp().max(1e-8);
                let pi_i = precisions[i].max(1e-6);
                let h_mu = (1.0 / p_var) + pi_i;

                let d_comp_dmu = (current_q.mean_latent[i] - prior.prior_mean[i]) / p_var;
                let d_acc_dmu = pi_i * (observation[i] - current_q.mean_latent[i]);
                let grad_mu = d_comp_dmu - d_acc_dmu;

                let var_q = current_q.log_variance[i].exp();
                let grad_s = 0.5 * (var_q / p_var - 1.0 + pi_i * var_q);
                let h_s = 0.5 * var_q * h_mu;

                if self.aura_v3_genesis {
                    // Subrutina 1 & 2 (AURA v3 Genesis): Microcircuito Cortical Canónico de 6 Capas
                    // (L4 entrada talámica -> L2/3 codificación predictiva lateral -> L5 salida -> L6 amortiguamiento apical)
                    // modulado por gradiente morfogenético 3D [SHH, Wnt, BMP] y poda por microglía de micro-ruido.
                    let morph_gain = Self::morphogen_gradient_gain(i, dim);
                    let eff_lr_mu = (0.58 * morph_gain / h_mu).min(0.92 / h_mu);
                    let target_lv = -h_mu.ln();
                    let lv_err = current_q.log_variance[i] - target_lv;
                    // Subrutina 3 (Microglía): poda de micro-oscilaciones residuales bajo umbral homeostático
                    let pruned_grad_mu = if grad_mu.abs() < 1e-5 { 0.0 } else { grad_mu };
                    grad_mu_vec.push(pruned_grad_mu);
                    grad_s_vec.push(grad_s);
                    step_mu_vec.push(eff_lr_mu * pruned_grad_mu);
                    step_s_vec.push(0.42 * lv_err);
                    max_grad_norm = max_grad_norm.max(pruned_grad_mu.abs()).max(lv_err.abs());
                } else {
                    let base_lr = self.learning_rate.max(1e-5);
                    let eff_lr_mu = base_lr / (1.0 + base_lr * h_mu);
                    let eff_lr_s = base_lr / (1.0 + base_lr * h_s);

                    grad_mu_vec.push(grad_mu);
                    grad_s_vec.push(grad_s);
                    step_mu_vec.push(eff_lr_mu * grad_mu);
                    step_s_vec.push(eff_lr_s * grad_s);

                    max_grad_norm = max_grad_norm.max(grad_mu.abs()).max(grad_s.abs());
                }
            }

            // Búsqueda lineal con garantía estricta de no-incremento de Energía Libre
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..5 {
                let mut candidate = current_q.clone();
                for i in 0..dim {
                    candidate.mean_latent[i] -= alpha * step_mu_vec[i];
                    candidate.log_variance[i] =
                        (candidate.log_variance[i] - alpha * step_s_vec[i]).clamp(-12.0, 6.0);
                }
                let cand_fe = self.evaluate(observation, &candidate, prior, precisions).variational_free_energy;
                if cand_fe <= res.variational_free_energy + 1e-12 {
                    current_q = candidate;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }

            if !accepted || max_grad_norm < self.convergence_epsilon {
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

    #[test]
    fn test_high_precision_spelke_stability() {
        let dim = 8;
        let opt = FreeEnergyOptimizer::new(0.08, 40, 1e-5);
        let obs = vec![1.0, -0.8, 0.5, -0.3, 1.2, -1.1, 0.2, 0.9];
        let precisions = vec![50.0; dim];
        let q_init = VariationalBelief::new(dim, "q_spelke");
        let prior = PriorBelief::standard_normal(dim);

        let initial_fe = opt.evaluate(&obs, &q_init, &prior, &precisions).variational_free_energy;
        let (q_opt, history) = opt.optimize_posterior(&obs, &q_init, &prior, &precisions);
        let final_res = opt.evaluate(&obs, &q_opt, &prior, &precisions);

        assert!(final_res.variational_free_energy.is_finite());
        assert!(final_res.variational_free_energy < initial_fe);
        assert!(history.windows(2).all(|w| w[1] <= w[0] + 1e-9));
        assert!((final_res.evidence_lower_bound + final_res.variational_free_energy).abs() < 1e-9);
    }
}
