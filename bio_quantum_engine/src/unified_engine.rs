//! Motor Unificado: BioQuantumConstellationEngine
//!
//! Orquesta y unifica los 5 hitos cuantitativos y teóricos en una sola arquitectura ejecutable:
//! 1. [Bio-Constelación]: L2/3 y L5 piramidales con Ley de Dale (0 violaciones), BTSP y Tsodyks-Markram.
//! 2. [Constelación v4]: Formulación residual puntual equivariante $\hat{q}_p(t+1) = q_p(t) + \Delta \hat{q}_p$.
//! 3. [ÓRBITA vs TimesFM 2.5]: Operador espectral de Koopman con transporte óptimo Brenier y sesgo afín.
//! 4. [CDT Cuenca]: Memoria simplicial con cota formal de olvido acotado $\Delta R \ge -\varepsilon$ en sueño SWR.
//! 5. [Wheeler-Feynman]: Handshake de interferencia de onda retardada y avanzada en el plano complejo $\mathbb{C}$.

use crate::cdt_basin_memory::CdtSimplicialBasinMemory;
use crate::dendritic_koopman_dale::CorticalPyramidalUnit;
use crate::equivariant_residual::{AutonomousThalamocorticalPrior, EquivariantBasisExtractor};
use crate::handshake_phasor::WheelerFeynmanHandshakeEngine;
use crate::koopman_brenier_spectral::KoopmanBrenierSpectralOperator;
use crate::rng::Xoshiro256StarStar;

/// Configuración global de la arquitectura unificada
#[derive(Debug, Clone)]
pub struct UnifiedEngineConfig {
    pub n_points: usize,
    pub d_coords: usize,
    pub d_basis: usize,
    pub d_latent: usize,
    pub epsilon_basin: f64,
    pub temperature: f64,
    pub eta_btsp: f64,
    pub eta_slow: f64,
}

impl Default for UnifiedEngineConfig {
    fn default() -> Self {
        Self {
            n_points: 6,
            d_coords: 12,
            d_basis: 18,
            d_latent: 10, // 5 bloques rotacionales 2x2
            epsilon_basin: 0.05,
            temperature: 0.2,
            eta_btsp: 0.08,
            eta_slow: 0.01,
        }
    }
}

/// Núcleo de la Arquitectura Bio-Cuántica-Geométrica
#[derive(Debug, Clone)]
pub struct BioQuantumConstellationEngine {
    pub config: UnifiedEngineConfig,
    /// Extractor de base equivariante por punto $\phi(q_p)$
    pub basis_extractor: EquivariantBasisExtractor,
    /// Prior tálamo-cortical de 2 capas autónomo (100% Dale)
    pub thalamocortical_prior: AutonomousThalamocorticalPrior,
    /// Motor de Handshake y fasores complejos $\mathbb{C}$
    pub handshake_engine: WheelerFeynmanHandshakeEngine,
    /// Columnas piramidales corticales multicompartimentales (una por punto)
    pub pyramidal_units: Vec<CorticalPyramidalUnit>,
    /// Operador espectral Koopman-Brenier (TimesFM beater)
    pub spectral_operator: KoopmanBrenierSpectralOperator,
    /// Memoria topológica CDT con teorema de cuenca y olvido acotado
    pub cdt_memory: CdtSimplicialBasinMemory,
    /// Pesos lentos de proyección persistente $W^P = W^{P,E} - W^{P,I}$ (Dale compliant)
    pub w_proj_e: Vec<Vec<f64>>,
    pub w_proj_i: Vec<Vec<f64>>,
    /// Campo rápido adaptativo BTSP / Cholesky: $W^{F,E} \ge 0, W^{F,I} \ge 0$ (Dale compliant)
    pub wf_exc: Vec<f64>,
    pub wf_inh: Vec<f64>,
    /// Reloj de ciclo circadiano: 0.0 = vigilia plena, 1.0 = sueño profundo
    pub circadian_phase: f64,
}

impl BioQuantumConstellationEngine {
    pub fn new(config: UnifiedEngineConfig, rng: &mut Xoshiro256StarStar) -> Self {
        let basis_extractor = EquivariantBasisExtractor::new(config.n_points);
        let thalamocortical_prior = AutonomousThalamocorticalPrior::new(config.d_basis, 16, 2, rng);
        let handshake_engine = WheelerFeynmanHandshakeEngine::new(config.d_basis, config.temperature);

        let mut pyramidal_units = Vec::with_capacity(config.n_points);
        for p in 0..config.n_points {
            pyramidal_units.push(CorticalPyramidalUnit::new(p, config.d_basis, config.d_latent, rng));
        }

        let spectral_operator = KoopmanBrenierSpectralOperator::new(config.d_latent, rng);
        let cdt_memory = CdtSimplicialBasinMemory::new(config.d_latent, config.epsilon_basin);

        let mut w_proj_e = vec![vec![0.0; config.d_latent]; 2];
        let mut w_proj_i = vec![vec![0.0; config.d_latent]; 2];
        let scale = 1.0 / (config.d_latent as f64).sqrt();

        for coord in 0..2 {
            for j in 0..config.d_latent {
                w_proj_e[coord][j] = (rng.next_f64() * 0.4 + 0.1) * scale;
                w_proj_i[coord][j] = (rng.next_f64() * 0.2 + 0.05) * scale;
            }
        }

        // Pesos rápidos: 2 coordenadas (dx, dy) x d_basis
        let total_fast = 2 * config.d_basis;
        let wf_exc = vec![1e-4; total_fast];
        let wf_inh = vec![1e-4; total_fast];

        Self {
            config,
            basis_extractor,
            thalamocortical_prior,
            handshake_engine,
            pyramidal_units,
            spectral_operator,
            cdt_memory,
            w_proj_e,
            w_proj_i,
            wf_exc,
            wf_inh,
            circadian_phase: 0.0,
        }
    }

    /// Reinicia los campos rápidos antes de un nuevo episodio
    pub fn begin_episode(&mut self) {
        for v in self.wf_exc.iter_mut() {
            *v = 1e-4;
        }
        for v in self.wf_inh.iter_mut() {
            *v = 1e-4;
        }
    }

    /// Adaptación Rápida Residual Óptima por Cholesky con Proyección a Ley de Dale:
    /// Se ejecuta sobre las primeras $K_{\text{obs}}=4$ observaciones del episodio.
    /// Resuelve $(X^T X + \lambda I) W = X^T Y$ y proyecta sobre $W^{F,E} \ge 0, W^{F,I} \ge 0$.
    pub fn adapt_to_observations(&mut self, frames: &[Vec<f64>]) {
        if frames.len() < 2 {
            return;
        }

        let d = self.config.d_basis;
        let n_trans = frames.len() - 1;

        let mut x_mat = Vec::with_capacity(n_trans * self.config.n_points);
        let mut y_mat = Vec::with_capacity(n_trans * self.config.n_points);

        for t in 0..n_trans {
            let q_curr = &frames[t];
            let q_next = &frames[t + 1];
            let bary = self.basis_extractor.compute_barycenter(q_curr);

            for p in 0..self.config.n_points {
                let x = q_curr[p * 2];
                let y = q_curr[p * 2 + 1];
                let tx = q_next[p * 2];
                let ty = q_next[p * 2 + 1];

                let phi = self.basis_extractor.extract_point_basis(q_curr, p, bary);
                let prior = self.thalamocortical_prior.forward(&phi);

                let true_dx = tx - x;
                let true_dy = ty - y;

                let residual_dx = true_dx - prior[0] * 0.1;
                let residual_dy = true_dy - prior[1] * 0.1;

                x_mat.push(phi);
                y_mat.push([residual_dx, residual_dy]);
            }
        }

        // Cholesky Ridge Regression: $A = X^T X + \lambda I$, $B = X^T Y$
        let lam = 1e-11;
        let mut a = vec![vec![0.0; d]; d];
        let mut b = vec![[0.0; 2]; d];

        for (phi, dy) in x_mat.iter().zip(y_mat.iter()) {
            for i in 0..d {
                for j in 0..d {
                    a[i][j] += phi[i] * phi[j];
                }
                b[i][0] += phi[i] * dy[0];
                b[i][1] += phi[i] * dy[1];
            }
        }

        for i in 0..d {
            a[i][i] += lam;
        }

        // Factorización de Cholesky $A = L L^T$
        let mut l_mat = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i][j];
                for k in 0..j {
                    sum -= l_mat[i][k] * l_mat[j][k];
                }
                if i == j {
                    l_mat[i][j] = sum.max(1e-12).sqrt();
                } else {
                    l_mat[i][j] = sum / l_mat[j][j];
                }
            }
        }

        // Resolver $L z = B$, luego $L^T W = z$
        let mut w_opt = vec![[0.0; 2]; d];
        for c in 0..2 {
            let mut z = vec![0.0; d];
            for i in 0..d {
                let mut sum = b[i][c];
                for k in 0..i {
                    sum -= l_mat[i][k] * z[k];
                }
                z[i] = sum / l_mat[i][i];
            }
            for i in (0..d).rev() {
                let mut sum = z[i];
                for k in (i + 1)..d {
                    sum -= l_mat[k][i] * w_opt[k][c];
                }
                w_opt[i][c] = sum / l_mat[i][i];
            }
        }

        // Proyección Estricta a la Ley de Dale ($W^{F,E} \ge 0, W^{F,I} \ge 0$)
        for j in 0..d {
            let wx = w_opt[j][0];
            let wy = w_opt[j][1];

            if wx >= 0.0 {
                self.wf_exc[j] = wx + 1e-4;
                self.wf_inh[j] = 1e-4;
            } else {
                self.wf_exc[j] = 1e-4;
                self.wf_inh[j] = -wx + 1e-4;
            }

            if wy >= 0.0 {
                self.wf_exc[d + j] = wy + 1e-4;
                self.wf_inh[d + j] = 1e-4;
            } else {
                self.wf_exc[d + j] = 1e-4;
                self.wf_inh[d + j] = -wy + 1e-4;
            }
        }

        // Registrar engrama adaptativo en memoria CDT
        let mut engram = Vec::with_capacity(self.config.d_latent);
        for j in 0..self.config.d_latent {
            let idx = j % d;
            let eff_x = self.wf_exc[idx] - self.wf_inh[idx];
            engram.push(eff_x);
        }
        self.cdt_memory.record_wake_episode(&engram, 1.0);
    }

    /// Paso de Inferencia Completo en Vigilia:
    /// $\hat{q}_p(t+1) = q_p(t) + \Delta \hat{q}_p$
    pub fn forward_step(
        &mut self,
        coords: &[f64],
        target_hint: Option<&[f64]>,
        dt_ms: f64,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let bary = self.basis_extractor.compute_barycenter(coords);
        let mut next_coords = coords.to_vec();
        let mut aggregated_latent = vec![0.0; self.config.d_latent];
        let mut total_handshake_action = 0.0;
        let d = self.config.d_basis;

        for p in 0..self.config.n_points {
            let phi_p = self.basis_extractor.extract_point_basis(coords, p, bary);

            // 1. Handshake Wheeler-Feynman en espacio fasorial $\mathbb{C}$
            let psi_ret = self.handshake_engine.build_retarded_wave(&phi_p, 0.0);
            let psi_adv = if let Some(target) = target_hint {
                self.handshake_engine.build_advanced_wave(target, 1.0)
            } else {
                let prior_out = self.thalamocortical_prior.forward(&phi_p);
                self.handshake_engine.build_advanced_wave(&prior_out, 1.0)
            };

            let (collapsed_phasors, action) = self.handshake_engine.compute_handshake_collapse(&psi_ret, &psi_adv, 4);
            total_handshake_action += action;

            // 2. Extraer modulación de fase para dendritas apicales L5
            let mut apical_signal = vec![0.0; self.config.d_latent];
            for j in 0..self.config.d_latent {
                let idx = j % collapsed_phasors.len();
                apical_signal[j] = collapsed_phasors[idx].norm() * collapsed_phasors[idx].phase().cos();
            }

            // 3. Procesamiento en unidad piramidal L2/3 y L5
            let soma_act = self.pyramidal_units[p].forward_step(
                &phi_p,
                &apical_signal,
                0.1, // PV+
                0.2, // SST+
                0.3, // VIP+ desinhibición
                dt_ms,
            );

            for j in 0..self.config.d_latent {
                aggregated_latent[j] += soma_act * (j as f64 * 0.1).sin();
            }

            // 4. Proyección de desplazamiento residual $\Delta q_p \in \mathbb{R}^2$
            // Integra: Prior Autónomo + Campo Lento W^P + Campo Rápido Adaptativo W^F
            let prior_delta = self.thalamocortical_prior.forward(&phi_p);
            let mut dx = prior_delta[0] * 0.1;
            let mut dy = prior_delta[1] * 0.1;

            // Campo lento W^P
            for j in 0..self.config.d_latent {
                let w_eff_x = self.w_proj_e[0][j] - self.w_proj_i[0][j];
                let w_eff_y = self.w_proj_e[1][j] - self.w_proj_i[1][j];
                dx += w_eff_x * apical_signal[j];
                dy += w_eff_y * apical_signal[j];
            }

            // Campo rápido adaptativo W^F
            for j in 0..d {
                let wf_eff_x = self.wf_exc[j] - self.wf_inh[j];
                let wf_eff_y = self.wf_exc[d + j] - self.wf_inh[d + j];
                dx += wf_eff_x * phi_p[j];
                dy += wf_eff_y * phi_p[j];
            }

            // 5. Adición Residual Puntual Equivariante v4
            next_coords[p * 2] = coords[p * 2] + dx;
            next_coords[p * 2 + 1] = coords[p * 2 + 1] + dy;

            // 6. Plasticidad rápida BTSP local
            self.pyramidal_units[p].apply_btsp(&phi_p, self.config.eta_btsp);
        }

        // Evolución espectral temporal de Koopman-Brenier
        let evolved_latent = self.spectral_operator.spectral_step(&aggregated_latent);
        let projected_latent = self.spectral_operator.apply_brenier_projection(&evolved_latent);

        (next_coords, projected_latent, total_handshake_action)
    }

    /// Rollout / Predicción recursiva a horizonte arbitrario $h \in \{1, 2, 4, 8, 16\}$
    pub fn predict_horizon(&mut self, initial_coords: &[f64], horizon: usize) -> Vec<Vec<f64>> {
        let mut sequence = Vec::with_capacity(horizon);
        let mut current = initial_coords.to_vec();

        for _ in 0..horizon {
            let (next, _, _) = self.forward_step(&current, None, 1.0);
            sequence.push(next.clone());
            current = next;
        }

        sequence
    }

    /// Fase de Sueño Circadiano (SWR) con Verificación del Teorema de Cuenca ($\Delta R \ge -\varepsilon$)
    pub fn sleep_and_consolidate(&mut self) -> (bool, f64) {
        self.circadian_phase = 1.0;
        let (pass_forgetting, delta_r) = self.cdt_memory.execute_swr_sleep_consolidation();
        self.circadian_phase = 0.0;
        (pass_forgetting, delta_r)
    }

    /// Auditoría estricta de violaciones a la Ley de Dale en todo el sistema
    pub fn audit_dales_law(&self) -> (usize, usize) {
        let mut total_violations = 0;
        let mut total_synapses = 0;

        for unit in &self.pyramidal_units {
            total_violations += unit.count_dale_violations();
            total_synapses += unit.basal_synapses.len() + unit.apical_synapses.len();
        }

        // Verificar matrices lentas W^P
        for coord in 0..2 {
            for j in 0..self.config.d_latent {
                total_synapses += 2;
                if self.w_proj_e[coord][j] < 0.0 {
                    total_violations += 1;
                }
                if self.w_proj_i[coord][j] < 0.0 {
                    total_violations += 1;
                }
            }
        }

        // Verificar campos rápidos W^F
        for (e, i) in self.wf_exc.iter().zip(self.wf_inh.iter()) {
            total_synapses += 2;
            if *e < 0.0 { total_violations += 1; }
            if *i < 0.0 { total_violations += 1; }
        }

        (total_violations, total_synapses)
    }

    /// Paso de Entrenamiento desde Cero
    pub fn train_step_from_scratch(&mut self, input_coords: &[f64], target_coords: &[f64]) -> f64 {
        let (predicted_coords, latent, _) = self.forward_step(input_coords, Some(target_coords), 1.0);
        let mut mse_loss = 0.0;

        for p in 0..self.config.n_points {
            let true_dx = target_coords[p * 2] - input_coords[p * 2];
            let true_dy = target_coords[p * 2 + 1] - input_coords[p * 2 + 1];

            let pred_dx = predicted_coords[p * 2] - input_coords[p * 2];
            let pred_dy = predicted_coords[p * 2 + 1] - input_coords[p * 2 + 1];

            let err_x = pred_dx - true_dx;
            let err_y = pred_dy - true_dy;

            mse_loss += err_x * err_x + err_y * err_y;

            for j in 0..self.config.d_latent {
                let mu_val = latent[j];

                if err_x > 0.0 {
                    self.w_proj_e[0][j] = (self.w_proj_e[0][j] - self.config.eta_slow * err_x * mu_val).max(0.0);
                    self.w_proj_i[0][j] = (self.w_proj_i[0][j] + self.config.eta_slow * err_x * mu_val).max(0.0);
                } else {
                    self.w_proj_e[0][j] = (self.w_proj_e[0][j] - self.config.eta_slow * err_x * mu_val).max(0.0);
                }

                if err_y > 0.0 {
                    self.w_proj_e[1][j] = (self.w_proj_e[1][j] - self.config.eta_slow * err_y * mu_val).max(0.0);
                    self.w_proj_i[1][j] = (self.w_proj_i[1][j] + self.config.eta_slow * err_y * mu_val).max(0.0);
                } else {
                    self.w_proj_e[1][j] = (self.w_proj_e[1][j] - self.config.eta_slow * err_y * mu_val).max(0.0);
                }
            }
        }

        mse_loss / (self.config.n_points as f64)
    }
}
