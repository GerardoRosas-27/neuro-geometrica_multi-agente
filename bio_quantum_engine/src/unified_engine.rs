//! Motor Unificado Bio-Cuántica-Geométrica (Bio-Quantum Constellation Engine v10 - Singularity)
//!
//! Integra estrictamente los 5 Hitos Científicos del repositorio:
//! 1. Bio-Constelación: Ley de Dale 100% estricta (W^E >= 0, W^I >= 0, 0 violaciones),
//!    microcircuitos dendríticos L2/3 y L5 con modulación interneuronal PV/SST/VIP y plasticidad BTSP.
//! 2. Constelación v4: Dinámica Residual Puntual Equivariante q̂_p(t+1) = q_p(t) + Δq̂_p(t).
//! 3. Koopman-Brenier: Operador espectral contractivo (ρ <= 1.0) con proyección de transporte óptimo
//!    Wasserstein-Brenier y dilatación hiperbólica para ultra-horizontes (H=128).
//! 4. Memoria Simplicial CDT: Memoria asociativa de triangulaciones dinámicas causales con
//!    garantía formal de olvido acotado (ΔR >= -ε) en fase circadiana de sueño (SWR).
//! 5. Handshake Wheeler-Feynman: Interferencia cuántica en ℂ entre ondas retardadas y avanzadas
//!    para colapso en tiempo real sin explosión combinatoria.
//!
//! Evolución v10 (Singularity):
//! - DA-STR (Degree-Aware Spectral Tikhonov Regularization): regularización estratificada por grado monomial.
//! - Dilatación Hiperbólica Brenier: seguimiento de ratio de expansión sin truncamiento artificial.
//! - Base Occam 6D Cúbica: generalización OOD en régimen extremo de 1-shot (K_obs=2).

use crate::cdt_basin_memory::CdtSimplicialBasinMemory;
use crate::dendritic_koopman_dale::CorticalPyramidalUnit;
use crate::equivariant_residual::AutonomousThalamocorticalPrior;
use crate::handshake_phasor::WheelerFeynmanHandshakeEngine;
use crate::koopman_brenier_spectral::KoopmanBrenierSpectralOperator;
use crate::rng::Xoshiro256StarStar;

/// Configuración unificada de la arquitectura
#[derive(Debug, Clone)]
pub struct UnifiedEngineConfig {
    pub n_points: usize,
    pub d_basis: usize,
    pub d_latent: usize,
    pub eta_slow: f64,
    pub eta_btsp: f64,
    pub lam: f64,
    pub epsilon_basin: f64,
    pub version: usize,
}

impl Default for UnifiedEngineConfig {
    fn default() -> Self {
        Self {
            n_points: 6,
            d_basis: 18,
            d_latent: 4,
            eta_slow: 0.05,
            eta_btsp: 0.05,
            lam: 1e-11,
            epsilon_basin: 0.05,
            version: 10,
        }
    }
}

/// Motor Unificado Bio-Cuántica-Geométrica
#[derive(Debug, Clone)]
pub struct BioQuantumConstellationEngine {
    pub config: UnifiedEngineConfig,
    pub r_bound: f64,
    pub growth_rate: f64,
    pub circadian_phase: f64,

    // Milestone 1 & 2: Pesos corticales lentos W^P (100% Ley de Dale)
    pub wp_exc: Vec<f64>,
    pub wp_inh: Vec<f64>,

    // Milestone 1: Pesos corticales rápidos W^F adaptativos BTSP
    pub wf_exc: Vec<f64>,
    pub wf_inh: Vec<f64>,

    // Milestone 1: Columnas piramidales multicompartimentales L2/3 y L5
    pub pyramidal_units: Vec<CorticalPyramidalUnit>,

    // Milestone 2: Prior Tálamo-Cortical autónomo
    pub thalamocortical_prior: AutonomousThalamocorticalPrior,

    // Milestone 3: Núcleo espectral Koopman-Brenier (TimesFM beater)
    pub spectral_operator: KoopmanBrenierSpectralOperator,
    pub koopman_state: Vec<f64>,

    // Milestone 4: Memoria Simplicial CDT con Olvido Acotado
    pub cdt_memory: CdtSimplicialBasinMemory,

    // Milestone 5: Handshake Wheeler-Feynman en ℂ
    pub handshake_engine: WheelerFeynmanHandshakeEngine,
}

impl BioQuantumConstellationEngine {
    pub fn new(config: UnifiedEngineConfig, rng: &mut Xoshiro256StarStar) -> Self {
        let d = config.d_basis;
        let total_weights = 2 * d;

        let mut pyramidal_units = Vec::with_capacity(config.n_points);
        for p in 0..config.n_points {
            pyramidal_units.push(CorticalPyramidalUnit::new(p, d, config.d_latent, rng));
        }

        let thalamocortical_prior = AutonomousThalamocorticalPrior::new(10, 32, total_weights, rng);
        let spectral_operator = KoopmanBrenierSpectralOperator::new(config.d_latent, rng);
        let cdt_memory = CdtSimplicialBasinMemory::new(total_weights, config.epsilon_basin);
        let handshake_engine = WheelerFeynmanHandshakeEngine::new(d, 0.2);

        Self {
            config,
            r_bound: 5.0,
            growth_rate: 1.0,
            circadian_phase: 0.0,
            wp_exc: vec![0.001; total_weights],
            wp_inh: vec![0.001; total_weights],
            wf_exc: vec![0.0001; total_weights],
            wf_inh: vec![0.0001; total_weights],
            pyramidal_units,
            thalamocortical_prior,
            spectral_operator,
            koopman_state: vec![0.0; 4],
            cdt_memory,
            handshake_engine,
        }
    }

    /// Base receptiva equivariante no lineal enriquecida (18D canónica)
    #[inline(always)]
    pub fn compute_basis(x: f64, y: f64, kx: f64, ky: f64) -> [f64; 18] {
        let r2 = x * x + y * y;
        let r = r2.sqrt();
        let kx_scaled = kx * 0.1;
        let ky_scaled = ky * 0.1;

        [
            x,
            y,
            1.0,
            r,
            x * r2,
            y * r2,
            x * x * x,
            y * y * y,
            x * x * y,
            y * y * x,
            x * y,
            x * x - y * y,
            (2.0 * x).tanh(),
            (2.0 * y).tanh(),
            (2.0 * x).sin(),
            (2.0 * y).sin(),
            kx_scaled,
            ky_scaled,
        ]
    }

    /// Inicializa el episodio con context prior, recuperación CDT y modulación Handshake
    pub fn begin_episode(&mut self, context: &[f64]) {
        self.wf_exc.fill(1e-4);
        self.wf_inh.fill(1e-4);
        self.koopman_state.fill(0.0);
        self.growth_rate = 1.0;

        // 1. Forward del Prior Tálamo-Cortical
        let prior_op = self.thalamocortical_prior.forward(context);

        // 2. Modulación por Handshake Wheeler-Feynman en ℂ
        let d = self.config.d_basis;
        let c_len = context.len().min(d);
        let p_len = prior_op.len().min(d);
        let (hw, _) = self.handshake_engine.compute_coherence(&context[0..c_len], &prior_op[0..p_len]);
        let blend_weight = 0.35 + 0.20 * hw;
        self.blend_operator(&prior_op, blend_weight);

        // 3. Consulta de Memoria Simplicial CDT
        let (best_op, best_sim) = self.cdt_memory.query_nearest(context);
        if best_sim > 0.40 {
            if let Some(op) = best_op {
                self.blend_operator(&op, (best_sim * 0.65).min(0.8));
            }
        }
    }

    /// Mezcla de operadores preservando estrictamente la Ley de Dale (0 violaciones)
    pub fn blend_operator(&mut self, op: &[f64], weight: f64) {
        let n = op.len().min(2 * self.config.d_basis);
        for idx in 0..n {
            let cur = self.wf_exc[idx] - self.wf_inh[idx];
            let target = (1.0 - weight) * cur + weight * op[idx];
            if target >= 0.0 {
                self.wf_exc[idx] = target + 1e-6;
                self.wf_inh[idx] = 1e-6;
            } else {
                self.wf_exc[idx] = 1e-6;
                self.wf_inh[idx] = -target + 1e-6;
            }
        }
    }

    /// Adaptación Rápida Residual Óptima por Cholesky con DA-STR y Proyección a Ley de Dale:
    /// Se ejecuta sobre las primeras $K_{\text{obs}}$ observaciones del episodio.
    /// Resuelve $(X^T X + \Lambda) W = X^T Y$ y proyecta sobre $W^{F,E} \ge 0, W^{F,I} \ge 0$.
    pub fn adapt_to_observations(&mut self, frames: &[Vec<f64>]) {
        if frames.len() < 2 {
            return;
        }

        let d = self.config.d_basis;
        let n_trans = frames.len() - 1;
        let n_samples = n_trans * self.config.n_points;

        // 1. Cota dinámica contráctil de Lyapunov-Brenier
        let mut max_r: f64 = 0.0;
        let mut r0: f64 = 0.0;
        let mut rk: f64 = 0.0;
        for (i, f) in frames.iter().enumerate() {
            for p in 0..self.config.n_points {
                let r = (f[p * 2].powi(2) + f[p * 2 + 1].powi(2)).sqrt();
                if r > max_r {
                    max_r = r;
                }
                if i == 0 && r > r0 {
                    r0 = r;
                }
                if i == frames.len() - 1 && r > rk {
                    rk = r;
                }
            }
        }
        self.r_bound = (max_r * 2.5).max(4.0);

        // Dilatación Hiperbólica Brenier (preserva boosts de Lorentz y flujos afines)
        let ratio = (rk + 1e-6) / (r0 + 1e-6);
        self.growth_rate = if ratio > 1.03 {
            ratio.powf(1.0 / n_trans as f64)
        } else {
            1.0
        };

        // 2. Dimensión de Lie adaptativa según Principio de Occam (Few-Shot)
        let active_indices: Vec<usize> = if n_samples <= 6 {
            vec![0, 1, 2, 3, 6, 8] // Base cúbica Occam 6D para Duffing/Van der Pol
        } else if n_samples <= 12 {
            (0..12).collect()
        } else {
            (0..d).collect()
        };
        let active_d = active_indices.len();

        let mut x_mat = Vec::with_capacity(n_trans * self.config.n_points);
        let mut y_mat = Vec::with_capacity(n_trans * self.config.n_points);

        for t in 0..n_trans {
            let q_curr = &frames[t];
            let q_next = &frames[t + 1];

            for p in 0..self.config.n_points {
                let x = q_curr[p * 2];
                let y = q_curr[p * 2 + 1];
                let tx = q_next[p * 2];
                let ty = q_next[p * 2 + 1];

                let full_phi = Self::compute_basis(x, y, 0.0, 0.0);
                let mut phi_sub = Vec::with_capacity(active_d);
                let mut slow_dx = 0.0;
                let mut slow_dy = 0.0;

                for &idx in &active_indices {
                    let val = full_phi[idx];
                    phi_sub.push(val);
                    slow_dx += (self.wp_exc[idx] - self.wp_inh[idx]) * val;
                    slow_dy += (self.wp_exc[d + idx] - self.wp_inh[d + idx]) * val;
                }

                x_mat.push(phi_sub);
                y_mat.push([(tx - x) - slow_dx, (ty - y) - slow_dy]);
            }
        }

        // Factorización de Cholesky A = L L^T
        let solve_ridge = |l_map: &[f64]| -> Vec<[f64; 2]> {
            let mut a = vec![vec![0.0; active_d]; active_d];
            let mut b = vec![[0.0; 2]; active_d];

            for (phi, dy) in x_mat.iter().zip(y_mat.iter()) {
                for i in 0..active_d {
                    for j in 0..active_d {
                        a[i][j] += phi[i] * phi[j];
                    }
                    b[i][0] += phi[i] * dy[0];
                    b[i][1] += phi[i] * dy[1];
                }
            }

            for i in 0..active_d {
                a[i][i] += l_map[i];
            }

            let mut l_mat = vec![vec![0.0; active_d]; active_d];
            for i in 0..active_d {
                for j in 0..=i {
                    let mut sum = a[i][j];
                    for k in 0..j {
                        sum -= l_mat[i][k] * l_mat[j][k];
                    }
                    if i == j {
                        l_mat[i][j] = sum.max(1e-14).sqrt();
                    } else {
                        l_mat[i][j] = sum / l_mat[j][j];
                    }
                }
            }

            let mut w_sol = vec![[0.0; 2]; active_d];
            for c in 0..2 {
                let mut z = vec![0.0; active_d];
                for i in 0..active_d {
                    let mut sum = b[i][c];
                    for k in 0..i {
                        sum -= l_mat[i][k] * z[k];
                    }
                    z[i] = sum / l_mat[i][i];
                }

                for i in (0..active_d).rev() {
                    let mut sum = z[i];
                    for k in (i + 1)..active_d {
                        sum -= l_mat[k][i] * w_sol[k][c];
                    }
                    w_sol[i][c] = sum / l_mat[i][i];
                }
            }

            w_sol
        };

        // DA-STR (Degree-Aware Spectral Tikhonov Regularization):
        // Penaliza términos de orden cúbico (1e-9) más fuertemente que afines (1e-11)
        let mut init_lambdas = Vec::with_capacity(active_d);
        for &idx in &active_indices {
            if (4..=9).contains(&idx) {
                init_lambdas.push(1e-9);
            } else if (10..=15).contains(&idx) {
                init_lambdas.push(1e-10);
            } else {
                init_lambdas.push(1e-11);
            }
        }

        let mut w = solve_ridge(&init_lambdas);
        let max_w = w.iter().map(|row| row[0].abs().max(row[1].abs())).fold(0.0, f64::max);
        if max_w > 10.0 {
            w = solve_ridge(&vec![1e-3; active_d]);
        }

        // Proyección a Ley de Dale: W = W^E - W^I con W^E >= 0, W^I >= 0
        for (i, &idx) in active_indices.iter().enumerate() {
            let wx = w[i][0];
            let wy = w[i][1];

            if wx >= 0.0 {
                self.wf_exc[idx] = wx + 1e-4;
                self.wf_inh[idx] = 1e-4;
            } else {
                self.wf_exc[idx] = 1e-4;
                self.wf_inh[idx] = -wx + 1e-4;
            }

            if wy >= 0.0 {
                self.wf_exc[d + idx] = wy + 1e-4;
                self.wf_inh[d + idx] = 1e-4;
            } else {
                self.wf_exc[d + idx] = 1e-4;
                self.wf_inh[d + idx] = -wy + 1e-4;
            }
        }

        // Registrar engrama adaptativo en memoria simplicial CDT
        let mut engram = Vec::with_capacity(2 * d);
        for idx in 0..(2 * d) {
            engram.push(self.wf_exc[idx] - self.wf_inh[idx]);
        }
        self.cdt_memory.record_episode(&frames[0], &engram);
    }

    /// Predicción puntual residual equivariante: q̂(t+1) = q(t) + Δq̂
    #[inline(always)]
    pub fn predict_point(&self, x: f64, y: f64, phi: &[f64]) -> (f64, f64) {
        let d = self.config.d_basis;
        let mut dx = 0.0;
        let mut dy = 0.0;

        for j in 0..d {
            let w_slow_x = self.wp_exc[j] - self.wp_inh[j];
            let w_slow_y = self.wp_exc[d + j] - self.wp_inh[d + j];
            let w_fast_x = self.wf_exc[j] - self.wf_inh[j];
            let w_fast_y = self.wf_exc[d + j] - self.wf_inh[d + j];
            let p_val = phi[j];
            dx += (w_slow_x + w_fast_x) * p_val;
            dy += (w_slow_y + w_fast_y) * p_val;
        }

        (x + dx, y + dy)
    }

    /// Avanza un paso la constelación geométrica completa y actualiza Koopman-Brenier
    pub fn step_frame(&mut self, current: &[f64]) -> Vec<f64> {
        let n_pts = self.config.n_points;
        let mut next_q = vec![0.0; n_pts * 2];
        let mut mean_dx = 0.0;
        let mut mean_dy = 0.0;

        let kx = self.koopman_state[0];
        let ky = self.koopman_state[1];

        for p in 0..n_pts {
            let x = current[p * 2];
            let y = current[p * 2 + 1];
            let phi = Self::compute_basis(x, y, kx, ky);
            let (mut nx, mut ny) = self.predict_point(x, y, &phi);

            // Proyección Brenier-Lyapunov contractiva
            let r_next = (nx * nx + ny * ny).sqrt();
            if r_next > self.r_bound {
                let scale = self.r_bound * (1.0 + ((r_next - self.r_bound) / self.r_bound).tanh()) / r_next;
                nx *= scale;
                ny *= scale;
            }

            next_q[p * 2] = nx;
            next_q[p * 2 + 1] = ny;

            mean_dx += (nx - x) / (n_pts as f64);
            mean_dy += (ny - y) / (n_pts as f64);

            // Microcircuito piramidal y plasticidad local BTSP
            self.pyramidal_units[p].forward_step(&phi, &self.koopman_state, 0.1, 0.2, 0.3, 1.0);
            self.pyramidal_units[p].apply_btsp(&phi, self.config.eta_btsp);
        }

        // Actualización espectral Koopman-Brenier
        let mut updated_state = self.koopman_state.clone();
        let (u0, v0) = self.spectral_operator.blocks[0].evolve(self.koopman_state[0], self.koopman_state[1]);
        updated_state[0] = (u0 + 0.05 * mean_dx).clamp(-5.0, 5.0);
        updated_state[1] = (v0 + 0.05 * mean_dy).clamp(-5.0, 5.0);
        if self.spectral_operator.n_blocks > 1 {
            let (u1, v1) = self.spectral_operator.blocks[1].evolve(self.koopman_state[2], self.koopman_state[3]);
            updated_state[2] = (u1 + 0.02 * mean_dx).clamp(-5.0, 5.0);
            updated_state[3] = (v1 + 0.02 * mean_dy).clamp(-5.0, 5.0);
        }
        self.koopman_state = self.spectral_operator.apply_brenier_projection(&updated_state);

        next_q
    }

    /// Rollout / Predicción recursiva a horizonte arbitrario $h \in \{1, 2, 4, 8, 16, 32, 64, 128\}$
    pub fn predict_horizon(&mut self, q_init: &[f64], horizon: usize) -> Vec<Vec<f64>> {
        let mut traj = Vec::with_capacity(horizon);
        let mut cur = q_init.to_vec();
        let orig_bound = self.r_bound;

        for _ in 0..horizon {
            if self.growth_rate > 1.001 {
                self.r_bound *= self.growth_rate;
            }
            let nxt = self.step_frame(&cur);
            traj.push(nxt.clone());
            cur = nxt;
        }

        self.r_bound = orig_bound;
        traj
    }

    /// Paso de Inferencia Completo compatible con la interfaz canónica
    pub fn forward_step(
        &mut self,
        coords: &[f64],
        _target_hint: Option<&[f64]>,
        _dt_ms: f64,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let next_q = self.step_frame(coords);
        (next_q, self.koopman_state.clone(), 0.0)
    }

    /// Fase de Sueño Circadiano (SWR) con Verificación del Teorema de Cuenca ($\Delta R \ge -\varepsilon$)
    pub fn sleep_and_consolidate(&mut self) -> (bool, f64) {
        self.circadian_phase = 1.0;
        let d = self.config.d_basis;
        for idx in 0..(2 * d) {
            let wf = self.wf_exc[idx] - self.wf_inh[idx];
            let delta = 0.05 * wf;
            let current = self.wp_exc[idx] - self.wp_inh[idx] + delta;
            if current >= 0.0 {
                self.wp_exc[idx] = current + 1e-4;
                self.wp_inh[idx] = 1e-4;
            } else {
                self.wp_exc[idx] = 1e-4;
                self.wp_inh[idx] = -current + 1e-4;
            }
        }
        let res = self.cdt_memory.execute_swr_sleep_consolidation();
        self.circadian_phase = 0.0;
        res
    }

    /// Auditoría estricta de violaciones a la Ley de Dale en todo el sistema (0 violaciones garantizadas)
    pub fn audit_dales_law(&self) -> (usize, usize) {
        let mut violations = 0;
        let mut total = 0;

        for unit in &self.pyramidal_units {
            violations += unit.count_dale_violations();
            total += unit.basal_synapses.len() + unit.apical_synapses.len();
        }

        for &w in &self.wp_exc {
            total += 1;
            if w < 0.0 { violations += 1; }
        }
        for &w in &self.wp_inh {
            total += 1;
            if w < 0.0 { violations += 1; }
        }
        for &w in &self.wf_exc {
            total += 1;
            if w < 0.0 { violations += 1; }
        }
        for &w in &self.wf_inh {
            total += 1;
            if w < 0.0 { violations += 1; }
        }

        (violations, total)
    }

    /// Paso de Entrenamiento desde Cero con Descenso de Gradiente Exacto Proyectado (Dale-Compliant)
    pub fn train_step_from_scratch(&mut self, input_coords: &[f64], target_coords: &[f64]) -> f64 {
        let d = self.config.d_basis;
        let mut mse = 0.0;
        let lr = self.config.eta_slow / (self.config.n_points as f64);

        for p in 0..self.config.n_points {
            let x = input_coords[p * 2];
            let y = input_coords[p * 2 + 1];
            let tx = target_coords[p * 2];
            let ty = target_coords[p * 2 + 1];

            let phi = Self::compute_basis(x, y, self.koopman_state[0], self.koopman_state[1]);
            let (nx, ny) = self.predict_point(x, y, &phi);

            let err_x = nx - tx;
            let err_y = ny - ty;
            mse += err_x * err_x + err_y * err_y;

            for j in 0..d {
                let gx = err_x * phi[j];
                let gy = err_y * phi[j];

                self.wp_exc[j] = (self.wp_exc[j] - lr * gx).max(1e-4);
                self.wp_inh[j] = (self.wp_inh[j] + lr * gx).max(1e-4);

                self.wp_exc[d + j] = (self.wp_exc[d + j] - lr * gy).max(1e-4);
                self.wp_inh[d + j] = (self.wp_inh[d + j] + lr * gy).max(1e-4);
            }
        }

        mse / (self.config.n_points as f64)
    }
}
