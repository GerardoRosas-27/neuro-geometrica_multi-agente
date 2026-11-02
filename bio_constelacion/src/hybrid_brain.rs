//! Arquitectura Cerebral Híbrida: Bio-Constelación
//!
//! Integra lo mejor de Constelación v4 y Neocortex (BioBrain):
//! 1. Dinámica residual puntual equivariante: $\hat{q}_p(t+1) = q_p(t) + \Delta \hat{q}_p$.
//! 2. Base receptiva cortical L2/3 $\phi(q_p)$ con 10 componentes no lineales.
//! 3. Campo Dendrítico Rápido BTSP ($W^F = W^{F,E} - W^{F,I}$, 100% Ley de Dale).
//! 4. Columnas piramidales de 3 compartimentos con bloques Koopman contractivos $K_j = \rho_j R(\omega_j)$.
//! 5. Microcircuito de interneuronas corticales: PV+ (divisiva somática), SST+ (cancelación apical),
//!    VIP+ (desinhibición por novedad/ACh).
//! 6. Prior Tálamo-Cortical autónomo ($W^{\text{prior}}$) que resuelve el horizonte $h=1$ en E33
//!    sin necesidad de pesos externos congelados de ÓRBITA.
//! 7. Memoria Episódica Hipocampal (DG-CA3-CA1) con reactivación y consolidación en Sueño SWR.

use crate::dale_stp_synapse::FastDendriticBtspField;
use crate::dendritic_koopman::{CompartmentalUnitState, DendriticKoopmanNeuron};
use crate::hippocampus_cdt_sleep::{CdtTransition, HybridHippocampusCdt};
use crate::rng::Xoshiro256StarStar;

/// Configuración para la arquitectura híbrida Bio-Constelación.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Número de puntos en el objeto 2D (constelación de 6 puntos = 12 floats)
    pub n_points: usize,
    /// Dimensión de la base receptiva por punto $\phi(q_p)$
    pub d_basis: usize,
    /// Dimensión del espacio latente Koopman por columna
    pub d_koopman: usize,
    /// Dimensión oculta de levantamiento no lineal
    pub d_lifting: usize,
    /// Tasa de plasticidad rápida BTSP ($\eta_{\text{fast}}$)
    pub eta_fast: f64,
    /// Tasa de aprendizaje lenta de consolidación ($\eta_{\text{slow}}$)
    pub eta_slow: f64,
    /// Ciclos de relajación dendrítica durante observación de adaptación
    pub n_btsp_cycles: usize,
    /// Dimensión del Giro Dentado (DG)
    pub dg_dim: usize,
    /// Capacidad máxima de engramas en CA3
    pub ca3_capacity: usize,
    /// Capacidad máxima del búfer CDT
    pub cdt_capacity: usize,
    /// Tasa de consolidación sináptica en sueño (Frey-Morris)
    pub consolidation_rate: f64,
    /// Banderas de ablación para verificación científica:
    pub enable_fast_btsp: bool,
    pub enable_hippocampus: bool,
    pub enable_sleep: bool,
    pub enable_vip_sst: bool,
    pub enable_context_prior: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            n_points: 6,
            d_basis: 10,
            d_koopman: 4,
            d_lifting: 8,
            eta_fast: 0.25,
            eta_slow: 0.005,
            n_btsp_cycles: 24,
            dg_dim: 128,
            ca3_capacity: 256,
            cdt_capacity: 1024,
            consolidation_rate: 0.05,
            enable_fast_btsp: true,
            enable_hippocampus: true,
            enable_sleep: true,
            enable_vip_sst: true,
            enable_context_prior: true,
        }
    }
}

/// Red Tálamo-Cortical de Prior Contextual.
/// Mapea el vector de contexto de la tarea $c \in \mathbb{R}^{10}$
/// a un operador de desplazamiento base $W^{\text{prior}} \in \mathbb{R}^{2 \times d_{\text{basis}}}$.
/// Cumple 100% con la Ley de Dale ($W^E \ge 0, W^I \ge 0$).
#[derive(Debug, Clone)]
pub struct ThalamoCorticalPrior {
    pub hidden_dim: usize,
    pub w1_e: Vec<f64>,
    pub w1_i: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2_e: Vec<f64>,
    pub w2_i: Vec<f64>,
    pub b2: Vec<f64>,
}

impl ThalamoCorticalPrior {
    pub fn new(hidden_dim: usize, d_basis: usize, rng: &mut Xoshiro256StarStar) -> Self {
        let n_in = 10;
        let n_out = 2 * d_basis;

        let mut w1_e = vec![0.0; hidden_dim * n_in];
        let mut w1_i = vec![0.0; hidden_dim * n_in];
        let b1 = vec![0.0; hidden_dim];

        let scale1 = (2.0 / n_in as f64).sqrt() * 0.2;
        for val in w1_e.iter_mut() {
            *val = (rng.randn().abs() * scale1).max(0.0);
        }
        for val in w1_i.iter_mut() {
            *val = (rng.randn().abs() * scale1).max(0.0);
        }

        let mut w2_e = vec![0.0; n_out * hidden_dim];
        let mut w2_i = vec![0.0; n_out * hidden_dim];
        let b2 = vec![0.0; n_out];

        let scale2 = (2.0 / hidden_dim as f64).sqrt() * 0.1;
        for val in w2_e.iter_mut() {
            *val = (rng.randn().abs() * scale2).max(0.0);
        }
        for val in w2_i.iter_mut() {
            *val = (rng.randn().abs() * scale2).max(0.0);
        }

        Self {
            hidden_dim,
            w1_e,
            w1_i,
            b1,
            w2_e,
            w2_i,
            b2,
        }
    }

    /// Genera la matriz de pesos prior $W^{\text{prior}} \in \mathbb{R}^{2 \times d_{\text{basis}}}$.
    pub fn forward(&self, context: &[f64; 10], d_basis: usize) -> Vec<f64> {
        let mut hidden = vec![0.0; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let mut sum = self.b1[h];
            for c in 0..10 {
                let w = self.w1_e[h * 10 + c] - self.w1_i[h * 10 + c];
                sum += w * context[c];
            }
            hidden[h] = sum.max(0.0); // ReLU
        }

        let n_out = 2 * d_basis;
        let mut out = vec![0.0; n_out];
        for o in 0..n_out {
            let mut sum = self.b2[o];
            for h in 0..self.hidden_dim {
                let w = self.w2_e[o * self.hidden_dim + h] - self.w2_i[o * self.hidden_dim + h];
                sum += w * hidden[h];
            }
            out[o] = sum;
        }

        out
    }

    /// Actualización Hebbiana de 3 factores guiada por Dopamina.
    pub fn update(&mut self, context: &[f64; 10], target_operator: &[f64], da: f64, lr: f64) {
        if da.abs() < 1e-4 {
            return;
        }

        let mut hidden = vec![0.0; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let mut sum = self.b1[h];
            for c in 0..10 {
                let w = self.w1_e[h * 10 + c] - self.w1_i[h * 10 + c];
                sum += w * context[c];
            }
            hidden[h] = sum.max(0.0);
        }

        let n_out = target_operator.len();
        let mut pred = vec![0.0; n_out];
        for o in 0..n_out {
            let mut sum = self.b2[o];
            for h in 0..self.hidden_dim {
                let w = self.w2_e[o * self.hidden_dim + h] - self.w2_i[o * self.hidden_dim + h];
                sum += w * hidden[h];
            }
            pred[o] = sum;
        }

        for o in 0..n_out {
            let err = target_operator[o] - pred[o];
            let delta = (lr * da * err).clamp(-0.05, 0.05);
            for h in 0..self.hidden_dim {
                let post = hidden[h];
                if delta > 0.0 {
                    self.w2_e[o * self.hidden_dim + h] += delta * post;
                } else {
                    self.w2_i[o * self.hidden_dim + h] -= delta * post;
                }
            }
            self.b2[o] += delta * 0.05;
        }

        for val in self.w2_e.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }
        for val in self.w2_i.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }
    }
}

/// Cerebro Híbrido: Bio-Constelación.
#[derive(Debug, Clone)]
pub struct BioConstelacion {
    /// Configuración arquitectónica
    pub config: HybridConfig,
    /// Neurona de Koopman dendrítica compartimental (Capa L5)
    pub koopman_col: DendriticKoopmanNeuron,
    /// Estado dinámico de los compartimentos
    pub comp_state: CompartmentalUnitState,
    /// Campo dendrítico rápido BTSP y sinapsis lenta de proyección ($W^F$ y $W^P$)
    pub fast_btsp: FastDendriticBtspField,
    /// Memoria hipocampal, neuromoduladores y consolidación SWR
    pub hippocampus: HybridHippocampusCdt,
    /// Prior contextual tálamo-cortical autónomo
    pub thalamocortical_prior: ThalamoCorticalPrior,
}

impl BioConstelacion {
    /// Construye e inicializa la arquitectura híbrida completa.
    pub fn new(config: HybridConfig, rng: &mut Xoshiro256StarStar) -> Self {
        let d_basis = config.d_basis;
        let d_koopman = config.d_koopman;
        let d_lifting = config.d_lifting;

        let koopman_col = DendriticKoopmanNeuron::new(rng, d_koopman, d_lifting, 0.05);

        let comp_state = CompartmentalUnitState::fresh(d_koopman, d_lifting);

        let fast_btsp = FastDendriticBtspField::new(d_basis);

        let hippocampus = HybridHippocampusCdt::new(
            config.dg_dim,
            config.ca3_capacity,
            config.cdt_capacity,
            config.consolidation_rate,
            rng,
        );

        let thalamocortical_prior = ThalamoCorticalPrior::new(32, d_basis, rng);

        Self {
            config,
            koopman_col,
            comp_state,
            fast_btsp,
            hippocampus,
            thalamocortical_prior,
        }
    }

    /// Calcula la base de características corticales L2/3 $\phi(q_p)$ para un punto 2D $(x_p, y_p)$.
    /// Dimensión = 10:
    /// [x, y, 1, r, tanh(2x), tanh(2y), sin(2x), sin(2y), mu_koop_x, mu_koop_y]
    pub fn compute_basis(&self, x: f64, y: f64) -> Vec<f64> {
        let r = (x * x + y * y).sqrt();
        let mu_x = self.comp_state.u_soma.first().copied().unwrap_or(0.0) * 0.1;
        let mu_y = self.comp_state.u_soma.get(1).copied().unwrap_or(0.0) * 0.1;

        vec![
            x,
            y,
            1.0,
            r,
            (2.0 * x).tanh(),
            (2.0 * y).tanh(),
            (2.0 * x).sin(),
            (2.0 * y).sin(),
            mu_x,
            mu_y,
        ]
    }

    /// Inicializa un nuevo episodio de predicción con el contexto de observación inicial.
    /// Recupera engramas episódicos de CA3 y activa el prior tálamo-cortical.
    pub fn begin_episode(&mut self, context: &[f64; 10]) {
        self.fast_btsp.reset_fast_field();
        self.comp_state.reset();

        // 1. Prior Tálamo-Cortical
        if self.config.enable_context_prior {
            let prior_op = self
                .thalamocortical_prior
                .forward(context, self.config.d_basis);
            self.fast_btsp.set_prior_operator(&prior_op, 0.45);
        }

        // 2. Consulta y Recuperación en Hipocampo (DG -> CA3)
        if self.config.enable_hippocampus {
            let dg_code = self.hippocampus.encode_dg(context, 16);
            if let Some((ca3_op, sim)) = self.hippocampus.recall_ca3(&dg_code, 0.40) {
                let blend = (sim * 0.65).clamp(0.20, 0.80);
                self.fast_btsp.set_prior_operator(&ca3_op, blend);
            }
        }
    }

    /// Predice el siguiente estado de los 6 puntos $q(t+1)$ de forma puntual equivariante.
    pub fn step_predict_frame(&mut self, current_q: &[f64]) -> Vec<f64> {
        let n_pts = self.config.n_points;
        let mut next_q = vec![0.0; n_pts * 2];

        let mut basal_drive = vec![0.0; self.config.d_koopman];
        let apical_drive = vec![0.0; self.config.d_koopman];

        for p in 0..n_pts {
            let pt_curr = [current_q[p * 2], current_q[p * 2 + 1]];
            let phi = self.compute_basis(pt_curr[0], pt_curr[1]);

            let pt_pred =
                self.fast_btsp
                    .predict_point(&pt_curr, &phi, self.config.enable_fast_btsp);

            next_q[p * 2] = pt_pred[0];
            next_q[p * 2 + 1] = pt_pred[1];

            // Acumular energía basal para la columna Koopman
            if self.config.d_koopman >= 2 {
                basal_drive[0] += (pt_pred[0] - pt_curr[0]) / (n_pts as f64);
                basal_drive[1] += (pt_pred[1] - pt_curr[1]) / (n_pts as f64);
            }
        }

        // Relajación de la columna dendrítica Koopman con microcircuito PV+/VIP+/SST+
        let pv_inh = 0.1;
        let vip_gate = self.hippocampus.neuromod.ach;
        let step_size = vec![0.05; self.config.d_koopman];

        self.koopman_col.step_compartments(
            &mut self.comp_state,
            &basal_drive,
            &apical_drive,
            pv_inh,
            vip_gate,
            &step_size,
            self.config.enable_vip_sst,
        );

        next_q
    }

    /// Realiza una tirada autoregresiva de horizonte `horizon` (ej. 16 pasos)
    /// a partir del estado inicial `q_init` de 12 floats.
    pub fn rollout(&mut self, q_init: &[f64], horizon: usize) -> Vec<Vec<f64>> {
        let mut traj = Vec::with_capacity(horizon);
        let mut current = q_init.to_vec();

        for _ in 0..horizon {
            let next = self.step_predict_frame(&current);
            traj.push(next.clone());
            current = next;
        }

        traj
    }

    /// Adapta el campo dendrítico rápido $W^F$ con las transiciones observadas
    /// en la fase de observación inicial (ej. $K_{\text{obs}} = 4$ fotogramas).
    pub fn adapt_to_observation_frames(
        &mut self,
        obs_frames: &[Vec<f64>],
        context: &[f64; 10],
        family_id: usize,
    ) {
        if obs_frames.len() < 2 {
            return;
        }

        let n_pts = self.config.n_points;
        let n_transitions = obs_frames.len() - 1;

        for t in 0..n_transitions {
            let q_curr = &obs_frames[t];
            let q_next = &obs_frames[t + 1];

            let mut q_prevs = Vec::with_capacity(n_pts);
            let mut phis = Vec::with_capacity(n_pts);
            let mut q_targets = Vec::with_capacity(n_pts);

            for p in 0..n_pts {
                let pt = [q_curr[p * 2], q_curr[p * 2 + 1]];
                let target = [q_next[p * 2], q_next[p * 2 + 1]];
                let phi = self.compute_basis(pt[0], pt[1]);

                q_prevs.push(pt);
                phis.push(phi);
                q_targets.push(target);
            }

            // Actualización rápida BTSP
            let vip_ach = self.hippocampus.neuromod.ach;
            self.fast_btsp.update_btsp_and_slow(
                &q_prevs,
                &phis,
                &q_targets,
                false, // slow weights train in sleep consolidation
                self.config.eta_slow,
                self.hippocampus.neuromod.da,
                self.config.enable_fast_btsp,
                self.config.eta_fast,
                vip_ach,
                self.config.n_btsp_cycles,
            );
        }

        // Evaluar error en la última transición para neuromoduladores y memoria episódica
        let q_last_prev = &obs_frames[n_transitions - 1];
        let q_last_tgt = &obs_frames[n_transitions];

        let mut sum_err = 0.0;
        for p in 0..n_pts {
            let pt_prev = [q_last_prev[p * 2], q_last_prev[p * 2 + 1]];
            let phi = self.compute_basis(pt_prev[0], pt_prev[1]);
            let pt_pred =
                self.fast_btsp
                    .predict_point(&pt_prev, &phi, self.config.enable_fast_btsp);

            let dx = q_last_tgt[p * 2] - pt_pred[0];
            let dy = q_last_tgt[p * 2 + 1] - pt_pred[1];
            sum_err += (dx * dx + dy * dy).sqrt();
        }

        let avg_err = sum_err / (n_pts as f64);
        let success = avg_err < 0.05;
        self.hippocampus
            .neuromod
            .update_signals(avg_err, avg_err * 2.0, success);

        // Guardar transición en búfer CDT
        let trans = CdtTransition {
            context: *context,
            q_t: obs_frames[0].clone(),
            q_next: obs_frames[1].clone(),
            family_id,
            rel_err: avg_err,
            surprise: avg_err * 1.5,
        };
        self.hippocampus.push_cdt(trans);

        // Almacenar engrama en CA3
        if self.config.enable_hippocampus {
            let dg_code = self.hippocampus.encode_dg(context, 16);
            let cur_op = self.fast_btsp.current_fast_operator();
            let salience = (1.0 / (1.0 + avg_err)).clamp(0.1, 2.0);
            self.hippocampus
                .store_ca3(*context, dg_code, cur_op.clone(), salience);

            if self.config.enable_context_prior {
                self.thalamocortical_prior.update(
                    context,
                    &cur_op,
                    self.hippocampus.neuromod.da,
                    0.02,
                );
            }
        }
    }

    /// Ciclo de Sueño SWR (Sharp-Wave Ripples):
    /// Reactiva transiciones CDT y consolida los pesos rápidos en los pesos lentos ($W^P$).
    pub fn sleep_swr_consolidation(
        &mut self,
        n_ripples: usize,
        batch_size: usize,
        rng: &mut Xoshiro256StarStar,
    ) {
        if !self.config.enable_sleep {
            return;
        }

        self.hippocampus.neuromod.enter_sleep_mode();

        for _ in 0..n_ripples {
            let replay_batch = self.hippocampus.sample_swr_batch(batch_size, rng);
            for item in replay_batch {
                let n_pts = self.config.n_points;
                let mut q_prevs = Vec::with_capacity(n_pts);
                let mut phis = Vec::with_capacity(n_pts);
                let mut q_targets = Vec::with_capacity(n_pts);

                for p in 0..n_pts {
                    let pt = [item.q_t[p * 2], item.q_t[p * 2 + 1]];
                    let target = [item.q_next[p * 2], item.q_next[p * 2 + 1]];
                    let phi = self.compute_basis(pt[0], pt[1]);

                    q_prevs.push(pt);
                    phis.push(phi);
                    q_targets.push(target);
                }

                // Consolidación en W^P mediante actualización lenta
                self.fast_btsp.update_btsp_and_slow(
                    &q_prevs,
                    &phis,
                    &q_targets,
                    true, // entrenar pesos lentos W^P
                    self.config.eta_slow,
                    1.0,   // factor dopaminérgico unificado en sueño
                    false, // sin BTSP rápido en repetición
                    0.0,
                    0.0,
                    0,
                );
            }

            self.fast_btsp
                .consolidate_sleep(self.config.consolidation_rate);
        }

        self.hippocampus.neuromod.enter_wake_mode();
    }

    /// Auditoría de la Ley de Dale en todos los conjuntos de pesos sinápticos:
    /// devuelve `(violaciones, sinapsis_totales)`.
    pub fn audit_dale_law(&self) -> (usize, usize) {
        self.fast_btsp.audit_dale_law()
    }
}
