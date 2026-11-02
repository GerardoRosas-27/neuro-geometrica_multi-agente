//! Matriz Sináptica Biofísica: Ley de Dale, Plasticidad de Corto Plazo (Tsodyks-Markram),
//! Trazas de Elegibilidad Neo-Hebbianas de 3 Factores y Consolidación Estructural (Benna-Fusi / Frey-Morris).
//!
//! Fundamentación neurocientífica:
//! 1. **Tsodyks & Markram (1997)**, *PNAS*, 94(2), 719-723; **Mongillo, Barak & Tsodyks (2008)**,
//!    *Science*, 319(5869), 1543-1546 (*Synaptic Theory of Working Memory*):
//!    Cada terminal presináptica mantiene una fracción de vesículas neurotransmisoras disponibles
//!    $x(t) \in [0, 1]$ (depresión de corto plazo, $\tau_{\text{rec}}$) y una concentración de
//!    $\text{Ca}^{2+}$ residual que controla la probabilidad de liberación $u(t) \in [U_0, 1]$
//!    (facilitación de corto plazo, $\tau_{\text{facil}}$).
//! 2. **Ley de Dale (Eccles, 1954; Parisien, Anderson & Eliasmith, 2008, *Neural Computation*)**:
//!    Las neuronas biológicas no cambian el signo de sus neurotransmisores. Para lograr pesos
//!    efectivos de cualquier signo respetando estrictamente la Ley de Dale, cada proyección se
//!    descompone en una vía excitatoria glutamatérgica directa ($W^E \ge 0$) y una vía inhibitoria
//!    disináptica *feedforward* mediada por interneuronas locales ($W^I \ge 0$), de modo que
//!    el peso efectivo es $W_{\text{eff}} = W^E - W^I$ con $W^E \ge 0, W^I \ge 0$.
//! 3. **Gerstner et al. (2018)**, *Frontiers in Neural Circuits*, 12:53; **Bellec et al. (2020)**,
//!    *Nature Communications* (*e-prop*):
//!    Aprendizaje de 3 factores mediante trazas de elegibilidad exponenciales $e_{ij}(t)$ que
//!    registran la coincidencia entre actividad presináptica y potencial instructivo dendrítico
//!    postsináptico, convertidas en cambios de peso al llegar el tercer factor neuromodulador.
//! 4. **Frey & Morris (1997)**, *Nature*, 385, 533-536 (*Synaptic Tagging and Capture*);
//!    **Benna & Fusi (2016)**, *Nature Neuroscience*, 19, 1697-1706:
//!    Doble escala sináptica: peso rápido lábil (tráfico de receptores AMPA) acoplado a un
//!    andamiaje estructural consolidado ($W^{\text{cons}}$, remodelación de actina en espinas
//!    dendríticas) con rigidez metaplástica $\Omega_{ij}$ que previene el olvido catastrófico.

#![allow(clippy::needless_range_loop)]

use crate::rng::Xoshiro256StarStar;

/// Parámetros de la dinámica sináptica de corto y largo plazo.
#[derive(Clone, Debug)]
pub struct SynapseParams {
    /// Probabilidad basal de liberación de vesículas ($U_0 \in (0, 1)$).
    pub u0: f64,
    /// Tasa de recuperación de vesículas ($\Delta t / \tau_{\text{rec}}$).
    pub dt_over_tau_rec: f64,
    /// Tasa de decaimiento de facilitación por calcio ($\Delta t / \tau_{\text{facil}}$).
    pub dt_over_tau_facil: f64,
    /// Factor de ganancia de la modulación de corto plazo STP ($0 = \text{desactivado}$).
    pub stp_strength: f64,
    /// Factor de decaimiento temporal de la traza de elegibilidad ($\gamma_e \in [0, 1)$).
    pub eligibility_decay: f64,
    /// Regularización homeostática de Oja / decaimiento sináptico suave ($\lambda_{\text{Oja}}$).
    pub oja_decay: f64,
    /// Fuerza de anclaje elástico al andamiaje estructural consolidado ($\kappa_{\text{cons}}$).
    pub consolidation_pull: f64,
    /// Límite superior de magnitud de peso por sinapsis.
    pub w_max: f64,
}

impl Default for SynapseParams {
    fn default() -> Self {
        Self {
            u0: 0.30,
            dt_over_tau_rec: 0.15,
            dt_over_tau_facil: 0.08,
            stp_strength: 0.25,
            eligibility_decay: 0.50,
            oja_decay: 1e-4,
            consolidation_pull: 0.02,
            w_max: 4.0,
        }
    }
}

/// Matriz de proyección sináptica biofísica `n_out x n_in` que respeta la Ley de Dale.
#[derive(Clone, Debug)]
pub struct BioSynapseMatrix {
    pub n_out: usize,
    pub n_in: usize,
    /// Pesos excitatorios glutamatérgicos AMPA/NMDA ($W^E_{ij} \ge 0$).
    pub w_exc: Vec<f64>,
    /// Pesos inhibitorios GABAérgicos disinápticos feedforward ($W^I_{ij} \ge 0$).
    pub w_inh: Vec<f64>,
    /// Sesgo excitatorio/inhibitorio postsináptico por neurona ($b_i$).
    pub bias: Vec<f64>,
    /// Andamiaje estructural consolidado de largo plazo ($W^{\text{cons}}_{ij} = W^E - W^I$).
    pub w_consolidated: Vec<f64>,
    /// Sesgo estructural consolidado.
    pub bias_consolidated: Vec<f64>,
    /// Rigidez/madurez metaplástica de la espina dendrítica ($\Omega_{ij} \ge 0$).
    pub spine_maturity: Vec<f64>,
    /// Traza de elegibilidad de 3 factores por sinapsis ($e_{ij}$).
    pub eligibility: Vec<f64>,
    /// Traza de elegibilidad del sesgo ($e_{b, i}$).
    pub bias_eligibility: Vec<f64>,
    /// Acumulador de actividad sináptica reciente para etiquetado sináptico (*synaptic tagging*).
    pub synaptic_tag_acc: Vec<f64>,
    /// Probabilidad de liberación presináptica facilitada por $\text{Ca}^{2+}$ ($u_j$).
    pub stp_u: Vec<f64>,
    /// Fracción de vesículas neurotransmisoras disponibles ($x_j$).
    pub stp_x: Vec<f64>,
}

impl BioSynapseMatrix {
    /// Crea una proyección sináptica inicializada respetando la Ley de Dale ($W^E \ge 0, W^I \ge 0$).
    pub fn new(n_out: usize, n_in: usize, scale: f64, u0: f64, seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let std = scale / (n_in.max(1) as f64).sqrt();
        let total = n_out * n_in;
        let mut w_exc = vec![0.0; total];
        let mut w_inh = vec![0.0; total];
        let mut w_consolidated = vec![0.0; total];
        let baseline_co_tone = 0.05 * std;

        for idx in 0..total {
            let w = rng.randn() * std;
            if w >= 0.0 {
                w_exc[idx] = w + baseline_co_tone;
                w_inh[idx] = baseline_co_tone;
            } else {
                w_exc[idx] = baseline_co_tone;
                w_inh[idx] = -w + baseline_co_tone;
            }
            w_consolidated[idx] = w;
        }

        Self {
            n_out,
            n_in,
            w_exc,
            w_inh,
            bias: vec![0.0; n_out],
            w_consolidated,
            bias_consolidated: vec![0.0; n_out],
            spine_maturity: vec![0.0; total],
            eligibility: vec![0.0; total],
            bias_eligibility: vec![0.0; n_out],
            synaptic_tag_acc: vec![0.0; total],
            stp_u: vec![u0; n_in],
            stp_x: vec![1.0; n_in],
        }
    }

    /// Peso efectivo neto $W_{\text{eff}, ij} = W^E_{ij} - W^I_{ij}$ (con $W^E \ge 0, W^I \ge 0$).
    #[inline]
    pub fn effective_weight(&self, i: usize, j: usize) -> f64 {
        let idx = i * self.n_in + j;
        self.w_exc[idx] - self.w_inh[idx]
    }

    /// Reinicia el estado transitorio de corto plazo (vesículas STP y trazas de elegibilidad).
    pub fn reset_short_term_state(&mut self, u0: f64) {
        self.stp_u.fill(u0);
        self.stp_x.fill(1.0);
        self.eligibility.fill(0.0);
        self.bias_eligibility.fill(0.0);
    }

    /// Actualiza la dinámica presináptica de corto plazo (Tsodyks-Markram, 1997) y calcula la
    /// corriente postsináptica transmitida a cada neurona de salida.
    pub fn forward(&mut self, pre_rates: &[f64], params: &SynapseParams) -> Vec<f64> {
        debug_assert_eq!(pre_rates.len(), self.n_in);

        // Paso Tsodyks-Markram por canal presináptico
        let mut effective_pre = vec![0.0; self.n_in];
        for j in 0..self.n_in {
            let act = pre_rates[j].abs().clamp(0.0, 1.0);
            let u = self.stp_u[j];
            let x = self.stp_x[j];

            // Facilitación por entrada de Ca2+ y recuperación exponencial hacia U0
            let u_next =
                u + params.dt_over_tau_facil * (params.u0 - u) + params.u0 * (1.0 - u) * act;
            // Depresión por consumo de vesículas y recuperación con constante tau_rec
            let x_next = x + params.dt_over_tau_rec * (1.0 - x) - u_next * x * act;

            self.stp_u[j] = u_next.clamp(0.05, 0.99);
            self.stp_x[j] = x_next.clamp(0.05, 1.0);

            // Ganancia de liberación normalizada respecto al reposo (u0 * 1.0)
            let stp_gain = if params.u0 > 1e-6 {
                (self.stp_u[j] * self.stp_x[j]) / params.u0
            } else {
                1.0
            };
            let mod_gain = 1.0 + params.stp_strength * (stp_gain - 1.0);
            effective_pre[j] = pre_rates[j] * mod_gain;
        }

        let mut out = vec![0.0; self.n_out];
        for i in 0..self.n_out {
            let row = i * self.n_in;
            let mut sum = self.bias[i];
            for j in 0..self.n_in {
                let w_eff = self.w_exc[row + j] - self.w_inh[row + j];
                sum += w_eff * effective_pre[j];
            }
            out[i] = sum;
        }
        out
    }

    /// Calcula el flujo sináptico sin alterar el estado de vesículas (para predicciones puras).
    pub fn forward_stateless(&self, pre_rates: &[f64], params: &SynapseParams) -> Vec<f64> {
        let mut out = vec![0.0; self.n_out];
        for i in 0..self.n_out {
            let row = i * self.n_in;
            let mut sum = self.bias[i];
            for j in 0..self.n_in {
                let stp_gain = if params.u0 > 1e-6 {
                    (self.stp_u[j] * self.stp_x[j]) / params.u0
                } else {
                    1.0
                };
                let mod_gain = 1.0 + params.stp_strength * (stp_gain - 1.0);
                let w_eff = self.w_exc[row + j] - self.w_inh[row + j];
                sum += w_eff * pre_rates[j] * mod_gain;
            }
            out[i] = sum;
        }
        out
    }

    /// Proyección transpuesta ($W_{\text{eff}}^\top \cdot \delta$) para propagación dendrítica local
    /// o estimación de error predictivo entre capas adyacentes.
    pub fn backward_project(&self, post_signals: &[f64]) -> Vec<f64> {
        debug_assert_eq!(post_signals.len(), self.n_out);
        let mut back = vec![0.0; self.n_in];
        for i in 0..self.n_out {
            let sig = post_signals[i];
            if sig.abs() < 1e-12 {
                continue;
            }
            let row = i * self.n_in;
            for j in 0..self.n_in {
                let w_eff = self.w_exc[row + j] - self.w_inh[row + j];
                back[j] += w_eff * sig;
            }
        }
        back
    }

    /// Regla de aprendizaje Neo-Hebbiana de 3 factores (Gerstner et al., 2018) con
    /// respetabilidad estricta de la Ley de Dale ($W^E \ge 0, W^I \ge 0$) y protección
    /// metaplástica de espinas consolidadas (Benna & Fusi, 2016; Zenke et al., 2015).
    ///
    /// * `pre_rates`: actividad presináptica $r_{\text{pre}, j}$.
    /// * `post_teaching`: señal instructiva dendrítica postsináptica $\delta_{\text{dend}, i}$.
    /// * `post_rates`: tasa postsináptica $r_{\text{post}, i}$ (para regularización de Oja).
    /// * `third_factor_mod`: factor neuromodulador global/local $M(t)$ (Dopamina + Acetilcolina).
    /// * `lr`: tasa de aprendizaje base $\eta$.
    pub fn update_three_factor(
        &mut self,
        pre_rates: &[f64],
        post_teaching: &[f64],
        post_rates: &[f64],
        third_factor_mod: f64,
        lr: f64,
        params: &SynapseParams,
    ) {
        let eff_lr = lr * third_factor_mod;
        if eff_lr.abs() < 1e-12 {
            return;
        }

        for i in 0..self.n_out {
            let row = i * self.n_in;
            let teach_i = post_teaching[i].clamp(-3.0, 3.0);
            let r_post_sq = post_rates[i] * post_rates[i];

            // Actualizar traza de elegibilidad del sesgo
            self.bias_eligibility[i] =
                params.eligibility_decay * self.bias_eligibility[i] + teach_i;
            let bias_pull = params.consolidation_pull * (self.bias_consolidated[i] - self.bias[i]);
            self.bias[i] = (self.bias[i] + eff_lr * self.bias_eligibility[i] + lr * bias_pull)
                .clamp(-params.w_max, params.w_max);

            for j in 0..self.n_in {
                let idx = row + j;
                let pre_j = pre_rates[j];

                // 1. Traza de elegibilidad Hebbiana / Dendrítica (Pre * Señal Dendrítica Post)
                let hebb = teach_i * pre_j;
                self.eligibility[idx] = params.eligibility_decay * self.eligibility[idx] + hebb;

                // Acumular etiqueta sináptica (Frey & Morris, 1997) para futura consolidación
                self.synaptic_tag_acc[idx] += hebb.abs();

                // 2. Factor de plasticidad inversamente proporcional a la madurez estructural
                let maturity = self.spine_maturity[idx];
                let plasticity_gate = 1.0 / (1.0 + maturity);

                // 3. Peso efectivo actual y fuerza restauradora hacia el andamiaje consolidado
                let w_eff = self.w_exc[idx] - self.w_inh[idx];
                let cons_force =
                    params.consolidation_pull * maturity * (self.w_consolidated[idx] - w_eff);
                let oja_term = -params.oja_decay * r_post_sq * w_eff;

                let dw =
                    eff_lr * plasticity_gate * self.eligibility[idx] + lr * (cons_force + oja_term);

                // 4. Aplicar cambio respetando la Ley de Dale (W^E >= 0, W^I >= 0)
                if dw >= 0.0 {
                    self.w_exc[idx] += 0.5 * dw;
                    self.w_inh[idx] = (self.w_inh[idx] - 0.5 * dw).max(0.001);
                } else {
                    let abs_dw = -dw;
                    self.w_inh[idx] += 0.5 * abs_dw;
                    self.w_exc[idx] = (self.w_exc[idx] - 0.5 * abs_dw).max(0.001);
                }

                // Renormalización suave de co-contracción E/I para evitar deriva de ambos pesos
                let min_co = self.w_exc[idx].min(self.w_inh[idx]);
                if min_co > 0.02 {
                    let excess = min_co - 0.02;
                    self.w_exc[idx] -= excess;
                    self.w_inh[idx] -= excess;
                }

                // Acotar magnitud máxima
                let new_eff = self.w_exc[idx] - self.w_inh[idx];
                if new_eff > params.w_max {
                    self.w_exc[idx] = params.w_max + 0.02;
                    self.w_inh[idx] = 0.02;
                } else if new_eff < -params.w_max {
                    self.w_exc[idx] = 0.02;
                    self.w_inh[idx] = params.w_max + 0.02;
                }
            }
        }
    }

    /// Consolidación sináptica durante el sueño (Frey & Morris, 1997 *Synaptic Tagging & Capture*
    /// y Tononi & Cirelli, 2014 *Synaptic Homeostasis Hypothesis - SHY*):
    ///
    /// Transfiere los pesos rápidos de las sinapsis etiquetadas al andamiaje estructural
    /// de largo plazo e incrementa su madurez metaplástica $\Omega_{ij}$.
    pub fn consolidate_synapses(&mut self, consolidation_rate: f64, shy_downscale: f64) {
        let max_tag = self
            .synaptic_tag_acc
            .iter()
            .copied()
            .fold(1e-8_f64, f64::max);

        for idx in 0..self.w_exc.len() {
            let w_eff = self.w_exc[idx] - self.w_inh[idx];
            let norm_tag = (self.synaptic_tag_acc[idx] / max_tag).clamp(0.0, 1.0);

            // Sinapsis activamente involucradas consolidan su estado al andamiaje estructural
            let alpha = (consolidation_rate * (0.25 + 0.75 * norm_tag)).clamp(0.0, 1.0);
            self.w_consolidated[idx] += alpha * (w_eff - self.w_consolidated[idx]);

            // Incrementar rigidez de espina dendrítica en sinapsis relevantes
            self.spine_maturity[idx] = (self.spine_maturity[idx] + 1.5 * norm_tag).clamp(0.0, 12.0);

            // Hipótesis de Homeostasis Sináptica del Sueño (SHY, Tononi & Cirelli, 2014):
            // Las sinapsis débiles o no etiquetadas sufren un reescalado suave descendente
            if norm_tag < 0.15 && shy_downscale > 0.0 {
                let scale = 1.0 - shy_downscale * (1.0 - norm_tag);
                self.w_exc[idx] = (self.w_exc[idx] * scale).max(0.001);
                self.w_inh[idx] = (self.w_inh[idx] * scale).max(0.001);
            }

            self.synaptic_tag_acc[idx] = 0.0;
        }

        for i in 0..self.n_out {
            self.bias_consolidated[i] +=
                consolidation_rate * (self.bias[i] - self.bias_consolidated[i]);
        }
    }

    /// Número de parámetros efectivos en la matriz.
    pub fn n_params(&self) -> usize {
        self.n_out * self.n_in + self.n_out
    }
}
