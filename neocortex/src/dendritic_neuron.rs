//! Neurona Piramidal Multicompartimental con Computación Dendrítica Activa.
//!
//! Fundamentación neurocientífica:
//! 1. **Larkum (2013)**, *Trends in Neurosciences*, 36(3), 141-151:
//!    Acoplamiento BAC (*Backpropagating Action Potential Activated Calcium spike firing*):
//!    la coincidencia entre entrada ascendente en dendritas basales y entrada contextual/predictiva
//!    en el penacho apical distal desencadena un potencial de meseta de $\text{Ca}^{2+}$ que
//!    convierte espigas aisladas en ráfagas de alta frecuencia (*bursts*).
//! 2. **Sacramento, Ponte Costa, Bengio & Senn (2018)**, *NeurIPS 31*:
//!    Ecuaciones diferenciales de tres compartimentos electrotonicamente acoplados:
//!    - Compartimento Basal ($v_B$): recibe flujo *bottom-up* (sensorial / capa inferior).
//!    - Compartimento Apical Distal ($v_A$): recibe predicción *top-down* menos la cancelación
//!      inhibitoria de interneuronas SST (células de Martinotti), aislando el error local.
//!    - Compartimento Somático ($u_S$): integra conductancias de fuga, basal, apical e
//!      inhibición perisomática de interneuronas PV+.
//! 3. **Payeur, Guerguiev, Zenke, Richards & Naud (2021)**, *Nature Neuroscience*, 24, 1010-1019:
//!    Multiplexación de código neuronal:
//!    - **Tasa de eventos ($e_i$)**: impulsada por el potencial basal/somático (transmite la
//!      inferencia hacia adelante).
//!    - **Probabilidad de ráfaga ($p_i$)**: controlada por el potencial de meseta de $\text{Ca}^{2+}$
//!      en la dendrita apical (codifica la señal instructiva / gradiente local).
//!    - **Tasa de ráfagas ($b_i = e_i \cdot p_i$)**: diferencia entre tasa de ráfaga actual y
//!      esperada ($\Delta b_i = e_i (p_i - \bar{p}_i)$) guía la plasticidad sináptica sin
//!      necesidad de retropropagación global ni transporte simétrico de pesos.
//! 4. **Bienenstock, Cooper & Munro (1982)** & **Turrigiano & Nelson (2004)**:
//!    Umbral deslizante de metaplasticidad $\theta_M(t)$ y excitabilidad intrínseca homeostática.

/// Parámetros biofísicos de la neurona piramidal multicompartimental.
#[derive(Clone, Debug)]
pub struct PyramidalParams {
    /// Conductancia de fuga somática ($g_L$).
    pub g_leak: f64,
    /// Conductancia de acoplamiento dendrita basal -> soma ($g_B$).
    pub g_basal: f64,
    /// Conductancia de acoplamiento dendrita apical -> soma ($g_A$).
    pub g_apical: f64,
    /// Ganancia de amplificación no-lineal BAC (Larkum, 2013) cuando coinciden basal y apical.
    pub bac_coupling_gain: f64,
    /// Umbral de activación de espiga dendrítica de calcio ($\Theta_{\text{Ca}}$).
    pub ca_plateau_threshold: f64,
    /// Pendiente de la sigmoide del potencial de meseta de $\text{Ca}^{2+}$.
    pub ca_plateau_slope: f64,
    /// Probabilidad basal de ráfaga en equilibrio ($\bar{p}_0 \approx 0.5$ en Payeur et al., 2021).
    pub baseline_burst_prob: f64,
    /// Constante de tiempo de membrana somática ($\Delta t / \tau_m$).
    pub dt_over_tau_m: f64,
    /// Constante de tiempo del filtro de media móvil de probabilidad de ráfaga ($\Delta t / \tau_p$).
    pub dt_over_tau_burst: f64,
    /// Tasa de disparo objetivo homeostática ($r_{\text{target}}$, Turrigiano & Nelson, 2004).
    pub target_firing_rate: f64,
    /// Velocidad de adaptación intrínseca homeostática ($\eta_{\text{homeo}}$).
    pub homeostatic_lr: f64,
    /// Constante de tiempo del umbral deslizante BCM ($\Delta t / \tau_{\text{BCM}}$).
    pub bcm_tau_inv: f64,
}

impl Default for PyramidalParams {
    fn default() -> Self {
        Self {
            g_leak: 0.15,
            g_basal: 1.0,
            g_apical: 0.45,
            bac_coupling_gain: 0.35,
            ca_plateau_threshold: 0.0,
            ca_plateau_slope: 2.0,
            baseline_burst_prob: 0.5,
            dt_over_tau_m: 0.5,
            dt_over_tau_burst: 0.15,
            target_firing_rate: 0.25,
            homeostatic_lr: 0.005,
            bcm_tau_inv: 0.05,
        }
    }
}

/// Estado dinámico de una neurona piramidal de tres compartimentos.
#[derive(Clone, Debug)]
pub struct PyramidalNeuron {
    /// Potencial del compartimento dendrítico basal ($v_B$): entrada *bottom-up*.
    pub v_basal: f64,
    /// Potencial del compartimento dendrítico apical distal ($v_A$): entrada *top-down* neta.
    pub v_apical: f64,
    /// Potencial de membrana somático ($u_S$).
    pub u_soma: f64,
    /// Potencial de meseta de calcio activo en el penacho apical ($p_{\text{Ca}} \in [0, 1]$).
    pub ca_plateau: f64,
    /// Tasa de eventos somáticos ($e_i \in [-1, 1]$ o $[0, 1]$): señal transmitida hacia adelante.
    pub event_rate: f64,
    /// Probabilidad instantánea de ráfaga ($p_i \in (0, 1)$) modulada por el compartimento apical.
    pub burst_prob: f64,
    /// Media móvil exponencial de la probabilidad de ráfaga ($\bar{p}_i$).
    pub burst_prob_ema: f64,
    /// Tasa de ráfagas ($b_i = e_i \cdot p_i$, Payeur et al., 2021).
    pub burst_rate: f64,
    /// Señal de enseñanza dendrítica local ($\delta_{\text{dend}}$) que sustituye al gradiente de backprop.
    pub dendritic_teaching_signal: f64,
    /// Umbral deslizante de metaplasticidad de Bienenstock-Cooper-Munro ($\theta_M$).
    pub bcm_threshold: f64,
    /// Sesgo de excitabilidad intrínseca regulado por homeostasis sináptica ($b_{\text{homeo}}$).
    pub intrinsic_excitability: f64,
    /// Media móvil de la actividad absoluta para regulación homeostática.
    pub avg_activity: f64,
}

impl Default for PyramidalNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl PyramidalNeuron {
    pub fn new() -> Self {
        Self {
            v_basal: 0.0,
            v_apical: 0.0,
            u_soma: 0.0,
            ca_plateau: 0.5,
            event_rate: 0.0,
            burst_prob: 0.5,
            burst_prob_ema: 0.5,
            burst_rate: 0.0,
            dendritic_teaching_signal: 0.0,
            bcm_threshold: 0.1,
            intrinsic_excitability: 0.0,
            avg_activity: 0.25,
        }
    }

    /// Reinicia los potenciales rápidos preservando la homeostasis y metaplasticidad de largo plazo.
    pub fn reset_fast_state(&mut self) {
        self.v_basal = 0.0;
        self.v_apical = 0.0;
        self.u_soma = 0.0;
        self.ca_plateau = 0.5;
        self.event_rate = 0.0;
        self.burst_prob = 0.5;
        self.burst_rate = 0.0;
        self.dendritic_teaching_signal = 0.0;
    }

    /// Actualiza la dinámica de los 3 compartimentos en un paso temporal biofísico.
    ///
    /// * `basal_drive`: corriente sináptica *bottom-up* recibida en dendritas basales.
    /// * `apical_top_down`: corriente *top-down* de capas superiores o tálamo de orden superior.
    /// * `sst_apical_inhibition`: inhibición dendrítica de células de Martinotti (SST+) que
    ///   cancela la predicción esperada (Sacramento et al., 2018).
    /// * `pv_somatic_inhibition`: inhibición perisomática rápida de células en cesto (PV+).
    /// * `vip_disinhibition_gate`: factor de desinhibición apical ($[0, 2]$) mediado por
    ///   interneuronas VIP+ y acetilcolina (ACh).
    pub fn step(
        &mut self,
        basal_drive: f64,
        apical_top_down: f64,
        sst_apical_inhibition: f64,
        pv_somatic_inhibition: f64,
        vip_disinhibition_gate: f64,
        params: &PyramidalParams,
    ) {
        // 1. Compartimento Basal: recibe entrada ascendente + excitabilidad intrínseca homeostática
        let target_v_basal = basal_drive + self.intrinsic_excitability;
        self.v_basal += params.dt_over_tau_m * (target_v_basal - self.v_basal);

        // 2. Compartimento Apical: diferencia entre feedback top-down y predicción inhibitoria SST,
        // modulada por la compuerta de desinhibición VIP (Pfeffer et al., 2013; Sacramento et al., 2018).
        let effective_sst = sst_apical_inhibition / (1.0 + 0.5 * vip_disinhibition_gate.max(0.0));
        let net_apical = (apical_top_down - effective_sst) * (1.0 + 0.3 * vip_disinhibition_gate);
        self.v_apical += params.dt_over_tau_m * (net_apical - self.v_apical);

        // 3. Potencial de meseta de Calcio (Ca2+ spike) en el penacho apical (Larkum, 2013):
        let ca_arg = params.ca_plateau_slope * (self.v_apical - params.ca_plateau_threshold);
        self.ca_plateau = sigmoid(ca_arg);

        // 4. Acoplamiento no-lineal BAC (Backpropagating Action Potential Activated Ca2+ spike):
        // Cuando la dendrita basal está activa y la dendrita apical se despolariza, el soma
        // experimenta una ganancia asociativa no-lineal.
        let bac_term =
            params.bac_coupling_gain * self.v_basal.tanh() * (2.0 * self.ca_plateau - 1.0);

        // 5. Integración en el compartimento Somático (Sacramento et al., 2018):
        let denom =
            params.g_leak + params.g_basal + params.g_apical + pv_somatic_inhibition.max(0.0);
        let u_eq = (params.g_basal * self.v_basal + params.g_apical * self.v_apical + bac_term
            - 0.5 * pv_somatic_inhibition)
            / denom.max(1e-6);
        self.u_soma += params.dt_over_tau_m * (u_eq - self.u_soma);

        // 6. Tasa de eventos (inferencia feedforward) vs Tasa de ráfagas (Payeur et al., 2021):
        // La tasa de eventos depende primordialmente del flujo basal/somático;
        // la derivada local phi'(v_basal) actúa como factor de sensibilidad postsináptica.
        let basal_rate = self.v_basal.tanh();
        let soma_rate = self.u_soma.tanh();
        self.event_rate = 0.7 * basal_rate + 0.3 * soma_rate;

        // Probabilidad de que un evento somático se convierta en ráfaga de alta frecuencia (burst)
        self.burst_prob = self.ca_plateau;
        self.burst_rate = self.event_rate * self.burst_prob;

        // 7. Señal instructiva dendrítica local (Payeur et al., 2021; Sacramento et al., 2018):
        // phi'(v_basal) = 1 - tanh^2(v_basal). El desajuste apical v_apical (o p_i - p_bar)
        // ponderado por la sensibilidad somática/basal aproxima localmente el gradiente sin backprop.
        let phi_prime = (1.0 - basal_rate * basal_rate).max(0.05);
        self.dendritic_teaching_signal = self.v_apical * phi_prime;

        // Actualizar media móvil de probabilidad de ráfaga
        self.burst_prob_ema += params.dt_over_tau_burst * (self.burst_prob - self.burst_prob_ema);
    }

    /// Actualiza la homeostasis intrínseca (Turrigiano & Nelson, 2004) y el umbral deslizante
    /// de metaplasticidad BCM (Bienenstock, Cooper & Munro, 1982).
    pub fn update_homeostasis(&mut self, params: &PyramidalParams) {
        let act_abs = self.event_rate.abs();
        self.avg_activity += params.bcm_tau_inv * (act_abs - self.avg_activity);

        // Regla BCM: el umbral de modificación sináptica sigue al cuadrado de la actividad media
        let r_sq = self.event_rate * self.event_rate;
        self.bcm_threshold += params.bcm_tau_inv * (r_sq - self.bcm_threshold);
        self.bcm_threshold = self.bcm_threshold.clamp(0.01, 0.95);

        // Regulación de excitabilidad intrínseca para mantener la tasa de disparo objetivo
        let rate_error = params.target_firing_rate - self.avg_activity;
        self.intrinsic_excitability =
            (self.intrinsic_excitability + params.homeostatic_lr * rate_error).clamp(-0.5, 0.5);
    }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}
