//! Módulo de Microcircuitos Dendríticos Piramidales y Ley de Dale Estricta
//!
//! Implementa los principios biofísicos de `exp/hibrido-constelacion-neocortex` y `exp/neocortex-bioinspirado`:
//! 1. Neuronas piramidales multicompartimentales (Basal L2/3, Apical L5, Soma).
//! 2. 100% de cumplimiento con la Ley de Dale: $W_{\text{eff}} = W^E - W^I$ con $W^E \ge 0, W^I \ge 0$.
//! 3. Plasticidad Sináptica Conductual (BTSP) con mesetas de calcio apicales.
//! 4. Plasticidad de Corto Plazo (STP) modelo Tsodyks-Markram (variables $u$ y $x$).
//! 5. Microcircuito interneuronal PV+, SST+, VIP+ para modulación de ganancia y desinhibición.

use crate::rng::Xoshiro256StarStar;

/// Sinapsis biofísica con Ley de Dale estricta y dinámica Tsodyks-Markram (STP)
#[derive(Debug, Clone)]
pub struct DaleSynapse {
    /// Peso excitatorio no negativo ($W^E \ge 0$)
    pub weight_e: f64,
    /// Peso inhibitorio no negativo ($W^I \ge 0$)
    pub weight_i: f64,
    /// Fracción de utilización de neurotransmisor de calcio $u \in [0, 1]$
    pub u_ca: f64,
    /// Fracción de vesículas sinápticas disponibles $x \in [0, 1]$
    pub x_vesicles: f64,
    /// Parámetro de facilitación sináptica $U$
    pub u_baseline: f64,
    /// Constante de tiempo de recuperación de vesículas ($\tau_{\text{rec}}$ en ms)
    pub tau_rec: f64,
    /// Constante de tiempo de facilitación de calcio ($\tau_{\text{fac}}$ en ms)
    pub tau_fac: f64,
}

impl DaleSynapse {
    pub fn new(w_e: f64, w_i: f64, u_baseline: f64, tau_rec: f64, tau_fac: f64) -> Self {
        Self {
            weight_e: w_e.max(0.0),
            weight_i: w_i.max(0.0),
            u_ca: u_baseline,
            x_vesicles: 1.0,
            u_baseline,
            tau_rec,
            tau_fac,
        }
    }

    /// Retorna el peso efectivo algebraico respetando la Ley de Dale
    #[inline(always)]
    pub fn effective_weight(&self) -> f64 {
        self.weight_e - self.weight_i
    }

    /// Transmite una señal presináptica modulada por la dinámica Tsodyks-Markram
    pub fn transmit_spike(&mut self, pre_activity: f64, dt_ms: f64) -> f64 {
        // Recuperación hacia el estado basal
        self.x_vesicles += (1.0 - self.x_vesicles) * (dt_ms / self.tau_rec);
        self.u_ca += (self.u_baseline - self.u_ca) * (dt_ms / self.tau_fac);

        if pre_activity > 0.01 {
            // Aumento de utilización por flujo de calcio presináptico
            self.u_ca += self.u_baseline * (1.0 - self.u_ca) * pre_activity;
            self.u_ca = self.u_ca.clamp(0.01, 1.0);

            // Liberación fraccional de vesículas
            let release = self.u_ca * self.x_vesicles * pre_activity;
            self.x_vesicles = (self.x_vesicles - release).max(0.0);

            let w_eff = self.effective_weight();
            release * w_eff
        } else {
            0.0
        }
    }

    /// Actualización de plasticidad conductual BTSP (Bittner et al., 2017)
    /// Modula selectivamente $W^E$ y $W^I$ proyectando siempre a $\ge 0$.
    pub fn apply_btsp_update(&mut self, pre_act: f64, calcium_plateau: f64, eta: f64, decay: f64) {
        if calcium_plateau > 0.05 {
            // La meseta apical potencia la sinapsis excitatoria activa
            let delta = eta * calcium_plateau * pre_act;
            self.weight_e = (self.weight_e + delta - decay * self.weight_e).max(0.0);
        } else {
            // Homeostasis y LTD
            self.weight_e = (self.weight_e - decay * 0.1 * self.weight_e).max(0.0);
        }
    }
}

/// Estado funcional de los tres compartimentos de una neurona piramidal cortical
#[derive(Debug, Clone, Copy)]
pub struct PyramidalCompartments {
    /// Potencial dendrítico basal (feedforward sensorial L2/3)
    pub v_basal: f64,
    /// Potencial dendrítico apical (retroalimentación contextual / prior L5)
    pub v_apical: f64,
    /// Potencial somático integrado
    pub v_soma: f64,
    /// Probabilidad de disparo de meseta de calcio apical
    pub calcium_plateau_prob: f64,
}

/// Columna Piramidal Cortical de Tres Compartimentos con Microcircuito PV/SST/VIP
#[derive(Debug, Clone)]
pub struct CorticalPyramidalUnit {
    pub id: usize,
    /// Sinapsis dendríticas basales L2/3
    pub basal_synapses: Vec<DaleSynapse>,
    /// Sinapsis dendríticas apicales L5
    pub apical_synapses: Vec<DaleSynapse>,
    /// Acoplamiento BAC (Backpropagating Action Potential Activated Ca2+ Spike)
    pub bac_coupling_strength: f64,
    /// Estado actual de los compartimentos
    pub state: PyramidalCompartments,
}

impl CorticalPyramidalUnit {
    pub fn new(id: usize, n_basal_inputs: usize, n_apical_inputs: usize, rng: &mut Xoshiro256StarStar) -> Self {
        let mut basal_synapses = Vec::with_capacity(n_basal_inputs);
        for _ in 0..n_basal_inputs {
            let w_e = (rng.next_f64() * 0.4 + 0.1) / (n_basal_inputs as f64).sqrt();
            let w_i = (rng.next_f64() * 0.2 + 0.05) / (n_basal_inputs as f64).sqrt();
            basal_synapses.push(DaleSynapse::new(w_e, w_i, 0.2, 500.0, 200.0));
        }

        let mut apical_synapses = Vec::with_capacity(n_apical_inputs);
        for _ in 0..n_apical_inputs {
            let w_e = (rng.next_f64() * 0.3 + 0.1) / (n_apical_inputs as f64).sqrt();
            let w_i = (rng.next_f64() * 0.15 + 0.05) / (n_apical_inputs as f64).sqrt();
            apical_synapses.push(DaleSynapse::new(w_e, w_i, 0.15, 600.0, 300.0));
        }

        Self {
            id,
            basal_synapses,
            apical_synapses,
            bac_coupling_strength: 0.8,
            state: PyramidalCompartments {
                v_basal: 0.0,
                v_apical: 0.0,
                v_soma: 0.0,
                calcium_plateau_prob: 0.0,
            },
        }
    }

    /// Integra la actividad de los tres compartimentos bajo control interneuronal
    /// - `pv_inhibition`: Inhibición rápida perisomática al soma
    /// - `sst_inhibition`: Inhibición apical dendrítica
    /// - `vip_disinhibition`: Desinhibición que reduce la inhibición SST
    pub fn forward_step(
        &mut self,
        basal_in: &[f64],
        apical_in: &[f64],
        pv_inhibition: f64,
        sst_inhibition: f64,
        vip_disinhibition: f64,
        dt_ms: f64,
    ) -> f64 {
        // 1. Integración basal L2/3
        let mut i_basal = 0.0;
        for (i, syn) in self.basal_synapses.iter_mut().enumerate() {
            let x = if i < basal_in.len() { basal_in[i] } else { 0.0 };
            i_basal += syn.transmit_spike(x, dt_ms);
        }
        self.state.v_basal = (self.state.v_basal * 0.8 + i_basal * 0.2).clamp(-3.0, 3.0);

        // 2. Integración apical L5 modulada por interneuronas SST y VIP
        let effective_sst = (sst_inhibition - vip_disinhibition).max(0.0);
        let mut i_apical = 0.0;
        for (i, syn) in self.apical_synapses.iter_mut().enumerate() {
            let x = if i < apical_in.len() { apical_in[i] } else { 0.0 };
            i_apical += syn.transmit_spike(x, dt_ms);
        }
        // Gating apical por somatostatina
        let apical_gated = i_apical / (1.0 + effective_sst * 1.5);
        self.state.v_apical = (self.state.v_apical * 0.8 + apical_gated * 0.2).clamp(-3.0, 3.0);

        // 3. Meseta de Calcio Apical (Ca2+ Spike) si la despolarización apical supera el umbral
        let ca_thresh = 0.7;
        let ca_prob = 1.0 / (1.0 + (-(self.state.v_apical - ca_thresh) * 5.0).exp());
        self.state.calcium_plateau_prob = ca_prob;

        // 4. Integración somática con acoplamiento BAC
        // El soma se excita por la base y recibe un salto no lineal por la meseta apical
        let i_soma_raw = self.state.v_basal + self.bac_coupling_strength * self.state.v_apical * ca_prob;
        // Control de ganancia divisivo por interneuronas PV+
        let v_soma = i_soma_raw / (1.0 + pv_inhibition.max(0.0) * 1.2);
        self.state.v_soma = v_soma;

        // Disparo somático continuo acotado
        (v_soma * 1.5).tanh()
    }

    /// Aplica plasticidad rápida BTSP a todas las sinapsis basales
    pub fn apply_btsp(&mut self, basal_in: &[f64], eta: f64) {
        let ca = self.state.calcium_plateau_prob;
        for (i, syn) in self.basal_synapses.iter_mut().enumerate() {
            let x = if i < basal_in.len() { basal_in[i] } else { 0.0 };
            syn.apply_btsp_update(x, ca, eta, 0.001);
        }
    }

    /// Verifica que no exista ninguna violación a la Ley de Dale en sus sinapsis
    pub fn count_dale_violations(&self) -> usize {
        let mut violations = 0;
        for syn in &self.basal_synapses {
            if syn.weight_e < 0.0 || syn.weight_i < 0.0 {
                violations += 1;
            }
        }
        for syn in &self.apical_synapses {
            if syn.weight_e < 0.0 || syn.weight_i < 0.0 {
                violations += 1;
            }
        }
        violations
    }
}
