//! Columna Cortical Laminada Canónica (Capas L4, L2/3, L5/6 + Microcircuito Inhibitorio PV/SST/VIP).
//!
//! Fundamentación neurocientífica:
//! 1. **Mountcastle (1997)**, *Brain*, 120(4), 701-722; **Hawkins, Ahmad & Cui (2017)**,
//!    *Frontiers in Neural Circuits*, 11:81:
//!    La columna neocortical es la unidad computacional repetitiva de la corteza cerebral humana.
//! 2. **Bastos et al. (2012)**, *Neuron*, 76(4), 695-711 (*Canonical Microcircuits for Predictive Coding*):
//!    - **Capa 4 (L4, Granular)**: Recibe aferencias ascendentes (*bottom-up*) talámicas o de L2/3 inferior.
//!    - **Capa 2/3 (L2/3, Supragranular)**: Neuronas piramidales superficiales con dendritas apicales en
//!      Capa 1; integran la entrada de L4 con predicciones *top-down* y propagan hacia áreas superiores.
//!    - **Capa 5/6 (L5/6, Infragranular)**: Neuronas piramidales profundas que mantienen el estado
//!      generativo temporal y emiten predicciones descendentes (*top-down*) hacia la columna inferior.
//! 3. **Pfeffer et al. (2013)**, *Nature Neuroscience*, 16, 1068-1076; **Tremblay, Lee & Rudy (2016)**,
//!    *Neuron*, 91(2), 260-292; **Sacramento et al. (2018)**, *NeurIPS 31*:
//!    Microcircuito GABAérgico de tres clases de interneuronas:
//!    - **PV+ (Parvalbúmina)**: Inhibición perisomática rápida, control de ganancia divisivo
//!      (Carandini & Heeger, 2012) y dispersión competitiva lateral (*soft k-WTA*).
//!    - **SST+ (Somatostatina / Martinotti)**: Inhibición dendrítica apical que aprende a cancelar
//!      la predicción esperada para aislar el error instructivo en el penacho apical.
//!    - **VIP+ (Péptido Intestinal Vasoactivo)**: Inhibe a SST+ bajo señales colinérgicas/noradrenérgicas
//!      de novedad o atención (*desinhibición dendrítica*).

#![allow(clippy::needless_range_loop)]

use crate::dendritic_neuron::{PyramidalNeuron, PyramidalParams};
use crate::synapse_stp::{BioSynapseMatrix, SynapseParams};

/// Estado del microcircuito inhibitorio GABAérgico (PV+, SST+, VIP+) de una columna cortical.
#[derive(Clone, Debug)]
pub struct InhibitoryMicrocircuit {
    /// Actividad de interneuronas Parvalbúmina (PV+): inhibición perisomática divisiva.
    pub pv_activity: Vec<f64>,
    /// Actividad de interneuronas Somatostatina (SST+ / Martinotti): inhibición apical predictiva.
    pub sst_activity: Vec<f64>,
    /// Actividad de interneuronas VIP+: desinhibición (inhibe a SST+).
    pub vip_activity: Vec<f64>,
    /// Pesos plásticos locales Piramidal -> SST para que SST aprenda a predecir el flujo top-down
    /// (Sacramento, Ponte Costa, Bengio & Senn, 2018).
    pub w_pyr_to_sst: Vec<f64>,
    /// Fracción de neuronas activas permitidas por inhibición lateral PV+ (*sparsity* k-WTA).
    pub target_sparsity: f64,
}

impl InhibitoryMicrocircuit {
    pub fn new(n_neurons: usize, target_sparsity: f64) -> Self {
        Self {
            pv_activity: vec![0.0; n_neurons],
            sst_activity: vec![0.0; n_neurons],
            vip_activity: vec![0.0; n_neurons],
            // Inicialmente identidad escalada suave para cancelación basal
            w_pyr_to_sst: vec![0.05; n_neurons],
            target_sparsity: target_sparsity.clamp(0.15, 0.90),
        }
    }

    /// Calcula las tres corrientes del microcircuito inhibitorio:
    /// 1. `VIP+` impulsada por neuromodulación colinérgica/noradrenérgica y sorpresa.
    /// 2. `SST+` impulsada por piramidales locales e inhibida por `VIP+`.
    /// 3. `PV+` impulsada por la energía poblacional piramidal (normalización divisiva + k-WTA).
    pub fn step(
        &mut self,
        pyr_rates: &[f64],
        top_down_apical: &[f64],
        ach_level: f64,
        ne_level: f64,
        learn_sst: bool,
        sst_lr: f64,
    ) {
        let n = pyr_rates.len();
        if n == 0 {
            return;
        }

        // Energía media de la población excitatoria (balance E/I instantáneo)
        let mean_abs: f64 = pyr_rates.iter().map(|r| r.abs()).sum::<f64>() / (n as f64);

        // Umbral competitivo lateral (soft k-WTA mediado por células en cesto PV+)
        let mut abs_sorted: Vec<f64> = pyr_rates.iter().map(|r| r.abs()).collect();
        abs_sorted.sort_by(|a, b| b.total_cmp(a));
        let k_idx = ((n as f64 * self.target_sparsity).round() as usize).clamp(1, n - 1);
        let kwta_threshold = abs_sorted[k_idx];

        let vip_drive = (0.6 * ach_level + 0.4 * ne_level).clamp(0.0, 2.0);

        for i in 0..n {
            // 1. VIP+ responde a ACh (incertidumbre esperada) y NE (sorpresa/cambio de contexto)
            self.vip_activity[i] = vip_drive;

            // 2. SST+ (Martinotti) predice la entrada apical a partir de la tasa piramidal local,
            // pero es suprimida cuando VIP+ se activa (desinhibición dendrítica)
            let raw_sst = self.w_pyr_to_sst[i] * pyr_rates[i];
            let disinhibited_sst = raw_sst / (1.0 + 0.8 * self.vip_activity[i]);
            self.sst_activity[i] = disinhibited_sst;

            // Aprendizaje local de la sinapsis Piramidal -> SST (Sacramento et al., 2018):
            // SST ajusta su peso para cancelar la componente predecible del feedback apical,
            // dejando únicamente la sorpresa/error instructivo en el compartimento apical.
            if learn_sst && i < top_down_apical.len() {
                let apical_residual = top_down_apical[i] - raw_sst;
                self.w_pyr_to_sst[i] = (self.w_pyr_to_sst[i]
                    + sst_lr * apical_residual * pyr_rates[i])
                    .clamp(-2.0, 2.0);
            }

            // 3. PV+ (Parvalbúmina) aplica inhibición perisomática proporcional a la actividad
            // global de la capa más inhibición lateral sobre neuronas por debajo del umbral k-WTA
            let sub_kwta = (kwta_threshold - pyr_rates[i].abs()).max(0.0);
            self.pv_activity[i] = 0.25 * mean_abs + 0.60 * sub_kwta;
        }
    }
}

/// Columna Cortical Laminada con capas L4 (entrada granular), L2/3 (asociación/error superficial)
/// y L5/6 (representación profunda/predicción generativa).
#[derive(Clone, Debug)]
pub struct CorticalColumn {
    pub layer_index: usize,
    pub n_in: usize,
    pub n_hidden: usize,
    /// Capa 4 (Granular): proyección aferente *bottom-up* hacia el compartimento basal de L2/3.
    pub syn_l4_to_l23_basal: BioSynapseMatrix,
    /// Conexiones intracolumnares L2/3 -> L5/6 (flujo laminar canónico de Bastos et al., 2012).
    pub syn_l23_to_l56: BioSynapseMatrix,
    /// Proyección generativa descendente (*top-down*) desde L5/6 para predecir la capa inferior ($\hat{x}_{\ell-1}$).
    pub syn_l56_topdown_pred: BioSynapseMatrix,
    /// Neuronas piramidales multicompartimentales de Capa 2/3.
    pub l23_pyramidal: Vec<PyramidalNeuron>,
    /// Neuronas piramidales multicompartimentales de Capa 5/6 (salida y memoria local).
    pub l56_pyramidal: Vec<PyramidalNeuron>,
    /// Microcircuito inhibitorio GABAérgico (PV+, SST+, VIP+).
    pub interneuron_circuit: InhibitoryMicrocircuit,
    /// Última entrada aferente recibida por la Capa 4 ($x_{\ell-1}$).
    pub last_bottom_up_input: Vec<f64>,
    /// Predicción generativa descendente emitida por L5/6 hacia la capa inferior ($\hat{x}_{\ell-1}$).
    pub top_down_prediction_out: Vec<f64>,
    /// Error de predicción sensorial local ponderado por precisión ($\xi_\ell = \Pi_\ell (x - \hat{x})$).
    pub local_prediction_error: Vec<f64>,
    /// Precisión sináptica estimada (inversa de la varianza del error local, Friston 2010).
    pub error_precision: Vec<f64>,
}

impl CorticalColumn {
    pub fn new(layer_index: usize, n_in: usize, n_hidden: usize, u0: f64, seed: u64) -> Self {
        let syn_l4_to_l23_basal =
            BioSynapseMatrix::new(n_hidden, n_in, 1.15, u0, seed.wrapping_add(101));
        let syn_l23_to_l56 =
            BioSynapseMatrix::new(n_hidden, n_hidden, 0.90, u0, seed.wrapping_add(202));
        let syn_l56_topdown_pred =
            BioSynapseMatrix::new(n_in, n_hidden, 0.85, u0, seed.wrapping_add(303));

        Self {
            layer_index,
            n_in,
            n_hidden,
            syn_l4_to_l23_basal,
            syn_l23_to_l56,
            syn_l56_topdown_pred,
            l23_pyramidal: (0..n_hidden).map(|_| PyramidalNeuron::new()).collect(),
            l56_pyramidal: (0..n_hidden).map(|_| PyramidalNeuron::new()).collect(),
            interneuron_circuit: InhibitoryMicrocircuit::new(n_hidden, 0.65),
            last_bottom_up_input: vec![0.0; n_in],
            top_down_prediction_out: vec![0.0; n_in],
            local_prediction_error: vec![0.0; n_in],
            error_precision: vec![1.0; n_in],
        }
    }

    /// Reinicia estados rápidos neuronales y vesiculares entre episodios independientes.
    pub fn reset_short_term_state(&mut self, u0: f64) {
        self.syn_l4_to_l23_basal.reset_short_term_state(u0);
        self.syn_l23_to_l56.reset_short_term_state(u0);
        self.syn_l56_topdown_pred.reset_short_term_state(u0);
        for n in &mut self.l23_pyramidal {
            n.reset_fast_state();
        }
        for n in &mut self.l56_pyramidal {
            n.reset_fast_state();
        }
        self.last_bottom_up_input.fill(0.0);
        self.top_down_prediction_out.fill(0.0);
        self.local_prediction_error.fill(0.0);
    }

    /// Ejecuta un ciclo laminar dentro de la columna cortical:
    /// 1. L4 recibe `bottom_up_input` y excita las dendritas basales de L2/3.
    /// 2. El penacho apical en L1 recibe `apical_feedback` desde la columna cortical superior.
    /// 3. El microcircuito PV/SST/VIP regula inhibición perisomática y desinhibición apical.
    /// 4. L2/3 excita a L5/6, que genera la representación de salida y la predicción *top-down*
    ///    hacia la capa inferior (Rao & Ballard, 1999; Bastos et al., 2012).
    #[allow(clippy::too_many_arguments)]
    pub fn step_column(
        &mut self,
        bottom_up_input: &[f64],
        apical_feedback: &[f64],
        ach_level: f64,
        ne_level: f64,
        pyr_params: &PyramidalParams,
        syn_params: &SynapseParams,
        learn_microcircuit: bool,
    ) -> Vec<f64> {
        self.last_bottom_up_input.copy_from_slice(bottom_up_input);

        // 1. Flujo L4 -> Dendritas basales de L2/3 (con plasticidad de corto plazo Tsodyks-Markram)
        let l23_basal_drive = self
            .syn_l4_to_l23_basal
            .forward(bottom_up_input, syn_params);

        // 2. Actualizar microcircuito inhibitorio PV / SST / VIP
        let prev_l23_rates: Vec<f64> = self.l23_pyramidal.iter().map(|n| n.event_rate).collect();
        self.interneuron_circuit.step(
            &prev_l23_rates,
            apical_feedback,
            ach_level,
            ne_level,
            learn_microcircuit,
            0.015,
        );

        // 3. Paso biofísico de los 3 compartimentos de L2/3
        let mut l23_rates = vec![0.0; self.n_hidden];
        for i in 0..self.n_hidden {
            let ap_in = apical_feedback.get(i).copied().unwrap_or(0.0);
            let sst_inh = self.interneuron_circuit.sst_activity[i];
            let pv_inh = self.interneuron_circuit.pv_activity[i];
            let vip_gate = self.interneuron_circuit.vip_activity[i];

            self.l23_pyramidal[i].step(
                l23_basal_drive[i],
                ap_in,
                sst_inh,
                pv_inh,
                vip_gate,
                pyr_params,
            );
            l23_rates[i] = self.l23_pyramidal[i].event_rate;
        }

        // 4. Flujo laminar L2/3 -> L5/6 con conexión residual somática/basal
        let l56_drive = self.syn_l23_to_l56.forward(&l23_rates, syn_params);
        let mut l56_rates = vec![0.0; self.n_hidden];
        for i in 0..self.n_hidden {
            let ap_in = apical_feedback.get(i).copied().unwrap_or(0.0);
            let combined_basal = 0.65 * l23_rates[i] + 0.35 * l56_drive[i];
            let pv_inh = 0.5 * self.interneuron_circuit.pv_activity[i];
            let vip_gate = self.interneuron_circuit.vip_activity[i];

            self.l56_pyramidal[i].step(combined_basal, ap_in, 0.0, pv_inh, vip_gate, pyr_params);
            l56_rates[i] = self.l56_pyramidal[i].event_rate;
        }

        // 5. Predicción generativa descendente (L5/6 -> capa inferior) y error predictivo local
        // (Rao & Ballard, 1999; Friston, 2005)
        self.top_down_prediction_out = self
            .syn_l56_topdown_pred
            .forward_stateless(&l56_rates, syn_params);

        for j in 0..self.n_in {
            let raw_err = bottom_up_input[j] - self.top_down_prediction_out[j];
            // Estimación adaptativa de precisión (inversa de varianza móvil del error)
            let err_sq = raw_err * raw_err;
            let var_est = (1.0 / self.error_precision[j]).max(0.05);
            let new_var = (0.95 * var_est + 0.05 * err_sq).clamp(0.05, 4.0);
            self.error_precision[j] = 1.0 / new_var;
            self.local_prediction_error[j] = raw_err * self.error_precision[j].sqrt();
        }

        l56_rates
    }

    /// Aprendizaje local sin retropropagación global:
    /// Combina (1) la señal instructiva dendrítica apical (Sacramento et al., 2018; Payeur et al., 2021)
    /// y (2) la minimización de energía libre predictiva local (Rao & Ballard, 1999; Whittington & Bogacz, 2017),
    /// moduladas por el tercer factor neuromodulador (`third_factor_mod`).
    pub fn local_plasticity_step(
        &mut self,
        third_factor_mod: f64,
        lr: f64,
        pyr_params: &PyramidalParams,
        syn_params: &SynapseParams,
    ) -> Vec<f64> {
        let l23_rates: Vec<f64> = self.l23_pyramidal.iter().map(|n| n.event_rate).collect();
        let l56_rates: Vec<f64> = self.l56_pyramidal.iter().map(|n| n.event_rate).collect();

        // Señal instructiva dendrítica en L5/6 y L2/3 (derivada del compartimento apical)
        let l56_teaching: Vec<f64> = self
            .l56_pyramidal
            .iter()
            .map(|n| n.dendritic_teaching_signal)
            .collect();

        // Retroproyección intracolumnar L5/6 -> L2/3 combinada con el potencial apical directo de L2/3
        let intra_back = self.syn_l23_to_l56.backward_project(&l56_teaching);
        let pred_err_back = self
            .syn_l56_topdown_pred
            .backward_project(&self.local_prediction_error);

        let mut l23_teaching = vec![0.0; self.n_hidden];
        for i in 0..self.n_hidden {
            let phi_prime = (1.0 - l23_rates[i] * l23_rates[i]).max(0.05);
            l23_teaching[i] = 0.65 * self.l23_pyramidal[i].dendritic_teaching_signal
                + 0.25 * intra_back[i] * phi_prime
                + 0.10 * pred_err_back[i] * phi_prime;
        }

        // Propagar señal apical instructiva hacia la columna inferior antes de actualizar pesos
        let feedback_to_lower = self.syn_l4_to_l23_basal.backward_project(&l23_teaching);

        // 1. Actualizar sinapsis L4 -> L2/3 basal con regla de 3 factores
        self.syn_l4_to_l23_basal.update_three_factor(
            &self.last_bottom_up_input,
            &l23_teaching,
            &l23_rates,
            third_factor_mod,
            lr,
            syn_params,
        );

        // 2. Actualizar sinapsis intracolumnar L2/3 -> L5/6
        self.syn_l23_to_l56.update_three_factor(
            &l23_rates,
            &l56_teaching,
            &l56_rates,
            third_factor_mod,
            0.6 * lr,
            syn_params,
        );

        // 3. Actualizar predicción generativa top-down L5/6 -> Capa inferior (Rao & Ballard, 1999)
        self.syn_l56_topdown_pred.update_three_factor(
            &l56_rates,
            &self.local_prediction_error,
            &self.top_down_prediction_out,
            third_factor_mod.abs().max(0.25),
            0.4 * lr,
            syn_params,
        );

        // 4. Regulación homeostática y umbral deslizante BCM (Turrigiano & Nelson, 2004)
        for n in &mut self.l23_pyramidal {
            n.update_homeostasis(pyr_params);
        }
        for n in &mut self.l56_pyramidal {
            n.update_homeostasis(pyr_params);
        }

        feedback_to_lower
    }

    /// Consolida las sinapsis de las tres láminas durante el sueño NREM/REM.
    pub fn consolidate_sleep(&mut self, consolidation_rate: f64, shy_downscale: f64) {
        self.syn_l4_to_l23_basal
            .consolidate_synapses(consolidation_rate, shy_downscale);
        self.syn_l23_to_l56
            .consolidate_synapses(consolidation_rate, shy_downscale);
        self.syn_l56_topdown_pred
            .consolidate_synapses(consolidation_rate, shy_downscale);
    }

    pub fn l56_rates(&self) -> Vec<f64> {
        self.l56_pyramidal.iter().map(|n| n.event_rate).collect()
    }

    pub fn n_params(&self) -> usize {
        self.syn_l4_to_l23_basal.n_params()
            + self.syn_l23_to_l56.n_params()
            + self.syn_l56_topdown_pred.n_params()
            + self.n_hidden
    }
}
