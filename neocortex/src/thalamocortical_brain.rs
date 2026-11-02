//! Arquitectura Cerebral Integrada (`BioBrain`): Bucle Talamocortical Jerárquico,
//! Columnas Corticales Laminadas, Sistema Hipocampal (DG-CA3-CA1), Oscilaciones Theta-Gamma
//! y Consolidación en Sueño (Sharp-Wave Ripples + Homeostasis Sináptica SHY).
//!
//! Fundamentación neurocientífica:
//! 1. **Sherman (2016)**, *Nature Neuroscience*, 19, 533-541:
//!    El tálamo no es un simple relevo pasivo; el tálamo de primer orden (LGN/VPM) y el tálamo
//!    de orden superior (**Pulvinar** / **Núcleo Mediodorsal, MD**) controlan la ganancia y
//!    precisión de la transmisión entre áreas corticales según el contexto y la atención colinérgica.
//! 2. **Sacramento et al. (2018)**, *NeurIPS*; **Payeur et al. (2021)**, *Nature Neuroscience*;
//!    **Rao & Ballard (1999)**, *Nature Neuroscience*:
//!    Asignación de crédito multicapa 100% local a través de dendritas apicales y errores
//!    predictivos, sin retropropagación global de gradientes (*Backprop-Free*).
//! 3. **McClelland, McNaughton & O'Reilly (1995)**; **Buzsáki (2015)**:
//!    Durante la vigilia, la corteza aprende gradualmente mientras el hipocampo (DG-CA3-CA1)
//!    captura episodios rápidos; durante el sueño NREM, el tálamo bloquea las aferencias externas
//!    y las ráfagas **Sharp-Wave Ripples (SWR)** del hipocampo entrenan a la neocorteza para
//!    consolidar el conocimiento sin olvido catastrófico.

#![allow(clippy::needless_range_loop)]

use crate::cortical_column::CorticalColumn;
use crate::dendritic_neuron::PyramidalParams;
use crate::hippocampus_cls::HippocampalSystem;
use crate::oscillations_neuromod::{NeuromodulatorySystem, ThetaGammaOscillator};
use crate::rng::Xoshiro256StarStar;
use crate::synapse_stp::{BioSynapseMatrix, SynapseParams};

/// Interfaz general de inferencia predictiva para sistemas dinámicos y cognitivos.
pub trait Predictor {
    fn name(&self) -> &str;
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64>;
    fn params(&self) -> usize;
}

/// Compuerta Talámica de Primer Orden y Orden Superior (Pulvinar / Mediodorsal).
#[derive(Clone, Debug)]
pub struct ThalamicGate {
    pub sensory_dim: usize,
    pub context_dim: usize,
    /// Ganancia de precisión por canal sensorial en el núcleo de relevo talámico.
    pub sensory_precision_gain: Vec<f64>,
    /// Ganancia de modulación contextual en el núcleo Pulvinar / Mediodorsal (MD).
    pub pulvinar_context_gain: Vec<f64>,
    /// Nivel de inhibición del Núcleo Reticular Talámico (TRN): alto en sueño NREM para
    /// desconectar el mundo exterior durante la consolidación por ondas lentas y SWR.
    pub trn_sleep_gate: f64,
}

impl ThalamicGate {
    pub fn new(sensory_dim: usize, context_dim: usize) -> Self {
        Self {
            sensory_dim,
            context_dim,
            sensory_precision_gain: vec![1.0; sensory_dim],
            pulvinar_context_gain: vec![1.0; context_dim],
            trn_sleep_gate: 0.0,
        }
    }

    /// Combina y pondera la entrada sensorial y el contexto a través del tálamo.
    pub fn relay_input(&self, x: &[f64], c: &[f64], ach_level: f64) -> Vec<f64> {
        let wake_factor = (1.0 - self.trn_sleep_gate).clamp(0.0, 1.0);
        // La acetilcolina (ACh) suprime el núcleo reticular (TRN) y amplifica la transmisión sensorial
        let ach_boost = 0.85 + 0.15 * ach_level;
        let mut out = Vec::with_capacity(self.sensory_dim + self.context_dim);
        for (i, &val) in x.iter().enumerate() {
            let g = self.sensory_precision_gain.get(i).copied().unwrap_or(1.0);
            out.push(val * g * ach_boost * wake_factor);
        }
        for (i, &val) in c.iter().enumerate() {
            let g = self.pulvinar_context_gain.get(i).copied().unwrap_or(1.0);
            out.push(val * g * wake_factor);
        }
        out
    }
}

/// Métricas instantáneas de un ciclo cognitivo cerebral.
#[derive(Clone, Debug)]
pub struct BrainStepTelemetry {
    /// Predicción combinada neocortical + hipocampal.
    pub output: Vec<f64>,
    /// Predicción puramente neocortical (columnas lentas consolidadas).
    pub cortical_output: Vec<f64>,
    /// Predicción episódica recuperada por CA3-CA1.
    pub hippocampal_output: Vec<f64>,
    /// Novedad detectada por el comparador CA1 ($[0, 1]$).
    pub ca1_novelty: f64,
    /// Sorpresa predictiva media en las columnas corticales.
    pub cortical_free_energy: f64,
    /// Nivel fásico de Dopamina (VTA).
    pub dopamine: f64,
    /// Nivel de Acetilcolina (NBM).
    pub acetylcholine: f64,
    /// Nivel de Norepinefrina (Locus Coeruleus).
    pub norepinephrine: f64,
    /// Indicador de cambio de contexto detectado por el Locus Coeruleus.
    pub context_switch: bool,
}

/// Arquitectura Cerebral Completa Bioinspirada (`BioBrain`).
#[derive(Clone, Debug)]
pub struct BioBrain {
    pub sensory_dim: usize,
    pub context_dim: usize,
    pub output_dim: usize,
    /// Tálamo sensorial y Pulvinar/MD.
    pub thalamus: ThalamicGate,
    /// Jerarquía de columnas neocorticales laminadas (p. ej. V1/S1 -> Asociación -> Prefrontal).
    pub columns: Vec<CorticalColumn>,
    /// Proyección de lectura Cortico-Estriatal / Motora desde la capa L5/6 de todas las columnas.
    pub readout_synapses: BioSynapseMatrix,
    /// Proyección residual directa tálamo-cortical hacia la salida (vía piramidal tractal L5b).
    pub direct_pathway: BioSynapseMatrix,
    /// Sistema Hipocampal Trisináptico (EC -> DG -> CA3 -> CA1).
    pub hippocampus: HippocampalSystem,
    /// Reloj oscilatorio acoplado Theta-Gamma.
    pub oscillator: ThetaGammaOscillator,
    /// Núcleos neuromoduladores ascendentes (DA, ACh, NE, 5-HT).
    pub neuromod: NeuromodulatorySystem,
    /// Parámetros biofísicos de neuronas piramidales.
    pub pyr_params: PyramidalParams,
    /// Parámetros biofísicos de sinapsis.
    pub syn_params: SynapseParams,
    /// Tasa de aprendizaje base cortical.
    pub base_learning_rate: f64,
    /// Peso de mezcla entre memoria episódica hipocampal CA3-CA1 y neocorteza.
    pub hippocampal_blend: f64,
}

impl BioBrain {
    /// Construye una nueva arquitectura `BioBrain` con `column_widths` columnas corticales en jerarquía.
    pub fn new(
        sensory_dim: usize,
        context_dim: usize,
        output_dim: usize,
        column_widths: &[usize],
        seed: u64,
    ) -> Self {
        let total_in = sensory_dim + context_dim;
        let pyr_params = PyramidalParams::default();
        let syn_params = SynapseParams::default();

        let mut columns = Vec::with_capacity(column_widths.len());
        let mut prev_dim = total_in;
        for (idx, &w) in column_widths.iter().enumerate() {
            let col_seed = seed.wrapping_add((idx as u64 + 1) * 10_007);
            columns.push(CorticalColumn::new(
                idx,
                prev_dim,
                w,
                syn_params.u0,
                col_seed,
            ));
            prev_dim = w;
        }

        let total_l56_features: usize = column_widths.iter().sum();
        let readout_synapses = BioSynapseMatrix::new(
            output_dim,
            total_l56_features,
            0.90,
            syn_params.u0,
            seed.wrapping_add(90_001),
        );
        let direct_pathway = BioSynapseMatrix::new(
            output_dim,
            total_in,
            0.35,
            syn_params.u0,
            seed.wrapping_add(95_001),
        );

        let dg_dim = (total_in * 8).clamp(64, 512);
        let hippocampus = HippocampalSystem::new(
            total_in,
            dg_dim,
            output_dim,
            0.10,
            18.0,
            512,
            seed.wrapping_add(77_777),
        );

        Self {
            sensory_dim,
            context_dim,
            output_dim,
            thalamus: ThalamicGate::new(sensory_dim, context_dim),
            columns,
            readout_synapses,
            direct_pathway,
            hippocampus,
            oscillator: ThetaGammaOscillator::new(4),
            neuromod: NeuromodulatorySystem::new(),
            pyr_params,
            syn_params,
            base_learning_rate: 0.035,
            hippocampal_blend: 0.35,
        }
    }

    /// Reinicia los potenciales de membrana rápidos y vesículas sinápticas de corto plazo
    /// (preservando los pesos aprendidos, la consolidación estructural y los engramas de CA3).
    pub fn reset_working_memory(&mut self) {
        let u0 = self.syn_params.u0;
        for col in &mut self.columns {
            col.reset_short_term_state(u0);
        }
        self.readout_synapses.reset_short_term_state(u0);
        self.direct_pathway.reset_short_term_state(u0);
    }

    /// Ejecuta un ciclo Theta completo (con `n_gamma_cycles` subciclos Gamma de relajación
    /// recurrente talamocortical) para inferir la salida a partir de `x` y `c`.
    pub fn forward_cognitive_cycle(
        &mut self,
        x: &[f64],
        c: &[f64],
        n_gamma_cycles: usize,
    ) -> BrainStepTelemetry {
        let thalamic_in = self.thalamus.relay_input(x, c, self.neuromod.acetylcholine);

        // 1. Consulta rápida al circuito trisináptico hipocampal (EC -> DG -> CA3 -> CA1)
        let (_completed_ec, ca1_recalled, ca1_novelty) =
            self.hippocampus.ca3_pattern_completion(&thalamic_in);

        // 2. Relajación talamocortical a través de subciclos Gamma anidados en el ciclo Theta
        let steps = n_gamma_cycles.max(1);
        let n_cols = self.columns.len();

        // Retroalimentación apical inicial entre columnas (desde L5/6 de la capa superior)
        let mut apical_signals: Vec<Vec<f64>> = self
            .columns
            .iter()
            .map(|col| vec![0.0; col.n_hidden])
            .collect();

        for _ in 0..steps {
            self.oscillator.step_gamma_subcycle();
            let retrieval_gate = self.oscillator.retrieval_gate();

            // Propagación ascendente (L4 -> L2/3 -> L5/6) modulada por entradas apicales
            let mut bottom_up = thalamic_in.clone();
            for idx in 0..n_cols {
                let ap = &apical_signals[idx];
                let l56_out = self.columns[idx].step_column(
                    &bottom_up,
                    ap,
                    self.neuromod.acetylcholine,
                    self.neuromod.norepinephrine,
                    &self.pyr_params,
                    &self.syn_params,
                    false,
                );
                bottom_up = l56_out;
            }

            // Propagación descendente (L5/6 superior -> penacho apical L1 de la columna inferior)
            for idx in (0..n_cols.saturating_sub(1)).rev() {
                let pred_from_upper = &self.columns[idx + 1].local_prediction_error;
                for j in 0..apical_signals[idx].len().min(pred_from_upper.len()) {
                    apical_signals[idx][j] = -retrieval_gate * pred_from_upper[j];
                }
            }
        }

        // 3. Concatenar representaciones profundas L5/6 de todas las columnas corticales
        let mut all_l56 = Vec::new();
        let mut total_fe = 0.0;
        let mut fe_count = 0usize;
        for col in &self.columns {
            all_l56.extend(col.l56_rates());
            for &err in &col.local_prediction_error {
                total_fe += err * err;
                fe_count += 1;
            }
        }
        let cortical_free_energy = if fe_count > 0 {
            (total_fe / (fe_count as f64)).sqrt()
        } else {
            0.0
        };

        let readout_part = self.readout_synapses.forward(&all_l56, &self.syn_params);
        let direct_part = self.direct_pathway.forward(&thalamic_in, &self.syn_params);

        let mut cortical_output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            cortical_output[i] = readout_part[i] + direct_part[i];
        }

        // 4. Integración Complementaria Corteza + Hipocampo:
        // Si el hipocampo reconoce fuertemente un patrón episódico (baja novedad en CA1),
        // complementa a la corteza; si el episodio es completamente desconocido, predomina la corteza.
        let familiarity = (1.0 - ca1_novelty).clamp(0.0, 1.0);
        let eff_blend = (self.hippocampal_blend * familiarity * familiarity).clamp(0.0, 0.85);

        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            output[i] = (1.0 - eff_blend) * cortical_output[i] + eff_blend * ca1_recalled[i];
        }

        BrainStepTelemetry {
            output,
            cortical_output,
            hippocampal_output: ca1_recalled,
            ca1_novelty,
            cortical_free_energy,
            dopamine: self.neuromod.dopamine,
            acetylcholine: self.neuromod.acetylcholine,
            norepinephrine: self.neuromod.norepinephrine,
            context_switch: self.neuromod.context_switch_detected,
        }
    }

    /// Paso de aprendizaje biológico *online* 100% libre de retropropagación global:
    /// 1. Ejecuta el ciclo cognitivo Theta-Gamma.
    /// 2. Calcula la sorpresa respecto a la observación real `target` y actualiza los
    ///    4 núcleos neuromoduladores (DA, ACh, NE, 5-HT).
    /// 3. Inyecta el error de predicción en las dendritas apicales de la capa superior;
    ///    cada columna propaga localmente las señales dendríticas (Sacramento et al., 2018;
    ///    Payeur et al., 2021) y actualiza sus sinapsis mediante la regla de 3 factores.
    /// 4. Codifica el episodio en el complejo hipocampal DG-CA3-CA1 si hay saliencia suficiente.
    pub fn observe_and_learn_local(
        &mut self,
        x: &[f64],
        c: &[f64],
        target: &[f64],
        encode_in_hippocampus: bool,
    ) -> BrainStepTelemetry {
        let telemetry = self.forward_cognitive_cycle(x, c, 2);
        let thalamic_in = self.thalamus.relay_input(x, c, self.neuromod.acetylcholine);

        // Error de predicción en la salida cortical
        let mut pred_err = vec![0.0; self.output_dim];
        let mut mse = 0.0;
        for i in 0..self.output_dim {
            let e = target[i] - telemetry.cortical_output[i];
            pred_err[i] = e;
            mse += e * e;
        }
        let surprise = (mse / (self.output_dim.max(1) as f64)).sqrt();

        // Actualizar neuromoduladores (VTA Dopamina, NBM Acetilcolina, LC Norepinefrina, Rafe 5-HT)
        self.neuromod.update(surprise, Some(-surprise));
        let third_factor = self.neuromod.third_factor_plasticity_gain();
        let lr = self.base_learning_rate;

        // Concatenar tasas L5/6 actuales para actualizar lectura cortico-estriatal
        let mut all_l56 = Vec::new();
        for col in &self.columns {
            all_l56.extend(col.l56_rates());
        }

        // Proyectar error hacia las dendritas apicales de L5/6 de cada columna
        let apical_from_readout = self.readout_synapses.backward_project(&pred_err);

        // Actualizar sinapsis de lectura y vía directa con regla de 3 factores
        self.readout_synapses.update_three_factor(
            &all_l56,
            &pred_err,
            &telemetry.cortical_output,
            third_factor,
            lr,
            &self.syn_params,
        );
        self.direct_pathway.update_three_factor(
            &thalamic_in,
            &pred_err,
            &telemetry.cortical_output,
            third_factor,
            0.35 * lr,
            &self.syn_params,
        );

        // Asignación de crédito dendrítica local columna por columna (de arriba hacia abajo)
        let n_cols = self.columns.len();
        let mut col_offsets = Vec::with_capacity(n_cols);
        let mut acc = 0usize;
        for col in &self.columns {
            col_offsets.push(acc);
            acc += col.n_hidden;
        }

        let mut feedback_from_above = vec![0.0; self.columns.last().map_or(0, |c| c.n_hidden)];
        for idx in (0..n_cols).rev() {
            let offset = col_offsets[idx];
            let n_h = self.columns[idx].n_hidden;

            // Combinar la señal apical directa desde el estrato motor/lectura con la señal
            // descendente de la columna cortical inmediatamente superior
            let mut combined_apical = vec![0.0; n_h];
            for i in 0..n_h {
                let r_ap = apical_from_readout[offset + i];
                let u_ap = if idx + 1 < n_cols {
                    feedback_from_above.get(i).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                combined_apical[i] = r_ap + 0.65 * u_ap;
            }

            // Re-evaluar el compartimento apical con el error instructivo y el microcircuito SST/VIP
            let bottom_up = self.columns[idx].last_bottom_up_input.clone();
            self.columns[idx].step_column(
                &bottom_up,
                &combined_apical,
                self.neuromod.acetylcholine,
                self.neuromod.norepinephrine,
                &self.pyr_params,
                &self.syn_params,
                true,
            );

            feedback_from_above = self.columns[idx].local_plasticity_step(
                third_factor,
                lr,
                &self.pyr_params,
                &self.syn_params,
            );
        }

        // Codificación rápida en el circuito hipocampal DG-CA3-CA1 (McClelland et al., 1995)
        if encode_in_hippocampus {
            let salience = (0.5 + surprise + 0.3 * self.neuromod.norepinephrine).clamp(0.1, 5.0);
            self.hippocampus
                .encode_episode(&thalamic_in, target, salience);
        }

        BrainStepTelemetry {
            dopamine: self.neuromod.dopamine,
            acetylcholine: self.neuromod.acetylcholine,
            norepinephrine: self.neuromod.norepinephrine,
            context_switch: self.neuromod.context_switch_detected,
            ..telemetry
        }
    }

    /// **Fase de Sueño NREM + REM (Sharp-Wave Ripples + Consolidación Sináptica SHY)**
    /// (Buzsáki, 2015; Frey & Morris, 1997; Tononi & Cirelli, 2014):
    /// 1. El Núcleo Reticular Talámico (`trn_sleep_gate = 1.0`) desconecta las entradas externas.
    /// 2. CA3 emite ráfagas de *Sharp-Wave Ripples (SWR)* reproduciendo episodios almacenados
    ///    hacia las columnas neocorticales para transferir memorias rápidas al sustrato cortical.
    /// 3. Se ejecuta la captura de etiquetas sinápticas (*Synaptic Tagging & Capture*) hacia el
    ///    andamiaje estructural consolidado ($W^{\text{cons}}$) y el reescalado homeostático SHY.
    pub fn sleep_consolidation_cycle(
        &mut self,
        n_swr_ripples: usize,
        consolidation_rate: f64,
        shy_downscale: f64,
        seed: u64,
    ) -> f64 {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        self.thalamus.trn_sleep_gate = 1.0;

        let batch = self
            .hippocampus
            .sample_sharp_wave_ripple_batch(n_swr_ripples, &mut rng);

        // Desbloquear temporalmente el relé interno para reproducir los engramas desde CA3
        self.thalamus.trn_sleep_gate = 0.0;
        let mut total_replay_err = 0.0;
        let count = batch.len().max(1);

        for (ec_in, ca1_target, _salience) in batch {
            let x_part = &ec_in[..self.sensory_dim];
            let c_part = &ec_in[self.sensory_dim..];
            // Durante el sueño NREM no se re-codifican nuevos engramas en el hipocampo
            let tel = self.observe_and_learn_local(x_part, c_part, &ca1_target, false);
            let err: f64 = tel
                .cortical_output
                .iter()
                .zip(ca1_target.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / (self.output_dim.max(1) as f64);
            total_replay_err += err.sqrt();
        }

        // Consolidar espinas dendríticas y aplicar homeostasis sináptica del sueño (SHY)
        for col in &mut self.columns {
            col.consolidate_sleep(consolidation_rate, shy_downscale);
        }
        self.readout_synapses
            .consolidate_synapses(consolidation_rate, shy_downscale);
        self.direct_pathway
            .consolidate_synapses(consolidation_rate, shy_downscale);

        self.reset_working_memory();
        total_replay_err / (count as f64)
    }

    /// Inferencia sin mutar el estado (para compatibilidad con el trait `Predictor`).
    pub fn predict_immutable(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        let mut clone = self.clone();
        clone.forward_cognitive_cycle(x, c, 2).output
    }

    /// Número total de parámetros sinápticos efectivos de la corteza.
    pub fn total_synaptic_params(&self) -> usize {
        let cols_p: usize = self.columns.iter().map(|c| c.n_params()).sum();
        cols_p + self.readout_synapses.n_params() + self.direct_pathway.n_params()
    }
}

impl Predictor for BioBrain {
    fn name(&self) -> &str {
        "BioBrain_Neocortex_CLS"
    }

    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        self.predict_immutable(x, c)
    }

    fn params(&self) -> usize {
        self.total_synaptic_params()
    }
}
