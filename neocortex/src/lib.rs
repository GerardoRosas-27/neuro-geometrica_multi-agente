//! # `neocortex`: Arquitectura Neuronal Bioinspirada Fiel al Cerebro Humano
//!
//! Implementación completa, independiente y científicamente fundamentada de un sistema
//! talamocortical-hipocampal humano que reemplaza las abstracciones no biológicas del
//! Deep Learning estándar (neuronas puntuales, pesos sin signo, retropropagación simétrica global,
//! ausencia de escalas temporales sinápticas y olvido catastrófico) por sus contrapartes
//! neurobiológicas demostradas en la literatura experimental y computacional:
//!
//! 1. **[`dendritic_neuron`]**: Neuronas Piramidales Multicompartimentales (Basal $v_B$, Apical $v_A$,
//!    Soma $u_S$) con acoplamiento de espigas de calcio ($\text{Ca}^{2+}$ plateau potentials,
//!    *Larkum 2013*), multiplexación de eventos/ráfagas (*Payeur et al., 2021*), asignación de crédito
//!    dendrítico local (*Sacramento et al., 2018; Guerguiev et al., 2017*) y metaplasticidad homeostática
//!    BCM (*Bienenstock, Cooper & Munro, 1982; Turrigiano & Nelson, 2004*).
//! 2. **[`synapse_stp`]**: Sinapsis biofísicas sujetas a la **Ley de Dale** ($W_{\text{eff}} = W^E - W^I$
//!    con $W^E \ge 0, W^I \ge 0$, *Eccles 1954; Parisien et al., 2008*), plasticidad de corto plazo
//!    de **Tsodyks-Markram** (*1997*; *Mongillo et al., 2008*), trazas de elegibilidad Neo-Hebbianas
//!    de 3 factores (*Gerstner et al., 2018*) y consolidación estructural con etiquetado sináptico
//!    (*Frey & Morris, 1997; Benna & Fusi, 2016*).
//! 3. **[`cortical_column`]**: Columna Cortical Laminada Canónica (**L4** granular, **L2/3** superficial,
//!    **L5/6** profunda, *Mountcastle 1997; Bastos et al., 2012*) con microcircuito inhibitorio
//!    GABAérgico de tres poblaciones (**PV+** perisomática / k-WTA, **SST+** Martinotti apical
//!    predictiva y **VIP+** desinhibitoria, *Pfeffer et al., 2013; Tremblay et al., 2016*).
//! 4. **[`oscillations_neuromod`]**: Reloj de acoplamiento de fase-amplitud **Theta-Gamma**
//!    (*Buzsáki, 2006; Hasselmo et al., 2002; Lisman & Jensen, 2013*) y cuatro sistemas
//!    neuromoduladores (**Dopamina** VTA, **Acetilcolina** NBM, **Norepinefrina** Locus Coeruleus,
//!    **Serotonina** Rafe, *Doya 2002; Yu & Dayan 2005*).
//! 5. **[`hippocampus_cls`]**: Circuito Trisináptico Hipocampal (**Corteza Entorrinal $\rightarrow$
//!    Giro Dentado $\rightarrow$ CA3 $\rightarrow$ CA1**) con separación de patrones ultra-dispersa,
//!    completado de patrones por atractores continuos (*Rolls, 2013; Ramsauer et al., 2020*) y
//!    reproducción en sueño mediante **Sharp-Wave Ripples (SWR)** (*McClelland et al., 1995; Buzsáki, 2015*).
//! 6. **[`thalamocortical_brain`]**: Arquitectura unificada [`thalamocortical_brain::BioBrain`] con
//!    compuerta talámica (Pulvinar / TRN, *Sherman 2016*) e implementación del trait `Predictor`.

pub mod cortical_column;
pub mod dendritic_neuron;
pub mod hippocampus_cls;
pub mod oscillations_neuromod;
pub mod rng;
pub mod synapse_stp;
pub mod thalamocortical_brain;

pub use cortical_column::{CorticalColumn, InhibitoryMicrocircuit};
pub use dendritic_neuron::{PyramidalNeuron, PyramidalParams};
pub use hippocampus_cls::{EpisodicEngram, HippocampalSystem};
pub use oscillations_neuromod::{NeuromodulatorySystem, ThetaGammaOscillator};
pub use rng::Xoshiro256StarStar;
pub use synapse_stp::{BioSynapseMatrix, SynapseParams};
pub use thalamocortical_brain::{BioBrain, BrainStepTelemetry, Predictor, ThalamicGate};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dales_law_strictly_preserved() {
        let mut syn = BioSynapseMatrix::new(8, 6, 1.0, 0.3, 42);
        let params = SynapseParams::default();
        let pre = vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.9];
        let post_teach = vec![0.4, -0.6, 0.2, -0.8, 0.5, -0.1, 0.3, -0.4];
        let post_rate = vec![0.2; 8];

        for _ in 0..25 {
            let _ = syn.forward(&pre, &params);
            syn.update_three_factor(&pre, &post_teach, &post_rate, 1.2, 0.05, &params);
        }

        // Verificar que todas las sinapsis excitatorias e inhibitorias son estrictamente no-negativas
        for &we in &syn.w_exc {
            assert!(we >= 0.0, "Violación de la Ley de Dale en W^E: {we}");
        }
        for &wi in &syn.w_inh {
            assert!(wi >= 0.0, "Violación de la Ley de Dale en W^I: {wi}");
        }
    }

    #[test]
    fn test_tsodyks_markram_short_term_plasticity_working_memory() {
        let mut syn = BioSynapseMatrix::new(4, 4, 1.0, 0.30, 100);
        let params = SynapseParams::default();
        let u_rest = syn.stp_u[0];
        let x_rest = syn.stp_x[0];

        // Aplicar un tren de pulsos presinápticos intensos
        let pulse = vec![1.0, 0.0, 0.0, 0.0];
        for _ in 0..5 {
            let _ = syn.forward(&pulse, &params);
        }

        // El canal 0 debe mostrar facilitación de calcio (u > u_rest) y depresión vesicular (x < x_rest)
        assert!(
            syn.stp_u[0] > u_rest,
            "La facilitación por Ca2+ (u) debe aumentar tras actividad presináptica"
        );
        assert!(
            syn.stp_x[0] < x_rest,
            "Las vesículas disponibles (x) deben disminuir tras liberación repetida"
        );
        // El canal 1 inactivo permanece en reposo
        assert!((syn.stp_u[1] - u_rest).abs() < 1e-9);
        assert!((syn.stp_x[1] - x_rest).abs() < 1e-9);
    }

    #[test]
    fn test_dentate_gyrus_pattern_separation_and_ca3_completion() {
        let mut hippo = HippocampalSystem::new(8, 128, 4, 0.08, 22.0, 64, 777);
        let pat_a = vec![0.8, 0.6, -0.4, 0.2, -0.5, 0.7, 0.1, -0.3];
        let pat_b = vec![0.75, 0.55, -0.35, 0.25, -0.45, 0.65, 0.15, -0.25];
        let target_a = vec![1.0, -1.0, 0.5, -0.5];
        let target_b = vec![-1.0, 1.0, -0.5, 0.5];

        let dg_a = hippo.dentate_gyrus_pattern_separation(&pat_a);
        let dg_b = hippo.dentate_gyrus_pattern_separation(&pat_b);

        // Verificar dispersión estricta en el Giro Dentado
        let active_count = dg_a.iter().filter(|&&v| v > 0.0).count();
        assert!(
            (5..=15).contains(&active_count),
            "El Giro Dentado debe mantener código ultra-disperso: activos={active_count}"
        );
        // La similitud en DG debe ser menor o igual que la similitud de entrada (separación de patrones)
        let cos_in: f64 = pat_a.iter().zip(&pat_b).map(|(a, b)| a * b).sum::<f64>()
            / (pat_a.iter().map(|x| x * x).sum::<f64>().sqrt()
                * pat_b.iter().map(|x| x * x).sum::<f64>().sqrt());
        let cos_dg: f64 = dg_a.iter().zip(&dg_b).map(|(a, b)| a * b).sum();
        assert!(
            cos_dg < cos_in,
            "El Giro Dentado debe ortogonalizar patrones similares: cos_dg={cos_dg} < cos_in={cos_in}"
        );

        // Almacenar ambos episodios en CA3 y recuperar con clave parcialmente corrupta
        hippo.encode_episode(&pat_a, &target_a, 1.5);
        hippo.encode_episode(&pat_b, &target_b, 1.5);

        let noisy_a = vec![0.8, 0.6, 0.0, 0.2, -0.5, 0.0, 0.1, -0.3];
        let (_comp_ec, recalled_target, novelty) = hippo.ca3_pattern_completion(&noisy_a);
        let err_a: f64 = recalled_target
            .iter()
            .zip(&target_a)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            err_a < 0.25,
            "CA3 debe completar el patrón y recuperar target_a con bajo error: err={err_a}"
        );
        assert!(novelty < 0.35);
    }

    #[test]
    fn test_local_dendritic_learning_reduces_error_without_backprop() {
        let mut brain = BioBrain::new(6, 4, 4, &[24, 16], 2026);
        let x = vec![0.5, -0.4, 0.7, -0.2, 0.3, -0.6];
        let c = vec![1.0, 0.0, 0.4, -0.3];
        let target = vec![0.6, -0.5, 0.3, 0.8];

        let initial_pred = brain.predict(&x, &c);
        let initial_mse: f64 = initial_pred
            .iter()
            .zip(&target)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / 4.0;

        for _ in 0..30 {
            brain.observe_and_learn_local(&x, &c, &target, true);
        }

        let final_pred = brain.predict(&x, &c);
        let final_mse: f64 = final_pred
            .iter()
            .zip(&target)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / 4.0;

        assert!(
            final_mse < initial_mse * 0.05,
            "El aprendizaje dendrítico local de 3 factores debe reducir drásticamente el MSE: inicial={initial_mse:.6} -> final={final_mse:.6}"
        );
    }

    #[test]
    fn test_cls_sleep_consolidation_mitigates_catastrophic_forgetting() {
        let mut brain = BioBrain::new(4, 2, 2, &[20, 16], 999);
        let task_a_x = vec![0.8, -0.5, 0.3, 0.6];
        let task_a_c = vec![1.0, 0.0];
        let task_a_y = vec![0.75, -0.65];

        let task_b_x = vec![-0.6, 0.7, -0.4, 0.5];
        let task_b_c = vec![0.0, 1.0];
        let task_b_y = vec![-0.80, 0.70];

        // Fase 1: Aprender Tarea A y consolidar en sueño NREM SWR
        for _ in 0..25 {
            brain.observe_and_learn_local(&task_a_x, &task_a_c, &task_a_y, true);
        }
        brain.sleep_consolidation_cycle(20, 0.65, 0.02, 111);

        // Fase 2: Aprender Tarea B secuencialmente y consolidar en sueño NREM SWR
        for _ in 0..25 {
            brain.observe_and_learn_local(&task_b_x, &task_b_c, &task_b_y, true);
        }
        brain.sleep_consolidation_cycle(30, 0.50, 0.02, 222);

        let pred_a = brain.predict(&task_a_x, &task_a_c);
        let pred_b = brain.predict(&task_b_x, &task_b_c);

        let rmse_a: f64 = pred_a
            .iter()
            .zip(&task_a_y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let rmse_b: f64 = pred_b
            .iter()
            .zip(&task_b_y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            rmse_a < 0.20 && rmse_b < 0.20,
            "Ambas tareas deben recordarse tras aprendizaje secuencial + sueño SWR: rmse_a={rmse_a:.4}, rmse_b={rmse_b:.4}"
        );
    }
}
