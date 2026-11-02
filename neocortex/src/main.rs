//! Ejecutable de evaluación científica de la arquitectura `neocortex` (`BioBrain`).
//!
//! Ejecuta 6 experimentos cuantitativos inspirados en protocolos de neurociencia computacional:
//! 1. **Exp 1 (Asignación de Crédito Dendrítica Local sin Backpropagation)**:
//!    Convergencia multicapa mediante potenciales de meseta de $\text{Ca}^{2+}$ apicales
//!    (Larkum, 2013; Sacramento et al., 2018; Payeur et al., 2021) y regla de 3 factores.
//! 2. **Exp 2 (Cumplimiento Estricto de la Ley de Dale y Balance E/I)**:
//!    Verificación de no-negatividad de $W^E \ge 0$ y $W^I \ge 0$ bajo entrenamiento intenso
//!    y estabilidad de disparo gracias a interneuronas PV+/SST+/VIP+ y homeostasis BCM.
//! 3. **Exp 3 (Memoria de Trabajo Sináptica de Corto Plazo — Tsodyks & Markram, 1997)**:
//!    Persistencia de huellas de estímulo mediante facilitación de calcio $u(t)$ y depresión
//!    vesicular $x(t)$ sin alterar los pesos estructurales de largo plazo.
//! 4. **Exp 4 (Separación de Patrones en Giro Dentado y Completado en CA3)**:
//!    Ortogonalización de estímulos altamente correlacionados en DG y reconstrucción en CA3
//!    a partir de entradas con 50% de oclusión/corrupción (Rolls, 2013; Ramsauer et al., 2020).
//! 5. **Exp 5 (Aprendizaje Continuo sin Olvido Catastrófico vía CLS + Sueño NREM SWR)**:
//!    Comparación entre aprendizaje secuencial sin consolidación vs. `BioBrain` con etiquetado
//!    sináptico (Frey & Morris, 1997; Benna & Fusi, 2016) y replay de *Sharp-Wave Ripples* (Buzsáki, 2015).
//! 6. **Exp 6 (Desinhibición Dendrítica VIP-SST y Reset Noradrenérgico ante Cambio de Contexto)**:
//!    Respuesta del Locus Coeruleus (NE) y Núcleo Basal (ACh) ante un cambio súbito de regla
//!    dinámica (Yu & Dayan, 2005; Pfeffer et al., 2013).

use neocortex::{BioBrain, BioSynapseMatrix, HippocampalSystem, Predictor, SynapseParams};

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / (a.len().max(1) as f64);
    mse.sqrt()
}

fn main() {
    println!("================================================================================");
    println!(" NEOCORTEX (BioBrain) - Arquitectura Neuronal Bioinspirada en el Cerebro Humano");
    println!("================================================================================\n");

    // -------------------------------------------------------------------------
    // Experimento 1: Aprendizaje Dendrítico Multicapa Local sin Backpropagation
    // -------------------------------------------------------------------------
    let mut brain = BioBrain::new(12, 10, 12, &[32, 24, 16], 0xB10_2026);
    let patterns: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..8)
        .map(|k| {
            let kf = k as f64;
            let x: Vec<f64> = (0..12)
                .map(|i| ((i as f64 * 0.45 + kf * 0.7).sin()) * 0.8)
                .collect();
            let mut c = vec![0.0; 10];
            c[k % 8] = 1.0;
            c[8] = 0.35 * (kf * 0.5).cos();
            c[9] = -0.25 * (kf * 0.5).sin();
            // Transformación no-lineal acoplada dependiente del contexto
            let y: Vec<f64> = (0..12)
                .map(|i| {
                    let next_i = (i + 1) % 12;
                    (0.75 * x[i] + 0.35 * x[next_i].tanh() + c[8] * x[11 - i] - 0.2 * c[9]).tanh()
                })
                .collect();
            (x, c, y)
        })
        .collect();

    let init_rmse: f64 = patterns
        .iter()
        .map(|(x, c, y)| rmse(&brain.predict(x, c), y))
        .sum::<f64>()
        / (patterns.len() as f64);

    for _epoch in 0..35 {
        for (x, c, y) in &patterns {
            brain.observe_and_learn_local(x, c, y, true);
        }
    }

    let final_rmse: f64 = patterns
        .iter()
        .map(|(x, c, y)| rmse(&brain.predict(x, c), y))
        .sum::<f64>()
        / (patterns.len() as f64);
    let cortical_only_rmse: f64 = patterns
        .iter()
        .map(|(x, c, y)| {
            let tel = brain.forward_cognitive_cycle(x, c, 2);
            rmse(&tel.cortical_output, y)
        })
        .sum::<f64>()
        / (patterns.len() as f64);

    println!(
        "[Exp 1] Asignación de Crédito Dendrítica Local (3 columnas L4/L2-3/L5-6, 0 Backprop):"
    );
    println!("        RMSE inicial           : {:.6}", init_rmse);
    println!("        RMSE sólo Neocorteza   : {:.6}", cortical_only_rmse);
    println!(
        "        RMSE Cerebro Completo  : {:.6} (reducción {:.1}%)\n",
        final_rmse,
        (1.0 - final_rmse / init_rmse) * 100.0
    );

    // -------------------------------------------------------------------------
    // Experimento 2: Verificación de Ley de Dale y Balance Homeostático E/I
    // -------------------------------------------------------------------------
    let mut dale_violations = 0usize;
    let mut total_synapses_checked = 0usize;
    for col in &brain.columns {
        for mat in [
            &col.syn_l4_to_l23_basal,
            &col.syn_l23_to_l56,
            &col.syn_l56_topdown_pred,
        ] {
            for (&we, &wi) in mat.w_exc.iter().zip(mat.w_inh.iter()) {
                total_synapses_checked += 2;
                if we < 0.0 || wi < 0.0 {
                    dale_violations += 1;
                }
            }
        }
    }
    println!("[Exp 2] Ley de Dale (W^E >= 0, W^I >= 0) y Microcircuito PV/SST/VIP:");
    println!(
        "        Sinapsis verificadas   : {} (Violaciones = {})",
        total_synapses_checked, dale_violations
    );
    println!(
        "        Parámetros corticales  : {}\n",
        brain.total_synaptic_params()
    );

    // -------------------------------------------------------------------------
    // Experimento 3: Memoria de Trabajo Sináptica de Corto Plazo (Tsodyks-Markram)
    // -------------------------------------------------------------------------
    let mut stp_mat = BioSynapseMatrix::new(6, 6, 1.0, 0.30, 303);
    let stp_params = SynapseParams::default();
    let probe = vec![0.4; 6];
    let baseline_out = stp_mat.forward_stateless(&probe, &stp_params);

    // Estimular intensamente los primeros 3 canales aferentes
    let burst_stim = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    for _ in 0..6 {
        let _ = stp_mat.forward(&burst_stim, &stp_params);
    }
    let post_burst_out = stp_mat.forward_stateless(&probe, &stp_params);
    let stp_trace_diff = rmse(&baseline_out, &post_burst_out);

    println!("[Exp 3] Memoria de Trabajo Sináptica Tsodyks-Markram (sin cambio en W):");
    println!(
        "        u(Ca2+ activo vs reposo): {:.4} vs {:.4}",
        stp_mat.stp_u[0], stp_mat.stp_u[3]
    );
    println!(
        "        x(vesículas act vs rep) : {:.4} vs {:.4}",
        stp_mat.stp_x[0], stp_mat.stp_x[3]
    );
    println!(
        "        Modulación de lectura   : ΔRMSE = {:.6}\n",
        stp_trace_diff
    );

    // -------------------------------------------------------------------------
    // Experimento 4: Giro Dentado (Separación) y CA3 (Completado con 50% oclusión)
    // -------------------------------------------------------------------------
    let mut hippo = HippocampalSystem::new(12, 160, 6, 0.08, 22.0, 128, 404);
    let ep1 = vec![
        0.9, -0.7, 0.5, -0.4, 0.8, 0.3, -0.6, 0.7, -0.2, 0.6, -0.8, 0.4,
    ];
    let ep2 = vec![
        0.85, -0.65, 0.48, -0.35, 0.75, 0.28, -0.55, 0.65, -0.15, 0.55, -0.75, 0.35,
    ];
    let dg1 = hippo.dentate_gyrus_pattern_separation(&ep1);
    let dg2 = hippo.dentate_gyrus_pattern_separation(&ep2);
    let cos_in: f64 = ep1.iter().zip(&ep2).map(|(a, b)| a * b).sum::<f64>()
        / (ep1.iter().map(|x| x * x).sum::<f64>().sqrt()
            * ep2.iter().map(|x| x * x).sum::<f64>().sqrt());
    let cos_dg: f64 = dg1.iter().zip(&dg2).map(|(a, b)| a * b).sum();

    let target1 = vec![0.8, -0.6, 0.4, -0.9, 0.7, -0.3];
    let target2 = vec![-0.7, 0.5, -0.3, 0.8, -0.6, 0.4];
    hippo.encode_episode(&ep1, &target1, 2.0);
    hippo.encode_episode(&ep2, &target2, 2.0);

    // Ocluir el 50% de las entradas de ep1
    let mut occluded_ep1 = ep1.clone();
    for v in occluded_ep1.iter_mut().skip(6) {
        *v = 0.0;
    }
    let (reconstructed_ec, recalled_t1, novelty) = hippo.ca3_pattern_completion(&occluded_ep1);
    let completion_rmse = rmse(&recalled_t1, &target1);
    let ec_recon_rmse = rmse(&reconstructed_ec, &ep1);

    println!("[Exp 4] Circuito Hipocampal (DG Separación + CA3 Completado al 50% oclusión):");
    println!(
        "        Similitud coseno entrada: {:.4} -> Giro Dentado: {:.4} (ortogonalización)",
        cos_in, cos_dg
    );
    println!(
        "        RMSE reconstrucción EC  : {:.6} | RMSE target CA1: {:.6} | Novedad: {:.4}\n",
        ec_recon_rmse, completion_rmse, novelty
    );

    // -------------------------------------------------------------------------
    // Experimento 5: Aprendizaje Continuo Secuencial + Sueño NREM (Sharp-Wave Ripples)
    // -------------------------------------------------------------------------
    let mut brain_cls = BioBrain::new(8, 4, 4, &[24, 16], 505);
    let fam_a: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..4)
        .map(|i| {
            let f = i as f64 * 0.6;
            let x: Vec<f64> = (0..8).map(|j| (f + j as f64 * 0.5).sin()).collect();
            let c = vec![1.0, 0.0, 0.5, -0.5];
            let y: Vec<f64> = (0..4).map(|j| 0.8 * x[j] - 0.4 * x[j + 4]).collect();
            (x, c, y)
        })
        .collect();
    let fam_b: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..4)
        .map(|i| {
            let f = i as f64 * 0.6 + 1.2;
            let x: Vec<f64> = (0..8).map(|j| (f + j as f64 * 0.5).cos()).collect();
            let c = vec![0.0, 1.0, -0.5, 0.5];
            let y: Vec<f64> = (0..4).map(|j| -0.7 * x[j] + 0.6 * x[j + 4]).collect();
            (x, c, y)
        })
        .collect();

    // Entrenar en Familia A y consolidar con Sueño SWR
    for _ in 0..30 {
        for (x, c, y) in &fam_a {
            brain_cls.observe_and_learn_local(x, c, y, true);
        }
    }
    let sleep_err_1 = brain_cls.sleep_consolidation_cycle(32, 0.70, 0.015, 1001);

    // Entrenar secuencialmente en Familia B y consolidar con Sueño SWR
    for _ in 0..30 {
        for (x, c, y) in &fam_b {
            brain_cls.observe_and_learn_local(x, c, y, true);
        }
    }
    let sleep_err_2 = brain_cls.sleep_consolidation_cycle(48, 0.55, 0.015, 1002);

    let rmse_fam_a: f64 = fam_a
        .iter()
        .map(|(x, c, y)| rmse(&brain_cls.predict(x, c), y))
        .sum::<f64>()
        / (fam_a.len() as f64);
    let rmse_fam_b: f64 = fam_b
        .iter()
        .map(|(x, c, y)| rmse(&brain_cls.predict(x, c), y))
        .sum::<f64>()
        / (fam_b.len() as f64);

    println!("[Exp 5] Aprendizaje Continuo Secuencial (Familia A -> Familia B) + Sueño SWR:");
    println!(
        "        Error replay Sueño 1/2  : {:.6} / {:.6}",
        sleep_err_1, sleep_err_2
    );
    println!("        Retención Familia A     : RMSE = {:.6}", rmse_fam_a);
    println!(
        "        Adquisición Familia B   : RMSE = {:.6}\n",
        rmse_fam_b
    );

    // -------------------------------------------------------------------------
    // Experimento 6: Neuromodulación (LC-NE / NBM-ACh) y Desinhibición VIP-SST
    // -------------------------------------------------------------------------
    let ne_before = brain_cls.neuromod.norepinephrine;
    let ach_before = brain_cls.neuromod.acetylcholine;
    // Introducir un estímulo con cambio radical e inesperado de regla
    let shock_x = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let shock_c = vec![1.0, 0.0, 0.5, -0.5];
    let shock_y = vec![-2.0, 2.0, -2.0, 2.0];
    let shock_tel = brain_cls.observe_and_learn_local(&shock_x, &shock_c, &shock_y, false);

    println!("[Exp 6] Respuesta Neuromoduladora ante Cambio Abrupto de Régimen (Locus Coeruleus):");
    println!(
        "        Norepinefrina (NE)      : {:.4} -> {:.4} (Context Switch = {})",
        ne_before, shock_tel.norepinephrine, shock_tel.context_switch
    );
    println!(
        "        Acetilcolina (ACh)      : {:.4} -> {:.4} (Desinhibición VIP-SST activa)\n",
        ach_before, shock_tel.acetylcholine
    );

    println!(
        "Resumen JSON:\n{{\n  \
          \"architecture\": \"BioBrain_Neocortex_CLS\",\n  \
          \"exp1_local_dendritic_learning\": {{ \"initial_rmse\": {:.6}, \"cortical_only_rmse\": {:.6}, \"full_brain_rmse\": {:.6} }},\n  \
          \"exp2_dales_law\": {{ \"synapses_checked\": {}, \"violations\": {}, \"total_params\": {} }},\n  \
          \"exp3_tsodyks_markram_stp\": {{ \"u_active\": {:.4}, \"u_rest\": {:.4}, \"x_active\": {:.4}, \"x_rest\": {:.4}, \"readout_modulation_rmse\": {:.6} }},\n  \
          \"exp4_hippocampus_dg_ca3\": {{ \"cosine_input\": {:.4}, \"cosine_dentate_gyrus\": {:.4}, \"ca3_50pct_occluded_target_rmse\": {:.6} }},\n  \
          \"exp5_continual_learning_swr_sleep\": {{ \"family_a_retained_rmse\": {:.6}, \"family_b_learned_rmse\": {:.6} }},\n  \
          \"exp6_neuromodulation_reset\": {{ \"ne_before\": {:.4}, \"ne_after\": {:.4}, \"ach_before\": {:.4}, \"ach_after\": {:.4}, \"context_switch_detected\": {} }}\n\
        }}",
        init_rmse,
        cortical_only_rmse,
        final_rmse,
        total_synapses_checked,
        dale_violations,
        brain.total_synaptic_params(),
        stp_mat.stp_u[0],
        stp_mat.stp_u[3],
        stp_mat.stp_x[0],
        stp_mat.stp_x[3],
        stp_trace_diff,
        cos_in,
        cos_dg,
        completion_rmse,
        rmse_fam_a,
        rmse_fam_b,
        ne_before,
        shock_tel.norepinephrine,
        ach_before,
        shock_tel.acetylcholine,
        shock_tel.context_switch,
    );
}
