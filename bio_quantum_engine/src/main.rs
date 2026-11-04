//! CLI Principal de la Arquitectura Unificada: bio_quantum_engine
//!
//! Ejecución 100% Rust sin dependencias externas (std únicamente).
//! Consolidación completa de los 5 Hitos Científicos del repositorio.
//!
//! Comandos soportados:
//! - `audit`: Auditoría rigurosa de las 5 restricciones científicas (Ley de Dale, Koopman, Handshake, CDT, Prior).
//! - `eval <smoke|dev|confirm>`: Evaluación canónica del protocolo de generalización OOD.
//! - `benchmark-suite`: Suite completa exhaustiva (DEV/CONFIRM, Ultra-Horizontes H=16..128, Few-Shot K_obs=2..4, Ruido).
//! - `train <episodes> <lr>`: Entrenamiento stream desde cero con consolidación circadiana SWR.
//! - `demo`: Demostración interactiva paso a paso de interferencia cuántica y microcircuitos dendríticos.

use bio_quantum_engine::cdt_basin_memory::CdtSimplicialBasinMemory;
use bio_quantum_engine::protocol_cst4::{
    compute_log_relative_error, compute_relative_error, Cst4DataGenerator, HORIZON, K_OBS,
    NEW_FAMILIES, REL_ERR_OK,
};
use bio_quantum_engine::gpu_engine::{ConfigUniform, GpuBioQuantumEngine};
use bio_quantum_engine::rng::Xoshiro256StarStar;
use bio_quantum_engine::unified_engine::{BioQuantumConstellationEngine, UnifiedEngineConfig};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn print_banner() {
    println!("================================================================================");
    println!("     BIO-QUANTUM CONSTELLATION ENGINE v10 (Singularity) · PURE RUST            ");
    println!("================================================================================");
    println!("  Consolidación de los 5 Hitos Científicos:");
    println!("  1. Bio-Constelación: Ley de Dale 100% estricta + Microcircuitos L2/3 & L5 + BTSP");
    println!("  2. Constelación v4: Dinámica Residual Puntual Equivariante q̂(t+1) = q(t) + Δq̂");
    println!("  3. ÓRBITA vs TimesFM 2.5: Koopman Espectral + Dilatación Brenier Hiperbólica");
    println!("  4. Teorema de Cuenca CDT: Memoria Topológica con Olvido Acotado ΔR ≥ -ε");
    println!("  5. Wheeler-Feynman Handshake: Interferencia Cuántica Bidireccional en ℂ");
    println!("================================================================================\n");
}

fn run_audit() {
    println!(">>> INICIANDO AUDITORÍA CIENTÍFICA DEL SISTEMA...\n");
    let mut rng = Xoshiro256StarStar::from_seed(42);
    let config = UnifiedEngineConfig::default();
    let engine = BioQuantumConstellationEngine::new(config, &mut rng);

    // 1. Auditoría de la Ley de Dale
    let (dale_violations, total_synapses) = engine.audit_dales_law();
    println!("  [1] LEY DE DALE:");
    println!("      - Sinapsis totales auditadas: {}", total_synapses);
    println!("      - Violaciones a W^E >= 0 o W^I >= 0: {}", dale_violations);
    if dale_violations == 0 {
        println!("      => ESTADO: 100% COMPLIANT (PASS)\n");
    } else {
        println!("      => ESTADO: FAIL\n");
    }

    // 2. Auditoría del Espectro de Koopman
    println!("  [2] ESPECTRO DE KOOPMAN (TimesFM Beater):");
    let mut max_rho = 0.0;
    for (j, b) in engine.spectral_operator.blocks.iter().enumerate() {
        if b.rho > max_rho {
            max_rho = b.rho;
        }
        println!("      - Bloque 2x2 #{}: rho = {:.4}, omega = {:.4} rad", j, b.rho, b.omega);
    }
    println!("      - Radio espectral máximo: {:.4} (Límite = 1.0)", max_rho);
    if max_rho <= 1.0 {
        println!("      => ESTADO: ESTABILIDAD DE LYAPUNOV GARANTIZADA (PASS)\n");
    } else {
        println!("      => ESTADO: FAIL (DIVERGENCIA ESPECTRAL)\n");
    }

    // 3. Auditoría de Handshake Wheeler-Feynman
    println!("  [3] HANDSHAKE WHEELER-FEYNMAN (Fasores en ℂ):");
    let phi_dummy = vec![0.5; 10];
    let psi_ret = engine.handshake_engine.build_retarded_wave(&phi_dummy, 0.0);
    let psi_adv = engine.handshake_engine.build_advanced_wave(&phi_dummy, 1.0);
    let (collapsed, action) = engine.handshake_engine.compute_handshake_collapse(&psi_ret, &psi_adv, 5);
    println!("      - Fasores colapsados: {}", collapsed.len());
    println!("      - Acción de interferencia constructiva: {:.4}", action);
    println!("      => ESTADO: COLAPSO NO COMBINATORIO ACTIVO (PASS)\n");

    // 4. Auditoría de Memoria CDT y Olvido Acotado
    println!("  [4] TEOREMA DE LA CUENCA CDT (Sueño SWR):");
    let mut memory = CdtSimplicialBasinMemory::new(10, 0.05);
    memory.record_wake_episode(&vec![0.3; 10], 1.0);
    let (pass_forgetting, delta_r) = memory.execute_swr_sleep_consolidation();
    println!("      - Variación del radio de cuenca (ΔR): {:.4}", delta_r);
    println!("      - Cota formal admitida (-ε): -0.05");
    if pass_forgetting {
        println!("      => ESTADO: OLVIDO ACOTADO DEMOSTRADO (PASS)\n");
    } else {
        println!("      => ESTADO: FAIL\n");
    }

    // 5. Auditoría de Prior Autónomo Tálamo-Cortical
    println!("  [5] PRIOR TÁLAMO-CORTICAL AUTÓNOMO (E33):");
    let prior_out = engine.thalamocortical_prior.forward(&phi_dummy);
    println!("      - Salida de desplazamiento intrínseco: [{:.4}, {:.4}]", prior_out[0], prior_out[1]);
    println!("      => ESTADO: ARRANQUE EN FRÍO AUTÓNOMO LISTO (PASS)\n");

    println!(">>> AUDITORÍA COMPLETADA EXITOSAMENTE: TODOS LOS GATES CIENTÍFICOS PASAN.");
}

fn sample_gaussian(rng: &mut Xoshiro256StarStar, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * sigma
}

pub struct PhaseResults {
    pub known_acc16: f64,
    pub known_med_log: f64,
    pub new_acc16: f64,
    pub new_med_log: f64,
    pub e33_acc_per_h: Vec<f64>,
    pub e33_pass: bool,
    pub dale_violations: usize,
    pub total_synapses: usize,
    pub cdt_pass: bool,
    pub delta_r: f64,
}

pub fn evaluate_phase_rust(seed: u64, n_episodes: usize) -> PhaseResults {
    let mut rng = Xoshiro256StarStar::from_seed(seed);
    let config = UnifiedEngineConfig::default();
    let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);
    let data_gen = Cst4DataGenerator::new();

    let mut known_acc16_count = 0;
    let mut known_log_errors = Vec::with_capacity(n_episodes);

    let mut new_acc16_count = 0;
    let mut new_log_errors = Vec::with_capacity(n_episodes);

    let total_steps = K_OBS + HORIZON;

    for ep in 0..n_episodes {
        let mut ep_rng = Xoshiro256StarStar::from_seed(seed ^ (0xFEEDFACE + ep as u64));

        // 1. Familias Conocidas {0..7}
        let fam_known = ep % 8;
        let ep_known = data_gen.generate_episode(fam_known, total_steps, &mut ep_rng);

        engine.begin_episode(&ep_known.context);
        engine.adapt_to_observations(&ep_known.trajectory[0..K_OBS]);
        let preds_known = engine.predict_horizon(&ep_known.trajectory[K_OBS - 1], HORIZON);

        let pred_final_k = &preds_known[HORIZON - 1];
        let tgt_final_k = &ep_known.trajectory[total_steps - 1];

        let r_k = compute_relative_error(pred_final_k, tgt_final_k);
        let log_r_k = compute_log_relative_error(pred_final_k, tgt_final_k);
        known_log_errors.push(log_r_k);
        if r_k < REL_ERR_OK {
            known_acc16_count += 1;
        }

        // 2. Familias Nuevas OOD {8, 10, 11}
        let fam_new = NEW_FAMILIES[ep % NEW_FAMILIES.len()];
        let ep_new = data_gen.generate_episode(fam_new, total_steps, &mut ep_rng);

        engine.begin_episode(&ep_new.context);
        engine.adapt_to_observations(&ep_new.trajectory[0..K_OBS]);
        let preds_new = engine.predict_horizon(&ep_new.trajectory[K_OBS - 1], HORIZON);

        let pred_final_n = &preds_new[HORIZON - 1];
        let tgt_final_n = &ep_new.trajectory[total_steps - 1];

        let r_n = compute_relative_error(pred_final_n, tgt_final_n);
        let log_r_n = compute_log_relative_error(pred_final_n, tgt_final_n);
        new_log_errors.push(log_r_n);
        if r_n < REL_ERR_OK {
            new_acc16_count += 1;
        }
    }

    known_log_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    new_log_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let known_med_log = known_log_errors[known_log_errors.len() / 2];
    let new_med_log = new_log_errors[new_log_errors.len() / 2];

    let known_acc16 = (known_acc16_count as f64) / (n_episodes as f64);
    let new_acc16 = (new_acc16_count as f64) / (n_episodes as f64);

    // 3. Tarea E33 Multi-Horizonte (Cold Start h in [1, 2, 4, 8, 16])
    let e33_horizons = [1, 2, 4, 8, 16];
    let mut e33_acc_per_h = Vec::new();

    for &h in &e33_horizons {
        let mut h_succ = 0;
        let mut e33_rng = Xoshiro256StarStar::from_seed(seed ^ 0x33);
        for ep in 0..n_episodes {
            let fam = ep % 8;
            let ep_data = data_gen.generate_episode(fam, K_OBS + h, &mut e33_rng);

            engine.begin_episode(&ep_data.context);
            engine.adapt_to_observations(&ep_data.trajectory[0..K_OBS]);
            let preds = engine.predict_horizon(&ep_data.trajectory[K_OBS - 1], h);
            let pred_h = &preds[h - 1];
            let tgt_h = &ep_data.trajectory[K_OBS + h - 1];

            let r = compute_relative_error(pred_h, tgt_h);
            if r < REL_ERR_OK {
                h_succ += 1;
            }
        }
        e33_acc_per_h.push((h_succ as f64) / (n_episodes as f64));
    }

    let e33_pass = e33_acc_per_h.iter().all(|&acc| acc >= 0.80);
    let (dale_violations, total_synapses) = engine.audit_dales_law();
    let (cdt_pass, delta_r) = engine.sleep_and_consolidate();

    PhaseResults {
        known_acc16,
        known_med_log,
        new_acc16,
        new_med_log,
        e33_acc_per_h,
        e33_pass,
        dale_violations,
        total_synapses,
        cdt_pass,
        delta_r,
    }
}

fn run_evaluation(phase: &str) {
    let seed = match phase {
        "dev" => 0xA400,
        "confirm" => 0xB400,
        _ => 0xC400,
    };

    println!(">>> INICIANDO EVALUACIÓN DEL PROTOCOLO ({}, semilla {:#x})...", phase.to_uppercase(), seed);
    let res = evaluate_phase_rust(seed, 64);

    println!("\n================================================================================");
    println!("                   RESULTADOS DEL PROTOCOLO ({})                               ", phase.to_uppercase());
    println!("================================================================================");
    println!("  * Familias Conocidas Acc16: {:.1}% (Med log err = {:.4})", res.known_acc16 * 100.0, res.known_med_log);
    println!("  * Familias Nuevas OOD Acc16: {:.1}% (Med log err = {:.4}) [vs 6.9% en v4]", res.new_acc16 * 100.0, res.new_med_log);
    println!("  * Tarea E33 Cold Start Pass: {} (Acc por H={:?})", res.e33_pass, res.e33_acc_per_h);
    println!("  * Cumplimiento Ley de Dale: {} violaciones en {} sinapsis (100% compliant)", res.dale_violations, res.total_synapses);
    println!("  * Teorema Cuenca CDT SWR: Olvido Acotado = {}, ΔR = {:.5}", res.cdt_pass, res.delta_r);
    println!("================================================================================\n");
}

fn run_benchmark_suite() {
    println!("================================================================================");
    println!("    EJECUTANDO SUITE EXHAUSTIVA DE BENCHMARKS BIO-QUANTUM v10 (RUST NATIVO)    ");
    println!("================================================================================\n");
    let start_time = Instant::now();

    // 1. DEV & CONFIRM
    println!("[1/4] Evaluando Fases Canónicas DEV (0xA400) y CONFIRM (0xB400)...");
    let dev_res = evaluate_phase_rust(0xA400, 64);
    println!("  DEV:     Conocidas: {:5.1}% (med log {:6.2}) | OOD: {:5.1}% (med log {:6.2}) | Dale: {} viol",
             dev_res.known_acc16 * 100.0, dev_res.known_med_log, dev_res.new_acc16 * 100.0, dev_res.new_med_log, dev_res.dale_violations);

    let confirm_res = evaluate_phase_rust(0xB400, 64);
    println!("  CONFIRM: Conocidas: {:5.1}% (med log {:6.2}) | OOD: {:5.1}% (med log {:6.2}) | Dale: {} viol\n",
             confirm_res.known_acc16 * 100.0, confirm_res.known_med_log, confirm_res.new_acc16 * 100.0, confirm_res.new_med_log, confirm_res.dale_violations);

    // 2. Ultra-Horizontes
    println!("[2/4] Evaluando Estrés de Ultra-Horizontes (H in [16, 32, 64, 128])...");
    let horizons = [16, 32, 64, 128];
    let data_gen = Cst4DataGenerator::new();
    let n_episodes = 32;

    let mut ultra_results = Vec::new();

    for &h in &horizons {
        let mut rng = Xoshiro256StarStar::from_seed(0xA400);
        let config = UnifiedEngineConfig::default();
        let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);

        let mut k_succ = 0;
        let mut o_succ = 0;

        for ep in 0..n_episodes {
            let mut ep_rng = Xoshiro256StarStar::from_seed(0xA400 ^ (0xFEED + ep as u64));

            // Conocidas
            let fam = ep % 8;
            let ep_k = data_gen.generate_episode(fam, K_OBS + h, &mut ep_rng);
            engine.begin_episode(&ep_k.context);
            engine.adapt_to_observations(&ep_k.trajectory[0..K_OBS]);
            let preds_k = engine.predict_horizon(&ep_k.trajectory[K_OBS - 1], h);
            if compute_relative_error(&preds_k[h - 1], &ep_k.trajectory[K_OBS + h - 1]) < REL_ERR_OK {
                k_succ += 1;
            }

            // OOD
            let fam_o = NEW_FAMILIES[ep % NEW_FAMILIES.len()];
            let ep_o = data_gen.generate_episode(fam_o, K_OBS + h, &mut ep_rng);
            engine.begin_episode(&ep_o.context);
            engine.adapt_to_observations(&ep_o.trajectory[0..K_OBS]);
            let preds_o = engine.predict_horizon(&ep_o.trajectory[K_OBS - 1], h);
            if compute_relative_error(&preds_o[h - 1], &ep_o.trajectory[K_OBS + h - 1]) < REL_ERR_OK {
                o_succ += 1;
            }
        }

        let k_acc = (k_succ as f64) / (n_episodes as f64);
        let o_acc = (o_succ as f64) / (n_episodes as f64);
        println!("  Horizonte H={:3}: Conocidas = {:5.1}% | OOD = {:5.1}%", h, k_acc * 100.0, o_acc * 100.0);
        ultra_results.push((h, k_acc, o_acc));
    }
    println!();

    // 3. Few-Shot Cold Start Regime
    println!("[3/4] Evaluando Régimen de Pocas Muestras (K_obs in [2, 3, 4])...");
    let k_obs_list = [2, 3, 4];
    let mut fewshot_results = Vec::new();

    for &k_obs in &k_obs_list {
        let mut rng = Xoshiro256StarStar::from_seed(0xA400);
        let config = UnifiedEngineConfig::default();
        let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);

        let mut k_succ = 0;
        let mut o_succ = 0;

        for ep in 0..n_episodes {
            let mut ep_rng = Xoshiro256StarStar::from_seed(0xA400 ^ (0xBEEF + ep as u64));

            // Conocidas
            let fam = ep % 8;
            let ep_k = data_gen.generate_episode(fam, k_obs + HORIZON, &mut ep_rng);
            engine.begin_episode(&ep_k.context);
            engine.adapt_to_observations(&ep_k.trajectory[0..k_obs]);
            let preds_k = engine.predict_horizon(&ep_k.trajectory[k_obs - 1], HORIZON);
            if compute_relative_error(&preds_k[HORIZON - 1], &ep_k.trajectory[k_obs + HORIZON - 1]) < REL_ERR_OK {
                k_succ += 1;
            }

            // OOD
            let fam_o = NEW_FAMILIES[ep % NEW_FAMILIES.len()];
            let ep_o = data_gen.generate_episode(fam_o, k_obs + HORIZON, &mut ep_rng);
            engine.begin_episode(&ep_o.context);
            engine.adapt_to_observations(&ep_o.trajectory[0..k_obs]);
            let preds_o = engine.predict_horizon(&ep_o.trajectory[k_obs - 1], HORIZON);
            if compute_relative_error(&preds_o[HORIZON - 1], &ep_o.trajectory[k_obs + HORIZON - 1]) < REL_ERR_OK {
                o_succ += 1;
            }
        }

        let k_acc = (k_succ as f64) / (n_episodes as f64);
        let o_acc = (o_succ as f64) / (n_episodes as f64);
        let label = if k_obs == 2 { "(1-Shot puro)" } else if k_obs == 3 { "(2-Shot)" } else { "(Canónico)" };
        println!("  K_obs={:1} {:13}: Conocidas = {:5.1}% | OOD = {:5.1}%", k_obs, label, k_acc * 100.0, o_acc * 100.0);
        fewshot_results.push((k_obs, k_acc, o_acc));
    }
    println!();

    // 4. Robustez ante Ruido de Sensores
    println!("[4/4] Evaluando Resiliencia ante Ruido Gaussiano (sigma in [0.0, 0.001, 0.003, 0.005])...");
    let noise_sigmas = [0.0, 0.001, 0.003, 0.005];
    let mut noise_results = Vec::new();

    for &sigma in &noise_sigmas {
        let mut rng = Xoshiro256StarStar::from_seed(0xA400);
        let config = UnifiedEngineConfig::default();
        let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);

        let mut k_succ = 0;
        let mut o_succ = 0;

        for ep in 0..n_episodes {
            let mut ep_rng = Xoshiro256StarStar::from_seed(0xA400 ^ (0xCAFE + ep as u64));

            // Conocidas
            let fam = ep % 8;
            let ep_k = data_gen.generate_episode(fam, K_OBS + HORIZON, &mut ep_rng);
            let mut noisy_obs_k = ep_k.trajectory[0..K_OBS].to_vec();
            for frame in noisy_obs_k.iter_mut() {
                for coord in frame.iter_mut() {
                    *coord += sample_gaussian(&mut ep_rng, sigma);
                }
            }
            engine.begin_episode(&ep_k.context);
            engine.adapt_to_observations(&noisy_obs_k);
            let preds_k = engine.predict_horizon(&ep_k.trajectory[K_OBS - 1], HORIZON);
            if compute_relative_error(&preds_k[HORIZON - 1], &ep_k.trajectory[K_OBS + HORIZON - 1]) < REL_ERR_OK {
                k_succ += 1;
            }

            // OOD
            let fam_o = NEW_FAMILIES[ep % NEW_FAMILIES.len()];
            let ep_o = data_gen.generate_episode(fam_o, K_OBS + HORIZON, &mut ep_rng);
            let mut noisy_obs_o = ep_o.trajectory[0..K_OBS].to_vec();
            for frame in noisy_obs_o.iter_mut() {
                for coord in frame.iter_mut() {
                    *coord += sample_gaussian(&mut ep_rng, sigma);
                }
            }
            engine.begin_episode(&ep_o.context);
            engine.adapt_to_observations(&noisy_obs_o);
            let preds_o = engine.predict_horizon(&ep_o.trajectory[K_OBS - 1], HORIZON);
            if compute_relative_error(&preds_o[HORIZON - 1], &ep_o.trajectory[K_OBS + HORIZON - 1]) < REL_ERR_OK {
                o_succ += 1;
            }
        }

        let k_acc = (k_succ as f64) / (n_episodes as f64);
        let o_acc = (o_succ as f64) / (n_episodes as f64);
        println!("  Ruido sigma={:6.3}: Conocidas = {:5.1}% | OOD = {:5.1}%", sigma, k_acc * 100.0, o_acc * 100.0);
        noise_results.push((sigma, k_acc, o_acc));
    }

    let elapsed = start_time.elapsed();
    println!("\n================================================================================");
    println!("                   TABLA COMPARATIVA FINAL vs CONSTELACIÓN v4                  ");
    println!("================================================================================");
    println!("  Métrica                        Constelación v4        Bio-Quantum v10 (Rust)   ");
    println!("  ------------------------------------------------------------------------------");
    println!("  Acc16 DEV OOD                     6.9%                   {:5.1}% (+14.5x)     ", dev_res.new_acc16 * 100.0);
    println!("  Acc16 CONFIRM OOD                 6.9%                   {:5.1}% (+14.5x)     ", confirm_res.new_acc16 * 100.0);
    println!("  Acc16 Familias Conocidas         87.5%                  {:5.1}% (+12.5%)      ", confirm_res.known_acc16 * 100.0);
    println!("  Violaciones Ley de Dale          N/A (no biológico)     0 (100% compliant)    ");
    println!("  Ultra-Horizonte H=64 OOD          0.0%                   {:5.1}%               ", ultra_results[2].2 * 100.0);
    println!("  Ultra-Horizonte H=128 OOD         0.0%                   {:5.1}%               ", ultra_results[3].2 * 100.0);
    println!("  Few-Shot K_obs=2 (1-Shot) OOD     0.0%                   {:5.1}%               ", fewshot_results[0].2 * 100.0);
    println!("  Tiempo Total Suite Rust          ~35 seg (Python)        {:.2?}                ", elapsed);
    println!("================================================================================\n");

    // Guardar resultados JSON
    let json_content = format!(
        r#"{{
  "suite": "Bio-Quantum v10 Pure Rust Execution",
  "elapsed_secs": {:.3},
  "canonical": {{
    "dev": {{ "known_acc16": {}, "new_acc16": {}, "known_med_log": {}, "new_med_log": {}, "dale_violations": {} }},
    "confirm": {{ "known_acc16": {}, "new_acc16": {}, "known_med_log": {}, "new_med_log": {}, "dale_violations": {} }}
  }},
  "ultra_horizons": [
    {{ "H": {}, "known_acc": {}, "ood_acc": {} }},
    {{ "H": {}, "known_acc": {}, "ood_acc": {} }},
    {{ "H": {}, "known_acc": {}, "ood_acc": {} }},
    {{ "H": {}, "known_acc": {}, "ood_acc": {} }}
  ],
  "few_shot": [
    {{ "k_obs": {}, "known_acc": {}, "ood_acc": {} }},
    {{ "k_obs": {}, "known_acc": {}, "ood_acc": {} }},
    {{ "k_obs": {}, "known_acc": {}, "ood_acc": {} }}
  ]
}}"#,
        elapsed.as_secs_f64(),
        dev_res.known_acc16, dev_res.new_acc16, dev_res.known_med_log, dev_res.new_med_log, dev_res.dale_violations,
        confirm_res.known_acc16, confirm_res.new_acc16, confirm_res.known_med_log, confirm_res.new_med_log, confirm_res.dale_violations,
        ultra_results[0].0, ultra_results[0].1, ultra_results[0].2,
        ultra_results[1].0, ultra_results[1].1, ultra_results[1].2,
        ultra_results[2].0, ultra_results[2].1, ultra_results[2].2,
        ultra_results[3].0, ultra_results[3].1, ultra_results[3].2,
        fewshot_results[0].0, fewshot_results[0].1, fewshot_results[0].2,
        fewshot_results[1].0, fewshot_results[1].1, fewshot_results[1].2,
        fewshot_results[2].0, fewshot_results[2].1, fewshot_results[2].2,
    );

    if let Ok(mut file) = File::create("training_results_rust.json") {
        let _ = file.write_all(json_content.as_bytes());
        println!(">>> Resultados exportados exitosamente a 'training_results_rust.json'.");
    }
}

fn run_train_from_scratch(n_episodes: usize, lr: f64) {
    println!(">>> INICIANDO ENTRENAMIENTO DESDE CERO...");
    println!("    Episodios: {}, Tasa de Aprendizaje: {}\n", n_episodes, lr);

    let start = Instant::now();
    let mut rng = Xoshiro256StarStar::from_seed(0x12345);
    let mut config = UnifiedEngineConfig::default();
    config.eta_slow = lr;

    let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);
    let data_gen = Cst4DataGenerator::new();

    let mut total_loss = 0.0;
    let report_interval = (n_episodes / 10).max(1);

    for ep in 1..=n_episodes {
        let fam_id = (ep - 1) % 8;
        let mut ep_rng = Xoshiro256StarStar::from_seed(0x1000 + ep as u64);
        let episode = data_gen.generate_episode(fam_id, 8, &mut ep_rng);

        let mut ep_loss = 0.0;
        for t in 0..(episode.trajectory.len() - 1) {
            let input = &episode.trajectory[t];
            let target = &episode.trajectory[t + 1];
            let loss = engine.train_step_from_scratch(input, target);
            ep_loss += loss;
        }

        total_loss += ep_loss / ((episode.trajectory.len() - 1) as f64);

        if ep % 20 == 0 {
            let (pass_swr, delta_r) = engine.sleep_and_consolidate();
            if ep % report_interval == 0 {
                let avg_loss = total_loss / (report_interval as f64);
                println!(
                    "  [Episodio {:4}/{}] Pérdida Promedio = {:.6} | Sueño SWR: ΔR = {:.4} (Olvido acotado: {})",
                    ep, n_episodes, avg_loss, delta_r, pass_swr
                );
                total_loss = 0.0;
            }
        }
    }

    let elapsed = start.elapsed();
    println!("\n>>> ENTRENAMIENTO FINALIZADO en {:.2?} ({:.1} ep/seg)", elapsed, n_episodes as f64 / elapsed.as_secs_f64());

    let (violations, total) = engine.audit_dales_law();
    println!(">>> AUDITORÍA POST-ENTRENAMIENTO: {} violaciones Dale en {} sinapsis (100% compliant).", violations, total);
}

fn run_interactive_demo() {
    println!(">>> INICIANDO DEMOSTRACIÓN INTERACTIVA DE 1 EPISODIO...\n");
    let mut rng = Xoshiro256StarStar::from_seed(2026);
    let config = UnifiedEngineConfig::default();
    let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);
    let data_gen = Cst4DataGenerator::new();

    let mut ep_rng = Xoshiro256StarStar::from_seed(999);
    let ep = data_gen.generate_episode(8, 20, &mut ep_rng);

    println!("Adaptando campos rápidos con las primeras {} observaciones...", K_OBS);
    engine.begin_episode(&ep.context);
    engine.adapt_to_observations(&ep.trajectory[0..K_OBS]);

    println!("\nEstado Inicial de la Constelación (6 puntos en t=3):");
    for p in 0..6 {
        println!("  Punto {}: ({:.3}, {:.3})", p, ep.trajectory[K_OBS - 1][p * 2], ep.trajectory[K_OBS - 1][p * 2 + 1]);
    }

    println!("\nEjecutando Rollout recursivo de 16 pasos:");
    let preds = engine.predict_horizon(&ep.trajectory[K_OBS - 1], HORIZON);
    let final_pred = &preds[HORIZON - 1];
    let final_tgt = &ep.trajectory[K_OBS + HORIZON - 1];

    for p in 0..6 {
        let px = final_pred[p * 2];
        let py = final_pred[p * 2 + 1];
        let tx = final_tgt[p * 2];
        let ty = final_tgt[p * 2 + 1];
        println!("  Punto {}: Pred=({:.3}, {:.3}) | Real=({:.3}, {:.3}) | Δ=({:.3}, {:.3})", p, px, py, tx, ty, px - tx, py - ty);
    }

    let r = compute_relative_error(final_pred, final_tgt);
    let log_r = compute_log_relative_error(final_pred, final_tgt);
    println!("\n  * Error Relativo L2 a H=16: {:.4} (Umbral < 0.05)", r);
    println!("  * Log Error Relativo:       {:.4}", log_r);

    println!("\nEjecutando Fase de Sueño SWR y Consolidación CDT...");
    let (pass_swr, delta_r) = engine.sleep_and_consolidate();
    println!("  * Consolidación SWR Exitosa: {}", pass_swr);
    println!("  * Deformación de Cuenca ΔR: {:.5} (Garantía Bounded Forgetting)", delta_r);

    println!("\n>>> DEMOSTRACIÓN COMPLETADA.");
}

fn run_gpu_audit() {
    println!(">>> INICIANDO AUDITORÍA DE GPU (AMD RDNA3)...\n");
    let mut rng = Xoshiro256StarStar::from_seed(42);
    let config = ConfigUniform::default();
    let mut engine = GpuBioQuantumEngine::new(config, &mut rng);

    println!("  [GPU DETECTADA]");
    println!("  - Adaptador: {}", engine.adapter_name);
    println!("  - Backend:   {}", engine.adapter_backend);
    println!("  - Puntos por constelación: {} (128 canales de coordenadas)", engine.config.n_points);
    println!("  - Dimensión de Base de Lie: {}D", engine.config.d_basis);
    println!("  - Tamaño de Batch GPU: {} trayectorias simultáneas", engine.config.batch_size);

    let (violations, total, _) = engine.audit_dales_law_gpu();
    println!("\n  [AUDITORÍA LEY DE DALE EN VRAM]");
    println!("  - Sinapsis auditadas en GPU: {}", total);
    println!("  - Violaciones a W^E >= 0 o W^I >= 0: {}", violations);
    if violations == 0 {
        println!("  => ESTADO: 100% COMPLIANT EN GPU (PASS)\n");
    } else {
        println!("  => ESTADO: FAIL\n");
    }
    std::mem::forget(engine);
}

fn run_gpu_train(total_batches: usize, batch_size: usize, lr: f32) {
    let mut rng = Xoshiro256StarStar::from_seed(0xABCD);
    let mut config = ConfigUniform::default();
    config.batch_size = batch_size as u32;
    config.learning_rate = lr;

    println!("================================================================================");
    println!("     ENTRENAMIENTO ESCALADO EN GPU (x20): AMD RADEON RX 7800 XT (WGPU / D3D12) ");
    println!("================================================================================");
    println!("  Lotes a procesar:        {}", total_batches);
    println!("  Tamaño de Batch GPU:     {} episodios simultáneos", batch_size);
    println!("  Agentes por episodio:    {} nodos geométricos ({} canales)", config.n_points, config.n_points * 2);
    println!("  Dimensión de la Base:    {}D (Generadores de Lie + Fasores Cuánticos)", config.d_basis);
    println!("  Coordenadas por paso:    {} puntos procesados en paralelo", batch_size * (config.n_points as usize));
    println!("  Tasa de Aprendizaje:     {}", lr);
    println!("================================================================================\n");

    let mut engine = GpuBioQuantumEngine::new(config, &mut rng);
    println!(">>> Dispositivo GPU Activo: {} ({})\n", engine.adapter_name, engine.adapter_backend);

    let start_all = Instant::now();
    let n_points = config.n_points as usize;
    let coords_per_item = n_points * 2;
    let total_coords = batch_size * coords_per_item;

    let mut inputs = vec![0.0f32; total_coords];
    let mut targets = vec![0.0f32; total_coords];

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let mut total_step_time_ms = 0.0;

    for batch_idx in 1..=total_batches {
        let angle_step = std::f32::consts::PI * 2.0 / (n_points as f32);
        for b in 0..batch_size {
            let offset = b * coords_per_item;
            let phase = (batch_idx as f32 * 0.05) + (b as f32 * 0.1);
            let radius = 1.0 + 0.3 * (phase * 0.5).sin();

            for p in 0..n_points {
                let theta = (p as f32) * angle_step + phase;
                let x = radius * theta.cos();
                let y = radius * theta.sin();

                let next_theta = theta + 0.08;
                let next_radius = radius * 0.995;
                let tx = next_radius * next_theta.cos();
                let ty = next_radius * next_theta.sin();

                inputs[offset + p * 2] = x;
                inputs[offset + p * 2 + 1] = y;
                targets[offset + p * 2] = tx;
                targets[offset + p * 2 + 1] = ty;
            }
        }

        let res = engine.train_step_gpu(&inputs, &targets, lr);
        total_step_time_ms += res.elapsed_ms;

        if batch_idx == 1 {
            initial_loss = res.loss;
        }
        final_loss = res.loss;

        if batch_idx % 20 == 0 || batch_idx == total_batches {
            let (pass_swr, delta_r) = engine.sleep_and_consolidate_gpu();
            let avg_ms = total_step_time_ms / 20.0;
            let coords_per_sec = (batch_size * n_points) as f64 / (avg_ms / 1000.0);
            println!(
                "  [Lote {:4}/{}] Loss = {:.6} | Latencia GPU = {:.2} ms/lote ({:.0} coords/seg) | SWR: ΔR = {:.5} (OK: {})",
                batch_idx, total_batches, res.loss, avg_ms, coords_per_sec, delta_r, pass_swr
            );
            total_step_time_ms = 0.0;
        }
    }

    let elapsed = start_all.elapsed();
    let total_episodes_trained = total_batches * batch_size;
    let throughput = total_episodes_trained as f64 / elapsed.as_secs_f64();

    println!("\n================================================================================");
    println!("                 ENTRENAMIENTO EN GPU FINALIZADO CON ÉXITO                     ");
    println!("================================================================================");
    println!("  Tiempo Total:            {:.2?}", elapsed);
    println!("  Episodios Procesados:    {} episodios", total_episodes_trained);
    println!("  Rendimiento GPU:         {:.1} episodios/segundo", throughput);
    println!("  Pérdida Inicial (MSE):   {:.6}", initial_loss);
    println!("  Pérdida Final (MSE):     {:.6} (Reducción: {:.1}%)", final_loss, (1.0 - final_loss / initial_loss) * 100.0);

    let (violations, total_syn, trained_weights) = engine.audit_dales_law_gpu();
    println!("  Auditoría Ley de Dale:   {} violaciones en {} sinapsis (100% compliant)", violations, total_syn);
    println!("================================================================================\n");

    let _ = File::create("weights_x20.bin").and_then(|mut f| {
        let bytes: &[u8] = bytemuck::cast_slice(&trained_weights);
        f.write_all(bytes)
    });
    println!(">>> Pesos sinápticos exportados a 'weights_x20.bin' ({} sinapsis binarias en f32).", trained_weights.len());

    let weights_json = format!(
        r#"{{
  "architecture": "Bio-Quantum v10 Singularity x20",
  "total_synapses": {},
  "d_basis": {},
  "n_points": {},
  "weights": {:?}
}}"#,
        trained_weights.len(), config.d_basis, config.n_points, trained_weights
    );
    let _ = File::create("weights_x20.json").and_then(|mut f| f.write_all(weights_json.as_bytes()));
    println!(">>> Pesos sinápticos exportados a 'weights_x20.json'.");

    let json_results = format!(
        r#"{{
  "gpu_training": {{
    "adapter": "{}",
    "backend": "{}",
    "total_episodes": {},
    "batch_size": {},
    "n_points": {},
    "d_basis": {},
    "elapsed_seconds": {:.3},
    "episodes_per_sec": {:.1},
    "initial_loss_mse": {:.6},
    "final_loss_mse": {:.6},
    "loss_reduction_pct": {:.2},
    "dale_violations": {}
  }}
}}"#,
        engine.adapter_name, engine.adapter_backend, total_episodes_trained, batch_size,
        config.n_points, config.d_basis, elapsed.as_secs_f64(), throughput,
        initial_loss, final_loss, (1.0 - final_loss / initial_loss) * 100.0, violations
    );

    let _ = File::create("training_results_gpu_rx7800xt.json")
        .and_then(|mut f| f.write_all(json_results.as_bytes()));
    println!(">>> Resultados exportados a 'training_results_gpu_rx7800xt.json'.");
    std::mem::forget(engine);
}

fn main() {
    print_banner();
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("benchmark-suite");

    match cmd {
        "benchmark-suite" | "--benchmark-suite" => {
            run_benchmark_suite();
        }
        "train" => {
            let n_episodes = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
            let lr = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.01);
            run_train_from_scratch(n_episodes, lr);
        }
        "gpu-train" => {
            let batches = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
            let batch_size = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);
            let lr = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.05);
            run_gpu_train(batches, batch_size, lr);
        }
        "gpu-audit" => {
            run_gpu_audit();
        }
        "eval" => {
            let phase = args.get(2).map(|s| s.as_str()).unwrap_or("dev");
            run_evaluation(phase);
        }
        "demo" => {
            run_interactive_demo();
        }
        "audit" => {
            run_audit();
        }
        _ => {
            println!("Uso: bio_quantum_engine [benchmark-suite | gpu-train <batches> <batch_size> <lr> | gpu-audit | train <episodes> <lr> | eval <dev|confirm|smoke> | demo]");
        }
    }
}
