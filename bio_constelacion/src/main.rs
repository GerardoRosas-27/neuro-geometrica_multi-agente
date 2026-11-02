#![allow(clippy::needless_range_loop)]
#![allow(clippy::collapsible_match)]

//! Punto de Entrada CLI para la Arquitectura Híbrida Bio-Constelación
//!
//! Subcomandos:
//! - `paso0 [--seed <val>]`: Verificaciones de integridad, Ley de Dale, radio espectral y determinismo.
//! - `smoke [--seed <val>] [--episodes <n>]`: Evaluación rápida y reporte comparativo vs Constelación v4.
//! - `run --phase <dev|confirm|smoke> [--seed <val>] [--out <dir>]`: Evaluación completa y persistencia JSON.

use bio_constelacion::hybrid_brain::{BioConstelacion, HybridConfig};
use bio_constelacion::protocol_cst4::{
    evaluate_hybrid_on_protocol, CST4_FROZEN_CONFIRM, CST4_FROZEN_DEV,
};
use bio_constelacion::rng::Xoshiro256StarStar;
use std::env;
use std::fs;
use std::path::Path;

fn print_help() {
    println!("=== BIO-CONSTELACIÓN: Arquitectura Híbrida Neurobiológica y Geométrica ===");
    println!("Uso: cargo run -- [SUBCOMANDO] [OPCIONES]");
    println!();
    println!("Subcomandos:");
    println!("  paso0                Verificación de integridad, Ley de Dale y estabilidad.");
    println!(
        "  smoke                Evaluación rápida (32 episodios) y reporte vs Constelación v4."
    );
    println!("  run                  Ejecuta evaluación formal por fase.");
    println!();
    println!("Opciones:");
    println!("  --seed <HEX/DEC>     Semilla aleatoria (ej. 0xA400, 0xC400, 42).");
    println!("  --phase <NAME>       Fase: dev, confirm, smoke (por defecto: smoke).");
    println!("  --episodes <N>       Número de episodios de prueba (por defecto: 32).");
    println!("  --out <DIR>          Directorio para guardar artefactos JSON.");
    println!("  --ablation <ARM>     Ablaciones: all, no_fast_btsp, no_hippo, no_sleep, no_prior.");
}

fn parse_u64(arg: &str) -> u64 {
    if let Some(stripped) = arg.strip_prefix("0x").or_else(|| arg.strip_prefix("0X")) {
        u64::from_str_radix(stripped, 16).unwrap_or(0xC400)
    } else {
        arg.parse::<u64>().unwrap_or(0xC400)
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_help();
        return;
    }

    let subcmd = args[1].as_str();

    let mut seed: u64 = 0xC400;
    let mut phase = String::from("smoke");
    let mut episodes: usize = 32;
    let mut out_dir = String::from("artifacts/bio_constelacion");
    let mut ablation_arm = String::from("hybrid");

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                if i + 1 < args.len() {
                    seed = parse_u64(&args[i + 1]);
                    i += 1;
                }
            }
            "--phase" => {
                if i + 1 < args.len() {
                    phase = args[i + 1].clone();
                    i += 1;
                }
            }
            "--episodes" => {
                if i + 1 < args.len() {
                    episodes = args[i + 1].parse().unwrap_or(32);
                    i += 1;
                }
            }
            "--out" => {
                if i + 1 < args.len() {
                    out_dir = args[i + 1].clone();
                    i += 1;
                }
            }
            "--ablation" => {
                if i + 1 < args.len() {
                    ablation_arm = args[i + 1].clone();
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    match subcmd {
        "paso0" => run_paso0(seed),
        "smoke" => run_smoke(seed, episodes),
        "run" => run_formal(phase, seed, episodes, out_dir, ablation_arm),
        _ => {
            eprintln!("Subcomando desconocido: {}", subcmd);
            print_help();
        }
    }
}

fn configure_brain(arm: &str, seed: u64) -> BioConstelacion {
    let mut rng = Xoshiro256StarStar::new(seed);
    let mut config = HybridConfig::default();

    match arm {
        "no_fast_btsp" => config.enable_fast_btsp = false,
        "no_hippo" => config.enable_hippocampus = false,
        "no_sleep" => config.enable_sleep = false,
        "no_prior" => config.enable_context_prior = false,
        _ => {} // default: full hybrid
    }

    BioConstelacion::new(config, &mut rng)
}

fn run_paso0(seed: u64) {
    println!("\n=======================================================");
    println!("   PASO 0: VERIFICACIÓN DE INTEGRIDAD BIO-CONSTELACIÓN  ");
    println!("=======================================================");
    println!("Semilla: 0x{:X}", seed);

    let brain = configure_brain("hybrid", seed);

    // 1. Auditoría estricta de la Ley de Dale en todo el campo sináptico
    let (dale_violations, total_syn) = brain.audit_dale_law();
    println!(
        "[1] Ley de Dale en Sinapsis (Lentas W^P + Rápidas W^F): {} violaciones de {} pesos (100% compliant)",
        dale_violations, total_syn
    );
    assert_eq!(dale_violations, 0, "Fallo: Violación de la Ley de Dale");

    // 2. Auditoría Ley de Dale en Campo Rápido BTSP
    let mut btsp_neg = 0;
    for &w in brain
        .fast_btsp
        .wf_exc
        .iter()
        .chain(brain.fast_btsp.wf_inh.iter())
    {
        if w < 0.0 {
            btsp_neg += 1;
        }
    }
    println!(
        "[2] Ley de Dale en Campo Rápido BTSP: {} violaciones de {} pesos (100% compliant)",
        btsp_neg,
        brain.fast_btsp.wf_exc.len() + brain.fast_btsp.wf_inh.len()
    );
    assert_eq!(
        btsp_neg, 0,
        "Fallo: Pesos negativos en campo BTSP excitatorio/inhibitorio"
    );

    // 3. Radio Espectral del Operador Koopman
    let rho = brain.koopman_col.spectral_radius();
    println!(
        "[3] Radio Espectral del Bloque Koopman: rho = {:.4} (Requisito: rho <= 1.0)",
        rho
    );
    assert!(rho <= 1.0, "Fallo: Radio espectral supera 1.0 (inestable)");

    // 4. Dispersión k-WTA del Giro Dentado
    let test_context = [0.25; 10];
    let dg_code = brain.hippocampus.encode_dg(&test_context, 16);
    let active_dg = dg_code.iter().filter(|&&v| v > 1e-6).count();
    println!(
        "[4] Giro Dentado k-WTA Sparsity: {} unidades activas de {} (Exactamente k=16)",
        active_dg, brain.hippocampus.dg_dim
    );
    assert_eq!(active_dg, 16, "Fallo: Sparsity de DG no cumple k=16");

    // 5. Determinismo del Generador RNG
    let mut rng1 = Xoshiro256StarStar::new(seed);
    let mut rng2 = Xoshiro256StarStar::new(seed);
    let val1: Vec<f64> = (0..10).map(|_| rng1.gen_f64()).collect();
    let val2: Vec<f64> = (0..10).map(|_| rng2.gen_f64()).collect();
    println!("[5] Determinismo Pseudoaleatorio: Coincidencia 10/10 entre streams independientes");
    assert_eq!(
        val1, val2,
        "Fallo: Generador no determinista para semilla fija"
    );

    println!("\n>>> RESULTADO PASO 0: TODOS LOS CHEQUEOS BIOLÓGICOS Y GEOMÉTRICOS APROBADOS <<<\n");
}

fn run_smoke(seed: u64, episodes: usize) {
    println!("\n=======================================================");
    println!("     SMOKE TEST: BIO-CONSTELACIÓN VS CONSTELACIÓN V4    ");
    println!("=======================================================");
    println!("Semilla: 0x{:X} | Episodios: {}", seed, episodes);

    let mut brain = configure_brain("hybrid", seed);
    let metrics = evaluate_hybrid_on_protocol(&mut brain, seed, episodes);

    println!("\n--- RESULTADOS COMPARATIVOS DIRECTOS ---");
    println!(
        "{:<34} | {:<12} | {:<10} | {:<11} | {:<9} | {:<10} | {:<12}",
        "Arquitectura / Variante",
        "Conocidas ln",
        "Conocidas %",
        "Nuevas ln",
        "Nuevas %",
        "E33 Pass",
        "Ley de Dale"
    );
    println!("{:-<110}", "");

    // 1. Constelación v4 Congelada DEV
    println!(
        "{:<34} | {:<12.4} | {:<9.1}% | {:<11.4} | {:<8.1}% | {:<10} | {:<12}",
        CST4_FROZEN_DEV.name,
        CST4_FROZEN_DEV.known_med_log,
        CST4_FROZEN_DEV.known_acc16 * 100.0,
        CST4_FROZEN_DEV.new_med_log,
        CST4_FROZEN_DEV.new_acc16 * 100.0,
        CST4_FROZEN_DEV.e33_pass_ratio,
        "VIOLADA"
    );

    // 2. Constelación v4 Congelada CONFIRM
    println!(
        "{:<34} | {:<12.4} | {:<9.1}% | {:<11.4} | {:<8.1}% | {:<10} | {:<12}",
        CST4_FROZEN_CONFIRM.name,
        CST4_FROZEN_CONFIRM.known_med_log,
        CST4_FROZEN_CONFIRM.known_acc16 * 100.0,
        CST4_FROZEN_CONFIRM.new_med_log,
        CST4_FROZEN_CONFIRM.new_acc16 * 100.0,
        CST4_FROZEN_CONFIRM.e33_pass_ratio,
        "VIOLADA"
    );

    // 3. Bio-Constelación Híbrida Actual
    let e33_str = if metrics.e33_pass {
        "APROBADO"
    } else {
        "REVISAR"
    };
    let dale_str = if metrics.dale_violations == 0 {
        "CUMPLE 100%"
    } else {
        "VIOLADA"
    };

    println!(
        "{:<34} | {:<12.4} | {:<9.1}% | {:<11.4} | {:<8.1}% | {:<10} | {:<12}",
        "Bio-Constelación (Híbrido)",
        metrics.known_med_log,
        metrics.known_acc16 * 100.0,
        metrics.new_med_log,
        metrics.new_acc16 * 100.0,
        e33_str,
        dale_str
    );

    println!("{:-<110}", "");
    println!("\nDetalle E33 Precisión por Horizonte en Bio-Constelación:");
    println!(
        "h=1: {:.2}% | h=2: {:.2}% | h=4: {:.2}% | h=8: {:.2}% | h=16: {:.2}%",
        metrics.e33_acc_per_h[0] * 100.0,
        metrics.e33_acc_per_h[1] * 100.0,
        metrics.e33_acc_per_h[2] * 100.0,
        metrics.e33_acc_per_h[3] * 100.0,
        metrics.e33_acc_per_h[4] * 100.0,
    );
    println!();
}

fn run_formal(phase: String, seed: u64, episodes: usize, out_dir: String, ablation_arm: String) {
    println!("\n=== EJECUCIÓN FORMAL BIO-CONSTELACIÓN ===");
    println!(
        "Fase: {} | Semilla: 0x{:X} | Brazo: {} | Episodios: {}",
        phase, seed, ablation_arm, episodes
    );

    let mut brain = configure_brain(&ablation_arm, seed);
    let metrics = evaluate_hybrid_on_protocol(&mut brain, seed, episodes);

    let json_content = format!(
        r#"{{
  "seed": "0x{:X}",
  "phase": "{}",
  "ablation_arm": "{}",
  "episodes": {},
  "known_med_log": {:.6},
  "known_acc16": {:.4},
  "new_med_log": {:.6},
  "new_acc16": {:.4},
  "e33_pass": {},
  "e33_acc_per_h": [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}],
  "dale_violations": {},
  "total_synapses": {},
  "cst4_frozen_dev_known_med_log": {:.6},
  "cst4_frozen_dev_known_acc16": {:.4},
  "cst4_frozen_dev_new_med_log": {:.6},
  "cst4_frozen_dev_new_acc16": {:.4}
}}"#,
        metrics.seed,
        phase,
        ablation_arm,
        episodes,
        metrics.known_med_log,
        metrics.known_acc16,
        metrics.new_med_log,
        metrics.new_acc16,
        metrics.e33_pass,
        metrics.e33_acc_per_h[0],
        metrics.e33_acc_per_h[1],
        metrics.e33_acc_per_h[2],
        metrics.e33_acc_per_h[3],
        metrics.e33_acc_per_h[4],
        metrics.dale_violations,
        metrics.total_synapses,
        CST4_FROZEN_DEV.known_med_log,
        CST4_FROZEN_DEV.known_acc16,
        CST4_FROZEN_DEV.new_med_log,
        CST4_FROZEN_DEV.new_acc16
    );

    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!(
            "Advertencia: no se pudo crear el directorio {}: {}",
            out_dir, e
        );
    }

    let out_file = Path::new(&out_dir).join(format!("metrics_{:X}_{}.json", seed, ablation_arm));
    let done_file = Path::new(&out_dir).join(format!("{:X}_{}.done", seed, ablation_arm));

    if let Err(e) = fs::write(&out_file, &json_content) {
        eprintln!(
            "Error escribiendo archivo JSON {}: {}",
            out_file.display(),
            e
        );
    } else {
        println!("Guardado: {}", out_file.display());
    }

    let _ = fs::write(done_file, "DONE\n");
}
