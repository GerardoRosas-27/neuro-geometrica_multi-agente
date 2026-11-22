//! Binario de ejecución y generación del informe formal de Benchmarks FEP-MARCUS.
//!
//! Evalúa las 4 hipótesis fundamentales de la arquitectura basada 100% en:
//! 1. Maximización de la Energía Libre (ELBO).
//! 2. Reducción rigurosa de la Sorpresa Estructurada.
//! 3. Inferencia Activa y Selección Causal por G(pi).
//! 4. Generalización OOD acotada por invariantes ontológicos.

use marcus_neurosymbolic::benchmarks::{run_all_fep_benchmarks, run_all_fep_benchmarks_aura_v3_genesis};
use std::fs;
use std::path::Path;

fn main() {
    println!("============================================================");
    println!(" FEP-MARCUS v3 vs AURA v3 Genesis + FEP-MARCUS v3");
    println!(" 100% Basada en Maximizar la Energía Libre y Reducir la Sorpresa");
    println!(" Síntesis con Gary Marcus + 4 Subrutinas de ADN AURA v3 Genesis");
    println!("============================================================\n");

    println!("[1/5] Ejecutando BM-6: Minimización de Energía Libre Variacional y Maximización de ELBO...");
    println!("[2/5] Ejecutando BM-7: Reducción Multimodal de Sorpresa Estructurada (Spelke + Causal)...");
    println!("[3/5] Ejecutando BM-8: Inferencia Activa y Selección Causal por G(pi)...");
    println!("[4/5] Ejecutando BM-9: Generalización OOD con Cota Estricta de Energía Libre...");
    println!("[5/5] Ejecutando BM-10: Calibración Metacognitiva, Veto OOD y Esfuerzo Racional...\n");

    let report_fep = run_all_fep_benchmarks();
    let report_aura = run_all_fep_benchmarks_aura_v3_genesis();
    let table = format!(
        "{}\n{}\n",
        report_fep.render_markdown_table(),
        report_fep.render_versus_markdown_table(&report_aura)
    );
    println!("{}", table);

    // Guardar en docs/marcus_neurosymbolic/RESULTADOS_FEP_ACTIVE_INFERENCE.md
    let target_path = Path::new("../docs/marcus_neurosymbolic/RESULTADOS_FEP_ACTIVE_INFERENCE.md");
    if let Some(parent) = target_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::write(target_path, &table) {
        Ok(_) => println!("[INFO] Informe guardado exitosamente en: {:?}", target_path),
        Err(e) => eprintln!("[WARN] No se pudo escribir el informe en {:?}: {}", target_path, e),
    }

    // Guardar JSON en artifacts/marcus_neurosymbolic/fep_active_inference_results.json
    let json_path = Path::new("../artifacts/marcus_neurosymbolic/fep_active_inference_results.json");
    if let Some(parent) = json_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let combined = serde_json::json!({
        "fep_marcus_v3": report_fep,
        "aura_v3_genesis_fep_marcus_v3": report_aura,
    });
    if let Ok(json_str) = serde_json::to_string_pretty(&combined) {
        match fs::write(json_path, json_str) {
            Ok(_) => println!("[INFO] Métricas JSON guardadas exitosamente en: {:?}", json_path),
            Err(e) => eprintln!("[WARN] No se pudo escribir JSON en {:?}: {}", json_path, e),
        }
    }

    if report_fep.all_fep_passed && report_aura.all_fep_passed {
        println!("\n>>> VEREDICTO FINAL: TODAS LAS HIPÓTESIS H-FEP1 A H-FEP5 CONFIRMADAS AL 100% EN AMBAS ARQUITECTURAS <<<");
    } else {
        eprintln!("\n>>> ERROR: UNA O MÁS HIPÓTESIS FEP FALLARON <<<");
        std::process::exit(1);
    }
}
