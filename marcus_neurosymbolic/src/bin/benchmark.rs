//! Binario de ejecución y generación del informe formal de Benchmarks MARCUS-I.

use marcus_neurosymbolic::benchmarks::run_all_benchmarks;
use std::fs;
use std::path::Path;

fn main() {
    println!("============================================================");
    println!(" MARCUS-I: Arnés de Evaluación Experimental Neuro-Simbólico ");
    println!(" Validación Falsable de las Hipótesis de Gary Marcus         ");
    println!("============================================================\n");

    println!("[1/5] Ejecutando BM-1: Generalización Zero-Shot OOD en Variables...");
    println!("[2/5] Ejecutando BM-2: Sistematicidad e Invarianza Composicional...");
    println!("[3/5] Ejecutando BM-3: Física Núcleo y Permanencia Spelke...");
    println!("[4/5] Ejecutando BM-4: Inferencia Causal frente a Correlación Espuria...");
    println!("[5/5] Ejecutando BM-5: Vigilancia Epistémica y Supresión de Alucinaciones...\n");

    let report = run_all_benchmarks();
    let table = report.render_markdown_table();
    println!("{}", table);

    // Guardar en docs/marcus_neurosymbolic/RESULTADOS_BENCHMARK.md
    let target_path = Path::new("../docs/marcus_neurosymbolic/RESULTADOS_BENCHMARK.md");
    if let Some(parent) = target_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::write(target_path, &table) {
        Ok(_) => println!("[INFO] Informe guardado exitosamente en: {:?}", target_path),
        Err(e) => eprintln!("[WARN] No se pudo escribir el informe en {:?}: {}", target_path, e),
    }

    if report.all_passed {
        println!("\n>>> VEREDICTO FINAL: TODAS LAS HIPÓTESIS H1-H5 CONFIRMADAS AL 100% <<<");
    } else {
        eprintln!("\n>>> ERROR: UNA O MÁS HIPÓTESIS FALLARON <<<");
        std::process::exit(1);
    }
}
