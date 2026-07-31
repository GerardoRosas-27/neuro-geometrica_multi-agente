//! Benchmark pareado de la ruta fría vs workspace reutilizado de
//! `minimize_free_energy`.
//!
//! Ambas rutas restauran exactamente el mismo campo fasorial antes de cada
//! medición. La ruta fría libera los tres buffers antes de inferir; la warm
//! conserva su capacidad. Así se aísla el beneficio real del pool sin comparar
//! estados ya convergidos ni atribuirle optimizaciones de commits anteriores.

use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use std::time::Instant;

const REPS: usize = 9;

fn engine(nodes: usize, degree: usize) -> NativePhasorThermodynamicEngine {
    let mut engine = NativePhasorThermodynamicEngine::from_cdt_config(
        NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: nodes,
            spatial_degree: degree,
            temporal_degree: 1,
            temperature: 0.0,
            seed: 55_121 + nodes as u64 + degree as u64 * 7,
            ..NativeThermoCdtConfig::default()
        },
        NativePhasorConfig {
            coupling_strength: 1.0,
            radial_strength: 1.0,
            target_amplitude: 1.0,
            confinement: 0.02,
            entropy_weight: 0.0,
            temperature_scale: 0.0,
            noise_scale: 0.0,
            ..NativePhasorConfig::default()
        },
    )
    .expect("fixture del benchmark");
    engine.inject_pattern(&(0..24).collect::<Vec<_>>(), 1.5, 0.0);
    engine.inject_pattern(
        &(nodes / 2..nodes / 2 + 24).collect::<Vec<_>>(),
        1.5,
        std::f32::consts::PI,
    );
    engine
}

fn measure(
    engine: &mut NativePhasorThermodynamicEngine,
    config: NativePhasorMinimizerConfig,
) -> (
    f64,
    cdt_rqm_epr::native_phasor_thermodynamic_engine::NativePhasorMinimizationReport,
) {
    let started = Instant::now();
    let report = engine.minimize_free_energy(config);
    (started.elapsed().as_secs_f64() * 1_000.0, report)
}

fn run_case(nodes: usize, degree: usize, max_iterations: usize, tolerance: f32) -> String {
    let mut cold = engine(nodes, degree);
    let mut warm = cold.clone();
    let initial = cold.phasors.clone();
    let config = NativePhasorMinimizerConfig {
        max_iterations,
        residual_tolerance: tolerance,
        ..NativePhasorMinimizerConfig::default()
    };

    // Precalienta exclusivamente el workspace; el estado se restaura después.
    warm.minimize_free_energy(config);
    warm.phasors.copy_from_slice(&initial);

    let mut cold_timings = Vec::with_capacity(REPS);
    let mut warm_timings = Vec::with_capacity(REPS);
    let mut energy_drift = 0.0_f32;
    let mut iteration_mismatches = 0;
    let mut workspace_bytes = 0;
    for rep in 0..REPS {
        cold.phasors.copy_from_slice(&initial);
        cold.clear_minimizer_workspace();
        warm.phasors.copy_from_slice(&initial);

        let ((cold_ms, cold_report), (warm_ms, warm_report)) = if rep % 2 == 0 {
            let cold_result = measure(&mut cold, config);
            let warm_result = measure(&mut warm, config);
            (cold_result, warm_result)
        } else {
            let warm_result = measure(&mut warm, config);
            let cold_result = measure(&mut cold, config);
            (cold_result, warm_result)
        };
        cold_timings.push(cold_ms);
        warm_timings.push(warm_ms);
        workspace_bytes = warm.minimizer_workspace_capacity_bytes();
        energy_drift = energy_drift.max(
            (cold_report.final_report.free_energy - warm_report.final_report.free_energy).abs(),
        );
        iteration_mismatches += usize::from(cold_report.iterations != warm_report.iterations);
    }

    cold_timings.sort_by(f64::total_cmp);
    warm_timings.sort_by(f64::total_cmp);
    let cold_median = cold_timings[cold_timings.len() / 2];
    let warm_median = warm_timings[warm_timings.len() / 2];
    format!(
        "{nodes},{degree},{cold_median:.3},{warm_median:.3},{:.4},{workspace_bytes},\
         {energy_drift:.3e},{iteration_mismatches}",
        cold_median / warm_median
    )
}

fn main() {
    println!(
        "nodes,degree,cold_ms,warm_pool_ms,pool_speedup,workspace_bytes,energy_drift,\
         iteration_mismatches"
    );
    for (nodes, degree) in [(1_024usize, 2usize), (8_192, 2), (8_192, 4), (32_768, 2)] {
        println!("{}", run_case(nodes, degree, 300, 1.0e-3));
    }
}
