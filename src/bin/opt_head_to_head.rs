//! Benchmark pareado legacy vs optimizado para `minimize_free_energy`.
//! Misma fixture en ambas versiones: grafos CDT con semilla fija, dos patrones
//! en contrafase para frustrar el arranque y descenso Armijo completo.

use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use std::time::Instant;

const REPS: usize = 5;

fn run_case(nodes: usize, degree: usize, max_iterations: usize, tolerance: f32) -> String {
    let mut timings = Vec::with_capacity(REPS);
    let mut iterations = 0usize;
    let mut evaluations = 0usize;
    let mut final_energy = 0.0f64;
    let mut converged_count = 0usize;

    for rep in 0..REPS {
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
        engine.set_temperature_scale(0.0);

        let started = Instant::now();
        let result = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations,
            residual_tolerance: tolerance,
            ..NativePhasorMinimizerConfig::default()
        });
        timings.push(started.elapsed().as_secs_f64() * 1_000.0);
        if rep == 0 {
            iterations = result.iterations;
            evaluations = result.energy_evaluations;
            final_energy = f64::from(result.final_report.free_energy);
        }
        converged_count += usize::from(result.converged);
    }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = timings[timings.len() / 2];
    let min = timings[0];
    format!(
        "{nodes},{degree},{median:.3},{min:.3},{iterations},{evaluations},{final_energy:.6},{converged_count}/{REPS}"
    )
}

fn main() {
    println!("nodes,degree,median_ms,min_ms,iterations,energy_evaluations,final_F,converged");
    for (nodes, degree) in [(1_024usize, 2usize), (8_192, 2), (8_192, 4), (32_768, 2)] {
        println!("{}", run_case(nodes, degree, 300, 1.0e-3));
    }
}
