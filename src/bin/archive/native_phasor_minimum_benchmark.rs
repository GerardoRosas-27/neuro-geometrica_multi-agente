use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;
use std::io;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "nodes,edges,elapsed_ms,iterations,energy_evaluations,relative_gap,coherence,residual"
    );
    for nodes in [128, 640, 4_096, 16_384] {
        let mut engine = NativePhasorThermodynamicEngine::from_cdt_config(
            NativeThermoCdtConfig {
                slices: 1,
                nodes_per_slice: nodes,
                spatial_degree: 2,
                temporal_degree: 1,
                temperature: 0.0,
                seed: 91_003 + nodes as u64,
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
        )?;
        let exact = engine
            .analytic_unfrustrated_minimum()
            .ok_or_else(|| io::Error::other("el fixture dejó de tener mínimo analítico"))?;
        let started = Instant::now();
        let result = engine.minimize_free_energy(NativePhasorMinimizerConfig {
            max_iterations: 100,
            residual_tolerance: 2.0e-4,
            ..NativePhasorMinimizerConfig::default()
        });
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let relative_gap =
            (result.final_report.free_energy - exact).abs() as f64 / (exact.abs() as f64).max(1.0);
        println!(
            "{},{},{:.3},{},{},{:.3e},{:.8},{:.3e}",
            nodes,
            result.final_report.edges,
            elapsed_ms,
            result.iterations,
            result.energy_evaluations,
            relative_gap,
            result.final_report.phase_coherence,
            result.final_report.gradient_residual
        );
        if !result.converged || relative_gap > 2.0e-5 {
            return Err(io::Error::other(format!(
                "no alcanzó el mínimo para {nodes} nodos: {result:?}"
            ))
            .into());
        }
    }
    Ok(())
}
