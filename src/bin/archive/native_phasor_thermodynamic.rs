use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorMinimizerConfig, NativePhasorThermodynamicEngine,
    DEFAULT_PHASOR_NODES_PER_SLICE, DEFAULT_PHASOR_STARTUP_SLICES,
};
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cdt_config = NativeThermoCdtConfig {
        slices: DEFAULT_PHASOR_STARTUP_SLICES,
        nodes_per_slice: DEFAULT_PHASOR_NODES_PER_SLICE,
        spatial_degree: 4,
        temporal_degree: 3,
        temperature: 0.65,
        seed: 87_301,
        ..NativeThermoCdtConfig::default()
    };
    let mut engine = NativePhasorThermodynamicEngine::from_cdt_config(
        cdt_config,
        NativePhasorConfig {
            coupling_strength: 0.9,
            radial_strength: 1.2,
            dt: 0.015,
            ..NativePhasorConfig::default()
        },
    )?;

    let first_pattern = (0..24).collect::<Vec<_>>();
    let second_pattern = (160..184).collect::<Vec<_>>();
    engine.inject_pattern(&first_pattern, 1.5, 0.0);
    engine.inject_pattern(&second_pattern, 1.5, std::f32::consts::PI);

    let initial = engine.report();
    println!("Motor termodinámico fasorial independiente");
    print_report("inicial", initial);

    engine.set_temperature_scale(0.0);
    let minimization = engine.minimize_free_energy(NativePhasorMinimizerConfig {
        max_iterations: 600,
        residual_tolerance: 5.0e-3,
        ..NativePhasorMinimizerConfig::default()
    });
    print_report("final", minimization.final_report);
    println!(
        "iteraciones={} evaluaciones_F={} rechazados={} warm_start={} convergió={}",
        minimization.iterations,
        minimization.energy_evaluations,
        minimization.rejected_steps,
        minimization.warm_start_applied,
        minimization.converged
    );
    println!("fasores dominantes (nodo, amplitud, fase):");
    for (node, amplitude, phase) in engine.dominant_nodes(8) {
        println!("  {node:4}  {amplitude:.5}  {phase:.5}");
    }

    Ok(())
}

fn print_report(
    label: &str,
    report: cdt_rqm_epr::native_phasor_thermodynamic_engine::NativePhasorReport,
) {
    println!(
        "{label}: F={:.6} U={:.6} acoplamiento={:.6} entropía={:.6} \
         residuo={:.6} coherencia={:.6} amplitud={:.6} activos={}/{}",
        report.free_energy,
        report.internal_energy,
        report.coupling_energy,
        report.entropy,
        report.gradient_residual,
        report.phase_coherence,
        report.mean_amplitude,
        report.active_phasors,
        report.nodes
    );
}
