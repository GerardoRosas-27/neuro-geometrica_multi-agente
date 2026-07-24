use cdt_rqm_epr::thermodynamic_attractor_comparison::{
    compare_thermodynamic_attractors, AttractorArchitectureMetrics, AttractorComparisonConfig,
};

fn main() {
    let report = compare_thermodynamic_attractors(AttractorComparisonConfig {
        nodes: 256,
        patterns: 8,
        trials: 32,
        corruption_fraction: 0.25,
        old_max_steps: 2_500,
        phasor_max_iterations: 800,
        ..AttractorComparisonConfig::default()
    });

    println!("Comparación pareada: CDT térmico puro vs motor fasorial");
    println!("capas_legacy_excluidas=RQM,EPR,cognición,sueño,plasticidad");
    println!(
        "dataset_checksum={} nodos={} patrones={} aristas={} ensayos={}",
        report.dataset_checksum, report.nodes, report.patterns, report.trained_edges, report.trials
    );
    print_metrics("CDT anterior", report.old_cdt);
    print_metrics("Fasorial", report.phasor);
    println!(
        "speedup_pared={:.3}x reducción_iteraciones={:.2}%",
        report.wall_time_speedup(),
        100.0 * report.iteration_reduction()
    );

    let winner = if report.phasor.attractor_success_rate()
        > report.old_cdt.attractor_success_rate() + 1.0e-9
    {
        "fasorial: mayor recuperación de atractores"
    } else if report.old_cdt.attractor_success_rate()
        > report.phasor.attractor_success_rate() + 1.0e-9
    {
        "CDT anterior: mayor recuperación de atractores"
    } else if report.phasor.mean_final_common_energy() + 1.0e-9
        < report.old_cdt.mean_final_common_energy()
    {
        "fasorial: misma recuperación y menor energía común"
    } else if report.old_cdt.mean_final_common_energy() + 1.0e-9
        < report.phasor.mean_final_common_energy()
    {
        "CDT anterior: misma recuperación y menor energía común"
    } else if report.phasor.elapsed < report.old_cdt.elapsed {
        "fasorial: misma calidad y menor tiempo"
    } else {
        "CDT anterior: misma calidad y menor tiempo"
    };
    println!("resultado={winner}");
}

fn print_metrics(label: &str, metrics: AttractorArchitectureMetrics) {
    println!(
        "{label}: éxito={:.2}% convergencia={:.2}% exactitud={:.5} \
         F_común={:.6}->{:.6} residuo={:.3e} iteraciones={:.2} \
         evaluaciones_F={:.2} tiempo_ms={:.3}",
        100.0 * metrics.attractor_success_rate(),
        100.0 * metrics.convergence_rate(),
        metrics.mean_target_accuracy(),
        metrics.mean_initial_common_energy(),
        metrics.mean_final_common_energy(),
        metrics.mean_phase_residual(),
        metrics.mean_iterations(),
        metrics.mean_energy_evaluations(),
        metrics.elapsed.as_secs_f64() * 1_000.0,
    );
}
