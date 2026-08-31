use cdt_rqm_epr::native_hybrid_phasor_cdt_engine::{
    NativeHybridConfig, NativeHybridPhasorCdtEngine, NativePhasorCue,
};
use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    NativePhasorConfig, NativePhasorThermodynamicEngine,
};
use cdt_rqm_epr::native_rng::signed_unit;
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use num_complex::Complex32;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const NODES: usize = 256;
    const TRIALS: usize = 64;
    let core = NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: 1,
        nodes_per_slice: NODES,
        spatial_degree: 4,
        temporal_degree: 1,
        temperature: 0.0,
        seed: 81_771,
        ..NativeThermoCdtConfig::default()
    });
    let phasor_config = NativePhasorConfig {
        temperature_scale: 0.0,
        noise_scale: 0.0,
        entropy_weight: 0.0,
        ..NativePhasorConfig::default()
    };
    let hybrid_config = NativeHybridConfig {
        minimum_relative_energy_drop: 0.0,
        minimum_magnetic_coherence: -1.0,
        ..NativeHybridConfig::default()
    };
    let mut standalone = NativePhasorThermodynamicEngine::from_core(&core, phasor_config)?;
    let mut hybrid = NativeHybridPhasorCdtEngine::from_core(core, phasor_config, hybrid_config)?;

    let mut standalone_elapsed = Duration::ZERO;
    let mut hybrid_elapsed = Duration::ZERO;
    let mut max_energy_drift = 0.0_f32;
    let mut max_residual_drift = 0.0_f32;
    let mut iteration_mismatches = 0usize;
    for trial in 0..TRIALS {
        let cue = cue(NODES, trial);
        let standalone_report;
        let wake_report;
        if trial % 2 == 0 {
            let standalone_started = Instant::now();
            apply_cue(&mut standalone, &cue, hybrid_config.cue_as_boundary);
            standalone_report = standalone.minimize_free_energy(hybrid_config.minimizer);
            standalone_elapsed += standalone_started.elapsed();

            let hybrid_started = Instant::now();
            wake_report = hybrid.infer_and_stage(&cue)?;
            hybrid_elapsed += hybrid_started.elapsed();
        } else {
            let hybrid_started = Instant::now();
            wake_report = hybrid.infer_and_stage(&cue)?;
            hybrid_elapsed += hybrid_started.elapsed();

            let standalone_started = Instant::now();
            apply_cue(&mut standalone, &cue, hybrid_config.cue_as_boundary);
            standalone_report = standalone.minimize_free_energy(hybrid_config.minimizer);
            standalone_elapsed += standalone_started.elapsed();
        }

        max_energy_drift = max_energy_drift.max(
            (standalone_report.final_report.free_energy
                - wake_report.minimization.final_report.free_energy)
                .abs(),
        );
        max_residual_drift = max_residual_drift.max(
            (standalone_report.final_report.gradient_residual
                - wake_report.minimization.final_report.gradient_residual)
                .abs(),
        );
        iteration_mismatches +=
            usize::from(standalone_report.iterations != wake_report.minimization.iterations);
        black_box((&standalone_report, &wake_report));
    }

    let standalone_ms = standalone_elapsed.as_secs_f64() * 1_000.0;
    let hybrid_ms = hybrid_elapsed.as_secs_f64() * 1_000.0;
    let overhead = hybrid_ms / standalone_ms.max(f64::EPSILON);
    println!("Verificación de eficiencia fasorial antes/después de la integración");
    println!(
        "nodos={} aristas={} ensayos={} pendientes_wake={}",
        NODES,
        hybrid.core.edge_count(),
        TRIALS,
        hybrid.pending_attractors().len()
    );
    println!(
        "standalone_ms={standalone_ms:.3} híbrido_wake_ms={hybrid_ms:.3} \
         factor_overhead={overhead:.4}x"
    );
    println!(
        "drift_energía_max={max_energy_drift:.3e} drift_residuo_max={max_residual_drift:.3e} \
         diferencias_iteraciones={iteration_mismatches}"
    );
    println!(
        "paridad_numérica={} eficiencia_conservada={}",
        max_energy_drift < 1.0e-6 && max_residual_drift < 1.0e-7 && iteration_mismatches == 0,
        overhead <= 1.20
    );
    Ok(())
}

/// Replica exactamente cómo `infer_and_stage` presenta la cue, para que la
/// referencia aislada resuelva el mismo problema que wake.
fn apply_cue(
    engine: &mut NativePhasorThermodynamicEngine,
    cue: &[NativePhasorCue],
    as_boundary: bool,
) {
    engine.clear_stimulus();
    for item in cue {
        let field = Complex32::from_polar(item.amplitude, item.phase);
        engine.phasors[item.node] = field;
        if as_boundary {
            engine.stimulus[item.node] = field;
        }
    }
}

fn cue(nodes: usize, trial: usize) -> Vec<NativePhasorCue> {
    (0..nodes)
        .map(|node| {
            let base = if node % 3 == 0 { 0.12 } else { 2.7 };
            let jitter = 0.002 * signed_unit((trial as u64).rotate_left(23) ^ node as u64);
            NativePhasorCue {
                node,
                amplitude: 1.0,
                phase: base + jitter,
            }
        })
        .collect()
}
