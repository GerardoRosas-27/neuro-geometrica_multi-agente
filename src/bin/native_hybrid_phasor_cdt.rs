use cdt_rqm_epr::native_hybrid_phasor_cdt_engine::{
    NativeHybridConfig, NativeHybridPhasorCdtEngine, NativePhasorCue,
};
use cdt_rqm_epr::native_phasor_thermodynamic_engine::NativePhasorConfig;
use cdt_rqm_epr::native_thermodynamic_cdt::NativeThermoCdtConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = NativeHybridPhasorCdtEngine::new(
        NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: 64,
            temperature: 0.0,
            diffusion: 0.12,
            pilot_gain: 0.0,
            amplitude_decay: 0.0,
            seed: 73_119,
            ..NativeThermoCdtConfig::default()
        },
        NativePhasorConfig {
            temperature_scale: 0.0,
            noise_scale: 0.0,
            entropy_weight: 0.0,
            ..NativePhasorConfig::default()
        },
        NativeHybridConfig {
            minimum_stability: 0.90,
            minimum_relative_energy_drop: 0.0,
            ..NativeHybridConfig::default()
        },
    )?;

    let cue = (0..32)
        .map(|node| NativePhasorCue {
            node,
            amplitude: 1.0,
            phase: if node < 16 { 0.0 } else { std::f32::consts::PI },
        })
        .collect::<Vec<_>>();

    println!("Motor híbrido: inferencia fasorial wake + consolidación CDT sleep");
    let cdt_phase_before_wake = engine.core.phase.clone();
    for exposure in 1..=2 {
        let report = engine.infer_and_stage(&cue)?;
        println!(
            "wake={} gate={} F={:.6}->{:.6} residuo={:.3e} \
             coherencia={:.5} estabilidad=diferida confianza={:.5} \
             pendiente={:?} cola={} memoria_CDT={}",
            exposure,
            report.gate.passed,
            report.minimization.initial.free_energy,
            report.minimization.final_report.free_energy,
            report.minimization.final_report.gradient_residual,
            report.minimization.final_report.phase_coherence,
            report.confidence,
            report.pending_id,
            report.pending_count,
            engine.attractors().len(),
        );
    }
    println!(
        "wake_modificó_CDT={}",
        engine.core.phase != cdt_phase_before_wake
    );

    let sleep = engine.sleep_consolidate()?;
    println!(
        "sleep={} pendientes={} revalidados={} aceptados={} rechazados={} \
         estabilidad_media={:.5} aristas_consolidadas={} memoria={}->{} CDT_F={:.6}->{:.6}",
        sleep.sleep_cycle,
        sleep.pending_before,
        sleep.revalidated,
        sleep.accepted,
        sleep.rejected,
        sleep.mean_stability,
        sleep.consolidated_edges,
        sleep.memory_before,
        sleep.memory_size,
        sleep.cdt_before.free_energy_proxy,
        sleep.cdt_after.free_energy_proxy,
    );

    println!(
        "resultado: core_nodos={} core_aristas={} atractores_persistentes={} pendientes={}",
        engine.core.node_count(),
        engine.core.edge_count(),
        engine.attractors().len(),
        engine.pending_attractors().len(),
    );
    Ok(())
}
