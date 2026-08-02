//! Entrenamiento comparado de las tres estructuras temporales de inferencia.

use cdt_rqm_epr::transactional_training_experiment::{
    run_transactional_training_comparison, TransactionalTrainingConfig,
};

const SCHEDULE: [usize; 4] = [4, 12, 24, 48];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("experimento=estructura_temporal_de_la_inferencia");
    println!("evaluacion=cue_parcial_corrompida sin_meta sin_estimulo mismo_core_entrenado");
    println!("azar=0.50 la_evaluacion_nunca_ve_la_meta_ni_las_mascaras_de_entrenamiento");
    println!(
        "epocas,modo,moduladores,gate_pasados,wake_ciclos,sleep_aceptados,rechazos_fep,memoria,\
         phi,acuerdo_frontera,igniciones,train_s,exactitud,exactitud_gauge,tasa_exito,iteraciones"
    );

    for epochs in SCHEDULE {
        let report = run_transactional_training_comparison(TransactionalTrainingConfig {
            epochs,
            ..TransactionalTrainingConfig::default()
        })?;
        for run in &report.runs {
            println!(
                "{epochs},{},{},{}/{},{},{},{},{:.6},{:.6},{},{:.3},{:.4},{:.4},{:.4},{:.1}",
                run.mode.label(),
                run.modulators,
                run.gate_passed,
                run.wake_cycles,
                run.sleep_accepted,
                run.rejected_by_efficiency,
                run.memory_size,
                run.mean_integrated_information,
                run.mean_handshake_coherence,
                run.attention_ignitions,
                run.train_seconds,
                run.eval_accuracy,
                run.eval_gauge_invariant_accuracy,
                run.eval_success_rate,
                run.eval_mean_iterations,
            );
        }
        println!("epocas={epochs} decision={}", report.decision);
    }
    Ok(())
}
