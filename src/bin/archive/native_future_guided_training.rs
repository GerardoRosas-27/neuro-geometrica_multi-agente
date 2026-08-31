//! Validación sintética del aprendizaje por futuros postseleccionados.

use cdt_rqm_epr::future_guided_training::{
    run_synthetic_future_guided_comparison, FutureGuidedTrainingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("experimento=entrada_unica_futuros_gemma_sintetico_vs_interferencia");
    println!("control=mismo_core_estado_cues_semillas_y_presupuesto_maximo");
    println!("seleccion=futuro_con_menor_energia_libre");
    println!("consolidacion=gate_wake+revalidacion_sleep+delta_F_store");
    println!(
        "nodos,tareas,epocas,propuestas,modo,wake,propuestas_generadas,top1_correcto,\
         seleccionado_correcto,gate,consolidado,rechazo_FEP,evaluaciones_F,igniciones,\
         handshake,phi,segundos,recuerdo,tasa_exito,iteraciones_recuerdo"
    );

    for nodes in [64usize, 128, 256] {
        let report = run_synthetic_future_guided_comparison(FutureGuidedTrainingConfig {
            nodes,
            synthetic_tasks: 8,
            epochs: 16,
            proposals_per_input: 4,
            evaluation_trials: 8,
            candidate_iterations: 96,
            ..FutureGuidedTrainingConfig::default()
        })?;
        for item in &report.reports {
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.3},{:.6},{:.6},{:.2}",
                report.config.nodes,
                report.config.tasks,
                report.config.epochs,
                report.config.proposals,
                item.mode,
                item.wake_cycles,
                item.generated_proposals,
                item.correct_top1_futures,
                item.selected_correct_futures,
                item.gate_passed,
                item.consolidated,
                item.rejected_by_efficiency,
                item.energy_evaluations,
                item.attention_ignitions,
                item.mean_handshake_coherence,
                item.mean_phi,
                item.train_seconds,
                item.recall_accuracy,
                item.recall_success_rate,
                item.recall_iterations,
            );
        }
        println!("nodos={nodes} ganador={}", report.winner);
    }
    Ok(())
}
