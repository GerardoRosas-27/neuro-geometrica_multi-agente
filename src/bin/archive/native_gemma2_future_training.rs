//! Entrenamiento no supervisado desde una sola entrada y futuros de Gemma 2.
//!
//! Ejemplo:
//! `cargo run --release --bin native_gemma2_future_training -- \
//!    --input "0:+,3:-,8:+" --nodes 128 --epochs 4 --proposals 4`

use cdt_rqm_epr::future_guided_training::{
    future_training_engine, FutureGuidedTrainer, FutureGuidedTrainingConfig, FutureTrainingMode,
};
use cdt_rqm_epr::gemma_future_generator::GemmaFutureGenerator;
use cdt_rqm_epr::native_gemma2::resolve_gemma2_model_path;
use cdt_rqm_epr::native_gemma2_runtime::Gemma2GenerationConfig;
use cdt_rqm_epr::native_hybrid_phasor_cdt_engine::NativePhasorCue;
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let input = argument(&arguments, "--input")
        .ok_or("falta --input; formato: \"0:+,3:-,8:+\" (sólo la evidencia presente)")?;
    let nodes = parsed(&arguments, "--nodes", 128usize)?;
    let epochs = parsed(&arguments, "--epochs", 4usize)?;
    let proposals = parsed(&arguments, "--proposals", 4usize)?;
    let iterations = parsed(&arguments, "--iterations", 256usize)?;
    let max_tokens = parsed(&arguments, "--max-tokens", 256usize)?;
    let context = parsed(&arguments, "--context", 2_048usize)?;
    let seed = parsed(&arguments, "--seed", 0xF077_2026u64)?;
    let device = argument(&arguments, "--device").unwrap_or_else(|| "cpu".to_string());
    let explicit_model = argument(&arguments, "--model").map(PathBuf::from);
    let model_path = resolve_gemma2_model_path(explicit_model.as_deref())?;
    let cue = parse_input(&input, nodes)?;

    println!("entrenamiento=entrada_unica_futuros_gemma_postseleccion_F");
    println!("modelo={}", model_path.display());
    println!(
        "device={device} nodos={nodes} epocas={epochs} propuestas={proposals} iteraciones={iterations}"
    );
    println!("entrada={input}");
    println!("regla=Gemma_propone;F_selecciona;atencion_modula;delta_F_store_consolida");

    let config = FutureGuidedTrainingConfig {
        nodes,
        epochs,
        proposals_per_input: proposals,
        candidate_iterations: iterations,
        seed,
        ..FutureGuidedTrainingConfig::default()
    };
    let engine = future_training_engine(config, FutureTrainingMode::PredictedFutureAdaptive, seed)?;
    let generator = GemmaFutureGenerator::from_gguf(
        &model_path,
        &device,
        Gemma2GenerationConfig {
            max_tokens,
            context_limit: context,
            temperature: 0.75,
            top_p: 0.90,
            seed,
        },
    )?;
    let mut trainer = FutureGuidedTrainer::new(engine, generator, proposals);

    println!(
        "epoca,propuestas,seleccion,confianza,F,residuo,coherencia,caida_F,evaluaciones,\
         gate,consolidado,rechazo_FEP,igniciones,handshake,phi,tokens_gemma,decode_tok_s"
    );
    for epoch in 0..epochs {
        let episode_seed = seed ^ (epoch as u64).rotate_left(23);
        match trainer.learn_from_input(&cue, episode_seed)? {
            Some(report) => println!(
                "{},{},{},{:.4},{:.6},{:.3e},{:.6},{:.6},{},{},{},{},{},{:.6},{:.6},{},{:.2}",
                epoch + 1,
                report.proposals_generated,
                report.selected_latent_id,
                report.selected_confidence,
                report.selected_free_energy,
                report.final_residual,
                report.phase_coherence,
                report.relative_energy_drop,
                report.energy_evaluations,
                report.gate_passed,
                report.consolidated,
                report.rejected_by_efficiency,
                report.attention_ignitions,
                report.handshake_coherence,
                report.integrated_information,
                trainer.generator.last_metrics.generated_tokens,
                trainer.generator.last_metrics.decode_tokens_per_second(),
            ),
            None => {
                eprintln!(
                    "Gemma no produjo futuros parseables en época {}; salida={:?}; error={:?}",
                    epoch + 1,
                    trainer.generator.last_text,
                    trainer.generator.last_error
                );
            }
        }
    }
    println!("memorias_cdt={}", trainer.engine.attractors().len());
    Ok(())
}

fn argument(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .iter()
        .position(|value| value == name)
        .and_then(|index| arguments.get(index + 1))
        .cloned()
}

fn parsed<T: std::str::FromStr>(
    arguments: &[String],
    name: &str,
    default: T,
) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    argument(arguments, name).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|error| format!("{name} inválido: {error}").into())
    })
}

fn parse_input(
    input: &str,
    nodes: usize,
) -> Result<Vec<NativePhasorCue>, Box<dyn std::error::Error>> {
    let mut cue = Vec::new();
    for field in input.split(',') {
        let (node, sign) = field
            .trim()
            .split_once(':')
            .ok_or_else(|| format!("entrada inválida: {field}"))?;
        let node = node.trim().parse::<usize>()?;
        if node >= nodes {
            return Err(format!("nodo {node} fuera de 0..{nodes}").into());
        }
        let phase = match sign.trim() {
            "+" | "1" => 0.0,
            "-" | "0" => std::f32::consts::PI,
            other => return Err(format!("signo inválido: {other}").into()),
        };
        cue.push(NativePhasorCue {
            node,
            amplitude: 1.0,
            phase,
        });
    }
    if cue.is_empty() {
        return Err("la entrada no puede estar vacía".into());
    }
    Ok(cue)
}
