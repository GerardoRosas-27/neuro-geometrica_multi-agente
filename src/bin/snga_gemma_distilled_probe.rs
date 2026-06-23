use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::fs;

const STATE_PATH: &str = "data/snga_gemma_distilled_language.snga";
const PROGRESS_PATH: &str = "data/snga_gemma_distillation.progress";

fn main() {
    let mut learned = SimplicialNetwork::grid_3d(config(), 2);
    let baseline = SimplicialNetwork::grid_3d(config(), 2);
    let loaded = learned.load_persistent_state(STATE_PATH);

    println!("SNGA Gemma distilled probe");
    match loaded {
        Ok(report) => {
            println!(
                "loaded=true agents={} edges={} causal_edges={}",
                report.agents, report.edges, report.causal_edges
            );
        }
        Err(err) => {
            println!("loaded=false error={err}");
            println!("lectura: no hay sustrato de destilacion Gemma guardado");
            return;
        }
    }

    if let Ok(progress) = fs::read_to_string(PROGRESS_PATH) {
        println!("progress:\n{}", progress.trim());
    }

    let topics = curriculum_topics();
    let exercises = exercise_labels();
    let learned_topic = eval_prompts(&learned, &topics);
    let baseline_topic = eval_prompts(&baseline, &topics);
    let learned_exercise = eval_prompts(&learned, &exercises);
    let baseline_exercise = eval_prompts(&baseline, &exercises);

    println!(
        "topics: learned_nonzero={}/{} baseline_nonzero={}/{} learned_conf={:.3} baseline_conf={:.3}",
        learned_topic.nonzero,
        topics.len(),
        baseline_topic.nonzero,
        topics.len(),
        learned_topic.mean_confidence,
        baseline_topic.mean_confidence
    );
    println!(
        "exercise_types: learned_nonzero={}/{} baseline_nonzero={}/{} learned_conf={:.3} baseline_conf={:.3}",
        learned_exercise.nonzero,
        exercises.len(),
        baseline_exercise.nonzero,
        exercises.len(),
        learned_exercise.mean_confidence,
        baseline_exercise.mean_confidence
    );

    let plasticity = learned.plasticity_stats();
    println!(
        "network: active_edges={} associative_edges={} consolidated_edges={} causal_edges={} energy={:.1}",
        plasticity.active_edges,
        plasticity.associative_edges,
        plasticity.consolidated_edges,
        plasticity.causal_edges,
        learned.total_free_energy()
    );

    println!(
        "lectura: {}",
        if learned_topic.nonzero == topics.len()
            && learned_exercise.nonzero == exercises.len()
            && learned_topic.mean_confidence > baseline_topic.mean_confidence + 0.01
            && plasticity.causal_edges > 0
        {
            "hay aprendizaje linguistico destilado en SNGA sin usar Gemma"
        } else {
            "hay estado guardado, pero la evidencia de destilacion linguistica aun es debil"
        }
    );
}

struct EvalStats {
    nonzero: usize,
    mean_confidence: f32,
}

fn eval_prompts(network: &SimplicialNetwork, prompts: &[&str]) -> EvalStats {
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for prompt in prompts {
        let pattern = text_pattern(prompt, network.agents.len());
        let predicted = network.predict_next_pattern(&pattern, 1, 24);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        let conf = predicted.iter().map(|(_, score)| *score).sum::<f32>() / 24.0;
        confidence += conf;
        println!(
            "probe prompt={:?} predicted={} confidence={:.3}",
            prompt,
            predicted.len(),
            conf
        );
    }
    EvalStats {
        nonzero,
        mean_confidence: confidence / prompts.len().max(1) as f32,
    }
}

fn curriculum_topics() -> Vec<&'static str> {
    vec![
        "palabra como simbolo estable",
        "frase como secuencia de intencion",
        "sujeto accion objeto",
        "pregunta y respuesta",
        "resumen breve de memoria",
        "concepto como region geometrica",
        "categoria y rasgo compartido",
        "contradiccion entre conceptos",
        "jerarquia de ideas",
        "asociacion multimodal",
        "objeto en una escena",
        "causa y efecto local",
        "cambio por accion",
        "memoria episodica de evento",
        "prediccion del siguiente estado",
        "modelo interno del mundo",
        "plan a varios pasos",
        "objetivo y ruta causal",
        "incertidumbre y sorpresa",
        "aprendizaje continuo sin olvidar",
    ]
}

fn exercise_labels() -> Vec<&'static str> {
    vec![
        "definicion",
        "parafrasis",
        "pregunta_respuesta",
        "analogia",
        "correccion",
    ]
}

fn text_pattern(text: &str, nodes: usize) -> Vec<usize> {
    text.bytes()
        .enumerate()
        .map(|(i, byte)| ((byte as usize * 41) + i * 67 + text.len() * 13) % nodes)
        .collect()
}

fn config() -> SimplicialConfig {
    SimplicialConfig {
        width: 40,
        height: 24,
        spacing: 9.0,
        elasticity: 0.007,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 128,
        inhibition_decay: 0.05,
        max_spikes_per_step: 384,
        local_inhibition_decay: 0.76,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 512,
        replay_interval: 8,
        replay_batch: 8,
        replay_learning_rate: 0.06,
        causal_learning_rate: 0.20,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 313,
    }
}
