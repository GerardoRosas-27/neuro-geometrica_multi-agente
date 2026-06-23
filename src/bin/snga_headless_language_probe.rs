use snga::simplicial::{SimplicialConfig, SimplicialNetwork};

const STATE_PATH: &str = "data/snga_headless_language.snga";

struct ProbeCase {
    topic: &'static str,
    expected: &'static str,
}

fn main() {
    let mut network = SimplicialNetwork::grid_3d(config(), 2);
    let loaded = network.load_persistent_state(STATE_PATH);
    if let Err(err) = &loaded {
        println!("SNGA headless language probe");
        println!("loaded=false error={err}");
        println!("lectura: no hay sustrato entrenado aun; ejecuta snga_headless_language_trainer");
        return;
    }

    let cases = probe_cases();
    let mut top_recall = 0.0;
    let mut nonzero = 0;

    println!("SNGA headless language probe");
    println!("loaded=true cases={}", cases.len());
    for case in &cases {
        let topic = text_pattern(case.topic, network.agents.len());
        let expected = text_pattern(case.expected, network.agents.len());
        let predicted = network.predict_next_pattern(&topic, 1, expected.len());
        let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let recall = overlap(&predicted_ids, &expected) as f32 / expected.len().max(1) as f32;
        if !predicted.is_empty() {
            nonzero += 1;
        }
        top_recall += recall;
        println!(
            "case topic={:?} recall={:.1}% predicted={}",
            case.topic,
            recall * 100.0,
            predicted.len()
        );
    }

    let avg_recall = top_recall / cases.len().max(1) as f32;
    println!(
        "summary: avg_recall={:.1}% nonzero_predictions={}/{} energy={:.1}",
        avg_recall * 100.0,
        nonzero,
        cases.len(),
        network.total_free_energy()
    );
    println!(
        "lectura: {}",
        if avg_recall > 0.55 && nonzero == cases.len() {
            "hay indicios de aprendizaje linguistico interno sin usar Gemma"
        } else {
            "el sustrato tiene actividad, pero necesita mas entrenamiento ciclico"
        }
    );
}

fn probe_cases() -> Vec<ProbeCase> {
    vec![
        ProbeCase {
            topic: "palabra",
            expected: "una palabra estable activa una region geometrica repetible",
        },
        ProbeCase {
            topic: "frase",
            expected: "una frase organiza sujeto accion objeto en secuencia",
        },
        ProbeCase {
            topic: "concepto",
            expected: "un concepto es una region compacta dentro de la malla",
        },
        ProbeCase {
            topic: "causa",
            expected: "una causa predice un efecto si la ruta fue aprendida",
        },
        ProbeCase {
            topic: "modelo interno",
            expected: "el mundo interno simula futuros cortos en la geometria",
        },
        ProbeCase {
            topic: "plan",
            expected: "un plan selecciona rutas causales hacia un objetivo",
        },
    ]
}

fn text_pattern(text: &str, nodes: usize) -> Vec<usize> {
    text.bytes()
        .enumerate()
        .map(|(i, byte)| ((byte as usize * 43) + i * 71 + text.len() * 17) % nodes)
        .collect()
}

fn overlap(left: &[usize], right: &[usize]) -> usize {
    let right = right
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    left.iter().filter(|idx| right.contains(idx)).count()
}

fn config() -> SimplicialConfig {
    SimplicialConfig {
        width: 48,
        height: 28,
        spacing: 8.0,
        elasticity: 0.007,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.0002,
        max_active_agents: 160,
        inhibition_decay: 0.05,
        max_spikes_per_step: 512,
        local_inhibition_decay: 0.76,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 1024,
        replay_interval: 8,
        replay_batch: 8,
        replay_learning_rate: 0.06,
        causal_learning_rate: 0.20,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.0001,
        hyperbolic_curvature: 0.0,
        seed: 317,
    }
}
