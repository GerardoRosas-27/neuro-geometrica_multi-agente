use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

const STATE_PATH: &str = "data/snga_headless_language.snga";
const PROGRESS_PATH: &str = "data/snga_headless_language.progress";

#[derive(Clone, Copy)]
struct Lesson {
    stage: &'static str,
    topic: &'static str,
    text: &'static str,
}

fn main() {
    let hours = arg_value("--hours")
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(8.0);
    let save_every_secs = arg_value("--save-every-seconds")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(300);
    let sleep_ms = arg_value("--sleep-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(250);

    let mut network = SimplicialNetwork::grid_3d(config(), 2);
    let loaded = network.load_persistent_state(STATE_PATH).is_ok();
    network.enable_neural_oscillations();

    let lessons = curriculum();
    let mut progress = load_progress().unwrap_or_default();
    let start = Instant::now();
    let duration = Duration::from_secs_f64(hours * 3600.0);
    let mut last_save = Instant::now();

    println!("SNGA headless language trainer");
    println!(
        "loaded={} lessons={} duration_hours={:.2} save_every={}s sleep={}ms",
        loaded,
        lessons.len(),
        hours,
        save_every_secs,
        sleep_ms
    );

    while start.elapsed() < duration {
        let lesson = lessons[progress.lesson_idx % lessons.len()];
        train_lesson(&mut network, lesson);
        progress.lesson_idx += 1;
        progress.total_lessons += 1;

        if progress.total_lessons % lessons.len() == 0 {
            progress.epochs += 1;
            evaluate_brief(&mut network, &lessons);
        }

        if last_save.elapsed() >= Duration::from_secs(save_every_secs) {
            save_all(&network, &progress, "checkpoint");
            last_save = Instant::now();
        }

        if sleep_ms > 0 {
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    save_all(&network, &progress, "final");
}

fn train_lesson(network: &mut SimplicialNetwork, lesson: Lesson) {
    let topic = text_pattern(lesson.topic, network.agents.len());
    let text = text_pattern(lesson.text, network.agents.len());
    let stage = text_pattern(lesson.stage, network.agents.len());
    let mut fused = topic.clone();
    fused.extend(text.iter().copied());
    fused.extend(stage.iter().copied());
    fused.sort_unstable();
    fused.dedup();

    network.clear_activity();
    network.set_attention_goal(&text);
    network.inject_pattern(&topic, 1.15, 2);
    network.inject_pattern(&text, 0.95, 1);
    network.learn_transition(&topic, &text);
    network.learn_transition(&stage, &topic);
    network.reinforce_coactivation_if_useful(&fused, 0.065, 0.92);
    for _ in 0..10 {
        network.step();
    }
    network.clear_attention_goal();
    network.clear_activity();
    for _ in 0..4 {
        network.step();
    }
}

fn evaluate_brief(network: &mut SimplicialNetwork, lessons: &[Lesson]) {
    let mut total_recall = 0.0;
    for lesson in lessons.iter().take(8) {
        let topic = text_pattern(lesson.topic, network.agents.len());
        let expected = text_pattern(lesson.text, network.agents.len());
        let predicted = network.predict_next_pattern(&topic, 1, expected.len());
        let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        total_recall += overlap(&predicted_ids, &expected) as f32 / expected.len().max(1) as f32;
    }
    println!(
        "probe checkpoint: avg_recall={:.1}% energy={:.1}",
        total_recall / 8.0 * 100.0,
        network.total_free_energy()
    );
}

fn save_all(network: &SimplicialNetwork, progress: &Progress, label: &str) {
    match network.save_persistent_state(STATE_PATH) {
        Ok(report) => {
            if let Err(err) = save_progress(progress) {
                eprintln!("{label}: estado guardado, progreso fallo: {err}");
            }
            println!(
                "{label}: saved agents={} edges={} causal={} lessons={} epochs={}",
                report.agents,
                report.edges,
                report.causal_edges,
                progress.total_lessons,
                progress.epochs
            );
        }
        Err(err) => eprintln!("{label}: fallo guardando estado: {err}"),
    }
}

#[derive(Default)]
struct Progress {
    lesson_idx: usize,
    total_lessons: usize,
    epochs: usize,
}

fn load_progress() -> Option<Progress> {
    let text = fs::read_to_string(PROGRESS_PATH).ok()?;
    let mut progress = Progress::default();
    for line in text.lines() {
        let (key, value) = line.split_once('=')?;
        match key {
            "lesson_idx" => progress.lesson_idx = value.parse().ok()?,
            "total_lessons" => progress.total_lessons = value.parse().ok()?,
            "epochs" => progress.epochs = value.parse().ok()?,
            _ => {}
        }
    }
    Some(progress)
}

fn save_progress(progress: &Progress) -> std::io::Result<()> {
    if let Some(parent) = Path::new(PROGRESS_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        PROGRESS_PATH,
        format!(
            "lesson_idx={}\ntotal_lessons={}\nepochs={}\n",
            progress.lesson_idx, progress.total_lessons, progress.epochs
        ),
    )
}

fn curriculum() -> Vec<Lesson> {
    vec![
        Lesson {
            stage: "lenguaje",
            topic: "palabra",
            text: "una palabra estable activa una region geometrica repetible",
        },
        Lesson {
            stage: "lenguaje",
            topic: "frase",
            text: "una frase organiza sujeto accion objeto en secuencia",
        },
        Lesson {
            stage: "lenguaje",
            topic: "pregunta",
            text: "una pregunta busca una ruta desde intencion hasta respuesta",
        },
        Lesson {
            stage: "conceptos",
            topic: "concepto",
            text: "un concepto es una region compacta dentro de la malla",
        },
        Lesson {
            stage: "conceptos",
            topic: "categoria",
            text: "una categoria agrupa rasgos compartidos y separa distractores",
        },
        Lesson {
            stage: "conceptos",
            topic: "contradiccion",
            text: "una contradiccion aumenta energia y debe ser inhibida",
        },
        Lesson {
            stage: "entorno",
            topic: "objeto",
            text: "un objeto mantiene rasgos y relaciones dentro de una escena",
        },
        Lesson {
            stage: "entorno",
            topic: "causa",
            text: "una causa predice un efecto si la ruta fue aprendida",
        },
        Lesson {
            stage: "entorno",
            topic: "evento",
            text: "un evento episodico conecta estado contexto y consecuencia",
        },
        Lesson {
            stage: "mundo",
            topic: "modelo interno",
            text: "el mundo interno simula futuros cortos en la geometria",
        },
        Lesson {
            stage: "mundo",
            topic: "plan",
            text: "un plan selecciona rutas causales hacia un objetivo",
        },
        Lesson {
            stage: "mundo",
            topic: "incertidumbre",
            text: "la incertidumbre aparece como sorpresa que guia aprendizaje",
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

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
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
