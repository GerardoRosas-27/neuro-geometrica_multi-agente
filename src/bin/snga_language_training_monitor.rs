use macroquad::prelude::*;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

const STATE_PATH: &str = "data/snga_headless_language.snga";
const PROGRESS_PATH: &str = "data/snga_headless_language.progress";
const TRAIN_STEPS_PER_FRAME: usize = 2;
const SAVE_EVERY_SECONDS: u64 = 300;

#[derive(Clone, Copy)]
struct Lesson {
    stage: &'static str,
    topic: &'static str,
    text: &'static str,
}

#[derive(Default)]
struct Progress {
    lesson_idx: usize,
    total_lessons: usize,
    epochs: usize,
}

struct Monitor {
    network: SimplicialNetwork,
    lessons: Vec<Lesson>,
    progress: Progress,
    paused: bool,
    started: Instant,
    last_save: Instant,
    last_probe_recall: f32,
    status: String,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA - Entrenamiento lingüístico liviano".to_string(),
        window_width: 980,
        window_height: 560,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app = Monitor::new();
    loop {
        app.handle_input();
        if !app.paused {
            for _ in 0..TRAIN_STEPS_PER_FRAME {
                app.train_one();
            }
            if app.last_save.elapsed() >= Duration::from_secs(SAVE_EVERY_SECONDS) {
                app.save("checkpoint");
            }
        }
        app.draw();
        next_frame().await;
        if is_key_pressed(KeyCode::Escape) {
            app.save("guardado al salir");
            break;
        }
    }
}

impl Monitor {
    fn new() -> Self {
        let mut network = SimplicialNetwork::grid_3d(config(), 2);
        let loaded = network.load_persistent_state(STATE_PATH).is_ok();
        network.enable_neural_oscillations();
        let progress = load_progress().unwrap_or_default();
        let mut app = Self {
            network,
            lessons: curriculum(),
            progress,
            paused: false,
            started: Instant::now(),
            last_save: Instant::now(),
            last_probe_recall: 0.0,
            status: if loaded {
                "sustrato cargado".to_string()
            } else {
                "sustrato nuevo".to_string()
            },
        };
        app.last_probe_recall = app.probe();
        app
    }

    fn handle_input(&mut self) {
        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if is_key_pressed(KeyCode::S) {
            self.save("guardado manual");
        }
        if is_key_pressed(KeyCode::P) {
            self.last_probe_recall = self.probe();
        }
    }

    fn train_one(&mut self) {
        let lesson = self.lessons[self.progress.lesson_idx % self.lessons.len()];
        train_lesson(&mut self.network, lesson);
        self.progress.lesson_idx += 1;
        self.progress.total_lessons += 1;
        if self.progress.total_lessons % self.lessons.len() == 0 {
            self.progress.epochs += 1;
            self.last_probe_recall = self.probe();
        }
    }

    fn probe(&mut self) -> f32 {
        let mut total = 0.0;
        for lesson in self.lessons.iter().take(8) {
            let topic = text_pattern(lesson.topic, self.network.agents.len());
            let expected = text_pattern(lesson.text, self.network.agents.len());
            let predicted = self.network.predict_next_pattern(&topic, 1, expected.len());
            let predicted_ids = predicted.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
            total += overlap(&predicted_ids, &expected) as f32 / expected.len().max(1) as f32;
        }
        total / 8.0
    }

    fn save(&mut self, prefix: &str) {
        match self.network.save_persistent_state(STATE_PATH) {
            Ok(report) => {
                if let Err(err) = save_progress(&self.progress) {
                    self.status = format!("{prefix}: estado guardado, progreso fallo: {err}");
                } else {
                    self.status = format!(
                        "{prefix}: agentes={} aristas={} causales={}",
                        report.agents, report.edges, report.causal_edges
                    );
                }
                self.last_save = Instant::now();
            }
            Err(err) => self.status = format!("{prefix}: fallo guardando {err}"),
        }
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(6, 8, 14, 255));
        let stats = self.network.stats();
        let plasticity = self.network.plasticity_stats();
        let osc = self.network.oscillation_stats();
        let elapsed = self.started.elapsed().as_secs();
        let save_in = SAVE_EVERY_SECONDS.saturating_sub(self.last_save.elapsed().as_secs());

        draw_text(
            "SNGA entrenamiento lingüístico liviano",
            28.0,
            40.0,
            32.0,
            WHITE,
        );
        draw_text(
            "Sin render de malla: solo métricas, bajo consumo gráfico.",
            28.0,
            70.0,
            20.0,
            SKYBLUE,
        );

        let lines = [
            format!(
                "estado: {} | tiempo: {}s | proximo guardado: {}s",
                if self.paused { "pausa" } else { "entrenando" },
                elapsed,
                save_in
            ),
            format!(
                "lecciones={} epochs={} lesson_idx={}",
                self.progress.total_lessons, self.progress.epochs, self.progress.lesson_idx
            ),
            format!(
                "probe_recall={:.1}% energia={:.1}",
                self.last_probe_recall * 100.0,
                stats.total_free_energy
            ),
            format!(
                "nodos={} aristas={} asociativas={} consolidadas={} causales={}",
                self.network.agents.len(),
                self.network.edges.len(),
                plasticity.associative_edges,
                plasticity.consolidated_edges,
                plasticity.causal_edges
            ),
            format!(
                "activos={} spikes={} modo={:?} regiones beta/gamma={}/{}",
                stats.active_agents,
                stats.active_spikes,
                osc.mode,
                osc.beta_regions,
                osc.gamma_regions
            ),
            self.status.clone(),
            "controles: Space pausa | S guardar | P probe | Esc guardar/salir".to_string(),
        ];

        let mut y = 125.0;
        for line in lines {
            draw_text(&line, 36.0, y, 24.0, WHITE);
            y += 42.0;
        }

        draw_bar(
            36.0,
            445.0,
            700.0,
            22.0,
            self.last_probe_recall,
            GREEN,
            "recall lingüístico interno",
        );
        let phase = ((self.progress.total_lessons % self.lessons.len()) as f32)
            / self.lessons.len().max(1) as f32;
        draw_bar(
            36.0,
            495.0,
            700.0,
            22.0,
            phase,
            ORANGE,
            "avance de época curricular",
        );
    }
}

fn draw_bar(x: f32, y: f32, w: f32, h: f32, value: f32, color: Color, label: &str) {
    draw_text(label, x, y - 8.0, 18.0, GRAY);
    draw_rectangle(x, y, w, h, Color::from_rgba(35, 42, 56, 255));
    draw_rectangle(x, y, w * value.clamp(0.0, 1.0), h, color);
    draw_text(
        &format!("{:.1}%", value * 100.0),
        x + w + 16.0,
        y + h - 3.0,
        20.0,
        WHITE,
    );
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
