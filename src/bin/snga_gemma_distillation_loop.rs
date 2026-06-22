use macroquad::prelude::*;
use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::path::Path;

const STATE_PATH: &str = "data/snga_gemma_distilled_language.snga";
const PROGRESS_PATH: &str = "data/snga_gemma_distillation.progress";
const DISTILL_EVERY_FRAMES: u64 = 45;
const HEARTBEAT_SAVE_FRAMES: u64 = 180;
const QUIZ_EVERY_LESSONS: usize = 4;
const PASSES_TO_ADVANCE: usize = 3;
const MIN_LESSONS_TO_ADVANCE: usize = 8;
const MAX_MESSAGES: usize = 22;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CurriculumStage {
    Language,
    Concepts,
    Environment,
    World,
}

impl CurriculumStage {
    fn label(self) -> &'static str {
        match self {
            Self::Language => "lenguaje",
            Self::Concepts => "conceptos",
            Self::Environment => "entorno",
            Self::World => "mundo",
        }
    }

    fn topics(self) -> &'static [&'static str] {
        match self {
            Self::Language => &[
                "palabra como simbolo estable",
                "frase como secuencia de intencion",
                "sujeto accion objeto",
                "pregunta y respuesta",
                "resumen breve de memoria",
            ],
            Self::Concepts => &[
                "concepto como region geometrica",
                "categoria y rasgo compartido",
                "contradiccion entre conceptos",
                "jerarquia de ideas",
                "asociacion multimodal",
            ],
            Self::Environment => &[
                "objeto en una escena",
                "causa y efecto local",
                "cambio por accion",
                "memoria episodica de evento",
                "prediccion del siguiente estado",
            ],
            Self::World => &[
                "modelo interno del mundo",
                "plan a varios pasos",
                "objetivo y ruta causal",
                "incertidumbre y sorpresa",
                "aprendizaje continuo sin olvidar",
            ],
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Language => Self::Concepts,
            Self::Concepts => Self::Environment,
            Self::Environment => Self::World,
            Self::World => Self::World,
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "conceptos" => Self::Concepts,
            "entorno" => Self::Environment,
            "mundo" => Self::World,
            _ => Self::Language,
        }
    }
}

#[derive(Clone, Debug)]
struct CurriculumProgress {
    stage: CurriculumStage,
    topic_idx: usize,
    total_lessons: usize,
    stage_lessons: usize,
    quizzes: usize,
    passes: usize,
}

impl Default for CurriculumProgress {
    fn default() -> Self {
        Self {
            stage: CurriculumStage::Language,
            topic_idx: 0,
            total_lessons: 0,
            stage_lessons: 0,
            quizzes: 0,
            passes: 0,
        }
    }
}

struct DistillationApp {
    network: SimplicialNetwork,
    engine: OllamaGemmaEngine,
    messages: Vec<(String, String)>,
    progress: CurriculumProgress,
    frame: u64,
    status: String,
    paused: bool,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SNGA - Destilacion ciclica Gemma -> Red".to_string(),
        window_width: 1280,
        window_height: 820,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app = DistillationApp::new();
    loop {
        app.handle_input();
        if !app.paused {
            app.frame += 1;
            if app.frame % DISTILL_EVERY_FRAMES == 1 {
                app.distill_step();
            }
            for _ in 0..3 {
                app.network.step();
            }
            if app.frame % HEARTBEAT_SAVE_FRAMES == 0 {
                app.save("heartbeat guardado");
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

impl DistillationApp {
    fn new() -> Self {
        let mut network = SimplicialNetwork::grid_3d(distill_config(), 2);
        let substrate_status = match network.load_persistent_state(STATE_PATH) {
            Ok(report) => format!(
                "sustrato linguistico cargado: agentes={} aristas={} causales={}",
                report.agents, report.edges, report.causal_edges
            ),
            Err(_) => "sustrato linguistico limpio: iniciando destilacion".to_string(),
        };
        network.enable_neural_oscillations();
        let progress = load_progress(PROGRESS_PATH).unwrap_or_default();
        let status = format!(
            "{} | progreso: etapa={} lecciones={} passes={}",
            substrate_status,
            progress.stage.label(),
            progress.total_lessons,
            progress.passes
        );
        Self {
            network,
            engine: OllamaGemmaEngine {
                host: env::var("SNGA_OLLAMA_HOST")
                    .unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
                model: env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()),
            },
            messages: vec![(
                "sistema".to_string(),
                "Curriculo iniciado: lenguaje -> conceptos -> entorno -> mundo.".to_string(),
            )],
            progress,
            frame: 0,
            status,
            paused: false,
        }
    }

    fn handle_input(&mut self) {
        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if is_key_pressed(KeyCode::N) {
            self.distill_step();
        }
        if is_key_pressed(KeyCode::S) {
            self.save("guardado manual");
        }
    }

    fn distill_step(&mut self) {
        let topics = self.progress.stage.topics();
        let topic = topics[self.progress.topic_idx % topics.len()];
        self.progress.topic_idx += 1;

        let teacher_prompt = format!(
            "Eres maestro de una red SNGA. Etapa: {}. Tema: {topic}. \
             Escribe una sola frase didactica, concreta, maximo 18 palabras, sin lista.",
            self.progress.stage.label()
        );
        let teacher_context = LinguisticContext {
            user_prompt: teacher_prompt,
            inferred_intent: "ensenanza_linguistica".to_string(),
            geometric_projection: self.network.project_active_state(8),
            memory_summary:
                "Gemma actua como maestro linguistico; SNGA debe aprender la frase en su malla."
                    .to_string(),
        };
        let teacher = self
            .engine
            .generate(&teacher_context)
            .unwrap_or_else(|_| fallback_teacher(topic));

        self.learn_lesson(topic, &teacher.text);
        let student = self.student_response(topic);

        self.messages
            .push(("Gemma maestro".to_string(), teacher.text));
        self.messages.push(("SNGA estudiante".to_string(), student));
        truncate_messages(&mut self.messages);
        self.progress.total_lessons += 1;
        self.progress.stage_lessons += 1;

        if self.progress.total_lessons % QUIZ_EVERY_LESSONS == 0 {
            self.quiz_step();
        }
        self.maybe_advance_stage();
        self.save("autosave destilacion");
    }

    fn quiz_step(&mut self) {
        let topics = self.progress.stage.topics();
        let topic = topics[self.progress.topic_idx.saturating_sub(1) % topics.len()];
        let topic_pattern = text_pattern(topic, self.network.agents.len());
        let predicted = self.network.predict_next_pattern(&topic_pattern, 1, 18);
        let confidence = predicted.iter().map(|(_, score)| *score).sum::<f32>() / 18.0;
        let passed = predicted.len() >= 8 && confidence > 0.08;
        self.progress.quizzes += 1;
        if passed {
            self.progress.passes += 1;
        }
        self.messages.push((
            "Quiz".to_string(),
            format!(
                "Etapa {} tema '{}': predicciones={} confianza={:.3} resultado={}",
                self.progress.stage.label(),
                topic,
                predicted.len(),
                confidence,
                if passed { "aprobado" } else { "repasar" }
            ),
        ));
        truncate_messages(&mut self.messages);
    }

    fn maybe_advance_stage(&mut self) {
        if self.progress.stage == CurriculumStage::World {
            return;
        }
        if self.progress.stage_lessons >= MIN_LESSONS_TO_ADVANCE
            && self.progress.passes >= PASSES_TO_ADVANCE
        {
            let previous = self.progress.stage;
            self.progress.stage = self.progress.stage.next();
            self.progress.stage_lessons = 0;
            self.progress.passes = 0;
            self.progress.topic_idx = 0;
            self.messages.push((
                "Sistema".to_string(),
                format!(
                    "Avance curricular: {} -> {}",
                    previous.label(),
                    self.progress.stage.label()
                ),
            ));
            truncate_messages(&mut self.messages);
        }
    }

    fn learn_lesson(&mut self, topic: &str, lesson: &str) {
        let topic_pattern = text_pattern(topic, self.network.agents.len());
        let lesson_pattern = text_pattern(lesson, self.network.agents.len());
        let mut fused = topic_pattern.clone();
        fused.extend(lesson_pattern.iter().copied());
        fused.sort_unstable();
        fused.dedup();

        self.network.clear_activity();
        self.network.set_attention_goal(&lesson_pattern);
        self.network.inject_pattern(&topic_pattern, 1.15, 2);
        self.network.inject_pattern(&lesson_pattern, 0.95, 1);
        self.network
            .learn_transition(&topic_pattern, &lesson_pattern);
        self.network
            .reinforce_coactivation_if_useful(&fused, 0.07, 0.92);
        for _ in 0..12 {
            self.network.step();
        }
        self.network.clear_attention_goal();
    }

    fn student_response(&mut self, topic: &str) -> String {
        let topic_pattern = text_pattern(topic, self.network.agents.len());
        let predicted = self.network.predict_next_pattern(&topic_pattern, 1, 12);
        let projection = self.network.project_active_state(8);
        let predicted_summary = predicted
            .iter()
            .map(|(idx, score)| format!("{idx}:{score:.2}"))
            .collect::<Vec<_>>()
            .join(", ");
        let active_summary = projection
            .top_agents
            .iter()
            .map(|(idx, score)| format!("{idx}:{score:.2}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Aprendi sobre '{topic}'. Prediccion interna [{}]. Activacion geometrica [{}].",
            predicted_summary, active_summary
        )
    }

    fn save(&mut self, prefix: &str) {
        match self.persist() {
            Ok(report) => {
                self.status = format!(
                    "{prefix}: etapa={} lecciones={} agentes={} aristas={} causales={}",
                    self.progress.stage.label(),
                    self.progress.total_lessons,
                    report.agents,
                    report.edges,
                    report.causal_edges
                );
            }
            Err(err) => {
                self.status = format!("fallo guardando destilacion: {err}");
            }
        }
    }

    fn persist(&self) -> std::io::Result<snga::simplicial::PersistentStateReport> {
        let report = self.network.save_persistent_state(STATE_PATH)?;
        save_progress(PROGRESS_PATH, &self.progress)?;
        Ok(report)
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(6, 8, 14, 255));
        draw_text("Destilacion ciclica Gemma -> SNGA", 24.0, 34.0, 30.0, WHITE);
        draw_text(
            "Gemma enseña frases; SNGA aprende patrones en la malla persistente.",
            24.0,
            62.0,
            19.0,
            SKYBLUE,
        );
        draw_text(
            &format!(
                "estado={} | frame={} | lecciones={} | energia={:.1} | aristas={} | causales={}",
                if self.paused { "pausa" } else { "destilando" },
                self.frame,
                self.progress.total_lessons,
                self.network.total_free_energy(),
                self.network.edges.len(),
                self.network.plasticity_stats().causal_edges
            ),
            24.0,
            96.0,
            20.0,
            WHITE,
        );
        draw_text(
            &format!(
                "{} | etapa={} stage_lessons={} quizzes={} passes={}",
                self.status,
                self.progress.stage.label(),
                self.progress.stage_lessons,
                self.progress.quizzes,
                self.progress.passes
            ),
            24.0,
            126.0,
            18.0,
            GRAY,
        );

        let mut y = 170.0;
        for (speaker, msg) in &self.messages {
            let color = if speaker.starts_with("Gemma") {
                GREEN
            } else if speaker.starts_with("SNGA") {
                YELLOW
            } else {
                SKYBLUE
            };
            draw_text(&format!("{speaker}:"), 28.0, y, 22.0, color);
            y = draw_wrapped(msg, 42.0, y + 26.0, screen_width() - 84.0, 20.0, WHITE) + 16.0;
        }

        draw_text(
            "Controles: N siguiente leccion | Space pausa | S guardar | Esc guardar/salir",
            24.0,
            screen_height() - 22.0,
            18.0,
            GRAY,
        );
    }
}

impl Drop for DistillationApp {
    fn drop(&mut self) {
        let _ = self.persist();
    }
}

fn fallback_teacher(topic: &str) -> snga::linguistic_engine::LinguisticResponse {
    snga::linguistic_engine::LinguisticResponse {
        text: format!(
            "SNGA aprende {topic} cuando una frase activa regiones geometricas que luego se consolidan por replay."
        ),
        engine: "fallback-teacher".to_string(),
    }
}

fn load_progress(path: &str) -> Option<CurriculumProgress> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut progress = CurriculumProgress::default();
    for line in text.lines() {
        let (key, value) = line.split_once('=')?;
        match key {
            "stage" => progress.stage = CurriculumStage::from_label(value),
            "topic_idx" => progress.topic_idx = value.parse().ok()?,
            "total_lessons" => progress.total_lessons = value.parse().ok()?,
            "stage_lessons" => progress.stage_lessons = value.parse().ok()?,
            "quizzes" => progress.quizzes = value.parse().ok()?,
            "passes" => progress.passes = value.parse().ok()?,
            _ => {}
        }
    }
    Some(progress)
}

fn save_progress(path: &str, progress: &CurriculumProgress) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        path,
        format!(
            "stage={}\ntopic_idx={}\ntotal_lessons={}\nstage_lessons={}\nquizzes={}\npasses={}\n",
            progress.stage.label(),
            progress.topic_idx,
            progress.total_lessons,
            progress.stage_lessons,
            progress.quizzes,
            progress.passes
        ),
    )
}

fn truncate_messages(messages: &mut Vec<(String, String)>) {
    while messages.len() > MAX_MESSAGES {
        messages.remove(0);
    }
}

fn draw_wrapped(text: &str, x: f32, y: f32, max_width: f32, size: f32, color: Color) -> f32 {
    let mut line = String::new();
    let mut yy = y;
    for word in text.split_whitespace() {
        let candidate = if line.is_empty() {
            word.to_string()
        } else {
            format!("{line} {word}")
        };
        if measure_text(&candidate, None, size as u16, 1.0).width > max_width && !line.is_empty() {
            draw_text(&line, x, yy, size, color);
            yy += size + 6.0;
            line = word.to_string();
        } else {
            line = candidate;
        }
    }
    if !line.is_empty() {
        draw_text(&line, x, yy, size, color);
        yy += size + 6.0;
    }
    yy
}

fn text_pattern(text: &str, nodes: usize) -> Vec<usize> {
    text.bytes()
        .enumerate()
        .map(|(i, byte)| ((byte as usize * 41) + i * 67 + text.len() * 13) % nodes)
        .collect()
}

fn distill_config() -> SimplicialConfig {
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
