use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

const STATE_PATH: &str = "data/snga_gemma_distilled_language.snga";
const PROGRESS_PATH: &str = "data/snga_gemma_distillation.progress";
const QUIZ_EVERY_LESSONS: usize = 4;
const PASSES_TO_ADVANCE: usize = 3;
const MIN_LESSONS_TO_ADVANCE: usize = 8;

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

#[derive(Clone, Copy, Debug)]
enum ExerciseKind {
    Definition,
    Paraphrase,
    QuestionAnswer,
    Analogy,
    Correction,
}

impl ExerciseKind {
    fn from_lesson_idx(idx: usize) -> Self {
        match idx % 5 {
            0 => Self::Definition,
            1 => Self::Paraphrase,
            2 => Self::QuestionAnswer,
            3 => Self::Analogy,
            _ => Self::Correction,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Definition => "definicion",
            Self::Paraphrase => "parafrasis",
            Self::QuestionAnswer => "pregunta_respuesta",
            Self::Analogy => "analogia",
            Self::Correction => "correccion",
        }
    }
}

#[derive(Clone, Debug)]
struct Progress {
    stage: CurriculumStage,
    topic_idx: usize,
    total_lessons: usize,
    stage_lessons: usize,
    quizzes: usize,
    passes: usize,
}

impl Default for Progress {
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
    let buffer_lessons = arg_value("--buffer-lessons")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(24)
        .max(1);

    let engine = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()),
    };
    let mut network = SimplicialNetwork::grid_3d(config(), 2);
    let loaded = network.load_persistent_state(STATE_PATH).is_ok();
    network.enable_neural_oscillations();
    let mut progress = load_progress().unwrap_or_default();

    let start = Instant::now();
    let duration = Duration::from_secs_f64(hours * 3600.0);
    let mut last_save = Instant::now();
    let mut dirty_lessons = 0_usize;

    println!("SNGA Gemma headless open-language distillation");
    println!(
        "model={} loaded={} duration_hours={:.2} save_every={}s sleep={}ms buffer_lessons={}",
        engine.model, loaded, hours, save_every_secs, sleep_ms, buffer_lessons
    );

    while start.elapsed() < duration {
        let topics = progress.stage.topics();
        let topic = topics[progress.topic_idx % topics.len()];
        let exercise = ExerciseKind::from_lesson_idx(progress.total_lessons);
        progress.topic_idx += 1;

        let lesson = ask_gemma_lesson(&engine, progress.stage, topic, exercise);
        learn_lesson(&mut network, topic, exercise, &lesson);
        progress.total_lessons += 1;
        progress.stage_lessons += 1;
        dirty_lessons += 1;

        let mut quiz_line = String::new();
        if progress.total_lessons % QUIZ_EVERY_LESSONS == 0 {
            let (passed, confidence, predicted) = quiz(&network, topic);
            progress.quizzes += 1;
            if passed {
                progress.passes += 1;
            }
            quiz_line = format!(
                " quiz={} conf={:.3} predicted={}",
                if passed { "ok" } else { "repasar" },
                confidence,
                predicted
            );
        }
        if maybe_advance(&mut progress) {
            println!("avance curricular -> {}", progress.stage.label());
        }

        let stats = network.plasticity_stats();
        println!(
            "lesson={} stage={} kind={} topic={:?} edges={} assoc={} causal={} energy={:.1}{}",
            progress.total_lessons,
            progress.stage.label(),
            exercise.label(),
            topic,
            network.edges.len(),
            stats.associative_edges,
            stats.causal_edges,
            network.total_free_energy(),
            quiz_line
        );

        let buffer_full = dirty_lessons >= buffer_lessons;
        let time_to_save = last_save.elapsed() >= Duration::from_secs(save_every_secs);
        if buffer_full || time_to_save {
            let label = if buffer_full {
                "buffer_checkpoint"
            } else {
                "time_checkpoint"
            };
            save_all(&network, &progress, label);
            last_save = Instant::now();
            dirty_lessons = 0;
        }
        if sleep_ms > 0 {
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    save_all(&network, &progress, "final");
}

fn ask_gemma_lesson(
    engine: &OllamaGemmaEngine,
    stage: CurriculumStage,
    topic: &str,
    exercise: ExerciseKind,
) -> String {
    let prompt = match exercise {
        ExerciseKind::Definition => format!(
            "Eres maestro de SNGA. Etapa {}. Tema: {topic}. Define el tema en una frase clara de maximo 20 palabras.",
            stage.label()
        ),
        ExerciseKind::Paraphrase => format!(
            "Eres maestro de SNGA. Etapa {}. Tema: {topic}. Da dos parafrasis cortas separadas por ' | ', sin listas.",
            stage.label()
        ),
        ExerciseKind::QuestionAnswer => format!(
            "Eres maestro de SNGA. Etapa {}. Tema: {topic}. Escribe exactamente: Q: una pregunta breve | A: una respuesta breve.",
            stage.label()
        ),
        ExerciseKind::Analogy => format!(
            "Eres maestro de SNGA. Etapa {}. Tema: {topic}. Explica con una analogia simple en una frase corta.",
            stage.label()
        ),
        ExerciseKind::Correction => format!(
            "Eres maestro de SNGA. Etapa {}. Tema: {topic}. Escribe: Error: una idea incorrecta | Correccion: la idea correcta.",
            stage.label()
        ),
    };

    let context = LinguisticContext {
        user_prompt: prompt,
        inferred_intent: format!("ensenanza_{}", exercise.label()),
        geometric_projection: snga::simplicial::ConceptProjection {
            top_agents: Vec::new(),
        },
        memory_summary: "Gemma enseña lenguaje abierto; SNGA aprende patrones en la malla."
            .to_string(),
    };

    engine
        .generate(&context)
        .map(|response| response.text)
        .unwrap_or_else(|_| {
            format!(
                "SNGA aprende {topic} cuando una frase activa regiones geometricas que luego se consolidan por replay."
            )
        })
}

fn learn_lesson(
    network: &mut SimplicialNetwork,
    topic: &str,
    exercise: ExerciseKind,
    lesson: &str,
) {
    let topic_pattern = text_pattern(topic, network.agents.len());
    let lesson_pattern = text_pattern(lesson, network.agents.len());
    let exercise_pattern = text_pattern(exercise.label(), network.agents.len());
    let mut fused = topic_pattern.clone();
    fused.extend(lesson_pattern.iter().copied());
    fused.extend(exercise_pattern.iter().copied());
    fused.sort_unstable();
    fused.dedup();

    network.clear_activity();
    network.set_attention_goal(&lesson_pattern);
    network.inject_pattern(&topic_pattern, 1.15, 2);
    network.inject_pattern(&exercise_pattern, 0.85, 1);
    network.inject_pattern(&lesson_pattern, 0.95, 1);
    network.learn_transition(&topic_pattern, &lesson_pattern);
    network.learn_transition(&exercise_pattern, &lesson_pattern);
    network.reinforce_coactivation_if_useful(&fused, 0.07, 0.92);

    if let Some((question, answer)) = extract_qa_pair(lesson) {
        let question_pattern = text_pattern(&question, network.agents.len());
        let answer_pattern = text_pattern(&answer, network.agents.len());
        let mut qa_fused = question_pattern.clone();
        qa_fused.extend(answer_pattern.iter().copied());
        qa_fused.extend(topic_pattern.iter().copied());
        qa_fused.sort_unstable();
        qa_fused.dedup();
        network.learn_transition(&question_pattern, &answer_pattern);
        network.reinforce_coactivation_if_useful(&qa_fused, 0.08, 0.95);
    }

    for _ in 0..12 {
        network.step();
    }
    network.clear_attention_goal();
    network.clear_activity();
}

fn quiz(network: &SimplicialNetwork, topic: &str) -> (bool, f32, usize) {
    let topic_pattern = text_pattern(topic, network.agents.len());
    let predicted = network.predict_next_pattern(&topic_pattern, 1, 18);
    let confidence = predicted.iter().map(|(_, score)| *score).sum::<f32>() / 18.0;
    (
        predicted.len() >= 8 && confidence > 0.08,
        confidence,
        predicted.len(),
    )
}

fn maybe_advance(progress: &mut Progress) -> bool {
    if progress.stage == CurriculumStage::World {
        return false;
    }
    if progress.stage_lessons >= MIN_LESSONS_TO_ADVANCE && progress.passes >= PASSES_TO_ADVANCE {
        progress.stage = progress.stage.next();
        progress.stage_lessons = 0;
        progress.passes = 0;
        progress.topic_idx = 0;
        true
    } else {
        false
    }
}

fn extract_qa_pair(lesson: &str) -> Option<(String, String)> {
    let normalized = lesson.replace('\n', " ");
    if let Some((question_part, answer_part)) = normalized.split_once("| A:") {
        let question = question_part
            .trim()
            .trim_start_matches("Q:")
            .trim()
            .to_string();
        let answer = answer_part.trim().to_string();
        if !question.is_empty() && !answer.is_empty() {
            return Some((question, answer));
        }
    }
    if let Some((wrong, correction)) = normalized.split_once("| Correccion:") {
        let question = wrong.trim().trim_start_matches("Error:").trim().to_string();
        let answer = correction.trim().to_string();
        if !question.is_empty() && !answer.is_empty() {
            return Some((question, answer));
        }
    }
    None
}

fn save_all(network: &SimplicialNetwork, progress: &Progress, label: &str) {
    match network.save_persistent_state(STATE_PATH) {
        Ok(report) => {
            if let Err(err) = save_progress(progress) {
                eprintln!("{label}: estado guardado, progreso fallo: {err}");
            }
            println!(
                "{label}: saved agents={} edges={} causal={} lessons={} stage={}",
                report.agents,
                report.edges,
                report.causal_edges,
                progress.total_lessons,
                progress.stage.label()
            );
        }
        Err(err) => eprintln!("{label}: fallo guardando estado: {err}"),
    }
}

fn load_progress() -> Option<Progress> {
    let text = fs::read_to_string(PROGRESS_PATH).ok()?;
    let mut progress = Progress {
        stage: CurriculumStage::Language,
        topic_idx: 0,
        total_lessons: 0,
        stage_lessons: 0,
        quizzes: 0,
        passes: 0,
    };
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

fn save_progress(progress: &Progress) -> std::io::Result<()> {
    if let Some(parent) = Path::new(PROGRESS_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        PROGRESS_PATH,
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

fn text_pattern(text: &str, nodes: usize) -> Vec<usize> {
    text.bytes()
        .enumerate()
        .map(|(i, byte)| ((byte as usize * 41) + i * 67 + text.len() * 13) % nodes)
        .collect()
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
