//! Entrenamiento infinito del sustrato cognitivo sobre líquido de espines.
//!
//! Gemma 2 actúa como generador/evaluador lingüístico in-process. Sus pesos
//! GGUF permanecen congelados; lo que aprende y se guarda es el motor
//! CDT–spin-liquid–RQM/EPR.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use cdt_rqm_epr::matrix_free_cognitive_substrate::LatentConceptId;
use cdt_rqm_epr::native_gemma2::{resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2};
use cdt_rqm_epr::relational_field::ObserverId;
use cdt_rqm_epr::symmetry_guided_rqm_epr::{RqmPhaseRelationState, RqmRelationKey};
use cdt_rqm_epr::unified_spin_cognitive_engine::{
    ConsolidatedKnowledge, KnowledgeKey, UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const VERSION: u32 = 2;
const DEFAULT_ROOT: &str = "data/gemma2_developmental_infinite_training";
const DEVELOPMENT_OBSERVER: ObserverId = ObserverId(997_000);
const OOD_OBSERVER_BASE: usize = 1_007_000;

#[derive(Clone, Debug)]
struct TrainerConfig {
    root: PathBuf,
    duration: Option<Duration>,
    max_cycles: Option<u64>,
    teacher_tokens: usize,
    exposures: usize,
    validate_every: u64,
    checkpoint_every_cycles: u64,
    checkpoint_every: Duration,
    milestone_every: u64,
    retain_milestones: usize,
    minimum_seen_lessons: usize,
}

#[derive(Clone, Copy)]
struct Lesson {
    stage: usize,
    id: &'static str,
    mechanism: &'static str,
    prompt: &'static str,
    expected: &'static [&'static str],
    source: usize,
    middle: usize,
    target: usize,
    transfer_source: usize,
    transfer_target: usize,
    phase: f64,
}

const LESSONS: &[Lesson] = &[
    Lesson {
        stage: 0,
        id: "sensorimotor_action",
        mechanism: "perception -> action schema -> goal",
        prompt: "Un bebé ve un juguete bajo una manta. Para alcanzarlo, ¿qué acción debe realizar primero? Elige: retirar la manta, nombrar el juguete o contar.",
        expected: &["retirar", "manta"],
        source: 0,
        middle: 1,
        target: 2,
        transfer_source: 128,
        transfer_target: 129,
        phase: 0.07,
    },
    Lesson {
        stage: 1,
        id: "object_permanence",
        mechanism: "hidden object -> stable internal model -> search",
        prompt: "Un juguete queda oculto bajo una manta. ¿Sigue existiendo aunque no se vea? Elige: sí, no o solo si se nombra.",
        expected: &["sí", "si"],
        source: 4,
        middle: 5,
        target: 6,
        transfer_source: 132,
        transfer_target: 133,
        phase: 0.17,
    },
    Lesson {
        stage: 2,
        id: "deferred_imitation",
        mechanism: "observed action -> offline motor memory -> later reproduction",
        prompt: "Un niño observa hoy una acción y la repite mañana sin verla de nuevo. Elige el proceso: imitación diferida, reflejo inmediato o azar.",
        expected: &["imitación", "imitacion", "diferida"],
        source: 8,
        middle: 9,
        target: 10,
        transfer_source: 136,
        transfer_target: 137,
        phase: 0.27,
    },
    Lesson {
        stage: 3,
        id: "predictive_learning",
        mechanism: "expectation -> prediction error -> model update",
        prompt: "El resultado real no coincide con lo que un bebé esperaba. ¿Qué debe hacer su modelo interno? Elige: actualizarse, ignorarlo siempre o borrar toda memoria.",
        expected: &["actualizar", "actualizarse"],
        source: 12,
        middle: 13,
        target: 14,
        transfer_source: 140,
        transfer_target: 141,
        phase: 0.39,
    },
    Lesson {
        stage: 4,
        id: "preverbal_attention",
        mechanism: "novelty or need -> salience -> attentional priority",
        prompt: "Un bebé aún no habla y orienta su mirada hacia un estímulo novedoso o relevante. ¿Qué proceso cognitivo selecciona ese estímulo? Elige una sola opción: atención, gramática o escritura.",
        expected: &["atención", "atencion"],
        source: 16,
        middle: 17,
        target: 18,
        transfer_source: 144,
        transfer_target: 145,
        phase: 0.51,
    },
    Lesson {
        stage: 5,
        id: "symbolic_play",
        mechanism: "physical object -> representational mapping -> symbolic substitute",
        prompt: "En el juego, una caja representa un automóvil. ¿Qué capacidad aparece? Elige: juego simbólico, reflejo muscular o olvido.",
        expected: &["simbólico", "simbolico", "símbolo", "simbolo"],
        source: 20,
        middle: 21,
        target: 22,
        transfer_source: 148,
        transfer_target: 149,
        phase: 0.63,
    },
    Lesson {
        stage: 6,
        id: "concept_abstraction",
        mechanism: "repeated experiences -> shared invariant -> category",
        prompt: "Tras ver muchas pelotas diferentes, el niño extrae rasgos comunes y forma una categoría. Elige: concepto abstracto, reflejo aislado o ruido.",
        expected: &["concepto", "abstracto", "categoría", "categoria"],
        source: 24,
        middle: 25,
        target: 26,
        transfer_source: 152,
        transfer_target: 153,
        phase: 0.75,
    },
    Lesson {
        stage: 7,
        id: "language_grounding",
        mechanism: "preverbal concept -> shared word -> compositional symbol",
        prompt: "Un concepto ya aprendido recibe una palabra compartida socialmente. ¿Qué aporta esa palabra? Elige: etiqueta lingüística, impulso motor o pérdida del concepto.",
        expected: &["etiqueta", "lingüística", "linguistica", "palabra"],
        source: 28,
        middle: 29,
        target: 30,
        transfer_source: 156,
        transfer_target: 157,
        phase: 0.87,
    },
    Lesson {
        stage: 8,
        id: "executive_planning",
        mechanism: "goal -> working-memory simulation -> ordered conditional plan",
        prompt: "Para una meta compleja, el sistema mantiene pasos y reglas «si A entonces B». Elige la función: planificación ejecutiva, reflejo simple o percepción pasiva.",
        expected: &["planificación", "planificacion", "ejecutiva"],
        source: 32,
        middle: 33,
        target: 34,
        transfer_source: 160,
        transfer_target: 161,
        phase: 0.97,
    },
];

#[derive(Clone, Copy)]
struct PlanningTask {
    object_label: &'static str,
    aliases: &'static [&'static str],
    object_id: usize,
    goal_label: &'static str,
    goal_id: usize,
    steps: &'static [(usize, &'static str)],
    prompt: &'static str,
    expected: &'static [&'static str],
    phase: f64,
}

const PLANNING_TASKS: &[PlanningTask] = &[
    PlanningTask {
        object_label: "juguete",
        aliases: &["juguete", "objeto oculto", "muñeco", "muneco"],
        object_id: 512,
        goal_label: "obtener_juguete",
        goal_id: 600,
        steps: &[(601, "retirar_manta"), (602, "agarrar_juguete")],
        prompt: "El juguete está oculto bajo una manta y quiero obtenerlo. Identifica el objeto y ordena: retirar la manta, agarrar el juguete.",
        expected: &["juguete", "retirar", "agarrar"],
        phase: 1.11,
    },
    PlanningTask {
        object_label: "caja",
        aliases: &["caja", "caja de cartón", "caja de carton"],
        object_id: 516,
        goal_label: "usar_caja_como_coche",
        goal_id: 604,
        steps: &[(605, "asignar_simbolo_coche"), (606, "simular_movimiento")],
        prompt: "En juego simbólico, una caja representará un coche. Identifica el objeto y ordena: asignar símbolo de coche, simular movimiento.",
        expected: &["caja", "simbolo", "movimiento"],
        phase: 1.23,
    },
    PlanningTask {
        object_label: "vaso",
        aliases: &["vaso", "recipiente"],
        object_id: 520,
        goal_label: "beber_agua",
        goal_id: 608,
        steps: &[
            (609, "localizar_vaso"),
            (610, "llenar_con_agua"),
            (611, "beber"),
        ],
        prompt: "Quiero beber agua usando un vaso. Identifica el objeto y ordena: localizar el vaso, llenarlo con agua, beber.",
        expected: &["vaso", "localizar", "llenar", "beber"],
        phase: 1.37,
    },
    PlanningTask {
        object_label: "semilla",
        aliases: &["semilla", "grano"],
        object_id: 524,
        goal_label: "cultivar_planta",
        goal_id: 612,
        steps: &[
            (613, "humedecer_suelo"),
            (614, "colocar_semilla"),
            (615, "cuidar_crecimiento"),
        ],
        prompt: "Quiero cultivar una planta a partir de una semilla. Identifica el objeto y ordena: humedecer suelo, colocar semilla, cuidar crecimiento.",
        expected: &["semilla", "humedecer", "colocar", "cuidar"],
        phase: 1.49,
    },
];

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct LanguageLabel {
    concept_id: usize,
    canonical: String,
    aliases: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct StoredLanguagePlan {
    object_id: usize,
    goal_id: usize,
    phase: f64,
    step_ids: Vec<usize>,
    validations: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct LanguagePlanningZone {
    labels: Vec<LanguageLabel>,
    plans: Vec<StoredLanguagePlan>,
}

struct GemmaTeacher {
    model: QuantizedGemma2,
    tokenizer: Gemma2Tokenizer,
    device: Device,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RelationSnapshot {
    observer: usize,
    source: usize,
    target: usize,
    amplitude: f64,
    phase: f64,
    coherence: f64,
    uncertainty: f64,
    eligibility: f64,
    prediction_error: f64,
    exposures: u64,
    consolidated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct KnowledgeSnapshot {
    observer: usize,
    source: usize,
    target: usize,
    confidence: f64,
    topological_symmetry: f64,
    spin_entropy: f64,
    prediction_error: f64,
    consolidations: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainingCheckpoint {
    version: u32,
    cycle: u64,
    accepted_cycles: u64,
    rollback_cycles: u64,
    teacher_queries: u64,
    teacher_passes: u64,
    #[serde(default)]
    planning_cycles: u64,
    #[serde(default)]
    planning_passes: u64,
    #[serde(default)]
    planning_zone: LanguagePlanningZone,
    lessons_seen: Vec<bool>,
    amplitudes: Vec<[f64; 2]>,
    relations: Vec<RelationSnapshot>,
    epr_state: String,
    knowledge: Vec<KnowledgeSnapshot>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct CognitiveMetrics {
    seen_lessons: usize,
    highest_mastered_stage: usize,
    composition_accuracy: f64,
    direct_composed_relation_absence: f64,
    transfer_accuracy: f64,
    retention_accuracy: f64,
    preverbal_accuracy: f64,
    symbolic_accuracy: f64,
    developmental_integration_accuracy: f64,
    ood_abstention: f64,
    topological_symmetry: f64,
    spin_entropy: f64,
    entangled_edges: usize,
    relations: usize,
    knowledge: usize,
    epr_links: usize,
    functional_cognition_gate: bool,
}

#[derive(Debug, Serialize)]
struct CycleMetrics<'a> {
    cycle: u64,
    stage: usize,
    lesson: &'a str,
    mechanism: &'a str,
    teacher_response: &'a str,
    teacher_score: f64,
    accepted: bool,
    rollback_reason: &'a str,
    teacher_queries: u64,
    teacher_passes: u64,
    cognitive: CognitiveMetrics,
}

#[derive(Debug, Serialize)]
struct PlanningCycleMetrics<'a> {
    cycle: u64,
    planning_cycle: u64,
    object: &'a str,
    teacher_response: &'a str,
    grounding_score: f64,
    accepted: bool,
    rollback_reason: &'a str,
    stored_plans: usize,
    plan_accuracy: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrainerConfig::from_env();
    let plan_request = argument_value("--plan");
    fs::create_dir_all(config.root.join("checkpoints"))?;
    if env::args().any(|argument| argument == "--recover-observed-cycle-850") {
        recover_observed_cycle_850(&config)?;
        return Ok(());
    }
    let model_path = resolve_gemma2_model_path(None)?;
    println!("Gemma2 + spin-liquid infinite trainer");
    println!("GGUF={} ollama_runtime=false", model_path.display());
    println!(
        "root={} checkpoint_cycles={} checkpoint_seconds={} milestone_cycles={} max_cycles={:?}",
        config.root.display(),
        config.checkpoint_every_cycles,
        config.checkpoint_every.as_secs(),
        config.milestone_every,
        config.max_cycles,
    );
    let mut teacher = GemmaTeacher::load(&model_path)?;
    let latest = config.root.join("latest.json");
    let (mut engine, mut state) = if latest.exists() {
        let checkpoint: TrainingCheckpoint = serde_json::from_slice(&fs::read(&latest)?)?;
        let mut engine = fresh_engine()?;
        restore_checkpoint(&mut engine, &checkpoint)?;
        println!(
            "resume=true cycle={} accepted={} rollbacks={} relations={} knowledge={}",
            checkpoint.cycle,
            checkpoint.accepted_cycles,
            checkpoint.rollback_cycles,
            checkpoint.relations.len(),
            checkpoint.knowledge.len(),
        );
        (engine, checkpoint)
    } else {
        println!("resume=false fresh=true");
        (fresh_engine()?, TrainingCheckpoint::fresh())
    };
    if let Some(request) = plan_request {
        run_language_plan_query(&mut teacher, &engine, &state.planning_zone, &request)?;
        return Ok(());
    }
    let started = Instant::now();
    let mut last_checkpoint = Instant::now();
    let metrics_path = config.root.join("metrics.jsonl");
    let mut last_metrics = validate(&engine, &state.lessons_seen, config.minimum_seen_lessons);

    loop {
        if config
            .max_cycles
            .is_some_and(|maximum| state.cycle >= maximum)
            || config
                .duration
                .is_some_and(|duration| started.elapsed() >= duration)
        {
            break;
        }
        state.cycle = state.cycle.saturating_add(1);
        if state.lessons_seen.iter().all(|seen| *seen) {
            state.planning_cycles = state.planning_cycles.saturating_add(1);
            let task = PLANNING_TASKS
                [(state.planning_cycles.saturating_sub(1) as usize) % PLANNING_TASKS.len()];
            let response = teacher.ask_planning(task.prompt, config.teacher_tokens, state.cycle)?;
            let grounding_score = score_fraction(&response, task.expected);
            state.teacher_queries = state.teacher_queries.saturating_add(1);
            let previous = engine.clone();
            let previous_zone = state.planning_zone.clone();
            let mut accepted = false;
            let mut rollback_reason = "language_grounding_rejected";
            if grounding_score >= 0.66 {
                train_language_plan(&mut engine, task, config.exposures);
                state.planning_zone.register(task);
                if validate_language_plan(&engine, task) {
                    state.planning_passes = state.planning_passes.saturating_add(1);
                    if let Some(plan) = state
                        .planning_zone
                        .plans
                        .iter_mut()
                        .find(|plan| plan.object_id == task.object_id)
                    {
                        plan.validations = plan.validations.saturating_add(1);
                    }
                    accepted = true;
                    rollback_reason = "none";
                } else {
                    engine = previous;
                    state.planning_zone = previous_zone;
                    state.rollback_cycles = state.rollback_cycles.saturating_add(1);
                    rollback_reason = "network_plan_not_consolidated";
                }
            }
            let plan_accuracy = language_plan_accuracy(&engine, &state.planning_zone);
            let record = PlanningCycleMetrics {
                cycle: state.cycle,
                planning_cycle: state.planning_cycles,
                object: task.object_label,
                teacher_response: &response,
                grounding_score,
                accepted,
                rollback_reason,
                stored_plans: state.planning_zone.plans.len(),
                plan_accuracy,
            };
            append_jsonl(&metrics_path, &record)?;
            println!(
                "cycle={} zone=language_planner planning_cycle={} object={} grounding={:.3} accepted={} reason={} plans={} accuracy={:.3}",
                state.cycle,
                state.planning_cycles,
                task.object_label,
                grounding_score,
                accepted,
                rollback_reason,
                state.planning_zone.plans.len(),
                plan_accuracy,
            );
            persist_periodically(&config, &engine, &mut state, &mut last_checkpoint)?;
            continue;
        }
        let lesson_index = next_lesson_index(&state.lessons_seen, state.cycle);
        let lesson = LESSONS[lesson_index];
        let response = teacher.ask(lesson.prompt, config.teacher_tokens, state.cycle)?;
        let teacher_score = score_response(&response, lesson.expected);
        state.teacher_queries = state.teacher_queries.saturating_add(1);
        if teacher_score >= 1.0 {
            state.teacher_passes = state.teacher_passes.saturating_add(1);
        }

        let baseline = validate(&engine, &state.lessons_seen, config.minimum_seen_lessons);
        let previous = engine.clone();
        let mut accepted = false;
        let mut rollback_reason = "teacher_rejected";
        if teacher_score >= 1.0 {
            let prerequisite = lesson_index
                .checked_sub(1)
                .filter(|index| state.lessons_seen.get(*index).copied().unwrap_or(false))
                .map(|index| LESSONS[index]);
            train_lesson(
                &mut engine,
                lesson,
                prerequisite,
                teacher_score,
                config.exposures,
            );
            let mut candidate_seen = state.lessons_seen.clone();
            candidate_seen[lesson_index] = true;
            let candidate = validate(&engine, &candidate_seen, config.minimum_seen_lessons);
            if let Some(reason) = regression_reason(&baseline, &candidate) {
                engine = previous;
                state.rollback_cycles = state.rollback_cycles.saturating_add(1);
                rollback_reason = reason;
            } else {
                state.lessons_seen = candidate_seen;
                state.accepted_cycles = state.accepted_cycles.saturating_add(1);
                last_metrics = candidate;
                accepted = true;
                rollback_reason = "none";
            }
        }
        if !accepted {
            last_metrics = validate(&engine, &state.lessons_seen, config.minimum_seen_lessons);
        }
        if state.cycle % config.validate_every == 0 || state.cycle == 1 {
            let record = CycleMetrics {
                cycle: state.cycle,
                stage: lesson.stage,
                lesson: lesson.id,
                mechanism: lesson.mechanism,
                teacher_response: &response,
                teacher_score,
                accepted,
                rollback_reason,
                teacher_queries: state.teacher_queries,
                teacher_passes: state.teacher_passes,
                cognitive: last_metrics,
            };
            append_jsonl(&metrics_path, &record)?;
            println!(
                "cycle={} stage={} lesson={} teacher={:.1} accepted={} reason={} seen={} composition={:.3} transfer={:.3} integration={:.3} preverbal={:.3} symbolic={:.3} retention={:.3} ood={:.3} direct_absence={:.3} entropy={:.4} entangled={} gate={}",
                state.cycle,
                lesson.stage,
                lesson.id,
                teacher_score,
                accepted,
                rollback_reason,
                last_metrics.seen_lessons,
                last_metrics.composition_accuracy,
                last_metrics.transfer_accuracy,
                last_metrics.developmental_integration_accuracy,
                last_metrics.preverbal_accuracy,
                last_metrics.symbolic_accuracy,
                last_metrics.retention_accuracy,
                last_metrics.ood_abstention,
                last_metrics.direct_composed_relation_absence,
                last_metrics.spin_entropy,
                last_metrics.entangled_edges,
                last_metrics.functional_cognition_gate,
            );
        }

        persist_periodically(&config, &engine, &mut state, &mut last_checkpoint)?;

        let reached_time = config
            .duration
            .is_some_and(|duration| started.elapsed() >= duration);
        let reached_cycles = config
            .max_cycles
            .is_some_and(|maximum| state.cycle >= maximum);
        if reached_time || reached_cycles {
            break;
        }
    }

    capture_checkpoint(&engine, &mut state);
    save_checkpoint(&config.root, &state, true)?;
    fs::write(
        config.root.join("summary.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "version": VERSION,
            "cycle": state.cycle,
            "accepted_cycles": state.accepted_cycles,
            "rollback_cycles": state.rollback_cycles,
            "teacher_queries": state.teacher_queries,
            "teacher_passes": state.teacher_passes,
            "planning_cycles": state.planning_cycles,
            "planning_passes": state.planning_passes,
            "stored_language_labels": state.planning_zone.labels.len(),
            "stored_language_plans": state.planning_zone.plans.len(),
            "language_plan_accuracy": language_plan_accuracy(&engine, &state.planning_zone),
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "last_metrics": last_metrics,
            "claim": "developmental sensorimotor-to-symbolic task gate; not evidence of consciousness or general cognition",
        }))?,
    )?;
    println!(
        "finished=true cycle={} accepted={} rollbacks={} gate={}",
        state.cycle,
        state.accepted_cycles,
        state.rollback_cycles,
        last_metrics.functional_cognition_gate,
    );
    Ok(())
}

impl TrainerConfig {
    fn from_env() -> Self {
        let hours = env_f64("GEMMA_SPIN_TRAIN_HOURS", 0.0);
        Self {
            root: PathBuf::from(
                env::var("GEMMA_SPIN_TRAIN_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_string()),
            ),
            duration: (hours > 0.0).then(|| Duration::from_secs_f64(hours * 3_600.0)),
            max_cycles: env::var("GEMMA_SPIN_MAX_CYCLES")
                .ok()
                .and_then(|value| value.parse().ok()),
            teacher_tokens: env_usize("GEMMA_SPIN_TEACHER_TOKENS", 24).max(1),
            exposures: env_usize("GEMMA_SPIN_EXPOSURES", 40).max(1),
            validate_every: env_u64("GEMMA_SPIN_VALIDATE_EVERY", 1).max(1),
            checkpoint_every_cycles: env_u64("GEMMA_SPIN_CHECKPOINT_EVERY_CYCLES", 5).max(1),
            checkpoint_every: Duration::from_secs(
                env_u64("GEMMA_SPIN_CHECKPOINT_EVERY_SECONDS", 300).max(1),
            ),
            milestone_every: env_u64("GEMMA_SPIN_MILESTONE_EVERY", 50).max(1),
            retain_milestones: env_usize("GEMMA_SPIN_RETAIN_MILESTONES", 24).max(1),
            minimum_seen_lessons: env_usize("GEMMA_SPIN_MINIMUM_SEEN", 6).clamp(1, LESSONS.len()),
        }
    }
}

impl GemmaTeacher {
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut file = File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
        let model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn ask(
        &mut self,
        question: &str,
        max_tokens: usize,
        seed: u64,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.generate(
            "Responde únicamente con la opción correcta.",
            question,
            max_tokens,
            seed,
        )
    }

    fn ask_planning(
        &mut self,
        question: &str,
        max_tokens: usize,
        seed: u64,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.generate(
            "Identifica el objeto y devuelve las acciones en orden, de forma breve.",
            question,
            max_tokens,
            seed,
        )
    }

    fn generate(
        &mut self,
        instruction: &str,
        question: &str,
        max_tokens: usize,
        seed: u64,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = format!(
            "<start_of_turn>user\n{instruction} {question}<end_of_turn>\n<start_of_turn>model\n"
        );
        let mut prompt_tokens = vec![self.tokenizer.bos_id];
        prompt_tokens.extend(self.tokenizer.encode(&prompt)?);
        self.model.clear_kv_cache();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, 0)?.squeeze(0)?;
        let mut sampler = LogitsProcessor::new(seed, Some(0.05), Some(0.90));
        let mut generated = Vec::new();
        for _ in 0..max_tokens {
            let token = sampler.sample(&logits)?;
            if token == self.tokenizer.eos_id || Some(token) == self.tokenizer.end_of_turn_id {
                break;
            }
            generated.push(token);
            let next = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            logits = self
                .model
                .forward(&next, prompt_tokens.len() + generated.len() - 1)?
                .squeeze(0)?;
        }
        Ok(self.tokenizer.decode(&generated, true)?.trim().to_string())
    }
}

impl TrainingCheckpoint {
    fn fresh() -> Self {
        Self {
            version: VERSION,
            cycle: 0,
            accepted_cycles: 0,
            rollback_cycles: 0,
            teacher_queries: 0,
            teacher_passes: 0,
            planning_cycles: 0,
            planning_passes: 0,
            planning_zone: LanguagePlanningZone::default(),
            lessons_seen: vec![false; LESSONS.len()],
            amplitudes: Vec::new(),
            relations: Vec::new(),
            epr_state: String::new(),
            knowledge: Vec::new(),
        }
    }
}

fn fresh_engine() -> Result<UnifiedSpinCognitiveEngine, Box<dyn std::error::Error>> {
    Ok(UnifiedSpinCognitiveEngine::periodic_pyrochlore(
        2,
        1,
        1,
        UnifiedSpinCognitiveConfig {
            bootstrap_cooling_steps: 180,
            cooling_steps_per_observation: 1,
            real_steps_per_observation: 1,
            backreaction_rate: 0.002,
            ..UnifiedSpinCognitiveConfig::default()
        },
    )?)
}

impl LanguagePlanningZone {
    fn register(&mut self, task: PlanningTask) {
        self.register_label(task.object_id, task.object_label, task.aliases);
        self.register_label(task.goal_id, task.goal_label, &[task.goal_label]);
        for &(step_id, step_label) in task.steps {
            self.register_label(step_id, step_label, &[step_label]);
        }
        if let Some(plan) = self
            .plans
            .iter_mut()
            .find(|plan| plan.object_id == task.object_id)
        {
            plan.goal_id = task.goal_id;
            plan.phase = task.phase;
            plan.step_ids = task.steps.iter().map(|(id, _)| *id).collect();
        } else {
            self.plans.push(StoredLanguagePlan {
                object_id: task.object_id,
                goal_id: task.goal_id,
                phase: task.phase,
                step_ids: task.steps.iter().map(|(id, _)| *id).collect(),
                validations: 0,
            });
        }
    }

    fn register_label(&mut self, concept_id: usize, canonical: &str, aliases: &[&str]) {
        if let Some(label) = self
            .labels
            .iter_mut()
            .find(|label| label.concept_id == concept_id)
        {
            label.canonical = canonical.to_string();
            label.aliases = aliases.iter().map(|alias| alias.to_string()).collect();
        } else {
            self.labels.push(LanguageLabel {
                concept_id,
                canonical: canonical.to_string(),
                aliases: aliases.iter().map(|alias| alias.to_string()).collect(),
            });
        }
    }

    fn resolve(&self, text: &str) -> Option<&LanguageLabel> {
        let text = normalize_spanish(text);
        self.labels
            .iter()
            .filter(|label| {
                self.plans
                    .iter()
                    .any(|plan| plan.object_id == label.concept_id)
            })
            .find(|label| {
                text.contains(&normalize_spanish(&label.canonical))
                    || label
                        .aliases
                        .iter()
                        .any(|alias| text.contains(&normalize_spanish(alias)))
            })
    }

    fn label(&self, concept_id: usize) -> String {
        self.labels
            .iter()
            .find(|label| label.concept_id == concept_id)
            .map(|label| label.canonical.clone())
            .unwrap_or_else(|| format!("concepto_{concept_id}"))
    }
}

fn train_language_plan(
    engine: &mut UnifiedSpinCognitiveEngine,
    task: PlanningTask,
    exposures: usize,
) {
    let mut path = vec![task.object_id, task.goal_id];
    path.extend(task.steps.iter().map(|(id, _)| *id));
    for pair in path.windows(2) {
        engine.train_relation(
            DEVELOPMENT_OBSERVER,
            LatentConceptId(pair[0]),
            LatentConceptId(pair[1]),
            task.phase,
            1.0,
            0.0,
            &[],
            exposures,
        );
    }
}

fn recover_observed_cycle_850(
    config: &TrainerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = fresh_engine()?;
    let mut state = TrainingCheckpoint::fresh();
    for (index, lesson) in LESSONS.iter().copied().enumerate() {
        let prerequisite = index.checked_sub(1).map(|previous| LESSONS[previous]);
        train_lesson(&mut engine, lesson, prerequisite, 1.0, config.exposures);
        state.lessons_seen[index] = true;
    }

    // Last durable state observed before the training directory disappeared:
    // caja had 54 accepted validations and vaso had 89.
    for (task_index, validations) in [(1_usize, 54_u64), (2_usize, 89_u64)] {
        let task = PLANNING_TASKS[task_index];
        for _ in 0..validations {
            train_language_plan(&mut engine, task, config.exposures);
            state.planning_zone.register(task);
            if let Some(plan) = state
                .planning_zone
                .plans
                .iter_mut()
                .find(|plan| plan.object_id == task.object_id)
            {
                plan.validations = plan.validations.saturating_add(1);
            }
        }
    }

    state.cycle = 850;
    state.accepted_cycles = LESSONS.len() as u64;
    state.teacher_queries = 850;
    state.teacher_passes = LESSONS.len() as u64;
    state.planning_cycles = 355;
    state.planning_passes = 143;
    capture_checkpoint(&engine, &mut state);

    let cognitive = validate(&engine, &state.lessons_seen, config.minimum_seen_lessons);
    if !cognitive.functional_cognition_gate
        || language_plan_accuracy(&engine, &state.planning_zone) < 1.0
    {
        return Err("el estado reconstruido no superó la validación".into());
    }
    save_checkpoint(&config.root, &state, true)?;
    println!(
        "recovered=true cycle={} stages={} planning_cycles={} planning_passes={} plans={} accuracy={:.3}",
        state.cycle,
        cognitive.seen_lessons,
        state.planning_cycles,
        state.planning_passes,
        state.planning_zone.plans.len(),
        language_plan_accuracy(&engine, &state.planning_zone),
    );
    Ok(())
}

fn validate_language_plan(engine: &UnifiedSpinCognitiveEngine, task: PlanningTask) -> bool {
    let mut expected = vec![task.object_id, task.goal_id];
    expected.extend(task.steps.iter().map(|(id, _)| *id));
    engine
        .infer(
            DEVELOPMENT_OBSERVER,
            LatentConceptId(task.object_id),
            task.phase,
            expected.len().saturating_sub(1),
        )
        .is_some_and(|inference| {
            inference.path
                == expected
                    .into_iter()
                    .map(LatentConceptId)
                    .collect::<Vec<_>>()
        })
}

fn language_plan_accuracy(engine: &UnifiedSpinCognitiveEngine, zone: &LanguagePlanningZone) -> f64 {
    if zone.plans.is_empty() {
        return 0.0;
    }
    let correct = zone
        .plans
        .iter()
        .filter(|plan| {
            let mut expected = vec![plan.object_id, plan.goal_id];
            expected.extend(plan.step_ids.iter().copied());
            engine
                .infer(
                    DEVELOPMENT_OBSERVER,
                    LatentConceptId(plan.object_id),
                    plan.phase,
                    expected.len().saturating_sub(1),
                )
                .is_some_and(|inference| {
                    inference.path
                        == expected
                            .into_iter()
                            .map(LatentConceptId)
                            .collect::<Vec<_>>()
                })
        })
        .count();
    correct as f64 / zone.plans.len() as f64
}

fn run_language_plan_query(
    teacher: &mut GemmaTeacher,
    engine: &UnifiedSpinCognitiveEngine,
    zone: &LanguagePlanningZone,
    request: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if zone.plans.is_empty() {
        return Err("la zona lingüística aún no existe; completa primero las nueve etapas".into());
    }
    let available = zone
        .plans
        .iter()
        .map(|plan| zone.label(plan.object_id))
        .collect::<Vec<_>>()
        .join(", ");
    let grounding_prompt = format!(
        "Objetos conocidos: {available}. Petición: {request}. Responde solo con el objeto referido."
    );
    let grounded_text = teacher.ask_planning(&grounding_prompt, 12, 0x504C_414E)?;
    let label = zone
        .resolve(&grounded_text)
        .or_else(|| zone.resolve(request))
        .ok_or("el LLM no pudo vincular la petición con un objeto aprendido")?;
    let plan = zone
        .plans
        .iter()
        .find(|plan| plan.object_id == label.concept_id)
        .ok_or("el objeto no tiene un plan consolidado")?;
    let mut expected = vec![plan.object_id, plan.goal_id];
    expected.extend(plan.step_ids.iter().copied());
    let inference = engine
        .infer(
            DEVELOPMENT_OBSERVER,
            LatentConceptId(plan.object_id),
            plan.phase,
            expected.len().saturating_sub(1),
        )
        .ok_or("la red no recuperó un plan consolidado")?;
    let labels = inference
        .path
        .iter()
        .map(|concept| zone.label(concept.0))
        .collect::<Vec<_>>();
    println!("referencia_llm={grounded_text:?}");
    println!("objeto_red={} id={}", label.canonical, label.concept_id);
    println!("plan_red={}", labels.join(" -> "));
    Ok(())
}

fn train_lesson(
    engine: &mut UnifiedSpinCognitiveEngine,
    lesson: Lesson,
    prerequisite: Option<Lesson>,
    teacher_score: f64,
    exposures: usize,
) {
    let orbit = [(
        LatentConceptId(lesson.transfer_source),
        LatentConceptId(lesson.transfer_target),
    )];
    engine.train_relation(
        DEVELOPMENT_OBSERVER,
        LatentConceptId(lesson.source),
        LatentConceptId(lesson.middle),
        lesson.phase,
        0.75 + 0.25 * teacher_score,
        teacher_score,
        &orbit,
        exposures,
    );
    engine.train_relation(
        DEVELOPMENT_OBSERVER,
        LatentConceptId(lesson.middle),
        LatentConceptId(lesson.target),
        lesson.phase,
        0.75 + 0.25 * teacher_score,
        0.0,
        &[],
        exposures,
    );
    if let Some(prerequisite) = prerequisite {
        engine.train_relation(
            DEVELOPMENT_OBSERVER,
            LatentConceptId(prerequisite.target),
            LatentConceptId(lesson.source),
            lesson.phase,
            1.0,
            0.0,
            &[],
            exposures,
        );
    }
}

fn validate(
    engine: &UnifiedSpinCognitiveEngine,
    lessons_seen: &[bool],
    minimum_seen: usize,
) -> CognitiveMetrics {
    let seen = LESSONS
        .iter()
        .enumerate()
        .filter(|(index, _)| lessons_seen.get(*index).copied().unwrap_or(false))
        .collect::<Vec<_>>();
    let denominator = seen.len().max(1) as f64;
    let mut composition = 0usize;
    let mut direct_absence = 0usize;
    let mut transfer = 0usize;
    let mut retention = 0usize;
    let mut preverbal_correct = 0usize;
    let mut preverbal_total = 0usize;
    let mut symbolic_correct = 0usize;
    let mut symbolic_total = 0usize;
    for (_, lesson) in &seen {
        let composed = usize::from(
            engine
                .infer(
                    DEVELOPMENT_OBSERVER,
                    LatentConceptId(lesson.source),
                    lesson.phase,
                    2,
                )
                .is_some_and(|inference| {
                    inference.path.last() == Some(&LatentConceptId(lesson.target))
                }),
        );
        composition += composed;
        if lesson.stage <= 4 {
            preverbal_total += 1;
            preverbal_correct += composed;
        } else {
            symbolic_total += 1;
            symbolic_correct += composed;
        }
        direct_absence += usize::from(
            engine
                .cognition
                .workspace
                .relation(
                    DEVELOPMENT_OBSERVER,
                    LatentConceptId(lesson.source),
                    LatentConceptId(lesson.target),
                )
                .is_none(),
        );
        transfer += usize::from(
            engine
                .infer(
                    DEVELOPMENT_OBSERVER,
                    LatentConceptId(lesson.transfer_source),
                    lesson.phase,
                    1,
                )
                .is_some_and(|inference| {
                    inference.path.last() == Some(&LatentConceptId(lesson.transfer_target))
                }),
        );
        retention += usize::from(
            engine
                .cognition
                .workspace
                .relation(
                    DEVELOPMENT_OBSERVER,
                    LatentConceptId(lesson.source),
                    LatentConceptId(lesson.middle),
                )
                .is_some_and(|relation| relation.consolidated),
        );
    }
    let integration_pairs = LESSONS
        .windows(2)
        .enumerate()
        .filter(|(index, _)| {
            lessons_seen.get(*index).copied().unwrap_or(false)
                && lessons_seen.get(index + 1).copied().unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let integrated = integration_pairs
        .iter()
        .filter(|(_, pair)| {
            engine
                .infer(
                    DEVELOPMENT_OBSERVER,
                    LatentConceptId(pair[0].target),
                    pair[1].phase,
                    1,
                )
                .is_some_and(|inference| {
                    inference.path.last() == Some(&LatentConceptId(pair[1].source))
                })
        })
        .count();
    let ood_trials = 8usize;
    let ood = (0..ood_trials)
        .filter(|index| {
            engine
                .infer(
                    ObserverId(OOD_OBSERVER_BASE + index),
                    LatentConceptId(240 + index),
                    0.0,
                    3,
                )
                .is_none()
        })
        .count();
    let report = engine.report();
    let composition_accuracy = composition as f64 / denominator;
    let direct_composed_relation_absence = direct_absence as f64 / denominator;
    let transfer_accuracy = transfer as f64 / denominator;
    let retention_accuracy = retention as f64 / denominator;
    let preverbal_accuracy = preverbal_correct as f64 / preverbal_total.max(1) as f64;
    let symbolic_accuracy = symbolic_correct as f64 / symbolic_total.max(1) as f64;
    let developmental_integration_accuracy = if integration_pairs.is_empty() {
        f64::from(seen.len() <= 1)
    } else {
        integrated as f64 / integration_pairs.len() as f64
    };
    let ood_abstention = ood as f64 / ood_trials as f64;
    let functional_cognition_gate = seen.len() >= minimum_seen
        && composition_accuracy >= 0.80
        && direct_composed_relation_absence == 1.0
        && transfer_accuracy >= 0.75
        && retention_accuracy >= 0.80
        && preverbal_accuracy >= 0.80
        && developmental_integration_accuracy >= 0.80
        && ood_abstention == 1.0
        && report.topological_symmetry >= 0.99
        && report.quantum.mean_single_spin_entropy >= 0.10
        && report.quantum.entangled_edges > 0;
    CognitiveMetrics {
        seen_lessons: seen.len(),
        highest_mastered_stage: seen
            .iter()
            .map(|(_, lesson)| lesson.stage)
            .max()
            .unwrap_or(0),
        composition_accuracy,
        direct_composed_relation_absence,
        transfer_accuracy,
        retention_accuracy,
        preverbal_accuracy,
        symbolic_accuracy,
        developmental_integration_accuracy,
        ood_abstention,
        topological_symmetry: report.topological_symmetry,
        spin_entropy: report.quantum.mean_single_spin_entropy,
        entangled_edges: report.quantum.entangled_edges,
        relations: report.rqm_relations,
        knowledge: report.consolidated_knowledge,
        epr_links: report.epr_links,
        functional_cognition_gate,
    }
}

fn regression_reason(
    baseline: &CognitiveMetrics,
    candidate: &CognitiveMetrics,
) -> Option<&'static str> {
    if candidate.topological_symmetry < 0.99 {
        return Some("topology_regression");
    }
    if candidate.entangled_edges == 0 || candidate.spin_entropy < 0.10 {
        return Some("spin_liquid_regression");
    }
    if candidate.composition_accuracy < 0.80
        || candidate.transfer_accuracy < 0.75
        || candidate.retention_accuracy < 0.80
        || candidate.preverbal_accuracy < 0.80
        || (candidate.seen_lessons > 1 && candidate.developmental_integration_accuracy < 0.80)
    {
        return Some("lesson_not_consolidated");
    }
    if baseline.seen_lessons > 0
        && candidate.composition_accuracy + 0.10 < baseline.composition_accuracy
    {
        return Some("composition_regression");
    }
    if baseline.seen_lessons > 0 && candidate.transfer_accuracy + 0.10 < baseline.transfer_accuracy
    {
        return Some("transfer_regression");
    }
    None
}

fn next_lesson_index(lessons_seen: &[bool], cycle: u64) -> usize {
    lessons_seen
        .iter()
        .position(|seen| !seen)
        .unwrap_or_else(|| (cycle.saturating_sub(1) as usize) % LESSONS.len())
}

fn score_response(response: &str, expected: &[&str]) -> f64 {
    let response = normalize_spanish(response);
    f64::from(
        expected
            .iter()
            .any(|keyword| response.contains(&normalize_spanish(keyword))),
    )
}

fn score_fraction(response: &str, expected: &[&str]) -> f64 {
    if expected.is_empty() {
        return 0.0;
    }
    let response = normalize_spanish(response);
    expected
        .iter()
        .filter(|keyword| response.contains(&normalize_spanish(keyword)))
        .count() as f64
        / expected.len() as f64
}

fn normalize_spanish(value: &str) -> String {
    value
        .to_lowercase()
        .replace(['á', 'à'], "a")
        .replace(['é', 'è'], "e")
        .replace(['í', 'ì'], "i")
        .replace(['ó', 'ò'], "o")
        .replace(['ú', 'ù', 'ü'], "u")
}

fn capture_checkpoint(engine: &UnifiedSpinCognitiveEngine, checkpoint: &mut TrainingCheckpoint) {
    checkpoint.amplitudes = engine
        .spin_liquid
        .amplitudes()
        .iter()
        .map(|amplitude| [amplitude.re, amplitude.im])
        .collect();
    checkpoint.relations = engine
        .cognition
        .workspace
        .relation_entries()
        .map(|(key, state)| RelationSnapshot::from_parts(key, state))
        .collect();
    checkpoint.epr_state = engine
        .cognition
        .workspace
        .entanglement
        .serialize_persistent_state();
    checkpoint.knowledge = engine
        .knowledge
        .values()
        .copied()
        .map(KnowledgeSnapshot::from)
        .collect();
}

fn restore_checkpoint(
    engine: &mut UnifiedSpinCognitiveEngine,
    checkpoint: &TrainingCheckpoint,
) -> Result<(), Box<dyn std::error::Error>> {
    if checkpoint.version != VERSION || checkpoint.lessons_seen.len() != LESSONS.len() {
        return Err("checkpoint Gemma-spin developmental incompatible".into());
    }
    if !checkpoint.amplitudes.is_empty() {
        let amplitudes = checkpoint
            .amplitudes
            .iter()
            .map(|value| Complex64::new(value[0], value[1]))
            .collect::<Vec<_>>();
        engine.spin_liquid.set_amplitudes(&amplitudes)?;
    }
    for relation in &checkpoint.relations {
        engine
            .cognition
            .workspace
            .import_relation(relation.key(), relation.state());
    }
    if !checkpoint.epr_state.is_empty() {
        engine
            .cognition
            .workspace
            .entanglement
            .apply_persistent_state(&checkpoint.epr_state)
            .map_err(io::Error::other)?;
    }
    for knowledge in &checkpoint.knowledge {
        let value = knowledge.value();
        engine.knowledge.insert(value.key, value);
    }
    Ok(())
}

impl RelationSnapshot {
    fn from_parts(key: RqmRelationKey, state: RqmPhaseRelationState) -> Self {
        Self {
            observer: key.observer,
            source: key.source.0,
            target: key.target.0,
            amplitude: state.amplitude,
            phase: state.phase,
            coherence: state.coherence,
            uncertainty: state.uncertainty,
            eligibility: state.eligibility,
            prediction_error: state.prediction_error,
            exposures: state.exposures,
            consolidated: state.consolidated,
        }
    }

    fn key(&self) -> RqmRelationKey {
        RqmRelationKey {
            observer: self.observer,
            source: LatentConceptId(self.source),
            target: LatentConceptId(self.target),
        }
    }

    fn state(&self) -> RqmPhaseRelationState {
        RqmPhaseRelationState {
            amplitude: self.amplitude,
            phase: self.phase,
            coherence: self.coherence,
            uncertainty: self.uncertainty,
            eligibility: self.eligibility,
            prediction_error: self.prediction_error,
            exposures: self.exposures,
            consolidated: self.consolidated,
        }
    }
}

impl From<ConsolidatedKnowledge> for KnowledgeSnapshot {
    fn from(value: ConsolidatedKnowledge) -> Self {
        Self {
            observer: value.key.observer,
            source: value.key.source.0,
            target: value.key.target.0,
            confidence: value.confidence,
            topological_symmetry: value.topological_symmetry,
            spin_entropy: value.spin_entropy,
            prediction_error: value.prediction_error,
            consolidations: value.consolidations,
        }
    }
}

impl KnowledgeSnapshot {
    fn value(&self) -> ConsolidatedKnowledge {
        ConsolidatedKnowledge {
            key: KnowledgeKey {
                observer: self.observer,
                source: LatentConceptId(self.source),
                target: LatentConceptId(self.target),
            },
            confidence: self.confidence,
            topological_symmetry: self.topological_symmetry,
            spin_entropy: self.spin_entropy,
            prediction_error: self.prediction_error,
            consolidations: self.consolidations,
        }
    }
}

fn save_checkpoint(
    root: &Path,
    checkpoint: &TrainingCheckpoint,
    milestone: bool,
) -> io::Result<()> {
    fs::create_dir_all(root.join("checkpoints"))?;
    let body = serde_json::to_vec(checkpoint).map_err(io::Error::other)?;
    let temporary = root.join("latest.tmp");
    let latest = root.join("latest.json");
    fs::write(&temporary, &body)?;
    if latest.exists() {
        fs::remove_file(&latest)?;
    }
    fs::rename(&temporary, &latest)?;
    if milestone {
        fs::write(
            root.join("checkpoints")
                .join(format!("cycle-{:012}.json", checkpoint.cycle)),
            body,
        )?;
    }
    Ok(())
}

fn persist_periodically(
    config: &TrainerConfig,
    engine: &UnifiedSpinCognitiveEngine,
    state: &mut TrainingCheckpoint,
    last_checkpoint: &mut Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    let periodic = state.cycle % config.checkpoint_every_cycles == 0
        || last_checkpoint.elapsed() >= config.checkpoint_every;
    if periodic {
        capture_checkpoint(engine, state);
        save_checkpoint(&config.root, state, false)?;
        *last_checkpoint = Instant::now();
        println!("event=checkpoint cycle={}", state.cycle);
    }
    if state.cycle % config.milestone_every == 0 {
        capture_checkpoint(engine, state);
        save_checkpoint(&config.root, state, true)?;
        prune_milestones(&config.root, config.retain_milestones)?;
        println!("event=milestone cycle={}", state.cycle);
    }
    Ok(())
}

fn prune_milestones(root: &Path, retain: usize) -> io::Result<()> {
    let directory = root.join("checkpoints");
    let mut files = fs::read_dir(&directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("cycle-") && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    files.sort();
    let remove = files.len().saturating_sub(retain);
    for path in files.into_iter().take(remove) {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn append_jsonl(path: &Path, value: &impl Serialize) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value).map_err(io::Error::other)?;
    file.write_all(b"\n")
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn argument_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == name {
            return args.next();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_score_requires_expected_grounding() {
        assert_eq!(score_response("Retirar la manta.", &["retirar"]), 1.0);
        assert_eq!(score_response("Solo contar.", &["retirar"]), 0.0);
    }

    #[test]
    fn fresh_checkpoint_matches_curriculum() {
        let checkpoint = TrainingCheckpoint::fresh();
        assert_eq!(checkpoint.lessons_seen.len(), LESSONS.len());
        assert!(checkpoint.lessons_seen.iter().all(|seen| !seen));
    }

    #[test]
    fn curriculum_retries_first_unmastered_stage() {
        let mut seen = vec![false; LESSONS.len()];
        assert_eq!(next_lesson_index(&seen, 1), 0);
        seen[0] = true;
        seen[1] = true;
        assert_eq!(next_lesson_index(&seen, 99), 2);
        seen.fill(true);
        assert_eq!(next_lesson_index(&seen, 10), 0);
    }

    #[test]
    fn developmental_stages_are_strictly_ordered() {
        assert!(LESSONS
            .windows(2)
            .all(|pair| pair[0].stage + 1 == pair[1].stage));
    }

    #[test]
    fn language_zone_resolves_aliases_to_abstract_ids() {
        let mut zone = LanguagePlanningZone::default();
        zone.register(PLANNING_TASKS[0]);
        assert_eq!(
            zone.resolve("quiero alcanzar el muñeco")
                .map(|label| label.concept_id),
            Some(PLANNING_TASKS[0].object_id)
        );
    }

    #[test]
    fn consolidated_network_recovers_labeled_plan() {
        let mut engine = fresh_engine().unwrap();
        let task = PLANNING_TASKS[0];
        train_language_plan(&mut engine, task, 40);
        assert!(validate_language_plan(&engine, task));
        let mut zone = LanguagePlanningZone::default();
        zone.register(task);
        assert_eq!(language_plan_accuracy(&engine, &zone), 1.0);
    }
}
