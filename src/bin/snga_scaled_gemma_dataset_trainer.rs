use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::simplicial::{ConceptProjection, SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

const STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const PROGRESS_PATH: &str = "data/snga_scaled_gemma_language.progress";
const PATTERN_SIZE: usize = 12;

#[derive(Clone, Debug)]
struct Lesson {
    stage: String,
    topic: String,
    question: String,
    answer: String,
    relation: String,
}

#[derive(Default)]
struct Progress {
    batches: usize,
    lessons: usize,
    stage: TrainingStage,
    stage_batches: usize,
    passes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrainingStage {
    LinguisticBasics,
    GrammarTime,
    SimpleSentences,
    ComplexSentences,
    WorldKnowledge,
}

impl Default for TrainingStage {
    fn default() -> Self {
        Self::LinguisticBasics
    }
}

impl TrainingStage {
    fn label(self) -> &'static str {
        match self {
            Self::LinguisticBasics => "bases_linguisticas",
            Self::GrammarTime => "gramatica_tiempos",
            Self::SimpleSentences => "oraciones_basicas",
            Self::ComplexSentences => "oraciones_complejas",
            Self::WorldKnowledge => "conocimiento_general",
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "gramatica_tiempos" => Self::GrammarTime,
            "oraciones_basicas" => Self::SimpleSentences,
            "oraciones_complejas" => Self::ComplexSentences,
            "conocimiento_general" => Self::WorldKnowledge,
            _ => Self::LinguisticBasics,
        }
    }

    fn next(self) -> Self {
        match self {
            Self::LinguisticBasics => Self::GrammarTime,
            Self::GrammarTime => Self::SimpleSentences,
            Self::SimpleSentences => Self::ComplexSentences,
            Self::ComplexSentences => Self::WorldKnowledge,
            Self::WorldKnowledge => Self::WorldKnowledge,
        }
    }

    fn focus(self) -> &'static str {
        match self {
            Self::LinguisticBasics => {
                "letras, silabas, palabras, significado de palabra, sinonimos simples, nombre de objetos"
            }
            Self::GrammarTime => {
                "sujeto, verbo, objeto, singular, plural, pasado, presente, futuro, negacion, pregunta"
            }
            Self::SimpleSentences => {
                "oraciones cortas sujeto-verbo-objeto, preguntas y respuestas concretas, descripcion breve"
            }
            Self::ComplexSentences => {
                "oraciones con causa, condicion, comparacion, explicacion, porque, aunque, entonces"
            }
            Self::WorldKnowledge => {
                "ciencia, biologia, salud, matematicas, historia, geografia, tecnologia, sociedad, vida cotidiana"
            }
        }
    }
}

fn main() {
    let hours = arg_value("--hours").and_then(|v| v.parse::<f64>().ok());
    let batches = arg_value("--batches").and_then(|v| v.parse::<usize>().ok());
    let batch_size = arg_value("--batch-size")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1);
    let sleep_ms = arg_value("--sleep-ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let engine = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()),
    };

    let mut network = SimplicialNetwork::grid_3d(scaled_config(), 2);
    let loaded = network.load_persistent_state(STATE_PATH).is_ok();
    network.enable_neural_oscillations();
    let mut progress = load_progress().unwrap_or_default();
    let started = Instant::now();

    println!("SNGA scaled Gemma dataset trainer");
    println!(
        "loaded={} nodes={} batch_size={} batches={:?} hours={:?} model={}",
        loaded,
        network.agents.len(),
        batch_size,
        batches,
        hours,
        engine.model
    );

    loop {
        if let Some(max_batches) = batches {
            if progress.batches >= max_batches {
                break;
            }
        }
        if let Some(max_hours) = hours {
            if started.elapsed() >= Duration::from_secs_f64(max_hours * 3600.0) {
                break;
            }
        }

        let dataset = generate_dataset(&engine, batch_size, progress.batches, progress.stage);
        let dataset_len = dataset.len();
        for lesson in &dataset {
            train_lesson(&mut network, lesson);
            progress.lessons += 1;
        }
        drop(dataset);

        progress.batches += 1;
        progress.stage_batches += 1;
        let probe = probe_stage(&network, progress.stage);
        if probe.0 == probe.1 && probe.2 > 0.05 {
            progress.passes += 1;
        }
        if maybe_advance_stage(&mut progress) {
            println!("avance curricular -> {}", progress.stage.label());
        }
        save_all(&network, &progress, "batch");
        let stats = network.plasticity_stats();
        println!(
            "batch={} stage={} stage_batches={} passes={} trained={} total_lessons={} probe_nonzero={}/{} probe_conf={:.3} edges={} assoc={} causal={} energy={:.1}",
            progress.batches,
            progress.stage.label(),
            progress.stage_batches,
            progress.passes,
            dataset_len,
            progress.lessons,
            probe.0,
            probe.1,
            probe.2,
            network.edges.len(),
            stats.associative_edges,
            stats.causal_edges,
            network.total_free_energy()
        );

        if sleep_ms > 0 {
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    save_all(&network, &progress, "final");
}

fn generate_dataset(
    engine: &OllamaGemmaEngine,
    batch_size: usize,
    batch_idx: usize,
    stage: TrainingStage,
) -> Vec<Lesson> {
    let prompt = format!(
        "Genera {batch_size} lecciones en espanol para destilar aprendizaje guiado en SNGA.\n\
        Etapa actual: {}.\n\
        Enfoque estricto de esta etapa: {}.\n\
        No saltes a temas de etapas posteriores salvo que la etapa sea conocimiento_general.\n\
        Evita repetir tema dentro del lote.\n\
        Formato estricto por linea:\n\
        etapa | tema | pregunta | respuesta | relacion\n\
        La respuesta debe ser breve, correcta y concreta. No uses numeracion. Lote {batch_idx}.",
        stage.label(),
        stage.focus()
    );
    let context = LinguisticContext {
        user_prompt: prompt,
        inferred_intent: "generar_dataset_destilacion".to_string(),
        geometric_projection: ConceptProjection {
            top_agents: Vec::new(),
        },
        memory_summary: "Genera datos temporales; SNGA los aprende y el lote se descarta."
            .to_string(),
    };

    let text = engine
        .generate(&context)
        .map(|r| r.text)
        .unwrap_or_else(|_| fallback_dataset(batch_size, batch_idx, stage));
    let parsed = parse_dataset(&text);
    if parsed.is_empty() {
        parse_dataset(&fallback_dataset(batch_size, batch_idx, stage))
    } else {
        parsed
    }
}

fn parse_dataset(text: &str) -> Vec<Lesson> {
    text.lines()
        .filter_map(|line| {
            let clean = line
                .trim()
                .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == '-')
                .trim();
            let parts = clean.split('|').map(str::trim).collect::<Vec<_>>();
            (parts.len() >= 5).then(|| Lesson {
                stage: parts[0].to_string(),
                topic: parts[1].to_string(),
                question: parts[2].to_string(),
                answer: parts[3].to_string(),
                relation: parts[4].to_string(),
            })
        })
        .collect()
}

fn train_lesson(network: &mut SimplicialNetwork, lesson: &Lesson) {
    let stage = pattern("stage", &lesson.stage, network.agents.len());
    let topic = pattern("topic", &lesson.topic, network.agents.len());
    let question = pattern("question", &lesson.question, network.agents.len());
    let answer = pattern("answer", &lesson.answer, network.agents.len());
    let relation = pattern("relation", &lesson.relation, network.agents.len());

    network.learn_transition(&stage, &topic);
    network.learn_transition(&topic, &question);
    network.learn_transition(&question, &answer);
    network.learn_transition(&topic, &answer);
    network.learn_transition(&answer, &relation);

    let mut fused = Vec::new();
    fused.extend(stage.iter().copied());
    fused.extend(topic.iter().copied());
    fused.extend(question.iter().copied());
    fused.extend(answer.iter().copied());
    fused.extend(relation.iter().copied());
    fused.sort_unstable();
    fused.dedup();

    network.clear_activity();
    network.set_attention_goal(&answer);
    network.inject_pattern(&question, 1.15, 2);
    network.inject_pattern(&answer, 0.95, 1);
    network.reinforce_coactivation_if_useful(&fused, 0.055, 0.90);
    for _ in 0..8 {
        network.step();
    }
    network.clear_attention_goal();
    network.clear_activity();
    for _ in 0..2 {
        network.step();
    }
}

fn probe_stage(network: &SimplicialNetwork, stage: TrainingStage) -> (usize, usize, f32) {
    let prompts: &[&str] = match stage {
        TrainingStage::LinguisticBasics => &[
            "palabra",
            "silaba",
            "letra",
            "significado",
            "sinonimo",
            "objeto",
        ],
        TrainingStage::GrammarTime => {
            &["sujeto", "verbo", "objeto", "pasado", "presente", "futuro"]
        }
        TrainingStage::SimpleSentences => &[
            "oracion simple",
            "pregunta concreta",
            "respuesta breve",
            "descripcion",
            "accion",
            "lugar",
        ],
        TrainingStage::ComplexSentences => &[
            "causa",
            "condicion",
            "comparacion",
            "explicacion",
            "porque",
            "entonces",
        ],
        TrainingStage::WorldKnowledge => &[
            "lenguaje",
            "concepto",
            "causalidad",
            "memoria",
            "planificacion",
            "mundo",
            "herramienta",
            "emocion",
        ],
    };
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for prompt in prompts {
        let p = pattern("topic", prompt, network.agents.len());
        let predicted = network.predict_next_pattern(&p, 1, 32);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        confidence += predicted.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;
    }
    (nonzero, prompts.len(), confidence / prompts.len() as f32)
}

fn save_all(network: &SimplicialNetwork, progress: &Progress, label: &str) {
    match network.save_persistent_state(STATE_PATH) {
        Ok(report) => {
            if let Err(err) = save_progress(progress) {
                eprintln!("{label}: estado guardado, progreso fallo: {err}");
            }
            println!(
                "{label}: saved agents={} edges={} causal={} batches={} lessons={}",
                report.agents,
                report.edges,
                report.causal_edges,
                progress.batches,
                progress.lessons
            );
        }
        Err(err) => eprintln!("{label}: fallo guardando: {err}"),
    }
}

fn fallback_dataset(batch_size: usize, batch_idx: usize, stage: TrainingStage) -> String {
    let seeds: &[(&str, &str, &str, &str, &str)] = match stage {
        TrainingStage::LinguisticBasics => &[
            (
                "bases_linguisticas",
                "palabra",
                "que es una palabra",
                "una palabra es una senal estable para una idea",
                "simbolo-significado",
            ),
            (
                "bases_linguisticas",
                "silaba",
                "que es una silaba",
                "una silaba es un grupo de sonidos dentro de una palabra",
                "sonido-palabra",
            ),
            (
                "bases_linguisticas",
                "sinonimo",
                "que es un sinonimo",
                "un sinonimo es una palabra con significado parecido",
                "palabra-equivalencia",
            ),
            (
                "bases_linguisticas",
                "objeto",
                "que nombra una palabra",
                "una palabra puede nombrar un objeto accion o cualidad",
                "nombre-referencia",
            ),
        ],
        TrainingStage::GrammarTime => &[
            (
                "gramatica_tiempos",
                "sujeto",
                "que es un sujeto",
                "el sujeto indica quien realiza o recibe una accion",
                "sujeto-accion",
            ),
            (
                "gramatica_tiempos",
                "verbo",
                "que es un verbo",
                "el verbo expresa accion estado o cambio",
                "accion-tiempo",
            ),
            (
                "gramatica_tiempos",
                "pasado",
                "que expresa el pasado",
                "el pasado describe algo que ya ocurrio",
                "tiempo-anterior",
            ),
            (
                "gramatica_tiempos",
                "futuro",
                "que expresa el futuro",
                "el futuro describe algo que puede ocurrir despues",
                "tiempo-posterior",
            ),
        ],
        TrainingStage::SimpleSentences => &[
            (
                "oraciones_basicas",
                "oracion simple",
                "como se forma una oracion simple",
                "una oracion simple une sujeto verbo y objeto",
                "sujeto-verbo-objeto",
            ),
            (
                "oraciones_basicas",
                "pregunta concreta",
                "como se responde una pregunta concreta",
                "una respuesta concreta entrega el dato pedido",
                "pregunta-respuesta",
            ),
            (
                "oraciones_basicas",
                "descripcion",
                "como describir un objeto",
                "describir un objeto menciona rasgos visibles o utiles",
                "objeto-rasgo",
            ),
            (
                "oraciones_basicas",
                "lugar",
                "como indicar lugar",
                "el lugar dice donde ocurre una accion",
                "accion-ubicacion",
            ),
        ],
        TrainingStage::ComplexSentences => &[
            (
                "oraciones_complejas",
                "causa",
                "como explicar una causa",
                "una causa indica por que ocurre un efecto",
                "causa-efecto",
            ),
            (
                "oraciones_complejas",
                "condicion",
                "que expresa una condicion",
                "una condicion indica cuando algo puede suceder",
                "condicion-resultado",
            ),
            (
                "oraciones_complejas",
                "comparacion",
                "para que sirve comparar",
                "comparar muestra semejanzas y diferencias entre ideas",
                "idea-diferencia",
            ),
            (
                "oraciones_complejas",
                "explicacion",
                "como explicar una idea",
                "explicar conecta razones ejemplos y consecuencias",
                "razon-consecuencia",
            ),
        ],
        TrainingStage::WorldKnowledge => &[
            (
                "lenguaje",
                "palabra",
                "que es una palabra",
                "una palabra es una senal estable para una idea",
                "simbolo-significado",
            ),
            (
                "conceptos",
                "categoria",
                "como se agrupan ideas",
                "una categoria une rasgos compartidos y separa distractores",
                "rasgo-categoria",
            ),
            (
                "entorno",
                "causa",
                "que hace una causa",
                "una causa anticipa un efecto observado en el entorno",
                "causa-efecto",
            ),
            (
                "mundo",
                "plan",
                "para que sirve un plan",
                "un plan organiza pasos hacia un objetivo",
                "objetivo-ruta",
            ),
            (
                "fisica",
                "gravedad",
                "que es la gravedad",
                "la gravedad atrae masas y organiza orbitas planetarias",
                "masa-atraccion",
            ),
            (
                "biologia",
                "celula",
                "que es una celula",
                "una celula es una unidad viva con membrana y funciones internas",
                "vida-estructura",
            ),
            (
                "salud",
                "agua",
                "por que beber agua",
                "el agua ayuda a regular temperatura y transportar nutrientes",
                "hidratacion-cuerpo",
            ),
            (
                "matematicas",
                "proporcion",
                "que es una proporcion",
                "una proporcion compara relaciones entre cantidades",
                "cantidad-relacion",
            ),
            (
                "historia",
                "agricultura",
                "por que fue importante la agricultura",
                "la agricultura permitio asentamientos estables y crecimiento social",
                "alimento-sociedad",
            ),
            (
                "geografia",
                "rio",
                "que hace un rio",
                "un rio transporta agua y sedimentos desde zonas altas hacia mares o lagos",
                "agua-territorio",
            ),
            (
                "tecnologia",
                "sensor",
                "para que sirve un sensor",
                "un sensor convierte cambios del entorno en señales medibles",
                "entorno-medicion",
            ),
            (
                "sociedad",
                "cooperacion",
                "por que cooperan las personas",
                "la cooperacion permite resolver tareas que superan a un individuo",
                "grupo-objetivo",
            ),
            (
                "emociones",
                "miedo",
                "para que sirve el miedo",
                "el miedo prepara al organismo para evitar peligro",
                "amenaza-respuesta",
            ),
            (
                "vida cotidiana",
                "herramienta",
                "que es una herramienta",
                "una herramienta amplifica una accion humana para lograr una meta",
                "accion-meta",
            ),
        ],
    };
    (0..batch_size)
        .map(|i| {
            let item = seeds[(batch_idx + i) % seeds.len()];
            format!(
                "{} | {} | {} | {} | {}",
                item.0, item.1, item.2, item.3, item.4
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn pattern(prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = DefaultHasher::new();
            prefix.hash(&mut hasher);
            value.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn load_progress() -> Option<Progress> {
    let text = fs::read_to_string(PROGRESS_PATH).ok()?;
    let mut p = Progress::default();
    for line in text.lines() {
        let (key, value) = line.split_once('=')?;
        match key {
            "batches" => p.batches = value.parse().ok()?,
            "lessons" => p.lessons = value.parse().ok()?,
            "stage" => p.stage = TrainingStage::from_label(value),
            "stage_batches" => p.stage_batches = value.parse().ok()?,
            "passes" => p.passes = value.parse().ok()?,
            _ => {}
        }
    }
    Some(p)
}

fn save_progress(progress: &Progress) -> std::io::Result<()> {
    if let Some(parent) = Path::new(PROGRESS_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        PROGRESS_PATH,
        format!(
            "batches={}\nlessons={}\nstage={}\nstage_batches={}\npasses={}\n",
            progress.batches,
            progress.lessons,
            progress.stage.label(),
            progress.stage_batches,
            progress.passes
        ),
    )
}

fn maybe_advance_stage(progress: &mut Progress) -> bool {
    if progress.stage == TrainingStage::WorldKnowledge {
        return false;
    }
    if progress.stage_batches >= 3 && progress.passes >= 2 {
        progress.stage = progress.stage.next();
        progress.stage_batches = 0;
        progress.passes = 0;
        true
    } else {
        false
    }
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

fn scaled_config() -> SimplicialConfig {
    SimplicialConfig {
        width: 72,
        height: 40,
        spacing: 6.5,
        elasticity: 0.005,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00012,
        max_active_agents: 384,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 2048,
        replay_interval: 8,
        replay_batch: 12,
        replay_learning_rate: 0.05,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.00008,
        hyperbolic_curvature: 0.0,
        seed: 401,
    }
}
