use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{ConceptProjection, SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::thread;
use std::time::Duration;

const DEFAULT_BASE_STATE: &str = "data/snga_scaled_gemma_language_fractal_compressed.snga";
const DEFAULT_STATE_PATH: &str = "data/snga_fractal_gemma_spanish_curriculum.snga";
const DEFAULT_PROGRESS_PATH: &str = "data/snga_fractal_gemma_spanish_curriculum.progress";
const DEFAULT_AGENT_COUNT: usize = 5_760;
const DEFAULT_PATTERN_NODES: usize = 5_760;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;
const DEFAULT_COMPRESS_EVERY: usize = 5;
const DEFAULT_MAX_ASSOCIATIVE: usize = 500_000;

#[derive(Clone, Copy)]
enum Region {
    FineLetters,
    LocalSyllables,
    MediumWords,
    UpperSentences,
    AssociativeMeaning,
}

#[derive(Clone, Debug)]
struct Lesson {
    stage: String,
    unit: String,
    input: String,
    target: String,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum TrainingStage {
    #[default]
    Letters,
    Syllables,
    Words,
    WordUnions,
    Sentences,
    Grammar,
    MediumSpanish,
}

impl TrainingStage {
    fn label(self) -> &'static str {
        match self {
            Self::Letters => "letras",
            Self::Syllables => "silabas",
            Self::Words => "palabras",
            Self::WordUnions => "uniones_de_palabras",
            Self::Sentences => "oraciones",
            Self::Grammar => "gramatica_basica",
            Self::MediumSpanish => "espanol_medio",
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "silabas" => Self::Syllables,
            "palabras" => Self::Words,
            "uniones_de_palabras" => Self::WordUnions,
            "oraciones" => Self::Sentences,
            "gramatica_basica" => Self::Grammar,
            "espanol_medio" => Self::MediumSpanish,
            _ => Self::Letters,
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Letters => Self::Syllables,
            Self::Syllables => Self::Words,
            Self::Words => Self::WordUnions,
            Self::WordUnions => Self::Sentences,
            Self::Sentences => Self::Grammar,
            Self::Grammar => Self::MediumSpanish,
            Self::MediumSpanish => Self::MediumSpanish,
        }
    }

    fn focus(self) -> &'static str {
        match self {
            Self::Letters => {
                "ensenar letras espanolas normalizadas, vocales, consonantes, orden y sonido basico"
            }
            Self::Syllables => {
                "ensenar silabas simples: consonante-vocal, vocal-consonante, separacion de silabas"
            }
            Self::Words => "ensenar como letras y silabas forman palabras con significado concreto",
            Self::WordUnions => {
                "ensenar pares de palabras, articulo-nombre, sujeto-verbo, verbo-objeto"
            }
            Self::Sentences => {
                "ensenar oraciones simples en espanol: sujeto verbo objeto y preguntas cortas"
            }
            Self::Grammar => {
                "ensenar singular plural, genero, tiempo verbal, negacion, pregunta y respuesta"
            }
            Self::MediumSpanish => {
                "ensenar causa efecto, condicion, comparacion, explicacion, resumen y definicion"
            }
        }
    }
}

fn main() {
    let batch_size = arg_value("--batch-size")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(48)
        .max(1);
    let max_batches = arg_value("--batches").and_then(|value| value.parse::<usize>().ok());
    let sleep_ms = arg_value("--sleep-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let compress_every = arg_value("--compress-every")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_COMPRESS_EVERY);
    let max_associative = arg_value("--max-associative")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_ASSOCIATIVE);
    let base_state = arg_value("--base").unwrap_or_else(|| DEFAULT_BASE_STATE.to_string());
    let state_path = arg_value("--state").unwrap_or_else(|| DEFAULT_STATE_PATH.to_string());
    let progress_path =
        arg_value("--progress").unwrap_or_else(|| DEFAULT_PROGRESS_PATH.to_string());

    let engine = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()),
    };
    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    let loaded = if Path::new(&state_path).exists() {
        network.load_persistent_state(&state_path).is_ok()
    } else {
        network.load_persistent_state(&base_state).is_ok()
    };
    network.enable_neural_oscillations();
    let mut progress = load_progress(&progress_path).unwrap_or_default();

    println!("SNGA fractal Gemma Spanish curriculum trainer");
    println!(
        "loaded={} state={} base={} progress={} batch_size={} batches={:?} compress_every={} max_associative={} model={} stage={}",
        loaded,
        state_path,
        base_state,
        progress_path,
        batch_size,
        max_batches,
        compress_every,
        max_associative,
        engine.model,
        progress.stage.label()
    );

    loop {
        if let Some(limit) = max_batches {
            if progress.batches >= limit {
                break;
            }
        }

        let lessons = generate_dataset(&engine, batch_size, progress.batches, progress.stage);
        for lesson in &lessons {
            train_lesson(&mut network, lesson);
            progress.lessons += 1;
        }
        progress.batches += 1;
        progress.stage_batches += 1;

        let exam = generate_exam(&engine, progress.stage);
        let exam_score = run_exam(&network, &exam);
        let decision = gemma_decision(&engine, progress.stage, &exam, exam_score);
        if decision {
            progress.passes += 1;
        }
        if progress.passes >= 3
            && progress.stage_batches >= 3
            && progress.stage != TrainingStage::MediumSpanish
        {
            progress.stage = progress.stage.next();
            progress.stage_batches = 0;
            progress.passes = 0;
            println!("avance_curricular={}", progress.stage.label());
        }

        let adjusted = network.anneal_active_edge_rest_lengths(0.10, 1.05);
        let compression = if compress_every > 0 && progress.batches % compress_every == 0 {
            compress_network(&mut network, max_associative)
        } else {
            CompressionReport::default()
        };
        save_all(&network, &progress, &state_path, &progress_path, "batch");
        let stats = network.plasticity_stats();
        println!(
            "batch={} stage={} stage_batches={} passes={} lessons={} exam_nonzero={}/{} exam_conf={:.3} gemma_advance={} edges={} assoc={} causal={} adjusted={} compressed_assoc={} compressed_causal={} compressed_ok={} energy={:.1}",
            progress.batches,
            progress.stage.label(),
            progress.stage_batches,
            progress.passes,
            progress.lessons,
            exam_score.nonzero,
            exam_score.total,
            exam_score.confidence,
            decision,
            stats.active_edges,
            stats.associative_edges,
            stats.causal_edges,
            adjusted,
            compression.removed_associative,
            compression.removed_causal,
            compression.knowledge_preserved,
            network.total_free_energy()
        );

        if sleep_ms > 0 {
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    save_all(&network, &progress, &state_path, &progress_path, "final");
}

fn generate_dataset(
    engine: &OllamaGemmaEngine,
    batch_size: usize,
    batch_idx: usize,
    stage: TrainingStage,
) -> Vec<Lesson> {
    let prompt = format!(
        "Genera {batch_size} lecciones para ensenar linguistica espanola a una red SNGA fractal.\n\
        La red debe aprender desde bajo nivel, sin usar un LLM en inferencia.\n\
        Etapa actual: {}.\n\
        Enfoque estricto: {}.\n\
        Formato obligatorio, una leccion por linea:\n\
        etapa | unidad | entrada | objetivo | relacion\n\
        Para letras usa entrada como una letra o secuencia corta y objetivo como su categoria.\n\
        Para palabras usa entrada como letras/silabas y objetivo como palabra o significado.\n\
        Para oraciones usa entrada como palabras ordenadas y objetivo como oracion correcta.\n\
        No uses numeracion ni markdown. Lote {batch_idx}.",
        stage.label(),
        stage.focus()
    );
    let text = ask_gemma(engine, prompt, "generar_dataset_linguistico")
        .unwrap_or_else(|| fallback_dataset(batch_size, batch_idx, stage));
    let parsed = parse_lessons(&text);
    if parsed.is_empty() {
        parse_lessons(&fallback_dataset(batch_size, batch_idx, stage))
    } else {
        parsed
    }
}

fn generate_exam(engine: &OllamaGemmaEngine, stage: TrainingStage) -> Vec<Lesson> {
    let prompt = format!(
        "Genera 8 examenes breves para evaluar si SNGA aprendio la etapa {}.\n\
        Enfoque: {}.\n\
        Formato obligatorio, una prueba por linea:\n\
        etapa | unidad | entrada | objetivo | relacion\n\
        No incluyas explicaciones.",
        stage.label(),
        stage.focus()
    );
    let text = ask_gemma(engine, prompt, "generar_examen_linguistico")
        .unwrap_or_else(|| fallback_dataset(8, 0, stage));
    let parsed = parse_lessons(&text);
    if parsed.is_empty() {
        parse_lessons(&fallback_dataset(8, 0, stage))
    } else {
        parsed
    }
}

fn gemma_decision(
    engine: &OllamaGemmaEngine,
    stage: TrainingStage,
    exam: &[Lesson],
    score: ExamScore,
) -> bool {
    let sample = exam
        .iter()
        .take(4)
        .map(|lesson| format!("{} -> {}", lesson.input, lesson.target))
        .collect::<Vec<_>>()
        .join("; ");
    let prompt = format!(
        "Eres maestro evaluador de SNGA. Etapa: {}.\n\
        Examen aplicado: {}.\n\
        Resultado de la red: respuestas_con_senal={}/{} confianza_media={:.3}.\n\
        Decide si domina suficiente para avanzar.\n\
        Responde solo una linea: AVANZAR: si o AVANZAR: no.",
        stage.label(),
        sample,
        score.nonzero,
        score.total,
        score.confidence
    );
    let Some(text) = ask_gemma(engine, prompt, "evaluar_avance_curricular") else {
        return score.nonzero == score.total && score.confidence > 0.05;
    };
    let lower = text.to_lowercase();
    lower.contains("avanzar: si") || lower.contains("avanzar si")
}

fn ask_gemma(engine: &OllamaGemmaEngine, prompt: String, intent: &str) -> Option<String> {
    let context = LinguisticContext {
        user_prompt: prompt,
        inferred_intent: intent.to_string(),
        geometric_projection: ConceptProjection {
            top_agents: Vec::new(),
        },
        memory_summary: "Gemma genera datos o evalua avance; SNGA aprende en la malla fractal."
            .to_string(),
    };
    engine.generate(&context).ok().map(|response| response.text)
}

fn parse_lessons(text: &str) -> Vec<Lesson> {
    text.lines()
        .filter_map(|line| {
            let clean = line
                .trim()
                .trim_start_matches(|ch: char| ch.is_ascii_digit() || ch == '.' || ch == '-')
                .trim();
            let parts = clean.split('|').map(str::trim).collect::<Vec<_>>();
            (parts.len() >= 5).then(|| Lesson {
                stage: parts[0].to_string(),
                unit: parts[1].to_string(),
                input: parts[2].to_string(),
                target: parts[3].to_string(),
                relation: parts[4].to_string(),
            })
        })
        .collect()
}

fn train_lesson(network: &mut SimplicialNetwork, lesson: &Lesson) {
    let stage = pattern("stage", &lesson.stage, network.agents.len());
    let unit = pattern("unit", &lesson.unit, network.agents.len());
    let input = hierarchical_text_pattern("input", &lesson.input, network.agents.len());
    let target = hierarchical_text_pattern("target", &lesson.target, network.agents.len());
    let relation = pattern("relation", &lesson.relation, network.agents.len());

    network.learn_transition(&stage, &unit);
    network.learn_transition(&unit, &input);
    network.learn_transition(&input, &target);
    network.learn_transition(&unit, &target);
    network.learn_transition(&target, &relation);

    let mut fused = Vec::new();
    fused.extend(stage);
    fused.extend(unit);
    fused.extend(input.iter().copied());
    fused.extend(target.iter().copied());
    fused.extend(relation);
    fused.sort_unstable();
    fused.dedup();

    network.clear_activity();
    network.set_attention_goal(&target);
    network.inject_pattern(&input, 1.15, 2);
    network.inject_pattern(&target, 0.95, 1);
    network.reinforce_coactivation_if_useful(&fused, 0.045, 0.90);
    for _ in 0..5 {
        network.step();
    }
    network.clear_attention_goal();
    network.clear_activity();
}

#[derive(Clone, Copy)]
struct ExamScore {
    total: usize,
    nonzero: usize,
    confidence: f32,
}

fn run_exam(network: &SimplicialNetwork, exam: &[Lesson]) -> ExamScore {
    let mut nonzero = 0;
    let mut confidence = 0.0;
    for lesson in exam {
        let input = hierarchical_text_pattern("input", &lesson.input, network.agents.len());
        let predicted = network.predict_next_pattern(&input, 1, 32);
        if !predicted.is_empty() {
            nonzero += 1;
        }
        confidence += predicted.iter().map(|(_, score)| *score).sum::<f32>() / 32.0;
    }
    ExamScore {
        total: exam.len(),
        nonzero,
        confidence: confidence / exam.len().max(1) as f32,
    }
}

#[derive(Default)]
struct CompressionReport {
    removed_associative: usize,
    removed_causal: usize,
    knowledge_preserved: bool,
}

fn compress_network(network: &mut SimplicialNetwork, max_associative: usize) -> CompressionReport {
    let mut report = CompressionReport {
        removed_associative: 0,
        removed_causal: 0,
        knowledge_preserved: true,
    };
    let reference = validation_signature(network);
    let mut chunk = 200_000;

    while network.plasticity_stats().associative_edges > max_associative && chunk > 0 {
        let before = network.clone();
        let removed = network.prune_low_value_associative_edges(chunk);
        if removed == 0 {
            break;
        }

        if validation_signature(network) == reference {
            report.removed_associative += removed;
        } else {
            *network = before;
            chunk /= 2;
        }
    }

    let mut causal_chunk = 80_000;
    let mut causal_attempts = 0;
    while causal_chunk > 0 && causal_attempts < 8 {
        causal_attempts += 1;
        let before = network.clone();
        let removed = network.prune_low_value_causal_edges(causal_chunk);
        if removed == 0 {
            break;
        }
        if validation_signature(network) == reference {
            report.removed_causal += removed;
        } else {
            *network = before;
            causal_chunk /= 2;
        }
    }

    if report.removed_associative > 0 || report.removed_causal > 0 {
        network.anneal_active_edge_rest_lengths(1.0, 0.0);
        report.knowledge_preserved = validation_signature(network) == reference;
    }
    report
}

fn validation_signature(network: &SimplicialNetwork) -> Vec<Vec<usize>> {
    let mut signatures = Vec::new();
    for lesson in validation_lessons() {
        for regional in [true, false] {
            let input = if regional {
                hierarchical_text_pattern("input", &lesson.input, network.agents.len())
            } else {
                legacy_hierarchical_text_pattern("input", &lesson.input, network.agents.len())
            };
            signatures.push(
                network
                    .predict_next_pattern(&input, 1, 32)
                    .into_iter()
                    .map(|(idx, _)| idx)
                    .collect::<Vec<_>>(),
            );
        }
    }
    signatures
}

fn validation_lessons() -> Vec<Lesson> {
    vec![
        Lesson {
            stage: "letras".to_string(),
            unit: "vocal a".to_string(),
            input: "a".to_string(),
            target: "vocal abierta".to_string(),
            relation: "letra-categoria".to_string(),
        },
        Lesson {
            stage: "silabas".to_string(),
            unit: "ma".to_string(),
            input: "m a".to_string(),
            target: "ma".to_string(),
            relation: "letras-silaba".to_string(),
        },
        Lesson {
            stage: "palabras".to_string(),
            unit: "casa".to_string(),
            input: "c a s a".to_string(),
            target: "casa lugar para vivir".to_string(),
            relation: "letras-palabra".to_string(),
        },
        Lesson {
            stage: "uniones_de_palabras".to_string(),
            unit: "nino corre".to_string(),
            input: "nino corre".to_string(),
            target: "sujeto y verbo".to_string(),
            relation: "palabra-frase".to_string(),
        },
        Lesson {
            stage: "oraciones".to_string(),
            unit: "oracion simple".to_string(),
            input: "el nino come pan".to_string(),
            target: "sujeto verbo objeto".to_string(),
            relation: "frase-oracion".to_string(),
        },
        Lesson {
            stage: "gramatica_basica".to_string(),
            unit: "plural".to_string(),
            input: "los ninos comen".to_string(),
            target: "varios sujetos".to_string(),
            relation: "numero-gramatical".to_string(),
        },
        Lesson {
            stage: "espanol_medio".to_string(),
            unit: "causa".to_string(),
            input: "llueve entonces el suelo se moja".to_string(),
            target: "causa y efecto".to_string(),
            relation: "causalidad".to_string(),
        },
    ]
}

fn fallback_dataset(batch_size: usize, batch_idx: usize, stage: TrainingStage) -> String {
    let seeds = match stage {
        TrainingStage::Letters => vec![
            ("letras", "vocal a", "a", "vocal abierta", "letra-categoria"),
            (
                "letras",
                "consonante m",
                "m",
                "consonante nasal",
                "letra-categoria",
            ),
            ("letras", "vocal e", "e", "vocal media", "letra-categoria"),
        ],
        TrainingStage::Syllables => vec![
            ("silabas", "ma", "m a", "ma", "letras-silaba"),
            ("silabas", "pa", "p a", "pa", "letras-silaba"),
            ("silabas", "so", "s o", "so", "letras-silaba"),
        ],
        TrainingStage::Words => vec![
            (
                "palabras",
                "casa",
                "c a s a",
                "casa lugar para vivir",
                "letras-palabra",
            ),
            (
                "palabras",
                "mesa",
                "m e s a",
                "mesa objeto",
                "letras-palabra",
            ),
            (
                "palabras",
                "nino",
                "n i n o",
                "nino persona pequena",
                "letras-palabra",
            ),
        ],
        TrainingStage::WordUnions => vec![
            (
                "uniones_de_palabras",
                "la casa",
                "la casa",
                "articulo y nombre",
                "palabra-frase",
            ),
            (
                "uniones_de_palabras",
                "nino corre",
                "nino corre",
                "sujeto y verbo",
                "palabra-frase",
            ),
            (
                "uniones_de_palabras",
                "come pan",
                "come pan",
                "verbo y objeto",
                "palabra-frase",
            ),
        ],
        TrainingStage::Sentences => vec![
            (
                "oraciones",
                "oracion simple",
                "el nino come pan",
                "sujeto verbo objeto",
                "frase-oracion",
            ),
            (
                "oraciones",
                "pregunta",
                "que es una palabra",
                "pregunta por definicion",
                "pregunta-respuesta",
            ),
            (
                "oraciones",
                "respuesta",
                "una palabra nombra una idea",
                "respuesta definicion",
                "pregunta-respuesta",
            ),
        ],
        TrainingStage::Grammar => vec![
            (
                "gramatica_basica",
                "pasado",
                "ayer el nino corrio",
                "accion anterior",
                "tiempo-verbal",
            ),
            (
                "gramatica_basica",
                "plural",
                "los ninos comen",
                "varios sujetos",
                "numero-gramatical",
            ),
            (
                "gramatica_basica",
                "negacion",
                "el perro no ladra",
                "accion negada",
                "negacion",
            ),
        ],
        TrainingStage::MediumSpanish => vec![
            (
                "espanol_medio",
                "causa",
                "llueve entonces el suelo se moja",
                "causa y efecto",
                "causalidad",
            ),
            (
                "espanol_medio",
                "condicion",
                "si estudias aprendes",
                "condicion y resultado",
                "condicion",
            ),
            (
                "espanol_medio",
                "comparacion",
                "el gato es mas pequeno que el perro",
                "comparacion",
                "comparar",
            ),
        ],
    };
    (0..batch_size)
        .map(|idx| {
            let item = seeds[(idx + batch_idx) % seeds.len()];
            format!(
                "{} | {} | {} | {} | {}",
                item.0, item.1, item.2, item.3, item.4
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn hierarchical_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    // Conserva la firma global anterior y agrega firmas regionales por escala
    // para que el aprendizaje nuevo se ordene sin olvidar lo ya entrenado.
    let mut out = pattern(prefix, text, nodes);
    let normalized = normalize_text(text);
    out.extend(regional_pattern(
        prefix,
        &normalized,
        PATTERN_SIZE,
        nodes,
        text_region(&normalized),
    ));
    for (pos, ch) in normalized.chars().enumerate().take(24) {
        out.extend(letter_pattern(ch, pos, nodes));
    }
    for word in normalized.split_whitespace().take(12) {
        out.extend(pattern("word", word, nodes));
        out.extend(regional_pattern(
            "word",
            word,
            PATTERN_SIZE,
            nodes,
            Region::MediumWords,
        ));
    }
    for pair in normalized.split_whitespace().collect::<Vec<_>>().windows(2) {
        out.extend(pattern(
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
            nodes,
        ));
        out.extend(regional_pattern(
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
            PATTERN_SIZE,
            nodes,
            Region::UpperSentences,
        ));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn legacy_hierarchical_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    let mut out = pattern(prefix, text, nodes);
    let normalized = normalize_text(text);
    for (pos, ch) in normalized.chars().enumerate().take(24) {
        out.extend(legacy_letter_pattern(ch, pos, nodes));
    }
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    for word in words.iter().take(12) {
        out.extend(pattern("word", word, nodes));
    }
    for pair in words.windows(2) {
        out.extend(pattern(
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
            nodes,
        ));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn legacy_letter_pattern(ch: char, pos: usize, nodes: usize) -> Vec<usize> {
    let nodes = pattern_nodes(nodes);
    (0..LETTER_PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "letter".hash(&mut hasher);
            ch.hash(&mut hasher);
            pos.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn letter_pattern(ch: char, pos: usize, nodes: usize) -> Vec<usize> {
    let mut pattern = (0..LETTER_PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "letter".hash(&mut hasher);
            ch.hash(&mut hasher);
            pos.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect::<Vec<_>>();
    pattern.extend(regional_pattern(
        "letter",
        &format!("{ch}_{pos}"),
        LETTER_PATTERN_SIZE,
        nodes,
        Region::FineLetters,
    ));
    pattern.sort_unstable();
    pattern.dedup();
    pattern
}

fn pattern(prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    let nodes = pattern_nodes(nodes);
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalize_text(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
}

fn regional_pattern(
    prefix: &str,
    value: &str,
    size: usize,
    nodes: usize,
    region: Region,
) -> Vec<usize> {
    let coding_nodes = linguistic_nodes(nodes);
    let (start, len) = region_bounds(region, coding_nodes);
    (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "regional".hash(&mut hasher);
            prefix.hash(&mut hasher);
            normalize_text(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            start + (hasher.finish() as usize % len.max(1))
        })
        .collect()
}

fn text_region(normalized: &str) -> Region {
    let words = normalized.split_whitespace().count();
    let chars = normalized.chars().filter(|ch| !ch.is_whitespace()).count();
    if words <= 1 && chars <= 2 {
        Region::LocalSyllables
    } else if words <= 1 {
        Region::MediumWords
    } else if words <= 4 {
        Region::UpperSentences
    } else {
        Region::AssociativeMeaning
    }
}

fn region_bounds(region: Region, nodes: usize) -> (usize, usize) {
    let (start, end) = match region {
        Region::FineLetters => (0.00_f32, 0.20_f32),
        Region::LocalSyllables => (0.20, 0.40),
        Region::MediumWords => (0.40, 0.65),
        Region::UpperSentences => (0.65, 0.85),
        Region::AssociativeMeaning => (0.85, 1.00),
    };
    let start_idx = (nodes as f32 * start).floor() as usize;
    let end_idx = (nodes as f32 * end).ceil() as usize;
    (
        start_idx.min(nodes.saturating_sub(1)),
        end_idx.saturating_sub(start_idx).max(1),
    )
}

fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| match ch {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            other => other,
        })
        .filter(|ch| ch.is_ascii_alphanumeric() || ch.is_whitespace())
        .collect()
}

fn save_all(
    network: &SimplicialNetwork,
    progress: &Progress,
    state_path: &str,
    progress_path: &str,
    label: &str,
) {
    match network.save_persistent_state(state_path) {
        Ok(report) => {
            if let Err(err) = save_progress(progress_path, progress) {
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

fn load_progress(path: &str) -> Option<Progress> {
    let text = fs::read_to_string(path).ok()?;
    let mut progress = Progress::default();
    for line in text.lines() {
        let (key, value) = line.split_once('=')?;
        match key {
            "batches" => progress.batches = value.parse().ok()?,
            "lessons" => progress.lessons = value.parse().ok()?,
            "stage" => progress.stage = TrainingStage::from_label(value),
            "stage_batches" => progress.stage_batches = value.parse().ok()?,
            "passes" => progress.passes = value.parse().ok()?,
            _ => {}
        }
    }
    Some(progress)
}

fn save_progress(path: &str, progress: &Progress) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
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

fn fractal_mesh_config() -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: agent_count(),
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    }
}

fn agent_count() -> usize {
    env::var("SNGA_AGENT_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_AGENT_COUNT)
        .max(DEFAULT_AGENT_COUNT)
}

fn pattern_nodes(nodes: usize) -> usize {
    env::var("SNGA_PATTERN_NODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PATTERN_NODES)
        .min(nodes)
        .max(1)
}

fn linguistic_nodes(nodes: usize) -> usize {
    env::var("SNGA_LINGUISTIC_NODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(nodes)
        .min(nodes)
        .max(1)
}

fn config() -> SimplicialConfig {
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

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}
