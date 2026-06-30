use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::Path;

const DEFAULT_STATE_PATH: &str = "data/snga_fractal_semantic_executive_gemma_adapter.snga";
const DEFAULT_TEACHINGS_PATH: &str = "data/snga_semantic_executive_console_teachings.tsv";
const DEFAULT_REGION_SIZE: usize = 8_192;
const REGION_COUNT: usize = 12;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;
const MIN_LEXICAL_TRUST: f32 = 0.18;

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Region {
    SemanticHubAtl = 0,
    SemanticControl = 2,
    Planner = 5,
    LinguisticSlot = 10,
}

#[derive(Clone, Copy)]
struct AdapterLesson {
    user_input: &'static str,
    linguistic_intent: &'static str,
    internal_concepts: &'static str,
    control_task: &'static str,
    response_frame: &'static str,
    ideal_response: &'static str,
}

#[derive(Clone, Debug)]
struct LearnedTeaching {
    prompt: String,
    teaching: String,
    response: String,
}

struct Decision<'a> {
    lesson: Option<&'a AdapterLesson>,
    teaching: Option<&'a LearnedTeaching>,
    trusted: bool,
    answer: String,
    lexical_score: f32,
    confidence: f32,
}

fn main() {
    let turns = arg_value("--turns")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let save_every_lessons = arg_value("--save-every-lessons")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20)
        .max(1);
    let no_save = has_flag("--no-save");
    let offline_fallback = has_flag("--offline-fallback");
    let state_path =
        env::var("SNGA_SEMEXEC_ADAPTER_STATE").unwrap_or_else(|_| DEFAULT_STATE_PATH.to_string());
    let teachings_path =
        env::var("SNGA_SEMEXEC_TEACHINGS").unwrap_or_else(|_| DEFAULT_TEACHINGS_PATH.to_string());

    let teacher = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_TEACHER_MODEL")
            .or_else(|_| env::var("SNGA_GEMMA_MODEL"))
            .unwrap_or_else(|_| "gemma2:2b".to_string()),
    };
    let peripheral = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_PERIPHERAL_MODEL")
            .or_else(|_| env::var("SNGA_GEMMA_MODEL"))
            .unwrap_or_else(|_| "gemma2:2b".to_string()),
    };

    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes()));
    match network.load_persistent_state(&state_path) {
        Ok(report) => println!(
            "SNGA distillation chat cargado: state={} agentes={} aristas={} causales={}",
            state_path, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            eprintln!("No pude cargar {state_path}: {err}");
            return;
        }
    }
    let mut teachings = load_teachings(&teachings_path);
    println!(
        "modo: teacher={} peripheral={} teachings={} no_save={} offline_fallback={} turns={} save_every_lessons={}",
        teacher.model,
        peripheral.model,
        teachings.len(),
        no_save,
        offline_fallback,
        if turns == usize::MAX {
            "∞".to_string()
        } else {
            turns.to_string()
        },
        save_every_lessons
    );

    let mut dirty_teachings = 0_usize;
    for turn in 0..turns {
        let prompt = sanitize_prompt(&gemma_initiates(&teacher, turn, offline_fallback));
        println!("\nGemma-maestro> {prompt}");

        let decision = decide_with_snga(&network, &prompt, &teachings);
        if decision.trusted {
            let verbalized =
                peripheral_verbalizes(&peripheral, &prompt, &decision, offline_fallback);
            println!("SNGA-logica> {}", decision.answer);
            println!("LLM-acoplado> {verbalized}");
            println!(
                "diagnostico> trusted=true lexical={:.1}% confidence={:.3}",
                decision.lexical_score * 100.0,
                decision.confidence
            );
            continue;
        }

        println!(
            "SNGA-logica> no_conozco: no hay ruta confiable para \"{}\"",
            prompt
        );
        let teaching_text = sanitize_teaching(&gemma_teaches(&teacher, &prompt, offline_fallback));
        println!("LLM-acoplado pregunta> Gemma, enseña a SNGA: ¿qué significa y cómo responder?");
        println!("Gemma-maestro enseña> {teaching_text}");

        let teaching = LearnedTeaching {
            prompt: prompt.clone(),
            response: synthesize_response_from_teaching(&prompt, &teaching_text),
            teaching: teaching_text,
        };
        train_teaching_route(&mut network, &teaching);
        println!(
            "SNGA-aprende> \"{}\" -> {}",
            teaching.prompt, teaching.response
        );
        teachings.push(teaching);
        dirty_teachings += 1;

        if !no_save && dirty_teachings >= save_every_lessons {
            save_all(&network, &teachings, &state_path, &teachings_path);
            dirty_teachings = 0;
        }
    }

    if !no_save && dirty_teachings > 0 && turns != usize::MAX {
        save_all(&network, &teachings, &state_path, &teachings_path);
    }
}

fn gemma_initiates(engine: &OllamaGemmaEngine, turn: usize, offline: bool) -> String {
    if offline {
        return fallback_prompts()[turn % fallback_prompts().len()].to_string();
    }
    ask_engine(
        engine,
        "Propón UNA pregunta breve en español para enseñar conocimiento cotidiano a SNGA. Devuelve solo la pregunta, sin explicación.",
    )
    .unwrap_or_else(|| fallback_prompts()[turn % fallback_prompts().len()].to_string())
}

fn gemma_teaches(engine: &OllamaGemmaEngine, prompt: &str, offline: bool) -> String {
    if offline {
        return fallback_teaching(prompt);
    }
    ask_engine(
        engine,
        &format!(
            "SNGA no conoce esta pregunta: {prompt:?}. Enseña en una frase breve qué significa y cómo debe responder. Formato libre, sin JSON."
        ),
    )
    .unwrap_or_else(|| fallback_teaching(prompt))
}

fn ask_engine(engine: &OllamaGemmaEngine, prompt: &str) -> Option<String> {
    let context = LinguisticContext {
        user_prompt: prompt.to_string(),
        inferred_intent: "maestro_periferico_para_destilacion_snga".to_string(),
        geometric_projection: snga::simplicial::ConceptProjection {
            top_agents: Vec::new(),
        },
        memory_summary:
            "Actuas como maestro externo. SNGA aprende; tu respuesta sera usada para entrenar la red."
                .to_string(),
    };
    engine
        .generate(&context)
        .ok()
        .map(|response| clean_line(&response.text))
        .filter(|text| !text.is_empty())
}

fn peripheral_verbalizes(
    engine: &OllamaGemmaEngine,
    prompt: &str,
    decision: &Decision<'_>,
    offline: bool,
) -> String {
    if offline {
        return decision.answer.clone();
    }
    let source = if let Some(lesson) = decision.lesson {
        format!(
            "intencion={}; concepto={}; control={}; frame={}",
            lesson.linguistic_intent,
            lesson.internal_concepts,
            lesson.control_task,
            lesson.response_frame
        )
    } else if let Some(teaching) = decision.teaching {
        format!("ensenanza_usuario={}", teaching.teaching)
    } else {
        "sin_fuente".to_string()
    };
    ask_engine(
        engine,
        &format!(
            "El usuario/Gemma pregunto: {prompt:?}. SNGA ya decidio la respuesta logica: {:?}. Fuente: {source}. Verbaliza breve en español sin agregar hechos nuevos.",
            decision.answer
        ),
    )
    .unwrap_or_else(|| decision.answer.clone())
}

fn decide_with_snga<'a>(
    network: &SimplicialNetwork,
    prompt: &str,
    teachings: &'a [LearnedTeaching],
) -> Decision<'a> {
    if let Some(teaching) = match_teaching(prompt, teachings) {
        return Decision {
            lesson: None,
            teaching: Some(teaching),
            trusted: true,
            answer: teaching.response.clone(),
            lexical_score: lexical_overlap(prompt, &teaching.prompt)
                .max(lexical_overlap(prompt, &teaching.teaching))
                .max(exact_normalized_match(prompt, &teaching.prompt)),
            confidence: 1.0,
        };
    }

    let llm_input = linguistic_text_pattern("gemma_input", prompt, network.agents.len());
    let concept_prediction = network.infer_transitive_from(&llm_input, 3, 128);
    let frame_prediction = network.infer_transitive_from(&llm_input, 7, 256);
    let concept_ids = ids(&concept_prediction);
    let frame_ids = ids(&frame_prediction);
    let mut best = None::<(usize, f32, f32)>;
    for (idx, lesson) in validation_lessons().iter().enumerate() {
        let concept = regional_pattern(
            Region::SemanticHubAtl,
            "internal_concepts",
            lesson.internal_concepts,
            PATTERN_SIZE,
            network.agents.len(),
        );
        let planner = regional_pattern(
            Region::Planner,
            "response_frame",
            lesson.response_frame,
            PATTERN_SIZE,
            network.agents.len(),
        );
        let lexical = lexical_overlap(prompt, lesson.user_input)
            .max(lexical_overlap(prompt, lesson.internal_concepts))
            .max(lexical_overlap(prompt, lesson.response_frame));
        let score = overlap_ratio(&concept_ids, &concept) * 5.0
            + overlap_ratio(&frame_ids, &planner) * 5.0
            + lexical;
        match best {
            Some((_, best_score, _)) if best_score >= score => {}
            _ => best = Some((idx, score, lexical)),
        }
    }

    let (idx, _, lexical_score) = best.unwrap_or((0, 0.0, 0.0));
    let lesson = &validation_lessons()[idx];
    Decision {
        lesson: Some(lesson),
        teaching: None,
        trusted: lexical_score >= MIN_LEXICAL_TRUST,
        answer: lesson.ideal_response.to_string(),
        lexical_score,
        confidence: confidence(&concept_prediction) + confidence(&frame_prediction),
    }
}

fn train_teaching_route(network: &mut SimplicialNetwork, teaching: &LearnedTeaching) {
    let input = linguistic_text_pattern("teach_input", &teaching.prompt, network.agents.len());
    let meaning = regional_pattern(
        Region::SemanticHubAtl,
        "teach_meaning",
        &teaching.teaching,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let frame = regional_pattern(
        Region::Planner,
        "teach_response_frame",
        &teaching.response,
        PATTERN_SIZE,
        network.agents.len(),
    );
    for _ in 0..6 {
        network.learn_transition(&input, &meaning);
        network.learn_transition(&meaning, &frame);
        network.learn_from_prediction_error(&input, &meaning, 2, 128, 0.12);
        network.learn_from_prediction_error(&input, &frame, 3, 128, 0.10);
    }
    let mut fused = input.clone();
    fused.extend(meaning.iter().copied());
    fused.extend(frame.iter().copied());
    fused.sort_unstable();
    fused.dedup();
    network.reinforce_coactivation_if_useful(&fused, 0.07, 0.94);
}

fn save_all(
    network: &SimplicialNetwork,
    teachings: &[LearnedTeaching],
    state_path: &str,
    teachings_path: &str,
) {
    match network.save_persistent_state(state_path) {
        Ok(report) => println!(
            "guardado_snga> agentes={} aristas={} causales={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => eprintln!("guardado_snga> fallo guardando estado: {err}"),
    }
    if let Err(err) = save_teachings(teachings_path, teachings) {
        eprintln!("guardado_snga> fallo guardando enseñanzas: {err}");
    }
}

fn validation_lessons() -> &'static [AdapterLesson] {
    &[
        AdapterLesson {
            user_input: "que es una manzana roja",
            linguistic_intent: "pregunta definicion objeto",
            internal_concepts: "manzana fruta roja redonda comestible",
            control_task: "activar concepto desde palabra y rasgos",
            response_frame: "definir manzana con rasgos visuales y categoria",
            ideal_response: "Una manzana roja es una fruta comestible, redonda y dulce.",
        },
        AdapterLesson {
            user_input: "planea cena vegetariana sin carne",
            linguistic_intent: "instruccion plan con restriccion",
            internal_concepts: "vegetariano lechuga lentejas arroz excluir carne",
            control_task: "filtrar conceptos no permitidos por restriccion",
            response_frame: "proponer plato permitido y explicar exclusion",
            ideal_response: "Puedo combinar lentejas con arroz y dejar fuera la carne.",
        },
        AdapterLesson {
            user_input: "banco en el parque",
            linguistic_intent: "desambiguar palabra por contexto",
            internal_concepts: "banco asiento parque no financiero",
            control_task: "seleccionar significado secundario por contexto",
            response_frame: "explicar que banco significa asiento",
            ideal_response: "En ese contexto, banco significa un asiento para sentarse.",
        },
        AdapterLesson {
            user_input: "si llueve que pasa con el suelo",
            linguistic_intent: "pregunta causa efecto",
            internal_concepts: "lluvia agua suelo mojado",
            control_task: "seguir cadena causal simple",
            response_frame: "responder efecto probable de lluvia",
            ideal_response: "Si llueve, el agua cae y el suelo probablemente se moja.",
        },
        AdapterLesson {
            user_input: "explica tu plan antes de responder",
            linguistic_intent: "pedir planificacion",
            internal_concepts: "meta pasos memoria trabajo respuesta",
            control_task: "ordenar pasos antes de verbalizar",
            response_frame: "mencionar objetivo restriccion y accion",
            ideal_response:
                "Primero fijo el objetivo, luego reviso restricciones y finalmente doy la accion.",
        },
    ]
}

fn fallback_prompts() -> &'static [&'static str] {
    &[
        "hola",
        "que es el miedo",
        "que es estar triste",
        "que es mundo",
        "para que sirve dormir",
        "que significa aprender",
    ]
}

fn fallback_teaching(prompt: &str) -> String {
    match normalize_text(prompt).as_str() {
        "hola" => "hola es un saludo humano; puedes responder con hola".to_string(),
        text if text.contains("miedo") => {
            "miedo es una emocion ante peligro o incertidumbre; responde explicandolo breve"
                .to_string()
        }
        text if text.contains("triste") => {
            "estar triste es sentir pena o bajo animo; responde con una definicion breve"
                .to_string()
        }
        text if text.contains("mundo") => {
            "mundo es el entorno o planeta donde vivimos; responde con una definicion breve"
                .to_string()
        }
        _ => format!("{prompt} es una consulta nueva; responde con una explicacion breve."),
    }
}

fn synthesize_response_from_teaching(prompt: &str, teaching: &str) -> String {
    let normalized_prompt = normalize_text(prompt);
    let normalized_teaching = normalize_text(teaching);
    let response = if normalized_prompt == "hola" && normalized_teaching.contains("saludo") {
        "Hola.".to_string()
    } else if let Some((_, answer)) = teaching.split_once("puedes responder") {
        clean_line(answer).trim_end_matches('.').to_string() + "."
    } else if let Some((_, answer)) = teaching.split_once("responde") {
        clean_line(answer).trim_end_matches('.').to_string() + "."
    } else {
        clean_line(teaching).trim_end_matches('.').to_string() + "."
    };
    sanitize_response(&response)
}

fn match_teaching<'a>(
    prompt: &str,
    teachings: &'a [LearnedTeaching],
) -> Option<&'a LearnedTeaching> {
    teachings
        .iter()
        .map(|teaching| {
            let score = lexical_overlap(prompt, &teaching.prompt)
                .max(lexical_overlap(prompt, &teaching.teaching))
                .max(exact_normalized_match(prompt, &teaching.prompt));
            (teaching, score)
        })
        .filter(|(_, score)| *score >= MIN_LEXICAL_TRUST)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(teaching, _)| teaching)
}

fn load_teachings(path: &str) -> Vec<LearnedTeaching> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter_map(|line| {
            let parts = line.split('\t').collect::<Vec<_>>();
            (parts.len() == 3).then(|| LearnedTeaching {
                prompt: sanitize_prompt(&unescape_field(parts[0])),
                teaching: sanitize_teaching(&unescape_field(parts[1])),
                response: sanitize_response(&unescape_field(parts[2])),
            })
        })
        .collect()
}

fn save_teachings(path: &str, teachings: &[LearnedTeaching]) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    for teaching in teachings {
        out.push_str(&escape_field(&sanitize_prompt(&teaching.prompt)));
        out.push('\t');
        out.push_str(&escape_field(&sanitize_teaching(&teaching.teaching)));
        out.push('\t');
        out.push_str(&escape_field(&sanitize_response(&teaching.response)));
        out.push('\n');
    }
    fs::write(path, out)
}

fn sanitize_prompt(value: &str) -> String {
    sanitize_text(value)
        .trim_end_matches('.')
        .trim()
        .to_string()
}

fn sanitize_teaching(value: &str) -> String {
    sanitize_text(value)
}

fn sanitize_response(value: &str) -> String {
    let mut clean = sanitize_text(value);
    if !clean.ends_with('.') && !clean.ends_with('?') && !clean.ends_with('!') {
        clean.push('.');
    }
    clean
}

fn sanitize_text(value: &str) -> String {
    let mut clean = value
        .replace('\r', " ")
        .replace('\n', " ")
        .replace('\t', " ")
        .replace("\\n", " ")
        .replace("\\t", " ")
        .trim()
        .trim_matches('"')
        .trim_matches('“')
        .trim_matches('”')
        .trim_matches('`')
        .trim()
        .to_string();
    while clean.contains("  ") {
        clean = clean.replace("  ", " ");
    }
    while clean.ends_with('"')
        || clean.ends_with('“')
        || clean.ends_with('”')
        || clean.ends_with('`')
    {
        clean.pop();
        clean = clean.trim_end().to_string();
    }
    clean
}

fn linguistic_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    let mut out = regional_pattern(Region::LinguisticSlot, prefix, text, PATTERN_SIZE, nodes);
    let normalized = normalize_text(text);
    for (pos, ch) in normalized.chars().enumerate().take(32) {
        out.extend(letter_pattern(ch, pos, nodes));
    }
    for word in normalized.split_whitespace().take(16) {
        out.extend(regional_pattern(
            Region::LinguisticSlot,
            "word",
            word,
            PATTERN_SIZE,
            nodes,
        ));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn letter_pattern(ch: char, pos: usize, nodes: usize) -> Vec<usize> {
    regional_pattern(
        Region::LinguisticSlot,
        "letter",
        &format!("{ch}_{pos}"),
        LETTER_PATTERN_SIZE,
        nodes,
    )
}

fn regional_pattern(
    region: Region,
    prefix: &str,
    value: &str,
    size: usize,
    nodes: usize,
) -> Vec<usize> {
    let region_size = (nodes / REGION_COUNT).max(DEFAULT_REGION_SIZE);
    let start = region as usize * region_size;
    let len = region_size.min(nodes.saturating_sub(start)).max(1);
    let normalized = normalize_text(value);
    let mut out = pattern_in_range(region, prefix, &normalized, size, start, len);
    if let Some(legacy_region_size) = legacy_region_size(nodes) {
        let legacy_start = region as usize * legacy_region_size;
        if legacy_start < nodes {
            let legacy_len = legacy_region_size.min(nodes - legacy_start).max(1);
            out.extend(pattern_in_range(
                region,
                prefix,
                &normalized,
                size,
                legacy_start,
                legacy_len,
            ));
            out.sort_unstable();
            out.dedup();
        }
    }
    out
}

fn pattern_in_range(
    region: Region,
    prefix: &str,
    normalized: &str,
    size: usize,
    start: usize,
    len: usize,
) -> Vec<usize> {
    (0..size)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (region as usize).hash(&mut hasher);
            prefix.hash(&mut hasher);
            normalized.hash(&mut hasher);
            offset.hash(&mut hasher);
            start + (hasher.finish() as usize % len)
        })
        .collect()
}

fn legacy_region_size(nodes: usize) -> Option<usize> {
    let current = (nodes / REGION_COUNT).max(DEFAULT_REGION_SIZE);
    env::var("SNGA_SEMEXEC_LEGACY_REGION_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|size| *size > 0 && *size != current && *size * REGION_COUNT <= nodes)
}

fn ids(predicted: &[(usize, f32)]) -> Vec<usize> {
    predicted.iter().map(|(idx, _)| *idx).collect()
}

fn overlap_ratio(left: &[usize], right: &[usize]) -> f32 {
    let hits = left.iter().filter(|idx| right.contains(idx)).count();
    hits as f32 / right.len().max(1) as f32
}

fn confidence(predicted: &[(usize, f32)]) -> f32 {
    if predicted.is_empty() {
        return 0.0;
    }
    predicted.iter().map(|(_, score)| *score).sum::<f32>() / predicted.len() as f32
}

fn lexical_overlap(left: &str, right: &str) -> f32 {
    let left_words = content_words(left);
    let right_words = content_words(right);
    if left_words.is_empty() || right_words.is_empty() {
        return 0.0;
    }
    let hits = left_words
        .iter()
        .filter(|word| right_words.contains(word))
        .count();
    hits as f32 / left_words.len().max(right_words.len()).max(1) as f32
}

fn content_words(text: &str) -> Vec<String> {
    normalize_text(text)
        .split_whitespace()
        .filter(|word| !stop_words().contains(word))
        .map(normalize_word)
        .filter(|word| word.len() > 2)
        .collect()
}

fn normalize_word(word: &str) -> String {
    if word.len() > 4 && word.ends_with("es") {
        word[..word.len() - 2].to_string()
    } else if word.len() > 3 && word.ends_with('s') {
        word[..word.len() - 1].to_string()
    } else {
        word.to_string()
    }
}

fn stop_words() -> &'static [&'static str] {
    &[
        "que", "con", "para", "por", "una", "uno", "del", "los", "las", "sin", "como", "antes",
        "desde", "entonces", "son", "esta", "este", "de", "el", "la", "en", "un",
    ]
}

fn exact_normalized_match(left: &str, right: &str) -> f32 {
    if normalize_text(left).trim() == normalize_text(right).trim() {
        1.0
    } else {
        0.0
    }
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

fn clean_line(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or(text)
        .trim()
        .trim_matches('"')
        .trim_matches('¿')
        .trim()
        .to_string()
}

fn escape_field(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn unescape_field(value: &str) -> String {
    let mut out = String::new();
    let mut escaped = false;
    for ch in value.chars() {
        if escaped {
            match ch {
                't' => out.push('\t'),
                'n' => out.push('\n'),
                '\\' => out.push('\\'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn fractal_mesh_config(target_nodes: usize) -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 6,
        target_dimension: 2.72,
        target_nodes,
        base_radius: 0.0,
        lateral_link_weight: 0.32,
        parent_link_weight: 1.0,
    }
}

fn total_nodes() -> usize {
    env::var("SNGA_SEMEXEC_REGION_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REGION_SIZE)
        .max(DEFAULT_REGION_SIZE)
        * REGION_COUNT
}

fn config() -> SimplicialConfig {
    SimplicialConfig {
        width: 72,
        height: 40,
        spacing: 6.5,
        elasticity: 0.005,
        damping: 0.86,
        activation_threshold: 0.63,
        simplex_area_weight: 0.00012,
        max_active_agents: 448,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 14,
        rhythm_amplitude: 0.045,
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
        seed: 727,
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

fn has_flag(name: &str) -> bool {
    env::args().any(|arg| arg == name)
}
