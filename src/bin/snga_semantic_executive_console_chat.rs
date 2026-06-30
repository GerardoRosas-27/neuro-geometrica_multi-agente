use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
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
    ConceptBinder = 1,
    SemanticControl = 2,
    ExecutiveLogicDlpfc = 3,
    WorkingMemory = 4,
    Planner = 5,
    ControlGate = 6,
    VisualSlot = 7,
    AuditorySlot = 8,
    SomatosensorySlot = 9,
    LinguisticSlot = 10,
    EpisodicSlot = 11,
}

#[derive(Clone, Copy)]
struct AdapterLesson {
    user_input: &'static str,
    linguistic_intent: &'static str,
    internal_concepts: &'static str,
    control_task: &'static str,
    response_frame: &'static str,
    verbal_intent: &'static str,
    ideal_response: &'static str,
}

struct SngaChatDecision<'a> {
    lesson: &'a AdapterLesson,
    trusted: bool,
    lexical_score: f32,
    concept_overlap: f32,
    frame_overlap: f32,
    verification_overlap: f32,
    confidence: f32,
    concept_prediction: Vec<(usize, f32)>,
    frame_prediction: Vec<(usize, f32)>,
}

#[derive(Clone, Debug)]
struct LearnedTeaching {
    prompt: String,
    teaching: String,
    response: String,
}

fn main() {
    let state_path =
        env::var("SNGA_SEMEXEC_ADAPTER_STATE").unwrap_or_else(|_| DEFAULT_STATE_PATH.to_string());
    let teachings_path =
        env::var("SNGA_SEMEXEC_TEACHINGS").unwrap_or_else(|_| DEFAULT_TEACHINGS_PATH.to_string());
    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes()));
    match network.load_persistent_state(&state_path) {
        Ok(report) => println!(
            "SNGA semexec chat cargado: state={} agentes={} aristas={} causales={}",
            state_path, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            eprintln!("No pude cargar el estado semantico-ejecutivo {state_path}: {err}");
            return;
        }
    }
    let mut teachings = load_teachings(&teachings_path);
    if !teachings.is_empty() {
        println!(
            "ensenanzas consola cargadas: {} desde {}",
            teachings.len(),
            teachings_path
        );
    }

    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("--once") {
        let prompt = args.iter().skip(1).cloned().collect::<Vec<_>>().join(" ");
        if prompt.trim().is_empty() {
            eprintln!("Uso: cargo run --bin snga_semantic_executive_console_chat -- --once \"tu pregunta\"");
            return;
        }
        answer_once(&mut network, prompt.trim(), &teachings);
        return;
    }

    println!(
        "Chat SNGA + LLM periferico. La logica viene del sustrato; el LLM solo interpreta/verbaliza."
    );
    println!("Escribe 'salir' para terminar.");
    let mut pending_teach_prompt = None::<String>;
    loop {
        print!("usuario> ");
        io::stdout().flush().ok();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }
        if let Some(prompt_to_teach) = pending_teach_prompt.take() {
            teach_snga_from_user(
                &mut network,
                &mut teachings,
                &state_path,
                &teachings_path,
                &prompt_to_teach,
                input,
            );
            continue;
        }
        if !answer_once(&mut network, input, &teachings) {
            pending_teach_prompt = Some(input.to_string());
        }
    }
}

fn answer_once(
    network: &mut SimplicialNetwork,
    prompt: &str,
    teachings: &[LearnedTeaching],
) -> bool {
    if let Some(teaching) = match_teaching(prompt, teachings) {
        let context = build_teaching_context(network, prompt, teaching);
        let response = render_with_peripheral_or_symbolic(&context, Some(&teaching.response));
        println!("snga-logica> {}", teaching.response);
        println!("llm-periferico({})> {}", response.engine, response.text);
        println!(
            "diagnostico> trusted=true source=ensenanza_consola lexical={:.1}%",
            lexical_overlap(prompt, &teaching.prompt)
                .max(lexical_overlap(prompt, &teaching.teaching))
                * 100.0
        );
        return true;
    }

    let decision = decide_with_snga(network, prompt);
    let context = build_linguistic_context(network, prompt, &decision);
    let response = if decision.trusted {
        render_with_peripheral_or_symbolic(&context, Some(decision.lesson.ideal_response))
    } else {
        render_teaching_question(prompt, &context)
    };

    if decision.trusted {
        println!("snga-logica> {}", decision.lesson.ideal_response);
    } else {
        println!(
            "snga-logica> fuera_de_dominio: el sustrato no tiene una ruta logica confiable para esta consulta."
        );
    }
    println!("llm-periferico({})> {}", response.engine, response.text);
    println!(
        "diagnostico> trusted={} lexical={:.1}% intent=\"{}\" concept_overlap={:.1}% frame_overlap={:.1}% verify_overlap={:.1}% confidence={:.3}",
        decision.trusted,
        decision.lexical_score * 100.0,
        decision.lesson.linguistic_intent,
        decision.concept_overlap * 100.0,
        decision.frame_overlap * 100.0,
        decision.verification_overlap * 100.0,
        decision.confidence
    );
    !decision
        .trusted
        .then(|| {
            println!(
                "ensenanza> responde explicando que significa o como debo contestar a: \"{}\"",
                prompt
            );
        })
        .is_none()
}

fn decide_with_snga<'a>(network: &SimplicialNetwork, prompt: &str) -> SngaChatDecision<'a> {
    let lessons = validation_lessons();
    let llm_input = linguistic_text_pattern("gemma_input", prompt, network.agents.len());
    let concept_prediction = network.infer_transitive_from(&llm_input, 3, 128);
    let frame_prediction = network.infer_transitive_from(&llm_input, 7, 256);
    let concept_ids = ids(&concept_prediction);
    let frame_ids = ids(&frame_prediction);

    let mut best = None::<(usize, f32, f32, f32, f32)>;
    for (idx, lesson) in lessons.iter().enumerate() {
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
        let verification = regional_pattern(
            Region::SemanticControl,
            "verification_same_state",
            &format!("{} {}", lesson.internal_concepts, lesson.response_frame),
            PATTERN_SIZE,
            network.agents.len(),
        );
        let output_pattern =
            linguistic_text_pattern("gemma_output", lesson.ideal_response, network.agents.len());
        let output_prediction = network.predict_next_pattern(&output_pattern, 1, 128);
        let concept_overlap = overlap_ratio(&concept_ids, &concept);
        let frame_overlap = overlap_ratio(&frame_ids, &planner);
        let verification_overlap = overlap_ratio(&ids(&output_prediction), &verification);
        let lexical = lexical_overlap(prompt, lesson.user_input)
            .max(lexical_overlap(prompt, lesson.internal_concepts))
            .max(lexical_overlap(prompt, lesson.response_frame));
        let score =
            concept_overlap * 5.0 + frame_overlap * 5.0 + verification_overlap + lexical * 0.75;

        match best {
            Some((_, best_score, _, _, _)) if best_score >= score => {}
            _ => best = Some((idx, score, concept_overlap, frame_overlap, lexical)),
        }
    }

    let (idx, _, concept_overlap, frame_overlap, lexical_score) =
        best.unwrap_or((0, 0.0, 0.0, 0.0, 0.0));
    let lesson = &lessons[idx];
    let verification = regional_pattern(
        Region::SemanticControl,
        "verification_same_state",
        &format!("{} {}", lesson.internal_concepts, lesson.response_frame),
        PATTERN_SIZE,
        network.agents.len(),
    );
    let output_pattern =
        linguistic_text_pattern("gemma_output", lesson.ideal_response, network.agents.len());
    let output_prediction = network.predict_next_pattern(&output_pattern, 1, 128);
    let verification_overlap = overlap_ratio(&ids(&output_prediction), &verification);
    let confidence = confidence(&concept_prediction) + confidence(&frame_prediction);

    SngaChatDecision {
        lesson,
        trusted: lexical_score >= MIN_LEXICAL_TRUST,
        lexical_score,
        concept_overlap,
        frame_overlap,
        verification_overlap,
        confidence,
        concept_prediction,
        frame_prediction,
    }
}

fn build_linguistic_context(
    network: &SimplicialNetwork,
    prompt: &str,
    decision: &SngaChatDecision<'_>,
) -> LinguisticContext {
    let mut top_agents = decision
        .concept_prediction
        .iter()
        .chain(decision.frame_prediction.iter())
        .copied()
        .collect::<Vec<_>>();
    top_agents.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_agents.truncate(16);

    let (inferred_intent, memory_summary) = if decision.trusted {
        (
            format!(
            "SNGA_INTENCION={}; CONCEPTO={}; CONTROL={}; FRAME={}; VERBAL={}",
            decision.lesson.linguistic_intent,
            decision.lesson.internal_concepts,
            decision.lesson.control_task,
            decision.lesson.response_frame,
            decision.lesson.verbal_intent
            ),
            format!(
            "SNGA ya hizo la inferencia logica en el sustrato. Respuesta autorizada: '{}'. No agregues hechos nuevos; solo verbaliza esta conclusion. nodos={} energia={:.1} concept_overlap={:.1}% frame_overlap={:.1}% verify_overlap={:.1}%",
            decision.lesson.ideal_response,
            network.agents.len(),
            network.total_free_energy(),
            decision.concept_overlap * 100.0,
            decision.frame_overlap * 100.0,
            decision.verification_overlap * 100.0
            ),
        )
    } else {
        (
            "SNGA_SIN_RUTA_LOGICA_CONFIABLE".to_string(),
            format!(
                "El sustrato SNGA no encontro una ruta confiable para esta consulta. No uses ninguna respuesta memorizada de SNGA. Si respondes, hazlo como periferico linguistico general y aclara que no proviene de una inferencia SNGA validada. nodos={} energia={:.1} mejor_intento={} lexical={:.1}%",
                network.agents.len(),
                network.total_free_energy(),
                decision.lesson.linguistic_intent,
                decision.lexical_score * 100.0
            ),
        )
    };

    LinguisticContext {
        user_prompt: prompt.to_string(),
        inferred_intent,
        geometric_projection: snga::simplicial::ConceptProjection { top_agents },
        memory_summary,
    }
}

fn render_with_peripheral_or_symbolic(
    context: &LinguisticContext,
    forced_response: Option<&str>,
) -> snga::linguistic_engine::LinguisticResponse {
    if env::var("SNGA_SEMEXEC_SNG_ONLY").ok().as_deref() == Some("1") {
        return snga::linguistic_engine::LinguisticResponse {
            text: forced_response
                .unwrap_or("SNGA no tiene respuesta simbolica para esta consulta.")
                .to_string(),
            engine: "snga-semexec-symbolic".to_string(),
        };
    }

    let model = env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string());
    let host = env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string());
    let engine = OllamaGemmaEngine { host, model };
    engine.generate(context).unwrap_or_else(|err| {
        eprintln!("LLM periferico no disponible; usando respuesta SNGA simbolica: {err}");
        snga::linguistic_engine::LinguisticResponse {
            text: forced_response
                .unwrap_or("SNGA no tiene respuesta simbolica para esta consulta.")
                .to_string(),
            engine: "snga-semexec-symbolic".to_string(),
        }
    })
}

fn render_teaching_question(
    prompt: &str,
    context: &LinguisticContext,
) -> snga::linguistic_engine::LinguisticResponse {
    if env::var("SNGA_SEMEXEC_SNG_ONLY").ok().as_deref() == Some("1") {
        return snga::linguistic_engine::LinguisticResponse {
            text: format!(
                "No conozco una ruta SNGA para \"{prompt}\". Que significa y como debo responder?"
            ),
            engine: "snga-semexec-symbolic-teacher".to_string(),
        };
    }

    render_with_peripheral_or_symbolic(
        context,
        Some(&format!(
            "No conozco una ruta SNGA para \"{prompt}\". Que significa y como debo responder?"
        )),
    )
}

fn build_teaching_context(
    network: &SimplicialNetwork,
    prompt: &str,
    teaching: &LearnedTeaching,
) -> LinguisticContext {
    LinguisticContext {
        user_prompt: prompt.to_string(),
        inferred_intent: format!(
            "SNGA_ENSENANZA_APRENDIDA={}; RESPUESTA={}",
            teaching.teaching, teaching.response
        ),
        geometric_projection: network.project_active_state(8),
        memory_summary: format!(
            "Esta respuesta fue ensenada por el usuario y entrenada como ruta SNGA. No inventes hechos fuera de esta ensenanza. prompt_original='{}'",
            teaching.prompt
        ),
    }
}

fn teach_snga_from_user(
    network: &mut SimplicialNetwork,
    teachings: &mut Vec<LearnedTeaching>,
    state_path: &str,
    teachings_path: &str,
    prompt: &str,
    user_teaching: &str,
) {
    let response = synthesize_response_from_teaching(prompt, user_teaching);
    let teaching = LearnedTeaching {
        prompt: prompt.to_string(),
        teaching: user_teaching.to_string(),
        response,
    };

    train_teaching_route(network, &teaching);
    teachings.push(teaching.clone());

    match network.save_persistent_state(state_path) {
        Ok(report) => println!(
            "aprendido_snga> estado guardado agentes={} aristas={} causales={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => eprintln!("aprendido_snga> fallo guardando estado SNGA: {err}"),
    }
    if let Err(err) = save_teachings(teachings_path, teachings) {
        eprintln!("aprendido_snga> fallo guardando ensenanzas: {err}");
    }

    println!("aprendido_snga> \"{}\" -> {}", prompt, teaching.response);
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
    let output = linguistic_text_pattern("teach_output", &teaching.response, network.agents.len());
    for _ in 0..6 {
        network.learn_transition(&input, &meaning);
        network.learn_transition(&meaning, &frame);
        network.learn_transition(&frame, &output);
        network.learn_from_prediction_error(&input, &meaning, 2, 128, 0.12);
        network.learn_from_prediction_error(&input, &frame, 3, 128, 0.10);
    }

    let mut fused = input.clone();
    fused.extend(meaning.iter().copied());
    fused.extend(frame.iter().copied());
    fused.extend(output.iter().copied());
    fused.sort_unstable();
    fused.dedup();
    network.reinforce_coactivation_if_useful(&fused, 0.07, 0.94);
}

fn synthesize_response_from_teaching(prompt: &str, teaching: &str) -> String {
    let normalized_prompt = normalize_text(prompt);
    let normalized_teaching = normalize_text(teaching);
    if normalized_prompt == "hola" && normalized_teaching.contains("saludo") {
        "Hola.".to_string()
    } else if let Some((_, answer)) = teaching.split_once("puedes responder") {
        answer
            .trim()
            .trim_matches('"')
            .trim_end_matches('.')
            .to_string()
            + "."
    } else if let Some((_, answer)) = teaching.split_once("respuesta") {
        answer
            .trim()
            .trim_matches('"')
            .trim_end_matches('.')
            .to_string()
            + "."
    } else {
        teaching.trim().trim_end_matches('.').to_string() + "."
    }
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

fn exact_normalized_match(left: &str, right: &str) -> f32 {
    if normalize_text(left).trim() == normalize_text(right).trim() {
        1.0
    } else {
        0.0
    }
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
                prompt: unescape_field(parts[0]),
                teaching: unescape_field(parts[1]),
                response: unescape_field(parts[2]),
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
        out.push_str(&escape_field(&teaching.prompt));
        out.push('\t');
        out.push_str(&escape_field(&teaching.teaching));
        out.push('\t');
        out.push_str(&escape_field(&teaching.response));
        out.push('\n');
    }
    fs::write(path, out)
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

fn validation_lessons() -> &'static [AdapterLesson] {
    &[
        AdapterLesson {
            user_input: "que es una manzana roja",
            linguistic_intent: "pregunta definicion objeto",
            internal_concepts: "manzana fruta roja redonda comestible",
            control_task: "activar concepto desde palabra y rasgos",
            response_frame: "definir manzana con rasgos visuales y categoria",
            verbal_intent: "expresar definicion breve de manzana",
            ideal_response: "Una manzana roja es una fruta comestible, redonda y dulce.",
        },
        AdapterLesson {
            user_input: "planea cena vegetariana sin carne",
            linguistic_intent: "instruccion plan con restriccion",
            internal_concepts: "vegetariano lechuga lentejas arroz excluir carne",
            control_task: "filtrar conceptos no permitidos por restriccion",
            response_frame: "proponer plato permitido y explicar exclusion",
            verbal_intent: "expresar plan vegetariano simple",
            ideal_response: "Puedo combinar lentejas con arroz y dejar fuera la carne.",
        },
        AdapterLesson {
            user_input: "banco en el parque",
            linguistic_intent: "desambiguar palabra por contexto",
            internal_concepts: "banco asiento parque no financiero",
            control_task: "seleccionar significado secundario por contexto",
            response_frame: "explicar que banco significa asiento",
            verbal_intent: "expresar desambiguacion contextual",
            ideal_response: "En ese contexto, banco significa un asiento para sentarse.",
        },
        AdapterLesson {
            user_input: "si llueve que pasa con el suelo",
            linguistic_intent: "pregunta causa efecto",
            internal_concepts: "lluvia agua suelo mojado",
            control_task: "seguir cadena causal simple",
            response_frame: "responder efecto probable de lluvia",
            verbal_intent: "expresar consecuencia causal",
            ideal_response: "Si llueve, el agua cae y el suelo probablemente se moja.",
        },
        AdapterLesson {
            user_input: "explica tu plan antes de responder",
            linguistic_intent: "pedir planificacion",
            internal_concepts: "meta pasos memoria trabajo respuesta",
            control_task: "ordenar pasos antes de verbalizar",
            response_frame: "mencionar objetivo restriccion y accion",
            verbal_intent: "expresar marco de plan",
            ideal_response:
                "Primero fijo el objetivo, luego reviso restricciones y finalmente doy la accion.",
        },
    ]
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
    for pair in normalized.split_whitespace().collect::<Vec<_>>().windows(2) {
        out.extend(regional_pattern(
            Region::LinguisticSlot,
            "word_pair",
            &format!("{}_{}", pair[0], pair[1]),
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
    let (start, len) = region_range(region, nodes);
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
            start + (hasher.finish() as usize % len.max(1))
        })
        .collect()
}

fn region_range(region: Region, nodes: usize) -> (usize, usize) {
    let region_size = inferred_region_size(nodes);
    let start = region as usize * region_size;
    let end = (start + region_size).min(nodes);
    (start, end.saturating_sub(start).max(1))
}

fn inferred_region_size(nodes: usize) -> usize {
    (nodes / REGION_COUNT).max(DEFAULT_REGION_SIZE)
}

fn legacy_region_size(nodes: usize) -> Option<usize> {
    let current = inferred_region_size(nodes);
    env::var("SNGA_SEMEXEC_LEGACY_REGION_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|size| *size > 0 && *size != current && *size * REGION_COUNT <= nodes)
}

fn total_nodes() -> usize {
    env::var("SNGA_SEMEXEC_REGION_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REGION_SIZE)
        .max(DEFAULT_REGION_SIZE)
        * REGION_COUNT
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
        "desde", "entonces", "son", "est", "esta", "este", "de", "el", "la", "en", "un",
    ]
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
