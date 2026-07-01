use snga::linguistic_engine::{LinguisticContext, LinguisticEngine, OllamaGemmaEngine};
use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{ConceptProjection, SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::thread;
use std::time::Duration;

const DEFAULT_BASE_STATE: &str = "data/snga_fractal_semantic_executive_substrate.snga";
const DEFAULT_STATE_PATH: &str = "data/snga_fractal_semantic_executive_gemma_adapter.snga";
const DEFAULT_PROGRESS_PATH: &str = "data/snga_fractal_semantic_executive_gemma_adapter.progress";
const DEFAULT_REGION_SIZE: usize = 8_192;
const REGION_COUNT: usize = 12;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_COMPRESS_EVERY: usize = 5;
const DEFAULT_RELAX_EVERY: usize = 1;
const DEFAULT_SAVE_EVERY: usize = 1;

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

#[derive(Clone, Debug)]
struct AdapterLesson {
    user_input: String,
    linguistic_intent: String,
    internal_concepts: String,
    control_task: String,
    response_frame: String,
    verbal_intent: String,
    ideal_response: String,
}

#[derive(Default)]
struct Progress {
    batches: usize,
    lessons: usize,
    gemma_failures: usize,
}

#[derive(Default)]
struct CompressionReport {
    removed_associative: usize,
    removed_causal: usize,
    knowledge_preserved: bool,
}

#[derive(Clone, Copy)]
struct EvalReport {
    total: usize,
    input_to_concept_hits: usize,
    input_to_frame_hits: usize,
    frame_to_verbal_hits: usize,
    verification_hits: usize,
    confidence: f32,
}

struct SemanticFeatureDetector {
    role: &'static str,
    pattern: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
enum BridgeStrategy {
    ChainOnly,
    DirectShortcuts,
    SemanticAnchors,
    Hybrid,
}

impl BridgeStrategy {
    fn from_label(label: &str) -> Self {
        match label {
            "chain" => Self::ChainOnly,
            "direct" => Self::DirectShortcuts,
            "anchors" => Self::SemanticAnchors,
            _ => Self::Hybrid,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ChainOnly => "chain",
            Self::DirectShortcuts => "direct",
            Self::SemanticAnchors => "anchors",
            Self::Hybrid => "hybrid",
        }
    }
}

fn main() {
    let batch_size = arg_value("--batch-size")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BATCH_SIZE)
        .max(1);
    let max_batches = arg_value("--batches").and_then(|value| value.parse::<usize>().ok());
    let sleep_ms = arg_value("--sleep-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let compress_every = arg_value("--compress-every")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_COMPRESS_EVERY);
    let relax_every = arg_value("--relax-every")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_RELAX_EVERY);
    let save_every = arg_value("--save-every")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SAVE_EVERY)
        .max(1);
    let eval_every = arg_value("--eval-every")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let state_path = arg_value("--state").unwrap_or_else(|| DEFAULT_STATE_PATH.to_string());
    let base_state = arg_value("--base").unwrap_or_else(|| DEFAULT_BASE_STATE.to_string());
    let progress_path =
        arg_value("--progress").unwrap_or_else(|| DEFAULT_PROGRESS_PATH.to_string());
    let bridge_strategy = BridgeStrategy::from_label(
        &arg_value("--input-bridge").unwrap_or_else(|| "hybrid".to_string()),
    );
    let offline_fallback = has_flag("--offline-fallback");
    let skip_steps = has_flag("--skip-steps");

    let engine = OllamaGemmaEngine {
        host: env::var("SNGA_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string()),
        model: env::var("SNGA_GEMMA_MODEL").unwrap_or_else(|_| "gemma2:2b".to_string()),
    };

    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes()));
    let loaded = load_state_or_base(&mut network, &state_path, &base_state);
    network.enable_neural_oscillations();
    let mut progress = load_progress(&progress_path).unwrap_or_default();

    println!("SNGA semantic-executive Gemma linguistic adapter trainer");
    println!(
        "loaded={} model={} state={} base={} progress={} batch_size={} max_batches={:?} save_every={} eval_every={} relax_every={} compress_every={} input_bridge={} offline_fallback={} skip_steps={}",
        loaded,
        engine.model,
        state_path,
        base_state,
        progress_path,
        batch_size,
        max_batches,
        save_every,
        eval_every,
        relax_every,
        compress_every,
        bridge_strategy.label(),
        offline_fallback,
        skip_steps
    );

    loop {
        if let Some(limit) = max_batches {
            if progress.batches >= limit {
                break;
            }
        }

        let (lessons, used_fallback) = if offline_fallback {
            (fallback_lessons(batch_size, progress.batches), true)
        } else {
            generate_dataset(&engine, batch_size, progress.batches)
        };
        if used_fallback {
            progress.gemma_failures += 1;
        }

        for lesson in &lessons {
            train_adapter_lesson(&mut network, lesson, bridge_strategy, skip_steps);
            progress.lessons += 1;
        }
        progress.batches += 1;

        let eval = if eval_every > 0 && progress.batches % eval_every == 0 {
            evaluate_adapter(&network)
        } else {
            EvalReport {
                total: validation_lessons().len(),
                input_to_concept_hits: 0,
                input_to_frame_hits: 0,
                frame_to_verbal_hits: 0,
                verification_hits: 0,
                confidence: 0.0,
            }
        };
        let adjusted = if relax_every > 0 && progress.batches % relax_every == 0 {
            network.anneal_active_edge_rest_lengths(0.12, 1.05)
        } else {
            0
        };
        let compression = if compress_every > 0 && progress.batches % compress_every == 0 {
            compress_linguistic_adapter(&mut network)
        } else {
            CompressionReport {
                knowledge_preserved: true,
                ..CompressionReport::default()
            }
        };

        if progress.batches % save_every == 0 {
            save_all(&network, &progress, &state_path, &progress_path, "batch");
        }

        let stats = network.plasticity_stats();
        println!(
            "batch={} lessons={} gemma_fallback={} eval_input_concept={}/{} eval_input_frame={}/{} eval_frame_verbal={}/{} eval_verify={}/{} conf={:.3} edges={} assoc={} cells={} causal={} adjusted={} compressed_assoc={} compressed_causal={} compressed_ok={} energy={:.1}",
            progress.batches,
            progress.lessons,
            used_fallback,
            eval.input_to_concept_hits,
            eval.total,
            eval.input_to_frame_hits,
            eval.total,
            eval.frame_to_verbal_hits,
            eval.total,
            eval.verification_hits,
            eval.total,
            eval.confidence,
            stats.active_edges,
            stats.associative_edges,
            stats.semantic_cells,
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

fn load_state_or_base(network: &mut SimplicialNetwork, state_path: &str, base_state: &str) -> bool {
    if Path::new(state_path).exists() {
        return network.load_persistent_state(state_path).is_ok();
    }
    if Path::new(base_state).exists() {
        return network.load_persistent_state(base_state).is_ok();
    }
    println!("base_missing=true path={base_state}; using fresh fractal topology");
    false
}

fn generate_dataset(
    engine: &OllamaGemmaEngine,
    batch_size: usize,
    batch_idx: usize,
) -> (Vec<AdapterLesson>, bool) {
    let prompt = format!(
        "Genera {batch_size} lecciones para entrenar un adaptador linguistico SNGA con Gemma como periferico de entrada/salida.\n\
        Objetivo: texto de usuario -> patrones linguisticos -> nucleo semantico-ejecutivo -> marco interno -> intencion verbal -> respuesta Gemma -> verificacion semantica.\n\
        Incluye ejemplos de conceptos concretos, ambiguedad, restricciones, planificacion, causa-efecto y explicacion breve.\n\
        Formato obligatorio, una leccion por linea, 7 campos separados por |:\n\
        entrada_usuario | intencion_linguistica | conceptos_internos | tarea_control | marco_respuesta_interno | intencion_verbal | respuesta_ideal\n\
        No uses numeracion, markdown ni texto extra. Lote {batch_idx}."
    );

    match ask_gemma(engine, prompt, "generar_dataset_adaptador_linguistico") {
        Some(text) => {
            let parsed = parse_lessons(&text);
            if parsed.is_empty() {
                (fallback_lessons(batch_size, batch_idx), true)
            } else {
                (fill_to_batch(parsed, batch_size, batch_idx), false)
            }
        }
        None => (fallback_lessons(batch_size, batch_idx), true),
    }
}

fn train_adapter_lesson(
    network: &mut SimplicialNetwork,
    lesson: &AdapterLesson,
    bridge_strategy: BridgeStrategy,
    skip_steps: bool,
) {
    let llm_input =
        linguistic_text_pattern("gemma_input", &lesson.user_input, network.agents.len());
    let input_intent = regional_pattern(
        Region::LinguisticSlot,
        "linguistic_intent",
        &lesson.linguistic_intent,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let input_key = regional_pattern(
        Region::LinguisticSlot,
        "input_bridge_key",
        &lesson.user_input,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let concept = regional_pattern(
        Region::SemanticHubAtl,
        "internal_concepts",
        &lesson.internal_concepts,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let binding = regional_pattern(
        Region::ConceptBinder,
        "binding",
        &format!("{} {}", lesson.user_input, lesson.internal_concepts),
        PATTERN_SIZE,
        network.agents.len(),
    );
    let control = regional_pattern(
        Region::SemanticControl,
        "control_task",
        &lesson.control_task,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let logic = regional_pattern(
        Region::ExecutiveLogicDlpfc,
        "logic",
        &format!("{} {}", lesson.control_task, lesson.response_frame),
        PATTERN_SIZE,
        network.agents.len(),
    );
    let working = regional_pattern(
        Region::WorkingMemory,
        "working_frame",
        &lesson.response_frame,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let planner = regional_pattern(
        Region::Planner,
        "response_frame",
        &lesson.response_frame,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let verbal =
        linguistic_text_pattern("verbal_intent", &lesson.verbal_intent, network.agents.len());
    let llm_output =
        linguistic_text_pattern("gemma_output", &lesson.ideal_response, network.agents.len());
    let verification = regional_pattern(
        Region::SemanticControl,
        "verification_same_state",
        &format!("{} {}", lesson.internal_concepts, lesson.response_frame),
        PATTERN_SIZE,
        network.agents.len(),
    );
    let feature_detectors = semantic_feature_detectors(network.agents.len(), lesson);
    train_semantic_feature_detectors(
        network,
        &feature_detectors,
        &llm_input,
        &input_intent,
        &concept,
        &control,
        &planner,
    );

    network.learn_transition(&llm_input, &input_intent);
    network.learn_transition(&input_intent, &concept);
    network.learn_transition(&concept, &binding);
    network.learn_transition(&binding, &control);
    network.learn_transition(&control, &logic);
    network.learn_transition(&logic, &working);
    network.learn_transition(&working, &planner);
    network.learn_transition(&planner, &verbal);
    network.learn_transition(&verbal, &llm_output);
    network.learn_transition(&llm_output, &verification);
    network.learn_transition(&verification, &concept);

    match bridge_strategy {
        BridgeStrategy::ChainOnly => {}
        BridgeStrategy::DirectShortcuts => train_direct_input_bridge(
            network,
            &llm_input,
            &input_key,
            &input_intent,
            &concept,
            &control,
            &planner,
        ),
        BridgeStrategy::SemanticAnchors => train_semantic_anchor_bridge(
            network,
            lesson,
            &llm_input,
            &input_intent,
            &concept,
            &control,
            &planner,
        ),
        BridgeStrategy::Hybrid => {
            train_direct_input_bridge(
                network,
                &llm_input,
                &input_key,
                &input_intent,
                &concept,
                &control,
                &planner,
            );
            train_semantic_anchor_bridge(
                network,
                lesson,
                &llm_input,
                &input_intent,
                &concept,
                &control,
                &planner,
            );
        }
    }

    reinforce_fused(
        network,
        [&llm_input, &input_intent, &concept, &binding, &control],
        0.05,
    );
    reinforce_fused(
        network,
        [&logic, &working, &planner, &verbal, &llm_output],
        0.05,
    );
    reinforce_fused(
        network,
        [&llm_output, &verification, &concept, &control],
        0.04,
    );
    train_predictive_error_corrections(
        network,
        &llm_input,
        &concept,
        &planner,
        &verbal,
        &llm_output,
        &verification,
    );

    network.clear_activity();
    network.set_attention_goal(&planner);
    network.inject_pattern(&llm_input, 1.15, 2);
    network.inject_pattern(&concept, 0.9, 1);
    network.inject_pattern(&llm_output, 0.75, 1);
    if !skip_steps {
        for _ in 0..6 {
            network.step();
        }
    }
    network.clear_attention_goal();
    network.clear_activity();
}

fn semantic_feature_detectors(
    nodes: usize,
    lesson: &AdapterLesson,
) -> Vec<SemanticFeatureDetector> {
    let mut detectors = vec![
        SemanticFeatureDetector {
            role: "intent",
            pattern: regional_pattern(
                Region::LinguisticSlot,
                "feature_intent",
                &lesson.linguistic_intent,
                PATTERN_SIZE,
                nodes,
            ),
        },
        SemanticFeatureDetector {
            role: "control",
            pattern: regional_pattern(
                Region::SemanticControl,
                "feature_control",
                &lesson.control_task,
                PATTERN_SIZE,
                nodes,
            ),
        },
        SemanticFeatureDetector {
            role: "frame",
            pattern: regional_pattern(
                Region::WorkingMemory,
                "feature_frame",
                &lesson.response_frame,
                PATTERN_SIZE,
                nodes,
            ),
        },
    ];

    for keyword in semantic_keywords(lesson).into_iter().take(8) {
        detectors.push(SemanticFeatureDetector {
            role: "keyword",
            pattern: regional_pattern(
                Region::ConceptBinder,
                "feature_keyword",
                &keyword,
                PATTERN_SIZE,
                nodes,
            ),
        });
    }
    detectors
}

fn train_semantic_feature_detectors(
    network: &mut SimplicialNetwork,
    detectors: &[SemanticFeatureDetector],
    llm_input: &Vec<usize>,
    input_intent: &Vec<usize>,
    concept: &Vec<usize>,
    control: &Vec<usize>,
    planner: &Vec<usize>,
) {
    for detector in detectors {
        for _ in 0..3 {
            network.learn_transition(llm_input, &detector.pattern);
            network.learn_transition(input_intent, &detector.pattern);
            network.learn_transition(&detector.pattern, concept);
            network.learn_transition(&detector.pattern, control);
            if detector.role == "frame" || detector.role == "control" {
                network.learn_transition(&detector.pattern, planner);
            }
        }
        reinforce_fused(network, [llm_input, &detector.pattern, concept], 0.06);
        reinforce_fused(
            network,
            [&detector.pattern, concept, control, planner],
            0.055,
        );
    }

    let detector_union = detectors
        .iter()
        .flat_map(|detector| detector.pattern.iter().copied())
        .collect::<Vec<_>>();
    if detector_union.is_empty() {
        return;
    }

    let detector_union = compact_bridge_pattern(&detector_union, 96);
    network.reinforce_coactivation_if_useful(&detector_union, 0.05, 0.94);
    network.learn_from_prediction_error(llm_input, &detector_union, 1, 192, 0.08);
    network.learn_from_prediction_error(&detector_union, concept, 1, 192, 0.08);
    network.learn_from_prediction_error(&detector_union, planner, 2, 192, 0.06);
}

fn train_predictive_error_corrections(
    network: &mut SimplicialNetwork,
    llm_input: &Vec<usize>,
    concept: &Vec<usize>,
    planner: &Vec<usize>,
    verbal: &Vec<usize>,
    llm_output: &Vec<usize>,
    verification: &Vec<usize>,
) {
    network.learn_from_prediction_error(llm_input, concept, 3, 128, 0.12);
    network.learn_from_prediction_error(llm_input, planner, 7, 256, 0.10);
    network.learn_from_prediction_error(planner, verbal, 2, 128, 0.08);
    network.learn_from_prediction_error(llm_output, verification, 1, 128, 0.08);
}

fn train_direct_input_bridge(
    network: &mut SimplicialNetwork,
    llm_input: &Vec<usize>,
    input_key: &Vec<usize>,
    input_intent: &Vec<usize>,
    concept: &Vec<usize>,
    control: &Vec<usize>,
    planner: &Vec<usize>,
) {
    for _ in 0..4 {
        network.learn_transition(llm_input, input_key);
        network.learn_transition(input_key, concept);
        network.learn_transition(input_key, control);
        network.learn_transition(input_key, planner);
        network.learn_transition(llm_input, concept);
        network.learn_transition(input_intent, concept);
        network.learn_transition(input_intent, control);
    }
    reinforce_fused(
        network,
        [llm_input, input_key, input_intent, concept],
        0.075,
    );
    reinforce_fused(network, [input_key, concept, control, planner], 0.065);
}

fn train_semantic_anchor_bridge(
    network: &mut SimplicialNetwork,
    lesson: &AdapterLesson,
    llm_input: &Vec<usize>,
    input_intent: &Vec<usize>,
    concept: &Vec<usize>,
    control: &Vec<usize>,
    planner: &Vec<usize>,
) {
    let input_anchor = regional_pattern(
        Region::ConceptBinder,
        "input_semantic_anchor",
        &lesson.user_input,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let concept_anchor = regional_pattern(
        Region::SemanticControl,
        "concept_semantic_anchor",
        &lesson.internal_concepts,
        PATTERN_SIZE,
        network.agents.len(),
    );
    let frame_anchor = regional_pattern(
        Region::WorkingMemory,
        "frame_semantic_anchor",
        &lesson.response_frame,
        PATTERN_SIZE,
        network.agents.len(),
    );

    for _ in 0..4 {
        network.learn_transition(llm_input, &input_anchor);
        network.learn_transition(input_intent, &input_anchor);
        network.learn_transition(&input_anchor, concept);
        network.learn_transition(&input_anchor, control);
        network.learn_transition(&input_anchor, &concept_anchor);
        network.learn_transition(&concept_anchor, concept);
        network.learn_transition(&concept_anchor, control);
        network.learn_transition(&concept_anchor, &frame_anchor);
        network.learn_transition(&frame_anchor, planner);
    }

    for keyword in semantic_keywords(lesson).iter().take(10) {
        let word = regional_pattern(
            Region::LinguisticSlot,
            "semantic_keyword",
            keyword,
            PATTERN_SIZE,
            network.agents.len(),
        );
        network.learn_transition(&word, &input_anchor);
        network.learn_transition(&word, &concept_anchor);
        network.learn_transition(&word, concept);
        reinforce_fused(
            network,
            [&word, &input_anchor, &concept_anchor, concept],
            0.055,
        );
    }

    reinforce_fused(
        network,
        [
            llm_input,
            input_intent,
            &input_anchor,
            &concept_anchor,
            concept,
            control,
        ],
        0.06,
    );
    reinforce_fused(network, [&concept_anchor, &frame_anchor, planner], 0.055);
}

fn semantic_keywords(lesson: &AdapterLesson) -> Vec<String> {
    let mut words = Vec::new();
    for text in [
        &lesson.user_input,
        &lesson.linguistic_intent,
        &lesson.internal_concepts,
        &lesson.control_task,
        &lesson.response_frame,
    ] {
        for word in normalize_text(text).split_whitespace() {
            if word.len() <= 2 || stop_words().contains(&word) {
                continue;
            }
            let word = word.to_string();
            if !words.contains(&word) {
                words.push(word);
            }
        }
    }
    words
}

fn stop_words() -> &'static [&'static str] {
    &[
        "que", "con", "para", "por", "una", "uno", "del", "los", "las", "sin", "como", "antes",
        "desde", "entonces",
    ]
}

fn reinforce_fused<const N: usize>(
    network: &mut SimplicialNetwork,
    parts: [&Vec<usize>; N],
    learning_rate: f32,
) {
    let mut fused = Vec::new();
    for part in parts {
        fused.extend(part.iter().copied());
    }
    fused.sort_unstable();
    fused.dedup();
    let fused = compact_bridge_pattern(&fused, 96);
    network.reinforce_coactivation_if_useful(&fused, learning_rate, 0.92);
}

fn compact_bridge_pattern(pattern: &[usize], limit: usize) -> Vec<usize> {
    if pattern.len() <= limit {
        return pattern.to_vec();
    }
    let stride = (pattern.len() as f32 / limit as f32).ceil() as usize;
    pattern
        .iter()
        .step_by(stride.max(1))
        .take(limit)
        .copied()
        .collect()
}

fn evaluate_adapter(network: &SimplicialNetwork) -> EvalReport {
    let validation = validation_lessons();
    let mut report = EvalReport {
        total: validation.len(),
        input_to_concept_hits: 0,
        input_to_frame_hits: 0,
        frame_to_verbal_hits: 0,
        verification_hits: 0,
        confidence: 0.0,
    };

    for lesson in &validation {
        let llm_input =
            linguistic_text_pattern("gemma_input", &lesson.user_input, network.agents.len());
        let concept = regional_pattern(
            Region::SemanticHubAtl,
            "internal_concepts",
            &lesson.internal_concepts,
            PATTERN_SIZE,
            network.agents.len(),
        );
        let planner = regional_pattern(
            Region::Planner,
            "response_frame",
            &lesson.response_frame,
            PATTERN_SIZE,
            network.agents.len(),
        );
        let verbal =
            linguistic_text_pattern("verbal_intent", &lesson.verbal_intent, network.agents.len());
        let llm_output =
            linguistic_text_pattern("gemma_output", &lesson.ideal_response, network.agents.len());
        let verification = regional_pattern(
            Region::SemanticControl,
            "verification_same_state",
            &format!("{} {}", lesson.internal_concepts, lesson.response_frame),
            PATTERN_SIZE,
            network.agents.len(),
        );

        let input_pred = network.infer_transitive_from(&llm_input, 7, 256);
        let concept_pred = network.infer_transitive_from(&llm_input, 3, 128);
        let frame_pred = network.infer_transitive_from(&planner, 2, 128);
        let output_pred = network.predict_next_pattern(&llm_output, 1, 128);
        let input_ids = ids(&input_pred);
        let concept_ids = ids(&concept_pred);
        let frame_ids = ids(&frame_pred);
        let output_ids = ids(&output_pred);

        report.input_to_concept_hits += usize::from(overlap_ratio(&concept_ids, &concept) > 0.0);
        report.input_to_frame_hits += usize::from(overlap_ratio(&input_ids, &planner) > 0.0);
        report.frame_to_verbal_hits += usize::from(overlap_ratio(&frame_ids, &verbal) > 0.0);
        report.verification_hits += usize::from(overlap_ratio(&output_ids, &verification) > 0.0);
        report.confidence += confidence(&input_pred)
            + confidence(&concept_pred)
            + confidence(&frame_pred)
            + confidence(&output_pred);
    }

    report.confidence /= (report.total.max(1) * 4) as f32;
    report
}

fn compress_linguistic_adapter(network: &mut SimplicialNetwork) -> CompressionReport {
    let reference = validation_signature(network);
    let (start, end) = region_range(Region::LinguisticSlot, network.agents.len());
    let mut report = CompressionReport {
        removed_associative: 0,
        removed_causal: 0,
        knowledge_preserved: true,
    };

    let mut assoc_chunk = 60_000;
    let mut attempts = 0;
    while assoc_chunk > 0 && attempts < 6 {
        attempts += 1;
        let before = network.clone();
        let removed = network.prune_low_value_associative_edges_in_range(assoc_chunk, start, end);
        if removed == 0 {
            break;
        }
        if validation_signature(network) == reference {
            report.removed_associative += removed;
        } else {
            *network = before;
            assoc_chunk /= 2;
        }
    }

    let mut causal_chunk = 20_000;
    let mut causal_attempts = 0;
    while causal_chunk > 0 && causal_attempts < 6 {
        causal_attempts += 1;
        let before = network.clone();
        let removed = network.prune_low_value_causal_edges_in_range(causal_chunk, start, end);
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
        network.anneal_active_edge_rest_lengths(0.5, 1.05);
        report.knowledge_preserved = validation_signature(network) == reference;
    }
    report
}

fn validation_signature(network: &SimplicialNetwork) -> Vec<Vec<usize>> {
    let mut signatures = Vec::new();
    for lesson in validation_lessons() {
        let llm_input =
            linguistic_text_pattern("gemma_input", &lesson.user_input, network.agents.len());
        let planner = regional_pattern(
            Region::Planner,
            "response_frame",
            &lesson.response_frame,
            PATTERN_SIZE,
            network.agents.len(),
        );
        let llm_output =
            linguistic_text_pattern("gemma_output", &lesson.ideal_response, network.agents.len());
        signatures.push(ids(&network.infer_transitive_from(&llm_input, 7, 96)));
        signatures.push(ids(&network.predict_next_pattern(&planner, 1, 96)));
        signatures.push(ids(&network.predict_next_pattern(&llm_output, 1, 96)));
    }
    signatures
}

fn ask_gemma(engine: &OllamaGemmaEngine, prompt: String, intent: &str) -> Option<String> {
    let context = LinguisticContext {
        user_prompt: prompt,
        inferred_intent: intent.to_string(),
        geometric_projection: ConceptProjection {
            top_agents: Vec::new(),
        },
        memory_summary:
            "Gemma genera datos para el adaptador; SNGA debe conservar el estado interno."
                .to_string(),
    };
    engine.generate(&context).ok().map(|response| response.text)
}

fn parse_lessons(text: &str) -> Vec<AdapterLesson> {
    text.lines()
        .filter_map(|line| {
            let clean = line
                .trim()
                .trim_start_matches(|ch: char| ch.is_ascii_digit() || ch == '.' || ch == '-')
                .trim();
            let parts = clean.split('|').map(str::trim).collect::<Vec<_>>();
            (parts.len() >= 7).then(|| AdapterLesson {
                user_input: parts[0].to_string(),
                linguistic_intent: parts[1].to_string(),
                internal_concepts: parts[2].to_string(),
                control_task: parts[3].to_string(),
                response_frame: parts[4].to_string(),
                verbal_intent: parts[5].to_string(),
                ideal_response: parts[6].to_string(),
            })
        })
        .collect()
}

fn fill_to_batch(
    mut lessons: Vec<AdapterLesson>,
    batch_size: usize,
    batch_idx: usize,
) -> Vec<AdapterLesson> {
    if lessons.len() >= batch_size {
        lessons.truncate(batch_size);
        return lessons;
    }
    let fallback = fallback_lessons(batch_size, batch_idx);
    let mut idx = 0;
    while lessons.len() < batch_size {
        lessons.push(fallback[idx % fallback.len()].clone());
        idx += 1;
    }
    lessons
}

fn fallback_lessons(batch_size: usize, batch_idx: usize) -> Vec<AdapterLesson> {
    let seeds = validation_lessons();
    (0..batch_size)
        .map(|idx| seeds[(idx + batch_idx) % seeds.len()].clone())
        .collect()
}

fn validation_lessons() -> Vec<AdapterLesson> {
    vec![
        AdapterLesson {
            user_input: "que es una manzana roja".to_string(),
            linguistic_intent: "pregunta definicion objeto".to_string(),
            internal_concepts: "manzana fruta roja redonda comestible".to_string(),
            control_task: "activar concepto desde palabra y rasgos".to_string(),
            response_frame: "definir manzana con rasgos visuales y categoria".to_string(),
            verbal_intent: "expresar definicion breve de manzana".to_string(),
            ideal_response: "Una manzana roja es una fruta comestible, redonda y dulce."
                .to_string(),
        },
        AdapterLesson {
            user_input: "planea cena vegetariana sin carne".to_string(),
            linguistic_intent: "instruccion plan con restriccion".to_string(),
            internal_concepts: "vegetariano lechuga lentejas arroz excluir carne".to_string(),
            control_task: "filtrar conceptos no permitidos por restriccion".to_string(),
            response_frame: "proponer plato permitido y explicar exclusion".to_string(),
            verbal_intent: "expresar plan vegetariano simple".to_string(),
            ideal_response: "Puedo combinar lentejas con arroz y dejar fuera la carne.".to_string(),
        },
        AdapterLesson {
            user_input: "banco en el parque".to_string(),
            linguistic_intent: "desambiguar palabra por contexto".to_string(),
            internal_concepts: "banco asiento parque no financiero".to_string(),
            control_task: "seleccionar significado secundario por contexto".to_string(),
            response_frame: "explicar que banco significa asiento".to_string(),
            verbal_intent: "expresar desambiguacion contextual".to_string(),
            ideal_response: "En ese contexto, banco significa un asiento para sentarse."
                .to_string(),
        },
        AdapterLesson {
            user_input: "si llueve que pasa con el suelo".to_string(),
            linguistic_intent: "pregunta causa efecto".to_string(),
            internal_concepts: "lluvia agua suelo mojado".to_string(),
            control_task: "seguir cadena causal simple".to_string(),
            response_frame: "responder efecto probable de lluvia".to_string(),
            verbal_intent: "expresar consecuencia causal".to_string(),
            ideal_response: "Si llueve, el agua cae y el suelo probablemente se moja.".to_string(),
        },
        AdapterLesson {
            user_input: "explica tu plan antes de responder".to_string(),
            linguistic_intent: "pedir planificacion".to_string(),
            internal_concepts: "meta pasos memoria trabajo respuesta".to_string(),
            control_task: "ordenar pasos antes de verbalizar".to_string(),
            response_frame: "mencionar objetivo restriccion y accion".to_string(),
            verbal_intent: "expresar marco de plan".to_string(),
            ideal_response:
                "Primero fijo el objetivo, luego reviso restricciones y finalmente doy la accion."
                    .to_string(),
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

fn total_nodes() -> usize {
    DEFAULT_REGION_SIZE * REGION_COUNT
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
                "{label}: saved agents={} edges={} causal={} batches={} lessons={} gemma_failures={}",
                report.agents,
                report.edges,
                report.causal_edges,
                progress.batches,
                progress.lessons,
                progress.gemma_failures
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
            "gemma_failures" => progress.gemma_failures = value.parse().ok()?,
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
            "batches={}\nlessons={}\ngemma_failures={}\n",
            progress.batches, progress.lessons, progress.gemma_failures
        ),
    )
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
    env::args().skip(1).any(|arg| arg == name)
}
