//! Chat circadiano unificado: vigilia con Gemma adaptativo + sueño entrenando CTP.
//!
//! ```powershell
//! cargo run --release --bin native_gemma2_circadian_chat -- --chat dyamon
//! ```

use candle_core::quantized::gguf_file;
use candle_core::Tensor;
use cdt_rqm_epr::adaptive_gemma2::{
    AdaptiveGemma2Config, AdaptiveThermoMemory, RecalledLayerRoute,
};
use cdt_rqm_epr::gemma2_circadian_bridge::{
    load_or_create_hybrid, new_wake_record, persist_hybrid_session, run_sleep_phase,
    CircadianPaths, CircadianSleepConfig, DEFAULT_CIRCADIAN_ROOT, WakeJournal,
};
use cdt_rqm_epr::gemma2_thermo_hybrid_llm::{Gemma2ThermoHybridConfig, Gemma2ThermoHybridLlm};
use cdt_rqm_epr::gemma2_thermo_hybrid_session::sanitize_chat_name;
use cdt_rqm_epr::gemma_phasor_coupling::{GemmaPhasorCouplingConfig, GemmaPhasorWorker};
use cdt_rqm_epr::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2ForwardOutput, Gemma2Tokenizer,
    LayerExecutionMask, QuantizedGemma2,
};
use cdt_rqm_epr::native_gemma2_runtime::{chat_tokens, Gemma2GenerationConfig, Gemma2Session};
use cdt_rqm_epr::thermo_router::{ActivationFingerprint, TransformerActivationAdapter};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_CONTEXT: usize = 2_048;

#[derive(Clone, Debug)]
struct Config {
    model: Option<PathBuf>,
    chat_name: Option<String>,
    circadian_root: PathBuf,
    max_tokens: usize,
    context: usize,
    temperature: f64,
    top_p: f64,
    prompt: Option<String>,
    sleep_only: bool,
    device: String,
    thermo: bool,
}

struct PreparedForward {
    output: Gemma2ForwardOutput,
    mask: LayerExecutionMask,
    context_fingerprint: ActivationFingerprint,
    activation_fingerprint: ActivationFingerprint,
    route_id: Option<cdt_rqm_epr::thermo_router::RouteId>,
    quality: f32,
    fallback: bool,
    recalled_memory_tokens: usize,
}

struct CircadianSession {
    adaptive: AdaptiveThermoMemory,
    gemma_session: Gemma2Session,
    hybrid: Gemma2ThermoHybridLlm,
    journal: WakeJournal,
    paths: Option<CircadianPaths>,
    history: Vec<(String, String)>,
    turns: u64,
    thermo_created_at: u64,
    phasor: Option<GemmaPhasorWorker>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
    let model_id = format!("gemma2:{}", model_path.display());

    let paths = config
        .chat_name
        .as_deref()
        .map(|name| CircadianPaths::for_chat(&config.circadian_root, name))
        .transpose()?;

    let adaptive_root = paths
        .as_ref()
        .map(|paths| paths.adaptive_root.clone())
        .unwrap_or_else(|| PathBuf::from("data/native_gemma2_adaptive"));
    let mut adaptive = AdaptiveThermoMemory::load_or_new(
        &adaptive_root,
        model_id,
        AdaptiveGemma2Config::default(),
    )?;

    if config.sleep_only {
        if let Some(paths) = paths.as_ref() {
            let device = resolve_gemma2_device(&config.device)?;
            let mut file = File::open(&model_path)?;
            let content = gguf_file::Content::read(&mut file)?;
            let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
            let hybrid_config = Gemma2ThermoHybridConfig::default();
            let (mut hybrid, _) = load_or_create_hybrid(
                &model,
                paths,
                hybrid_config,
                &[],
                0,
                cdt_rqm_epr::gemma2_thermo_hybrid_session::unix_now(),
            )?;
            let journal = WakeJournal::open(paths.wake_journal.clone())?;
            let report = run_sleep_phase(
                &mut model,
                &mut hybrid,
                &mut adaptive,
                &journal,
                paths,
                &CircadianSleepConfig::default(),
            )?;
            persist_hybrid_session(paths, 0, 0, &[], &mut hybrid)?;
            print_sleep_report(report);
        } else {
            let report = adaptive.consolidate_sleep()?;
            println!(
                "sleep adaptativo: flushed={} routes={}",
                report.flushed, report.remaining_routes
            );
        }
        return Ok(());
    }

    print_banner();
    println!("GGUF: {}", model_path.display());
    if let Some(paths) = paths.as_ref() {
        println!("Sesión circadiana: {}", paths.chat_name);
        println!("Raíz: {}", paths.root.display());
    } else {
        println!("Modo efímero (usa --chat NOMBRE para persistir vigilia + sueño)");
    }

    let started = Instant::now();
    let device = resolve_gemma2_device(&config.device)?;
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
    let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
    let context_limit = config.context.min(model.max_context());

    let hybrid_config = Gemma2ThermoHybridConfig::default();
    let (hybrid, thermo_created_at) = if let Some(paths) = paths.as_ref() {
        load_or_create_hybrid(
            &model,
            paths,
            hybrid_config,
            &[],
            0,
            cdt_rqm_epr::gemma2_thermo_hybrid_session::unix_now(),
        )?
    } else {
        (
            Gemma2ThermoHybridLlm::for_gemma(&model, hybrid_config)?,
            cdt_rqm_epr::gemma2_thermo_hybrid_session::unix_now(),
        )
    };

    let journal = if let Some(paths) = paths.as_ref() {
        WakeJournal::open(paths.wake_journal.clone())?
    } else {
        WakeJournal::open(PathBuf::from("data/native_gemma2_circadian/_ephemeral/journal.jsonl"))?
    };

    let phasor = config
        .thermo
        .then(|| GemmaPhasorWorker::start(GemmaPhasorCouplingConfig::default()))
        .transpose()?;

    let mut session = CircadianSession {
        adaptive,
        gemma_session: Gemma2Session::new(),
        hybrid,
        journal,
        paths,
        history: Vec::new(),
        turns: 0,
        thermo_created_at,
        phasor,
    };

    println!(
        "Listo en {:.1}s | capas={} | contexto={} | atractores CTP={}",
        started.elapsed().as_secs_f64(),
        model.layer_count(),
        context_limit,
        session
            .hybrid
            .thermo_engine()
            .hybrid_engine()
            .attractors()
            .len(),
    );
    println!("Vigilia: Gemma adaptativo responde. Sueño (/sueño): entrena núcleo CTP.");

    if let Some(prompt) = config.prompt.as_deref() {
        wake_turn(&mut model, &tokenizer, &mut session, prompt, &config, context_limit)?;
        sleep_and_persist(&mut model, &mut session)?;
        return Ok(());
    }

    repl(&mut model, &tokenizer, &mut session, &config, context_limit)
}

fn repl(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    session: &mut CircadianSession,
    config: &Config,
    context_limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comandos: /sueño /limpiar /estado /salir");
    loop {
        print!("\nTú> ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        match input.to_ascii_lowercase().as_str() {
            "/salir" | "salir" | "exit" => break,
            "/limpiar" | "/clear" => {
                session.history.clear();
                session.gemma_session.reset(model);
                session.turns = 0;
                println!("Historial borrado (memorias adaptativa y CTP conservadas).");
                continue;
            }
            "/estado" | "/status" => {
                print_status(session);
                continue;
            }
            "/sueño" | "/sueno" | "/sleep" => {
                sleep_and_persist(model, session)?;
                continue;
            }
            "/ayuda" | "/help" => {
                print_help();
                continue;
            }
            _ if input.starts_with('/') => {
                println!("Comando desconocido. Usa /ayuda.");
                continue;
            }
            _ => {}
        }
        if let Err(error) = wake_turn(model, tokenizer, session, input, config, context_limit) {
            eprintln!("Error: {error}");
        }
    }
    sleep_and_persist(model, session)?;
    println!("Hasta pronto.");
    Ok(())
}

fn wake_turn(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    session: &mut CircadianSession,
    input: &str,
    config: &Config,
    context_limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let prompt_limit = context_limit.saturating_sub(config.max_tokens).max(32);
    let prompt_tokens = chat_tokens(tokenizer, &session.history, input, prompt_limit)?;
    let prepared = prepare_forward(model, &mut session.adaptive, &prompt_tokens)?;
    eprintln!(
        "[vigilia route={} layers={}/{} quality={:.3} fallback={}]",
        prepared
            .route_id
            .map(|route| route.0.to_string())
            .unwrap_or_else(|| "new".to_string()),
        prepared.mask.executed_count(),
        prepared.mask.layer_count(),
        prepared.quality,
        prepared.fallback,
    );

    session.gemma_session.adopt_prefill(
        &prompt_tokens,
        Some(&prepared.mask),
        prepared.output.logits,
    )?;
    print!("\nGemma> ");
    io::stdout().flush()?;

    let generation = session.gemma_session.generate_observed(
        model,
        tokenizer,
        &prompt_tokens,
        Some(&prepared.mask),
        Gemma2GenerationConfig {
            max_tokens: config.max_tokens,
            context_limit,
            temperature: config.temperature,
            top_p: config.top_p,
            seed: 0x4745_4D4D_4132,
        },
        |fragment| {
            print!("{fragment}");
            let _ = io::stdout().flush();
        },
        |token, position| {
            if let Some(worker) = session.phasor.as_ref() {
                worker.observe_token(token, position);
            }
        },
        |_| false,
    )?;
    println!();

    session.adaptive.observe(
        prepared.context_fingerprint,
        prepared.activation_fingerprint,
        prepared.mask.clone(),
        &prompt_tokens,
        prepared.quality,
        prepared.route_id,
        prepared.fallback,
    )?;

    session.turns = session.turns.saturating_add(1);
    let record = new_wake_record(
        session.turns,
        input,
        &generation.text,
        &prompt_tokens,
        &generation.token_ids,
        prepared.quality,
        prepared.mask.executed_count(),
    );
    session.journal.append(&record)?;

    session
        .history
        .push((input.to_string(), generation.text.clone()));
    Ok(())
}

fn sleep_and_persist(
    model: &mut QuantizedGemma2,
    session: &mut CircadianSession,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(paths) = session.paths.clone() {
        println!("Entrando en fase de sueño...");
        let report = run_sleep_phase(
            model,
            &mut session.hybrid,
            &mut session.adaptive,
            &session.journal,
            &paths,
            &CircadianSleepConfig::default(),
        )?;
        persist_hybrid_session(
            &paths,
            session.thermo_created_at,
            session.turns,
            &session.history,
            &mut session.hybrid,
        )?;
        session.adaptive.save()?;
        print_sleep_report(report);
        println!("Persistido: adaptive + thermo.cdt + dataset de sueño");
    } else {
        let report = session.adaptive.consolidate_sleep()?;
        println!(
            "Sueño adaptativo: flushed={} routes={}",
            report.flushed, report.remaining_routes
        );
    }
    Ok(())
}

fn print_sleep_report(report: cdt_rqm_epr::gemma2_circadian_bridge::CircadianSleepReport) {
    println!("── Sueño ──");
    println!(
        "  Adaptativo: flushed={} routes={} relaciones_podadas={}",
        report.adaptive.flushed, report.adaptive.remaining_routes, report.adaptive.pruned_relations
    );
    println!("  Dataset CTP: {} secuencias exportadas", report.dataset_entries);
    println!(
        "  Entrenamiento CTP: seq={} ventanas={} mse={:.4} atractores={}",
        report.thermo.sequences,
        report.thermo.windows,
        report.thermo.mean_alignment_mse,
        report.thermo.attractors_after,
    );
}

fn print_status(session: &CircadianSession) {
    let attractors = session
        .hybrid
        .thermo_engine()
        .hybrid_engine()
        .attractors()
        .len();
    println!("── Estado circadiano ──");
    println!("  Turnos vigilia:     {}", session.turns);
    println!("  Mensajes historial: {}", session.history.len());
    println!("  Atractores CTP:     {}", attractors);
    println!(
        "  Rutas adaptativas:  {}",
        session.adaptive.router.registry.routes().len()
    );
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CHAT CIRCADIANO — Gemma adaptativo (día) + CTP (sueño)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn print_help() {
    println!(
        r"
Comandos:
  /sueño, /sleep   Consolida adaptativo + exporta dataset + entrena CTP + guarda
  /limpiar         Borra historial visible (conserva memorias)
  /estado          Métricas de vigilia
  /salir           Sueño + guardado + salir

Persistencia (--chat NOMBRE):
  data/native_gemma2_circadian/NOMBRE/adaptive/   memoria adaptativa
  data/native_gemma2_circadian/NOMBRE/thermo.cdt  núcleo CTP entrenado
  data/native_gemma2_circadian/NOMBRE/wake/         experiencias del día
  data/native_gemma2_circadian/NOMBRE/sleep/        dataset + reporte entrenamiento

Comando unificado:
  cargo run --release --bin native_gemma2_circadian_chat -- --chat NOMBRE
"
    );
}

fn prepare_forward(
    model: &mut QuantizedGemma2,
    memory: &mut AdaptiveThermoMemory,
    prompt_tokens: &[u32],
) -> Result<PreparedForward, Box<dyn std::error::Error>> {
    let prompt = Tensor::new(prompt_tokens, model.device())?.unsqueeze(0)?;
    let context_fingerprint = memory.context_fingerprint(prompt_tokens);
    let recalled = memory.recall(&context_fingerprint, model.layer_count());
    if let Some(route) = recalled.as_ref().filter(|_| !memory.should_revalidate()) {
        model.clear_kv_cache();
        let output = model.forward_with_mask(&prompt, 0, Some(&route.mask), true, false)?;
        let logits = output.logits.squeeze(0)?.to_vec1::<f32>()?;
        let self_quality = output_confidence(&logits);
        if self_quality >= 0.05 && logits.iter().all(|value| value.is_finite()) {
            let activations = memory.activation_fingerprint(&context_fingerprint, &output.trace);
            return Ok(PreparedForward {
                output,
                mask: route.mask.clone(),
                context_fingerprint,
                activation_fingerprint: activations,
                route_id: Some(route.route_id),
                quality: self_quality.max(0.30),
                fallback: false,
                recalled_memory_tokens: route.memory_tokens.len(),
            });
        }
        return full_fallback(
            model,
            memory,
            &prompt,
            context_fingerprint,
            Some(route.clone()),
        );
    }

    memory.note_revalidation();
    model.clear_kv_cache();
    let full_mask = LayerExecutionMask::all(model.layer_count());
    let full = model.forward_with_mask(&prompt, 0, Some(&full_mask), true, false)?;
    let activation_fingerprint = memory.activation_fingerprint(&context_fingerprint, &full.trace);
    let candidates = memory.progressive_candidate_masks(&full.trace);
    if candidates.is_empty() {
        return Ok(PreparedForward {
            output: full,
            mask: full_mask,
            context_fingerprint,
            activation_fingerprint,
            route_id: recalled.as_ref().map(|route| route.route_id),
            quality: 1.0,
            fallback: false,
            recalled_memory_tokens: recalled
                .as_ref()
                .map(|route| route.memory_tokens.len())
                .unwrap_or(0),
        });
    }

    let full_logits = full.logits.squeeze(0)?.to_vec1::<f32>()?;
    let mut best = None::<(LayerExecutionMask, f32, Gemma2ForwardOutput)>;
    let mut last_quality = 0.0;
    let mut cache_matches_best = false;
    for candidate in candidates
        .into_iter()
        .take(memory.config.max_candidate_prefills)
    {
        model.clear_kv_cache();
        let sparse = model.forward_with_mask(&prompt, 0, Some(&candidate), false, false)?;
        let sparse_logits = sparse.logits.squeeze(0)?.to_vec1::<f32>()?;
        last_quality = logit_agreement(&full_logits, &sparse_logits);
        if last_quality < memory.config.min_verified_quality {
            cache_matches_best = false;
            break;
        }
        best = Some((candidate, last_quality, sparse));
        cache_matches_best = true;
    }
    if let Some((candidate, quality, sparse)) = best {
        let output = if cache_matches_best {
            sparse
        } else {
            model.clear_kv_cache();
            model.forward_with_mask(&prompt, 0, Some(&candidate), false, false)?
        };
        return Ok(PreparedForward {
            output,
            mask: candidate,
            context_fingerprint,
            activation_fingerprint,
            route_id: recalled.as_ref().map(|route| route.route_id),
            quality,
            fallback: false,
            recalled_memory_tokens: recalled
                .as_ref()
                .map(|route| route.memory_tokens.len())
                .unwrap_or(0),
        });
    }
    model.clear_kv_cache();
    let output = model.forward_with_mask(&prompt, 0, Some(&full_mask), true, false)?;
    Ok(PreparedForward {
        output,
        mask: full_mask,
        context_fingerprint,
        activation_fingerprint,
        route_id: recalled.as_ref().map(|route| route.route_id),
        quality: last_quality,
        fallback: true,
        recalled_memory_tokens: recalled
            .as_ref()
            .map(|route| route.memory_tokens.len())
            .unwrap_or(0),
    })
}

fn full_fallback(
    model: &mut QuantizedGemma2,
    memory: &AdaptiveThermoMemory,
    prompt: &Tensor,
    context_fingerprint: ActivationFingerprint,
    route: Option<RecalledLayerRoute>,
) -> Result<PreparedForward, Box<dyn std::error::Error>> {
    model.clear_kv_cache();
    let mask = LayerExecutionMask::all(model.layer_count());
    let output = model.forward_with_mask(prompt, 0, Some(&mask), true, false)?;
    let activations = memory.activation_fingerprint(&context_fingerprint, &output.trace);
    Ok(PreparedForward {
        output,
        mask,
        context_fingerprint,
        activation_fingerprint: activations,
        route_id: route.as_ref().map(|route| route.route_id),
        quality: 0.0,
        fallback: true,
        recalled_memory_tokens: route
            .as_ref()
            .map(|route| route.memory_tokens.len())
            .unwrap_or(0),
    })
}

fn output_confidence(logits: &[f32]) -> f32 {
    TransformerActivationAdapter::new(8)
        .capture(logits)
        .confidence
}

fn logit_agreement(full: &[f32], sparse: &[f32]) -> f32 {
    if full.len() != sparse.len() || full.is_empty() {
        return 0.0;
    }
    let full_top = full
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    let sparse_top = sparse
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    if full_top != sparse_top {
        return 0.0;
    }
    let (squared_error, squared_signal) =
        full.iter()
            .zip(sparse)
            .fold((0.0f64, 0.0f64), |(error, signal), (left, right)| {
                (
                    error + (*left as f64 - *right as f64).powi(2),
                    signal + (*left as f64).powi(2),
                )
            });
    let relative_rmse = (squared_error / squared_signal.max(f64::EPSILON)).sqrt();
    (-4.0 * relative_rmse).exp() as f32
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let chat_name = env::var("GEMMA2_CIRCADIAN_CHAT")
        .ok()
        .map(|value| sanitize_chat_name(&value))
        .transpose()?;

    let mut config = Config {
        model: None,
        chat_name,
        circadian_root: env::var("GEMMA2_CIRCADIAN_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_CIRCADIAN_ROOT)),
        max_tokens: DEFAULT_MAX_TOKENS,
        context: DEFAULT_CONTEXT,
        temperature: 0.8,
        top_p: 0.95,
        prompt: None,
        sleep_only: false,
        device: env::var("GEMMA2_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
        thermo: true,
    };

    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--chat" => {
                config.chat_name = Some(sanitize_chat_name(&required(&mut args, "--chat")?)?);
            }
            "--circadian-root" => {
                config.circadian_root =
                    PathBuf::from(required(&mut args, "--circadian-root")?);
            }
            "--model" => config.model = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--max-tokens" => config.max_tokens = required(&mut args, "--max-tokens")?.parse()?,
            "--context" => config.context = required(&mut args, "--context")?.parse()?,
            "--temperature" => config.temperature = required(&mut args, "--temperature")?.parse()?,
            "--top-p" => config.top_p = required(&mut args, "--top-p")?.parse()?,
            "--device" => config.device = required(&mut args, "--device")?,
            "--prompt" => config.prompt = Some(required(&mut args, "--prompt")?),
            "--sleep-only" => config.sleep_only = true,
            "--no-thermo" => config.thermo = false,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other if !other.starts_with('-') => config.prompt = Some(other.to_string()),
            flag => return Err(format!("argumento desconocido: {flag}").into()),
        }
    }
    Ok(config)
}

fn required(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("falta valor para {name}").into())
}
