//! Chat circadiano unificado: vigilia con Gemma adaptativo + sueño entrenando CTP.
//!
//! ```powershell
//! cargo run --release --bin native_gemma2_circadian_chat -- --chat dyamon
//! ```

use candle_core::quantized::gguf_file;
use cdt_rqm_epr::adaptive_gemma2::{AdaptiveGemma2Config, AdaptiveThermoMemory};
use cdt_rqm_epr::gemma2_circadian_bridge::{
    load_or_create_hybrid, new_wake_record, persist_hybrid_session, run_sleep_phase, wake_history,
    CircadianPaths, CircadianSleepConfig, WakeJournal, DEFAULT_CIRCADIAN_ROOT,
};
use cdt_rqm_epr::gemma2_thermo_hybrid_llm::{
    apply_wake_bias_tensor, Gemma2ThermoHybridConfig, Gemma2ThermoHybridLlm, Gemma2WakeBiasReport,
};
use cdt_rqm_epr::gemma2_thermo_hybrid_session::sanitize_chat_name;
use cdt_rqm_epr::gemma_phasor_coupling::{GemmaPhasorCouplingConfig, GemmaPhasorWorker};
use cdt_rqm_epr::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2,
};
use cdt_rqm_epr::native_gemma2_runtime::{
    chat_tokens_with_cache, Gemma2GenerationConfig, Gemma2Session,
};
use std::cell::Cell;
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

struct CircadianSession {
    adaptive: AdaptiveThermoMemory,
    gemma_session: Gemma2Session,
    hybrid: Gemma2ThermoHybridLlm,
    journal: WakeJournal,
    paths: Option<CircadianPaths>,
    history: Vec<(String, String)>,
    turns: u64,
    last_recalled_memory_tokens: usize,
    last_ctp: Gemma2WakeBiasReport,
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
            let journal = WakeJournal::open(paths.wake_journal.clone())?;
            let (journal_turns, journal_history) = wake_history(&journal)?;
            let mut loaded = load_or_create_hybrid(
                &model,
                paths,
                hybrid_config,
                &journal_history,
                journal_turns,
                cdt_rqm_epr::gemma2_thermo_hybrid_session::unix_now(),
            )?;
            let report = run_sleep_phase(
                &mut model,
                &mut loaded.hybrid,
                &mut adaptive,
                &journal,
                paths,
                &CircadianSleepConfig::default(),
            )?;
            persist_hybrid_session(
                paths,
                loaded.created_at_unix,
                loaded.turns,
                &loaded.history,
                &mut loaded.hybrid,
            )?;
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

    let journal = if let Some(paths) = paths.as_ref() {
        WakeJournal::open(paths.wake_journal.clone())?
    } else {
        WakeJournal::open(PathBuf::from(
            "data/native_gemma2_circadian/_ephemeral/journal.jsonl",
        ))?
    };
    let (journal_turns, journal_history) = wake_history(&journal)?;

    let hybrid_config = Gemma2ThermoHybridConfig::default();
    let (hybrid, thermo_created_at, turns, history, resumed) = if let Some(paths) = paths.as_ref() {
        let loaded = load_or_create_hybrid(
            &model,
            paths,
            hybrid_config,
            &journal_history,
            journal_turns,
            cdt_rqm_epr::gemma2_thermo_hybrid_session::unix_now(),
        )?;
        (
            loaded.hybrid,
            loaded.created_at_unix,
            loaded.turns,
            loaded.history,
            loaded.resumed,
        )
    } else {
        (
            Gemma2ThermoHybridLlm::for_gemma(&model, hybrid_config)?,
            cdt_rqm_epr::gemma2_thermo_hybrid_session::unix_now(),
            journal_turns,
            journal_history,
            false,
        )
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
        history,
        turns,
        last_recalled_memory_tokens: 0,
        last_ctp: Gemma2WakeBiasReport::default(),
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
    if resumed || session.turns > 0 || !session.history.is_empty() {
        println!(
            "Memoria cargada: {} turnos, {} mensajes, {} rutas adaptativas",
            session.turns,
            session.history.len(),
            session.adaptive.router.registry.routes().len(),
        );
    }
    println!("Vigilia: Gemma adaptativo responde. Sueño (/sueño): entrena núcleo CTP.");

    if let Some(prompt) = config.prompt.as_deref() {
        wake_turn(
            &mut model,
            &tokenizer,
            &mut session,
            prompt,
            &config,
            context_limit,
        )?;
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
                session.last_recalled_memory_tokens = 0;
                session.hybrid.reset_context();
                session.last_ctp = Gemma2WakeBiasReport::default();
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
    let prompt_tokens = chat_tokens_with_cache(
        tokenizer,
        &session.history,
        input,
        prompt_limit,
        session.gemma_session.cached_tokens(),
    )?;
    let cached_tokens = session.gemma_session.cached_tokens().to_vec();
    let cached_mask = session.gemma_session.active_mask().cloned();
    let cached_logits = session.gemma_session.last_logits().cloned();
    let prefill_started = Instant::now();
    let prepared = session.adaptive.prepare_forward(
        model,
        &prompt_tokens,
        &cached_tokens,
        cached_mask.as_ref(),
        cached_logits.as_ref(),
    )?;
    session.last_recalled_memory_tokens = prepared.recalled_memory_tokens;
    let route_label = if prepared.layer_route_hit {
        prepared
            .layer_route_id
            .map(|id| format!("lrc:{id}"))
            .unwrap_or_else(|| "lrc".to_string())
    } else if prepared.route_id.is_some() {
        prepared
            .route_id
            .map(|route| route.0.to_string())
            .unwrap_or_default()
    } else if prepared.recalled_memory_tokens > 0 {
        "working".to_string()
    } else {
        "miss".to_string()
    };
    eprintln!(
        "[vigilia route={} layers={}/{} quality={:.3} fallback={} memory_tokens={} prefill={} cache={} {:.3}s]",
        route_label,
        prepared.mask.executed_count(),
        prepared.mask.layer_count(),
        prepared.quality,
        prepared.fallback,
        prepared.recalled_memory_tokens,
        prepared.prefill_tokens,
        prepared.cache_reused,
        prefill_started.elapsed().as_secs_f64(),
    );

    session.gemma_session.adopt_prefill(
        &prompt_tokens,
        Some(&prepared.mask),
        prepared.output.logits,
    )?;

    let suffix = if prepared.cache_reused {
        &prompt_tokens[cached_tokens.len().min(prompt_tokens.len())..]
    } else {
        prompt_tokens.as_slice()
    };
    let ctp_started = Instant::now();
    let observed = session
        .hybrid
        .observe_prompt_tokens(model, suffix, prepared.cache_reused)?;
    let blend = session.hybrid.config().wake_blend;
    let (ctp_bias, mut ctp_report) = if blend > 0.0 && session.hybrid.context_len() > 0 {
        let (bias, report) = session.hybrid.compute_wake_bias(model)?;
        (Some(bias), report)
    } else {
        (None, Gemma2WakeBiasReport::default())
    };
    ctp_report.observed_tokens = observed;
    ctp_report.blend = blend;
    let ctp_seconds = ctp_started.elapsed().as_secs_f64();
    let mixed = Cell::new(0usize);

    print!("\nGemma> ");
    io::stdout().flush()?;

    let generation = session.gemma_session.generate_observed_with_logits(
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
        |logits, step| {
            let Some(ctp_bias) = ctp_bias.as_ref() else {
                return Ok(());
            };
            let changed = apply_wake_bias_tensor(logits, ctp_bias, blend)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if step == 0 {
                mixed.set(changed);
            }
            Ok(())
        },
        |_| false,
    )?;
    println!();
    session
        .hybrid
        .observe_generated_tokens(model, &generation.token_ids)?;
    ctp_report.mixed = mixed.get();
    session.last_ctp = ctp_report;
    if session.last_ctp.mixed > 0 {
        eprintln!(
            "[ctp blend={:.2} phi={:.3} bias={:.3} mixed={} ctx={} {:.3}s]",
            session.last_ctp.blend,
            session.last_ctp.phi_norm,
            session.last_ctp.ctp_bias_norm,
            session.last_ctp.mixed,
            session.last_ctp.context_length,
            ctp_seconds,
        );
    }
    eprintln!(
        "[decode {:.2} tok/s layers={}/{} route={} cache={}]",
        generation.metrics.decode_tokens_per_second(),
        prepared.mask.executed_count(),
        prepared.mask.layer_count(),
        if prepared.layer_route_hit {
            "hit"
        } else {
            "miss"
        },
        generation.metrics.cache_reused,
    );

    session.adaptive.observe(
        prepared.context_fingerprint,
        prepared.activation_fingerprint,
        prepared.mask.clone(),
        &prompt_tokens,
        prepared.quality,
        prepared.route_id,
        prepared.fallback,
    )?;
    session
        .adaptive
        .observe_layer_route_turn(&prompt_tokens, &prepared.mask, prepared.fallback);

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
        let report = session.adaptive.consolidate_sleep_with_model(model, &[])?;
        println!(
            "Sueño adaptativo: replay={} máscaras={} flushed={} working={} routes={} lrc={} kl={:.3} top1={:.2}",
            report.replayed,
            report.discovered_masks,
            report.flushed,
            report.retained_working,
            report.remaining_routes,
            report.lrc_promoted,
            report.sleep_mean_kl,
            report.sleep_top1_agree
        );
    }
    Ok(())
}

fn print_sleep_report(report: cdt_rqm_epr::gemma2_circadian_bridge::CircadianSleepReport) {
    println!("── Sueño ──");
    println!(
        "  Adaptativo: replay={} máscaras={} flushed={} working={} routes={} lrc={} kl={:.3} top1={:.2} relaciones_podadas={}",
        report.adaptive.replayed,
        report.adaptive.discovered_masks,
        report.adaptive.flushed,
        report.adaptive.retained_working,
        report.adaptive.remaining_routes,
        report.adaptive.lrc_promoted,
        report.adaptive.sleep_mean_kl,
        report.adaptive.sleep_top1_agree,
        report.adaptive.pruned_relations
    );
    println!(
        "  Dataset CTP: {} secuencias exportadas",
        report.dataset_entries
    );
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
    println!(
        "  Tokens recall:      {}",
        session.last_recalled_memory_tokens
    );
    println!("  Atractores CTP:     {}", attractors);
    println!(
        "  CTP vigilia:        blend={:.2} phi={:.3} mixed={}",
        session.last_ctp.blend, session.last_ctp.phi_norm, session.last_ctp.mixed
    );
    println!(
        "  Rutas adaptativas:  {}",
        session.adaptive.router.registry.routes().len()
    );
    println!(
        "  Rutas LRC:          {}",
        session.adaptive.layer_route_count()
    );
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CHAT CIRCADIANO — demo de ingeniería, no claim del preprint ║");
    println!("║  Idioma forzado: español. Identidad Dyamon fuera del historial║");
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
                config.circadian_root = PathBuf::from(required(&mut args, "--circadian-root")?);
            }
            "--model" => config.model = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--max-tokens" => config.max_tokens = required(&mut args, "--max-tokens")?.parse()?,
            "--context" => config.context = required(&mut args, "--context")?.parse()?,
            "--temperature" => {
                config.temperature = required(&mut args, "--temperature")?.parse()?
            }
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
