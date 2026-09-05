//! Chat circadiano unificado: vigilia con Gemma adaptativo + sueño entrenando CTP.
//!
//! ```powershell
//! cargo run --release --bin native_gemma2_circadian_chat -- --chat dyamon
//! ```

use candle_core::quantized::gguf_file;
use cdt_rqm_epr::adaptive_gemma2::{
    AdaptiveGemma2Config, AdaptiveThermoMemory, PreparedAdaptiveForward,
};
use cdt_rqm_epr::agent_graph::{
    AgentGraph, AgentGraphConfig, AgentRole, ABSTAIN_REPLY, AGENT_GRAPH_FILE, DENSE_TALKER_ID,
};
use cdt_rqm_epr::gemma2_circadian_bridge::{
    load_or_create_hybrid, new_wake_record, persist_hybrid_session, run_sleep_phase, wake_history,
    CircadianPaths, CircadianSleepConfig, WakeJournal, DEFAULT_CIRCADIAN_ROOT,
};
use cdt_rqm_epr::gemma2_thermo_hybrid_llm::{
    apply_wake_bias_tensor, Gemma2ThermoHybridConfig, Gemma2ThermoHybridLlm, Gemma2WakeBiasReport,
};
use cdt_rqm_epr::gemma2_thermo_hybrid_session::sanitize_chat_name;
use cdt_rqm_epr::gemma_phasor_coupling::{GemmaPhasorCouplingConfig, GemmaPhasorWorker};
use cdt_rqm_epr::layer_route_cache::fingerprint_wake;
use cdt_rqm_epr::native_gemma2::{
    init_gemma2_rayon_threads, resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer,
    LayerExecutionMask, QuantizedGemma2,
};
use cdt_rqm_epr::native_gemma2_runtime::{
    chat_tokens_with_cache, Gemma2Generation, Gemma2GenerationConfig, Gemma2Session,
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
    bench_routes: bool,
    device: String,
    thermo: bool,
}

struct CircadianSession {
    adaptive: AdaptiveThermoMemory,
    graph: AgentGraph,
    graph_path: PathBuf,
    gemma_session: Gemma2Session,
    hybrid: Gemma2ThermoHybridLlm,
    journal: WakeJournal,
    paths: Option<CircadianPaths>,
    history: Vec<(String, String)>,
    turns: u64,
    last_recalled_memory_tokens: usize,
    last_ctp: Gemma2WakeBiasReport,
    last_speaker: AgentRole,
    last_verify_passed: bool,
    thermo_created_at: u64,
    phasor: Option<GemmaPhasorWorker>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rayon_threads = init_gemma2_rayon_threads();
    let config = parse_args()?;
    if config.bench_routes {
        eprintln!("T1.3 rayon_threads={rayon_threads}");
        // T0: sin min(32). El default del chat es 256; el protocolo V8 pide 64 tokens.
        let generated_tokens = if config.max_tokens == DEFAULT_MAX_TOKENS {
            64
        } else {
            config.max_tokens.max(8)
        };
        let report = cdt_rqm_epr::layer_route_benchmark::run_route_speed_benchmark(
            cdt_rqm_epr::layer_route_benchmark::RouteSpeedConfig {
                generated_tokens,
                device: config.device.clone(),
                ..cdt_rqm_epr::layer_route_benchmark::RouteSpeedConfig::default()
            },
        )?;
        print!("{}", report.to_csv());
        eprintln!("{}", report.summary());
        return Ok(());
    }
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

    let graph_path = paths
        .as_ref()
        .map(|paths| paths.root.join(AGENT_GRAPH_FILE))
        .unwrap_or_else(|| {
            PathBuf::from("data/native_gemma2_circadian/_ephemeral").join(AGENT_GRAPH_FILE)
        });
    let graph = AgentGraph::load_or_new(&graph_path, AgentGraphConfig::default());

    let mut session = CircadianSession {
        adaptive,
        graph,
        graph_path,
        gemma_session: Gemma2Session::new(),
        hybrid,
        journal,
        paths,
        history,
        turns,
        last_recalled_memory_tokens: 0,
        last_ctp: Gemma2WakeBiasReport::default(),
        last_speaker: AgentRole::DenseTalker,
        last_verify_passed: true,
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
    println!(
        "Vigilia: Router→hablante→Verifier (sin LLM en el router). Sueño (/sueño): núcleo CTP."
    );

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
    let wake_fingerprint = fingerprint_wake(&prompt_tokens);
    let decision = session.graph.plan_turn(
        &wake_fingerprint,
        input,
        &session.adaptive.lrc,
        model.layer_count(),
    );
    session.last_speaker = decision.speaker;
    let prefill_started = Instant::now();
    let mut prepared = session.adaptive.prepare_forward_with_forced_mask(
        model,
        &prompt_tokens,
        &cached_tokens,
        cached_mask.as_ref(),
        cached_logits.as_ref(),
        Some(&decision.mask),
    )?;
    session.last_recalled_memory_tokens = prepared.recalled_memory_tokens;
    let mut speaker_label = match decision.speaker {
        AgentRole::FastTalker => "FastTalker",
        AgentRole::DenseTalker => "DenseTalker",
        AgentRole::Compiler => "Compiler",
        AgentRole::Router => "Router",
        AgentRole::Verifier => "Verifier",
        AgentRole::Memory => "Memory",
    };
    let route_label = if decision.lrc_hit {
        decision
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
        "[vigilia speaker={} route={} layers={}/{} quality={:.3} fallback={} memory_tokens={} prefill={} cache={} {:.3}s]",
        speaker_label,
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

    print!("\nGemma> ");
    io::stdout().flush()?;
    let turn_started = Instant::now();
    let (mut generation, mut ctp_seconds) = stream_generation(
        model,
        tokenizer,
        session,
        &prompt_tokens,
        &cached_tokens,
        &prepared,
        config,
        context_limit,
    )?;
    println!();

    let mut verify =
        session
            .graph
            .verify_reply(decision.speaker_id, &generation.text, decision.compiler);
    let mut dense_fallback = false;
    let mut speaker_id = decision.speaker_id;
    if !verify.passed() && decision.speaker != AgentRole::DenseTalker {
        eprintln!("[verificador: fail; un reintento DenseTalker]");
        session.gemma_session.reset(model);
        let dense = LayerExecutionMask::all(model.layer_count());
        prepared = session.adaptive.prepare_forward_with_forced_mask(
            model,
            &prompt_tokens,
            &[],
            None,
            None,
            Some(&dense),
        )?;
        print!("Gemma> ");
        io::stdout().flush()?;
        let retried = stream_generation(
            model,
            tokenizer,
            session,
            &prompt_tokens,
            &[],
            &prepared,
            config,
            context_limit,
        )?;
        println!();
        generation = retried.0;
        ctp_seconds = retried.1;
        speaker_id = DENSE_TALKER_ID;
        session.last_speaker = AgentRole::DenseTalker;
        speaker_label = "DenseTalker";
        dense_fallback = true;
        verify = session
            .graph
            .verify_reply(speaker_id, &generation.text, decision.compiler);
    }

    let mut reply_text = generation.text.clone();
    let mut reply_tokens = generation.token_ids.clone();
    if !verify.passed() {
        eprintln!("[verificador: segundo fail; abstencion]");
        reply_text = ABSTAIN_REPLY.to_string();
        reply_tokens.clear();
        println!("{ABSTAIN_REPLY}");
    }
    session.last_verify_passed = verify.passed();
    let latency_ms = turn_started.elapsed().as_secs_f64() * 1_000.0;
    session.graph.observe_turn(
        speaker_id,
        verify.passed(),
        latency_ms as f32,
        dense_fallback,
    );
    let _ = session.graph.save(&session.graph_path);

    eprintln!(
        "[decode {:.2} tok/s layers={}/{} route={} speaker={} verify={} cache={}]",
        generation.metrics.decode_tokens_per_second(),
        prepared.mask.executed_count(),
        prepared.mask.layer_count(),
        if decision.lrc_hit { "hit" } else { "miss" },
        speaker_label,
        if verify.passed() { "pass" } else { "fail" },
        generation.metrics.cache_reused,
    );
    if ctp_seconds > 0.0 && session.last_ctp.mixed > 0 {
        eprintln!(
            "[ctp blend={:.2} phi={:.3} mixed={} {:.3}s]",
            session.last_ctp.blend, session.last_ctp.phi_norm, session.last_ctp.mixed, ctp_seconds
        );
    }

    session.adaptive.observe(
        prepared.context_fingerprint.clone(),
        prepared.activation_fingerprint.clone(),
        prepared.mask.clone(),
        &prompt_tokens,
        prepared.quality,
        prepared.route_id,
        prepared.fallback || dense_fallback,
    )?;
    session.adaptive.observe_layer_route_turn(
        &prompt_tokens,
        &prepared.mask,
        prepared.fallback || dense_fallback,
    );

    session.turns = session.turns.saturating_add(1);
    let record = new_wake_record(
        session.turns,
        input,
        &reply_text,
        &prompt_tokens,
        &reply_tokens,
        prepared.quality,
        prepared.mask.executed_count(),
    );
    session.journal.append(&record)?;
    session.history.push((input.to_string(), reply_text));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_generation(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    session: &mut CircadianSession,
    prompt_tokens: &[u32],
    cached_tokens: &[u32],
    prepared: &PreparedAdaptiveForward,
    config: &Config,
    context_limit: usize,
) -> Result<(Gemma2Generation, f64), Box<dyn std::error::Error>> {
    session.gemma_session.adopt_prefill(
        prompt_tokens,
        Some(&prepared.mask),
        prepared.output.logits.clone(),
    )?;
    let suffix = if prepared.cache_reused {
        &prompt_tokens[cached_tokens.len().min(prompt_tokens.len())..]
    } else {
        prompt_tokens
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
    let generation = session.gemma_session.generate_observed_with_logits(
        model,
        tokenizer,
        prompt_tokens,
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
    session
        .hybrid
        .observe_generated_tokens(model, &generation.token_ids)?;
    ctp_report.mixed = mixed.get();
    session.last_ctp = ctp_report;
    Ok((generation, ctp_seconds))
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
        session.graph.save(&session.graph_path)?;
        print_sleep_report(report);
        println!("Persistido: adaptive + thermo.cdt + dataset de sueño + grafo");
    } else {
        session.graph.save(&session.graph_path)?;
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
    println!(
        "  Grafo agentes:      gen={} speaker={:?} verifier={}",
        session.graph.generation(),
        session.last_speaker,
        if session.last_verify_passed {
            "pass"
        } else {
            "fail"
        }
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
  --bench-routes   CSV V8: capas, KL, tok/s sparse vs 26/26, LRC, fallback, Ollama
  /sueño, /sleep   Consolida adaptativo + exporta dataset + entrena CTP + guarda
  /limpiar         Borra historial visible (conserva memorias)
  /estado          Metricas de vigilia (LRC + grafo de 6 agentes)
  /salir           Sueño + guardado + salir

Turno: Router (sin LLM) elige FastTalker, DenseTalker o Compiler; Verifier
aplica reglas. Un fallback denso por turno; segundo fail se abstiene.

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
        bench_routes: false,
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
            "--bench-routes" => config.bench_routes = true,
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
