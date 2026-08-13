//! Consola interactiva: Gemma 2 (periférico) + motor termodinámico CTP (núcleo).
//!
//! ```text
//!   Tú> hola
//!   Thermo> [respuesta en streaming]
//!   [φ=... entropy=... sleep=... tok/s=...]
//! ```
//!
//! Comandos:
//!   /ayuda      — muestra ayuda
//!   /limpiar    — borra historial y contexto CTP
//!   /estado     — métricas del motor termodinámico
//!   /sleep      — consolidación CDT inmediata
//!   /salir      — termina

use candle_core::quantized::gguf_file;
use cdt_rqm_epr::gemma2_thermo_hybrid_llm::{Gemma2ThermoHybridConfig, Gemma2ThermoHybridLlm};
use cdt_rqm_epr::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2,
};
use cdt_rqm_epr::native_gemma2_runtime::chat_tokens;
use candle_transformers::generation::LogitsProcessor;
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_CONTEXT: usize = 2_048;
const DEFAULT_TEMPERATURE: f64 = 0.8;
const DEFAULT_TOP_P: f64 = 0.95;

#[derive(Clone, Debug)]
struct ConsoleConfig {
    model: Option<PathBuf>,
    max_tokens: usize,
    context: usize,
    temperature: f64,
    top_p: f64,
    seed: u64,
    prompt: Option<String>,
    device: String,
    thermo_window: usize,
    sleep_every: usize,
    show_metrics: bool,
}

struct ConsoleSession {
    hybrid: Gemma2ThermoHybridLlm,
    processor: LogitsProcessor,
    history: Vec<(String, String)>,
    turns: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    print_banner();

    let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
    println!("Modelo GGUF: {}", model_path.display());
    println!("Dispositivo periférico (Candle): {}", config.device);
    println!("Cargando Gemma 2...");

    let load_started = Instant::now();
    let device = resolve_gemma2_device(&config.device)?;
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
    let model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;

    let hybrid_config = Gemma2ThermoHybridConfig {
        thermo_window: config.thermo_window,
        sleep_every_tokens: config.sleep_every,
        seed: config.seed,
        ..Default::default()
    };
    let hybrid = Gemma2ThermoHybridLlm::for_gemma(&model, hybrid_config)?;
    let context_limit = config.context.min(model.max_context());

    println!("Listo en {:.1}s", load_started.elapsed().as_secs_f64());
    print_engine_info(&model, &hybrid, context_limit, &config);

    let mut session = ConsoleSession {
        hybrid,
        processor: LogitsProcessor::new(
            config.seed,
            Some(config.temperature),
            Some(config.top_p),
        ),
        history: Vec::new(),
        turns: 0,
    };

    if let Some(prompt) = config.prompt.as_deref() {
        let response = converse(
            &model,
            &tokenizer,
            &mut session,
            prompt,
            &config,
            context_limit,
        )?;
        if response.is_empty() {
            println!();
        }
        return Ok(());
    }

    print_help_short();
    repl(&model, &tokenizer, &mut session, &config, context_limit)
}

fn repl(
    model: &QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    session: &mut ConsoleSession,
    config: &ConsoleConfig,
    context_limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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
            "/salir" | "/exit" | "salir" | "exit" => break,
            "/ayuda" | "/help" => {
                print_help_full();
                continue;
            }
            "/limpiar" | "/clear" => {
                session.history.clear();
                session.hybrid.reset();
                session.turns = 0;
                println!("Historial y contexto termodinámico reiniciados.");
                continue;
            }
            "/estado" | "/status" => {
                print_status(session);
                continue;
            }
            "/sleep" => match session.hybrid.force_sleep() {
                Ok(report) => println!(
                    "Consolidación CDT: aceptados={} rechazados={} memoria={}",
                    report.accepted, report.rejected, report.memory_size
                ),
                Err(error) => println!("Sleep falló: {error}"),
            },
            cmd if cmd.starts_with("/ventana ") => {
                if let Some(value) = cmd.strip_prefix("/ventana ") {
                    match value.trim().parse::<usize>() {
                        Ok(window) if window >= 4 => {
                            session.hybrid.config_mut().thermo_window = window;
                            println!("Ventana CTP ajustada a {window} embeddings.");
                        }
                        _ => println!("Uso: /ventana N  (N >= 4)"),
                    }
                }
                continue;
            }
            _ if input.starts_with('/') => {
                println!("Comando desconocido. Escribe /ayuda para ver comandos.");
                continue;
            }
            _ => {}
        }

        match converse(model, tokenizer, session, input, config, context_limit) {
            Ok(response) => {
                session.history.push((input.to_string(), response));
                session.turns += 1;
            }
            Err(error) => eprintln!("Error: {error}"),
        }
    }

    println!("Hasta pronto.");
    Ok(())
}

fn converse(
    model: &QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    session: &mut ConsoleSession,
    user_input: &str,
    config: &ConsoleConfig,
    context_limit: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt_limit = context_limit.saturating_sub(config.max_tokens).max(32);
    let prompt_tokens = chat_tokens(tokenizer, &session.history, user_input, prompt_limit)?;

    let stop_tokens = stop_tokens(tokenizer);
    let mut decode = tokenizer.decode_stream(true);
    let mut response = String::new();

    print!("\nThermo> ");
    io::stdout().flush()?;

    let started = Instant::now();
    let ttft_started = Instant::now();
    let mut ttft_ms = None::<f64>;

    let (generated, report) = session.hybrid.generate_streaming(
        model,
        &prompt_tokens,
        config.max_tokens,
        &stop_tokens,
        &mut session.processor,
        |token, _report| {
            if stop_tokens.contains(&token) {
                return true;
            }
            if ttft_ms.is_none() {
                ttft_ms = Some(ttft_started.elapsed().as_secs_f64() * 1_000.0);
            }
            if let Ok(Some(fragment)) = decode.step(token) {
                print!("{fragment}");
                let _ = io::stdout().flush();
                response.push_str(&fragment);
            }
            false
        },
    )?;

    let elapsed = started.elapsed();
    if config.show_metrics {
        eprintln!(
            "\n[thermo: φ={:.3} H={:.3} sleep={} ctx={} tok/s={:.2} TTFT={:.0}ms]",
            report.mean_phi_norm,
            report.mean_softmax_entropy,
            report.sleep_cycles,
            session.hybrid.context_len(),
            generated.len() as f64 / elapsed.as_secs_f64().max(1e-6),
            ttft_ms.unwrap_or(0.0),
        );
    } else if !response.ends_with('\n') {
        println!();
    }

    Ok(response)
}

fn stop_tokens(tokenizer: &Gemma2Tokenizer) -> Vec<u32> {
    let mut stops = vec![tokenizer.eos_id];
    if let Some(end_of_turn) = tokenizer.end_of_turn_id {
        stops.push(end_of_turn);
    }
    stops
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GEMMA 2 + MOTOR TERMODINÁMICO CTP  —  Consola Híbrida       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Periférico: W_emb·√d  →  logits  →  Softmax (Candle/GPU)    ║");
    println!("║  Núcleo:     RFF fasorial + Softmax CTP + Langevin + CDT     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_engine_info(
    model: &QuantizedGemma2,
    hybrid: &Gemma2ThermoHybridLlm,
    context_limit: usize,
    config: &ConsoleConfig,
) {
    let thermo = hybrid.thermo_engine().config();
    println!(
        "d_model={} | softcap={} | contexto={} | ventana CTP={} | RFF={} | nodos CDT={}",
        model.embedding_length(),
        model.final_softcap(),
        context_limit,
        config.thermo_window,
        thermo.rff.features,
        thermo.cdt_nodes,
    );
    println!(
        "max_tokens={} | T={} | top_p={} | sleep/cada={} tokens",
        config.max_tokens,
        config.temperature,
        config.top_p,
        config.sleep_every,
    );
}

fn print_status(session: &ConsoleSession) {
    let phasor = session.hybrid.thermo_engine().hybrid_engine().phasor.report();
    let hybrid = session.hybrid.thermo_engine().hybrid_engine();
    println!("── Estado termodinámico ──");
    println!("  Turnos de chat:       {}", session.turns);
    println!("  Tokens procesados:    {}", session.hybrid.tokens_processed());
    println!("  Contexto (ventana):   {}", session.hybrid.context_len());
    println!("  Ciclos sleep:         {}", session.hybrid.sleep_cycles());
    println!("  Atractores CDT:       {}", hybrid.attractors().len());
    println!("  Pendientes:           {}", hybrid.pending_attractors().len());
    println!("  F (energía libre):    {:.4}", phasor.free_energy);
    println!("  Coherencia de fase:   {:.4}", phasor.phase_coherence);
    println!("  Residuo gradiente:    {:.3e}", phasor.gradient_residual);
    println!("  Mensajes en historial: {}", session.history.len());
}

fn print_help_short() {
    println!("Escribe tu mensaje y pulsa Enter. Comandos: /ayuda /limpiar /estado /sleep /salir");
}

fn print_help_full() {
    println!(
        r"
Comandos de consola:
  /ayuda, /help     Muestra esta ayuda
  /limpiar, /clear  Borra historial de chat y contexto CTP
  /estado, /status  Métricas del motor termodinámico y CDT
  /sleep            Fuerza consolidación CDT (fase sueño)
  /ventana N        Ajusta ventana deslizante del reservorio (N >= 4)
  /salir, /exit     Termina la aplicación

Variables de entorno:
  GEMMA2_DEVICE              cpu | cuda:0
  GEMMA2_THERMO_WINDOW       ventana CTP (default 64)
  GEMMA2_THERMO_SLEEP_EVERY  tokens entre sleep (default 32)
  GEMMA2_THERMO_SEED         semilla de muestreo

Argumentos:
  --model PATH         Ruta al GGUF de Gemma 2
  --device DEVICE      Dispositivo Candle
  --max-tokens N       Tokens máximos por respuesta
  --context N          Límite de contexto del prompt
  --temperature T      Temperatura de muestreo
  --top-p P            Nucleus sampling
  --thermo-window N    Ventana del motor CTP
  --sleep-every N      Consolidación CDT cada N tokens
  --no-metrics         Oculta métricas termodinámicas
  --prompt TEXTO       Una sola pregunta (sin REPL interactivo)
"
    );
}

fn parse_args() -> Result<ConsoleConfig, Box<dyn std::error::Error>> {
    let mut config = ConsoleConfig {
        model: None,
        max_tokens: DEFAULT_MAX_TOKENS,
        context: DEFAULT_CONTEXT,
        temperature: DEFAULT_TEMPERATURE,
        top_p: DEFAULT_TOP_P,
        seed: env_u64("GEMMA2_THERMO_SEED", 0x4745_4D4D_4154_4845),
        prompt: None,
        device: env::var("GEMMA2_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
        thermo_window: env_usize("GEMMA2_THERMO_WINDOW", 64),
        sleep_every: env_usize("GEMMA2_THERMO_SLEEP_EVERY", 32),
        show_metrics: true,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => config.model = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--max-tokens" => config.max_tokens = required(&mut args, "--max-tokens")?.parse()?,
            "--context" => config.context = required(&mut args, "--context")?.parse()?,
            "--temperature" => config.temperature = required(&mut args, "--temperature")?.parse()?,
            "--top-p" => config.top_p = required(&mut args, "--top-p")?.parse()?,
            "--device" => config.device = required(&mut args, "--device")?,
            "--thermo-window" => {
                config.thermo_window = required(&mut args, "--thermo-window")?.parse()?
            }
            "--sleep-every" => config.sleep_every = required(&mut args, "--sleep-every")?.parse()?,
            "--seed" => config.seed = required(&mut args, "--seed")?.parse()?,
            "--prompt" => config.prompt = Some(required(&mut args, "--prompt")?),
            "--no-metrics" => config.show_metrics = false,
            "--help" | "-h" => {
                print_help_full();
                std::process::exit(0);
            }
            other if !other.starts_with('-') => config.prompt = Some(other.to_string()),
            flag => return Err(format!("argumento desconocido: {flag}").into()),
        }
    }

    if config.max_tokens == 0 {
        return Err("--max-tokens debe ser > 0".into());
    }
    if config.context < 32 {
        return Err("--context debe ser >= 32".into());
    }
    if !(0.0..=1.0).contains(&config.top_p) || config.top_p == 0.0 {
        return Err("--top-p debe estar en (0, 1]".into());
    }
    if config.temperature <= 0.0 {
        return Err("--temperature debe ser > 0".into());
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

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
