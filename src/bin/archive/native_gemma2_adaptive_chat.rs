//! Gemma 2 con enrutamiento adaptativo y memoria termodinámica de dos velocidades.

use candle_core::quantized::gguf_file;
use cdt_rqm_epr::adaptive_gemma2::{AdaptiveGemma2Config, AdaptiveThermoMemory};
use cdt_rqm_epr::gemma_phasor_coupling::{GemmaPhasorCouplingConfig, GemmaPhasorWorker};
use cdt_rqm_epr::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2,
};
use cdt_rqm_epr::native_gemma2_runtime::{
    chat_tokens_with_cache, Gemma2GenerationConfig, Gemma2Session,
};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_CONTEXT: usize = 2_048;
const DEFAULT_STATE_ROOT: &str = "data/native_gemma2_adaptive";

#[derive(Clone, Debug)]
struct Config {
    model: Option<PathBuf>,
    state_root: PathBuf,
    max_tokens: usize,
    context: usize,
    temperature: f64,
    top_p: f64,
    prompt: Option<String>,
    sleep_only: bool,
    device: String,
    thermo: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
    let model_id = format!("gemma2:{}", model_path.display());
    let mut memory = AdaptiveThermoMemory::load_or_new(
        &config.state_root,
        model_id,
        AdaptiveGemma2Config::default(),
    )?;
    if config.sleep_only {
        let device = resolve_gemma2_device(&config.device)?;
        let mut file = File::open(&model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
        let report = memory.consolidate_sleep_with_model(&mut model, &[])?;
        println!(
            "sleep=true replayed={} discovered={} flushed={} working={} pruned_routes={} remaining_routes={}",
            report.replayed,
            report.discovered_masks,
            report.flushed,
            report.retained_working,
            report.pruned_routes,
            report.remaining_routes
        );
        return Ok(());
    }

    println!("Gemma 2 adaptativo: Candle + memoria termodinámica");
    println!("GGUF: {}", model_path.display());
    println!("Memoria: {}", config.state_root.display());
    let started = Instant::now();
    let device = resolve_gemma2_device(&config.device)?;
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
    let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
    let mut session = Gemma2Session::new();
    let thermo = config
        .thermo
        .then(|| GemmaPhasorWorker::start(GemmaPhasorCouplingConfig::default()))
        .transpose()?;
    let context_limit = config.context.min(model.max_context());
    println!(
        "Listo en {:.1}s; capas={} contexto={}",
        started.elapsed().as_secs_f64(),
        model.layer_count(),
        context_limit
    );

    if let Some(prompt) = config.prompt.as_deref() {
        answer(
            &mut model,
            &tokenizer,
            &mut memory,
            &mut session,
            thermo.as_ref(),
            &[],
            prompt,
            &config,
            context_limit,
        )?;
        println!();
        memory.consolidate_sleep_with_model(&mut model, &[])?;
        return Ok(());
    }

    println!("Comandos: /limpiar, /sueño, /salir.");
    let mut history = Vec::<(String, String)>::new();
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
        if matches!(
            input.to_ascii_lowercase().as_str(),
            "/salir" | "salir" | "exit"
        ) {
            break;
        }
        if input.eq_ignore_ascii_case("/limpiar") {
            history.clear();
            session.reset(&mut model);
            println!("Historial y KV cache borrados.");
            continue;
        }
        if matches!(input.to_ascii_lowercase().as_str(), "/sueño" | "/sueno") {
            let report = memory.consolidate_sleep_with_model(&mut model, &[])?;
            println!(
                "Sueño: replay={} máscaras={} flushed={} working={} rutas podadas={}.",
                report.replayed,
                report.discovered_masks,
                report.flushed,
                report.retained_working,
                report.pruned_routes
            );
            continue;
        }
        let response = answer(
            &mut model,
            &tokenizer,
            &mut memory,
            &mut session,
            thermo.as_ref(),
            &history,
            input,
            &config,
            context_limit,
        )?;
        println!();
        history.push((input.to_string(), response));
    }
    let report = memory.consolidate_sleep_with_model(&mut model, &[])?;
    println!(
        "Memoria guardada: replay={} máscaras={} flushed={} routes={}",
        report.replayed, report.discovered_masks, report.flushed, report.remaining_routes
    );
    Ok(())
}

// La interacción adaptativa coordina cuatro estados con ciclos de vida
// distintos; mantenerlos visibles evita ocultar resets transaccionales.
#[allow(clippy::too_many_arguments)]
fn answer(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    memory: &mut AdaptiveThermoMemory,
    session: &mut Gemma2Session,
    thermo: Option<&GemmaPhasorWorker>,
    history: &[(String, String)],
    input: &str,
    config: &Config,
    context_limit: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt_limit = context_limit.saturating_sub(config.max_tokens).max(32);
    let prompt_tokens = chat_tokens_with_cache(
        tokenizer,
        history,
        input,
        prompt_limit,
        session.cached_tokens(),
    )?;
    let prepared = memory.prepare_forward(
        model,
        &prompt_tokens,
        session.cached_tokens(),
        session.active_mask(),
        session.last_logits(),
    )?;
    eprintln!(
        "[route={} layers={}/{} quality={:.3} fallback={} memory_tokens={} prefill={} cache={}]",
        prepared
            .route_id
            .map(|route| route.0.to_string())
            .unwrap_or_else(|| {
                if prepared.recalled_memory_tokens > 0 {
                    "working".to_string()
                } else {
                    "new".to_string()
                }
            }),
        prepared.mask.executed_count(),
        prepared.mask.layer_count(),
        prepared.quality,
        prepared.fallback,
        prepared.recalled_memory_tokens,
        prepared.prefill_tokens,
        prepared.cache_reused,
    );
    session.adopt_prefill(&prompt_tokens, Some(&prepared.mask), prepared.output.logits)?;
    print!("\nGemma> ");
    io::stdout().flush()?;
    let generation = session.generate_observed(
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
            if let Some(worker) = thermo {
                worker.observe_token(token, position);
            }
        },
        |_| false,
    )?;
    eprintln!(
        "\n[TTFT={:.3}s decode={:.2} tok/s tokens={}]",
        generation.metrics.time_to_first_token_seconds,
        generation.metrics.decode_tokens_per_second(),
        generation.metrics.generated_tokens,
    );
    if let Some(report) = thermo.and_then(|worker| worker.snapshot(Duration::from_millis(100))) {
        eprintln!(
            "[fasores concurrentes: tokens={} steps={} F={:.4} coherencia={:.4}]",
            report.observed_tokens,
            report.phasor_steps,
            report.state.free_energy,
            report.state.phase_coherence,
        );
    }
    memory.observe(
        prepared.context_fingerprint,
        prepared.activation_fingerprint,
        prepared.mask,
        &prompt_tokens,
        prepared.quality,
        prepared.route_id,
        prepared.fallback,
    )?;
    Ok(generation.text)
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config {
        model: None,
        state_root: PathBuf::from(DEFAULT_STATE_ROOT),
        max_tokens: DEFAULT_MAX_TOKENS,
        context: DEFAULT_CONTEXT,
        temperature: 0.8,
        top_p: 0.95,
        prompt: None,
        sleep_only: false,
        device: "cpu".to_string(),
        thermo: true,
    };
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => config.model = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--state-root" => {
                config.state_root = PathBuf::from(required(&mut args, "--state-root")?)
            }
            "--max-tokens" => config.max_tokens = required(&mut args, "--max-tokens")?.parse()?,
            "--context" => config.context = required(&mut args, "--context")?.parse()?,
            "--temperature" => {
                config.temperature = required(&mut args, "--temperature")?.parse()?
            }
            "--top-p" => config.top_p = required(&mut args, "--top-p")?.parse()?,
            "--prompt" => config.prompt = Some(required(&mut args, "--prompt")?),
            "--sleep-only" => config.sleep_only = true,
            "--device" => config.device = required(&mut args, "--device")?,
            "--no-thermo" => config.thermo = false,
            "--help" | "-h" => {
                println!(
                    "Uso: native_gemma2_adaptive_chat [--model GGUF] [--state-root DIR] \
                     [--prompt TEXTO] [--max-tokens N] [--context N] [--device cpu|cuda:N] \\
                     [--sleep-only] [--no-thermo]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("argumento desconocido: {argument}").into()),
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
