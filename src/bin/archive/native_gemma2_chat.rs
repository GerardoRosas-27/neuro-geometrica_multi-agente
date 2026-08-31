//! Chat de terminal con Gemma 2 2B ejecutado localmente por Candle/Rust.
//! No usa el proceso, API ni servidor de Ollama.

use candle_core::quantized::gguf_file;
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
const DEFAULT_TEMPERATURE: f64 = 0.8;
const DEFAULT_TOP_P: f64 = 0.95;

#[derive(Clone, Debug)]
struct ChatConfig {
    model: Option<PathBuf>,
    max_tokens: usize,
    context: usize,
    temperature: f64,
    top_p: f64,
    prompt: Option<String>,
    device: String,
    thermo: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
    println!("Gemma 2 2B nativo en Rust");
    println!("GGUF: {}", model_path.display());
    println!(
        "Motor: Candle {}; Ollama API/servidor: no usado",
        config.device
    );
    println!("Cargando pesos cuantizados...");

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
        "Listo en {:.1}s; contexto={} tokens; salida máxima={} tokens",
        started.elapsed().as_secs_f64(),
        context_limit,
        config.max_tokens
    );

    if let Some(prompt) = config.prompt.as_deref() {
        let _ = answer(
            &mut model,
            &tokenizer,
            &mut session,
            thermo.as_ref(),
            &[],
            prompt,
            &config,
            context_limit,
        )?;
        println!();
        return Ok(());
    }

    println!("Comandos: /limpiar borra el historial; /salir termina.");
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
            println!("Historial borrado.");
            continue;
        }
        let response = answer(
            &mut model,
            &tokenizer,
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
    Ok(())
}

// El helper reúne explícitamente los estados persistentes del REPL y los
// parámetros de una interacción; agrupar referencias aquí no reduce ownership.
#[allow(clippy::too_many_arguments)]
fn answer(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    session: &mut Gemma2Session,
    thermo: Option<&GemmaPhasorWorker>,
    history: &[(String, String)],
    input: &str,
    config: &ChatConfig,
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
    print!("\nGemma> ");
    io::stdout().flush()?;
    let generation = session.generate_observed(
        model,
        tokenizer,
        &prompt_tokens,
        None,
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
        "\n[TTFT={:.3}s prefill={:.2} tok/s decode={:.2} tok/s tokens={} KV_reused={}]",
        generation.metrics.time_to_first_token_seconds,
        generation.metrics.prefill_tokens_per_second(),
        generation.metrics.decode_tokens_per_second(),
        generation.metrics.generated_tokens,
        generation.metrics.cache_reused,
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
    Ok(generation.text)
}

fn parse_args() -> Result<ChatConfig, Box<dyn std::error::Error>> {
    let mut config = ChatConfig {
        model: None,
        max_tokens: DEFAULT_MAX_TOKENS,
        context: DEFAULT_CONTEXT,
        temperature: DEFAULT_TEMPERATURE,
        top_p: DEFAULT_TOP_P,
        prompt: None,
        device: "cpu".to_string(),
        thermo: true,
    };
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => config.model = Some(PathBuf::from(required_value(&mut args, "--model")?)),
            "--max-tokens" => {
                config.max_tokens = required_value(&mut args, "--max-tokens")?.parse()?
            }
            "--context" => config.context = required_value(&mut args, "--context")?.parse()?,
            "--temperature" => {
                config.temperature = required_value(&mut args, "--temperature")?.parse()?
            }
            "--top-p" => config.top_p = required_value(&mut args, "--top-p")?.parse()?,
            "--prompt" => config.prompt = Some(required_value(&mut args, "--prompt")?),
            "--device" => config.device = required_value(&mut args, "--device")?,
            "--no-thermo" => config.thermo = false,
            "--help" | "-h" => {
                println!(
                    "Uso: native_gemma2_chat [--model GGUF] [--device cpu|cuda:N] \\
                     [--max-tokens N] [--context N] [--temperature N] [--top-p N] \\
                     [--prompt TEXTO] [--no-thermo]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("argumento desconocido: {argument}").into()),
        }
    }
    if config.max_tokens == 0 || config.context < 32 {
        return Err("--max-tokens debe ser > 0 y --context >= 32".into());
    }
    if !(0.0..=1.0).contains(&config.top_p) || config.top_p == 0.0 {
        return Err("--top-p debe estar en (0, 1]".into());
    }
    if config.temperature <= 0.0 {
        return Err("--temperature debe ser > 0".into());
    }
    Ok(config)
}

fn required_value(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("falta valor para {name}").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_model_blob_from_local_oci_manifest() {
        let path = resolve_gemma2_model_path(None);
        if std::path::Path::new("ollama-models").exists() {
            assert!(path.unwrap().is_file());
        }
    }
}
