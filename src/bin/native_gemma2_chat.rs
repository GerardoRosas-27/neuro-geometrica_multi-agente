//! Chat de terminal con Gemma 2 2B ejecutado localmente por Candle/Rust.
//! No usa el proceso, API ni servidor de Ollama.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use cdt_rqm_epr::native_gemma2::{resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2};
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
struct ChatConfig {
    model: Option<PathBuf>,
    max_tokens: usize,
    context: usize,
    temperature: f64,
    top_p: f64,
    prompt: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
    println!("Gemma 2 2B nativo en Rust");
    println!("GGUF: {}", model_path.display());
    println!("Motor: Candle CPU; Ollama API/servidor: no usado");
    println!("Cargando pesos cuantizados...");

    let started = Instant::now();
    let device = Device::Cpu;
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
    let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
    let context_limit = config.context.min(model.max_context());
    println!(
        "Listo en {:.1}s; contexto={} tokens; salida máxima={} tokens",
        started.elapsed().as_secs_f64(),
        context_limit,
        config.max_tokens
    );

    if let Some(prompt) = config.prompt.as_deref() {
        let _ = answer(&mut model, &tokenizer, &[], prompt, &config, context_limit)?;
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
            model.clear_kv_cache();
            println!("Historial borrado.");
            continue;
        }
        let response = answer(
            &mut model,
            &tokenizer,
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

fn answer(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    history: &[(String, String)],
    input: &str,
    config: &ChatConfig,
    context_limit: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt_limit = context_limit.saturating_sub(config.max_tokens).max(32);
    let prompt_tokens = chat_tokens(tokenizer, history, input, prompt_limit)?;
    model.clear_kv_cache();
    let prompt = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    print!("\nGemma> ");
    io::stdout().flush()?;
    let prefill = Instant::now();
    let mut logits = model.forward(&prompt, 0)?.squeeze(0)?;
    let prefill_elapsed = prefill.elapsed();
    let mut sampler = LogitsProcessor::new(
        0x4745_4D4D_4132,
        Some(config.temperature),
        Some(config.top_p),
    );
    let mut generated = Vec::<u32>::new();
    let mut rendered = String::new();
    let decode_started = Instant::now();
    for _ in 0..config.max_tokens {
        let token = sampler.sample(&logits)?;
        if token == tokenizer.eos_id || Some(token) == tokenizer.end_of_turn_id {
            break;
        }
        generated.push(token);
        let decoded = tokenizer.decode(&generated, true)?;
        if let Some(delta) = decoded.strip_prefix(&rendered) {
            print!("{delta}");
            io::stdout().flush()?;
        }
        rendered = decoded;
        if prompt_tokens.len() + generated.len() >= context_limit {
            break;
        }
        let next = Tensor::new(&[token], &Device::Cpu)?.unsqueeze(0)?;
        logits = model
            .forward(&next, prompt_tokens.len() + generated.len() - 1)?
            .squeeze(0)?;
    }
    let decode_elapsed = decode_started.elapsed();
    let speed = generated.len() as f64 / decode_elapsed.as_secs_f64().max(f64::EPSILON);
    eprintln!(
        "\n[prefill={:.2}s, generados={}, {:.2} tok/s]",
        prefill_elapsed.as_secs_f64(),
        generated.len(),
        speed
    );
    Ok(rendered.trim().to_string())
}

fn chat_tokens(
    tokenizer: &Gemma2Tokenizer,
    history: &[(String, String)],
    input: &str,
    limit: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    for skip in 0..=history.len() {
        let mut prompt = String::new();
        for (user, assistant) in &history[skip..] {
            prompt.push_str("<start_of_turn>user\n");
            prompt.push_str(user);
            prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
            prompt.push_str(assistant);
            prompt.push_str("<end_of_turn>\n");
        }
        prompt.push_str("<start_of_turn>user\n");
        prompt.push_str(input);
        prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(&prompt)?);
        if tokens.len() <= limit {
            return Ok(tokens);
        }
    }
    Err(
        format!("el mensaje actual excede el límite de contexto disponible ({limit} tokens)")
            .into(),
    )
}

fn parse_args() -> Result<ChatConfig, Box<dyn std::error::Error>> {
    let mut config = ChatConfig {
        model: None,
        max_tokens: DEFAULT_MAX_TOKENS,
        context: DEFAULT_CONTEXT,
        temperature: DEFAULT_TEMPERATURE,
        top_p: DEFAULT_TOP_P,
        prompt: None,
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
            "--help" | "-h" => {
                println!(
                    "Uso: native_gemma2_chat [--model GGUF] [--max-tokens N] [--context N] \\
                     [--temperature N] [--top-p N] [--prompt TEXTO]"
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
