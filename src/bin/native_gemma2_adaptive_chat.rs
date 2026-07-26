//! Gemma 2 con enrutamiento adaptativo y memoria termodinámica de dos velocidades.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use cdt_rqm_epr::adaptive_gemma2::{
    AdaptiveGemma2Config, AdaptiveThermoMemory, RecalledLayerRoute,
};
use cdt_rqm_epr::native_gemma2::{
    resolve_gemma2_model_path, Gemma2ForwardOutput, Gemma2Tokenizer, LayerExecutionMask,
    QuantizedGemma2,
};
use cdt_rqm_epr::thermo_router::{ActivationFingerprint, TransformerActivationAdapter};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

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
        let report = memory.consolidate_sleep()?;
        println!(
            "sleep=true flushed={} pruned_routes={} pruned_relations={} remaining_routes={}",
            report.flushed, report.pruned_routes, report.pruned_relations, report.remaining_routes
        );
        return Ok(());
    }

    println!("Gemma 2 adaptativo: Candle + memoria termodinámica");
    println!("GGUF: {}", model_path.display());
    println!("Memoria: {}", config.state_root.display());
    let started = Instant::now();
    let device = Device::Cpu;
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
    let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
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
            &[],
            prompt,
            &config,
            context_limit,
        )?;
        println!();
        memory.consolidate_sleep()?;
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
            model.clear_kv_cache();
            println!("Historial y KV cache borrados.");
            continue;
        }
        if matches!(input.to_ascii_lowercase().as_str(), "/sueño" | "/sueno") {
            let report = memory.consolidate_sleep()?;
            println!(
                "Sueño: {} memorias consolidadas, {} rutas y {} relaciones podadas.",
                report.flushed, report.pruned_routes, report.pruned_relations
            );
            continue;
        }
        let response = answer(
            &mut model,
            &tokenizer,
            &mut memory,
            &history,
            input,
            &config,
            context_limit,
        )?;
        println!();
        history.push((input.to_string(), response));
    }
    let report = memory.consolidate_sleep()?;
    println!(
        "Memoria guardada: flushed={} routes={}",
        report.flushed, report.remaining_routes
    );
    Ok(())
}

fn answer(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    memory: &mut AdaptiveThermoMemory,
    history: &[(String, String)],
    input: &str,
    config: &Config,
    context_limit: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt_limit = context_limit.saturating_sub(config.max_tokens).max(32);
    let prompt_tokens = chat_tokens(tokenizer, history, input, prompt_limit)?;
    let prepared = prepare_forward(model, memory, &prompt_tokens)?;
    eprintln!(
        "[route={} layers={}/{} quality={:.3} fallback={} memory_tokens={}]",
        prepared
            .route_id
            .map(|route| route.0.to_string())
            .unwrap_or_else(|| "new".to_string()),
        prepared.mask.executed_count(),
        prepared.mask.layer_count(),
        prepared.quality,
        prepared.fallback,
        prepared.recalled_memory_tokens
    );
    let rendered = generate(
        model,
        tokenizer,
        prepared.output.logits,
        &prepared.mask,
        prompt_tokens.len(),
        config,
        context_limit,
    )?;
    memory.observe(
        prepared.context_fingerprint,
        prepared.activation_fingerprint,
        prepared.mask,
        &prompt_tokens,
        prepared.quality,
        prepared.route_id,
        prepared.fallback,
    )?;
    Ok(rendered)
}

fn prepare_forward(
    model: &mut QuantizedGemma2,
    memory: &mut AdaptiveThermoMemory,
    prompt_tokens: &[u32],
) -> Result<PreparedForward, Box<dyn std::error::Error>> {
    let prompt = Tensor::new(prompt_tokens, &Device::Cpu)?.unsqueeze(0)?;
    let context_fingerprint = memory.context_fingerprint(prompt_tokens);
    let recalled = memory.recall(&context_fingerprint, model.layer_count());
    if let Some(route) = recalled.as_ref().filter(|_| !memory.should_revalidate()) {
        model.clear_kv_cache();
        let output = model.forward_with_mask(&prompt, 0, Some(&route.mask), true)?;
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
    let full = model.forward_with_mask(&prompt, 0, Some(&full_mask), true)?;
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
    let mut best = None::<(LayerExecutionMask, f32)>;
    let mut last_quality = 0.0;
    for candidate in candidates {
        model.clear_kv_cache();
        let sparse = model.forward_with_mask(&prompt, 0, Some(&candidate), false)?;
        let sparse_logits = sparse.logits.squeeze(0)?.to_vec1::<f32>()?;
        last_quality = logit_agreement(&full_logits, &sparse_logits);
        if last_quality < memory.config.min_verified_quality {
            break;
        }
        best = Some((candidate, last_quality));
    }
    if let Some((candidate, quality)) = best {
        model.clear_kv_cache();
        let output = model.forward_with_mask(&prompt, 0, Some(&candidate), true)?;
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
    let output = model.forward_with_mask(&prompt, 0, Some(&full_mask), true)?;
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
    let output = model.forward_with_mask(prompt, 0, Some(&mask), true)?;
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

fn generate(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    logits: Tensor,
    mask: &LayerExecutionMask,
    prompt_length: usize,
    config: &Config,
    context_limit: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    print!("\nGemma> ");
    io::stdout().flush()?;
    let mut logits = logits.squeeze(0)?;
    let mut sampler = LogitsProcessor::new(
        0x4745_4D4D_4132,
        Some(config.temperature),
        Some(config.top_p),
    );
    let mut generated = Vec::<u32>::new();
    let mut rendered = String::new();
    let started = Instant::now();
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
        if prompt_length + generated.len() >= context_limit {
            break;
        }
        let next = Tensor::new(&[token], &Device::Cpu)?.unsqueeze(0)?;
        logits = model
            .forward_with_mask(
                &next,
                prompt_length + generated.len() - 1,
                Some(mask),
                false,
            )?
            .logits
            .squeeze(0)?;
    }
    let speed = generated.len() as f64 / started.elapsed().as_secs_f64().max(f64::EPSILON);
    eprintln!("\n[generados={}, {:.2} tok/s]", generated.len(), speed);
    Ok(rendered.trim().to_string())
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
    Err(format!("el mensaje excede el límite de {limit} tokens").into())
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
            "--help" | "-h" => {
                println!(
                    "Uso: native_gemma2_adaptive_chat [--model GGUF] [--state-root DIR] \
                     [--prompt TEXTO] [--max-tokens N] [--context N] [--sleep-only]"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_logits_have_full_agreement() {
        assert!((logit_agreement(&[0.1, 0.9, -0.2], &[0.1, 0.9, -0.2]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn changed_top_token_forces_fallback() {
        assert_eq!(logit_agreement(&[0.1, 0.9], &[0.9, 0.1]), 0.0);
    }
}
