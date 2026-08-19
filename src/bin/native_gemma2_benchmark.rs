//! Benchmark reproducible del runtime Gemma 2 nativo.
//!
//! Mide carga, RSS, TTFT, throughput de prefill/decode y reutilización KV.

use candle_core::quantized::gguf_file;
use cdt_rqm_epr::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2,
};
use cdt_rqm_epr::native_gemma2_runtime::{
    Gemma2GenerationConfig, Gemma2GenerationMetrics, Gemma2Session,
};
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;
use sysinfo::{get_current_pid, ProcessesToUpdate, System};

const DEFAULT_GENERATED_TOKENS: usize = 64;
const DEFAULT_REPETITIONS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_argument = arg_value("--model").map(PathBuf::from);
    let generated_tokens = arg_usize("--generated", DEFAULT_GENERATED_TOKENS).max(1);
    let repetitions = arg_usize("--repetitions", DEFAULT_REPETITIONS).max(1);
    let device_name = arg_value("--device").unwrap_or_else(|| "cpu".to_string());
    let prompt_lengths = arg_value("--prompt-lengths")
        .map(|value| parse_lengths(&value))
        .transpose()?
        .unwrap_or_else(|| vec![32, 256, 1_024, 2_048]);
    let model_path = resolve_gemma2_model_path(model_argument.as_deref())?;

    let rss_before = current_rss_bytes();
    let load_started = Instant::now();
    let device = resolve_gemma2_device(&device_name)?;
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
    let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let rss_after = current_rss_bytes();
    println!(
        "kind,model,device,load_s,rss_before_mib,rss_after_mib,rss_delta_mib,embedding_mib,\
         embedding_if_f32_mib"
    );
    println!(
        "load,{},{device_name},{load_seconds:.6},{:.3},{:.3},{:.3},{:.3},{:.3}",
        model_path.display(),
        mib(rss_before),
        mib(rss_after),
        mib(rss_after.saturating_sub(rss_before)),
        mib(model.embedding_storage_bytes() as u64),
        mib(model.embedding_logical_f32_bytes() as u64),
    );
    println!(
        "kind,prompt_tokens,generated_tokens,median_ttft_ms,median_prefill_tok_s,\
         median_decode_tok_s,median_prefill_ms,model_decode_ms,logits_ms,text_decode_ms,kv_reused"
    );

    for prompt_length in prompt_lengths {
        if prompt_length + generated_tokens > model.max_context() {
            eprintln!(
                "omitido prompt={prompt_length}: excede contexto {} con generated={generated_tokens}",
                model.max_context()
            );
            continue;
        }
        let prompt = synthetic_prompt(&tokenizer, prompt_length)?;
        let context_limit = model.max_context();
        let mut reports = Vec::with_capacity(repetitions);
        let mut reuse_reports = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            let mut session = Gemma2Session::new();
            session.reset(&mut model);
            let generation = session.generate(
                &mut model,
                &tokenizer,
                &prompt,
                None,
                Gemma2GenerationConfig {
                    max_tokens: generated_tokens,
                    context_limit,
                    temperature: 0.01,
                    top_p: 1.0,
                    seed: 0x4745_4D4D_4132,
                },
                |_| {},
            )?;
            reports.push(generation.metrics);
            let mut extension = session.cached_tokens().to_vec();
            if extension.len() + 9 < context_limit {
                extension.extend(tokenizer.encode(" Continúa con una conclusión breve.")?);
                extension.truncate(context_limit.saturating_sub(8));
                let reused = session.generate(
                    &mut model,
                    &tokenizer,
                    &extension,
                    None,
                    Gemma2GenerationConfig {
                        max_tokens: 8,
                        context_limit,
                        temperature: 0.01,
                        top_p: 1.0,
                        seed: 0x5255_5345_4B56,
                    },
                    |_| {},
                )?;
                reuse_reports.push(reused.metrics);
            }
        }
        reports.sort_by(|left, right| {
            left.time_to_first_token_seconds
                .total_cmp(&right.time_to_first_token_seconds)
        });
        let median = reports[reports.len() / 2];
        print_report("inference", prompt_length, median);
        if !reuse_reports.is_empty() {
            reuse_reports.sort_by(|left, right| {
                left.time_to_first_token_seconds
                    .total_cmp(&right.time_to_first_token_seconds)
            });
            let median = reuse_reports[reuse_reports.len() / 2];
            print_report("session_reuse", prompt_length, median);
        }
    }

    Ok(())
}

fn print_report(kind: &str, prompt_length: usize, metrics: Gemma2GenerationMetrics) {
    println!(
        "{kind},{prompt_length},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
        metrics.generated_tokens,
        metrics.time_to_first_token_seconds * 1_000.0,
        metrics.prefill_tokens_per_second(),
        metrics.decode_tokens_per_second(),
        metrics.prefill_seconds * 1_000.0,
        metrics.model_decode_seconds * 1_000.0,
        metrics.logits_processing_seconds * 1_000.0,
        metrics.text_decode_seconds * 1_000.0,
        metrics.cache_reused,
    );
}

fn synthetic_prompt(
    tokenizer: &Gemma2Tokenizer,
    target: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let seed = tokenizer.encode(
        "<start_of_turn>user\nExplica brevemente la relación entre energía, información y \
         causalidad en un sistema físico. ",
    )?;
    let mut tokens = Vec::with_capacity(target);
    tokens.push(tokenizer.bos_id);
    while tokens.len() < target {
        let remaining = target - tokens.len();
        tokens.extend(seed.iter().copied().take(remaining));
    }
    tokens.truncate(target);
    Ok(tokens)
}

fn current_rss_bytes() -> u64 {
    let Ok(pid) = get_current_pid() else {
        return 0;
    };
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map_or(0, |process| process.memory())
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn parse_lengths(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let lengths = value
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if lengths.is_empty() || lengths.contains(&0) {
        return Err("--prompt-lengths requiere enteros positivos".into());
    }
    Ok(lengths)
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == name {
            return args.next();
        }
    }
    None
}

fn arg_usize(name: &str, default: usize) -> usize {
    arg_value(name)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
