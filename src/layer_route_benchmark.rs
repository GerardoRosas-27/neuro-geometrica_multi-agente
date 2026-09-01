//! Benchmark V8: capas, KL, tok/s sparse vs 26/26, hit LRC, fallback,
//! y comparación opcional con Ollama (`gemma2:2b` por defecto).

use crate::adaptive_gemma2::{conservative_candidate_mask, AdaptiveGemma2Config};
use crate::layer_route_cache::{
    fingerprint_wake, is_sparse_mask, logits_kl, LayerRouteCache, LayerRouteCacheConfig,
};
use crate::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, LayerExecutionMask,
    QuantizedGemma2,
};
use crate::native_gemma2_runtime::{Gemma2GenerationConfig, Gemma2Session};
use candle_core::quantized::gguf_file;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::Duration;

pub const BENCH_PROMPTS: &[&str] = &[
    "Explica en una frase qué es un residual.",
    "Explica en una frase qué es un residual en una red.",
    "Responde siempre en español. ¿Quién eres?",
];

const DEFAULT_GENERATED: usize = 64;
const DEFAULT_OLLAMA_HOST: &str = "127.0.0.1";
const DEFAULT_OLLAMA_PORT: u16 = 11434;
const DEFAULT_OLLAMA_MODEL: &str = "gemma2:2b";

#[derive(Clone, Debug)]
pub struct RouteSpeedConfig {
    pub generated_tokens: usize,
    pub prompt_count: usize,
    pub repetitions: usize,
    pub ollama_host: String,
    pub ollama_port: u16,
    pub ollama_model: String,
    pub compare_ollama: bool,
    pub device: String,
}

impl Default for RouteSpeedConfig {
    fn default() -> Self {
        Self {
            generated_tokens: DEFAULT_GENERATED,
            prompt_count: BENCH_PROMPTS.len(),
            repetitions: 2,
            ollama_host: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| DEFAULT_OLLAMA_HOST.into()),
            ollama_port: std::env::var("OLLAMA_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_OLLAMA_PORT),
            ollama_model: std::env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.into()),
            compare_ollama: true,
            device: std::env::var("GEMMA2_DEVICE").unwrap_or_else(|_| "cpu".into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouteSpeedRow {
    pub backend: String,
    pub prompt_id: usize,
    pub executed_layers: usize,
    pub layer_count: usize,
    pub kl_vs_dense: f32,
    pub decode_tok_s: f64,
    /// generated / model_decode_seconds. Excluye sample + decode UTF-8.
    /// En la fila `ollama` coincide con `decode_tok_s` (eval tok/s de Ollama ya es decode del modelo).
    pub model_decode_tok_s: f64,
    pub ttft_seconds: f64,
    pub lrc_hit: bool,
    pub fallback: bool,
    pub generated_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouteSpeedReport {
    pub rows: Vec<RouteSpeedRow>,
    pub native_dense_mean_tok_s: f64,
    pub native_sparse_mean_tok_s: f64,
    pub ollama_mean_tok_s: Option<f64>,
    pub mean_kl: f32,
    pub mean_executed_layers: f32,
    pub layer_count: usize,
    pub lrc_hit_rate: f32,
    pub fallback_rate: f32,
    pub native_sparse_faster_than_dense: bool,
    pub native_faster_than_ollama: Option<bool>,
}

impl RouteSpeedReport {
    pub fn csv_header() -> &'static str {
        "backend,prompt_id,executed_layers,layer_count,kl_vs_dense,decode_tok_s,model_decode_tok_s,ttft_s,lrc_hit,fallback,generated_tokens"
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::from(Self::csv_header());
        out.push('\n');
        for row in &self.rows {
            out.push_str(&format!(
                "{},{},{},{},{:.6},{:.4},{:.4},{:.4},{},{},{}\n",
                row.backend,
                row.prompt_id,
                row.executed_layers,
                row.layer_count,
                row.kl_vs_dense,
                row.decode_tok_s,
                row.model_decode_tok_s,
                row.ttft_seconds,
                u8::from(row.lrc_hit),
                u8::from(row.fallback),
                row.generated_tokens
            ));
        }
        out
    }

    pub fn summary(&self) -> String {
        format!(
            "layers={:.1}/{} kl={:.4} tok/s dense={:.2} sparse={:.2} ollama={} lrc_hit={:.2} fallback={:.2} sparse>dense={} native>ollama={}",
            self.mean_executed_layers,
            self.layer_count,
            self.mean_kl,
            self.native_dense_mean_tok_s,
            self.native_sparse_mean_tok_s,
            self.ollama_mean_tok_s
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "n/a".into()),
            self.lrc_hit_rate,
            self.fallback_rate,
            self.native_sparse_faster_than_dense,
            self.native_faster_than_ollama
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into())
        )
    }
}

pub fn aggregate_rows(rows: &[RouteSpeedRow]) -> RouteSpeedReport {
    let dense: Vec<&RouteSpeedRow> = rows
        .iter()
        .filter(|row| row.backend == "native_dense")
        .collect();
    let sparse: Vec<&RouteSpeedRow> = rows
        .iter()
        .filter(|row| row.backend == "native_sparse")
        .collect();
    let ollama: Vec<&RouteSpeedRow> = rows.iter().filter(|row| row.backend == "ollama").collect();
    let mean = |values: &[&RouteSpeedRow], pick: fn(&RouteSpeedRow) -> f64| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().map(|row| pick(row)).sum::<f64>() / values.len() as f64
        }
    };
    let native_dense_mean_tok_s = mean(&dense, |row| row.decode_tok_s);
    let native_sparse_mean_tok_s = mean(&sparse, |row| row.decode_tok_s);
    let ollama_mean_tok_s = if ollama.is_empty() {
        None
    } else {
        Some(mean(&ollama, |row| row.decode_tok_s))
    };
    let mean_kl = if sparse.is_empty() {
        0.0
    } else {
        sparse.iter().map(|row| row.kl_vs_dense).sum::<f32>() / sparse.len() as f32
    };
    let mean_executed = if sparse.is_empty() {
        0.0
    } else {
        sparse
            .iter()
            .map(|row| row.executed_layers as f32)
            .sum::<f32>()
            / sparse.len() as f32
    };
    let hits = sparse.iter().filter(|row| row.lrc_hit).count();
    let fallbacks = sparse.iter().filter(|row| row.fallback).count();
    let n = sparse.len().max(1);
    RouteSpeedReport {
        layer_count: rows.first().map(|row| row.layer_count).unwrap_or(0),
        rows: rows.to_vec(),
        native_dense_mean_tok_s,
        native_sparse_mean_tok_s,
        ollama_mean_tok_s,
        mean_kl,
        mean_executed_layers: mean_executed,
        lrc_hit_rate: hits as f32 / n as f32,
        fallback_rate: fallbacks as f32 / n as f32,
        native_sparse_faster_than_dense: native_sparse_mean_tok_s > native_dense_mean_tok_s,
        native_faster_than_ollama: ollama_mean_tok_s
            .map(|ollama| native_sparse_mean_tok_s.max(native_dense_mean_tok_s) > ollama),
    }
}

fn socket_addr(host: &str, port: u16) -> Result<std::net::SocketAddr, String> {
    (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("resolver {host}:{port}: {error}"))?
        .next()
        .ok_or_else(|| format!("sin dirección para {host}:{port}"))
}

pub fn ollama_is_reachable(host: &str, port: u16) -> bool {
    socket_addr(host, port)
        .ok()
        .and_then(|addr| TcpStream::connect_timeout(&addr, Duration::from_millis(400)).ok())
        .is_some()
}

fn connect(host: &str, port: u16, timeout: Duration) -> Result<TcpStream, String> {
    let addr = socket_addr(host, port)?;
    TcpStream::connect_timeout(&addr, timeout)
        .map_err(|error| format!("ollama no responde en {host}:{port}: {error}"))
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    prompt_eval_duration: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OllamaSpeedSample {
    pub decode_tok_s: f64,
    pub ttft_seconds: f64,
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    pub text_len: usize,
}

pub fn parse_ollama_generate_json(body: &str) -> Result<OllamaSpeedSample, String> {
    let parsed: OllamaGenerateResponse =
        serde_json::from_str(body).map_err(|error| format!("json ollama: {error}"))?;
    let decode_seconds = parsed.eval_duration as f64 / 1.0e9;
    let prefill_seconds = parsed.prompt_eval_duration as f64 / 1.0e9;
    Ok(OllamaSpeedSample {
        decode_tok_s: parsed.eval_count as f64 / decode_seconds.max(f64::EPSILON),
        ttft_seconds: prefill_seconds,
        generated_tokens: parsed.eval_count as usize,
        prompt_tokens: parsed.prompt_eval_count as usize,
        text_len: parsed.response.chars().count(),
    })
}

pub fn probe_ollama(
    config: &RouteSpeedConfig,
    prompt: &str,
    generated_tokens: usize,
) -> Result<OllamaSpeedSample, String> {
    let payload = serde_json::json!({
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_predict": generated_tokens.max(1),
            "temperature": 0.01,
            "top_p": 1.0,
            "seed": 42
        }
    })
    .to_string();
    let mut stream = connect(
        &config.ollama_host,
        config.ollama_port,
        Duration::from_secs(3),
    )?;
    stream
        .set_read_timeout(Some(Duration::from_secs(180)))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|error| error.to_string())?;
    let request = format!(
        "POST /api/generate HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        config.ollama_host,
        config.ollama_port,
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| error.to_string())?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|error| error.to_string())?;
    let text = String::from_utf8_lossy(&raw);
    let body = text
        .split("\r\n\r\n")
        .nth(1)
        .or_else(|| text.split("\n\n").nth(1))
        .ok_or_else(|| {
            format!(
                "respuesta HTTP sin cuerpo: {}",
                text.chars().take(120).collect::<String>()
            )
        })?;
    if text.starts_with("HTTP/1.1 4") || text.starts_with("HTTP/1.0 4") {
        return Err(format!(
            "ollama HTTP error: {}",
            body.chars().take(200).collect::<String>()
        ));
    }
    parse_ollama_generate_json(body.trim())
}

pub fn run_route_speed_benchmark(
    config: RouteSpeedConfig,
) -> Result<RouteSpeedReport, Box<dyn std::error::Error>> {
    let model_path = resolve_gemma2_model_path(None)?;
    run_route_speed_benchmark_with_model(&model_path, config)
}

pub fn run_route_speed_benchmark_with_model(
    model_path: &Path,
    config: RouteSpeedConfig,
) -> Result<RouteSpeedReport, Box<dyn std::error::Error>> {
    let device = resolve_gemma2_device(&config.device)?;
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
    let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
    let layer_count = model.layer_count();
    let dense_mask = LayerExecutionMask::all(layer_count);
    let mut lrc = LayerRouteCache::new(LayerRouteCacheConfig::default());
    let gen_config = Gemma2GenerationConfig {
        max_tokens: config.generated_tokens.max(1),
        context_limit: model.max_context(),
        temperature: 0.01,
        top_p: 1.0,
        seed: 0x4745_4D4D_4132,
    };
    let mut rows = Vec::new();
    let prompts = &BENCH_PROMPTS[..config.prompt_count.clamp(1, BENCH_PROMPTS.len())];
    for (prompt_id, prompt) in prompts.iter().copied().enumerate() {
        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(prompt)?);
        let input = Tensor::new(tokens.as_slice(), model.device())?.unsqueeze(0)?;
        model.clear_kv_cache();
        let dense_prefill = model.forward_with_mask(&input, 0, Some(&dense_mask), true, false)?;
        let dense_logits = dense_prefill
            .logits
            .squeeze(0)?
            .to_vec1::<f32>()
            .unwrap_or_default();
        let sparse_mask =
            conservative_candidate_mask(&dense_prefill.trace, &AdaptiveGemma2Config::default());
        model.clear_kv_cache();
        let sparse_prefill =
            model.forward_with_mask(&input, 0, Some(&sparse_mask), false, false)?;
        let sparse_logits = sparse_prefill
            .logits
            .squeeze(0)?
            .to_vec1::<f32>()
            .unwrap_or_default();
        let kl = logits_kl(&dense_logits, &sparse_logits);
        let fingerprint = fingerprint_wake(&tokens);
        let lrc_hit = lrc.lookup_confident(&fingerprint).is_some();
        let fallback = !is_sparse_mask(&sparse_mask) || kl > lrc.config().max_kl_promote;
        if is_sparse_mask(&sparse_mask) {
            let _ = lrc.promote(
                fingerprint.clone(),
                sparse_mask.clone(),
                kl,
                crate::layer_route_cache::top1_agree(&dense_logits, &sparse_logits),
                prompt_id as u64 + 1,
            );
        }

        let dense_metrics = timed_generate_median(
            &mut model,
            &tokenizer,
            &tokens,
            None,
            &gen_config,
            config.repetitions,
        )?;
        rows.push(RouteSpeedRow {
            backend: "native_dense".into(),
            prompt_id,
            executed_layers: layer_count,
            layer_count,
            kl_vs_dense: 0.0,
            decode_tok_s: dense_metrics.decode_tok_s,
            model_decode_tok_s: dense_metrics.model_decode_tok_s,
            ttft_seconds: dense_metrics.ttft_seconds,
            lrc_hit: false,
            fallback: false,
            generated_tokens: dense_metrics.generated_tokens,
        });

        let sparse_metrics = timed_generate_median(
            &mut model,
            &tokenizer,
            &tokens,
            Some(&sparse_mask),
            &gen_config,
            config.repetitions,
        )?;
        rows.push(RouteSpeedRow {
            backend: "native_sparse".into(),
            prompt_id,
            executed_layers: sparse_mask.executed_count(),
            layer_count,
            kl_vs_dense: kl,
            decode_tok_s: sparse_metrics.decode_tok_s,
            model_decode_tok_s: sparse_metrics.model_decode_tok_s,
            ttft_seconds: sparse_metrics.ttft_seconds,
            lrc_hit,
            fallback,
            generated_tokens: sparse_metrics.generated_tokens,
        });

        if config.compare_ollama {
            match probe_ollama_median(&config, prompt, config.generated_tokens, config.repetitions)
            {
                Ok(sample) => rows.push(RouteSpeedRow {
                    backend: "ollama".into(),
                    prompt_id,
                    executed_layers: layer_count,
                    layer_count,
                    kl_vs_dense: 0.0,
                    decode_tok_s: sample.decode_tok_s,
                    // eval tok/s de Ollama ya es decode del modelo; la columna queda numerica.
                    model_decode_tok_s: sample.decode_tok_s,
                    ttft_seconds: sample.ttft_seconds,
                    lrc_hit: false,
                    fallback: false,
                    generated_tokens: sample.generated_tokens,
                }),
                Err(error) => eprintln!("ollama omitido prompt={prompt_id}: {error}"),
            }
        }
    }
    Ok(aggregate_rows(&rows))
}

struct TimedSample {
    decode_tok_s: f64,
    model_decode_tok_s: f64,
    ttft_seconds: f64,
    generated_tokens: usize,
}

fn median_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        n if n % 2 == 1 => sorted[n / 2],
        n => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

fn timed_generate(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    tokens: &[u32],
    mask: Option<&LayerExecutionMask>,
    config: &Gemma2GenerationConfig,
) -> Result<TimedSample, Box<dyn std::error::Error>> {
    let mut session = Gemma2Session::new();
    session.reset(model);
    let generation = session.generate(model, tokenizer, tokens, mask, *config, |_| {})?;
    let generated = generation.metrics.generated_tokens;
    Ok(TimedSample {
        decode_tok_s: generation.metrics.decode_tokens_per_second(),
        model_decode_tok_s: generated as f64
            / generation.metrics.model_decode_seconds.max(f64::EPSILON),
        ttft_seconds: generation.metrics.time_to_first_token_seconds,
        generated_tokens: generated,
    })
}

fn timed_generate_median(
    model: &mut QuantizedGemma2,
    tokenizer: &Gemma2Tokenizer,
    tokens: &[u32],
    mask: Option<&LayerExecutionMask>,
    config: &Gemma2GenerationConfig,
    repetitions: usize,
) -> Result<TimedSample, Box<dyn std::error::Error>> {
    let reps = repetitions.max(1);
    let mut decode = Vec::with_capacity(reps);
    let mut model_decode = Vec::with_capacity(reps);
    let mut ttft = Vec::with_capacity(reps);
    let mut generated = 0usize;
    for _ in 0..reps {
        let sample = timed_generate(model, tokenizer, tokens, mask, config)?;
        decode.push(sample.decode_tok_s);
        model_decode.push(sample.model_decode_tok_s);
        ttft.push(sample.ttft_seconds);
        generated = sample.generated_tokens;
    }
    Ok(TimedSample {
        decode_tok_s: median_f64(&decode),
        model_decode_tok_s: median_f64(&model_decode),
        ttft_seconds: median_f64(&ttft),
        generated_tokens: generated,
    })
}

fn probe_ollama_median(
    config: &RouteSpeedConfig,
    prompt: &str,
    generated_tokens: usize,
    repetitions: usize,
) -> Result<OllamaSpeedSample, String> {
    let reps = repetitions.max(1);
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        samples.push(probe_ollama(config, prompt, generated_tokens)?);
    }
    let decode: Vec<f64> = samples.iter().map(|s| s.decode_tok_s).collect();
    let ttft: Vec<f64> = samples.iter().map(|s| s.ttft_seconds).collect();
    let last = samples.last().cloned().expect("repetitions >= 1");
    Ok(OllamaSpeedSample {
        decode_tok_s: median_f64(&decode),
        ttft_seconds: median_f64(&ttft),
        generated_tokens: last.generated_tokens,
        prompt_tokens: last.prompt_tokens,
        text_len: last.text_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_generated_tokens_is_64() {
        assert_eq!(RouteSpeedConfig::default().generated_tokens, 64);
        assert_eq!(RouteSpeedConfig::default().repetitions, 2);
        assert_eq!(
            RouteSpeedConfig::default().prompt_count,
            BENCH_PROMPTS.len()
        );
    }

    #[test]
    fn median_of_two_runs_is_midpoint() {
        assert!((median_f64(&[10.0, 12.0]) - 11.0).abs() < 1.0e-9);
        assert!((median_f64(&[7.0]) - 7.0).abs() < 1.0e-9);
    }

    #[test]
    fn csv_header_lists_required_columns() {
        let header = RouteSpeedReport::csv_header();
        for column in [
            "backend",
            "executed_layers",
            "kl_vs_dense",
            "decode_tok_s",
            "model_decode_tok_s",
            "lrc_hit",
            "fallback",
            "generated_tokens",
        ] {
            assert!(header.contains(column), "{header}");
        }
    }

    #[test]
    fn aggregate_computes_hit_fallback_and_speed_verdict() {
        let rows = vec![
            RouteSpeedRow {
                backend: "native_dense".into(),
                prompt_id: 0,
                executed_layers: 26,
                layer_count: 26,
                kl_vs_dense: 0.0,
                decode_tok_s: 10.0,
                model_decode_tok_s: 10.0,
                ttft_seconds: 1.0,
                lrc_hit: false,
                fallback: false,
                generated_tokens: 16,
            },
            RouteSpeedRow {
                backend: "native_sparse".into(),
                prompt_id: 0,
                executed_layers: 23,
                layer_count: 26,
                kl_vs_dense: 0.04,
                decode_tok_s: 12.0,
                model_decode_tok_s: 12.0,
                ttft_seconds: 0.9,
                lrc_hit: false,
                fallback: false,
                generated_tokens: 16,
            },
            RouteSpeedRow {
                backend: "native_sparse".into(),
                prompt_id: 1,
                executed_layers: 23,
                layer_count: 26,
                kl_vs_dense: 0.05,
                decode_tok_s: 13.0,
                model_decode_tok_s: 13.0,
                ttft_seconds: 0.8,
                lrc_hit: true,
                fallback: true,
                generated_tokens: 16,
            },
            RouteSpeedRow {
                backend: "ollama".into(),
                prompt_id: 0,
                executed_layers: 26,
                layer_count: 26,
                kl_vs_dense: 0.0,
                decode_tok_s: 8.0,
                model_decode_tok_s: 8.0,
                ttft_seconds: 1.2,
                lrc_hit: false,
                fallback: false,
                generated_tokens: 16,
            },
        ];
        let report = aggregate_rows(&rows);
        assert!((report.lrc_hit_rate - 0.5).abs() < 1.0e-6);
        assert!((report.fallback_rate - 0.5).abs() < 1.0e-6);
        assert!((report.native_sparse_mean_tok_s - 12.5).abs() < 1.0e-6);
        assert!(report.native_sparse_faster_than_dense);
        assert_eq!(report.native_faster_than_ollama, Some(true));
        assert!(report.to_csv().starts_with(RouteSpeedReport::csv_header()));
    }

    #[test]
    fn ollama_parser_extracts_eval_throughput() {
        let body = r#"{"response":"hola","eval_count":32,"eval_duration":2000000000,"prompt_eval_count":8,"prompt_eval_duration":500000000}"#;
        let sample = parse_ollama_generate_json(body).unwrap();
        assert_eq!(sample.generated_tokens, 32);
        assert!((sample.decode_tok_s - 16.0).abs() < 1.0e-6);
        assert!((sample.ttft_seconds - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn native_sparse_vs_dense_and_ollama_report_layers_kl_and_tok_s() {
        let path = match resolve_gemma2_model_path(None) {
            Ok(path) => path,
            Err(error) => {
                eprintln!("GGUF ausente, se omite: {error}");
                return;
            }
        };
        let compare_ollama = ollama_is_reachable("127.0.0.1", 11434);
        if !compare_ollama {
            eprintln!("Ollama no responde en 127.0.0.1:11434; se mide solo nativo");
        }
        let report = run_route_speed_benchmark_with_model(
            &path,
            RouteSpeedConfig {
                generated_tokens: 32,
                prompt_count: 3,
                compare_ollama,
                ..RouteSpeedConfig::default()
            },
        )
        .expect("benchmark nativo");
        eprintln!("{}", report.to_csv());
        eprintln!("{}", report.summary());
        assert_eq!(report.layer_count, 26);
        assert!(report.mean_executed_layers < 26.0, "{}", report.summary());
        assert!(report.mean_kl.is_finite());
        assert!(report.native_dense_mean_tok_s > 0.0);
        assert!(report.native_sparse_mean_tok_s > 0.0);
        let native_rows = report
            .rows
            .iter()
            .filter(|row| row.backend.starts_with("native_"))
            .count();
        assert_eq!(native_rows, 6, "{}", report.to_csv());
        assert!(
            report
                .rows
                .iter()
                .all(|row| row.model_decode_tok_s.is_finite() && row.model_decode_tok_s > 0.0),
            "{}",
            report.to_csv()
        );
        assert!(
            report
                .rows
                .iter()
                .filter(|row| row.backend.starts_with("native_"))
                .all(|row| row.generated_tokens > 0),
            "{}",
            report.to_csv()
        );
        if compare_ollama {
            assert!(report.ollama_mean_tok_s.is_some(), "{}", report.summary());
            assert!(
                report.native_faster_than_ollama.is_some(),
                "{}",
                report.summary()
            );
        }
    }
}
