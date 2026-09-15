//! Entrenador de consola: dataset general enorme, sustrato ancho, guarda cada hora.

use cdt_rqm_epr::field_corpus::N_TOPICS;
use cdt_rqm_epr::field_encoder::{train_encoder_with, EncoderTrainConfig};
use cdt_rqm_epr::field_substrate::ComplexT;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let hours = arg_f64(&args, "--hours", "FIELD_TRAIN_HOURS", 8.0);
    let checkpoint_hours = arg_f64(&args, "--checkpoint-hours", "FIELD_CHECKPOINT_HOURS", 1.0);
    let dataset_size = arg_usize(&args, "--dataset-size", "FIELD_DATASET_SIZE", 100_000);
    let scale = arg_usize(&args, "--scale", "FIELD_SCALE", 2);
    let clusters = arg_usize(&args, "--clusters", "FIELD_CLUSTERS", N_TOPICS);
    let steps = arg_usize(&args, "--steps", "FIELD_STEPS", 10_000_000);
    let seed = arg_u64(&args, "--seed", "FIELD_SEED", 0xE4C0);
    let dir = arg_string(
        &args,
        "--dir",
        "FIELD_TRAIN_DIR",
        "data/field_console_training/general",
    );
    let prefer_gemma = !has_flag(&args, "--no-gemma");
    let t = ComplexT::scaled(scale);

    eprintln!("campo-entrenador");
    eprintln!("  horas={hours}  guarda cada {checkpoint_hours} h");
    eprintln!(
        "  geometria=scaled({scale})  vertices={}  aristas={}  caras={}",
        t.n_vertices,
        t.n_edges(),
        t.faces.len()
    );
    eprintln!("  dataset={dataset_size} frases  clusters={clusters}  temas={N_TOPICS}");
    eprintln!(
        "  gemma={}  dir={dir}",
        if prefer_gemma { "si (GGUF)" } else { "hash" }
    );
    eprintln!("  Ctrl+C guarda y sale. Resume al relanzar el mismo --dir.");

    let stop = Arc::new(AtomicBool::new(false));
    let stop_h = stop.clone();
    let _ = ctrlc::set_handler(move || {
        eprintln!("señal: guardando checkpoint...");
        stop_h.store(true, Ordering::SeqCst);
    });

    let cfg = EncoderTrainConfig {
        steps,
        seed,
        checkpoint_dir: Some(PathBuf::from(&dir)),
        checkpoint_every: 200,
        resume: !has_flag(&args, "--no-resume"),
        prefer_frozen_gemma: prefer_gemma,
        geometry_level: scale,
        corpus_size: dataset_size,
        n_clusters: clusters,
        checkpoint_every_secs: (checkpoint_hours * 3600.0).max(1.0) as u64,
        max_hours: hours,
        eval_every: 100,
        log: true,
        stop: Some(stop),
    };
    let (_enc, report) = train_encoder_with(cfg).map_err(std::io::Error::other)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn arg_usize(args: &[String], flag: &str, env: &str, default: usize) -> usize {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(i + 1) {
            if let Ok(n) = v.parse() {
                return n;
            }
        }
    }
    std::env::var(env)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_u64(args: &[String], flag: &str, env: &str, default: u64) -> u64 {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(i + 1) {
            if let Ok(n) = v.parse() {
                return n;
            }
        }
    }
    std::env::var(env)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_f64(args: &[String], flag: &str, env: &str, default: f64) -> f64 {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(i + 1) {
            if let Ok(n) = v.parse() {
                return n;
            }
        }
    }
    std::env::var(env)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_string(args: &[String], flag: &str, env: &str, default: &str) -> String {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(i + 1) {
            return v.clone();
        }
    }
    std::env::var(env).unwrap_or_else(|_| default.to_string())
}
