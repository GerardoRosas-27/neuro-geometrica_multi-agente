use cdt_rqm_epr::field_encoder::{train_encoder_with, EncoderTrainConfig, TrainReport};
use cdt_rqm_epr::field_substrate::{
    run_dataset_training, run_experiment_table, run_two_pattern_comparison, DatasetTrainConfig,
};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Serialize)]
struct FieldExperimentReport {
    protocol: &'static str,
    note: &'static str,
    seeds: Vec<u64>,
    comparison: Vec<cdt_rqm_epr::field_substrate::ComparisonReport>,
    mean_recon_field: f64,
    mean_recon_do_nothing: f64,
    mean_recon_neighbor: f64,
    mean_recon_hopfield: f64,
    mean_recon_hebb: f64,
    two_pattern: Option<TwoPatternMeans>,
    encoder: Option<TrainReport>,
    training: Option<cdt_rqm_epr::field_substrate::TrainingProgress>,
}

#[derive(Serialize)]
struct TwoPatternMeans {
    first_field: f64,
    first_hopfield: f64,
    second_field: f64,
    second_hopfield: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dataset_size = arg_or_env_usize(&args, "--dataset-size", "FIELD_DATASET_SIZE", 32);
    let checkpoint_every =
        arg_or_env_usize(&args, "--checkpoint-every", "FIELD_CHECKPOINT_EVERY", 0);
    let dataset_id = arg_or_env_string(
        &args,
        "--dataset-id",
        "FIELD_DATASET_ID",
        "synthetic-unit-phasors",
    );
    let checkpoint_dir = arg_or_env_string(
        &args,
        "--checkpoint-dir",
        "FIELD_CHECKPOINT_DIR",
        &format!("data/field_substrate_training/{dataset_id}"),
    );
    let resume = env_flag("FIELD_RESUME", true) && !has_flag(&args, "--no-resume");
    let skip_train = has_flag(&args, "--no-train");
    let skip_encoder = has_flag(&args, "--no-encoder");
    let prefer_gemma = !has_flag(&args, "--no-gemma");
    let encoder_steps = arg_or_env_usize(&args, "--encoder-steps", "FIELD_ENCODER_STEPS", 120);
    let encoder_dir = arg_or_env_string(
        &args,
        "--encoder-dir",
        "FIELD_ENCODER_DIR",
        "data/field_substrate_training/encoder-corpus",
    );

    let seeds: Vec<u64> = (0..8).collect();
    let comparison = run_experiment_table(&seeds);
    let n = comparison.len() as f64;
    let mean = |f: fn(&cdt_rqm_epr::field_substrate::ComparisonReport) -> f64| {
        comparison.iter().map(f).sum::<f64>() / n
    };

    let (a, b) = run_two_pattern_comparison(0);
    let two_pattern = TwoPatternMeans {
        first_field: a.recon_field,
        first_hopfield: a.recon_hopfield,
        second_field: b.recon_field,
        second_hopfield: b.recon_hopfield,
    };

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = stop.clone();
    let _ = ctrlc::set_handler(move || {
        stop_handler.store(true, Ordering::SeqCst);
    });

    let encoder = if skip_encoder {
        None
    } else {
        let enc_cfg = EncoderTrainConfig {
            steps: encoder_steps,
            seed: 0xE4C0,
            checkpoint_dir: Some(PathBuf::from(&encoder_dir)),
            checkpoint_every: (encoder_steps / 4).max(1),
            resume,
            prefer_frozen_gemma: prefer_gemma,
            stop: Some(stop.clone()),
            ..EncoderTrainConfig::default()
        };
        let (_enc, report) = train_encoder_with(enc_cfg).map_err(std::io::Error::other)?;
        Some(report)
    };

    let training = if skip_train {
        None
    } else {
        let mut cfg = DatasetTrainConfig::synthetic(PathBuf::from(&checkpoint_dir), dataset_size);
        cfg.dataset_id = dataset_id;
        cfg.resume = resume;
        cfg.stop = Some(stop);
        if checkpoint_every > 0 {
            cfg.checkpoint_every = checkpoint_every;
        }
        Some(run_dataset_training(cfg).map_err(std::io::Error::other)?)
    };

    let report = FieldExperimentReport {
        protocol: "field_substrate_v2",
        note: "experimento paralelo, no es el preprint de cuenca; merge a main = estos archivos, no LRC/#9",
        seeds,
        mean_recon_field: mean(|r| r.recon_field),
        mean_recon_do_nothing: mean(|r| r.recon_do_nothing),
        mean_recon_neighbor: mean(|r| r.recon_neighbor),
        mean_recon_hopfield: mean(|r| r.recon_hopfield),
        mean_recon_hebb: mean(|r| r.recon_hebb),
        comparison,
        two_pattern: Some(two_pattern),
        encoder,
        training,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn arg_or_env_usize(args: &[String], flag: &str, env: &str, default: usize) -> usize {
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

fn arg_or_env_string(args: &[String], flag: &str, env: &str, default: &str) -> String {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(i + 1) {
            return v.clone();
        }
    }
    std::env::var(env).unwrap_or_else(|_| default.to_string())
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"),
        Err(_) => default,
    }
}
