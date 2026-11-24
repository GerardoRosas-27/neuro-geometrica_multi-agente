//! CLI entrypoint for `adaptive-spectral-collapse-v1`.
//!
//! Usage:
//!   adaptive-spectral-collapse-v1 run <seed_hex_or_dec> <phase> <out_dir> [--export-traces]

use std::env;
use std::path::PathBuf;
use std::process;

use adaptive_spectral_collapse_v1::suites::run_seed;

fn parse_seed(raw: &str) -> Result<u64, String> {
    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex seed '{}': {}", raw, e))
    } else {
        raw.parse::<u64>()
            .map_err(|e| format!("invalid decimal seed '{}': {}", raw, e))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args[1] != "run" {
        eprintln!(
            "Usage: {} run <seed> <phase> <out_dir> [--export-traces]",
            args.first().map(String::as_str).unwrap_or("adaptive-spectral-collapse-v1")
        );
        process::exit(2);
    }

    let seed = match parse_seed(&args[2]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(2);
        }
    };
    let phase = &args[3];
    let out_dir = PathBuf::from(&args[4]);
    let export_traces = args.iter().skip(5).any(|a| a == "--export-traces");

    if let Err(e) = run_seed(seed, phase, &out_dir, export_traces) {
        eprintln!("Execution error: {}", e);
        process::exit(1);
    }
}
