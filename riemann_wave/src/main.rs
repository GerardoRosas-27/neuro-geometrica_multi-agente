use std::env;
use std::path::PathBuf;

use riemann_wave_v1::riemann::{compute_riemann_zeros, evaluate_prime_wave_discrimination};
use riemann_wave_v1::suites::run_all_suites;

fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("valid hex seed")
    } else {
        s.parse::<u64>().expect("valid decimal seed")
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage:\n  {} run <seed> <phase> <out_dir> [--export-traces]\n  {} verify-zeros [count]",
            args[0], args[0]
        );
        std::process::exit(2);
    }

    match args[1].as_str() {
        "run" => {
            if args.len() < 5 {
                eprintln!("Expected: run <seed> <phase> <out_dir> [--export-traces]");
                std::process::exit(2);
            }
            let seed = parse_seed(&args[2]);
            let phase = &args[3];
            let out_dir = PathBuf::from(&args[4]);
            let export_traces = args.iter().any(|a| a == "--export-traces");
            run_all_suites(seed, phase, &out_dir, export_traces);
            println!(
                "DONE seed=0x{:04X} phase={} traces={} out={}",
                seed,
                phase,
                export_traces,
                out_dir.display()
            );
        }
        "verify-zeros" => {
            let count = args
                .get(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(128);
            let zeros = compute_riemann_zeros(count);
            let (auc, snr) = evaluate_prime_wave_discrimination(&zeros, 60);
            println!(
                "Computed {} exact Riemann zeta zeros: gamma_1={:.10}, gamma_{}={:.10}",
                zeros.len(),
                zeros[0],
                zeros.len(),
                zeros[zeros.len() - 1]
            );
            println!("Prime-power wave interference (x<=60): AUC={:.6}, SNR={:.4}", auc, snr);
        }
        other => {
            eprintln!("Unknown subcommand: {}", other);
            std::process::exit(2);
        }
    }
}
