use std::env;
use std::path::PathBuf;

use riemann_collapse_v1::collapse::{
    eval_range_discrimination, eval_transactional_collapse, optimize_collapse_config,
    TrainWindowCache,
};
use riemann_collapse_v1::riemann::compute_riemann_zeros;
use riemann_collapse_v1::rng::Rng;
use riemann_collapse_v1::suites::run_all_suites;

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
            "Usage:\n  {} run <seed> <phase> <out_dir> [--export-traces]\n  {} find-config [seed]",
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
        "find-config" => {
            let seed = args.get(2).map(|s| parse_seed(s)).unwrap_or(0xA800);
            let zeros = compute_riemann_zeros(128);
            let cache = TrainWindowCache::new(2, 60);
            let mut rng = Rng::from_seed(seed, 203);
            let res = optimize_collapse_config(&cache, &zeros, &mut rng, false);
            let ho = eval_range_discrimination(61, 150, |x| {
                eval_transactional_collapse(x, &zeros, &res.best_cfg)
            });
            println!("Discovered config (evals={}): {:?}", res.evals_used, res.best_cfg);
            println!(
                "Held-out [61..150]: prime_power_auc={:.6}, prime_only_auc={:.6}, snr={:.4}",
                ho.prime_power_auc, ho.prime_only_auc, ho.prime_power_snr
            );
        }
        other => {
            eprintln!("Unknown subcommand: {}", other);
            std::process::exit(2);
        }
    }
}
