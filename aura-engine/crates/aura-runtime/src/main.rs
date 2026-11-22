//! CLI runner `aura-v2` for all 6 AURA v2 Training-Bottleneck Benchmarks (B1–B6).

use aura_runtime::{bench_b1, bench_b2, bench_b3, bench_b4, bench_b5, bench_b6};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("invalid hex seed")
    } else {
        s.parse::<u64>().expect("invalid decimal seed")
    }
}

fn write_b1(dir: &Path, r: &bench_b1::B1Report) {
    let mut f = File::create(dir.join("b1.csv")).unwrap();
    writeln!(
        f,
        "seed,max_abs_diff,mem_ratio_1024,packed_bytes_1024,dense_bytes_1024,\
         us_256_d67_naive,us_256_d67_f32,us_256_d67_lut4_avx2,\
         us_1024_d30_naive,us_1024_d30_f32,us_1024_d30_lut4_avx2,\
         us_1024_d67_naive,us_1024_d67_f32,us_1024_d67_lut4_avx2,\
         us_1024_b16_naive,us_1024_b16_f32,us_1024_b16_lut4_tiled,\
         speedup_b16_vs_naive,speedup_b16_vs_f32,\
         fabric_scatter_tps,fabric_tiled_tps,fabric_tiled_speedup,\
         fw_naive_ups,fw_nlms_ups,fw_nlms_speedup,fw_naive_large_err,fw_nlms_large_err"
    )
    .unwrap();
    writeln!(
        f,
        "{},{},{:.4},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.4},{:.2},{:.2},{:.4},{},{}",
        r.seed,
        r.max_abs_diff,
        r.mem_ratio_1024,
        r.packed_bytes_1024,
        r.dense_bytes_1024,
        r.us_256_d67_naive,
        r.us_256_d67_f32,
        r.us_256_d67_lut4_avx2,
        r.us_1024_d30_naive,
        r.us_1024_d30_f32,
        r.us_1024_d30_lut4_avx2,
        r.us_1024_d67_naive,
        r.us_1024_d67_f32,
        r.us_1024_d67_lut4_avx2,
        r.us_1024_b16_naive,
        r.us_1024_b16_f32,
        r.us_1024_b16_lut4_tiled,
        r.speedup_b16_vs_naive,
        r.speedup_b16_vs_f32,
        r.fabric_scatter_tps,
        r.fabric_tiled_tps,
        r.fabric_tiled_speedup,
        r.fw_naive_ups,
        r.fw_nlms_ups,
        r.fw_nlms_speedup,
        r.fw_naive_large_err,
        r.fw_nlms_large_err
    )
    .unwrap();
}

fn write_b2(dir: &Path, rows: &[bench_b2::B2ArmRow]) {
    let mut f = File::create(dir.join("b2.csv")).unwrap();
    writeln!(
        f,
        "seed,arm,avg_ticks_per_window,total_ticks,wall_ms,late_fe,late_sensory_fe,\
         firing_rate_pct,ne_pre_shift,ne_post_shift,ne_shift_delta,scan_fwd_max_diff,scan_bwd_max_diff"
    )
    .unwrap();
    for r in rows {
        writeln!(
            f,
            "{},{},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.8e},{:.8e}",
            r.seed,
            r.arm,
            r.avg_ticks_per_window,
            r.total_ticks,
            r.wall_ms,
            r.late_fe,
            r.late_sensory_fe,
            r.firing_rate_pct,
            r.ne_pre_shift,
            r.ne_post_shift,
            r.ne_shift_delta,
            r.scan_fwd_max_diff,
            r.scan_bwd_max_diff
        )
        .unwrap();
    }
}

fn write_b3(dir: &Path, rows: &[bench_b3::B3ArmRow]) {
    let mut f = File::create(dir.join("b3.csv")).unwrap();
    writeln!(
        f,
        "seed,arm,acc_a_pre,acc_a_post,forgetting_a,acc_b_post,combined_ab,weight_flips,nonzero_frac"
    )
    .unwrap();
    for r in rows {
        writeln!(
            f,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.6}",
            r.seed,
            r.arm,
            r.acc_a_pre,
            r.acc_a_post,
            r.forgetting_a,
            r.acc_b_post,
            r.combined_ab,
            r.weight_flips,
            r.nonzero_frac
        )
        .unwrap();
    }
}

fn write_b4(dir: &Path, rows: &[bench_b4::B4ArmRow]) {
    let mut f = File::create(dir.join("b4.csv")).unwrap();
    writeln!(
        f,
        "seed,arm,acc_d128,acc_d512,acc_d2048,acc_d5120,hipp_store_frac_pct,total_episodes"
    )
    .unwrap();
    for r in rows {
        writeln!(
            f,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
            r.seed,
            r.arm,
            r.acc_d128,
            r.acc_d512,
            r.acc_d2048,
            r.acc_d5120,
            r.hipp_store_frac_pct,
            r.total_episodes
        )
        .unwrap();
    }
}

fn write_b5(dir: &Path, rows: &[bench_b5::B5ArmRow]) {
    let mut f = File::create(dir.join("b5.csv")).unwrap();
    writeln!(
        f,
        "seed,arm,total_patches,compression_ratio,bpp_predictable,bpp_static,\
         p1_err,p2_err,p4a_err,effective_rank,familiar_reducible_epi,static_reducible_epi,\
         ood_reducible_epi,static_aleatoric,ood_aleatoric,static_hipp_writes,ood_hipp_writes"
    )
    .unwrap();
    for r in rows {
        writeln!(
            f,
            "{},{},{},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
            r.seed,
            r.arm,
            r.total_patches,
            r.compression_ratio,
            r.bpp_predictable,
            r.bpp_static,
            r.p1_err,
            r.p2_err,
            r.p4a_err,
            r.effective_rank,
            r.familiar_reducible_epi,
            r.static_reducible_epi,
            r.ood_reducible_epi,
            r.static_aleatoric,
            r.ood_aleatoric,
            r.static_hipp_writes,
            r.ood_hipp_writes
        )
        .unwrap();
    }
}

fn write_b6(dir: &Path, res: &bench_b6::B6Result) {
    let mut f = File::create(dir.join("b6.csv")).unwrap();
    writeln!(
        f,
        "seed,arm,train_init_bpb,train_final_bpb,eval_md_bpb,shift_rust_bpb,\
         compression_ratio,effective_rank,train_bytes_per_sec,packed_ternary_bytes,dense_f32_bytes,total_flips"
    )
    .unwrap();
    for r in &res.rows {
        writeln!(
            f,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.2},{},{},{}",
            r.seed,
            r.arm,
            r.train_init_bpb,
            r.train_final_bpb,
            r.eval_md_bpb,
            r.shift_rust_bpb,
            r.compression_ratio,
            r.effective_rank,
            r.train_bytes_per_sec,
            r.packed_ternary_bytes,
            r.dense_f32_bytes,
            r.total_flips
        )
        .unwrap();
    }
    if !res.sample_md_gen.is_empty() {
        let mut sf = File::create(dir.join("b6_samples.txt")).unwrap();
        writeln!(sf, "train_md_bytes={}", res.train_md_bytes).unwrap();
        writeln!(sf, "eval_md_bytes={}", res.eval_md_bytes).unwrap();
        writeln!(sf, "shift_rust_bytes={}", res.shift_rust_bytes).unwrap();
        writeln!(sf, "PROMPT_MD: ## Resultados de ").unwrap();
        writeln!(sf, "GEN_MD:    {}", res.sample_md_gen).unwrap();
        writeln!(sf, "PROMPT_RS: pub fn ").unwrap();
        writeln!(sf, "GEN_RS:    {}", res.sample_rust_gen).unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage:\n  aura-v2 all <seed> <out_dir> [--repo <repo_root>] [--samples]\n  \
             aura-v2 b1 <seed>"
        );
        std::process::exit(1);
    }

    let cmd = args[1].as_str();
    let seed = parse_seed(&args[2]);

    if cmd == "b1" {
        let r = bench_b1::run_b1(seed);
        println!("=== B1 Standalone Zero-Mul + Cache Tiling Benchmark (seed=0x{seed:X}) ===");
        println!("max_abs_diff          = {}", r.max_abs_diff);
        println!(
            "mem_ratio_1024        = {:.1}x ({} B packed 2-bit vs {} B f32)",
            r.mem_ratio_1024, r.packed_bytes_1024, r.dense_bytes_1024
        );
        println!(
            "256x256 d=67% B=1     = naive {:.2} us | f32 {:.2} us | lut4_avx2 {:.2} us",
            r.us_256_d67_naive, r.us_256_d67_f32, r.us_256_d67_lut4_avx2
        );
        println!(
            "1024x1024 d=30% B=1   = naive {:.2} us | f32 {:.2} us | lut4_avx2 {:.2} us",
            r.us_1024_d30_naive, r.us_1024_d30_f32, r.us_1024_d30_lut4_avx2
        );
        println!(
            "1024x1024 d=67% B=1   = naive {:.2} us | f32 {:.2} us | lut4_avx2 {:.2} us",
            r.us_1024_d67_naive, r.us_1024_d67_f32, r.us_1024_d67_lut4_avx2
        );
        println!(
            "1024x1024 d=67% B=16  = naive {:.2} us | f32 {:.2} us | lut4_tiled {:.2} us ({:.2}x vs naive, {:.2}x vs f32)",
            r.us_1024_b16_naive,
            r.us_1024_b16_f32,
            r.us_1024_b16_lut4_tiled,
            r.speedup_b16_vs_naive,
            r.speedup_b16_vs_f32
        );
        println!(
            "CorticalFabric 2048   = scatter {:.0} tps | tiled {:.0} tps ({:.2}x)",
            r.fabric_scatter_tps, r.fabric_tiled_tps, r.fabric_tiled_speedup
        );
        println!(
            "FastWeight 256 NLMS   = naive {:.0} ups (err={}) | nlms_avx2 {:.0} ups (err={}, {:.2}x)",
            r.fw_naive_ups,
            r.fw_naive_large_err,
            r.fw_nlms_ups,
            r.fw_nlms_large_err,
            r.fw_nlms_speedup
        );
        return;
    }

    if cmd == "all" {
        if args.len() < 4 {
            eprintln!("Missing <out_dir>");
            std::process::exit(1);
        }
        let out_dir = PathBuf::from(&args[3]);
        fs::create_dir_all(&out_dir).unwrap();

        let mut repo_root = env::current_dir().unwrap();
        let mut emit_samples = false;
        let mut i = 4usize;
        while i < args.len() {
            if args[i] == "--repo" && i + 1 < args.len() {
                repo_root = PathBuf::from(&args[i + 1]);
                i += 2;
            } else if args[i] == "--samples" {
                emit_samples = true;
                i += 1;
            } else {
                i += 1;
            }
        }

        let r1 = bench_b1::run_b1(seed);
        write_b1(&out_dir, &r1);

        let r2 = bench_b2::run_b2(seed);
        write_b2(&out_dir, &r2);

        let r3 = bench_b3::run_b3(seed);
        write_b3(&out_dir, &r3);

        let r4 = bench_b4::run_b4(seed);
        write_b4(&out_dir, &r4);

        let r5 = bench_b5::run_b5(seed);
        write_b5(&out_dir, &r5);

        let r6 = bench_b6::run_b6(seed, &repo_root, emit_samples);
        write_b6(&out_dir, &r6);
    } else {
        eprintln!("Unknown command: {cmd}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_smoke_all_six_benchmarks_are_finite_and_exact() {
        let r1 = bench_b1::run_b1(0xA500);
        assert_eq!(r1.max_abs_diff, 0);
        assert!((r1.mem_ratio_1024 - 16.0).abs() < 1e-6);

        let r2 = bench_b2::run_b2(0xA500);
        assert_eq!(r2.len(), 5);
        assert!(r2[0].scan_fwd_max_diff < 1e-5);
        assert!(r2[0].scan_bwd_max_diff < 1e-5);

        let r3 = bench_b3::run_b3(0xA500);
        assert_eq!(r3.len(), 4);

        let r4 = bench_b4::run_b4(0xA500);
        assert_eq!(r4.len(), 3);

        let r5 = bench_b5::run_b5(0xA500);
        assert_eq!(r5.len(), 4);
    }
}
