//! CLI determinista `aura-exp` para ejecutar la suite científica AURA + CorticalFabric v1.
//!
//! Uso:
//!   aura-exp run <seed_u64> <DEV|CONFIRM> <out_dir>
//!   aura-exp bench [seed_u64]

use aura_runtime::{exp_e1, exp_e2, exp_e3, exp_e4, exp_e5};
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("invalid hex seed")
    } else {
        s.parse::<u64>().expect("invalid decimal seed")
    }
}

fn run_seed(seed: u64, phase: &str, out_dir: &Path) {
    fs::create_dir_all(out_dir).expect("create out_dir");

    let e1_rows = exp_e1::run_e1_for_seed(seed, phase);
    let e2_rows = exp_e2::run_e2_for_seed(seed, phase);
    let e3_rows = exp_e3::run_e3_for_seed(seed, phase);
    let e4_rows = exp_e4::run_e4_for_seed(seed, phase);
    let e5 = exp_e5::run_e5_verification(seed, 500, 60);

    // 1. Escribir metrics_e1.csv
    {
        let mut f = fs::File::create(out_dir.join("metrics_e1.csv")).expect("create metrics_e1");
        writeln!(
            f,
            "seed,phase,arm,total_patches,pred_bytes_per_patch,high_ent_bytes_per_patch,compression_ratio,phase1_norm_err,phase2_shift_norm_err,phase2_recovery_steps,phase3_static_norm_err,phase3_static_coherence,phase2_shift_coherence,phase4_recall_norm_err,familiar_epistemic,static_epistemic,ood_epistemic,effective_rank,mean_latent_norm,eval_ms"
        )
        .unwrap();
        for r in &e1_rows {
            writeln!(
                f,
                "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4}",
                r.seed,
                r.phase,
                r.arm,
                r.total_patches,
                r.pred_bytes_per_patch,
                r.high_ent_bytes_per_patch,
                r.compression_ratio,
                r.phase1_norm_err,
                r.phase2_shift_norm_err,
                r.phase2_recovery_steps,
                r.phase3_static_norm_err,
                r.phase3_static_coherence,
                r.phase2_shift_coherence,
                r.phase4_recall_norm_err,
                r.familiar_epistemic,
                r.static_epistemic,
                r.ood_epistemic,
                r.effective_rank,
                r.mean_latent_norm,
                r.eval_ms
            )
            .unwrap();
        }
    }

    // 2. Escribir metrics_e2.csv
    {
        let mut f = fs::File::create(out_dir.join("metrics_e2.csv")).expect("create metrics_e2");
        writeln!(
            f,
            "seed,phase,arm,phase1_return,phase1_success,phase2_shift_return,phase2_shift_success,phase3_ood_return,combined_p2_p3_return,abstain_rate_trap,abstain_rate_familiar,system1_frac_familiar,mean_rollouts_per_step,eval_ms"
        )
        .unwrap();
        for r in &e2_rows {
            writeln!(
                f,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4}",
                r.seed,
                r.phase,
                r.arm,
                r.phase1_return,
                r.phase1_success,
                r.phase2_shift_return,
                r.phase2_shift_success,
                r.phase3_ood_return,
                r.combined_p2_p3_return,
                r.abstain_rate_trap,
                r.abstain_rate_familiar,
                r.system1_frac_familiar,
                r.mean_rollouts_per_step,
                r.eval_ms
            )
            .unwrap();
        }
    }

    // 3. Escribir metrics_e3.csv
    {
        let mut f = fs::File::create(out_dir.join("metrics_e3.csv")).expect("create metrics_e3");
        writeln!(
            f,
            "seed,phase,arm,acc_a_pre,acc_a_post,forgetting_a,acc_b_post,combined_ab,weight_flips,nonzero_frac,eval_ms"
        )
        .unwrap();
        for r in &e3_rows {
            writeln!(
                f,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.4}",
                r.seed,
                r.phase,
                r.arm,
                r.acc_a_pre,
                r.acc_a_post,
                r.forgetting_a,
                r.acc_b_post,
                r.combined_ab,
                r.weight_flips,
                r.nonzero_frac,
                r.eval_ms
            )
            .unwrap();
        }
    }

    // 4. Escribir metrics_e4.csv
    {
        let mut f = fs::File::create(out_dir.join("metrics_e4.csv")).expect("create metrics_e4");
        writeln!(
            f,
            "seed,phase,arm,fe_regime1_early,fe_regime1_late,fe_regime2_early,fe_regime2_late,fe_late_mean,sensory_fe_late,mean_spike_rate_pct,ne_pre_shift,ne_post_shift,ne_shift_delta,ticks_per_sec,eval_ms"
        )
        .unwrap();
        for r in &e4_rows {
            writeln!(
                f,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2},{:.4}",
                r.seed,
                r.phase,
                r.arm,
                r.fe_regime1_early,
                r.fe_regime1_late,
                r.fe_regime2_early,
                r.fe_regime2_late,
                r.fe_late_mean,
                r.sensory_fe_late,
                r.mean_spike_rate_pct,
                r.ne_pre_shift,
                r.ne_post_shift,
                r.ne_shift_delta,
                r.ticks_per_sec,
                r.eval_ms
            )
            .unwrap();
        }
    }

    // 5. Escribir bench_e5.json
    {
        let mut f = fs::File::create(out_dir.join("bench_e5.json")).expect("create bench_e5");
        writeln!(
            f,
            "{{\n  \"seed\": {},\n  \"phase\": \"{}\",\n  \"max_abs_diff_zero_mul\": {},\n  \"memory_ratio_f32_over_packed\": {},\n  \"dim256_zero_mul_us\": {:.4},\n  \"dim256_f32_dense_us\": {:.4},\n  \"dim1024_zero_mul_us\": {:.4},\n  \"dim1024_f32_dense_us\": {:.4},\n  \"dim1024_packed_bytes\": {},\n  \"dim1024_f32_bytes\": {}\n}}",
            seed,
            phase,
            e5.max_abs_diff_zero_mul,
            e5.memory_ratio_f32_over_packed,
            e5.dim256_zero_mul_us,
            e5.dim256_f32_dense_us,
            e5.dim1024_zero_mul_us,
            e5.dim1024_f32_dense_us,
            e5.dim1024_packed_bytes,
            e5.dim1024_f32_bytes
        )
        .unwrap();
    }

    // 6. Traza resumida legible para auditoría.
    {
        let mut f = fs::File::create(out_dir.join("trace.txt")).expect("create trace");
        writeln!(f, "AURA + CorticalFabric v1 | seed=0x{seed:04X} ({seed}) | phase={phase}").unwrap();
        writeln!(f, "[E1] aura_full shift_err={:.4} vs no_fast={:.4} | recall={:.4} vs no_hip={:.4} | erank={:.2} vs no_vicreg={:.2}",
            e1_rows[0].phase2_shift_norm_err,
            e1_rows[1].phase2_shift_norm_err,
            e1_rows[0].phase4_recall_norm_err,
            e1_rows[2].phase4_recall_norm_err,
            e1_rows[0].effective_rank,
            e1_rows[4].effective_rank,
        ).unwrap();
        writeln!(f, "[E2] aura_full p2+p3={:.3} (p2_succ={:.2}, trap_abs={:.2}, fam_abs={:.2}) vs sys1={:.3} vs no_fast={:.3}",
            e2_rows[0].combined_p2_p3_return,
            e2_rows[0].phase2_shift_success,
            e2_rows[0].abstain_rate_trap,
            e2_rows[0].abstain_rate_familiar,
            e2_rows[1].combined_p2_p3_return,
            e2_rows[2].combined_p2_p3_return,
        ).unwrap();
        writeln!(f, "[E3] sleep_full A_pre={:.3} A_post={:.3} (forg={:.3}) B={:.3} mean={:.3} | no_replay A_post={:.3} B={:.3} | no_hyst A_post={:.3} B={:.3}",
            e3_rows[0].acc_a_pre,
            e3_rows[0].acc_a_post,
            e3_rows[0].forgetting_a,
            e3_rows[0].acc_b_post,
            e3_rows[0].combined_ab,
            e3_rows[1].acc_a_post,
            e3_rows[1].acc_b_post,
            e3_rows[2].acc_a_post,
            e3_rows[2].acc_b_post,
        ).unwrap();
        writeln!(f, "[E4] cortical_full FE_late={:.1} (spikes={:.2}%, NE_delta={:+.2}) vs frozen={:.1} vs no_phase={:.1} vs no_nm={:.1}",
            e4_rows[0].fe_late_mean,
            e4_rows[0].mean_spike_rate_pct,
            e4_rows[0].ne_shift_delta,
            e4_rows[1].fe_late_mean,
            e4_rows[2].fe_late_mean,
            e4_rows[3].fe_late_mean,
        ).unwrap();
        writeln!(f, "[E5] zero_mul_max_diff={} mem_ratio={}x 256={:.2}us(zm)/{:.2}us(f32) 1024={:.2}us(zm)/{:.2}us(f32)",
            e5.max_abs_diff_zero_mul,
            e5.memory_ratio_f32_over_packed,
            e5.dim256_zero_mul_us,
            e5.dim256_f32_dense_us,
            e5.dim1024_zero_mul_us,
            e5.dim1024_f32_dense_us,
        ).unwrap();
    }

    println!(
        "DONE seed=0x{seed:04X} phase={phase} | E1_shift={:.4}/{:.4} | E2_ret={:.2}/{:.2} | E3_AB={:.3}/{:.3} | E4_FE={:.0}/{:.0}",
        e1_rows[0].phase2_shift_norm_err,
        e1_rows[1].phase2_shift_norm_err,
        e2_rows[0].combined_p2_p3_return,
        e2_rows[1].combined_p2_p3_return,
        e3_rows[0].combined_ab,
        e3_rows[1].combined_ab,
        e4_rows[0].fe_late_mean,
        e4_rows[1].fe_late_mean,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:\n  aura-exp run <seed> <DEV|CONFIRM> <out_dir>\n  aura-exp bench [seed]");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "run" => {
            if args.len() != 5 {
                eprintln!("Usage: aura-exp run <seed> <DEV|CONFIRM> <out_dir>");
                std::process::exit(1);
            }
            let seed = parse_seed(&args[2]);
            let phase = &args[3];
            let out_dir = Path::new(&args[4]);
            run_seed(seed, phase, out_dir);
        }
        "bench" => {
            let seed = args.get(2).map(|s| parse_seed(s)).unwrap_or(0xA400);
            let r = exp_e5::run_e5_verification(seed, 2000, 200);
            println!("=== AURA E5 Zero-Mul Engineering Verification ===");
            println!("max_abs_diff_zero_mul        : {}", r.max_abs_diff_zero_mul);
            println!("memory_ratio (f32 / packed)  : {}x ({} B vs {} B for 1024x1024)", r.memory_ratio_f32_over_packed, r.dim1024_packed_bytes, r.dim1024_f32_bytes);
            println!("256x256  zero-mul vs f32     : {:.2} us vs {:.2} us", r.dim256_zero_mul_us, r.dim256_f32_dense_us);
            println!("1024x1024 zero-mul vs f32    : {:.2} us vs {:.2} us", r.dim1024_zero_mul_us, r.dim1024_f32_dense_us);
        }
        other => {
            eprintln!("Unknown subcommand: {other}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aura_metacog::MetacogConfig;
    use aura_perception::{EntropyPatcher, PerceptionEncoder};
    use aura_runtime::agent::AuraAgent;

    #[test]
    fn aura_agent_end_to_end_smoke_deterministic() {
        let make_run = || {
            let p = EntropyPatcher::new(0xA400, 1, 6, 420, true);
            let mut agent = AuraAgent::new(p, 0xA400, 6, true, MetacogConfig::default());
            let goal = aura_math::LatentVec::zero();
            let mut out = Vec::new();
            let stream = b"SEQ:K0->abcdSEQ:K0->abcdSEQ:M4->efgh";
            let mut prev_act = None;
            for &b in stream {
                if let Some(z) = agent.perception.push_byte(b) {
                    let (act, rep) = agent.cognitive_step(z, &goal, &[0, 1, 2, 3], 0, prev_act, 0, 1);
                    prev_act = Some(act);
                    out.push((act, rep.surprise_l1, rep.epistemic_uncertainty));
                }
            }
            out
        };
        let r1 = make_run();
        let r2 = make_run();
        assert!(!r1.is_empty());
        assert_eq!(r1, r2, "AuraAgent must be bit-exact deterministic");
    }

    #[test]
    fn e5_zero_mul_verification_passes() {
        let r = exp_e5::run_e5_verification(0xA400, 20, 5);
        assert_eq!(r.max_abs_diff_zero_mul, 0);
        assert_eq!(r.memory_ratio_f32_over_packed, 16);
    }
}
