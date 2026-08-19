//! Benchmark pareado legacy (híbrido fasorial-CDT directo) vs nueva arquitectura
//! (RFF fasorial + Softmax CTP + reservorio Langevin + consolidación CDT).

use cdt_rqm_epr::hybrid_thermo_attention_comparison::{
    run_hybrid_legacy_comparison, HybridLegacyComparisonConfig,
};

fn main() {
    let trials = env_usize("HYBRID_LEGACY_TRIALS", 12);
    let seq_len = env_usize("HYBRID_LEGACY_SEQ_LEN", 16);
    let d_model = env_usize("HYBRID_LEGACY_D_MODEL", 16);
    let d_v = env_usize("HYBRID_LEGACY_D_V", 8);
    let planted = env_usize("HYBRID_LEGACY_PLANTED", 4);
    let nodes = env_usize("HYBRID_LEGACY_CDT_NODES", 128);

    let config = HybridLegacyComparisonConfig {
        sequence_length: seq_len,
        d_model,
        d_v,
        planted_pairs: planted,
        trials,
        cdt_nodes: nodes,
        ..Default::default()
    };

    let report = run_hybrid_legacy_comparison(config);

    println!("benchmark=hybrid_thermo_attention_vs_legacy");
    println!(
        "config,trials={trials},seq_len={seq_len},d_model={d_model},d_v={d_v},\
         planted={planted},cdt_nodes={nodes}"
    );
    println!();
    println!("metric,legacy,hybrid,delta,ratio");
    print_row(
        "handshake_top1",
        report.legacy.mean_top1(),
        report.hybrid.mean_top1(),
    );
    print_row(
        "handshake_mrr",
        report.legacy.mean_mrr(),
        report.hybrid.mean_mrr(),
    );
    print_row(
        "handshake_planted_mass",
        report.legacy.mean_planted_mass(),
        report.hybrid.mean_planted_mass(),
    );
    print_row(
        "softmax_entropy",
        report.legacy.mean_entropy(),
        report.hybrid.mean_entropy(),
    );
    print_row(
        "rff_softmax_correlation",
        report.legacy.mean_rff_correlation(),
        report.hybrid.mean_rff_correlation(),
    );
    print_row(
        "wall_ms",
        report.legacy.mean_wall_ms(),
        report.hybrid.mean_wall_ms(),
    );
    print_row(
        "minimizer_iterations",
        report.legacy.mean_iterations(),
        report.hybrid.mean_iterations(),
    );
    print_row(
        "wake_pass_rate",
        report.legacy.wake_pass_rate(),
        report.hybrid.wake_pass_rate(),
    );
    print_row(
        "sleep_accepted",
        report.legacy.mean_sleep_accepted(),
        report.hybrid.mean_sleep_accepted(),
    );
    print_row(
        "recall_coherence",
        report.legacy.mean_recall_coherence(),
        report.hybrid.mean_recall_coherence(),
    );
    print_row(
        "ctp_bias_norm",
        report.legacy.ctp_bias_norm_sum / report.legacy.trials.max(1) as f64,
        report.hybrid.ctp_bias_norm_sum / report.hybrid.trials.max(1) as f64,
    );
    println!();
    println!(
        "summary,handshake_top1_delta={:+.4},handshake_mrr_delta={:+.4},\
         wall_time_ratio={:.3},sleep_accepted_delta={:+.2},recall_coherence_delta={:+.4}",
        report.handshake_top1_delta(),
        report.handshake_mrr_delta(),
        report.wall_time_ratio(),
        report.sleep_accepted_delta(),
        report.recall_coherence_delta(),
    );
}

fn print_row(name: &str, legacy: f64, hybrid: f64) {
    let delta = hybrid - legacy;
    let ratio = hybrid / legacy.max(f64::EPSILON);
    println!("{name},{legacy:.6},{hybrid:.6},{delta:+.6},{ratio:.4}");
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
