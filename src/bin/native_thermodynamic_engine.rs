use cdt_rqm_epr::native_thermodynamic_engine::{
    run_consolidated_native_engine, EngineBenchmark, NativeEngineConfig, NativeSleepReport,
    DEFAULT_TRAINED_STATE,
};
use std::env;

fn main() {
    let state =
        env::var("NATIVE_THERMO_STATE").unwrap_or_else(|_| DEFAULT_TRAINED_STATE.to_string());
    let config = NativeEngineConfig {
        eval_repeats: env_usize("NATIVE_THERMO_EVAL_REPEATS", 24),
        sleep_attempts: env_usize("NATIVE_THERMO_SLEEP_ATTEMPTS", 8),
        sleep_replay_passes: env_usize("NATIVE_THERMO_SLEEP_REPLAY_PASSES", 2),
    };

    let report = match run_consolidated_native_engine(&state, config) {
        Ok(report) => report,
        Err(err) => {
            println!("Native thermodynamic engine");
            println!("loaded=false state={state} error={err}");
            return;
        }
    };

    println!("Native thermodynamic engine");
    println!(
        "loaded=true state={} repeats={} sleep_attempts={} sleep_replay_passes={}",
        state, config.eval_repeats, config.sleep_attempts, config.sleep_replay_passes
    );
    println!(
        "migration: legacy_relations={} imported_relations={} nodes={} imported_edges={} epr_links={}",
        report.migration.legacy_relations,
        report.migration.imported_relations,
        report.migration.nodes,
        report.migration.imported_edges,
        report.migration.epr_links
    );
    print_benchmark("previous_cdt_rqm_epr", report.previous);
    print_benchmark(
        "native_thermo_rqm_epr_before_sleep",
        report.native_before_sleep,
    );
    print_sleep("native_sleep", report.sleep);
    print_benchmark(
        "native_thermo_rqm_epr_after_sleep",
        report.native_after_sleep,
    );
    println!("decision: {}", report.decision.as_str());
}

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn print_benchmark(label: &str, benchmark: EngineBenchmark) {
    println!(
        "{}: accuracy={:.1}% leakage={:.1}% margin={:.3} dynamics={:.3} cases={} elapsed_ms={:.3} us_per_case={:.3} relations={} epr_links={} energy={:.3}",
        label,
        benchmark.metrics.accuracy() * 100.0,
        benchmark.metrics.leakage() * 100.0,
        benchmark.metrics.margin(),
        benchmark.metrics.dynamics(),
        benchmark.metrics.cases,
        benchmark.elapsed.as_secs_f64() * 1_000.0,
        benchmark.elapsed.as_secs_f64() * 1_000_000.0 / benchmark.metrics.cases.max(1) as f64,
        benchmark.relations,
        benchmark.epr_links,
        benchmark.energy
    );
}

fn print_sleep(label: &str, report: NativeSleepReport) {
    println!(
        "{}: attempts={} accepted={} accuracy={:.1}%->{:.1}% leakage={:.1}%->{:.1}% margin={:.3}->{:.3} dynamics={:.3}->{:.3} energy={:.3}->{:.3} epr_links={}->{}",
        label,
        report.attempts,
        report.accepted,
        report.before.accuracy() * 100.0,
        report.after.accuracy() * 100.0,
        report.before.leakage() * 100.0,
        report.after.leakage() * 100.0,
        report.before.margin(),
        report.after.margin(),
        report.before.dynamics(),
        report.after.dynamics(),
        report.before_energy,
        report.after_energy,
        report.before_epr_links,
        report.after_epr_links
    );
}
