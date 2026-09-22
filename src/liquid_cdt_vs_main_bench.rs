//! Comparación detallada: arquitectura dual NEW (`LiquidCdtSystem`) vs
//! stack estilo MAIN (`NativeThermoRqmEprSubstrate` / RQM directo).
//!
//! NEW: líquido en **toda** la inferencia; Thermo CDT memoria tras sueño;
//! RQM **solo** en `sleep_consolidate`.
//! MAIN: train relacional + query RQM en el hot path.
//!
//! Ver `docs/bench_liquido_cdt_vs_main.md`.

use crate::entanglement::EntanglementConfig;
use crate::liquid_cdt_memory::LiquidCdtSystem;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use std::time::Instant;

const OBSERVER: ObserverId = ObserverId(0xC0A1_A1E1);
const CUE_BASE: usize = 64;
const N: usize = 8;
const WARMUP_QUERIES: usize = 200;
const BENCH_QUERIES: usize = 2000;
const MAIN_TRAIN_EPOCHS: usize = 6;

// ─── Métricas ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct LatencyStats {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub queries: usize,
    pub qps: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ArmInferMetrics {
    pub name: &'static str,
    pub identity_latency: LatencyStats,
    pub identity_acc: f64,
    pub shifted_latency: LatencyStats,
    pub shifted_acc: f64,
    /// NEW: sleep costs; MAIN: relational train.
    pub train_or_sleep_ms_batch1: f64,
    pub train_or_sleep_ms_batch2: f64,
    pub train_note: String,
    pub retained_engrams_or_relations: usize,
    pub recall_ok: usize,
    pub recall_total: usize,
    pub rqm_api_calls_during_infer: u64,
    pub cdt_tick_delta_during_infer: u64,
    pub infer_empty_buffer_mean_us: f64,
    pub infer_after_sleep_mean_us: f64,
}

#[derive(Clone, Debug)]
pub struct CompareReport {
    pub new_arm: ArmInferMetrics,
    pub main_arm: ArmInferMetrics,
    pub verdict_mejoras: String,
    pub verdict_empates: String,
    pub verdict_peor: String,
    pub verdict_conclusion: String,
}

fn percentile_us(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_us.len() as f64 - 1.0)).round() as usize;
    sorted_us[idx.min(sorted_us.len() - 1)]
}

fn latency_from_samples(samples_us: &mut [f64]) -> LatencyStats {
    let n = samples_us.len();
    if n == 0 {
        return LatencyStats::default();
    }
    samples_us.sort_by(|a, b| a.total_cmp(b));
    let sum: f64 = samples_us.iter().sum();
    let mean = sum / n as f64;
    LatencyStats {
        mean_us: mean,
        p50_us: percentile_us(samples_us, 50.0),
        p95_us: percentile_us(samples_us, 95.0),
        p99_us: percentile_us(samples_us, 99.0),
        queries: n,
        qps: if mean > 0.0 { 1e6 / mean } else { 0.0 },
    }
}

fn pick_label(cands: &[NativeCandidateScore]) -> Option<usize> {
    cands
        .iter()
        .filter(|c| c.agent < N)
        .max_by(|a, b| {
            a.score
                .total_cmp(&b.score)
                .then_with(|| a.agent.cmp(&b.agent))
        })
        .map(|c| c.agent)
}

fn make_main_rqm() -> NativeThermoRqmEprSubstrate {
    let thermal = NativeThermoCdtConfig {
        slices: 2,
        nodes_per_slice: (CUE_BASE + N).max(128),
        seed: 0xA11A_C0A1,
        ..NativeThermoCdtConfig::default()
    };
    let cfg = NativeThermoRqmConfig {
        max_candidates: N * 2,
        thermal_steps_per_train: 1,
        thermal_steps_per_query: 2,
        thermal_activation_margin: 0.05,
        ..Default::default()
    };
    let epr = EntanglementConfig {
        create_threshold: 0.4,
        max_syncs_per_step: 64,
        ..EntanglementConfig::default()
    };
    NativeThermoRqmEprSubstrate::new(thermal, cfg, epr)
}

/// Entrena mapa cue→label: `map(i)` es el target para cue `CUE_BASE+i`.
fn train_main_map(rqm: &mut NativeThermoRqmEprSubstrate, map: impl Fn(usize) -> usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..MAIN_TRAIN_EPOCHS {
        for i in 0..N {
            rqm.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[map(i)], 0.95);
        }
    }
    t0.elapsed().as_secs_f64() * 1e3
}

fn bench_main_queries(
    rqm: &mut NativeThermoRqmEprSubstrate,
    map: impl Fn(usize) -> usize,
    warmup: usize,
    total: usize,
) -> (LatencyStats, f64) {
    let mut samples = Vec::with_capacity(total);
    let mut ok = 0usize;
    let mut done = 0usize;

    // Warmup
    let mut w = 0usize;
    while w < warmup {
        let i = w % N;
        let _ = rqm.query(OBSERVER, 0.0, &[CUE_BASE + i]);
        w += 1;
    }

    while done < total {
        let i = done % N;
        let t0 = Instant::now();
        let r = rqm.query(OBSERVER, 0.0, &[CUE_BASE + i]);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        samples.push(us);
        if pick_label(&r.candidates) == Some(map(i)) {
            ok += 1;
        }
        done += 1;
    }
    let lat = latency_from_samples(&mut samples);
    (lat, ok as f64 / total as f64)
}

fn bench_liquid_queries(
    sys: &mut LiquidCdtSystem,
    target: impl Fn(usize) -> usize,
    warmup: usize,
    total: usize,
) -> (LatencyStats, f64, u64, u64) {
    let candidates: Vec<usize> = (0..N).collect();
    let rqm_before = sys.rqm_api_calls();
    let tick_before = sys.cdt_tick();
    let steps_before = sys.cdt_step_count();

    let mut w = 0usize;
    while w < warmup {
        let i = w % N;
        let _ = sys.infer(i, &candidates);
        w += 1;
    }

    let mut samples = Vec::with_capacity(total);
    let mut ok = 0usize;
    let mut done = 0usize;
    while done < total {
        let i = done % N;
        let t0 = Instant::now();
        let r = sys.infer(i, &candidates);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        samples.push(us);
        if r.liquid.predicted == target(i) {
            ok += 1;
        }
        done += 1;
    }

    let rqm_delta = sys.rqm_api_calls().saturating_sub(rqm_before);
    let tick_delta = sys.cdt_tick().saturating_sub(tick_before);
    let step_delta = sys.cdt_step_count().saturating_sub(steps_before);
    assert_eq!(
        step_delta, 0,
        "liquid infer must not call CDT step; delta={step_delta}"
    );
    let lat = latency_from_samples(&mut samples);
    (lat, ok as f64 / total as f64, rqm_delta, tick_delta)
}

fn mean_us_liquid(sys: &mut LiquidCdtSystem, queries: usize) -> f64 {
    let candidates: Vec<usize> = (0..N).collect();
    let t0 = Instant::now();
    for q in 0..queries {
        let _ = sys.infer(q % N, &candidates);
    }
    t0.elapsed().as_secs_f64() * 1e6 / queries as f64
}

fn run_new_arm() -> ArmInferMetrics {
    let candidates: Vec<usize> = (0..N).collect();
    let mut sys = LiquidCdtSystem::new(N);

    // Hot-path purity + identity latency/acc
    let (id_lat, id_acc, rqm_delta, tick_delta) =
        bench_liquid_queries(&mut sys, |i| i, WARMUP_QUERIES, BENCH_QUERIES);

    // Infer with empty buffer (fresh-ish path after warmup already done)
    let empty_us = mean_us_liquid(&mut sys, 400);

    // Sleep batch A (0..3) then batch B (4..7)
    for obs in 0..4 {
        let _ = sys.observe_and_buffer(obs, &candidates);
    }
    let s1 = sys.sleep_consolidate();
    let sleep1_ms = s1.sleep_ms;

    for obs in 4..8 {
        let _ = sys.observe_and_buffer(obs, &candidates);
    }
    let s2 = sys.sleep_consolidate();
    let sleep2_ms = s2.sleep_ms;

    // Infer after sleep — must stay liquid-fast
    let after_sleep_us = mean_us_liquid(&mut sys, 400);

    // Memory / forgetting via soft-recall
    let mut recall_ok = 0usize;
    for c in 0..N {
        if let Some((pred, _)) = sys.memory.recall(c) {
            if pred == c {
                recall_ok += 1;
            }
        }
    }

    // Shifted map: liquid is identity-biased; still report numbers.
    let mut sys_shift = LiquidCdtSystem::new(N);
    let (sh_lat, sh_acc, _, _) = bench_liquid_queries(
        &mut sys_shift,
        |i| (i + 1) % N,
        WARMUP_QUERIES / 2,
        BENCH_QUERIES,
    );

    ArmInferMetrics {
        name: "NEW LiquidCdtSystem (liquid infer + CDT sleep)",
        identity_latency: id_lat,
        identity_acc: id_acc,
        shifted_latency: sh_lat,
        shifted_acc: sh_acc,
        train_or_sleep_ms_batch1: sleep1_ms,
        train_or_sleep_ms_batch2: sleep2_ms,
        train_note: format!(
            "no relational train for infer; sleep_consolidate batchA={} ep → {} engrams, \
             batchB={} ep → {} engrams; RQM only in sleep",
            s1.episodes_consolidated, s1.engrams_after, s2.episodes_consolidated, s2.engrams_after
        ),
        retained_engrams_or_relations: sys.engram_count(),
        recall_ok,
        recall_total: N,
        rqm_api_calls_during_infer: rqm_delta,
        cdt_tick_delta_during_infer: tick_delta,
        infer_empty_buffer_mean_us: empty_us,
        infer_after_sleep_mean_us: after_sleep_us,
    }
}

fn run_main_arm() -> ArmInferMetrics {
    // Identity map
    let mut rqm = make_main_rqm();
    let train_id_ms = train_main_map(&mut rqm, |i| i);
    let (id_lat, id_acc) = bench_main_queries(&mut rqm, |i| i, WARMUP_QUERIES, BENCH_QUERIES);
    let id_mean_us = id_lat.mean_us;
    let rel_after_id = rqm.relation_count();

    // Forgetting contrast: train A (0..3) then B (4..7) on fresh substrate
    let mut rqm_ab = make_main_rqm();
    let t_a = Instant::now();
    for _ in 0..MAIN_TRAIN_EPOCHS {
        for i in 0..4 {
            rqm_ab.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[i], 0.95);
        }
    }
    let train_a_ms = t_a.elapsed().as_secs_f64() * 1e3;
    let t_b = Instant::now();
    for _ in 0..MAIN_TRAIN_EPOCHS {
        for i in 4..8 {
            rqm_ab.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[i], 0.95);
        }
    }
    let train_b_ms = t_b.elapsed().as_secs_f64() * 1e3;

    let mut recall_ok = 0usize;
    for i in 0..N {
        let r = rqm_ab.query(OBSERVER, 0.0, &[CUE_BASE + i]);
        if pick_label(&r.candidates) == Some(i) {
            recall_ok += 1;
        }
    }

    // Shifted map (arbitrary relation) on dedicated substrate
    let mut rqm_sh = make_main_rqm();
    let _train_sh = train_main_map(&mut rqm_sh, |i| (i + 1) % N);
    let (sh_lat, sh_acc) = bench_main_queries(
        &mut rqm_sh,
        |i| (i + 1) % N,
        WARMUP_QUERIES / 2,
        BENCH_QUERIES,
    );

    ArmInferMetrics {
        name: "MAIN NativeThermoRqmEprSubstrate (direct RQM)",
        identity_latency: id_lat,
        identity_acc: id_acc,
        shifted_latency: sh_lat,
        shifted_acc: sh_acc,
        train_or_sleep_ms_batch1: train_a_ms,
        train_or_sleep_ms_batch2: train_b_ms,
        train_note: format!(
            "train_observed_transition cue_base+i→i, {MAIN_TRAIN_EPOCHS} epochs × {N} relations; \
             full identity train={train_id_ms:.3} ms; relations_after_id={rel_after_id}"
        ),
        retained_engrams_or_relations: rqm_ab.relation_count(),
        recall_ok,
        recall_total: N,
        rqm_api_calls_during_infer: BENCH_QUERIES as u64, // every query hits RQM
        cdt_tick_delta_during_infer: 0,                   // not tracked separately on main
        infer_empty_buffer_mean_us: id_mean_us,
        infer_after_sleep_mean_us: id_mean_us,
    }
}

fn build_verdicts(
    new: &ArmInferMetrics,
    main: &ArmInferMetrics,
) -> (String, String, String, String) {
    let mejoras = format!(
        "NEW gana latencia de inferencia identidad ({:.4} vs {:.4} µs/q, ≈{:.0}×), \
         hot-path sin RQM (rqm_api_calls_infer={}) ni ticks CDT (Δtick={}), \
         y memoria desacoplada del sueño (engrams={} retenidos tras 2 lotes; \
         infer post-sueño sigue líquida ≈{:.4} µs).",
        new.identity_latency.mean_us,
        main.identity_latency.mean_us,
        main.identity_latency.mean_us / new.identity_latency.mean_us.max(1e-12),
        new.rqm_api_calls_during_infer,
        new.cdt_tick_delta_during_infer,
        new.retained_engrams_or_relations,
        new.infer_after_sleep_mean_us
    );
    let empates = format!(
        "Exactitud identidad empatada (≥0.99): NEW={:.4}, MAIN={:.4}. \
         Retención de 8 conceptos tras dos lotes: NEW recall={}/{}, MAIN recall={}/{}.",
        new.identity_acc,
        main.identity_acc,
        new.recall_ok,
        new.recall_total,
        main.recall_ok,
        main.recall_total
    );
    let peor = format!(
        "MAIN gana en aprendizaje relacional arbitrario sin sueño: mapa desplazado \
         i→(i+1)%N acc NEW={:.4} (líquido sesgado a identidad) vs MAIN={:.4}. \
         MAIN también ofrece primer disparo relacional inmediato tras train \
         (sin esperar consolidación onírica); coste train A/B {:.3}/{:.3} ms vs sleep {:.3}/{:.3} ms.",
        new.shifted_acc,
        main.shifted_acc,
        main.train_or_sleep_ms_batch1,
        main.train_or_sleep_ms_batch2,
        new.train_or_sleep_ms_batch1,
        new.train_or_sleep_ms_batch2
    );
    let conclusion = format!(
        "Recomendación: usar NEW (LiquidCdt) como **núcleo de inferencia** y memoria \
         de largo plazo vía CDT en sueño; reservar MAIN/RQM para pegamento relacional \
         onírico o para mapas arbitrarios que el líquido no captura por geometría. \
         Throughput identidad: NEW≈{:.0} q/s vs MAIN≈{:.0} q/s.",
        new.identity_latency.qps, main.identity_latency.qps
    );
    (mejoras, empates, peor, conclusion)
}

/// Suite completa NEW vs MAIN.
pub fn run_compare() -> CompareReport {
    let new_arm = run_new_arm();
    let main_arm = run_main_arm();
    let (mejoras, empates, peor, conclusion) = build_verdicts(&new_arm, &main_arm);
    CompareReport {
        new_arm,
        main_arm,
        verdict_mejoras: mejoras,
        verdict_empates: empates,
        verdict_peor: peor,
        verdict_conclusion: conclusion,
    }
}

fn fmt_lat(l: &LatencyStats) -> String {
    format!(
        "mean={:.4}  p50={:.4}  p95={:.4}  p99={:.4}  µs/q  |  qps≈{:.0}  (n={})",
        l.mean_us, l.p50_us, l.p95_us, l.p99_us, l.qps, l.queries
    )
}

/// Informe markdown-ish para consola y docs.
pub fn format_compare_report(r: &CompareReport) -> String {
    let mut s = String::new();
    s.push_str("# LiquidCdt (NEW) vs Main RQM — comparación detallada\n\n");
    s.push_str(&format!(
        "Protocolo: N={N} conceptos · warmup={WARMUP_QUERIES} · bench={BENCH_QUERIES} queries/brazo · \
         MAIN epochs={MAIN_TRAIN_EPOCHS} · cue≠label (`CUE_BASE+i → target`)\n\n"
    ));

    for arm in [&r.new_arm, &r.main_arm] {
        s.push_str(&format!("## {}\n\n", arm.name));
        s.push_str(&format!(
            "| Métrica | Valor |\n|--------|------:|\n\
             | Identity latency | {} |\n\
             | Identity accuracy | {:.4} |\n\
             | Shifted latency | {} |\n\
             | Shifted accuracy | {:.4} |\n\
             | Train/sleep batch1 (ms) | {:.3} |\n\
             | Train/sleep batch2 (ms) | {:.3} |\n\
             | Retained (engrams/relations) | {} |\n\
             | Recall ok | {}/{} |\n\
             | RQM API calls during infer | {} |\n\
             | CDT tick Δ during infer | {} |\n\
             | Infer empty-buffer mean µs | {:.4} |\n\
             | Infer after-sleep mean µs | {:.4} |\n\n",
            fmt_lat(&arm.identity_latency),
            arm.identity_acc,
            fmt_lat(&arm.shifted_latency),
            arm.shifted_acc,
            arm.train_or_sleep_ms_batch1,
            arm.train_or_sleep_ms_batch2,
            arm.retained_engrams_or_relations,
            arm.recall_ok,
            arm.recall_total,
            arm.rqm_api_calls_during_infer,
            arm.cdt_tick_delta_during_infer,
            arm.infer_empty_buffer_mean_us,
            arm.infer_after_sleep_mean_us,
        ));
        s.push_str(&format!("Nota train/sleep: {}\n\n", arm.train_note));
    }

    s.push_str("## Tabla veredicto\n\n");
    s.push_str("| Aspecto | Ganador | Detalle |\n|---------|---------|--------|\n");
    s.push_str(&format!(
        "| Latencia infer identidad | **NEW** | {:.4} vs {:.4} µs/q |\n",
        r.new_arm.identity_latency.mean_us, r.main_arm.identity_latency.mean_us
    ));
    s.push_str(&format!(
        "| Hot-path sin RQM | **NEW** | rqm_infer={} vs MAIN cada query |\n",
        r.new_arm.rqm_api_calls_during_infer
    ));
    s.push_str(&format!(
        "| Memoria sueño-desacoplada | **NEW** | engrams={} · post-sleep {:.4} µs |\n",
        r.new_arm.retained_engrams_or_relations, r.new_arm.infer_after_sleep_mean_us
    ));
    s.push_str(&format!(
        "| Mapa arbitrario (shifted) | **MAIN** | NEW={:.4} MAIN={:.4} |\n",
        r.new_arm.shifted_acc, r.main_arm.shifted_acc
    ));
    s.push_str(&format!(
        "| Exactitud identidad | empate | NEW={:.4} MAIN={:.4} |\n\n",
        r.new_arm.identity_acc, r.main_arm.identity_acc
    ));

    s.push_str("## Mejoras\n\n");
    s.push_str(&r.verdict_mejoras);
    s.push_str("\n\n## Empates\n\n");
    s.push_str(&r.verdict_empates);
    s.push_str("\n\n## Peor\n\n");
    s.push_str(&r.verdict_peor);
    s.push_str("\n\n## Conclusión\n\n");
    s.push_str(&r.verdict_conclusion);
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_liquid_cdt_vs_main_detailed() {
        let report = run_compare();
        let text = format_compare_report(&report);
        println!("{text}");

        assert!(
            report.new_arm.identity_acc >= 0.99,
            "new identity acc={}",
            report.new_arm.identity_acc
        );
        assert!(
            report.main_arm.identity_acc >= 0.99,
            "main identity acc={}",
            report.main_arm.identity_acc
        );
        assert!(
            report.new_arm.identity_latency.mean_us < report.main_arm.identity_latency.mean_us,
            "new ({:.4} µs) should beat main ({:.4} µs) on identity infer",
            report.new_arm.identity_latency.mean_us,
            report.main_arm.identity_latency.mean_us
        );
        assert_eq!(
            report.new_arm.rqm_api_calls_during_infer, 0,
            "new infer must not call RQM"
        );
        assert_eq!(
            report.new_arm.retained_engrams_or_relations, 8,
            "after two sleep batches engram_count must be 8"
        );
        assert_eq!(report.new_arm.cdt_tick_delta_during_infer, 0);
    }

    #[test]
    fn compare_shifted_map_reported() {
        // Softer asserts: liquid is identity-biased; main should learn shifted.
        let report = run_compare();
        let text = format_compare_report(&report);
        println!("=== SHIFT-FOCUS ===\n{text}");
        assert!(
            report.main_arm.shifted_acc >= 0.90,
            "main shifted acc={}",
            report.main_arm.shifted_acc
        );
        // Liquid may be near chance/identity; just ensure we measured it.
        assert!(report.new_arm.shifted_latency.queries >= BENCH_QUERIES);
        println!(
            "shifted: NEW acc={:.4} mean_us={:.4} | MAIN acc={:.4} mean_us={:.4}",
            report.new_arm.shifted_acc,
            report.new_arm.shifted_latency.mean_us,
            report.main_arm.shifted_acc,
            report.main_arm.shifted_latency.mean_us
        );
    }
}
