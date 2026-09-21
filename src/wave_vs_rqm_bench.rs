//! Bench: `wave_predict_core` vs RQM nativo estilo `main`.
//!
//! Misma tarea: observación → elegir continuación correcta entre K candidatos.
//! Wave: interferencia analítica pasado×futuro.
//! Main: `NativeThermoRqmEprSubstrate` cue→label (sin spin).

use crate::entanglement::EntanglementConfig;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use crate::wave_predict_core::WavePredictCore;
use std::time::Instant;

const OBSERVER: ObserverId = ObserverId(0xA11E);
const N_LABELS: usize = 8;
const TRAIN_REPEATS: usize = 6;
const INFER_LOOPS: usize = 80;
const CUE_BASE: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct ArmReport {
    pub name: &'static str,
    pub train_ms: f64,
    pub infer_ms: f64,
    pub us_per_query: f64,
    pub accuracy: f64,
    pub queries: usize,
    pub p50_us: f64,
    pub p99_us: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn pick_label(cands: &[NativeCandidateScore]) -> Option<usize> {
    cands.iter()
        .filter(|c| c.agent < N_LABELS)
        .max_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.agent.cmp(&b.agent)))
        .map(|c| c.agent)
}

fn rqm_substrate() -> NativeThermoRqmEprSubstrate {
    let thermal = NativeThermoCdtConfig {
        slices: 2,
        nodes_per_slice: (CUE_BASE + N_LABELS).max(128),
        seed: 0xBEEF_01,
        ..NativeThermoCdtConfig::default()
    };
    let mut rqm = NativeThermoRqmConfig::default();
    rqm.max_candidates = N_LABELS * 2;
    rqm.thermal_steps_per_train = 1;
    rqm.thermal_steps_per_query = 2;
    rqm.thermal_activation_margin = 0.05;
    let epr = EntanglementConfig {
        create_threshold: 0.4,
        max_syncs_per_step: 64,
        ..EntanglementConfig::default()
    };
    NativeThermoRqmEprSubstrate::new(thermal, rqm, epr)
}

fn bench_main_rqm() -> ArmReport {
    let mut rqm = rqm_substrate();
    // warmup
    for i in 0..N_LABELS {
        rqm.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[i], 0.95);
        let _ = rqm.query(OBSERVER, 0.0, &[CUE_BASE + i]);
    }

    let t0 = Instant::now();
    for _ in 0..TRAIN_REPEATS {
        for i in 0..N_LABELS {
            rqm.train_observed_transition(OBSERVER, 0.0, &[CUE_BASE + i], &[i], 0.95);
        }
    }
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut lat = Vec::with_capacity(INFER_LOOPS * N_LABELS);
    let mut ok = 0usize;
    let t1 = Instant::now();
    for _ in 0..INFER_LOOPS {
        for i in 0..N_LABELS {
            let tq = Instant::now();
            let report = rqm.query(OBSERVER, 0.0, &[CUE_BASE + i]);
            lat.push(tq.elapsed().as_secs_f64() * 1e6);
            if pick_label(&report.candidates) == Some(i) {
                ok += 1;
            }
        }
    }
    let infer_ms = t1.elapsed().as_secs_f64() * 1e3;
    let queries = lat.len();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ArmReport {
        name: "main_rqm_direct",
        train_ms,
        infer_ms,
        us_per_query: (infer_ms * 1e3) / queries as f64,
        accuracy: ok as f64 / queries as f64,
        queries,
        p50_us: percentile(&lat, 50.0),
        p99_us: percentile(&lat, 99.0),
    }
}

fn bench_wave_predict() -> ArmReport {
    let mut core = WavePredictCore::new();
    let candidates: Vec<usize> = (0..N_LABELS).collect();
    // warmup (sin train pesado: la geometría de canal es el “modelo”)
    let _ = core.predict_from_observation(0, &candidates);

    let train_ms = 0.0; // sin entrenamiento relacional

    let mut lat = Vec::with_capacity(INFER_LOOPS * N_LABELS);
    let mut ok = 0usize;
    let t1 = Instant::now();
    for _ in 0..INFER_LOOPS {
        for i in 0..N_LABELS {
            let tq = Instant::now();
            let r = core.predict_from_observation(i, &candidates);
            lat.push(tq.elapsed().as_secs_f64() * 1e6);
            if r.best_content == i {
                ok += 1;
            }
        }
    }
    let infer_ms = t1.elapsed().as_secs_f64() * 1e3;
    let queries = lat.len();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ArmReport {
        name: "wave_predict_core",
        train_ms,
        infer_ms,
        us_per_query: (infer_ms * 1e3) / queries as f64,
        accuracy: ok as f64 / queries as f64,
        queries,
        p50_us: percentile(&lat, 50.0),
        p99_us: percentile(&lat, 99.0),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompareReport {
    pub main: ArmReport,
    pub wave: ArmReport,
    pub infer_speedup_wave_over_main: f64,
}

pub fn run_wave_vs_rqm() -> CompareReport {
    let main = bench_main_rqm();
    let wave = bench_wave_predict();
    let infer_speedup_wave_over_main = if wave.us_per_query > 1e-12 {
        main.us_per_query / wave.us_per_query
    } else {
        f64::INFINITY
    };
    CompareReport {
        main,
        wave,
        infer_speedup_wave_over_main,
    }
}

pub fn format_compare(c: &CompareReport) -> String {
    format!(
        "wave vs main RQM | labels={n} train_repeats={tr} infer_loops={il} queries/brazo={q}\n\
         | brazo | train_ms | infer_ms | µs/q | p50 µs | p99 µs | accuracy |\n\
         |---|---:|---:|---:|---:|---:|---:|\n\
         | {m} | {mt:.3} | {mi:.3} | {mu:.3} | {mp50:.3} | {mp99:.3} | {ma:.3} |\n\
         | {w} | {wt:.3} | {wi:.3} | {wu:.3} | {wp50:.3} | {wp99:.3} | {wa:.3} |\n\
         speedup infer wave/main ×{sp:.2}\n",
        n = N_LABELS,
        tr = TRAIN_REPEATS,
        il = INFER_LOOPS,
        q = c.main.queries,
        m = c.main.name,
        mt = c.main.train_ms,
        mi = c.main.infer_ms,
        mu = c.main.us_per_query,
        mp50 = c.main.p50_us,
        mp99 = c.main.p99_us,
        ma = c.main.accuracy,
        w = c.wave.name,
        wt = c.wave.train_ms,
        wi = c.wave.infer_ms,
        wu = c.wave.us_per_query,
        wp50 = c.wave.p50_us,
        wp99 = c.wave.p99_us,
        wa = c.wave.accuracy,
        sp = c.infer_speedup_wave_over_main,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_predict_vs_main_rqm_head_to_head() {
        let c = run_wave_vs_rqm();
        let report = format_compare(&c);
        println!("{report}");
        assert!(
            c.main.accuracy >= 0.99,
            "main RQM should solve identity: {:?}",
            c.main
        );
        assert!(
            c.wave.accuracy >= 0.99,
            "wave predict should solve continuation: {:?}",
            c.wave
        );
        assert!(
            c.wave.us_per_query < c.main.us_per_query,
            "wave analytic should be faster than RQM query: {:?}",
            c
        );
    }
}
