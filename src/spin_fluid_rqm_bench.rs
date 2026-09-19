//! Bench: inferencia colapso→RQM (exp) vs RQM nativo directo (estilo `main`).
//!
//! Misma tarea: mapa identidad de 4 etiquetas. Mismo motor
//! `NativeThermoRqmEprSubstrate` y mismo tamaño térmico. La diferencia es
//! si la señal de entrada pasa por el colapso de spin o es el id crudo.

use crate::entanglement::EntanglementConfig;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use crate::spin_fluid_rqm_infer::{
    SpinFluidRqmInfer, FEATURE_END, NUM_LABELS,
};
use std::time::Instant;

const OBSERVER: ObserverId = ObserverId(0xBEE1);
const TRAIN_REPEATS: usize = 6;
const INFER_LOOPS: usize = 50; // 50 × 4 = 200 queries

#[derive(Clone, Copy, Debug, Default)]
pub struct BenchArm {
    pub name: &'static str,
    pub train_ms: f64,
    pub infer_ms: f64,
    pub infer_per_query_us: f64,
    pub accuracy: f64,
    pub relations: usize,
    pub queries: usize,
}

/// Nodos cue estilo main (entrada cruda, sin colapso). Distintos de 0..NUM_LABELS.
pub const MAIN_CUE_BASE: usize = FEATURE_END; // 96..99

fn shared_thermal() -> NativeThermoCdtConfig {
    NativeThermoCdtConfig {
        slices: 2,
        nodes_per_slice: (MAIN_CUE_BASE + NUM_LABELS).max(128),
        seed: 0xBEEF_CAFE,
        ..NativeThermoCdtConfig::default()
    }
}

fn shared_rqm() -> NativeThermoRqmConfig {
    let mut c = NativeThermoRqmConfig::default();
    c.max_candidates = NUM_LABELS * 4;
    c.thermal_steps_per_train = 1;
    c.thermal_steps_per_query = 2;
    c.thermal_activation_margin = 0.05;
    c
}

fn shared_epr() -> EntanglementConfig {
    EntanglementConfig {
        create_threshold: 0.4,
        max_syncs_per_step: 64,
        ..EntanglementConfig::default()
    }
}

fn pick_label(candidates: &[NativeCandidateScore]) -> Option<usize> {
    candidates
        .iter()
        .filter(|c| c.agent < NUM_LABELS)
        .max_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.agent.cmp(&b.agent)))
        .map(|c| c.agent)
}

/// Brazo estilo `main`: cue = nodo de entrada (≠ etiqueta), sin dinámica de spin.
fn bench_main_style_rqm() -> BenchArm {
    let mut rqm =
        NativeThermoRqmEprSubstrate::new(shared_thermal(), shared_rqm(), shared_epr());

    // warmup — RQM ignora source==target, así que cue y label son nodos distintos.
    for i in 0..NUM_LABELS {
        let cue = MAIN_CUE_BASE + i;
        rqm.train_observed_transition(OBSERVER, 0.0, &[cue], &[i], 0.95);
        let _ = rqm.query(OBSERVER, 0.0, &[cue]);
    }

    let t0 = Instant::now();
    for _ in 0..TRAIN_REPEATS {
        for i in 0..NUM_LABELS {
            let cue = MAIN_CUE_BASE + i;
            rqm.train_observed_transition(OBSERVER, 0.0, &[cue], &[i], 0.95);
        }
    }
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let mut ok = 0usize;
    let mut queries = 0usize;
    for _ in 0..INFER_LOOPS {
        for i in 0..NUM_LABELS {
            let cue = MAIN_CUE_BASE + i;
            let report = rqm.query(OBSERVER, 0.0, &[cue]);
            queries += 1;
            if pick_label(&report.candidates) == Some(i) {
                ok += 1;
            }
        }
    }
    let infer_ms = t1.elapsed().as_secs_f64() * 1e3;
    BenchArm {
        name: "main_rqm_direct",
        train_ms,
        infer_ms,
        infer_per_query_us: (infer_ms * 1e3) / queries as f64,
        accuracy: ok as f64 / queries as f64,
        relations: rqm.relation_count(),
        queries,
    }
}

/// Brazo experimento: colapso de spin → features → mismo RQM.
fn bench_spin_collapse_rqm() -> BenchArm {
    let mut eng = SpinFluidRqmInfer::new();
    // alinear configs térmicas con el brazo main
    eng.rqm = NativeThermoRqmEprSubstrate::new(shared_thermal(), shared_rqm(), shared_epr());

    // warmup
    let _ = eng.train_pair(0, 0, 0.95);
    let _ = eng.infer(0);

    let t0 = Instant::now();
    let _ = eng.train_identity_epoch(TRAIN_REPEATS);
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let mut ok = 0usize;
    let mut queries = 0usize;
    for _ in 0..INFER_LOOPS {
        for i in 0..NUM_LABELS {
            let r = eng.infer(i);
            queries += 1;
            if r.predicted == Some(i) {
                ok += 1;
            }
        }
    }
    let infer_ms = t1.elapsed().as_secs_f64() * 1e3;
    BenchArm {
        name: "exp_spin_collapse_rqm",
        train_ms,
        infer_ms,
        infer_per_query_us: (infer_ms * 1e3) / queries as f64,
        accuracy: ok as f64 / queries as f64,
        relations: eng.rqm.relation_count(),
        queries,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BenchCompare {
    pub main: BenchArm,
    pub spin: BenchArm,
    pub train_slowdown: f64,
    pub infer_slowdown: f64,
}

pub fn run_performance_compare() -> BenchCompare {
    let main = bench_main_style_rqm();
    let spin = bench_spin_collapse_rqm();
    BenchCompare {
        train_slowdown: if main.train_ms > 1e-9 {
            spin.train_ms / main.train_ms
        } else {
            f64::INFINITY
        },
        infer_slowdown: if main.infer_ms > 1e-9 {
            spin.infer_ms / main.infer_ms
        } else {
            f64::INFINITY
        },
        main,
        spin,
    }
}

pub fn format_compare(c: &BenchCompare) -> String {
    format!(
        "bench identity map labels={n} train_repeats={tr} infer_loops={il} (queries/brazo={q})\n\
         | brazo | train_ms | infer_ms | µs/query | accuracy | relations |\n\
         |---|---:|---:|---:|---:|---:|\n\
         | {m} | {mt:.3} | {mi:.3} | {mu:.1} | {ma:.3} | {mr} |\n\
         | {s} | {st:.3} | {si:.3} | {su:.1} | {sa:.3} | {sr} |\n\
         slowdown train×{ts:.2}  infer×{is:.2}\n",
        n = NUM_LABELS,
        tr = TRAIN_REPEATS,
        il = INFER_LOOPS,
        q = c.main.queries,
        m = c.main.name,
        mt = c.main.train_ms,
        mi = c.main.infer_ms,
        mu = c.main.infer_per_query_us,
        ma = c.main.accuracy,
        mr = c.main.relations,
        s = c.spin.name,
        st = c.spin.train_ms,
        si = c.spin.infer_ms,
        su = c.spin.infer_per_query_us,
        sa = c.spin.accuracy,
        sr = c.spin.relations,
        ts = c.train_slowdown,
        is = c.infer_slowdown,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn performance_spin_rqm_vs_main_rqm() {
        let c = run_performance_compare();
        let report = format_compare(&c);
        println!("{report}");
        // Ambos deben resolver la tarea; el spin es más caro (colapso 16³).
        assert!(
            c.main.accuracy >= 0.99,
            "main-style RQM should be exact on identity: {:?}",
            c.main
        );
        assert!(
            c.spin.accuracy >= 0.99,
            "spin→RQM should stay exact on identity: {:?}",
            c.spin
        );
        assert!(
            c.infer_slowdown > 1.0,
            "spin path should be slower on infer (includes collapse): {:?}",
            c
        );
    }
}
