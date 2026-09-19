//! Bench detallado: colapso→RQM (exp) vs RQM nativo directo (estilo `main`).
//!
//! Misma tarea base: mapa de 4 etiquetas. Mismo `NativeThermoRqmEprSubstrate`
//! y mismo tamaño térmico. Reporta desglose de tiempos, curva de train,
//! latencias, confusión y escalado con pasos de colapso.

use crate::entanglement::EntanglementConfig;
use crate::native_thermo_rqm_epr::{
    NativeCandidateScore, NativeThermoRqmConfig, NativeThermoRqmEprSubstrate,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use crate::spin_fluid_rqm_infer::{SpinFluidRqmInfer, FEATURE_END, NUM_LABELS};
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

fn new_spin_engine() -> SpinFluidRqmInfer {
    let mut eng = SpinFluidRqmInfer::new();
    eng.rqm = NativeThermoRqmEprSubstrate::new(shared_thermal(), shared_rqm(), shared_epr());
    eng
}

fn pick_label(candidates: &[NativeCandidateScore]) -> Option<usize> {
    candidates
        .iter()
        .filter(|c| c.agent < NUM_LABELS)
        .max_by(|a, b| a.score.total_cmp(&b.score).then_with(|| a.agent.cmp(&b.agent)))
        .map(|c| c.agent)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Brazo estilo `main`: cue = nodo de entrada (≠ etiqueta), sin dinámica de spin.
fn bench_main_style_rqm() -> BenchArm {
    let mut rqm = NativeThermoRqmEprSubstrate::new(shared_thermal(), shared_rqm(), shared_epr());

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
    let mut eng = new_spin_engine();
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

// ─── detalle ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseBreakdown {
    pub encode_collapse_us: f64,
    pub extract_features_us: f64,
    pub rqm_query_us: f64,
    pub rqm_train_us: f64,
    pub total_infer_us: f64,
    pub collapse_fraction: f64,
    pub samples: usize,
}

/// Cronometra fases internas del pipeline spin (sin contar setup del motor).
pub fn bench_spin_phase_breakdown(samples: usize) -> PhaseBreakdown {
    let mut eng = new_spin_engine();
    let _ = eng.train_identity_epoch(2);

    let mut enc = 0.0;
    let mut feat = 0.0;
    let mut qry = 0.0;
    let mut trn = 0.0;
    let mut tot = 0.0;

    for s in 0..samples {
        let input = s % NUM_LABELS;
        let t_all = Instant::now();

        let t0 = Instant::now();
        let mut spin = eng.encode_input(input);
        eng.collapse(&mut spin);
        enc += t0.elapsed().as_secs_f64() * 1e6;

        let t1 = Instant::now();
        let features = eng.extract_features(&spin);
        let nodes = SpinFluidRqmInfer::feature_nodes(&features);
        feat += t1.elapsed().as_secs_f64() * 1e6;

        let phase = features.phase as f32;
        let t2 = Instant::now();
        let _ = eng.rqm.query(OBSERVER, phase, &nodes);
        qry += t2.elapsed().as_secs_f64() * 1e6;

        let t3 = Instant::now();
        eng.rqm
            .train_observed_transition(OBSERVER, phase, &nodes, &[input], 0.5);
        trn += t3.elapsed().as_secs_f64() * 1e6;

        tot += t_all.elapsed().as_secs_f64() * 1e6;
    }

    let n = samples as f64;
    let encode_collapse_us = enc / n;
    let extract_features_us = feat / n;
    let rqm_query_us = qry / n;
    let rqm_train_us = trn / n;
    let total_infer_us = tot / n;
    let collapse_fraction = if total_infer_us > 1e-9 {
        encode_collapse_us / total_infer_us
    } else {
        0.0
    };
    PhaseBreakdown {
        encode_collapse_us,
        extract_features_us,
        rqm_query_us,
        rqm_train_us,
        total_infer_us,
        collapse_fraction,
        samples,
    }
}

#[derive(Clone, Debug)]
pub struct LatencyReport {
    pub arm: &'static str,
    pub n: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
}

pub fn bench_infer_latency_main(samples: usize) -> LatencyReport {
    let mut rqm = NativeThermoRqmEprSubstrate::new(shared_thermal(), shared_rqm(), shared_epr());
    for i in 0..NUM_LABELS {
        rqm.train_observed_transition(OBSERVER, 0.0, &[MAIN_CUE_BASE + i], &[i], 0.95);
    }
    let mut xs = Vec::with_capacity(samples);
    for s in 0..samples {
        let i = s % NUM_LABELS;
        let t0 = Instant::now();
        let _ = rqm.query(OBSERVER, 0.0, &[MAIN_CUE_BASE + i]);
        xs.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    summarize_latency("main_rqm_direct", xs)
}

pub fn bench_infer_latency_spin(samples: usize) -> LatencyReport {
    let mut eng = new_spin_engine();
    let _ = eng.train_identity_epoch(TRAIN_REPEATS);
    let mut xs = Vec::with_capacity(samples);
    for s in 0..samples {
        let i = s % NUM_LABELS;
        let t0 = Instant::now();
        let _ = eng.infer(i);
        xs.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    summarize_latency("exp_spin_collapse_rqm", xs)
}

fn summarize_latency(arm: &'static str, mut xs: Vec<f64>) -> LatencyReport {
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n as f64;
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    LatencyReport {
        arm,
        n,
        mean_us: mean,
        p50_us: percentile(&xs, 50.0),
        p90_us: percentile(&xs, 90.0),
        p99_us: percentile(&xs, 99.0),
        min_us: xs[0],
        max_us: xs[n - 1],
    }
}

#[derive(Clone, Debug)]
pub struct TrainCurvePoint {
    pub epoch: usize,
    pub accuracy: f64,
    pub relations: usize,
    pub epoch_ms: f64,
}

pub fn bench_train_curve(max_epochs: usize) -> Vec<TrainCurvePoint> {
    let mut eng = new_spin_engine();
    let mut out = Vec::with_capacity(max_epochs + 1);
    out.push(TrainCurvePoint {
        epoch: 0,
        accuracy: eng.identity_accuracy(),
        relations: eng.rqm.relation_count(),
        epoch_ms: 0.0,
    });
    for e in 1..=max_epochs {
        let t0 = Instant::now();
        let _ = eng.train_identity_epoch(1);
        let epoch_ms = t0.elapsed().as_secs_f64() * 1e3;
        out.push(TrainCurvePoint {
            epoch: e,
            accuracy: eng.identity_accuracy(),
            relations: eng.rqm.relation_count(),
            epoch_ms,
        });
    }
    out
}

#[derive(Clone, Debug)]
pub struct ConfusionReport {
    pub arm: &'static str,
    /// rows = true label, cols = predicted (None bucket as NUM_LABELS)
    pub matrix: Vec<Vec<usize>>,
    pub accuracy: f64,
    pub per_label_recall: Vec<f64>,
}

fn empty_confusion() -> Vec<Vec<usize>> {
    vec![vec![0usize; NUM_LABELS + 1]; NUM_LABELS]
}

pub fn bench_confusion_main(repeats: usize) -> ConfusionReport {
    let mut rqm = NativeThermoRqmEprSubstrate::new(shared_thermal(), shared_rqm(), shared_epr());
    for _ in 0..TRAIN_REPEATS {
        for i in 0..NUM_LABELS {
            rqm.train_observed_transition(OBSERVER, 0.0, &[MAIN_CUE_BASE + i], &[i], 0.95);
        }
    }
    let mut matrix = empty_confusion();
    let mut ok = 0usize;
    let mut total = 0usize;
    for _ in 0..repeats {
        for i in 0..NUM_LABELS {
            let pred = pick_label(
                &rqm
                    .query(OBSERVER, 0.0, &[MAIN_CUE_BASE + i])
                    .candidates,
            );
            let col = pred.unwrap_or(NUM_LABELS);
            matrix[i][col] += 1;
            total += 1;
            if pred == Some(i) {
                ok += 1;
            }
        }
    }
    let mut recall = vec![0.0; NUM_LABELS];
    for i in 0..NUM_LABELS {
        let row_sum: usize = matrix[i].iter().sum();
        recall[i] = if row_sum > 0 {
            matrix[i][i] as f64 / row_sum as f64
        } else {
            0.0
        };
    }
    ConfusionReport {
        arm: "main_rqm_direct",
        matrix,
        accuracy: ok as f64 / total as f64,
        per_label_recall: recall,
    }
}

pub fn bench_confusion_spin(repeats: usize) -> ConfusionReport {
    let mut eng = new_spin_engine();
    let _ = eng.train_identity_epoch(TRAIN_REPEATS);
    let mut matrix = empty_confusion();
    let mut ok = 0usize;
    let mut total = 0usize;
    for _ in 0..repeats {
        for i in 0..NUM_LABELS {
            let r = eng.infer(i);
            let col = r.predicted.unwrap_or(NUM_LABELS);
            matrix[i][col] += 1;
            total += 1;
            if r.predicted == Some(i) {
                ok += 1;
            }
        }
    }
    let mut recall = vec![0.0; NUM_LABELS];
    for i in 0..NUM_LABELS {
        let row_sum: usize = matrix[i].iter().sum();
        recall[i] = if row_sum > 0 {
            matrix[i][i] as f64 / row_sum as f64
        } else {
            0.0
        };
    }
    ConfusionReport {
        arm: "exp_spin_collapse_rqm",
        matrix,
        accuracy: ok as f64 / total as f64,
        per_label_recall: recall,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CollapseScalePoint {
    pub steps: usize,
    pub infer_us: f64,
    pub accuracy: f64,
    pub amp_mean: f64,
}

pub fn bench_collapse_steps_scaling(step_grid: &[usize], infer_repeats: usize) -> Vec<CollapseScalePoint> {
    let mut out = Vec::new();
    for &steps in step_grid {
        let mut eng = new_spin_engine();
        eng.collapse_steps = steps;
        let _ = eng.train_identity_epoch(TRAIN_REPEATS);
        let mut ok = 0usize;
        let mut amp = 0.0;
        let t0 = Instant::now();
        let n = infer_repeats * NUM_LABELS;
        for _ in 0..infer_repeats {
            for i in 0..NUM_LABELS {
                let r = eng.infer(i);
                amp += r.features.max_amp;
                if r.predicted == Some(i) {
                    ok += 1;
                }
            }
        }
        let infer_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        out.push(CollapseScalePoint {
            steps,
            infer_us,
            accuracy: ok as f64 / n as f64,
            amp_mean: amp / n as f64,
        });
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct ShiftedMapBench {
    pub train_ms: f64,
    pub infer_ms: f64,
    pub accuracy: f64,
    pub relations: usize,
}

pub fn bench_shifted_map(train_repeats: usize, infer_loops: usize) -> ShiftedMapBench {
    let mut eng = new_spin_engine();
    let t0 = Instant::now();
    for _ in 0..train_repeats {
        for i in 0..NUM_LABELS {
            let _ = eng.train_pair(i, (i + 1) % NUM_LABELS, 0.95);
        }
    }
    let train_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let mut ok = 0usize;
    let mut q = 0usize;
    for _ in 0..infer_loops {
        for i in 0..NUM_LABELS {
            let r = eng.infer(i);
            q += 1;
            if r.predicted == Some((i + 1) % NUM_LABELS) {
                ok += 1;
            }
        }
    }
    let infer_ms = t1.elapsed().as_secs_f64() * 1e3;
    ShiftedMapBench {
        train_ms,
        infer_ms,
        accuracy: ok as f64 / q as f64,
        relations: eng.rqm.relation_count(),
    }
}

fn format_confusion(c: &ConfusionReport) -> String {
    let mut s = format!(
        "confusion {} accuracy={:.3} recall={:?}\n  pred→",
        c.arm, c.accuracy, c.per_label_recall
    );
    for j in 0..NUM_LABELS {
        s.push_str(&format!(" {j}"));
    }
    s.push_str(" ?\n");
    for i in 0..NUM_LABELS {
        s.push_str(&format!("  true {i}"));
        for j in 0..=NUM_LABELS {
            s.push_str(&format!(" {}", c.matrix[i][j]));
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn performance_spin_rqm_vs_main_rqm() {
        let c = run_performance_compare();
        println!("{}", format_compare(&c));
        assert!(c.main.accuracy >= 0.99, "{:?}", c.main);
        assert!(c.spin.accuracy >= 0.99, "{:?}", c.spin);
        assert!(c.infer_slowdown > 1.0, "{:?}", c);
    }

    #[test]
    fn detailed_phase_breakdown_collapse_dominates() {
        let b = bench_spin_phase_breakdown(40);
        println!(
            "phase_breakdown samples={} encode+collapse={:.1}µs feat={:.1}µs rqm_query={:.1}µs rqm_train={:.1}µs total={:.1}µs collapse_frac={:.3}",
            b.samples,
            b.encode_collapse_us,
            b.extract_features_us,
            b.rqm_query_us,
            b.rqm_train_us,
            b.total_infer_us,
            b.collapse_fraction
        );
        assert!(
            b.collapse_fraction > 0.85,
            "NLS collapse should dominate infer time: {:?}",
            b
        );
        assert!(b.rqm_query_us < b.encode_collapse_us * 0.05);
    }

    #[test]
    fn detailed_infer_latency_percentiles() {
        let main = bench_infer_latency_main(200);
        let spin = bench_infer_latency_spin(80);
        println!(
            "latency main n={} mean={:.2} p50={:.2} p90={:.2} p99={:.2} min={:.2} max={:.2} µs",
            main.n, main.mean_us, main.p50_us, main.p90_us, main.p99_us, main.min_us, main.max_us
        );
        println!(
            "latency spin n={} mean={:.1} p50={:.1} p90={:.1} p99={:.1} min={:.1} max={:.1} µs",
            spin.n, spin.mean_us, spin.p50_us, spin.p90_us, spin.p99_us, spin.min_us, spin.max_us
        );
        assert!(main.p99_us < 200.0, "main p99 should stay tiny: {:?}", main);
        assert!(
            spin.mean_us > main.mean_us * 50.0,
            "spin should be >> main: main={:?} spin={:?}",
            main,
            spin
        );
    }

    #[test]
    fn detailed_train_curve_reaches_perfect() {
        let curve = bench_train_curve(6);
        for p in &curve {
            println!(
                "train_curve epoch={} acc={:.2} rel={} epoch_ms={:.2}",
                p.epoch, p.accuracy, p.relations, p.epoch_ms
            );
        }
        assert_eq!(curve[0].accuracy, 0.0);
        assert!(
            curve.last().unwrap().accuracy >= 0.99,
            "should reach ~1.0 by epoch 6: {:?}",
            curve
        );
        let first_perfect = curve.iter().find(|p| p.accuracy >= 0.99).map(|p| p.epoch);
        assert!(first_perfect.is_some());
        println!("first_perfect_epoch={first_perfect:?}");
    }

    #[test]
    fn detailed_confusion_matrices() {
        let main = bench_confusion_main(25);
        let spin = bench_confusion_spin(25);
        print!("{}", format_confusion(&main));
        print!("{}", format_confusion(&spin));
        assert!(main.accuracy >= 0.99);
        assert!(spin.accuracy >= 0.99);
        for r in main.per_label_recall.iter().chain(spin.per_label_recall.iter()) {
            assert!(*r >= 0.99);
        }
    }

    #[test]
    fn detailed_collapse_steps_scaling() {
        let grid = [4usize, 12, 28, 48];
        let pts = bench_collapse_steps_scaling(&grid, 10);
        for p in &pts {
            println!(
                "scale steps={} infer_us={:.1} acc={:.3} amp_mean={:.3}",
                p.steps, p.infer_us, p.accuracy, p.amp_mean
            );
        }
        // Más pasos ⇒ más tiempo (monótono no estricto por ruido, pero 48 > 4).
        assert!(
            pts.last().unwrap().infer_us > pts.first().unwrap().infer_us * 2.0,
            "more collapse steps should cost more: {:?}",
            pts
        );
        // Con pocos pasos aún puede aprender si features espaciales ya diferencian.
        assert!(pts.iter().all(|p| p.accuracy >= 0.75), "{:?}", pts);
    }

    #[test]
    fn detailed_shifted_map_cost() {
        let b = bench_shifted_map(8, 30);
        println!(
            "shifted_map train_ms={:.2} infer_ms={:.2} acc={:.3} rel={}",
            b.train_ms, b.infer_ms, b.accuracy, b.relations
        );
        assert!(b.accuracy >= 0.99, "{:?}", b);
        assert!(b.relations > 0);
    }
}
