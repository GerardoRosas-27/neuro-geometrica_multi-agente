//! POC: núcleo de inferencia — WavePredictCore (líquido/ondas) vs NativeThermoCdtSubstrate.
//!
//! Misma tarea N=8: continuación identidad (obs → predecir el mismo contenido
//! entre 8 candidatos). Mide latencia µs/query, exactitud y veredicto.

use crate::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use crate::wave_predict_core::WavePredictCore;
use std::f32::consts::TAU;
use std::time::Instant;

const N: usize = 8;
const THERMO_STEPS: usize = 8;
const PILOT_AMP: f32 = 1.2;
const WARMUP_QUERIES: usize = 32;
const BENCH_LOOPS: usize = 80; // 80 * 8 = 640 queries por brazo

#[derive(Clone, Debug)]
pub struct ArmMetrics {
    pub name: &'static str,
    pub mean_us: f64,
    pub accuracy: f64,
    pub queries: usize,
    /// Solo thermo: tiempo de entrenamiento una vez (ms).
    pub train_ms: Option<f64>,
    pub config_note: String,
}

#[derive(Clone, Debug)]
pub struct PocReport {
    pub liquid: ArmMetrics,
    pub thermo_small: ArmMetrics,
    pub thermo_mid: Option<ArmMetrics>,
    pub verdict: String,
}

fn concept_nodes(concept: usize, node_count: usize, n_concepts: usize) -> Vec<usize> {
    let per = node_count / n_concepts;
    debug_assert!(per >= 1, "need at least one node per concept");
    let start = (concept % n_concepts) * per;
    let end = if concept % n_concepts == n_concepts - 1 {
        node_count
    } else {
        start + per
    };
    (start..end).collect()
}

fn concept_phase(concept: usize, n_concepts: usize) -> f32 {
    (concept % n_concepts) as f32 * (TAU / n_concepts as f32)
}

fn soft_reset(sub: &mut NativeThermoCdtSubstrate) {
    sub.clear_activation();
    let n = sub.node_count();
    for i in 0..n {
        sub.amplitude[i] = 0.5;
        sub.phase[i] = 0.0;
        sub.thermal_state[i] = 0.0;
        sub.pilot_force[i] = 0.0;
    }
}

fn snapshot_owned(sub: &NativeThermoCdtSubstrate, nodes: &[usize]) -> Vec<f64> {
    let mut v = Vec::with_capacity(nodes.len() * 3);
    for &i in nodes {
        let a = sub.amplitude[i] as f64;
        let p = sub.phase[i] as f64;
        v.push(a);
        v.push(p.cos());
        v.push(p.sin());
    }
    v
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

fn small_thermo_config() -> NativeThermoCdtConfig {
    NativeThermoCdtConfig {
        slices: 2,
        nodes_per_slice: 48,
        spatial_degree: 3,
        temporal_degree: 1,
        temperature: 0.2,
        seed: 0x70C0,
        ..NativeThermoCdtConfig::default()
    }
}

fn mid_thermo_config() -> NativeThermoCdtConfig {
    NativeThermoCdtConfig {
        slices: 4,
        nodes_per_slice: 80,
        spatial_degree: 4,
        temporal_degree: 2,
        temperature: 0.25,
        seed: 0x70C0,
        ..NativeThermoCdtConfig::default()
    }
}

struct ThermoMemory {
    /// Plantilla por concepto: amp+cos+sin de sus nodos (mismo tamaño cada una).
    templates: Vec<Vec<f64>>,
    node_sets: Vec<Vec<usize>>,
}

fn train_thermo(sub: &mut NativeThermoCdtSubstrate, n_concepts: usize) -> ThermoMemory {
    let nc = sub.node_count();
    let mut node_sets = Vec::with_capacity(n_concepts);
    for c in 0..n_concepts {
        node_sets.push(concept_nodes(c, nc, n_concepts));
    }
    let mut templates = Vec::with_capacity(n_concepts);
    for (c, nodes) in node_sets.iter().enumerate().take(n_concepts) {
        soft_reset(sub);
        sub.inject_pilot_pattern(nodes, PILOT_AMP, concept_phase(c, n_concepts));
        for _ in 0..THERMO_STEPS {
            let _ = sub.step();
        }
        templates.push(snapshot_owned(sub, nodes));
    }
    ThermoMemory {
        templates,
        node_sets,
    }
}

fn predict_thermo(
    sub: &mut NativeThermoCdtSubstrate,
    mem: &ThermoMemory,
    observation: usize,
    n_concepts: usize,
) -> usize {
    let cue = observation % n_concepts;
    soft_reset(sub);
    let nodes = &mem.node_sets[cue];
    sub.inject_pilot_pattern(nodes, PILOT_AMP, concept_phase(cue, n_concepts));
    for _ in 0..THERMO_STEPS {
        let _ = sub.step();
    }
    // Firma del cue (nodos del concepto observado) vs todas las plantillas.
    let feat = snapshot_owned(sub, nodes);
    let mut best_c = 0usize;
    let mut best_s = f64::NEG_INFINITY;
    for c in 0..n_concepts {
        let s = cosine_sim(&feat, &mem.templates[c]);
        if s > best_s {
            best_s = s;
            best_c = c;
        }
    }
    best_c
}

fn bench_liquid(warmup: usize, loops: usize) -> ArmMetrics {
    let mut core = WavePredictCore::new();
    let candidates: Vec<usize> = (0..N).collect();

    for _ in 0..warmup {
        for obs in 0..N {
            let _ = core.predict_from_observation(obs, &candidates);
        }
    }

    let mut ok = 0usize;
    let mut total = 0usize;
    let t0 = Instant::now();
    for _ in 0..loops {
        for obs in 0..N {
            let r = core.predict_from_observation(obs, &candidates);
            total += 1;
            if r.best_content == obs {
                ok += 1;
            }
        }
    }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
    ArmMetrics {
        name: "WavePredictCore (liquid/analytic)",
        mean_us: elapsed_us / total as f64,
        accuracy: ok as f64 / total as f64,
        queries: total,
        train_ms: None,
        config_note: format!("N={N} identity, analytic interference, no NLS grid"),
    }
}

fn bench_thermo(
    config: NativeThermoCdtConfig,
    label: &'static str,
    warmup: usize,
    loops: usize,
) -> ArmMetrics {
    let note = format!(
        "slices={} nodes/slice={} deg={}x{} T={} seed={:#x} steps={}",
        config.slices,
        config.nodes_per_slice,
        config.spatial_degree,
        config.temporal_degree,
        config.temperature,
        config.seed,
        THERMO_STEPS
    );
    let mut sub = NativeThermoCdtSubstrate::new(config);
    let t_train = Instant::now();
    let mem = train_thermo(&mut sub, N);
    let train_ms = t_train.elapsed().as_secs_f64() * 1e3;

    for _ in 0..warmup {
        for obs in 0..N {
            let _ = predict_thermo(&mut sub, &mem, obs, N);
        }
    }

    let mut ok = 0usize;
    let mut total = 0usize;
    let t0 = Instant::now();
    for _ in 0..loops {
        for obs in 0..N {
            let pred = predict_thermo(&mut sub, &mem, obs, N);
            total += 1;
            if pred == obs {
                ok += 1;
            }
        }
    }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
    ArmMetrics {
        name: label,
        mean_us: elapsed_us / total as f64,
        accuracy: ok as f64 / total as f64,
        queries: total,
        train_ms: Some(train_ms),
        config_note: note,
    }
}

/// Ejecuta el POC completo (liquid + thermo small + thermo mid latency).
pub fn run_poc() -> PocReport {
    let liquid = bench_liquid(WARMUP_QUERIES / N, BENCH_LOOPS);
    let thermo_small = bench_thermo(
        small_thermo_config(),
        "NativeThermoCdt (small POC)",
        WARMUP_QUERIES / N,
        BENCH_LOOPS,
    );
    // Mid-size solo ballpark de latencia (menos loops).
    let thermo_mid = Some(bench_thermo(
        mid_thermo_config(),
        "NativeThermoCdt (mid ~4×80)",
        2,
        8,
    ));

    let verdict = if liquid.mean_us < thermo_small.mean_us
        && liquid.accuracy >= 0.99
        && thermo_small.accuracy >= 0.99
    {
        "Para eficiencia de núcleo de inferencia puro → gana el líquido (WavePredictCore). \
         Thermo CDT es para memoria aprendida durable / dinámica, no el core de query más rápido."
            .to_string()
    } else if liquid.mean_us < thermo_small.mean_us {
        "El líquido es más rápido; revisar exactitud del brazo termo si < 0.99.".to_string()
    } else {
        "Resultado inesperado: thermo no debería ganar en latencia en este POC.".to_string()
    };

    PocReport {
        liquid,
        thermo_small,
        thermo_mid,
        verdict,
    }
}

pub fn format_poc_report(r: &PocReport) -> String {
    let mut s = String::new();
    s.push_str("=== POC: líquido (ondas) vs termo CDT — núcleo de inferencia ===\n");
    s.push_str(&format!(
        "Tarea: N={N} continuación identidad (obs → mismo contenido entre 8 candidatos)\n\n"
    ));
    for arm in [&r.liquid, &r.thermo_small] {
        s.push_str(&format_arm(arm));
    }
    if let Some(ref mid) = r.thermo_mid {
        s.push_str("--- ballpark motor completo ---\n");
        s.push_str(&format_arm(mid));
    }
    s.push_str(&format!("VEREDICTO: {}\n", r.verdict));
    s
}

fn format_arm(a: &ArmMetrics) -> String {
    let train = a
        .train_ms
        .map(|ms| format!(" train_once={ms:.3} ms"))
        .unwrap_or_default();
    format!(
        "[{}]\n  mean_us={:.4} µs/query  accuracy={:.4}  queries={}{}\n  {}\n",
        a.name, a.mean_us, a.accuracy, a.queries, train, a.config_note
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poc_liquid_vs_thermo_core() {
        let report = run_poc();
        let text = format_poc_report(&report);
        println!("{text}");
        assert!(
            report.liquid.accuracy >= 0.99,
            "liquid acc={}",
            report.liquid.accuracy
        );
        assert!(
            report.thermo_small.accuracy >= 0.99,
            "thermo_small acc={}",
            report.thermo_small.accuracy
        );
        assert!(
            report.liquid.mean_us < report.thermo_small.mean_us,
            "liquid ({:.4} µs) should beat thermo_small ({:.4} µs)",
            report.liquid.mean_us,
            report.thermo_small.mean_us
        );
    }
}
