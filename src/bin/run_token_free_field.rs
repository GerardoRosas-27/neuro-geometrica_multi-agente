//! Runner reproducible: paridad Python↔Rust y latencia oficial del core.
//! Uso: run_token_free_field <artifact.json> <parity_corpus.json> [reps]
use cdt_rqm_epr::token_free_field::{FieldState, LiquidParams, RolloutConfig, Scratch};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Corpus {
    psi0: Vec<Vec<f32>>,
    ops: Vec<Vec<usize>>,
    /// Estados de referencia del framework en cada paso: [query][step][N].
    reference: Vec<Vec<Vec<f32>>>,
    slots: usize,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 - 1.0) * q).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("uso: run_token_free_field <artifact.json> <corpus.json> [reps]");
        std::process::exit(2);
    }
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let p = LiquidParams::from_json(&std::fs::read_to_string(&args[1]).expect("artifact"))
        .expect("parse artifact");
    let c: Corpus =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).expect("corpus")).expect("corpus");
    let steps = c.ops[0].len();
    let cfg = RolloutConfig { max_steps: steps };
    let mut s = Scratch::default();
    let mut per_step_err: Vec<Vec<f64>> = vec![Vec::new(); steps];
    let mut disc_match = vec![0usize; steps];
    let mut tot = 0usize;
    for (q, psi0) in c.psi0.iter().enumerate() {
        let (st, tr) = p.rollout(&FieldState(psi0.clone()), &c.ops[q], cfg, &mut s);
        assert_eq!(tr.token_ids_seen + tr.llm_calls, 0);
        for t in 0..steps {
            let r = &c.reference[q][t];
            for (a, b) in st[t].0.iter().zip(r) {
                per_step_err[t].push((a - b).abs() as f64);
            }
            if p.decode(&st[t], c.slots) == p.decode(&FieldState(r.clone()), c.slots) {
                disc_match[t] += 1;
            }
        }
        tot += 1;
    }
    let mut report = serde_json::Map::new();
    for t in [0usize, steps - 1] {
        let e = &mut per_step_err[t];
        let mean = e.iter().sum::<f64>() / e.len() as f64;
        let max = e.iter().cloned().fold(0.0, f64::max);
        let p99 = pct(e, 0.99);
        report.insert(
            format!("step{}", t + 1),
            serde_json::json!({"max_abs": max, "mean_abs": mean, "p99_abs": p99,
                "discrete_match": disc_match[t] as f64 / tot as f64}),
        );
    }
    // Latencia oficial: una consulta = 1 paso de un Ψ (single-query, 1 hilo).
    let mut lat = Vec::new();
    for _ in 0..reps {
        for (q, psi0) in c.psi0.iter().enumerate() {
            let psi = FieldState(psi0.clone());
            let t0 = Instant::now();
            let (st, _) = p.rollout(&psi, &c.ops[q][..1], cfg, &mut s);
            std::hint::black_box(&st);
            lat.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
    }
    let mut dec = Vec::new();
    for psi0 in c.psi0.iter() {
        let psi = FieldState(psi0.clone());
        let t0 = Instant::now();
        std::hint::black_box(p.decode(&psi, c.slots));
        dec.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    report.insert(
        "latency_us_step".into(),
        serde_json::json!({"p50": pct(&mut lat, 0.5), "p95": pct(&mut lat, 0.95),
            "p99": pct(&mut lat, 0.99)}),
    );
    report.insert(
        "latency_us_decoder".into(),
        serde_json::json!({"p50": pct(&mut dec, 0.5), "p99": pct(&mut dec, 0.99)}),
    );
    report.insert("params".into(), p.param_count().into());
    report.insert("params_sha256".into(), p.sha256.clone().into());
    report.insert("token_ids_seen_by_core".into(), 0.into());
    report.insert("llm_calls_after_encoder".into(), 0.into());
    println!("{}", serde_json::Value::Object(report));
}
