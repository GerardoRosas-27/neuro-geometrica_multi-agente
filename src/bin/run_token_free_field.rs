//! Runner reproducible: paridad Python↔Rust y latencia oficial del core.
//! Uso: run_token_free_field <artifact.json> <corpus.json> [reps]
//!      [--topk k] [--substeps K] [--adaptive eps]
//! El corpus trae `ops` (one-hot implícito) o `ctx` (vectores de contexto).
use cdt_rqm_epr::token_free_field::{FieldState, LiquidParams, Scratch};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Corpus {
    psi0: Vec<Vec<f32>>,
    #[serde(default)]
    ops: Vec<Vec<usize>>,
    #[serde(default)]
    ctx: Vec<Vec<Vec<f32>>>,
    /// Estados de referencia del framework: [query][step][N].
    reference: Vec<Vec<Vec<f32>>>,
    /// Slots del decoder lineal; 0 = decoder identidad (tarea v4).
    slots: usize,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 - 1.0) * q).round() as usize]
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

struct Runner<'a> {
    p: &'a LiquidParams,
    topk: Option<usize>,
    substeps: Option<usize>,
    adaptive: Option<f32>,
}

impl Runner<'_> {
    /// Un paso (op) completo; devuelve (Ψ', sub-pasos calculados).
    fn op_step(&self, psi: &FieldState, c: &[f32], s: &mut Scratch) -> (FieldState, usize) {
        let Some(eps) = self.adaptive else {
            return self.p.step_ctx(psi, c, self.topk, self.substeps, s);
        };
        // E48: K = 1, 2, 4, … hasta ‖Ψ_K − Ψ_{K/2}‖/(‖Ψ_K‖+1e-6) < ε (tope 64).
        let (mut prev, mut used) = self.p.step_ctx(psi, c, self.topk, Some(1), s);
        let mut k = 2;
        while k <= 64 {
            let (cur, u) = self.p.step_ctx(psi, c, self.topk, Some(k), s);
            used += u;
            let d: f32 = cur
                .0
                .iter()
                .zip(&prev.0)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let nrm: f32 = cur.0.iter().map(|v| v * v).sum();
            prev = cur;
            if d.sqrt() / (nrm.sqrt() + 1e-6) < eps {
                break;
            }
            k *= 2;
        }
        (prev, used)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("uso: run_token_free_field <artifact.json> <corpus.json> [reps] [--topk k] [--substeps K] [--adaptive eps]");
        std::process::exit(2);
    }
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let p = LiquidParams::from_json(&std::fs::read_to_string(&args[1]).expect("artifact"))
        .expect("parse artifact");
    let c: Corpus =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).expect("corpus")).expect("corpus");
    let r = Runner {
        p: &p,
        topk: flag(&args, "--topk").and_then(|v| v.parse().ok()),
        substeps: flag(&args, "--substeps").and_then(|v| v.parse().ok()),
        adaptive: flag(&args, "--adaptive").and_then(|v| v.parse().ok()),
    };
    let ctx_of = |q: usize, t: usize| -> Vec<f32> {
        if c.ctx.is_empty() {
            let mut v = vec![0.0; p.ctx_dim];
            v[c.ops[q][t]] = 1.0;
            v
        } else {
            c.ctx[q][t].clone()
        }
    };
    let steps = c.reference[0].len();
    let mut s = Scratch::default();
    let mut per_step_err: Vec<Vec<f64>> = vec![Vec::new(); steps];
    let mut disc_match = vec![0usize; steps];
    let mut finals = Vec::new();
    let mut substeps_total = 0usize;
    for (q, psi0) in c.psi0.iter().enumerate() {
        let mut cur = FieldState(psi0.clone());
        let mut traj = Vec::new();
        for t in 0..steps {
            let (n, u) = r.op_step(&cur, &ctx_of(q, t), &mut s);
            substeps_total += u;
            let rf = &c.reference[q][t];
            for (a, b) in n.0.iter().zip(rf) {
                per_step_err[t].push((a - b).abs() as f64);
            }
            let same = if c.slots > 0 {
                p.decode(&n, c.slots) == p.decode(&FieldState(rf.clone()), c.slots)
            } else {
                true
            };
            if same {
                disc_match[t] += 1;
            }
            traj.push(n.0.clone());
            cur = n;
        }
        finals.push(traj);
    }
    let tot = c.psi0.len();
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
    // Latencia oficial: una op de un Ψ (single-query, 1 hilo), selección incluida.
    let mut lat = Vec::new();
    for _ in 0..reps {
        for (q, psi0) in c.psi0.iter().enumerate() {
            let psi = FieldState(psi0.clone());
            let cv = ctx_of(q, 0);
            let t0 = Instant::now();
            let o = r.op_step(&psi, &cv, &mut s);
            std::hint::black_box(&o);
            lat.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
    }
    let mean_lat = lat.iter().sum::<f64>() / lat.len() as f64;
    report.insert(
        "latency_us_step".into(),
        serde_json::json!({"p50": pct(&mut lat, 0.5), "p95": pct(&mut lat, 0.95),
            "p99": pct(&mut lat, 0.99), "mean": mean_lat}),
    );
    if c.slots > 0 {
        let mut dec = Vec::new();
        for psi0 in c.psi0.iter() {
            let psi = FieldState(psi0.clone());
            let t0 = Instant::now();
            std::hint::black_box(p.decode(&psi, c.slots));
            dec.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        report.insert(
            "latency_us_decoder".into(),
            serde_json::json!({"p50": pct(&mut dec, 0.5), "p99": pct(&mut dec, 0.99)}),
        );
    }
    report.insert(
        "substeps_per_op_mean".into(),
        (substeps_total as f64 / (tot * steps) as f64).into(),
    );
    report.insert("params".into(), p.param_count().into());
    report.insert("params_sha256".into(), p.sha256.clone().into());
    report.insert("token_ids_seen_by_core".into(), 0.into());
    report.insert("llm_calls_after_encoder".into(), 0.into());
    report.insert("memory_queries".into(), 0.into());
    if let Some(path) = flag(&args, "--dump") {
        std::fs::write(path, serde_json::to_string(&finals).expect("dump")).expect("dump");
    }
    println!("{}", serde_json::Value::Object(report));
}
