//! V4 Dφ / teacher distillation (docs/plan_v4_dphi_teacher_distillation.md,
//! pre-registration docs/preregistro_v4.md §12+).
//!
//! * `DIST_A` — distillation ceiling: teacher direct vs Dφ trained directly on
//!   the `LearningSignal` teacher output vs the existing consolidation route.
//! * `DIST_B` — capacity sweep (hidden 64/136/272/544 ≈ 962/2k/4k/8k params).
//!
//! The teacher is consolidated from the CDT store, used only during training
//! and dropped before TEST. Every teacher target is checked by the auditor
//! against TEST (canonical hash + 5 % ball). The teacher→Dφ diagnostic is
//! computed *after* TEST evaluation from a freshly re-consolidated signal and
//! never reaches a model.

#![allow(clippy::needless_range_loop)]

use crate::field_autonomy_v4::{
    adapt_dynamics, collect_experience, consolidate, evaluate, replay_sample, triples, AuditFlags,
    Auditor, ConsolidateMode, EvalOut, ExpMode, LearningSignal, Row, RunOpts, K_POINTS, MARGIN,
    REPLAY_FRACTION,
};
use crate::v4_controls::{Dphi, Predictor, TrainBudget, Triple, HIDDEN};
use crate::v4_dataset::{generate_and_seal, DatasetConfig, Example, Sealed, CTX_DIM, N_FAM};
use crate::v4_metrics as m;
use crate::v4_provenance::QueryCounters;
use rand::Rng;

pub const CAPACITY_HIDDEN: [usize; 4] = [64, 136, 272, 544];

fn fam_params(c: &[f64]) -> (usize, [f64; 2]) {
    let fam = (0..N_FAM).find(|&f| c[f] > 0.5).unwrap_or(0);
    (fam, [c[N_FAM], c[N_FAM + 1]])
}

/// Train `d` on teacher outputs only: with prob REPLAY_FRACTION a generative
/// replay state, else a TRAIN input relabelled by the teacher (ground truth is
/// never used). Returns the number of auditor hits (must be 0).
pub fn distill(
    d: &mut Dphi,
    sig: &LearningSignal,
    train: &[Triple],
    budget: TrainBudget,
    seed: u64,
    k: usize,
    aud: &Auditor,
    flags: &mut AuditFlags,
) {
    let n = train.len();
    let mut hits: Vec<u64> = vec![];
    d.train_with(budget, seed, |r| {
        let t = if r.gen::<f64>() < REPLAY_FRACTION {
            replay_sample(sig, r, k)
        } else {
            let (x, c, _) = &train[r.gen_range(0..n)];
            let (f, p) = fam_params(c);
            let y = sig.predict(f, p, x).unwrap_or_else(|| x.clone());
            (x.clone(), c.clone(), y)
        };
        hits.extend(aud.hits(&t.2));
        t
    });
    for h in hits {
        if h == u64::MAX {
            flags.any_equiv_global = true;
        } else {
            flags.equiv_hits.insert(h);
        }
    }
}

struct Teacher<'a>(&'a LearningSignal);
impl Predictor for Teacher<'_> {
    fn name(&self) -> &str {
        "TEACHER"
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        let (f, p) = fam_params(c);
        self.0.predict(f, p, x).unwrap_or_else(|| x.to_vec())
    }
    fn params(&self) -> usize {
        self.0.n_numbers()
    }
}

/// Post-hoc diagnostics (after TEST eval): teacher→model rel. error and
/// median σ_max(J_step) on TEST inputs.
fn diag(d: &dyn Predictor, sig: &LearningSignal, test: &[Example]) -> (f64, f64) {
    let t = Teacher(sig);
    let te: Vec<f64> = test
        .iter()
        .map(|e| {
            m::rel_err(
                &d.predict(&e.input, &e.context),
                &t.predict(&e.input, &e.context),
            )
            .min(10.0)
        })
        .collect();
    let sg: Vec<f64> = test
        .iter()
        .step_by(2)
        .map(|e| m::sigma_max(&m::jacobian(|w| d.predict(w, &e.context), &e.input)))
        .collect();
    (m::mean(&te), m::median(&sg))
}

fn mk_row(
    exp: &str,
    seed: u64,
    cond: &str,
    ev: &EvalOut,
    params: usize,
    compute: usize,
    notes: String,
) -> Row {
    Row {
        experiment: exp.into(),
        seed,
        condition: cond.into(),
        partition: "TEST".into(),
        n_examples: ev.n,
        params,
        compute,
        accuracy: ev.acc,
        cosine: ev.disp_cos,
        energy: ev.energy_ratio,
        manifold_distance: ev.manifold,
        stability: ev.rel_err,
        leakage: ev.leaked,
        q: ev.queries,
        status: if ev.leaked > 0 {
            "LEAKED".into()
        } else {
            "CLEAN".into()
        },
        notes,
    }
}

fn gate_row(exp: &str, seed: u64, pass: bool, notes: String) -> Row {
    Row {
        experiment: exp.into(),
        seed,
        condition: "GATE".into(),
        partition: "TEST".into(),
        n_examples: 0,
        params: 0,
        compute: 0,
        accuracy: 0.0,
        cosine: 0.0,
        energy: 0.0,
        manifold_distance: 0.0,
        stability: 0.0,
        leakage: 0,
        q: QueryCounters::default(),
        status: if pass { "PASS".into() } else { "FAIL".into() },
        notes,
    }
}

/// Fresh-process evaluation of a checkpoint (same CLI as E37). Returns
/// (acc, queries_total, pid).
fn fresh_eval(opts: &RunOpts, s: &Sealed, seed: u64, d: &Dphi, tag: &str) -> (f64, f64, f64) {
    let (Some(dir), Some(exe)) = (&opts.ckpt_dir, &opts.exe) else {
        return (f64::NAN, f64::NAN, f64::NAN);
    };
    std::fs::create_dir_all(dir).ok();
    let p = dir.join(format!("dphi_{tag}_0x{seed:X}.json"));
    std::fs::write(&p, d.to_ckpt()).ok();
    let o = std::process::Command::new(exe)
        .args([
            "eval-checkpoint",
            "--ckpt",
            p.to_str().unwrap_or(""),
            "--seed",
            &format!("{seed}"),
            "--mode",
            "P0",
            "--test-sha",
            &s.manifest.test_sha256,
        ])
        .output();
    let v: Option<serde_json::Value> = o.ok().and_then(|o| {
        let t = String::from_utf8_lossy(&o.stdout).to_string();
        serde_json::from_str(t.lines().last()?).ok()
    });
    let g = |k: &str| v.as_ref().and_then(|x| x[k].as_f64()).unwrap_or(f64::NAN);
    (g("acc"), g("queries_total"), g("pid"))
}

/// Train one model by `route` ("consol" | "distill" | "train"), audit it,
/// evaluate on TEST and return (model, eval, wall seconds).
fn train_route(
    route: &str,
    mut d: Dphi,
    s: &Sealed,
    seed: u64,
    budget: TrainBudget,
    aud: &Auditor,
) -> (Dphi, EvalOut, f64) {
    let k = K_POINTS;
    let tr = triples(&s.ds.train);
    let mut flags = AuditFlags::default();
    let t0 = std::time::Instant::now();
    match route {
        "train" => {
            d.train_on(&tr, budget, seed ^ 0x7A);
        }
        _ => {
            let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
            flags.cdt_canon = cdt.canonical_targets();
            let sig = consolidate(&cdt, ConsolidateMode::Full);
            if route == "consol" {
                adapt_dynamics(
                    &mut d,
                    &tr,
                    Some(&sig),
                    None,
                    budget,
                    seed ^ 0x7A,
                    k,
                    aud,
                    &mut flags,
                );
            } else {
                distill(&mut d, &sig, &tr, budget, seed ^ 0x7A, k, aud, &mut flags);
            }
            drop(sig);
            drop(cdt);
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    s.verify_test().expect("TEST sealed");
    let ev = evaluate(&d, &s.ds.test, 1, Some((aud, &flags)), None);
    (d, ev, wall)
}

/// Auditor-side teacher, rebuilt after TEST for diagnostics only.
fn posthoc_teacher(s: &Sealed, seed: u64) -> LearningSignal {
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    consolidate(&cdt, ConsolidateMode::Full)
}

fn clean(ev: &EvalOut) -> bool {
    ev.leaked == 0 && ev.queries.total() == 0
}

pub fn run(seed: u64, exp: &str, opts: &RunOpts) -> Vec<Row> {
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
    let aud = Auditor::new(&s);
    let b1 = TrainBudget::lock();
    let compute = b1.examples_seen();
    let mut rows = vec![];
    match exp {
        "DIST_A" => {
            let init = |lbl: &str| -> Dphi {
                if lbl == "mlp" {
                    Dphi::mlp_direct(2 * K_POINTS, CTX_DIM, seed ^ 0xD0)
                } else {
                    Dphi::new(2 * K_POINTS, CTX_DIM, seed ^ 0xD0)
                }
            };
            let (a, ev_a, _) = train_route("train", init("d"), &s, seed, b1, &aud);
            let (b, ev_b, _) = train_route("consol", init("d"), &s, seed, b1, &aud);
            let (d1, ev_d1, _) = train_route("distill", init("d"), &s, seed, b1, &aud);
            let (mlp, ev_m, _) = train_route("distill", init("mlp"), &s, seed, b1, &aud);
            let b4 = TrainBudget {
                steps: 4 * b1.steps,
                ..b1
            };
            let (d4, ev_d4, _) = train_route("distill", init("d"), &s, seed, b4, &aud);
            let (fp_acc, fp_q, fp_pid) = fresh_eval(opts, &s, seed, &d1, "DIST_A_D1");
            // post-hoc (auditor side, after all TEST evaluations)
            let sig = posthoc_teacher(&s, seed);
            let ev_t = evaluate(&Teacher(&sig), &s.ds.test, 1, None, None);
            let mut cap = vec![];
            for (cond, mdl, ev, c) in [
                ("A_train_only", &a as &dyn Predictor, &ev_a, compute),
                ("B_consolidation_route", &b, &ev_b, compute),
                ("D1_teacher_distill", &d1, &ev_d1, compute),
                ("M1_mlp_direct_distill", &mlp, &ev_m, compute),
                ("D4_teacher_distill_4xsteps", &d4, &ev_d4, 4 * compute),
            ] {
                let (te, sg) = diag(mdl, &sig, &s.ds.test);
                cap.push(ev.acc / ev_t.acc.max(1e-9));
                rows.push(mk_row(
                    "DIST_A",
                    seed,
                    cond,
                    ev,
                    mdl.params(),
                    c,
                    format!(
                        "teacher_to_model_rel={te:.4} sigma_med={sg:.4} capture={:.3}",
                        ev.acc / ev_t.acc.max(1e-9)
                    ),
                ));
            }
            rows.push(mk_row(
                "DIST_A",
                seed,
                "T_teacher_direct",
                &ev_t,
                sig.n_numbers(),
                0,
                "posthoc auditor-side; structured hypothesis class".into(),
            ));
            let all_clean = [&ev_a, &ev_b, &ev_d1, &ev_m, &ev_d4]
                .iter()
                .all(|e| clean(e));
            let fresh_ok = (fp_acc - ev_d1.acc).abs() < 1e-12 && fp_q == 0.0;
            let pass = ev_d1.acc >= ev_b.acc + MARGIN && all_clean && fresh_ok;
            rows.push(gate_row(
                "DIST_A",
                seed,
                pass,
                format!(
                    "T={:.4} A={:.4} B={:.4} D1={:.4} M1={:.4} D4={:.4} relerr_B={:.4} relerr_D1={:.4} relerr_T={:.4} fresh_D1={fp_acc:.4}(q={fp_q},pid={fp_pid}) clean={all_clean} ckpt={}",
                    ev_t.acc, ev_a.acc, ev_b.acc, ev_d1.acc, ev_m.acc, ev_d4.acc, ev_b.rel_err, ev_d1.rel_err, ev_t.rel_err,
                    &d1.weights_hash()[..12]
                ),
            ));
        }
        "DIST_B" => {
            let route = std::env::var("V4_DIST_ROUTE").unwrap_or_else(|_| "distill".into());
            let mut accs = vec![];
            let mut models: Vec<(String, Dphi, EvalOut, f64)> = vec![];
            for &h in &CAPACITY_HIDDEN {
                let (d, ev, w) = train_route(
                    &route,
                    Dphi::with_hidden(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0),
                    &s,
                    seed,
                    b1,
                    &aud,
                );
                accs.push(ev.acc);
                models.push((format!("DPHI_H{h}"), d, ev, w));
                let (d, ev, w) = train_route(
                    &route,
                    Dphi::mlp_direct_h(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0),
                    &s,
                    seed,
                    b1,
                    &aud,
                );
                models.push((format!("MLP_H{h}"), d, ev, w));
            }
            let (bc, ev_bc, _) = train_route(
                "consol",
                Dphi::new(2 * K_POINTS, CTX_DIM, seed ^ 0xD0),
                &s,
                seed,
                b1,
                &aud,
            );
            let sig = posthoc_teacher(&s, seed);
            let ev_t = evaluate(&Teacher(&sig), &s.ds.test, 1, None, None);
            let mut all_clean = clean(&ev_bc);
            for (name, d, ev, w) in &models {
                let (te, sg) = diag(d, &sig, &s.ds.test);
                all_clean &= clean(ev);
                let mem = d.params() * 8 * 4;
                rows.push(mk_row("DIST_B", seed, name, ev, d.params(), compute, format!("route={route} wall_s={w:.2} mem_bytes={mem} teacher_to_model_rel={te:.4} sigma_med={sg:.4}")));
            }
            let (te, sg) = diag(&bc, &sig, &s.ds.test);
            rows.push(mk_row(
                "DIST_B",
                seed,
                "B_consolidation_route_H64",
                &ev_bc,
                bc.params(),
                compute,
                format!("teacher_to_model_rel={te:.4} sigma_med={sg:.4}"),
            ));
            rows.push(mk_row(
                "DIST_B",
                seed,
                "T_teacher_direct",
                &ev_t,
                sig.n_numbers(),
                0,
                String::new(),
            ));
            let best = (1..accs.len())
                .max_by(|&i, &j| accs[i].total_cmp(&accs[j]))
                .unwrap_or(1);
            rows.push(gate_row(
                "DIST_B",
                seed,
                accs[best] >= accs[0] + MARGIN && all_clean,
                format!(
                    "acc_H64={:.4} acc_H136={:.4} acc_H272={:.4} acc_H544={:.4} T={:.4} clean={all_clean}",
                    accs[0], accs[1], accs[2], accs[3], ev_t.acc
                ),
            ));
        }
        _ => rows.push(gate_row(exp, seed, false, "UNKNOWN_EXP".into())),
    }
    let _ = HIDDEN;
    rows
}
