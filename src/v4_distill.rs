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
#![allow(clippy::too_many_arguments)]

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
        "DIST_C" => rows.extend(run_c(seed, &s, &aud, opts)),
        "DIST_D" => rows.extend(run_d(seed, &s, &aud)),
        "DIST_E" => rows.extend(run_e(seed, &s, &aud, opts)),
        _ => rows.push(gate_row(exp, seed, false, "UNKNOWN_EXP".into())),
    }
    let _ = HIDDEN;
    rows
}

/// Fase C recipe without any env override (used by §16).
pub fn onestep_budget_base() -> TrainBudget {
    TrainBudget {
        steps: 32000,
        batch: 32,
        lr: 3e-3,
    }
}

/// Fase C pre-registered recipe (preregistro §14).
pub fn onestep_budget() -> TrainBudget {
    let mult = std::env::var("V4_ONESTEP_MULT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);
    TrainBudget {
        steps: 32000 * mult,
        batch: 32,
        lr: 3e-3,
    }
}
pub const ONESTEP_LR_FINAL: f64 = 3e-5;

pub fn phase_hidden() -> usize {
    std::env::var("V4_DPHI_H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(HIDDEN)
}

/// Consolidation-route labels (TRAIN ground truth ∪ teacher replay), trained
/// with the explicit one-step recipe. Teacher/CDT dropped before return.
pub fn train_onestep_route(d: &mut Dphi, s: &Sealed, seed: u64, aud: &Auditor) -> AuditFlags {
    train_onestep_route_b(d, s, seed, aud, onestep_budget())
}

pub fn train_onestep_route_b(
    d: &mut Dphi,
    s: &Sealed,
    seed: u64,
    aud: &Auditor,
    budget: TrainBudget,
) -> AuditFlags {
    let k = K_POINTS;
    let tr = triples(&s.ds.train);
    let n = tr.len();
    let mut flags = AuditFlags::default();
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    flags.cdt_canon = cdt.canonical_targets();
    let sig = consolidate(&cdt, ConsolidateMode::Full);
    let mut hits: Vec<u64> = vec![];
    d.train_onestep(budget, seed ^ 0x7A, ONESTEP_LR_FINAL, true, |r| {
        if r.gen::<f64>() < REPLAY_FRACTION {
            let t = replay_sample(&sig, r, k);
            hits.extend(aud.hits(&t.2));
            t
        } else {
            tr[r.gen_range(0..n)].clone()
        }
    });
    drop(sig);
    drop(cdt);
    for h in hits {
        if h == u64::MAX {
            flags.any_equiv_global = true;
        } else {
            flags.equiv_hits.insert(h);
        }
    }
    flags
}

/// Per-family / per-region one-step metrics on TEST (after evaluation).
pub fn onestep_breakdown(d: &dyn Predictor, test: &[Example]) -> String {
    let mut parts = vec![];
    let stat = |xs: &[&Example]| -> (f64, f64, f64, f64, usize) {
        let (mut re, mut ab, mut co, mut md) = (vec![], vec![], vec![], vec![]);
        for e in xs {
            let z = d.predict(&e.input, &e.context);
            re.push(m::rel_err(&z, &e.expected).min(10.0));
            ab.push(crate::v4_dataset::l2(&z, &e.expected));
            co.push(m::disp_cosine(&e.input, &z, &e.expected));
            md.push(m::manifold_distance(&e.input, &z));
        }
        (
            m::mean(&re),
            m::mean(&ab),
            m::mean(&co),
            m::mean(&md),
            xs.len(),
        )
    };
    for f in 0..N_FAM {
        let xs: Vec<&Example> = test.iter().filter(|e| e.family_id == f).collect();
        let (r, a, c, md, _) = stat(&xs);
        parts.push(format!(
            "{}:rel={r:.4},abs={a:.4},cos={c:.3},manif={md:.4}",
            crate::v4_dataset::FAMILIES[f]
        ));
    }
    for reg in ["in_dist", "interp_hole"] {
        let xs: Vec<&Example> = test.iter().filter(|e| e.subset == reg).collect();
        let (r, a, c, md, n) = stat(&xs);
        parts.push(format!(
            "{reg}(n={n}):rel={r:.4},abs={a:.4},cos={c:.3},manif={md:.4}"
        ));
    }
    parts.join(" ")
}

fn run_c(seed: u64, s: &Sealed, aud: &Auditor, opts: &RunOpts) -> Vec<Row> {
    let h = phase_hidden();
    let b1 = TrainBudget::lock();
    let cb = onestep_budget();
    let mut rows = vec![];
    // C0: reference at H (lock budget, consolidation route)
    let (c0, ev0, _) = train_route(
        "consol",
        Dphi::with_hidden(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0),
        s,
        seed,
        b1,
        aud,
    );
    let mut models: Vec<(&str, Dphi, EvalOut, f64)> = vec![("C0_lock_budget", c0, ev0, 0.0)];
    for (name, init) in [
        (
            "C1_onestep_recipe",
            Dphi::with_hidden(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0),
        ),
        (
            "M1_mlp_onestep_recipe",
            Dphi::mlp_direct_h(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0),
        ),
    ] {
        let mut d = init;
        let t0 = std::time::Instant::now();
        let flags = train_onestep_route(&mut d, s, seed, aud);
        let w = t0.elapsed().as_secs_f64();
        s.verify_test().expect("TEST sealed");
        let ev = evaluate(&d, &s.ds.test, 1, Some((aud, &flags)), None);
        models.push((name, d, ev, w));
    }
    let (fp_acc, fp_q, _) = fresh_eval(opts, s, seed, &models[1].1, "DIST_C_C1");
    let sig = posthoc_teacher(s, seed);
    let mut all_clean = true;
    for (name, d, ev, w) in &models {
        let (te, sg) = diag(d, &sig, &s.ds.test);
        all_clean &= clean(ev);
        let comp = if *name == "C0_lock_budget" {
            b1.examples_seen()
        } else {
            cb.examples_seen()
        };
        rows.push(mk_row(
            "DIST_C",
            seed,
            name,
            ev,
            d.params(),
            comp,
            format!(
                "H={h} wall_s={w:.2} teacher_to_model_rel={te:.4} sigma_med={sg:.4} {}",
                onestep_breakdown(d, &s.ds.test)
            ),
        ));
    }
    let c1 = &models[1].2;
    let fresh_ok = (fp_acc - c1.acc).abs() < 1e-12 && fp_q == 0.0;
    rows.push(gate_row(
        "DIST_C",
        seed,
        c1.rel_err < 0.05 && all_clean && fresh_ok,
        format!(
            "H={h} rel_err1 C0={:.4} C1={:.4} M1={:.4} acc C0={:.4} C1={:.4} M1={:.4} fresh_C1={fp_acc:.4}(q={fp_q}) clean={all_clean}",
            models[0].2.rel_err, c1.rel_err, models[2].2.rel_err, models[0].2.acc, c1.acc, models[2].2.acc
        ),
    ));
    rows
}

/// Fase D: rollout stability of the Fase-C model (E33 protocol, iterable
/// families rotate+reflect, free-run). Variants: V0 pure, V1 per-step
/// similarity(+mirror) re-projection onto x0's orbit, V3 V1+spectral limit.
/// Controls: M1 (MLP, same recipe) pure; C0 (lock budget) pure.
fn run_d(seed: u64, s: &Sealed, aud: &Auditor) -> Vec<Row> {
    use crate::field_autonomy_v4::{target_h, HORIZONS, ITERABLE_FAMS};
    let h = phase_hidden();
    let mut c1 = Dphi::with_hidden(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0);
    let f1 = train_onestep_route(&mut c1, s, seed, aud);
    let mut m1 = Dphi::mlp_direct_h(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0);
    let f2 = train_onestep_route(&mut m1, s, seed, aud);
    let (c0, ev_c0, _) = train_route(
        "consol",
        Dphi::with_hidden(2 * K_POINTS, CTX_DIM, h, seed ^ 0xD0),
        s,
        seed,
        TrainBudget::lock(),
        aud,
    );
    s.verify_test().expect("TEST sealed");
    let ev1 = evaluate(&c1, &s.ds.test, 1, Some((aud, &f1)), None);
    let ev2 = evaluate(&m1, &s.ds.test, 1, Some((aud, &f2)), None);
    let all_clean = clean(&ev1) && clean(&ev2) && clean(&ev_c0);
    let iter_ex: Vec<&Example> =
        s.ds.test
            .iter()
            .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
            .collect();
    let mut rows = vec![];
    let mut rng =
        <rand_xoshiro::Xoshiro256StarStar as rand_xoshiro::rand_core::SeedableRng>::seed_from_u64(
            seed ^ 0xD33,
        );
    let deltas: Vec<Vec<f64>> = iter_ex
        .iter()
        .map(|e| {
            let v: Vec<f64> = e.input.iter().map(|_| rng.gen_range(-1.0..1.0)).collect();
            let nv = m::norm(&v).max(1e-12);
            v.iter()
                .map(|x| x / nv * 0.01 * m::norm(&e.input))
                .collect()
        })
        .collect();
    for (label, d, vi) in [
        ("V0_pure_C1", &c1, 0usize),
        ("V1_reproj_C1", &c1, 1),
        ("V3_reproj_spectral_C1", &c1, 3),
        ("V0_pure_M1_mlp", &m1, 0),
        ("V0_pure_C0_lock", &c0, 0),
    ] {
        let mut zs: Vec<Vec<f64>> = iter_ex.iter().map(|e| e.input.clone()).collect();
        let mut zp: Vec<Vec<f64>> = iter_ex
            .iter()
            .zip(&deltas)
            .map(|(e, dl)| e.input.iter().zip(dl).map(|(a, b)| a + b).collect())
            .collect();
        let (mut region, mut contiguous) = (0usize, true);
        let mut sig_all = vec![];
        let mut notes = vec![];
        let hmax = *HORIZONS.last().unwrap();
        for t in 1..=hmax {
            for (i, e) in iter_ex.iter().enumerate() {
                let (z, sg) =
                    crate::v4_stability::step_variant(d, &zs[i], &e.context, &e.input, vi);
                zs[i] = z;
                sig_all.push(sg);
                zp[i] = crate::v4_stability::step_variant(d, &zp[i], &e.context, &e.input, vi).0;
            }
            if HORIZONS.contains(&t) {
                let (mut ok, mut re, mut dr, mut md, mut en, mut pg) =
                    (0usize, vec![], vec![], vec![], vec![], vec![]);
                for (i, e) in iter_ex.iter().enumerate() {
                    let y = target_h(e, t);
                    ok += m::correct(&zs[i], &y) as usize;
                    re.push(m::rel_err(&zs[i], &y).min(10.0));
                    dr.push(m::norm(&zs[i]) / m::norm(&y).max(1e-12));
                    en.push(m::energy(&zs[i]) / m::energy(&y).max(1e-12));
                    md.push(m::manifold_distance(&e.input, &zs[i]));
                    pg.push(crate::v4_dataset::l2(&zp[i], &zs[i]) / m::norm(&deltas[i]).max(1e-12));
                }
                let acc = ok as f64 / iter_ex.len() as f64;
                if contiguous && acc >= 0.5 {
                    region = t;
                } else {
                    contiguous = false;
                }
                notes.push(format!("h{t}:acc={acc:.3},rel={:.3},norm={:.3},energy={:.3},manif={:.3},pert_growth={:.3}", m::mean(&re), m::median(&dr), m::median(&en), m::median(&md), m::median(&pg)));
                rows.push(Row {
                    experiment: "DIST_D".into(),
                    seed,
                    condition: format!("{label}_h{t}"),
                    partition: "TEST_iterable".into(),
                    n_examples: iter_ex.len(),
                    params: d.params(),
                    compute: 0,
                    accuracy: acc,
                    cosine: m::median(&pg),
                    energy: m::median(&en),
                    manifold_distance: m::median(&md),
                    stability: m::mean(&re),
                    leakage: 0,
                    q: QueryCounters::default(),
                    status: "CLEAN".into(),
                    notes: format!(
                        "norm_ratio={:.4} pert_growth={:.4}",
                        m::median(&dr),
                        m::median(&pg)
                    ),
                });
            }
        }
        rows.push(Row {
            partition: label.into(),
            ..gate_row(
                "DIST_D",
                seed,
                region >= 8 && all_clean,
                format!("{label} region={region} sigma_med={:.4} rel_err1_all={:.4}/{:.4}/{:.4} clean={all_clean} {}", m::median(&sig_all), ev1.rel_err, ev2.rel_err, ev_c0.rel_err, notes.join(" ")),
            )
        });
    }
    rows
}

/// §16: per-family consolidation fit residual (in-sample, computed at
/// consolidation time from the CDT experiences — never from DEV/TEST).
pub fn family_fit_residual(
    sig: &LearningSignal,
    exps: &[crate::field_autonomy_v4::Experience],
) -> Vec<f64> {
    let mut v = vec![vec![]; N_FAM];
    for e in exps {
        if let Some(y) = sig.predict(e.fam, e.params, &e.x) {
            v[e.fam].push(m::rel_err(&y, &e.y));
        }
    }
    v.iter()
        .map(|x| {
            if x.is_empty() {
                f64::INFINITY
            } else {
                m::mean(x)
            }
        })
        .collect()
}

pub const TRUST_TAU: f64 = 0.01;

/// §16 trusted-replay route: TRAIN ground truth (prob 1-REPLAY_FRACTION) +
/// teacher replay restricted to families whose consolidation fit residual
/// < `tau`. Returns flags and the trusted mask.
pub fn train_onestep_trusted(
    d: &mut Dphi,
    s: &Sealed,
    seed: u64,
    aud: &Auditor,
    tau: f64,
) -> (AuditFlags, Vec<bool>) {
    train_onestep_trusted_b(d, s, seed, aud, tau, onestep_budget())
}

pub fn train_onestep_trusted_b(
    d: &mut Dphi,
    s: &Sealed,
    seed: u64,
    aud: &Auditor,
    tau: f64,
    budget: TrainBudget,
) -> (AuditFlags, Vec<bool>) {
    train_onestep_trusted_b_ck(d, s, seed, aud, tau, budget, None)
}

/// [`train_onestep_trusted_b`] with optional bit-exact checkpoint (infra only).
/// `hits` is kept as a set (order-free: it only feeds a HashSet/flag below).
pub fn train_onestep_trusted_b_ck(
    d: &mut Dphi,
    s: &Sealed,
    seed: u64,
    aud: &Auditor,
    tau: f64,
    budget: TrainBudget,
    ck: Option<(std::path::PathBuf, String)>,
) -> (AuditFlags, Vec<bool>) {
    let k = K_POINTS;
    let tr = triples(&s.ds.train);
    let n = tr.len();
    let mut flags = AuditFlags::default();
    let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
    flags.cdt_canon = cdt.canonical_targets();
    let sig = consolidate(&cdt, ConsolidateMode::Full);
    let res = family_fit_residual(&sig, cdt.read_all());
    let trusted: Vec<bool> = res.iter().map(|r| *r < tau).collect();
    let any = trusted.iter().any(|t| *t);
    let hits: std::cell::RefCell<std::collections::BTreeSet<u64>> = Default::default();
    let save = || serde_json::json!(hits.borrow().iter().copied().collect::<Vec<u64>>());
    let load = |v: &serde_json::Value| {
        let mut h = hits.borrow_mut();
        h.clear();
        h.extend(
            v.as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_u64()),
        );
    };
    let tc = ck.map(|(p, key)| crate::train_ckpt::TrainCkpt::new(p, key, &save, &load));
    d.train_onestep_ck(
        budget,
        seed ^ 0x7A,
        ONESTEP_LR_FINAL,
        true,
        |r| {
            if any && r.gen::<f64>() < REPLAY_FRACTION {
                loop {
                    let t = replay_sample(&sig, r, k);
                    if trusted[fam_params(&t.1).0] {
                        hits.borrow_mut().extend(aud.hits(&t.2));
                        return t;
                    }
                }
            }
            tr[r.gen_range(0..n)].clone()
        },
        tc.as_ref(),
    );
    drop(tc);
    let hits = hits.into_inner();
    drop(sig);
    drop(cdt);
    for h in hits {
        if h == u64::MAX {
            flags.any_equiv_global = true;
        } else {
            flags.equiv_hits.insert(h);
        }
    }
    (flags, trusted)
}

/// §16 pre-registered constants.
pub const E_HIDDEN: usize = 136;
pub const E_MULT: usize = 8;

/// E33-protocol rollout (rotate+reflect TEST, free-run) for one variant.
/// Returns (region, per-horizon rows, notes).
fn e33_rollout(
    exp: &str,
    seed: u64,
    label: &str,
    d: &Dphi,
    vi: usize,
    iter_ex: &[&Example],
    deltas: &[Vec<f64>],
) -> (usize, Vec<Row>, String) {
    use crate::field_autonomy_v4::{target_h, HORIZONS};
    let mut zs: Vec<Vec<f64>> = iter_ex.iter().map(|e| e.input.clone()).collect();
    let mut zp: Vec<Vec<f64>> = iter_ex
        .iter()
        .zip(deltas)
        .map(|(e, dl)| e.input.iter().zip(dl).map(|(a, b)| a + b).collect())
        .collect();
    let (mut region, mut contiguous) = (0usize, true);
    let (mut rows, mut notes, mut sig_all) = (vec![], vec![], vec![]);
    for t in 1..=*HORIZONS.last().unwrap() {
        for (i, e) in iter_ex.iter().enumerate() {
            let (z, sg) = crate::v4_stability::step_variant(d, &zs[i], &e.context, &e.input, vi);
            zs[i] = z;
            sig_all.push(sg);
            zp[i] = crate::v4_stability::step_variant(d, &zp[i], &e.context, &e.input, vi).0;
        }
        if HORIZONS.contains(&t) {
            let (mut ok, mut re, mut dr, mut md, mut en, mut pg) =
                (0usize, vec![], vec![], vec![], vec![], vec![]);
            let mut fam_ok = [0usize; 2];
            let mut fam_n = [0usize; 2];
            for (i, e) in iter_ex.iter().enumerate() {
                let y = target_h(e, t);
                let c = m::correct(&zs[i], &y);
                ok += c as usize;
                let fi = if e.family_id == 1 { 0 } else { 1 };
                fam_n[fi] += 1;
                fam_ok[fi] += c as usize;
                re.push(m::rel_err(&zs[i], &y).min(10.0));
                dr.push(m::norm(&zs[i]) / m::norm(&y).max(1e-12));
                en.push(m::energy(&zs[i]) / m::energy(&y).max(1e-12));
                md.push(m::manifold_distance(&e.input, &zs[i]));
                pg.push(crate::v4_dataset::l2(&zp[i], &zs[i]) / m::norm(&deltas[i]).max(1e-12));
            }
            let acc = ok as f64 / iter_ex.len() as f64;
            if contiguous && acc >= 0.5 {
                region = t;
            } else {
                contiguous = false;
            }
            notes.push(format!(
                "h{t}:acc={acc:.3}(rot={:.2},refl={:.2}),rel={:.3},norm={:.3},pert={:.3}",
                fam_ok[0] as f64 / fam_n[0].max(1) as f64,
                fam_ok[1] as f64 / fam_n[1].max(1) as f64,
                m::mean(&re),
                m::median(&dr),
                m::median(&pg)
            ));
            rows.push(Row {
                experiment: exp.into(),
                seed,
                condition: format!("{label}_h{t}"),
                partition: "TEST_iterable".into(),
                n_examples: iter_ex.len(),
                params: d.params(),
                compute: 0,
                accuracy: acc,
                cosine: m::median(&pg),
                energy: m::median(&en),
                manifold_distance: m::median(&md),
                stability: m::mean(&re),
                leakage: 0,
                q: QueryCounters::default(),
                status: "CLEAN".into(),
                notes: format!(
                    "norm_ratio={:.4} pert_growth={:.4} acc_rot={:.4} acc_refl={:.4}",
                    m::median(&dr),
                    m::median(&pg),
                    fam_ok[0] as f64 / fam_n[0].max(1) as f64,
                    fam_ok[1] as f64 / fam_n[1].max(1) as f64
                ),
            });
        }
    }
    (
        region,
        rows,
        format!("sigma_med={:.4} {}", m::median(&sig_all), notes.join(" ")),
    )
}

fn run_e(seed: u64, s: &Sealed, aud: &Auditor, opts: &RunOpts) -> Vec<Row> {
    use crate::field_autonomy_v4::ITERABLE_FAMS;
    let z = 2 * K_POINTS;
    let mut rows = vec![];
    let mut models: Vec<(&str, Dphi, EvalOut, String)> = vec![];
    // E0: Fase C model (reference)
    {
        let mut d = Dphi::new(z, CTX_DIM, seed ^ 0xD0);
        let f = train_onestep_route_b(&mut d, s, seed, aud, onestep_budget_base());
        s.verify_test().expect("TEST sealed");
        let ev = evaluate(&d, &s.ds.test, 1, Some((aud, &f)), None);
        models.push(("E0_faseC_H64", d, ev, "route=consol mult=1".into()));
    }
    let specs: [(&str, Dphi, bool); 4] = [
        (
            "E1_plain_trusted",
            Dphi::with_hidden(z, CTX_DIM, E_HIDDEN, seed ^ 0xD0),
            true,
        ),
        (
            "E2_gated_consol",
            Dphi::gated_h(z, CTX_DIM, E_HIDDEN, true, seed ^ 0xD0),
            false,
        ),
        (
            "E3_gated_trusted",
            Dphi::gated_h(z, CTX_DIM, E_HIDDEN, true, seed ^ 0xD0),
            true,
        ),
        (
            "M3_mlp_gated_trusted",
            Dphi::gated_h(z, CTX_DIM, E_HIDDEN, false, seed ^ 0xD0),
            true,
        ),
    ];
    for (name, mut d, trusted) in specs {
        let eb = TrainBudget {
            steps: onestep_budget_base().steps * E_MULT,
            ..onestep_budget_base()
        };
        let (f, note) = {
            if trusted {
                let (f, t) = train_onestep_trusted_b(&mut d, s, seed, aud, TRUST_TAU, eb);
                let ts: String = t.iter().map(|b| if *b { '1' } else { '0' }).collect();
                (f, format!("route=trusted mask={ts} mult={E_MULT}"))
            } else {
                (
                    train_onestep_route_b(&mut d, s, seed, aud, eb),
                    format!("route=consol mult={E_MULT}"),
                )
            }
        };
        s.verify_test().expect("TEST sealed");
        let ev = evaluate(&d, &s.ds.test, 1, Some((aud, &f)), None);
        models.push((name, d, ev, note));
    }
    let e3i = 3;
    let (fp_acc, fp_q, _) = fresh_eval(opts, s, seed, &models[e3i].1, "DIST_E_E3");
    let sig = posthoc_teacher(s, seed);
    let all_clean = models.iter().all(|m| clean(&m.2));
    for (name, d, ev, note) in &models {
        let (te, sg) = diag(d, &sig, &s.ds.test);
        rows.push(mk_row(
            "DIST_E",
            seed,
            name,
            ev,
            d.params(),
            0,
            format!(
                "{note} teacher_to_model_rel={te:.4} sigma_med={sg:.4} {}",
                onestep_breakdown(d, &s.ds.test)
            ),
        ));
    }
    let iter_ex: Vec<&Example> =
        s.ds.test
            .iter()
            .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
            .collect();
    let mut rng =
        <rand_xoshiro::Xoshiro256StarStar as rand_xoshiro::rand_core::SeedableRng>::seed_from_u64(
            seed ^ 0xD33,
        );
    let deltas: Vec<Vec<f64>> = iter_ex
        .iter()
        .map(|e| {
            let v: Vec<f64> = e.input.iter().map(|_| rng.gen_range(-1.0..1.0)).collect();
            let nv = m::norm(&v).max(1e-12);
            v.iter()
                .map(|x| x / nv * 0.01 * m::norm(&e.input))
                .collect()
        })
        .collect();
    let fresh_ok = (fp_acc - models[e3i].2.acc).abs() < 1e-12 && fp_q == 0.0;
    for (mi, vi, label) in [
        (0usize, 0usize, "R_E0_V0"),
        (1, 0, "R_E1_V0"),
        (2, 0, "R_E2_V0"),
        (3, 0, "R_E3_V0_pure"),
        (3, 1, "R_E3_V1_reproj"),
        (4, 0, "R_M3_V0"),
    ] {
        let (region, rr, notes) =
            e33_rollout("DIST_E", seed, label, &models[mi].1, vi, &iter_ex, &deltas);
        rows.extend(rr);
        let extra = if mi == e3i { fresh_ok } else { true };
        rows.push(Row {
            partition: label.into(),
            ..gate_row("DIST_E", seed, region >= 8 && all_clean && extra, format!("{label} region={region} rel_err1={:.4} clean={all_clean} fresh_E3={fp_acc:.4}(q={fp_q}) {notes}", models[mi].2.rel_err))
        });
    }
    rows
}
