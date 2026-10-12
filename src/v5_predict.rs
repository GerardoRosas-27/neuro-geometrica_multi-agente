//! Predictive field v5 (branch `exp/predictive-field-v5`).
//!
//! Same gated Dφ as v4 §16, trained on a different objective: the iterated
//! field must match the true future of a TRAIN episode at horizons 1, 2, 4
//! and 8. Episodes live in a CDT buffer that is dropped before TEST. There is
//! no ridge teacher and no trusted-replay mask.
//!
//! Pre-registration: `docs/preregistro_v5.md`. Smoke is not evidence.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::field_autonomy_v4::{
    evaluate, target_h, AuditFlags, Auditor, Row, RunOpts, CSV_HEADER, HORIZONS, ITERABLE_FAMS,
    K_POINTS,
};
use crate::v4_controls::{Dphi, Predictor, TrainBudget};
use crate::v4_dataset::{
    apply_rule, generate_and_seal, l2, sha256_hex, DatasetConfig, Example, Sealed, CTX_DIM,
    FAMILIES, N_FAM,
};
use crate::v4_distill::{onestep_breakdown, train_onestep_trusted_b_ck, TRUST_TAU};
use crate::v4_metrics as m;
use crate::v4_provenance::{count_query, snapshot, QueryCounters, Store};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Horizons inside the training loss. E33 still evaluates 1…64.
pub const LOSS_AT: [usize; 4] = [1, 2, 4, 8];
pub const EPISODE_H: usize = 8;
pub const V5_HIDDEN: usize = 136;
/// Same step budget as v4 §16 E3: 32_000 × 8.
pub const V5_STEP_MULT: usize = 8;
pub const V5_LR_FINAL: f64 = 3e-5;
/// Smoke only. Not a result.
pub const SMOKE_STEPS: usize = 200;

pub fn budget(steps: usize) -> TrainBudget {
    TrainBudget {
        steps,
        batch: 32,
        lr: 3e-3,
    }
}

pub fn full_steps() -> usize {
    32_000 * V5_STEP_MULT
}

#[derive(Clone)]
struct Episode {
    fam: usize,
    context: Vec<f64>,
    /// `states[k]` is the field after k applications of the train rule. `states[0]` is the input.
    states: Vec<Vec<f64>>,
}

struct CdtEpisodes {
    eps: Vec<Episode>,
}

impl CdtEpisodes {
    fn sample(&self, rng: &mut Xoshiro256StarStar) -> Episode {
        count_query(Store::Cdt, 1);
        self.eps[rng.gen_range(0..self.eps.len())].clone()
    }
}

fn episodes_from_train(train: &[Example]) -> Vec<Episode> {
    train
        .iter()
        .map(|e| {
            let mut states = Vec::with_capacity(EPISODE_H + 1);
            states.push(e.input.clone());
            for _ in 0..EPISODE_H {
                let nxt = apply_rule(e.family_id, e.params, states.last().unwrap());
                states.push(nxt);
            }
            debug_assert!(l2(&states[1], &e.expected) < 1e-9);
            Episode {
                fam: e.family_id,
                context: e.context.clone(),
                states,
            }
        })
        .collect()
}

/// Permute future frames inside each family. The past (`states[0]`) stays.
fn shuffle_futures(eps: &mut [Episode], seed: u64) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5F1F);
    for f in 0..N_FAM {
        let idx: Vec<usize> = eps
            .iter()
            .enumerate()
            .filter(|(_, e)| e.fam == f)
            .map(|(i, _)| i)
            .collect();
        for k in 1..=EPISODE_H {
            let mut frames: Vec<Vec<f64>> = idx.iter().map(|&i| eps[i].states[k].clone()).collect();
            for i in (1..frames.len()).rev() {
                frames.swap(i, rng.gen_range(0..=i));
            }
            for (j, &i) in idx.iter().enumerate() {
                eps[i].states[k] = frames[j].clone();
            }
        }
    }
}

fn episode_hash(eps: &[Episode]) -> String {
    let mut buf = Vec::new();
    for e in eps {
        buf.extend_from_slice(&(e.fam as u64).to_le_bytes());
        for s in &e.states {
            for v in s {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    sha256_hex(&buf)
}

/// Frames k≥2 whose field matches a sealed TEST target. Frame 1 is the
/// ordinary one-step TRAIN label already used by v4.
fn future_leak_frames(eps: &[Episode], aud: &Auditor) -> usize {
    let mut n = 0usize;
    for e in eps {
        for s in e.states.iter().skip(2) {
            if !aud.hits(s).is_empty() {
                n += 1;
            }
        }
    }
    n
}

fn future_of(ep: &Episode) -> Vec<Vec<f64>> {
    ep.states[1..=EPISODE_H].to_vec()
}

fn train_arm(
    residual: bool,
    store: &CdtEpisodes,
    seed: u64,
    steps: usize,
    rng_salt: u64,
    ck: Option<(std::path::PathBuf, String)>,
) -> (Dphi, f64, f64, u64) {
    let z = 2 * K_POINTS;
    let mut d = Dphi::gated_h(z, CTX_DIM, V5_HIDDEN, residual, seed ^ 0xD0);
    let before = snapshot();
    let nosave = || serde_json::Value::Null;
    let noload = |_: &serde_json::Value| {};
    let tc = ck.map(|(p, k)| crate::train_ckpt::TrainCkpt::new(p, k, &nosave, &noload));
    let (loss, max_g) = d.train_horizon_ck(
        budget(steps),
        seed ^ rng_salt,
        V5_LR_FINAL,
        &LOSS_AT,
        |rng| {
            let ep = store.sample(rng);
            (ep.states[0].clone(), ep.context.clone(), future_of(&ep))
        },
        tc.as_ref(),
    );
    let reads = snapshot().delta(&before).cdt;
    (d, loss, max_g, reads)
}

fn fam_rel_notes(d: &Dphi, test: &[Example]) -> String {
    let mut acc: Vec<Vec<f64>> = (0..N_FAM).map(|_| vec![]).collect();
    for e in test {
        let p = d.step(&e.input, &e.context);
        acc[e.family_id].push(m::rel_err(&p, &e.expected));
    }
    FAMILIES
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{n}:rel={:.4}", m::mean(&acc[i])))
        .collect::<Vec<_>>()
        .join(" ")
}

fn e33(
    seed: u64,
    label: &str,
    d: &Dphi,
    iter_ex: &[&Example],
    deltas: &[Vec<f64>],
) -> (usize, Vec<Row>, String) {
    let mut zs: Vec<Vec<f64>> = iter_ex.iter().map(|e| e.input.clone()).collect();
    let mut zp: Vec<Vec<f64>> = iter_ex
        .iter()
        .zip(deltas)
        .map(|(e, dl)| e.input.iter().zip(dl).map(|(a, b)| a + b).collect())
        .collect();
    let (mut region, mut contiguous) = (0usize, true);
    let (mut rows, mut notes) = (vec![], vec![]);
    for t in 1..=*HORIZONS.last().unwrap() {
        for (i, e) in iter_ex.iter().enumerate() {
            zs[i] = d.step(&zs[i], &e.context);
            zp[i] = d.step(&zp[i], &e.context);
        }
        if HORIZONS.contains(&t) {
            let (mut ok, mut re, mut dr, mut pg) = (0usize, vec![], vec![], vec![]);
            let (mut fam_ok, mut fam_n) = ([0usize; 2], [0usize; 2]);
            for (i, e) in iter_ex.iter().enumerate() {
                let y = target_h(e, t);
                let c = m::correct(&zs[i], &y);
                ok += c as usize;
                let fi = if e.family_id == 1 { 0 } else { 1 };
                fam_n[fi] += 1;
                fam_ok[fi] += c as usize;
                re.push(m::rel_err(&zs[i], &y).min(10.0));
                dr.push(m::norm(&zs[i]) / m::norm(&y).max(1e-12));
                pg.push(l2(&zp[i], &zs[i]) / m::norm(&deltas[i]).max(1e-12));
            }
            let acc = ok as f64 / iter_ex.len().max(1) as f64;
            if contiguous && acc >= 0.5 {
                region = t;
            } else {
                contiguous = false;
            }
            let (ar, af) = (
                fam_ok[0] as f64 / fam_n[0].max(1) as f64,
                fam_ok[1] as f64 / fam_n[1].max(1) as f64,
            );
            notes.push(format!(
                "h{t}:acc={acc:.3}(rot={ar:.2},refl={af:.2}),rel={:.3},norm={:.3},pert={:.3}",
                m::mean(&re),
                m::median(&dr),
                m::median(&pg)
            ));
            rows.push(Row {
                experiment: "V5_P2".into(),
                seed,
                condition: format!("{label}_h{t}"),
                partition: "TEST_iterable".into(),
                n_examples: iter_ex.len(),
                params: d.params(),
                compute: 0,
                accuracy: acc,
                cosine: m::median(&pg),
                energy: 0.0,
                manifold_distance: 0.0,
                stability: m::mean(&re),
                leakage: 0,
                q: QueryCounters::default(),
                status: "CLEAN".into(),
                notes: format!(
                    "norm_ratio={:.4} pert_growth={:.4} acc_rot={ar:.4} acc_refl={af:.4}",
                    m::median(&dr),
                    m::median(&pg)
                ),
            });
        }
    }
    (region, rows, notes.join(" "))
}

fn fresh_eval(opts: &RunOpts, s: &Sealed, seed: u64, d: &Dphi, tag: &str) -> (f64, f64) {
    let (Some(dir), Some(exe)) = (&opts.ckpt_dir, &opts.exe) else {
        return (f64::NAN, f64::NAN);
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
    (g("acc"), g("queries_total"))
}

#[derive(Clone, Copy)]
enum ArmKind {
    V5 { residual: bool, shuf: bool },
    E3,
}

struct Trained {
    name: String,
    d: Dphi,
    loss: f64,
    max_g: f64,
    reads: u64,
    ev: crate::field_autonomy_v4::EvalOut,
    train_s: f64,
    lat: f64,
    steps: usize,
    note: String,
}

/// Median single-step inference latency (µs) over TEST, 5 passes.
fn latency_us(d: &Dphi, test: &[Example]) -> f64 {
    let mut v = Vec::with_capacity(test.len() * 5);
    for _ in 0..5 {
        for e in test {
            let t = std::time::Instant::now();
            let p = d.step(&e.input, &e.context);
            std::hint::black_box(p);
            v.push(t.elapsed().as_secs_f64() * 1e6);
        }
    }
    m::median(&v)
}

pub struct SeedOut {
    pub rows: Vec<Row>,
    pub manifest_json: String,
}

pub fn run_seed(seed: u64, steps: usize, opts: &RunOpts) -> SeedOut {
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
    s.verify_test().expect("TEST sealed");
    let aud = Auditor::new(&s);
    let real = episodes_from_train(&s.ds.train);
    let leaks = future_leak_frames(&real, &aud);
    let real_hash = episode_hash(&real);
    let mut shuffled = real.clone();
    shuffle_futures(&mut shuffled, seed);
    let cdt_real = CdtEpisodes { eps: real };
    let cdt_shuf = CdtEpisodes { eps: shuffled };

    // Equal-terms comparison (docs/comparacion_v4_v5.md): same seed, same
    // sealed split, same net (gated_h H=136, init seed^0xD0), same audit and
    // the same E33 harness for every arm. `E3_v4` is main's §16 recipe
    // retrained here with the same optimizer-step budget; `P2_gated_eqc` is
    // v5 with steps/EPISODE_H so its Dφ forward+backward count matches E3.
    let arms: [(&str, ArmKind, u64, usize); 5] = [
        (
            "P2_gated",
            ArmKind::V5 {
                residual: true,
                shuf: false,
            },
            0x71,
            steps,
        ),
        (
            "P2_mlp",
            ArmKind::V5 {
                residual: false,
                shuf: false,
            },
            0x73,
            steps,
        ),
        (
            "P2_shuffled",
            ArmKind::V5 {
                residual: true,
                shuf: true,
            },
            0x72,
            steps,
        ),
        ("E3_v4", ArmKind::E3, 0x7A, steps),
        (
            "P2_gated_eqc",
            ArmKind::V5 {
                residual: true,
                shuf: false,
            },
            0x74,
            (steps / EPISODE_H).max(1),
        ),
    ];
    let mut trained: Vec<Trained> = vec![];
    for (name, kind, salt, st) in arms {
        let t0 = std::time::Instant::now();
        // Infra-only training checkpoint (see crate::train_ckpt).
        let ck = opts.train_ckpt.as_ref().map(|dir| {
            (
                dir.join(format!("0x{seed:X}_{name}.json")),
                format!("v5ck1 seed={seed} arm={name} steps={st} salt={salt}"),
            )
        });
        let (d, loss, max_g, reads, flags, note) = match kind {
            ArmKind::V5 { residual, shuf } => {
                let store = if shuf { &cdt_shuf } else { &cdt_real };
                let (d, l, g, r) = train_arm(residual, store, seed, st, salt, ck);
                (
                    d,
                    l,
                    g,
                    r,
                    AuditFlags::default(),
                    String::from("route=v5_horizon"),
                )
            }
            ArmKind::E3 => {
                let mut d = Dphi::gated_h(2 * K_POINTS, CTX_DIM, V5_HIDDEN, true, seed ^ 0xD0);
                let before = snapshot();
                let (f, mask) =
                    train_onestep_trusted_b_ck(&mut d, &s, seed, &aud, TRUST_TAU, budget(st), ck);
                let reads = snapshot().delta(&before).cdt;
                let ms: String = mask.iter().map(|b| if *b { '1' } else { '0' }).collect();
                (
                    d,
                    f64::NAN,
                    f64::NAN,
                    reads,
                    f,
                    format!("route=v4_trusted mask={ms}"),
                )
            }
        };
        let train_s = t0.elapsed().as_secs_f64();
        s.verify_test().expect("TEST sealed");
        let ev = evaluate(&d, &s.ds.test, 1, Some((&aud, &flags)), None);
        let lat = latency_us(&d, &s.ds.test);
        trained.push(Trained {
            name: name.into(),
            d,
            loss,
            max_g,
            reads,
            ev,
            train_s,
            lat,
            steps: st,
            note,
        });
    }
    // Buffer is not readable past this point.
    drop(cdt_real);
    drop(cdt_shuf);

    let fresh: Vec<(f64, f64, bool)> = trained
        .iter()
        .map(|t| {
            let (a, q) = fresh_eval(opts, &s, seed, &t.d, &t.name);
            (a, q, (a - t.ev.acc).abs() < 1e-9 && q == 0.0)
        })
        .collect();

    let mut rows = Vec::new();
    for t in &trained {
        let q_test = t.ev.queries.total();
        rows.push(Row {
            experiment: "V5_P2".into(),
            seed,
            condition: t.name.clone(),
            partition: "TEST".into(),
            n_examples: t.ev.n,
            params: t.d.params(),
            compute: t.steps,
            accuracy: t.ev.acc,
            cosine: t.ev.disp_cos,
            energy: t.ev.energy_ratio,
            manifold_distance: t.ev.manifold,
            stability: t.ev.rel_err,
            leakage: leaks + t.ev.leaked,
            q: t.ev.queries,
            status: if leaks == 0 && q_test == 0 && t.ev.leaked == 0 {
                "CLEAN".into()
            } else {
                "LEAKED".into()
            },
            notes: format!(
                "{} loss={:.6} max_g={:.4} train_cdt_reads={} q_test={q_test} train_s={:.2} lat_us={:.3} {} {}",
                t.note,
                t.loss,
                t.max_g,
                t.reads,
                t.train_s,
                t.lat,
                fam_rel_notes(&t.d, &s.ds.test),
                onestep_breakdown(&t.d, &s.ds.test)
            ),
        });
    }

    let iter_ex: Vec<&Example> =
        s.ds.test
            .iter()
            .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
            .collect();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xD33);
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

    for (i, t) in trained.iter().enumerate() {
        let label = format!("R_{}", t.name);
        let (region, rr, notes) = e33(seed, &label, &t.d, &iter_ex, &deltas);
        rows.extend(rr);
        let ev = &t.ev;
        let clean = leaks == 0 && ev.queries.total() == 0 && ev.leaked == 0;
        let (fp_acc, fp_q, fresh_ok) = fresh[i];
        let pass = region >= 8 && clean && fresh_ok;
        rows.push(Row {
            experiment: "V5_P2".into(),
            seed,
            condition: "GATE".into(),
            partition: label.clone(),
            n_examples: iter_ex.len(),
            params: t.d.params(),
            compute: t.steps,
            accuracy: ev.acc,
            cosine: 0.0,
            energy: 0.0,
            manifold_distance: 0.0,
            stability: ev.rel_err,
            leakage: leaks,
            q: ev.queries,
            status: if !clean {
                "LEAKED".into()
            } else if pass {
                "PASS".into()
            } else {
                "FAIL".into()
            },
            notes: format!(
                "{label} region={region} rel_err1={:.4} clean={clean} fresh_ok={fresh_ok} fresh_acc={fp_acc:.4}(q={fp_q:.0}) leaks={leaks} train_s={:.2} lat_us={:.3} {notes}",
                ev.rel_err, t.train_s, t.lat
            ),
        });
    }

    let manifest_json = format!(
        "{{\"seed\":\"0x{seed:X}\",\"n_episodes\":{},\"horizon\":{EPISODE_H},\"leak_frames\":{leaks},\"episode_sha256\":\"{real_hash}\",\"test_sha256\":\"{}\"}}",
        s.ds.train.len(),
        s.manifest.test_sha256
    );
    SeedOut {
        rows,
        manifest_json,
    }
}

pub fn csv_header() -> &'static str {
    CSV_HEADER
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4_dataset::DatasetConfig;

    #[test]
    fn horizon_grad_matches_finite_difference() {
        let mut d = Dphi::gated_h(2, 4, 4, true, 1);
        let x0 = vec![0.3, -0.2];
        let c = vec![1.0, 0.0, 0.2, -0.1];
        let future = vec![vec![0.25, -0.05], vec![0.15, 0.02]];
        let loss_at = [1usize, 2];
        let (l0, g) = d.horizon_grad(&x0, &c, &future, &loss_at, 1);
        assert!(l0.is_finite() && g.iter().all(|v| v.is_finite()));
        let eps = 1e-6;
        let mut p = d.net.flat();
        let i = 3usize;
        p[i] += eps;
        d.net.set_flat(&p);
        let (lp, _) = d.horizon_grad(&x0, &c, &future, &loss_at, 1);
        p[i] -= 2.0 * eps;
        d.net.set_flat(&p);
        let (lm, _) = d.horizon_grad(&x0, &c, &future, &loss_at, 1);
        let num = (lp - lm) / (2.0 * eps);
        let err = (num - g[i]).abs();
        assert!(err < 1e-4, "numeric {num} analytic {} err {err}", g[i]);
    }

    #[test]
    fn episode_step_one_matches_train_label() {
        let s = generate_and_seal(0x5A00, &DatasetConfig::standard(6));
        let eps = episodes_from_train(&s.ds.train);
        assert_eq!(eps.len(), s.ds.train.len());
        for (ep, e) in eps.iter().zip(&s.ds.train) {
            assert!(l2(&ep.states[1], &e.expected) < 1e-8);
            assert_eq!(ep.states.len(), EPISODE_H + 1);
        }
        let mut sh = eps.clone();
        shuffle_futures(&mut sh, 0x5A00);
        let changed = sh
            .iter()
            .zip(&eps)
            .any(|(a, b)| l2(&a.states[2], &b.states[2]) > 1e-6);
        assert!(changed);
    }
}
