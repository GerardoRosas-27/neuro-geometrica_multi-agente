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
    evaluate, target_h, Auditor, Row, RunOpts, CSV_HEADER, HORIZONS, ITERABLE_FAMS, K_POINTS,
};
use crate::v4_controls::{Dphi, Predictor, TrainBudget};
use crate::v4_dataset::{
    apply_rule, generate_and_seal, l2, sha256_hex, DatasetConfig, Example, Sealed, CTX_DIM,
    FAMILIES, N_FAM,
};
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
) -> (Dphi, f64, f64, u64) {
    let z = 2 * K_POINTS;
    let mut d = Dphi::gated_h(z, CTX_DIM, V5_HIDDEN, residual, seed ^ 0xD0);
    let before = snapshot();
    let (loss, max_g) = d.train_horizon(
        budget(steps),
        seed ^ rng_salt,
        V5_LR_FINAL,
        &LOSS_AT,
        |rng| {
            let ep = store.sample(rng);
            (ep.states[0].clone(), ep.context.clone(), future_of(&ep))
        },
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

fn fresh_eval(opts: &RunOpts, s: &Sealed, seed: u64, d: &Dphi) -> (f64, f64) {
    let (Some(dir), Some(exe)) = (&opts.ckpt_dir, &opts.exe) else {
        return (f64::NAN, f64::NAN);
    };
    std::fs::create_dir_all(dir).ok();
    let p = dir.join(format!("dphi_V5_P2_0x{seed:X}.json"));
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

    let arms = [
        ("P2_gated", true, false, 0x71u64),
        ("P2_mlp", false, false, 0x73),
        ("P2_shuffled", true, true, 0x72),
    ];
    let mut trained: Vec<(
        String,
        Dphi,
        f64,
        f64,
        u64,
        crate::field_autonomy_v4::EvalOut,
    )> = vec![];
    for (name, residual, shuf, salt) in arms {
        let store = if shuf { &cdt_shuf } else { &cdt_real };
        let (d, loss, max_g, reads) = train_arm(residual, store, seed, steps, salt);
        s.verify_test().expect("TEST sealed");
        let ev = evaluate(&d, &s.ds.test, 1, Some((&aud, &Default::default())), None);
        trained.push((name.into(), d, loss, max_g, reads, ev));
    }
    // Buffer is not readable past this point.
    drop(cdt_real);
    drop(cdt_shuf);

    let (fp_acc, fp_q) = fresh_eval(opts, &s, seed, &trained[0].1);
    let fresh_ok = (fp_acc - trained[0].5.acc).abs() < 1e-9 && fp_q == 0.0;

    let mut rows = Vec::new();
    for (name, d, loss, max_g, reads, ev) in &trained {
        let q_test = ev.queries.total();
        rows.push(Row {
            experiment: "V5_P2".into(),
            seed,
            condition: name.clone(),
            partition: "TEST".into(),
            n_examples: ev.n,
            params: d.params(),
            compute: steps,
            accuracy: ev.acc,
            cosine: ev.disp_cos,
            energy: ev.energy_ratio,
            manifold_distance: ev.manifold,
            stability: ev.rel_err,
            leakage: leaks + ev.leaked,
            q: ev.queries,
            status: if leaks == 0 && q_test == 0 {
                "CLEAN".into()
            } else {
                "LEAKED".into()
            },
            notes: format!(
                "loss={loss:.6} max_g={max_g:.4} train_cdt_reads={reads} q_test={q_test} {}",
                fam_rel_notes(d, &s.ds.test)
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

    let labels = ["R_P2_gated", "R_P2_mlp", "R_P2_shuffled"];
    for (i, label) in labels.iter().enumerate() {
        let (region, rr, notes) = e33(seed, label, &trained[i].1, &iter_ex, &deltas);
        rows.extend(rr);
        let ev = &trained[i].5;
        let clean = leaks == 0 && ev.queries.total() == 0 && ev.leaked == 0;
        let extra = if i == 0 { fresh_ok } else { true };
        let pass = region >= 8 && clean && extra;
        rows.push(Row {
            experiment: "V5_P2".into(),
            seed,
            condition: "GATE".into(),
            partition: (*label).into(),
            n_examples: iter_ex.len(),
            params: trained[i].1.params(),
            compute: steps,
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
                "{label} region={region} rel_err1={:.4} clean={clean} fresh_ok={fresh_ok} fresh_acc={fp_acc:.4}(q={fp_q:.0}) leaks={leaks} {notes}",
                ev.rel_err
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
