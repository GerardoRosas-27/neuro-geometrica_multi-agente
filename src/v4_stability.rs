//! Follow-up 2 — E33 stability variants on the v4 Dφ (pre-registered in
//! docs/preregistro_v4.md §9). Dφ is trained exactly like E35-A; only the
//! *rollout* operator changes:
//! * V0 baseline free-run (residual Dφ, as in E33);
//! * V1 manifold re-projection each step onto the similarity(+reflection)
//!   orbit of the input x0 (uses only x0 and the current state, never the target);
//! * V2 tangent spectral limit: δ ← δ / max(1, σ_max(J_step(z)));
//! * V3 V1+V2.
//! Non-saturating encoder: not applicable here (identity encoder).

#![allow(clippy::needless_range_loop)]

use crate::field_autonomy_v4::{target_h, triples, Row, HORIZONS, ITERABLE_FAMS, K_POINTS};
use crate::v4_controls::{Dphi, TrainBudget};
use crate::v4_dataset::{generate_and_seal, DatasetConfig, Example, CTX_DIM};
use crate::v4_metrics as m;
use crate::v4_provenance::QueryCounters;

/// Best similarity (rotation+uniform scale+translation, optionally after a
/// mirror of x0) mapping x0 onto z; returns the projected point set.
pub fn project_similarity(x0: &[f64], z: &[f64]) -> Vec<f64> {
    let k = x0.len() / 2;
    let mut best = (f64::MAX, z.to_vec());
    for mirror in [false, true] {
        let a: Vec<(f64, f64)> = (0..k)
            .map(|i| {
                (
                    x0[2 * i],
                    if mirror {
                        -x0[2 * i + 1]
                    } else {
                        x0[2 * i + 1]
                    },
                )
            })
            .collect();
        let (ax, ay) = (
            a.iter().map(|p| p.0).sum::<f64>() / k as f64,
            a.iter().map(|p| p.1).sum::<f64>() / k as f64,
        );
        let (bx, by) = (
            (0..k).map(|i| z[2 * i]).sum::<f64>() / k as f64,
            (0..k).map(|i| z[2 * i + 1]).sum::<f64>() / k as f64,
        );
        let (mut nr, mut ni, mut den) = (0.0, 0.0, 0.0);
        for i in 0..k {
            let (pr, pi) = (a[i].0 - ax, a[i].1 - ay);
            let (qr, qi) = (z[2 * i] - bx, z[2 * i + 1] - by);
            nr += qr * pr + qi * pi;
            ni += qi * pr - qr * pi;
            den += pr * pr + pi * pi;
        }
        let (sr, si) = (nr / den.max(1e-12), ni / den.max(1e-12));
        let mut out = vec![0.0; 2 * k];
        let mut res = 0.0;
        for i in 0..k {
            let (pr, pi) = (a[i].0 - ax, a[i].1 - ay);
            out[2 * i] = bx + sr * pr - si * pi;
            out[2 * i + 1] = by + sr * pi + si * pr;
            res += (out[2 * i] - z[2 * i]).powi(2) + (out[2 * i + 1] - z[2 * i + 1]).powi(2);
        }
        if res < best.0 {
            best = (res, out);
        }
    }
    best.1
}

fn step_variant(d: &Dphi, z: &[f64], c: &[f64], x0: &[f64], v: usize) -> (Vec<f64>, f64) {
    let raw = d.step(z, c);
    let sig = m::sigma_max(&m::jacobian(|w| d.step(w, c), z));
    let mut out = raw.clone();
    if v == 2 || v == 3 {
        let s = 1.0 / sig.max(1.0);
        out = z.iter().zip(&raw).map(|(a, b)| a + s * (b - a)).collect();
    }
    if v == 1 || v == 3 {
        out = project_similarity(x0, &out);
    }
    (out, sig)
}

pub const VARIANTS: [&str; 4] = [
    "V0_baseline",
    "V1_manifold_reproj",
    "V2_spectral_limit",
    "V3_reproj+spectral",
];

pub fn run(seed: u64) -> Vec<Row> {
    let k = K_POINTS;
    let s = generate_and_seal(seed, &DatasetConfig::standard(k));
    let mut a = Dphi::new(2 * k, CTX_DIM, seed ^ 0xD0);
    a.train_on(&triples(&s.ds.train), TrainBudget::lock(), seed ^ 0x7A);
    let iter_ex: Vec<&Example> =
        s.ds.test
            .iter()
            .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
            .collect();
    let mut rows = vec![];
    let mut regions = vec![];
    for (vi, vname) in VARIANTS.iter().enumerate() {
        let mut region = 0usize;
        let mut contiguous = true;
        let mut notes = vec![];
        let hmax = *HORIZONS.last().unwrap();
        // one rollout to h64 per example, scoring at each horizon
        let mut zs: Vec<Vec<f64>> = iter_ex.iter().map(|e| e.input.clone()).collect();
        let mut sig_all = vec![];
        let mut per_h = vec![];
        for t in 1..=hmax {
            for (i, e) in iter_ex.iter().enumerate() {
                let (z, sg) = step_variant(&a, &zs[i], &e.context, &e.input, vi);
                zs[i] = z;
                sig_all.push(sg);
            }
            if HORIZONS.contains(&t) {
                let (mut ok, mut re, mut dr, mut md) = (0usize, vec![], vec![], vec![]);
                for (i, e) in iter_ex.iter().enumerate() {
                    let y = target_h(e, t);
                    ok += m::correct(&zs[i], &y) as usize;
                    re.push(m::rel_err(&zs[i], &y).min(10.0));
                    dr.push(m::norm(&zs[i]) / m::norm(&y));
                    md.push(m::manifold_distance(&e.input, &zs[i]));
                }
                let acc = ok as f64 / iter_ex.len() as f64;
                per_h.push(acc);
                if contiguous && acc >= 0.5 {
                    region = t;
                } else {
                    contiguous = false;
                }
                notes.push(format!(
                    "h{t}:acc={acc:.3},rel={:.3},norm={:.3},manif={:.3}",
                    m::mean(&re),
                    m::median(&dr),
                    m::median(&md)
                ));
                rows.push(Row {
                    experiment: format!("E33S_{}", &vname[..2]),
                    seed,
                    condition: format!("{vname}_h{t}"),
                    partition: "TEST_iterable".into(),
                    n_examples: iter_ex.len(),
                    params: a.params(),
                    compute: 0,
                    accuracy: acc,
                    cosine: 0.0,
                    energy: m::median(&dr).powi(2),
                    manifold_distance: m::median(&md),
                    stability: m::mean(&re),
                    leakage: 0,
                    q: QueryCounters::default(),
                    status: "CLEAN".into(),
                    notes: String::new(),
                });
            }
        }
        regions.push(region);
        rows.push(Row {
            experiment: format!("E33S_{}", &vname[..2]),
            seed,
            condition: "GATE".into(),
            partition: vname.to_string(),
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
            status: if region >= 8 {
                "PASS".into()
            } else {
                "FAIL".into()
            },
            notes: format!(
                "{vname} region={region} median_sigma={:.3} {}",
                m::median(&sig_all),
                notes.join(" ")
            ),
        });
    }
    let _ = regions;
    rows
}
