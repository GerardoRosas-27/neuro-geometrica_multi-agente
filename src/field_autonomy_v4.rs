//! Field Autonomy v4 — E31–E44 harness (docs/plan_autonomia_campo_v4.md,
//! docs/protocolo_v4_agente.md, pre-registration docs/preregistro_v4.md).
//!
//! Architecture: periphery/encoder = identity on point coordinates (frozen,
//! no parameters), Dφ = residual dynamics, decoder = identity (independent,
//! parameter-free, therefore no decoder lookup is possible).
//!
//! E35 pipeline is physically split into `collect_experience`,
//! `consolidate` and `adapt_dynamics`; the CDT store is dropped before
//! TEST, and TEST runs under the thread-local provenance guard.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::v4_controls::{Dphi, LinearCtl, NnCtl, Predictor, StaticCtl, TrainBudget, Triple};
use crate::v4_dataset::{
    apply_rule, canonical_hash, canonical_points, context_of, generate_and_seal, l2, norm,
    sha256_hex, DatasetConfig, Example, ObjectClass, Partition, Sealed, CTX_DIM, FAMILIES,
    NEAR_DUP_REL, NEAR_DUP_TOL, N_FAM,
};
use crate::v4_metrics as m;
use crate::v4_provenance::{count_query, snapshot, Provenance, QueryCounters, Store};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const DEV_SEEDS_V4: [u64; 16] = [
    0xA400, 0xA401, 0xA402, 0xA403, 0xA404, 0xA405, 0xA406, 0xA407, 0xA408, 0xA409, 0xA40A, 0xA40B,
    0xA40C, 0xA40D, 0xA40E, 0xA40F,
];
pub const CONFIRM_SEEDS_V4: [u64; 16] = [
    0xB400, 0xB401, 0xB402, 0xB403, 0xB404, 0xB405, 0xB406, 0xB407, 0xB408, 0xB409, 0xB40A, 0xB40B,
    0xB40C, 0xB40D, 0xB40E, 0xB40F,
];
pub const SMOKE_SEED_V4: u64 = 0x5A00;

/// Pre-registered constants (docs/preregistro_v4.md §3). Frozen before DEV.
pub const K_POINTS: usize = 6;
pub const MARGIN: f64 = 0.05;
pub const REPLAY_FRACTION: f64 = 0.5;
pub const HORIZONS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
pub const ITERABLE_FAMS: [usize; 2] = [1, 2];
pub const EPSILONS: [f64; 5] = [0.001, 0.01, 0.05, 0.10, 0.20];
pub const E42_K: [usize; 5] = [4, 8, 16, 32, 64];
pub const E40_SEQ: [usize; 4] = [1, 4, 5, 6];

pub fn hp_lock_json() -> String {
    let b = TrainBudget::lock();
    format!(
        "{{\"k_points\":{},\"hidden\":{},\"steps\":{},\"batch\":{},\"lr\":{},\"margin\":{},\"replay_fraction\":{},\"rel_err_ok\":{},\"train_per_family\":24,\"dev_per_family\":12,\"test_per_family\":32}}",
        K_POINTS,
        crate::v4_controls::HIDDEN,
        b.steps,
        b.batch,
        b.lr,
        MARGIN,
        REPLAY_FRACTION,
        m::REL_ERR_OK
    )
}

pub fn hp_lock_hash() -> String {
    sha256_hex(hp_lock_json().as_bytes())
}

pub fn triples(xs: &[Example]) -> Vec<Triple> {
    xs.iter()
        .map(|e| (e.input.clone(), e.context.clone(), e.expected.clone()))
        .collect()
}

// ───────────────────────────── audit ─────────────────────────────

/// Holds TEST canonical forms; only the auditor (never a model) sees them.
pub struct Auditor {
    test_canon: HashSet<u64>,
    test_cp: Vec<(u64, Vec<f64>, f64)>,
    pub train_canon: HashSet<u64>,
    pub dev_canon: HashSet<u64>,
}

impl Auditor {
    pub fn new(s: &Sealed) -> Self {
        Self::with_test(s, &s.ds.test)
    }

    pub fn with_test(s: &Sealed, test: &[Example]) -> Self {
        Self {
            test_canon: test.iter().map(|e| e.canonical_target_hash).collect(),
            test_cp: test
                .iter()
                .map(|e| {
                    (
                        e.instance_id,
                        canonical_points(&e.expected),
                        norm(&e.expected),
                    )
                })
                .collect(),
            train_canon: s.ds.train.iter().map(|e| e.canonical_target_hash).collect(),
            dev_canon: s.ds.dev.iter().map(|e| e.canonical_target_hash).collect(),
        }
    }

    /// TEST instance ids whose target is exact/canonical/correctness-ball
    /// equivalent to `target`.
    pub fn hits(&self, target: &[f64]) -> Vec<u64> {
        let h = canonical_hash(target);
        let cp = canonical_points(target);
        self.test_cp
            .iter()
            .filter(|(_, t, n)| {
                t.len() == cp.len() && (l2(t, &cp) < NEAR_DUP_TOL.max(NEAR_DUP_REL * n))
            })
            .map(|(id, _, _)| *id)
            .chain(if self.test_canon.contains(&h) {
                Some(u64::MAX)
            } else {
                None
            })
            .collect()
    }
}

#[derive(Default, Clone)]
pub struct AuditFlags {
    pub cdt_canon: HashSet<u64>,
    pub equiv_hits: HashSet<u64>,
    pub any_equiv_global: bool,
}

// ───────────────────────────── evaluation ─────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct EvalOut {
    pub n: usize,
    pub acc: f64,
    pub rel_err: f64,
    pub disp_cos: f64,
    pub energy_ratio: f64,
    pub manifold: f64,
    pub fam_acc: Vec<f64>,
    pub fam_n: Vec<usize>,
    pub correct: Vec<bool>,
    pub queries: QueryCounters,
    pub leaked: usize,
    pub provenance: Vec<String>,
}

pub fn target_h(e: &Example, h: usize) -> Vec<f64> {
    let mut y = e.input.clone();
    for _ in 0..h {
        y = apply_rule(e.family_id, e.params, &y);
    }
    y
}

/// Sealed TEST evaluation under the provenance guard. Free-run for h>1:
/// the model's own output is fed back; the target is used only for metrics.
pub fn evaluate(
    model: &dyn Predictor,
    exs: &[Example],
    h: usize,
    aud: Option<(&Auditor, &AuditFlags)>,
    tag: Option<(&str, u64, &str)>,
) -> EvalOut {
    let mut out = EvalOut {
        fam_acc: vec![0.0; N_FAM],
        fam_n: vec![0; N_FAM],
        ..Default::default()
    };
    let (mut rels, mut dcs, mut ens, mut mds) = (vec![], vec![], vec![], vec![]);
    for e in exs {
        let before = snapshot();
        let mut z = e.input.clone();
        for _ in 0..h {
            z = model.predict(&z, &e.context);
        }
        let q = snapshot().delta(&before);
        let y = target_h(e, h);
        let ok = m::correct(&z, &y);
        rels.push(m::rel_err(&z, &y).min(10.0));
        dcs.push(m::disp_cosine(&e.input, &z, &y));
        ens.push(m::energy(&z) / m::energy(&y).max(1e-12));
        if z.iter().all(|v| v.is_finite()) {
            mds.push(m::manifold_distance(&e.input, &z));
        }
        out.correct.push(ok);
        out.fam_n[e.family_id] += 1;
        if ok {
            out.fam_acc[e.family_id] += 1.0;
        }
        out.queries.add(&q);
        let mut p = Provenance {
            queries: q,
            instance_id: e.instance_id,
            ..Default::default()
        };
        if let Some((a, f)) = aud {
            p.target_seen_training = a.train_canon.contains(&e.canonical_target_hash);
            p.target_seen_dev = a.dev_canon.contains(&e.canonical_target_hash);
            p.target_seen_cdt = f.cdt_canon.contains(&e.canonical_target_hash);
            p.target_equivalent_seen = f.equiv_hits.contains(&e.instance_id) || f.any_equiv_global;
        }
        p.target_seen_nn = false;
        p.retrieval_on = q.total() > 0;
        if p.leaked() {
            out.leaked += 1;
        }
        if let Some((exp, seed, cond)) = tag {
            p.experiment = exp.into();
            p.seed = seed;
            p.condition = cond.into();
            out.provenance.push(p.to_jsonl());
        }
    }
    out.n = exs.len();
    out.acc = out.correct.iter().filter(|b| **b).count() as f64 / out.n.max(1) as f64;
    for f in 0..N_FAM {
        out.fam_acc[f] = if out.fam_n[f] > 0 {
            out.fam_acc[f] / out.fam_n[f] as f64
        } else {
            f64::NAN
        };
    }
    out.rel_err = m::mean(&rels);
    out.disp_cos = m::mean(&dcs);
    out.energy_ratio = m::median(&ens);
    out.manifold = m::mean(&mds);
    out
}

// ───────────────────────────── CDT experience path ─────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Experience {
    pub fam: usize,
    pub params: [f64; 2],
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// CDT experience store. Every read is counted as a CDT query.
pub struct CdtStore {
    exps: Vec<Experience>,
}

impl CdtStore {
    pub fn read_all(&self) -> &[Experience] {
        count_query(Store::Cdt, self.exps.len() as u64);
        &self.exps
    }

    pub fn len(&self) -> usize {
        self.exps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exps.is_empty()
    }

    pub fn canonical_targets(&self) -> HashSet<u64> {
        self.exps.iter().map(|e| canonical_hash(&e.y)).collect()
    }

    /// Lookup (positive control only — E38 P3).
    pub fn nearest_answer(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        count_query(Store::Cdt, 1);
        let mut best = (f64::MAX, 0usize);
        for (i, e) in self.exps.iter().enumerate() {
            let cc = context_of(e.fam, e.params);
            let d = l2(&e.x, x).powi(2) + 4.0 * l2(&cc, c).powi(2);
            if d < best.0 {
                best = (d, i);
            }
        }
        let e = &self.exps[best.1];
        x.iter()
            .zip(&e.x)
            .zip(&e.y)
            .map(|((a, b), y)| a + (y - b))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpMode {
    Real,
    /// E36-E: same x/params, targets from unrelated fixed random maps.
    Irrelevant,
    /// E36-G: y shuffled among experiences of the same family.
    ShuffledOrder,
}

pub fn collect_experience(train: &[Example], mode: ExpMode, seed: u64) -> CdtStore {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0xC011);
    let mut exps: Vec<Experience> = train
        .iter()
        .map(|e| Experience {
            fam: e.family_id,
            params: e.params,
            x: e.input.clone(),
            y: e.expected.clone(),
        })
        .collect();
    match mode {
        ExpMode::Real => {}
        ExpMode::Irrelevant => {
            let maps: Vec<[f64; 6]> = (0..N_FAM)
                .map(|_| {
                    let mut a = [0.0; 6];
                    for v in &mut a {
                        *v = rng.gen_range(-0.4..0.4);
                    }
                    a
                })
                .collect();
            for e in &mut exps {
                let a = maps[e.fam];
                for i in 0..e.x.len() / 2 {
                    let (px, py) = (e.x[2 * i], e.x[2 * i + 1]);
                    e.y[2 * i] = px + a[0] * px + a[1] * py + a[4];
                    e.y[2 * i + 1] = py + a[2] * px + a[3] * py + a[5];
                }
            }
        }
        ExpMode::ShuffledOrder => {
            for f in 0..N_FAM {
                let idx: Vec<usize> = (0..exps.len()).filter(|&i| exps[i].fam == f).collect();
                let mut ys: Vec<Vec<f64>> = idx.iter().map(|&i| exps[i].y.clone()).collect();
                for i in (1..ys.len()).rev() {
                    ys.swap(i, rng.gen_range(0..=i));
                }
                for (j, &i) in idx.iter().enumerate() {
                    exps[i].y = ys[j].clone();
                }
            }
        }
    }
    CdtStore { exps }
}

/// Consolidated regularity for one family: pointwise affine rule whose six
/// entries are quadratic in the parameters, plus input/parameter statistics.
/// It contains no (input → answer) pairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FamilyRule {
    pub fam: usize,
    pub coef_x: Vec<f64>,
    pub coef_y: Vec<f64>,
    pub p_min: [f64; 2],
    pub p_max: [f64; 2],
    pub center_mean: [f64; 2],
    pub center_std: [f64; 2],
    pub r_min: f64,
    pub r_max: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningSignal {
    pub kind: String,
    pub rules: Vec<FamilyRule>,
    /// Item-1 generic consolidation (no generator structure), when enabled.
    pub generic: Option<GenericField>,
}

impl LearningSignal {
    pub fn n_numbers(&self) -> usize {
        match &self.generic {
            Some(g) => g.n_numbers(),
            None => self.rules.len() * (36 + 12),
        }
    }

    /// Consolidated predictor (teacher) — used by T1 and diagnostics.
    pub fn predict(&self, fam: usize, p: [f64; 2], x: &[f64]) -> Option<Vec<f64>> {
        if let Some(g) = &self.generic {
            if !g.fams.iter().any(|s| s.fam == fam) {
                return None;
            }
            return Some(g.apply(fam, p, x));
        }
        let r = self.rules.iter().find(|r| r.fam == fam)?;
        Some(rule_apply(r, p, x, self.kind.starts_with("StatsOnly")))
    }
}

/// Global switch for the consolidation family (set once by the runner).
pub static GENERIC_CONSOLIDATION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub const RFF_D: usize = 1600;
pub const RFF_LEN: f64 = 1.4;
pub const RFF_LAM: f64 = 1e-8;

fn env_or(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Generic consolidation: random-Fourier-feature kernel ridge on
/// `u = [q, context]` → Δq, shared across points. Knows nothing about
/// affine/rotation structure. Replay inputs: per-family independent Gaussian
/// per point (mean/std of experienced points) — no object-shape model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenericField {
    pub omega: Vec<f64>,
    pub phase: Vec<f64>,
    pub w: Vec<Vec<f64>>,
    pub use_q: bool,
    pub ctx_shift: bool,
    pub fams: Vec<GenericStats>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenericStats {
    pub fam: usize,
    pub p_min: [f64; 2],
    pub p_max: [f64; 2],
    pub mean: [f64; 2],
    pub std: [f64; 2],
}

impl GenericField {
    fn n_numbers(&self) -> usize {
        self.omega.len() + self.phase.len() + self.w.len() * 2 + self.fams.len() * 8
    }

    fn feats(&self, q: (f64, f64), c: &[f64]) -> Vec<f64> {
        let mut u = vec![
            if self.use_q { q.0 } else { 0.0 },
            if self.use_q { q.1 } else { 0.0 },
        ];
        u.extend_from_slice(c);
        let d = u.len();
        let mut f: Vec<f64> = (0..self.phase.len())
            .map(|i| {
                (2.0 / self.phase.len() as f64).sqrt()
                    * ((0..d).map(|j| self.omega[i * d + j] * u[j]).sum::<f64>() + self.phase[i])
                        .cos()
            })
            .collect();
        f.extend_from_slice(&u);
        f.push(1.0);
        f
    }

    pub fn apply(&self, fam: usize, p: [f64; 2], x: &[f64]) -> Vec<f64> {
        let nf = if self.ctx_shift {
            (fam + 1) % N_FAM
        } else {
            fam
        };
        let c = context_of(nf, p);
        let mut y = x.to_vec();
        for i in 0..x.len() / 2 {
            let f = self.feats((x[2 * i], x[2 * i + 1]), &c);
            for k in 0..2 {
                y[2 * i + k] += f.iter().zip(&self.w).map(|(a, w)| a * w[k]).sum::<f64>();
            }
        }
        y
    }
}

pub fn consolidate_generic(
    exps: &[Experience],
    mode: ConsolidateMode,
    seed: u64,
) -> LearningSignal {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x6E6E);
    let d = 2 + CTX_DIM;
    let nd = env_or("V4_RFF_D", RFF_D as f64) as usize;
    let len = env_or("V4_RFF_LEN", RFF_LEN);
    let lam = env_or("V4_RFF_LAM", RFF_LAM);
    let mut g = GenericField {
        omega: (0..nd * d)
            .map(|_| {
                let (u1, u2): (f64, f64) = (rng.gen_range(1e-12..1.0), rng.gen_range(0.0..1.0));
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos() / len
            })
            .collect(),
        phase: (0..nd)
            .map(|_| rng.gen_range(0.0..std::f64::consts::TAU))
            .collect(),
        w: vec![],
        use_q: mode != ConsolidateMode::StatsOnly,
        ctx_shift: false,
        fams: vec![],
    };
    let (mut feats, mut ys) = (vec![], vec![]);
    for e in exps {
        let c = context_of(e.fam, e.params);
        for i in 0..e.x.len() / 2 {
            feats.push(g.feats((e.x[2 * i], e.x[2 * i + 1]), &c));
            ys.push(vec![
                e.y[2 * i] - e.x[2 * i],
                e.y[2 * i + 1] - e.x[2 * i + 1],
            ]);
        }
    }
    g.w = crate::v4_controls::ridge(&feats, &ys, lam * feats.len() as f64);
    for f in 0..N_FAM {
        let fx: Vec<&Experience> = exps.iter().filter(|e| e.fam == f).collect();
        if fx.is_empty() {
            continue;
        }
        let (mut xs, mut yv) = (vec![], vec![]);
        let (mut pmin, mut pmax) = ([f64::MAX; 2], [f64::MIN; 2]);
        for e in &fx {
            for i in 0..e.x.len() / 2 {
                xs.push(e.x[2 * i]);
                yv.push(e.x[2 * i + 1]);
            }
            for k in 0..2 {
                pmin[k] = pmin[k].min(e.params[k]);
                pmax[k] = pmax[k].max(e.params[k]);
            }
        }
        g.fams.push(GenericStats {
            fam: f,
            p_min: pmin,
            p_max: pmax,
            mean: [m::mean(&xs), m::mean(&yv)],
            std: [m::std(&xs), m::std(&yv)],
        });
    }
    if mode == ConsolidateMode::Corrupt {
        g.ctx_shift = true;
    }
    LearningSignal {
        kind: format!("Generic{mode:?}"),
        rules: vec![],
        generic: Some(g),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsolidateMode {
    Full,
    /// E36-F: statistics without topology (translation part only, A = I).
    StatsOnly,
    /// E36-D: coefficients permuted across families.
    Corrupt,
}

fn basis(p: [f64; 2]) -> [f64; 6] {
    [1.0, p[0], p[1], p[0] * p[0], p[1] * p[1], p[0] * p[1]]
}

fn rule_feats(p: [f64; 2], q: (f64, f64), stats_only: bool) -> Vec<f64> {
    let g = basis(p);
    let mut f = Vec::with_capacity(18);
    for &b in &g {
        f.push(if stats_only { 0.0 } else { b * q.0 });
        f.push(if stats_only { 0.0 } else { b * q.1 });
        f.push(b);
    }
    f
}

pub fn consolidate(store: &CdtStore, mode: ConsolidateMode) -> LearningSignal {
    let exps = store.read_all();
    if GENERIC_CONSOLIDATION.load(std::sync::atomic::Ordering::Relaxed) {
        return consolidate_generic(exps, mode, 0x6E);
    }
    let mut rules = Vec::new();
    for f in 0..N_FAM {
        let fx: Vec<&Experience> = exps.iter().filter(|e| e.fam == f).collect();
        if fx.is_empty() {
            continue;
        }
        let so = mode == ConsolidateMode::StatsOnly;
        let (mut feats, mut tx, mut ty) = (vec![], vec![], vec![]);
        let (mut cs, mut rs) = (vec![], vec![]);
        let (mut pmin, mut pmax) = ([f64::MAX; 2], [f64::MIN; 2]);
        for e in &fx {
            let k = e.x.len() / 2;
            let (mut cx, mut cy) = (0.0, 0.0);
            for i in 0..k {
                let q = (e.x[2 * i], e.x[2 * i + 1]);
                feats.push(rule_feats(e.params, q, so));
                // StatsOnly learns displacement on top of identity.
                tx.push(vec![e.y[2 * i] - if so { q.0 } else { 0.0 }]);
                ty.push(vec![e.y[2 * i + 1] - if so { q.1 } else { 0.0 }]);
                cx += q.0;
                cy += q.1;
            }
            cx /= k as f64;
            cy /= k as f64;
            cs.push([cx, cy]);
            for i in 0..k {
                rs.push(((e.x[2 * i] - cx).powi(2) + (e.x[2 * i + 1] - cy).powi(2)).sqrt());
            }
            for d in 0..2 {
                pmin[d] = pmin[d].min(e.params[d]);
                pmax[d] = pmax[d].max(e.params[d]);
            }
        }
        let lam = 1e-4 * feats.len() as f64;
        let wx = crate::v4_controls::ridge(&feats, &tx, lam);
        let wy = crate::v4_controls::ridge(&feats, &ty, lam);
        let cxs: Vec<f64> = cs.iter().map(|c| c[0]).collect();
        let cys: Vec<f64> = cs.iter().map(|c| c[1]).collect();
        rules.push(FamilyRule {
            fam: f,
            coef_x: wx.iter().map(|r| r[0]).collect(),
            coef_y: wy.iter().map(|r| r[0]).collect(),
            p_min: pmin,
            p_max: pmax,
            center_mean: [m::mean(&cxs), m::mean(&cys)],
            center_std: [m::std(&cxs), m::std(&cys)],
            r_min: rs.iter().cloned().fold(f64::MAX, f64::min),
            r_max: rs.iter().cloned().fold(f64::MIN, f64::max),
        });
    }
    let mut kind = format!("{mode:?}");
    if mode == ConsolidateMode::Corrupt && rules.len() > 1 {
        let n = rules.len();
        let cx: Vec<Vec<f64>> = rules.iter().map(|r| r.coef_x.clone()).collect();
        let cy: Vec<Vec<f64>> = rules.iter().map(|r| r.coef_y.clone()).collect();
        for i in 0..n {
            rules[i].coef_x = cx[(i + 1) % n].clone();
            rules[i].coef_y = cy[(i + 1) % n].clone();
        }
        kind.push_str("(perm+1)");
    }
    LearningSignal {
        kind,
        rules,
        generic: None,
    }
}

pub fn rule_apply(r: &FamilyRule, p: [f64; 2], x: &[f64], stats_only: bool) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() / 2 {
        let q = (x[2 * i], x[2 * i + 1]);
        let f = rule_feats(p, q, stats_only);
        let a: f64 = f.iter().zip(&r.coef_x).map(|(u, v)| u * v).sum();
        let b: f64 = f.iter().zip(&r.coef_y).map(|(u, v)| u * v).sum();
        y[2 * i] = a + if stats_only { q.0 } else { 0.0 };
        y[2 * i + 1] = b + if stats_only { q.1 } else { 0.0 };
    }
    y
}

/// Generative replay from the learning signal (fresh synthetic states;
/// never a stored experience).
pub fn replay_sample(sig: &LearningSignal, rng: &mut Xoshiro256StarStar, k: usize) -> Triple {
    if let Some(g) = &sig.generic {
        let st = &g.fams[rng.gen_range(0..g.fams.len())];
        let p = [
            rng.gen_range(st.p_min[0]..=st.p_max[0]),
            rng.gen_range(st.p_min[1]..=st.p_max[1]),
        ];
        let x: Vec<f64> = (0..2 * k)
            .map(|i| st.mean[i % 2] + st.std[i % 2] * rng.gen_range(-1.7..1.7))
            .collect();
        let y = g.apply(st.fam, p, &x);
        return (x, context_of(st.fam, p), y);
    }
    let r = &sig.rules[rng.gen_range(0..sig.rules.len())];
    let p = [
        rng.gen_range(r.p_min[0]..=r.p_max[0]),
        rng.gen_range(r.p_min[1]..=r.p_max[1]),
    ];
    let cx = r.center_mean[0] + r.center_std[0] * rng.gen_range(-1.7..1.7);
    let cy = r.center_mean[1] + r.center_std[1] * rng.gen_range(-1.7..1.7);
    let mut angs: Vec<f64> = (0..k)
        .map(|_| rng.gen_range(0.0..std::f64::consts::TAU))
        .collect();
    angs.sort_by(|a, b| a.total_cmp(b));
    let mut x = vec![0.0; 2 * k];
    for i in 0..k {
        let rr = rng.gen_range(r.r_min..=r.r_max.max(r.r_min + 1e-9));
        x[2 * i] = cx + rr * angs[i].cos();
        x[2 * i + 1] = cy + rr * angs[i].sin();
    }
    let y = rule_apply(r, p, &x, sig.kind.starts_with("StatsOnly"));
    (x, context_of(r.fam, p), y)
}

/// Generative pseudo-experiences from a consolidated signal (used when the
/// original episodes were already deleted, e.g. E45 phase 2).
pub fn pseudo_experiences(sig: &LearningSignal, n: usize, k: usize, seed: u64) -> Vec<Example> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x95E0);
    (0..n)
        .map(|i| {
            let (x, c, y) = replay_sample(sig, &mut rng, k);
            let fam = (0..N_FAM).find(|&f| c[f] > 0.5).unwrap_or(0);
            let params = [c[N_FAM], c[N_FAM + 1]];
            Example {
                rule_id: fam,
                family_id: fam,
                instance_id: u64::MAX - i as u64,
                object_class: ObjectClass::Blob,
                params,
                canonical_target_hash: canonical_hash(&y),
                input: x,
                context: c,
                expected: y,
                partition: Partition::Train,
                subset: "pseudo",
            }
        })
        .collect()
}

/// Adapt Dφ with the same budget as the control: each sample is drawn from
/// TRAIN with prob 1-REPLAY_FRACTION, else from consolidated replay. All
/// replay targets are passed to the auditor.
pub fn adapt_dynamics(
    d: &mut Dphi,
    train: &[Triple],
    sig: Option<&LearningSignal>,
    raw: Option<&CdtStore>,
    budget: TrainBudget,
    seed: u64,
    k: usize,
    aud: &Auditor,
    flags: &mut AuditFlags,
) {
    let raw_exps: Option<Vec<Triple>> = raw.map(|s| {
        s.read_all()
            .iter()
            .map(|e| (e.x.clone(), context_of(e.fam, e.params), e.y.clone()))
            .collect()
    });
    let n = train.len();
    let mut hits: Vec<u64> = vec![];
    d.train_with(budget, seed, |r| {
        if r.gen::<f64>() < REPLAY_FRACTION {
            if let Some(s) = sig {
                let t = replay_sample(s, r, k);
                hits.extend(aud.hits(&t.2));
                return t;
            }
            if let Some(re) = &raw_exps {
                return re[r.gen_range(0..re.len())].clone();
            }
        }
        train[r.gen_range(0..n)].clone()
    });
    for h in hits {
        if h == u64::MAX {
            flags.any_equiv_global = true;
        } else {
            flags.equiv_hits.insert(h);
        }
    }
}

// ───────────────────────────── rows ─────────────────────────────

#[derive(Clone, Debug)]
pub struct Row {
    pub experiment: String,
    pub seed: u64,
    pub condition: String,
    pub partition: String,
    pub n_examples: usize,
    pub params: usize,
    pub compute: usize,
    pub accuracy: f64,
    pub cosine: f64,
    pub energy: f64,
    pub manifold_distance: f64,
    pub stability: f64,
    pub leakage: usize,
    pub q: QueryCounters,
    pub status: String,
    pub notes: String,
}

pub const CSV_HEADER: &str = "experiment,seed,condition,partition,n_examples,params,compute,accuracy,cosine,energy,manifold_distance,stability,leakage,cdt_queries,rqm_queries,table_queries,nn_queries,attractor_queries,direct_memory_queries,status,notes";

impl Row {
    pub fn csv(&self) -> String {
        format!(
            "{},0x{:X},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{},\"{}\"",
            self.experiment,
            self.seed,
            self.condition,
            self.partition,
            self.n_examples,
            self.params,
            self.compute,
            self.accuracy,
            self.cosine,
            self.energy,
            self.manifold_distance,
            self.stability,
            self.leakage,
            self.q.cdt,
            self.q.rqm,
            self.q.table,
            self.q.nn,
            self.q.attractor,
            self.q.direct_memory,
            self.status,
            self.notes.replace('"', "'")
        )
    }
}

fn row(
    exp: &str,
    seed: u64,
    cond: &str,
    part: &str,
    ev: &EvalOut,
    params: usize,
    compute: usize,
) -> Row {
    Row {
        experiment: exp.into(),
        seed,
        condition: cond.into(),
        partition: part.into(),
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
        notes: String::new(),
    }
}

fn gate(exp: &str, seed: u64, pass: bool, notes: String) -> Row {
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

// ───────────────────────────── per-seed run ─────────────────────────────

pub struct SeedOut {
    pub rows: Vec<Row>,
    pub provenance: Vec<String>,
    pub manifest_json: String,
    pub e32_trace: Vec<String>,
}

fn fam_str(v: &[f64]) -> String {
    v.iter()
        .map(|x| {
            if x.is_nan() {
                "-".into()
            } else {
                format!("{x:.2}")
            }
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// E35 core for one dataset: returns (A, B, eval A, eval B, flags, signal).
pub struct E35Out {
    pub a: Dphi,
    pub b: Dphi,
    pub ev_a: EvalOut,
    pub ev_b: EvalOut,
    pub flags: AuditFlags,
    pub signal_numbers: usize,
    pub cdt_size: usize,
    pub cdt_dropped: bool,
    pub signal_json: String,
}

pub fn run_e35_core(s: &Sealed, seed: u64, k: usize, exp: &str, prov: bool) -> E35Out {
    let budget = TrainBudget::lock();
    let aud = Auditor::new(s);
    let tr = triples(&s.ds.train);
    // 3. control A (same init seed as B)
    let mut a = Dphi::new(2 * k, CTX_DIM, seed ^ 0xD0);
    a.train_on(&tr, budget, seed ^ 0x7A);
    // 4–7. experience → CDT → learning signal → adapt B
    let mut flags = AuditFlags::default();
    let mut b = Dphi::new(2 * k, CTX_DIM, seed ^ 0xD0);
    let (signal_numbers, cdt_size, signal_json);
    {
        let cdt = collect_experience(&s.ds.train, ExpMode::Real, seed);
        flags.cdt_canon = cdt.canonical_targets();
        let sig = consolidate(&cdt, ConsolidateMode::Full);
        signal_numbers = sig.n_numbers();
        cdt_size = cdt.len();
        signal_json = serde_json::to_string(&sig).unwrap_or_default();
        adapt_dynamics(
            &mut b,
            &tr,
            Some(&sig),
            None,
            budget,
            seed ^ 0x7A,
            k,
            &aud,
            &mut flags,
        );
        // 9. DROP CDT / learning signal / (no RAM/RQM/index objects exist in this path)
        drop(sig);
        drop(cdt);
    }
    // 10–13. clean evaluation, counters checked per prediction
    s.verify_test().expect("TEST sealed");
    let tag_a = if prov {
        Some((exp, seed, "A_control"))
    } else {
        None
    };
    let tag_b = if prov {
        Some((exp, seed, "B_cdt_experience"))
    } else {
        None
    };
    let ev_a = evaluate(
        &a,
        &s.ds.test,
        1,
        Some((&aud, &AuditFlags::default())),
        tag_a,
    );
    let ev_b = evaluate(&b, &s.ds.test, 1, Some((&aud, &flags)), tag_b);
    E35Out {
        a,
        b,
        ev_a,
        ev_b,
        flags,
        signal_numbers,
        cdt_size,
        cdt_dropped: true,
        signal_json,
    }
}

pub struct RunOpts {
    pub exps: HashSet<String>,
    pub ckpt_dir: Option<std::path::PathBuf>,
    pub exe: Option<std::path::PathBuf>,
}

impl RunOpts {
    fn on(&self, e: &str) -> bool {
        self.exps.is_empty() || self.exps.contains(e)
    }
}

pub fn run_seed(seed: u64, opts: &RunOpts) -> SeedOut {
    let k = K_POINTS;
    let budget = TrainBudget::lock();
    let compute = budget.examples_seen();
    let cfg = DatasetConfig::standard(k);
    let s = generate_and_seal(seed, &cfg);
    let mut rows = vec![];
    let mut prov = vec![];
    let mut trace = vec![];
    let manifest_json = s.manifest.to_json();
    if s.manifest.audit_status != "CLEAN" {
        rows.push(gate("DATASET", seed, false, "DATASET_INVALID".into()));
        return SeedOut {
            rows,
            provenance: prov,
            manifest_json,
            e32_trace: trace,
        };
    }
    let aud = Auditor::new(&s);
    let noflags = AuditFlags::default();
    let tr = triples(&s.ds.train);

    if opts.exps.len() == 1 && opts.on("E33S") {
        rows.extend(crate::v4_stability::run(seed));
        return SeedOut {
            rows,
            provenance: prov,
            manifest_json,
            e32_trace: trace,
        };
    }
    if opts.exps.len() == 1 && opts.on("E45") {
        rows.extend(crate::v4_e45::run(seed));
        return SeedOut {
            rows,
            provenance: prov,
            manifest_json,
            e32_trace: trace,
        };
    }
    // E35 (also provides Dφ_A for E31–E34)
    let e35 = run_e35_core(&s, seed, k, "E35", true);
    let a = &e35.a;

    // ── E31 rule learning ──
    if opts.on("E31") {
        let mut sh = tr.clone();
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5F);
        for i in (1..sh.len()).rev() {
            let j = rng.gen_range(0..=i);
            let t = sh[i].2.clone();
            sh[i].2 = sh[j].2.clone();
            sh[j].2 = t;
        }
        let mut shuf = Dphi::new(2 * k, CTX_DIM, seed ^ 0xD0);
        shuf.label = "SHUFFLED_LABEL".into();
        shuf.train_on(&sh, budget, seed ^ 0x7A);
        let rnd = Dphi::random_dynamics(2 * k, CTX_DIM, seed ^ 0xD0);
        let lin = LinearCtl::fit(&tr);
        let nn = NnCtl::new(&tr);
        let ev_d = &e35.ev_a;
        let ctrls: Vec<(&dyn Predictor, usize)> = vec![
            (&StaticCtl, 0),
            (&lin, 0),
            (&rnd, 0),
            (&shuf, compute),
            (&nn, 0),
        ];
        rows.push(
            row("E31", seed, "DPHI", "TEST", ev_d, a.params(), compute)
                .with_notes(fam_str(&ev_d.fam_acc)),
        );
        let mut best = [0.0f64; N_FAM];
        let mut best_acc = 0.0f64;
        for (c, comp) in &ctrls {
            let ev = evaluate(*c, &s.ds.test, 1, Some((&aud, &noflags)), None);
            for f in 0..N_FAM {
                best[f] = best[f].max(ev.fam_acc[f]);
            }
            best_acc = best_acc.max(ev.acc);
            let mut r = row("E31", seed, c.name(), "TEST", &ev, c.params(), *comp)
                .with_notes(fam_str(&ev.fam_acc));
            if c.name() == "NN_LOOKUP" {
                r.status = "LOOKUP_CONTROL".into();
            }
            rows.push(r);
        }
        let won: Vec<&str> = (0..N_FAM)
            .filter(|&f| ev_d.fam_acc[f] >= best[f] + MARGIN)
            .map(|f| FAMILIES[f])
            .collect();
        let pass = won.len() >= 3 && ev_d.leaked == 0;
        rows.push(gate(
            "E31",
            seed,
            pass,
            format!(
                "families_won={} [{}] dphi_acc={:.3} best_ctrl_acc={:.3}",
                won.len(),
                won.join(" "),
                ev_d.acc,
                best_acc
            ),
        ));
        let dev_ev = evaluate(a, &s.ds.dev, 1, None, None);
        rows.push(row(
            "E31",
            seed,
            "DPHI",
            "DEV",
            &dev_ev,
            a.params(),
            compute,
        ));
    }

    // ── E32 Jacobian / stability ──
    if opts.on("E32") {
        let mut ok_fams = 0;
        let mut notes = vec![];
        for f in 0..N_FAM {
            let mut ratios = vec![];
            let (mut sd, mut st) = (vec![], vec![]);
            for e in s.ds.test.iter().filter(|e| e.family_id == f) {
                let jd = m::jacobian(|z| a.step(z, &e.context), &e.input);
                let jt = m::jacobian(|z| apply_rule(f, e.params, z), &e.input);
                let (a1, t1) = (m::sigma_max(&jd), m::sigma_max(&jt));
                sd.push(a1);
                st.push(t1);
                ratios.push((a1 - t1).abs() / t1.max(1e-9));
            }
            let med = m::median(&ratios);
            if med <= 0.10 {
                ok_fams += 1;
            }
            notes.push(format!(
                "{}:σD={:.3}/σT={:.3}",
                FAMILIES[f],
                m::median(&sd),
                m::median(&st)
            ));
        }
        // per-step trace on rotate family (h=1..64)
        let rot: Vec<&Example> = s.ds.test.iter().filter(|e| e.family_id == 1).collect();
        let mut sig_gt1 = 0usize;
        let mut sig_all = 0usize;
        for t in 1..=64usize {
            let (mut nz, mut nd, mut cs, mut en, mut md, mut te, mut sg) =
                (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
            let mut pert: Vec<Vec<f64>> = vec![vec![]; EPSILONS.len()];
            for e in &rot {
                let zprev = a.rollout(&e.input, &e.context, t - 1);
                let z = a.step(&zprev, &e.context);
                let y = target_h(e, t);
                let yprev = target_h(e, t - 1);
                nz.push(m::norm(&z) / m::norm(&y));
                nd.push(l2(&z, &zprev));
                cs.push(m::cosine(&z, &y));
                en.push(m::energy(&z) / m::energy(&y));
                md.push(m::manifold_distance(&e.input, &z));
                te.push(m::rel_err(&a.step(&yprev, &e.context), &y));
                let sgv = m::sigma_max(&m::jacobian(|w| a.step(w, &e.context), &zprev));
                sg.push(sgv);
                sig_all += 1;
                if sgv > 1.0 {
                    sig_gt1 += 1;
                }
                if [1usize, 8, 32, 64].contains(&t) {
                    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ e.instance_id);
                    let mut u: Vec<f64> = (0..2 * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
                    let un = m::norm(&u);
                    u.iter_mut().for_each(|v| *v /= un);
                    for (ei, eps) in EPSILONS.iter().enumerate() {
                        let x2: Vec<f64> =
                            e.input.iter().zip(&u).map(|(a, b)| a + eps * b).collect();
                        let z2 = a.rollout(&x2, &e.context, t);
                        pert[ei].push(l2(&z2, &z) / eps);
                    }
                }
            }
            let ps: Vec<String> = pert
                .iter()
                .map(|p| {
                    if p.is_empty() {
                        "-".into()
                    } else {
                        format!("{:.3}", m::median(p))
                    }
                })
                .collect();
            trace.push(format!(
                "0x{:X},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                seed,
                t,
                m::median(&nz),
                m::median(&nd),
                m::median(&cs),
                m::median(&en),
                m::median(&md),
                m::median(&te),
                m::median(&sg),
                ps.join(";")
            ));
        }
        let pass = ok_fams >= 3;
        rows.push(gate(
            "E32",
            seed,
            pass,
            format!(
                "fams_sigma_within10%={ok_fams} frac_steps_sigma>1={:.3} {}",
                sig_gt1 as f64 / sig_all.max(1) as f64,
                notes.join(" ")
            ),
        ));
    }

    // ── E33 free-run ──
    if opts.on("E33") {
        let mut region = 0usize;
        let mut contiguous = true;
        let iter_ex: Vec<Example> =
            s.ds.test
                .iter()
                .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
                .cloned()
                .collect();
        let mut accs = vec![];
        for &h in &HORIZONS {
            let ev = evaluate(a, &s.ds.test, h, None, None);
            let evi = evaluate(a, &iter_ex, h, None, None);
            accs.push(format!("h{h}:{:.2}/{:.2}", evi.acc, ev.acc));
            let mut r = row(
                "E33",
                seed,
                &format!("DPHI_h{h}"),
                "TEST",
                &ev,
                a.params(),
                compute,
            );
            r.notes = format!("iterable_acc={:.3} fam={}", evi.acc, fam_str(&ev.fam_acc));
            rows.push(r);
            if contiguous && evi.acc >= 0.5 {
                region = h;
            } else {
                contiguous = false;
            }
        }
        let pass = region >= 8;
        rows.push(gate(
            "E33",
            seed,
            pass,
            format!(
                "stability_region_h={region} iterable/all {}",
                accs.join(" ")
            ),
        ));
    }

    // ── E34 paired static vs dynamic ──
    if opts.on("E34") {
        let mut md = Dphi::mlp_direct(2 * k, CTX_DIM, seed ^ 0xD0);
        md.train_on(&tr, budget, seed ^ 0x7A);
        let iter_ex: Vec<Example> =
            s.ds.test
                .iter()
                .filter(|e| ITERABLE_FAMS.contains(&e.family_id))
                .cloned()
                .collect();
        let d1 = &e35.ev_a;
        let m1 = evaluate(&md, &s.ds.test, 1, None, None);
        let d4 = evaluate(a, &iter_ex, 4, None, None);
        let m4 = evaluate(&md, &iter_ex, 4, None, None);
        let st = evaluate(&StaticCtl, &s.ds.test, 1, None, None);
        rows.push(row("E34", seed, "DPHI_h1", "TEST", d1, a.params(), compute));
        rows.push(row(
            "E34",
            seed,
            "MLP_DIRECT_h1",
            "TEST",
            &m1,
            md.params(),
            compute,
        ));
        rows.push(row(
            "E34",
            seed,
            "DPHI_h4_iter",
            "TEST",
            &d4,
            a.params(),
            compute,
        ));
        rows.push(row(
            "E34",
            seed,
            "MLP_DIRECT_h4_iter",
            "TEST",
            &m4,
            md.params(),
            compute,
        ));
        rows.push(row("E34", seed, "STATIC", "TEST", &st, 0, 0));
        let pass = d1.acc >= m1.acc + MARGIN && d4.acc >= m4.acc && d1.acc >= st.acc + MARGIN;
        rows.push(gate(
            "E34",
            seed,
            pass,
            format!(
                "dphi_h1={:.3} mlp_h1={:.3} dphi_h4={:.3} mlp_h4={:.3} static={:.3} params={}/{}",
                d1.acc,
                m1.acc,
                d4.acc,
                m4.acc,
                st.acc,
                a.params(),
                md.params()
            ),
        ));
    }

    // ── E35 gate ──
    if opts.on("E35") {
        let (ea, eb) = (&e35.ev_a, &e35.ev_b);
        let mut ra = row("E35", seed, "A_control", "TEST", ea, a.params(), compute);
        ra.notes = format!("fam={}", fam_str(&ea.fam_acc));
        let mut rb = row(
            "E35",
            seed,
            "B_cdt_experience",
            "TEST",
            eb,
            e35.b.params(),
            compute,
        );
        rb.notes = format!(
            "fam={} cdt_size={} signal_numbers={} cdt_dropped={} equiv_hits={} any_equiv={}",
            fam_str(&eb.fam_acc),
            e35.cdt_size,
            e35.signal_numbers,
            e35.cdt_dropped,
            e35.flags.equiv_hits.len(),
            e35.flags.any_equiv_global
        );
        rows.push(ra);
        rows.push(rb);
        prov.extend(ea.provenance.iter().cloned());
        prov.extend(eb.provenance.iter().cloned());
        let target_seen_cdt =
            s.ds.test
                .iter()
                .any(|e| e35.flags.cdt_canon.contains(&e.canonical_target_hash));
        let equiv = !e35.flags.equiv_hits.is_empty() || e35.flags.any_equiv_global;
        let clean = eb.queries.total() == 0
            && ea.queries.total() == 0
            && eb.leaked == 0
            && ea.leaked == 0
            && !target_seen_cdt
            && !equiv;
        let pass = eb.acc >= ea.acc + MARGIN && clean;
        rows.push(gate("E35", seed, pass, format!("A={:.4} B={:.4} diff={:.4} relerrA={:.4} relerrB={:.4} cdt_q={} rqm_q={} table_q={} nn_q={} attr_q={} direct_q={} leaked={} target_seen_cdt={} target_equivalent_seen={}", ea.acc, eb.acc, eb.acc - ea.acc, ea.rel_err, eb.rel_err, eb.queries.cdt, eb.queries.rqm, eb.queries.table, eb.queries.nn, eb.queries.attractor, eb.queries.direct_memory, eb.leaked + ea.leaked, target_seen_cdt, equiv)));
        let dev_a = evaluate(a, &s.ds.dev, 1, None, None);
        let dev_b = evaluate(&e35.b, &s.ds.dev, 1, None, None);
        rows.push(row(
            "E35",
            seed,
            "A_control",
            "DEV",
            &dev_a,
            a.params(),
            compute,
        ));
        rows.push(row(
            "E35",
            seed,
            "B_cdt_experience",
            "DEV",
            &dev_b,
            a.params(),
            compute,
        ));
    }

    // ── E36 consolidation ablation ──
    if opts.on("E36") {
        let conds: [(&str, ExpMode, Option<ConsolidateMode>); 5] = [
            ("B_raw_experience", ExpMode::Real, None),
            (
                "D_corrupt_cdt",
                ExpMode::Real,
                Some(ConsolidateMode::Corrupt),
            ),
            (
                "E_irrelevant_cdt",
                ExpMode::Irrelevant,
                Some(ConsolidateMode::Full),
            ),
            (
                "F_stats_no_topology",
                ExpMode::Real,
                Some(ConsolidateMode::StatsOnly),
            ),
            (
                "G_shuffled_order",
                ExpMode::ShuffledOrder,
                Some(ConsolidateMode::Full),
            ),
        ];
        let c_acc = e35.ev_b.acc;
        rows.push(row(
            "E36",
            seed,
            "A_no_experience",
            "TEST",
            &e35.ev_a,
            a.params(),
            compute,
        ));
        rows.push(row(
            "E36",
            seed,
            "C_consolidated",
            "TEST",
            &e35.ev_b,
            a.params(),
            compute,
        ));
        let mut others = vec![];
        let mut raw_acc = 0.0;
        for (name, em, cm) in conds {
            let cdt = collect_experience(&s.ds.train, em, seed);
            let mut fl = AuditFlags {
                cdt_canon: cdt.canonical_targets(),
                ..Default::default()
            };
            let mut d = Dphi::new(2 * k, CTX_DIM, seed ^ 0xD0);
            match cm {
                Some(mode) => {
                    let sig = consolidate(&cdt, mode);
                    adapt_dynamics(
                        &mut d,
                        &tr,
                        Some(&sig),
                        None,
                        budget,
                        seed ^ 0x7A,
                        k,
                        &aud,
                        &mut fl,
                    );
                }
                None => adapt_dynamics(
                    &mut d,
                    &tr,
                    None,
                    Some(&cdt),
                    budget,
                    seed ^ 0x7A,
                    k,
                    &aud,
                    &mut fl,
                ),
            }
            drop(cdt);
            let ev = evaluate(&d, &s.ds.test, 1, Some((&aud, &fl)), None);
            if name.starts_with("B_raw") {
                raw_acc = ev.acc;
            } else {
                others.push((name, ev.acc));
            }
            rows.push(row("E36", seed, name, "TEST", &ev, d.params(), compute));
        }
        let best_other = others.iter().map(|o| o.1).fold(0.0, f64::max);
        let pass = c_acc >= raw_acc + MARGIN && c_acc >= best_other + MARGIN;
        rows.push(gate(
            "E36",
            seed,
            pass,
            format!(
                "C={c_acc:.3} B_raw={raw_acc:.3} A={:.3} {}",
                e35.ev_a.acc,
                others
                    .iter()
                    .map(|(n, v)| format!("{n}={v:.3}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        ));
    }

    // ── E37 / E38: delete-after-learning + restart in a fresh process ──
    if opts.on("E37") || opts.on("E38") {
        if let (Some(dir), Some(exe)) = (&opts.ckpt_dir, &opts.exe) {
            std::fs::create_dir_all(dir).ok();
            let p_b = dir.join(format!("dphi_B_0x{seed:X}.json"));
            let p_cdt = dir.join(format!("cdt_0x{seed:X}.json"));
            std::fs::write(&p_b, e35.b.to_ckpt()).ok();
            // CDT serialised separately (P1/P3) — then the in-process objects are gone.
            let cdt_tmp = collect_experience(&s.ds.train, ExpMode::Real, seed);
            std::fs::write(&p_cdt, serde_json::to_string(&cdt_tmp.exps).unwrap()).ok();
            drop(cdt_tmp);
            let run = |mode: &str| -> Option<serde_json::Value> {
                let o = std::process::Command::new(exe)
                    .args([
                        "eval-checkpoint",
                        "--ckpt",
                        p_b.to_str()?,
                        "--cdt",
                        p_cdt.to_str()?,
                        "--seed",
                        &format!("{seed}"),
                        "--mode",
                        mode,
                        "--test-sha",
                        &s.manifest.test_sha256,
                    ])
                    .output()
                    .ok()?;
                let txt = String::from_utf8_lossy(&o.stdout);
                serde_json::from_str(txt.lines().last()?).ok()
            };
            let pre = e35.ev_b.acc;
            let a_acc = e35.ev_a.acc;
            let p0 = run("P0");
            let get = |v: &Option<serde_json::Value>, k: &str| {
                v.as_ref().and_then(|x| x[k].as_f64()).unwrap_or(f64::NAN)
            };
            let p0_acc = get(&p0, "acc");
            let p0_q = get(&p0, "queries_total");
            if opts.on("E37") {
                let mut r = gate(
                    "E37",
                    seed,
                    (p0_acc - pre).abs() <= 0.01 && p0_acc >= a_acc + MARGIN && p0_q == 0.0,
                    String::new(),
                );
                r.notes = format!("pre_delete={pre:.4} post_delete_fresh_process={p0_acc:.4} A={a_acc:.4} fresh_process_queries={p0_q} ckpt_sha={}", e35.b.weights_hash());
                rows.push(r);
            }
            if opts.on("E38") {
                let p0n = run("P0_NEWTEST");
                let p1 = run("P1");
                let p3 = run("P3");
                let a_new = {
                    let t2 = new_test(&s, seed);
                    evaluate(a, &t2, 1, None, None).acc
                };
                let (p0n_acc, p1_acc, p3_acc) =
                    (get(&p0n, "acc"), get(&p1, "acc"), get(&p3, "acc"));
                let pass = p0_acc >= a_acc + MARGIN
                    && p0n_acc >= a_new + MARGIN
                    && (p1_acc - p0_acc).abs() < 1e-12
                    && get(&p1, "queries_total") == 0.0
                    && get(&p0n, "queries_total") == 0.0;
                rows.push(gate("E38", seed, pass, format!("P0={p0_acc:.4} P0_newtest={p0n_acc:.4} A_newtest={a_new:.4} P1={p1_acc:.4}(q={}) P2=P0(identity encoder) P3_lookup={p3_acc:.4}(cdt_q={}) A={a_acc:.4}", get(&p1, "queries_total"), get(&p3, "queries_total"))));
            }
        }
    }

    // ── E39 transfer circle → triangle ──
    if opts.on("E39") {
        let mut cfg2 = DatasetConfig::standard(k);
        cfg2.train_class = ObjectClass::Circle;
        cfg2.test_class = ObjectClass::Triangle;
        let s2 = generate_and_seal(seed ^ 0x39, &cfg2);
        if s2.manifest.audit_status == "CLEAN" {
            let o = run_e35_core(&s2, seed, k, "E39", false);
            let st = evaluate(&StaticCtl, &s2.ds.test, 1, None, None);
            rows.push(row(
                "E39",
                seed,
                "A_no_consolidation",
                "TEST_triangle",
                &o.ev_a,
                o.a.params(),
                compute,
            ));
            rows.push(row(
                "E39",
                seed,
                "B_consolidated",
                "TEST_triangle",
                &o.ev_b,
                o.b.params(),
                compute,
            ));
            let clean = o.ev_b.leaked == 0 && o.ev_b.queries.total() == 0;
            rows.push(gate(
                "E39",
                seed,
                o.ev_b.acc >= o.ev_a.acc + MARGIN && o.ev_b.acc > st.acc && clean,
                format!(
                    "A={:.3} B={:.3} static={:.3}",
                    o.ev_a.acc, o.ev_b.acc, st.acc
                ),
            ));
        } else {
            rows.push(gate("E39", seed, false, "DATASET_INVALID".into()));
        }
    }

    // ── E40 continual ──
    if opts.on("E40") {
        rows.extend(run_e40(&s, seed, k, &aud));
    }

    // ── E41 causal intervention + rollback ──
    if opts.on("E41") {
        rows.extend(run_e41(&s, seed, &e35.b));
    }

    // ── E42 scaling ──
    if opts.on("E42") {
        let mut all = true;
        let mut notes = vec![];
        for &kk in &E42_K {
            let cfgk = DatasetConfig::standard(kk);
            let sk = generate_and_seal(seed ^ (0x4200 + kk as u64), &cfgk);
            if sk.manifest.audit_status != "CLEAN" {
                all = false;
                notes.push(format!("N{}:INVALID", 2 * kk));
                continue;
            }
            let o = run_e35_core(&sk, seed, kk, "E42", false);
            let clean = o.ev_b.leaked == 0 && o.ev_b.queries.total() == 0;
            all &= o.ev_b.acc >= o.ev_a.acc + MARGIN && clean;
            notes.push(format!(
                "N{}:A={:.3},B={:.3}",
                2 * kk,
                o.ev_a.acc,
                o.ev_b.acc
            ));
            rows.push(row(
                "E42",
                seed,
                &format!("A_N{}", 2 * kk),
                "TEST",
                &o.ev_a,
                o.a.params(),
                compute,
            ));
            rows.push(row(
                "E42",
                seed,
                &format!("B_N{}", 2 * kk),
                "TEST",
                &o.ev_b,
                o.b.params(),
                compute,
            ));
        }
        rows.push(gate("E42", seed, all, notes.join(" ")));
    }

    SeedOut {
        rows,
        provenance: prov,
        manifest_json,
        e32_trace: trace,
    }
}

impl Row {
    fn with_notes(mut self, n: String) -> Self {
        self.notes = n;
        self
    }
}

/// Fresh TEST' for E38 (new stream, audited against TRAIN/DEV/TEST).
pub fn new_test(s: &Sealed, seed: u64) -> Vec<Example> {
    let mut cfg = s.ds.config.clone();
    cfg.train_per_family = 0;
    cfg.dev_per_family = 0;
    let d2 = crate::v4_dataset::generate(seed ^ 0x3838_3838, &cfg);
    let seen: HashSet<u64> =
        s.ds.train
            .iter()
            .chain(&s.ds.dev)
            .map(|e| e.canonical_target_hash)
            .collect();
    d2.test
        .into_iter()
        .filter(|e| !seen.contains(&e.canonical_target_hash))
        .map(|mut e| {
            e.partition = Partition::Test;
            e
        })
        .collect()
}

fn run_e40(s: &Sealed, seed: u64, k: usize, aud: &Auditor) -> Vec<Row> {
    let total = TrainBudget::lock();
    let block = TrainBudget {
        steps: total.steps / E40_SEQ.len(),
        ..total
    };
    let test: Vec<Example> =
        s.ds.test
            .iter()
            .filter(|e| E40_SEQ.contains(&e.family_id))
            .cloned()
            .collect();
    let mut rows = vec![];
    let mut res: Vec<(String, Vec<Vec<f64>>)> = vec![];
    for cond in ["A_sequential", "B_cdt_replay", "D_corrupt_replay"] {
        let mut d = Dphi::new(2 * k, CTX_DIM, seed ^ 0xD0);
        let mut seen_exps: Vec<Example> = vec![];
        let mut hist = vec![];
        let mut fl = AuditFlags::default();
        for (bi, &f) in E40_SEQ.iter().enumerate() {
            let blk: Vec<Example> =
                s.ds.train
                    .iter()
                    .filter(|e| e.family_id == f)
                    .cloned()
                    .collect();
            seen_exps.extend(blk.iter().cloned());
            let tr = triples(&blk);
            if cond == "A_sequential" || bi == 0 {
                d.train_on(&tr, block, seed ^ (0x40 + bi as u64));
            } else {
                // CDT persists during training only; consolidated over all blocks so far.
                let cdt = collect_experience(&seen_exps, ExpMode::Real, seed);
                let mode = if cond == "B_cdt_replay" {
                    ConsolidateMode::Full
                } else {
                    ConsolidateMode::Corrupt
                };
                let sig = consolidate(&cdt, mode);
                adapt_dynamics(
                    &mut d,
                    &tr,
                    Some(&sig),
                    None,
                    block,
                    seed ^ (0x40 + bi as u64),
                    k,
                    aud,
                    &mut fl,
                );
            }
            let ev = evaluate(&d, &test, 1, None, None);
            hist.push(E40_SEQ.iter().map(|&g| ev.fam_acc[g]).collect::<Vec<f64>>());
        }
        let ev = evaluate(&d, &test, 1, Some((aud, &fl)), None);
        let mut r = row(
            "E40",
            seed,
            cond,
            "TEST",
            &ev,
            d.params(),
            total.examples_seen(),
        );
        r.notes = format!("final_per_rule={}", fam_str(&hist[3]));
        rows.push(r);
        res.push((cond.to_string(), hist));
    }
    let forget = |h: &Vec<Vec<f64>>| -> f64 {
        m::mean(&(0..3).map(|i| h[i][i] - h[3][i]).collect::<Vec<_>>())
    };
    let fin = |h: &Vec<Vec<f64>>| -> f64 { m::mean(&h[3]) };
    let (a, b, dd) = (&res[0].1, &res[1].1, &res[2].1);
    let pass =
        fin(b) >= fin(a) + MARGIN && forget(b) <= forget(a) - MARGIN && fin(b) >= fin(dd) + MARGIN;
    rows.push(gate(
        "E40",
        seed,
        pass,
        format!(
            "final A={:.3} B={:.3} D={:.3} forgetting A={:.3} B={:.3} D={:.3}",
            fin(a),
            fin(b),
            fin(dd),
            forget(a),
            forget(b),
            forget(dd)
        ),
    ));
    rows
}

fn run_e41(s: &Sealed, seed: u64, base: &Dphi) -> Vec<Row> {
    let rot: Vec<Example> =
        s.ds.test
            .iter()
            .filter(|e| e.family_id == 1)
            .cloned()
            .collect();
    let as_translate: Vec<Example> = rot
        .iter()
        .map(|e| {
            let mut e2 = e.clone();
            e2.family_id = 0;
            e2.expected = apply_rule(0, e.params, &e.input);
            e2
        })
        .collect();
    let tr_test: Vec<Example> =
        s.ds.test
            .iter()
            .filter(|e| e.family_id == 0)
            .cloned()
            .collect();
    let base_hash = base.weights_hash();
    let base_rot = evaluate(base, &rot, 1, None, None).acc;
    let base_tr = evaluate(base, &tr_test, 1, None, None).acc;
    // structured: swap the family-one-hot input columns translate(0) <-> rotate(1)
    let mut st = base.clone();
    let din = st.net.din;
    let mut delta2 = 0.0;
    for j in 0..st.net.dh {
        let (c0, c1) = (crate::v4_controls::PD, crate::v4_controls::PD + 1);
        let (v0, v1) = (st.net.w1[j * din + c0], st.net.w1[j * din + c1]);
        st.net.w1[j * din + c0] = v1;
        st.net.w1[j * din + c1] = v0;
        delta2 += 2.0 * (v0 - v1).powi(2);
    }
    // inputs with rotate context: does Dφ now produce translate(p)?
    let ctx_rot: Vec<Example> = as_translate
        .iter()
        .map(|e| {
            let mut e2 = e.clone();
            e2.context = context_of(1, e.params);
            e2
        })
        .collect();
    let redirect = |d: &Dphi| -> f64 {
        let ok = ctx_rot
            .iter()
            .filter(|e| m::correct(&d.step(&e.input, &e.context), &e.expected))
            .count();
        ok as f64 / ctx_rot.len().max(1) as f64
    };
    let st_redirect = redirect(&st);
    let st_rot = evaluate(&st, &rot, 1, None, None).acc;
    // random control: same Frobenius magnitude on random w1 entries
    let mut rd = base.clone();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x41);
    let n = rd.net.w1.len();
    let idx: Vec<usize> = (0..2 * rd.net.dh).map(|_| rng.gen_range(0..n)).collect();
    let per = (delta2 / idx.len() as f64).sqrt();
    for &i in &idx {
        rd.net.w1[i] += if rng.gen::<bool>() { per } else { -per };
    }
    let rd_redirect = redirect(&rd);
    let rd_rot = evaluate(&rd, &rot, 1, None, None).acc;
    // rollback = reload checkpoint bytes
    let rb = Dphi::from_ckpt(&base.to_ckpt()).expect("ckpt");
    let rb_ok = rb.weights_hash() == base_hash
        && (evaluate(&rb, &rot, 1, None, None).acc - base_rot).abs() < 1e-12;
    let pass = st_redirect >= 0.5 * base_tr && rd_redirect <= 0.10 && rb_ok && base_tr > 0.0;
    vec![gate("E41", seed, pass, format!("base_rot={base_rot:.3} base_translate={base_tr:.3} structured: redirect={st_redirect:.3} rot_acc={st_rot:.3} | random(|Δ|F equal): redirect={rd_redirect:.3} rot_acc={rd_rot:.3} | rollback_exact={rb_ok}"))]
}

/// Fresh-process evaluator (E37/E38). Receives only checkpoint paths.
pub fn eval_checkpoint_cli(args: &[String]) -> i32 {
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let seed: u64 = get("--seed").and_then(|v| v.parse().ok()).unwrap_or(0);
    let mode = get("--mode").unwrap_or_else(|| "P0".into());
    let ck = get("--ckpt").unwrap_or_default();
    let d: Dphi = match std::fs::read_to_string(&ck)
        .ok()
        .and_then(|t| Dphi::from_ckpt(&t))
    {
        Some(d) => d,
        None => {
            println!("{{\"error\":\"ckpt\"}}");
            return 2;
        }
    };
    let k = d.zdim / 2;
    let s = generate_and_seal(seed, &DatasetConfig::standard(k));
    if let Some(sha) = get("--test-sha") {
        if sha != s.manifest.test_sha256 {
            println!("{{\"error\":\"test_sha_mismatch\"}}");
            return 3;
        }
    }
    let before = snapshot();
    let test = if mode == "P0_NEWTEST" {
        new_test(&s, seed)
    } else {
        s.ds.test.clone()
    };
    let acc = match mode.as_str() {
        "P1" => {
            // CDT loaded, retrieval OFF: never read.
            let _cdt: Vec<Experience> = get("--cdt")
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default();
            evaluate(&d, &test, 1, None, None).acc
        }
        "P3" => {
            let exps: Vec<Experience> = get("--cdt")
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default();
            let store = CdtStore { exps };
            struct Look<'a>(&'a CdtStore);
            impl Predictor for Look<'_> {
                fn name(&self) -> &str {
                    "CDT_LOOKUP"
                }
                fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
                    self.0.nearest_answer(x, c)
                }
                fn params(&self) -> usize {
                    0
                }
            }
            evaluate(&Look(&store), &test, 1, None, None).acc
        }
        _ => evaluate(&d, &test, 1, None, None).acc,
    };
    let q = snapshot().delta(&before);
    println!(
        "{{\"mode\":\"{}\",\"acc\":{},\"n\":{},\"queries_total\":{},\"cdt_queries\":{},\"ckpt_sha\":\"{}\",\"pid\":{}}}",
        mode,
        acc,
        test.len(),
        q.total(),
        q.cdt,
        d.weights_hash(),
        std::process::id()
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_e35_guard_counts_cdt_and_drops_store() {
        let cfg = DatasetConfig::standard(4);
        let s = generate_and_seal(SMOKE_SEED_V4, &cfg);
        let cdt = collect_experience(&s.ds.train, ExpMode::Real, 1);
        let before = snapshot();
        let sig = consolidate(&cdt, ConsolidateMode::Full);
        assert_eq!(snapshot().delta(&before).cdt, cdt.len() as u64);
        assert_eq!(sig.rules.len(), N_FAM);
        // learning signal is coefficients only (no pairs)
        assert_eq!(sig.n_numbers(), N_FAM * 48);
        drop(cdt);
        let d = Dphi::new(8, CTX_DIM, 1);
        let ev = evaluate(&d, &s.ds.test, 1, None, None);
        assert_eq!(ev.queries.total(), 0);
    }
}
