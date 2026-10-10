//! E45 — Liquid-vs-Thermo consolidation benchmark (plan v4 §24,
//! preregistro §7). Same 96 experiences, two phases, episodes/CDT deleted
//! before TEST, same audit counters for every brain.

#![allow(clippy::needless_range_loop)]

use crate::field_autonomy_v4::{
    adapt_dynamics, collect_experience, consolidate, evaluate, rule_apply, triples, AuditFlags,
    Auditor, ConsolidateMode, EvalOut, ExpMode, LearningSignal, Row, K_POINTS, MARGIN,
};
use crate::v4_controls::{Adam, Dphi, Predictor, StaticCtl, TrainBudget, Triple, PD};
use crate::v4_dataset::{generate_and_seal, DatasetConfig, Example, CTX_DIM};
use crate::v4_metrics as m;
use crate::v4_provenance::QueryCounters;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::time::Instant;

pub const RANK: usize = 4;
const SET1: [usize; 4] = [0, 1, 2, 3];

/// L2: frozen base Dφ + fast low-rank memory on the output layer (W2 + U Vᵀ, b2 + δ).
#[derive(Clone)]
pub struct LowRank {
    pub base: Dphi,
    pub u: Vec<f64>, // dout × r
    pub v: Vec<f64>, // dh × r
    pub db: Vec<f64>,
}

impl LowRank {
    pub fn new(base: Dphi, seed: u64) -> Self {
        let mut r = Xoshiro256StarStar::seed_from_u64(seed);
        let dh = base.net.dh;
        let dout = base.net.dout;
        Self {
            u: vec![0.0; dout * RANK],
            v: (0..dh * RANK).map(|_| r.gen_range(-0.1..0.1)).collect(),
            db: vec![0.0; dout],
            base,
        }
    }

    pub fn n_fast(&self) -> usize {
        self.u.len() + self.v.len() + self.db.len()
    }

    fn point(&self, q: &[f64], c: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut inp = q.to_vec();
        inp.extend_from_slice(c);
        let (h, mut o) = self.base.net.forward(&inp);
        let dh = h.len();
        let vh: Vec<f64> = (0..RANK)
            .map(|k| (0..dh).map(|j| self.v[j * RANK + k] * h[j]).sum())
            .collect();
        for i in 0..o.len() {
            o[i] +=
                self.db[i] + (0..RANK).map(|k| self.u[i * RANK + k] * vh[k]).sum::<f64>() + q[i];
        }
        (h, vh, o)
    }

    pub fn rollback(&mut self) {
        self.u.iter_mut().for_each(|x| *x = 0.0);
        self.db.iter_mut().for_each(|x| *x = 0.0);
    }

    pub fn train(&mut self, data: &[Triple], b: TrainBudget, seed: u64) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let n = self.n_fast();
        let mut opt = Adam::new(n, b.lr);
        let dh = self.base.net.dh;
        for _ in 0..b.steps {
            let mut g = vec![0.0; n];
            for _ in 0..b.batch {
                let (x, c, y) = &data[rng.gen_range(0..data.len())];
                for p in 0..x.len() / PD {
                    let (h, vh, o) = self.point(&x[p * PD..(p + 1) * PD], c);
                    for i in 0..PD {
                        let d = 2.0 * (o[i] - y[p * PD + i]) / (x.len() * b.batch) as f64;
                        for k in 0..RANK {
                            g[i * RANK + k] += d * vh[k];
                            for j in 0..dh {
                                g[self.u.len() + j * RANK + k] += d * self.u[i * RANK + k] * h[j];
                            }
                        }
                        g[self.u.len() + self.v.len() + i] += d;
                    }
                }
            }
            let mut p: Vec<f64> = self
                .u
                .iter()
                .chain(&self.v)
                .chain(&self.db)
                .cloned()
                .collect();
            opt.step(&mut p, &g);
            let (a, bb) = (self.u.len(), self.v.len());
            self.u.copy_from_slice(&p[..a]);
            self.v.copy_from_slice(&p[a..a + bb]);
            self.db.copy_from_slice(&p[a + bb..]);
        }
    }
}

impl Predictor for LowRank {
    fn name(&self) -> &str {
        "L2_LOWRANK"
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        (0..x.len() / PD)
            .flat_map(|p| self.point(&x[p * PD..(p + 1) * PD], c).2)
            .collect()
    }
    fn params(&self) -> usize {
        self.base.params() + self.n_fast()
    }
}

/// T1: inference directly from the consolidated CDT rules (episodes deleted).
pub struct Thermo {
    pub sig: LearningSignal,
}

impl Predictor for Thermo {
    fn name(&self) -> &str {
        "T1_CDT"
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        let fam = (0..c.len() - 2).find(|&i| c[i] > 0.5).unwrap_or(0);
        match self.sig.rules.iter().find(|r| r.fam == fam) {
            Some(r) => rule_apply(r, [c[c.len() - 2], c[c.len() - 1]], x, false),
            None => x.to_vec(),
        }
    }
    fn params(&self) -> usize {
        self.sig.n_numbers()
    }
}

fn changed(a: &Dphi, b: &Dphi) -> usize {
    a.net
        .flat()
        .iter()
        .zip(b.net.flat())
        .filter(|(x, y)| (*x - y).abs() > 0.0)
        .count()
}

fn latency_us(p: &dyn Predictor, xs: &[Example]) -> f64 {
    let t = Instant::now();
    for e in xs {
        std::hint::black_box(p.predict(&e.input, &e.context));
    }
    t.elapsed().as_secs_f64() * 1e6 / xs.len().max(1) as f64
}

struct Brain {
    name: &'static str,
    ev: EvalOut,
    set1_p1: f64,
    set1_p2: f64,
    set2: f64,
    retention: f64,
    cons_ms: f64,
    changed: usize,
    bytes: usize,
    lat_us: f64,
    updates: usize,
}

pub fn run(seed: u64) -> Vec<Row> {
    let k = K_POINTS;
    let mut cfg = DatasetConfig::standard(k);
    cfg.train_per_family = 12;
    let s = generate_and_seal(seed ^ 0x45, &cfg);
    let aud = Auditor::new(&s);
    let lock = TrainBudget::lock();
    let half = TrainBudget {
        steps: lock.steps / 2,
        ..lock
    };
    let p1: Vec<Example> =
        s.ds.train
            .iter()
            .filter(|e| SET1.contains(&e.family_id))
            .cloned()
            .collect();
    let p2: Vec<Example> =
        s.ds.train
            .iter()
            .filter(|e| !SET1.contains(&e.family_id))
            .cloned()
            .collect();
    let t1: Vec<Example> =
        s.ds.test
            .iter()
            .filter(|e| SET1.contains(&e.family_id))
            .cloned()
            .collect();
    let t2: Vec<Example> =
        s.ds.test
            .iter()
            .filter(|e| !SET1.contains(&e.family_id))
            .cloned()
            .collect();
    // auditor's private copy of the (deleted) episodes for retention
    let episodes: Vec<Example> = s.ds.train.clone();
    let acc = |p: &dyn Predictor, xs: &[Example]| evaluate(p, xs, 1, None, None).acc;
    let mut brains: Vec<Brain> = vec![];
    let init = Dphi::new(2 * k, CTX_DIM, seed ^ 0xD0);

    // L1 — liquid, error-gated direct consolidation
    let mut l1 = init.clone();
    let t = Instant::now();
    let mut upd = l1.train_gated(&triples(&p1), half, seed ^ 0x71, m::REL_ERR_OK);
    let l1_p1 = l1.clone();
    let a11 = acc(&l1, &t1);
    upd += l1.train_gated(&triples(&p2), half, seed ^ 0x72, m::REL_ERR_OK);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    brains.push(Brain {
        name: "L1_liquid_gated",
        ev: evaluate(
            &l1,
            &s.ds.test,
            1,
            Some((&aud, &AuditFlags::default())),
            None,
        ),
        set1_p1: a11,
        set1_p2: acc(&l1, &t1),
        set2: acc(&l1, &t2),
        retention: acc(&l1, &episodes),
        cons_ms: ms,
        changed: changed(&init, &l1),
        bytes: l1.params() * 8,
        lat_us: latency_us(&l1, &s.ds.test),
        updates: upd,
    });

    // L2 — long-term = L1 after phase 1; phase 2 into fast UVᵀ only
    let mut l2 = LowRank::new(l1_p1.clone(), seed ^ 0x2F);
    let t = Instant::now();
    l2.train(&triples(&p2), half, seed ^ 0x72);
    let ms2 = t.elapsed().as_secs_f64() * 1e3;
    let mut rb = l2.clone();
    rb.rollback();
    let rollback_ok = (acc(&rb, &t1) - a11).abs() < 1e-12;
    brains.push(Brain {
        name: "L2_liquid_lowrank",
        ev: evaluate(
            &l2,
            &s.ds.test,
            1,
            Some((&aud, &AuditFlags::default())),
            None,
        ),
        set1_p1: a11,
        set1_p2: acc(&l2, &t1),
        set2: acc(&l2, &t2),
        retention: acc(&l2, &episodes),
        cons_ms: brains[0].cons_ms / 2.0 + ms2,
        changed: changed(&init, &l1_p1) + l2.n_fast(),
        bytes: l2.params() * 8,
        lat_us: latency_us(&l2, &s.ds.test),
        updates: half.examples_seen(),
    });

    // T1 — CDT inference + consolidation (episodes deleted after each phase)
    let t = Instant::now();
    let cdt1 = collect_experience(&p1, ExpMode::Real, seed);
    let sig1 = consolidate(&cdt1, ConsolidateMode::Full);
    drop(cdt1);
    let th1 = Thermo { sig: sig1 };
    let a_t1 = acc(&th1, &t1);
    let cdt2 = collect_experience(&p2, ExpMode::Real, seed);
    let mut sig = consolidate(&cdt2, ConsolidateMode::Full);
    drop(cdt2);
    sig.rules.extend(th1.sig.rules.iter().cloned());
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let th = Thermo { sig };
    brains.push(Brain {
        name: "T1_thermo_cdt",
        ev: evaluate(
            &th,
            &s.ds.test,
            1,
            Some((&aud, &AuditFlags::default())),
            None,
        ),
        set1_p1: a_t1,
        set1_p2: acc(&th, &t1),
        set2: acc(&th, &t2),
        retention: acc(&th, &episodes),
        cons_ms: ms,
        changed: th.params(),
        bytes: th.params() * 8,
        lat_us: latency_us(&th, &s.ds.test),
        updates: episodes.len(),
    });

    // H — liquid inference + CDT consolidation (replay of all consolidated rules)
    let mut h = init.clone();
    let mut fl = AuditFlags::default();
    let t = Instant::now();
    let c1 = collect_experience(&p1, ExpMode::Real, seed);
    let s1 = consolidate(&c1, ConsolidateMode::Full);
    drop(c1);
    adapt_dynamics(
        &mut h,
        &triples(&p1),
        Some(&s1),
        None,
        half,
        seed ^ 0x71,
        k,
        &aud,
        &mut fl,
    );
    let ah1 = acc(&h, &t1);
    let c12 = collect_experience(&s.ds.train, ExpMode::Real, seed);
    let s12 = consolidate(&c12, ConsolidateMode::Full);
    drop(c12);
    adapt_dynamics(
        &mut h,
        &triples(&p2),
        Some(&s12),
        None,
        half,
        seed ^ 0x72,
        k,
        &aud,
        &mut fl,
    );
    drop((s1, s12));
    let ms = t.elapsed().as_secs_f64() * 1e3;
    brains.push(Brain {
        name: "H_liquid_inf_cdt_cons",
        ev: evaluate(&h, &s.ds.test, 1, Some((&aud, &fl)), None),
        set1_p1: ah1,
        set1_p2: acc(&h, &t1),
        set2: acc(&h, &t2),
        retention: acc(&h, &episodes),
        cons_ms: ms,
        changed: changed(&init, &h),
        bytes: h.params() * 8,
        lat_us: latency_us(&h, &s.ds.test),
        updates: lock.examples_seen(),
    });
    let st = acc(&StaticCtl, &s.ds.test);

    let mut rows = vec![];
    for b in &brains {
        let gain = (b.ev.acc - st).max(1e-3);
        rows.push(Row {
            experiment: "E45".into(),
            seed,
            condition: b.name.into(),
            partition: "TEST".into(),
            n_examples: b.ev.n,
            params: b.bytes / 8,
            compute: b.updates,
            accuracy: b.ev.acc,
            cosine: b.ev.disp_cos,
            energy: b.ev.energy_ratio,
            manifold_distance: b.ev.manifold,
            stability: b.ev.rel_err,
            leakage: b.ev.leaked,
            q: b.ev.queries,
            status: if b.ev.leaked > 0 { "LEAKED".into() } else { "CLEAN".into() },
            notes: format!(
                "set1_after_p1={:.3} set1_after_p2={:.3} forgetting={:.3} set2={:.3} retention_deleted_episodes={:.3} consolidation_ms={:.1} params_changed={} bytes={} latency_us={:.2} updates={} ms_per_acc_point={:.2}",
                b.set1_p1, b.set1_p2, b.set1_p1 - b.set1_p2, b.set2, b.retention, b.cons_ms, b.changed, b.bytes, b.lat_us, b.updates, b.cons_ms / (100.0 * gain)
            ),
        });
    }
    let best_l = brains[0].ev.acc.max(brains[1].ev.acc);
    let clean = brains
        .iter()
        .all(|b| b.ev.queries.total() == 0 && b.ev.leaked == 0);
    let pass = best_l >= brains[2].ev.acc - MARGIN && clean;
    let q = QueryCounters::default();
    rows.push(Row {
        experiment: "E45".into(),
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
        q,
        status: if pass { "PASS".into() } else { "FAIL".into() },
        notes: format!(
            "L1={:.3} L2={:.3} T1={:.3} H={:.3} static={st:.3} L2_rollback_exact={rollback_ok} all_queries_zero={clean}",
            brains[0].ev.acc, brains[1].ev.acc, brains[2].ev.acc, brains[3].ev.acc
        ),
    });
    rows
}
