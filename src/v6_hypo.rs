//! v6 (branch `exp/v6-hipotesis`): hipótesis H1–H7 de `docs/hipotesis_math.md`
//! aplicadas sobre la arquitectura ganadora v4 §16 (Dφ gated local-lineal
//! por punto, teacher CDT consolidado, replay de confianza).
//!
//! Every arm shares: sealed split per seed, init `seed ^ 0xD0`, sampler salt
//! `0x7A` (common random numbers), H = 136, budget 256 000 × 32 (v4 §16 E3),
//! cosine lr 3e-3 → 3e-5, relative one-step loss. Hypotheses are additive
//! loss terms / sampler changes switched by [`Hyp`]. With every flag off the
//! training loop is bit-identical to `v4_distill::train_onestep_trusted_b`
//! (unit test `baseline_bit_identical_to_v4`).
//!
//! Pre-registration: `docs/preregistro_v6.md`. All constants below are fixed
//! a priori (no tuning phase, no re-tuning between DEV and CONFIRM).

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field_autonomy_v4::{
    collect_experience, consolidate, evaluate, replay_sample, target_h, triples, AuditFlags,
    Auditor, ConsolidateMode, EvalOut, ExpMode, Row, RunOpts, CSV_HEADER, HORIZONS, ITERABLE_FAMS,
    K_POINTS, REPLAY_FRACTION,
};
use crate::train_ckpt::{LoopState, TrainCkpt};
use crate::v4_controls::{Dphi, Mlp, Predictor, TrainBudget, PD};
use crate::v4_dataset::{
    apply_rule, generate_and_seal, l2, DatasetConfig, Example, Sealed, CTX_DIM, FAMILIES, N_FAM,
};
use crate::v4_distill::{family_fit_residual, onestep_breakdown, ONESTEP_LR_FINAL, TRUST_TAU};
use crate::v4_metrics as m;
use crate::v4_provenance::{count_query, snapshot, Store};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::cell::RefCell;
use std::collections::BTreeSet;

// ───────────── pre-registered constants (docs/preregistro_v6.md §3) ─────────────
pub const DEV_SEEDS_V6: [u64; 16] = [
    0xC600, 0xC601, 0xC602, 0xC603, 0xC604, 0xC605, 0xC606, 0xC607, 0xC608, 0xC609, 0xC60A, 0xC60B,
    0xC60C, 0xC60D, 0xC60E, 0xC60F,
];
pub const CONFIRM_SEEDS_V6: [u64; 16] = [
    0xD600, 0xD601, 0xD602, 0xD603, 0xD604, 0xD605, 0xD606, 0xD607, 0xD608, 0xD609, 0xD60A, 0xD60B,
    0xD60C, 0xD60D, 0xD60E, 0xD60F,
];
pub const SMOKE_SEED_V6: u64 = 0x5A60;
pub const V6_HIDDEN: usize = 136;
/// v4 §16 E3: 32 000 × E_MULT(8) optimizer steps, batch 32, lr 3e-3 → 3e-5.
pub const V6_STEPS: usize = 32_000 * 8;
pub const SMOKE_STEPS: usize = 200;
pub const SALT_ONESTEP: u64 = 0x7A;
pub const SALT_HORIZON: u64 = 0x74;
/// H1: weight of ‖E_batch[(Δθ−Δ_T)/‖y‖]‖².
pub const LAMBDA_H1: f64 = 1.0;
/// H2: horizons in the loss and episode length (same as v5 P2_gated_eqc).
pub const LOSS_AT: [usize; 4] = [1, 2, 4, 8];
pub const EPISODE_H: usize = 8;
/// H2: a prefix step is "correct" (PRM label) iff rel_err < this (= m::REL_ERR_OK).
pub const PRM_TAU: f64 = m::REL_ERR_OK;
/// H3: finite replay buffer sizes = mult × H3_N_BASE triples (trusted families only).
pub const H3_N_BASE: usize = 4096;
pub const H3_MULTS: [f64; 4] = [0.5, 1.0, 2.0, 4.0];
/// H4: Sobolev weight and finite-difference step (model and teacher).
pub const LAMBDA_H4: f64 = 0.1;
pub const FD_EPS: f64 = 1e-3;
/// H5a: continuous weight w_f = exp(-r_f / H5_TAU).
pub const H5_TAU: f64 = TRUST_TAU;
/// H5b: contamination fractions of replay targets, relative noise σ.
pub const H5_ETAS: [f64; 4] = [0.01, 0.02, 0.04, 0.08];
pub const H5_SIGMA: f64 = 0.1;
/// H6: semigroup weight (s ∈ {1, ½} as extra network input).
pub const LAMBDA_H6: f64 = 1.0;
/// H7: normal-contraction hinge weight and target contraction ρ.
pub const LAMBDA_H7: f64 = 1e-3;
pub const RHO_H7: f64 = 0.9;

pub fn budget(steps: usize) -> TrainBudget {
    TrainBudget {
        steps,
        batch: 32,
        lr: 3e-3,
    }
}

/// Hypothesis switches for the one-step (§16) route.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hyp {
    pub h1: bool,
    /// 0 = infinite generative replay stream (v4); >0 = finite buffer mult.
    pub h3_mult: f64,
    pub h4: bool,
    pub h5_soft: bool,
    pub h5_eta: f64,
    pub h6: bool,
    pub h7: bool,
}

impl Hyp {
    pub fn all() -> Self {
        Hyp {
            h1: true,
            h4: true,
            h5_soft: true,
            h6: true,
            h7: true,
            ..Default::default()
        }
    }
    pub fn tag(&self) -> String {
        format!(
            "h1={} h3={} h4={} h5soft={} h5eta={} h6={} h7={}",
            self.h1 as u8,
            self.h3_mult,
            self.h4 as u8,
            self.h5_soft as u8,
            self.h5_eta,
            self.h6 as u8,
            self.h7 as u8
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Route {
    OneStep(Hyp),
    /// H2: multi-horizon on TRAIN episodes with steps/EPISODE_H (equal Dφ evals).
    Horizon {
        prm: bool,
    },
    /// v6 + H2: half the Dφ evaluations one-step(Hyp), half PRM horizon.
    OneThenPrm(Hyp),
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub name: &'static str,
    pub residual: bool,
    pub shuffled: bool,
    pub route: Route,
    /// Also score E33 with V1 manifold reprojection (H7 comparator).
    pub reproj: bool,
    pub pending: bool,
}

/// The pre-registered arm list (fixed order; docs/preregistro_v6.md §4).
pub fn arms() -> Vec<Arm> {
    let a = |name, residual, route, shuffled| Arm {
        name,
        residual,
        shuffled,
        route,
        reproj: false,
        pending: false,
    };
    let one = |h: Hyp| Route::OneStep(h);
    let d = Hyp::default();
    let mut v = vec![
        Arm {
            reproj: true,
            ..a("E3_v4", true, one(d), false)
        },
        a("E3_mlp", false, one(d), false),
        a("E3_shuffled", true, one(d), true),
        a("H1_bias", true, one(Hyp { h1: true, ..d }), false),
        Arm {
            pending: true,
            ..a("H2_mh_eqc", true, Route::Horizon { prm: false }, false)
        },
        Arm {
            pending: true,
            ..a("H2_prm_eqc", true, Route::Horizon { prm: true }, false)
        },
    ];
    for (i, mult) in H3_MULTS.iter().enumerate() {
        let n = ["H3_x0.5", "H3_x1", "H3_x2", "H3_x4"][i];
        v.push(a(
            n,
            true,
            one(Hyp {
                h3_mult: *mult,
                ..d
            }),
            false,
        ));
    }
    v.push(a("H4_sobolev", true, one(Hyp { h4: true, ..d }), false));
    v.push(a(
        "H4_sobolev_mlp",
        false,
        one(Hyp { h4: true, ..d }),
        false,
    ));
    v.push(a("H5_soft", true, one(Hyp { h5_soft: true, ..d }), false));
    for (i, eta) in H5_ETAS.iter().enumerate() {
        let n = ["H5_eta1", "H5_eta2", "H5_eta4", "H5_eta8"][i];
        v.push(a(n, true, one(Hyp { h5_eta: *eta, ..d }), false));
    }
    v.push(a("H6_semigroup", true, one(Hyp { h6: true, ..d }), false));
    v.push(a("H7_normal", true, one(Hyp { h7: true, ..d }), false));
    v.push(a("V6_all", true, one(Hyp::all()), false));
    // MLP gated control with every term definable for a non-residual map
    // (H6 needs Φ_{1/2} = Ψ + ½Δ, undefined for the direct MLP).
    v.push(a(
        "V6_all_mlp",
        false,
        one(Hyp {
            h6: false,
            ..Hyp::all()
        }),
        false,
    ));
    v.push(a("V6_shuffled", true, one(Hyp::all()), true));
    v.push(Arm {
        pending: true,
        ..a("V6_all_H2", true, Route::OneThenPrm(Hyp::all()), false)
    });
    v
}

// ─────────────────────────── point-level field ───────────────────────────

/// One field point evaluation cache.
#[derive(Clone)]
struct Pt {
    inp: Vec<f64>,
    h: Vec<f64>,
    raw: Vec<f64>,
    out: [f64; 2],
}

/// Field Φ_s on one point. `s_in`: s is the last network input (H6 arms).
/// For s = 1 without `s_in` this is exactly `Dphi::combine` (same fp order).
fn pt_fwd(net: &Mlp, residual: bool, s_in: bool, q: [f64; 2], c: &[f64], s: f64) -> Pt {
    let mut inp = Vec::with_capacity(net.din);
    inp.push(q[0]);
    inp.push(q[1]);
    inp.extend_from_slice(c);
    if s_in {
        inp.push(s);
    }
    let (h, o) = net.forward(&inp);
    let r = if residual { 1.0 } else { 0.0 };
    let sc = if residual { s } else { 1.0 };
    let out = if sc == 1.0 {
        [
            r * q[0] + o[0] * q[0] + o[1] * q[1] + o[4],
            r * q[1] + o[2] * q[0] + o[3] * q[1] + o[5],
        ]
    } else {
        [
            r * q[0] + sc * (o[0] * q[0] + o[1] * q[1] + o[4]),
            r * q[1] + sc * (o[2] * q[0] + o[3] * q[1] + o[5]),
        ]
    };
    Pt {
        inp,
        h,
        raw: o,
        out,
    }
}

/// Backprop `g_out` = dL/d out through one point. Adds parameter grads into
/// `g`; returns dL/dq when `need_dq`.
fn pt_bwd(
    net: &Mlp,
    residual: bool,
    pt: &Pt,
    s: f64,
    g_out: [f64; 2],
    g: &mut [f64],
    need_dq: bool,
) -> [f64; 2] {
    let q = [pt.inp[0], pt.inp[1]];
    let sc = if residual { s } else { 1.0 };
    let (g0, g1) = if sc == 1.0 {
        (g_out[0], g_out[1])
    } else {
        (sc * g_out[0], sc * g_out[1])
    };
    let d_raw = [g0 * q[0], g0 * q[1], g1 * q[0], g1 * q[1], g0, g1];
    net.backward(&pt.inp, &pt.h, &d_raw, g);
    if !need_dq {
        return [0.0, 0.0];
    }
    let r = if residual { 1.0 } else { 0.0 };
    let o = &pt.raw;
    let mut dq = [
        r * g_out[0] + g0 * o[0] + g1 * o[2],
        r * g_out[1] + g0 * o[1] + g1 * o[3],
    ];
    // through the MLP input (first PD inputs)
    let mut dh = vec![0.0; net.dh];
    for k in 0..net.dout {
        if d_raw[k] == 0.0 {
            continue;
        }
        for j in 0..net.dh {
            dh[j] += d_raw[k] * net.w2[k * net.dh + j];
        }
    }
    for j in 0..net.dh {
        let dz = dh[j] * (1.0 - pt.h[j] * pt.h[j]);
        dq[0] += dz * net.w1[j * net.din];
        dq[1] += dz * net.w1[j * net.din + 1];
    }
    dq
}

fn state_fwd(net: &Mlp, residual: bool, s_in: bool, z: &[f64], c: &[f64], s: f64) -> Vec<Pt> {
    (0..z.len() / PD)
        .map(|p| pt_fwd(net, residual, s_in, [z[p * PD], z[p * PD + 1]], c, s))
        .collect()
}

fn outs(pts: &[Pt]) -> Vec<f64> {
    pts.iter().flat_map(|p| p.out).collect()
}

/// Inference Dφ from a (possibly s-input) training net: s = 1 is folded into
/// the first-layer bias, so the exported model has the standard 2590-param
/// `Dphi` layout and uses the unchanged `Dphi::step` / fresh-process eval.
fn export(net: &Mlp, residual: bool, s_in: bool, label: &str) -> Dphi {
    let mut d = Dphi::gated_h(2 * K_POINTS, CTX_DIM, net.dh, residual, 0);
    if s_in {
        let din = net.din;
        let mut w1 = Vec::with_capacity((din - 1) * net.dh);
        let mut b1 = net.b1.clone();
        for j in 0..net.dh {
            w1.extend_from_slice(&net.w1[j * din..j * din + din - 1]);
            b1[j] += net.w1[j * din + din - 1];
        }
        d.net.w1 = w1;
        d.net.b1 = b1;
        d.net.w2 = net.w2.clone();
        d.net.b2 = net.b2.clone();
    } else {
        d.net = net.clone();
    }
    d.label = label.into();
    d
}

// ─────────────────────────── Adam (same formula as v4_controls::Adam) ─────

struct Opt {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
    lr: f64,
}

impl Opt {
    fn new(n: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
        }
    }
    fn step(&mut self, p: &mut [f64], g: &[f64]) {
        self.t += 1;
        let (b1, b2) = (0.9f64, 0.999f64);
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        for i in 0..p.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= self.lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

fn cosine_lr(step: usize, steps: usize, lr: f64, lr_final: f64) -> f64 {
    let t = step as f64 / steps.max(1) as f64;
    lr_final + 0.5 * (lr - lr_final) * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Stateless per-(seed, step, b, salt) RNG for auxiliary draws (H5b noise,
/// H7 directions) so the main sampler stream is untouched and checkpoints
/// need no extra RNG state.
fn aux_rng(seed: u64, step: usize, b: usize, salt: u64) -> Xoshiro256StarStar {
    let mut x = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = x.wrapping_add((step as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    x ^= (b as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    Xoshiro256StarStar::seed_from_u64(x)
}

fn fam_params(c: &[f64]) -> (usize, [f64; 2]) {
    let fam = (0..N_FAM).find(|&f| c[f] > 0.5).unwrap_or(0);
    (fam, [c[N_FAM], c[N_FAM + 1]])
}

fn sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

// ─────────────────────────── one-step (§16) route ───────────────────────────

struct Sample {
    x: Vec<f64>,
    c: Vec<f64>,
    y: Vec<f64>,
    w: f64,
    replay: bool,
}

pub struct TrainOut {
    pub flags: AuditFlags,
    pub mask: Vec<bool>,
    pub fit_res: Vec<f64>,
    pub loss: f64,
    pub aux: [f64; 5],
    pub leak_frames: usize,
}

fn mk_ckpt<'a>(
    ck: &Option<(std::path::PathBuf, String)>,
    save: &'a dyn Fn() -> serde_json::Value,
    load: &'a dyn Fn(&serde_json::Value),
) -> Option<TrainCkpt<'a>> {
    ck.as_ref()
        .map(|(p, k)| TrainCkpt::new(p.clone(), k.clone(), save, load))
}

fn maybe_save(
    ck: Option<&TrainCkpt>,
    step: usize,
    steps: usize,
    rng: &Xoshiro256StarStar,
    opt: &Opt,
    p: &[f64],
    last: f64,
    base: &crate::v4_provenance::QueryCounters,
) {
    if let Some(c) = ck {
        if step + 1 == steps || c.due_at(step + 1) {
            let st = LoopState {
                next_step: step + 1,
                rng: rng.clone(),
                m: opt.m.clone(),
                v: opt.v.clone(),
                t: opt.t,
                p: p.to_vec(),
                last,
                max_gn: 0.0,
            };
            crate::train_ckpt::save(c, &st, base);
            // Infra test hook: simulate a crash right after a checkpoint.
            if let Some(n) = std::env::var("V6_ABORT_AT_STEP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                if step + 1 == n {
                    eprintln!("V6_ABORT_AT_STEP={n}: exiting after checkpoint");
                    std::process::exit(86);
                }
            }
        }
    }
}

fn try_resume(
    ck: Option<&TrainCkpt>,
    rng: &mut Xoshiro256StarStar,
    opt: &mut Opt,
    p: &mut Vec<f64>,
    last: &mut f64,
) -> usize {
    if let Some(st) = ck.and_then(crate::train_ckpt::load) {
        *rng = st.rng;
        opt.m = st.m;
        opt.v = st.v;
        opt.t = st.t;
        *p = st.p;
        *last = st.last;
        return st.next_step;
    }
    0
}

/// §16 trusted-replay one-step training with hypothesis terms. Returns the
/// training net (din may include s for H6) and diagnostics.
pub fn train_onestep_v6(
    net: &mut Mlp,
    residual: bool,
    train: &[Example],
    seed: u64,
    aud: &Auditor,
    hyp: Hyp,
    bud: TrainBudget,
    ck: Option<(std::path::PathBuf, String)>,
) -> TrainOut {
    let k = K_POINTS;
    let s_in = hyp.h6;
    let tr = triples(train);
    let n_tr = tr.len();
    let mut flags = AuditFlags::default();
    let cdt = collect_experience(train, ExpMode::Real, seed);
    flags.cdt_canon = cdt.canonical_targets();
    let sig = consolidate(&cdt, ConsolidateMode::Full);
    let res = family_fit_residual(&sig, cdt.read_all());
    let trusted: Vec<bool> = res.iter().map(|r| *r < TRUST_TAU).collect();
    let wsoft: Vec<f64> = res.iter().map(|r| (-r / H5_TAU).exp()).collect();
    let any = trusted.iter().any(|t| *t);
    let replay_on = if hyp.h5_soft { true } else { any };
    // H3 finite buffer (trusted families only), deterministic from seed.
    let buffer: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = if hyp.h3_mult > 0.0 && any {
        let n = (hyp.h3_mult * H3_N_BASE as f64).round() as usize;
        let mut r = Xoshiro256StarStar::seed_from_u64(seed ^ 0xB0F3);
        let mut b = Vec::with_capacity(n);
        while b.len() < n {
            let t = replay_sample(&sig, &mut r, k);
            if trusted[fam_params(&t.1).0] {
                b.push(t);
            }
        }
        b
    } else {
        vec![]
    };
    let hits: RefCell<BTreeSet<u64>> = Default::default();
    if !buffer.is_empty() {
        for t in &buffer {
            hits.borrow_mut().extend(aud.hits(&t.2));
        }
    }
    // Extra checkpoint state: auditor hits + last-step diagnostics (bit-exact).
    let aux_cell: RefCell<[f64; 5]> = RefCell::new([0.0; 5]);
    let save = || {
        serde_json::json!({
            "hits": hits.borrow().iter().copied().collect::<Vec<u64>>(),
            "aux": crate::train_ckpt::bits(&aux_cell.borrow()[..]),
        })
    };
    let load = |v: &serde_json::Value| {
        let mut h = hits.borrow_mut();
        h.clear();
        h.extend(
            v["hits"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_u64()),
        );
        let a = crate::train_ckpt::unbits(&v["aux"]);
        if a.len() == 5 {
            aux_cell.borrow_mut().copy_from_slice(&a);
        }
    };
    let tc = mk_ckpt(&ck, &save, &load);

    let np = net.n_params();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ SALT_ONESTEP);
    let mut opt = Opt::new(np, bud.lr);
    let mut p = net.flat();
    let mut last = 0.0;
    let base = snapshot();
    let start = try_resume(tc.as_ref(), &mut rng, &mut opt, &mut p, &mut last);
    net.set_flat(&p);
    let bsz = bud.batch as f64;
    let mut g = vec![0.0; np];
    for step in start..bud.steps {
        opt.lr = cosine_lr(step, bud.steps, bud.lr, ONESTEP_LR_FINAL);
        g.fill(0.0);
        let mut loss = 0.0;
        let mut a_step = [0.0f64; 5];
        // H1 bookkeeping: per-sample normalized error vectors and caches.
        let mut h1_cache: Vec<(Vec<Pt>, Vec<f64>, f64)> = vec![];
        for b in 0..bud.batch {
            // ---- sampler (E3 order of RNG draws preserved) ----
            let smp = {
                let r = &mut rng;
                if replay_on && r.gen::<f64>() < REPLAY_FRACTION {
                    let t = if !buffer.is_empty() {
                        buffer[r.gen_range(0..buffer.len())].clone()
                    } else if hyp.h5_soft {
                        let t = replay_sample(&sig, r, k);
                        hits.borrow_mut().extend(aud.hits(&t.2));
                        t
                    } else {
                        loop {
                            let t = replay_sample(&sig, r, k);
                            if trusted[fam_params(&t.1).0] {
                                hits.borrow_mut().extend(aud.hits(&t.2));
                                break t;
                            }
                        }
                    };
                    let w = if hyp.h5_soft {
                        wsoft[fam_params(&t.1).0]
                    } else {
                        1.0
                    };
                    Sample {
                        x: t.0,
                        c: t.1,
                        y: t.2,
                        w,
                        replay: true,
                    }
                } else {
                    let t = tr[r.gen_range(0..n_tr)].clone();
                    Sample {
                        x: t.0,
                        c: t.1,
                        y: t.2,
                        w: 1.0,
                        replay: false,
                    }
                }
            };
            let Sample {
                x,
                c,
                y: y_clean,
                w,
                replay,
            } = smp;
            // H5b: contaminate replay targets (aux RNG, main stream untouched).
            let y = if hyp.h5_eta > 0.0 && replay {
                let mut ar = aux_rng(seed, step, b, 0x5B);
                if ar.gen::<f64>() < hyp.h5_eta {
                    let nv: Vec<f64> = y_clean.iter().map(|_| ar.gen_range(-1.0..1.0)).collect();
                    let s = H5_SIGMA * sq(&y_clean).sqrt() / sq(&nv).sqrt().max(1e-12);
                    y_clean.iter().zip(&nv).map(|(a, b)| a + s * b).collect()
                } else {
                    y_clean.clone()
                }
            } else {
                y_clean.clone()
            };
            // ---- main relative one-step term (bit-identical to train_onestep) ----
            let scale = w * (1.0 / y.iter().map(|v| v * v).sum::<f64>().max(1e-6));
            let npts = x.len() / PD;
            let mut main_pts = Vec::with_capacity(npts);
            for pt in 0..npts {
                let q = [x[pt * PD], x[pt * PD + 1]];
                let ptc = pt_fwd(net, residual, s_in, q, &c, 1.0);
                let o = ptc.out;
                let d = [
                    2.0 * (o[0] - y[pt * PD]) * scale / bud.batch as f64,
                    2.0 * (o[1] - y[pt * PD + 1]) * scale / bud.batch as f64,
                ];
                loss += (0..PD)
                    .map(|i| (o[i] - y[pt * PD + i]).powi(2))
                    .sum::<f64>()
                    * scale;
                pt_bwd(net, residual, &ptc, 1.0, d, &mut g, false);
                main_pts.push(ptc);
            }
            let fx = outs(&main_pts);
            // ---- H4: Sobolev / Jacobian distillation on teacher replay ----
            let h4_w = if hyp.h5_soft { w } else { 1.0 };
            if hyp.h4 && replay && h4_w > 0.0 {
                let (fam, prm) = fam_params(&c);
                if let Some(y_t) = sig.predict(fam, prm, &x) {
                    let mut l4 = 0.0;
                    for pt in 0..npts {
                        for j in 0..PD {
                            let mut xp = x.clone();
                            xp[pt * PD + j] += FD_EPS;
                            let yp = sig.predict(fam, prm, &xp).unwrap_or_else(|| y_t.clone());
                            let colt = [
                                (yp[pt * PD] - y_t[pt * PD]) / FD_EPS,
                                (yp[pt * PD + 1] - y_t[pt * PD + 1]) / FD_EPS,
                            ];
                            let mut q = [x[pt * PD], x[pt * PD + 1]];
                            q[j] += FD_EPS;
                            let pp = pt_fwd(net, residual, s_in, q, &c, 1.0);
                            let o = main_pts[pt].out;
                            let r = [
                                (pp.out[0] - o[0]) / FD_EPS - colt[0],
                                (pp.out[1] - o[1]) / FD_EPS - colt[1],
                            ];
                            l4 += r[0] * r[0] + r[1] * r[1];
                            let f = 2.0 * LAMBDA_H4 * h4_w / (FD_EPS * npts as f64 * bsz);
                            pt_bwd(net, residual, &pp, 1.0, [f * r[0], f * r[1]], &mut g, false);
                            pt_bwd(
                                net,
                                residual,
                                &main_pts[pt],
                                1.0,
                                [-f * r[0], -f * r[1]],
                                &mut g,
                                false,
                            );
                        }
                    }
                    a_step[1] += LAMBDA_H4 * h4_w * l4 / npts as f64;
                }
            }
            // ---- H6: semigroup Φ_1(q) ≈ Φ_½(Φ_½(q)) ----
            if hyp.h6 && residual {
                let ys = 1.0 / sq(&y).max(1e-6);
                for pt in 0..npts {
                    let q = [x[pt * PD], x[pt * PD + 1]];
                    let a = pt_fwd(net, residual, s_in, q, &c, 0.5);
                    let bb = pt_fwd(net, residual, s_in, a.out, &c, 0.5);
                    let f = main_pts[pt].out;
                    let r = [f[0] - bb.out[0], f[1] - bb.out[1]];
                    a_step[2] += LAMBDA_H6 * (r[0] * r[0] + r[1] * r[1]) * ys;
                    let gf = [
                        2.0 * LAMBDA_H6 * r[0] * ys / bsz,
                        2.0 * LAMBDA_H6 * r[1] * ys / bsz,
                    ];
                    pt_bwd(net, residual, &main_pts[pt], 1.0, gf, &mut g, false);
                    let da = pt_bwd(net, residual, &bb, 0.5, [-gf[0], -gf[1]], &mut g, true);
                    pt_bwd(net, residual, &a, 0.5, da, &mut g, false);
                }
            }
            // ---- H7: contraction of the normal (off-similarity-manifold) gain ----
            if hyp.h7 {
                let tb = tangent_basis(&x);
                let mut ar = aux_rng(seed, step, b, 0x77);
                let mut v: Vec<f64> = x.iter().map(|_| ar.gen_range(-1.0..1.0)).collect();
                proj_out(&mut v, &tb);
                let nv = sq(&v).sqrt();
                if nv > 1e-9 {
                    v.iter_mut().for_each(|a| *a /= nv);
                    let xp: Vec<f64> = x.iter().zip(&v).map(|(a, b)| a + FD_EPS * b).collect();
                    let pp = state_fwd(net, residual, s_in, &xp, &c, 1.0);
                    let fp = outs(&pp);
                    let mut wv: Vec<f64> =
                        fp.iter().zip(&fx).map(|(a, b)| (a - b) / FD_EPS).collect();
                    proj_out(&mut wv, &tb);
                    let nn = sq(&wv).sqrt();
                    if nn > RHO_H7 {
                        a_step[3] += LAMBDA_H7 * (nn.ln() - RHO_H7.ln());
                        let f = LAMBDA_H7 / (nn * nn * FD_EPS * bsz);
                        for pt in 0..npts {
                            let gp = [f * wv[pt * PD], f * wv[pt * PD + 1]];
                            pt_bwd(net, residual, &pp[pt], 1.0, gp, &mut g, false);
                            pt_bwd(
                                net,
                                residual,
                                &main_pts[pt],
                                1.0,
                                [-gp[0], -gp[1]],
                                &mut g,
                                false,
                            );
                        }
                    }
                }
            }
            if hyp.h1 {
                let ny = sq(&y).sqrt().max(1e-3);
                let e: Vec<f64> = fx.iter().zip(&y).map(|(a, b)| (a - b) / ny).collect();
                h1_cache.push((main_pts, e, ny));
            }
        }
        // ---- H1: squared batch-mean signed relative error ----
        if hyp.h1 && !h1_cache.is_empty() {
            let zd = h1_cache[0].1.len();
            let mut eb = vec![0.0; zd];
            for (_, e, _) in &h1_cache {
                for i in 0..zd {
                    eb[i] += e[i] / bsz;
                }
            }
            a_step[0] += LAMBDA_H1 * sq(&eb);
            for (pts, _, ny) in &h1_cache {
                for (pt, ptc) in pts.iter().enumerate() {
                    let f = 2.0 * LAMBDA_H1 / (bsz * ny);
                    pt_bwd(
                        net,
                        residual,
                        ptc,
                        1.0,
                        [f * eb[pt * PD], f * eb[pt * PD + 1]],
                        &mut g,
                        false,
                    );
                }
            }
        }
        opt.step(&mut p, &g);
        net.set_flat(&p);
        last = loss / bsz;
        aux_cell.borrow_mut().copy_from_slice(&a_step);
        maybe_save(tc.as_ref(), step, bud.steps, &rng, &opt, &p, last, &base);
    }
    let aux_final: [f64; 5] = *aux_cell.borrow();
    drop(sig);
    drop(cdt);
    for h in hits.borrow().iter() {
        if *h == u64::MAX {
            flags.any_equiv_global = true;
        } else {
            flags.equiv_hits.insert(*h);
        }
    }
    TrainOut {
        flags,
        mask: trusted,
        fit_res: res,
        loss: last,
        aux: aux_final,
        leak_frames: 0,
    }
}

/// Orthonormal basis of the similarity tangent space at `x`
/// (span{u, i·u, 1, i}, u = x − centroid) — the space V1 reprojects onto.
fn tangent_basis(x: &[f64]) -> Vec<Vec<f64>> {
    let k = x.len() / 2;
    let (mut cx, mut cy) = (0.0, 0.0);
    for i in 0..k {
        cx += x[2 * i] / k as f64;
        cy += x[2 * i + 1] / k as f64;
    }
    let mut raw = vec![vec![0.0; 2 * k]; 4];
    for i in 0..k {
        let (ux, uy) = (x[2 * i] - cx, x[2 * i + 1] - cy);
        raw[0][2 * i] = ux;
        raw[0][2 * i + 1] = uy;
        raw[1][2 * i] = -uy;
        raw[1][2 * i + 1] = ux;
        raw[2][2 * i] = 1.0;
        raw[3][2 * i + 1] = 1.0;
    }
    let mut basis: Vec<Vec<f64>> = vec![];
    for mut v in raw {
        proj_out(&mut v, &basis);
        let n = sq(&v).sqrt();
        if n > 1e-9 {
            v.iter_mut().for_each(|a| *a /= n);
            basis.push(v);
        }
    }
    basis
}

fn proj_out(v: &mut [f64], basis: &[Vec<f64>]) {
    for t in basis {
        let d: f64 = v.iter().zip(t).map(|(a, b)| a * b).sum();
        for (a, b) in v.iter_mut().zip(t) {
            *a -= d * b;
        }
    }
}

// ─────────────────────────── H2: multi-horizon / PRM ───────────────────────────

/// TRAIN episodes: states[k] = rule^k(input). Same construction as v5.
fn episodes(train: &[Example]) -> Vec<(Vec<f64>, Vec<Vec<f64>>)> {
    train
        .iter()
        .map(|e| {
            let mut st = vec![e.input.clone()];
            for _ in 0..EPISODE_H {
                let nx = apply_rule(e.family_id, e.params, st.last().unwrap());
                st.push(nx);
            }
            (e.context.clone(), st)
        })
        .collect()
}

fn future_leak_frames(eps: &[(Vec<f64>, Vec<Vec<f64>>)], aud: &Auditor) -> usize {
    eps.iter()
        .map(|(_, st)| {
            st.iter()
                .skip(2)
                .filter(|s| !aud.hits(s).is_empty())
                .count()
        })
        .sum()
}

/// One episode: forward EPISODE_H steps, relative loss at LOSS_AT; with `prm`
/// a horizon k only counts if every earlier step j<k had rel_err < PRM_TAU.
/// Returns (loss, number of masked horizons).
fn horizon_episode(
    net: &Mlp,
    residual: bool,
    c: &[f64],
    st: &[Vec<f64>],
    prm: bool,
    bsz: f64,
    g: &mut [f64],
) -> (f64, usize) {
    let hmax = EPISODE_H;
    let mut traj: Vec<Vec<Pt>> = Vec::with_capacity(hmax);
    let mut z = st[0].clone();
    let mut rel = vec![0.0; hmax + 1];
    for t in 1..=hmax {
        let pts = state_fwd(net, residual, false, &z, c, 1.0);
        z = outs(&pts);
        rel[t] = m::rel_err(&z, &st[t]);
        traj.push(pts);
    }
    let mut active = vec![true; hmax + 1];
    if prm {
        let mut ok = true;
        for t in 1..=hmax {
            active[t] = ok;
            ok = ok && rel[t] < PRM_TAU;
        }
    }
    let npts = st[0].len() / PD;
    let mut d_state = vec![0.0; st[0].len()];
    let (mut loss, mut masked) = (0.0, 0usize);
    for t in (1..=hmax).rev() {
        if LOSS_AT.contains(&t) {
            if active[t] {
                let y = &st[t];
                let zt = outs(&traj[t - 1]);
                let scale = 1.0 / sq(y).max(1e-6);
                for i in 0..zt.len() {
                    let diff = zt[i] - y[i];
                    loss += diff * diff * scale;
                    d_state[i] += 2.0 * diff * scale / bsz;
                }
            } else {
                masked += 1;
            }
        }
        if d_state.iter().all(|v| *v == 0.0) {
            continue;
        }
        let mut d_prev = vec![0.0; d_state.len()];
        for p in 0..npts {
            let dq = pt_bwd(
                net,
                residual,
                &traj[t - 1][p],
                1.0,
                [d_state[p * PD], d_state[p * PD + 1]],
                g,
                t > 1,
            );
            d_prev[p * PD] = dq[0];
            d_prev[p * PD + 1] = dq[1];
        }
        d_state = d_prev;
    }
    (loss, masked)
}

fn train_horizon_v6(
    net: &mut Mlp,
    residual: bool,
    eps: &[(Vec<f64>, Vec<Vec<f64>>)],
    seed: u64,
    prm: bool,
    bud: TrainBudget,
    ck: Option<(std::path::PathBuf, String)>,
) -> (f64, f64) {
    let mf: RefCell<f64> = RefCell::new(0.0);
    let save = || serde_json::json!(mf.borrow().to_bits());
    let load = |v: &serde_json::Value| {
        if let Some(b) = v.as_u64() {
            *mf.borrow_mut() = f64::from_bits(b);
        }
    };
    let tc = mk_ckpt(&ck, &save, &load);
    let np = net.n_params();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ SALT_HORIZON);
    let mut opt = Opt::new(np, bud.lr);
    let mut p = net.flat();
    let mut last = 0.0;
    let base = snapshot();
    let start = try_resume(tc.as_ref(), &mut rng, &mut opt, &mut p, &mut last);
    net.set_flat(&p);
    let mut g = vec![0.0; np];
    for step in start..bud.steps {
        opt.lr = cosine_lr(step, bud.steps, bud.lr, ONESTEP_LR_FINAL);
        g.fill(0.0);
        let (mut loss, mut masked) = (0.0, 0usize);
        for _ in 0..bud.batch {
            count_query(Store::Cdt, 1);
            let (c, st) = &eps[rng.gen_range(0..eps.len())];
            let (l, mk) = horizon_episode(net, residual, c, st, prm, bud.batch as f64, &mut g);
            loss += l;
            masked += mk;
        }
        opt.step(&mut p, &g);
        net.set_flat(&p);
        last = loss / bud.batch as f64;
        *mf.borrow_mut() = masked as f64 / (bud.batch * LOSS_AT.len()) as f64;
        maybe_save(tc.as_ref(), step, bud.steps, &rng, &opt, &p, last, &base);
    }
    let masked_frac = *mf.borrow();
    (last, masked_frac)
}

// ─────────────────────────── evaluation helpers ───────────────────────────

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

/// E33 free-run (rotate+reflect TEST) to h64; `vi` = 0 pure, 1 V1 reprojection.
fn e33(
    seed: u64,
    label: &str,
    d: &Dphi,
    vi: usize,
    iter_ex: &[&Example],
    deltas: &[Vec<f64>],
) -> (usize, Vec<Row>, String, [f64; 3]) {
    let stepf = |z: &[f64], e: &Example| {
        let o = d.step(z, &e.context);
        if vi == 1 {
            crate::v4_stability::project_similarity(&e.input, &o)
        } else {
            o
        }
    };
    let mut zs: Vec<Vec<f64>> = iter_ex.iter().map(|e| e.input.clone()).collect();
    let mut zp: Vec<Vec<f64>> = iter_ex
        .iter()
        .zip(deltas)
        .map(|(e, dl)| e.input.iter().zip(dl).map(|(a, b)| a + b).collect())
        .collect();
    let (mut region, mut contiguous) = (0usize, true);
    let (mut rows, mut notes) = (vec![], vec![]);
    let mut key = [f64::NAN; 3]; // acc h16, rel h8, pert h8
    for t in 1..=*HORIZONS.last().unwrap() {
        for (i, e) in iter_ex.iter().enumerate() {
            zs[i] = stepf(&zs[i], e);
            zp[i] = stepf(&zp[i], e);
        }
        if HORIZONS.contains(&t) {
            let (mut ok, mut re, mut pg) = (0usize, vec![], vec![]);
            for (i, e) in iter_ex.iter().enumerate() {
                let y = target_h(e, t);
                ok += m::correct(&zs[i], &y) as usize;
                re.push(m::rel_err(&zs[i], &y).min(10.0));
                pg.push(l2(&zp[i], &zs[i]) / m::norm(&deltas[i]).max(1e-12));
            }
            let acc = ok as f64 / iter_ex.len().max(1) as f64;
            if contiguous && acc >= 0.5 {
                region = t;
            } else {
                contiguous = false;
            }
            if t == 16 {
                key[0] = acc;
            }
            if t == 8 {
                key[1] = m::mean(&re);
                key[2] = m::median(&pg);
            }
            notes.push(format!(
                "h{t}:acc={acc:.4},rel={:.4},pert={:.4}",
                m::mean(&re),
                m::median(&pg)
            ));
            rows.push(Row {
                experiment: "V6".into(),
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
                q: Default::default(),
                status: "CLEAN".into(),
                notes: String::new(),
            });
        }
    }
    (region, rows, notes.join(" "), key)
}

/// H1 metric: signed per-step bias on the TRUE trajectory (t = 1..16).
/// Returns (norm bias, angle bias [rad], mean rel err).
fn signed_bias(d: &Dphi, iter_ex: &[&Example]) -> (f64, f64, f64) {
    let (mut nb, mut ab, mut re) = (vec![], vec![], vec![]);
    for e in iter_ex {
        let mut z = e.input.clone();
        for _ in 0..16 {
            let y = apply_rule(e.family_id, e.params, &z);
            let p = d.step(&z, &e.context);
            nb.push((m::norm(&p) - m::norm(&y)) / m::norm(&y).max(1e-12));
            // signed angle of the complex LS fit p_c ≈ s·y_c (centered)
            let k = y.len() / 2;
            let (mut yx, mut yy, mut px, mut py) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..k {
                yx += y[2 * i] / k as f64;
                yy += y[2 * i + 1] / k as f64;
                px += p[2 * i] / k as f64;
                py += p[2 * i + 1] / k as f64;
            }
            let (mut re_, mut im_) = (0.0, 0.0);
            for i in 0..k {
                let (a, b) = (y[2 * i] - yx, y[2 * i + 1] - yy);
                let (c, dd) = (p[2 * i] - px, p[2 * i + 1] - py);
                re_ += a * c + b * dd;
                im_ += a * dd - b * c;
            }
            ab.push(im_.atan2(re_));
            re.push(m::rel_err(&p, &y));
            z = y;
        }
    }
    (m::mean(&nb), m::mean(&ab), m::mean(&re))
}

/// H4 metric: median |σmax(Jθ) − σmax(J*)| at iterable TEST inputs.
fn sigma_gap(d: &Dphi, iter_ex: &[&Example]) -> f64 {
    let v: Vec<f64> = iter_ex
        .iter()
        .map(|e| {
            let jt = m::jacobian(|w| d.step(w, &e.context), &e.input);
            let js = m::jacobian(|w| apply_rule(e.family_id, e.params, w), &e.input);
            (m::sigma_max(&jt) - m::sigma_max(&js)).abs()
        })
        .collect();
    m::median(&v)
}

/// H5b metric: RMS relative one-step difference vs a reference model on TEST.
fn dstep(a: &Dphi, b: &Dphi, test: &[Example]) -> f64 {
    let v: Vec<f64> = test
        .iter()
        .map(|e| {
            let (pa, pb) = (a.step(&e.input, &e.context), b.step(&e.input, &e.context));
            sq(&pa.iter().zip(&pb).map(|(x, y)| x - y).collect::<Vec<_>>())
                / sq(&e.expected).max(1e-12)
        })
        .collect();
    m::mean(&v).sqrt()
}

fn shuffle_train(train: &[Example], seed: u64) -> Vec<Example> {
    let mut out = train.to_vec();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x5F1F);
    for f in 0..N_FAM {
        let idx: Vec<usize> = (0..out.len()).filter(|&i| out[i].family_id == f).collect();
        let mut ys: Vec<Vec<f64>> = idx.iter().map(|&i| out[i].expected.clone()).collect();
        for i in (1..ys.len()).rev() {
            ys.swap(i, rng.gen_range(0..=i));
        }
        for (j, &i) in idx.iter().enumerate() {
            out[i].expected = ys[j].clone();
        }
    }
    out
}

fn fam_rel_notes(d: &Dphi, test: &[Example]) -> String {
    let mut acc: Vec<Vec<f64>> = (0..N_FAM).map(|_| vec![]).collect();
    for e in test {
        acc[e.family_id].push(m::rel_err(&d.step(&e.input, &e.context), &e.expected));
    }
    FAMILIES
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{n}:rel={:.4}", m::mean(&acc[i])))
        .collect::<Vec<_>>()
        .join(" ")
}

// ─────────────────────────── per-seed driver ───────────────────────────

pub struct SeedOut {
    pub rows: Vec<Row>,
    pub manifest_json: String,
}

struct Trained {
    arm: Arm,
    d: Dphi,
    train_params: usize,
    ev: EvalOut,
    flags_leak: usize,
    train_s: f64,
    note: String,
}

pub fn run_seed(
    seed: u64,
    steps: usize,
    opts: &RunOpts,
    ck_dir: Option<&std::path::Path>,
    only: Option<&[String]>,
) -> SeedOut {
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
    s.verify_test().expect("TEST sealed");
    let aud = Auditor::new(&s);
    let shuf_train = shuffle_train(&s.ds.train, seed);
    let eps = episodes(&s.ds.train);
    let ep_leaks = future_leak_frames(&eps, &aud);
    let z = 2 * K_POINTS;
    let mut trained: Vec<Trained> = vec![];
    for arm in arms() {
        if let Some(o) = only {
            if !o.iter().any(|n| n == arm.name) {
                continue;
            }
        }
        let t0 = std::time::Instant::now();
        let ckp = |phase: &str, st: usize| {
            ck_dir.map(|dir| {
                (
                    dir.join(format!("0x{seed:X}_{}{phase}.json", arm.name)),
                    format!(
                        "v6ck1 seed={seed} arm={} phase={phase} steps={st}",
                        arm.name
                    ),
                )
            })
        };
        let train: &[Example] = if arm.shuffled {
            &shuf_train
        } else {
            &s.ds.train
        };
        let (d, train_params, flags, leak, note) = match arm.route {
            Route::OneStep(h) => {
                let mut net = Dphi::gated_h(z, CTX_DIM, V6_HIDDEN, arm.residual, seed ^ 0xD0).net;
                if h.h6 {
                    net = Mlp::new(PD + CTX_DIM + 1, V6_HIDDEN, 3 * PD, 0.1, seed ^ 0xD0);
                }
                let tp = net.n_params();
                let o = train_onestep_v6(
                    &mut net,
                    arm.residual,
                    train,
                    seed,
                    &aud,
                    h,
                    budget(steps),
                    ckp("", steps),
                );
                let ms: String = o.mask.iter().map(|b| if *b { '1' } else { '0' }).collect();
                let note = format!(
                    "route=onestep {} mask={ms} fit_res={} loss={:.6} aux_h1={:.3e} aux_h4={:.3e} aux_h6={:.3e} aux_h7={:.3e}",
                    h.tag(),
                    o.fit_res.iter().map(|r| format!("{r:.4}")).collect::<Vec<_>>().join("/"),
                    o.loss,
                    o.aux[0],
                    o.aux[1],
                    o.aux[2],
                    o.aux[3]
                );
                (
                    export(&net, arm.residual, h.h6, arm.name),
                    tp,
                    o.flags,
                    o.leak_frames,
                    note,
                )
            }
            Route::Horizon { prm } => {
                let mut net = Dphi::gated_h(z, CTX_DIM, V6_HIDDEN, arm.residual, seed ^ 0xD0).net;
                let tp = net.n_params();
                let st = (steps / EPISODE_H).max(1);
                let (l, mf) = train_horizon_v6(
                    &mut net,
                    arm.residual,
                    &eps,
                    seed,
                    prm,
                    budget(st),
                    ckp("", st),
                );
                let note = format!(
                    "route=horizon prm={prm} steps={st} loss={l:.6} masked_frac_last={mf:.3}"
                );
                (
                    export(&net, arm.residual, false, arm.name),
                    tp,
                    AuditFlags::default(),
                    ep_leaks,
                    note,
                )
            }
            Route::OneThenPrm(h) => {
                let mut net = Mlp::new(
                    PD + CTX_DIM + h.h6 as usize,
                    V6_HIDDEN,
                    3 * PD,
                    0.1,
                    seed ^ 0xD0,
                );
                let tp = net.n_params();
                let sa = (steps / 2).max(1);
                let o = train_onestep_v6(
                    &mut net,
                    arm.residual,
                    train,
                    seed,
                    &aud,
                    h,
                    budget(sa),
                    ckp("_A", sa),
                );
                // phase B on the exported (s = 1 folded) net
                let mut d = export(&net, arm.residual, h.h6, arm.name);
                let sb = (steps / (2 * EPISODE_H)).max(1);
                let (l, mf) = train_horizon_v6(
                    &mut d.net,
                    arm.residual,
                    &eps,
                    seed,
                    true,
                    budget(sb),
                    ckp("_B", sb),
                );
                let note = format!("route=onestep+prm {} stepsA={sa} stepsB={sb} lossA={:.6} lossB={l:.6} masked_frac_last={mf:.3}", h.tag(), o.loss);
                (d, tp, o.flags, ep_leaks, note)
            }
        };
        let train_s = t0.elapsed().as_secs_f64();
        s.verify_test().expect("TEST sealed");
        let ev = evaluate(&d, &s.ds.test, 1, Some((&aud, &flags)), None);
        eprintln!(
            "seed 0x{seed:X} arm {} acc1={:.4} ({train_s:.1}s)",
            arm.name, ev.acc
        );
        trained.push(Trained {
            arm,
            d,
            train_params,
            ev,
            flags_leak: leak,
            train_s,
            note,
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
    let e3 = trained
        .iter()
        .find(|t| t.arm.name == "E3_v4")
        .map(|t| t.d.clone());
    let mut rows = vec![];
    for t in &trained {
        let (fa, fq) = fresh_eval(opts, &s, seed, &t.d, t.arm.name);
        let fresh_ok = (fa - t.ev.acc).abs() < 1e-9 && fq == 0.0;
        let q_test = t.ev.queries.total();
        let leak = t.flags_leak + t.ev.leaked;
        let clean = leak == 0 && q_test == 0;
        rows.push(Row {
            experiment: "V6".into(),
            seed,
            condition: t.arm.name.into(),
            partition: "TEST".into(),
            n_examples: t.ev.n,
            params: t.d.params(),
            compute: steps,
            accuracy: t.ev.acc,
            cosine: t.ev.disp_cos,
            energy: t.ev.energy_ratio,
            manifold_distance: t.ev.manifold,
            stability: t.ev.rel_err,
            leakage: leak,
            q: t.ev.queries,
            status: if clean {
                "CLEAN".into()
            } else {
                "LEAKED".into()
            },
            notes: format!(
                "{} train_params={} train_s={:.2} q_test={q_test} {} {}",
                t.note,
                t.train_params,
                t.train_s,
                fam_rel_notes(&t.d, &s.ds.test),
                onestep_breakdown(&t.d, &s.ds.test)
            ),
        });
        let variants: &[usize] = if t.arm.reproj { &[0, 1] } else { &[0] };
        for &vi in variants {
            let label = if vi == 1 {
                format!("R_{}_V1reproj", t.arm.name)
            } else {
                format!("R_{}", t.arm.name)
            };
            let (region, rr, notes, key) = e33(seed, &label, &t.d, vi, &iter_ex, &deltas);
            rows.extend(rr);
            let (nb, ab, br) = signed_bias(&t.d, &iter_ex);
            let sg = sigma_gap(&t.d, &iter_ex);
            let ds = e3
                .as_ref()
                .map(|r| dstep(&t.d, r, &s.ds.test))
                .unwrap_or(f64::NAN);
            let pass = region >= 8 && clean && fresh_ok;
            rows.push(Row {
                experiment: "V6".into(),
                seed,
                condition: "GATE".into(),
                partition: label.clone(),
                n_examples: iter_ex.len(),
                params: t.d.params(),
                compute: steps,
                accuracy: t.ev.acc,
                cosine: 0.0,
                energy: 0.0,
                manifold_distance: 0.0,
                stability: t.ev.rel_err,
                leakage: leak,
                q: t.ev.queries,
                status: if !clean { "LEAKED".into() } else if pass { "PASS".into() } else { "FAIL".into() },
                notes: format!(
                    "{label} pending={} region={region} acc1={:.4} rel_err1={:.4} acc_h16={:.4} rel_h8={:.4} pert_h8={:.4} bias_norm={nb:.5} bias_ang={ab:.5} bias_rel={br:.5} sigma_gap={sg:.4} dstep_vs_E3={ds:.5} clean={clean} leak={leak} q_test={q_test} fresh_ok={fresh_ok} fresh_acc={fa:.4} fresh_q={fq} train_s={:.2} {notes}",
                    t.arm.pending, t.ev.acc, t.ev.rel_err, key[0], key[1], key[2], t.train_s
                ),
            });
        }
    }
    let manifest_json = format!(
        "{{\"seed\":\"0x{seed:X}\",\"n_train\":{},\"episode_leak_frames\":{ep_leaks},\"test_sha256\":\"{}\"}}",
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

    #[test]
    fn baseline_bit_identical_to_v4() {
        let seed = 0x5A60;
        let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS));
        let aud = Auditor::new(&s);
        let z = 2 * K_POINTS;
        let mut a = Dphi::gated_h(z, CTX_DIM, V6_HIDDEN, true, seed ^ 0xD0);
        crate::v4_distill::train_onestep_trusted_b(&mut a, &s, seed, &aud, TRUST_TAU, budget(30));
        let mut net = Dphi::gated_h(z, CTX_DIM, V6_HIDDEN, true, seed ^ 0xD0).net;
        train_onestep_v6(
            &mut net,
            true,
            &s.ds.train,
            seed,
            &aud,
            Hyp::default(),
            budget(30),
            None,
        );
        assert_eq!(a.net.flat(), net.flat());
    }

    #[test]
    fn pt_bwd_dq_matches_fd() {
        let net = Mlp::new(PD + 3, 5, 6, 0.3, 7);
        let c = [0.2, -0.1, 1.0];
        for s in [1.0, 0.5] {
            let q = [0.3, -0.7];
            let pt = pt_fwd(&net, true, true, q, &c, s);
            let gout = [0.7, -0.4];
            let mut g = vec![0.0; net.n_params()];
            let dq = pt_bwd(&net, true, &pt, s, gout, &mut g, true);
            for j in 0..2 {
                let mut qp = q;
                qp[j] += 1e-6;
                let mut qm = q;
                qm[j] -= 1e-6;
                let fp = pt_fwd(&net, true, true, qp, &c, s).out;
                let fm = pt_fwd(&net, true, true, qm, &c, s).out;
                let num = (gout[0] * (fp[0] - fm[0]) + gout[1] * (fp[1] - fm[1])) / 2e-6;
                assert!((num - dq[j]).abs() < 1e-6, "s={s} j={j} {num} {}", dq[j]);
            }
        }
    }

    #[test]
    fn export_folds_s_exactly_enough() {
        let net = Mlp::new(PD + CTX_DIM + 1, 8, 6, 0.3, 3);
        let d = export(&net, true, true, "t");
        let x: Vec<f64> = (0..2 * K_POINTS).map(|i| 0.1 * i as f64 - 0.4).collect();
        let c = vec![0.0; CTX_DIM];
        let a = d.step(&x, &c);
        let b = outs(&state_fwd(&net, true, true, &x, &c, 1.0));
        assert!(a.iter().zip(&b).all(|(u, v)| (u - v).abs() < 1e-12));
    }
}
