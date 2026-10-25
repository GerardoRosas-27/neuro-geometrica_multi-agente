//! ECO (docs/eco/PREREG.md): ÓRBITA-v1 congelada (predictor ŷ, sorpresa s) + LAT con árbitro + Dφ §16 E3 (corrector y = ŷ + Δ)
//! con CDT, sueño enmascarado (residuo de ajuste ≥ 1 % no se consolida), checkpoint, compuerta de no-daño y rollback.
//!   orbita eco-paso0 --weights W [--weights W2 ...] --seed 0xC400 --v4 artifacts/v4/distill_e   (JSON a stdout)
//!   orbita eco-run --phase smoke|dev|confirm --weights W --seed S --out DIR
//! Núcleo ÓRBITA congelado (se verifica `raw` al final). Lee ds.test v4 sólo para E33 (como `orbita eval`).
#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::type_complexity)]
use crate::fam::reid;
use crate::latent::{fit_gn, lat_step, npar};
use crate::model::{CtxParams, Orbita};
use cdt_rqm_epr::field_autonomy_v4::{evaluate, target_h, K_POINTS, ITERABLE_FAMS};
use cdt_rqm_epr::v4_controls::{Adam, Dphi, Mlp};
use cdt_rqm_epr::v4_dataset::{apply_rule, context_of, generate_and_seal, make_object, DatasetConfig, Example, ObjectClass, N_FAM};
use cdt_rqm_epr::v4_metrics as mt;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::path::PathBuf;

// ─── hiperparámetros congelados (PREREG §4) ──────────────────────────────────
pub const K: usize = 4; // pasos observados
pub const HORIZON: usize = 16;
pub const NEW_FAMS: [usize; 3] = [8, 10, 11];
pub const N_NEW_PER_FAM: usize = 48; // flujo: 144 nuevas
pub const N_KNOWN_STREAM: usize = 48; // + 48 conocidas (6/familia) => 192
pub const SLEEP_EVERY: usize = 24; // 8 sueños
pub const SLEEP_STEPS: usize = 2000;
pub const SLEEP_BATCH: usize = 32;
pub const LR0: f64 = 3e-3; // = E3 (coseno a LR1)
pub const LR1: f64 = 3e-5;
pub const HID: usize = 136; // = v4_distill::E_HIDDEN
pub const CDIM: usize = 15; // [q_p(2), s(1), ŷ_p−q_p(2), ĉ(10)]
pub const TRUST_TAU: f64 = 0.01; // = v4_distill::TRUST_TAU
pub const REPLAY: f64 = 0.5; // = REPLAY_FRACTION
pub const N_GATE_KNOWN: usize = 64;
pub const GATE_TOL_KNOWN: f64 = 0.05; // log rel_err
pub const LAT_LAM0: f64 = 1e-4;
pub const H_PER_FAM: usize = 16;
pub const EVAL_H: [usize; 5] = [1, 2, 4, 8, 16];
const XOR_HELD: u64 = 0x4E0A_501D; // H idéntico a Nova/ESTELA
const XOR_STREAM: u64 = 0xEC05_7EA4;
const XOR_GATE: u64 = 0xEC06_A7E0;
const XOR_DPHI: u64 = 0xEC0D_0F10;
const XOR_OPT: u64 = 0xEC00_0B70;
const XOR_SHUF: u64 = 0xEC05_F1F0;
pub fn lat_tau(k: usize) -> f64 { match k { 2 => 0.014655674499999998, 4 => 0.0220202535, _ => 0.037142499499999995 } }


#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Arm { Eco, NoSleep, NoRollback, Shuf, Pos, E2, M3, Zero }
impl Arm {
    pub fn all() -> [Arm; 8] { [Arm::Eco, Arm::NoSleep, Arm::NoRollback, Arm::Shuf, Arm::Pos, Arm::E2, Arm::M3, Arm::Zero] }
    pub fn name(self) -> &'static str { match self { Arm::Eco => "eco", Arm::NoSleep => "eco_nosleep", Arm::NoRollback => "eco_norollback", Arm::Shuf => "eco_shuf", Arm::Pos => "eco_pos", Arm::E2 => "eco_e2", Arm::M3 => "eco_m3", Arm::Zero => "eco_zero" } }
}

// ─── familias (copia fiel de exp/estela orbita/src/estela/mod.rs @ db42e6a) ──
fn rot(t: f64, x: f64, y: f64) -> (f64, f64) { (t.cos() * x - t.sin() * y, t.sin() * x + t.cos() * y) }
pub fn rule(f: usize, p: [f64; 2], o: &[f64]) -> Vec<f64> {
    if f < N_FAM { return apply_rule(f, p, o); }
    let mut v = o.to_vec();
    for i in 0..o.len() / 2 {
        let (x, y) = (o[2 * i], o[2 * i + 1]);
        let (a, b) = if f == 8 { let (u, w) = rot(0.6 * p[0], x, y); (u + 0.25 * p[1] * (2.0 * w).sin(), w + 0.25 * p[1] * (2.0 * u).sin()) }
            else if f == 9 { let r = (x * x + y * y).sqrt(); let (u, w) = rot(0.4 * p[0] + 0.8 * p[1] * (r - 0.7), x, y); let s = (0.1 * p[1]).exp(); (s * u, s * w) }
            else if f == 10 { let (r, th) = ((x * x + y * y).sqrt(), y.atan2(x)); let r2 = r * (1.0 + 0.15 * p[1] * (3.0 * th).sin()); let t2 = th + 0.5 * p[0]; (r2 * t2.cos(), r2 * t2.sin()) }
            else { let u = x + 0.3 * p[0] * (2.0 * y).tanh(); (u, y + 0.3 * p[1] * (2.0 * u).tanh()) };
        v[2 * i] = a; v[2 * i + 1] = b;
    }
    v
}
#[derive(Clone)]
pub struct Ep { pub fam: usize, pub p: [f64; 2], pub x0: Vec<f64>, pub ys: Vec<Vec<f64>> }
pub fn make_ep(rng: &mut Xoshiro256StarStar, fam: usize) -> Ep {
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
    let _c = if fam < N_FAM { context_of(fam, p) } else { context_of(0, p) }; // mismo consumo de RNG que estela::make_ep
    let ci = rng.gen_range(0..3);
    let x0 = make_object(rng, K_POINTS, cls[ci]);
    let mut ys = Vec::with_capacity(HORIZON); let mut y = x0.clone();
    for _ in 0..HORIZON { y = rule(fam, p, &y); ys.push(y.clone()); }
    Ep { fam, p, x0, ys }
}
pub fn gen_heldout(seed: u64, n_per_fam: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_HELD);
    let fams: Vec<usize> = (0..N_FAM).chain(NEW_FAMS.iter().copied()).collect();
    let mut out = vec![]; for &f in &fams { for _ in 0..n_per_fam { out.push(make_ep(&mut rng, f)); } } out
}
/// flujo precuencial: 48×{8,10,11} + 6×{0..7}, orden aleatorio (Fisher–Yates)
pub fn gen_stream(seed: u64, n_new: usize, n_known: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_STREAM);
    let mut fams: Vec<usize> = vec![]; for &f in &NEW_FAMS { for _ in 0..n_new { fams.push(f); } } for i in 0..n_known { fams.push(i % N_FAM); }
    for i in (1..fams.len()).rev() { let j = rng.gen_range(0..=i); fams.swap(i, j); }
    fams.iter().map(|&f| make_ep(&mut rng, f)).collect()
}
pub fn gen_gate_known(seed: u64, n: usize) -> Vec<Ep> { let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_GATE); (0..n).map(|i| make_ep(&mut rng, i % N_FAM)).collect() }
fn fin(z: &[f64]) -> bool { z.iter().all(|v| v.is_finite()) }
pub fn clamp_err(z: &[f64], y: &[f64]) -> f64 { if fin(z) { mt::rel_err(z, y).min(10.0) } else { 10.0 } }
pub fn log_rel(e: f64) -> f64 { e.max(1e-12).ln() }
fn mean(v: &[f64]) -> f64 { if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 } }
fn median(mut v: Vec<f64>) -> f64 { if v.is_empty() { return f64::NAN; } v.sort_by(|a, b| a.total_cmp(b)); mt::median(&v) }

// ─── base = ÓRBITA+LAT (0c: ĉ en el formato D3-num) ──────────────────────────
/// 0c (PREREG §0c): ĉ = [one-hot(g) ∈ R^8, p̂_1, p̂_2] con p̂ recortado a [−1,1] — exactamente el vector de `orbita/llm/d3.py::ctx`.
pub fn d3num_ctx(g: usize, p: [f64; 2]) -> Vec<f64> { let mut v = vec![0.0; N_FAM]; v[g] = 1.0; v.push(p[0].clamp(-1.0, 1.0)); v.push(p[1].clamp(-1.0, 1.0)); v }
pub struct Base { pub c: Vec<f64>, pub cp: CtxParams, pub g: usize, pub sk: f64, pub sf: f64, pub lat: Option<Vec<f64>> }
impl Base {
    pub fn step(&self, d: &Orbita, x: &[f64]) -> Vec<f64> { match &self.lat { Some(a) => lat_step(d, &self.cp, &self.c, a, x), None => d.step(x, &self.c) } }
}
pub fn s_known(d: &Orbita, c: &[f64], x0: &[f64], ys: &[Vec<f64>], k: usize) -> f64 { let mut z = x0.to_vec(); let mut r = 0.0; for t in 0..k { z = d.step(&z, c); r += clamp_err(&z, &ys[t]); } r / k as f64 }
pub fn sfeat(sk: f64) -> f64 { (sk.max(1e-12) / lat_tau(K)).ln().clamp(-8.0, 8.0) / 4.0 }
/// ĉ: conocidas → contexto verdadero; nuevas → reid con ÓRBITA (candidato p̂ = p, como el parser de D3-num), = ctx_for_eval de Nova/ESTELA.
/// LAT (= LAT_EP): si s_known > τ_LAT(k), A ajustado por GN/LM en los k pasos (prior 0, λ0), contexto ĉ.
pub fn make_base(d: &Orbita, fam: usize, p: [f64; 2], x0: &[f64], ys: &[Vec<f64>], k: usize, known_ctx: Option<&[f64]>) -> Base {
    let (c, g) = match known_ctx { Some(c) => (c.to_vec(), (0..N_FAM).find(|&i| c[i] == 1.0).unwrap_or(0)),
        None => if fam < N_FAM { (d3num_ctx(fam, p), fam) } else { let (g, pp, _) = reid(d, x0, ys, k, p); (d3num_ctx(g, pp), g) } };
    let cp = d.ctx(&c); let sk = s_known(d, &c, x0, ys, k);
    let lat = if sk <= lat_tau(k) { None } else { Some(fit_gn(&|th, z| lat_step(d, &cp, &c, th, z), &vec![0.0; npar()], LAT_LAM0, x0, ys, k)) };
    Base { c, cp, g, sk, sf: sfeat(sk), lat }
}

// ─── Dφ corrector (§16 E3: campo local-lineal con compuerta, pesos compartidos por punto) ──
#[derive(Clone)]
pub struct Corr { pub net: Mlp, pub residual: bool }
impl Corr {
    /// residual: capa de salida = 0 ⇒ Δ ≡ 0 (Paso 0b). M3 (no residual): init v4 (escala 0,1), como M3_mlp_gated_trusted.
    pub fn new(seed: u64, residual: bool) -> Self {
        let d = Dphi::gated_h(2, CDIM, HID, residual, seed); let mut net = d.net;
        if residual { for w in net.w2.iter_mut() { *w = 0.0; } for b in net.b2.iter_mut() { *b = 0.0; } }
        Corr { net, residual }
    }
    fn inp(yh: &[f64], q: &[f64], sf: f64, c: &[f64]) -> Vec<f64> { let mut v = vec![yh[0], yh[1], q[0], q[1], sf, yh[0] - q[0], yh[1] - q[1]]; v.extend_from_slice(c); v }
    fn comb(&self, y: &[f64], o: &[f64]) -> [f64; 2] { let r = if self.residual { 1.0 } else { 0.0 };
        [r * y[0] + o[0] * y[0] + o[1] * y[1] + o[4], r * y[1] + o[2] * y[0] + o[3] * y[1] + o[5]] }
    /// y = ŷ + Δ(ŷ, q, s, ĉ) por punto
    pub fn apply(&self, q: &[f64], yh: &[f64], sf: f64, c: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; q.len()];
        for p in 0..q.len() / 2 { let (_, o) = self.net.forward(&Self::inp(&yh[2 * p..2 * p + 2], &q[2 * p..2 * p + 2], sf, c)); let y = self.comb(&yh[2 * p..2 * p + 2], &o); out[2 * p] = y[0]; out[2 * p + 1] = y[1]; }
        out
    }
    pub fn hash(&self) -> u64 { let mut h: u64 = 0xcbf29ce484222325; for v in self.net.flat() { h ^= v.to_bits(); h = h.wrapping_mul(0x100000001b3); } h }
    /// un paso de entrenamiento de un minibatch (q, ŷ, s, ĉ, y): pérdida relativa por estado ‖y'−y‖²/‖y‖² (= receta E3, rel_loss)
    fn grad(&self, b: &[(Vec<f64>, Vec<f64>, f64, Vec<f64>, Vec<f64>)], g: &mut [f64]) -> f64 {
        let mut loss = 0.0; let nb = b.len() as f64;
        for (q, yh, sf, c, y) in b {
            let sc = 1.0 / y.iter().map(|v| v * v).sum::<f64>().max(1e-6);
            for p in 0..q.len() / 2 {
                let yp = &yh[2 * p..2 * p + 2]; let inp = Self::inp(yp, &q[2 * p..2 * p + 2], *sf, c);
                let (h, o) = self.net.forward(&inp); let out = self.comb(yp, &o);
                let dd = [2.0 * (out[0] - y[2 * p]) * sc / nb, 2.0 * (out[1] - y[2 * p + 1]) * sc / nb];
                loss += ((out[0] - y[2 * p]).powi(2) + (out[1] - y[2 * p + 1]).powi(2)) * sc / nb;
                let draw = [dd[0] * yp[0], dd[0] * yp[1], dd[1] * yp[0], dd[1] * yp[1], dd[0], dd[1]];
                self.net.backward(&inp, &h, &draw, g);
            }
        }
        loss
    }
}
/// rollout de 16 pasos: base, o base+Δ. Devuelve la trayectoria.
pub fn rollout(d: &Orbita, b: &Base, x0: &[f64], n: usize, corr: Option<&Corr>) -> Vec<Vec<f64>> {
    let mut x = x0.to_vec(); let mut out = Vec::with_capacity(n);
    for _ in 0..n { if !fin(&x) { out.push(x.clone()); continue; } let yh = b.step(d, &x); x = match corr { Some(c) => c.apply(&x, &yh, b.sf, &b.c), None => yh }; out.push(x.clone()); }
    out
}
/// árbitro retenido: Δ se usa sólo si mejora estrictamente el error en los k pasos observados (Dφ no vio este episodio)
pub fn eco_traj(d: &Orbita, b: &Base, x0: &[f64], ys: &[Vec<f64>], k: usize, corr: &Corr) -> (Vec<Vec<f64>>, bool) {
    let tb = rollout(d, b, x0, HORIZON, None); let tc = rollout(d, b, x0, HORIZON, Some(corr));
    let e = |t: &[Vec<f64>]| (0..k).map(|i| clamp_err(&t[i], &ys[i])).sum::<f64>();
    if e(&tc) < e(&tb) { (tc, true) } else { (tb, false) }
}

// ─── CDT + sueño ─────────────────────────────────────────────────────────────
pub struct Rec { pub ep: Ep, pub base: Base, pub teacher: Vec<f64>, pub fit_err: f64, pub success: bool, pub gate: bool }
pub fn teacher_of(d: &Orbita, b: &Base, e: &Ep) -> (Vec<f64>, f64) {
    let a = fit_gn(&|th, z| lat_step(d, &b.cp, &b.c, th, z), &vec![0.0; npar()], LAT_LAM0, &e.x0, &e.ys, HORIZON);
    let mut z = e.x0.clone(); let mut r = 0.0; for t in 0..HORIZON { z = lat_step(d, &b.cp, &b.c, &a, &z); r += clamp_err(&z, &e.ys[t]); }
    (a, r / HORIZON as f64)
}
type Sample = (Vec<f64>, Vec<f64>, f64, Vec<f64>, Vec<f64>);
fn sample(d: &Orbita, cdt: &[Rec], gt: &[usize], tr: &[usize], perm: Option<&[usize]>, rng: &mut Xoshiro256StarStar) -> Sample {
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let teach = !tr.is_empty() && rng.gen::<f64>() < REPLAY;
    let (e, q, y) = if teach {
        let e = tr[rng.gen_range(0..tr.len())]; let r = &cdt[e]; let ci = rng.gen_range(0..3); let mut x = make_object(rng, K_POINTS, cls[ci]);
        let nt = rng.gen_range(0..=3); for _ in 0..nt { x = lat_step(d, &r.base.cp, &r.base.c, &r.teacher, &x); }
        let y = lat_step(d, &r.base.cp, &r.base.c, &r.teacher, &x); (e, x, y)
    } else {
        let e = gt[rng.gen_range(0..gt.len())]; let r = &cdt[e]; let t = rng.gen_range(0..HORIZON);
        let q = if t == 0 { r.ep.x0.clone() } else { r.ep.ys[t - 1].clone() }; (e, q, r.ep.ys[t].clone())
    };
    let ce = perm.map_or(e, |p| p[e]); let b = &cdt[ce].base; let yh = b.step(d, &q);
    (q, yh, b.sf, b.c.clone(), y)
}
fn fit(d: &Orbita, corr: &mut Corr, steps: usize, seed: u64, mut draw: impl FnMut(&mut Xoshiro256StarStar) -> Sample) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed); let n = corr.net.n_params();
    let mut opt = Adam::new(n, LR0); let mut p = corr.net.flat(); let _ = d;
    for s in 0..steps {
        let t = s as f64 / steps.max(1) as f64; opt.lr = LR1 + 0.5 * (LR0 - LR1) * (1.0 + (std::f64::consts::PI * t).cos());
        let b: Vec<Sample> = (0..SLEEP_BATCH).map(|_| draw(&mut rng)).filter(|x| fin(&x.1) && fin(&x.4) && fin(&x.0)).collect();
        if b.is_empty() { continue; }
        let mut g = vec![0.0; n]; corr.grad(&b, &mut g);
        if !g.iter().all(|x| x.is_finite()) { continue; }
        opt.step(&mut p, &g); corr.net.set_flat(&p);
    }
}
fn gate_loss(d: &Orbita, corr: &Corr, eps: &[(&Ep, &Base)]) -> f64 { mean(&eps.iter().map(|(e, b)| log_rel(clamp_err(&rollout(d, b, &e.x0, HORIZON, Some(corr))[HORIZON - 1], &e.ys[HORIZON - 1]))).collect::<Vec<_>>()) }

pub struct ArmState { pub arm: Arm, pub corr: Corr, pub stream_err: Vec<f64>, pub used: Vec<bool>, pub sleeps: Vec<serde_json::Value> }

/// un sueño: candidato = checkpoint + entrenamiento; compuerta de no-daño (flujo retenido + conocidas fijas); rollback si falla
fn sleep(d: &Orbita, st: &mut ArmState, cdt: &[Rec], gk: &[(&Ep, &Base)], seed: u64, no: usize, steps: usize) {
    let train: Vec<usize> = (0..cdt.len()).filter(|&i| !cdt[i].gate && (st.arm != Arm::Pos || cdt[i].success)).collect();
    let tr: Vec<usize> = train.iter().copied().filter(|&i| st.arm == Arm::E2 || cdt[i].fit_err < TRUST_TAU).collect();
    let perm: Option<Vec<usize>> = if st.arm == Arm::Shuf { let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_SHUF ^ no as u64); let mut p: Vec<usize> = (0..cdt.len()).collect();
        let mut tix = train.clone(); for i in (1..tix.len()).rev() { let j = rng.gen_range(0..=i); tix.swap(i, j); } for (a, b) in train.iter().zip(&tix) { p[*a] = *b; } Some(p) } else { None };
    let ckpt = st.corr.clone(); let h0 = ckpt.hash();
    if train.is_empty() { st.sleeps.push(serde_json::json!({"no": no, "skipped": true})); return; }
    let mut cand = st.corr.clone();
    fit(d, &mut cand, steps, seed ^ XOR_OPT ^ ((no as u64) << 16), |r| sample(d, cdt, &train, &tr, perm.as_deref(), r));
    let gs: Vec<(&Ep, &Base)> = cdt.iter().filter(|r| r.gate).map(|r| (&r.ep, &r.base)).collect();
    let (ls0, ls1) = (gate_loss(d, &ckpt, &gs), gate_loss(d, &cand, &gs)); let (lk0, lk1) = (gate_loss(d, &ckpt, gk), gate_loss(d, &cand, gk));
    let pass = ls1 < ls0 && lk1 <= lk0 + GATE_TOL_KNOWN;
    let accept = pass || st.arm == Arm::NoRollback;
    if accept { st.corr = cand; } else { st.corr = ckpt; assert_eq!(st.corr.hash(), h0, "rollback no exacto"); }
    st.sleeps.push(serde_json::json!({"no": no, "n_train": train.len(), "n_trusted": tr.len(), "gate_stream": [ls0, ls1], "gate_known": [lk0, lk1], "pass": pass, "accepted": accept}));
}

// ─── evaluación final ────────────────────────────────────────────────────────
fn eval_h(d: &Orbita, held: &[(Ep, Base)], corr: Option<&Corr>) -> serde_json::Value {
    let f = |known: bool| { let mut acc = 0.0; let mut lg = vec![]; let mut n = 0.0; let mut used = 0.0;
        for (e, b) in held.iter().filter(|(e, _)| (e.fam < N_FAM) == known) {
            let (tr, u) = match corr { Some(c) => eco_traj(d, b, &e.x0, &e.ys, K, c), None => (rollout(d, b, &e.x0, HORIZON, None), false) };
            let er = clamp_err(&tr[HORIZON - 1], &e.ys[HORIZON - 1]); acc += (er < mt::REL_ERR_OK) as u8 as f64; lg.push(log_rel(er)); n += 1.0; used += u as u8 as f64; }
        (acc / n, median(lg), used / n) };
    let (ka, kl, ku) = f(true); let (na, nl, nu) = f(false);
    serde_json::json!({"known_acc16": ka, "known_med_log": kl, "known_delta_used": ku, "new_acc16": na, "new_med_log": nl, "new_delta_used": nu})
}
/// E33 (§16): rotate+reflect del TEST v4 iterable, contexto verdadero, rollout libre; ŷ con LAT si s_known(4 pasos verdaderos) > τ.
fn e33(d: &Orbita, iter_ex: &[(Vec<Vec<f64>>, Base, &Example)], corr: Option<&Corr>, base_lat: bool) -> serde_json::Value {
    let mut accs = vec![0.0; EVAL_H.len()]; let mut rels = vec![0.0; EVAL_H.len()];
    for (ys, b, e) in iter_ex {
        let tr = if !base_lat { let mut z = e.input.clone(); (0..HORIZON).map(|_| { z = d.step(&z, &e.context); z.clone() }).collect::<Vec<_>>() }
            else { match corr { Some(c) => eco_traj(d, b, &e.input, ys, K, c).0, None => rollout(d, b, &e.input, HORIZON, None) } };
        for (j, &h) in EVAL_H.iter().enumerate() { let y = target_h(e, h); accs[j] += (fin(&tr[h - 1]) && mt::correct(&tr[h - 1], &y)) as u8 as f64; rels[j] += clamp_err(&tr[h - 1], &y); }
    }
    let n = iter_ex.len() as f64; let accs: Vec<f64> = accs.iter().map(|a| a / n).collect();
    let mut region = 0; for (j, &h) in EVAL_H.iter().enumerate() { if accs[j] >= 0.5 { region = h; } else { break; } }
    serde_json::json!({"acc": accs, "rel": rels.iter().map(|r| r / n).collect::<Vec<_>>(), "region": region, "pass": region >= 8, "n": iter_ex.len()})
}

// ─── ejecución por seed ──────────────────────────────────────────────────────
pub struct RunCfg { pub n_new: usize, pub n_known: usize, pub hpf: usize, pub sleep_steps: usize, pub do_e33: bool }
pub fn run_seed(d: &Orbita, seed: u64, cfg: &RunCfg, arms: &[Arm]) -> serde_json::Value {
    let t0 = std::time::Instant::now();
    let stream = gen_stream(seed, cfg.n_new, cfg.n_known);
    let gate_eps = gen_gate_known(seed, N_GATE_KNOWN);
    let gate_b: Vec<Base> = gate_eps.iter().map(|e| make_base(d, e.fam, e.p, &e.x0, &e.ys, K, None)).collect();
    let gk: Vec<(&Ep, &Base)> = gate_eps.iter().zip(&gate_b).collect();
    let mut sts: Vec<ArmState> = arms.iter().map(|&a| ArmState { arm: a, corr: Corr::new(seed ^ XOR_DPHI, a != Arm::M3), stream_err: vec![], used: vec![], sleeps: vec![] }).collect();
    let (mut e_orb, mut e_lat, mut cdt): (Vec<f64>, Vec<f64>, Vec<Rec>) = (vec![], vec![], vec![]);
    let mut parity_bad = 0usize; let mut nb_ep = 0usize;
    for (i, e) in stream.iter().enumerate() {
        let b = make_base(d, e.fam, e.p, &e.x0, &e.ys, K, None);
        let mut z = e.x0.clone(); for _ in 0..HORIZON { z = d.step(&z, &b.c); } e_orb.push(clamp_err(&z, &e.ys[HORIZON - 1]));
        let tl = rollout(d, &b, &e.x0, HORIZON, None); let el = clamp_err(&tl[HORIZON - 1], &e.ys[HORIZON - 1]); e_lat.push(el);
        for st in sts.iter_mut() { // precuencial: predicción ANTES de aprender del episodio
            let (tr, u) = eco_traj(d, &b, &e.x0, &e.ys, K, &st.corr);
            if st.arm == Arm::Zero { for (p, q) in tr.iter().zip(&tl) { for (a, c) in p.iter().zip(q) { if a.to_bits() != c.to_bits() { parity_bad += 1; } } } nb_ep += 1; }
            st.stream_err.push(clamp_err(&tr[HORIZON - 1], &e.ys[HORIZON - 1])); st.used.push(u);
        }
        let (teacher, fit_err) = teacher_of(d, &b, e);
        cdt.push(Rec { ep: e.clone(), base: b, teacher, fit_err, success: el < mt::REL_ERR_OK, gate: i % 4 == 3 });
        for st in sts.iter_mut() {
            match st.arm {
                Arm::Zero => {}
                Arm::NoSleep => if i % 4 != 3 { let per = cfg.sleep_steps / (SLEEP_EVERY * 3 / 4); let one = [cdt.len() - 1];
                    let mut c2 = st.corr.clone(); fit(d, &mut c2, per, seed ^ XOR_OPT ^ ((i as u64) << 24), |r| sample(d, &cdt, &one, &[], None, r)); st.corr = c2; },
                _ => if (i + 1) % SLEEP_EVERY == 0 { let no = (i + 1) / SLEEP_EVERY; sleep(d, st, &cdt, &gk, seed, no, cfg.sleep_steps); }
            }
        }
    }
    let held: Vec<(Ep, Base)> = gen_heldout(seed, cfg.hpf).into_iter().map(|e| { let b = make_base(d, e.fam, e.p, &e.x0, &e.ys, K, None); (e, b) }).collect();
    let newi: Vec<usize> = (0..stream.len()).filter(|&i| stream[i].fam >= N_FAM).collect();
    let knwi: Vec<usize> = (0..stream.len()).filter(|&i| stream[i].fam < N_FAM).collect();
    let auc = |v: &[f64], ix: &[usize]| mean(&ix.iter().map(|&i| log_rel(v[i])).collect::<Vec<_>>());
    let curve = |v: &[f64]| newi.iter().map(|&i| log_rel(v[i])).collect::<Vec<_>>();
    // E33 sobre el TEST v4 iterable
    let mut e33s = serde_json::Map::new(); let mut test_sha = String::new();
    if cfg.do_e33 {
        let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS)); s.verify_test().expect("TEST sealed"); test_sha = s.manifest.test_sha256.clone();
        let it: Vec<(Vec<Vec<f64>>, Base, &Example)> = s.ds.test.iter().filter(|e| ITERABLE_FAMS.contains(&e.family_id)).map(|e| {
            let ys: Vec<Vec<f64>> = (1..=HORIZON).map(|h| target_h(e, h)).collect(); let b = make_base(d, 0, [0.0; 2], &e.input, &ys, K, Some(&e.context)); (ys, b, e) }).collect();
        e33s.insert("orbita".into(), e33(d, &it, None, false)); e33s.insert("orbita_lat".into(), e33(d, &it, None, true));
        for st in &sts { e33s.insert(st.arm.name().into(), e33(d, &it, Some(&st.corr), true)); }
        // e33 libera `s` aquí; ninguna consulta CDT/RQM/tabla
        let _ = &s;
    }
    let mut arms_j = serde_json::Map::new();
    arms_j.insert("orbita".into(), serde_json::json!({"auc_new": auc(&e_orb, &newi), "auc_known": auc(&e_orb, &knwi), "curve_new": curve(&e_orb), "held": eval_h(d, &held_orbita(d, &held), None)}));
    arms_j.insert("orbita_lat".into(), serde_json::json!({"auc_new": auc(&e_lat, &newi), "auc_known": auc(&e_lat, &knwi), "curve_new": curve(&e_lat), "held": eval_h(d, &held, None)}));
    for st in &sts {
        let used_new = mean(&newi.iter().map(|&i| st.used[i] as u8 as f64).collect::<Vec<_>>());
        arms_j.insert(st.arm.name().into(), serde_json::json!({"auc_new": auc(&st.stream_err, &newi), "auc_known": auc(&st.stream_err, &knwi), "curve_new": curve(&st.stream_err),
            "delta_used_new": used_new, "held": eval_h(d, &held, Some(&st.corr)), "sleeps": st.sleeps, "n_accepted": st.sleeps.iter().filter(|s| s["accepted"] == true).count(), "corr_hash": format!("{:016x}", st.corr.hash())}));
    }
    let trusted = cdt.iter().filter(|r| r.fit_err < TRUST_TAU).count(); let succ = cdt.iter().filter(|r| r.success).count();
    serde_json::json!({"seed": format!("0x{seed:X}"), "arms": arms_j, "e33": e33s, "test_sha256": test_sha,
        "parity_zero_vs_orbita_lat": {"mismatch": parity_bad, "episodes": nb_ep}, "cdt": {"n": cdt.len(), "trusted": trusted, "success": succ,
        "fit_err_median": median(cdt.iter().map(|r| r.fit_err).collect())}, "lat_trigger_new": mean(&newi.iter().map(|&i| cdt[i].base.lat.is_some() as u8 as f64).collect::<Vec<_>>()),
        "wall_s": t0.elapsed().as_secs_f64()})
}
/// H para el brazo «solo ÓRBITA»: misma ĉ, sin LAT
fn held_orbita(d: &Orbita, held: &[(Ep, Base)]) -> Vec<(Ep, Base)> {
    held.iter().map(|(e, b)| (e.clone(), Base { c: b.c.clone(), cp: d.ctx(&b.c), g: b.g, sk: b.sk, sf: b.sf, lat: None })).collect()
}

fn arg_of(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned() }
pub fn parse_seed(s: &str) -> u64 { u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap() }

pub fn run(a: &[String]) {
    let wpath = arg_of(a, "--weights").expect("--weights"); let wtxt = std::fs::read_to_string(&wpath).unwrap();
    let d = Orbita::from_json(&wtxt); let raw0 = d.raw.clone();
    let seed = parse_seed(&arg_of(a, "--seed").expect("--seed"));
    let phase = arg_of(a, "--phase").unwrap_or_else(|| "dev".into());
    let smoke = phase == "smoke";
    if phase == "dev" { assert!((0xA400..=0xA40F).contains(&seed)); } else if phase == "confirm" { assert!((0xB400..=0xB40F).contains(&seed)); } else { assert!(smoke && !(0xA400..=0xA40F).contains(&seed) && !(0xB400..=0xB40F).contains(&seed), "smoke sólo fuera de DEV/CONFIRM"); }
    let g = |k: &str, dflt: usize| arg_of(a, k).map(|s| s.parse().unwrap()).unwrap_or(dflt);
    let cfg = RunCfg { n_new: g("--n-new", N_NEW_PER_FAM), n_known: g("--n-known", N_KNOWN_STREAM), hpf: g("--h-per-fam", H_PER_FAM), sleep_steps: g("--sleep-steps", SLEEP_STEPS), do_e33: true };
    if !smoke { assert!(cfg.n_new == N_NEW_PER_FAM && cfg.n_known == N_KNOWN_STREAM && cfg.hpf == H_PER_FAM && cfg.sleep_steps == SLEEP_STEPS, "DEV/CONFIRM usan los valores congelados"); }
    let out = PathBuf::from(arg_of(a, "--out").expect("--out")).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip 0x{seed:X}"); return; }
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let mut r = run_seed(&d, seed, &cfg, &Arm::all());
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before);
    assert_eq!(d.raw, raw0, "núcleo ÓRBITA modificado");
    assert_eq!(r["parity_zero_vs_orbita_lat"]["mismatch"], 0, "Paso 0b falla en línea");
    r["test_queries"] = serde_json::json!(q.total()); r["phase"] = serde_json::json!(phase); r["weights"] = serde_json::json!(wpath);
    r["git_sha"] = serde_json::json!(std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default());
    assert_eq!(q.total(), 0, "consultas TEST ≠ 0");
    std::fs::write(out.join(format!("metrics_0x{seed:X}.json")), serde_json::to_string(&r).unwrap()).unwrap();
    std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok wall={:.1}s auc_new eco={:.4} orbita_lat={:.4}", r["wall_s"].as_f64().unwrap(), r["arms"]["eco"]["auc_new"].as_f64().unwrap(), r["arms"]["orbita_lat"]["auc_new"].as_f64().unwrap());
}

// ─── Paso 0 ──────────────────────────────────────────────────────────────────
/// 0a: re-evalúa los checkpoints E3 de artifacts/v4/distill_e/{dev,confirm} y compara con metrics.csv.
pub fn paso0a(v4: &str) -> serde_json::Value {
    let mut out = serde_json::Map::new(); let mut ok = true;
    for ph in ["dev", "confirm"] {
        let csv = std::fs::read_to_string(format!("{v4}/{ph}/metrics.csv")).unwrap();
        let mut rows: Vec<(u64, f64)> = vec![];
        for l in csv.lines().skip(1) { let f: Vec<&str> = l.splitn(9, ',').collect(); if f.len() > 8 && f[2] == "E3_gated_trusted" && f[3] == "TEST" { rows.push((parse_seed(f[1]), f[7].parse().unwrap())); } }
        let mut per = vec![]; let (mut s_csv, mut s_re) = (0.0, 0.0); let mut maxd: f64 = 0.0; let mut qs = 0u64;
        for (seed, acc_csv) in &rows {
            let ck = std::fs::read_to_string(format!("{v4}/{ph}/checkpoints/dphi_DIST_E_E3_0x{seed:X}.json")).unwrap();
            let dp = Dphi::from_ckpt(&ck).unwrap(); let s = generate_and_seal(*seed, &DatasetConfig::standard(dp.zdim / 2));
            let b = cdt_rqm_epr::v4_provenance::snapshot(); let ev = evaluate(&dp, &s.ds.test, 1, None, None); qs += cdt_rqm_epr::v4_provenance::snapshot().delta(&b).total();
            maxd = maxd.max((ev.acc - acc_csv).abs()); s_csv += acc_csv; s_re += ev.acc;
            per.push(serde_json::json!({"seed": format!("0x{seed:X}"), "acc_csv": acc_csv, "acc_reeval": ev.acc, "rel_err1": ev.rel_err, "n": ev.n}));
        }
        let n = rows.len() as f64; let pass = rows.len() == 16 && maxd <= 5e-5 && qs == 0; ok &= pass;
        out.insert(ph.into(), serde_json::json!({"n_seeds": rows.len(), "mean_acc_csv": s_csv / n, "mean_acc_reeval": s_re / n, "max_abs_diff_vs_csv_4dp": maxd, "queries": qs, "pass": pass, "per_seed": per}));
    }
    out.insert("pass".into(), serde_json::json!(ok)); serde_json::Value::Object(out)
}
/// 0b: Δ ≡ 0 ⇒ ECO == ÓRBITA+LAT bit a bit (flujo reducido, H y corrector aplicado directamente)
pub fn paso0b(d: &Orbita, seed: u64) -> (usize, usize) {
    let c = Corr::new(seed ^ XOR_DPHI, true); let (mut bad, mut n) = (0, 0);
    let mut eps = gen_stream(seed, 4, 8); eps.extend(gen_heldout(seed, 2));
    for e in &eps {
        let b = make_base(d, e.fam, e.p, &e.x0, &e.ys, K, None);
        let tl = rollout(d, &b, &e.x0, HORIZON, None); let tz = rollout(d, &b, &e.x0, HORIZON, Some(&c)); let (ta, _) = eco_traj(d, &b, &e.x0, &e.ys, K, &c);
        for t in 0..HORIZON { for j in 0..tl[t].len() { n += 2; if tl[t][j].to_bits() != tz[t][j].to_bits() { bad += 1; } if tl[t][j].to_bits() != ta[t][j].to_bits() { bad += 1; } } }
    }
    (bad, n)
}
pub fn paso0(a: &[String]) {
    let ws: Vec<String> = a.iter().enumerate().filter(|(_, x)| *x == "--weights").map(|(i, _)| a[i + 1].clone()).collect();
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC400".into()));
    assert!(!(0xA400..=0xA40F).contains(&seed) && !(0xB400..=0xB40F).contains(&seed));
    let v4 = arg_of(a, "--v4").unwrap_or_else(|| "artifacts/v4/distill_e".into());
    let a0 = paso0a(&v4);
    let mut res = vec![]; let mut b_ok = true;
    for w in &ws { let d = Orbita::from_json(&std::fs::read_to_string(w).unwrap()); let r = paso0b(&d, seed); b_ok &= r.0 == 0; res.push(serde_json::json!({"weights": w, "mismatch": r.0, "values": r.1})); }
    let c_ok = crate::eco::d3_layout_ok();
    let all = a0["pass"] == true && b_ok && c_ok;
    println!("{}", serde_json::json!({"paso0_ok": all, "0a": a0, "0b": {"pass": b_ok, "data_seed": format!("0x{seed:X}"), "checks": res}, "0c": {"pass": c_ok}}));
    if !all { std::process::exit(1); }
}
/// 0c: ĉ de ECO == vector D3-num de d3.py (one-hot argmax en 0..8, p̂ en 8..10, recorte a [−1,1]) == context_of para p ∈ [−1,1]
pub fn d3_layout_ok() -> bool {
    let mut ok = true;
    for g in 0..N_FAM { for &p in &[[0.3, -0.7], [-1.0, 1.0], [0.0, 0.123]] { ok &= d3num_ctx(g, p) == context_of(g, p); } }
    ok &= d3num_ctx(2, [1.7, -3.0]) == vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0];
    let inp = Corr::inp(&[1.0, 2.0], &[0.5, 0.25], 0.1, &context_of(3, [0.2, 0.4]));
    ok &= inp.len() == 2 + CDIM && inp[7..] == context_of(3, [0.2, 0.4])[..];
    ok
}

#[cfg(test)]
mod tests;
