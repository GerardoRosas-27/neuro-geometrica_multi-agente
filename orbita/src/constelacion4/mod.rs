//! CONSTELACIÓN v4 (docs/constelacion_v4: v3 + cabeza 1-paso informada por ÓRBITA para E33 + más entrenamiento; orb_prior=false ≡ v3). CONSTELACIÓN v3 (docs/constelacion_v3; v3=false ≡ v2 bit a bit). Base v2 (docs/constelacion_v2): red jerárquica de ÓRBITAS-neurona (Koopman bloque rot-escala + levantamiento
//! monótono, fijos y aleatorios) con predictive coding, plasticidad local, precisiones, τ/poda y sueño CDT (sin rollback).
//!   orbita cst-paso0 --seed 0xC400 --weights W           (JSON)
//!   orbita cst-f0 --seed S        orbita cst-f1 --seed S --weights W
//!   orbita cst-run --phase dev|confirm|smoke --seed S --weights W --out DIR
#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::type_complexity)]
use crate::fam::reid;
use crate::model::Orbita;
use cdt_rqm_epr::field_autonomy_v4::{target_h, ITERABLE_FAMS, K_POINTS};
use cdt_rqm_epr::v4_dataset::{apply_rule, context_of, generate_and_seal, make_object, DatasetConfig, Example, ObjectClass, N_FAM};
use cdt_rqm_epr::v4_metrics as mt;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde_json::json;
use std::path::PathBuf;

pub const K: usize = 4;
pub const HORIZON: usize = 16;
pub const NEW_FAMS: [usize; 3] = [8, 10, 11];
pub const EVAL_H: [usize; 5] = [1, 2, 4, 8, 16];
const XOR_HELD: u64 = 0x4E0A_501D; // H idéntico a ESTELA/ECO
const XOR_STREAM: u64 = 0xEC05_7EA4; // flujo B idéntico a ECO
const XOR_KNOWN: u64 = 0xC057_A11A; // flujo A (conocidas)
const XOR_INIT: u64 = 0xC057_1417;
const XOR_KBUF: u64 = 0xC057_B0F0;
const XOR_SLEEP: u64 = 0xC057_5EE9;

// ─── familias (copia fiel de exp/eco @ b731ff3, = estela @ db42e6a) ─────────
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
impl Ep { pub fn frame(&self, t: usize) -> &[f64] { if t == 0 { &self.x0 } else { &self.ys[t - 1] } } }
pub fn make_ep(rng: &mut Xoshiro256StarStar, fam: usize) -> Ep {
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
    let _c = if fam < N_FAM { context_of(fam, p) } else { context_of(0, p) };
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
pub fn gen_stream(seed: u64, n_new: usize, n_known: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_STREAM);
    let mut fams: Vec<usize> = vec![]; for &f in &NEW_FAMS { for _ in 0..n_new { fams.push(f); } } for i in 0..n_known { fams.push(i % N_FAM); }
    for i in (1..fams.len()).rev() { let j = rng.gen_range(0..=i); fams.swap(i, j); }
    fams.iter().map(|&f| make_ep(&mut rng, f)).collect()
}
/// flujo A: sólo conocidas (o una familia), barajado
pub fn gen_known_stream(seed: u64, xor: u64, n: usize, fams: &[usize]) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ xor);
    let mut fs: Vec<usize> = (0..n).map(|i| fams[i % fams.len()]).collect();
    for i in (1..fs.len()).rev() { let j = rng.gen_range(0..=i); fs.swap(i, j); }
    fs.iter().map(|&f| make_ep(&mut rng, f)).collect()
}
fn fin(z: &[f64]) -> bool { z.iter().all(|v| v.is_finite()) }
pub fn clamp_err(z: &[f64], y: &[f64]) -> f64 { if fin(z) { mt::rel_err(z, y).min(10.0) } else { 10.0 } }
pub fn log_rel(e: f64) -> f64 { e.max(1e-12).ln() }
fn mean(v: &[f64]) -> f64 { if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 } }
fn median(mut v: Vec<f64>) -> f64 { if v.is_empty() { return f64::NAN; } v.sort_by(|a, b| a.total_cmp(b)); mt::median(&v) }
fn gauss(rng: &mut Xoshiro256StarStar) -> f64 { let u1: f64 = rng.gen_range(1e-300..1.0); let u2: f64 = rng.gen(); (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos() }
fn sig(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

// ─── configuración ───────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct Cfg {
    pub m: usize, pub h: usize, pub layers: Vec<usize>, pub identity_init: bool, pub eta_b: f64,
    pub t_inf: usize, pub kappa: f64, pub tol: f64,
    pub eta_w: f64, pub lam: f64, pub w_init: f64,
    pub eta_pi: f64, pub pi_min: f64, pub pi_max: f64,
    pub tau_min: f64, pub tau_max: f64, pub zeta: f64, pub rho_tau: f64,
    pub p_every: usize, pub w_min: f64, pub u_min_rel: f64,
    pub b_in: f64, pub rho_max: f64, pub mu_clip: f64,
    pub n_s: usize, pub n_replay: usize, pub alpha_s: f64, pub delta_k: f64, pub n_kbuf: usize,
    // flags de control
    pub errors: bool, pub precision: bool, pub prune: bool, pub sleep: bool, pub plastic_w: bool, pub backprop: bool,
    // v3: cabeza de expectativa temporal residual (x̂(t+1) = x(t) + (W^P+W^F)φ), campo rápido por episodio, Π normalizada, rollout reinyectado
    pub v3: bool, pub feat_x: bool, pub feat_s: f64, pub eta_p: f64, pub eta_f: f64, pub t_f: usize, pub pi_norm: bool, pub fast: bool, pub pip_min: f64, pub pip_max: f64,
    // v4: x̂(1) = ÓRBITA(x0, c) cuando el contexto está disponible y aún no se ha observado ninguna transición (W^F vacío)
    pub orb_prior: bool,
}
impl Cfg {
    pub fn base() -> Cfg { Cfg { m: 8, h: 8, layers: vec![16, 8, 4], identity_init: false, eta_b: 0.1, t_inf: 8, kappa: 0.1, tol: 1e-3,
        eta_w: 0.01, lam: 1e-6, w_init: 0.1, eta_pi: 1e-3, pi_min: 0.01, pi_max: 100.0, tau_min: 1.0, tau_max: 10.0, zeta: 0.01, rho_tau: 0.01,
        p_every: 500, w_min: 1e-4, u_min_rel: 0.01, b_in: 10.0, rho_max: 0.99, mu_clip: 10.0,
        n_s: 16, n_replay: 16, alpha_s: 0.1, delta_k: 0.05, n_kbuf: 16,
        errors: true, precision: true, prune: true, sleep: true, plastic_w: true, backprop: false,
        v3: false, feat_x: false, feat_s: 1.0, eta_p: 0.0, eta_f: 0.0, t_f: 0, pi_norm: false, fast: false, pip_min: 0.1, pip_max: 10.0, orb_prior: false } }
    /// v3 congelada (valores fijados en docs/constelacion_v3/PREREG.md)
    pub fn v3() -> Cfg { let mut c = Cfg::base(); c.v3 = true; c.w_init = 1.0; c.b_in = 40.0; c.pi_min = V3_PI_MIN; c.pi_max = V3_PI_MAX; c.pi_norm = true; c.fast = true;
        c.feat_x = V3_FEAT_X; c.feat_s = V3_FEAT_S; c.eta_p = V3_ETA_P; c.eta_f = V3_ETA_F; c.t_f = V3_T_F; c }
    /// v4 congelada (docs/constelacion_v4/PREREG.md)
    pub fn v4() -> Cfg { let mut c = Cfg::v3(); c.orb_prior = true; c.t_f = V4_T_F; c }
    pub fn to_json(&self) -> serde_json::Value { json!(format!("{:?}", self)) }
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Arm { Cst, NoErr, NoPrec, T1, NoPrune, NoSleep, FixedW, OneLayer, Backprop, NoFast }
impl Arm {
    pub fn all() -> [Arm; 9] { [Arm::Cst, Arm::NoErr, Arm::NoPrec, Arm::T1, Arm::NoPrune, Arm::NoSleep, Arm::FixedW, Arm::OneLayer, Arm::Backprop] }
    pub fn all_v3() -> [Arm; 10] { [Arm::Cst, Arm::NoErr, Arm::NoPrec, Arm::T1, Arm::NoPrune, Arm::NoSleep, Arm::FixedW, Arm::OneLayer, Arm::Backprop, Arm::NoFast] }
    pub fn name(self) -> &'static str { match self { Arm::Cst => "cst", Arm::NoErr => "no_err", Arm::NoPrec => "no_prec", Arm::T1 => "t1", Arm::NoPrune => "no_prune", Arm::NoSleep => "no_sleep", Arm::FixedW => "fixed_w", Arm::OneLayer => "one_layer", Arm::Backprop => "backprop", Arm::NoFast => "no_fast" } }
    pub fn cfg(self, b: &Cfg) -> Cfg { let mut c = b.clone(); match self {
        Arm::Cst => {}, Arm::NoErr => c.errors = false, Arm::NoPrec => c.precision = false, Arm::T1 => c.t_inf = 1, Arm::NoPrune => c.prune = false,
        Arm::NoSleep => c.sleep = false, Arm::FixedW => c.plastic_w = false, Arm::OneLayer => c.layers = vec![28], Arm::Backprop => c.backprop = true, Arm::NoFast => c.fast = false } c }
}

// ─── ÓRBITA-neurona ──────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct Neuron { pub rho: Vec<f64>, pub om: Vec<f64>, pub kb: Vec<[f64; 4]>, pub r: Vec<f64>, pub b: Vec<f64>, pub eta_b: f64, m: usize, h: usize }
impl Neuron {
    pub fn new(rng: &mut Xoshiro256StarStar, c: &Cfg) -> Neuron {
        let (m, h) = (c.m, c.h); let nb = m / 2;
        let (rho, om): (Vec<f64>, Vec<f64>) = if c.identity_init { (vec![1.0; nb], vec![0.0; nb]) } else {
            (0..nb).map(|_| (rng.gen_range(0.90..0.99), rng.gen_range(-std::f64::consts::FRAC_PI_4..std::f64::consts::FRAC_PI_4))).unzip() };
        let kb = rho.iter().zip(&om).map(|(r, w)| [r * w.cos(), -r * w.sin(), r * w.sin(), r * w.cos()]).collect();
        let sd = (1.0 / h as f64).sqrt();
        let r = (0..h * m).map(|_| gauss(rng) * sd).collect(); let b = (0..h).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Neuron { rho, om, kb, r, b, eta_b: if c.identity_init { 0.0 } else { c.eta_b }, m, h }
    }
    fn kmul(&self, x: &[f64], v: &mut [f64]) { for j in 0..self.m / 2 { let k = &self.kb[j]; v[2 * j] = k[0] * x[2 * j] + k[1] * x[2 * j + 1]; v[2 * j + 1] = k[2] * x[2 * j] + k[3] * x[2 * j + 1]; } }
    fn ktmul(&self, x: &[f64], v: &mut [f64]) { for j in 0..self.m / 2 { let k = &self.kb[j]; v[2 * j] = k[0] * x[2 * j] + k[2] * x[2 * j + 1]; v[2 * j + 1] = k[1] * x[2 * j] + k[3] * x[2 * j + 1]; } }
    /// out = ψ(Kμ); s = σ(R K μ + b)
    pub fn f(&self, mu: &[f64], s: &mut [f64], out: &mut [f64]) {
        let mut v = [0.0; 16]; self.kmul(mu, &mut v[..self.m]);
        for i in 0..self.h { let row = &self.r[i * self.m..(i + 1) * self.m]; let mut a = self.b[i]; for j in 0..self.m { a += row[j] * v[j]; } s[i] = sig(a); }
        for j in 0..self.m { out[j] = v[j]; }
        if self.eta_b != 0.0 { for i in 0..self.h { let c = self.eta_b * s[i]; let row = &self.r[i * self.m..(i + 1) * self.m]; for j in 0..self.m { out[j] += c * row[j]; } } }
    }
    /// Jᵀ g = Kᵀ (g + η_b Rᵀ diag(s(1−s)) R g)
    pub fn jt(&self, s: &[f64], g: &[f64], out: &mut [f64]) {
        let mut u = [0.0; 16]; for j in 0..self.m { u[j] = g[j]; }
        if self.eta_b != 0.0 { for i in 0..self.h { let row = &self.r[i * self.m..(i + 1) * self.m]; let mut a = 0.0; for j in 0..self.m { a += row[j] * g[j]; } let c = self.eta_b * s[i] * (1.0 - s[i]) * a; for j in 0..self.m { u[j] += c * row[j]; } } }
        self.ktmul(&u[..self.m], out);
    }
    /// J v = (I + η_b Rᵀ diag(s') R) K v
    pub fn jv(&self, s: &[f64], x: &[f64], out: &mut [f64]) {
        let mut v = [0.0; 16]; self.kmul(x, &mut v[..self.m]); for j in 0..self.m { out[j] = v[j]; }
        if self.eta_b != 0.0 { for i in 0..self.h { let row = &self.r[i * self.m..(i + 1) * self.m]; let mut a = 0.0; for j in 0..self.m { a += row[j] * v[j]; } let c = self.eta_b * s[i] * (1.0 - s[i]) * a; for j in 0..self.m { out[j] += c * row[j]; } } }
    }
    pub fn spec_k(&self) -> f64 { self.rho.iter().cloned().fold(0.0, f64::max) }
    /// cota ‖J_ψ‖ ≤ 1 + η_b·máx σ′·‖R‖² (máx σ′ = 1/4), ‖R‖ espectral por iteración de potencias
    pub fn jpsi_bound(&self) -> f64 {
        let mut v = vec![1.0; self.m]; let mut nrm = 0.0;
        for _ in 0..200 { let mut w = vec![0.0; self.h]; for i in 0..self.h { for j in 0..self.m { w[i] += self.r[i * self.m + j] * v[j]; } }
            let mut z = vec![0.0; self.m]; for i in 0..self.h { for j in 0..self.m { z[j] += self.r[i * self.m + j] * w[i]; } }
            nrm = z.iter().map(|a| a * a).sum::<f64>().sqrt(); for j in 0..self.m { v[j] = z[j] / nrm; } }
        1.0 + self.eta_b * 0.25 * nrm
    }
}

// ─── red ─────────────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct Net {
    pub c: Cfg, pub dims: Vec<usize>, pub neu: Vec<Vec<Neuron>>,
    pub w: Vec<Vec<f64>>, pub cm: Vec<Vec<f64>>, pub tau: Vec<Vec<f64>>, pub u: Vec<Vec<f64>>, pub low: Vec<Vec<u8>>, pub act: Vec<Vec<f64>>,
    pub pi: Vec<Vec<f64>>, pub pit: Vec<Vec<f64>>, pub uin: Vec<f64>, pub eta_scale: f64, pub steps: usize, pub nan: bool,
    pub n_pruned: usize, pub n_grown: usize, pub n_rescale_row: usize, pub n_rescale_spec: usize,
    pub curv: Vec<Vec<f64>>, pub wp: Vec<f64>, pub taup: Vec<f64>, pub up: Vec<f64>, pub wf: Vec<f64>, pub pip: Vec<f64>, pub dp: usize,
}
#[derive(Clone)]
pub struct PState { pub mu: Vec<Vec<f64>>, pub started: bool, pub xl: Vec<f64> }
pub struct StepInfo { pub e0: f64, pub ef: f64, pub e_layer: Vec<f64>, pub iters: f64 }

impl Net {
    pub fn new(c: &Cfg, seed: u64) -> Net {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_INIT);
        let l = c.layers.len(); let mut dims = vec![2]; for &n in &c.layers { dims.push(n * c.m); }
        let dp = if c.feat_x { 2 } else { 0 } + dims[1] + 1;
        let neu: Vec<Vec<Neuron>> = c.layers.iter().map(|&n| (0..n).map(|_| Neuron::new(&mut rng, c)).collect()).collect();
        let mut w = vec![]; for i in 1..=l { let fi = dims[i] as f64; w.push((0..dims[i - 1] * dims[i]).map(|_| gauss(&mut rng) * (1.0 / fi).sqrt() * c.w_init).collect::<Vec<f64>>()); }
        let sz = |i: usize| dims[i - 1] * dims[i];
        let uin = (0..dims[l] * 2).map(|_| gauss(&mut rng) * 0.5).collect();
        let mut net = Net { dims: dims.clone(), neu, cm: (1..=l).map(|i| vec![1.0; sz(i)]).collect(), tau: (1..=l).map(|i| vec![c.tau_min; sz(i)]).collect(),
            u: (1..=l).map(|i| vec![0.0; sz(i)]).collect(), low: (1..=l).map(|i| vec![0; sz(i)]).collect(), act: (1..=l).map(|i| vec![0.0; sz(i)]).collect(),
            pi: (0..l).map(|i| vec![1.0; dims[i]]).collect(), pit: (1..=l).map(|i| vec![1.0; dims[i]]).collect(), w, uin, eta_scale: 1.0, steps: 0, nan: false,
            n_pruned: 0, n_grown: 0, n_rescale_row: 0, n_rescale_spec: 0, c: c.clone(),
            curv: vec![], wp: vec![0.0; 2 * dp], taup: vec![c.tau_min; 2 * dp], up: vec![0.0; 2 * dp], wf: vec![0.0; 2 * dp], pip: vec![1.0; 2], dp };
        net.update_curv(); net
    }
    pub fn begin_episode(&mut self) { for v in self.wf.iter_mut() { *v = 0.0; } self.update_curv(); }
    /// curvatura diagonal de la energía respecto de μ_l por el término de abajo: Σ_i Π_{l−1,i} (C⊙W_l)_ij²  (se recalcula al empezar cada episodio y tras cada paso de plasticidad)
    pub fn update_curv(&mut self) { if !self.c.v3 { return; } let l = self.nl();
        self.curv = (1..=l).map(|li| { let (r, cc) = (self.dims[li - 1], self.dims[li]); (0..cc).map(|j| (0..r).map(|i| { let w = self.w[li - 1][i * cc + j] * self.cm[li - 1][i * cc + j]; self.pi[li - 1][i] * w * w }).sum()).collect() }).collect(); }
    /// rasgos presinápticos de la cabeza temporal: [x(t)?; s/√d·F_1(μ_1); 1]
    pub fn feat(&self, st: &PState) -> Vec<f64> { let mut f = Vec::with_capacity(self.dp); if self.c.feat_x { f.extend_from_slice(&st.xl); }
        let s = self.c.feat_s / (self.dims[1] as f64).sqrt(); f.extend(self.fl(1, &st.mu[0]).0.iter().map(|v| v * s)); f.push(1.0); f }
    /// x̂(t+1) = x(t) + (W^P + W^F) φ  (v3)
    pub fn pred_from(&self, st: &PState, phi: &[f64]) -> Vec<f64> { let d = self.dp; (0..2).map(|i| st.xl[i] + (0..d).map(|j| (self.wp[i * d + j] + self.wf[i * d + j]) * phi[j]).sum::<f64>()).collect() }
    pub fn predict1(&self, st: &PState) -> Vec<f64> { if !self.c.v3 { return self.predict(st, 1); } if !st.started { return vec![0.0; 2]; } self.pred_from(st, &self.feat(st)) }
    pub fn nl(&self) -> usize { self.c.layers.len() }
    pub fn n_w(&self) -> usize { self.w.iter().map(|x| x.len()).sum() }
    pub fn hash(&self) -> u64 { let mut h: u64 = 0xcbf29ce484222325; for v in self.w.iter().chain(&self.pi).chain(&self.pit).chain(&self.tau).chain(&self.cm).flatten() { h ^= v.to_bits(); h = h.wrapping_mul(0x100000001b3); }
        if self.c.v3 { for v in self.wp.iter().chain(&self.taup).chain(&self.pip) { h ^= v.to_bits(); h = h.wrapping_mul(0x100000001b3); } } h }
    pub fn fresh_state(&self) -> PState { PState { mu: (1..=self.nl()).map(|i| vec![0.0; self.dims[i]]).collect(), started: false, xl: vec![0.0; 2] } }
    /// F_l aplicado por neurona; devuelve (F(μ), s)
    pub fn fl(&self, l: usize, mu: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (m, h) = (self.c.m, self.c.h); let n = self.c.layers[l - 1];
        let mut out = vec![0.0; n * m]; let mut s = vec![0.0; n * h];
        for k in 0..n { self.neu[l - 1][k].f(&mu[k * m..(k + 1) * m], &mut s[k * h..(k + 1) * h], &mut out[k * m..(k + 1) * m]); }
        (out, s)
    }
    fn jtl(&self, l: usize, s: &[f64], g: &[f64]) -> Vec<f64> {
        let (m, h) = (self.c.m, self.c.h); let n = self.c.layers[l - 1]; let mut out = vec![0.0; n * m];
        for k in 0..n { self.neu[l - 1][k].jt(&s[k * h..(k + 1) * h], &g[k * m..(k + 1) * m], &mut out[k * m..(k + 1) * m]); } out
    }
    fn jvl(&self, l: usize, s: &[f64], x: &[f64]) -> Vec<f64> {
        let (m, h) = (self.c.m, self.c.h); let n = self.c.layers[l - 1]; let mut out = vec![0.0; n * m];
        for k in 0..n { self.neu[l - 1][k].jv(&s[k * h..(k + 1) * h], &x[k * m..(k + 1) * m], &mut out[k * m..(k + 1) * m]); } out
    }
    /// x̂_{l−1} = (C⊙W_l) v
    fn wv(&self, l: usize, v: &[f64]) -> Vec<f64> {
        let (r, c) = (self.dims[l - 1], self.dims[l]); let w = &self.w[l - 1]; let cm = &self.cm[l - 1];
        (0..r).map(|i| { let mut a = 0.0; for j in 0..c { a += cm[i * c + j] * w[i * c + j] * v[j]; } a }).collect()
    }
    fn wtv(&self, l: usize, g: &[f64]) -> Vec<f64> {
        let (r, c) = (self.dims[l - 1], self.dims[l]); let w = &self.w[l - 1]; let cm = &self.cm[l - 1]; let mut out = vec![0.0; c];
        for i in 0..r { let gi = g[i]; if gi == 0.0 { continue; } for j in 0..c { out[j] += cm[i * c + j] * w[i * c + j] * gi; } } out
    }
    /// predicción descendente: μ_L avanza h pasos con F_L, luego x̂_{l−1} = W_l F_l(μ_l) hasta x̂_0
    pub fn predict(&self, st: &PState, h: usize) -> Vec<f64> {
        let l = self.nl(); let mut top = st.mu[l - 1].clone();
        for _ in 0..h { top = self.fl(l, &top).0; }
        let mut cur = top; for ll in (1..=l).rev() { let (f, _) = self.fl(ll, &cur); cur = self.wv(ll, &f); } cur
    }
    fn clipv(&self, v: &mut [f64]) { let c = self.c.mu_clip; for x in v.iter_mut() { *x = x.clamp(-c, c); } }
}

/// Acumuladores de plasticidad de un paso (promedio sobre puntos)
struct Acc { gw: Vec<Vec<f64>>, e2: Vec<Vec<f64>>, et2: Vec<Vec<f64>>, red: f64, np: f64, nt: f64 }

/// inferencia iterativa de un punto (la misma función en train y eval)
fn infer_point(net: &Net, st: &mut PState, x: &[f64], acc: Option<&mut Acc>) -> (f64, f64, Vec<f64>, f64) {
    let l = net.nl(); let c = &net.c;
    let started = st.started;
    if !c.errors {
        // control «sin neuronas de error»: la capa superior recibe x directamente (reservorio), sin ε ascendente
        let top = &mut st.mu[l - 1]; let mut nt = if started { net.fl(l, top).0 } else { vec![0.0; top.len()] };
        for i in 0..nt.len() { nt[i] += net.uin[i * 2] * x[0] + net.uin[i * 2 + 1] * x[1]; }
        *top = nt; net.clipv(top);
        let mut cur = st.mu[l - 1].clone();
        for ll in (2..=l).rev() { let (f, _) = net.fl(ll, &cur); cur = net.wv(ll, &f); net.clipv(&mut cur); st.mu[ll - 2] = cur.clone(); }
        let (f1, _) = net.fl(1, &st.mu[0]); let xh = net.wv(1, &f1); let e: Vec<f64> = (0..2).map(|i| x[i] - xh[i]).collect();
        let en = 0.5 * (e[0] * e[0] + e[1] * e[1]);
        if let Some(a) = acc { for i in 0..2 { for j in 0..f1.len() { a.gw[0][i * f1.len() + j] += e[i] * f1[j]; } a.e2[0][i] += e[i] * e[i]; } a.np += 1.0; }
        st.started = true; return (en, en, vec![en], 1.0);
    }
    let prior: Vec<Vec<f64>> = if started { (1..=l).map(|i| net.fl(i, &st.mu[i - 1]).0).collect() } else { st.mu.iter().map(|v| vec![0.0; v.len()]).collect() };
    for i in 0..l { st.mu[i] = prior[i].clone(); }
    let mut e_first = f64::NAN; let mut e_prev = f64::NAN; let mut iters = 0.0;
    let mut fs: Vec<(Vec<f64>, Vec<f64>)>; let mut errs: Vec<Vec<f64>>; let mut el: Vec<f64>;
    let mut it = 0;
    loop {
        fs = (1..=l).map(|i| net.fl(i, &st.mu[i - 1])).collect();
        errs = (0..l).map(|i| { let below: &[f64] = if i == 0 { x } else { &st.mu[i - 1] }; let xh = net.wv(i + 1, &fs[i].0); below.iter().zip(&xh).map(|(a, b)| a - b).collect() }).collect();
        el = vec![0.0; l + 1];
        for i in 0..l { el[i] += 0.5 * errs[i].iter().zip(&net.pi[i]).map(|(e, p)| p * e * e).sum::<f64>(); }
        if started { for i in 1..=l { el[i] += 0.5 * st.mu[i - 1].iter().zip(&prior[i - 1]).zip(&net.pit[i - 1]).map(|((a, b), p)| p * (a - b) * (a - b)).sum::<f64>(); } }
        let e: f64 = el.iter().sum();
        if it == 0 { e_first = e; }
        if !e.is_finite() { break; }
        if it >= c.t_inf || (it > 0 && (e_prev - e).abs() / e.max(1e-300) < c.tol) { break; }
        e_prev = e;
        let eps: Vec<Vec<f64>> = (0..l).map(|i| errs[i].iter().zip(&net.pi[i]).map(|(e, p)| p * e).collect()).collect();
        for i in 1..=l {
            let back = net.jtl(i, &fs[i - 1].1, &net.wtv(i, &eps[i - 1]));
            let pc = if c.v3 { Some(&net.curv[i - 1]) } else { None };
            let mu = &mut st.mu[i - 1];
            for j in 0..mu.len() {
                let mut g = -back[j];
                if i < l { g += eps[i][j]; }
                if started { g += net.pit[i - 1][j] * (mu[j] - prior[i - 1][j]); }
                // v3: paso precondicionado por la curvatura diagonal local (Π propia + Π^τ + Σ_i Π_{l−1,i} W_ij²): estable para cualquier Π
                let k = match pc { Some(cv) => c.kappa / (cv[j] + if i < l { net.pi[i][j] } else { 0.0 } + if started { net.pit[i - 1][j] } else { 0.0 }).max(1.0), None => c.kappa };
                mu[j] -= k * g;
            }
            net.clipv(&mut st.mu[i - 1]);
        }
        it += 1; iters += 1.0;
    }
    let ef: f64 = el.iter().sum();
    if let Some(a) = acc {
        for i in 1..=l { let (r, cc) = (net.dims[i - 1], net.dims[i]); let f = &fs[i - 1].0; let pe: Vec<f64> = errs[i - 1].iter().zip(&net.pi[i - 1]).map(|(e, p)| p * e).collect();
            for rr in 0..r { for j in 0..cc { a.gw[i - 1][rr * cc + j] += pe[rr] * f[j]; } } }
        for i in 0..l { for j in 0..errs[i].len() { a.e2[i][j] += errs[i][j] * errs[i][j]; } }
        if started { for i in 1..=l { for j in 0..st.mu[i - 1].len() { let d = st.mu[i - 1][j] - prior[i - 1][j]; a.et2[i - 1][j] += d * d; } } a.nt += 1.0; }
        a.red += (e_first - ef).max(0.0); a.np += 1.0;
    }
    st.started = true;
    (e_first, ef, el, iters)
}

/// gradiente BPTT truncado (1 paso) de ½‖x_t − x̂_0(t | μ(t−1))‖² respecto de W (control «backprop»)
fn bp_grad(net: &Net, prev: &PState, x: &[f64], g: &mut [Vec<f64>]) {
    let l = net.nl(); let mut top = prev.mu[l - 1].clone(); top = net.fl(l, &top).0;
    // forward guardando
    let mut ins = vec![vec![]; l + 1]; let mut fss = vec![(vec![], vec![]); l + 1]; ins[l] = top;
    for ll in (1..=l).rev() { let fs = net.fl(ll, &ins[ll]); let below = net.wv(ll, &fs.0); fss[ll] = fs; ins[ll - 1] = below; }
    let mut d: Vec<f64> = (0..2).map(|i| ins[0][i] - x[i]).collect(); // dL/dx̂
    for ll in 1..=l { let (r, cc) = (net.dims[ll - 1], net.dims[ll]); let f = &fss[ll].0;
        for rr in 0..r { for j in 0..cc { g[ll - 1][rr * cc + j] -= d[rr] * f[j] * net.cm[ll - 1][rr * cc + j]; } }
        if ll < l { d = net.jtl(ll, &fss[ll].1, &net.wtv(ll, &d)); } }
}

/// EL paso de CONSTELACIÓN (Paso 0a: única función para train y eval).
/// train=false ⇒ sólo inferencia. lr_mult = α_s en sueño; prec_only = episodios − del sueño.
pub fn constelacion_step(net: &mut Net, sts: &mut [PState], x: &[f64], train: bool, lr_mult: f64, prec_only: bool) -> StepInfo { step_ex(net, sts, x, train, lr_mult, prec_only, false) }
/// imagine=true: cuadro imaginado (rollout reinyectado): sin campo rápido ni plasticidad.
pub fn step_ex(net: &mut Net, sts: &mut [PState], x: &[f64], train: bool, lr_mult: f64, prec_only: bool, imagine: bool) -> StepInfo {
    let np = x.len() / 2; let l = net.nl();
    // v3: expectativa temporal previa (antes de inferir x_t): φ(t−1), e^P = x_t − x̂(t)
    let pre: Vec<Option<Vec<f64>>> = if net.c.v3 { sts.iter().map(|s| if s.started { Some(net.feat(s)) } else { None }).collect() } else { vec![] };
    let eprior: Vec<Option<Vec<f64>>> = if net.c.v3 { (0..np).map(|p| pre[p].as_ref().map(|phi| { let xh = net.pred_from(&sts[p], phi); vec![x[2 * p] - xh[0], x[2 * p + 1] - xh[1]] })).collect() } else { vec![] };
    let mut acc = Acc { gw: net.w.iter().map(|w| vec![0.0; w.len()]).collect(), e2: net.pi.iter().map(|p| vec![0.0; p.len()]).collect(), et2: net.pit.iter().map(|p| vec![0.0; p.len()]).collect(), red: 0.0, np: 0.0, nt: 0.0 };
    let prevs: Option<Vec<PState>> = if train && net.c.backprop { Some(sts.to_vec()) } else { None };
    let (mut e0, mut ef, mut it) = (0.0, 0.0, 0.0); let mut elay = vec![0.0; l + 1];
    for p in 0..np {
        let r = infer_point(net, &mut sts[p], &x[2 * p..2 * p + 2], if train { Some(&mut acc) } else { None });
        e0 += r.0 / np as f64; ef += r.1 / np as f64; it += r.3 / np as f64; for i in 0..r.2.len() { elay[i] += r.2[i] / np as f64; }
    }
    if !ef.is_finite() || sts.iter().any(|s| s.mu.iter().any(|v| !fin(v))) { net.nan = true; }
    if train && !net.nan { plasticity(net, &acc, lr_mult, prec_only, prevs.as_deref(), x, sts); net.update_curv(); }
    if net.c.v3 && !net.nan {
        let n_obs = pre.iter().filter(|p| p.is_some()).count();
        if n_obs > 0 { head_update(net, sts, x, &pre, &eprior, train, lr_mult, prec_only, imagine); }
        for p in 0..np { sts[p].xl = x[2 * p..2 * p + 2].to_vec(); }
    }
    StepInfo { e0, ef, e_layer: elay, iters: it }
}

fn plasticity(net: &mut Net, a: &Acc, lr_mult: f64, prec_only: bool, prevs: Option<&[PState]>, x: &[f64], sts: &[PState]) {
    let c = net.c.clone(); let l = net.nl(); let np = a.np.max(1.0);
    // precisiones (regla local de máxima verosimilitud)
    if c.precision && c.errors {
        for i in 0..l { for j in 0..net.pi[i].len() { let p = &mut net.pi[i][j]; *p = (*p + c.eta_pi * 0.5 * (1.0 / *p - a.e2[i][j] / np)).clamp(c.pi_min, c.pi_max); } }
        if a.nt > 0.0 { for i in 0..l { for j in 0..net.pit[i].len() { let p = &mut net.pit[i][j]; *p = (*p + c.eta_pi * 0.5 * (1.0 / *p - a.et2[i][j] / a.nt)).clamp(c.pi_min, c.pi_max); } } }
        if c.pi_norm { for v in net.pi.iter_mut().chain(net.pit.iter_mut()) { let g = (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp(); for x in v.iter_mut() { *x /= g; } } }
    }
    if prec_only || !c.plastic_w { net.steps += 1; return; }
    let eta = c.eta_w * net.eta_scale * lr_mult; let red = a.red / np;
    let mut gbp: Option<Vec<Vec<f64>>> = None;
    if c.backprop { let mut g: Vec<Vec<f64>> = net.w.iter().map(|w| vec![0.0; w.len()]).collect(); if let Some(pv) = prevs { if pv[0].started { for p in 0..pv.len() { bp_grad(net, &pv[p], &x[2 * p..2 * p + 2], &mut g); } } } gbp = Some(g); }
    for li in 0..l {
        let only_out = !c.errors && li > 0; if only_out { continue; }
        for k in 0..net.w[li].len() {
            if net.cm[li][k] == 0.0 { continue; }
            let g = match &gbp { Some(gb) => gb[li][k] / np, None => a.gw[li][k] / np };
            let tau = if c.prune && !c.backprop { net.tau[li][k] } else { 1.0 };
            let dw = eta / tau * g - c.lam * net.w[li][k];
            net.w[li][k] += dw;
            net.act[li][k] = (1.0 - c.zeta) * net.act[li][k] + c.zeta * (a.gw[li][k] / np).abs();
            if c.prune { net.u[li][k] = (1.0 - c.zeta) * net.u[li][k] + c.zeta * dw.abs() * net.w[li][k].abs() * red; }
        }
        if c.prune {
            let act: Vec<usize> = (0..net.u[li].len()).filter(|&k| net.cm[li][k] != 0.0).collect();
            let ub = act.iter().map(|&k| net.u[li][k]).sum::<f64>() / act.len().max(1) as f64;
            if ub > 0.0 { for &k in &act { net.tau[li][k] = (net.tau[li][k] * (1.0 + c.rho_tau * (net.u[li][k] / ub - 1.0))).clamp(c.tau_min, c.tau_max); } }
        }
        // límite de entrada por fila
        let (r, cc) = (net.dims[li], net.dims[li + 1]);
        for rr in 0..r { let s: f64 = (0..cc).map(|j| (net.w[li][rr * cc + j] * net.cm[li][rr * cc + j]).abs()).sum(); if s > c.b_in { let f = c.b_in / s; for j in 0..cc { net.w[li][rr * cc + j] *= f; } net.n_rescale_row += 1; } }
    }
    net.steps += 1;
    if net.steps % c.p_every == 0 {
        // radio espectral σ_max(W_l J_l) en el estado medio, potencias
        for li in 1..=l {
            let mut mbar = vec![0.0; net.dims[li]]; for s in sts { for j in 0..mbar.len() { mbar[j] += s.mu[li - 1][j] / sts.len() as f64; } }
            let (_, sv) = net.fl(li, &mbar);
            let mut v = vec![1.0 / (net.dims[li] as f64).sqrt(); net.dims[li]]; let mut sg = 0.0;
            for _ in 0..30 { let a1 = net.wv(li, &net.jvl(li, &sv, &v)); let b1 = net.jtl(li, &sv, &net.wtv(li, &a1)); let nb = b1.iter().map(|z| z * z).sum::<f64>().sqrt(); if nb == 0.0 { break; } sg = nb.sqrt(); v = b1.iter().map(|z| z / nb).collect(); }
            if sg > c.rho_max { let f = c.rho_max / sg; for w in net.w[li - 1].iter_mut() { *w *= f; } net.n_rescale_spec += 1; }
        }
        if c.prune {
            for li in 0..l {
                let act: Vec<usize> = (0..net.u[li].len()).filter(|&k| net.cm[li][k] != 0.0).collect();
                let ub = act.iter().map(|&k| net.u[li][k]).sum::<f64>() / act.len().max(1) as f64;
                for &k in &act { if net.w[li][k].abs() < c.w_min && net.u[li][k] < c.u_min_rel * ub { net.low[li][k] += 1; if net.low[li][k] >= 2 { net.cm[li][k] = 0.0; net.low[li][k] = 0; net.n_pruned += 1; } } else { net.low[li][k] = 0; } }
                // crecimiento: el par inactivo con mayor |ε|·|F| promedio (act) se reactiva con W=0 (tope = tamaño inicial)
                if let Some(best) = (0..net.u[li].len()).filter(|&k| net.cm[li][k] == 0.0).max_by(|&x, &y| net.act[li][x].total_cmp(&net.act[li][y])) {
                    net.cm[li][best] = 1.0; net.w[li][best] = 0.0; net.tau[li][best] = c.tau_min; net.u[li][best] = 0.0; net.n_grown += 1; }
            }
        }
    }
}

// ─── episodios ───────────────────────────────────────────────────────────────
pub struct EpOut { pub h1_log: f64, pub h16_err: f64, pub e_mean: f64, pub e_layer: Vec<f64>, pub preds: Vec<Vec<f64>> }
/// train: los 17 cuadros con plasticidad. eval: observa 0..=k, sin plasticidad.
/// h16 = predicción de y16 hecha tras observar 0..=K (rollout descendente h=16−K). preds[h−1] (h∈1..16): h≤K una-paso filtrada; h>K rollout.
pub fn run_episode(net: &mut Net, e: &Ep, train: bool, lr_mult: f64, prec_only: bool, k_obs: usize) -> EpOut {
    let np = e.x0.len() / 2; net.begin_episode(); let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect();
    let last = if train { HORIZON } else { k_obs };
    let mut h1 = vec![]; let mut es = 0.0; let mut el = vec![0.0; net.nl() + 1]; let mut h16 = 10.0; let mut preds = vec![vec![]; HORIZON];
    for t in 0..=last {
        if t >= 1 { let p: Vec<f64> = sts.iter().flat_map(|s| net.predict1(s)).collect(); h1.push(log_rel(clamp_err(&p, e.frame(t)))); if t <= k_obs { preds[t - 1] = p; } }
        let si = constelacion_step(net, &mut sts, e.frame(t), train, lr_mult, prec_only);
        es += si.ef; for i in 0..el.len() { el[i] += si.e_layer[i]; }
        if t == k_obs {
            if net.c.v3 { // rollout reinyectado (spec §4): x̂(t+1) se observa como si fuera real, sin campo rápido ni plasticidad
                let mut s2 = sts.clone(); let wf = net.wf.clone(); let nan0 = net.nan;
                for hh in (k_obs + 1)..=HORIZON { let p: Vec<f64> = s2.iter().flat_map(|s| net.predict1(s)).collect(); preds[hh - 1] = p.clone(); if !fin(&p) { break; } step_ex(net, &mut s2, &p, false, 0.0, false, true); }
                debug_assert!(wf == net.wf); net.nan = nan0;
                if !fin(&preds[HORIZON - 1]) || preds[HORIZON - 1].is_empty() { preds[HORIZON - 1] = vec![f64::NAN; 2 * np]; }
            } else { for hh in (k_obs + 1)..=HORIZON { preds[hh - 1] = sts.iter().flat_map(|s| net.predict(s, hh - k_obs)).collect(); } }
            h16 = clamp_err(&preds[HORIZON - 1], &e.ys[HORIZON - 1]); }
        if net.nan { break; }
    }
    let n = (last + 1) as f64;
    EpOut { h1_log: mean(&h1), h16_err: h16, e_mean: es / n, e_layer: el.iter().map(|v| v / n).collect(), preds }
}

// ─── CDT + sueño ─────────────────────────────────────────────────────────────
pub struct Cdt { pub eps: Vec<(Ep, f64, bool)>, pub ema: Option<f64>, pub kbuf: Vec<Ep>, pub kprev: Option<f64>, pub kbad: usize, pub log: Vec<serde_json::Value>, pub n_sleeps: usize }
impl Cdt {
    pub fn new(seed: u64, c: &Cfg) -> Cdt { Cdt { eps: vec![], ema: None, kbuf: gen_known_stream(seed, XOR_KBUF, c.n_kbuf, &(0..N_FAM).collect::<Vec<_>>()), kprev: None, kbad: 0, log: vec![], n_sleeps: 0 } }
    pub fn push(&mut self, e: &Ep, o: &EpOut) {
        let v = o.h1_log; let sign = match self.ema { Some(m) => v < m, None => false };
        self.ema = Some(match self.ema { Some(m) => 0.9 * m + 0.1 * v, None => v });
        self.eps.push((e.clone(), o.e_mean, sign));
    }
}
fn kbuf_energy(net: &mut Net, kb: &[Ep]) -> f64 { mean(&kb.iter().map(|e| run_episode(net, e, false, 0.0, false, HORIZON).e_mean).collect::<Vec<_>>()) }
pub fn sleep(net: &mut Net, cdt: &mut Cdt, seed: u64) {
    let c = net.c.clone(); cdt.n_sleeps += 1;
    let mut es: Vec<f64> = cdt.eps.iter().map(|x| x.1).filter(|v| v.is_finite()).collect(); es.sort_by(|a, b| a.total_cmp(b));
    if es.is_empty() { return; }
    let p95 = es[((es.len() as f64 * 0.95) as usize).min(es.len() - 1)];
    let cand: Vec<usize> = (0..cdt.eps.len()).filter(|&i| cdt.eps[i].1.is_finite() && cdt.eps[i].1 <= p95).collect();
    let tot: f64 = cand.iter().map(|&i| cdt.eps[i].1.max(1e-12)).sum();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_SLEEP ^ (cdt.n_sleeps as u64) << 20);
    let (mut npos, mut nneg) = (0, 0);
    for _ in 0..c.n_replay { let mut r = rng.gen::<f64>() * tot; let mut pick = cand[cand.len() - 1];
        for &i in &cand { r -= cdt.eps[i].1.max(1e-12); if r <= 0.0 { pick = i; break; } }
        let (e, _, s) = cdt.eps[pick].clone();
        if s { run_episode(net, &e, true, c.alpha_s, false, K); npos += 1; } else { run_episode(net, &e, true, 1.0, true, K); nneg += 1; } }
    let ek = kbuf_energy(net, &cdt.kbuf.clone());
    if let Some(p) = cdt.kprev { if ek > p * (1.0 + c.delta_k) { cdt.kbad += 1; } else { cdt.kbad = 0; } }
    let mut halved = false; if cdt.kbad >= 3 { net.eta_scale *= 0.5; cdt.kbad = 0; halved = true; }
    cdt.kprev = Some(ek);
    cdt.log.push(json!({"n": cdt.n_sleeps, "pos": npos, "neg": nneg, "kbuf_E": ek, "halved": halved, "eta_scale": net.eta_scale}));
}

/// stream con plasticidad + sueño (precuencial: la predicción h16 se registra antes de ver 5..16)
pub fn run_stream(net: &mut Net, cdt: &mut Cdt, stream: &[Ep], seed: u64) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let (mut h1, mut h16, mut fam) = (vec![], vec![], vec![]);
    for e in stream {
        let o = run_episode(net, e, true, 1.0, false, K);
        h1.push(o.h1_log); h16.push(log_rel(o.h16_err)); fam.push(e.fam);
        cdt.push(e, &o);
        if net.c.sleep && cdt.eps.len() % net.c.n_s == 0 { sleep(net, cdt, seed); }
        if net.nan { break; }
    }
    (h1, h16, fam)
}

/// evaluación H (k=4, sin plasticidad): acc16 y mediana log h16 por grupo
pub fn eval_held(net: &mut Net, held: &[Ep]) -> serde_json::Value {
    let mut r = serde_json::Map::new();
    for (name, known) in [("known", true), ("new", false)] {
        let errs: Vec<f64> = held.iter().filter(|e| (e.fam < N_FAM) == known).map(|e| run_episode(net, e, false, 0.0, false, K).h16_err).collect();
        if errs.is_empty() { continue; }
        r.insert(format!("{name}_acc16"), json!(errs.iter().filter(|&&x| x < mt::REL_ERR_OK).count() as f64 / errs.len() as f64));
        r.insert(format!("{name}_med_log"), json!(median(errs.iter().map(|&x| log_rel(x)).collect())));
    }
    serde_json::Value::Object(r)
}
/// ÓRBITA-v1 congelada sola (= brazo `orbita` de ESTELA k=4): conocidas ctx verdadero, nuevas reid(k=4); rollout 16 desde x0
pub fn orbita_held(d: &Orbita, held: &[Ep]) -> serde_json::Value {
    let mut r = serde_json::Map::new();
    for (name, known) in [("known", true), ("new", false)] {
        let errs: Vec<f64> = held.iter().filter(|e| (e.fam < N_FAM) == known).map(|e| {
            let c = if e.fam < N_FAM { context_of(e.fam, e.p) } else { let (g, pp, _) = reid(d, &e.x0, &e.ys, K, e.p); context_of(g, pp) };
            let mut z = e.x0.clone(); for _ in 0..HORIZON { z = d.step(&z, &c); } clamp_err(&z, &e.ys[HORIZON - 1]) }).collect();
        if errs.is_empty() { continue; }
        r.insert(format!("{name}_acc16"), json!(errs.iter().filter(|&&x| x < mt::REL_ERR_OK).count() as f64 / errs.len() as f64));
        r.insert(format!("{name}_med_log"), json!(median(errs.iter().map(|&x| log_rel(x)).collect())));
    }
    serde_json::Value::Object(r)
}
/// colapso: varianza de μ por capa en H conocidas, y fracción de Π en límites
pub fn collapse(net: &mut Net, held: &[Ep]) -> serde_json::Value {
    let l = net.nl(); let mut vs = vec![vec![]; l];
    for e in held.iter().filter(|e| e.fam < N_FAM).take(32) { net.begin_episode(); let np = e.x0.len() / 2; let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect();
        for t in 0..=K { constelacion_step(net, &mut sts, e.frame(t), false, 0.0, false); } for s in &sts { for i in 0..l { vs[i].push(s.mu[i].clone()); } } }
    let var: Vec<f64> = vs.iter().map(|v| { let d = v[0].len(); let n = v.len() as f64; (0..d).map(|j| { let m = v.iter().map(|x| x[j]).sum::<f64>() / n; v.iter().map(|x| (x[j] - m).powi(2)).sum::<f64>() / n }).sum::<f64>() / d as f64 }).collect();
    let c = &net.c; let at = |ps: &Vec<Vec<f64>>| ps.iter().map(|p| p.iter().filter(|&&x| x <= c.pi_min * 1.0001 || x >= c.pi_max * 0.9999).count() as f64 / p.len() as f64).collect::<Vec<_>>();
    let (pa, pta) = (at(&net.pi), at(&net.pit));
    let collapsed = var.iter().any(|&v| v < 1e-10) || pa.iter().chain(&pta).any(|&f| f > 0.9);
    json!({"mu_var": var, "pi_at_bound": pa, "pit_at_bound": pta, "collapsed": collapsed})
}
/// E33 (§16): rotate+reflect del TEST v4 iterable (lectura como `orbita eval`/ECO; contador de consultas = 0). k=4 observados.
fn e33_examples(seed: u64) -> (Vec<Example>, String) {
    let s = generate_and_seal(seed, &DatasetConfig::standard(K_POINTS)); s.verify_test().expect("TEST sealed");
    (s.ds.test.iter().filter(|e| ITERABLE_FAMS.contains(&e.family_id)).cloned().collect(), s.manifest.test_sha256.clone())
}
fn e33_score(trs: &[(Vec<Vec<f64>>, &Example)]) -> serde_json::Value {
    let mut accs = vec![0.0; EVAL_H.len()];
    for (tr, e) in trs { for (j, &h) in EVAL_H.iter().enumerate() { let y = target_h(e, h); accs[j] += (fin(&tr[h - 1]) && mt::correct(&tr[h - 1], &y)) as u8 as f64; } }
    let n = trs.len() as f64; let accs: Vec<f64> = accs.iter().map(|a| a / n).collect();
    let mut region = 0; for (j, &h) in EVAL_H.iter().enumerate() { if accs[j] >= 0.5 { region = h; } else { break; } }
    json!({"acc": accs, "region": region, "pass": region >= 8, "n": trs.len()})
}
/// E33 de la red. Devuelve (con cabeza ÓRBITA-1-paso si orb_prior, sin ella = v3). La cabeza sólo sustituye x̂(1): la predicción hecha
/// tras observar x0 y antes de cualquier transición (W^F = 0), usando el contexto del ejemplo, igual que ÓRBITA. No toca dinámica ni plasticidad.
pub fn e33_net2(net: &mut Net, d: &Orbita, ex: &[Example]) -> (serde_json::Value, serde_json::Value) {
    let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex.iter().map(|e| { let ys: Vec<Vec<f64>> = (1..=HORIZON).map(|h| target_h(e, h)).collect();
        let ep = Ep { fam: e.family_id, p: [0.0; 2], x0: e.input.clone(), ys }; (run_episode(net, &ep, false, 0.0, false, K).preds, e) }).collect();
    let raw = e33_score(&trs);
    if !net.c.orb_prior { return (raw.clone(), raw); }
    let trs2: Vec<(Vec<Vec<f64>>, &Example)> = trs.iter().map(|(p, e)| { let mut p = p.clone(); p[0] = d.step(&e.input, &e.context); (p, *e) }).collect();
    (e33_score(&trs2), raw)
}
pub fn e33_net(net: &mut Net, ex: &[Example]) -> serde_json::Value {
    let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex.iter().map(|e| { let ys: Vec<Vec<f64>> = (1..=HORIZON).map(|h| target_h(e, h)).collect();
        let ep = Ep { fam: e.family_id, p: [0.0; 2], x0: e.input.clone(), ys }; (run_episode(net, &ep, false, 0.0, false, K).preds, e) }).collect();
    e33_score(&trs)
}
pub fn e33_orbita(d: &Orbita, ex: &[Example]) -> serde_json::Value {
    let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex.iter().map(|e| { let mut z = e.input.clone(); ((0..HORIZON).map(|_| { z = d.step(&z, &e.context); z.clone() }).collect(), e) }).collect();
    e33_score(&trs)
}

fn arg_of(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned() }
pub fn parse_seed(s: &str) -> u64 { u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap() }
fn slope(v: &[f64]) -> f64 { let n = v.len() as f64; let mx = (n - 1.0) / 2.0; let my = mean(v); let (mut a, mut b) = (0.0, 0.0); for (i, y) in v.iter().enumerate() { a += (i as f64 - mx) * (y - my); b += (i as f64 - mx).powi(2); } a / b }
fn git_sha() -> String { std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default() }

// ─── constantes de protocolo ─────────────────────────────────────────────────
pub const V4_T_F: usize = 128; // v4: 128 iteraciones NLMS por frame (v3: 32)
pub const N_A: usize = 3072; // v4: flujo A ×4 (v3: 768)
pub const N_NEW: usize = 192; pub const N_KNOWN_B: usize = 192; // v4: flujo B ×4 (v3: 48+48)
pub const HPF: usize = 16;

/// F0: una ÓRBITA-neurona (L=1, n=1) en la familia 0, η vs η=0 vs trivial
pub fn f0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); assert!((0xC400..=0xC4FF).contains(&seed));
    let n: usize = arg_of(a, "--n").map(|s| s.parse().unwrap()).unwrap_or(256);
    let mut c = cfg_from_args(a); c.layers = vec![1]; c.sleep = false;
    let stream = gen_known_stream(seed, XOR_KNOWN, n, &[0]);
    let mut out = serde_json::Map::new();
    for name in ["eta", "eta0_slow", "eta0"] {
        let mut cc = c.clone(); if name != "eta" { cc.eta_w = 0.0; cc.eta_p = 0.0; } if name == "eta0" { cc.eta_f = 0.0; } let mut net = Net::new(&cc, seed); let mut cdt = Cdt::new(seed, &cc);
        let (h1, h16, _) = run_stream(&mut net, &mut cdt, &stream, seed);
        let q = n / 4;
        out.insert(name.into(), json!({"h1_curve": h1, "h16_curve": h16, "slope_h1": slope(&h1), "slope_h16": slope(&h16), "last_q_h1": mean(&h1[n - q..]), "last_q_h16": mean(&h16[n - q..]), "nan": net.nan}));
    }
    // trivial: x̂(t+1) = x(t); h16: x̂(16) = x(K)
    let th1: Vec<f64> = stream.iter().map(|e| mean(&(1..=HORIZON).map(|t| log_rel(clamp_err(e.frame(t - 1), e.frame(t)))).collect::<Vec<_>>())).collect();
    let th16: Vec<f64> = stream.iter().map(|e| log_rel(clamp_err(e.frame(K), &e.ys[HORIZON - 1]))).collect();
    let q = n / 4; out.insert("trivial".into(), json!({"last_q_h1": mean(&th1[n - q..]), "last_q_h16": mean(&th16[n - q..])}));
    println!("{}", json!({"seed": format!("0x{seed:X}"), "n": n, "cfg": c.to_json(), "f0": out}));
}

/// F1: CONSTELACIÓN (L=3) en conocidas 0–7 (flujo A) vs ÓRBITA-v1 congelada en H conocidas
pub fn f1(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); assert!((0xC400..=0xC4FF).contains(&seed));
    let d = Orbita::from_json(&std::fs::read_to_string(arg_of(a, "--weights").unwrap()).unwrap());
    let c = cfg_from_args(a);
    let t0 = std::time::Instant::now();
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
    let held: Vec<Ep> = gen_heldout(seed, HPF).into_iter().filter(|e| e.fam < N_FAM).collect();
    let pre = eval_held(&mut net, &held);
    let stream = gen_known_stream(seed, XOR_KNOWN, N_A, &(0..N_FAM).collect::<Vec<_>>());
    let (h1, h16, _) = run_stream(&mut net, &mut cdt, &stream, seed);
    let post = eval_held(&mut net, &held); let orb = orbita_held(&d, &held); let col = collapse(&mut net, &held);
    let (ex, sha) = e33_examples(seed); let (e33c, e33c_v3) = e33_net2(&mut net, &d, &ex); let e33o = e33_orbita(&d, &ex);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(); assert_eq!(q, 0);
    let gap = post["known_med_log"].as_f64().unwrap() - orb["known_med_log"].as_f64().unwrap();
    println!("{}", json!({"seed": format!("0x{seed:X}"), "cfg": c.to_json(), "pre": pre, "post": post, "orbita": orb, "gap_log": gap, "stop_gap": gap > 1.0, "collapse": col,
        "e33": {"cst": e33c, "cst_v3head": e33c_v3, "orbita": e33o}, "test_sha256": sha, "test_queries": q, "h1_curve": h1, "h16_curve": h16, "slope_h16": slope(&h16), "nan": net.nan,
        "prune": {"pruned": net.n_pruned, "grown": net.n_grown, "row_rescale": net.n_rescale_row, "spec_rescale": net.n_rescale_spec}, "sleeps": cdt.log, "wall_s": t0.elapsed().as_secs_f64()}));
}
pub const V3_PI_MIN: f64 = 0.1; pub const V3_PI_MAX: f64 = 10.0;
pub const V3_FEAT_X: bool = true; pub const V3_FEAT_S: f64 = 0.3; pub const V3_ETA_P: f64 = 0.001; pub const V3_ETA_F: f64 = 1.0; pub const V3_T_F: usize = 32;
const TUNE_FLAGS: [&str; 12] = ["--pi-max", "--mu-clip", "--eta-w", "--kappa", "--eta-pi", "--w-init", "--eta-p", "--eta-f", "--t-f", "--feat-x", "--feat-s", "--b-in"];
fn cfg_from_args(a: &[String]) -> Cfg {
    let mut c = if a.iter().any(|x| x == "--v2") { Cfg::base() } else if a.iter().any(|x| x == "--v3") { Cfg::v3() } else { Cfg::v4() };
    if let Some(v) = arg_of(a, "--eta-p") { c.eta_p = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--eta-f") { c.eta_f = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--t-f") { c.t_f = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--feat-x") { c.feat_x = v == "1"; }
    if let Some(v) = arg_of(a, "--feat-s") { c.feat_s = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--b-in") { c.b_in = v.parse().unwrap(); }
    if a.iter().any(|x| x == "--no-pinorm") { c.pi_norm = false; }
    if let Some(v) = arg_of(a, "--pi-max") { c.pi_max = v.parse().unwrap(); c.pi_min = 1.0 / c.pi_max; }
    if let Some(v) = arg_of(a, "--mu-clip") { c.mu_clip = v.parse().unwrap(); }
    if a.iter().any(|x| x == "--no-fast") { c.fast = false; }
    if let Some(v) = arg_of(a, "--eta-w") { c.eta_w = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--kappa") { c.kappa = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--eta-pi") { c.eta_pi = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--w-init") { c.w_init = v.parse().unwrap(); }
    c
}

/// F2: por semilla, todos los brazos en el mismo binario
pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let phase = arg_of(a, "--phase").unwrap();
    match phase.as_str() { "dev" => assert!((0xA400..=0xA40F).contains(&seed)), "confirm" => assert!((0xB400..=0xB40F).contains(&seed)),
        _ => assert!(phase == "smoke" && (0xC400..=0xD4FF).contains(&seed)) }
    if phase != "smoke" { assert!(TUNE_FLAGS.iter().all(|f| arg_of(a, f).is_none()) && !a.iter().any(|x| x.starts_with("--no-") || x == "--v2" || x == "--v3"), "DEV/CONFIRM: valores congelados"); }
    let wpath = arg_of(a, "--weights").unwrap(); let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap()); let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip"); return; }
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let base = cfg_from_args(a);
    let held = gen_heldout(seed, HPF); let held_known: Vec<Ep> = held.iter().filter(|e| e.fam < N_FAM).cloned().collect();
    let na: usize = if phase == "smoke" { arg_of(a, "--na").map(|v| v.parse().unwrap()).unwrap_or(N_A) } else { N_A };
    let nb: usize = if phase == "smoke" { arg_of(a, "--nb").map(|v| v.parse().unwrap()).unwrap_or(N_NEW) } else { N_NEW };
    let sa = gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, nb, nb); assert!(N_NEW == N_KNOWN_B);
    let (ex, sha) = e33_examples(seed);
    let mut arms = serde_json::Map::new();
    let arms_list: Vec<Arm> = if base.v3 { Arm::all_v3().to_vec() } else { Arm::all().to_vec() };
    for arm in arms_list {
        if let Some(o) = &only { if !o.iter().any(|x| x == arm.name()) { continue; } }
        let ta = std::time::Instant::now(); let c = arm.cfg(&base);
        let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
        let (_, h16a, _) = run_stream(&mut net, &mut cdt, &sa, seed);
        let pre = eval_held(&mut net, &held_known);
        let (_, h16b, fb) = run_stream(&mut net, &mut cdt, &sb, seed);
        let post = eval_held(&mut net, &held); let col = collapse(&mut net, &held); let (e33, e33_v3head) = e33_net2(&mut net, &d, &ex);
        let auc_new = mean(&h16b.iter().zip(&fb).filter(|(_, &f)| f >= N_FAM).map(|(v, _)| *v).collect::<Vec<_>>());
        arms.insert(arm.name().into(), json!({"pre_known": pre, "held": post, "collapse": col, "e33": e33, "e33_v3head": e33_v3head, "auc_new_preq": auc_new, "streamA_last_q_h16": mean(&h16a[h16a.len() * 3 / 4..]),
            "nan": net.nan, "wp_rms": rms(&net.wp), "pip": net.pip.clone(), "streamB_new_preq_last": h16b.iter().zip(&fb).filter(|(_, &f)| f >= N_FAM).map(|(v, _)| *v).collect::<Vec<_>>().iter().rev().take(16).sum::<f64>() / 16.0, "hash": format!("{:016x}", net.hash()), "eta_scale": net.eta_scale, "sleeps": cdt.log.len(), "prune": [net.n_pruned, net.n_grown, net.n_rescale_row, net.n_rescale_spec], "wall_s": ta.elapsed().as_secs_f64()}));
        eprintln!("0x{seed:X} {} {:.1}s", arm.name(), ta.elapsed().as_secs_f64());
    }
    let orb = orbita_held(&d, &held); let e33o = e33_orbita(&d, &ex);
    let e33p = { let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex.iter().map(|e| (vec![e.input.clone(); HORIZON], e)).collect(); e33_score(&trs) };
    let triv = { let mut r = serde_json::Map::new(); for (g, k) in [("known", true), ("new", false)] { r.insert(format!("{g}_med_log"), json!(median(held.iter().filter(|e| (e.fam < N_FAM) == k).map(|e| log_rel(clamp_err(e.frame(K), &e.ys[HORIZON - 1]))).collect()))); } serde_json::Value::Object(r) };
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(d.raw, raw0); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let r = json!({"seed": format!("0x{seed:X}"), "phase": phase, "weights": wpath, "orbita": orb, "trivial": triv, "e33_orbita": e33o, "e33_persist": e33p, "arms": arms, "test_queries": q, "test_sha256": sha, "git_sha": git_sha(), "cfg": base.to_json(), "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out.join(format!("metrics_0x{seed:X}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok {:.0}s", t0.elapsed().as_secs_f64());
}

// ─── Paso 0 ──────────────────────────────────────────────────────────────────
fn trace(net: &mut Net, eps: &[Ep], train: bool) -> Vec<u64> {
    let mut v = vec![]; for e in eps { let o = run_episode(net, e, train, 1.0, false, HORIZON); for p in o.preds.iter().flatten() { v.push(p.to_bits()); } v.push(o.e_mean.to_bits()); v.push(o.h1_log.to_bits()); } v
}
pub fn paso0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC400".into())); assert!((0xC400..=0xC4FF).contains(&seed));
    let base = cfg_from_args(a); let mut res = serde_json::Map::new();
    let eps = gen_known_stream(seed, XOR_KNOWN, 16, &(0..N_FAM).collect::<Vec<_>>());
    // 0f: v3=false ≡ v2 bit a bit (mismo flujo que 0b de v2, en proceso, contra crate::constelacion)
    { let b2 = Cfg::base(); let s48 = gen_known_stream(seed, XOR_KNOWN, 48, &(0..N_FAM).collect::<Vec<_>>());
      let mut n3 = Net::new(&b2, seed); let mut c3 = Cdt::new(seed, &b2); let (a1, a16, _) = run_stream(&mut n3, &mut c3, &s48, seed);
      let b2o = crate::constelacion::Cfg::base(); let mut n2 = crate::constelacion::Net::new(&b2o, seed); let mut c2 = crate::constelacion::Cdt::new(seed, &b2o);
      let s48o = crate::constelacion::gen_known_stream(seed, XOR_KNOWN, 48, &(0..N_FAM).collect::<Vec<_>>()); let (b1, b16, _) = crate::constelacion::run_stream(&mut n2, &mut c2, &s48o, seed);
      let same = n3.hash() == n2.hash() && a1.iter().chain(&a16).zip(b1.iter().chain(&b16)).all(|(x, y)| x.to_bits() == y.to_bits());
      let mut hm: u64 = 0xcbf29ce484222325; for v in a1.iter().chain(&a16) { hm ^= v.to_bits(); hm = hm.wrapping_mul(0x100000001b3); }
      res.insert("0f_v3off_eq_v2".into(), json!({"pass": same, "weights_hash_v3off": format!("{:016x}", n3.hash()), "weights_hash_v2": format!("{:016x}", n2.hash()), "metrics_hash_v3off": format!("{hm:016x}"), "v2_paso0_ref": {"weights": "8e14c5880d854d46", "metrics": "a139b204454bcfbc"}})); }
    // 0g: (análogo de «γ=0 ≡ base») con W^P = W^F = 0 y η_P = η_F = 0 la predicción es exactamente la persistencia x̂(t+1)=x(t), filtrada y en rollout
    { let mut c = base.clone(); c.eta_p = 0.0; c.eta_f = 0.0; let mut n = Net::new(&c, seed); let mut bad = 0usize; let mut nv = 0usize;
      for e in &eps { let o = run_episode(&mut n, e, false, 0.0, false, K); for h in 1..=HORIZON { let want = if h <= K { e.frame(h - 1) } else { e.frame(K) }; for (x, y) in o.preds[h - 1].iter().zip(want) { nv += 1; if x.to_bits() != y.to_bits() { bad += 1; } } } }
      res.insert("0g_zero_head_eq_persistence".into(), json!({"values": nv, "mismatch": bad, "pass": bad == 0})); }
    // 0a: train con plasticidad off (tasas 0, sin poda/sueño) ≡ eval, por brazo
    let mut bad = 0usize; let mut nv = 0usize;
    let arms_list: Vec<Arm> = if base.v3 { Arm::all_v3().to_vec() } else { Arm::all().to_vec() };
    for arm in arms_list { let mut c = arm.cfg(&base); c.eta_w = 0.0; c.eta_p = 0.0; c.eta_pi = 0.0; c.lam = 0.0; c.zeta = 0.0; c.rho_tau = 0.0; c.prune = false; c.p_every = usize::MAX;
        let mut n1 = Net::new(&c, seed); let mut n2 = Net::new(&c, seed); let t1 = trace(&mut n1, &eps, true); let t2 = trace(&mut n2, &eps, false);
        nv += t1.len(); bad += t1.iter().zip(&t2).filter(|(x, y)| x != y).count() + t1.len().abs_diff(t2.len()); }
    res.insert("0a".into(), json!({"values": nv, "mismatch": bad, "pass": bad == 0}));
    // 0b: hash de pesos tras un flujo corto (comparado entre 2 procesos por el script)
    let mut net = Net::new(&base, seed); let mut cdt = Cdt::new(seed, &base); let (h1, h16, _) = run_stream(&mut net, &mut cdt, &gen_known_stream(seed, XOR_KNOWN, 48, &(0..N_FAM).collect::<Vec<_>>()), seed);
    let mut hm: u64 = 0xcbf29ce484222325; for v in h1.iter().chain(&h16) { hm ^= v.to_bits(); hm = hm.wrapping_mul(0x100000001b3); }
    let mut hd: u64 = 0xcbf29ce484222325; for e in gen_heldout(seed, HPF).iter().chain(&gen_stream(seed, N_NEW, N_KNOWN_B)) { for v in e.x0.iter().chain(e.ys.iter().flatten()) { hd ^= v.to_bits(); hd = hd.wrapping_mul(0x100000001b3); } }
    res.insert("0b".into(), json!({"weights_hash": format!("{:016x}", net.hash()), "metrics_hash": format!("{hm:016x}"), "data_hash": format!("{hd:016x}"), "sleeps": cdt.log.len()}));
    // 0c: estabilidad de inicialización
    let n0 = Net::new(&base, seed); let mut maxk: f64 = 0.0; let mut maxj: f64 = 0.0; let mut nn = 0;
    for ly in &n0.neu { for ne in ly { maxk = maxk.max(ne.spec_k()); maxj = maxj.max(ne.jpsi_bound()); nn += 1; } }
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed); let mut maxnorm: f64 = 0.0; let mut finite = true;
    for l in 1..=n0.nl() { let mut mu: Vec<f64> = (0..n0.dims[l]).map(|_| rng.gen_range(-1.0..1.0)).collect(); for _ in 0..1000 { mu = n0.fl(l, &mu).0; finite &= fin(&mu); maxnorm = maxnorm.max(mu.iter().map(|x| x * x).sum::<f64>().sqrt()); } }
    let mut nf = Net::new(&base, seed); let mut sts: Vec<PState> = (0..K_POINTS).map(|_| nf.fresh_state()).collect(); let e = &eps[0];
    let mut maxe: f64 = 0.0; for t in 0..1000 { let si = constelacion_step(&mut nf, &mut sts, e.frame(t % (HORIZON + 1)), false, 0.0, false); maxe = maxe.max(si.ef); finite &= si.ef.is_finite(); }
    let pass_c = maxk < 1.0 && finite && !nf.nan && maxe < 1e6;
    res.insert("0c".into(), json!({"neurons": nn, "max_rho_K": maxk, "max_jpsi_bound": maxj, "free_run_1000_max_norm": maxnorm, "inference_1000_max_E": maxe, "finite": finite, "pass": pass_c}));
    // 0d: ΔW de PC (T grande, Π_0 pequeño, Π^τ arriba grande) vs −∂L/∂W por diferencias finitas, red diminuta L=2 n=(2,1)
    let mut c = Cfg::base(); c.layers = vec![2, 1]; c.t_inf = 20000; c.tol = 0.0; c.kappa = 0.02; c.w_init = 1.0; c.prune = false; c.sleep = false;
    let mut corr = vec![];
    for trial in 0..5u64 {
        let mut tn = Net::new(&c, seed ^ (trial << 8)); tn.pi[0] = vec![0.01; 2]; tn.pi[1] = vec![1.0; tn.dims[1]]; tn.pit[0] = vec![1e-9; tn.dims[1]]; tn.pit[1] = vec![50.0; tn.dims[2]];
        let mut r2 = Xoshiro256StarStar::seed_from_u64(seed ^ trial); let mu2: Vec<f64> = (0..tn.dims[2]).map(|_| r2.gen_range(-1.0..1.0)).collect(); let x = [r2.gen_range(-1.0..1.0), r2.gen_range(-1.0..1.0)];
        let mut st = tn.fresh_state(); st.mu[1] = mu2.clone(); st.mu[0] = vec![0.0; tn.dims[1]]; st.started = true;
        // pérdida: L(W) = ½‖x − W1 F(W2 F(F(μ2)))‖² (predicción hacia adelante desde el prior)
        let loss = |n: &Net| { let p = n.predict(&PState { mu: vec![vec![0.0; n.dims[1]], mu2.clone()], started: true, xl: vec![0.0; 2] }, 1); 0.5 * ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2)) };
        // nota: el prior de capa 1 viene de μ1(t−1)=0 con Π^τ_1≈0 ⇒ libre; capa 2 anclada en F(μ2)
        let mut acc = Acc { gw: tn.w.iter().map(|w| vec![0.0; w.len()]).collect(), e2: tn.pi.iter().map(|p| vec![0.0; p.len()]).collect(), et2: tn.pit.iter().map(|p| vec![0.0; p.len()]).collect(), red: 0.0, np: 0.0, nt: 0.0 };
        infer_point(&tn, &mut st, &x, Some(&mut acc));
        let pc: Vec<f64> = acc.gw.iter().flatten().cloned().collect();
        let mut fd = vec![]; for li in 0..tn.w.len() { for k in 0..tn.w[li].len() { let mut a1 = tn.clone(); a1.w[li][k] += 1e-6; let mut b1 = tn.clone(); b1.w[li][k] -= 1e-6; fd.push(-(loss(&a1) - loss(&b1)) / 2e-6); } }
        let (ma, mb) = (mean(&pc), mean(&fd)); let cov: f64 = pc.iter().zip(&fd).map(|(a, b)| (a - ma) * (b - mb)).sum(); let va: f64 = pc.iter().map(|a| (a - ma).powi(2)).sum(); let vb: f64 = fd.iter().map(|b| (b - mb).powi(2)).sum();
        corr.push(cov / (va * vb).sqrt());
    }
    res.insert("0d".into(), json!({"corr": corr, "mean_corr": mean(&corr), "pass": mean(&corr) > 0.9}));
    // 0e: costo μs por punto-paso (1 hilo; el binario no usa rayon aquí)
    let mut nt = Net::new(&base, seed); let mut sts: Vec<PState> = (0..K_POINTS).map(|_| nt.fresh_state()).collect(); let t0 = std::time::Instant::now(); let reps = 2000;
    for t in 0..reps { constelacion_step(&mut nt, &mut sts, eps[1].frame(t % (HORIZON + 1)), true, 1.0, false); }
    let us_train = t0.elapsed().as_secs_f64() * 1e6 / (reps * K_POINTS) as f64;
    let t1 = std::time::Instant::now(); for t in 0..reps { constelacion_step(&mut nt, &mut sts, eps[1].frame(t % (HORIZON + 1)), false, 1.0, false); }
    let us_eval = t1.elapsed().as_secs_f64() * 1e6 / (reps * K_POINTS) as f64;
    res.insert("0e".into(), json!({"us_per_point_step_train": us_train, "us_per_point_step_eval": us_eval, "n_w": nt.n_w(), "neuron_fixed_params": 4 + base.m * base.h + base.h}));
    let ok = res["0a"]["pass"] == true && res["0c"]["pass"] == true && (!base.v3 || (res["0f_v3off_eq_v2"]["pass"] == true && res["0g_zero_head_eq_persistence"]["pass"] == true));
    println!("{}", json!({"seed": format!("0x{seed:X}"), "git_sha": git_sha(), "paso0_ok_0a_0c": ok, "res": res}));
}

// ─── DIAGNÓSTICO (v3; sólo semillas 0xC4xx/0xD4xx) ──────────────────────────
fn rms(v: &[f64]) -> f64 { (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt() }
/// ridge: minimiza ‖Φw − y‖² + λ‖w‖², por Cholesky (d pequeño). Φ: n×d. devuelve d×k
fn ridge(phi: &[Vec<f64>], y: &[Vec<f64>], lam: f64) -> Vec<Vec<f64>> {
    let d = phi[0].len(); let k = y[0].len();
    let mut a = vec![0.0; d * d]; let mut b = vec![0.0; d * k];
    for (p, t) in phi.iter().zip(y) { for i in 0..d { if p[i] == 0.0 { continue; } for j in 0..d { a[i * d + j] += p[i] * p[j]; } for j in 0..k { b[i * k + j] += p[i] * t[j]; } } }
    let tr = (0..d).map(|i| a[i * d + i]).sum::<f64>() / d as f64; for i in 0..d { a[i * d + i] += lam * tr.max(1e-12); }
    let mut l = vec![0.0; d * d];
    for i in 0..d { for j in 0..=i { let mut s = a[i * d + j]; for q in 0..j { s -= l[i * d + q] * l[j * d + q]; } if i == j { l[i * d + i] = s.max(1e-300).sqrt(); } else { l[i * d + j] = s / l[j * d + j]; } } }
    let mut w = vec![vec![0.0; k]; d];
    for c in 0..k { let mut z = vec![0.0; d]; for i in 0..d { let mut s = b[i * k + c]; for q in 0..i { s -= l[i * d + q] * z[q]; } z[i] = s / l[i * d + i]; }
        for i in (0..d).rev() { let mut s = z[i]; for q in i + 1..d { s -= l[q * d + i] * w[q][c]; } w[i][c] = s / l[i * d + i]; } }
    w
}
fn apply(w: &[Vec<f64>], p: &[f64]) -> Vec<f64> { let k = w[0].len(); let mut o = vec![0.0; k]; for i in 0..p.len() { for c in 0..k { o[c] += p[i] * w[i][c]; } } o }
/// oráculo: lectura lineal óptima (ridge) por punto desde rasgos φ(ep, punto) → objetivo por punto; ajusta en la mitad par, evalúa med log rel_err h en la impar
fn oracle(eps: &[Ep], feats: &dyn Fn(&Ep) -> Vec<Vec<f64>>, tgt: &dyn Fn(&Ep) -> Vec<f64>) -> f64 {
    let fs: Vec<Vec<Vec<f64>>> = eps.iter().map(|e| feats(e)).collect();
    let (mut px, mut py) = (vec![], vec![]);
    for (i, e) in eps.iter().enumerate() { if i % 2 == 0 { let t = tgt(e); for (p, f) in fs[i].iter().enumerate() { let mut f1 = f.clone(); f1.push(1.0); px.push(f1); py.push(vec![t[2 * p], t[2 * p + 1]]); } } }
    let w = ridge(&px, &py, 1e-4);
    let errs: Vec<f64> = eps.iter().enumerate().filter(|(i, _)| i % 2 == 1).map(|(i, e)| { let pr: Vec<f64> = fs[i].iter().flat_map(|f| { let mut f1 = f.clone(); f1.push(1.0); apply(&w, &f1) }).collect(); log_rel(clamp_err(&pr, &tgt(e))) }).collect();
    median(errs)
}
/// estados por punto tras observar 0..=K (sin plasticidad)
fn states_at_k(net: &mut Net, e: &Ep) -> Vec<PState> { net.begin_episode(); let np = e.x0.len() / 2; let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect(); for t in 0..=K { constelacion_step(net, &mut sts, e.frame(t), false, 0.0, false); } sts }
pub fn diag(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); assert!((0xC400..=0xC4FF).contains(&seed) || (0xD400..=0xD4FF).contains(&seed), "diag sólo en semillas de ajuste");
    let mut c = cfg_from_args(a);
    if let Some(v) = arg_of(a, "--t-inf") { c.t_inf = v.parse().unwrap(); }
    if let Some(v) = arg_of(a, "--layers") { c.layers = v.split(',').map(|x| x.parse().unwrap()).collect(); }
    if a.iter().any(|x| x == "--no-prune") { c.prune = false; } if a.iter().any(|x| x == "--no-prec") { c.precision = false; }
    if a.iter().any(|x| x == "--no-sleep") { c.sleep = false; } if a.iter().any(|x| x == "--no-spec") { c.rho_max = f64::INFINITY; c.b_in = f64::INFINITY; }
    apply_v3_flags(&mut c, a);
    let n_a: usize = arg_of(a, "--n").map(|s| s.parse().unwrap()).unwrap_or(N_A);
    let fams: Vec<usize> = arg_of(a, "--fams").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or((0..N_FAM).collect());
    let held: Vec<Ep> = gen_heldout(seed, HPF).into_iter().filter(|e| fams.contains(&e.fam)).collect();
    let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
    let snap = |net: &mut Net, held: &[Ep]| -> serde_json::Value {
        let l = net.nl(); let (mut rx, mut rp1, mut rerr1) = (vec![], vec![], vec![]); let mut rmu = vec![vec![]; l]; let (mut ered, mut e8, mut e512) = (vec![], vec![], vec![]);
        for e in held.iter().take(32) { let sts = states_at_k(net, e);
            let p: Vec<f64> = sts.iter().flat_map(|s| net.predict(s, 1)).collect(); rx.push(rms(e.frame(K + 1))); rp1.push(rms(&p)); rerr1.push(log_rel(clamp_err(&p, e.frame(K + 1))));
            for i in 0..l { rmu[i].push(rms(&sts.iter().flat_map(|s| s.mu[i].clone()).collect::<Vec<_>>())); }
            // convergencia de la inferencia: mismo estado previo, T=8 vs T=512 en el cuadro K+1
            let mut s8 = sts.clone(); let mut s512 = sts.clone(); let mut n8 = net.clone(); n8.c.tol = 0.0; let mut n512 = n8.clone(); n512.c.t_inf = 512;
            let a8 = constelacion_step(&mut n8, &mut s8, e.frame(K + 1), false, 0.0, false); let a512 = constelacion_step(&mut n512, &mut s512, e.frame(K + 1), false, 0.0, false);
            ered.push(1.0 - a8.ef / a8.e0.max(1e-300)); e8.push(a8.ef); e512.push(a512.ef); }
        let wn: Vec<f64> = net.w.iter().map(|w| rms(w)).collect();
        let pis: Vec<serde_json::Value> = net.pi.iter().chain(&net.pit).map(|p| { let mut v = p.clone(); v.sort_by(|a, b| a.total_cmp(b)); json!([v[0], v[v.len() / 2], v[v.len() - 1]]) }).collect();
        let cm_act: Vec<f64> = net.cm.iter().map(|m| m.iter().sum::<f64>() / m.len() as f64).collect();
        // unidades muertas: columnas de W_1 (presinápticas) sin ninguna conexión activa
        let dead: Vec<usize> = (0..l).map(|li| { let (r, cc) = (net.dims[li], net.dims[li + 1]); (0..cc).filter(|&j| (0..r).all(|i| net.cm[li][i * cc + j] == 0.0)).count() }).collect();
        json!({"rms_x": mean(&rx), "rms_pred1": mean(&rp1), "pred_over_x": mean(&rp1) / mean(&rx), "h1_med_log": median(rerr1), "rms_mu": rmu.iter().map(|v| mean(v)).collect::<Vec<_>>(),
            "inf_E_reduction_T8": mean(&ered), "E_T8": mean(&e8), "E_T512": mean(&e512), "w_rms": wn, "pi_min_med_max": pis, "conn_active": cm_act, "dead_presyn": dead,
            "eval": eval_held(net, held), "eta_scale": net.eta_scale, "prune": [net.n_pruned, net.n_grown, net.n_rescale_row, net.n_rescale_spec]})
    };
    let s0 = snap(&mut net, &held);
    let stream = gen_known_stream(seed, XOR_KNOWN, n_a, &fams);
    let (h1, h16, _) = run_stream(&mut net, &mut cdt, &stream, seed);
    let s1 = snap(&mut net, &held);
    // oráculos (lecturas lineales óptimas, ajuste/eval en H conocidas mitad/mitad)
    let nref = &mut net; let l = nref.nl();
    let mut cache: std::collections::HashMap<usize, Vec<PState>> = Default::default();
    for (i, e) in held.iter().enumerate() { cache.insert(i, states_at_k(nref, e)); }
    let idx = |e: &Ep| held.iter().position(|x| std::ptr::eq(x, e)).unwrap();
    let net2 = nref.clone();
    let y16 = |e: &Ep| e.ys[HORIZON - 1].clone(); let y5 = |e: &Ep| e.frame(K + 1).to_vec();
    let o_l1_h1 = oracle(&held, &|e| cache[&idx(e)].iter().map(|s| net2.fl(1, &net2.fl(1, &s.mu[0]).0).0).collect(), &y5);
    let o_all_h16 = oracle(&held, &|e| cache[&idx(e)].iter().map(|s| s.mu.iter().flatten().cloned().collect()).collect(), &y16);
    let o_top_h16 = oracle(&held, &|e| cache[&idx(e)].iter().map(|s| { let mut t = s.mu[l - 1].clone(); for _ in 0..HORIZON - K { t = net2.fl(l, &t).0; } net2.fl(l, &t).0 }).collect(), &y16);
    let o_x_h1 = oracle(&held, &|e| (0..e.x0.len() / 2).map(|p| vec![e.frame(K)[2 * p], e.frame(K)[2 * p + 1], e.frame(K - 1)[2 * p], e.frame(K - 1)[2 * p + 1]]).collect(), &y5);
    let o_x_h16 = oracle(&held, &|e| (0..e.x0.len() / 2).map(|p| (0..=K).flat_map(|t| vec![e.frame(t)[2 * p], e.frame(t)[2 * p + 1]]).collect()).collect(), &y16);
    let triv16 = median(held.iter().map(|e| log_rel(clamp_err(e.frame(K), &e.ys[HORIZON - 1]))).collect());
    let triv1 = median(held.iter().map(|e| log_rel(clamp_err(e.frame(K), e.frame(K + 1)))).collect());
    let q = h1.len() / 4;
    println!("{}", json!({"seed": format!("0x{seed:X}"), "cfg": c.to_json(), "n": n_a, "before": s0, "after": s1, "collapse": collapse(nref, &held),
        "stream_last_q": {"h1": mean(&h1[h1.len() - q..]), "h16": mean(&h16[h16.len() - q..])}, "stream_first_q": {"h1": mean(&h1[..q]), "h16": mean(&h16[..q])},
        "oracle_med_log": {"layer1_FF_to_x5": o_l1_h1, "all_mu_to_y16": o_all_h16, "top_rollout_to_y16": o_top_h16, "x4x3_to_x5": o_x_h1, "x0..4_to_y16": o_x_h16, "trivial_h1": triv1, "trivial_h16": triv16, "zero": 0.0}}));
}
fn apply_v3_flags(_c: &mut Cfg, _a: &[String]) {}
/// oráculo en contexto: por episodio, ajuste afín x(t+1)−x(t) = A x(t) + c sobre los cuadros 0..=K (24 muestras), rollout a 16
pub fn oracle_icl(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); assert!((0xC400..=0xC4FF).contains(&seed) || (0xD400..=0xD4FF).contains(&seed));
    let held = gen_heldout(seed, HPF); let mut r = serde_json::Map::new();
    for lam in [1e-6, 1e-3, 1e-2, 1e-1] {
        let mut by = std::collections::BTreeMap::<String, Vec<f64>>::new();
        for e in &held { let np = e.x0.len() / 2; let (mut px, mut py) = (vec![], vec![]);
            for t in 0..K { for p in 0..np { let x = e.frame(t); let y = e.frame(t + 1); px.push(vec![x[2 * p], x[2 * p + 1], 1.0]); py.push(vec![y[2 * p] - x[2 * p], y[2 * p + 1] - x[2 * p + 1]]); } }
            let w = ridge(&px, &py, lam); let mut z = e.frame(K).to_vec();
            for _ in K..HORIZON { let mut n = z.clone(); for p in 0..np { let d = apply(&w, &[z[2 * p], z[2 * p + 1], 1.0]); n[2 * p] += d[0]; n[2 * p + 1] += d[1]; } z = n; }
            let g = if e.fam < N_FAM { "known" } else { "new" }; by.entry(g.into()).or_default().push(log_rel(clamp_err(&z, &e.ys[HORIZON - 1]))); }
        for (g, v) in by { r.insert(format!("icl_affine_lam{lam}_{g}"), json!(median(v))); }
    }
    for g in ["known", "new"] { r.insert(format!("trivial_{g}"), json!(median(held.iter().filter(|e| (e.fam < N_FAM) == (g == "known")).map(|e| log_rel(clamp_err(e.frame(K), &e.ys[HORIZON - 1]))).collect()))); }
    println!("{}", serde_json::Value::Object(r));
}

/// v3: cabeza temporal. Rápido (estado de episodio, NLMS sobre la transición actual, t_f iteraciones) + lento (tres factores: η_P/τ · Π^P·e · φ).
fn head_update(net: &mut Net, sts: &[PState], x: &[f64], pre: &[Option<Vec<f64>>], ep: &[Option<Vec<f64>>], train: bool, lr_mult: f64, prec_only: bool, imagine: bool) {
    let c = net.c.clone(); let d = net.dp; let np = pre.len();
    let obs: Vec<usize> = (0..np).filter(|&p| pre[p].is_some()).collect(); let no = obs.len() as f64;
    if train {
        if c.precision { for i in 0..2 { let e2 = obs.iter().map(|&p| ep[p].as_ref().unwrap()[i].powi(2)).sum::<f64>() / no; net.pip[i] = (net.pip[i] + c.eta_pi * 0.5 * (1.0 / net.pip[i] - e2)).clamp(c.pip_min, c.pip_max); }
            if c.pi_norm { let g = (net.pip.iter().map(|v| v.ln()).sum::<f64>() / 2.0).exp(); for v in net.pip.iter_mut() { *v /= g; } } }
        if !prec_only && c.plastic_w {
            let eta = c.eta_p * net.eta_scale * lr_mult;
            let mut g = vec![0.0; 2 * d];
            for &p in &obs { let phi = pre[p].as_ref().unwrap(); let e = ep[p].as_ref().unwrap(); for i in 0..2 { let pe = if c.backprop { e[i] } else { net.pip[i] * e[i] }; for j in 0..d { g[i * d + j] += pe * phi[j] / no; } } }
            let mut ub = 0.0;
            for k in 0..2 * d { let tau = if c.prune && !c.backprop { net.taup[k] } else { 1.0 }; let dw = eta / tau * g[k] - c.lam * net.wp[k]; net.wp[k] += dw;
                if c.prune { net.up[k] = (1.0 - c.zeta) * net.up[k] + c.zeta * dw.abs() * net.wp[k].abs(); } ub += net.up[k] / (2 * d) as f64; }
            if c.prune && ub > 0.0 { for k in 0..2 * d { net.taup[k] = (net.taup[k] * (1.0 + c.rho_tau * (net.up[k] / ub - 1.0))).clamp(c.tau_min, c.tau_max); } }
        }
    }
    let _ = sts;
    if c.fast && !imagine && c.eta_f > 0.0 {
        // NLMS en lote sobre los puntos del cuadro (campo compartido), t_f iteraciones; base = x(t−1) guardado en el estado
        let nrm: f64 = obs.iter().map(|&p| pre[p].as_ref().unwrap().iter().map(|v| v * v).sum::<f64>()).sum::<f64>() + 1e-9;
        for _ in 0..c.t_f { let mut g = vec![0.0; 2 * d];
            for &p in &obs { let phi = pre[p].as_ref().unwrap(); let xh = net.pred_from(&sts[p], phi); for i in 0..2 { let e = x[2 * p + i] - xh[i]; for j in 0..d { g[i * d + j] += e * phi[j]; } } }
            for k in 0..2 * d { net.wf[k] += c.eta_f * g[k] / nrm; } }
    }
}
