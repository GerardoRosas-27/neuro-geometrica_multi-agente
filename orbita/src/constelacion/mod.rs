//! CONSTELACIÓN v2 (docs/constelacion_v2): red jerárquica de ÓRBITAS-neurona (Koopman bloque rot-escala + levantamiento
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
}
impl Cfg {
    pub fn base() -> Cfg { Cfg { m: 8, h: 8, layers: vec![16, 8, 4], identity_init: false, eta_b: 0.1, t_inf: 8, kappa: 0.1, tol: 1e-3,
        eta_w: 0.01, lam: 1e-6, w_init: 0.1, eta_pi: 1e-3, pi_min: 0.01, pi_max: 100.0, tau_min: 1.0, tau_max: 10.0, zeta: 0.01, rho_tau: 0.01,
        p_every: 500, w_min: 1e-4, u_min_rel: 0.01, b_in: 10.0, rho_max: 0.99, mu_clip: 10.0,
        n_s: 16, n_replay: 16, alpha_s: 0.1, delta_k: 0.05, n_kbuf: 16,
        errors: true, precision: true, prune: true, sleep: true, plastic_w: true, backprop: false } }
    pub fn to_json(&self) -> serde_json::Value { json!(format!("{:?}", self)) }
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Arm { Cst, NoErr, NoPrec, T1, NoPrune, NoSleep, FixedW, OneLayer, Backprop }
impl Arm {
    pub fn all() -> [Arm; 9] { [Arm::Cst, Arm::NoErr, Arm::NoPrec, Arm::T1, Arm::NoPrune, Arm::NoSleep, Arm::FixedW, Arm::OneLayer, Arm::Backprop] }
    pub fn name(self) -> &'static str { match self { Arm::Cst => "cst", Arm::NoErr => "no_err", Arm::NoPrec => "no_prec", Arm::T1 => "t1", Arm::NoPrune => "no_prune", Arm::NoSleep => "no_sleep", Arm::FixedW => "fixed_w", Arm::OneLayer => "one_layer", Arm::Backprop => "backprop" } }
    pub fn cfg(self, b: &Cfg) -> Cfg { let mut c = b.clone(); match self {
        Arm::Cst => {}, Arm::NoErr => c.errors = false, Arm::NoPrec => c.precision = false, Arm::T1 => c.t_inf = 1, Arm::NoPrune => c.prune = false,
        Arm::NoSleep => c.sleep = false, Arm::FixedW => c.plastic_w = false, Arm::OneLayer => c.layers = vec![28], Arm::Backprop => c.backprop = true } c }
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
}
#[derive(Clone)]
pub struct PState { pub mu: Vec<Vec<f64>>, pub started: bool }
pub struct StepInfo { pub e0: f64, pub ef: f64, pub e_layer: Vec<f64>, pub iters: f64 }

impl Net {
    pub fn new(c: &Cfg, seed: u64) -> Net {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_INIT);
        let l = c.layers.len(); let mut dims = vec![2]; for &n in &c.layers { dims.push(n * c.m); }
        let neu: Vec<Vec<Neuron>> = c.layers.iter().map(|&n| (0..n).map(|_| Neuron::new(&mut rng, c)).collect()).collect();
        let mut w = vec![]; for i in 1..=l { let fi = dims[i] as f64; w.push((0..dims[i - 1] * dims[i]).map(|_| gauss(&mut rng) * (1.0 / fi).sqrt() * c.w_init).collect::<Vec<f64>>()); }
        let sz = |i: usize| dims[i - 1] * dims[i];
        let uin = (0..dims[l] * 2).map(|_| gauss(&mut rng) * 0.5).collect();
        Net { dims: dims.clone(), neu, cm: (1..=l).map(|i| vec![1.0; sz(i)]).collect(), tau: (1..=l).map(|i| vec![c.tau_min; sz(i)]).collect(),
            u: (1..=l).map(|i| vec![0.0; sz(i)]).collect(), low: (1..=l).map(|i| vec![0; sz(i)]).collect(), act: (1..=l).map(|i| vec![0.0; sz(i)]).collect(),
            pi: (0..l).map(|i| vec![1.0; dims[i]]).collect(), pit: (1..=l).map(|i| vec![1.0; dims[i]]).collect(), w, uin, eta_scale: 1.0, steps: 0, nan: false,
            n_pruned: 0, n_grown: 0, n_rescale_row: 0, n_rescale_spec: 0, c: c.clone() }
    }
    pub fn nl(&self) -> usize { self.c.layers.len() }
    pub fn n_w(&self) -> usize { self.w.iter().map(|x| x.len()).sum() }
    pub fn hash(&self) -> u64 { let mut h: u64 = 0xcbf29ce484222325; for v in self.w.iter().chain(&self.pi).chain(&self.pit).chain(&self.tau).chain(&self.cm).flatten() { h ^= v.to_bits(); h = h.wrapping_mul(0x100000001b3); } h }
    pub fn fresh_state(&self) -> PState { PState { mu: (1..=self.nl()).map(|i| vec![0.0; self.dims[i]]).collect(), started: false } }
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
            let mu = &mut st.mu[i - 1];
            for j in 0..mu.len() {
                let mut g = -back[j];
                if i < l { g += eps[i][j]; }
                if started { g += net.pit[i - 1][j] * (mu[j] - prior[i - 1][j]); }
                mu[j] -= c.kappa * g;
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
pub fn constelacion_step(net: &mut Net, sts: &mut [PState], x: &[f64], train: bool, lr_mult: f64, prec_only: bool) -> StepInfo {
    let np = x.len() / 2; let l = net.nl();
    let mut acc = Acc { gw: net.w.iter().map(|w| vec![0.0; w.len()]).collect(), e2: net.pi.iter().map(|p| vec![0.0; p.len()]).collect(), et2: net.pit.iter().map(|p| vec![0.0; p.len()]).collect(), red: 0.0, np: 0.0, nt: 0.0 };
    let prevs: Option<Vec<PState>> = if train && net.c.backprop { Some(sts.to_vec()) } else { None };
    let (mut e0, mut ef, mut it) = (0.0, 0.0, 0.0); let mut elay = vec![0.0; l + 1];
    for p in 0..np {
        let r = infer_point(net, &mut sts[p], &x[2 * p..2 * p + 2], if train { Some(&mut acc) } else { None });
        e0 += r.0 / np as f64; ef += r.1 / np as f64; it += r.3 / np as f64; for i in 0..r.2.len() { elay[i] += r.2[i] / np as f64; }
    }
    if !ef.is_finite() || sts.iter().any(|s| s.mu.iter().any(|v| !fin(v))) { net.nan = true; }
    if train && !net.nan { plasticity(net, &acc, lr_mult, prec_only, prevs.as_deref(), x, sts); }
    StepInfo { e0, ef, e_layer: elay, iters: it }
}

fn plasticity(net: &mut Net, a: &Acc, lr_mult: f64, prec_only: bool, prevs: Option<&[PState]>, x: &[f64], sts: &[PState]) {
    let c = net.c.clone(); let l = net.nl(); let np = a.np.max(1.0);
    // precisiones (regla local de máxima verosimilitud)
    if c.precision && c.errors {
        for i in 0..l { for j in 0..net.pi[i].len() { let p = &mut net.pi[i][j]; *p = (*p + c.eta_pi * 0.5 * (1.0 / *p - a.e2[i][j] / np)).clamp(c.pi_min, c.pi_max); } }
        if a.nt > 0.0 { for i in 0..l { for j in 0..net.pit[i].len() { let p = &mut net.pit[i][j]; *p = (*p + c.eta_pi * 0.5 * (1.0 / *p - a.et2[i][j] / a.nt)).clamp(c.pi_min, c.pi_max); } } }
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
    let np = e.x0.len() / 2; let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect();
    let last = if train { HORIZON } else { k_obs };
    let mut h1 = vec![]; let mut es = 0.0; let mut el = vec![0.0; net.nl() + 1]; let mut h16 = 10.0; let mut preds = vec![vec![]; HORIZON];
    for t in 0..=last {
        if t >= 1 { let p: Vec<f64> = sts.iter().flat_map(|s| net.predict(s, 1)).collect(); h1.push(log_rel(clamp_err(&p, e.frame(t)))); if t <= k_obs { preds[t - 1] = p; } }
        let si = constelacion_step(net, &mut sts, e.frame(t), train, lr_mult, prec_only);
        es += si.ef; for i in 0..el.len() { el[i] += si.e_layer[i]; }
        if t == k_obs { for hh in (k_obs + 1)..=HORIZON { preds[hh - 1] = sts.iter().flat_map(|s| net.predict(s, hh - k_obs)).collect(); } h16 = clamp_err(&preds[HORIZON - 1], &e.ys[HORIZON - 1]); }
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
    for e in held.iter().filter(|e| e.fam < N_FAM).take(32) { let np = e.x0.len() / 2; let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect();
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
pub const N_A: usize = 768; // flujo A: 96 por familia conocida
pub const N_NEW: usize = 48; pub const N_KNOWN_B: usize = 48; // flujo B = ECO (192)
pub const HPF: usize = 16;

/// F0: una ÓRBITA-neurona (L=1, n=1) en la familia 0, η vs η=0 vs trivial
pub fn f0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); assert!((0xC400..=0xC4FF).contains(&seed));
    let n: usize = arg_of(a, "--n").map(|s| s.parse().unwrap()).unwrap_or(256);
    let mut c = Cfg::base(); c.layers = vec![1]; c.sleep = false;
    let stream = gen_known_stream(seed, XOR_KNOWN, n, &[0]);
    let mut out = serde_json::Map::new();
    for (name, eta) in [("eta", c.eta_w), ("eta0", 0.0)] {
        let mut cc = c.clone(); cc.eta_w = eta; let mut net = Net::new(&cc, seed); let mut cdt = Cdt::new(seed, &cc);
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
    let (ex, sha) = e33_examples(seed); let e33c = e33_net(&mut net, &ex); let e33o = e33_orbita(&d, &ex);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(); assert_eq!(q, 0);
    let gap = post["known_med_log"].as_f64().unwrap() - orb["known_med_log"].as_f64().unwrap();
    println!("{}", json!({"seed": format!("0x{seed:X}"), "cfg": c.to_json(), "pre": pre, "post": post, "orbita": orb, "gap_log": gap, "stop_gap": gap > 1.0, "collapse": col,
        "e33": {"cst": e33c, "orbita": e33o}, "test_sha256": sha, "test_queries": q, "h1_curve": h1, "h16_curve": h16, "slope_h16": slope(&h16), "nan": net.nan,
        "prune": {"pruned": net.n_pruned, "grown": net.n_grown, "row_rescale": net.n_rescale_row, "spec_rescale": net.n_rescale_spec}, "sleeps": cdt.log, "wall_s": t0.elapsed().as_secs_f64()}));
}
fn cfg_from_args(a: &[String]) -> Cfg {
    let mut c = Cfg::base();
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
    if phase != "smoke" { assert!(arg_of(a, "--eta-w").is_none() && arg_of(a, "--kappa").is_none() && arg_of(a, "--eta-pi").is_none() && arg_of(a, "--w-init").is_none(), "DEV/CONFIRM: valores congelados"); }
    let wpath = arg_of(a, "--weights").unwrap(); let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap()); let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip"); return; }
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let base = cfg_from_args(a);
    let held = gen_heldout(seed, HPF); let held_known: Vec<Ep> = held.iter().filter(|e| e.fam < N_FAM).cloned().collect();
    let sa = gen_known_stream(seed, XOR_KNOWN, N_A, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, N_NEW, N_KNOWN_B);
    let (ex, sha) = e33_examples(seed);
    let mut arms = serde_json::Map::new();
    for arm in Arm::all() {
        if let Some(o) = &only { if !o.iter().any(|x| x == arm.name()) { continue; } }
        let ta = std::time::Instant::now(); let c = arm.cfg(&base);
        let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
        let (_, h16a, _) = run_stream(&mut net, &mut cdt, &sa, seed);
        let pre = eval_held(&mut net, &held_known);
        let (_, h16b, fb) = run_stream(&mut net, &mut cdt, &sb, seed);
        let post = eval_held(&mut net, &held); let col = collapse(&mut net, &held); let e33 = e33_net(&mut net, &ex);
        let auc_new = mean(&h16b.iter().zip(&fb).filter(|(_, &f)| f >= N_FAM).map(|(v, _)| *v).collect::<Vec<_>>());
        arms.insert(arm.name().into(), json!({"pre_known": pre, "held": post, "collapse": col, "e33": e33, "auc_new_preq": auc_new, "streamA_last_q_h16": mean(&h16a[h16a.len() * 3 / 4..]),
            "nan": net.nan, "hash": format!("{:016x}", net.hash()), "eta_scale": net.eta_scale, "sleeps": cdt.log.len(), "prune": [net.n_pruned, net.n_grown, net.n_rescale_row, net.n_rescale_spec], "wall_s": ta.elapsed().as_secs_f64()}));
        eprintln!("0x{seed:X} {} {:.1}s", arm.name(), ta.elapsed().as_secs_f64());
    }
    let orb = orbita_held(&d, &held); let e33o = e33_orbita(&d, &ex);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(d.raw, raw0); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let r = json!({"seed": format!("0x{seed:X}"), "phase": phase, "weights": wpath, "orbita": orb, "e33_orbita": e33o, "arms": arms, "test_queries": q, "test_sha256": sha, "git_sha": git_sha(), "cfg": base.to_json(), "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out.join(format!("metrics_0x{seed:X}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok {:.0}s", t0.elapsed().as_secs_f64());
}

// ─── Paso 0 ──────────────────────────────────────────────────────────────────
fn trace(net: &mut Net, eps: &[Ep], train: bool) -> Vec<u64> {
    let mut v = vec![]; for e in eps { let o = run_episode(net, e, train, 1.0, false, HORIZON); for p in o.preds.iter().flatten() { v.push(p.to_bits()); } v.push(o.e_mean.to_bits()); v.push(o.h1_log.to_bits()); } v
}
pub fn paso0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC400".into())); assert!((0xC400..=0xC4FF).contains(&seed));
    let base = Cfg::base(); let mut res = serde_json::Map::new();
    let eps = gen_known_stream(seed, XOR_KNOWN, 16, &(0..N_FAM).collect::<Vec<_>>());
    // 0a: train con plasticidad off (tasas 0, sin poda/sueño) ≡ eval, por brazo
    let mut bad = 0usize; let mut nv = 0usize;
    for arm in Arm::all() { let mut c = arm.cfg(&base); c.eta_w = 0.0; c.eta_pi = 0.0; c.lam = 0.0; c.zeta = 0.0; c.rho_tau = 0.0; c.prune = false; c.p_every = usize::MAX;
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
        let loss = |n: &Net| { let p = n.predict(&PState { mu: vec![vec![0.0; n.dims[1]], mu2.clone()], started: true }, 1); 0.5 * ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2)) };
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
    let ok = res["0a"]["pass"] == true && res["0c"]["pass"] == true;
    println!("{}", json!({"seed": format!("0x{seed:X}"), "git_sha": git_sha(), "paso0_ok_0a_0c": ok, "res": res}));
}
