//! ESTELA v0 (docs/estela/PREREG.md). Arquitectura de estados sin tokens, complementaria a ÓRBITA.
//!   orbita estela-paso0 --weights W [--weights W2 ...]                         compuerta de paridad (JSON a stdout)
//!   orbita estela-run --phase smoke|dev|confirm --weights W --seed S --out DIR [--steps N --h-per-fam H]
//! Núcleo ÓRBITA congelado (sólo se leen sus pesos; se verifica `raw` al final). No lee ds.test.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub mod ad;
pub mod model;
use crate::fam::reid;
use crate::latent::{fit_gn, ctx8, lat_step, npar};
use crate::model::{Orbita, M};
use ad::{Var, S};
use cdt_rqm_epr::field_autonomy_v4::K_POINTS;
use cdt_rqm_epr::v4_dataset::{apply_rule, context_of, make_object, ObjectClass, N_FAM};
use cdt_rqm_epr::v4_metrics as mt;
use model::{Frozen, Lay, Variant};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::Write;
use std::path::PathBuf;

// ─── hiperparámetros congelados a priori (PREREG §4) ─────────────────────────
pub const STEPS: usize = 3000; // pasos de Adam por variante y seed
pub const BATCH: usize = 16; // episodios por paso
pub const HTRAIN: usize = 4; // horizonte multi-paso del objetivo JEPA
pub const LR: f64 = 1e-3;
pub const CLIP: f64 = 1.0; // norma máxima del gradiente
pub const LAM_SIG: f64 = 0.1; // SIGReg
pub const LAM_PROBE: f64 = 0.1; // sonda de estado físico (poco peso)
pub const LAM_REC: f64 = 1.0; // sólo control ESTELA-REC
pub const N_SLICES: usize = 16; // direcciones de SIGReg por paso
pub const N_TRAIN: usize = 2048; // trayectorias de entrenamiento (familias 0–7, contexto verdadero)
pub const N_CAL: usize = 256; // calibración de τ_E (familias 0–7) — no se usa H
pub const TAU_Q: f64 = 0.95; // cuantil para τ_E(k)
pub const LAT_LAM0: f64 = 1e-4; // = ÓRBITA+LAT
pub const KS: [usize; 3] = [2, 4, 8];
pub const PRIMARY_NEW: [usize; 3] = [8, 10, 11];
pub const H_PER_FAM: usize = 16;
pub const HORIZON: usize = 16;
// XOR de RNG: H idéntico a Nova (0x4E0A_501D); entrenamiento y calibración propios
const XOR_HELD: u64 = 0x4E0A_501D;
const XOR_TRAIN: u64 = 0xE57E_7A11;
const XOR_CAL: u64 = 0xE57E_CA11;
const XOR_OPT: u64 = 0xE57E_0B70;

/// τ_LAT(k) de ÓRBITA+LAT (congelado en el ciclo latente; idéntico a nova2::lat_tau)
pub fn lat_tau(k: usize) -> f64 { match k { 2 => 0.014655674499999998, 4 => 0.0220202535, _ => 0.037142499499999995 } }
pub fn k_fit_val(k: usize) -> (usize, usize) { match k { 2 => (1, 1), 4 => (3, 1), 8 => (6, 2), _ => ((k * 3) / 4, k - (k * 3) / 4) } }

// ─── familias (0–7 v4; 8–11 de fam.rs/Nova, portadas sin cambios) ────────────
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
pub struct Ep { pub fam: usize, pub p: [f64; 2], pub c: Vec<f64>, pub x0: Vec<f64>, pub ys: Vec<Vec<f64>> }
/// mismo generador que nova_train::make_ep (un único generador para entrenamiento, calibración y evaluación)
pub fn make_ep(rng: &mut Xoshiro256StarStar, fam: usize, horizon: usize) -> Ep {
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
    let c = if fam < N_FAM { context_of(fam, p) } else { context_of(0, p) };
    let ci = rng.gen_range(0..3);
    let x0 = make_object(rng, K_POINTS, cls[ci]);
    let mut ys = Vec::with_capacity(horizon); let mut y = x0.clone();
    for _ in 0..horizon { y = rule(fam, p, &y); ys.push(y.clone()); }
    Ep { fam, p, c, x0, ys }
}
pub fn gen_heldout(seed: u64, n_per_fam: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_HELD);
    let fams: Vec<usize> = (0..N_FAM).chain(PRIMARY_NEW.iter().copied()).collect();
    let mut out = vec![]; for &f in &fams { for _ in 0..n_per_fam { out.push(make_ep(&mut rng, f, HORIZON)); } } out
}
pub fn gen_known(seed: u64, xor: u64, n: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ xor);
    (0..n).map(|i| make_ep(&mut rng, i % N_FAM, HORIZON)).collect()
}
fn fin(z: &[f64]) -> bool { z.iter().all(|v| v.is_finite()) }
fn clamp_err(z: &[f64], y: &[f64]) -> f64 { if fin(z) { mt::rel_err(z, y).min(10.0) } else { 10.0 } }
fn log_rel(e: f64) -> f64 { e.max(1e-12).ln() }
fn median(mut v: Vec<f64>) -> f64 { if v.is_empty() { return f64::NAN; } v.sort_by(|a, b| a.total_cmp(b)); mt::median(&v) }
fn quantile(mut v: Vec<f64>, q: f64) -> f64 { v.sort_by(|a, b| a.total_cmp(b)); let i = ((q * (v.len() as f64 - 1.0)).round() as usize).min(v.len() - 1); v[i] }

// ─── ruta única de predicción (evaluación) ───────────────────────────────────
/// trayectoria q̂_1..q̂_n desde x0 con la MISMA función `model::step` que usa el entrenamiento
pub fn rollout(th: &[f64], l: &Lay, fz: &Frozen, x0: &[f64], n: usize, lat: Option<&[f64]>) -> Vec<Vec<f64>> {
    let mut x = x0.to_vec(); let mut out = Vec::with_capacity(n);
    for _ in 0..n { if !fin(&x) { out.push(x.clone()); continue; } x = model::step(th, l, fz, &x, lat).q; out.push(x.clone()); }
    out
}
/// sorpresa por energía (E3): media en los k pasos observados de E(z_obs) − E(ẑ*), predicción a un paso desde lo observado
pub fn surprise_e(th: &[f64], l: &Lay, fz: &Frozen, e: &Ep, k: usize) -> f64 {
    let mut s = 0.0; let mut n = 0.0;
    for t in 0..k {
        let x = if t == 0 { &e.x0 } else { &e.ys[t - 1] };
        let o = model::step(th, l, fz, x, None);
        for i in 0..x.len() / 2 {
            let zo = model::enc(th, l, [e.ys[t][2 * i], e.ys[t][2 * i + 1]], &fz.c);
            let v = model::energy(th, l, &zo, &o.z0[i]) - model::energy(th, l, &o.zhat[i], &o.z0[i]);
            s += if v.is_finite() { v } else { 1e6 }; n += 1.0;
        }
    }
    s / n
}

// ─── entrenamiento ───────────────────────────────────────────────────────────
pub struct TrainLog { pub first: [f64; 4], pub last: [f64; 4], pub tape_max: usize }
/// pérdida de un minibatch. `th` son hojas Var; la predicción usa `model::step` (la misma que evaluación).
/// Devuelve (total, [jepa, sig, probe, rec]) y, si `rec_preds`, las predicciones q̂ (para la prueba de paridad).
pub fn batch_loss(d: &Orbita, th: &[Var], l: &Lay, v: Variant, eps: &[(&Ep, usize)], dirs: &[Vec<f64>], rec_preds: Option<&mut Vec<Vec<f64>>>) -> (Var, [f64; 4]) {
    let thf: Vec<f64> = th.iter().map(|x| x.v()).collect(); // sg(·): codificador objetivo = mismos pesos, sin gradiente
    batch_loss_tgt(d, th, &thf, l, v, eps, dirs, rec_preds)
}
/// igual que `batch_loss` con los pesos del codificador objetivo explícitos (para la prueba de diferencias finitas)
pub fn batch_loss_tgt(d: &Orbita, th: &[Var], thf: &[f64], l: &Lay, v: Variant, eps: &[(&Ep, usize)], dirs: &[Vec<f64>], rec_preds: Option<&mut Vec<Vec<f64>>>) -> (Var, [f64; 4]) {
    let (mut lj, mut lp, mut lr) = (Var::c(0.0), Var::c(0.0), Var::c(0.0));
    let mut emb: Vec<[Var; M]> = vec![];
    let mut preds = vec![];
    let nb = eps.len() as f64;
    for (e, t0) in eps {
        let fz = Frozen::new(d, &e.c, v);
        let st = |t: usize| if t == 0 { &e.x0 } else { &e.ys[t - 1] }; // estado en tiempo t
        let mut x: Vec<Var> = st(*t0).iter().map(|&q| Var::c(q)).collect();
        for i in 0..K_POINTS { emb.push(model::enc(th, l, [x[2 * i], x[2 * i + 1]], &e.c)); }
        if v == Variant::Rec { // reconstrucción del estado actual desde su latente (P·E(s_t) ≈ s_t)
            for i in 0..K_POINTS { let z = model::enc(th, l, [x[2 * i], x[2 * i + 1]], &e.c); for o in 0..2 { let q = model::dot_pub(&th[l.pr + o * M..l.pr + (o + 1) * M], &z) - x[2 * i + o]; lr = lr + q * q; } }
        }
        for h in 1..=HTRAIN {
            let o = model::step(th, l, &fz, &x, None);
            let y = st(t0 + h);
            let yn = mt::norm(y).max(1e-9);
            for i in 0..K_POINTS {
                let zt = model::enc(thf, l, [y[2 * i], y[2 * i + 1]], &e.c);
                for j in 0..M { let dd = o.zhat[i][j].add_c(-zt[j]); lj = lj + dd * dd; }
            }
            let mut pe = Var::c(0.0); for (a, b) in o.q.iter().zip(y) { let dd = a.add_c(-b); pe = pe + dd * dd; }
            lp = lp + pe.mul_c(1.0 / (yn * yn));
            if preds.len() < 64 { preds.push(o.q.iter().map(|q| q.v()).collect()); }
            for i in 0..K_POINTS { if h < HTRAIN { emb.push(model::enc(th, l, [o.q[2 * i], o.q[2 * i + 1]], &e.c)); } }
            x = o.q;
        }
    }
    if let Some(r) = rec_preds { *r = preds; }
    let lj = lj.mul_c(1.0 / (nb * (HTRAIN * K_POINTS * M) as f64));
    let lp = lp.mul_c(1.0 / (nb * HTRAIN as f64));
    let lr = lr.mul_c(1.0 / (nb * K_POINTS as f64));
    let ls = sigreg(&emb, dirs);
    let mut tot = lj + lp.mul_c(LAM_PROBE);
    if v != Variant::NoSig { tot = tot + ls.mul_c(LAM_SIG); }
    if v == Variant::Rec { tot = tot + (lp + lr).mul_c(LAM_REC); }
    (tot, [lj.v(), ls.v(), lp.v(), lr.v()])
}
/// SIGReg (LeJEPA): Epps–Pulley sobre proyecciones 1D de las coordenadas aprendibles z[2..8] contra N(0,1)
pub fn sigreg<T: S>(emb: &[[T; M]], dirs: &[Vec<f64>]) -> T {
    let nt = 17; let tmax = 4.0; let n = emb.len() as f64;
    let mut tot = T::c(0.0);
    for a in dirs {
        let pr: Vec<T> = emb.iter().map(|z| { let mut s = z[2].mul_c(a[0]); for j in 1..M - 2 { s = s + z[2 + j].mul_c(a[j]); } s }).collect();
        for ti in 0..nt {
            let t = tmax * ti as f64 / (nt - 1) as f64;
            let w = (if ti == 0 || ti == nt - 1 { 0.5 } else { 1.0 }) * (tmax / (nt - 1) as f64) * (-0.5 * t * t).exp();
            let (mut cr, mut ci) = (T::c(0.0), T::c(0.0));
            for p in &pr { let tp = p.mul_c(t); cr = cr + tp.cos(); ci = ci + tp.sin(); }
            let dr = cr.mul_c(1.0 / n).add_c(-(-0.5 * t * t).exp()); let di = ci.mul_c(1.0 / n);
            tot = tot + (dr * dr + di * di).mul_c(w);
        }
    }
    tot.mul_c(1.0 / dirs.len() as f64)
}
fn rand_dirs(rng: &mut Xoshiro256StarStar) -> Vec<Vec<f64>> {
    (0..N_SLICES).map(|_| { let mut a: Vec<f64> = (0..M - 2).map(|_| { let u1: f64 = rng.gen::<f64>().max(1e-300); let u2: f64 = rng.gen(); (-2.0 * u1.ln()).sqrt() * (6.283185307179586 * u2).cos() }).collect(); let n = mt::norm(&a); for x in &mut a { *x /= n; } a }).collect()
}
/// gradiente del minibatch respecto de θ (máscara de congelados aplicada)
pub fn grad(d: &Orbita, th: &[f64], l: &Lay, v: Variant, eps: &[(&Ep, usize)], dirs: &[Vec<f64>]) -> (Vec<f64>, [f64; 4], usize) {
    ad::reset();
    let tv: Vec<Var> = th.iter().map(|&x| ad::leaf(x)).collect();
    let (loss, parts) = batch_loss(d, &tv, l, v, eps, dirs, None);
    let tl = ad::tape_len();
    let g = ad::backward(loss);
    let mut out: Vec<f64> = tv.iter().map(|x| g[x.i as usize]).collect();
    if !v.uses_r() { for i in l.r1..l.rb2 + M { out[i] = 0.0; } }
    ad::reset();
    (out, parts, tl)
}
pub fn train(d: &Orbita, l: &Lay, v: Variant, seed: u64, data: &[Ep], steps: usize) -> (Vec<f64>, TrainLog) {
    let mut th = model::init(d, l, seed);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_OPT); // mismo flujo de minibatches para todas las variantes
    let (mut m1, mut m2) = (vec![0.0; l.n], vec![0.0; l.n]);
    let (b1, b2, epsa) = (0.9, 0.999, 1e-8);
    let mut log = TrainLog { first: [0.0; 4], last: [0.0; 4], tape_max: 0 };
    let mut acc = [0.0; 4]; let win = (steps / 10).max(1);
    for s in 0..steps {
        let batch: Vec<(&Ep, usize)> = (0..BATCH).map(|_| (&data[rng.gen_range(0..data.len())], rng.gen_range(0..=HORIZON - HTRAIN))).collect();
        let dirs = rand_dirs(&mut rng);
        let (mut g, parts, tl) = grad(d, &th, l, v, &batch, &dirs);
        log.tape_max = log.tape_max.max(tl);
        if !g.iter().all(|x| x.is_finite()) { continue; }
        let gn = g.iter().map(|x| x * x).sum::<f64>().sqrt(); if gn > CLIP { for x in &mut g { *x *= CLIP / gn; } }
        let t = (s + 1) as i32;
        for i in 0..l.n { m1[i] = b1 * m1[i] + (1.0 - b1) * g[i]; m2[i] = b2 * m2[i] + (1.0 - b2) * g[i] * g[i];
            let mh = m1[i] / (1.0 - b1.powi(t)); let vh = m2[i] / (1.0 - b2.powi(t)); th[i] -= LR * mh / (vh.sqrt() + epsa); }
        if !v.uses_r() { for i in l.r1..l.rb2 + M { th[i] = 0.0; } }
        for j in 0..4 { acc[j] += parts[j]; }
        if s + 1 == win { for j in 0..4 { log.first[j] = acc[j] / win as f64; } }
        if s + 1 > steps - win { for j in 0..4 { log.last[j] += parts[j] / win as f64; } }
    }
    (th, log)
}

// ─── evaluación ──────────────────────────────────────────────────────────────
/// contexto de evaluación (= nova_train::ctx_for_eval): conocidas → verdadero; nuevas → reid con ÓRBITA
pub fn ctx_for_eval(d: &Orbita, e: &Ep, k: usize) -> (Vec<f64>, usize) {
    if e.fam < N_FAM { (context_of(e.fam, e.p), e.fam) } else { let (g, pp, _) = reid(d, &e.x0, &e.ys, k, e.p); (context_of(g, pp), g) }
}
fn s_known(d: &Orbita, cg: &[f64], e: &Ep, k: usize) -> f64 {
    let mut z = e.x0.clone(); let mut r = 0.0; for t in 0..k { z = d.step(&z, cg); r += clamp_err(&z, &e.ys[t]); } r / k as f64
}
fn err16(tr: &[Vec<f64>], e: &Ep) -> f64 { clamp_err(&tr[15], &e.ys[15]) }
/// ÓRBITA+LAT (= LAT_EP de nova2: τ_LAT(k), λ0, contexto ctx8(p̂) para el ajuste)
fn orbita_lat(d: &Orbita, e: &Ep, k: usize, cg: &[f64], gs: usize) -> (f64, bool) {
    if s_known(d, cg, e, k) <= lat_tau(k) { let mut z = e.x0.clone(); for _ in 0..16 { z = d.step(&z, cg); } return (clamp_err(&z, &e.ys[15]), false); }
    let cx = ctx8(d, e.p);
    let aa = fit_gn(&|th, z| lat_step(d, &cx.cps[gs], &cx.cs[gs], th, z), &vec![0.0; npar()], LAT_LAM0, &e.x0, &e.ys, k);
    let mut z = e.x0.clone(); for _ in 0..16 { z = lat_step(d, &cx.cps[gs], &cx.cs[gs], &aa, &z); } (clamp_err(&z, &e.ys[15]), true)
}
/// ESTELA (+E3/E4): sorpresa > τ_E(k) dispara LAT sobre ESTELA; árbitro = mejora en pasos retenidos (k_fit_val) frente a sin adaptar
pub struct EstOut { pub base: f64, pub full: f64, pub trig: bool, pub acc: bool, pub s: f64 }
fn estela_ep(th: &[f64], l: &Lay, fz: &Frozen, e: &Ep, k: usize, tau_e: f64) -> EstOut {
    let tr0 = rollout(th, l, fz, &e.x0, 16, None); let base = err16(&tr0, e);
    let s = surprise_e(th, l, fz, e, k);
    if s <= tau_e { return EstOut { base, full: base, trig: false, acc: false, s }; }
    let (kf, kv) = k_fit_val(k);
    let stp = |a: &[f64], x: &[f64]| model::step(th, l, fz, x, Some(a)).q;
    let a1 = fit_gn(&stp, &vec![0.0; model::npar_lat()], LAT_LAM0, &e.x0, &e.ys, kf);
    let tra = rollout(th, l, fz, &e.x0, k, Some(&a1));
    let ve = |tr: &[Vec<f64>]| (kf..kf + kv).map(|t| clamp_err(&tr[t], &e.ys[t])).sum::<f64>();
    if !(ve(&tra) < ve(&tr0[..k])) { return EstOut { base, full: base, trig: true, acc: false, s }; }
    let a = fit_gn(&stp, &vec![0.0; model::npar_lat()], LAT_LAM0, &e.x0, &e.ys, k);
    let tr = rollout(th, l, fz, &e.x0, 16, Some(&a));
    EstOut { base, full: err16(&tr, e), trig: true, acc: true, s }
}
fn auroc(pos: &[f64], neg: &[f64]) -> f64 {
    let mut s = 0.0; for p in pos { for n in neg { s += if p > n { 1.0 } else if p == n { 0.5 } else { 0.0 }; } } s / (pos.len() * neg.len()) as f64
}
/// rango efectivo exp(H(λ/Σλ)) de la covarianza de z (8×8) — Jacobi
pub fn erank(zs: &[[f64; M]]) -> f64 {
    let n = zs.len() as f64; let mut mu = [0.0; M]; for z in zs { for j in 0..M { mu[j] += z[j] / n; } }
    let mut a = [[0.0; M]; M]; for z in zs { for i in 0..M { for j in 0..M { a[i][j] += (z[i] - mu[i]) * (z[j] - mu[j]) / n; } } }
    for _ in 0..100 { for p in 0..M { for q in p + 1..M { if a[p][q].abs() < 1e-300 { continue; }
        let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]); let (c, s) = (th.cos(), th.sin());
        for k in 0..M { let (x, y) = (a[k][p], a[k][q]); a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
        for k in 0..M { let (x, y) = (a[p][k], a[q][k]); a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; } } } }
    let ev: Vec<f64> = (0..M).map(|i| a[i][i].max(0.0)).collect(); let t: f64 = ev.iter().sum();
    if t <= 0.0 { return 0.0; } (-ev.iter().filter(|&&x| x > 0.0).map(|x| (x / t) * (x / t).ln()).sum::<f64>()).exp()
}

/// serialización exacta: serde_json sin `float_roundtrip` NO devuelve los mismos bits al releer f64
/// (detectado por paso0_params_roundtrip); se guardan los bits u64 en hex junto al valor legible.
pub fn params_to_json(th: &[f64]) -> String { serde_json::json!({"bits": th.iter().map(|x| format!("{:016x}", x.to_bits())).collect::<Vec<_>>(), "approx": th}).to_string() }
pub fn params_from_json(s: &str) -> Vec<f64> { let v: serde_json::Value = serde_json::from_str(s).unwrap(); v["bits"].as_array().unwrap().iter().map(|b| f64::from_bits(u64::from_str_radix(b.as_str().unwrap(), 16).unwrap())).collect() }
fn arg_of(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned() }
fn parse_seed(s: &str) -> u64 { u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap() }

/// Paso 0 (CLI): por archivo de pesos, (a) ESTELA@init == ÓRBITA bit a bit en H (16 pasos, todas las familias),
/// (b) forward con cinta (entrenamiento) == forward f64 (evaluación) bit a bit con parámetros perturbados.
pub fn paso0(a: &[String]) {
    let ws: Vec<String> = a.iter().enumerate().filter(|(_, x)| *x == "--weights").map(|(i, _)| a[i + 1].clone()).collect();
    let mut res = vec![]; let mut all_ok = true;
    for w in &ws {
        let d = Orbita::from_json(&std::fs::read_to_string(w).unwrap()); let l = Lay::new(d.hg);
        let r = paso0_check(&d, &l, 0xA400, 4);
        all_ok &= r.0 == 0 && r.1 == 0; res.push(serde_json::json!({"weights": w, "init_vs_orbita_mismatch": r.0, "tape_vs_f64_mismatch": r.1, "values_compared": r.2}));
    }
    println!("{}", serde_json::json!({"paso0_ok": all_ok, "checks": res}));
    if !all_ok { std::process::exit(1); }
}
/// devuelve (#discrepancias init-vs-ÓRBITA, #discrepancias cinta-vs-f64, #valores comparados)
pub fn paso0_check(d: &Orbita, l: &Lay, seed: u64, hpf: usize) -> (usize, usize, usize) {
    let held = gen_heldout(seed, hpf); let th0 = model::init(d, l, seed);
    let (mut bad_a, mut bad_b, mut n) = (0, 0, 0);
    let mut rng = Xoshiro256StarStar::seed_from_u64(7);
    let thp: Vec<f64> = th0.iter().map(|x| x + 0.05 * (rng.gen::<f64>() - 0.5)).collect();
    for e in &held {
        let (cg, _) = ctx_for_eval(d, e, 4);
        for v in Variant::all() {
            let fz = Frozen::new(d, &cg, v);
            if v.uses_k() { // (a): sólo variantes con K (MLP sin K no es ÓRBITA por diseño)
                let tr = rollout(&th0, l, &fz, &e.x0, 16, None); let mut z = e.x0.clone();
                for t in 0..16 { z = d.step(&z, &cg); for (p, q) in tr[t].iter().zip(&z) { n += 1; if p.to_bits() != q.to_bits() { bad_a += 1; } } }
            }
            // (b): cinta vs f64 con parámetros perturbados (r_φ, h_ψ, P ≠ init)
            let tr = rollout(&thp, l, &fz, &e.x0, 4, None);
            ad::reset(); let tv: Vec<Var> = thp.iter().map(|&x| ad::leaf(x)).collect();
            let mut x: Vec<Var> = e.x0.iter().map(|&q| Var::c(q)).collect();
            for t in 0..4 { let o = model::step(&tv, l, &fz, &x, None); for (p, q) in o.q.iter().zip(&tr[t]) { n += 1; if p.v().to_bits() != q.to_bits() { bad_b += 1; } } x = o.q; }
            ad::reset();
        }
    }
    (bad_a, bad_b, n)
}

pub fn run(a: &[String]) {
    let wpath = arg_of(a, "--weights").expect("--weights"); let wtxt = std::fs::read_to_string(&wpath).unwrap();
    let d = Orbita::from_json(&wtxt); let raw0 = d.raw.clone(); let l = Lay::new(d.hg);
    let seed = parse_seed(&arg_of(a, "--seed").expect("--seed"));
    let phase = arg_of(a, "--phase").unwrap_or_else(|| "dev".into()); let smoke = phase == "smoke";
    let steps: usize = arg_of(a, "--steps").map(|s| s.parse().unwrap()).unwrap_or(STEPS);
    let hpf: usize = arg_of(a, "--h-per-fam").map(|s| s.parse().unwrap()).unwrap_or(H_PER_FAM);
    if !smoke { assert!(steps == STEPS && hpf == H_PER_FAM, "DEV/CONFIRM usan los valores congelados"); }
    let out = PathBuf::from(arg_of(a, "--out").expect("--out")).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip 0x{seed:X}"); return; }
    let before = cdt_rqm_epr::v4_provenance::snapshot(); let t0 = std::time::Instant::now();
    // Paso 0 en línea: la compuerta se repite en cada seed antes de entrenar (aborta si falla)
    let p0 = paso0_check(&d, &l, seed, 1); assert!(p0.0 == 0 && p0.1 == 0, "Paso 0 falla: {p0:?}");
    let data = gen_known(seed, XOR_TRAIN, N_TRAIN); let cal = gen_known(seed, XOR_CAL, N_CAL); let held = gen_heldout(seed, hpf);
    let ctxs: Vec<Vec<(Vec<f64>, usize)>> = held.iter().map(|e| KS.iter().map(|&k| ctx_for_eval(&d, e, k)).collect()).collect();
    let mut variants = serde_json::Map::new(); let mut arms: serde_json::Map<String, serde_json::Value> = Default::default();
    let mut timing = serde_json::Map::new();
    // brazos ÓRBITA y ÓRBITA+LAT
    let ts = std::time::Instant::now();
    for (ki, &k) in KS.iter().enumerate() {
        let (mut o, mut ol, mut sk) = (vec![], vec![], vec![]);
        for (j, e) in held.iter().enumerate() {
            let (cg, gs) = &ctxs[j][ki];
            let mut z = e.x0.clone(); for _ in 0..16 { z = d.step(&z, cg); } o.push(clamp_err(&z, &e.ys[15]));
            ol.push(orbita_lat(&d, e, k, cg, *gs).0); sk.push(s_known(&d, cg, e, k));
        }
        let kk = format!("k{k}"); let ent = arms.entry(kk).or_insert_with(|| serde_json::json!({}));
        ent["orbita"] = summarize(&held, &o, None); ent["orbita_lat"] = summarize(&held, &ol, None);
        ent["auroc_orbita"] = serde_json::json!(det_auroc(&held, &sk));
    }
    timing.insert("orbita_arms_s".into(), serde_json::json!(ts.elapsed().as_secs_f64()));
    for v in Variant::all() {
        let ts = std::time::Instant::now();
        let (th, log) = train(&d, &l, v, seed, &data, steps);
        let ttrain = ts.elapsed().as_secs_f64();
        let pp = out.join(format!("params_{}_0x{seed:X}.json", v.name()));
        std::fs::write(&pp, params_to_json(&th)).unwrap();
        // round-trip: se evalúan los parámetros releídos del disco (bits exactos, ver params_to_json)
        let th2 = params_from_json(&std::fs::read_to_string(&pp).unwrap()); assert!(th.iter().zip(&th2).all(|(a, b)| a.to_bits() == b.to_bits())); let th = th2;
        let mut tau = serde_json::Map::new(); let mut zs = vec![];
        for e in held.iter().filter(|e| e.fam < N_FAM) { let fz = Frozen::new(&d, &e.c, v); for i in 0..K_POINTS { zs.push(model::enc(&th, &l, [e.x0[2 * i], e.x0[2 * i + 1]], &fz.c)); } }
        for (ki, &k) in KS.iter().enumerate() {
            let tau_e = quantile(cal.iter().map(|e| surprise_e(&th, &l, &Frozen::new(&d, &e.c, v), e, k)).collect(), TAU_Q);
            tau.insert(format!("k{k}"), serde_json::json!(tau_e));
            let (mut base, mut full, mut trig, mut acc, mut ss) = (vec![], vec![], vec![], vec![], vec![]);
            for (j, e) in held.iter().enumerate() {
                let fz = Frozen::new(&d, &ctxs[j][ki].0, v); let r = estela_ep(&th, &l, &fz, e, k, tau_e);
                base.push(r.base); full.push(r.full); trig.push(r.trig); acc.push(r.acc); ss.push(r.s);
            }
            let ent = arms.get_mut(&format!("k{k}")).unwrap();
            ent[v.name()] = summarize(&held, &full, Some((&trig, &acc)));
            ent[format!("{}_base", v.name())] = summarize(&held, &base, None);
            ent[format!("auroc_{}", v.name())] = serde_json::json!(det_auroc(&held, &ss));
        }
        variants.insert(v.name().into(), serde_json::json!({"loss_first": log.first, "loss_last": log.last, "loss_parts": ["jepa","sigreg","probe","rec"],
            "tau_e": tau, "erank": erank(&zs), "train_s": ttrain, "eval_s": ts.elapsed().as_secs_f64() - ttrain, "tape_max": log.tape_max,
            "resid_norm": th[l.r1..l.rb2 + M].iter().skip(RH_SKIP(&l)).map(|x| x * x).sum::<f64>().sqrt(), "energy_out_norm": th[l.e2..l.e2 + model::EH].iter().map(|x| x * x).sum::<f64>().sqrt()}));
        eprintln!("0x{seed:X} {} train={ttrain:.0}s total={:.0}s", v.name(), t0.elapsed().as_secs_f64());
    }
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    let rep = serde_json::json!({"seed": format!("0x{seed:X}"), "phase": phase, "prereg": "docs/estela/PREREG.md", "steps": steps, "h_per_fam": hpf,
        "paso0": {"init_vs_orbita_mismatch": p0.0, "tape_vs_f64_mismatch": p0.1, "values": p0.2}, "arms": arms, "variants": variants,
        "timing": timing, "total_s": t0.elapsed().as_secs_f64(), "test_queries": q, "smoke": smoke});
    let mut f = std::fs::File::create(out.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    assert_eq!(Orbita::from_json(&wtxt).raw, raw0, "núcleo ÓRBITA modificado"); assert_eq!(q, 0, "consultas TEST ≠ 0");
    std::fs::write(&done, b"ok\n").unwrap();
}
#[allow(non_snake_case)]
fn RH_SKIP(l: &Lay) -> usize { l.r2 - l.r1 } // norma sólo de la capa de salida de r_φ (W2, b2)
fn summarize(held: &[Ep], err: &[f64], tr: Option<(&[bool], &[bool])>) -> serde_json::Value {
    let (mut ln, mut lk, mut an, mut ak, mut nn, mut nk) = (vec![], vec![], 0usize, 0usize, 0usize, 0usize);
    let (mut tn, mut tk, mut cn, mut ck) = (0usize, 0usize, 0usize, 0usize);
    for (j, e) in held.iter().enumerate() {
        let ok = err[j] < 0.05;
        let (t, c) = tr.map_or((false, false), |(t, c)| (t[j], c[j]));
        if PRIMARY_NEW.contains(&e.fam) { ln.push(log_rel(err[j])); nn += 1; an += ok as usize; tn += t as usize; cn += c as usize; }
        else if e.fam < N_FAM { lk.push(log_rel(err[j])); nk += 1; ak += ok as usize; tk += t as usize; ck += c as usize; }
    }
    let mut v = serde_json::json!({"new_med_log": median(ln), "new_acc16": an as f64 / nn.max(1) as f64, "known_med_log": median(lk), "known_acc16": ak as f64 / nk.max(1) as f64, "n_new": nn, "n_known": nk});
    if tr.is_some() { v["trigger_new"] = serde_json::json!(tn as f64 / nn.max(1) as f64); v["trigger_known"] = serde_json::json!(tk as f64 / nk.max(1) as f64);
        v["accept_new"] = serde_json::json!(cn as f64 / nn.max(1) as f64); v["accept_known"] = serde_json::json!(ck as f64 / nk.max(1) as f64); }
    v
}
fn det_auroc(held: &[Ep], s: &[f64]) -> f64 {
    let pos: Vec<f64> = held.iter().zip(s).filter(|(e, _)| PRIMARY_NEW.contains(&e.fam)).map(|(_, x)| *x).collect();
    let neg: Vec<f64> = held.iter().zip(s).filter(|(e, _)| e.fam < N_FAM).map(|(_, x)| *x).collect();
    auroc(&pos, &neg)
}

#[cfg(test)]
mod tests;
