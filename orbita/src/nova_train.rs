//! ÓRBITA-Nova — harness completo DEV+CONFIRM (docs/orbita_nova/PREREG.md f64b289).
//!
//!   nova-eval --calibrate-tau --wdir DIR --out DIR
//!   nova-eval --full --phase dev|confirm --weights W --seed S --tau T --out DIR
//!             [--n-stream N] [--consolidate-every N] [--smoke] [--resume]
//!
//! Flujo 2400, C6 real (N=50), brazos NOVA/NOVA_SHUF/MLP/LAT_EP/LAT/ÓRBITA(+C3)
//! + ablaciones NOVA−C3/C5/C6 + ORACLE_EP. Held-out H; τ calibrado una vez en H DEV fam 1–2.
//! No toca run_v6. Núcleo ÓRBITA congelado (campo `raw`).
#![allow(clippy::too_many_arguments)]
use crate::fam::{reid, rule};
use crate::latent::{fit_gn, lat_decide, lat_step, npar, ctx8, Entry as LatEntry};
use crate::learn::Module;
use crate::model::{Orbita, M};
use crate::nova::{
    arbitrate, gate_nodamage, lr_step, npar_lr, LibEntry, OperatorLib, DEFAULT_RANK,
};
use cdt_rqm_epr::field_autonomy_v4::K_POINTS;
use cdt_rqm_epr::v4_dataset::{context_of, make_object, ObjectClass, N_FAM};
use cdt_rqm_epr::v4_metrics as m;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Familias nuevas primarias (PREREG §1). Fam 9 excluida del primario.
pub const PRIMARY_NEW: [usize; 3] = [8, 10, 11];
pub const STREAM_N: usize = 2400;
pub const H_PER_FAM: usize = 16;
pub const CONSOLIDATE_EVERY: usize = 50;
pub const CKPTS: [usize; 6] = [0, 150, 300, 600, 1200, 2400];
pub const HORIZON: usize = 16;
pub const LAM_C4: f64 = 1e-6;
pub const EPS_SPEC: f64 = 0.5;
/// Hiperparámetros LAT congelados del ciclo latente (selected.json).
pub const LAT_LAM0: f64 = 1e-4;
pub const LAT_LAMM: f64 = 1e-3;
pub const OEP_DEG: usize = 2;
pub const OEP_LAM: f64 = 1e-4;
/// XOR documentados: flujo S vs held-out H (PREREG §2).
const XOR_STREAM: u64 = 0x4E0A_F100;
const XOR_HELD: u64 = 0x4E0A_501D;
const XOR_CALIB: u64 = 0x4E0A_CA11;

pub(crate) fn fin(z: &[f64]) -> bool {
    z.iter().all(|v| v.is_finite())
}
pub(crate) fn k_fit_val(k: usize) -> (usize, usize) {
    match k {
        2 => (1, 1),
        4 => (3, 1),
        8 => (6, 2),
        _ => ((k * 3) / 4, k - (k * 3) / 4),
    }
}

pub(crate) fn log_rel(e: f64) -> f64 {
    e.max(1e-12).ln()
}

fn rel_at(step: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], ys: &[Vec<f64>], h: usize) -> f64 {
    let mut z = x0.to_vec();
    for t in 0..h {
        z = step(&z);
        if t + 1 == h {
            return if fin(&z) {
                m::rel_err(&z, &ys[h - 1]).min(10.0)
            } else {
                10.0
            };
        }
    }
    10.0
}

pub(crate) fn med_f64(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    m::median(&v.to_vec())
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let i = ((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize;
    v[i.min(v.len() - 1)]
}

// ─── Episodios ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Ep {
    pub fam: usize,
    pub p: [f64; 2],
    pub c: Vec<f64>,
    pub x0: Vec<f64>,
    pub ys: Vec<Vec<f64>>,
    pub k: usize,
}

fn make_ep(rng: &mut Xoshiro256StarStar, fam: usize, k: usize, horizon: usize) -> Ep {
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
    let c = if fam < N_FAM {
        context_of(fam, p)
    } else {
        // nuevas: contexto one-hot g=0 + p (ÓRBITA no tiene slots 8–11)
        context_of(0, p)
    };
    let ci = rng.gen_range(0..3);
    let x0 = make_object(rng, K_POINTS, cls[ci]);
    let mut ys = Vec::with_capacity(horizon);
    let mut y = x0.clone();
    for _ in 0..horizon {
        y = rule(fam, p, &y);
        ys.push(y.clone());
    }
    Ep { fam, p, c, x0, ys, k }
}

/// Flujo S: 80 % conocidas 0–7 / 20 % primarias {8,10,11}; k ∼ U{2,4,8}.
pub fn gen_stream(seed: u64, n: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_STREAM);
    let ks = [2usize, 4, 8];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let fam = if rng.gen::<f64>() < 0.8 {
            rng.gen_range(0..N_FAM)
        } else {
            PRIMARY_NEW[rng.gen_range(0..PRIMARY_NEW.len())]
        };
        let k = ks[rng.gen_range(0..3)];
        out.push(make_ep(&mut rng, fam, k, HORIZON));
    }
    out
}

/// Held-out H: 16 ep/familia en 0–7 ∪ {8,10,11} (=176). RNG distinta del flujo.
#[allow(dead_code)]
pub fn gen_heldout(seed: u64) -> Vec<Ep> {
    gen_heldout_n(seed, H_PER_FAM)
}

pub fn gen_heldout_n(seed: u64, n_per_fam: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_HELD);
    let fams: Vec<usize> = (0..N_FAM).chain(PRIMARY_NEW.iter().copied()).collect();
    let mut out = Vec::with_capacity(fams.len() * n_per_fam);
    for &fam in &fams {
        for _ in 0..n_per_fam {
            // k_eval se elige al evaluar; guardamos k=4 como etiqueta primaria
            out.push(make_ep(&mut rng, fam, 4, HORIZON));
        }
    }
    out
}

/// H para calibración τ: sólo familias 1–2 (PREREG §5).
pub fn gen_calib_h(seed: u64, n_per: usize) -> Vec<Ep> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_CALIB);
    let mut out = Vec::with_capacity(2 * n_per);
    for fam in [1usize, 2] {
        for _ in 0..n_per {
            out.push(make_ep(&mut rng, fam, 4, HORIZON));
        }
    }
    out
}

pub(crate) fn lora_prior(seed: u64, tag: u64, rank: usize) -> Vec<f64> {
    let mut prior = vec![0.0; npar_lr(rank)];
    let mut prng = Xoshiro256StarStar::seed_from_u64(
        seed ^ tag ^ (prior.len() as u64).wrapping_mul(0x9E3779B97F4A7C15),
    );
    let nu = rank * M;
    for i in 0..nu {
        let u: f64 = (0..6).map(|_| prng.gen::<f64>()).sum::<f64>() / 6.0;
        prior[i] = (u - 0.5) * 2.0 * 3.0f64.sqrt() * 1e-3;
    }
    prior
}

/// MLP control: θ∈R^{32} llena las primeras 32 entradas de Δ ∈ R^{M×M} (sin UVᵀ).
fn mlp_step(d: &Orbita, c: &[f64], th: &[f64], x: &[f64]) -> Vec<f64> {
    let mut delta = vec![0.0; M * M];
    for (i, v) in th.iter().enumerate().take(M * M) {
        delta[i] = *v;
    }
    let cp = d.ctx(c);
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() / 2 {
        let z = d.encode(&x[2 * i..2 * i + 2], c);
        let mut u = d.core(&cp, &z);
        for r in 0..M {
            let mut s = 0.0;
            for cidx in 0..M {
                s += delta[r * M + cidx] * z[cidx];
            }
            u[r] += s;
        }
        let w = d.correct(&cp, &u);
        out[2 * i] = w[0];
        out[2 * i + 1] = w[1];
    }
    out
}

pub(crate) fn fit_c4(d: &Orbita, e: &Ep, prior: &[f64], rank: usize, k: usize) -> Vec<f64> {
    let cp = d.ctx(&e.c);
    fit_gn(
        &|th, z| lr_step(d, &cp, &e.c, th, rank, z),
        prior,
        LAM_C4,
        &e.x0,
        &e.ys,
        k,
    )
}

pub(crate) fn ctx_for_eval(d: &Orbita, e: &Ep, k: usize) -> (Vec<f64>, usize, [f64; 2]) {
    if e.fam < N_FAM {
        (context_of(e.fam, e.p), e.fam, e.p)
    } else {
        let (g, pp, _) = reid(d, &e.x0, &e.ys, k, e.p);
        (context_of(g, pp), g, pp)
    }
}

// ─── Buffer episódico + consolidación C6 ─────────────────────────────────────

#[derive(Clone)]
pub(crate) struct BufItem {
    pub signature: Vec<f64>,
    pub th: Vec<f64>,
    pub _fam: usize,
    pub known: bool,
    pub val_err: f64,
    pub base_err: f64,
}

pub(crate) fn consolidate(
    lib: &mut OperatorLib,
    buffer: &mut Vec<BufItem>,
    tau: f64,
) -> usize {
    if buffer.is_empty() {
        return 0;
    }
    // clustering greedy: asignar a centroide si dist ≤ tau; si no, nuevo cluster
    let mut clusters: Vec<Vec<usize>> = vec![];
    let mut cents: Vec<Vec<f64>> = vec![];
    for (i, b) in buffer.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (ci, c) in cents.iter().enumerate() {
            let d = OperatorLib::sig_dist(&b.signature, c);
            if d <= tau && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((ci, d));
            }
        }
        if let Some((ci, _)) = best {
            clusters[ci].push(i);
            // actualizar centroide incremental (media de firmas)
            let n = clusters[ci].len() as f64;
            for (a, b) in cents[ci].iter_mut().zip(&buffer[i].signature) {
                *a = ((*a) * (n - 1.0) + b) / n;
            }
        } else {
            clusters.push(vec![i]);
            cents.push(b.signature.clone());
        }
    }
    let mut admitted = 0usize;
    for (ci, members) in clusters.iter().enumerate() {
        // ΔK medio
        let npar = buffer[members[0]].th.len();
        let mut th = vec![0.0; npar];
        let mut damaged = false;
        for &i in members {
            for (a, b) in th.iter_mut().zip(&buffer[i].th) {
                *a += b;
            }
            if buffer[i].known && buffer[i].val_err > buffer[i].base_err + 1e-12 {
                damaged = true;
            }
        }
        let n = members.len() as f64;
        for v in th.iter_mut() {
            *v /= n;
        }
        let mean_val: f64 = members.iter().map(|&i| buffer[i].val_err).sum::<f64>() / n;
        let mean_base: f64 = members.iter().map(|&i| buffer[i].base_err).sum::<f64>() / n;
        let held_ok = mean_val < mean_base - 1e-12;
        let no_damage = !damaged;
        let entry = LibEntry {
            signature: cents[ci].clone(),
            th,
            n_support: members.len(),
        };
        if lib.admit(entry, held_ok, no_damage) {
            admitted += 1;
        }
    }
    buffer.clear();
    admitted
}

// ─── Predicción por brazo ────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Orbita,
    OrbitaC3,
    LatEp,
    Lat,
    Nova,
    NovaNoC3,
    NovaNoC5,
    NovaNoC6,
    NovaShuf,
    Mlp,
    OracleEp,
}

impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Orbita => "orbita",
            Arm::OrbitaC3 => "orbita_c3",
            Arm::LatEp => "lat_ep",
            Arm::Lat => "lat",
            Arm::Nova => "nova",
            Arm::NovaNoC3 => "nova_m_c3",
            Arm::NovaNoC5 => "nova_m_c5",
            Arm::NovaNoC6 => "nova_m_c6",
            Arm::NovaShuf => "nova_shuf",
            Arm::Mlp => "mlp",
            Arm::OracleEp => "oracle_ep",
        }
    }
    pub fn all() -> &'static [Arm] {
        &[
            Arm::Orbita,
            Arm::OrbitaC3,
            Arm::LatEp,
            Arm::Lat,
            Arm::Nova,
            Arm::NovaNoC3,
            Arm::NovaNoC5,
            Arm::NovaNoC6,
            Arm::NovaShuf,
            Arm::Mlp,
            Arm::OracleEp,
        ]
    }
}

pub(crate) fn err_h16(step: &dyn Fn(&[f64]) -> Vec<f64>, e: &Ep) -> (f64, bool) {
    let err = rel_at(step, &e.x0, &e.ys, 16);
    (err, err < 0.05)
}

/// Evalúa un brazo en un episodio held-out (biblioteca/memoria congeladas).
fn eval_arm(
    d: &Orbita,
    e: &Ep,
    k: usize,
    arm: Arm,
    lib: &OperatorLib,
    lat_lib: &[LatEntry],
    tau: f64,
    rank: usize,
    seed: u64,
) -> (f64, bool) {
    let (cg, gs, _pp) = ctx_for_eval(d, e, k);
    let (kf, kv) = k_fit_val(k);
    match arm {
        Arm::Orbita => {
            let (err, ok) = err_h16(&|z| d.step(z, &cg), e);
            (err, ok)
        }
        Arm::OrbitaC3 => {
            // C3 corrige estado en los k pasos observados; el free-run h16 usa el
            // operador base desde el estado filtrado en y_{k-1} (si k>=1).
            let mut filt = crate::nova::InnovFilter::new(1e-4);
            let mut x = e.x0.clone();
            for t in 0..k.min(e.ys.len()) {
                let (xn, _, _) = filt.update(d, &cg, &x, &e.ys[t]);
                x = xn;
            }
            let start = if k == 0 { e.x0.clone() } else { x };
            let err = {
                let mut z = start;
                for _ in k..16 {
                    z = d.step(&z, &cg);
                }
                if fin(&z) {
                    m::rel_err(&z, &e.ys[15]).min(10.0)
                } else {
                    10.0
                }
            };
            (err, err < 0.05)
        }
        Arm::LatEp => {
            let cx = ctx8(d, e.p);
            let sk = {
                let mut z = e.x0.clone();
                let mut r = 0.0;
                for t in 0..k {
                    z = d.step(&z, &cg);
                    r += if fin(&z) {
                        m::rel_err(&z, &e.ys[t]).min(10.0)
                    } else {
                        10.0
                    };
                }
                r / k as f64
            };
            if sk <= tau {
                err_h16(&|z| d.step(z, &cg), e)
            } else {
                let aa = fit_gn(
                    &|th, z| lat_step(d, &cx.cps[gs], &cx.cs[gs], th, z),
                    &vec![0.0; npar()],
                    LAT_LAM0,
                    &e.x0,
                    &e.ys,
                    k,
                );
                err_h16(&|z| lat_step(d, &cx.cps[gs], &cx.cs[gs], &aa, z), e)
            }
        }
        Arm::Lat => {
            let cx = ctx8(d, e.p);
            let sk = {
                let mut z = e.x0.clone();
                let mut r = 0.0;
                for t in 0..k {
                    z = d.step(&z, &cg);
                    r += if fin(&z) {
                        m::rel_err(&z, &e.ys[t]).min(10.0)
                    } else {
                        10.0
                    };
                }
                r / k as f64
            };
            // construir Ep de latent.rs
            let lep = crate::latent::Ep {
                idx: 0,
                tid: String::new(),
                fam: e.fam,
                ph: e.p,
                c: e.c.clone(),
                x0: e.x0.clone(),
                ys: e.ys.clone(),
            };
            let (dec, _, _) = lat_decide(d, &cx, lat_lib, &lep, k, sk, gs, tau, LAT_LAM0, LAT_LAMM);
            match dec {
                None => err_h16(&|z| d.step(z, &cg), e),
                Some((g0, aa)) => err_h16(&|z| lat_step(d, &cx.cps[g0], &cx.cs[g0], &aa, z), e),
            }
        }
        Arm::OracleEp => {
            let mut oep = Module::new(None, OEP_DEG, OEP_LAM);
            oep.add(d, &e.x0, &e.ys, k, e.p, e.fam);
            err_h16(&|z| oep.step(d, z, e.p), e)
        }
        Arm::Mlp => {
            let prior = vec![0.0; npar_lr(rank)];
            let th = fit_gn(
                &|th, z| mlp_step(d, &cg, th, z),
                &prior,
                LAM_C4,
                &e.x0,
                &e.ys,
                k,
            );
            err_h16(&|z| mlp_step(d, &cg, &th, z), e)
        }
        Arm::Nova | Arm::NovaNoC3 | Arm::NovaNoC5 | Arm::NovaNoC6 | Arm::NovaShuf => {
            let use_c5 = !matches!(arm, Arm::NovaNoC5);
            let use_lib = !matches!(arm, Arm::NovaNoC6);
            let lib_use: OperatorLib = if matches!(arm, Arm::NovaShuf) {
                lib.shuffled_signatures(seed ^ (e.fam as u64) << 16)
            } else {
                lib.clone()
            };
            // firma preliminar: C4 desde prior 0 (o vacío)
            let prior0 = lora_prior(seed, e.fam as u64, rank);
            let mut e_ctx = e.clone();
            e_ctx.c = cg.clone();
            let th0 = fit_c4(d, &e_ctx, &prior0, rank, k);
            let sig = th0.clone();
            let prior = if use_lib {
                if let Some(ent) = lib_use.retrieve(&sig, tau) {
                    ent.th.clone()
                } else {
                    prior0
                }
            } else {
                prior0
            };
            let th = fit_c4(d, &e_ctx, &prior, rank, k);
            let cp = d.ctx(&cg);
            let ver = arbitrate(d, &cp, &cg, &th, rank, &e.x0, &e.ys, kf, kv, EPS_SPEC);
            let accept = if use_c5 {
                ver.accept && gate_nodamage(ver.val_err, ver.base_val)
            } else {
                true
            };
            if accept {
                err_h16(&|z| lr_step(d, &cp, &cg, &th, rank, z), e)
            } else {
                err_h16(&|z| d.step(z, &cg), e)
            }
        }
    }
}

fn aggregate_held(
    d: &Orbita,
    held: &[Ep],
    lib: &OperatorLib,
    lat_lib: &[LatEntry],
    tau: f64,
    rank: usize,
    seed: u64,
    k_eval: usize,
    smoke: bool,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let arms: &[Arm] = if smoke {
        &[Arm::Orbita, Arm::Nova, Arm::LatEp, Arm::Mlp]
    } else {
        Arm::all()
    };
    for &arm in arms {
        let mut logs_new = vec![];
        let mut logs_known = vec![];
        let mut acc_new = 0usize;
        let mut acc_known = 0usize;
        let mut n_new = 0usize;
        let mut n_known = 0usize;
        for e in held {
            let (err, ok) = eval_arm(d, e, k_eval, arm, lib, lat_lib, tau, rank, seed);
            let lr = log_rel(err);
            if PRIMARY_NEW.contains(&e.fam) {
                logs_new.push(lr);
                n_new += 1;
                if ok {
                    acc_new += 1;
                }
            } else if e.fam < N_FAM {
                logs_known.push(lr);
                n_known += 1;
                if ok {
                    acc_known += 1;
                }
            }
        }
        out.insert(
            arm.name().into(),
            serde_json::json!({
                "new_med_log": med_f64(&mut logs_new),
                "new_acc16": if n_new > 0 { acc_new as f64 / n_new as f64 } else { f64::NAN },
                "known_med_log": med_f64(&mut logs_known),
                "known_acc16": if n_known > 0 { acc_known as f64 / n_known as f64 } else { f64::NAN },
                "n_new": n_new,
                "n_known": n_known,
            }),
        );
    }
    serde_json::Value::Object(out)
}

// ─── Paso de flujo (actualiza libs) ──────────────────────────────────────────

fn stream_step(
    d: &Orbita,
    e: &Ep,
    lib: &mut OperatorLib,
    buffer: &mut Vec<BufItem>,
    lat_lib: &mut Vec<LatEntry>,
    tau: f64,
    rank: usize,
    seed: u64,
    pos: usize,
    consolidate_every: usize,
) {
    let k = e.k;
    let (cg, gs, _) = ctx_for_eval(d, e, k);
    let (kf, kv) = k_fit_val(k);

    // —— LAT memoria ——
    {
        let cx = ctx8(d, e.p);
        let sk = {
            let mut z = e.x0.clone();
            let mut r = 0.0;
            for t in 0..k {
                z = d.step(&z, &cg);
                r += if fin(&z) {
                    m::rel_err(&z, &e.ys[t]).min(10.0)
                } else {
                    10.0
                };
            }
            r / k as f64
        };
        let lep = crate::latent::Ep {
            idx: pos,
            tid: String::new(),
            fam: e.fam,
            ph: e.p,
            c: e.c.clone(),
            x0: e.x0.clone(),
            ys: e.ys.clone(),
        };
        let (dec, _, _) = lat_decide(d, &cx, lat_lib, &lep, k, sk, gs, tau, LAT_LAM0, LAT_LAMM);
        if let Some((g0, aa)) = dec {
            lat_lib.push(LatEntry {
                g0,
                a: aa,
                fam: e.fam,
            });
        }
    }

    // —— NOVA: C4 + C5 + buffer C6 ——
    let mut e_ctx = e.clone();
    e_ctx.c = cg.clone();
    let prior0 = lora_prior(seed, (pos as u64) << 8 ^ e.fam as u64, rank);
    let th0 = fit_c4(d, &e_ctx, &prior0, rank, k);
    let sig = th0.clone();
    let prior = if let Some(ent) = lib.retrieve(&sig, tau) {
        ent.th.clone()
    } else {
        prior0
    };
    let th = fit_c4(d, &e_ctx, &prior, rank, k);
    let cp = d.ctx(&cg);
    let ver = arbitrate(d, &cp, &cg, &th, rank, &e.x0, &e.ys, kf, kv, EPS_SPEC);
    let accept = ver.accept && gate_nodamage(ver.val_err, ver.base_val);
    if accept {
        buffer.push(BufItem {
            signature: sig,
            th,
            _fam: e.fam,
            known: e.fam < N_FAM,
            val_err: ver.val_err,
            base_err: ver.base_val,
        });
    }
    if consolidate_every > 0 && (pos + 1) % consolidate_every == 0 {
        let _n = consolidate(lib, buffer, tau);
    }
}

// ─── Calibración τ ───────────────────────────────────────────────────────────

pub fn calibrate_tau(wdir: &Path, out_dir: &Path) -> f64 {
    std::fs::create_dir_all(out_dir).unwrap();
    let seeds: Vec<u64> = (0..16).map(|i| 0xA400 + i).collect();
    let rank = DEFAULT_RANK;
    let mut dists = vec![];
    for seed in seeds {
        let wpath = wdir.join(format!("weights_0x{seed:X}.json"));
        let wtxt = std::fs::read_to_string(&wpath).unwrap_or_else(|_| {
            panic!("missing {}", wpath.display());
        });
        let d = Orbita::from_json(&wtxt);
        let eps = gen_calib_h(seed, H_PER_FAM);
        // firmas por episodio
        let mut items: Vec<(usize, Vec<f64>)> = vec![];
        for (i, e) in eps.iter().enumerate() {
            let prior = lora_prior(seed, i as u64, rank);
            let th = fit_c4(&d, e, &prior, rank, 4);
            items.push((e.fam, th));
        }
        // distancia al vecino de familia incorrecta
        for (i, (fi, si)) in items.iter().enumerate() {
            let mut best = f64::INFINITY;
            for (j, (fj, sj)) in items.iter().enumerate() {
                if i == j || fi == fj {
                    continue;
                }
                let d = OperatorLib::sig_dist(si, sj);
                if d < best {
                    best = d;
                }
            }
            if best.is_finite() {
                dists.push(best);
            }
        }
        assert_eq!(Orbita::from_json(&wtxt).raw, d.raw);
    }
    let tau = percentile(dists.clone(), 95.0);
    let report = serde_json::json!({
        "tau": tau,
        "n_dists": dists.len(),
        "rule": "P95 distancia L2 firma C4 al vecino de familia incorrecta; H DEV fam 1–2; 16 seeds pooled",
        "prereg": "docs/orbita_nova/PREREG.md",
        "prereg_commit": "f64b289",
        "p50": percentile(dists.clone(), 50.0),
        "p90": percentile(dists.clone(), 90.0),
        "p95": tau,
        "p99": percentile(dists, 99.0),
    });
    let mut f = std::fs::File::create(out_dir.join("tau.json")).unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&report).unwrap()).unwrap();
    eprintln!("tau_calib={tau:.8} n={}", report["n_dists"]);
    tau
}

// ─── Seed completa ───────────────────────────────────────────────────────────

pub fn run_seed_full(
    d: &Orbita,
    seed: u64,
    tau: f64,
    n_stream: usize,
    consolidate_every: usize,
    out_dir: &Path,
    resume: bool,
    h_per_fam: usize,
    smoke: bool,
) -> serde_json::Value {
    std::fs::create_dir_all(out_dir).unwrap();
    let done = out_dir.join(format!("0x{seed:X}.done"));
    if resume && done.exists() {
        eprintln!("skip 0x{seed:X} (.done)");
        let p = out_dir.join(format!("metrics_0x{seed:X}.json"));
        if p.exists() {
            return serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        }
    }
    let rank = DEFAULT_RANK;
    let stream = gen_stream(seed, n_stream);
    let held = gen_heldout_n(seed, h_per_fam);
    let mut lib = OperatorLib::new();
    let mut buffer: Vec<BufItem> = vec![];
    let mut lat_lib: Vec<LatEntry> = vec![];
    let mut ckpt_rows = vec![];

    let ckpt_set: std::collections::HashSet<usize> = CKPTS.iter().copied().filter(|&c| c <= n_stream).collect();

    let eval_ckpt = |d: &Orbita, held: &[Ep], lib: &OperatorLib, lat_lib: &[LatEntry], n: usize, ckpt_rows: &mut Vec<serde_json::Value>| {
        let agg4 = aggregate_held(d, held, lib, lat_lib, tau, rank, seed, 4, smoke);
        let (agg2, agg8) = if smoke {
            (serde_json::Value::Null, serde_json::Value::Null)
        } else {
            (
                aggregate_held(d, held, lib, lat_lib, tau, rank, seed, 2, smoke),
                aggregate_held(d, held, lib, lat_lib, tau, rank, seed, 8, smoke),
            )
        };
        ckpt_rows.push(serde_json::json!({
            "n": n,
            "lib_size": lib.entries.len(),
            "lat_lib_size": lat_lib.len(),
            "k4": agg4,
            "k2": agg2,
            "k8": agg8,
        }));
    };

    // n=0 antes de ver nada
    if ckpt_set.contains(&0) {
        eval_ckpt(d, &held, &lib, &lat_lib, 0, &mut ckpt_rows);
    }

    for (pos, e) in stream.iter().enumerate() {
        stream_step(
            d,
            e,
            &mut lib,
            &mut buffer,
            &mut lat_lib,
            tau,
            rank,
            seed,
            pos,
            consolidate_every,
        );
        let n = pos + 1;
        if ckpt_set.contains(&n) {
            // consolidar resto del buffer antes de evaluar
            if !buffer.is_empty() {
                consolidate(&mut lib, &mut buffer, tau);
            }
            eval_ckpt(d, &held, &lib, &lat_lib, n, &mut ckpt_rows);
            // checkpoint parcial
            let partial = serde_json::json!({
                "seed": format!("0x{seed:X}"),
                "n": n,
                "lib": lib.to_json(),
            });
            let mut pf = std::fs::File::create(out_dir.join(format!("ckpt_0x{seed:X}_n{n}.json"))).unwrap();
            writeln!(pf, "{}", serde_json::to_string(&partial).unwrap()).unwrap();
            eprintln!("0x{seed:X} ckpt n={n} lib={} lat={}", lib.entries.len(), lat_lib.len());
        }
    }
    if !buffer.is_empty() {
        consolidate(&mut lib, &mut buffer, tau);
    }
    // evaluar siempre al final del flujo (cubre smoke / n_stream ∉ CKPTS)
    if !ckpt_set.contains(&n_stream) {
        eval_ckpt(d, &held, &lib, &lat_lib, n_stream, &mut ckpt_rows);
    }

    let report = serde_json::json!({
        "seed": format!("0x{seed:X}"),
        "n_stream": n_stream,
        "tau": tau,
        "rank": rank,
        "consolidate_every": consolidate_every,
        "lib_size_final": lib.entries.len(),
        "lat_lib_size_final": lat_lib.len(),
        "ckpts": ckpt_rows,
        "prereg": "f64b289",
        "primary_new": PRIMARY_NEW.to_vec(),
        "arms": Arm::all().iter().map(|a| a.name()).collect::<Vec<_>>(),
    });
    let mut f = std::fs::File::create(out_dir.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&report).unwrap()).unwrap();
    std::fs::write(&done, b"ok\n").unwrap();
    report
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

fn arg_of(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}
fn seed_of_cli(a: &[String]) -> u64 {
    let s = arg_of(a, "--seed").expect("--seed");
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).expect("seed hex")
    } else {
        s.parse().expect("seed")
    }
}

pub fn run(a: &[String]) {
    if a.iter().any(|x| x == "--calibrate-tau") {
        let wdir = PathBuf::from(arg_of(a, "--wdir").unwrap_or_else(|| "artifacts/orbita/dev".into()));
        let out = PathBuf::from(arg_of(a, "--out").unwrap_or_else(|| "artifacts/orbita/nova_full".into()));
        let _tau = calibrate_tau(&wdir, &out);
        return;
    }
    if !a.iter().any(|x| x == "--full") {
        eprintln!("nova_train: use --calibrate-tau o --full");
        return;
    }
    let wpath = arg_of(a, "--weights").expect("--weights");
    let wtxt = std::fs::read_to_string(&wpath).unwrap();
    let d = Orbita::from_json(&wtxt);
    let seed = seed_of_cli(a);
    let tau = arg_of(a, "--tau")
        .map(|s| s.parse().unwrap())
        .unwrap_or_else(|| {
            // intentar leer tau.json del --out padre
            let out = arg_of(a, "--out").unwrap_or_else(|| "artifacts/orbita/nova_full".into());
            let p = PathBuf::from(&out);
            let candidates = [p.join("tau.json"), p.parent().unwrap_or(Path::new(".")).join("tau.json")];
            for c in &candidates {
                if let Ok(t) = std::fs::read_to_string(c) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                        if let Some(tau) = v.get("tau").and_then(|x| x.as_f64()) {
                            return tau;
                        }
                    }
                }
            }
            panic!("--tau requerido (o tau.json en --out)");
        });
    let smoke = a.iter().any(|x| x == "--smoke");
    let n_stream = if smoke {
        arg_of(a, "--n-stream")
            .map(|s| s.parse().unwrap())
            .unwrap_or(12)
    } else {
        arg_of(a, "--n-stream")
            .map(|s| s.parse().unwrap())
            .unwrap_or(STREAM_N)
    };
    let h_per_fam = if smoke {
        arg_of(a, "--h-per-fam").map(|s| s.parse().unwrap()).unwrap_or(2)
    } else {
        H_PER_FAM
    };
    let consolidate_every = arg_of(a, "--consolidate-every")
        .map(|s| s.parse().unwrap())
        .unwrap_or(CONSOLIDATE_EVERY);
    let phase = arg_of(a, "--phase").unwrap_or_else(|| "dev".into());
    let out_root = PathBuf::from(arg_of(a, "--out").unwrap_or_else(|| "artifacts/orbita/nova_full".into()));
    let out_dir = out_root.join(&phase);
    let resume = a.iter().any(|x| x == "--resume");
    let before_raw = d.raw.clone();
    let rep = run_seed_full(
        &d,
        seed,
        tau,
        n_stream,
        consolidate_every,
        &out_dir,
        resume,
        h_per_fam,
        smoke,
    );
    assert_eq!(Orbita::from_json(&wtxt).raw, before_raw, "núcleo ÓRBITA modificado");
    println!("{}", serde_json::to_string_pretty(&rep).unwrap());
}
