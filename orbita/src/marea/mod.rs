//! MAREA (docs/marea/PREREG.md): JEPA latente congelado + memoria test-time (Titans/ATLAS) con regla
//! de tres factores meta-aprendida + Hopfield ± con rollback + actor recursivo (TRM) que envía c_t al predictor.
//! Todo nuevo: no usa pesos ÓRBITA ni v4. ÓRBITA+LAT sólo como línea base externa.
//! Datos: generador de familias de exp/eco (copia, orbita/src/eco/mod.rs).
use crate::eco::{clamp_err, gen_stream, log_rel, make_base, make_ep, parse_seed, Ep, HORIZON, K, N_KNOWN_STREAM, N_NEW_PER_FAM};
use crate::model::Orbita;
use cdt_rqm_epr::v4_dataset::N_FAM;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::path::PathBuf;

pub const DX: usize = 12; pub const DZ: usize = 16; pub const DC: usize = 4;
pub const HE: usize = 64; pub const HP: usize = 64; pub const HA: usize = 32; pub const HD: usize = 64;
pub const R_ACT: usize = 3;
pub const PRE_FAMS: [usize; 6] = [0, 1, 2, 3, 4, 5];
pub const META_FAMS: [usize; 2] = [6, 7];
pub const PRE_SEED: u64 = 0xD400; pub const META_SEED: u64 = 0xD401; pub const VAL_SEED: u64 = 0xD402; pub const RAND_SEED: u64 = 0xD403;
pub const PRE_STEPS: usize = 200_000; pub const PRE_BATCH: usize = 32;
pub const LR0: f64 = 1e-3; pub const LR1: f64 = 1e-5; pub const EMA: f64 = 0.996; pub const LAM_SIG: f64 = 1.0;
pub const META_GENS: usize = 40; pub const META_POP: usize = 8; pub const META_EPS: usize = 32; pub const META_SIGMA: f64 = 0.3;
pub const BLOCK: usize = 4; pub const RB_MARGIN: f64 = 0.05; pub const HOP_BETA: f64 = 8.0; pub const HOP_CAP: usize = 64;
const WN: usize = DZ * (DZ + 1);
fn mean(v: &[f64]) -> f64 { if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 } }
fn xs_of(e: &Ep) -> Vec<Vec<f64>> { let mut v = vec![e.x0.clone()]; v.extend(e.ys.iter().cloned()); v }

// ─── MLP de una capa oculta (tanh) con gradiente de entrada ──────────────────
#[derive(Clone)]
pub struct Net { pub di: usize, pub dh: usize, pub dout: usize, pub skip: bool, pub p: Vec<f64> }
impl Net {
    pub fn new(di: usize, dh: usize, dout: usize, out_scale: f64, rng: &mut Xoshiro256StarStar) -> Net { Net::new_s(di, dh, dout, out_scale, false, rng) }
    /// skip=true: y += S·x (camino lineal, S ∈ R^{dout×di}, init aleatoria 1/√di)
    pub fn new_s(di: usize, dh: usize, dout: usize, out_scale: f64, skip: bool, rng: &mut Xoshiro256StarStar) -> Net {
        let base = dh * di + dh + dout * dh + dout;
        let mut p = vec![0.0; base + if skip { dout * di } else { 0 }];
        let s1 = (1.0 / di as f64).sqrt(); let s2 = out_scale * (1.0 / dh as f64).sqrt();
        for i in 0..dh * di { p[i] = rng.gen_range(-1.0..1.0) * s1 * 1.7; }
        let o = dh * di + dh; for i in 0..dout * dh { p[o + i] = rng.gen_range(-1.0..1.0) * s2 * 1.7; }
        if skip { for i in 0..dout * di { p[base + i] = rng.gen_range(-1.0..1.0) * s1 * 1.7; } }
        Net { di, dh, dout, skip, p }
    }
    pub fn base(&self) -> usize { self.dh * self.di + self.dh + self.dout * self.dh + self.dout }
    pub fn fwd(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (di, dh, dout) = (self.di, self.dh, self.dout); let p = &self.p;
        let mut h = vec![0.0; dh];
        for j in 0..dh { let mut s = p[dh * di + j]; let r = &p[j * di..(j + 1) * di]; for i in 0..di { s += r[i] * x[i]; } h[j] = s.tanh(); }
        let o = dh * di + dh; let mut y = vec![0.0; dout];
        for k in 0..dout { let mut s = p[o + dout * dh + k]; let r = &p[o + k * dh..o + (k + 1) * dh]; for j in 0..dh { s += r[j] * h[j]; } y[k] = s; }
        if self.skip { let b = self.base(); for k in 0..dout { let r = &p[b + k * di..b + (k + 1) * di]; for i in 0..di { y[k] += r[i] * x[i]; } } }
        (h, y)
    }
    pub fn out(&self, x: &[f64]) -> Vec<f64> { self.fwd(x).1 }
    /// acumula grad de parámetros en g, devuelve dL/dx
    pub fn bwd(&self, x: &[f64], h: &[f64], dy: &[f64], g: &mut [f64]) -> Vec<f64> {
        let (di, dh, dout) = (self.di, self.dh, self.dout); let p = &self.p; let o = dh * di + dh;
        let mut dh_ = vec![0.0; dh];
        for k in 0..dout { g[o + dout * dh + k] += dy[k]; for j in 0..dh { g[o + k * dh + j] += dy[k] * h[j]; dh_[j] += dy[k] * p[o + k * dh + j]; } }
        let mut dx = vec![0.0; di];
        if self.skip { let b = self.base(); for k in 0..dout { for i in 0..di { g[b + k * di + i] += dy[k] * x[i]; dx[i] += dy[k] * p[b + k * di + i]; } } }
        for j in 0..dh { let da = dh_[j] * (1.0 - h[j] * h[j]); g[dh * di + j] += da; for i in 0..di { g[j * di + i] += da * x[i]; dx[i] += da * p[j * di + i]; } }
        dx
    }
}
pub struct Adam { m: Vec<f64>, v: Vec<f64>, t: i32 }
impl Adam {
    pub fn new(n: usize) -> Adam { Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 } }
    pub fn step(&mut self, p: &mut [f64], g: &[f64], lr: f64) {
        self.t += 1; let (b1, b2) = (0.9f64, 0.999f64); let c1 = 1.0 - b1.powi(self.t); let c2 = 1.0 - b2.powi(self.t);
        for i in 0..p.len() { let gi = g[i].clamp(-10.0, 10.0); self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi; self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi; p[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8); }
    }
}

// ─── modelo JEPA ─────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct Model { pub enc: Net, pub pred: Net, pub act: Net, pub dec: Net }
impl Model {
    pub fn new(seed: u64) -> Model {
        let mut r = Xoshiro256StarStar::seed_from_u64(seed);
        Model { enc: Net::new_s(DX, HE, DZ, 1.0, true, &mut r), pred: Net::new(DZ + DC, HP, DZ, 0.1, &mut r), act: Net::new(2 * DZ + DC, HA, DC, 0.5, &mut r), dec: Net::new_s(DZ, HD, DX, 1.0, true, &mut r) }
    }
    pub fn to_json(&self) -> serde_json::Value { serde_json::json!({"enc": self.enc.p, "pred": self.pred.p, "act": self.act.p, "dec": self.dec.p}) }
    pub fn from_json(v: &serde_json::Value) -> Model {
        let mut m = Model::new(0); let f = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>();
        m.enc.p = f("enc"); m.pred.p = f("pred"); m.act.p = f("act"); m.dec.p = f("dec"); m
    }
    pub fn hash(&self) -> String { let mut h: u64 = 0xcbf29ce484222325; for n in [&self.enc, &self.pred, &self.act, &self.dec] { for x in &n.p { h ^= x.to_bits(); h = h.wrapping_mul(0x100000001b3); } } format!("{h:016x}") }
}
/// resumen de transiciones observadas: media de [z_j, z_{j+1}−z_j], j<t
/// z_j = E(x_j/s_j), z'_j = E(x_{j+1}/s_j), s_j = rms(x_j) (envoltura equivariante a escala; la métrica es rel_err)
pub fn sc(x: &[f64]) -> f64 { (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt().max(1e-6) }
pub fn nx(x: &[f64], s: f64) -> Vec<f64> { x.iter().map(|v| v / s).collect() }
pub fn summary(z: &[Vec<f64>], zp: &[Vec<f64>]) -> Vec<f64> {
    let mut s = vec![0.0; 2 * DZ]; let n = zp.len(); if n == 0 { return s; }
    for j in 0..n { for i in 0..DZ { s[i] += z[j][i]; s[DZ + i] += zp[j][i] - z[j][i]; } }
    for v in s.iter_mut() { *v /= n as f64; } s
}
/// actor recursivo (TRM): y ← y + A([s, y]), R veces. Devuelve c y las entradas/ocultas para backprop.
pub fn actor(a: &Net, s: &[f64], r: usize) -> (Vec<f64>, Vec<(Vec<f64>, Vec<f64>)>) {
    let mut y = vec![0.0; DC]; let mut tape = vec![];
    for _ in 0..r { let mut inp = s.to_vec(); inp.extend(&y); let (h, o) = a.fwd(&inp); for i in 0..DC { y[i] += o[i]; } tape.push((inp, h)); }
    (y, tape)
}
/// observaciones hasta t (z_0..z_t, z'_0..z'_t) y código c_t del actor (sólo transiciones j<t)
pub fn obs_ctx(m: &Model, xs: &[Vec<f64>], t: usize, r: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let z: Vec<Vec<f64>> = (0..=t).map(|j| m.enc.out(&nx(&xs[j], sc(&xs[j])))).collect();
    let zp: Vec<Vec<f64>> = (0..=t).map(|j| m.enc.out(&nx(&xs[j + 1], sc(&xs[j])))).collect();
    let c = actor(&m.act, &summary(&z[..t], &zp[..t]), r).0; (z, zp, c)
}
pub fn dec_x(m: &Model, zh: &[f64], s: f64) -> Vec<f64> { m.dec.out(zh).iter().map(|v| v * s).collect() }
fn cat(a: &[f64], b: &[f64]) -> Vec<f64> { let mut v = a.to_vec(); v.extend(b); v }
/// predicción latente congelada: ẑ = z + P([z, c])
pub fn pred0(m: &Model, z: &[f64], c: &[f64]) -> Vec<f64> { let d = m.pred.out(&cat(z, c)); z.iter().zip(&d).map(|(a, b)| a + b).collect() }

// ─── preentrenamiento JEPA (EMA target + SIGReg, sonda decodificadora con stop-grad) ──
pub fn pretrain(seed: u64, steps: usize, log: &mut dyn FnMut(usize, f64, f64)) -> Model {
    let mut m = Model::new(seed ^ 0x11); let mut ema = m.enc.clone();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let mut opt = [Adam::new(m.enc.p.len()), Adam::new(m.pred.p.len()), Adam::new(m.act.p.len()), Adam::new(m.dec.p.len())];
    for st in 0..steps {
        let lr = LR1 + 0.5 * (LR0 - LR1) * (1.0 + (std::f64::consts::PI * st as f64 / steps as f64).cos());
        let mut ge = vec![0.0; m.enc.p.len()]; let mut gp = vec![0.0; m.pred.p.len()]; let mut ga = vec![0.0; m.act.p.len()]; let mut gd = vec![0.0; m.dec.p.len()];
        let mut zb = vec![]; let mut eb = vec![]; let (mut lj, mut ld) = (0.0, 0.0);
        for _ in 0..PRE_BATCH {
            let f = PRE_FAMS[rng.gen_range(0..PRE_FAMS.len())]; let e = make_ep(&mut rng, f); let xs = xs_of(&e); let t = rng.gen_range(0..HORIZON);
            let st_ = sc(&xs[t]); let xin = nx(&xs[t], st_); let (eh, z) = m.enc.fwd(&xin);
            let zt: Vec<Vec<f64>> = (0..t).map(|j| ema.out(&nx(&xs[j], sc(&xs[j])))).collect();
            let ztp: Vec<Vec<f64>> = (0..=t).map(|j| ema.out(&nx(&xs[j + 1], sc(&xs[j])))).collect();
            let s = summary(&zt, &ztp[..t]); let (c, tape) = actor(&m.act, &s, R_ACT); let tgt = &ztp[t];
            let pin = cat(&z, &c); let (ph, pd) = m.pred.fwd(&pin);
            let zh: Vec<f64> = (0..DZ).map(|i| z[i] + pd[i]).collect();
            let dzh: Vec<f64> = (0..DZ).map(|i| 2.0 * (zh[i] - tgt[i]) / DZ as f64).collect();
            lj += (0..DZ).map(|i| (zh[i] - tgt[i]).powi(2)).sum::<f64>() / DZ as f64;
            let dpin = m.pred.bwd(&pin, &ph, &dzh, &mut gp);
            let mut dz: Vec<f64> = (0..DZ).map(|i| dzh[i] + dpin[i]).collect();
            let mut dy = dpin[DZ..].to_vec();
            for (inp, h) in tape.iter().rev() { let dinp = m.act.bwd(inp, h, &dy, &mut ga); for i in 0..DC { dy[i] += dinp[2 * DZ + i]; } }
            // sonda: dec(sg(zh)) y dec(sg(enc(x_{t+1}))) → x_{t+1}
            let _ = &mut gd; ld += 0.0;
            for v in dz.iter_mut() { *v /= 1.0; }
            zb.push(z); eb.push((xin, eh, dz));
        }
        // SIGReg (momentos): ||μ||² + ||C−I||²_F / DZ sobre el batch de z
        let b = PRE_BATCH as f64; let mu: Vec<f64> = (0..DZ).map(|i| zb.iter().map(|z| z[i]).sum::<f64>() / b).collect();
        let mut cv = vec![0.0; DZ * DZ]; for z in &zb { for i in 0..DZ { for j in 0..DZ { cv[i * DZ + j] += (z[i] - mu[i]) * (z[j] - mu[j]) / b; } } }
        for i in 0..DZ { cv[i * DZ + i] -= 1.0; }
        for (k, (x, eh, dz)) in eb.iter_mut().enumerate() {
            for i in 0..DZ { let mut g = 2.0 * mu[i] / b; for j in 0..DZ { g += 4.0 / (b * DZ as f64) * cv[i * DZ + j] * (zb[k][j] - mu[j]); } dz[i] += LAM_SIG * g; }
            m.enc.bwd(x, eh, dz, &mut ge);
        }
        for g in [&mut ge, &mut gp, &mut ga, &mut gd] { for v in g.iter_mut() { *v /= b; } }
        opt[0].step(&mut m.enc.p, &ge, lr); opt[1].step(&mut m.pred.p, &gp, lr); opt[2].step(&mut m.act.p, &ga, lr); let _ = &gd; // sonda conjunta desactivada: se entrena post-hoc (train_probe)
        for (a, c) in ema.p.iter_mut().zip(&m.enc.p) { *a = EMA * *a + (1.0 - EMA) * c; }
        if st % 1000 == 0 || st + 1 == steps { log(st, lj / b, ld / (2.0 * b)); }
    }
    train_probe(ridge_probe(m, seed ^ 0x44), seed ^ 0x45, steps / 2)
}
/// «JEPA aleatorio»: enc/pred/actor aleatorios; sólo se entrena la sonda dec (mismo presupuesto/4)
pub fn pretrain_probe_only(seed: u64, steps: usize) -> Model { train_probe(ridge_probe(Model::new(seed), seed ^ 0x44), seed, steps) }
pub const RIDGE_N: usize = 20_000; pub const RIDGE_LAM: f64 = 1e-4;
/// sonda post-hoc lineal (ridge, mínimos cuadrados) de [ẑ,1] y [z_{t+1},1] → x_{t+1}; dec = camino lineal ridge + MLP con salida a cero (afinado después por train_probe)
pub fn ridge_probe(mut m: Model, seed: u64) -> Model {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed); let (mut f, mut y) = (vec![], vec![]);
    while f.len() < RIDGE_N { let fam = PRE_FAMS[rng.gen_range(0..PRE_FAMS.len())]; let e = make_ep(&mut rng, fam); let xs = xs_of(&e); let t = rng.gen_range(0..HORIZON);
        let (z, zp, c) = obs_ctx(&m, &xs, t, R_ACT); let st_ = sc(&xs[t]); let tg = nx(&xs[t + 1], st_);
        f.push(cat(&pred0(&m, &z[t], &c), &[1.0])); y.push(tg.clone()); f.push(cat(&zp[t], &[1.0])); y.push(tg); }
    let w = cdt_rqm_epr::v4_controls::ridge(&f, &y, RIDGE_LAM); let d = &mut m.dec; let b = d.base(); let o = d.dh * d.di + d.dh;
    for v in d.p[o..].iter_mut() { *v = 0.0; } // capa oculta aleatoria intacta; salida MLP a cero
    for k in 0..DX { for i in 0..DZ { d.p[b + k * DZ + i] = w[i][k]; } d.p[o + DX * d.dh + k] = w[DZ][k]; }
    m
}
/// sonda post-hoc: sólo dec, sobre el JEPA congelado
pub fn train_probe(mut m: Model, seed: u64, steps: usize) -> Model { let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x22); let mut opt = Adam::new(m.dec.p.len());
    for st in 0..steps {
        let lr = LR1 + 0.5 * (LR0 - LR1) * (1.0 + (std::f64::consts::PI * st as f64 / steps as f64).cos());
        let mut gd = vec![0.0; m.dec.p.len()];
        for _ in 0..PRE_BATCH {
            let f = PRE_FAMS[rng.gen_range(0..PRE_FAMS.len())]; let e = make_ep(&mut rng, f); let xs = xs_of(&e); let t = rng.gen_range(0..HORIZON);
            let (z, zp, c) = obs_ctx(&m, &xs, t, R_ACT); let tg = nx(&xs[t + 1], sc(&xs[t]));
            let zh = pred0(&m, &z[t], &c); let zn = zp[t].clone();
            for zz in [&zh, &zn] { let (dh, xo) = m.dec.fwd(zz); let nn = tg.iter().map(|v| v * v).sum::<f64>() + 1e-3; let dxo: Vec<f64> = (0..DX).map(|i| 2.0 * (xo[i] - tg[i]) / nn / PRE_BATCH as f64).collect(); m.dec.bwd(zz, &dh, &dxo, &mut gd); }
        }
        opt.step(&mut m.dec.p, &gd, lr);
    }
    m
}

// ─── memoria test-time + Hopfield ± ──────────────────────────────────────────
/// meta-parámetros θ = [ln η, logit β_mom, ln λ, a, b, logit ρ]
pub fn mp_decode(t: &[f64]) -> (f64, f64, f64, f64, f64, f64) { let sg = |x: f64| 1.0 / (1.0 + (-x).exp()); (t[0].exp(), sg(t[1]), t[2].exp(), t[3], t[4], sg(t[5])) }
pub const MP_INIT: [f64; 6] = [-3.0, 0.0, -4.0, 1.0, -3.0, -2.0];
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Arm { Marea, Eta0, MRand, NoRollback, NoHopfield, ShufKeys, RandJepa, K1 }
impl Arm {
    pub fn all() -> [Arm; 8] { [Arm::Marea, Arm::Eta0, Arm::MRand, Arm::NoRollback, Arm::NoHopfield, Arm::ShufKeys, Arm::RandJepa, Arm::K1] }
    pub fn name(self) -> &'static str { match self { Arm::Marea => "marea", Arm::Eta0 => "marea_eta0", Arm::MRand => "marea_mrand", Arm::NoRollback => "marea_norollback", Arm::NoHopfield => "marea_nohopfield", Arm::ShufKeys => "marea_shufkeys", Arm::RandJepa => "marea_randjepa", Arm::K1 => "marea_k1" } }
}
pub struct Mem { pub wslow: Vec<f64>, pub pats: Vec<(Vec<f64>, Vec<f64>, f64, f64)>, pub n_rb: usize, pub n_ret: usize, pub n_plus: usize }
impl Mem { pub fn new() -> Mem { Mem { wslow: vec![0.0; WN], pats: vec![], n_rb: 0, n_ret: 0, n_plus: 0 } } }
fn norm(v: &[f64]) -> Vec<f64> { let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12); v.iter().map(|x| x / n).collect() }
fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
pub struct EpOut { pub err: Vec<f64>, pub err0: Vec<f64>, pub xh: Vec<Vec<f64>> }
/// un episodio precuencial de 16 pasos: en t se predice x_{t+1} desde x_t verdadero, luego se observa x_{t+1} y se aprende.
pub fn episode(m: &Model, th: &[f64], arm: Arm, e: &Ep, mem: &mut Mem, rng: &mut Xoshiro256StarStar) -> EpOut {
    let (eta, beta, lam, a, bb, rho) = mp_decode(th);
    let eta_on = arm != Arm::Eta0; let rb = arm != Arm::NoRollback; let hop = arm != Arm::NoHopfield; let r_act = if arm == Arm::K1 { 1 } else { R_ACT };
    let xs = xs_of(e); let mut w = mem.wslow.clone(); let mut s = vec![0.0; WN]; let mut snap = w.clone();
    let (mut zs, mut zps): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (vec![], vec![]); let mut out = EpOut { err: vec![], err0: vec![], xh: vec![] }; let mut c = vec![0.0; DC];
    for t in 0..HORIZON {
        let st_ = sc(&xs[t]); zs.push(m.enc.out(&nx(&xs[t], st_))); c = actor(&m.act, &summary(&zs[..t], &zps), r_act).0; let z = zs[t].clone();
        if t == K && hop && !mem.pats.is_empty() {
            let q = norm(&c); let perm: Vec<usize> = { let mut p: Vec<usize> = (0..mem.pats.len()).collect(); if arm == Arm::ShufKeys { for i in (1..p.len()).rev() { let j = rng.gen_range(0..=i); p.swap(i, j); } } p };
            let sims: Vec<f64> = mem.pats.iter().map(|p| HOP_BETA * dot(&q, &p.0)).collect();
            let bp = (0..sims.len()).filter(|&i| mem.pats[i].2 > 0.0).map(|i| sims[i]).fold(f64::NEG_INFINITY, f64::max);
            let bn = (0..sims.len()).filter(|&i| mem.pats[i].2 < 0.0).map(|i| sims[i]).fold(f64::NEG_INFINITY, f64::max);
            if bp.is_finite() && bp >= bn {
                let ws: Vec<f64> = (0..sims.len()).map(|i| if mem.pats[i].2 > 0.0 { (sims[i] - bp).exp() } else { 0.0 }).collect(); let zsum: f64 = ws.iter().sum();
                let mut ret = vec![0.0; WN]; for i in 0..sims.len() { if ws[i] > 0.0 { let v = &mem.pats[perm[i]].1; for k in 0..WN { ret[k] += ws[i] / zsum * v[k]; } } }
                for k in 0..WN { w[k] = 0.5 * w[k] + 0.5 * ret[k]; } mem.n_ret += 1;
            }
        }
        let zh0 = pred0(m, &z, &c); let mut zh = zh0.clone();
        for i in 0..DZ { let r = &w[i * (DZ + 1)..(i + 1) * (DZ + 1)]; let mut acc = r[DZ]; for j in 0..DZ { acc += r[j] * z[j]; } zh[i] += acc; }
        let xh = dec_x(m, &zh, st_); let xh0 = dec_x(m, &zh0, st_);
        let er = clamp_err(&xh, &xs[t + 1]); let e0 = clamp_err(&xh0, &xs[t + 1]);
        out.err.push(er); out.err0.push(e0); out.xh.push(xh);
        let zn = m.enc.out(&nx(&xs[t + 1], st_)); zps.push(zn.clone());
        if eta_on {
            let mf = if arm == Arm::MRand { rng.gen_range(-1.0..1.0) } else { (a * (bb - log_rel(er))).tanh() };
            for i in 0..DZ { let ri = (zn[i] - zh[i]).clamp(-3.0, 3.0); for j in 0..=DZ { let pj = if j < DZ { z[j] } else { 1.0 }; let k = i * (DZ + 1) + j; s[k] = beta * s[k] + eta * mf * ri * pj; w[k] = (1.0 - lam) * w[k] + s[k]; } }
            if (t + 1) % BLOCK == 0 {
                if rb { let lb = mean(&out.err[t + 1 - BLOCK..].iter().map(|&x| log_rel(x)).collect::<Vec<_>>()); let l0 = mean(&out.err0[t + 1 - BLOCK..].iter().map(|&x| log_rel(x)).collect::<Vec<_>>());
                    if lb > l0 + RB_MARGIN { w = snap.clone(); s.iter_mut().for_each(|v| *v = 0.0); mem.n_rb += 1; } }
                snap = w.clone();
            }
        }
    }
    let gain = mean(&out.err0.iter().map(|&x| log_rel(x)).collect::<Vec<_>>()) - mean(&out.err.iter().map(|&x| log_rel(x)).collect::<Vec<_>>());
    if eta_on {
        if hop { mem.pats.push((norm(&c), w.clone(), if gain > 0.0 { 1.0 } else { -1.0 }, gain.abs()));
            if mem.pats.len() > HOP_CAP { let (i, _) = mem.pats.iter().enumerate().min_by(|a, b| a.1 .3.total_cmp(&b.1 .3)).unwrap(); mem.pats.remove(i); } }
        if gain > 0.0 { mem.n_plus += 1; for k in 0..WN { mem.wslow[k] += rho * (w[k] - mem.wslow[k]); } }
    }
    out
}
/// pendiente OLS de ln err_t vs t
pub fn slope(err: &[f64]) -> f64 { let n = err.len() as f64; let tm = (n - 1.0) / 2.0; let y: Vec<f64> = err.iter().map(|&e| log_rel(e)).collect(); let ym = mean(&y);
    let (mut sxy, mut sxx) = (0.0, 0.0); for (t, v) in y.iter().enumerate() { sxy += (t as f64 - tm) * (v - ym); sxx += (t as f64 - tm).powi(2); } sxy / sxx }
pub fn meta_objective(m: &Model, th: &[f64], seed: u64) -> f64 {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed); let mut mem = Mem::new(); let mut tot = 0.0;
    for i in 0..META_EPS { let e = make_ep(&mut rng, META_FAMS[i % 2]); let o = episode(m, th, Arm::Marea, &e, &mut mem, &mut rng); tot += mean(&o.err.iter().map(|&x| log_rel(x)).collect::<Vec<_>>()); }
    tot / META_EPS as f64
}
/// ES (1+λ) sobre θ en familias 6–7 (no vistas en el preentrenamiento JEPA), semillas META ≠ DEV/CONFIRM
pub fn meta_learn(m: &Model, log: &mut dyn FnMut(usize, f64)) -> (Vec<f64>, f64, f64) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(META_SEED ^ 0x33); let mut best = MP_INIT.to_vec();
    let f0 = meta_objective(m, &best, META_SEED); let mut fb = f0;
    for g in 0..META_GENS {
        for _ in 0..META_POP { let cand: Vec<f64> = best.iter().map(|x| x + META_SIGMA * (rng.gen_range(-1.0f64..1.0) + rng.gen_range(-1.0..1.0) + rng.gen_range(-1.0..1.0))).collect();
            let fc = meta_objective(m, &cand, META_SEED); if fc < fb { fb = fc; best = cand; } }
        log(g, fb);
    }
    (best, f0, fb)
}

// ─── Paso 0 ──────────────────────────────────────────────────────────────────
fn jacobi_eig(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    for _ in 0..100 { let mut off = 0.0; for p in 0..n { for q in p + 1..n { off += a[p * n + q].powi(2); } } if off < 1e-20 { break; }
        for p in 0..n { for q in p + 1..n { let apq = a[p * n + q]; if apq.abs() < 1e-15 { continue; }
            let th = 0.5 * (2.0 * apq).atan2(a[q * n + q] - a[p * n + p]); let (c, s) = (th.cos(), th.sin());
            for k in 0..n { let (akp, akq) = (a[k * n + p], a[k * n + q]); a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq; }
            for k in 0..n { let (apk, aqk) = (a[p * n + k], a[q * n + k]); a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk; } } } }
    (0..n).map(|i| a[i * n + i]).collect()
}
/// 0a: sin colapso (erank ≥ DZ/2, std mín ≥ 0.05) y el predictor congelado bate a la copia x_{t+1}=x_t en familias 0–5 de validación
pub fn paso0a(m: &Model) -> serde_json::Value {
    let mut rng = Xoshiro256StarStar::seed_from_u64(VAL_SEED); let mut zs = vec![]; let (mut lp, mut lc, mut lr) = (vec![], vec![], vec![]);
    for i in 0..192 { let e = make_ep(&mut rng, PRE_FAMS[i % 6]); let xs = xs_of(&e); let (z, zp, _) = obs_ctx(m, &xs, HORIZON - 1, R_ACT);
        for t in 0..HORIZON { let c = actor(&m.act, &summary(&z[..t], &zp[..t]), R_ACT).0; let s_ = sc(&xs[t]); let xh = dec_x(m, &pred0(m, &z[t], &c), s_); lp.push(log_rel(clamp_err(&xh, &xs[t + 1]))); lc.push(log_rel(clamp_err(&xs[t], &xs[t + 1]))); lr.push(log_rel(clamp_err(&dec_x(m, &zp[t], s_), &xs[t + 1]))); zs.push(z[t].clone()); } }
    let n = zs.len() as f64; let mu: Vec<f64> = (0..DZ).map(|i| zs.iter().map(|z| z[i]).sum::<f64>() / n).collect();
    let mut cv = vec![0.0; DZ * DZ]; for z in &zs { for i in 0..DZ { for j in 0..DZ { cv[i * DZ + j] += (z[i] - mu[i]) * (z[j] - mu[j]) / n; } } }
    let sd_min = (0..DZ).map(|i| cv[i * DZ + i].sqrt()).fold(f64::INFINITY, f64::min);
    let ev: Vec<f64> = jacobi_eig(cv, DZ).into_iter().map(|x| x.max(0.0)).collect(); let tr: f64 = ev.iter().sum();
    let erank = (-ev.iter().filter(|&&x| x > 0.0).map(|&x| { let p = x / tr; p * p.ln() }).sum::<f64>()).exp();
    let (mp, mc) = (mean(&lp), mean(&lc));
    serde_json::json!({"erank": erank, "sd_min": sd_min, "eig": ev, "val_mean_log_err_pred": mp, "val_mean_log_err_copy": mc, "val_mean_log_err_recon": mean(&lr), "pass": erank >= DZ as f64 / 2.0 && sd_min >= 0.05 && mp < mc})
}
/// 0b: η=0 ≡ predictor congelado, bit a bit (todo el flujo de un seed)
pub fn frozen_xh(m: &Model, e: &Ep) -> Vec<Vec<f64>> { let xs = xs_of(e); let (z, zp, _) = obs_ctx(m, &xs, HORIZON - 1, R_ACT);
    (0..HORIZON).map(|t| { let c = actor(&m.act, &summary(&z[..t], &zp[..t]), R_ACT).0; dec_x(m, &pred0(m, &z[t], &c), sc(&xs[t])) }).collect() }
pub fn paso0b(m: &Model, th: &[f64], seed: u64) -> (usize, usize) {
    let st = gen_stream(seed, N_NEW_PER_FAM, N_KNOWN_STREAM); let mut mem = Mem::new(); let mut rng = Xoshiro256StarStar::seed_from_u64(seed); let (mut bad, mut n) = (0, 0);
    for e in &st { let o = episode(m, th, Arm::Eta0, e, &mut mem, &mut rng); let f = frozen_xh(m, e); for (a, b) in o.xh.iter().zip(&f) { for (x, y) in a.iter().zip(b) { n += 1; if x.to_bits() != y.to_bits() { bad += 1; } } } }
    (bad, n)
}

// ─── CLI ─────────────────────────────────────────────────────────────────────
fn arg_of(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned() }
fn git_sha() -> String { std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default() }
/// marea-pretrain --out DIR [--steps N]: JEPA + sonda + JEPA aleatorio + meta-aprendizaje → DIR/model.json
pub fn pretrain_cli(a: &[String]) {
    let out = PathBuf::from(arg_of(a, "--out").expect("--out")); std::fs::create_dir_all(&out).unwrap();
    let steps: usize = arg_of(a, "--steps").map(|s| s.parse().unwrap()).unwrap_or(PRE_STEPS);
    let t0 = std::time::Instant::now(); let mut lg = std::fs::File::create(out.join("pretrain_log.txt")).unwrap();
    use std::io::Write;
    let m = pretrain(PRE_SEED, steps, &mut |s, lj, ld| { let _ = writeln!(lg, "step {s} jepa {lj:.5} probe {ld:.5} t={:.0}s", t0.elapsed().as_secs_f64()); let _ = lg.flush(); });
    let t_pre = t0.elapsed().as_secs_f64();
    let rj = pretrain_probe_only(RAND_SEED, steps / 2); let t_rand = t0.elapsed().as_secs_f64() - t_pre;
    let mut lm = std::fs::File::create(out.join("meta_log.txt")).unwrap();
    let (th, f0, fb) = meta_learn(&m, &mut |g, f| { let _ = writeln!(lm, "gen {g} best {f:.5}"); let _ = lm.flush(); });
    let t_meta = t0.elapsed().as_secs_f64() - t_pre - t_rand;
    let p0a = paso0a(&m); let p0a_rand = paso0a(&rj);
    let j = serde_json::json!({"model": m.to_json(), "rand_model": rj.to_json(), "theta": th, "theta_decoded": format!("{:?}", mp_decode(&th)), "meta_obj_init": f0, "meta_obj_best": fb,
        "steps": steps, "batch": PRE_BATCH, "hash": m.hash(), "rand_hash": rj.hash(), "paso0a": p0a, "paso0a_randjepa": p0a_rand, "t_pretrain_s": t_pre, "t_rand_s": t_rand, "t_meta_s": t_meta, "git_sha": git_sha()});
    std::fs::write(out.join("model.json"), serde_json::to_string(&j).unwrap()).unwrap();
    println!("{}", serde_json::json!({"hash": m.hash(), "paso0a": j["paso0a"], "theta": j["theta_decoded"], "meta": [f0, fb], "t": [t_pre, t_rand, t_meta]}));
}
pub struct Loaded { pub m: Model, pub r: Model, pub th: Vec<f64> }
pub fn load(p: &str) -> Loaded { let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    Loaded { m: Model::from_json(&v["model"]), r: Model::from_json(&v["rand_model"]), th: v["theta"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() } }
pub fn run_seed(l: &Loaded, d: &Orbita, seed: u64) -> serde_json::Value {
    let t0 = std::time::Instant::now(); let st = gen_stream(seed, N_NEW_PER_FAM, N_KNOWN_STREAM);
    let newi: Vec<usize> = (0..st.len()).filter(|&i| st[i].fam >= N_FAM).collect(); let knwi: Vec<usize> = (0..st.len()).filter(|&i| st[i].fam < N_FAM).collect();
    let mut arms = serde_json::Map::new(); let mut parity = (0usize, 0usize);
    for arm in Arm::all() {
        let m = if arm == Arm::RandJepa { &l.r } else { &l.m }; let mut mem = Mem::new(); let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ 0x3A2E_A000);
        let outs: Vec<EpOut> = st.iter().map(|e| episode(m, &l.th, arm, e, &mut mem, &mut rng)).collect();
        if arm == Arm::Eta0 { for (o, e) in outs.iter().zip(&st) { for (a, b) in o.xh.iter().zip(&frozen_xh(m, e)) { for (x, y) in a.iter().zip(b) { parity.1 += 1; if x.to_bits() != y.to_bits() { parity.0 += 1; } } } } }
        let sl = |ix: &[usize]| mean(&ix.iter().map(|&i| slope(&outs[i].err)).collect::<Vec<_>>());
        let ml = |ix: &[usize], t0: usize| mean(&ix.iter().map(|&i| mean(&outs[i].err[t0..].iter().map(|&x| log_rel(x)).collect::<Vec<_>>())).collect::<Vec<_>>());
        let curve: Vec<f64> = (0..HORIZON).map(|t| mean(&newi.iter().map(|&i| log_rel(outs[i].err[t])).collect::<Vec<_>>())).collect();
        arms.insert(arm.name().into(), serde_json::json!({"slope_new": sl(&newi), "slope_known": sl(&knwi), "mlog_new": ml(&newi, 0), "mlog_known": ml(&knwi, 0), "mlog_new_post": ml(&newi, K), "mlog_known_post": ml(&knwi, K),
            "curve_new": curve, "n_rollback": mem.n_rb, "n_retrieve": mem.n_ret, "n_plus": mem.n_plus, "n_pats": mem.pats.len()}));
    }
    // línea base externa ÓRBITA+LAT / ÓRBITA: 1 paso desde x_t verdadero, t ≥ K (LAT ajustado en los K primeros)
    let (mut ol, mut oo, mut olk, mut ook) = (vec![], vec![], vec![], vec![]);
    for (i, e) in st.iter().enumerate() { let b = make_base(d, e.fam, e.p, &e.x0, &e.ys, K, None); let xs = xs_of(e);
        let a = mean(&(K..HORIZON).map(|t| log_rel(clamp_err(&b.step(d, &xs[t]), &xs[t + 1]))).collect::<Vec<_>>());
        let o = mean(&(K..HORIZON).map(|t| log_rel(clamp_err(&d.step(&xs[t], &b.c), &xs[t + 1]))).collect::<Vec<_>>());
        if e.fam >= N_FAM { ol.push(a); oo.push(o); } else { olk.push(a); ook.push(o); } let _ = i; }
    arms.insert("orbita_lat".into(), serde_json::json!({"mlog_new_post": mean(&ol), "mlog_known_post": mean(&olk)}));
    arms.insert("orbita".into(), serde_json::json!({"mlog_new_post": mean(&oo), "mlog_known_post": mean(&ook)}));
    serde_json::json!({"seed": format!("0x{seed:X}"), "arms": arms, "parity_eta0_vs_frozen": {"mismatch": parity.0, "values": parity.1}, "n_new": newi.len(), "n_known": knwi.len(), "wall_s": t0.elapsed().as_secs_f64()})
}
/// marea-run --phase smoke|dev|confirm --model M --weights W(ÓRBITA, sólo línea base) --seed S --out DIR
pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").expect("--seed")); let phase = arg_of(a, "--phase").expect("--phase");
    if phase == "dev" { assert!((0xA400..=0xA40F).contains(&seed)); } else if phase == "confirm" { assert!((0xB400..=0xB40F).contains(&seed)); } else { assert!(phase == "smoke" && !(0xA400..=0xA40F).contains(&seed) && !(0xB400..=0xB40F).contains(&seed)); }
    let out = PathBuf::from(arg_of(a, "--out").expect("--out")); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip"); return; }
    let mp = arg_of(a, "--model").expect("--model"); let l = load(&mp); let h0 = l.m.hash();
    let wp = arg_of(a, "--weights").expect("--weights"); let d = Orbita::from_json(&std::fs::read_to_string(&wp).unwrap());
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let mut r = run_seed(&l, &d, seed);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    assert_eq!(l.m.hash(), h0, "JEPA modificado"); assert_eq!(q, 0, "consultas TEST ≠ 0"); assert_eq!(r["parity_eta0_vs_frozen"]["mismatch"], 0, "Paso 0b falla en línea");
    r["test_queries"] = serde_json::json!(q); r["phase"] = serde_json::json!(phase); r["model_hash"] = serde_json::json!(h0); r["weights"] = serde_json::json!(wp); r["git_sha"] = serde_json::json!(git_sha());
    std::fs::write(out.join(format!("metrics_0x{seed:X}.json")), serde_json::to_string(&r).unwrap()).unwrap(); std::fs::write(&done, "").unwrap();
    println!("0x{seed:X} ok wall={:.1}s slope_new marea={:.5} eta0={:.5}", r["wall_s"].as_f64().unwrap(), r["arms"]["marea"]["slope_new"].as_f64().unwrap(), r["arms"]["marea_eta0"]["slope_new"].as_f64().unwrap());
}
/// marea-paso0 --model M --seed 0xC400: 0a (del model.json, recalculado), 0b paridad, 0c consultas TEST
pub fn paso0(a: &[String]) {
    let l = load(&arg_of(a, "--model").expect("--model")); let seed = parse_seed(&arg_of(a, "--seed").unwrap_or("0xC400".into()));
    assert!(!(0xA400..=0xA40F).contains(&seed) && !(0xB400..=0xB40F).contains(&seed));
    let before = cdt_rqm_epr::v4_provenance::snapshot();
    let p0a = paso0a(&l.m); let (bad, n) = paso0b(&l.m, &l.th, seed); let (badr, nr) = paso0b(&l.r, &l.th, seed);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total();
    let ok = p0a["pass"] == true && bad == 0 && badr == 0 && q == 0;
    println!("{}", serde_json::json!({"0a": p0a, "0b": {"mismatch": bad, "values": n, "randjepa_mismatch": badr, "randjepa_values": nr, "pass": bad == 0 && badr == 0}, "0c": {"test_queries": q, "pass": q == 0}, "paso0_ok": ok, "data_seed": format!("0x{seed:X}"), "git_sha": git_sha()}));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn grad_check() { let mut r = Xoshiro256StarStar::seed_from_u64(1); let n = Net::new(5, 7, 3, 1.0, &mut r); let x = vec![0.3, -0.2, 0.5, 0.1, -0.7];
        let l = |n: &Net, x: &[f64]| n.out(x).iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v).sum::<f64>();
        let (h, _) = n.fwd(&x); let mut g = vec![0.0; n.p.len()]; let dx = n.bwd(&x, &h, &[1.0, 2.0, 3.0], &mut g);
        for k in [0, 10, 40, n.p.len() - 1] { let mut n2 = n.clone(); n2.p[k] += 1e-6; assert!(((l(&n2, &x) - l(&n, &x)) / 1e-6 - g[k]).abs() < 1e-4); }
        for i in 0..5 { let mut x2 = x.clone(); x2[i] += 1e-6; assert!(((l(&n, &x2) - l(&n, &x)) / 1e-6 - dx[i]).abs() < 1e-4); } }
    #[test] fn eta0_parity_small() { let m = pretrain(0xD9, 200, &mut |_, _, _| {}); let (bad, n) = paso0b(&m, &MP_INIT, 0xC401); assert!(n > 0); assert_eq!(bad, 0); }
    #[test] fn slope_sign() { let e: Vec<f64> = (0..16).map(|t| (-0.1 * t as f64).exp()).collect(); assert!((slope(&e) + 0.1).abs() < 1e-9); }
}
