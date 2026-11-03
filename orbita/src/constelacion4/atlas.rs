//! ATLAS-FW (docs/atlas_fw/PREREG.md): aprendiz en contexto sobre la red v4 entrenada (sin tocar su entrenamiento).
//! Cada mecanismo es un interruptor en `Mech`; con todos apagados la predicción es v4 bit a bit (verificado en `atlas-paso0` y en tests).
//!  M1 biblioteca de generadores Koopman/SINDy: ψ = [φ_v4 ; s·g(x)] con g = polinomios ≤3, Fourier ω=2, tanh(2·), r·x
//!  M2 estimador disperso iterado (ridge + umbral secuencial STLSQ, N_IT pasos) sobre todas las transiciones de contexto, en lugar de NLMS
//!  M3 vector de contexto c_e (ajuste afín Δx=Ax+b, ⟨r⟩, ⟨|Δx|⟩, 1) + hiperred lineal Θ0 = H·c_e meta-entrenada
//!  M4 meta-entrenamiento con familias sintéticas procedurales (composiciones de primitivas), nunca {8,10,11}, con auditoría de fuga
//!  M5 objetivo de rollout reinyectado h(K+1)..h16 en el meta-entrenamiento (en vez de 1 paso)
//!  M6 compuerta CLS: si el residuo precuencial en contexto del camino v4 ≤ el de ATLAS, se usa la predicción v4 (sólo cuadros 0..K)
//!   orbita atlas-h0 --seeds 0xA400..0xA40F [--v4 DIR]   (cota oráculo de identificabilidad; sin red)
//!   orbita atlas-paso0 --seed 0xC4A0 [--na N --nb N]
//!   orbita atlas-run --phase dev|confirm|smoke --seed S --weights W --out DIR [--arms v4,M1,..] [--na --nb sólo smoke]
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mech { pub m1: bool, pub m2: bool, pub m3: bool, pub m4: bool, pub m5: bool, pub m6: bool, pub m7: bool, pub m8: bool, pub da: u32 }
impl Mech {
    pub fn from_list(s: &str) -> Mech { let mut n = Mech::default(); for t in s.split(',').filter(|t| !t.is_empty()) { match t { "M1" => n.m1 = true, "M2" => n.m2 = true, "M3" => n.m3 = true, "M4" => n.m4 = true, "M5" => n.m5 = true, "M6" => n.m6 = true, "M7" => n.m7 = true, "M8" => n.m8 = true, _ => panic!("mecanismo {t}") } } n }
    pub fn all() -> Mech { Mech { m1: true, m2: true, m3: true, m4: true, m5: true, m6: true, ..Mech::default() } }
    /// Bio-Quantum v10 portado: M1+M2+M6 (núcleo ATLAS sin M3–M5) + M7 DA-STR (factor da) [+ M8 cota Brenier]
    pub fn bq(da: u32, m8: bool, m7: bool) -> Mech { let mut m = Mech::from_list("M1,M2,M6"); m.m7 = m7; m.m8 = m8; m.da = if m7 { da } else { 0 }; m }
    /// ablación leave-one-out (docs/atlas_fw/PREREG_ABLACIONES.md)
    pub fn without(mut self, i: usize) -> Mech { match i { 1 => self.m1 = false, 2 => self.m2 = false, 3 => self.m3 = false, 4 => self.m4 = false, 5 => self.m5 = false, 6 => self.m6 = false, _ => panic!("mecanismo {i}") } self }
    pub fn off(&self) -> bool { *self == Mech::default() }
    /// hay prior meta-aprendido Θ0 (M3: contextual; M4/M5 solos: Θ̄ constante)
    pub fn meta(&self) -> bool { self.m3 || self.m4 || self.m5 }
}
pub fn arms() -> Vec<(&'static str, Mech)> { vec![("v4", Mech::default()), ("M1", Mech::from_list("M1")), ("M2", Mech::from_list("M2")), ("M3", Mech::from_list("M3")),
    ("M4", Mech::from_list("M4")), ("M5", Mech::from_list("M5")), ("M6", Mech::from_list("M6")), ("ALL", Mech::all()),
    ("ALL-M1", Mech::all().without(1)), ("ALL-M2", Mech::all().without(2)), ("ALL-M3", Mech::all().without(3)),
    ("ALL-M4", Mech::all().without(4)), ("ALL-M5", Mech::all().without(5)), ("ALL-M6", Mech::all().without(6)),
    ("BQ3", Mech::bq(3, false, true)), ("BQ10", Mech::bq(10, false, true)), ("BQ30", Mech::bq(30, false, true)),
    ("BQR3", Mech::bq(3, true, true)), ("BQR10", Mech::bq(10, true, true)), ("BQR30", Mech::bq(30, true, true)), ("BQ-M7R", Mech::bq(0, true, false)), ("CORE", Mech::bq(0, false, false))] }

// ─── constantes congeladas (PREREG) ─────────────────────────────────────────
pub const N_LIB: usize = 17; pub const LIB_S: f64 = 0.3;
pub const N_IT: usize = 3; pub const LAMS: [f64; 3] = [1e-3, 1e-2, 1e-1]; pub const THRS: [f64; 2] = [0.0, 0.05]; pub const ALPHAS: [f64; 3] = [0.0, 0.5, 1.0];
pub const LAM_STAR: f64 = 1e-3; pub const LAM_H: f64 = 1e-2; pub const DC: usize = 9;
pub const N_FIT: usize = 192; pub const N_VAL: usize = 96;
pub const AUDIT_TOL: f64 = 0.05;
/// M7 DA-STR (v10): grado de cada función de biblioteca; penalización ridge λ_j = λ·da^(grado−1) vía reescalado de columnas
pub const LIB_DEG: [i32; N_LIB] = [1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2];
/// M8 cota Brenier-Lyapunov (v10): r_b = RB_K·max r(contexto)·g^(h−K), g = (r_K/r_0)^(1/K) si >1.03; compresión suave tanh
pub const RB_K: f64 = 2.5;
fn colscale(m: &Mech, dp: usize, d: usize) -> Vec<f64> { (0..d).map(|j| if !m.m7 || j < dp { 1.0 } else { (m.da as f64).powf(-((LIB_DEG[j - dp] - 1) as f64) / 2.0) }).collect() }
fn brenier(p: &mut [f64], rb: f64) { for q in 0..p.len() / 2 { let r = (p[2 * q].powi(2) + p[2 * q + 1].powi(2)).sqrt(); if r > rb && r.is_finite() { let sc = rb * (1.0 + ((r - rb) / rb).tanh()) / r; p[2 * q] *= sc; p[2 * q + 1] *= sc; } } }
fn rmax(f: &[f64]) -> f64 { (0..f.len() / 2).map(|q| (f[2 * q].powi(2) + f[2 * q + 1].powi(2)).sqrt()).fold(0.0, f64::max) }
const XOR_META: u64 = 0xA71A_5F00; const XOR_SYN: u64 = 0xA71A_5A11;
pub fn seeds(phase: &str) -> std::ops::RangeInclusive<u64> { match phase { "dev" => 0xA480..=0xA48F, "confirm" => 0xB480..=0xB48F, "smoke" => 0xC4A0..=0xC4AF, "abl_dev" => 0xA490..=0xA49F, "abl_confirm" => 0xB490..=0xB49F, "bq_tune" => 0xC4B0..=0xC4B3, "bq_dev" => 0xA400..=0xA40F, "bq_confirm" => 0xB400..=0xB40F, _ => panic!("fase") } }

/// g(x): biblioteca genérica (no contiene las fórmulas de 8/10/11)
pub fn lib(x: &[f64]) -> [f64; N_LIB] { let (a, b) = (x[0], x[1]); let r = (a * a + b * b).sqrt();
    [a, b, a * a, a * b, b * b, a * a * a, a * a * b, a * b * b, b * b * b, (2.0 * a).sin(), (2.0 * b).sin(), (2.0 * a).cos() - 1.0, (2.0 * b).cos() - 1.0, (2.0 * a).tanh(), (2.0 * b).tanh(), r * a, r * b] }
fn psi_of(m: &Mech, phi: &[f64], xl: &[f64]) -> Vec<f64> { let mut v = phi.to_vec(); if m.m1 { v.extend(lib(xl).iter().map(|z| z * LIB_S)); } v }
/// x̂(t+1) = x(t) + Σ_j (W^P_j + Θ_j) ψ_j   (W^P = 0 en la parte de biblioteca; mismo orden de suma que v4 `pred_from`)
fn pred(wp: &[f64], dp: usize, xl: &[f64], psi: &[f64], th: &[f64]) -> Vec<f64> { let d = psi.len();
    (0..2).map(|i| xl[i] + (0..d).map(|j| (if j < dp { wp[i * dp + j] } else { 0.0 } + th[i * d + j]) * psi[j]).sum::<f64>()).collect() }
/// c_e: ajuste afín Δx = A x + b (ridge 1e-3) sobre las transiciones vistas, ⟨r⟩, ⟨|Δx|⟩, 1
pub fn ctx(frames: &[Vec<f64>]) -> Vec<f64> {
    let np = frames[0].len() / 2; let mut c = vec![0.0; DC]; c[DC - 1] = 1.0;
    let rs: Vec<f64> = frames.iter().flat_map(|f| (0..np).map(move |p| (f[2 * p].powi(2) + f[2 * p + 1].powi(2)).sqrt())).collect(); c[6] = mean(&rs);
    if frames.len() < 2 { return c; }
    let (mut px, mut py, mut dn) = (vec![], vec![], vec![]);
    for t in 1..frames.len() { for p in 0..np { let (x, y) = (&frames[t - 1], &frames[t]); px.push(vec![x[2 * p], x[2 * p + 1], 1.0]); let d = [y[2 * p] - x[2 * p], y[2 * p + 1] - x[2 * p + 1]]; dn.push((d[0] * d[0] + d[1] * d[1]).sqrt()); py.push(d.to_vec()); } }
    let w = ridge(&px, &py, 1e-3); for i in 0..3 { for k in 0..2 { c[i * 2 + k] = w[i][k]; } } c[7] = mean(&dn); c
}

// ─── modelo meta-aprendido ──────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct Model { pub m: Mech, pub d: usize, pub h: Option<Vec<Vec<f64>>>, pub tbar: Option<Vec<f64>>, pub lam: f64, pub thr: f64, pub alpha: f64 }
impl Model {
    pub fn plain(m: Mech, d: usize) -> Model { Model { m, d, h: None, tbar: None, lam: LAMS[0], thr: THRS[0], alpha: 0.0 } }
    fn prior(&self, frames: &[Vec<f64>]) -> Option<Vec<f64>> {
        if self.alpha == 0.0 { return None; }
        if let Some(h) = &self.h { let c = ctx(frames); return Some(apply(h, &c).iter().map(|v| v * self.alpha).collect()); }
        self.tbar.as_ref().map(|t| t.iter().map(|v| v * self.alpha).collect())
    }
}
/// ridge + STLSQ: w (2·d, fila i = salida i)
fn sparse_fit(x: &[Vec<f64>], y: &[Vec<f64>], lam: f64, thr: f64) -> Vec<f64> {
    let d = x[0].len(); let w = ridge(x, y, lam); let mut out = vec![0.0; 2 * d]; for j in 0..d { for i in 0..2 { out[i * d + j] = w[j][i]; } }
    if thr <= 0.0 { return out; }
    let sc: Vec<f64> = (0..d).map(|j| (x.iter().map(|r| r[j] * r[j]).sum::<f64>() / x.len() as f64).sqrt()).collect();
    for i in 0..2 { let mut sup: Vec<usize> = (0..d).collect();
        for _ in 0..N_IT { let a: Vec<f64> = sup.iter().map(|&j| out[i * d + j].abs() * sc[j]).collect(); let mx = a.iter().cloned().fold(0.0, f64::max); if mx <= 0.0 { break; }
            let ns: Vec<usize> = sup.iter().zip(&a).filter(|(_, &v)| v >= thr * mx).map(|(&j, _)| j).collect(); if ns.is_empty() { break; }
            let xr: Vec<Vec<f64>> = x.iter().map(|r| ns.iter().map(|&j| r[j]).collect()).collect(); let yr: Vec<Vec<f64>> = y.iter().map(|r| vec![r[i]]).collect();
            let wr = ridge(&xr, &yr, lam); for j in 0..d { out[i * d + j] = 0.0; } for (q, &j) in ns.iter().enumerate() { out[i * d + j] = wr[q][0]; } let done = ns.len() == sup.len(); sup = ns; if done { break; } } }
    out
}
/// estado en línea del aprendiz rápido de un episodio
struct Online<'a> { md: &'a Model, wp: &'a [f64], dp: usize, eta_f: f64, t_f: usize, th: Vec<f64>, delta: Vec<f64>, xs: Vec<Vec<f64>>, ys: Vec<Vec<f64>>, frames: Vec<Vec<f64>> }
impl<'a> Online<'a> {
    fn new(md: &'a Model, net: &'a Net, x0: &[f64]) -> Online<'a> { let d = md.d; let mut o = Online { md, wp: &net.wp, dp: net.dp, eta_f: net.c.eta_f, t_f: net.c.t_f, th: vec![0.0; 2 * d], delta: vec![0.0; 2 * d], xs: vec![], ys: vec![], frames: vec![x0.to_vec()] }; o.refresh(); o }
    fn refresh(&mut self) { let pr = self.md.prior(&self.frames);
        if self.md.m.m2 { let d = self.md.d; let w = if self.xs.is_empty() { vec![0.0; 2 * d] } else {
                let yr: Vec<Vec<f64>> = match &pr { Some(p) => self.xs.iter().zip(&self.ys).map(|(x, y)| (0..2).map(|i| y[i] - (0..d).map(|j| p[i * d + j] * x[j]).sum::<f64>()).collect()).collect(), None => self.ys.clone() };
                if self.md.m.m7 { let sc = colscale(&self.md.m, self.dp, d); let xs2: Vec<Vec<f64>> = self.xs.iter().map(|x| x.iter().zip(&sc).map(|(a, b)| a * b).collect()).collect();
                    let mut w = sparse_fit(&xs2, &yr, self.md.lam, self.md.thr); for i in 0..2 { for j in 0..d { w[i * d + j] *= sc[j]; } } w } else { sparse_fit(&self.xs, &yr, self.md.lam, self.md.thr) } };
            self.th = match &pr { Some(p) => p.iter().zip(&w).map(|(a, b)| a + b).collect(), None => w };
        } else { self.th = match &pr { Some(p) => p.iter().zip(&self.delta).map(|(a, b)| a + b).collect(), None => self.delta.clone() }; } }
    /// observa x(t) dados (x(t−1), ψ(t−1)) de los puntos ya iniciados
    fn observe(&mut self, obs: &[(Vec<f64>, Vec<f64>)], x: &[f64], pts: &[usize]) {
        self.frames.push(x.to_vec()); if obs.is_empty() { self.refresh(); return; }
        let d = self.md.d;
        if self.md.m.m2 { for (q, (xl, psi)) in obs.iter().enumerate() { let p = pts[q]; let base = pred(self.wp, self.dp, xl, psi, &vec![0.0; 2 * d]); self.xs.push(psi.clone()); self.ys.push(vec![x[2 * p] - base[0], x[2 * p + 1] - base[1]]); } self.refresh(); return; }
        // NLMS de v4 (mismo orden aritmético que head_update), sobre ψ y alrededor del prior
        let nrm: f64 = obs.iter().map(|(_, psi)| psi.iter().map(|v| v * v).sum::<f64>()).sum::<f64>() + 1e-9;
        let pr = self.md.prior(&self.frames[..self.frames.len() - 1]);
        for _ in 0..self.t_f { let mut g = vec![0.0; 2 * d];
            let th: Vec<f64> = match &pr { Some(p) => p.iter().zip(&self.delta).map(|(a, b)| a + b).collect(), None => self.delta.clone() };
            for (q, (xl, psi)) in obs.iter().enumerate() { let p = pts[q]; let xh = pred(self.wp, self.dp, xl, psi, &th); for i in 0..2 { let e = x[2 * p + i] - xh[i]; for j in 0..d { g[i * d + j] += e * psi[j]; } } }
            for k in 0..2 * d { self.delta[k] += self.eta_f * g[k] / nrm; } }
        self.refresh();
    }
}
pub struct AOut { pub preds: Vec<Vec<f64>>, pub ctx_res: f64, pub nan: bool }
/// episodio de evaluación ATLAS (sin plasticidad; la red v4 sólo infiere estados). Con Mech::default() ≡ run_episode(eval) de v4 bit a bit.
pub fn run_ep(net0: &Net, md: &Model, e: &Ep, k_obs: usize) -> AOut {
    let mut net = net0.clone(); net.c.fast = false; net.begin_episode(); let np = e.x0.len() / 2;
    let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect(); let mut preds = vec![vec![]; HORIZON];
    let mut on = Online::new(md, net0, &e.x0); let mut res = vec![];
    let predict = |net: &Net, sts: &[PState], th: &[f64]| -> Vec<f64> { sts.iter().flat_map(|s| if !s.started { vec![0.0; 2] } else { let phi = net.feat(s); pred(&net.wp, net.dp, &s.xl, &psi_of(&md.m, &phi, &s.xl), th) }).collect() };
    for t in 0..=k_obs {
        if t >= 1 { let p = predict(&net, &sts, &on.th); res.push(log_rel(clamp_err(&p, e.frame(t)))); preds[t - 1] = p; }
        let pts: Vec<usize> = (0..np).filter(|&p| sts[p].started).collect();
        let obs: Vec<(Vec<f64>, Vec<f64>)> = pts.iter().map(|&p| { let s = &sts[p]; let phi = net.feat(s); (s.xl.clone(), psi_of(&md.m, &phi, &s.xl)) }).collect();
        constelacion_step(&mut net, &mut sts, e.frame(t), false, 0.0, false);
        if net.nan { break; }
        if t >= 1 { on.observe(&obs, e.frame(t), &pts); }
        if t == k_obs { let mut s2 = sts.clone(); let nan0 = net.nan;
            let (r0, rk) = (rmax(e.frame(0)), rmax(e.frame(k_obs))); let rb0 = RB_K * (0..=k_obs).map(|t| rmax(e.frame(t))).fold(0.0, f64::max); let ratio = (rk + 1e-6) / (r0 + 1e-6); let g = if ratio > 1.03 { ratio.powf(1.0 / k_obs as f64) } else { 1.0 };
            for hh in (k_obs + 1)..=HORIZON { let mut p = predict(&net, &s2, &on.th); if md.m.m8 { brenier(&mut p, rb0 * g.powi((hh - k_obs) as i32)); } preds[hh - 1] = p.clone(); if !fin(&p) { break; } step_ex(&mut net, &mut s2, &p, false, 0.0, false, true); }
            net.nan = nan0; if !fin(&preds[HORIZON - 1]) || preds[HORIZON - 1].is_empty() { preds[HORIZON - 1] = vec![f64::NAN; 2 * np]; } }
    }
    AOut { preds, ctx_res: mean(&res), nan: net.nan }
}
/// predicción final del brazo (M6: compuerta CLS frente a v4 con el residuo precuencial de los cuadros 1..K)
pub fn arm_preds(net: &Net, md: &Model, e: &Ep) -> (Vec<Vec<f64>>, bool, bool) {
    if md.m.off() { let mut n = net.clone(); let o = run_episode(&mut n, e, false, 0.0, false, K); return (o.preds, false, n.nan); }
    let a = run_ep(net, md, e, K);
    if md.m.m6 { let mut n = net.clone(); let o = run_episode(&mut n, e, false, 0.0, false, K);
        let rv = mean(&(1..=K).map(|t| log_rel(clamp_err(&o.preds[t - 1], e.frame(t)))).collect::<Vec<_>>());
        if !(rv > a.ctx_res) { return (o.preds, true, n.nan); } }
    (a.preds, false, a.nan)
}
pub fn eval_held_atlas(net: &Net, md: &Model, held: &[Ep]) -> (serde_json::Value, serde_json::Value, bool) {
    let mut r = serde_json::Map::new(); let mut g = serde_json::Map::new(); let mut nan = false;
    for (name, known) in [("known", true), ("new", false)] {
        let (mut errs, mut nv4) = (vec![], 0usize);
        for e in held.iter().filter(|e| (e.fam < N_FAM) == known) { let (p, used, nn) = arm_preds(net, md, e); nan |= nn; nv4 += used as usize; errs.push(clamp_err(&p[HORIZON - 1], &e.ys[HORIZON - 1])); }
        if errs.is_empty() { continue; }
        r.insert(format!("{name}_acc16"), json!(errs.iter().filter(|&&x| x < mt::REL_ERR_OK).count() as f64 / errs.len() as f64));
        r.insert(format!("{name}_med_log"), json!(median(errs.iter().map(|&x| log_rel(x)).collect())));
        g.insert(format!("{name}_frac_v4"), json!(nv4 as f64 / errs.len() as f64));
    }
    (serde_json::Value::Object(r), serde_json::Value::Object(g), nan)
}
pub fn e33_atlas(net: &Net, md: &Model, d: &Orbita, ex: &[Example]) -> (serde_json::Value, serde_json::Value) {
    let trs: Vec<(Vec<Vec<f64>>, &Example)> = ex.iter().map(|e| { let ys: Vec<Vec<f64>> = (1..=HORIZON).map(|h| target_h(e, h)).collect();
        let ep = Ep { fam: e.family_id, p: [0.0; 2], x0: e.input.clone(), ys }; (arm_preds(net, md, &ep).0, e) }).collect();
    let raw = e33_score(&trs);
    if !net.c.orb_prior { return (raw.clone(), raw); }
    let trs2: Vec<(Vec<Vec<f64>>, &Example)> = trs.iter().map(|(p, e)| { let mut p = p.clone(); p[0] = d.step(&e.input, &e.context); (p, *e) }).collect();
    (e33_score(&trs2), raw)
}

// ─── M4: familias sintéticas procedurales ───────────────────────────────────
#[derive(Clone, Debug)]
pub enum Prim { Rot(f64), Scale(f64), Shear(bool, f64), SinShear(bool, f64, f64), TanhShear(bool, f64, f64), Radial(f64, f64, f64), Twist(f64, f64), Quad(bool, f64) }
pub fn prim_apply(pr: &Prim, x: f64, y: f64) -> (f64, f64) { match *pr {
    Prim::Rot(a) => rot(a, x, y), Prim::Scale(s) => (s.exp() * x, s.exp() * y),
    Prim::Shear(o, c) => if o { (x + c * y, y) } else { (x, y + c * x) },
    Prim::SinShear(o, c, w) => if o { (x + c * (w * y).sin(), y) } else { (x, y + c * (w * x).sin()) },
    Prim::TanhShear(o, c, w) => if o { (x + c * (w * y).tanh(), y) } else { (x, y + c * (w * x).tanh()) },
    Prim::Radial(c, n, ph) => { let (r, th) = ((x * x + y * y).sqrt(), y.atan2(x)); let r2 = r * (1.0 + c * (n * th + ph).cos()); (r2 * th.cos(), r2 * th.sin()) }
    Prim::Twist(a, b) => { let r = (x * x + y * y).sqrt(); rot(a + b * (r - 0.7), x, y) }
    Prim::Quad(o, c) => if o { (x + c * y * y, y) } else { (x, y + c * x * x) } } }
pub fn syn_rule(t: &[Prim], o: &[f64]) -> Vec<f64> { let mut v = o.to_vec(); for i in 0..o.len() / 2 { let (mut a, mut b) = (o[2 * i], o[2 * i + 1]); for p in t { let (u, w) = prim_apply(p, a, b); a = u; b = w; } v[2 * i] = a; v[2 * i + 1] = b; } v }
fn sample_prim(rng: &mut Xoshiro256StarStar) -> Prim { let o = rng.gen::<bool>(); let ws = [1.0, 2.0, 3.0]; let w = ws[rng.gen_range(0..3)];
    match rng.gen_range(0..8) { 0 => Prim::Rot(rng.gen_range(-0.8..0.8)), 1 => Prim::Scale(rng.gen_range(-0.15..0.15)), 2 => Prim::Shear(o, rng.gen_range(-0.4..0.4)),
        3 => Prim::SinShear(o, rng.gen_range(-0.3..0.3), w), 4 => Prim::TanhShear(o, rng.gen_range(-0.3..0.3), w),
        5 => { let ns = [1.0, 2.0, 4.0, 5.0]; Prim::Radial(rng.gen_range(-0.2..0.2), ns[rng.gen_range(0..4)], rng.gen_range(-3.14..3.14)) } // n=3 excluido
        6 => Prim::Twist(rng.gen_range(-0.5..0.5), rng.gen_range(-0.6..0.6)), _ => Prim::Quad(o, rng.gen_range(-0.3..0.3)) } }
/// auditoría de fuga: ¿el mapa de 1 paso está a rel_err < AUDIT_TOL de alguna familia retenida {8,10,11} en una rejilla 9×9 de p?
pub fn leaks(t: &[Prim], probe: &[f64]) -> bool { let y = syn_rule(t, probe);
    for &f in &NEW_FAMS { for a in 0..9 { for b in 0..9 { let p = [-1.0 + 0.25 * a as f64, -1.0 + 0.25 * b as f64]; if clamp_err(&y, &rule(f, p, probe)) < AUDIT_TOL { return true; } } } } false }
pub fn make_syn_ep(rng: &mut Xoshiro256StarStar, probe: &[f64], rej: &mut usize) -> (Ep, Vec<Prim>) {
    let cls = [ObjectClass::Blob, ObjectClass::Circle, ObjectClass::Triangle];
    loop { let n = rng.gen_range(1..=3); let t: Vec<Prim> = (0..n).map(|_| sample_prim(rng)).collect();
        if leaks(&t, probe) { *rej += 1; continue; }
        let ci = rng.gen_range(0..3); let x0 = make_object(rng, K_POINTS, cls[ci]); let mut ys = vec![]; let mut y = x0.clone();
        for _ in 0..HORIZON { y = syn_rule(&t, &y); ys.push(y.clone()); }
        let n0 = x0.iter().map(|v| v * v).sum::<f64>().sqrt(); let nl = ys[HORIZON - 1].iter().map(|v| v * v).sum::<f64>().sqrt();
        if !fin(&ys[HORIZON - 1]) || nl > 10.0 * n0.max(1e-9) || nl < 0.1 * n0 { *rej += 1; continue; }
        return (Ep { fam: 100, p: [0.0; 2], x0, ys }, t); }
}
pub fn meta_eps(seed: u64, m: &Mech, n: usize, xor: u64) -> (Vec<Ep>, usize) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_META ^ xor); let mut prng = Xoshiro256StarStar::seed_from_u64(seed ^ XOR_SYN);
    let probe = make_object(&mut prng, K_POINTS, ObjectClass::Blob); let mut rej = 0;
    let eps = (0..n).map(|i| if m.m4 && i % 2 == 1 { make_syn_ep(&mut rng, &probe, &mut rej).0 } else { make_ep(&mut rng, i % N_FAM) }).collect();
    for e in &eps { let e: &Ep = e; assert!(!NEW_FAMS.contains(&e.fam), "fuga: familia retenida en meta-entrenamiento"); }
    (eps, rej)
}

// ─── meta-entrenamiento ─────────────────────────────────────────────────────
struct Cache { obs: Vec<Vec<(Vec<f64>, Vec<f64>)>>, pts: Vec<Vec<usize>>, sts_k: Vec<PState>, ep: Ep }
fn cache_ep(net0: &Net, m: &Mech, e: &Ep) -> Cache { let mut net = net0.clone(); net.c.fast = false; net.begin_episode(); let np = e.x0.len() / 2;
    let mut sts: Vec<PState> = (0..np).map(|_| net.fresh_state()).collect(); let (mut obs, mut pts) = (vec![], vec![]); let mut sk = vec![];
    for t in 0..=HORIZON { let pt: Vec<usize> = (0..np).filter(|&p| sts[p].started).collect();
        obs.push(pt.iter().map(|&p| { let s = &sts[p]; let phi = net.feat(s); (s.xl.clone(), psi_of(m, &phi, &s.xl)) }).collect()); pts.push(pt);
        constelacion_step(&mut net, &mut sts, e.frame(t), false, 0.0, false); if t == K { sk = sts.clone(); } }
    Cache { obs, pts, sts_k: sk, ep: e.clone() } }
fn theta_at_k(net: &Net, md: &Model, c: &Cache) -> Vec<f64> { let mut on = Online::new(md, net, &c.ep.x0); on.frames.clear(); on.frames.push(c.ep.x0.clone()); on.refresh();
    for t in 1..=K { on.observe(&c.obs[t], c.ep.frame(t), &c.pts[t]); } on.th }
fn objective(net0: &Net, md: &Model, c: &Cache) -> f64 { let th = theta_at_k(net0, md, c); let wp = &net0.wp; let dp = net0.dp;
    let pr = |net: &Net, s: &[PState]| -> Vec<f64> { s.iter().flat_map(|s| { let phi = net.feat(s); pred(wp, dp, &s.xl, &psi_of(&md.m, &phi, &s.xl), &th) }).collect() };
    if !md.m.m5 { let p = pr(net0, &c.sts_k); return log_rel(clamp_err(&p, c.ep.frame(K + 1))); }
    let mut net = net0.clone(); net.c.fast = false; let mut s2 = c.sts_k.clone(); let mut v = vec![];
    for hh in (K + 1)..=HORIZON { let p = pr(&net, &s2); v.push(log_rel(clamp_err(&p, c.ep.frame(hh)))); if !fin(&p) { for _ in hh..HORIZON { v.push(log_rel(10.0)); } break; } step_ex(&mut net, &mut s2, &p, false, 0.0, false, true); }
    mean(&v) }
pub fn meta_train(net: &Net, m: Mech, seed: u64) -> (Model, serde_json::Value) {
    let d = net.dp + if m.m1 { N_LIB } else { 0 }; let mut md = Model::plain(m, d);
    if !(m.m2 || m.meta()) { return (md, json!({"meta": false})); }
    let (fit, rej1) = meta_eps(seed, &m, N_FIT, 0x1); let (val, rej2) = meta_eps(seed, &m, N_VAL, 0x2);
    let cf: Vec<Cache> = if m.meta() { fit.iter().map(|e| cache_ep(net, &m, e)).collect() } else { vec![] }; let cv: Vec<Cache> = val.iter().map(|e| cache_ep(net, &m, e)).collect();
    if m.meta() { // Θ*_e por episodio (ridge sobre residuos de transiciones 1..=K+1, o 1..=16 con M5) → H (M3) o Θ̄
        let tmax = if m.m5 { HORIZON } else { K + 1 }; let zero = vec![0.0; 2 * d]; let (mut cs, mut ts) = (vec![], vec![]);
        for c in &cf { let (mut xs, mut ys) = (vec![], vec![]);
            for t in 1..=tmax { for (q, (xl, psi)) in c.obs[t].iter().enumerate() { let p = c.pts[t][q]; let b = pred(&net.wp, net.dp, xl, psi, &zero); let x = c.ep.frame(t); xs.push(psi.clone()); ys.push(vec![x[2 * p] - b[0], x[2 * p + 1] - b[1]]); } }
            let w = ridge(&xs, &ys, LAM_STAR); let mut th = vec![0.0; 2 * d]; for j in 0..d { for i in 0..2 { th[i * d + j] = w[j][i]; } }
            cs.push(ctx(&(0..=K).map(|t| c.ep.frame(t).to_vec()).collect::<Vec<_>>())); ts.push(th); }
        if m.m3 { md.h = Some(ridge(&cs, &ts, LAM_H)); } else { md.tbar = Some((0..2 * d).map(|k| ts.iter().map(|t| t[k]).sum::<f64>() / ts.len() as f64).collect()); }
    }
    let lams: Vec<f64> = if m.m2 { LAMS.to_vec() } else { vec![LAMS[0]] }; let thrs: Vec<f64> = if m.m2 { THRS.to_vec() } else { vec![THRS[0]] }; let als: Vec<f64> = if m.meta() { ALPHAS.to_vec() } else { vec![0.0] };
    let mut best = (f64::INFINITY, 0.0, 0.0, 0.0); let mut grid = vec![];
    for &a in &als { for &l in &lams { for &th in &thrs { let mut mm = md.clone(); mm.alpha = a; mm.lam = l; mm.thr = th;
        let o = mean(&cv.iter().map(|c| objective(net, &mm, c)).collect::<Vec<_>>()); grid.push(json!([a, l, th, o])); if o < best.0 { best = (o, a, l, th); } } } }
    md.alpha = best.1; md.lam = best.2; md.thr = best.3; let (al, la, tr) = (md.alpha, md.lam, md.thr);
    (md, json!({"meta": true, "d": d, "alpha": al, "lam": la, "thr": tr, "objective": if m.m5 { "rollout_h5_h16" } else { "one_step_h5" }, "best_obj": best.0, "grid": grid, "syn_rejected": rej1 + rej2, "n_fit": N_FIT, "n_val": N_VAL}))
}

// ─── H0: cota oráculo de identificabilidad (sin red) ────────────────────────
fn ctx_loss(f: usize, p: [f64; 2], e: &Ep) -> f64 { let mut s = 0.0; for t in 0..K { let y = rule(f, p, e.frame(t)); for (a, b) in y.iter().zip(e.frame(t + 1)) { s += (a - b).powi(2); } } s }
/// p̂ = argmin_p Σ_{t<K} ‖f_p(x_t) − x_{t+1}‖² (rejilla 41×41 + búsqueda por patrones hasta paso 1e-12, p ∈ [−1,1]²)
pub fn fit_p(f: usize, e: &Ep) -> ([f64; 2], f64) { let mut best = ([0.0, 0.0], f64::INFINITY);
    for a in 0..41 { for b in 0..41 { let p = [-1.0 + 0.05 * a as f64, -1.0 + 0.05 * b as f64]; let l = ctx_loss(f, p, e); if l < best.1 { best = (p, l); } } }
    let mut st = 0.05; while st > 1e-12 { let mut imp = false; for (da, db) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] { let p = [(best.0[0] + da * st).clamp(-1.0, 1.0), (best.0[1] + db * st).clamp(-1.0, 1.0)]; let l = ctx_loss(f, p, e); if l < best.1 { best = (p, l); imp = true; } } if !imp { st *= 0.5; } }
    best }
fn roll(f: usize, p: [f64; 2], e: &Ep) -> f64 { let mut z = e.frame(K).to_vec(); for _ in K..HORIZON { z = rule(f, p, &z); } clamp_err(&z, &e.ys[HORIZON - 1]) }
pub fn h0(a: &[String]) {
    let s = arg_of(a, "--seeds").unwrap_or_else(|| "0xA400..0xA40F".into()); let (lo, hi) = s.split_once("..").unwrap(); let (lo, hi) = (parse_seed(lo), parse_seed(hi));
    let v4dir = arg_of(a, "--v4"); let before = cdt_rqm_epr::v4_provenance::snapshot(); let mut rows = vec![];
    for seed in lo..=hi { let held: Vec<Ep> = gen_heldout(seed, HPF).into_iter().filter(|e| e.fam >= N_FAM).collect();
        let (mut true_f, mut any_f, mut libo, mut ptrue, mut fam_ok) = (vec![], vec![], vec![], vec![], 0usize);
        for e in &held { let (p, _) = fit_p(e.fam, e); true_f.push(log_rel(roll(e.fam, p, e))); ptrue.push(log_rel(roll(e.fam, e.p, e)));
            let cands: Vec<usize> = (0..N_FAM).chain(NEW_FAMS.iter().copied()).collect(); let (bf, bp, _) = cands.iter().map(|&f| { let (p, l) = fit_p(f, e); (f, p, l) }).min_by(|x, y| x.2.total_cmp(&y.2)).unwrap();
            fam_ok += (bf == e.fam) as usize; any_f.push(log_rel(roll(bf, bp, e)));
            // biblioteca genérica M1 (ridge sobre g(x) con k=4 transiciones, residual Δx), λ=1e-3
            let np = e.x0.len() / 2; let (mut xs, mut ys) = (vec![], vec![]); for t in 0..K { let x = e.frame(t); let y = e.frame(t + 1); for p in 0..np { let mut g = lib(&x[2 * p..2 * p + 2]).to_vec(); g.push(1.0); xs.push(g); ys.push(vec![y[2 * p] - x[2 * p], y[2 * p + 1] - x[2 * p + 1]]); } }
            let w = ridge(&xs, &ys, 1e-3); let mut z = e.frame(K).to_vec(); for _ in K..HORIZON { let mut n = z.clone(); for p in 0..np { let mut g = lib(&z[2 * p..2 * p + 2]).to_vec(); g.push(1.0); let dd = apply(&w, &g); n[2 * p] += dd[0]; n[2 * p + 1] += dd[1]; } z = n; }
            libo.push(log_rel(clamp_err(&z, &e.ys[HORIZON - 1]))); }
        let v4 = v4dir.as_ref().and_then(|d| std::fs::read_to_string(format!("{d}/metrics_0x{seed:X}.json")).ok()).map(|t| { let v: serde_json::Value = serde_json::from_str(&t).unwrap(); v["arms"]["cst"]["held"]["new_med_log"].as_f64().unwrap() });
        rows.push(json!({"seed": format!("0x{seed:X}"), "n_new": held.len(), "oracle_true_family_med_log": median(true_f.clone()), "oracle_family_search_med_log": median(any_f.clone()), "family_search_correct": fam_ok,
            "oracle_true_params_med_log": median(ptrue.clone()), "oracle_generic_library_med_log": median(libo.clone()), "v4_new_med_log": v4})); }
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(); assert_eq!(q, 0);
    println!("{}", json!({"h0": rows, "test_queries": q, "git_sha": git_sha()}));
}

// ─── protocolo ──────────────────────────────────────────────────────────────
fn weights_hash(net: &Net) -> String { format!("{:016x}", net.hash()) }
pub fn run(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap()); let phase = arg_of(a, "--phase").unwrap(); assert!(seeds(&phase).contains(&seed), "semilla fuera de rango de la fase");
    let allowed = ["--seed", "--phase", "--weights", "--out", "--arms", "--na", "--nb"];
    for (i, x) in a.iter().enumerate() { if x.starts_with("--") { assert!(allowed.contains(&x.as_str()), "flag no permitido {x}"); if phase != "smoke" { assert!(x != "--na" && x != "--nb", "DEV/CONFIRM: presupuesto congelado"); } } else { assert!(i <= 1 || a[i - 1].starts_with("--"), "argumento suelto {x}"); } }
    let wpath = arg_of(a, "--weights").unwrap(); let d = Orbita::from_json(&std::fs::read_to_string(&wpath).unwrap()); let raw0 = d.raw.clone();
    let out = PathBuf::from(arg_of(a, "--out").unwrap()).join(&phase); std::fs::create_dir_all(&out).unwrap();
    let done = out.join(format!("0x{seed:X}.done")); if done.exists() { eprintln!("skip"); return; }
    let only: Option<Vec<String>> = arg_of(a, "--arms").map(|s| s.split(',').map(String::from).collect());
    if let Some(o) = &only { for x in o { assert!(arms().iter().any(|(n, _)| n == x), "brazo desconocido {x}"); } assert!(o.iter().any(|x| x == "v4"), "v4 siempre corre (contraste pareado)"); }
    let t0 = std::time::Instant::now(); let before = cdt_rqm_epr::v4_provenance::snapshot();
    let c = Cfg::v4(); let held = gen_heldout(seed, HPF); let held_known: Vec<Ep> = held.iter().filter(|e| e.fam < N_FAM).cloned().collect();
    let na: usize = if phase == "smoke" { arg_of(a, "--na").map(|v| v.parse().unwrap()).unwrap_or(N_A) } else { N_A };
    let nb: usize = if phase == "smoke" { arg_of(a, "--nb").map(|v| v.parse().unwrap()).unwrap_or(N_NEW) } else { N_NEW };
    let sa = gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()); let sb = gen_stream(seed, nb, nb);
    let (ex, sha) = e33_examples(seed);
    // entrenamiento v4 idéntico a `cst4-run` brazo cst (una vez por semilla; todos los brazos lo reutilizan)
    let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
    let (_, h16a, _) = run_stream(&mut net, &mut cdt, &sa, seed); let pre = eval_held(&mut net, &held_known);
    let (_, h16b, fb) = run_stream(&mut net, &mut cdt, &sb, seed);
    let train_s = t0.elapsed().as_secs_f64(); eprintln!("0x{seed:X} train {train_s:.1}s");
    let col = collapse(&mut net.clone(), &held); let hash = weights_hash(&net);
    let mut armsj = serde_json::Map::new();
    for (name, m) in arms() {
        if let Some(o) = &only { if !o.iter().any(|x| x == name) { continue; } }
        let ta = std::time::Instant::now();
        let (held_j, gate_j, nan_e, e33, e33raw, meta) = if m.off() { let mut n = net.clone(); let h = eval_held(&mut n, &held); let (e, r) = e33_net2(&mut n, &d, &ex); (h, json!({}), n.nan, e, r, json!({"meta": false})) }
            else { let (md, meta) = meta_train(&net, m, seed); let (h, g, nn) = eval_held_atlas(&net, &md, &held); let (e, r) = e33_atlas(&net, &md, &d, &ex); (h, g, nn, e, r, meta) };
        armsj.insert(name.into(), json!({"mech": format!("{m:?}"), "pre_known": pre, "held": held_j, "gate_cls": gate_j, "collapse": col, "e33": e33, "e33_v3head": e33raw, "nan": net.nan || nan_e, "meta": meta, "wall_s": ta.elapsed().as_secs_f64()}));
        eprintln!("0x{seed:X} {name} {:.1}s", ta.elapsed().as_secs_f64());
    }
    let orb = orbita_held(&d, &held); let e33o = e33_orbita(&d, &ex);
    let q = cdt_rqm_epr::v4_provenance::snapshot().delta(&before).total(); assert_eq!(d.raw, raw0); assert_eq!(q, 0, "consultas TEST ≠ 0");
    let r = json!({"seed": format!("0x{seed:X}"), "phase": phase, "weights": wpath, "orbita": orb, "e33_orbita": e33o, "arms": armsj, "test_queries": q, "test_sha256": sha, "git_sha": git_sha(),
        "v4_train": {"hash": hash, "eta_scale": net.eta_scale, "sleeps": cdt.log.len(), "streamA_last_q_h16": mean(&h16a[h16a.len() * 3 / 4..]), "auc_new_preq": mean(&h16b.iter().zip(&fb).filter(|(_, &f)| f >= N_FAM).map(|(v, _)| *v).collect::<Vec<_>>()), "train_s": train_s, "nan": net.nan},
        "wall_s": t0.elapsed().as_secs_f64()});
    let tmp = out.join(format!("metrics_0x{seed:X}.json.tmp")); std::fs::write(&tmp, serde_json::to_string(&r).unwrap()).unwrap(); std::fs::rename(&tmp, out.join(format!("metrics_0x{seed:X}.json"))).unwrap();
    std::fs::write(&done, "").unwrap(); println!("0x{seed:X} ok {:.0}s", t0.elapsed().as_secs_f64());
}

// ─── Paso 0: equivalencias y determinismo ───────────────────────────────────
pub fn bits(v: &[Vec<f64>]) -> Vec<u64> { v.iter().flatten().map(|x| x.to_bits()).collect() }
pub fn paso0(a: &[String]) {
    let seed = parse_seed(&arg_of(a, "--seed").unwrap_or_else(|| "0xC4A0".into())); assert!(seeds("smoke").contains(&seed));
    let na: usize = arg_of(a, "--na").map(|v| v.parse().unwrap()).unwrap_or(48); let nb: usize = arg_of(a, "--nb").map(|v| v.parse().unwrap()).unwrap_or(8);
    let c = Cfg::v4(); let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c);
    run_stream(&mut net, &mut cdt, &gen_known_stream(seed, XOR_KNOWN, na, &(0..N_FAM).collect::<Vec<_>>()), seed); run_stream(&mut net, &mut cdt, &gen_stream(seed, nb, nb), seed);
    let held = gen_heldout(seed, 4); let mut res = serde_json::Map::new();
    // (a) runner ATLAS genérico con Mech::default() ≡ run_episode v4 (eval), bit a bit, en preds h1..h16
    let md = Model::plain(Mech::default(), net.dp); let (mut nv, mut bad) = (0usize, 0usize);
    for e in &held { let mut n = net.clone(); let o = run_episode(&mut n, e, false, 0.0, false, K); let p = run_ep(&net, &md, e, K); let (x, y) = (bits(&o.preds), bits(&p.preds)); nv += x.len(); bad += x.iter().zip(&y).filter(|(a, b)| a != b).count() + x.len().abs_diff(y.len()); }
    res.insert("a_generic_off_eq_v4".into(), json!({"values": nv, "mismatch": bad, "pass": bad == 0 && nv > 0}));
    // (b) eval_held_atlas(off) ≡ eval_held v4 y E33
    let mut n = net.clone(); let hv = eval_held(&mut n, &held); let (hv2, _, _) = eval_held_atlas(&net, &Model::plain(Mech::default(), net.dp), &held);
    let mut n2 = net.clone(); let md0 = Model { m: Mech::from_list("M6"), ..Model::plain(Mech::default(), net.dp) }; let (hv3, g3, _) = eval_held_atlas(&net, &md0, &held); let _ = &mut n2;
    res.insert("b_held_off_eq_v4".into(), json!({"v4": hv, "atlas_off": hv2, "m6_alone": hv3, "m6_gate": g3, "pass": hv == hv2}));
    // (c) M6 solo: su ruta ATLAS es v4 ⇒ mismas cifras (no-op por construcción)
    res.insert("c_m6_alone_eq_v4".into(), json!({"pass": hv == hv3}));
    // (d) determinismo de meta-entrenamiento y sin NaN por brazo (rápido: sin E33)
    let mut info = serde_json::Map::new(); let mut ok_d = true;
    for (name, m) in arms() { if m.off() { continue; } let t = std::time::Instant::now(); let (m1, j1) = meta_train(&net, m, seed); let (m2, _) = meta_train(&net, m, seed);
        let same = format!("{m1:?}") == format!("{m2:?}"); let (h, g, nan) = eval_held_atlas(&net, &m1, &held); ok_d &= same && !nan;
        info.insert(name.into(), json!({"deterministic": same, "nan": nan, "held": h, "gate": g, "meta": {"alpha": j1["alpha"], "lam": j1["lam"], "thr": j1["thr"], "syn_rejected": j1["syn_rejected"], "best_obj": j1["best_obj"]}, "s": t.elapsed().as_secs_f64()})); }
    res.insert("d_meta_deterministic_no_nan".into(), json!({"pass": ok_d, "arms": info}));
    // (e) M4: ninguna familia retenida en meta-datos; tasa de rechazo de la auditoría
    let (eps, rej) = meta_eps(seed, &Mech::all(), N_FIT, 0x1); res.insert("e_m4_no_heldout_fams".into(), json!({"pass": eps.iter().all(|e| !NEW_FAMS.contains(&e.fam)), "n_syn": eps.iter().filter(|e| e.fam == 100).count(), "rejected": rej}));
    let ok = res.values().all(|v| v["pass"] == true);
    println!("{}", json!({"seed": format!("0x{seed:X}"), "git_sha": git_sha(), "paso0_ok": ok, "weights_hash": weights_hash(&net), "res": res}));
}

#[cfg(test)]
mod tests {
    use super::*;
    fn small_net(seed: u64) -> Net { let c = Cfg::v4(); let mut net = Net::new(&c, seed); let mut cdt = Cdt::new(seed, &c); run_stream(&mut net, &mut cdt, &gen_known_stream(seed, XOR_KNOWN, 6, &(0..N_FAM).collect::<Vec<_>>()), seed); net }
    #[test] fn off_is_v4_bit_exact() { let net = small_net(0xC4A1); let md = Model::plain(Mech::default(), net.dp);
        for e in gen_heldout(0xC4A1, 1) { let mut n = net.clone(); let o = run_episode(&mut n, &e, false, 0.0, false, K); let p = run_ep(&net, &md, &e, K); assert_eq!(bits(&o.preds), bits(&p.preds)); let (q, _, _) = arm_preds(&net, &md, &e); assert_eq!(bits(&o.preds), bits(&q)); } }
    #[test] fn m6_alone_is_v4() { let net = small_net(0xC4A2); let md = Model { m: Mech::from_list("M6"), ..Model::plain(Mech::default(), net.dp) };
        for e in gen_heldout(0xC4A2, 1).iter().take(4) { let mut n = net.clone(); let o = run_episode(&mut n, e, false, 0.0, false, K); assert_eq!(bits(&o.preds), bits(&arm_preds(&net, &md, e).0)); } }
    #[test] fn sparse_fit_recovers_sparse() { let mut r = Xoshiro256StarStar::seed_from_u64(1); let xs: Vec<Vec<f64>> = (0..60).map(|_| (0..6).map(|_| r.gen_range(-1.0..1.0)).collect()).collect();
        let ys: Vec<Vec<f64>> = xs.iter().map(|x| vec![2.0 * x[1], -0.5 * x[4]]).collect(); let w = sparse_fit(&xs, &ys, 1e-8, 0.05);
        for j in 0..6 { assert!((w[j] - if j == 1 { 2.0 } else { 0.0 }).abs() < 1e-5); assert!((w[6 + j] - if j == 4 { -0.5 } else { 0.0 }).abs() < 1e-5); } }
    #[test] fn synthetic_excludes_heldout() { let (eps, _) = meta_eps(0xC4A3, &Mech::all(), 32, 0x9); assert!(eps.iter().all(|e| !NEW_FAMS.contains(&e.fam))); assert!(eps.iter().any(|e| e.fam == 100));
        let (k, _) = meta_eps(0xC4A3, &Mech::default(), 16, 0x9); assert!(k.iter().all(|e| e.fam < N_FAM)); }
    #[test] fn audit_flags_heldout_rule() { let mut r = Xoshiro256StarStar::seed_from_u64(5); let probe = make_object(&mut r, K_POINTS, ObjectClass::Blob);
        // fam 11 con p=(0.5,0.5) es exactamente TanhShear(x,0.15,2)∘TanhShear(y,0.15,2): la auditoría debe rechazarla
        assert!(leaks(&[Prim::TanhShear(true, 0.15, 2.0), Prim::TanhShear(false, 0.15, 2.0)], &probe)); assert!(!leaks(&[Prim::Scale(0.1)], &probe)); }
    #[test] fn h0_fit_recovers_params() { for e in gen_heldout(0xC4A4, 1).into_iter().filter(|e| e.fam >= N_FAM) { let (p, l) = fit_p(e.fam, &e); assert!(l < 1e-12, "fam {} loss {l}", e.fam); let _ = p; } }
    #[test] fn ctx_dims() { let f = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.15, 0.2, 0.35, 0.4]]; let c = ctx(&f); assert_eq!(c.len(), DC); assert!(c.iter().all(|v| v.is_finite())); }
}
