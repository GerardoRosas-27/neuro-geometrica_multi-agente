//! ESTELA v0 · forward único (genérico en el escalar `S`). Entrenamiento (`Var`) y evaluación (`f64`)
//! llaman a ESTAS funciones; no existe una segunda implementación del forward (Paso 0).
//!
//! Por punto q∈R² con contexto c (como ÓRBITA):
//!   E1  z = [q; g_θ(q,c)] ∈ R^8                (g_θ iniciado con el lifting de ÓRBITA w3,b3,w4,b4)
//!   E2  u = K_ÓRBITA(c)·z + r_φ(z) [+ Aᵀξ]      (K congelado; r_φ MLP 8→32→8 con capa final = 0)
//!       ẑ0 = B(u)                              (B = corrector Brenier de ÓRBITA, congelado, 8 dims)
//!   E3  ẑ ← ẑ − α(∇h_ψ(ẑ) + (ẑ − ẑ0)), NREF veces (energía E(z)=½‖z−ẑ0‖²+h_ψ(z); h_ψ salida = 0 al inicio)
//!   E5  q̂' = P ẑ                               (sonda lineal 2×8, iniciada en [I 0])
//!   rollout: el siguiente paso re-codifica q̂' (igual que ÓRBITA). Con r=0, h=0, P=[I 0] y g=ÓRBITA,
//!   q̂' es bit a bit ÓRBITA.step (Paso 0).
use super::ad::S;
use crate::model::{CtxParams, Orbita, M, NB};

pub const RH: usize = 32; // ancho del residual r_φ
pub const EH: usize = 16; // ancho de h_ψ
pub const NREF: usize = 2; // pasos de refinamiento por energía
pub const ALPHA: f64 = 0.5; // paso del refinamiento
pub const NX: usize = 12; // entrada del encoder: q(2) + c(10)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant { Full, NoSig, Rec, Mlp, R0 }
impl Variant {
    pub fn name(self) -> &'static str { match self { Variant::Full => "estela", Variant::NoSig => "estela_nosig", Variant::Rec => "estela_rec", Variant::Mlp => "estela_mlp", Variant::R0 => "estela_r0" } }
    pub fn all() -> [Variant; 5] { [Variant::Full, Variant::NoSig, Variant::Rec, Variant::Mlp, Variant::R0] }
    pub fn uses_k(self) -> bool { self != Variant::Mlp }
    pub fn uses_r(self) -> bool { self != Variant::R0 }
}

/// disposición del vector plano de parámetros entrenables
#[derive(Clone, Copy, Debug)]
pub struct Lay { pub hg: usize, pub w3: usize, pub b3: usize, pub w4: usize, pub b4: usize, pub r1: usize, pub rb1: usize, pub r2: usize, pub rb2: usize, pub e1: usize, pub eb1: usize, pub e2: usize, pub pr: usize, pub n: usize }
impl Lay {
    pub fn new(hg: usize) -> Self {
        let mut o = 0; let mut nx = |k: usize| { let s = o; o += k; s };
        let w3 = nx(hg * NX); let b3 = nx(hg); let w4 = nx((M - 2) * hg); let b4 = nx(M - 2);
        let r1 = nx(RH * M); let rb1 = nx(RH); let r2 = nx(M * RH); let rb2 = nx(M);
        let e1 = nx(EH * M); let eb1 = nx(EH); let e2 = nx(EH); let pr = nx(2 * M);
        Lay { hg, w3, b3, w4, b4, r1, rb1, r2, rb2, e1, eb1, e2, pr, n: o }
    }
}

/// parámetros congelados por contexto (derivados de ÓRBITA en f64, idénticos a model.rs)
pub struct Frozen<'a> { pub d: &'a Orbita, pub cp: CtxParams, pub c: Vec<f64>, pub v: Variant }
impl<'a> Frozen<'a> { pub fn new(d: &'a Orbita, c: &[f64], v: Variant) -> Self { assert_eq!(d.variant, "full"); Frozen { d, cp: d.ctx(c), c: c.to_vec(), v } } }

#[inline]
pub fn dot_pub<T: S>(w: &[T], x: &[T]) -> T { dot(w, x) }
#[inline]
fn dot<T: S>(w: &[T], x: &[T]) -> T { let mut s = w[0] * x[0]; for j in 1..x.len() { s = s + w[j] * x[j]; } s }

/// E1: z = [q; g(q,c)] — mismo orden de operaciones que `Orbita::encode`
pub fn enc<T: S>(th: &[T], l: &Lay, q: [T; 2], c: &[f64]) -> [T; M] {
    let mut inp = [T::c(0.0); NX]; inp[0] = q[0]; inp[1] = q[1]; for j in 0..10 { inp[2 + j] = T::c(c[j]); }
    let mut h = vec![T::c(0.0); l.hg];
    for i in 0..l.hg { h[i] = (dot(&th[l.w3 + i * NX..l.w3 + (i + 1) * NX], &inp) + th[l.b3 + i]).tanh(); }
    let mut z = [T::c(0.0); M]; z[0] = q[0]; z[1] = q[1];
    for i in 0..M - 2 { z[2 + i] = dot(&th[l.w4 + i * l.hg..l.w4 + (i + 1) * l.hg], &h) + th[l.b4 + i]; }
    z
}
/// K(c)·z — mismo orden que `Orbita::core` (variante full)
fn core<T: S>(cp: &CtxParams, z: &[T; M]) -> [T; M] {
    let mut y = [T::c(0.0); M];
    for i in 0..M { let mut s = z[0].mul_c(cp.q[0][i]); for j in 1..M { s = s + z[j].mul_c(cp.q[j][i]); } y[i] = s; }
    for j in 0..NB {
        let (a, b) = (y[2 * j], y[2 * j + 1]);
        let (cs, sn) = (cp.om[j].cos(), cp.om[j].sin());
        y[2 * j] = (a.mul_c(cs) - b.mul_c(sn)).mul_c(cp.rho[j]);
        y[2 * j + 1] = (a.mul_c(sn) + b.mul_c(cs)).mul_c(cp.rho[j]);
    }
    let mut u = [T::c(0.0); M];
    for i in 0..M { let mut s = y[0].mul_c(cp.q[i][0]); for j in 1..M { s = s + y[j].mul_c(cp.q[i][j]); } u[i] = s.add_c(cp.beta[i]); }
    u
}
/// Brenier — mismo orden que `Orbita::correct`
fn correct<T: S>(d: &Orbita, cp: &CtxParams, u0: [T; M]) -> [T; M] {
    let mut u = u0;
    for _ in 0..d.r {
        let mut gate = vec![T::c(0.0); d.kc];
        for k in 0..d.kc {
            let mut s = u[0].mul_c(d.wc[k][0]); for i in 1..M { s = s + u[i].mul_c(d.wc[k][i]); }
            let pre = s.add_c(cp.bk[k]);
            let sig = T::c(1.0) / (-pre).exp().add_c(1.0);
            gate[k] = sig.mul_c(cp.ak[k]);
        }
        for i in 0..M { let mut s = gate[0].mul_c(d.wc[0][i]); for k in 1..d.kc { s = s + gate[k].mul_c(d.wc[k][i]); } u[i] = u[i] + s.mul_c(d.eta); }
    }
    u
}
fn residual<T: S>(th: &[T], l: &Lay, z: &[T; M]) -> [T; M] {
    let mut h = [T::c(0.0); RH];
    for i in 0..RH { h[i] = (dot(&th[l.r1 + i * M..l.r1 + (i + 1) * M], z) + th[l.rb1 + i]).tanh(); }
    let mut r = [T::c(0.0); M];
    for i in 0..M { r[i] = dot(&th[l.r2 + i * RH..l.r2 + (i + 1) * RH], &h) + th[l.rb2 + i]; }
    r
}
/// h_ψ(z) = Σ_k e2_k tanh(e1_k·z + eb1_k) y su gradiente (expresado con operaciones de S para que el entrenamiento derive a través)
pub fn energy_h<T: S>(th: &[T], l: &Lay, z: &[T; M]) -> (T, [T; M]) {
    let mut hv = T::c(0.0); let mut g = [T::c(0.0); M];
    for k in 0..EH {
        let w = &th[l.e1 + k * M..l.e1 + (k + 1) * M];
        let t = (dot(w, z) + th[l.eb1 + k]).tanh();
        hv = hv + th[l.e2 + k] * t;
        let s = th[l.e2 + k] * (T::c(1.0) - t * t);
        for j in 0..M { g[j] = g[j] + s * w[j]; }
    }
    (hv, g)
}
fn refine<T: S>(th: &[T], l: &Lay, z0: [T; M]) -> [T; M] {
    let mut z = z0;
    for _ in 0..NREF { let (_, g) = energy_h(th, l, &z); for j in 0..M { z[j] = z[j] - (g[j] + (z[j] - z0[j])).mul_c(ALPHA); } }
    z
}
/// E(z) = ½‖z − ẑ0‖² + h_ψ(z)
pub fn energy<T: S>(th: &[T], l: &Lay, z: &[T; M], z0: &[T; M]) -> T {
    let (h, _) = energy_h(th, l, z); let mut s = h; for j in 0..M { let d = z[j] - z0[j]; s = s + (d * d).mul_c(0.5); } s
}

pub struct StepOut<T> { pub zhat: Vec<[T; M]>, pub z0: Vec<[T; M]>, pub q: Vec<T> }
/// un paso de ESTELA sobre una nube x (6 puntos). `lat` = adaptador LAT opcional (f64, como latent.rs: ξ=[z, monomios grado 2, 1]).
pub fn step<T: S>(th: &[T], l: &Lay, fz: &Frozen, x: &[T], lat: Option<&[f64]>) -> StepOut<T> {
    let np = x.len() / 2; let mut out = StepOut { zhat: Vec::with_capacity(np), z0: Vec::with_capacity(np), q: vec![T::c(0.0); x.len()] };
    for i in 0..np {
        let z = enc(th, l, [x[2 * i], x[2 * i + 1]], &fz.c);
        let mut u = if fz.v.uses_k() { core(&fz.cp, &z) } else { z };
        if fz.v.uses_r() { let r = residual(th, l, &z); for j in 0..M { u[j] = u[j] + r[j]; } }
        if let Some(a) = lat {
            let (qx, qy) = (x[2 * i], x[2 * i + 1]);
            let mut xi: Vec<T> = z.to_vec(); xi.push(qx * qx); xi.push(qx * qy); xi.push(qy * qy); xi.push(T::c(1.0));
            for (r, v) in xi.iter().enumerate() { for j in 0..M { u[j] = u[j] + v.mul_c(a[r * M + j]); } }
        }
        let z0 = correct(fz.d, &fz.cp, u);
        let zh = refine(th, l, z0);
        for o in 0..2 { out.q[2 * i + o] = dot(&th[l.pr + o * M..l.pr + (o + 1) * M], &zh); }
        out.zhat.push(zh); out.z0.push(z0);
    }
    out
}
pub fn npar_lat() -> usize { (M + 4) * M }

/// parámetros iniciales: g = lifting de ÓRBITA; r_φ: W1 ~ U(±1/√8) (rng), W2 = b2 = 0; h_ψ: W1 ~ U(±1/√8), e2 = 0; P = [I 0]
pub fn init(d: &Orbita, l: &Lay, seed: u64) -> Vec<f64> {
    use rand::Rng; use rand_xoshiro::rand_core::SeedableRng;
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed ^ 0xE57E_1A00);
    let mut th = vec![0.0; l.n];
    for i in 0..l.hg { for j in 0..NX { th[l.w3 + i * NX + j] = d.w3[i][j]; } th[l.b3 + i] = d.b3[i]; }
    for i in 0..M - 2 { for j in 0..l.hg { th[l.w4 + i * l.hg + j] = d.w4[i][j]; } th[l.b4 + i] = d.b4[i]; }
    let s = 1.0 / (M as f64).sqrt();
    for v in &mut th[l.r1..l.r1 + RH * M] { *v = rng.gen_range(-s..s); }
    for v in &mut th[l.e1..l.e1 + EH * M] { *v = rng.gen_range(-s..s); }
    th[l.pr] = 1.0; th[l.pr + M + 1] = 1.0;
    th
}
