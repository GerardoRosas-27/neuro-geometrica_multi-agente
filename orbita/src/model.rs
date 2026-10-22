//! ÓRBITA forward (inferencia, f64). Debe coincidir con `train_orbita.py::Orbita.step`.
//! Por punto q∈R², contexto c∈R^10 (campo compartido entre puntos, como Dφ):
//!   head(c) → ω(4), s(4), A(28: antisimétrica 8×8), β(8), bk(KC), ak(KC)
//!   z = [q; g(q,c)] ∈ R^8 ;  Q = Cayley(A) ;  y = Q·blk(ρ_j R(ω_j))·Qᵀ z + β
//!   ρ_j = exp(s_j) (bloques tangentes j<2), exp(-softplus(s_j)) (normales j≥2)
//!   corrector Brenier r veces: u ← u + η Wcᵀ (softplus(ak) ⊙ σ(Wc u + bk))
//!   salida q' = u[0..2]
use cdt_rqm_epr::v4_controls::Predictor;
use serde_json::Value;

pub const M: usize = 8;
pub const NB: usize = 4;
pub const NT: usize = 2;

#[derive(Clone)]
pub struct Orbita {
    pub hh: usize,
    pub hg: usize,
    pub kc: usize,
    pub r: usize,
    pub eta: f64,
    pub w1: Vec<Vec<f64>>, pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>, pub b2: Vec<f64>,
    pub w3: Vec<Vec<f64>>, pub b3: Vec<f64>,
    pub w4: Vec<Vec<f64>>, pub b4: Vec<f64>,
    pub wc: Vec<Vec<f64>>,
    pub raw: String,
    pub variant: String,
}

fn mat(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect()
}
fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn lin(w: &[Vec<f64>], b: Option<&[f64]>, x: &[f64]) -> Vec<f64> {
    w.iter().enumerate().map(|(i, row)| row.iter().zip(x).map(|(a, c)| a * c).sum::<f64>() + b.map_or(0.0, |b| b[i])).collect()
}
fn softplus(x: f64) -> f64 { if x > 30.0 { x } else { x.exp().ln_1p() } }
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Resuelve (I - S/2) Q = (I + S/2) por eliminación gaussiana con pivoteo.
fn cayley(s: &[[f64; M]; M]) -> [[f64; M]; M] {
    let mut a = [[0.0; M]; M];
    let mut b = [[0.0; M]; M];
    for i in 0..M { for j in 0..M {
        let id = if i == j { 1.0 } else { 0.0 };
        a[i][j] = id - 0.5 * s[i][j];
        b[i][j] = id + 0.5 * s[i][j];
    }}
    for col in 0..M {
        let p = (col..M).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs())).unwrap();
        a.swap(col, p); b.swap(col, p);
        let d = a[col][col];
        for j in 0..M { a[col][j] /= d; b[col][j] /= d; }
        for i in 0..M { if i != col {
            let f = a[i][col];
            if f != 0.0 { for j in 0..M { a[i][j] -= f * a[col][j]; b[i][j] -= f * b[col][j]; } }
        }}
    }
    b
}

pub struct CtxParams { pub kf: Vec<f64>, pub q: [[f64; M]; M], pub rho: [f64; NB], pub om: [f64; NB], pub beta: Vec<f64>, pub bk: Vec<f64>, pub ak: Vec<f64> }

impl Orbita {
    pub fn from_json(t: &str) -> Self {
        let v: Value = serde_json::from_str(t).expect("weights json");
        let g = |k: &str| &v[k];
        Self {
            hh: g("hh").as_u64().unwrap() as usize, hg: g("hg").as_u64().unwrap() as usize,
            kc: g("kc").as_u64().unwrap() as usize, r: g("r").as_u64().unwrap() as usize,
            eta: g("eta").as_f64().unwrap(),
            w1: mat(g("w1")), b1: vecf(g("b1")), w2: mat(g("w2")), b2: vecf(g("b2")),
            w3: mat(g("w3")), b3: vecf(g("b3")), w4: mat(g("w4")), b4: vecf(g("b4")),
            wc: mat(g("wc")), raw: t.to_string(), variant: g("variant").as_str().unwrap_or("full").to_string(),
        }
    }
    pub fn n_params(&self) -> usize {
        let m = |w: &Vec<Vec<f64>>| w.len() * w[0].len();
        m(&self.w1) + self.b1.len() + m(&self.w2) + self.b2.len() + m(&self.w3) + self.b3.len() + m(&self.w4) + self.b4.len() + m(&self.wc)
    }
    pub fn ctx(&self, c: &[f64]) -> CtxParams {
        let h: Vec<f64> = lin(&self.w1, Some(&self.b1), c).into_iter().map(f64::tanh).collect();
        let o = lin(&self.w2, Some(&self.b2), &h);
        if self.variant == "A" || self.variant == "B" {
            let mut k = if self.variant == "B" { M * M } else { 0 };
            let kf = o[..k].to_vec();
            let beta = o[k..k + M].to_vec(); k += M;
            let bk = o[k..k + self.kc].to_vec(); k += self.kc;
            let ak: Vec<f64> = o[k..k + self.kc].iter().map(|x| softplus(*x)).collect();
            return CtxParams { kf, q: [[0.0; M]; M], rho: [0.0; NB], om: [0.0; NB], beta, bk, ak };
        }
        let mut k = 0;
        let mut om = [0.0; NB]; for j in 0..NB { om[j] = o[k]; k += 1; }
        let mut rho = [0.0; NB];
        for j in 0..NB { let s = o[k]; k += 1; rho[j] = if j < NT { s.exp() } else { (-softplus(s)).exp() }; }
        let mut s = [[0.0; M]; M];
        for i in 0..M { for j in (i + 1)..M { s[i][j] = o[k]; s[j][i] = -o[k]; k += 1; } }
        let beta = o[k..k + M].to_vec(); k += M;
        let bk = o[k..k + self.kc].to_vec(); k += self.kc;
        let ak: Vec<f64> = o[k..k + self.kc].iter().map(|x| softplus(*x)).collect();
        CtxParams { kf: vec![], q: cayley(&s), rho, om, beta, bk, ak }
    }
    pub fn encode(&self, q: &[f64], c: &[f64]) -> Vec<f64> {
        let mut inp = q.to_vec(); inp.extend_from_slice(c);
        let h: Vec<f64> = lin(&self.w3, Some(&self.b3), &inp).into_iter().map(f64::tanh).collect();
        let g = lin(&self.w4, Some(&self.b4), &h);
        let mut z = q.to_vec(); z.extend(g); z
    }
    pub fn core(&self, cp: &CtxParams, z: &[f64]) -> Vec<f64> {
        if self.variant == "A" { return z.to_vec(); }
        if self.variant == "B" {
            return (0..M).map(|i| z[i] + (0..M).map(|j| cp.kf[i * M + j] * z[j]).sum::<f64>() + cp.beta[i]).collect();
        }
        // y = Qᵀ z
        let mut y = [0.0; M];
        for i in 0..M { y[i] = (0..M).map(|j| cp.q[j][i] * z[j]).sum(); }
        for j in 0..NB {
            let (a, b) = (y[2 * j], y[2 * j + 1]);
            let (cs, sn) = (cp.om[j].cos(), cp.om[j].sin());
            y[2 * j] = cp.rho[j] * (cs * a - sn * b);
            y[2 * j + 1] = cp.rho[j] * (sn * a + cs * b);
        }
        (0..M).map(|i| (0..M).map(|j| cp.q[i][j] * y[j]).sum::<f64>() + cp.beta[i]).collect()
    }
    pub fn correct(&self, cp: &CtxParams, u0: &[f64]) -> Vec<f64> {
        let mut u = u0.to_vec();
        for _ in 0..self.r {
            let pre = lin(&self.wc, Some(&cp.bk), &u);
            let gate: Vec<f64> = pre.iter().zip(&cp.ak).map(|(p, a)| a * sigmoid(*p)).collect();
            for i in 0..M { u[i] += self.eta * (0..self.kc).map(|k| self.wc[k][i] * gate[k]).sum::<f64>(); }
        }
        u
    }
    pub fn step(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        let cp = self.ctx(c);
        let mut out = vec![0.0; x.len()];
        for p in 0..x.len() / 2 {
            let z = self.encode(&x[2 * p..2 * p + 2], c);
            let u = self.correct(&cp, &self.core(&cp, &z));
            out[2 * p] = u[0]; out[2 * p + 1] = u[1];
        }
        out
    }
}

impl Predictor for Orbita {
    fn name(&self) -> &str { "ORBITA" }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> { self.step(x, c) }
    fn params(&self) -> usize { self.n_params() }
}
