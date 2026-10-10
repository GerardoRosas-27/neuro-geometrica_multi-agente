//! v4 models and paired controls (protocolo §2 `v4_controls`).
//!
//! * `Dphi`: residual field dynamics `z' = z + F_θ([z, c])` (2-layer tanh MLP).
//!   Fresh random init per seed; never loaded from E11–E17/v3.
//! * `MlpDirect`: same capacity, non-residual `y = G_θ([x, c])`.
//! * `LinearCtl`: ridge least squares on `[x, c, 1]`.
//! * `NnCtl`: 1-nearest-neighbour displacement transfer (LOOKUP control;
//!   every query is counted as `nn_queries` by the provenance guard).
//! * `StaticCtl`: `Decoder(Encoder(x)) = x` (no transformation).
//! * random dynamics = untrained `Dphi` with full-scale random output layer.
//! * shuffled-label = `Dphi` trained with targets permuted across TRAIN.

#![allow(clippy::needless_range_loop)]

use crate::v4_provenance::{count_query, Store};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};

pub trait Predictor {
    fn name(&self) -> &str;
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64>;
    fn params(&self) -> usize;
}

fn randn(rng: &mut Xoshiro256StarStar) -> f64 {
    let u1: f64 = rng.gen_range(1e-12..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mlp {
    pub din: usize,
    pub dh: usize,
    pub dout: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

impl Mlp {
    pub fn new(din: usize, dh: usize, dout: usize, out_scale: f64, seed: u64) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let s1 = 1.0 / (din as f64).sqrt();
        let s2 = out_scale / (dh as f64).sqrt();
        Self {
            din,
            dh,
            dout,
            w1: (0..din * dh).map(|_| randn(&mut rng) * s1).collect(),
            b1: vec![0.0; dh],
            w2: (0..dh * dout).map(|_| randn(&mut rng) * s2).collect(),
            b2: vec![0.0; dout],
        }
    }

    pub fn n_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    pub fn forward(&self, inp: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut h = self.b1.clone();
        for j in 0..self.dh {
            let row = &self.w1[j * self.din..(j + 1) * self.din];
            let mut a = h[j];
            for i in 0..self.din {
                a += row[i] * inp[i];
            }
            h[j] = a.tanh();
        }
        let mut o = self.b2.clone();
        for k in 0..self.dout {
            let row = &self.w2[k * self.dh..(k + 1) * self.dh];
            let mut a = o[k];
            for j in 0..self.dh {
                a += row[j] * h[j];
            }
            o[k] = a;
        }
        (h, o)
    }

    pub fn flat(&self) -> Vec<f64> {
        let mut v = self.w1.clone();
        v.extend_from_slice(&self.b1);
        v.extend_from_slice(&self.w2);
        v.extend_from_slice(&self.b2);
        v
    }

    pub fn set_flat(&mut self, v: &[f64]) {
        let (a, b, c) = (self.w1.len(), self.b1.len(), self.w2.len());
        self.w1.copy_from_slice(&v[..a]);
        self.b1.copy_from_slice(&v[a..a + b]);
        self.w2.copy_from_slice(&v[a + b..a + b + c]);
        self.b2.copy_from_slice(&v[a + b + c..]);
    }

    /// Accumulate MSE gradient for one example; `d_out` = dL/d(out).
    pub fn backward(&self, inp: &[f64], h: &[f64], d_out: &[f64], g: &mut [f64]) {
        let (a, b, c) = (self.w1.len(), self.b1.len(), self.w2.len());
        let mut dh = vec![0.0; self.dh];
        for k in 0..self.dout {
            let dk = d_out[k];
            if dk == 0.0 {
                continue;
            }
            for j in 0..self.dh {
                g[a + b + k * self.dh + j] += dk * h[j];
                dh[j] += dk * self.w2[k * self.dh + j];
            }
            g[a + b + c + k] += dk;
        }
        for j in 0..self.dh {
            let dz = dh[j] * (1.0 - h[j] * h[j]);
            g[a + j] += dz;
            for i in 0..self.din {
                g[j * self.din + i] += dz * inp[i];
            }
        }
    }
}

/// Adam optimiser state for a flat parameter vector.
pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
    pub lr: f64,
}

impl Adam {
    pub fn new(n: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
        }
    }

    pub fn step(&mut self, p: &mut [f64], g: &[f64]) {
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

/// One supervised transition (input state, context, target state).
pub type Triple = (Vec<f64>, Vec<f64>, Vec<f64>);

#[derive(Clone, Copy, Debug)]
pub struct TrainBudget {
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
}

impl TrainBudget {
    /// Pre-registered v4 lock (docs/preregistro_v4.md). Not changed after DEV.
    pub fn lock() -> Self {
        Self {
            steps: 8000,
            batch: 16,
            lr: 3e-3,
        }
    }

    pub fn examples_seen(&self) -> usize {
        self.steps * self.batch
    }
}

/// Field dynamics Dφ (residual). `residual=false` gives the MLP-direct control.
///
/// Dφ is a *field*: the same learned vector field `F_θ(q, c)` acts on every
/// 2-D point `q` of the state (weights shared across points), so the state
/// dimension `zdim = 2k` can change without changing θ (used by E42).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dphi {
    pub label: String,
    pub residual: bool,
    pub zdim: usize,
    pub net: Mlp,
}

/// Point dimension of the field.
pub const PD: usize = 2;

pub const HIDDEN: usize = 64;

impl Dphi {
    pub fn new(zdim: usize, cdim: usize, seed: u64) -> Self {
        Self {
            label: "DPHI".into(),
            residual: true,
            zdim,
            net: Mlp::new(PD + cdim, HIDDEN, PD, 0.1, seed),
        }
    }

    pub fn mlp_direct(zdim: usize, cdim: usize, seed: u64) -> Self {
        Self {
            label: "MLP_DIRECT".into(),
            residual: false,
            zdim,
            net: Mlp::new(PD + cdim, HIDDEN, PD, 0.1, seed),
        }
    }

    pub fn random_dynamics(zdim: usize, cdim: usize, seed: u64) -> Self {
        Self {
            label: "RANDOM_DYN".into(),
            residual: true,
            zdim,
            net: Mlp::new(PD + cdim, HIDDEN, PD, 1.0, seed),
        }
    }

    fn input(z: &[f64], c: &[f64]) -> Vec<f64> {
        let mut v = z.to_vec();
        v.extend_from_slice(c);
        v
    }

    pub fn step(&self, z: &[f64], c: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; z.len()];
        for p in 0..z.len() / PD {
            let q = &z[p * PD..(p + 1) * PD];
            let (_, o) = self.net.forward(&Self::input(q, c));
            for i in 0..PD {
                out[p * PD + i] = o[i] + if self.residual { q[i] } else { 0.0 };
            }
        }
        out
    }

    pub fn rollout(&self, z0: &[f64], c: &[f64], h: usize) -> Vec<f64> {
        let mut z = z0.to_vec();
        for _ in 0..h {
            z = self.step(&z, c);
        }
        z
    }

    /// Mini-batch Adam on one-step MSE. `sampler` supplies each example
    /// (lets E35-B mix TRAIN with consolidated replay under the same budget).
    pub fn train_with<F>(&mut self, budget: TrainBudget, seed: u64, mut sampler: F) -> f64
    where
        F: FnMut(&mut Xoshiro256StarStar) -> Triple,
    {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let n = self.net.n_params();
        let mut opt = Adam::new(n, budget.lr);
        let mut p = self.net.flat();
        let mut last = 0.0;
        for _ in 0..budget.steps {
            let mut g = vec![0.0; n];
            let mut loss = 0.0;
            for _ in 0..budget.batch {
                let (x, c, y) = sampler(&mut rng);
                let npts = x.len() / PD;
                for p in 0..npts {
                    let q = &x[p * PD..(p + 1) * PD];
                    let inp = Self::input(q, &c);
                    let (h, mut o) = self.net.forward(&inp);
                    if self.residual {
                        for i in 0..PD {
                            o[i] += q[i];
                        }
                    }
                    let d: Vec<f64> = (0..PD)
                        .map(|i| 2.0 * (o[i] - y[p * PD + i]) / (x.len() * budget.batch) as f64)
                        .collect();
                    loss += (0..PD).map(|i| (o[i] - y[p * PD + i]).powi(2)).sum::<f64>()
                        / x.len() as f64;
                    self.net.backward(&inp, &h, &d, &mut g);
                }
            }
            opt.step(&mut p, &g);
            self.net.set_flat(&p);
            last = loss / budget.batch as f64;
        }
        last
    }

    /// E45-L1: predictive-error-gated plasticity. Examples whose one-step
    /// prediction is already within `tau` relative error produce no gradient.
    /// Returns the number of examples that actually updated the weights.
    pub fn train_gated(
        &mut self,
        data: &[Triple],
        budget: TrainBudget,
        seed: u64,
        tau: f64,
    ) -> usize {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let n = self.net.n_params();
        let mut opt = Adam::new(n, budget.lr);
        let mut p = self.net.flat();
        let mut updates = 0usize;
        for _ in 0..budget.steps {
            let mut g = vec![0.0; n];
            let mut any = false;
            for _ in 0..budget.batch {
                let (x, c, y) = &data[rng.gen_range(0..data.len())];
                let pred = self.step(x, c);
                let err = pred
                    .iter()
                    .zip(y)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
                    / y.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
                if err < tau {
                    continue;
                }
                any = true;
                updates += 1;
                for pt in 0..x.len() / PD {
                    let q = &x[pt * PD..(pt + 1) * PD];
                    let inp = Self::input(q, c);
                    let (h, _) = self.net.forward(&inp);
                    let d: Vec<f64> = (0..PD)
                        .map(|i| {
                            2.0 * (pred[pt * PD + i] - y[pt * PD + i])
                                / (x.len() * budget.batch) as f64
                        })
                        .collect();
                    self.net.backward(&inp, &h, &d, &mut g);
                }
            }
            if any {
                opt.step(&mut p, &g);
                self.net.set_flat(&p);
            }
        }
        updates
    }

    pub fn train_on(&mut self, data: &[Triple], budget: TrainBudget, seed: u64) -> f64 {
        let n = data.len();
        self.train_with(budget, seed, |r| data[r.gen_range(0..n)].clone())
    }

    /// Bit-exact checkpoint (f64 bits as u64; JSON floats are not exact).
    pub fn to_ckpt(&self) -> String {
        let bits: Vec<u64> = self.net.flat().iter().map(|v| v.to_bits()).collect();
        serde_json::json!({
            "label": self.label, "residual": self.residual, "zdim": self.zdim,
            "din": self.net.din, "dh": self.net.dh, "dout": self.net.dout, "bits": bits
        })
        .to_string()
    }

    pub fn from_ckpt(t: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(t).ok()?;
        let (din, dh, dout) = (
            v["din"].as_u64()? as usize,
            v["dh"].as_u64()? as usize,
            v["dout"].as_u64()? as usize,
        );
        let mut net = Mlp::new(din, dh, dout, 0.0, 0);
        let bits: Vec<f64> = v["bits"]
            .as_array()?
            .iter()
            .map(|b| f64::from_bits(b.as_u64().unwrap_or(0)))
            .collect();
        net.set_flat(&bits);
        Some(Self {
            label: v["label"].as_str()?.into(),
            residual: v["residual"].as_bool()?,
            zdim: v["zdim"].as_u64()? as usize,
            net,
        })
    }

    pub fn weights_hash(&self) -> String {
        let mut b = Vec::new();
        for v in self.net.flat() {
            b.extend_from_slice(&v.to_le_bytes());
        }
        crate::v4_dataset::sha256_hex(&b)
    }
}

impl Predictor for Dphi {
    fn name(&self) -> &str {
        &self.label
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        self.step(x, c)
    }
    fn params(&self) -> usize {
        self.net.n_params()
    }
}

pub struct StaticCtl;

impl Predictor for StaticCtl {
    fn name(&self) -> &str {
        "STATIC"
    }
    fn predict(&self, x: &[f64], _c: &[f64]) -> Vec<f64> {
        x.to_vec()
    }
    fn params(&self) -> usize {
        0
    }
}

/// Solve (A + λI) w = b for symmetric A by Gaussian elimination with pivoting.
pub fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-14 {
            continue;
        }
        for r in 0..n {
            if r != col {
                let f = a[r][col] / d;
                if f != 0.0 {
                    for k in col..n {
                        a[r][k] -= f * a[col][k];
                    }
                    for k in 0..b[r].len() {
                        b[r][k] -= f * b[col][k];
                    }
                }
            }
        }
    }
    for r in 0..n {
        let d = a[r][r];
        for k in 0..b[r].len() {
            b[r][k] = if d.abs() < 1e-14 { 0.0 } else { b[r][k] / d };
        }
    }
    b
}

/// Ridge regression rows: returns W (features × outputs).
pub fn ridge(feats: &[Vec<f64>], ys: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let d = feats[0].len();
    let m = ys[0].len();
    let mut a = vec![vec![0.0; d]; d];
    let mut b = vec![vec![0.0; m]; d];
    for (f, y) in feats.iter().zip(ys) {
        for i in 0..d {
            if f[i] == 0.0 {
                continue;
            }
            for j in 0..d {
                a[i][j] += f[i] * f[j];
            }
            for k in 0..m {
                b[i][k] += f[i] * y[k];
            }
        }
    }
    for i in 0..d {
        a[i][i] += lambda;
    }
    solve(a, b)
}

pub struct LinearCtl {
    w: Vec<Vec<f64>>,
}

fn lin_feat(x: &[f64], c: &[f64]) -> Vec<f64> {
    let mut f = x.to_vec();
    f.extend_from_slice(c);
    f.push(1.0);
    f
}

/// Pointwise linear control (same field factorisation as Dφ).
impl LinearCtl {
    pub fn fit(data: &[Triple]) -> Self {
        let mut feats = vec![];
        let mut ys = vec![];
        for (x, c, y) in data {
            for p in 0..x.len() / PD {
                feats.push(lin_feat(&x[p * PD..(p + 1) * PD], c));
                ys.push(y[p * PD..(p + 1) * PD].to_vec());
            }
        }
        Self {
            w: ridge(&feats, &ys, 1e-3),
        }
    }
}

impl Predictor for LinearCtl {
    fn name(&self) -> &str {
        "LINEAR"
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        let mut out = vec![];
        for p in 0..x.len() / PD {
            let f = lin_feat(&x[p * PD..(p + 1) * PD], c);
            for k in 0..PD {
                out.push((0..f.len()).map(|i| f[i] * self.w[i][k]).sum());
            }
        }
        out
    }
    fn params(&self) -> usize {
        self.w.len() * self.w[0].len()
    }
}

/// Lookup control: nearest TRAIN example in `[x, c]`, transfer its displacement.
pub struct NnCtl {
    data: Vec<Triple>,
}

impl NnCtl {
    pub fn new(data: &[Triple]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

impl Predictor for NnCtl {
    fn name(&self) -> &str {
        "NN_LOOKUP"
    }
    fn predict(&self, x: &[f64], c: &[f64]) -> Vec<f64> {
        count_query(Store::Nn, 1);
        let mut best = (f64::MAX, 0usize);
        for (i, (xx, cc, _)) in self.data.iter().enumerate() {
            let d: f64 = xx
                .iter()
                .zip(x)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                + 4.0
                    * cc.iter()
                        .zip(c)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>();
            if d < best.0 {
                best = (d, i);
            }
        }
        let (xx, _, yy) = &self.data[best.1];
        x.iter()
            .zip(xx)
            .zip(yy)
            .map(|((a, b), y)| a + (y - b))
            .collect()
    }
    fn params(&self) -> usize {
        self.data.len() * (self.data[0].0.len() * 2 + self.data[0].1.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_dphi_learns_translation_smoke() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1);
        let data: Vec<Triple> = (0..64)
            .map(|_| {
                let x: Vec<f64> = (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let y = x.iter().map(|v| v + 0.3).collect();
                (x, vec![1.0], y)
            })
            .collect();
        let mut d = Dphi::new(4, 1, 2);
        let b = TrainBudget {
            steps: 300,
            batch: 8,
            lr: 3e-3,
        };
        let loss = d.train_on(&data, b, 3);
        assert!(loss < 0.01, "loss {loss}");
        let lin = LinearCtl::fit(&data);
        let p = lin.predict(&data[0].0, &data[0].1);
        assert!((p[0] - data[0].2[0]).abs() < 1e-3);
    }
}
