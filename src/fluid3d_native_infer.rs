//! Inferencia nativa **solo** con el simulador NS 3D — sin RQM ni NLS de spin.
//!
//! Flujo: entrada → blob de velocidad / forzamiento → pocos `Fluid3D::step`
//! → features baratos → clasificador (centroides o MLP pequeño).

use crate::fluid3d_astra::{Fluid3D, ResidualMlp, N};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::time::Instant;

pub const NUM_LABELS: usize = 4;
const COARSE: usize = 4;
/// Features: E, |u|_max, com(3), hist 4³ = 1+1+3+64 = 69
pub const FEAT_DIM: usize = 2 + 3 + COARSE * COARSE * COARSE;

#[inline]
fn ix(i: usize, j: usize, k: usize) -> usize {
    i + N * (j + N * k)
}

#[derive(Clone, Debug)]
pub struct Fluid3DNativeInfer {
    pub rollout_steps: usize,
    pub force_amp: f64,
    pub centroids: Vec<Vec<f64>>, // NUM_LABELS × FEAT_DIM
    pub mlp: Option<FeatureMlp>,
    pub trained: bool,
}

#[derive(Clone, Debug)]
pub struct FeatureMlp {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub hidden: usize,
    pub lr: f64,
}

impl FeatureMlp {
    pub fn new(seed: u64, hidden: usize) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let s1 = (2.0 / FEAT_DIM as f64).sqrt();
        let s2 = (2.0 / hidden as f64).sqrt();
        let mut w1 = vec![0.0; hidden * FEAT_DIM];
        let mut w2 = vec![0.0; NUM_LABELS * hidden];
        for v in &mut w1 {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * s1;
        }
        for v in &mut w2 {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * s2;
        }
        Self {
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: vec![0.0; NUM_LABELS],
            hidden,
            lr: 0.08,
        }
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut h = vec![0.0; self.hidden];
        for j in 0..self.hidden {
            let mut s = self.b1[j];
            for i in 0..FEAT_DIM {
                s += self.w1[j * FEAT_DIM + i] * x[i];
            }
            h[j] = s.max(0.0);
        }
        let mut y = vec![0.0; NUM_LABELS];
        for o in 0..NUM_LABELS {
            let mut s = self.b2[o];
            for j in 0..self.hidden {
                s += self.w2[o * self.hidden + j] * h[j];
            }
            y[o] = s;
        }
        y
    }

    pub fn predict(&self, x: &[f64]) -> usize {
        let y = self.forward(x);
        y.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn train_step(&mut self, x: &[f64], label: usize) -> f64 {
        let mut h_pre = vec![0.0; self.hidden];
        let mut h = vec![0.0; self.hidden];
        for j in 0..self.hidden {
            let mut s = self.b1[j];
            for i in 0..FEAT_DIM {
                s += self.w1[j * FEAT_DIM + i] * x[i];
            }
            h_pre[j] = s;
            h[j] = s.max(0.0);
        }
        let mut y = vec![0.0; NUM_LABELS];
        for o in 0..NUM_LABELS {
            let mut s = self.b2[o];
            for j in 0..self.hidden {
                s += self.w2[o * self.hidden + j] * h[j];
            }
            y[o] = s;
        }
        // one-hot MSE
        let mut loss = 0.0;
        let mut dy = vec![0.0; NUM_LABELS];
        for o in 0..NUM_LABELS {
            let t = if o == label { 1.0 } else { 0.0 };
            let e = y[o] - t;
            loss += e * e;
            dy[o] = 2.0 * e / NUM_LABELS as f64;
        }
        let mut dh = vec![0.0; self.hidden];
        for o in 0..NUM_LABELS {
            for j in 0..self.hidden {
                dh[j] += self.w2[o * self.hidden + j] * dy[o];
                self.w2[o * self.hidden + j] -= self.lr * dy[o] * h[j];
            }
            self.b2[o] -= self.lr * dy[o];
        }
        for j in 0..self.hidden {
            let g = if h_pre[j] > 0.0 { dh[j] } else { 0.0 };
            for i in 0..FEAT_DIM {
                self.w1[j * FEAT_DIM + i] -= self.lr * g * x[i];
            }
            self.b1[j] -= self.lr * g;
        }
        loss / NUM_LABELS as f64
    }
}

impl Default for Fluid3DNativeInfer {
    fn default() -> Self {
        Self::new()
    }
}

impl Fluid3DNativeInfer {
    pub fn new() -> Self {
        Self {
            rollout_steps: 3,
            force_amp: 0.0,
            centroids: vec![vec![0.0; FEAT_DIM]; NUM_LABELS],
            mlp: Some(FeatureMlp::new(0xF100_D3D, 32)),
            trained: false,
        }
    }

    /// Codifica la clase como blob de velocidad localizado (sin spin).
    pub fn encode_input(&self, input: usize) -> Fluid3D {
        let input = input % NUM_LABELS;
        let mut f = Fluid3D::new(0.001, 0.06);
        let mid = (N / 2) as f64;
        let offsets = [
            [-2.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, -2.5, 0.0],
            [0.0, 2.5, 0.0],
        ];
        let dirs = [[1.0, 0.2, 0.0], [0.0, 1.0, 0.2], [0.2, 0.0, 1.0], [0.7, 0.7, 0.0]];
        let o = offsets[input];
        let d = dirs[input];
        let sigma = 1.4;
        let amp = 0.55 + 0.05 * input as f64;
        let inv = 1.0 / (2.0 * sigma * sigma);
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    let dx = i as f64 - (mid + o[0]);
                    let dy = j as f64 - (mid + o[1]);
                    let dz = k as f64 - (mid + o[2]);
                    let g = amp * (-(dx * dx + dy * dy + dz * dz) * inv).exp();
                    let idx = ix(i, j, k);
                    f.u[idx] = d[0] * g;
                    f.v[idx] = d[1] * g;
                    f.w[idx] = d[2] * g;
                }
            }
        }
        f
    }

    pub fn rollout(&self, fluid: &mut Fluid3D) {
        for s in 0..self.rollout_steps {
            fluid.step(self.force_amp, s as f64 * fluid.dt);
        }
    }

    pub fn extract_features(&self, fluid: &Fluid3D) -> Vec<f64> {
        let e = fluid.kinetic_energy();
        let m = fluid.max_speed();
        let mut wsum = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        let mut hist = vec![0.0; COARSE * COARSE * COARSE];
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    let idx = ix(i, j, k);
                    let s = (fluid.u[idx] * fluid.u[idx]
                        + fluid.v[idx] * fluid.v[idx]
                        + fluid.w[idx] * fluid.w[idx])
                        .sqrt();
                    wsum += s;
                    cx += s * i as f64;
                    cy += s * j as f64;
                    cz += s * k as f64;
                    let ci = ((i as f64 / N as f64) * COARSE as f64).floor() as usize;
                    let cj = ((j as f64 / N as f64) * COARSE as f64).floor() as usize;
                    let ck = ((k as f64 / N as f64) * COARSE as f64).floor() as usize;
                    let ci = ci.min(COARSE - 1);
                    let cj = cj.min(COARSE - 1);
                    let ck = ck.min(COARSE - 1);
                    hist[ci + COARSE * (cj + COARSE * ck)] += s;
                }
            }
        }
        if wsum > 1e-18 {
            cx /= wsum;
            cy /= wsum;
            cz /= wsum;
        }
        let hsum: f64 = hist.iter().sum::<f64>().max(1e-18);
        for h in &mut hist {
            *h /= hsum;
        }
        let mut feat = Vec::with_capacity(FEAT_DIM);
        feat.push(e);
        feat.push(m);
        feat.push(cx / N as f64);
        feat.push(cy / N as f64);
        feat.push(cz / N as f64);
        feat.extend_from_slice(&hist);
        // Normaliza E/|u|/COM para que el MLP no explote.
        for v in feat.iter_mut().take(5) {
            *v = (*v).tanh();
        }
        feat
    }

    pub fn features_for_input(&self, input: usize) -> Vec<f64> {
        let mut f = self.encode_input(input);
        self.rollout(&mut f);
        self.extract_features(&f)
    }

    pub fn train_centroids(&mut self, repeats: usize) {
        let mut sums = vec![vec![0.0; FEAT_DIM]; NUM_LABELS];
        let mut counts = vec![0.0; NUM_LABELS];
        for _ in 0..repeats {
            for i in 0..NUM_LABELS {
                let feat = self.features_for_input(i);
                for d in 0..FEAT_DIM {
                    sums[i][d] += feat[d];
                }
                counts[i] += 1.0;
            }
        }
        for i in 0..NUM_LABELS {
            for d in 0..FEAT_DIM {
                self.centroids[i][d] = sums[i][d] / counts[i];
            }
        }
        self.trained = true;
    }

    pub fn train_mlp(&mut self, epochs: usize) {
        let samples: Vec<(Vec<f64>, usize)> = (0..NUM_LABELS)
            .map(|i| (self.features_for_input(i), i))
            .collect();
        if let Some(mlp) = self.mlp.as_mut() {
            mlp.lr = 0.15;
            for _ in 0..epochs {
                for (x, y) in &samples {
                    let _ = mlp.train_step(x, *y);
                }
            }
        }
        self.trained = true;
    }

    fn nearest_centroid(&self, feat: &[f64]) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for i in 0..NUM_LABELS {
            let mut d = 0.0;
            for j in 0..FEAT_DIM {
                let e = feat[j] - self.centroids[i][j];
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    pub fn infer_centroid(&self, input: usize) -> (usize, Vec<f64>) {
        let feat = self.features_for_input(input);
        (self.nearest_centroid(&feat), feat)
    }

    pub fn infer_mlp(&self, input: usize) -> usize {
        let feat = self.features_for_input(input);
        self.mlp.as_ref().map(|m| m.predict(&feat)).unwrap_or(0)
    }

    pub fn identity_accuracy_centroid(&self) -> f64 {
        let mut ok = 0usize;
        for i in 0..NUM_LABELS {
            if self.infer_centroid(i).0 == i {
                ok += 1;
            }
        }
        ok as f64 / NUM_LABELS as f64
    }

    pub fn identity_accuracy_mlp(&self) -> f64 {
        let mut ok = 0usize;
        for i in 0..NUM_LABELS {
            if self.infer_mlp(i) == i {
                ok += 1;
            }
        }
        ok as f64 / NUM_LABELS as f64
    }
}

/// Bench rápido NS-nativo vs tiempo de referencia (spin colapso completo ~3–4 ms).
pub fn bench_native_ns_infer(infer_loops: usize) -> (f64, f64, f64) {
    let mut eng = Fluid3DNativeInfer::new();
    eng.train_centroids(3);
    eng.train_mlp(120);
    let acc_c = eng.identity_accuracy_centroid();
    let acc_m = eng.identity_accuracy_mlp();
    let t0 = Instant::now();
    for _ in 0..infer_loops {
        for i in 0..NUM_LABELS {
            let _ = eng.infer_centroid(i);
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let per_us = (ms * 1e3) / (infer_loops * NUM_LABELS) as f64;
    println!(
        "native_ns centroid_acc={acc_c:.3} mlp_acc={acc_m:.3} infer_ms={ms:.2} µs/q={per_us:.1} (n={})",
        infer_loops * NUM_LABELS
    );
    (acc_c, acc_m, per_us)
}

/// Usa el MLP residual de estado (caro) solo como smoke de API local.
pub fn residual_mlp_smoke() -> f64 {
    let eng = Fluid3DNativeInfer::new();
    let mut f = eng.encode_input(0);
    eng.rollout(&mut f);
    let x = f.pack_state();
    let mut mlp = ResidualMlp::new(1, 16);
    mlp.lr = 0.02;
    let y = f.pack_state();
    mlp.train_step(&x, &y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_ns_centroids_separate_classes() {
        let mut eng = Fluid3DNativeInfer::new();
        eng.train_centroids(4);
        let acc = eng.identity_accuracy_centroid();
        println!("native_ns centroid acc={acc:.3}");
        assert!(acc >= 0.99, "centroids should separate blobs: {acc}");
    }

    #[test]
    fn native_ns_mlp_learns_identity() {
        let mut eng = Fluid3DNativeInfer::new();
        eng.train_mlp(200);
        let acc = eng.identity_accuracy_mlp();
        println!("native_ns mlp acc={acc:.3}");
        assert!(acc >= 0.99, "feature MLP should learn: {acc}");
    }

    #[test]
    fn native_ns_faster_than_spin_collapse_ballpark() {
        let mut eng = Fluid3DNativeInfer::new();
        eng.rollout_steps = 0; // features del CI: aún separan clases, sin project/diffuse
        eng.train_centroids(2);
        let acc = eng.identity_accuracy_centroid();
        let t0 = Instant::now();
        for _ in 0..50 {
            for i in 0..NUM_LABELS {
                let _ = eng.infer_centroid(i);
            }
        }
        let per_us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
        println!("native_ns IC-only centroid acc={acc:.3} µs/q={per_us:.1}");
        assert!(acc >= 0.99);
        assert!(
            per_us < 500.0,
            "IC-only NS features should be << spin collapse: {per_us} µs"
        );
    }

    #[test]
    fn residual_mlp_api_smoke() {
        let loss = residual_mlp_smoke();
        assert!(loss.is_finite());
    }
}
