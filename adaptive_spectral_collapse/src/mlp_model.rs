//! Conventional end-to-end neural baseline (`mlp_conventional`) with matched/generous
//! parameter capacity (`1,118` base params vs `586` for spectral arms) and matched compute budget.
//!
//! Architecture: `16 -> 20 (ReLU) -> 16 (ReLU)` with multi-label abduction head `16 -> K` (BCE)
//! and nonlinear outcome prediction head `16 -> 2` (MSE).

use std::f64::consts::PI;

use crate::rng::{AdamState, Rng};
use crate::world::{UnknownWorld, DIM_X, K_BASE};

pub const MLP_H1: usize = 20;
pub const MLP_H2: usize = 16;

#[derive(Clone, Debug)]
pub struct MlpModel {
    pub k_classes: usize,
    /// Flat parameter vector:
    /// `W1` (`20 x 16 = 320`), `b1` (`20`),
    /// `W2` (`16 x 20 = 320`), `b2` (`16`),
    /// `W_pr` (`2 x 16 = 32`), `b_pr` (`2`),
    /// `W_ab` (`K x 16`), `b_ab` (`K`).
    pub params: Vec<f64>,
}

impl MlpModel {
    pub const OFF_W1: usize = 0;
    pub const OFF_B1: usize = MLP_H1 * DIM_X; // 320
    pub const OFF_W2: usize = Self::OFF_B1 + MLP_H1; // 340
    pub const OFF_B2: usize = Self::OFF_W2 + MLP_H2 * MLP_H1; // 660
    pub const OFF_WPR: usize = Self::OFF_B2 + MLP_H2; // 676
    pub const OFF_BPR: usize = Self::OFF_WPR + 2 * MLP_H2; // 708
    pub const OFF_WAB: usize = Self::OFF_BPR + 2; // 710

    pub fn param_count(k_classes: usize) -> usize {
        Self::OFF_WAB + k_classes * MLP_H2 + k_classes
    }

    pub fn off_bab(&self) -> usize {
        Self::OFF_WAB + self.k_classes * MLP_H2
    }

    pub fn new(k_classes: usize, rng: &mut Rng) -> Self {
        let n = Self::param_count(k_classes);
        let mut p = vec![0.0_f64; n];
        let s1 = (2.0 / (DIM_X as f64)).sqrt();
        for v in &mut p[Self::OFF_W1..Self::OFF_B1] {
            *v = s1 * rng.normal();
        }
        let s2 = (2.0 / (MLP_H1 as f64)).sqrt();
        for v in &mut p[Self::OFF_W2..Self::OFF_B2] {
            *v = s2 * rng.normal();
        }
        let sp = (1.0 / (MLP_H2 as f64)).sqrt();
        for v in &mut p[Self::OFF_WPR..Self::OFF_BPR] {
            *v = sp * rng.normal();
        }
        let off_bab = Self::OFF_WAB + k_classes * MLP_H2;
        for v in &mut p[Self::OFF_WAB..off_bab] {
            *v = sp * rng.normal();
        }
        Self { k_classes, params: p }
    }

    /// Expands abduction output head from `k_classes` to `new_k` preserving all existing weights.
    pub fn expand_classes(&mut self, new_k: usize, rng: &mut Rng) {
        assert!(new_k > self.k_classes);
        let old_k = self.k_classes;
        let old_off_bab = self.off_bab();
        let new_off_bab = Self::OFF_WAB + new_k * MLP_H2;
        let mut new_p = vec![0.0_f64; Self::param_count(new_k)];
        // Copy trunk + prediction head + old W_ab
        new_p[..old_off_bab].copy_from_slice(&self.params[..old_off_bab]);
        // Initialize new rows of W_ab
        let sp = (1.0 / (MLP_H2 as f64)).sqrt();
        for v in &mut new_p[old_off_bab..new_off_bab] {
            *v = sp * rng.normal();
        }
        // Copy old b_ab
        new_p[new_off_bab..(new_off_bab + old_k)]
            .copy_from_slice(&self.params[old_off_bab..(old_off_bab + old_k)]);
        // Initialize new biases slightly negative to match low base prior
        for v in &mut new_p[(new_off_bab + old_k)..] {
            *v = -1.5;
        }
        self.k_classes = new_k;
        self.params = new_p;
    }

    #[inline]
    pub fn forward(&self, x: &[f64; DIM_X]) -> ([f64; MLP_H1], [f64; MLP_H2], [f64; 2], Vec<f64>) {
        let mut h1 = [0.0_f64; MLP_H1];
        for i in 0..MLP_H1 {
            let mut s = self.params[Self::OFF_B1 + i];
            let row = Self::OFF_W1 + i * DIM_X;
            for c in 0..DIM_X {
                s += self.params[row + c] * x[c];
            }
            h1[i] = s.max(0.0);
        }
        let mut h2 = [0.0_f64; MLP_H2];
        for j in 0..MLP_H2 {
            let mut s = self.params[Self::OFF_B2 + j];
            let row = Self::OFF_W2 + j * MLP_H1;
            for i in 0..MLP_H1 {
                s += self.params[row + i] * h1[i];
            }
            h2[j] = s.max(0.0);
        }
        let mut y = [0.0_f64; 2];
        for o in 0..2 {
            let mut s = self.params[Self::OFF_BPR + o];
            let row = Self::OFF_WPR + o * MLP_H2;
            for j in 0..MLP_H2 {
                s += self.params[row + j] * h2[j];
            }
            y[o] = s;
        }
        let off_bab = self.off_bab();
        let mut logits = vec![0.0_f64; self.k_classes];
        for k in 0..self.k_classes {
            let mut s = self.params[off_bab + k];
            let row = Self::OFF_WAB + k * MLP_H2;
            for j in 0..MLP_H2 {
                s += self.params[row + j] * h2[j];
            }
            logits[k] = s;
        }
        (h1, h2, y, logits)
    }

    /// Predicts top-2 distinct causes `(a_hat, b_hat)` and nonlinear outcome `(y1, y2, cls)` directly from `x_obs`.
    pub fn infer(&self, x_obs: &[f64; DIM_X]) -> ((usize, usize), (f64, f64, usize)) {
        let (_, _, y, logits) = self.forward(x_obs);
        let mut best1 = (0usize, f64::NEG_INFINITY);
        let mut best2 = (1usize, f64::NEG_INFINITY);
        for (k, &lv) in logits.iter().enumerate() {
            if lv > best1.1 {
                best2 = best1;
                best1 = (k, lv);
            } else if lv > best2.1 {
                best2 = (k, lv);
            }
        }
        let (pa, pb) = if best1.0 < best2.0 {
            (best1.0, best2.0)
        } else {
            (best2.0, best1.0)
        };
        let cls = (if y[0] >= 0.0 { 2 } else { 0 }) + (if y[1] >= 0.0 { 1 } else { 0 });
        ((pa, pb), (y[0], y[1], cls))
    }

    /// Accumulates analytical gradient for one sample (optionally with pair outcome target `(ty1, ty2)`).
    pub fn loss_and_grad_sample(
        &self,
        x: &[f64; DIM_X],
        active_labels: &[usize],
        outcome_target: Option<(f64, f64)>,
        grads: &mut [f64],
        scale: f64,
    ) -> f64 {
        let (h1, h2, y, logits) = self.forward(x);
        let off_bab = self.off_bab();

        let mut loss = 0.0;
        let mut dh2 = [0.0_f64; MLP_H2];

        // 1. Multi-label BCE on abduction logits
        let k_f = self.k_classes as f64;
        for k in 0..self.k_classes {
            let t = if active_labels.contains(&k) { 1.0 } else { 0.0 };
            let z = logits[k];
            // Numerically stable softplus(z) - t*z
            let bce = if z > 0.0 {
                z + (-z).exp().ln_1p() - t * z
            } else {
                z.exp().ln_1p() - t * z
            };
            // Upweight positive targets slightly for class balance (2 active out of 24)
            let w_pos = if t > 0.5 { 4.0 } else { 1.0 };
            loss += w_pos * bce / k_f;
            let sig = 1.0 / (1.0 + (-z).exp());
            let dz = scale * w_pos * (sig - t) / k_f;
            grads[off_bab + k] += dz;
            let row = Self::OFF_WAB + k * MLP_H2;
            for j in 0..MLP_H2 {
                grads[row + j] += dz * h2[j];
                dh2[j] += dz * self.params[row + j];
            }
        }

        // 2. MSE on nonlinear outcome prediction head (when observing a pair)
        if let Some((ty1, ty2)) = outcome_target {
            let e0 = y[0] - ty1;
            let e1 = y[1] - ty2;
            loss += 0.5 * (e0 * e0 + e1 * e1);
            let dy = [scale * e0, scale * e1];
            for o in 0..2 {
                grads[Self::OFF_BPR + o] += dy[o];
                let row = Self::OFF_WPR + o * MLP_H2;
                for j in 0..MLP_H2 {
                    grads[row + j] += dy[o] * h2[j];
                    dh2[j] += dy[o] * self.params[row + j];
                }
            }
        }

        // Backprop through H2 (ReLU)
        let mut dh1 = [0.0_f64; MLP_H1];
        for j in 0..MLP_H2 {
            if h2[j] <= 0.0 {
                continue;
            }
            let dz = dh2[j];
            grads[Self::OFF_B2 + j] += dz;
            let row = Self::OFF_W2 + j * MLP_H1;
            for i in 0..MLP_H1 {
                grads[row + i] += dz * h1[i];
                dh1[i] += dz * self.params[row + i];
            }
        }

        // Backprop through H1 (ReLU)
        for i in 0..MLP_H1 {
            if h1[i] <= 0.0 {
                continue;
            }
            let dz = dh1[i];
            grads[Self::OFF_B1 + i] += dz;
            let row = Self::OFF_W1 + i * DIM_X;
            for c in 0..DIM_X {
                grads[row + c] += dz * x[c];
            }
        }

        scale * loss
    }

    /// Trains `mlp_conventional` end-to-end with Adam on all 24 singletons + 110 `TRAIN_PAIRS`
    /// (generous matched step budget: `800` epochs with cosine decay & L2 weight decay).
    pub fn train_base(&mut self, world: &UnknownWorld, rng: &mut Rng) {
        let steps = 800usize;
        let mut adam = AdamState::new(self.params.len());
        let mut grads = vec![0.0_f64; self.params.len()];
        let sigma_train = 0.015_f64;
        let n_tot = (K_BASE + world.train_pairs.len()) as f64;
        let scale = 1.0 / n_tot;

        for step in 0..steps {
            grads.fill(0.0);
            // Atomic prototypes (helps MLP learn individual concept signatures cleanly)
            for k in 0..K_BASE {
                let mut x = world.xs[k];
                for xi in &mut x {
                    *xi += sigma_train * rng.normal();
                }
                self.loss_and_grad_sample(&x, &[k], None, &mut grads, scale * 0.5);
            }
            // Training pairs (110 seen pairs)
            for &(a, b) in &world.train_pairs {
                let x = world.observe_pair(a, b, sigma_train, rng);
                let (ty1, ty2, _) = world.eval_outcome(a, b);
                self.loss_and_grad_sample(&x, &[a, b], Some((ty1, ty2)), &mut grads, scale);
            }
            let lr_t = 0.012 * (0.5 * (1.0 + (PI * (step as f64) / (steps as f64)).cos())).max(0.08);
            adam.update(&mut self.params, &grads, lr_t, 1e-4);
        }
    }

    /// Suite A4: Continual adaptation to 4 novel concepts (`24..27`) with experience replay
    /// under matched continual step budget (`280` steps).
    pub fn adapt_continual(&mut self, world: &UnknownWorld, rng: &mut Rng) {
        self.expand_classes(world.xs.len(), rng);
        let steps = 280usize;
        let mut adam = AdamState::new(self.params.len());
        let mut grads = vec![0.0_f64; self.params.len()];
        let sigma_train = 0.015_f64;

        let n_items = (world.xs.len() + world.train_pairs.len() + world.novel_train_pairs.len() * 2) as f64;
        let scale = 1.0 / n_items;

        for step in 0..steps {
            grads.fill(0.0);
            for k in 0..world.xs.len() {
                let mut x = world.xs[k];
                for xi in &mut x {
                    *xi += sigma_train * rng.normal();
                }
                let w = if k >= K_BASE { 1.0 } else { 0.4 };
                self.loss_and_grad_sample(&x, &[k], None, &mut grads, scale * w);
            }
            // Experience replay of old training pairs
            for &(a, b) in &world.train_pairs {
                let x = world.observe_pair(a, b, sigma_train, rng);
                let (ty1, ty2, _) = world.eval_outcome(a, b);
                self.loss_and_grad_sample(&x, &[a, b], Some((ty1, ty2)), &mut grads, scale);
            }
            // Novel training pairs (16 pairs, upweighted 2x identically to spectral arms)
            for &(a, b) in &world.novel_train_pairs {
                let x = world.observe_pair(a, b, sigma_train, rng);
                let (ty1, ty2, _) = world.eval_outcome(a, b);
                self.loss_and_grad_sample(&x, &[a, b], Some((ty1, ty2)), &mut grads, scale * 2.0);
            }
            let lr_t = 0.009 * (0.5 * (1.0 + (PI * (step as f64) / (steps as f64)).cos())).max(0.1);
            adam.update(&mut self.params, &grads, lr_t, 1e-4);
        }
    }
}
