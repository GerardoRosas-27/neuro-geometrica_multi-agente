//! Surrogate del colapso de spin: predice features post-colapso sin integrar NLS.
//!
//! Teacher: `SpinFluid3D` (early-exit opcional). Student: MLP sobre parámetros
//! del paquete / one-hot de clase → (peak, amp, phase bins).

use crate::fluid3d_astra::N;
use crate::spin_fluid3d::SpinFluid3D;
use crate::spin_fluid_rqm_infer::{
    CollapseFeatures, SpinFluidRqmInfer, AMP_BINS, COARSE, FEATURE_BASE, NUM_LABELS, PHASE_BINS,
    SPATIAL_FEATURES,
};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::time::Instant;

const IN_DIM: usize = NUM_LABELS; // one-hot
const OUT_DIM: usize = 5; // peak_x/N, peak_y/N, peak_z/N, amp/2, phase01

#[derive(Clone, Debug)]
pub struct CollapseSurrogate {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub hidden: usize,
    pub lr: f64,
    /// Targets teacher en OUT_DIM (para clasificar por vecino L2).
    pub teacher_targets: Vec<[f64; OUT_DIM]>,
    pub teacher_spatial: Vec<usize>,
}

impl CollapseSurrogate {
    pub fn new(seed: u64, hidden: usize) -> Self {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let s1 = (2.0 / IN_DIM as f64).sqrt();
        let s2 = (2.0 / hidden as f64).sqrt();
        let mut w1 = vec![0.0; hidden * IN_DIM];
        let mut w2 = vec![0.0; OUT_DIM * hidden];
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
            b2: vec![0.0; OUT_DIM],
            hidden,
            lr: 0.12,
            teacher_targets: vec![[0.0; OUT_DIM]; NUM_LABELS],
            teacher_spatial: vec![0; NUM_LABELS],
        }
    }

    fn one_hot(input: usize) -> [f64; IN_DIM] {
        let mut x = [0.0; IN_DIM];
        x[input % NUM_LABELS] = 1.0;
        x
    }

    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut h_pre = vec![0.0; self.hidden];
        let mut h = vec![0.0; self.hidden];
        for j in 0..self.hidden {
            let mut s = self.b1[j];
            for i in 0..IN_DIM {
                s += self.w1[j * IN_DIM + i] * x[i];
            }
            h_pre[j] = s;
            h[j] = s.max(0.0);
        }
        let mut y = vec![0.0; OUT_DIM];
        for o in 0..OUT_DIM {
            let mut s = self.b2[o];
            for j in 0..self.hidden {
                s += self.w2[o * self.hidden + j] * h[j];
            }
            y[o] = s;
        }
        (h_pre, h, y)
    }

    pub fn predict_raw(&self, input: usize) -> [f64; OUT_DIM] {
        let x = Self::one_hot(input);
        let (_, _, y) = self.forward(&x);
        let mut out = [0.0; OUT_DIM];
        out.copy_from_slice(&y[..OUT_DIM]);
        out
    }

    fn target_from_features(f: &CollapseFeatures) -> [f64; OUT_DIM] {
        let phase01 = (f.phase + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
        [
            f.peak[0] / N as f64,
            f.peak[1] / N as f64,
            f.peak[2] / N as f64,
            (f.max_amp / 2.0).clamp(0.0, 1.5),
            phase01.clamp(0.0, 1.0),
        ]
    }

    pub fn train_step(&mut self, input: usize, target: &[f64; OUT_DIM]) -> f64 {
        let x = Self::one_hot(input);
        let (h_pre, h, y) = self.forward(&x);
        let mut loss = 0.0;
        let mut dy = vec![0.0; OUT_DIM];
        for o in 0..OUT_DIM {
            let e = y[o] - target[o];
            loss += e * e;
            dy[o] = 2.0 * e / OUT_DIM as f64;
        }
        let mut dh = vec![0.0; self.hidden];
        for o in 0..OUT_DIM {
            for j in 0..self.hidden {
                dh[j] += self.w2[o * self.hidden + j] * dy[o];
                self.w2[o * self.hidden + j] -= self.lr * dy[o] * h[j];
            }
            self.b2[o] -= self.lr * dy[o];
        }
        for j in 0..self.hidden {
            let g = if h_pre[j] > 0.0 { dh[j] } else { 0.0 };
            for i in 0..IN_DIM {
                self.w1[j * IN_DIM + i] -= self.lr * g * x[i];
            }
            self.b1[j] -= self.lr * g;
        }
        loss / OUT_DIM as f64
    }

    pub fn features_from_prediction(&self, input: usize) -> CollapseFeatures {
        let y = self.predict_raw(input);
        let peak = [y[0] * N as f64, y[1] * N as f64, y[2] * N as f64];
        let max_amp = y[3] * 2.0;
        let phase = y[4] * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
        let cx = ((peak[0] / N as f64) * COARSE as f64).floor() as usize;
        let cy = ((peak[1] / N as f64) * COARSE as f64).floor() as usize;
        let cz = ((peak[2] / N as f64) * COARSE as f64).floor() as usize;
        let cx = cx.min(COARSE - 1);
        let cy = cy.min(COARSE - 1);
        let cz = cz.min(COARSE - 1);
        let spatial = FEATURE_BASE + cx + COARSE * (cy + COARSE * cz);
        let amp_bin = ((max_amp / 2.0) * AMP_BINS as f64).floor() as usize;
        let amp_bin = amp_bin.min(AMP_BINS - 1);
        let phase_bin = (y[4].clamp(0.0, 0.999) * PHASE_BINS as f64).floor() as usize;
        CollapseFeatures {
            peak,
            max_amp,
            phase,
            norm: 0.0,
            spatial_node: spatial,
            amp_node: FEATURE_BASE + SPATIAL_FEATURES + amp_bin,
            phase_node: FEATURE_BASE + SPATIAL_FEATURES + AMP_BINS + phase_bin,
        }
    }

    /// Entrena con teacher (colapso early-exit).
    pub fn distill_from_teacher(&mut self, epochs: usize, use_early: bool) {
        let teacher = SpinFluidRqmInfer::new();
        let mut dataset = Vec::new();
        for i in 0..NUM_LABELS {
            let mut spin = teacher.encode_input(i);
            if use_early {
                let _ = spin.collapse_early(teacher.collapse_steps, 1e-4, 3, None, true, 5);
            } else {
                teacher.collapse(&mut spin);
            }
            let feat = teacher.extract_features(&spin);
            let tgt = Self::target_from_features(&feat);
            self.teacher_targets[i] = tgt;
            self.teacher_spatial[i] = feat.spatial_node;
            dataset.push((i, tgt));
        }
        for _ in 0..epochs {
            for (i, tgt) in &dataset {
                let _ = self.train_step(*i, tgt);
            }
        }
    }

    /// Clasifica por vecino L2 al target teacher (robusto a bins espaciales colisionados).
    pub fn infer_label(&self, input: usize) -> usize {
        let y = self.predict_raw(input);
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for i in 0..NUM_LABELS {
            let mut d = 0.0;
            for o in 0..OUT_DIM {
                let e = y[o] - self.teacher_targets[i][o];
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    pub fn identity_accuracy(&self) -> f64 {
        let mut ok = 0usize;
        for i in 0..NUM_LABELS {
            if self.infer_label(i) == i {
                ok += 1;
            }
        }
        ok as f64 / NUM_LABELS as f64
    }
}

pub fn bench_surrogate_vs_teacher(infer_loops: usize) -> (f64, f64, f64, f64) {
    let mut surr = CollapseSurrogate::new(0x50A7, 24);
    surr.distill_from_teacher(400, true);
    let acc = surr.identity_accuracy();

    let t0 = Instant::now();
    for _ in 0..infer_loops {
        for i in 0..NUM_LABELS {
            let _ = surr.infer_label(i);
        }
    }
    let surr_ms = t0.elapsed().as_secs_f64() * 1e3;
    let surr_us = (surr_ms * 1e3) / (infer_loops * NUM_LABELS) as f64;

    let teacher = SpinFluidRqmInfer::new();
    let t1 = Instant::now();
    for _ in 0..infer_loops {
        for i in 0..NUM_LABELS {
            let mut spin = teacher.encode_input(i);
            let _ = spin.collapse_early(28, 1e-4, 3, None, true, 5);
            let _ = teacher.extract_features(&spin);
        }
    }
    let teach_ms = t1.elapsed().as_secs_f64() * 1e3;
    let teach_us = (teach_ms * 1e3) / (infer_loops * NUM_LABELS) as f64;

    println!(
        "surrogate acc={acc:.3} µs/q={surr_us:.2} | teacher_early µs/q={teach_us:.1} speedup×{:.1}",
        teach_us / surr_us.max(1e-9)
    );
    (acc, surr_us, teach_us, teach_us / surr_us.max(1e-9))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_distills_and_classifies() {
        let mut s = CollapseSurrogate::new(0x50A7, 24);
        s.distill_from_teacher(400, true);
        let acc = s.identity_accuracy();
        println!("surrogate identity acc={acc:.3} teacher_spatial={:?}", s.teacher_spatial);
        assert!(acc >= 0.99, "surrogate should match teacher spatial map: {acc}");
    }

    #[test]
    fn surrogate_much_faster_than_teacher() {
        let (acc, surr_us, teach_us, speedup) = bench_surrogate_vs_teacher(30);
        assert!(acc >= 0.99);
        assert!(
            speedup > 20.0,
            "surrogate should be >>20× teacher: surr={surr_us} teach={teach_us} ×{speedup}"
        );
    }

    #[test]
    fn teacher_early_exit_runs() {
        let mut spin = SpinFluid3D::new(0.35, 2.5, 0.02);
        let mid = (N / 2) as f64;
        spin.add_gaussian_packet([mid, mid, mid], 1.6, 0.8, [0.0, 0.0, 0.0]);
        let r = spin.collapse_early(40, 1e-4, 3, None, true, 5);
        assert!(r.steps_run <= 40);
        assert!(r.amp_end.is_finite());
    }
}
