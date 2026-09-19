//! Experimento aislado: fluidos 3D incompresibles + modelo aprendido.
//!
//! Forma matemática: Navier–Stokes 3D incompresible
//!   ∂u/∂t + (u·∇)u = −∇p + ν Δu + f ,   ∇·u = 0
//!
//! Contexto: GPT‑6 Astra formalizó en Lean (sep 2026) una singularidad
//! de NS 3D (vórtice que se estira). Aquí **no** se demuestra el blowup.
//! Se simula la dinámica NS con el método Stable Fluids (Stam 1999) y se
//! entrena un residual sobre trayectorias, con fuerza que estira un vórtice.
//!
//! Experimento autocontenido: cero acoplos al sustrato de campo / Gemma.


use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

pub const N: usize = 16;
pub const CELLS: usize = N * N * N;
#[inline]
fn ix(i: usize, j: usize, k: usize) -> usize {
    i + N * (j + N * k)
}

#[inline]
fn clamp_idx(v: f64) -> usize {
    let x = v.floor() as isize;
    x.clamp(0, (N - 1) as isize) as usize
}

/// Campos de velocidad (MAC simplificado: colocalizados en centros).
#[derive(Clone, Debug)]
pub struct Fluid3D {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
    pub p: Vec<f64>,
    pub visc: f64,
    pub dt: f64,
}

impl Fluid3D {
    pub fn new(visc: f64, dt: f64) -> Self {
        Self {
            u: vec![0.0; CELLS],
            v: vec![0.0; CELLS],
            w: vec![0.0; CELLS],
            p: vec![0.0; CELLS],
            visc,
            dt,
        }
    }

    pub fn kinetic_energy(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..CELLS {
            e += self.u[i] * self.u[i] + self.v[i] * self.v[i] + self.w[i] * self.w[i];
        }
        0.5 * e / CELLS as f64
    }

    pub fn max_speed(&self) -> f64 {
        let mut m = 0.0;
        for i in 0..CELLS {
            let s = (self.u[i] * self.u[i] + self.v[i] * self.v[i] + self.w[i] * self.w[i]).sqrt();
            if s > m {
                m = s;
            }
        }
        m
    }

    pub fn divergence_l2(&self) -> f64 {
        let mut acc = 0.0;
        let mut n = 0.0f64;
        for k in 1..N - 1 {
            for j in 1..N - 1 {
                for i in 1..N - 1 {
                    let du = self.u[ix(i + 1, j, k)] - self.u[ix(i - 1, j, k)];
                    let dv = self.v[ix(i, j + 1, k)] - self.v[ix(i, j - 1, k)];
                    let dw = self.w[ix(i, j, k + 1)] - self.w[ix(i, j, k - 1)];
                    let d = 0.5 * (du + dv + dw);
                    acc += d * d;
                    n += 1.0;
                }
            }
        }
        (acc / n.max(1.0)).sqrt()
    }

    /// Fuerza tipo vórtice que se estira (inspirada en la geometría del blowup NS).
    /// No pretende ser la fuerza analítica del proof; solo un régimen de estiramiento.
    pub fn apply_stretching_vortex_force(&mut self, amp: f64, t: f64) {
        let c = (N as f64 - 1.0) * 0.5;
        let grow = 1.0 + 0.15 * t;
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    let x = (i as f64 - c) / c;
                    let y = (j as f64 - c) / c;
                    let z = (k as f64 - c) / c;
                    let r2 = x * x + y * y;
                    let envelope = (-8.0 * r2 - 2.0 * z * z).exp();
                    let idx = ix(i, j, k);
                    // Rotación azimutal + estiramiento axial.
                    self.u[idx] += self.dt * amp * grow * (-y) * envelope;
                    self.v[idx] += self.dt * amp * grow * (x) * envelope;
                    self.w[idx] += self.dt * amp * grow * (-0.35 * z) * envelope;
                }
            }
        }
    }

    fn set_bnd_vel(&mut self) {
        // Paredes: velocidad normal ~ 0 (caja cerrada).
        for j in 0..N {
            for k in 0..N {
                self.u[ix(0, j, k)] = 0.0;
                self.u[ix(N - 1, j, k)] = 0.0;
            }
        }
        for i in 0..N {
            for k in 0..N {
                self.v[ix(i, 0, k)] = 0.0;
                self.v[ix(i, N - 1, k)] = 0.0;
            }
        }
        for i in 0..N {
            for j in 0..N {
                self.w[ix(i, j, 0)] = 0.0;
                self.w[ix(i, j, N - 1)] = 0.0;
            }
        }
    }

    fn diffuse(x: &mut [f64], x0: &[f64], diff: f64, dt: f64, iters: usize) {
        let a = dt * diff * ((N - 2) * (N - 2)) as f64;
        for _ in 0..iters {
            for k in 1..N - 1 {
                for j in 1..N - 1 {
                    for i in 1..N - 1 {
                        let s = x[ix(i - 1, j, k)]
                            + x[ix(i + 1, j, k)]
                            + x[ix(i, j - 1, k)]
                            + x[ix(i, j + 1, k)]
                            + x[ix(i, j, k - 1)]
                            + x[ix(i, j, k + 1)];
                        x[ix(i, j, k)] = (x0[ix(i, j, k)] + a * s) / (1.0 + 6.0 * a);
                    }
                }
            }
        }
    }

    fn advect(d: &mut [f64], d0: &[f64], u: &[f64], v: &[f64], w: &[f64], dt: f64) {
        let dt0 = dt * (N - 2) as f64;
        for k in 1..N - 1 {
            for j in 1..N - 1 {
                for i in 1..N - 1 {
                    let mut x = i as f64 - dt0 * u[ix(i, j, k)];
                    let mut y = j as f64 - dt0 * v[ix(i, j, k)];
                    let mut z = k as f64 - dt0 * w[ix(i, j, k)];
                    x = x.clamp(0.5, (N as f64) - 1.5);
                    y = y.clamp(0.5, (N as f64) - 1.5);
                    z = z.clamp(0.5, (N as f64) - 1.5);
                    let i0 = clamp_idx(x);
                    let i1 = (i0 + 1).min(N - 1);
                    let j0 = clamp_idx(y);
                    let j1 = (j0 + 1).min(N - 1);
                    let k0 = clamp_idx(z);
                    let k1 = (k0 + 1).min(N - 1);
                    let s1 = x - i0 as f64;
                    let s0 = 1.0 - s1;
                    let t1 = y - j0 as f64;
                    let t0 = 1.0 - t1;
                    let u1 = z - k0 as f64;
                    let u0 = 1.0 - u1;
                    d[ix(i, j, k)] = s0
                        * (t0 * (u0 * d0[ix(i0, j0, k0)] + u1 * d0[ix(i0, j0, k1)])
                            + t1 * (u0 * d0[ix(i0, j1, k0)] + u1 * d0[ix(i0, j1, k1)]))
                        + s1 * (t0 * (u0 * d0[ix(i1, j0, k0)] + u1 * d0[ix(i1, j0, k1)])
                            + t1 * (u0 * d0[ix(i1, j1, k0)] + u1 * d0[ix(i1, j1, k1)]));
                }
            }
        }
    }

    fn project(&mut self, iters: usize) {
        let mut div = vec![0.0; CELLS];
        for k in 1..N - 1 {
            for j in 1..N - 1 {
                for i in 1..N - 1 {
                    div[ix(i, j, k)] = -0.5
                        * (self.u[ix(i + 1, j, k)] - self.u[ix(i - 1, j, k)]
                            + self.v[ix(i, j + 1, k)]
                            - self.v[ix(i, j - 1, k)]
                            + self.w[ix(i, j, k + 1)]
                            - self.w[ix(i, j, k - 1)])
                        / N as f64;
                    self.p[ix(i, j, k)] = 0.0;
                }
            }
        }
        for _ in 0..iters {
            for k in 1..N - 1 {
                for j in 1..N - 1 {
                    for i in 1..N - 1 {
                        self.p[ix(i, j, k)] = (div[ix(i, j, k)]
                            + self.p[ix(i - 1, j, k)]
                            + self.p[ix(i + 1, j, k)]
                            + self.p[ix(i, j - 1, k)]
                            + self.p[ix(i, j + 1, k)]
                            + self.p[ix(i, j, k - 1)]
                            + self.p[ix(i, j, k + 1)])
                            / 6.0;
                    }
                }
            }
        }
        for k in 1..N - 1 {
            for j in 1..N - 1 {
                for i in 1..N - 1 {
                    self.u[ix(i, j, k)] -=
                        0.5 * N as f64 * (self.p[ix(i + 1, j, k)] - self.p[ix(i - 1, j, k)]);
                    self.v[ix(i, j, k)] -=
                        0.5 * N as f64 * (self.p[ix(i, j + 1, k)] - self.p[ix(i, j - 1, k)]);
                    self.w[ix(i, j, k)] -=
                        0.5 * N as f64 * (self.p[ix(i, j, k + 1)] - self.p[ix(i, j, k - 1)]);
                }
            }
        }
        self.set_bnd_vel();
    }

    /// Un paso Stable Fluids: force → diffuse → advect → project.
    pub fn step(&mut self, force_amp: f64, t: f64) {
        self.apply_stretching_vortex_force(force_amp, t);
        self.set_bnd_vel();

        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let w0 = self.w.clone();
        Self::diffuse(&mut self.u, &u0, self.visc, self.dt, 8);
        Self::diffuse(&mut self.v, &v0, self.visc, self.dt, 8);
        Self::diffuse(&mut self.w, &w0, self.visc, self.dt, 8);
        self.project(20);

        let u0 = self.u.clone();
        let v0 = self.v.clone();
        let w0 = self.w.clone();
        Self::advect(&mut self.u, &u0, &u0, &v0, &w0, self.dt);
        Self::advect(&mut self.v, &v0, &u0, &v0, &w0, self.dt);
        Self::advect(&mut self.w, &w0, &u0, &v0, &w0, self.dt);
        self.project(20);
    }

    pub fn pack_state(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(3 * CELLS);
        out.extend_from_slice(&self.u);
        out.extend_from_slice(&self.v);
        out.extend_from_slice(&self.w);
        out
    }

    pub fn unpack_into(&mut self, state: &[f64]) {
        assert_eq!(state.len(), 3 * CELLS);
        self.u.copy_from_slice(&state[0..CELLS]);
        self.v.copy_from_slice(&state[CELLS..2 * CELLS]);
        self.w.copy_from_slice(&state[2 * CELLS..]);
    }
}

/// MLP residual: predice Δ(u,v,w) a partir del estado. Sin candle: Vec puro.
#[derive(Clone, Debug)]
pub struct ResidualMlp {
    pub in_dim: usize,
    pub hidden: usize,
    pub out_dim: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub lr: f64,
}

impl ResidualMlp {
    pub fn new(seed: u64, hidden: usize) -> Self {
        let in_dim = 3 * CELLS;
        let out_dim = 3 * CELLS;
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let s1 = (2.0 / in_dim as f64).sqrt();
        let s2 = (2.0 / hidden as f64).sqrt();
        let mut w1 = vec![0.0; hidden * in_dim];
        let mut w2 = vec![0.0; out_dim * hidden];
        for v in w1.iter_mut() {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * s1;
        }
        for v in w2.iter_mut() {
            *v = (rng.gen::<f64>() * 2.0 - 1.0) * s2;
        }
        Self {
            in_dim,
            hidden,
            out_dim,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: vec![0.0; out_dim],
            lr: 0.01,
        }
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut h = vec![0.0; self.hidden];
        for j in 0..self.hidden {
            let mut s = self.b1[j];
            for i in 0..self.in_dim {
                s += self.w1[j * self.in_dim + i] * x[i];
            }
            h[j] = Self::relu(s);
        }
        let mut y = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let mut s = self.b2[o];
            for j in 0..self.hidden {
                s += self.w2[o * self.hidden + j] * h[j];
            }
            y[o] = s;
        }
        y
    }

    /// Predice el siguiente estado: x + residual.
    pub fn predict_next(&self, x: &[f64]) -> Vec<f64> {
        let d = self.forward(x);
        x.iter().zip(d.iter()).map(|(a, b)| a + b).collect()
    }

    pub fn train_step(&mut self, x: &[f64], target_next: &[f64]) -> f64 {
        let mut h_pre = vec![0.0; self.hidden];
        let mut h = vec![0.0; self.hidden];
        for j in 0..self.hidden {
            let mut s = self.b1[j];
            for i in 0..self.in_dim {
                s += self.w1[j * self.in_dim + i] * x[i];
            }
            h_pre[j] = s;
            h[j] = Self::relu(s);
        }
        let mut y = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let mut s = self.b2[o];
            for j in 0..self.hidden {
                s += self.w2[o * self.hidden + j] * h[j];
            }
            y[o] = s;
        }
        // Entrena el residual Δ = target - x
        let mut loss = 0.0;
        let mut dy = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let target_d = target_next[o] - x[o];
            let e = y[o] - target_d;
            loss += e * e;
            dy[o] = 2.0 * e / self.out_dim as f64;
        }
        loss /= self.out_dim as f64;

        // grads w2, b2
        let mut dh = vec![0.0; self.hidden];
        for o in 0..self.out_dim {
            self.b2[o] -= self.lr * dy[o];
            for j in 0..self.hidden {
                self.w2[o * self.hidden + j] -= self.lr * dy[o] * h[j];
                dh[j] += self.w2[o * self.hidden + j] * dy[o];
            }
        }
        // grads w1, b1 through ReLU
        for j in 0..self.hidden {
            let g = if h_pre[j] > 0.0 { dh[j] } else { 0.0 };
            self.b1[j] -= self.lr * g;
            for i in 0..self.in_dim {
                self.w1[j * self.in_dim + i] -= self.lr * g * x[i];
            }
        }
        loss
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimReport {
    pub steps: usize,
    pub energy_start: f64,
    pub energy_end: f64,
    pub max_speed_end: f64,
    pub div_end: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainReport {
    pub pairs: usize,
    pub epochs: usize,
    pub mse_before: f64,
    pub mse_after: f64,
    pub persistence_mse: f64,
}

pub fn run_vortex_rollout(steps: usize, seed: u64) -> (Fluid3D, SimReport) {
    let _ = seed;
    let mut fluid = Fluid3D::new(0.0008, 0.08);
    let e0 = fluid.kinetic_energy();
    for s in 0..steps {
        fluid.step(2.2, s as f64 * fluid.dt);
    }
    let report = SimReport {
        steps,
        energy_start: e0,
        energy_end: fluid.kinetic_energy(),
        max_speed_end: fluid.max_speed(),
        div_end: fluid.divergence_l2(),
    };
    (fluid, report)
}

/// Pares (u_t, u_{t+skip}) bajo forzamiento de vórtice estirado.
pub fn collect_pairs(steps: usize, force: f64, skip: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let skip = skip.max(1);
    let mut fluid = Fluid3D::new(0.0008, 0.08);
    let mut pairs = Vec::with_capacity(steps);
    let mut s = 0usize;
    for _ in 0..steps {
        let a = fluid.pack_state();
        for _ in 0..skip {
            fluid.step(force, s as f64 * fluid.dt);
            s += 1;
        }
        let b = fluid.pack_state();
        pairs.push((a, b));
    }
    pairs
}

pub fn mse_next(model: &ResidualMlp, pairs: &[(Vec<f64>, Vec<f64>)]) -> f64 {
    let mut acc = 0.0;
    for (x, y) in pairs {
        let p = model.predict_next(x);
        let mut e = 0.0;
        for i in 0..y.len() {
            let d = p[i] - y[i];
            e += d * d;
        }
        acc += e / y.len() as f64;
    }
    acc / pairs.len() as f64
}

pub fn persistence_mse(pairs: &[(Vec<f64>, Vec<f64>)]) -> f64 {
    let mut acc = 0.0;
    for (x, y) in pairs {
        let mut e = 0.0;
        for i in 0..y.len() {
            let d = x[i] - y[i];
            e += d * d;
        }
        acc += e / y.len() as f64;
    }
    acc / pairs.len() as f64
}

pub fn train_on_simulator(seed: u64) -> (ResidualMlp, TrainReport, SimReport) {
    let (_, sim) = run_vortex_rollout(40, seed);
    // Predicción multi-paso (skip=6): persistencia deja de ser casi óptima.
    let train = collect_pairs(48, 3.0, 6);
    let test = collect_pairs(24, 3.0, 6);
    let mut model = ResidualMlp::new(seed, 128);
    model.lr = 0.12;
    let mse0 = mse_next(&model, &test);
    let persist = persistence_mse(&test);
    let epochs = 40;
    for _ in 0..epochs {
        for (x, y) in &train {
            let _ = model.train_step(x, y);
        }
    }
    let mse1 = mse_next(&model, &test);
    (
        model,
        TrainReport {
            pairs: train.len(),
            epochs,
            mse_before: mse0,
            mse_after: mse1,
            persistence_mse: persist,
        },
        sim,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_keeps_divergence_small() {
        let mut f = Fluid3D::new(0.001, 0.05);
        for i in 0..CELLS {
            f.u[i] = 0.3;
            f.v[i] = -0.2;
            f.w[i] = 0.1;
        }
        f.project(40);
        let d = f.divergence_l2();
        assert!(d < 0.12, "div too large: {d}");
    }

    #[test]
    fn stretching_vortex_grows_energy() {
        let (_, r) = run_vortex_rollout(30, 1);
        println!(
            "sim steps={} E {:.6}->{:.6} |u|_max={:.4} div={:.4}",
            r.steps, r.energy_start, r.energy_end, r.max_speed_end, r.div_end
        );
        assert!(r.energy_end > r.energy_start + 1e-6);
        assert!(r.div_end < 0.08);
    }

    #[test]
    fn residual_model_beats_persistence() {
        let (_m, tr, sim) = train_on_simulator(0xA57A);
        println!(
            "train pairs={} epochs={} mse {:.6}->{:.6} persist={:.6} | sim E {:.4}->{:.4}",
            tr.pairs,
            tr.epochs,
            tr.mse_before,
            tr.mse_after,
            tr.persistence_mse,
            sim.energy_start,
            sim.energy_end
        );
        assert!(tr.mse_after < tr.mse_before, "model should learn: {:?}", tr);
        assert!(
            tr.mse_after < tr.persistence_mse,
            "should beat persistence baseline: {:?}",
            tr
        );
    }
}
