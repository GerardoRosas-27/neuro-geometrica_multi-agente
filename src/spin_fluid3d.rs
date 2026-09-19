//! Fluido de spin continuum sobre la misma rejilla que `fluid3d_astra`.
//!
//! Modelo: envolvente compleja ψ = ψ_r + i ψ_i (ondas de spin transversales)
//! con ecuación de Schrödinger no lineal enfocante (tipo Gross–Pitaevskii / NLS):
//!
//!   i ∂_t ψ = −α ∇²ψ − β |ψ|² ψ − i (u·∇)ψ
//!
//! - α > 0: dispersión (propagación)
//! - β > 0: no linealidad **enfocante** → colapso de onda si la amplitud es
//!   supercrítica en 3D
//! - u: velocidad opcional del `Fluid3D` (advección)
//!
//! Objetivo del experimento: colapsos, propagación e interferencia.
//! No acopla a RQM ni al líquido XXZ cuántico.

use crate::fluid3d_astra::{Fluid3D, CELLS, N};
use num_complex::Complex64;

#[inline]
fn ix(i: usize, j: usize, k: usize) -> usize {
    i + N * (j + N * k)
}

#[inline]
fn wrap(i: isize) -> usize {
    let n = N as isize;
    (((i % n) + n) % n) as usize
}

/// Fluido de spin: campo complejo ψ en rejilla N³ (+ magnetización longitudinal mz).
#[derive(Clone, Debug)]
pub struct SpinFluid3D {
    pub psi: Vec<Complex64>,
    /// Componente longitudinal (≈1 lejos de excitaciones fuertes).
    pub mz: Vec<f64>,
    pub alpha: f64,
    pub beta: f64,
    pub dt: f64,
    /// Amortiguamiento débil de |ψ| (estabilidad numérica; 0 = Hamiltonian).
    pub gamma: f64,
}

impl SpinFluid3D {
    pub fn new(alpha: f64, beta: f64, dt: f64) -> Self {
        Self {
            psi: vec![Complex64::new(0.0, 0.0); CELLS],
            mz: vec![1.0; CELLS],
            alpha,
            beta,
            dt,
            gamma: 0.0,
        }
    }

    pub fn clear(&mut self) {
        for z in self.psi.iter_mut() {
            *z = Complex64::new(0.0, 0.0);
        }
        for m in self.mz.iter_mut() {
            *m = 1.0;
        }
    }

    /// Paquete gaussiano con fase de momento k (propagación).
    pub fn add_gaussian_packet(
        &mut self,
        center: [f64; 3],
        sigma: f64,
        amplitude: f64,
        k: [f64; 3],
    ) {
        let inv = 1.0 / (2.0 * sigma * sigma);
        for kk in 0..N {
            for jj in 0..N {
                for ii in 0..N {
                    let x = ii as f64;
                    let y = jj as f64;
                    let z = kk as f64;
                    let dx = x - center[0];
                    let dy = y - center[1];
                    let dz = z - center[2];
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let amp = amplitude * (-r2 * inv).exp();
                    let phase = k[0] * dx + k[1] * dy + k[2] * dz;
                    let idx = ix(ii, jj, kk);
                    self.psi[idx] += Complex64::from_polar(amp, phase);
                    // Pequeña reducción de mz donde hay excitación transversal.
                    let a2 = self.psi[idx].norm_sqr();
                    self.mz[idx] = (1.0 - a2).max(0.0).sqrt();
                }
            }
        }
    }

    pub fn max_amplitude(&self) -> f64 {
        self.psi
            .iter()
            .map(|z| z.norm())
            .fold(0.0_f64, f64::max)
    }

    pub fn l2_norm_sq(&self) -> f64 {
        self.psi.iter().map(|z| z.norm_sqr()).sum::<f64>() / CELLS as f64
    }

    pub fn peak_index(&self) -> usize {
        let mut best = 0usize;
        let mut best_a = -1.0;
        for (i, z) in self.psi.iter().enumerate() {
            let a = z.norm();
            if a > best_a {
                best_a = a;
                best = i;
            }
        }
        best
    }

    pub fn peak_coords(&self) -> [f64; 3] {
        let i = self.peak_index();
        let x = (i % N) as f64;
        let y = ((i / N) % N) as f64;
        let z = (i / (N * N)) as f64;
        [x, y, z]
    }

    /// Centro de masa de |ψ|² (para medir propagación).
    pub fn center_of_mass(&self) -> [f64; 3] {
        let mut wsum = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for kk in 0..N {
            for jj in 0..N {
                for ii in 0..N {
                    let w = self.psi[ix(ii, jj, kk)].norm_sqr();
                    wsum += w;
                    cx += w * ii as f64;
                    cy += w * jj as f64;
                    cz += w * kk as f64;
                }
            }
        }
        if wsum < 1e-18 {
            return [0.0, 0.0, 0.0];
        }
        [cx / wsum, cy / wsum, cz / wsum]
    }

    fn laplacian(&self, out: &mut [Complex64]) {
        for k in 0..N {
            for j in 0..N {
                for i in 0..N {
                    let c = self.psi[ix(i, j, k)];
                    let xp = self.psi[ix(wrap(i as isize + 1), j, k)];
                    let xm = self.psi[ix(wrap(i as isize - 1), j, k)];
                    let yp = self.psi[ix(i, wrap(j as isize + 1), k)];
                    let ym = self.psi[ix(i, wrap(j as isize - 1), k)];
                    let zp = self.psi[ix(i, j, wrap(k as isize + 1))];
                    let zm = self.psi[ix(i, j, wrap(k as isize - 1))];
                    out[ix(i, j, k)] = xp + xm + yp + ym + zp + zm - c * 6.0;
                }
            }
        }
    }

    /// Gradiente central de ψ (para advección).
    fn grad_psi(&self, i: usize, j: usize, k: usize) -> [Complex64; 3] {
        let dx = (self.psi[ix(wrap(i as isize + 1), j, k)]
            - self.psi[ix(wrap(i as isize - 1), j, k)])
            * 0.5;
        let dy = (self.psi[ix(i, wrap(j as isize + 1), k)]
            - self.psi[ix(i, wrap(j as isize - 1), k)])
            * 0.5;
        let dz = (self.psi[ix(i, j, wrap(k as isize + 1))]
            - self.psi[ix(i, j, wrap(k as isize - 1))])
            * 0.5;
        [dx, dy, dz]
    }

    /// ∂t ψ = i α ∇²ψ + i β |ψ|² ψ − (u·∇)ψ − γ ψ
    /// (forma que hace i∂tψ = −α∇²ψ − β|ψ|²ψ en el límite sin advección).
    fn rhs(&self, fluid: Option<&Fluid3D>, out: &mut [Complex64]) {
        let mut lap = vec![Complex64::new(0.0, 0.0); CELLS];
        self.laplacian(&mut lap);
        let i_unit = Complex64::new(0.0, 1.0);
        for k in 0..N {
            for j in 0..N {
                for ii in 0..N {
                    let idx = ix(ii, j, k);
                    let psi = self.psi[idx];
                    let a2 = psi.norm_sqr();
                    let mut d = i_unit * (self.alpha * lap[idx] + self.beta * a2 * psi);
                    if let Some(f) = fluid {
                        let g = self.grad_psi(ii, j, k);
                        let adv = f.u[idx] * g[0] + f.v[idx] * g[1] + f.w[idx] * g[2];
                        d -= adv;
                    }
                    if self.gamma > 0.0 {
                        d -= Complex64::new(self.gamma, 0.0) * psi;
                    }
                    out[idx] = d;
                }
            }
        }
    }

    /// Un paso RK2.
    pub fn step(&mut self, fluid: Option<&Fluid3D>) {
        let mut k1 = vec![Complex64::new(0.0, 0.0); CELLS];
        let mut k2 = vec![Complex64::new(0.0, 0.0); CELLS];
        self.rhs(fluid, &mut k1);

        let backup = self.psi.clone();
        for i in 0..CELLS {
            self.psi[i] = backup[i] + k1[i] * self.dt;
        }
        self.rhs(fluid, &mut k2);
        for i in 0..CELLS {
            self.psi[i] = backup[i] + (k1[i] + k2[i]) * (0.5 * self.dt);
            let a2 = self.psi[i].norm_sqr();
            self.mz[i] = (1.0 - a2).max(0.0).sqrt();
        }
    }

    pub fn step_n(&mut self, n: usize, fluid: Option<&Fluid3D>) {
        for _ in 0..n {
            self.step(fluid);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CollapseReport {
    pub steps: usize,
    pub amp_start: f64,
    pub amp_end: f64,
    pub amp_max_seen: f64,
    pub norm_start: f64,
    pub norm_end: f64,
}

/// Rollout enfocante: paquete centrado, sin momento (colapso radial).
pub fn run_focusing_collapse(amplitude: f64, steps: usize) -> (SpinFluid3D, CollapseReport) {
    let mut s = SpinFluid3D::new(0.35, 2.5, 0.02);
    let mid = (N / 2) as f64;
    s.add_gaussian_packet([mid, mid, mid], 1.6, amplitude, [0.0, 0.0, 0.0]);
    let amp0 = s.max_amplitude();
    let n0 = s.l2_norm_sq();
    let mut amp_max = amp0;
    for _ in 0..steps {
        s.step(None);
        amp_max = amp_max.max(s.max_amplitude());
    }
    let report = CollapseReport {
        steps,
        amp_start: amp0,
        amp_end: s.max_amplitude(),
        amp_max_seen: amp_max,
        norm_start: n0,
        norm_end: s.l2_norm_sq(),
    };
    (s, report)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PropagationReport {
    pub steps: usize,
    pub com_start: [f64; 3],
    pub com_end: [f64; 3],
    pub displacement: f64,
}

pub fn run_packet_propagation(steps: usize) -> (SpinFluid3D, PropagationReport) {
    // β pequeño → casi lineal: el paquete viaja sin colapsar.
    let mut s = SpinFluid3D::new(0.55, 0.02, 0.03);
    let mid = (N / 2) as f64;
    s.add_gaussian_packet([mid, mid, mid], 1.8, 0.30, [0.85, 0.0, 0.0]);
    let c0 = s.center_of_mass();
    s.step_n(steps, None);
    let c1 = s.center_of_mass();
    let dx = c1[0] - c0[0];
    let dy = c1[1] - c0[1];
    let dz = c1[2] - c0[2];
    let report = PropagationReport {
        steps,
        com_start: c0,
        com_end: c1,
        displacement: (dx * dx + dy * dy + dz * dz).sqrt(),
    };
    (s, report)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InterferenceReport {
    pub peak_single: f64,
    pub peak_double: f64,
    pub constructive_ratio: f64,
}

/// Dos paquetes que se solapan en fase → interferencia constructiva.
pub fn run_two_packet_interference() -> (SpinFluid3D, InterferenceReport) {
    let mid = (N / 2) as f64;
    let sigma = 1.7;
    let amp = 0.28;
    let k = [0.0, 0.0, 0.0];

    let mut single = SpinFluid3D::new(0.4, 0.2, 0.02);
    single.add_gaussian_packet([mid - 1.2, mid, mid], sigma, amp, k);
    single.step_n(8, None);
    let peak_single = single.max_amplitude();

    let mut both = SpinFluid3D::new(0.4, 0.2, 0.02);
    both.add_gaussian_packet([mid - 1.2, mid, mid], sigma, amp, k);
    both.add_gaussian_packet([mid + 1.2, mid, mid], sigma, amp, k);
    both.step_n(8, None);
    let peak_double = both.max_amplitude();

    let report = InterferenceReport {
        peak_single,
        peak_double,
        constructive_ratio: if peak_single > 1e-12 {
            peak_double / peak_single
        } else {
            0.0
        },
    };
    (both, report)
}

/// Acopla un paso de NS (vórtice) + un paso de spin advectado.
pub fn run_spin_advected_by_fluid(steps: usize) -> (SpinFluid3D, Fluid3D, f64) {
    let mut fluid = Fluid3D::new(0.0008, 0.08);
    let mut spin = SpinFluid3D::new(0.35, 0.4, 0.02);
    let mid = (N / 2) as f64;
    spin.add_gaussian_packet([mid, mid, mid], 1.5, 0.4, [0.0, 0.3, 0.0]);
    let amp0 = spin.max_amplitude();
    for s in 0..steps {
        fluid.step(2.0, s as f64 * fluid.dt);
        spin.step(Some(&fluid));
    }
    (spin, fluid, amp0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_packet_propagates() {
        let (_s, r) = run_packet_propagation(40);
        println!(
            "prop steps={} com {:.2?}->{:.2?} |disp|={:.3}",
            r.steps, r.com_start, r.com_end, r.displacement
        );
        assert!(
            r.displacement > 0.45,
            "packet should move along +x: {:?}",
            r
        );
        // Principalmente en x
        let dx = (r.com_end[0] - r.com_start[0]).abs();
        let dy = (r.com_end[1] - r.com_start[1]).abs();
        assert!(dx > dy, "motion should be dominated by x: {:?}", r);
    }

    #[test]
    fn two_packets_interfere_constructively() {
        let (_s, r) = run_two_packet_interference();
        println!(
            "interfer peak_single={:.4} peak_double={:.4} ratio={:.3}",
            r.peak_single, r.peak_double, r.constructive_ratio
        );
        assert!(
            r.constructive_ratio > 1.25,
            "overlapping in-phase packets should raise the peak: {:?}",
            r
        );
    }

    #[test]
    fn supercritical_wave_collapses() {
        let (_weak, weak_r) = run_focusing_collapse(0.22, 40);
        let (_strong, strong_r) = run_focusing_collapse(0.85, 40);
        println!(
            "collapse weak amp {:.4}->{:.4} (max {:.4}) | strong {:.4}->{:.4} (max {:.4})",
            weak_r.amp_start,
            weak_r.amp_end,
            weak_r.amp_max_seen,
            strong_r.amp_start,
            strong_r.amp_end,
            strong_r.amp_max_seen
        );
        // Paquete fuerte: el pico visto debe crecer claramente (colapso enfocante).
        assert!(
            strong_r.amp_max_seen > strong_r.amp_start * 1.15,
            "supercritical packet should focus: {:?}",
            strong_r
        );
        // El fuerte enfoca más que el débil (ratio max/start).
        let rw = weak_r.amp_max_seen / weak_r.amp_start.max(1e-12);
        let rs = strong_r.amp_max_seen / strong_r.amp_start.max(1e-12);
        assert!(
            rs > rw,
            "stronger packet should focus more (rs={rs:.3} rw={rw:.3})"
        );
    }

    #[test]
    fn spin_survives_fluid_advection() {
        let (spin, fluid, amp0) = run_spin_advected_by_fluid(20);
        let amp1 = spin.max_amplitude();
        println!(
            "advect amp {:.4}->{:.4} fluid_E={:.5} div={:.4}",
            amp0,
            amp1,
            fluid.kinetic_energy(),
            fluid.divergence_l2()
        );
        assert!(amp1.is_finite());
        assert!(amp1 > 0.05, "spin amplitude should remain: {amp1}");
        assert!(fluid.kinetic_energy() > 0.0);
    }
}
