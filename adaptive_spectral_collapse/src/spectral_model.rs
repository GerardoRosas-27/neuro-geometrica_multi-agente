//! Unified Multi-Channel Adaptive Spectral Collapse Engine implementing all 6 spectral arms:
//! `adaptive_spectral`, `riemann_fixed`, `fourier_fixed`, `dyson_gue_fixed`,
//! `poisson_fixed`, and `qmc_halton_fixed`.
//!
//! Matched 1:1 capacity vs `mlp_conventional` (`1,082` spectral params vs `1,118` MLP params).
//! Zero hardcoded primes or Riemann zeros are ever passed to `adaptive_spectral`.

use std::f64::consts::PI;

use crate::rng::{AdamState, Rng};
use crate::spectra::{
    dyson_gue_frequencies, fourier_harmonics, mean_spacing_ratio, poisson_frequencies,
    qmc_halton_frequencies, scaled_riemann_frequencies,
};
use crate::world::{UnknownWorld, DIM_TRAIT, DIM_X, K_BASE};

pub const DIM_Z: usize = 3;
pub const M_1D: usize = 24;
pub const M_3D: usize = 96;
pub const CAUSAL_HIDDEN: usize = 8;
pub const SCAN_1D_PTS: usize = 120;
pub const CAVITY_LIMIT: f64 = 2.05;
const VOXEL_N: usize = 11;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralArm {
    AdaptiveSpectral,
    RiemannFixed,
    FourierFixed,
    DysonGueFixed,
    PoissonFixed,
    QmcHaltonFixed,
}

impl SpectralArm {
    pub const ALL: [SpectralArm; 6] = [
        SpectralArm::AdaptiveSpectral,
        SpectralArm::RiemannFixed,
        SpectralArm::FourierFixed,
        SpectralArm::DysonGueFixed,
        SpectralArm::PoissonFixed,
        SpectralArm::QmcHaltonFixed,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            SpectralArm::AdaptiveSpectral => "adaptive_spectral",
            SpectralArm::RiemannFixed => "riemann_fixed",
            SpectralArm::FourierFixed => "fourier_fixed",
            SpectralArm::DysonGueFixed => "dyson_gue_fixed",
            SpectralArm::PoissonFixed => "poisson_fixed",
            SpectralArm::QmcHaltonFixed => "qmc_halton_fixed",
        }
    }

    pub const fn adapts_frequencies(self) -> bool {
        matches!(self, SpectralArm::AdaptiveSpectral)
    }
}

/// 1D spectral wave field with `M_1D = 24` cosine modes: `W(u) = DC_BG + sum_m a_m cos(w_m u + phi_m)`.
#[derive(Clone, Debug)]
pub struct Wave1D {
    pub omega: [f64; M_1D],
    pub amp: [f64; M_1D],
    pub phi: [f64; M_1D],
}

impl Wave1D {
    pub const DC_BG: f64 = -0.30;

    #[inline]
    pub fn eval(&self, u: f64) -> f64 {
        let mut s = Self::DC_BG;
        for m in 0..M_1D {
            s += self.amp[m] * (self.omega[m] * u + self.phi[m]).cos();
        }
        s
    }

    #[inline]
    pub fn grad_u(&self, u: f64) -> f64 {
        let mut g = 0.0;
        for m in 0..M_1D {
            g -= self.amp[m] * self.omega[m] * (self.omega[m] * u + self.phi[m]).sin();
        }
        g
    }

    #[inline]
    pub fn accum_grad(
        &self,
        u: f64,
        d_out: f64,
        g_w: &mut [f64; M_1D],
        g_a: &mut [f64; M_1D],
        g_p: &mut [f64; M_1D],
        adapt_freq: bool,
    ) {
        for m in 0..M_1D {
            let ang = self.omega[m] * u + self.phi[m];
            let c = ang.cos();
            let s = ang.sin();
            g_a[m] += d_out * c;
            let d_ang = -d_out * self.amp[m] * s;
            g_p[m] += d_ang;
            if adapt_freq {
                g_w[m] += d_ang * u;
            }
        }
    }
}

/// 3D cross-channel coherence wave field with `M_3D = 96` planar wavevectors `k_m in R^3`:
/// `W_3D(u) = DC_BG + sum_m a_m cos(k_m . u + phi_m)`.
#[derive(Clone, Debug)]
pub struct Wave3D {
    pub k: [[f64; DIM_Z]; M_3D],
    pub amp: [f64; M_3D],
    pub phi: [f64; M_3D],
}

impl Wave3D {
    pub const DC_BG: f64 = -0.35;

    #[inline]
    pub fn eval(&self, u: &[f64; DIM_Z]) -> f64 {
        let mut sum = Self::DC_BG;
        for m in 0..M_3D {
            let dot = self.k[m][0] * u[0] + self.k[m][1] * u[1] + self.k[m][2] * u[2];
            sum += self.amp[m] * (dot + self.phi[m]).cos();
        }
        sum
    }

    #[inline]
    pub fn grad_u(&self, u: &[f64; DIM_Z]) -> [f64; DIM_Z] {
        let mut g = [0.0; DIM_Z];
        for m in 0..M_3D {
            let dot = self.k[m][0] * u[0] + self.k[m][1] * u[1] + self.k[m][2] * u[2];
            let coeff = -self.amp[m] * (dot + self.phi[m]).sin();
            g[0] += coeff * self.k[m][0];
            g[1] += coeff * self.k[m][1];
            g[2] += coeff * self.k[m][2];
        }
        g
    }

    #[inline]
    pub fn accum_grad(
        &self,
        u: &[f64; DIM_Z],
        d_out: f64,
        g_k: &mut [[f64; DIM_Z]; M_3D],
        g_a: &mut [f64; M_3D],
        g_p: &mut [f64; M_3D],
        adapt_freq: bool,
    ) {
        for m in 0..M_3D {
            let dot = self.k[m][0] * u[0] + self.k[m][1] * u[1] + self.k[m][2] * u[2];
            let ang = dot + self.phi[m];
            let c = ang.cos();
            let s = ang.sin();
            g_a[m] += d_out * c;
            let d_ang = -d_out * self.amp[m] * s;
            g_p[m] += d_ang;
            if adapt_freq {
                g_k[m][0] += d_ang * u[0];
                g_k[m][1] += d_ang * u[1];
                g_k[m][2] += d_ang * u[2];
            }
        }
    }
}

/// Learned compositional causal readout with per-discovered-state 4D trait embeddings (`338` base params):
/// `v_hat[k] in R^4` (`K * 4` params) + symmetric MLP `[v_a+v_b, v_a*v_b, (v_a-v_b)^2] (12) -> 16 (tanh) -> 2` (`242` params).
#[derive(Clone, Debug)]
pub struct CausalReadout {
    pub k_states: usize,
    /// Layout:
    /// `[0..192)`: `W_h` (`16 x 12`)
    /// `[192..208)`: `b_h` (`16`)
    /// `[208..240)`: `W_o` (`2 x 16`)
    /// `[240..242)`: `b_o` (`2`)
    /// `[242..242 + 4*K)`: per-state latent causal trait vectors `v_hat[k] in R^4`
    pub params: Vec<f64>,
}

pub const CAUSAL_MLP_PARAMS: usize = CAUSAL_HIDDEN * 12 + CAUSAL_HIDDEN + 2 * CAUSAL_HIDDEN + 2; // 242

impl CausalReadout {
    pub const OFF_WH: usize = 0;
    pub const OFF_BH: usize = CAUSAL_HIDDEN * 12; // 192
    pub const OFF_WO: usize = Self::OFF_BH + CAUSAL_HIDDEN; // 208
    pub const OFF_BO: usize = Self::OFF_WO + 2 * CAUSAL_HIDDEN; // 240
    pub const OFF_V: usize = Self::OFF_BO + 2; // 242

    pub fn param_count(k_states: usize) -> usize {
        Self::OFF_V + k_states * DIM_TRAIT
    }

    pub fn new(k_states: usize, rng: &mut Rng) -> Self {
        let mut p = vec![0.0; Self::param_count(k_states)];
        let s_h = (1.0 / 12.0_f64).sqrt();
        for v in &mut p[Self::OFF_WH..Self::OFF_BH] {
            *v = s_h * rng.normal();
        }
        let s_o = (1.0 / (CAUSAL_HIDDEN as f64)).sqrt();
        for v in &mut p[Self::OFF_WO..Self::OFF_BO] {
            *v = s_o * rng.normal();
        }
        for v in &mut p[Self::OFF_V..] {
            *v = 0.5 * rng.normal();
        }
        Self { k_states, params: p }
    }

    pub fn expand_states(&mut self, new_k: usize, rng: &mut Rng) {
        assert!(new_k > self.k_states);
        let old_len = self.params.len();
        let new_len = Self::param_count(new_k);
        self.params.resize(new_len, 0.0);
        for v in &mut self.params[old_len..] {
            *v = 0.3 * rng.normal();
        }
        self.k_states = new_k;
    }

    #[inline]
    fn trait_of(&self, k: usize) -> [f64; DIM_TRAIT] {
        let off = Self::OFF_V + k * DIM_TRAIT;
        [
            self.params[off],
            self.params[off + 1],
            self.params[off + 2],
            self.params[off + 3],
        ]
    }

    #[inline]
    pub fn predict_pair(&self, a: usize, b: usize) -> (f64, f64, usize) {
        let va = self.trait_of(a);
        let vb = self.trait_of(b);
        let mut feat = [0.0_f64; 12];
        for i in 0..4 {
            feat[i] = va[i] + vb[i];
            feat[4 + i] = va[i] * vb[i];
            let d = va[i] - vb[i];
            feat[8 + i] = d * d;
        }
        let mut h = [0.0_f64; CAUSAL_HIDDEN];
        for j in 0..CAUSAL_HIDDEN {
            let mut s = self.params[Self::OFF_BH + j];
            let row = Self::OFF_WH + j * 12;
            for i in 0..12 {
                s += self.params[row + i] * feat[i];
            }
            h[j] = s.tanh();
        }
        let mut y = [0.0_f64; 2];
        for o in 0..2 {
            let mut s = self.params[Self::OFF_BO + o];
            let row = Self::OFF_WO + o * CAUSAL_HIDDEN;
            for j in 0..CAUSAL_HIDDEN {
                s += self.params[row + j] * h[j];
            }
            y[o] = s;
        }
        let cls = (if y[0] >= 0.0 { 2 } else { 0 }) + (if y[1] >= 0.0 { 1 } else { 0 });
        (y[0], y[1], cls)
    }

    pub fn loss_and_grad_pair(
        &self,
        a: usize,
        b: usize,
        ty1: f64,
        ty2: f64,
        grads: &mut [f64],
        scale: f64,
        freeze_mlp_and_old_k: Option<usize>,
    ) -> f64 {
        let va = self.trait_of(a);
        let vb = self.trait_of(b);
        let mut feat = [0.0_f64; 12];
        for i in 0..4 {
            feat[i] = va[i] + vb[i];
            feat[4 + i] = va[i] * vb[i];
            let d = va[i] - vb[i];
            feat[8 + i] = d * d;
        }
        let mut h = [0.0_f64; CAUSAL_HIDDEN];
        for j in 0..CAUSAL_HIDDEN {
            let mut s = self.params[Self::OFF_BH + j];
            let row = Self::OFF_WH + j * 12;
            for i in 0..12 {
                s += self.params[row + i] * feat[i];
            }
            h[j] = s.tanh();
        }
        let mut y = [0.0_f64; 2];
        for o in 0..2 {
            let mut s = self.params[Self::OFF_BO + o];
            let row = Self::OFF_WO + o * CAUSAL_HIDDEN;
            for j in 0..CAUSAL_HIDDEN {
                s += self.params[row + j] * h[j];
            }
            y[o] = s;
        }
        let e0 = y[0] - ty1;
        let e1 = y[1] - ty2;
        let loss = 0.5 * (e0 * e0 + e1 * e1);
        let dy = [scale * e0, scale * e1];

        let upd_mlp = freeze_mlp_and_old_k.is_none();
        let mut dh = [0.0_f64; CAUSAL_HIDDEN];
        for o in 0..2 {
            if upd_mlp {
                grads[Self::OFF_BO + o] += dy[o];
            }
            let row = Self::OFF_WO + o * CAUSAL_HIDDEN;
            for j in 0..CAUSAL_HIDDEN {
                if upd_mlp {
                    grads[row + j] += dy[o] * h[j];
                }
                dh[j] += dy[o] * self.params[row + j];
            }
        }
        let mut dfeat = [0.0_f64; 12];
        for j in 0..CAUSAL_HIDDEN {
            let dz = dh[j] * (1.0 - h[j] * h[j]);
            if upd_mlp {
                grads[Self::OFF_BH + j] += dz;
            }
            let row = Self::OFF_WH + j * 12;
            for i in 0..12 {
                if upd_mlp {
                    grads[row + i] += dz * feat[i];
                }
                dfeat[i] += dz * self.params[row + i];
            }
        }
        let off_a = Self::OFF_V + a * DIM_TRAIT;
        let off_b = Self::OFF_V + b * DIM_TRAIT;
        let upd_a = freeze_mlp_and_old_k.map_or(true, |k_old| a >= k_old);
        let upd_b = freeze_mlp_and_old_k.map_or(true, |k_old| b >= k_old);
        for i in 0..4 {
            let diff = va[i] - vb[i];
            if upd_a {
                grads[off_a + i] += dfeat[i] + dfeat[4 + i] * vb[i] + 2.0 * dfeat[8 + i] * diff;
            }
            if upd_b {
                grads[off_b + i] += dfeat[i] + dfeat[4 + i] * va[i] - 2.0 * dfeat[8 + i] * diff;
            }
        }
        scale * loss
    }
}

/// Complete Multi-Channel Adaptive Spectral Collapse model (`1,082` base params vs `1,118` MLP params).
#[derive(Clone, Debug)]
pub struct SpectralModel {
    pub arm: SpectralArm,
    pub w_enc: [[f64; DIM_X]; DIM_Z],
    pub waves_1d: [Wave1D; DIM_Z],
    pub wave_3d: Wave3D,
    pub causal: CausalReadout,
    pub active_k: usize,
    pub state_coords: Vec<[f64; DIM_Z]>,
    pub active_voxels: Vec<([f64; DIM_Z], f64)>,
    pub initial_spacing_r: f64,
    pub learned_spacing_r: f64,
}

impl SpectralModel {
    /// Total base parameter count: `48 (W_enc) + 216 (1D) + 480 (3D) + 338 (Causal) = 1,082` params.
    pub const TOTAL_PARAMS: usize =
        DIM_Z * DIM_X + DIM_Z * M_1D * 3 + M_3D * (DIM_Z + 2) + CAUSAL_MLP_PARAMS + K_BASE * DIM_TRAIT;

    pub fn init(arm: SpectralArm, riemann_zeros: Option<&[f64]>, rng: &mut Rng) -> Self {
        if arm == SpectralArm::AdaptiveSpectral {
            assert!(
                riemann_zeros.is_none(),
                "AdaptiveSpectral must never receive Riemann zeros"
            );
        }

        let mut w_enc = [[0.0_f64; DIM_X]; DIM_Z];
        for d in 0..DIM_Z {
            for c in 0..DIM_X {
                w_enc[d][c] = 0.65 * rng.normal();
            }
        }

        let f1_min = 1.2_f64;
        let f1_max = 15.0_f64;
        let period_1d = 10.0_f64;
        let k3_min = 1.2_f64;
        let k3_max = 4.6_f64;

        let build_1d_freqs = |d: usize, rng: &mut Rng| -> [f64; M_1D] {
            let v: Vec<f64> = match arm {
                SpectralArm::AdaptiveSpectral | SpectralArm::PoissonFixed => {
                    poisson_frequencies(M_1D, f1_min, f1_max, rng)
                }
                SpectralArm::RiemannFixed => {
                    let zs = riemann_zeros.expect("RiemannFixed requires zeros");
                    scaled_riemann_frequencies(&zs[d * M_1D..(d + 1) * M_1D], M_1D, f1_min, f1_max)
                }
                SpectralArm::FourierFixed => fourier_harmonics(M_1D, period_1d),
                SpectralArm::DysonGueFixed => dyson_gue_frequencies(M_1D, f1_min, f1_max, rng),
                SpectralArm::QmcHaltonFixed => {
                    let bases = [2usize, 3, 5];
                    qmc_halton_frequencies(M_1D, bases[d], f1_min, f1_max)
                }
            };
            let mut arr = [0.0; M_1D];
            arr.copy_from_slice(&v[..M_1D]);
            arr
        };

        let mut waves_1d = [
            Wave1D { omega: [0.0; M_1D], amp: [0.15; M_1D], phi: [0.0; M_1D] },
            Wave1D { omega: [0.0; M_1D], amp: [0.15; M_1D], phi: [0.0; M_1D] },
            Wave1D { omega: [0.0; M_1D], amp: [0.15; M_1D], phi: [0.0; M_1D] },
        ];
        for d in 0..DIM_Z {
            waves_1d[d].omega = build_1d_freqs(d, rng);
            for m in 0..M_1D {
                waves_1d[d].amp[m] = 0.15 * rng.normal();
                waves_1d[d].phi[m] = rng.uniform(0.0, 2.0 * PI);
            }
        }

        let mut k3 = [[0.0_f64; DIM_Z]; M_3D];
        match arm {
            SpectralArm::AdaptiveSpectral | SpectralArm::PoissonFixed => {
                for m in 0..M_3D {
                    let r = rng.uniform(k3_min, k3_max);
                    let mut dir = [rng.normal(), rng.normal(), rng.normal()];
                    let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-9);
                    for d in 0..DIM_Z {
                        dir[d] /= n;
                        k3[m][d] = r * dir[d];
                    }
                }
            }
            SpectralArm::RiemannFixed => {
                let zs = riemann_zeros.expect("RiemannFixed requires zeros");
                let radii = scaled_riemann_frequencies(zs, M_3D, k3_min, k3_max);
                let golden = PI * (3.0 - 5.0_f64.sqrt());
                for m in 0..M_3D {
                    let y = 1.0 - 2.0 * ((m as f64) + 0.5) / (M_3D as f64);
                    let rr = (1.0 - y * y).max(0.0).sqrt();
                    let th = golden * (m as f64);
                    let r = radii[m];
                    k3[m] = [r * rr * th.cos(), r * y, r * rr * th.sin()];
                }
            }
            SpectralArm::FourierFixed => {
                let base_k = 2.0 * PI / 6.2;
                let mut lat = Vec::new();
                for nx in -4i32..=4 {
                    for ny in -4i32..=4 {
                        for nz in 0i32..=4 {
                            let n2 = nx * nx + ny * ny + nz * nz;
                            if n2 == 0 {
                                continue;
                            }
                            if nz == 0 && (ny < 0 || (ny == 0 && nx < 0)) {
                                continue;
                            }
                            lat.push((n2, nx, ny, nz));
                        }
                    }
                }
                lat.sort_unstable();
                for m in 0..M_3D {
                    let (_, nx, ny, nz) = lat[m];
                    k3[m] = [
                        base_k * (nx as f64),
                        base_k * (ny as f64),
                        base_k * (nz as f64),
                    ];
                }
            }
            SpectralArm::DysonGueFixed => {
                let radii = dyson_gue_frequencies(M_3D, k3_min, k3_max, rng);
                let golden = PI * (3.0 - 5.0_f64.sqrt());
                for m in 0..M_3D {
                    let y = 1.0 - 2.0 * ((m as f64) + 0.5) / (M_3D as f64);
                    let rr = (1.0 - y * y).max(0.0).sqrt();
                    let th = golden * (m as f64);
                    let r = radii[m];
                    k3[m] = [r * rr * th.cos(), r * y, r * rr * th.sin()];
                }
            }
            SpectralArm::QmcHaltonFixed => {
                for m in 0..M_3D {
                    let h0 = crate::spectra::halton(m + 1, 2) * 2.0 - 1.0;
                    let h1 = crate::spectra::halton(m + 1, 3) * 2.0 - 1.0;
                    let h2 = crate::spectra::halton(m + 1, 5) * 2.0 - 1.0;
                    k3[m] = [3.8 * h0, 3.8 * h1, 3.8 * h2];
                }
            }
        }

        let mut amp3 = [0.0; M_3D];
        let mut phi3 = [0.0; M_3D];
        for m in 0..M_3D {
            amp3[m] = 0.12 * rng.normal();
            phi3[m] = rng.uniform(0.0, 2.0 * PI);
        }

        let init_r = (mean_spacing_ratio(&waves_1d[0].omega)
            + mean_spacing_ratio(&waves_1d[1].omega)
            + mean_spacing_ratio(&waves_1d[2].omega))
            / 3.0;

        Self {
            arm,
            w_enc,
            waves_1d,
            wave_3d: Wave3D { k: k3, amp: amp3, phi: phi3 },
            causal: CausalReadout::new(K_BASE, rng),
            active_k: K_BASE,
            state_coords: Vec::new(),
            active_voxels: Vec::new(),
            initial_spacing_r: init_r,
            learned_spacing_r: init_r,
        }
    }

    #[inline]
    pub fn project(&self, x: &[f64; DIM_X]) -> [f64; DIM_Z] {
        let mut z = [0.0; DIM_Z];
        for d in 0..DIM_Z {
            let mut s = 0.0;
            for c in 0..DIM_X {
                s += self.w_enc[d][c] * x[c];
            }
            z[d] = s;
        }
        z
    }

    #[inline]
    pub fn project_composite(&self, x_obs: &[f64; DIM_X]) -> [f64; DIM_Z] {
        let s2 = std::f64::consts::SQRT_2;
        let z = self.project(x_obs);
        [z[0] * s2, z[1] * s2, z[2] * s2]
    }

    pub fn refresh_state_coords(&mut self, xs: &[[f64; DIM_X]], k_active: usize) {
        self.active_k = k_active;
        self.state_coords.clear();
        for k in 0..k_active {
            self.state_coords.push(self.project(&xs[k]));
        }
    }

    /// Pre-rasterizes the continuous wave field `W_3D(g)` on a uniform 11x11x11 spatial grid
    /// spanning `[-1.80, 1.80]^3`, retaining positive-amplitude spatial voxels (ZERO concept lookup!).
    pub fn rebuild_voxel_raster(&mut self) {
        self.active_voxels.clear();
        let lo = -1.80_f64;
        let hi = 1.80_f64;
        let step = (hi - lo) / ((VOXEL_N - 1) as f64);
        let mut all_vox = Vec::with_capacity(VOXEL_N * VOXEL_N * VOXEL_N);
        for ix in 0..VOXEL_N {
            let ux = lo + (ix as f64) * step;
            for iy in 0..VOXEL_N {
                let uy = lo + (iy as f64) * step;
                for iz in 0..VOXEL_N {
                    let uz = lo + (iz as f64) * step;
                    let g = [ux, uy, uz];
                    let v = self.wave_3d.eval(&g);
                    if v > 0.05 {
                        self.active_voxels.push((g, v));
                    }
                    all_vox.push((g, v));
                }
            }
        }
        if self.active_voxels.len() < 32 {
            all_vox.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            self.active_voxels = all_vox.into_iter().take(64).collect();
        }
    }

    pub fn train_encoder(
        &mut self,
        xs: &[[f64; DIM_X]],
        k_active: usize,
        train_pairs: &[(usize, usize)],
        anchor_old: Option<&[[f64; DIM_Z]]>,
        steps: usize,
        lr: f64,
        rng: &mut Rng,
    ) {
        let n_params = DIM_Z * DIM_X;
        let mut flat = vec![0.0; n_params];
        for d in 0..DIM_Z {
            for c in 0..DIM_X {
                flat[d * DIM_X + c] = self.w_enc[d][c];
            }
        }
        let mut adam = AdamState::new(n_params);
        let mut zs = vec![[0.0_f64; DIM_Z]; k_active];
        let mut dz = vec![[0.0_f64; DIM_Z]; k_active];
        let mut grads = vec![0.0_f64; n_params];

        let inv_2r1 = 0.5 / (0.85 * 0.85);
        let inv_2r2 = 0.5 / (0.48 * 0.48);
        let bound = 1.65_f64;

        for step in 0..steps {
            for k in 0..k_active {
                for d in 0..DIM_Z {
                    let mut s = 0.0;
                    let row = d * DIM_X;
                    for c in 0..DIM_X {
                        s += flat[row + c] * xs[k][c];
                    }
                    zs[k][d] = s;
                }
                dz[k] = [0.0; DIM_Z];
            }

            let w_atom = 1.8 / (k_active as f64);
            for i in 0..k_active {
                for j in (i + 1)..k_active {
                    let dx0 = zs[i][0] - zs[j][0];
                    let dx1 = zs[i][1] - zs[j][1];
                    let dx2 = zs[i][2] - zs[j][2];
                    let d2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2;
                    let e = (-d2 * inv_2r1).exp();
                    let f = -2.0 * w_atom * inv_2r1 * e;
                    dz[i][0] += f * dx0;
                    dz[i][1] += f * dx1;
                    dz[i][2] += f * dx2;
                    dz[j][0] -= f * dx0;
                    dz[j][1] -= f * dx1;
                    dz[j][2] -= f * dx2;
                }
            }

            if train_pairs.len() >= 2 {
                let n_samp = 72usize;
                let w_sum = 3.0 / (n_samp as f64);
                for _ in 0..n_samp {
                    let p1 = train_pairs[rng.usize_range(train_pairs.len())];
                    let p2 = train_pairs[rng.usize_range(train_pairs.len())];
                    if p1 == p2 {
                        continue;
                    }
                    let (a, b) = p1;
                    let (c, e_idx) = p2;
                    let du0 = (zs[a][0] + zs[b][0]) - (zs[c][0] + zs[e_idx][0]);
                    let du1 = (zs[a][1] + zs[b][1]) - (zs[c][1] + zs[e_idx][1]);
                    let du2 = (zs[a][2] + zs[b][2]) - (zs[c][2] + zs[e_idx][2]);
                    let d2 = du0 * du0 + du1 * du1 + du2 * du2;
                    let ex = (-d2 * inv_2r2).exp();
                    let f = -2.0 * w_sum * inv_2r2 * ex;
                    for &idx in &[a, b] {
                        dz[idx][0] += f * du0;
                        dz[idx][1] += f * du1;
                        dz[idx][2] += f * du2;
                    }
                    for &idx in &[c, e_idx] {
                        dz[idx][0] -= f * du0;
                        dz[idx][1] -= f * du1;
                        dz[idx][2] -= f * du2;
                    }
                }
            }

            for k in 0..k_active {
                for d in 0..DIM_Z {
                    if zs[k][d] > bound {
                        dz[k][d] += 5.0 * (zs[k][d] - bound) / (k_active as f64);
                    } else if zs[k][d] < -bound {
                        dz[k][d] += 5.0 * (zs[k][d] + bound) / (k_active as f64);
                    }
                }
            }
            for d in 0..DIM_Z {
                let mut mu = 0.0;
                for k in 0..k_active {
                    mu += zs[k][d];
                }
                mu /= k_active as f64;
                let mut var = 0.0;
                for k in 0..k_active {
                    let diff = zs[k][d] - mu;
                    var += diff * diff;
                }
                var /= k_active as f64;
                let dv = 2.0 * (var - 0.85);
                for k in 0..k_active {
                    dz[k][d] += 0.5 * (2.0 * mu / (k_active as f64))
                        + 1.8 * dv * (2.0 * (zs[k][d] - mu) / (k_active as f64));
                }
            }

            if let Some(old_z) = anchor_old {
                let w_anc = 22.0 / (old_z.len() as f64);
                for (k, oz) in old_z.iter().enumerate() {
                    for d in 0..DIM_Z {
                        dz[k][d] += 2.0 * w_anc * (zs[k][d] - oz[d]);
                    }
                }
            }

            grads.fill(0.0);
            for k in 0..k_active {
                for d in 0..DIM_Z {
                    let row = d * DIM_X;
                    let g = dz[k][d];
                    for c in 0..DIM_X {
                        grads[row + c] += g * xs[k][c];
                    }
                }
            }
            let lr_t = lr * (0.5 * (1.0 + (PI * (step as f64) / (steps as f64)).cos())).max(0.1);
            adam.update(&mut flat, &grads, lr_t, 0.045);
        }

        for d in 0..DIM_Z {
            for c in 0..DIM_X {
                self.w_enc[d][c] = flat[d * DIM_X + c];
            }
        }
        self.refresh_state_coords(xs, k_active);
    }

    pub fn train_wave_fields(
        &mut self,
        train_pairs: &[(usize, usize)],
        steps_1d: usize,
        steps_3d: usize,
        rng: &mut Rng,
    ) {
        let adapt_freq = self.arm.adapts_frequencies();
        let k_active = self.state_coords.len();

        let mut row_std = [0.0_f64; DIM_Z];
        for d in 0..DIM_Z {
            let n2: f64 = self.w_enc[d].iter().map(|v| v * v).sum();
            row_std[d] = (std::f64::consts::SQRT_2 * n2.sqrt() * 0.018).clamp(0.03, 0.11);
        }

        // --- 2A. Adapt 3 1D spectral channels (M_1D = 24) ---
        for d in 0..DIM_Z {
            let n_p = 3 * M_1D;
            let mut flat = vec![0.0; n_p];
            for m in 0..M_1D {
                flat[m] = self.waves_1d[d].omega[m];
                flat[M_1D + m] = self.waves_1d[d].amp[m];
                flat[2 * M_1D + m] = self.waves_1d[d].phi[m];
            }
            let mut adam = AdamState::new(n_p);
            let mut grads = vec![0.0; n_p];
            let coords_d: Vec<f64> = self.state_coords.iter().map(|z| z[d]).collect();

            for step in 0..steps_1d {
                for m in 0..M_1D {
                    self.waves_1d[d].omega[m] = flat[m];
                    self.waves_1d[d].amp[m] = flat[M_1D + m];
                    self.waves_1d[d].phi[m] = flat[2 * M_1D + m];
                }
                let mut g_w = [0.0; M_1D];
                let mut g_a = [0.0; M_1D];
                let mut g_p = [0.0; M_1D];

                let w_pos = 1.0 / (k_active as f64);
                for &zk in &coords_d {
                    let uj = zk + 0.5 * row_std[d] * rng.normal();
                    let val = self.waves_1d[d].eval(uj);
                    let margin = 1.25 - val;
                    if margin > 0.0 {
                        self.waves_1d[d].accum_grad(uj, -2.0 * w_pos * margin, &mut g_w, &mut g_a, &mut g_p, adapt_freq);
                    }
                }

                let n_neg = 56usize;
                let w_neg = 1.3 / (n_neg as f64);
                for _ in 0..n_neg {
                    let un = rng.uniform(-CAVITY_LIMIT, CAVITY_LIMIT);
                    let mut min_dist = f64::INFINITY;
                    for &zk in &coords_d {
                        let dd = (un - zk).abs();
                        if dd < min_dist {
                            min_dist = dd;
                        }
                    }
                    if min_dist < 0.08 {
                        continue;
                    }
                    let val = self.waves_1d[d].eval(un);
                    let excess = val + 0.35;
                    if excess > 0.0 {
                        self.waves_1d[d].accum_grad(un, 2.0 * w_neg * excess, &mut g_w, &mut g_a, &mut g_p, adapt_freq);
                    }
                }

                if adapt_freq {
                    for m in 0..M_1D {
                        let wm = self.waves_1d[d].omega[m];
                        if wm < 1.0 {
                            g_w[m] += 2.0 * (wm - 1.0);
                        } else if wm > 18.0 {
                            g_w[m] += 2.0 * (wm - 18.0);
                        }
                        for n in (m + 1)..M_1D {
                            let diff = wm - self.waves_1d[d].omega[n];
                            let ex = (-diff * diff / 0.18).exp();
                            let f = -0.02 * (2.0 * diff / 0.18) * ex;
                            g_w[m] += f;
                            g_w[n] -= f;
                        }
                    }
                }

                for m in 0..M_1D {
                    grads[m] = if adapt_freq { g_w[m] } else { 0.0 };
                    grads[M_1D + m] = g_a[m] + 0.002 * flat[M_1D + m];
                    grads[2 * M_1D + m] = g_p[m];
                }
                let lr_t = 0.04 * (0.5 * (1.0 + (PI * (step as f64) / (steps_1d as f64)).cos())).max(0.15);
                adam.update(&mut flat, &grads, lr_t, 0.0);
            }
            for m in 0..M_1D {
                self.waves_1d[d].omega[m] = flat[m];
                self.waves_1d[d].amp[m] = flat[M_1D + m];
                self.waves_1d[d].phi[m] = flat[2 * M_1D + m];
            }
        }

        // --- 2B. Adapt 3D cross-channel coherence wave (M_3D = 96) ---
        let n_p3 = M_3D * 5;
        let mut flat3 = vec![0.0; n_p3];
        for m in 0..M_3D {
            flat3[m * 5] = self.wave_3d.k[m][0];
            flat3[m * 5 + 1] = self.wave_3d.k[m][1];
            flat3[m * 5 + 2] = self.wave_3d.k[m][2];
            flat3[m * 5 + 3] = self.wave_3d.amp[m];
            flat3[m * 5 + 4] = self.wave_3d.phi[m];
        }
        let mut adam3 = AdamState::new(n_p3);
        let mut grads3 = vec![0.0; n_p3];

        for step in 0..steps_3d {
            for m in 0..M_3D {
                self.wave_3d.k[m] = [flat3[m * 5], flat3[m * 5 + 1], flat3[m * 5 + 2]];
                self.wave_3d.amp[m] = flat3[m * 5 + 3];
                self.wave_3d.phi[m] = flat3[m * 5 + 4];
            }
            let mut g_k = [[0.0; DIM_Z]; M_3D];
            let mut g_a = [0.0; M_3D];
            let mut g_p = [0.0; M_3D];

            let w_pos = 1.6 / (k_active as f64);
            for k_idx in 0..k_active {
                let z = self.state_coords[k_idx];
                let v0 = self.wave_3d.eval(&z);
                if v0 < 1.8 {
                    self.wave_3d.accum_grad(&z, -2.0 * 0.55 * w_pos * (1.8 - v0), &mut g_k, &mut g_a, &mut g_p, adapt_freq);
                }
                let u = [
                    z[0] + 0.75 * row_std[0] * rng.normal(),
                    z[1] + 0.75 * row_std[1] * rng.normal(),
                    z[2] + 0.75 * row_std[2] * rng.normal(),
                ];
                let vj = self.wave_3d.eval(&u);
                if vj < 1.4 {
                    self.wave_3d.accum_grad(&u, -2.0 * 0.45 * w_pos * (1.4 - vj), &mut g_k, &mut g_a, &mut g_p, adapt_freq);
                }
            }

            let n_void = 64usize;
            let n_ghost = 48usize;
            let n_pair_neg = if train_pairs.is_empty() { 0 } else { 40 };
            let w_neg = 2.4 / ((n_void + n_ghost + n_pair_neg).max(1) as f64);

            for idx in 0..(n_void + n_ghost + n_pair_neg) {
                let cand = if idx < n_void {
                    [
                        rng.uniform(-CAVITY_LIMIT, CAVITY_LIMIT),
                        rng.uniform(-CAVITY_LIMIT, CAVITY_LIMIT),
                        rng.uniform(-CAVITY_LIMIT, CAVITY_LIMIT),
                    ]
                } else if idx < n_void + n_ghost {
                    let i = rng.usize_range(k_active);
                    let j = rng.usize_range(k_active);
                    let mut l = rng.usize_range(k_active);
                    if i == j && j == l {
                        l = (l + 1) % k_active;
                    }
                    [
                        self.state_coords[i][0] + 0.04 * rng.normal(),
                        self.state_coords[j][1] + 0.04 * rng.normal(),
                        self.state_coords[l][2] + 0.04 * rng.normal(),
                    ]
                } else {
                    let (a, b) = train_pairs[rng.usize_range(train_pairs.len())];
                    let mut c = rng.usize_range(k_active);
                    if c == a || c == b {
                        c = (c + 2) % k_active;
                    }
                    [
                        self.state_coords[a][0] + self.state_coords[b][0] - self.state_coords[c][0],
                        self.state_coords[a][1] + self.state_coords[b][1] - self.state_coords[c][1],
                        self.state_coords[a][2] + self.state_coords[b][2] - self.state_coords[c][2],
                    ]
                };

                if cand[0].abs() > CAVITY_LIMIT || cand[1].abs() > CAVITY_LIMIT || cand[2].abs() > CAVITY_LIMIT {
                    continue;
                }
                let mut min_d2 = f64::INFINITY;
                for z in &self.state_coords {
                    let d2 = (cand[0] - z[0]).powi(2) + (cand[1] - z[1]).powi(2) + (cand[2] - z[2]).powi(2);
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                }
                if min_d2 < 0.14 {
                    continue;
                }
                let val = self.wave_3d.eval(&cand);
                let excess = val + 0.65;
                if excess > 0.0 {
                    self.wave_3d.accum_grad(&cand, 2.0 * w_neg * excess, &mut g_k, &mut g_a, &mut g_p, adapt_freq);
                }
            }

            if adapt_freq {
                for m in 0..M_3D {
                    let norm2 = self.wave_3d.k[m][0].powi(2)
                        + self.wave_3d.k[m][1].powi(2)
                        + self.wave_3d.k[m][2].powi(2);
                    if norm2 > 25.0 {
                        let excess = norm2 - 25.0;
                        for d in 0..DIM_Z {
                            g_k[m][d] += 0.15 * excess * self.wave_3d.k[m][d];
                        }
                    }
                }
            }

            for m in 0..M_3D {
                grads3[m * 5] = if adapt_freq { g_k[m][0] } else { 0.0 };
                grads3[m * 5 + 1] = if adapt_freq { g_k[m][1] } else { 0.0 };
                grads3[m * 5 + 2] = if adapt_freq { g_k[m][2] } else { 0.0 };
                grads3[m * 5 + 3] = g_a[m] + 0.002 * flat3[m * 5 + 3];
                grads3[m * 5 + 4] = g_p[m];
            }
            let lr_t = 0.04 * (0.5 * (1.0 + (PI * (step as f64) / (steps_3d as f64)).cos())).max(0.15);
            adam3.update(&mut flat3, &grads3, lr_t, 0.0);
        }

        for m in 0..M_3D {
            self.wave_3d.k[m] = [flat3[m * 5], flat3[m * 5 + 1], flat3[m * 5 + 2]];
            self.wave_3d.amp[m] = flat3[m * 5 + 3];
            self.wave_3d.phi[m] = flat3[m * 5 + 4];
        }

        self.rebuild_voxel_raster();

        self.learned_spacing_r = (mean_spacing_ratio(&self.waves_1d[0].omega)
            + mean_spacing_ratio(&self.waves_1d[1].omega)
            + mean_spacing_ratio(&self.waves_1d[2].omega))
            / 3.0;
    }

    pub fn train_causal_readout(
        &mut self,
        world: &UnknownWorld,
        train_pairs: &[(usize, usize)],
        steps: usize,
        lr: f64,
        freeze_mlp_and_old_k: Option<usize>,
    ) {
        let mut adam = AdamState::new(self.causal.params.len());
        let mut grads = vec![0.0_f64; self.causal.params.len()];
        let scale = 1.0 / (train_pairs.len().max(1) as f64);

        for step in 0..steps {
            grads.fill(0.0);
            for &(a, b) in train_pairs {
                let (ty1, ty2, _) = world.eval_outcome(a, b);
                self.causal.loss_and_grad_pair(
                    a,
                    b,
                    ty1,
                    ty2,
                    &mut grads,
                    scale,
                    freeze_mlp_and_old_k,
                );
            }
            let lr_t = lr * (0.5 * (1.0 + (PI * (step as f64) / (steps as f64)).cos())).max(0.1);
            // Zero weight decay on frozen entries when continual-adapting novel traits
            if let Some(k_old) = freeze_mlp_and_old_k {
                let start = CausalReadout::OFF_V + k_old * DIM_TRAIT;
                adam.update(&mut self.causal.params, &grads, lr_t, 0.0);
                // Keep mild regularization only on newly learned traits
                for v in &mut self.causal.params[start..] {
                    *v *= 1.0 - 0.008 * lr_t;
                }
            } else {
                adam.update(&mut self.causal.params, &grads, lr_t, 0.010);
            }
        }
    }

    pub fn train_base(&mut self, world: &UnknownWorld, rng: &mut Rng) {
        self.train_encoder(&world.xs, K_BASE, &world.train_pairs, None, 350, 0.035, rng);
        self.train_wave_fields(&world.train_pairs, 300, 480, rng);
        self.train_causal_readout(world, &world.train_pairs, 650, 0.018, None);
    }

    pub fn adapt_continual(&mut self, world: &UnknownWorld, rng: &mut Rng) {
        let old_coords = self.state_coords.clone();
        let mut mixed_pairs = world.novel_train_pairs.clone();
        let n_replay = 32.min(world.train_pairs.len());
        for i in 0..n_replay {
            let idx = (i * 3) % world.train_pairs.len();
            mixed_pairs.push(world.train_pairs[idx]);
        }

        self.train_encoder(
            &world.xs,
            world.xs.len(),
            &mixed_pairs,
            Some(&old_coords),
            100,
            0.015,
            rng,
        );
        self.train_wave_fields(&mixed_pairs, 120, 180, rng);

        // Expand per-state trait table for the 4 novel concepts (24..27) and infer their 4D traits
        // directly from their 4 interactions while keeping the learned causal law & old traits intact!
        self.causal.expand_states(world.xs.len(), rng);
        self.train_causal_readout(world, &world.novel_train_pairs, 260, 0.035, Some(K_BASE));
    }

    /// Purely continuous 1D + 3D Past x Future Wave-Packet Collapse decoder (ZERO pair/concept lookup!).
    pub fn decode_phase(&self, u_obs: &[f64; DIM_Z]) -> (usize, usize, [f64; DIM_Z]) {
        let mut seeds: Vec<([f64; DIM_Z], f64)> = Vec::with_capacity(128);

        for &(g, wp) in &self.active_voxels {
            let comp = [u_obs[0] - g[0], u_obs[1] - g[1], u_obs[2] - g[2]];
            if comp[0].abs() > CAVITY_LIMIT || comp[1].abs() > CAVITY_LIMIT || comp[2].abs() > CAVITY_LIMIT {
                continue;
            }
            let wf = self.wave_3d.eval(&comp);
            let sc = if wf > 0.0 {
                wp * wf
            } else {
                0.1 * (wp + wf) - 1.0
            };
            seeds.push((g, sc));
        }

        // Continuous 1D channel Past x Future peak intersections (top-3 per channel)
        let mut peaks_1d: [Vec<f64>; DIM_Z] = [Vec::new(), Vec::new(), Vec::new()];
        let step = (2.0 * CAVITY_LIMIT) / ((SCAN_1D_PTS - 1) as f64);
        for d in 0..DIM_Z {
            let mut sc = [0.0_f64; SCAN_1D_PTS];
            let mut us = [0.0_f64; SCAN_1D_PTS];
            for i in 0..SCAN_1D_PTS {
                let u = -CAVITY_LIMIT + (i as f64) * step;
                us[i] = u;
                let wp = self.waves_1d[d].eval(u).max(0.0);
                let wf = if wp > 0.0 {
                    self.waves_1d[d].eval(u_obs[d] - u).max(0.0)
                } else {
                    0.0
                };
                sc[i] = wp * wf;
            }
            let mut loc: Vec<(f64, f64)> = Vec::new();
            for i in 0..SCAN_1D_PTS {
                if sc[i] > 0.0
                    && (i == 0 || sc[i] >= sc[i - 1])
                    && (i + 1 == SCAN_1D_PTS || sc[i] >= sc[i + 1])
                {
                    loc.push((us[i], sc[i]));
                }
            }
            loc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (u, _) in loc {
                if peaks_1d[d].iter().all(|&p| (p - u).abs() > 0.12) {
                    peaks_1d[d].push(u);
                    if peaks_1d[d].len() >= 3 {
                        break;
                    }
                }
            }
        }
        for &u0 in &peaks_1d[0] {
            for &u1 in &peaks_1d[1] {
                for &u2 in &peaks_1d[2] {
                    let g = [u0, u1, u2];
                    let comp = [u_obs[0] - u0, u_obs[1] - u1, u_obs[2] - u2];
                    let wp = self.wave_3d.eval(&g);
                    let wf = self.wave_3d.eval(&comp);
                    let sc = if wp > 0.0 && wf > 0.0 {
                        wp * wf
                    } else {
                        0.1 * (wp + wf) - 1.0
                    };
                    seeds.push((g, sc));
                }
            }
        }

        if seeds.is_empty() {
            seeds.push(([u_obs[0] * 0.5, u_obs[1] * 0.5, u_obs[2] * 0.5], -10.0));
        }

        seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut top_seeds: Vec<[f64; DIM_Z]> = Vec::with_capacity(6);
        for (g, _) in seeds {
            if top_seeds.iter().all(|s| {
                (s[0] - g[0]).powi(2) + (s[1] - g[1]).powi(2) + (s[2] - g[2]).powi(2) > 0.16
            }) {
                top_seeds.push(g);
                if top_seeds.len() >= 6 {
                    break;
                }
            }
        }

        let eval_collapse = |u: &[f64; DIM_Z]| -> (f64, [f64; DIM_Z]) {
            let comp = [u_obs[0] - u[0], u_obs[1] - u[1], u_obs[2] - u[2]];
            let wp = self.wave_3d.eval(u);
            let wf = self.wave_3d.eval(&comp);
            let gp = self.wave_3d.grad_u(u);
            let gf = self.wave_3d.grad_u(&comp);
            let (mut val, mut grad) = if wp > 0.0 && wf > 0.0 {
                (
                    wp * wf,
                    [
                        gp[0] * wf - wp * gf[0],
                        gp[1] * wf - wp * gf[1],
                        gp[2] * wf - wp * gf[2],
                    ],
                )
            } else {
                (
                    0.25 * (wp + wf) - 0.5,
                    [
                        0.25 * (gp[0] - gf[0]),
                        0.25 * (gp[1] - gf[1]),
                        0.25 * (gp[2] - gf[2]),
                    ],
                )
            };
            for d in 0..DIM_Z {
                let p1 = self.waves_1d[d].eval(u[d]);
                let f1 = self.waves_1d[d].eval(comp[d]);
                if p1 > 0.0 && f1 > 0.0 {
                    val += 0.06 * p1 * f1;
                    let g1p = self.waves_1d[d].grad_u(u[d]);
                    let g1f = self.waves_1d[d].grad_u(comp[d]);
                    grad[d] += 0.06 * (g1p * f1 - p1 * g1f);
                }
            }
            (val, grad)
        };

        let mut best_u = top_seeds[0];
        let mut best_val = f64::NEG_INFINITY;

        for start in top_seeds {
            let mut cur = start;
            let (mut cur_val, mut cur_grad) = eval_collapse(&cur);
            let mut step_sz = 0.08_f64;
            for _ in 0..14 {
                let gnorm = (cur_grad[0].powi(2) + cur_grad[1].powi(2) + cur_grad[2].powi(2)).sqrt();
                if gnorm < 1e-6 {
                    break;
                }
                let mut cand = [0.0; DIM_Z];
                for d in 0..DIM_Z {
                    cand[d] = (cur[d] + step_sz * (cur_grad[d] / gnorm)).clamp(-CAVITY_LIMIT, CAVITY_LIMIT);
                }
                let (cv, cg) = eval_collapse(&cand);
                if cv > cur_val {
                    cur = cand;
                    cur_val = cv;
                    cur_grad = cg;
                } else {
                    step_sz *= 0.5;
                    if step_sz < 0.004 {
                        break;
                    }
                }
            }
            if cur_val > best_val {
                best_val = cur_val;
                best_u = cur;
            }
        }

        let u2 = [u_obs[0] - best_u[0], u_obs[1] - best_u[1], u_obs[2] - best_u[2]];
        let (pa, pb) = self.project_to_distinct_pair(&best_u, &u2);
        (pa, pb, best_u)
    }

    pub fn decode_sensory(&self, x_obs: &[f64; DIM_X]) -> (usize, usize, [f64; DIM_Z]) {
        let u_obs = self.project_composite(x_obs);
        self.decode_phase(&u_obs)
    }

    fn nearest_two(&self, u: &[f64; DIM_Z]) -> ((usize, f64), (usize, f64)) {
        let mut b1 = (0usize, f64::INFINITY);
        let mut b2 = (1usize, f64::INFINITY);
        for (k, z) in self.state_coords.iter().enumerate() {
            let d2 = (u[0] - z[0]).powi(2) + (u[1] - z[1]).powi(2) + (u[2] - z[2]).powi(2);
            if d2 < b1.1 {
                b2 = b1;
                b1 = (k, d2);
            } else if d2 < b2.1 {
                b2 = (k, d2);
            }
        }
        (b1, b2)
    }

    fn project_to_distinct_pair(&self, u1: &[f64; DIM_Z], u2: &[f64; DIM_Z]) -> (usize, usize) {
        let (n1a, n1b) = self.nearest_two(u1);
        let (n2a, n2b) = self.nearest_two(u2);
        let (a, b) = if n1a.0 != n2a.0 {
            (n1a.0, n2a.0)
        } else if n1a.1 + n2b.1 <= n2a.1 + n1b.1 {
            (n1a.0, n2b.0)
        } else {
            (n1b.0, n2a.0)
        };
        if a < b { (a, b) } else { (b, a) }
    }

    pub fn collision_free_rate(&self) -> f64 {
        let k = self.state_coords.len();
        let mut sums = Vec::with_capacity(k * (k - 1) / 2);
        for a in 0..k {
            for b in (a + 1)..k {
                sums.push([
                    self.state_coords[a][0] + self.state_coords[b][0],
                    self.state_coords[a][1] + self.state_coords[b][1],
                    self.state_coords[a][2] + self.state_coords[b][2],
                ]);
            }
        }
        let thresh2 = 0.10 * 0.10;
        let mut ok = 0usize;
        for i in 0..sums.len() {
            let mut clean = true;
            for j in 0..sums.len() {
                if i == j {
                    continue;
                }
                let d2 = (sums[i][0] - sums[j][0]).powi(2)
                    + (sums[i][1] - sums[j][1]).powi(2)
                    + (sums[i][2] - sums[j][2]).powi(2);
                if d2 < thresh2 {
                    clean = false;
                    break;
                }
            }
            if clean {
                ok += 1;
            }
        }
        (ok as f64) / (sums.len().max(1) as f64)
    }

    pub fn field_contrast_snr(&self, rng: &mut Rng) -> f64 {
        let mut sig = 0.0;
        for z in &self.state_coords {
            sig += self.wave_3d.eval(z);
        }
        sig /= self.state_coords.len().max(1) as f64;
        let mut void_pos_sq = 0.0;
        let n = 400usize;
        for _ in 0..n {
            let u = [
                rng.uniform(-CAVITY_LIMIT, CAVITY_LIMIT),
                rng.uniform(-CAVITY_LIMIT, CAVITY_LIMIT),
                rng.uniform(-CAVITY_LIMIT, CAVITY_LIMIT),
            ];
            let v = self.wave_3d.eval(&u).max(0.0);
            void_pos_sq += v * v;
        }
        let rms = (void_pos_sq / (n as f64)).sqrt().max(1e-3);
        sig.max(0.0) / rms
    }
}
