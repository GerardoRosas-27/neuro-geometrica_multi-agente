//! Spectral ensembles & repulsion dynamics:
//! - Dyson Langevin Brownian gas (beta=2 GUE and beta=1 GOE) + Wigner semicircle unfolding
//! - 2D isotropic Coulomb/Ginibre log-gas (beta=2 2D DPP)
//! - Complex Hermitian GUE random matrix reference sampler
//! - Exact Wigner GUE & Poisson spacing CDFs and Kolmogorov-Smirnov diagnostics

use std::f64::consts::PI;

use crate::linalg::hermitian_eigenvalues;
use crate::riemann::C64;
use crate::rng::{erf, Rng};

/// Exact Wigner semicircle CDF mapped from `[-R, R]` to `[0, 1]`.
#[inline]
pub fn wigner_semicircle_cdf(w: f64, radius: f64) -> f64 {
    let z = (w / radius).clamp(-0.999_99, 0.999_99);
    0.5 + (1.0 / PI) * (z * (1.0 - z * z).sqrt() + z.asin())
}

/// Maps sorted eigenvalues in the Wigner semicircle `[-2*sqrt(M), 2*sqrt(M)]`
/// onto frequencies in `[1.0, M]` with mean spacing 1.
pub fn unfold_semicircle_to_freq(w_sorted: &[f64]) -> Vec<f64> {
    let m = w_sorted.len();
    let radius = 2.0 * (m as f64).sqrt();
    w_sorted
        .iter()
        .map(|&w| 1.0 + ((m - 1) as f64) * wigner_semicircle_cdf(w, radius))
        .collect()
}

/// Maps sorted eigenvalues onto open `(0, 1)` quantiles via Wigner semicircle CDF,
/// using finite-sample support `[0.5/M, 1 - 0.5/M]` to avoid Tracy-Widom edge clamping spikes
/// into extreme Gaussian tails (`inv_normal_cdf(1e-6)`).
pub fn unfold_semicircle_to_quantiles(w_sorted: &[f64]) -> Vec<f64> {
    let m = w_sorted.len();
    let mf = m as f64;
    let radius = 2.0 * mf.sqrt();
    w_sorted
        .iter()
        .map(|&w| {
            let cdf = wigner_semicircle_cdf(w, radius);
            (0.5 + (mf - 1.0) * cdf) / mf
        })
        .collect()
}

pub struct DysonResult {
    pub frequencies: Vec<f64>,
    pub raw_eigenvalues: Vec<f64>,
    pub history: Vec<Vec<f64>>,
    pub equilibrium_spacings: Vec<f64>,
}

/// Consolidates `n_modes` frequencies via Dyson overdamped Langevin dynamics:
///   dw_i = (-0.5*w_i + sum_{j!=i} (w_i - w_j)/((w_i - w_j)^2 + eps^2)) dt + sqrt(2*D*dt) * xi_i
/// Setting `beta = 2.0` gives `D = 0.5` (exact GUE/Riemann spacing law);
/// setting `beta = 1.0` gives `D = 1.0` (GOE linear repulsion).
pub fn consolidate_dyson(
    n_modes: usize,
    steps: usize,
    dt: f64,
    beta: f64,
    record_history: bool,
    rng: &mut Rng,
) -> DysonResult {
    let radius = 2.0 * (n_modes as f64).sqrt();
    let mut w: Vec<f64> = (0..n_modes)
        .map(|_| rng.uniform(-0.85 * radius, 0.85 * radius))
        .collect();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut history = Vec::new();
    if record_history {
        history.push(w.clone());
    }
    let mut eq_spacings = Vec::new();

    let eps2 = 0.03 * 0.03;
    let noise_scale = ((2.0 / beta) * dt).sqrt();
    let mut next_w = vec![0.0; n_modes];

    for step in 0..steps {
        for i in 0..n_modes {
            let wi = w[i];
            let mut rep = 0.0;
            for j in 0..n_modes {
                if i == j {
                    continue;
                }
                let d = wi - w[j];
                rep += d / (d * d + eps2);
            }
            let drift = -0.5 * wi + rep;
            next_w[i] = wi + drift * dt + noise_scale * rng.normal();
        }
        std::mem::swap(&mut w, &mut next_w);

        if record_history && (step + 1) % 40 == 0 {
            let mut snap = w.clone();
            snap.sort_by(|a, b| a.partial_cmp(b).unwrap());
            history.push(snap);
        }

        if step >= steps / 2 && (step + 1) % 50 == 0 {
            let mut snap = w.clone();
            snap.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let freqs = unfold_semicircle_to_freq(&snap);
            let trim = (n_modes / 12).max(3).min(n_modes / 4);
            for k in trim..(n_modes - trim - 1) {
                eq_spacings.push(freqs[k + 1] - freqs[k]);
            }
        }
    }

    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let frequencies = unfold_semicircle_to_freq(&w);

    if !eq_spacings.is_empty() {
        let mean = eq_spacings.iter().sum::<f64>() / eq_spacings.len() as f64;
        for s in &mut eq_spacings {
            *s /= mean.max(1e-12);
        }
    }

    DysonResult {
        frequencies,
        raw_eigenvalues: w,
        history,
        equilibrium_spacings: eq_spacings,
    }
}

/// Samples normalized bulk level spacings from `n_matrices` complex Hermitian GUE matrices of size `dim`.
pub fn sample_gue_matrix_spacings(dim: usize, n_matrices: usize, rng: &mut Rng) -> Vec<f64> {
    let mut spacings = Vec::new();
    let inv_scale = 1.0 / (2.0 * (dim as f64).sqrt());
    for _ in 0..n_matrices {
        let mut h = vec![C64::new(0.0, 0.0); dim * dim];
        for r in 0..dim {
            // Diagonal: (a + a*) / (2 sqrt(dim)) = Re(a) / sqrt(dim) -> variance 1/dim
            let d = rng.normal() / (dim as f64).sqrt();
            h[r * dim + r] = C64::new(d, 0.0);
            for c in (r + 1)..dim {
                // H = (A + A^H)/(2 sqrt(dim)): off-diag Re and Im each have variance 1/(2*dim)
                let re = (rng.normal() + rng.normal()) * inv_scale;
                let im = (rng.normal() - rng.normal()) * inv_scale;
                h[r * dim + c] = C64::new(re, im);
                h[c * dim + r] = C64::new(re, -im);
            }
        }
        let (evals, _) = hermitian_eigenvalues(&h, dim);
        // Bulk within (-1.45, 1.45) of Wigner semicircle [-2, 2]
        let bulk: Vec<f64> = evals
            .into_iter()
            .filter(|&e| e > -1.45 && e < 1.45)
            .collect();
        let unfolded: Vec<f64> = bulk
            .iter()
            .map(|&e| (dim as f64) * wigner_semicircle_cdf(e, 2.0))
            .collect();
        for w in unfolded.windows(2) {
            spacings.push(w[1] - w[0]);
        }
    }
    let mean = spacings.iter().sum::<f64>() / spacings.len().max(1) as f64;
    for s in &mut spacings {
        *s /= mean.max(1e-12);
    }
    spacings
}

/// Exact Wigner GUE surmise CDF: `F_GUE(s) = erf(2*s/sqrt(pi)) - (4/pi)*s*exp(-(4/pi)*s^2)`.
#[inline]
pub fn wigner_gue_cdf(s: f64) -> f64 {
    if s <= 0.0 {
        return 0.0;
    }
    let a = 4.0 / PI;
    (erf(2.0 * s / PI.sqrt()) - a * s * (-a * s * s).exp()).clamp(0.0, 1.0)
}

/// Exact Poisson spacing CDF: `F_Poisson(s) = 1 - exp(-s)`.
#[inline]
pub fn poisson_cdf(s: f64) -> f64 {
    if s <= 0.0 {
        return 0.0;
    }
    1.0 - (-s).exp()
}

/// One-sample Kolmogorov-Smirnov statistic `sup_s |F_emp(s) - F_ref(s)|`.
pub fn ks_distance<F: Fn(f64) -> f64>(samples: &[f64], cdf: F) -> f64 {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len() as f64;
    let mut max_d = 0.0_f64;
    for (i, &x) in s.iter().enumerate() {
        let f_ref = cdf(x);
        let f_lo = (i as f64) / n;
        let f_hi = ((i + 1) as f64) / n;
        max_d = max_d.max((f_ref - f_lo).abs()).max((f_hi - f_ref).abs());
    }
    max_d
}

/// Fraction of spacings below threshold `thresh` (level repulsion near `s = 0`).
pub fn small_spacing_fraction(samples: &[f64], thresh: f64) -> f64 {
    let count = samples.iter().filter(|&&s| s < thresh).count();
    count as f64 / samples.len().max(1) as f64
}

/// 2D isotropic Coulomb/Ginibre log-gas (`beta = 2`, 2D DPP extension of Dyson)
/// mapped radially to a 2D Gaussian spectral density `N(0, sigma^2 I_2)`.
///
/// In the Ginibre disk equilibrium (`U = 0.5 sum |z_i|^2 - sum_{i<j} ln |z_i - z_j|`, `D = 0.5`),
/// particles populate a uniform disk of radius `R = sqrt(M)` with `beta = 2` repulsion.
/// Mapping radial coordinate `u_r = clip(|z|^2 / R^2, eps, 1-eps)` through the Rayleigh inverse CDF
/// `r_gauss = sigma * sqrt(-2 * ln(1 - u_r))` preserves angular isotropy and 2D local repulsion
/// while exactly matching the target 2D Gaussian marginal `N(0, sigma^2 I_2)`.
pub fn consolidate_ginibre_2d_rff(
    n_modes: usize,
    sigma: f64,
    steps: usize,
    dt: f64,
    rng: &mut Rng,
) -> Vec<(f64, f64)> {
    let radius = (n_modes as f64).sqrt();
    let mut pts: Vec<(f64, f64)> = (0..n_modes)
        .map(|_| {
            let r = 0.85 * radius * rng.uniform01().sqrt();
            let th = 2.0 * PI * rng.uniform01();
            (r * th.cos(), r * th.sin())
        })
        .collect();

    let eps2 = 0.04 * 0.04;
    let noise_scale = dt.sqrt(); // D = 1/2 -> beta = 2
    let mut next_pts = vec![(0.0, 0.0); n_modes];

    for _ in 0..steps {
        for i in 0..n_modes {
            let (xi, yi) = pts[i];
            let mut fx = -xi;
            let mut fy = -yi;
            for j in 0..n_modes {
                if i == j {
                    continue;
                }
                let dx = xi - pts[j].0;
                let dy = yi - pts[j].1;
                let inv_r2 = 1.0 / (dx * dx + dy * dy + eps2);
                fx += dx * inv_r2;
                fy += dy * inv_r2;
            }
            next_pts[i] = (
                xi + fx * dt + noise_scale * rng.normal(),
                yi + fy * dt + noise_scale * rng.normal(),
            );
        }
        std::mem::swap(&mut pts, &mut next_pts);
    }

    // Rank-calibrated radial mapping to exact 2D Gaussian marginal while preserving 2D repulsion angles
    let mut indexed: Vec<(usize, f64, f64)> = pts
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| (i, x * x + y * y, y.atan2(x)))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut out = vec![(0.0, 0.0); n_modes];
    let r2_disk = n_modes as f64;
    for (rank, &(orig_i, r2, theta)) in indexed.iter().enumerate() {
        // Blend exact Ginibre disk CDF `r^2 / M` with empirical rank safeguard at boundary
        let u_disk = (r2 / r2_disk).clamp(0.0, 0.999_999);
        let u_rank = (rank as f64 + 0.5) / n_modes as f64;
        // Pure Ginibre disk CDF inside bulk (clamped smoothly near boundary)
        let u = if u_disk >= 0.985 { u_rank } else { u_disk.max(1e-6) };
        let r_gauss = sigma * (-2.0 * (1.0 - u).ln()).sqrt();
        out[orig_i] = (r_gauss * theta.cos(), r_gauss * theta.sin());
    }
    out
}
