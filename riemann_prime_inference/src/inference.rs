//! Continuous wave fields and retrodictive / bidirectional inference operators
//! for the 4 preregistered collapse media (`riemann_prime_collapse`,
//! `random_collapse`, `dyson_gue_collapse`, `fourier_linear_collapse`).

use std::f64::consts::PI;

use crate::riemann::{dyson_gue_frequencies, first_k_primes};
use crate::rng::Rng;

/// Number of spectral wave modes across all 4 arms (PLAN.md §1).
pub const N_MODES: usize = 128;

/// Number of continuous scan points for retrodictive wave collapse (PLAN.md §2 Suite I1).
pub const N_SCAN_POINTS: usize = 400;

/// Supported wave-collapse inference arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollapseArm {
    RiemannPrime,
    Random,
    DysonGue,
    FourierLinear,
}

impl CollapseArm {
    pub const ALL: [CollapseArm; 4] = [
        CollapseArm::RiemannPrime,
        CollapseArm::Random,
        CollapseArm::DysonGue,
        CollapseArm::FourierLinear,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            CollapseArm::RiemannPrime => "riemann_prime_collapse",
            CollapseArm::Random => "random_collapse",
            CollapseArm::DysonGue => "dyson_gue_collapse",
            CollapseArm::FourierLinear => "fourier_linear_collapse",
        }
    }

    pub const fn is_linear(self) -> bool {
        matches!(self, CollapseArm::FourierLinear)
    }
}

/// Pre-tabulated continuous 1D wave field `W(x)` evaluated via linear interpolation
/// inside the physical alphabet cavity `[cavity_min, cavity_max]`.
#[derive(Clone, Debug)]
pub struct WaveField {
    pub arm: CollapseArm,
    pub k_vocab: usize,
    pub coords: Vec<f64>,
    pub primes: Vec<usize>,
    pub cavity_min: f64,
    pub cavity_max: f64,
    x_min: f64,
    dx: f64,
    values: Vec<f64>,
}

impl WaveField {
    /// Builds a continuous Fejer-windowed wave field (`sigma_w = 0.0`) for the given arm.
    pub fn build(
        arm: CollapseArm,
        k_vocab: usize,
        riemann_zeros: &[f64],
        rng: &mut Rng,
    ) -> Self {
        Self::build_damped(arm, k_vocab, riemann_zeros, 0.0, rng)
    }

    /// Builds a continuous Fejer-windowed wave field analytically convolved with a Gaussian
    /// phase-noise kernel of width `sigma_w` (`exp(-0.5 * omega_n^2 * sigma_w^2)` per mode).
    pub fn build_damped(
        arm: CollapseArm,
        k_vocab: usize,
        riemann_zeros: &[f64],
        sigma_w: f64,
        rng: &mut Rng,
    ) -> Self {
        assert!(riemann_zeros.len() >= N_MODES);
        let primes = first_k_primes(k_vocab);

        match arm {
            CollapseArm::RiemannPrime | CollapseArm::Random | CollapseArm::DysonGue => {
                let coords: Vec<f64> = primes.iter().map(|&p| (p as f64).ln()).collect();
                let cavity_min = 1.7_f64.ln();
                let cavity_max = 8.5_f64;

                let (freqs, phases): (Vec<f64>, Vec<f64>) = match arm {
                    CollapseArm::RiemannPrime => {
                        (riemann_zeros[..N_MODES].to_vec(), vec![0.0; N_MODES])
                    }
                    CollapseArm::Random => {
                        let f_min = riemann_zeros[0];
                        let f_max = riemann_zeros[N_MODES - 1];
                        let f: Vec<f64> = (0..N_MODES)
                            .map(|_| rng.uniform(f_min, f_max))
                            .collect();
                        let ph: Vec<f64> = (0..N_MODES)
                            .map(|_| rng.uniform(0.0, 2.0 * PI))
                            .collect();
                        (f, ph)
                    }
                    CollapseArm::DysonGue => {
                        let f_min = riemann_zeros[0];
                        let f_max = riemann_zeros[N_MODES - 1];
                        let f = dyson_gue_frequencies(N_MODES, f_min, f_max, rng);
                        (f, vec![0.0; N_MODES])
                    }
                    CollapseArm::FourierLinear => unreachable!(),
                };

                let x_min = 0.0_f64;
                let x_max = 8.6_f64;
                let dx = 0.0005_f64;
                let n_grid = ((x_max - x_min) / dx).ceil() as usize + 2;
                let mut values = Vec::with_capacity(n_grid);
                let weights: Vec<f64> = (0..N_MODES)
                    .map(|n| {
                        let fejer = 1.0 - (n as f64) / (N_MODES as f64);
                        if sigma_w > 0.0 {
                            let damp = (-0.5 * freqs[n] * freqs[n] * sigma_w * sigma_w).exp();
                            fejer * damp
                        } else {
                            fejer
                        }
                    })
                    .collect();

                for i in 0..n_grid {
                    let u = x_min + (i as f64) * dx;
                    let mut sum = 0.0;
                    for m in 0..N_MODES {
                        sum += weights[m] * (freqs[m] * u + phases[m]).cos();
                    }
                    values.push(-2.0 * sum);
                }

                Self {
                    arm,
                    k_vocab,
                    coords,
                    primes,
                    cavity_min,
                    cavity_max,
                    x_min,
                    dx,
                    values,
                }
            }
            CollapseArm::FourierLinear => {
                // Linear additive grid: c_k -> k + 1 in {1..=K}, period L = 2*K + 4 (= 28 for K=12).
                let coords: Vec<f64> = (1..=k_vocab).map(|k| k as f64).collect();
                let period = (2 * k_vocab + 4) as f64;
                let cavity_min = 0.5_f64;
                let cavity_max = (k_vocab as f64) + 0.5;

                // 128 exact integer Fourier harmonics of fundamental 2*pi/L synthesized
                // for the linear alphabet comb sum_{k=1}^K delta_L(s - k):
                // W_F(s) = 2 * sum_{m=1}^{128} (1 - (m-1)/128) * (A_m cos(w_m s) + B_m sin(w_m s)).
                let mut a_coeff = Vec::with_capacity(N_MODES);
                let mut b_coeff = Vec::with_capacity(N_MODES);
                let mut omegas = Vec::with_capacity(N_MODES);
                for m_idx in 0..N_MODES {
                    let m = (m_idx + 1) as f64;
                    let w = 2.0 * PI * m / period;
                    let fejer = 1.0 - (m_idx as f64) / (N_MODES as f64);
                    let mut ac = 0.0;
                    let mut bc = 0.0;
                    for &sk in &coords {
                        ac += (w * sk).cos();
                        bc += (w * sk).sin();
                    }
                    omegas.push(w);
                    a_coeff.push(2.0 * fejer * ac);
                    b_coeff.push(2.0 * fejer * bc);
                }

                let x_min = 0.0_f64;
                let x_max = period + 1.0;
                let dx = 0.002_f64;
                let n_grid = ((x_max - x_min) / dx).ceil() as usize + 2;
                let mut values = Vec::with_capacity(n_grid);
                for i in 0..n_grid {
                    let s = x_min + (i as f64) * dx;
                    let mut sum = 0.0;
                    for m in 0..N_MODES {
                        let ang = omegas[m] * s;
                        sum += a_coeff[m] * ang.cos() + b_coeff[m] * ang.sin();
                    }
                    values.push(sum);
                }

                Self {
                    arm,
                    k_vocab,
                    coords,
                    primes,
                    cavity_min,
                    cavity_max,
                    x_min,
                    dx,
                    values,
                }
            }
        }
    }

    /// Evaluates the raw continuous wave `W(x)` at `x`, returning `0.0` outside the physical cavity.
    #[inline]
    pub fn eval_cavity(&self, x: f64) -> f64 {
        if x < self.cavity_min || x > self.cavity_max {
            return 0.0;
        }
        self.eval_raw(x)
    }

    /// Evaluates the underlying continuous wave `W(x)` without cavity truncation (for tests/traces).
    #[inline]
    pub fn eval_raw(&self, x: f64) -> f64 {
        if x < self.x_min {
            return 0.0;
        }
        let pos = (x - self.x_min) / self.dx;
        let idx = pos as usize;
        if idx + 1 >= self.values.len() {
            return 0.0;
        }
        let frac = pos - (idx as f64);
        self.values[idx] * (1.0 - frac) + self.values[idx + 1] * frac
    }

    /// Evaluates the Past x Future retrodictive collapse score at candidate origin `u`
    /// given composite observation `u_obs`:
    /// `S_origen(u | u_obs) = max(0, W(u)) * max(0, W(u_obs - u))`.
    #[inline]
    pub fn collapse_score(&self, u: f64, u_obs: f64) -> f64 {
        let w_past = self.eval_cavity(u).max(0.0);
        if w_past <= 0.0 {
            return 0.0;
        }
        let w_future = self.eval_cavity(u_obs - u).max(0.0);
        w_past * w_future
    }

    /// Projects a continuous coordinate `x` to the nearest atomic concept index `0..k_vocab`.
    #[inline]
    pub fn nearest_concept(&self, x: f64) -> usize {
        let mut best_k = 0;
        let mut best_d = f64::INFINITY;
        for (k, &ck) in self.coords.iter().enumerate() {
            let d = (x - ck).abs();
            if d < best_d {
                best_d = d;
                best_k = k;
            }
        }
        best_k
    }

    /// Composed continuous coordinate of atomic pair `(a, b)`:
    /// `ln(p_a * p_b) = ln(p_a) + ln(p_b)` for log arms, `s_a + s_b` for Fourier linear.
    #[inline]
    pub fn compose_pair(&self, a: usize, b: usize) -> f64 {
        self.coords[a] + self.coords[b]
    }
}

/// Output of a single retrodictive origin-tracing collapse query.
#[derive(Clone, Debug)]
pub struct TraceResult {
    pub pred_a: usize,
    pub pred_b: usize,
    pub peak_u1: f64,
    pub peak_u2: f64,
    pub peak_score: f64,
    pub peak_margin_snr: f64,
}

/// Smooth continuous Past-prior landscape `pi_pasado(u)` built via regularized
/// Nadaraya-Watson kernel smoothing over atomic priors `P(c_k)` with baseline `min_p`.
/// Strictly bounded in `[min_p, max_p]` everywhere on R so it never extrapolates boundary
/// prime priors into out-of-vocabulary coordinates (`p > p_K`) or flat Voronoi mesas.
#[derive(Clone, Debug)]
pub struct SmoothPriorField {
    coords: Vec<f64>,
    probs: Vec<f64>,
    bg_prob: f64,
    eps_bg: f64,
    inv_2tau2: f64,
}

impl SmoothPriorField {
    pub fn new(coords: &[f64], probs: &[f64], tau: f64) -> Self {
        assert_eq!(coords.len(), probs.len());
        let bg_prob = probs.iter().cloned().fold(f64::INFINITY, f64::min);
        Self {
            coords: coords.to_vec(),
            probs: probs.to_vec(),
            bg_prob,
            eps_bg: 0.05,
            inv_2tau2: 0.5 / (tau * tau),
        }
    }

    #[inline]
    pub fn eval_single(&self, u: f64) -> f64 {
        let mut num = self.eps_bg * self.bg_prob;
        let mut den = self.eps_bg;
        for (&ck, &pk) in self.coords.iter().zip(self.probs.iter()) {
            let d = u - ck;
            let w = (-(d * d) * self.inv_2tau2).exp();
            num += pk * w;
            den += w;
        }
        num / den
    }

    #[inline]
    pub fn eval_pair(&self, u1: f64, u2: f64) -> f64 {
        self.eval_single(u1) * self.eval_single(u2)
    }
}

/// Runs the continuous 400-point retrodictive wave-collapse scan over the active
/// cavity support `u in [max(cavity_min, u_obs - cavity_max), u_obs / 2]`.
///
/// If `prior` is `Some`, multiplies the collapse field by `pi_pasado(u, u_obs - u)`
/// (Suite I3 Bayesian transactional collapse).
///
/// If multiple distinct atomic pairs achieve local maxima within 98% of the global peak
/// (as occurs systematically for additive sum collisions in `fourier_linear_collapse`),
/// breaks ties uniformly using `rng`.
pub fn trace_origin(
    field: &WaveField,
    u_obs: f64,
    true_pair: Option<(usize, usize)>,
    prior: Option<&SmoothPriorField>,
    rng: &mut Rng,
) -> TraceResult {
    let scan_lo = field.cavity_min.max(u_obs - field.cavity_max);
    let scan_hi = (u_obs * 0.5).min(field.cavity_max);

    if scan_hi <= scan_lo + 1e-9 {
        let a = rng.usize_range(field.k_vocab);
        let mut b = rng.usize_range(field.k_vocab - 1);
        if b >= a {
            b += 1;
        }
        let (pa, pb) = if a < b { (a, b) } else { (b, a) };
        return TraceResult {
            pred_a: pa,
            pred_b: pb,
            peak_u1: field.coords[pa],
            peak_u2: field.coords[pb],
            peak_score: 0.0,
            peak_margin_snr: 0.0,
        };
    }

    let mut us = Vec::with_capacity(N_SCAN_POINTS);
    let mut scores = Vec::with_capacity(N_SCAN_POINTS);
    let step = (scan_hi - scan_lo) / ((N_SCAN_POINTS - 1) as f64);
    let mut max_score = 0.0_f64;

    for i in 0..N_SCAN_POINTS {
        let u = scan_lo + (i as f64) * step;
        let base = field.collapse_score(u, u_obs);
        let s = match prior {
            Some(p) if base > 0.0 => base * p.eval_pair(u, u_obs - u),
            _ => base,
        };
        if s > max_score {
            max_score = s;
        }
        us.push(u);
        scores.push(s);
    }

    if max_score <= 1e-12 {
        let a = rng.usize_range(field.k_vocab);
        let mut b = rng.usize_range(field.k_vocab - 1);
        if b >= a {
            b += 1;
        }
        let (pa, pb) = if a < b { (a, b) } else { (b, a) };
        return TraceResult {
            pred_a: pa,
            pred_b: pb,
            peak_u1: field.coords[pa],
            peak_u2: field.coords[pb],
            peak_score: 0.0,
            peak_margin_snr: 0.0,
        };
    }

    // Refine local maxima with a 5-point continuous micro-search around each grid peak
    // so narrow peaks are never clipped by grid phase.
    let eval_at = |u: f64| -> f64 {
        let base = field.collapse_score(u, u_obs);
        match prior {
            Some(p) if base > 0.0 => base * p.eval_pair(u, u_obs - u),
            _ => base,
        }
    };

    let mut local_peaks: Vec<(f64, f64, usize, usize)> = Vec::new();
    let mut refined_max = 0.0_f64;
    for i in 0..N_SCAN_POINTS {
        let s = scores[i];
        if s <= 0.0 {
            continue;
        }
        let left_ok = i == 0 || s >= scores[i - 1];
        let right_ok = i + 1 == N_SCAN_POINTS || s >= scores[i + 1];
        if left_ok && right_ok {
            let mut best_u = us[i];
            let mut best_s = s;
            let lo = (us[i] - step).max(scan_lo);
            let hi = (us[i] + step).min(scan_hi);
            for j in 0..=8 {
                let cand_u = lo + (hi - lo) * (j as f64) / 8.0;
                let cand_s = eval_at(cand_u);
                if cand_s > best_s {
                    best_s = cand_s;
                    best_u = cand_u;
                }
            }
            if best_s > refined_max {
                refined_max = best_s;
            }
            let k1 = field.nearest_concept(best_u);
            let k2 = field.nearest_concept(u_obs - best_u);
            let (pa, pb) = if k1 <= k2 { (k1, k2) } else { (k2, k1) };
            local_peaks.push((best_u, best_s, pa, pb));
        }
    }

    // Deduplicate by projected atomic pair, keeping the strongest peak per distinct pair.
    let mut best_by_pair: Vec<(f64, f64, usize, usize)> = Vec::new();
    for &(u, s, pa, pb) in &local_peaks {
        if let Some(existing) = best_by_pair
            .iter_mut()
            .find(|entry| entry.2 == pa && entry.3 == pb)
        {
            if s > existing.1 {
                *existing = (u, s, pa, pb);
            }
        } else {
            best_by_pair.push((u, s, pa, pb));
        }
    }

    // Compute peak_margin_snr = true_peak / max_spurious_peak (if true_pair provided).
    let peak_margin_snr = if let Some((ta, tb)) = true_pair {
        let mut true_peak = 0.0_f64;
        let mut spurious_peak = 0.0_f64;
        for &(_, s, pa, pb) in &best_by_pair {
            if pa == ta && pb == tb {
                if s > true_peak {
                    true_peak = s;
                }
            } else if s > spurious_peak {
                spurious_peak = s;
            }
        }
        // Floor spurious denominator slightly to prevent division by zero when wave is zero elsewhere.
        let denom = spurious_peak.max(1e-3 * refined_max.max(1e-6));
        true_peak / denom
    } else {
        0.0
    };

    // Select among distinct pairs within 98% of refined_max (fair uniform tie-breaking).
    let threshold = 0.98 * refined_max;
    let tied: Vec<(f64, f64, usize, usize)> = best_by_pair
        .into_iter()
        .filter(|entry| entry.1 >= threshold)
        .collect();

    let chosen = if tied.len() == 1 {
        tied[0]
    } else {
        tied[rng.usize_range(tied.len())]
    };

    TraceResult {
        pred_a: chosen.2,
        pred_b: chosen.3,
        peak_u1: chosen.0,
        peak_u2: u_obs - chosen.0,
        peak_score: chosen.1,
        peak_margin_snr,
    }
}

/// Evaluates the continuous collapse profile on a fixed grid (for plotting & visual traces).
pub fn sample_collapse_profile(
    field: &WaveField,
    u_obs: f64,
    n_points: usize,
) -> Vec<(f64, f64, f64, f64)> {
    let lo = field.cavity_min.max(u_obs - field.cavity_max);
    let hi = (u_obs * 0.5).min(field.cavity_max);
    let mut out = Vec::with_capacity(n_points);
    if hi <= lo + 1e-9 || n_points < 2 {
        return out;
    }
    let step = (hi - lo) / ((n_points - 1) as f64);
    for i in 0..n_points {
        let u = lo + (i as f64) * step;
        let wp = field.eval_cavity(u);
        let wf = field.eval_cavity(u_obs - u);
        let sc = wp.max(0.0) * wf.max(0.0);
        out.push((u, wp, wf, sc));
    }
    out
}

/// Nonlinear downstream causal law "on the other side" (PLAN.md §2 Suite I2).
/// Input indices are ordered as `a <= b`.
#[inline]
pub fn eval_causal_outcome(a: usize, b: usize) -> (f64, f64, usize) {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let af = lo as f64;
    let bf = hi as f64;
    let y1 = (0.9 * af + 1.3 * bf).sin() + 0.45 * (((lo + 1) * (hi + 1)) as f64 / 4.0).cos();
    let sign = if (lo + hi) % 2 == 0 { 1.0 } else { -1.0 };
    let y2 = (1.1 * af - 0.7 * bf).cos() + 0.35 * sign;
    let c1 = if y1 >= 0.0 { 2 } else { 0 };
    let c2 = if y2 >= 0.0 { 1 } else { 0 };
    (y1, y2, c1 + c2)
}

/// Computes pair-level partial F1 between predicted pair `(pa, pb)` and true distinct pair `(ta, tb)`.
#[inline]
pub fn pair_partial_f1(pa: usize, pb: usize, ta: usize, tb: usize) -> f64 {
    if pa == pb {
        // Predicted a degenerate singleton pair {pa}; compare multiset intersection size 0 or 1.
        if pa == ta || pa == tb {
            0.5
        } else {
            0.0
        }
    } else {
        let mut hits = 0usize;
        if pa == ta || pa == tb {
            hits += 1;
        }
        if pb == ta || pb == tb {
            hits += 1;
        }
        (hits as f64) / 2.0
    }
}

/// Returns all `K*(K-1)/2` distinct ordered pairs `(a, b)` with `0 <= a < b < k_vocab`.
pub fn all_distinct_pairs(k_vocab: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(k_vocab * (k_vocab - 1) / 2);
    for a in 0..k_vocab {
        for b in (a + 1)..k_vocab {
            pairs.push((a, b));
        }
    }
    pairs
}

/// Computes the minimum separation between any two distinct products `p_a * p_b` in log-space.
pub fn min_prime_product_log_gap(k_vocab: usize) -> f64 {
    let primes = first_k_primes(k_vocab);
    let mut vals: Vec<f64> = Vec::with_capacity(k_vocab * (k_vocab - 1) / 2);
    for a in 0..k_vocab {
        for b in (a + 1)..k_vocab {
            vals.push(((primes[a] * primes[b]) as f64).ln());
        }
    }
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut min_gap = f64::INFINITY;
    for w in vals.windows(2) {
        let g = w[1] - w[0];
        if g < min_gap {
            min_gap = g;
        }
    }
    min_gap
}
