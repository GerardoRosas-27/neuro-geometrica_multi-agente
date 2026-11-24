//! Past-Future wave collapse physics, all 4 collapse operators, 7-parameter coarse-to-fine
//! exact configuration search, and blind Guinand-Weil dual resonance zero discovery.

use std::f64::consts::PI;

use crate::riemann::{is_prime_int, is_prime_power_int, von_mangoldt_table, C64};
use crate::rng::Rng;

/// Physical configuration of the Past-Future wave collapse medium (7 continuous parameters).
#[derive(Clone, Copy, Debug)]
pub struct CollapseConfig {
    /// Time slice relative to rendezvous (exact theoretical optimum: 0.0).
    pub tau: f64,
    /// Sum of past and future frequency scales `k_+ + k_-` (exact optimum: 1.0).
    pub k_sum: f64,
    /// Relative collapse phase `phi_+ - phi_-` in `[-pi, pi]` (exact optimum: pi).
    pub delta_phi: f64,
    /// Total spatial scaling exponent `x^{sigma_tot}` (log-space critical compensation: +0.5; x-space: -0.5).
    pub sigma_tot: f64,
    /// Residual quadratic dispersion mismatch `-beta_disp * gamma_n^2` (exact optimum: 0.0).
    pub beta_disp: f64,
    /// Spectral window exponent `w_n = (1 - n/M)^{p_win}` (Fejer = 1.0).
    pub p_win: f64,
    /// Past vs future propagation velocity asymmetry `(1 + v_asym) * gamma_n * tau`.
    pub v_asym: f64,
}

impl CollapseConfig {
    /// Reference analytic collapse configuration (for Suite C1 & C3).
    pub fn analytic_ref(sigma_tot: f64) -> Self {
        Self {
            tau: 0.0,
            k_sum: 1.0,
            delta_phi: PI,
            sigma_tot,
            beta_disp: 0.0,
            p_win: 1.0,
            v_asym: 0.0,
        }
    }

    pub fn bounds() -> [(f64, f64); 7] {
        [
            (-1.0, 1.0),    // tau
            (0.70, 1.30),   // k_sum
            (-PI, PI),      // delta_phi
            (-0.75, 0.75),  // sigma_tot
            (-0.03, 0.03),  // beta_disp
            (0.0, 3.0),     // p_win
            (-0.40, 0.40),  // v_asym
        ]
    }

    pub fn from_array(a: [f64; 7]) -> Self {
        Self {
            tau: a[0],
            k_sum: a[1],
            delta_phi: wrap_pi(a[2]),
            sigma_tot: a[3],
            beta_disp: a[4],
            p_win: a[5],
            v_asym: a[6],
        }
    }

    pub fn to_array(self) -> [f64; 7] {
        [
            self.tau,
            self.k_sum,
            wrap_pi(self.delta_phi),
            self.sigma_tot,
            self.beta_disp,
            self.p_win,
            self.v_asym,
        ]
    }

    pub fn random_in_bounds(rng: &mut Rng) -> Self {
        let b = Self::bounds();
        let mut a = [0.0; 7];
        for i in 0..7 {
            a[i] = rng.uniform(b[i].0, b[i].1);
        }
        Self::from_array(a)
    }
}

/// Wraps angle into `(-pi, pi]`.
#[inline]
pub fn wrap_pi(mut a: f64) -> f64 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a <= -PI {
        a += 2.0 * PI;
    }
    a
}

/// Shortest angular distance between `a` and `target` in `[0, pi]`.
#[inline]
pub fn angular_error(a: f64, target: f64) -> f64 {
    wrap_pi(a - target).abs()
}

/// Physical dispersion relation `omega_n = (1 + v_asym) * gamma_n + 0.008 * gamma_n^2`.
#[inline]
pub fn dispersion_omega(gamma: f64, v_asym: f64) -> f64 {
    (1.0 + v_asym) * gamma + 0.008 * gamma * gamma
}

/// Explicit complex past & future mode amplitudes for mode `n` at `(x, cfg)`.
#[inline]
pub fn past_future_mode_pair(
    x: f64,
    gamma: f64,
    weight: f64,
    cfg: &CollapseConfig,
) -> (C64, C64) {
    let ln_x = x.ln();
    let sqrt_w = weight.max(0.0).sqrt();
    let amp_p = sqrt_w * x.powf(0.5 * cfg.sigma_tot);
    let amp_f = sqrt_w * x.powf(0.5 * cfg.sigma_tot);
    let omega = dispersion_omega(gamma, cfg.v_asym);
    let k_half = 0.5 * cfg.k_sum;
    let phi_p = k_half * gamma * ln_x - 0.5 * omega * cfg.tau - 0.5 * cfg.beta_disp * gamma * gamma + 0.5 * cfg.delta_phi;
    let phi_f = -k_half * gamma * ln_x + 0.5 * omega * cfg.tau + 0.5 * cfg.beta_disp * gamma * gamma - 0.5 * cfg.delta_phi;
    (C64::exp_i(phi_p).scale(amp_p), C64::exp_i(phi_f).scale(amp_f))
}

/// Operator 1: Strict-emergence transactional half-frequency modal collapse:
/// `I_trans(x, tau) = 2 * Re sum_n psi_past_n(x, tau) * conj(psi_future_n(x, tau))`.
pub fn eval_transactional_collapse(x: f64, freqs: &[f64], cfg: &CollapseConfig) -> f64 {
    let m = freqs.len() as f64;
    let ln_x = x.ln();
    let x_scale = 2.0 * x.powf(cfg.sigma_tot);
    let mut acc = 0.0;
    for (n, &g) in freqs.iter().enumerate() {
        let w = (1.0 - (n as f64) / m).powf(cfg.p_win);
        let omega = dispersion_omega(g, cfg.v_asym);
        let phase = cfg.k_sum * g * ln_x - omega * cfg.tau - cfg.beta_disp * g * g + cfg.delta_phi;
        acc += w * phase.cos();
    }
    x_scale * acc
}

/// Evaluates transactional collapse directly via explicit complex past & future wave pairs
/// (used in unit test to prove algebraic equivalence to `eval_transactional_collapse`).
pub fn eval_transactional_explicit_complex(x: f64, freqs: &[f64], cfg: &CollapseConfig) -> f64 {
    let m = freqs.len() as f64;
    let mut acc = 0.0;
    for (n, &g) in freqs.iter().enumerate() {
        let w = (1.0 - (n as f64) / m).powf(cfg.p_win);
        let (p, f) = past_future_mode_pair(x, g, w, cfg);
        acc += p.mul(f.conj()).re;
    }
    2.0 * acc
}

/// Isolated past half-frequency wave alone (`2 * Re sum_n psi_past_n`, flipped to positive sign for fair AUC test).
pub fn eval_past_wave_alone(x: f64, freqs: &[f64], cfg: &CollapseConfig) -> f64 {
    let m = freqs.len() as f64;
    let ln_x = x.ln();
    let amp = -2.0 * x.powf(0.5 * cfg.sigma_tot);
    let mut acc = 0.0;
    for (n, &g) in freqs.iter().enumerate() {
        let w = (1.0 - (n as f64) / m).powf(cfg.p_win).sqrt();
        let omega = dispersion_omega(g, cfg.v_asym);
        let phase = 0.5 * cfg.k_sum * g * ln_x - 0.5 * omega * cfg.tau;
        acc += w * phase.cos();
    }
    amp * acc
}

/// Isolated future half-frequency wave alone (`2 * Re sum_n psi_future_n`, flipped to positive sign for fair AUC test).
pub fn eval_future_wave_alone(x: f64, freqs: &[f64], cfg: &CollapseConfig) -> f64 {
    let m = freqs.len() as f64;
    let ln_x = x.ln();
    let amp = -2.0 * x.powf(0.5 * cfg.sigma_tot);
    let mut acc = 0.0;
    for (n, &g) in freqs.iter().enumerate() {
        let w = (1.0 - (n as f64) / m).powf(cfg.p_win).sqrt();
        let omega = dispersion_omega(g, cfg.v_asym);
        let phase = -0.5 * cfg.k_sum * g * ln_x + 0.5 * omega * cfg.tau;
        acc += w * phase.cos();
    }
    amp * acc
}

/// Operator 2: Dispersive linear counter-propagating superposition `Psi_past + Psi_future`.
pub fn eval_linear_counterprop(x: f64, freqs: &[f64], cfg: &CollapseConfig) -> (f64, f64, f64) {
    let m = freqs.len() as f64;
    let ln_x = x.ln();
    let amp = x.powf(cfg.sigma_tot);
    let mut past = 0.0;
    let mut future = 0.0;
    for (n, &g) in freqs.iter().enumerate() {
        let w = (1.0 - (n as f64) / m).powf(cfg.p_win);
        let omega = dispersion_omega(g, cfg.v_asym);
        let p_phase = cfg.k_sum * g * ln_x - omega * cfg.tau + cfg.delta_phi;
        let f_phase = cfg.k_sum * g * ln_x + omega * cfg.tau + cfg.delta_phi;
        past += w * p_phase.cos();
        future += w * f_phase.cos();
    }
    let wp = amp * past;
    let wf = amp * future;
    (wp, wf, wp + wf)
}

/// Operator 3: Holographic Born-rule interference `|Psi_p + Psi_f|^2 - |Psi_p|^2 - |Psi_f|^2`
/// between retarded spectral past wave and smooth advanced future carrier `-x^{sigma_-}`.
pub fn eval_holographic_born(x: f64, freqs: &[f64], cfg: &CollapseConfig) -> f64 {
    let m = freqs.len() as f64;
    let ln_x = x.ln();
    let mut psi_p = C64::new(0.0, 0.0);
    let amp_p = x.powf(0.5 * cfg.sigma_tot);
    for (n, &g) in freqs.iter().enumerate() {
        let w = (1.0 - (n as f64) / m).powf(cfg.p_win);
        let omega = dispersion_omega(g, cfg.v_asym);
        let phase = cfg.k_sum * g * ln_x - omega * cfg.tau - cfg.beta_disp * g * g + cfg.delta_phi;
        psi_p = psi_p.add(C64::exp_i(phase).scale(amp_p * w));
    }
    let psi_f = C64::new(x.powf(0.5 * cfg.sigma_tot), 0.0);
    let total_sq = psi_p.add(psi_f).norm_sq();
    total_sq - psi_p.norm_sq() - psi_f.norm_sq()
}

/// Operator 4 (Negative control): Global uncoupled bilinear product `(sum psi_past) * conj(sum psi_future)`.
pub fn eval_naive_bilinear(x: f64, freqs: &[f64], cfg: &CollapseConfig) -> f64 {
    let m = freqs.len() as f64;
    let mut sum_p = C64::new(0.0, 0.0);
    let mut sum_f = C64::new(0.0, 0.0);
    for (n, &g) in freqs.iter().enumerate() {
        let w = (1.0 - (n as f64) / m).powf(cfg.p_win);
        let (p, f) = past_future_mode_pair(x, g, w, cfg);
        sum_p = sum_p.add(p);
        sum_f = sum_f.add(f);
    }
    2.0 * sum_p.mul(sum_f.conj()).re
}

#[derive(Clone, Copy, Debug)]
pub struct DiscrimMetrics {
    pub prime_power_auc: f64,
    pub prime_only_auc: f64,
    pub prime_power_snr: f64,
    pub prime_only_snr: f64,
}

fn roc_auc_and_snr(pos: &[f64], neg: &[f64]) -> (f64, f64) {
    let mut wins = 0.0;
    for &p in pos {
        for &c in neg {
            if p > c {
                wins += 1.0;
            } else if (p - c).abs() < 1e-15 {
                wins += 0.5;
            }
        }
    }
    let auc = wins / ((pos.len() * neg.len()).max(1) as f64);
    let mu_p = pos.iter().sum::<f64>() / pos.len().max(1) as f64;
    let mu_n = neg.iter().sum::<f64>() / neg.len().max(1) as f64;
    let var_n = neg
        .iter()
        .map(|&v| (v - mu_n) * (v - mu_n))
        .sum::<f64>()
        / neg.len().max(1) as f64;
    let snr = (mu_p - mu_n) / var_n.sqrt().max(1e-12);
    (auc, snr)
}

/// Evaluates prime-only and prime-power ROC-AUC and SNR over integers `n in n_lo..=n_hi`.
pub fn eval_range_discrimination<F: Fn(f64) -> f64>(
    n_lo: usize,
    n_hi: usize,
    field: F,
) -> DiscrimMetrics {
    let mut pp_pos = Vec::new();
    let mut pp_neg = Vec::new();
    let mut p_pos = Vec::new();
    let mut p_neg = Vec::new();
    for n in n_lo..=n_hi {
        let v = field(n as f64);
        if is_prime_power_int(n) {
            pp_pos.push(v);
        } else {
            pp_neg.push(v);
        }
        if is_prime_int(n) {
            p_pos.push(v);
        } else {
            p_neg.push(v);
        }
    }
    let (pp_auc, pp_snr) = roc_auc_and_snr(&pp_pos, &pp_neg);
    let (p_auc, p_snr) = roc_auc_and_snr(&p_pos, &p_neg);
    DiscrimMetrics {
        prime_power_auc: pp_auc,
        prime_only_auc: p_auc,
        prime_power_snr: pp_snr,
        prime_only_snr: p_snr,
    }
}

/// Precomputed integer tables for blazing-fast objective evaluation during configuration search.
pub struct TrainWindowCache {
    pub ns: Vec<f64>,
    pub ln_ns: Vec<f64>,
    pub is_pp: Vec<bool>,
}

impl TrainWindowCache {
    pub fn new(n_lo: usize, n_hi: usize) -> Self {
        let mut ns = Vec::new();
        let mut ln_ns = Vec::new();
        let mut is_pp = Vec::new();
        for n in n_lo..=n_hi {
            ns.push(n as f64);
            ln_ns.push((n as f64).ln());
            is_pp.push(is_prime_power_int(n));
        }
        Self { ns, ln_ns, is_pp }
    }

    pub fn objective(&self, freqs: &[f64], m_use: usize, cfg: &CollapseConfig) -> f64 {
        self.objective_up_to(freqs, m_use, 60.0, cfg)
    }

    /// Evaluates scale-balanced objective strictly within `[2, x_cap]` (`x_cap <= 60`).
    pub fn objective_up_to(
        &self,
        freqs: &[f64],
        m_use: usize,
        x_cap: f64,
        cfg: &CollapseConfig,
    ) -> f64 {
        let m = m_use.min(freqs.len());
        let mf = m as f64;
        let mut weights = Vec::with_capacity(m);
        let mut base_phases = Vec::with_capacity(m);
        let mut k_freqs = Vec::with_capacity(m);
        for n in 0..m {
            let g = freqs[n];
            weights.push((1.0 - (n as f64) / mf).powf(cfg.p_win));
            let omega = dispersion_omega(g, cfg.v_asym);
            base_phases.push(-omega * cfg.tau - cfg.beta_disp * g * g + cfg.delta_phi);
            k_freqs.push(cfg.k_sum * g);
        }

        let mid_x = 0.42 * x_cap;
        let mut pos_all = Vec::with_capacity(28);
        let mut neg_all = Vec::with_capacity(36);
        let mut pos_lo = Vec::with_capacity(16);
        let mut neg_lo = Vec::with_capacity(16);
        let mut pos_hi = Vec::with_capacity(16);
        let mut neg_hi = Vec::with_capacity(24);
        for i in 0..self.ns.len() {
            if self.ns[i] > x_cap {
                break;
            }
            let ln_x = self.ln_ns[i];
            let mut acc = 0.0;
            for k in 0..m {
                acc += weights[k] * (k_freqs[k] * ln_x + base_phases[k]).cos();
            }
            let val = 2.0 * self.ns[i].powf(cfg.sigma_tot) * acc;
            let hi_half = self.ns[i] >= mid_x;
            if self.is_pp[i] {
                pos_all.push(val);
                if hi_half {
                    pos_hi.push(val);
                } else {
                    pos_lo.push(val);
                }
            } else {
                neg_all.push(val);
                if hi_half {
                    neg_hi.push(val);
                } else {
                    neg_lo.push(val);
                }
            }
        }
        let (auc_all, snr_all) = roc_auc_and_snr(&pos_all, &neg_all);
        if pos_lo.len() >= 2 && pos_hi.len() >= 2 && neg_lo.len() >= 2 && neg_hi.len() >= 2 {
            let (auc_lo, snr_lo) = roc_auc_and_snr(&pos_lo, &neg_lo);
            let (auc_hi, snr_hi) = roc_auc_and_snr(&pos_hi, &neg_hi);
            let snr_bal = 0.45 * snr_all + 0.55 * snr_lo.min(snr_hi);
            let auc_bal = 0.5 * auc_all + 0.25 * auc_lo + 0.25 * auc_hi;
            snr_bal + 20.0 * (auc_bal - 0.5)
        } else {
            snr_all + 20.0 * (auc_all - 0.5)
        }
    }
}

pub struct SearchOutcome {
    pub best_cfg: CollapseConfig,
    pub best_obj: f64,
    pub evals_used: usize,
    pub history: Vec<(usize, f64, f64, f64, f64)>, // (eval, obj, tau, k_sum, delta_phi)
}

fn clamp_vec(v: &mut [f64; 7]) {
    let b = CollapseConfig::bounds();
    for i in 0..7 {
        if i == 2 {
            v[i] = wrap_pi(v[i]);
        } else {
            v[i] = v[i].clamp(b[i].0, b[i].1);
        }
    }
}

/// Computes zero-mean binary prime-power target weights `t_i` on `x in [2, 60]` (pure training labels).
fn centered_train_targets(cache: &TrainWindowCache) -> Vec<f64> {
    let n = cache.ns.len() as f64;
    let p_frac = cache.is_pp.iter().filter(|&&b| b).count() as f64 / n;
    cache
        .is_pp
        .iter()
        .map(|&is_p| if is_p { 1.0 - p_frac } else { -p_frac })
        .collect()
}

/// Computes complex modal matched-filter projections `Z_n(k) = sum_{x in [2,60]} t(x) e^{i k gamma_n ln x}`
/// for the first `m_use` modes. Depends ONLY on the single 1D parameter `k_sum`.
fn modal_projections(
    cache: &TrainWindowCache,
    targets: &[f64],
    freqs: &[f64],
    m_use: usize,
    k_sum: f64,
) -> Vec<(f64, f64)> {
    let m = m_use.min(freqs.len());
    let mut z = Vec::with_capacity(m);
    for n in 0..m {
        let kf = k_sum * freqs[n];
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..cache.ns.len() {
            let a = kf * cache.ln_ns[i];
            re += targets[i] * a.cos();
            im += targets[i] * a.sin();
        }
        z.push((re, im));
    }
    z
}

/// Three-stage Coherent Wavefront Autofocus + Joint 7D DE + Full M=128 Polish.
///   1) 1D Spectral Scale Autofocus (`k_sum` over full `[0.70, 1.30]`, 260 evals)
///   2) 2D Time-Dispersion Autofocus (`(tau, beta_disp)` over full box with closed-form phase `delta_phi`, 720 evals)
///   3) Joint 7D DE on `M = 64` (1170 evals) + Full `M = 128` Coordinate/Ridge Polish (850 evals)
/// Total budget = EXACTLY `3000` evaluations on `[2, 60]`, zero prior target knowledge, zero leakage.
pub fn optimize_collapse_config(
    cache: &TrainWindowCache,
    freqs: &[f64],
    rng: &mut Rng,
    record_history: bool,
) -> SearchOutcome {
    let b = CollapseConfig::bounds();
    let targets = centered_train_targets(cache);
    let mut evals = 0usize;
    let mut history = Vec::new();
    let m_auto = 24usize;
    let mf_auto = m_auto as f64;
    let w_auto: Vec<f64> = (0..m_auto).map(|n| 1.0 - (n as f64) / mf_auto).collect();

    // --- Stage 1: 1D Spectral Scale Autofocus on k_sum in [0.70, 1.30] (260 evals) ---
    let mut best_k = rng.uniform(b[1].0, b[1].1);
    let mut best_spec_e = f64::NEG_INFINITY;
    let n_k_coarse = 200usize;
    let dk = (b[1].1 - b[1].0) / n_k_coarse as f64;
    for i in 0..n_k_coarse {
        let k_cand = b[1].0 + (i as f64 + rng.uniform01()) * dk;
        let zs = modal_projections(cache, &targets, freqs, m_auto, k_cand);
        let energy: f64 = zs
            .iter()
            .enumerate()
            .map(|(n, &(re, im))| w_auto[n] * (re * re + im * im))
            .sum();
        evals += 1;
        if energy > best_spec_e {
            best_spec_e = energy;
            best_k = k_cand;
        }
    }
    // Fine 1D zoom around discovered k_sum (60 evals -> 260 total)
    let zoom_lo = (best_k - 2.0 * dk).max(b[1].0);
    let zoom_hi = (best_k + 2.0 * dk).min(b[1].1);
    let dk_fine = (zoom_hi - zoom_lo) / 60.0;
    for i in 0..60 {
        let k_cand = zoom_lo + (i as f64 + 0.5) * dk_fine;
        let zs = modal_projections(cache, &targets, freqs, m_auto, k_cand);
        let energy: f64 = zs
            .iter()
            .enumerate()
            .map(|(n, &(re, im))| w_auto[n] * (re * re + im * im))
            .sum();
        evals += 1;
        if energy > best_spec_e {
            best_spec_e = energy;
            best_k = k_cand;
        }
    }

    // --- Stage 2: 2D Coherent Time-Dispersion Autofocus over (tau, beta_disp) (720 evals) ---
    // Note: evaluates at seed-random v_asym_0 (at tau=0, v_asym drops out physically).
    let v_asym_0 = rng.uniform(b[6].0, b[6].1);
    let z_locked = modal_projections(cache, &targets, freqs, m_auto, best_k);
    let eval_coh = |tau: f64, beta: f64| -> (f64, f64) {
        let mut cre = 0.0;
        let mut cim = 0.0;
        for n in 0..m_auto {
            let g = freqs[n];
            let omega = dispersion_omega(g, v_asym_0);
            let ang = -(omega * tau + beta * g * g);
            let ca = ang.cos();
            let sa = ang.sin();
            let (zre, zim) = z_locked[n];
            cre += w_auto[n] * (zre * ca - zim * sa);
            cim += w_auto[n] * (zre * sa + zim * ca);
        }
        let mag = (cre * cre + cim * cim).sqrt();
        let phi_opt = wrap_pi(-cim.atan2(cre));
        (mag, phi_opt)
    };

    let mut best_tau = 0.0;
    let mut best_beta = 0.0;
    let mut best_phi = 0.0;
    let mut best_coh = f64::NEG_INFINITY;
    let g2d = 24usize;
    let dtau_g = (b[0].1 - b[0].0) / g2d as f64;
    let dbeta_g = (b[4].1 - b[4].0) / g2d as f64;
    for it in 0..g2d {
        for ib in 0..g2d {
            let t_c = b[0].0 + (it as f64 + rng.uniform01()) * dtau_g;
            let b_c = b[4].0 + (ib as f64 + rng.uniform01()) * dbeta_g;
            let (mag, phi) = eval_coh(t_c, b_c);
            evals += 1;
            if mag > best_coh {
                best_coh = mag;
                best_tau = t_c;
                best_beta = b_c;
                best_phi = phi;
            }
        }
    }
    // Fine 2D zoom (12x12 = 144 evals -> 720 total in Stage 2, 980 cumulative)
    let t_lo = (best_tau - 1.5 * dtau_g).max(b[0].0);
    let t_hi = (best_tau + 1.5 * dtau_g).min(b[0].1);
    let b_lo = (best_beta - 1.5 * dbeta_g).max(b[4].0);
    let b_hi = (best_beta + 1.5 * dbeta_g).min(b[4].1);
    for it in 0..12 {
        for ib in 0..12 {
            let t_c = t_lo + (it as f64 + 0.5) * (t_hi - t_lo) / 12.0;
            let b_c = b_lo + (ib as f64 + 0.5) * (b_hi - b_lo) / 12.0;
            let (mag, phi) = eval_coh(t_c, b_c);
            evals += 1;
            if mag > best_coh {
                best_coh = mag;
                best_tau = t_c;
                best_beta = b_c;
                best_phi = phi;
            }
        }
    }

    let mut best_vec = [
        best_tau,
        best_k,
        best_phi,
        rng.uniform(b[3].0, b[3].1),
        best_beta,
        rng.uniform(b[5].0, b[5].1),
        v_asym_0,
    ];
    clamp_vec(&mut best_vec);
    if record_history {
        let full_obj = cache.objective(freqs, freqs.len(), &CollapseConfig::from_array(best_vec));
        history.push((evals, full_obj, best_vec[0], best_vec[1], wrap_pi(best_vec[2])));
    }

    // --- Stage 3: Joint 7D DE on M = 64 over full [2, 60] (26 * 45 = 1170 evals -> 2150 cumulative) ---
    let pop_size = 26usize;
    let gens = 45usize;
    let m_de = 64usize;
    let mut pop: Vec<[f64; 7]> = Vec::with_capacity(pop_size);
    pop.push(best_vec);
    while pop.len() < pop_size {
        let mut cand = best_vec;
        // Tight local search on locked phase params + full global exploration on amplitude/shaping params
        cand[0] += rng.normal() * 0.025;
        cand[1] += rng.normal() * 0.004;
        cand[2] = wrap_pi(cand[2] + rng.normal() * 0.40);
        cand[3] = rng.uniform(b[3].0, b[3].1);
        cand[4] += rng.normal() * 0.0012;
        cand[5] = rng.uniform(b[5].0, b[5].1);
        cand[6] = rng.uniform(b[6].0, b[6].1);
        clamp_vec(&mut cand);
        pop.push(cand);
    }

    let mut fit = vec![0.0; pop_size];
    let mut best_obj_de = f64::NEG_INFINITY;
    for i in 0..pop_size {
        let f = cache.objective(freqs, m_de, &CollapseConfig::from_array(pop[i]));
        fit[i] = f;
        evals += 1;
        if f > best_obj_de {
            best_obj_de = f;
            best_vec = pop[i];
        }
    }

    for g in 0..(gens - 1) {
        for i in 0..pop_size {
            let mut r1 = rng.usize_range(pop_size);
            while r1 == i {
                r1 = rng.usize_range(pop_size);
            }
            let mut r2 = rng.usize_range(pop_size);
            while r2 == i || r2 == r1 {
                r2 = rng.usize_range(pop_size);
            }
            let f_mut = rng.uniform(0.45, 0.85);
            let cr = 0.85;
            let j_rand = rng.usize_range(7);
            let mut trial = pop[i];
            for d in 0..7 {
                if d == j_rand || rng.uniform01() < cr {
                    let diff_best = if d == 2 {
                        wrap_pi(best_vec[d] - pop[i][d])
                    } else {
                        best_vec[d] - pop[i][d]
                    };
                    let diff_r = if d == 2 {
                        wrap_pi(pop[r1][d] - pop[r2][d])
                    } else {
                        pop[r1][d] - pop[r2][d]
                    };
                    trial[d] = pop[i][d] + f_mut * diff_best + f_mut * diff_r;
                }
            }
            clamp_vec(&mut trial);
            let f_trial = cache.objective(freqs, m_de, &CollapseConfig::from_array(trial));
            evals += 1;
            if f_trial >= fit[i] {
                pop[i] = trial;
                fit[i] = f_trial;
                if f_trial > best_obj_de {
                    best_obj_de = f_trial;
                    best_vec = trial;
                }
            }
        }
        if record_history && (g % 5 == 0 || g == gens - 2) {
            let full_obj = cache.objective(freqs, freqs.len(), &CollapseConfig::from_array(best_vec));
            history.push((evals, full_obj, best_vec[0], best_vec[1], wrap_pi(best_vec[2])));
        }
    }

    // --- Stage 4: Full M = 128 Adaptive Coordinate + Coupled-Ridge Polish (remaining 850 evals -> exact 3000) ---
    let target_total = 3000usize;
    let m_full = freqs.len();
    let mut cur = best_vec;
    let mut cur_obj = cache.objective(freqs, m_full, &CollapseConfig::from_array(cur));
    evals += 1;

    let mut steps = [0.008, 0.0015, 0.20, 0.20, 0.0008, 0.25, 0.06];
    let mut ridge_step = 0.010_f64;

    while evals < target_total {
        let mut improved = false;
        for d in 0..7 {
            if evals >= target_total {
                break;
            }
            for &sign in &[1.0, -1.0] {
                if evals >= target_total {
                    break;
                }
                let mut cand = cur;
                cand[d] += sign * steps[d];
                clamp_vec(&mut cand);
                let obj = cache.objective(freqs, m_full, &CollapseConfig::from_array(cand));
                evals += 1;
                if obj > cur_obj {
                    cur = cand;
                    cur_obj = obj;
                    improved = true;
                    break;
                }
            }
        }
        for &sign in &[1.0, -1.0] {
            if evals >= target_total {
                break;
            }
            let mut cand = cur;
            let dtau = sign * ridge_step;
            cand[0] += dtau;
            cand[4] -= 0.008 * dtau;
            clamp_vec(&mut cand);
            let obj = cache.objective(freqs, m_full, &CollapseConfig::from_array(cand));
            evals += 1;
            if obj > cur_obj {
                cur = cand;
                cur_obj = obj;
                improved = true;
                break;
            }
        }
        if !improved {
            for s in &mut steps {
                *s *= 0.55;
            }
            ridge_step *= 0.55;
        }
        if record_history {
            history.push((evals, cur_obj, cur[0], cur[1], wrap_pi(cur[2])));
        }
    }

    SearchOutcome {
        best_cfg: CollapseConfig::from_array(cur),
        best_obj: cur_obj,
        evals_used: evals,
        history,
    }
}

/// Uniform random search control with the EXACT same budget of `3000` full evaluations.
pub fn random_search_config(
    cache: &TrainWindowCache,
    freqs: &[f64],
    budget: usize,
    rng: &mut Rng,
) -> SearchOutcome {
    let mut best_cfg = CollapseConfig::random_in_bounds(rng);
    let mut best_obj = cache.objective(freqs, freqs.len(), &best_cfg);
    for _ in 1..budget {
        let cand = CollapseConfig::random_in_bounds(rng);
        let obj = cache.objective(freqs, freqs.len(), &cand);
        if obj > best_obj {
            best_obj = obj;
            best_cfg = cand;
        }
    }
    SearchOutcome {
        best_cfg,
        best_obj,
        evals_used: budget,
        history: Vec::new(),
    }
}

/// Precomputed Guinand-Weil dual resonance field `R(w)` and exact analytical gradient `R'(w)`
/// constructed PURELY from the elementary Sieve of Eratosthenes on `[2, x_max]` (ZERO zeta calls!).
pub struct GuinandWeilField {
    prime_u: Vec<f64>,
    prime_c: Vec<f64>,
    quad_u: Vec<f64>,
    quad_q: Vec<f64>,
}

impl GuinandWeilField {
    pub fn from_sieve(x_max: usize) -> Self {
        let lam = von_mangoldt_table(x_max);
        let u_max = (x_max as f64).ln();
        let hann = |u: f64| -> f64 {
            let s = (PI * u / u_max).sin();
            s * s
        };

        let mut prime_u = Vec::new();
        let mut prime_c = Vec::new();
        for n in 2..=x_max {
            if lam[n] > 0.0 {
                let nf = n as f64;
                let u = nf.ln();
                prime_u.push(u);
                prime_c.push(hann(u) * lam[n] / nf.sqrt());
            }
        }

        // Composite Simpson's rule over u in [0, u_max] (even n_quad = 600)
        let n_quad = 600usize;
        let du = u_max / n_quad as f64;
        let mut quad_u = Vec::with_capacity(n_quad + 1);
        let mut quad_q = Vec::with_capacity(n_quad + 1);
        for j in 0..=n_quad {
            let u = j as f64 * du;
            let sw = if j == 0 || j == n_quad {
                1.0
            } else if j % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let w_simpson = sw * du / 3.0;
            quad_u.push(u);
            quad_q.push(w_simpson * hann(u) * (0.5 * u).exp());
        }

        Self {
            prime_u,
            prime_c,
            quad_u,
            quad_q,
        }
    }

    /// Evaluates `(R(w), dR/dw)` analytically with zero reference to `zeta(s)`.
    pub fn eval_and_grad(&self, w: f64) -> (f64, f64) {
        let mut val = 0.0;
        let mut grad = 0.0;
        for i in 0..self.prime_u.len() {
            let u = self.prime_u[i];
            let c = self.prime_c[i];
            let a = w * u;
            val -= c * a.cos();
            grad += c * u * a.sin();
        }
        for j in 0..self.quad_u.len() {
            let u = self.quad_u[j];
            let q = self.quad_q[j];
            let a = w * u;
            val += q * a.cos();
            grad -= q * u * a.sin();
        }
        (val, grad)
    }
}

/// Blindly discovers `k_modes` collapse frequencies from random initialization on `[w_lo, w_hi]`
/// using ONLY the elementary prime-sieve Guinand-Weil resonance field `R(w)` + optional short-range
/// Dyson spectral repulsion (preventing duplicate peak collapse without pushing modes into empty walls).
pub fn blind_discover_zeros(
    field: &GuinandWeilField,
    init_freqs: &[f64],
    w_lo: f64,
    w_hi: f64,
    repulsion_init: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let k = init_freqs.len();
    let mut w = init_freqs.to_vec();
    let total_steps = 2000usize;
    let phase_a = 1600usize;
    let eps2 = 0.22 * 0.22;
    let inv_two_sig2 = 1.0 / (2.0 * 2.0 * 2.0);
    let mut next_w = vec![0.0; k];

    for step in 0..total_steps {
        let in_explore = step < phase_a;
        let prog = if in_explore {
            step as f64 / phase_a as f64
        } else {
            1.0
        };
        let lr = if in_explore { 0.015 } else { 0.012 };
        let base_noise = if in_explore {
            0.09 * (1.0 - prog) + 0.003 * prog
        } else {
            0.0
        };
        let rep_eta = if in_explore {
            repulsion_init * (1.0 - 0.72 * prog)
        } else {
            0.0
        };

        for i in 0..k {
            let (r_val, g) = field.eval_and_grad(w[i]);
            // Thermal teleportation escape if stuck in a dead non-resonant valley (R < 1.3) during exploration
            // (100% identical in Dyson and No-Repulsion arms!)
            if in_explore && step % 40 == 0 && r_val < 1.3 {
                next_w[i] = rng.uniform(w_lo + 0.5, w_hi - 0.5);
                continue;
            }
            let mut delta = (lr * g).clamp(-0.35, 0.35);
            if rep_eta > 0.0 {
                let mut rep = 0.0;
                for j in 0..k {
                    if i == j {
                        continue;
                    }
                    let d = w[i] - w[j];
                    let d2 = d * d;
                    rep += (d / (d2 + eps2)) * (-d2 * inv_two_sig2).exp();
                }
                delta += (rep_eta * rep).clamp(-0.65, 0.65);
            }
            if in_explore {
                let esc_noise = if r_val < 1.4 {
                    base_noise.max(0.45 * (1.0 - 0.60 * prog))
                } else {
                    base_noise
                };
                delta += esc_noise * rng.normal();
            }
            let mut nw = w[i] + delta;
            if nw < w_lo {
                nw = w_lo + (w_lo - nw).min(1.0);
            } else if nw > w_hi {
                nw = w_hi - (nw - w_hi).min(1.0);
            }
            next_w[i] = nw.clamp(w_lo, w_hi);
        }
        std::mem::swap(&mut w, &mut next_w);
    }

    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    w
}

/// Evaluates how many distinct reference zeros `true_zeros` are captured within `tol`
/// and computes 1-to-1 sorted mean absolute error.
pub fn evaluate_zero_discovery(discovered: &[f64], true_zeros: &[f64], tol: f64) -> (usize, f64) {
    assert_eq!(discovered.len(), true_zeros.len());
    let mut found = 0usize;
    for &tz in true_zeros {
        if discovered.iter().any(|&dw| (dw - tz).abs() <= tol) {
            found += 1;
        }
    }
    let mut ds = discovered.to_vec();
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_err = ds
        .iter()
        .zip(true_zeros.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / true_zeros.len() as f64;
    (found, mean_err)
}

/// Predicts Chebyshev fluctuation `Delta psi(x) = psi(x) - x` from frequencies `freqs`
/// using `-2 * sqrt(x) * sum_k (0.5*cos(w_k ln x) + w_k*sin(w_k ln x)) / (0.25 + w_k^2)`.
pub fn predict_chebyshev_fluctuation(x: f64, freqs: &[f64]) -> f64 {
    let ln_x = x.ln();
    let mut acc = 0.0;
    for &w in freqs {
        let a = w * ln_x;
        acc += (0.5 * a.cos() + w * a.sin()) / (0.25 + w * w);
    }
    -2.0 * x.sqrt() * acc
}

pub fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    cov / (va * vb).sqrt().max(1e-15)
}
