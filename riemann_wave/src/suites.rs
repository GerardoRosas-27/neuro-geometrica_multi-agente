//! Five preregistered experimental suites (R1-R5) evaluating Riemann / Dyson GUE
//! wave frequencies as an inference medium, plus trace exporters for visualization.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::dyson::{
    consolidate_dyson, consolidate_ginibre_2d_rff, ks_distance, poisson_cdf,
    sample_gue_matrix_spacings, small_spacing_fraction, unfold_semicircle_to_quantiles,
    wigner_gue_cdf,
};
use crate::linalg::{rmse, vec_norm, Mat, RlsState, ThinSvd};
use crate::riemann::{
    compute_riemann_zeros, evaluate_prime_wave_discrimination, prime_wave_amplitude,
    riemann_zeros_as_frequencies, riemann_zeros_as_quantiles, unfolded_riemann_spacings,
};
use crate::rng::{halton, inv_normal_cdf, Rng};

const PHI_GOLDEN: f64 = 1.618_033_988_749_895;

/// Evaluates target wave concepts in logarithmic Riemann geometry `u in [0, 2*pi)`.
pub fn eval_concepts_u(u_grid: &[f64], concept_u: &[f64], width_u: f64) -> Vec<f64> {
    let inv_w = 1.0 / width_u;
    u_grid
        .iter()
        .map(|&u| {
            let mut s = 0.0;
            for &uc in concept_u {
                let d = (u - uc) * inv_w;
                s += (-0.5 * d * d).exp();
            }
            s
        })
        .collect()
}

/// Builds wave feature matrix `[1, cos(u * f_k + phi_k), sin(u * f_k + phi_k)]` of shape `N x (2M + 1)`.
pub fn build_wave_basis(u_grid: &[f64], freqs: &[f64], phase_jitter: Option<&[f64]>) -> Mat {
    let n = u_grid.len();
    let m = freqs.len();
    let p = 2 * m + 1;
    let mut phi = Mat::zeros(n, p);
    for (r, &u) in u_grid.iter().enumerate() {
        phi.set(r, 0, 1.0);
        for k in 0..m {
            let jitter = phase_jitter.map_or(0.0, |j| j[k]);
            let angle = u * freqs[k] + jitter;
            phi.set(r, 1 + k, angle.cos());
            phi.set(r, 1 + m + k, angle.sin());
        }
    }
    phi
}

pub fn build_wave_row(u: f64, freqs: &[f64], phase_jitter: Option<&[f64]>) -> Vec<f64> {
    let m = freqs.len();
    let mut row = vec![0.0; 2 * m + 1];
    row[0] = 1.0;
    for k in 0..m {
        let jitter = phase_jitter.map_or(0.0, |j| j[k]);
        let angle = u * freqs[k] + jitter;
        row[1 + k] = angle.cos();
        row[1 + m + k] = angle.sin();
    }
    row
}

pub fn min_frequency_gap(freqs: &[f64]) -> f64 {
    let mut s = freqs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut best = f64::INFINITY;
    for w in s.windows(2) {
        let g = (w[1] - w[0]).abs();
        if g < best {
            best = g;
        }
    }
    best
}

pub fn fourier_uniform_freqs(m: usize) -> Vec<f64> {
    (1..=m).map(|k| k as f64).collect()
}

pub fn golden_jitter_freqs(m: usize) -> Vec<f64> {
    (1..=m)
        .map(|k| {
            let base = 1.0 + ((m - 1) as f64) * ((k as f64 - 0.5) / m as f64);
            let frac = ((k as f64) * PHI_GOLDEN).fract();
            base + 0.15 * (frac - 0.5)
        })
        .collect()
}

pub fn poisson_freqs(m: usize, rng: &mut Rng) -> Vec<f64> {
    let mut f: Vec<f64> = (0..m).map(|_| rng.uniform(1.0, m as f64)).collect();
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    f
}

fn sample_well_separated_concepts(
    n_concepts: usize,
    lo: f64,
    hi: f64,
    min_sep: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_concepts);
    let mut attempts = 0;
    while out.len() < n_concepts && attempts < 20_000 {
        attempts += 1;
        let cand = rng.uniform(lo, hi);
        if out.iter().all(|&c: &f64| (c - cand).abs() >= min_sep) {
            out.push(cand);
        }
    }
    assert_eq!(
        out.len(),
        n_concepts,
        "Failed to sample separated concepts"
    );
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

fn linspace(start: f64, end: f64, n: usize, endpoint: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let denom = if endpoint { (n - 1) as f64 } else { n as f64 };
    (0..n)
        .map(|i| start + (end - start) * (i as f64) / denom)
        .collect()
}

/// Runs all 5 preregistered suites (R1-R5) for a single `(seed, phase)` and writes CSVs into `out_dir`.
pub fn run_all_suites(seed: u64, phase: &str, out_dir: &Path, export_traces: bool) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");

    // Precompute exact first 128 Riemann zeta zeros (deterministic mathematical constants)
    let zeros_128 = compute_riemann_zeros(128);

    // =========================================================================
    // SUITE R1: Spectral Verification & Prime Wave Interference
    // =========================================================================
    let t_r1_start = Instant::now();
    let mut f_r1 = BufWriter::new(File::create(out_dir.join("metrics_r1.csv")).unwrap());
    writeln!(
        f_r1,
        "seed,phase,arm,prime_auc,prime_snr,ks_vs_gue,ks_vs_poisson,p_small_02,mean_spacing,eval_ms"
    )
    .unwrap();

    // Arm 1: Exact Riemann zeros (M = 128)
    let t0 = Instant::now();
    let (auc_r, snr_r) = evaluate_prime_wave_discrimination(&zeros_128, 60);
    let sp_riemann = unfolded_riemann_spacings(&zeros_128);
    let ks_r_gue = ks_distance(&sp_riemann, wigner_gue_cdf);
    let ks_r_poi = ks_distance(&sp_riemann, poisson_cdf);
    let ps_r = small_spacing_fraction(&sp_riemann, 0.2);
    let mean_r = sp_riemann.iter().sum::<f64>() / sp_riemann.len() as f64;
    writeln!(
        f_r1,
        "{},{},riemann_zeros,{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
        seed,
        phase,
        auc_r,
        snr_r,
        ks_r_gue,
        ks_r_poi,
        ps_r,
        mean_r,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // Arm 2: Poisson random frequencies over [gamma_1, gamma_128]
    let t0 = Instant::now();
    let mut rng_r1_poi = Rng::from_seed(seed, 101);
    let mut poi_prime_freqs: Vec<f64> = (0..128)
        .map(|_| rng_r1_poi.uniform(zeros_128[0], zeros_128[127]))
        .collect();
    poi_prime_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (auc_p, snr_p) = evaluate_prime_wave_discrimination(&poi_prime_freqs, 60);
    let mut sp_poi: Vec<f64> = poi_prime_freqs.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_sp_p = sp_poi.iter().sum::<f64>() / sp_poi.len() as f64;
    for s in &mut sp_poi {
        *s /= mean_sp_p;
    }
    let ks_p_gue = ks_distance(&sp_poi, wigner_gue_cdf);
    let ks_p_poi = ks_distance(&sp_poi, poisson_cdf);
    let ps_p = small_spacing_fraction(&sp_poi, 0.2);
    writeln!(
        f_r1,
        "{},{},poisson_prime_ctrl,{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
        seed,
        phase,
        auc_p,
        snr_p,
        ks_p_gue,
        ks_p_poi,
        ps_p,
        1.0,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // Arm 3: Dyson Langevin gas beta = 2 (M = 60, 4000 steps, dt = 0.002)
    let t0 = Instant::now();
    let mut rng_dyson = Rng::from_seed(seed, 102);
    let dyson_res = consolidate_dyson(60, 4000, 0.002, 2.0, export_traces, &mut rng_dyson);
    let ks_d_gue = ks_distance(&dyson_res.equilibrium_spacings, wigner_gue_cdf);
    let ks_d_poi = ks_distance(&dyson_res.equilibrium_spacings, poisson_cdf);
    let ps_d = small_spacing_fraction(&dyson_res.equilibrium_spacings, 0.2);
    let mean_d = dyson_res.equilibrium_spacings.iter().sum::<f64>()
        / dyson_res.equilibrium_spacings.len() as f64;
    writeln!(
        f_r1,
        "{},{},dyson_gue,0.00000000,0.00000000,{:.8},{:.8},{:.8},{:.8},{:.3}",
        seed,
        phase,
        ks_d_gue,
        ks_d_poi,
        ps_d,
        mean_d,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // Arm 4: Random Hermitian GUE matrices (dim=48, 20 matrices)
    let t0 = Instant::now();
    let mut rng_gue = Rng::from_seed(seed, 103);
    let sp_gue = sample_gue_matrix_spacings(48, 20, &mut rng_gue);
    let ks_g_gue = ks_distance(&sp_gue, wigner_gue_cdf);
    let ks_g_poi = ks_distance(&sp_gue, poisson_cdf);
    let ps_g = small_spacing_fraction(&sp_gue, 0.2);
    writeln!(
        f_r1,
        "{},{},gue_matrix,0.00000000,0.00000000,{:.8},{:.8},{:.8},1.00000000,{:.3}",
        seed,
        phase,
        ks_g_gue,
        ks_g_poi,
        ps_g,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // Arm 5: Dyson Langevin gas beta = 1 (GOE intermediate control)
    let t0 = Instant::now();
    let mut rng_goe = Rng::from_seed(seed, 104);
    let goe_res = consolidate_dyson(60, 4000, 0.002, 1.0, false, &mut rng_goe);
    let ks_o_gue = ks_distance(&goe_res.equilibrium_spacings, wigner_gue_cdf);
    let ks_o_poi = ks_distance(&goe_res.equilibrium_spacings, poisson_cdf);
    let ps_o = small_spacing_fraction(&goe_res.equilibrium_spacings, 0.2);
    writeln!(
        f_r1,
        "{},{},goe_beta1,0.00000000,0.00000000,{:.8},{:.8},{:.8},1.00000000,{:.3}",
        seed,
        phase,
        ks_o_gue,
        ks_o_poi,
        ps_o,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();
    f_r1.flush().unwrap();
    let _r1_ms = t_r1_start.elapsed().as_secs_f64() * 1e3;

    // =========================================================================
    // SUITE R2 & R3: Wave Inference Stability, Anti-Aliasing & Irregular Sampling
    // =========================================================================
    let m_modes = 60usize;
    let ridge_lambda = 1e-2;
    let mut rng_r2_data = Rng::from_seed(seed, 201);
    let mut rng_r2_poi = Rng::from_seed(seed, 202);
    let mut rng_r2_jit = Rng::from_seed(seed, 203);
    let mut rng_r3_irreg = Rng::from_seed(seed, 301);

    let concepts_u = sample_well_separated_concepts(5, 0.55, 2.0 * PI - 0.55, 0.50, &mut rng_r2_data);
    let u_train = linspace(0.0, 2.0 * PI, 260, false);
    let y_train_clean = eval_concepts_u(&u_train, &concepts_u, 0.055);
    let y_obs: Vec<f64> = y_train_clean
        .iter()
        .map(|&y| y + 0.04 * rng_r2_data.normal())
        .collect();

    let u_eval = linspace(0.0, 2.0 * PI, 600, false);
    let y_eval_true = eval_concepts_u(&u_eval, &concepts_u, 0.055);

    // Out-of-period ghost window u in [2*pi, 4*pi) aligned with u_eval for exact periodic-echo check
    let u_ghost: Vec<f64> = u_eval.iter().map(|&u| u + 2.0 * PI).collect();

    // Shared phase jitter vectors across all 6 families
    let unit_jitter: Vec<f64> = (0..m_modes).map(|_| rng_r2_jit.normal()).collect();
    let jit_015: Vec<f64> = unit_jitter.iter().map(|&z| 0.015 * z).collect();
    let jit_050: Vec<f64> = unit_jitter.iter().map(|&z| 0.050 * z).collect();

    // Irregular async sampling dataset (N = 180)
    let mut u_irreg: Vec<f64> = (0..180).map(|_| rng_r3_irreg.uniform(0.0, 2.0 * PI)).collect();
    u_irreg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y_irreg_clean = eval_concepts_u(&u_irreg, &concepts_u, 0.055);
    let y_irreg_obs: Vec<f64> = y_irreg_clean
        .iter()
        .map(|&y| y + 0.04 * rng_r3_irreg.normal())
        .collect();

    let families: [(&str, Vec<f64>); 6] = [
        ("dyson_gue", dyson_res.frequencies.clone()),
        ("riemann_zeros", riemann_zeros_as_frequencies(&zeros_128, m_modes)),
        ("poisson", poisson_freqs(m_modes, &mut rng_r2_poi)),
        ("goe_beta1", goe_res.frequencies.clone()),
        ("fourier_uniform", fourier_uniform_freqs(m_modes)),
        ("golden_jitter", golden_jitter_freqs(m_modes)),
    ];

    let mut f_r2 = BufWriter::new(File::create(out_dir.join("metrics_r2.csv")).unwrap());
    writeln!(
        f_r2,
        "seed,phase,family,log10_cond,df_ridge,norm_ols,norm_ridge,rmse_ols_015,rmse_ols_050,rmse_ridge_015,rmse_ridge_050,rmse_ridge_000,min_gap,eval_ms"
    )
    .unwrap();

    let mut f_r3 = BufWriter::new(File::create(out_dir.join("metrics_r3.csv")).unwrap());
    writeln!(
        f_r3,
        "seed,phase,family,in_domain_rmse,ghost_peak,ghost_rms,periodic_echo_corr,joint_score,irreg_log10_cond,irreg_rmse_ols,irreg_rmse_ridge,eval_ms"
    )
    .unwrap();

    let mut trace_y_ols: Vec<(&str, Vec<f64>)> = Vec::new();
    let mut trace_y_ghost: Vec<(&str, Vec<f64>)> = Vec::new();

    for (name, freqs) in &families {
        let t0 = Instant::now();
        let phi_tr = build_wave_basis(&u_train, freqs, None);
        let svd = ThinSvd::compute(&phi_tr);
        let log10_cond = svd.cond().log10();
        let df_ridge = svd.effective_df(ridge_lambda);

        let w_ols = svd.solve_ols(&y_obs);
        let w_ridge = svd.solve_ridge(&y_obs, ridge_lambda);
        let norm_ols = vec_norm(&w_ols);
        let norm_ridge = vec_norm(&w_ridge);

        let phi_ev_000 = build_wave_basis(&u_eval, freqs, None);
        let phi_ev_015 = build_wave_basis(&u_eval, freqs, Some(&jit_015));
        let phi_ev_050 = build_wave_basis(&u_eval, freqs, Some(&jit_050));

        let y_ols_015 = phi_ev_015.mul_vec(&w_ols);
        let y_ols_050 = phi_ev_050.mul_vec(&w_ols);
        let y_ridge_000 = phi_ev_000.mul_vec(&w_ridge);
        let y_ridge_015 = phi_ev_015.mul_vec(&w_ridge);
        let y_ridge_050 = phi_ev_050.mul_vec(&w_ridge);

        let rmse_ols_015 = rmse(&y_ols_015, &y_eval_true);
        let rmse_ols_050 = rmse(&y_ols_050, &y_eval_true);
        let rmse_ridge_000 = rmse(&y_ridge_000, &y_eval_true);
        let rmse_ridge_015 = rmse(&y_ridge_015, &y_eval_true);
        let rmse_ridge_050 = rmse(&y_ridge_050, &y_eval_true);
        let min_gap = min_frequency_gap(freqs);

        writeln!(
            f_r2,
            "{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            name,
            log10_cond,
            df_ridge,
            norm_ols,
            norm_ridge,
            rmse_ols_015,
            rmse_ols_050,
            rmse_ridge_015,
            rmse_ridge_050,
            rmse_ridge_000,
            min_gap,
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();

        // Suite R3a: Out-of-period ghost aliasing
        let t3 = Instant::now();
        let phi_ghost = build_wave_basis(&u_ghost, freqs, Some(&jit_015));
        let y_ghost = phi_ghost.mul_vec(&w_ridge);
        let ghost_peak = y_ghost
            .iter()
            .map(|&v| v.abs())
            .fold(0.0_f64, |a, b| a.max(b));
        let ghost_rms = (y_ghost.iter().map(|&v| v * v).sum::<f64>() / y_ghost.len() as f64).sqrt();
        let periodic_echo_corr = {
            let n_g = y_ghost.len() as f64;
            let mg = y_ghost.iter().sum::<f64>() / n_g;
            let mt = y_eval_true.iter().sum::<f64>() / n_g;
            let mut cov = 0.0;
            let mut vg = 0.0;
            let mut vt = 0.0;
            for i in 0..y_ghost.len() {
                let dg = y_ghost[i] - mg;
                let dt = y_eval_true[i] - mt;
                cov += dg * dt;
                vg += dg * dg;
                vt += dt * dt;
            }
            cov / (vg * vt).sqrt().max(1e-15)
        };
        let joint_score = rmse_ridge_015 + ghost_rms;

        // Suite R3b: Irregular sampling
        let phi_irreg = build_wave_basis(&u_irreg, freqs, None);
        let svd_irreg = ThinSvd::compute(&phi_irreg);
        let irreg_log10_cond = svd_irreg.cond().log10();
        let w_irreg_ols = svd_irreg.solve_ols(&y_irreg_obs);
        let w_irreg_ridge = svd_irreg.solve_ridge(&y_irreg_obs, ridge_lambda);
        let irreg_rmse_ols = rmse(&phi_ev_015.mul_vec(&w_irreg_ols), &y_eval_true);
        let irreg_rmse_ridge = rmse(&phi_ev_015.mul_vec(&w_irreg_ridge), &y_eval_true);

        writeln!(
            f_r3,
            "{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            name,
            rmse_ridge_015,
            ghost_peak,
            ghost_rms,
            periodic_echo_corr,
            joint_score,
            irreg_log10_cond,
            irreg_rmse_ols,
            irreg_rmse_ridge,
            t3.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();

        if export_traces {
            trace_y_ols.push((name, y_ols_015));
            trace_y_ghost.push((name, y_ghost));
        }
    }
    f_r2.flush().unwrap();
    f_r3.flush().unwrap();

    // =========================================================================
    // SUITE R4: Streaming Online Wave Memory & Causal Repulsion Ablation
    // =========================================================================
    let mut f_r4 = BufWriter::new(File::create(out_dir.join("metrics_r4.csv")).unwrap());
    writeln!(
        f_r4,
        "seed,phase,arm,stream_rmse,first_concept_rmse,log10_cond,weight_norm,min_gap,eval_ms"
    )
    .unwrap();

    let mut rng_r4_data = Rng::from_seed(seed, 401);
    let mut rng_r4_jit = Rng::from_seed(seed, 402);
    let stream_concepts = sample_well_separated_concepts(8, 0.35, 2.0 * PI - 0.35, 0.45, &mut rng_r4_data);
    let stream_jit: Vec<f64> = (0..m_modes).map(|_| 0.02 * rng_r4_jit.normal()).collect();

    // Pre-generate exact 8 streaming episodes (identical across all arms)
    let mut episodes: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(8);
    for ep in 0..8 {
        let uc = stream_concepts[ep];
        let mut us = Vec::with_capacity(48);
        let mut ys = Vec::with_capacity(48);
        // 32 local observations around new concept + 16 background observations
        for _ in 0..32 {
            let u = (uc + rng_r4_data.uniform(-0.22, 0.22)).clamp(0.0, 2.0 * PI - 1e-6);
            let y_true = eval_concepts_u(&[u], &stream_concepts[..=ep], 0.055)[0];
            us.push(u);
            ys.push(y_true + 0.03 * rng_r4_data.normal());
        }
        for _ in 0..16 {
            let u = rng_r4_data.uniform(0.0, 2.0 * PI);
            let y_true = eval_concepts_u(&[u], &stream_concepts[..=ep], 0.055)[0];
            us.push(u);
            ys.push(y_true + 0.03 * rng_r4_data.normal());
        }
        episodes.push((us, ys));
    }

    let y_stream_true = eval_concepts_u(&u_eval, &stream_concepts, 0.055);
    let first_c = stream_concepts[0];
    let first_mask: Vec<usize> = u_eval
        .iter()
        .enumerate()
        .filter(|(_, &u)| (u - first_c).abs() <= 0.20)
        .map(|(i, _)| i)
        .collect();

    // Part A: Fixed-basis zero-replay streaming RLS
    let fixed_r4_arms: [(&str, &Vec<f64>); 4] = [
        ("dyson_gue", &families[0].1),
        ("riemann_zeros", &families[1].1),
        ("poisson", &families[2].1),
        ("fourier_uniform", &families[4].1),
    ];

    for (name, freqs) in &fixed_r4_arms {
        let t0 = Instant::now();
        let mut rls = RlsState::new(2 * m_modes + 1, 10.0);
        for (ep_u, ep_y) in &episodes {
            for (&u, &y) in ep_u.iter().zip(ep_y.iter()) {
                let row = build_wave_row(u, freqs, None);
                rls.update(&row, y);
            }
        }
        let phi_ev = build_wave_basis(&u_eval, freqs, Some(&stream_jit));
        let y_pred = phi_ev.mul_vec(&rls.w);
        let s_rmse = rmse(&y_pred, &y_stream_true);
        let f_err = {
            let a: Vec<f64> = first_mask.iter().map(|&i| y_pred[i]).collect();
            let b: Vec<f64> = first_mask.iter().map(|&i| y_stream_true[i]).collect();
            rmse(&a, &b)
        };
        let phi_cond = build_wave_basis(&u_train, freqs, None);
        let lcond = ThinSvd::compute(&phi_cond).cond().log10();
        writeln!(
            f_r4,
            "{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            name,
            s_rmse,
            f_err,
            lcond,
            vec_norm(&rls.w),
            min_frequency_gap(freqs),
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();
    }

    // Part B: Causal ablation of Dyson repulsion during online spectral co-adaptation
    // Both arms start from IDENTICAL disordered frequencies (families[2] = poisson).
    let u_probe = linspace(0.0, 2.0 * PI, 180, false);
    for &(arm_name, eta_rep) in &[("adaptive_dyson", 0.008_f64), ("adaptive_no_repulsion", 0.0_f64)] {
        let t0 = Instant::now();
        let mut freqs = families[2].1.clone();
        let mut rls = RlsState::new(2 * m_modes + 1, 10.0);

        for (ep_u, ep_y) in &episodes {
            // 1. Update weights online with incoming episode
            for (&u, &y) in ep_u.iter().zip(ep_y.iter()) {
                let row = build_wave_row(u, &freqs, None);
                rls.update(&row, y);
            }
            // Capture current consolidated internal wave field before moving frequencies (no external past data!)
            let phi_probe_old = build_wave_basis(&u_probe, &freqs, None);
            let y_probe_mem = phi_probe_old.mul_vec(&rls.w);

            // 2. Co-adapt frequencies via episode MSE gradient + optional circular Dyson repulsion
            let mut grad = vec![0.0; m_modes];
            let eps2 = 0.05 * 0.05;
            let period_scale = 2.0 * PI / (m_modes as f64);
            for _step in 0..25 {
                for g in &mut grad {
                    *g = 0.0;
                }
                let n_ep = ep_u.len() as f64;
                for (&u, &y) in ep_u.iter().zip(ep_y.iter()) {
                    let mut pred = rls.w[0];
                    for k in 0..m_modes {
                        let a = u * freqs[k];
                        pred += rls.w[1 + k] * a.cos() + rls.w[1 + m_modes + k] * a.sin();
                    }
                    let resid = pred - y;
                    let factor = 2.0 * resid * u / n_ep;
                    for k in 0..m_modes {
                        let a = u * freqs[k];
                        grad[k] += factor * (-rls.w[1 + k] * a.sin() + rls.w[1 + m_modes + k] * a.cos());
                    }
                }
                for k in 0..m_modes {
                    let mut df = (-0.015 * grad[k]).clamp(-0.10, 0.10);
                    if eta_rep > 0.0 {
                        // Circular Dyson (CUE) repulsion on [1, M]: exact 1/delta locally, uniform equilibrium density
                        let mut rep = 0.0;
                        let th_k = (freqs[k] - 1.0) * period_scale;
                        for j in 0..m_modes {
                            if j == k {
                                continue;
                            }
                            let dth = th_k - (freqs[j] - 1.0) * period_scale;
                            rep += dth.sin() / (2.0 * (1.0 - dth.cos()) + eps2);
                        }
                        df += (eta_rep * rep).clamp(-0.20, 0.20);
                    }
                    freqs[k] = (freqs[k] + df).clamp(1.0, m_modes as f64);
                }
            }

            // 3. Re-project consolidated internal wave field onto updated basis (self-distillation)
            let phi_probe_new = build_wave_basis(&u_probe, &freqs, None);
            let svd_new = ThinSvd::compute(&phi_probe_new);
            rls.w = svd_new.solve_ridge(&y_probe_mem, 1e-3);
            // Refresh covariance gently to stay responsive to remaining episodes
            for r in 0..rls.p {
                for c in 0..rls.p {
                    rls.cov.set(r, c, if r == c { 5.0 } else { 0.0 });
                }
            }
            for (&u, &y) in ep_u.iter().zip(ep_y.iter()) {
                let row = build_wave_row(u, &freqs, None);
                rls.update(&row, y);
            }
        }

        let phi_ev = build_wave_basis(&u_eval, &freqs, Some(&stream_jit));
        let y_pred = phi_ev.mul_vec(&rls.w);
        let s_rmse = rmse(&y_pred, &y_stream_true);
        let f_err = {
            let a: Vec<f64> = first_mask.iter().map(|&i| y_pred[i]).collect();
            let b: Vec<f64> = first_mask.iter().map(|&i| y_stream_true[i]).collect();
            rmse(&a, &b)
        };
        let phi_cond = build_wave_basis(&u_train, &freqs, None);
        let lcond = ThinSvd::compute(&phi_cond).cond().log10();
        writeln!(
            f_r4,
            "{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            arm_name,
            s_rmse,
            f_err,
            lcond,
            vec_norm(&rls.w),
            min_frequency_gap(&freqs),
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();
    }
    f_r4.flush().unwrap();

    // =========================================================================
    // SUITE R5: Wave-Kernel Inference (RFF / DPP-GUE) on 1D & 2D Nonlinear Dynamics
    // =========================================================================
    let mut f_r5 = BufWriter::new(File::create(out_dir.join("metrics_r5.csv")).unwrap());
    writeln!(
        f_r5,
        "seed,phase,task,arm,kernel_rel_frob,rmse_1step,rmse_rollout8,eval_ms"
    )
    .unwrap();

    // --- R5a + R5b: 1D RBF Kernel & 1D Nonlinear Multi-Scale Dynamics ---
    let m_rff1 = 48usize;
    let ell_1d = 0.45_f64;
    let inv_ell_1d = 1.0 / ell_1d;

    let mut rng_r5_spec = Rng::from_seed(seed, 501);
    let mut rng_r5_dyn = Rng::from_seed(seed, 502);
    let mut rng_r5_test = Rng::from_seed(seed, 503);

    let dyson_48 = consolidate_dyson(m_rff1, 4000, 0.002, 2.0, false, &mut rng_r5_spec);
    let q_dyson_48 = unfold_semicircle_to_quantiles(&dyson_48.raw_eigenvalues);
    let q_riemann_48 = riemann_zeros_as_quantiles(&zeros_128, m_rff1);

    let w_poisson_1d: Vec<f64> = (0..m_rff1)
        .map(|_| inv_normal_cdf(rng_r5_spec.uniform01()) * inv_ell_1d)
        .collect();
    let w_dyson_1d: Vec<f64> = q_dyson_48
        .iter()
        .map(|&u| inv_normal_cdf(u) * inv_ell_1d)
        .collect();
    let w_riemann_1d: Vec<f64> = q_riemann_48
        .iter()
        .map(|&u| inv_normal_cdf(u) * inv_ell_1d)
        .collect();
    let w_qmc_1d: Vec<f64> = (0..m_rff1)
        .map(|k| inv_normal_cdf((k as f64 + 0.5) / m_rff1 as f64) * inv_ell_1d)
        .collect();

    // 1D Kernel grid
    let k_grid = linspace(-2.0, 2.0, 80, true);
    let mut k_exact_1d = vec![0.0; 80 * 80];
    let mut k_exact_norm_sq = 0.0;
    for i in 0..80 {
        for j in 0..80 {
            let d = k_grid[i] - k_grid[j];
            let v = (-0.5 * (d * inv_ell_1d) * (d * inv_ell_1d)).exp();
            k_exact_1d[i * 80 + j] = v;
            k_exact_norm_sq += v * v;
        }
    }
    let k_exact_norm = k_exact_norm_sq.sqrt();

    // 1D Nonlinear dynamics parameters per seed
    let a1 = rng_r5_dyn.uniform(0.22, 0.34);
    let a2 = rng_r5_dyn.uniform(-0.16, 0.16);
    let step_1d = |x: f64| -> f64 { 0.72 * x + a1 * (2.4 * x).sin() + a2 * (3.8 * x).cos() - 0.14 * x * x * x };

    let x_tr_1d: Vec<f64> = (0..120).map(|_| rng_r5_dyn.uniform(-1.6, 1.6)).collect();
    let y_tr_1d: Vec<f64> = x_tr_1d
        .iter()
        .map(|&x| step_1d(x) + 0.015 * rng_r5_dyn.normal())
        .collect();
    let x_te_1d: Vec<f64> = (0..120).map(|_| rng_r5_test.uniform(-1.5, 1.5)).collect();
    let y_te_1d: Vec<f64> = x_te_1d.iter().map(|&x| step_1d(x)).collect();
    let x_init_1d: Vec<f64> = (0..32).map(|_| rng_r5_test.uniform(-1.3, 1.3)).collect();

    let arms_1d: [(&str, &Vec<f64>); 4] = [
        ("poisson_rff", &w_poisson_1d),
        ("dyson_rff", &w_dyson_1d),
        ("riemann_rff", &w_riemann_1d),
        ("qmc_rff", &w_qmc_1d),
    ];

    let build_rff_1d = |xs: &[f64], omegas: &[f64]| -> Mat {
        let n = xs.len();
        let m = omegas.len();
        let scale = 1.0 / (m as f64).sqrt();
        let mut mat = Mat::zeros(n, 2 * m);
        for (r, &x) in xs.iter().enumerate() {
            for k in 0..m {
                let a = omegas[k] * x;
                mat.set(r, k, scale * a.cos());
                mat.set(r, m + k, scale * a.sin());
            }
        }
        mat
    };

    for (name, omegas) in &arms_1d {
        let t0 = Instant::now();
        // Kernel relative Frobenius error
        let phi_k = build_rff_1d(&k_grid, omegas);
        let mut diff_sq = 0.0;
        for i in 0..80 {
            for j in 0..80 {
                let mut dot = 0.0;
                for c in 0..phi_k.n {
                    dot += phi_k.get(i, c) * phi_k.get(j, c);
                }
                let d = dot - k_exact_1d[i * 80 + j];
                diff_sq += d * d;
            }
        }
        let k_rel_frob = diff_sq.sqrt() / k_exact_norm;

        // Train 1D dynamics predictor with Ridge (lambda = 1e-3)
        let phi_tr = build_rff_1d(&x_tr_1d, omegas);
        let svd = ThinSvd::compute(&phi_tr);
        let w_dyn = svd.solve_ridge(&y_tr_1d, 1e-3);

        let pred_1s = build_rff_1d(&x_te_1d, omegas).mul_vec(&w_dyn);
        let rmse_1s = rmse(&pred_1s, &y_te_1d);

        // 8-step autoregressive rollout
        let mut sq_roll = 0.0;
        let mut count_roll = 0usize;
        let scale = 1.0 / (m_rff1 as f64).sqrt();
        for &x0 in &x_init_1d {
            let mut xt = x0;
            let mut xp = x0;
            for _ in 0..8 {
                xt = step_1d(xt);
                let mut next_p = 0.0;
                for k in 0..m_rff1 {
                    let a = omegas[k] * xp;
                    next_p += scale * (w_dyn[k] * a.cos() + w_dyn[m_rff1 + k] * a.sin());
                }
                xp = next_p.clamp(-10.0, 10.0);
                let d = xp - xt;
                sq_roll += d * d;
                count_roll += 1;
            }
        }
        let rmse_r8 = (sq_roll / count_roll as f64).sqrt();

        writeln!(
            f_r5,
            "{},{},r5_1d,{},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            name,
            k_rel_frob,
            rmse_1s,
            rmse_r8,
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();
    }

    // --- R5c: 2D Isotropic Kernel & 2D Nonlinear Vortex-Duffing Dynamics ---
    let m_rff2 = 64usize;
    let ell_2d = 0.85_f64;
    let inv_ell_2d = 1.0 / ell_2d;

    let mut rng_r5_2d_spec = Rng::from_seed(seed, 511);
    let mut rng_r5_2d_dyn = Rng::from_seed(seed, 512);
    let mut rng_r5_2d_test = Rng::from_seed(seed, 513);

    let w_poi_2d: Vec<(f64, f64)> = (0..m_rff2)
        .map(|_| {
            (
                rng_r5_2d_spec.normal() * inv_ell_2d,
                rng_r5_2d_spec.normal() * inv_ell_2d,
            )
        })
        .collect();
    let w_gin_2d = consolidate_ginibre_2d_rff(m_rff2, inv_ell_2d, 1500, 0.004, &mut rng_r5_2d_spec);
    let w_qmc_2d: Vec<(f64, f64)> = (1..=m_rff2)
        .map(|k| {
            let u1 = halton(k, 2).clamp(1e-6, 1.0 - 1e-6);
            let u2 = halton(k, 3);
            let r = inv_ell_2d * (-2.0 * (1.0 - u1).ln()).sqrt();
            let th = 2.0 * PI * u2;
            (r * th.cos(), r * th.sin())
        })
        .collect();

    // 2D Kernel test grid (9x9 = 81 points in [-1.4, 1.4]^2)
    let g1 = linspace(-1.4, 1.4, 9, true);
    let mut grid_2d = Vec::with_capacity(81);
    for &x in &g1 {
        for &y in &g1 {
            grid_2d.push((x, y));
        }
    }
    let mut k_exact_2d = vec![0.0; 81 * 81];
    let mut k2_norm_sq = 0.0;
    for i in 0..81 {
        for j in 0..81 {
            let dx = grid_2d[i].0 - grid_2d[j].0;
            let dy = grid_2d[i].1 - grid_2d[j].1;
            let v = (-0.5 * (dx * dx + dy * dy) * inv_ell_2d * inv_ell_2d).exp();
            k_exact_2d[i * 81 + j] = v;
            k2_norm_sq += v * v;
        }
    }
    let k2_norm = k2_norm_sq.sqrt();

    // 2D Nonlinear Vortex-Duffing dynamics per seed
    let theta0 = rng_r5_2d_dyn.uniform(0.35, 0.65);
    let c1 = rng_r5_2d_dyn.uniform(0.14, 0.24);
    let c2 = rng_r5_2d_dyn.uniform(-0.24, -0.14);
    let step_2d = |p: (f64, f64)| -> (f64, f64) {
        let (x, y) = p;
        let r2 = x * x + y * y;
        let th = theta0 + 0.35 * r2;
        let ux = 0.84 * x + c1 * (2.0 * y).sin() - 0.08 * x * x * x;
        let uy = 0.84 * y + c2 * (2.0 * x).sin() - 0.08 * y * y * y;
        (th.cos() * ux - th.sin() * uy, th.sin() * ux + th.cos() * uy)
    };

    let z_tr_2d: Vec<(f64, f64)> = (0..220)
        .map(|_| (rng_r5_2d_dyn.uniform(-1.4, 1.4), rng_r5_2d_dyn.uniform(-1.4, 1.4)))
        .collect();
    let mut yx_tr_2d = Vec::with_capacity(220);
    let mut yy_tr_2d = Vec::with_capacity(220);
    for &p in &z_tr_2d {
        let (nx, ny) = step_2d(p);
        yx_tr_2d.push(nx + 0.015 * rng_r5_2d_dyn.normal());
        yy_tr_2d.push(ny + 0.015 * rng_r5_2d_dyn.normal());
    }
    let z_te_2d: Vec<(f64, f64)> = (0..160)
        .map(|_| (rng_r5_2d_test.uniform(-1.3, 1.3), rng_r5_2d_test.uniform(-1.3, 1.3)))
        .collect();
    let y_te_2d: Vec<(f64, f64)> = z_te_2d.iter().map(|&p| step_2d(p)).collect();
    let z_init_2d: Vec<(f64, f64)> = (0..24)
        .map(|_| (rng_r5_2d_test.uniform(-1.1, 1.1), rng_r5_2d_test.uniform(-1.1, 1.1)))
        .collect();

    let build_rff_2d = |pts: &[(f64, f64)], omegas: &[(f64, f64)]| -> Mat {
        let n = pts.len();
        let m = omegas.len();
        let scale = 1.0 / (m as f64).sqrt();
        let mut mat = Mat::zeros(n, 2 * m);
        for (r, &(x, y)) in pts.iter().enumerate() {
            for k in 0..m {
                let a = omegas[k].0 * x + omegas[k].1 * y;
                mat.set(r, k, scale * a.cos());
                mat.set(r, m + k, scale * a.sin());
            }
        }
        mat
    };

    let arms_2d: [(&str, &Vec<(f64, f64)>); 3] = [
        ("poisson_2d_rff", &w_poi_2d),
        ("ginibre_2d_rff", &w_gin_2d),
        ("qmc_2d_rff", &w_qmc_2d),
    ];

    for (name, omegas) in &arms_2d {
        let t0 = Instant::now();
        let phi_k = build_rff_2d(&grid_2d, omegas);
        let mut diff_sq = 0.0;
        for i in 0..81 {
            for j in 0..81 {
                let mut dot = 0.0;
                for c in 0..phi_k.n {
                    dot += phi_k.get(i, c) * phi_k.get(j, c);
                }
                let d = dot - k_exact_2d[i * 81 + j];
                diff_sq += d * d;
            }
        }
        let k_rel = diff_sq.sqrt() / k2_norm;

        let phi_tr = build_rff_2d(&z_tr_2d, omegas);
        let svd = ThinSvd::compute(&phi_tr);
        let wx = svd.solve_ridge(&yx_tr_2d, 1e-3);
        let wy = svd.solve_ridge(&yy_tr_2d, 1e-3);

        let phi_te = build_rff_2d(&z_te_2d, omegas);
        let px = phi_te.mul_vec(&wx);
        let py = phi_te.mul_vec(&wy);
        let mut sq_1s = 0.0;
        for i in 0..z_te_2d.len() {
            let dx = px[i] - y_te_2d[i].0;
            let dy = py[i] - y_te_2d[i].1;
            sq_1s += 0.5 * (dx * dx + dy * dy);
        }
        let rmse_1s = (sq_1s / z_te_2d.len() as f64).sqrt();

        // 8-step 2D rollout
        let mut sq_roll = 0.0;
        let mut cnt = 0usize;
        let scale = 1.0 / (m_rff2 as f64).sqrt();
        for &z0 in &z_init_2d {
            let mut zt = z0;
            let mut zp = z0;
            for _ in 0..8 {
                zt = step_2d(zt);
                let mut nx = 0.0;
                let mut ny = 0.0;
                for k in 0..m_rff2 {
                    let a = omegas[k].0 * zp.0 + omegas[k].1 * zp.1;
                    let ca = scale * a.cos();
                    let sa = scale * a.sin();
                    nx += wx[k] * ca + wx[m_rff2 + k] * sa;
                    ny += wy[k] * ca + wy[m_rff2 + k] * sa;
                }
                zp = (nx.clamp(-10.0, 10.0), ny.clamp(-10.0, 10.0));
                let dx = zp.0 - zt.0;
                let dy = zp.1 - zt.1;
                sq_roll += 0.5 * (dx * dx + dy * dy);
                cnt += 1;
            }
        }
        let rmse_r8 = (sq_roll / cnt as f64).sqrt();

        writeln!(
            f_r5,
            "{},{},r5_2d,{},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            name,
            k_rel,
            rmse_1s,
            rmse_r8,
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();
    }
    f_r5.flush().unwrap();

    // =========================================================================
    // OPTIONAL VISUALIZATION TRACES (single representative seed)
    // =========================================================================
    if export_traces {
        // 1. Prime interference wave trace
        let mut fp = BufWriter::new(File::create(out_dir.join("trace_primes.csv")).unwrap());
        writeln!(fp, "x,riemann_wave,poisson_wave").unwrap();
        for x in linspace(1.5, 35.0, 1400, true) {
            writeln!(
                fp,
                "{:.6},{:.8},{:.8}",
                x,
                prime_wave_amplitude(&zeros_128, x),
                prime_wave_amplitude(&poi_prime_freqs, x)
            )
            .unwrap();
        }
        fp.flush().unwrap();

        // 2. Spacing distributions trace
        let mut fs = BufWriter::new(File::create(out_dir.join("trace_spacings.csv")).unwrap());
        writeln!(fs, "source,spacing").unwrap();
        for &s in &sp_riemann {
            writeln!(fs, "riemann,{:.8}", s).unwrap();
        }
        for &s in &dyson_res.equilibrium_spacings {
            writeln!(fs, "dyson_gue,{:.8}", s).unwrap();
        }
        for &s in &sp_gue {
            writeln!(fs, "gue_matrix,{:.8}", s).unwrap();
        }
        for &s in &sp_poi {
            writeln!(fs, "poisson,{:.8}", s).unwrap();
        }
        fs.flush().unwrap();

        // 3. Wave inference OLS & Ghost aliasing traces
        let mut fw = BufWriter::new(File::create(out_dir.join("trace_wave_inference.csv")).unwrap());
        writeln!(
            fw,
            "x,u,y_true,dyson_ols,riemann_ols,poisson_ols,fourier_ols"
        )
        .unwrap();
        let x_min = 1.5_f64;
        let x_max = 35.0_f64;
        for i in 0..u_eval.len() {
            let u = u_eval[i];
            let x = x_min * (x_max / x_min).powf(u / (2.0 * PI));
            writeln!(
                fw,
                "{:.6},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8}",
                x,
                u,
                y_eval_true[i],
                trace_y_ols[0].1[i],
                trace_y_ols[1].1[i],
                trace_y_ols[2].1[i],
                trace_y_ols[4].1[i],
            )
            .unwrap();
        }
        fw.flush().unwrap();

        let mut fg = BufWriter::new(File::create(out_dir.join("trace_ghost_aliasing.csv")).unwrap());
        writeln!(
            fg,
            "u,dyson_ghost,riemann_ghost,poisson_ghost,fourier_ghost,golden_ghost"
        )
        .unwrap();
        for i in 0..u_ghost.len() {
            writeln!(
                fg,
                "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8}",
                u_ghost[i],
                trace_y_ghost[0].1[i],
                trace_y_ghost[1].1[i],
                trace_y_ghost[2].1[i],
                trace_y_ghost[4].1[i],
                trace_y_ghost[5].1[i],
            )
            .unwrap();
        }
        fg.flush().unwrap();

        // 4. Dyson trajectory trace
        let mut ft = BufWriter::new(File::create(out_dir.join("trace_dyson_trajectories.csv")).unwrap());
        write!(ft, "step").unwrap();
        for k in 0..m_modes {
            write!(ft, ",w{}", k).unwrap();
        }
        writeln!(ft).unwrap();
        for (idx, snap) in dyson_res.history.iter().enumerate() {
            write!(ft, "{}", idx * 40).unwrap();
            for &v in snap {
                write!(ft, ",{:.6}", v).unwrap();
            }
            writeln!(ft).unwrap();
        }
        ft.flush().unwrap();
    }
}
