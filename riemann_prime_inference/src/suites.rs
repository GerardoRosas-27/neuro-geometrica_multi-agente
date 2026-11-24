//! Implementation of the 4 preregistered experimental suites (`I1`–`I4`)
//! and optional visual trace export for `exp/riemann-prime-inference`.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::inference::{
    all_distinct_pairs, eval_causal_outcome, min_prime_product_log_gap, pair_partial_f1,
    sample_collapse_profile, trace_origin, CollapseArm, SmoothPriorField, WaveField, N_MODES,
};
use crate::riemann::compute_riemann_zeros;
use crate::rng::Rng;

const K_DEFAULT: usize = 12;
const SIGMA_LINEAR_STD: f64 = 0.05;
const PRIOR_SMOOTH_TAU: f64 = 0.03;

fn build_all_fields(k_vocab: usize, zeros: &[f64], seed: u64) -> [WaveField; 4] {
    let mut rng_r = Rng::from_seed(seed, 101);
    let mut rng_rand = Rng::from_seed(seed, 102);
    let mut rng_dyson = Rng::from_seed(seed, 103);
    let mut rng_four = Rng::from_seed(seed, 104);

    [
        WaveField::build(CollapseArm::RiemannPrime, k_vocab, zeros, &mut rng_r),
        WaveField::build(CollapseArm::Random, k_vocab, zeros, &mut rng_rand),
        WaveField::build(CollapseArm::DysonGue, k_vocab, zeros, &mut rng_dyson),
        WaveField::build(CollapseArm::FourierLinear, k_vocab, zeros, &mut rng_four),
    ]
}

/// Runs all 4 suites (`I1`–`I4`) for a single `(seed, phase)` and writes CSV outputs to `out_dir`.
pub fn run_seed(
    seed: u64,
    phase: &str,
    out_dir: &Path,
    export_traces: bool,
) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;

    let zeros = compute_riemann_zeros(N_MODES);
    let fields = build_all_fields(K_DEFAULT, &zeros, seed);

    run_suite_i1(seed, phase, &fields, out_dir)?;
    run_suite_i2(seed, phase, &fields, out_dir)?;
    run_suite_i3(seed, phase, &zeros, out_dir)?;
    run_suite_i4(seed, phase, &fields, &zeros, out_dir)?;

    if export_traces {
        export_visual_traces(seed, &fields, &zeros, out_dir)?;
    }

    Ok(())
}

/// Suite I1: Retrodictive Origin Tracing (`N -> {c_a, c_b}`) across sigma in {0.0, 0.002, 0.006}.
fn run_suite_i1(
    seed: u64,
    phase: &str,
    fields: &[WaveField; 4],
    out_dir: &Path,
) -> std::io::Result<()> {
    let pairs = all_distinct_pairs(K_DEFAULT);
    let sigmas = [0.0_f64, 0.002, 0.006];

    let mut w = BufWriter::new(File::create(out_dir.join("metrics_i1.csv"))?);
    writeln!(
        w,
        "seed,phase,arm,sigma,n_pairs,origin_exact_acc,origin_partial_f1,peak_margin_snr"
    )?;

    for (s_idx, &sigma) in sigmas.iter().enumerate() {
        // Shared standardized Gaussian noise draws across all 4 arms for paired comparison.
        let mut noise_rng = Rng::from_seed(seed, 1000 + s_idx as u64);
        let z_draws: Vec<f64> = (0..pairs.len()).map(|_| noise_rng.normal()).collect();

        for (arm_idx, field) in fields.iter().enumerate() {
            let mut tie_rng = Rng::from_seed(seed, 1100 + (s_idx as u64) * 16 + arm_idx as u64);
            let sigma_eff = if sigma == 0.0 {
                0.0
            } else if field.arm.is_linear() {
                SIGMA_LINEAR_STD
            } else {
                sigma
            };

            let mut exact_hits = 0usize;
            let mut f1_sum = 0.0_f64;
            let mut snr_sum = 0.0_f64;

            for (j, &(a, b)) in pairs.iter().enumerate() {
                let u_obs = field.compose_pair(a, b) + sigma_eff * z_draws[j];
                let res = trace_origin(field, u_obs, Some((a, b)), None, &mut tie_rng);
                if res.pred_a == a && res.pred_b == b {
                    exact_hits += 1;
                }
                f1_sum += pair_partial_f1(res.pred_a, res.pred_b, a, b);
                snr_sum += res.peak_margin_snr;
            }

            let n = pairs.len() as f64;
            writeln!(
                w,
                "0x{:X},{},{},{:.4},{},{:.6},{:.6},{:.6}",
                seed,
                phase,
                field.arm.as_str(),
                sigma,
                pairs.len(),
                (exact_hits as f64) / n,
                f1_sum / n,
                snr_sum / n
            )?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Suite I2: "On the Other Side" — Downstream nonlinear prediction + counterfactual `do(c_a -> c_new)`.
fn run_suite_i2(
    seed: u64,
    phase: &str,
    fields: &[WaveField; 4],
    out_dir: &Path,
) -> std::io::Result<()> {
    let pairs = all_distinct_pairs(K_DEFAULT);
    let sigma = 0.002_f64;

    let mut shared_rng = Rng::from_seed(seed, 2000);
    let z_draws: Vec<f64> = (0..pairs.len()).map(|_| shared_rng.normal()).collect();
    // Choose counterfactual cause c_new not in {a, b}, shared identically across all 4 arms.
    let c_news: Vec<usize> = pairs
        .iter()
        .map(|&(a, b)| {
            let mut cand = shared_rng.usize_range(K_DEFAULT - 2);
            if cand >= a {
                cand += 1;
            }
            if cand >= b {
                cand += 1;
            }
            cand
        })
        .collect();

    let mut w = BufWriter::new(File::create(out_dir.join("metrics_i2.csv"))?);
    writeln!(
        w,
        "seed,phase,arm,n_pairs,origin_exact_acc,pred_rmse,class_acc,cf_exact_acc,cf_rmse"
    )?;

    for (arm_idx, field) in fields.iter().enumerate() {
        let mut tie_rng = Rng::from_seed(seed, 2100 + arm_idx as u64);
        let sigma_eff = if field.arm.is_linear() {
            SIGMA_LINEAR_STD
        } else {
            sigma
        };

        let mut origin_hits = 0usize;
        let mut pred_sq_err = 0.0_f64;
        let mut class_hits = 0usize;
        let mut cf_hits = 0usize;
        let mut cf_sq_err = 0.0_f64;

        for (j, &(a, b)) in pairs.iter().enumerate() {
            let u_obs = field.compose_pair(a, b) + sigma_eff * z_draws[j];
            let res = trace_origin(field, u_obs, Some((a, b)), None, &mut tie_rng);

            if res.pred_a == a && res.pred_b == b {
                origin_hits += 1;
            }

            // Forward prediction "on the other side" from traced origin.
            let (ty1, ty2, tcls) = eval_causal_outcome(a, b);
            let (py1, py2, pcls) = eval_causal_outcome(res.pred_a, res.pred_b);
            pred_sq_err += (py1 - ty1).powi(2) + (py2 - ty2).powi(2);
            if pcls == tcls {
                class_hits += 1;
            }

            // Counterfactual intervention do(c_a -> c_new) via spectral phase shift & re-collapse:
            // u_cf = u_obs - coord[pred_a] + coord[c_new].
            let c_new = c_news[j];
            let (cf_ta, cf_tb) = if c_new < b { (c_new, b) } else { (b, c_new) };
            let u_cf = u_obs - field.coords[res.pred_a] + field.coords[c_new];
            let cf_res = trace_origin(field, u_cf, Some((cf_ta, cf_tb)), None, &mut tie_rng);

            if cf_res.pred_a == cf_ta && cf_res.pred_b == cf_tb {
                cf_hits += 1;
            }
            let (cty1, cty2, _) = eval_causal_outcome(cf_ta, cf_tb);
            let (cpy1, cpy2, _) = eval_causal_outcome(cf_res.pred_a, cf_res.pred_b);
            cf_sq_err += (cpy1 - cty1).powi(2) + (cpy2 - cty2).powi(2);
        }

        let n = pairs.len() as f64;
        let pred_rmse = (pred_sq_err / (2.0 * n)).sqrt();
        let cf_rmse = (cf_sq_err / (2.0 * n)).sqrt();

        writeln!(
            w,
            "0x{:X},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            phase,
            field.arm.as_str(),
            pairs.len(),
            (origin_hits as f64) / n,
            pred_rmse,
            (class_hits as f64) / n,
            (cf_hits as f64) / n,
            cf_rmse
        )?;
    }

    w.flush()?;
    Ok(())
}

/// Suite I3: Bayesian Transactional Disambiguation (Past Prior x Future Evidence)
/// under heavy observation noise `sigma = 0.015` (200 events/seed).
fn run_suite_i3(
    seed: u64,
    phase: &str,
    zeros: &[f64],
    out_dir: &Path,
) -> std::io::Result<()> {
    let n_events = 200usize;
    let sigma = 0.015_f64;
    let sigma_w = sigma * 0.5;

    let mut data_rng = Rng::from_seed(seed, 3000);
    // Random permutation of the 12 causes to assign Zipf prior ranks per seed.
    let mut perm: Vec<usize> = (0..K_DEFAULT).collect();
    for i in (1..K_DEFAULT).rev() {
        let j = data_rng.usize_range(i + 1);
        perm.swap(i, j);
    }

    let mut probs = vec![0.0_f64; K_DEFAULT];
    let mut z_sum = 0.0_f64;
    for (rank, &concept) in perm.iter().enumerate() {
        let w = ((rank + 1) as f64).powf(-1.4);
        probs[concept] = w;
        z_sum += w;
    }
    for p in &mut probs {
        *p /= z_sum;
    }

    // Build discrete pair distribution P(a, b) proportional to P(c_a) * P(c_b) for a < b.
    let pairs = all_distinct_pairs(K_DEFAULT);
    let mut pair_cdf = Vec::with_capacity(pairs.len());
    let mut acc = 0.0_f64;
    let mut map_pair = pairs[0];
    let mut map_prob = -1.0_f64;
    for &(a, b) in &pairs {
        let pp = probs[a] * probs[b];
        if pp > map_prob {
            map_prob = pp;
            map_pair = (a, b);
        }
        acc += pp;
        pair_cdf.push(acc);
    }
    for v in &mut pair_cdf {
        *v /= acc;
    }

    // Sample 200 events (true_pair, noise) shared across all 4 arms.
    let mut events: Vec<((usize, usize), f64)> = Vec::with_capacity(n_events);
    for _ in 0..n_events {
        let u = data_rng.uniform01();
        let idx = pair_cdf
            .iter()
            .position(|&c| u <= c)
            .unwrap_or(pairs.len() - 1);
        let eta = sigma * data_rng.normal();
        events.push((pairs[idx], eta));
    }

    // Analytical noise-matched Gaussian-convolved wave fields (sigma_w = sigma / 2 = 0.0075)
    // applied identically to all 3 wave arms in Suite I3.
    let mut rng_rb = Rng::from_seed(seed, 3051);
    let mut rng_randb = Rng::from_seed(seed, 3052);
    let riemann_field =
        WaveField::build_damped(CollapseArm::RiemannPrime, K_DEFAULT, zeros, sigma_w, &mut rng_rb);
    let random_field =
        WaveField::build_damped(CollapseArm::Random, K_DEFAULT, zeros, sigma_w, &mut rng_randb);
    let smooth_prior = SmoothPriorField::new(&riemann_field.coords, &probs, PRIOR_SMOOTH_TAU);

    let mut w = BufWriter::new(File::create(out_dir.join("metrics_i3.csv"))?);
    writeln!(
        w,
        "seed,phase,arm,sigma,n_events,origin_exact_acc,origin_partial_f1"
    )?;

    let arms: [(&str, Option<(&WaveField, Option<&SmoothPriorField>)>); 4] = [
        (
            "riemann_bayes_collapse",
            Some((&riemann_field, Some(&smooth_prior))),
        ),
        ("riemann_uniform_collapse", Some((&riemann_field, None))),
        (
            "random_bayes_collapse",
            Some((&random_field, Some(&smooth_prior))),
        ),
        ("prior_only_ctrl", None),
    ];

    for (arm_idx, &(arm_name, cfg)) in arms.iter().enumerate() {
        let mut tie_rng = Rng::from_seed(seed, 3100 + arm_idx as u64);
        let mut exact_hits = 0usize;
        let mut f1_sum = 0.0_f64;

        for &((a, b), eta) in &events {
            let (pa, pb) = match cfg {
                Some((field, prior_opt)) => {
                    let u_obs = field.compose_pair(a, b) + eta;
                    let res = trace_origin(field, u_obs, Some((a, b)), prior_opt, &mut tie_rng);
                    (res.pred_a, res.pred_b)
                }
                None => map_pair,
            };

            if pa == a && pb == b {
                exact_hits += 1;
            }
            f1_sum += pair_partial_f1(pa, pb, a, b);
        }

        let n = n_events as f64;
        writeln!(
            w,
            "0x{:X},{},{},{:.4},{},{:.6},{:.6}",
            seed,
            phase,
            arm_name,
            sigma,
            n_events,
            (exact_hits as f64) / n,
            f1_sum / n
        )?;
    }

    w.flush()?;
    Ok(())
}

/// Suite I4: Compositional Analogy (`A:B::C:?`), Noise Limit Curve, and 1D Capacity Limit Curve.
fn run_suite_i4(
    seed: u64,
    phase: &str,
    fields: &[WaveField; 4],
    zeros: &[f64],
    out_dir: &Path,
) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(out_dir.join("metrics_i4.csv"))?);
    writeln!(
        w,
        "seed,phase,subtask,arm,param_name,param_value,n_events,exact_acc,aux_metric"
    )?;

    // --- I4a: Analogy A:B::C:? (80 distinct 4-tuples (a, b, x, y) per seed) ---
    let n_analogies = 80usize;
    let sigma_analogy = 0.002_f64;
    let mut an_rng = Rng::from_seed(seed, 4000);
    let mut tuples: Vec<(usize, usize, usize, usize, f64, f64, f64)> =
        Vec::with_capacity(n_analogies);
    for _ in 0..n_analogies {
        let mut pool: Vec<usize> = (0..K_DEFAULT).collect();
        for i in 0..4 {
            let j = i + an_rng.usize_range(K_DEFAULT - i);
            pool.swap(i, j);
        }
        let (a, b, x, y) = (pool[0], pool[1], pool[2], pool[3]);
        tuples.push((a, b, x, y, an_rng.normal(), an_rng.normal(), an_rng.normal()));
    }

    for (arm_idx, field) in fields.iter().enumerate() {
        let mut tie_rng = Rng::from_seed(seed, 4100 + arm_idx as u64);
        let sigma_eff = if field.arm.is_linear() {
            SIGMA_LINEAR_STD
        } else {
            sigma_analogy
        };

        let mut hits = 0usize;
        let mut f1_sum = 0.0_f64;
        for &(a, b, x, y, za, zb, zc) in &tuples {
            let u_a = field.compose_pair(a, x) + sigma_eff * za;
            let u_b = field.compose_pair(b, x) + sigma_eff * zb;
            let u_c = field.compose_pair(a, y) + sigma_eff * zc;
            // Spectral analogy shift: u_D = u_C + (u_B - u_A).
            let u_d = u_c + (u_b - u_a);
            let (tb, ty) = if b < y { (b, y) } else { (y, b) };
            let res = trace_origin(field, u_d, Some((tb, ty)), None, &mut tie_rng);
            if res.pred_a == tb && res.pred_b == ty {
                hits += 1;
            }
            f1_sum += pair_partial_f1(res.pred_a, res.pred_b, tb, ty);
        }

        let n = n_analogies as f64;
        writeln!(
            w,
            "0x{:X},{},analogy,{},sigma,{:.4},{},{:.6},{:.6}",
            seed,
            phase,
            field.arm.as_str(),
            sigma_analogy,
            n_analogies,
            (hits as f64) / n,
            f1_sum / n
        )?;
    }

    // --- I4b: Honest physical noise curve on riemann_prime_collapse (K = 12) ---
    let noise_levels = [0.0_f64, 0.001, 0.003, 0.008, 0.015, 0.030];
    let pairs_12 = all_distinct_pairs(K_DEFAULT);
    let riemann_12 = &fields[0];

    for (n_idx, &sig) in noise_levels.iter().enumerate() {
        let mut nrng = Rng::from_seed(seed, 4200 + n_idx as u64);
        let mut tie_rng = Rng::from_seed(seed, 4300 + n_idx as u64);
        let mut hits = 0usize;
        let mut snr_sum = 0.0_f64;

        for &(a, b) in &pairs_12 {
            let u_obs = riemann_12.compose_pair(a, b) + sig * nrng.normal();
            let res = trace_origin(riemann_12, u_obs, Some((a, b)), None, &mut tie_rng);
            if res.pred_a == a && res.pred_b == b {
                hits += 1;
            }
            snr_sum += res.peak_margin_snr;
        }
        let n = pairs_12.len() as f64;
        writeln!(
            w,
            "0x{:X},{},noise_sweep,riemann_prime_collapse,sigma,{:.4},{},{:.6},{:.6}",
            seed,
            phase,
            sig,
            pairs_12.len(),
            (hits as f64) / n,
            snr_sum / n
        )?;
    }

    // --- I4c: Honest 1D capacity sweep K in {8, 12, 18} at sigma = 0.003, M = 128 ---
    let ks = [8usize, 12, 18];
    let sigma_cap = 0.003_f64;
    for (k_idx, &k_vocab) in ks.iter().enumerate() {
        let mut brng = Rng::from_seed(seed, 4400 + k_idx as u64);
        let field_k = WaveField::build(CollapseArm::RiemannPrime, k_vocab, zeros, &mut brng);
        let pairs_k = all_distinct_pairs(k_vocab);
        let min_gap = min_prime_product_log_gap(k_vocab);

        let mut nrng = Rng::from_seed(seed, 4500 + k_idx as u64);
        let mut tie_rng = Rng::from_seed(seed, 4600 + k_idx as u64);
        let mut hits = 0usize;

        for &(a, b) in &pairs_k {
            let u_obs = field_k.compose_pair(a, b) + sigma_cap * nrng.normal();
            let res = trace_origin(&field_k, u_obs, Some((a, b)), None, &mut tie_rng);
            if res.pred_a == a && res.pred_b == b {
                hits += 1;
            }
        }
        let n = pairs_k.len() as f64;
        writeln!(
            w,
            "0x{:X},{},capacity_sweep,riemann_prime_collapse,k_vocab,{},{},{:.6},{:.6}",
            seed,
            phase,
            k_vocab,
            pairs_k.len(),
            (hits as f64) / n,
            min_gap
        )?;
    }

    w.flush()?;
    Ok(())
}

/// Exports representative continuous wave-collapse profiles for visualization (Figures 1 and 3).
fn export_visual_traces(
    seed: u64,
    fields: &[WaveField; 4],
    zeros: &[f64],
    out_dir: &Path,
) -> std::io::Result<()> {
    // 1. Retrodictive collapse profiles for N = 7*11 = 77 (a=3, b=4) and N = 17*19 = 323 (a=6, b=7).
    let mut w = BufWriter::new(File::create(out_dir.join("trace_retro_example.csv"))?);
    writeln!(w, "case_id,arm,u_coord,u_true1,u_true2,w_past,w_future,collapse_score")?;

    let cases = [("N77_7x11", 3usize, 4usize), ("N323_17x19", 6usize, 7usize)];
    let mut nrng = Rng::from_seed(seed, 9001);
    for &(case_id, a, b) in &cases {
        let z = nrng.normal();
        for field in fields {
            let sig = if field.arm.is_linear() {
                SIGMA_LINEAR_STD
            } else {
                0.002
            };
            let u_obs = field.compose_pair(a, b) + sig * z;
            let prof = sample_collapse_profile(field, u_obs, 360);
            for (u, wp, wf, sc) in prof {
                writeln!(
                    w,
                    "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                    case_id,
                    field.arm.as_str(),
                    u,
                    field.coords[a],
                    field.coords[b],
                    wp,
                    wf,
                    sc
                )?;
            }
        }
    }
    w.flush()?;

    // 2. Bayesian disambiguation profile: near-degenerate 17x19 = 323 vs 11x29 = 319
    //    (Delta u = 0.0125) when noise pushes u_obs = ln(323) - 0.008 closer to 319,
    //    while the Past prior favors (17, 19) over (11, 29).
    let mut rng_tr = Rng::from_seed(seed, 9002);
    let riemann = WaveField::build_damped(
        CollapseArm::RiemannPrime,
        K_DEFAULT,
        zeros,
        0.0075,
        &mut rng_tr,
    );
    let mut demo_probs = vec![0.04_f64; K_DEFAULT];
    demo_probs[6] = 0.30; // p=17 high past prior
    demo_probs[7] = 0.26; // p=19 high past prior
    demo_probs[4] = 0.03; // p=11 low past prior
    demo_probs[9] = 0.03; // p=29 low past prior
    let ssum: f64 = demo_probs.iter().sum();
    for p in &mut demo_probs {
        *p /= ssum;
    }
    let prior = SmoothPriorField::new(&riemann.coords, &demo_probs, PRIOR_SMOOTH_TAU);

    let u_obs_ambig = riemann.compose_pair(6, 7) - 0.008;
    let lo = riemann.cavity_min.max(u_obs_ambig - riemann.cavity_max);
    let hi = (u_obs_ambig * 0.5).min(riemann.cavity_max);

    let mut wb = BufWriter::new(File::create(out_dir.join("trace_bayesian_example.csv"))?);
    writeln!(wb, "u_coord,uniform_score,prior_weight,bayes_score")?;
    let n_pts = 400usize;
    let step = (hi - lo) / ((n_pts - 1) as f64);
    for i in 0..n_pts {
        let u = lo + (i as f64) * step;
        let unif = riemann.collapse_score(u, u_obs_ambig);
        let pw = prior.eval_pair(u, u_obs_ambig - u);
        writeln!(wb, "{:.6},{:.6},{:.6},{:.6}", u, unif, pw, unif * pw)?;
    }
    wb.flush()?;

    Ok(())
}
