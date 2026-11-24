//! Four preregistered experimental suites (C1-C4) for Past-Future Wave Collapse
//! and Exact Configuration Search toward Riemann Primes, plus visualization trace exporters.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::collapse::{
    angular_error, blind_discover_zeros, eval_future_wave_alone, eval_holographic_born,
    eval_linear_counterprop, eval_naive_bilinear, eval_past_wave_alone, eval_range_discrimination,
    eval_transactional_collapse, evaluate_zero_discovery, optimize_collapse_config, pearson_corr,
    predict_chebyshev_fluctuation, random_search_config, CollapseConfig, GuinandWeilField,
    TrainWindowCache,
};
use crate::riemann::{chebyshev_psi_table, compute_riemann_zeros};
use crate::rng::Rng;

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * (i as f64) / (n - 1) as f64)
        .collect()
}

pub fn run_all_suites(seed: u64, phase: &str, out_dir: &Path, export_traces: bool) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");

    // Precompute exact 256 Riemann zeros (used for C1/C2/C3 and ONLY as ground-truth evaluator in C4)
    let zeros_256 = compute_riemann_zeros(256);
    let zeros_128 = &zeros_256[..128];
    let zeros_20 = &zeros_256[..20];

    // =========================================================================
    // SUITE C1: Comparison of 4 Physical Collapse Operators & Strict Emergence
    // =========================================================================
    let mut f_c1 = BufWriter::new(File::create(out_dir.join("metrics_c1.csv")).unwrap());
    writeln!(
        f_c1,
        "seed,phase,operator,prime_only_auc,prime_power_auc,prime_snr,past_alone_prime_auc,future_alone_prime_auc,off_time_auc,emergence_gain,eval_ms"
    )
    .unwrap();

    // Exact stationary log-space critical-line reference (sigma_tot = 0.0 keeps composite variance flat while prime peaks ln(p)/sqrt(p) stay ~[0.41, 0.74])
    let ref_cfg = CollapseConfig::analytic_ref(0.0);
    let mut off_cfg = ref_cfg;
    off_cfg.tau = 0.25;

    // 1. transactional_half_freq (strict emergence)
    let t0 = Instant::now();
    let m_trans = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, zeros_128, &ref_cfg));
    let m_past = eval_range_discrimination(2, 100, |x| eval_past_wave_alone(x, zeros_128, &ref_cfg));
    let m_fut = eval_range_discrimination(2, 100, |x| eval_future_wave_alone(x, zeros_128, &ref_cfg));
    let m_off = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, zeros_128, &off_cfg));
    let emerg = m_trans.prime_only_auc - m_past.prime_only_auc.max(m_fut.prime_only_auc);
    writeln!(
        f_c1,
        "{},{},transactional_half_freq,{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
        seed,
        phase,
        m_trans.prime_only_auc,
        m_trans.prime_power_auc,
        m_trans.prime_power_snr,
        m_past.prime_only_auc,
        m_fut.prime_only_auc,
        m_off.prime_power_auc,
        emerg,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // 2. linear_counterprop (dispersive D'Alembert superposition)
    let t0 = Instant::now();
    let m_lin = eval_range_discrimination(2, 100, |x| eval_linear_counterprop(x, zeros_128, &ref_cfg).2);
    let m_lin_p = eval_range_discrimination(2, 100, |x| eval_linear_counterprop(x, zeros_128, &off_cfg).0);
    let m_lin_f = eval_range_discrimination(2, 100, |x| eval_linear_counterprop(x, zeros_128, &off_cfg).1);
    let m_lin_off = eval_range_discrimination(2, 100, |x| eval_linear_counterprop(x, zeros_128, &off_cfg).2);
    let emerg_lin = m_lin.prime_only_auc - m_lin_p.prime_only_auc.max(m_lin_f.prime_only_auc);
    writeln!(
        f_c1,
        "{},{},linear_counterprop,{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
        seed,
        phase,
        m_lin.prime_only_auc,
        m_lin.prime_power_auc,
        m_lin.prime_power_snr,
        m_lin_p.prime_only_auc,
        m_lin_f.prime_only_auc,
        m_lin_off.prime_power_auc,
        emerg_lin,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // 3. holographic_born (Born-rule interference with smooth advanced carrier)
    let t0 = Instant::now();
    let m_born = eval_range_discrimination(2, 100, |x| eval_holographic_born(x, zeros_128, &ref_cfg));
    let m_born_off = eval_range_discrimination(2, 100, |x| eval_holographic_born(x, zeros_128, &off_cfg));
    writeln!(
        f_c1,
        "{},{},holographic_born,{:.8},{:.8},{:.8},0.50000000,0.50000000,{:.8},{:.8},{:.3}",
        seed,
        phase,
        m_born.prime_only_auc,
        m_born.prime_power_auc,
        m_born.prime_power_snr,
        m_born_off.prime_power_auc,
        m_born.prime_only_auc - 0.5,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // 4. naive_bilinear_ctrl (negative control: global product without modal pairing)
    let t0 = Instant::now();
    let m_naive = eval_range_discrimination(2, 100, |x| eval_naive_bilinear(x, zeros_128, &ref_cfg));
    let m_naive_off = eval_range_discrimination(2, 100, |x| eval_naive_bilinear(x, zeros_128, &off_cfg));
    writeln!(
        f_c1,
        "{},{},naive_bilinear_ctrl,{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
        seed,
        phase,
        m_naive.prime_only_auc,
        m_naive.prime_power_auc,
        m_naive.prime_power_snr,
        m_past.prime_only_auc,
        m_fut.prime_only_auc,
        m_naive_off.prime_power_auc,
        m_naive.prime_only_auc - m_past.prime_only_auc.max(m_fut.prime_only_auc),
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // 5. poisson_freq_ctrl (negative spectral control per seed)
    let t0 = Instant::now();
    let mut rng_c1 = Rng::from_seed(seed, 101);
    let mut poi_freqs: Vec<f64> = (0..128)
        .map(|_| rng_c1.uniform(zeros_128[0], zeros_128[127]))
        .collect();
    poi_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m_poi = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, &poi_freqs, &ref_cfg));
    let m_poi_off = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, &poi_freqs, &off_cfg));
    writeln!(
        f_c1,
        "{},{},poisson_freq_ctrl,{:.8},{:.8},{:.8},0.50000000,0.50000000,{:.8},{:.8},{:.3}",
        seed,
        phase,
        m_poi.prime_only_auc,
        m_poi.prime_power_auc,
        m_poi.prime_power_snr,
        m_poi_off.prime_power_auc,
        m_poi.prime_only_auc - 0.5,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();
    f_c1.flush().unwrap();

    // =========================================================================
    // SUITE C2: Automatic Search for the Exact Collapse Configuration (7D)
    // =========================================================================
    let mut f_c2 = BufWriter::new(File::create(out_dir.join("metrics_c2.csv")).unwrap());
    writeln!(
        f_c2,
        "seed,phase,arm,tau,k_sum,delta_phi,sigma_tot,beta_disp,p_win,v_asym,err_tau,err_k_sum,err_phi,err_beta,train_obj,train_auc,heldout_pp_auc,heldout_p_auc,heldout_snr,eval_ms"
    )
    .unwrap();

    let train_cache = TrainWindowCache::new(2, 60);
    let mut rng_c2_init = Rng::from_seed(seed, 201);
    let mut rng_c2_rand = Rng::from_seed(seed, 202);
    let mut rng_c2_opt = Rng::from_seed(seed, 203);

    let write_c2_row = |f: &mut BufWriter<File>, arm: &str, cfg: &CollapseConfig, eval_ms: f64| {
        let train_obj = train_cache.objective(zeros_128, 128, cfg);
        let tr = eval_range_discrimination(2, 60, |x| eval_transactional_collapse(x, zeros_128, cfg));
        let ho = eval_range_discrimination(61, 150, |x| eval_transactional_collapse(x, zeros_128, cfg));
        writeln!(
            f,
            "{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            arm,
            cfg.tau,
            cfg.k_sum,
            cfg.delta_phi,
            cfg.sigma_tot,
            cfg.beta_disp,
            cfg.p_win,
            cfg.v_asym,
            cfg.tau.abs(),
            (cfg.k_sum - 1.0).abs(),
            angular_error(cfg.delta_phi, PI),
            cfg.beta_disp.abs(),
            train_obj,
            tr.prime_power_auc,
            ho.prime_power_auc,
            ho.prime_only_auc,
            ho.prime_power_snr,
            eval_ms
        )
        .unwrap();
    };

    // Arm 1: Random uncalibrated initial configuration
    let t0 = Instant::now();
    let cfg_init = CollapseConfig::random_in_bounds(&mut rng_c2_init);
    write_c2_row(&mut f_c2, "init_random", &cfg_init, t0.elapsed().as_secs_f64() * 1e3);

    // Arm 2: Uniform random search control (exact 3000 evals)
    let t0 = Instant::now();
    let res_rand = random_search_config(&train_cache, zeros_128, 3000, &mut rng_c2_rand);
    write_c2_row(&mut f_c2, "random_search_ctrl", &res_rand.best_cfg, t0.elapsed().as_secs_f64() * 1e3);

    // Arm 3: Coarse-to-Fine Multi-Scale DE + Coordinate Pattern Polish (exact 3000 evals)
    let t0 = Instant::now();
    let res_opt = optimize_collapse_config(&train_cache, zeros_128, &mut rng_c2_opt, export_traces);
    write_c2_row(&mut f_c2, "collapse_optimizer", &res_opt.best_cfg, t0.elapsed().as_secs_f64() * 1e3);

    // Arm 4: Fixed naive x^{-0.5} exponent control on the discovered optimal phase/time config
    let t0 = Instant::now();
    let mut cfg_neg_half = res_opt.best_cfg;
    cfg_neg_half.sigma_tot = -0.5;
    write_c2_row(&mut f_c2, "fixed_sigma_neg_half", &cfg_neg_half, t0.elapsed().as_secs_f64() * 1e3);
    f_c2.flush().unwrap();

    // =========================================================================
    // SUITE C3: Collapse Stability Landscape, Temporal FWHM & Mode Count M
    // =========================================================================
    let mut f_c3 = BufWriter::new(File::create(out_dir.join("metrics_c3.csv")).unwrap());
    writeln!(f_c3, "seed,phase,probe,param_val,auc_2_100,auc_61_150,snr_2_100,eval_ms").unwrap();

    // Temporal FWHM probe around tau = 0
    let t0 = Instant::now();
    let tau_grid = linspace(-0.30, 0.30, 241);
    let mut fwhm_count = 0usize;
    let dt_tau = 0.60 / 240.0;
    for &tau in &tau_grid {
        let mut c = ref_cfg;
        c.tau = tau;
        let m = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, zeros_128, &c));
        if m.prime_power_auc >= 0.85 {
            fwhm_count += 1;
        }
    }
    let fwhm_tau = fwhm_count as f64 * dt_tau;
    writeln!(
        f_c3,
        "{},{},temporal_fwhm_085,{:.8},1.00000000,1.00000000,0.00000000,{:.3}",
        seed,
        phase,
        fwhm_tau,
        t0.elapsed().as_secs_f64() * 1e3
    )
    .unwrap();

    // Phase symmetry probes
    for &(label, phi) in &[
        ("phase_pi", PI),
        ("phase_zero", 0.0),
        ("phase_half_pi", 0.5 * PI),
        ("phase_neg_half_pi", -0.5 * PI),
    ] {
        let t0 = Instant::now();
        let mut c = ref_cfg;
        c.delta_phi = phi;
        let m1 = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, zeros_128, &c));
        let m2 = eval_range_discrimination(61, 150, |x| eval_transactional_collapse(x, zeros_128, &c));
        writeln!(
            f_c3,
            "{},{},{},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            label,
            phi,
            m1.prime_power_auc,
            m2.prime_power_auc,
            m1.prime_power_snr,
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();
    }

    // Frequency scale detuning probes
    for &dk in &[0.0_f64, 0.002, 0.005, 0.010, 0.020, 0.050] {
        let t0 = Instant::now();
        let mut c = ref_cfg;
        c.k_sum = 1.0 + dk;
        let m1 = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, zeros_128, &c));
        let m2 = eval_range_discrimination(61, 150, |x| eval_transactional_collapse(x, zeros_128, &c));
        writeln!(
            f_c3,
            "{},{},detune_k_{:.3},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            dk,
            dk,
            m1.prime_power_auc,
            m2.prime_power_auc,
            m1.prime_power_snr,
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();
    }

    // Mode count M Rayleigh resolution transition
    for &m_modes in &[8usize, 16, 32, 64, 128, 256] {
        let t0 = Instant::now();
        let slice = &zeros_256[..m_modes];
        let m1 = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, slice, &ref_cfg));
        let m2 = eval_range_discrimination(61, 150, |x| eval_transactional_collapse(x, slice, &ref_cfg));
        writeln!(
            f_c3,
            "{},{},modes_M_{},{:.1},{:.8},{:.8},{:.8},{:.3}",
            seed,
            phase,
            m_modes,
            m_modes as f64,
            m1.prime_power_auc,
            m2.prime_power_auc,
            m1.prime_power_snr,
            t0.elapsed().as_secs_f64() * 1e3
        )
        .unwrap();
    }
    f_c3.flush().unwrap();

    // =========================================================================
    // SUITE C4: Blind Discovery of Riemann Zeros from Elementary Primes Only (0 Zeta Calls)
    // =========================================================================
    let mut f_c4 = BufWriter::new(File::create(out_dir.join("metrics_c4.csv")).unwrap());
    writeln!(
        f_c4,
        "seed,phase,arm,distinct_zeros_found,mean_zero_err,auc_2_45,snr_2_45,extrap_psi_corr,eval_ms"
    )
    .unwrap();

    let gw_field = GuinandWeilField::from_sieve(600);
    let psi_table = chebyshev_psi_table(1200);
    let extrap_xs: Vec<f64> = (601..=1200).map(|n| n as f64).collect();
    let extrap_true: Vec<f64> = (601..=1200).map(|n| psi_table[n] - n as f64).collect();

    let mut rng_c4_init = Rng::from_seed(seed, 401);
    let mut rng_c4_dyson = Rng::from_seed(seed, 402);
    let mut rng_c4_norep = Rng::from_seed(seed, 402); // IDENTICAL thermal noise stream for causal ablation!

    let init_20: Vec<f64> = (0..20).map(|_| rng_c4_init.uniform(12.5, 78.0)).collect();

    let eval_and_write_c4 = |f: &mut BufWriter<File>, arm: &str, freqs: &[f64], ms: f64| {
        let (found, mean_err) = evaluate_zero_discovery(freqs, zeros_20, 0.35);
        let cfg20 = CollapseConfig::analytic_ref(0.0);
        let m45 = eval_range_discrimination(2, 45, |x| eval_transactional_collapse(x, freqs, &cfg20));
        let pred: Vec<f64> = extrap_xs
            .iter()
            .map(|&x| predict_chebyshev_fluctuation(x, freqs))
            .collect();
        let corr = pearson_corr(&pred, &extrap_true);
        writeln!(
            f,
            "{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.3}",
            seed, phase, arm, found, mean_err, m45.prime_power_auc, m45.prime_power_snr, corr, ms
        )
        .unwrap();
    };

    // 1. random_freq_ctrl
    let t0 = Instant::now();
    eval_and_write_c4(&mut f_c4, "random_freq_ctrl", &init_20, t0.elapsed().as_secs_f64() * 1e3);

    // 2. blind_collapse_dyson (Guinand-Weil resonance + short-range Dyson spectral repulsion)
    let t0 = Instant::now();
    let disc_dyson = blind_discover_zeros(&gw_field, &init_20, 12.0, 78.5, 0.32, &mut rng_c4_dyson);
    eval_and_write_c4(&mut f_c4, "blind_collapse_dyson", &disc_dyson, t0.elapsed().as_secs_f64() * 1e3);

    // 3. blind_collapse_no_repulsion (exact causal ablation: repulsion = 0)
    let t0 = Instant::now();
    let disc_norep = blind_discover_zeros(&gw_field, &init_20, 12.0, 78.5, 0.0, &mut rng_c4_norep);
    eval_and_write_c4(&mut f_c4, "blind_collapse_no_repulsion", &disc_norep, t0.elapsed().as_secs_f64() * 1e3);

    // 4. exact_20_zeros (theoretical ceiling)
    let t0 = Instant::now();
    eval_and_write_c4(&mut f_c4, "exact_20_zeros", zeros_20, t0.elapsed().as_secs_f64() * 1e3);
    f_c4.flush().unwrap();

    // =========================================================================
    // OPTIONAL VISUALIZATION TRACES (representative seed)
    // =========================================================================
    if export_traces {
        // 1. Past alone vs Future alone vs Exact Collapse across x in [1.5, 55.0]
        let mut fw = BufWriter::new(File::create(out_dir.join("trace_past_future_collapse.csv")).unwrap());
        writeln!(fw, "x,past_alone,future_alone,collapse_exact,collapse_off_time,naive_bilinear").unwrap();
        for x in linspace(1.5, 55.0, 1600) {
            writeln!(
                fw,
                "{:.6},{:.8},{:.8},{:.8},{:.8},{:.8}",
                x,
                eval_past_wave_alone(x, zeros_128, &ref_cfg),
                eval_future_wave_alone(x, zeros_128, &ref_cfg),
                eval_transactional_collapse(x, zeros_128, &ref_cfg),
                eval_transactional_collapse(x, zeros_128, &off_cfg),
                eval_naive_bilinear(x, zeros_128, &ref_cfg)
            )
            .unwrap();
        }
        fw.flush().unwrap();

        // 2. Spacetime collapse heatmap (tau in [-0.15, 0.15] x x in [2, 35])
        let mut fh = BufWriter::new(File::create(out_dir.join("trace_spacetime_collapse.csv")).unwrap());
        writeln!(fh, "tau,x,amplitude").unwrap();
        for tau in linspace(-0.12, 0.12, 49) {
            let mut c = ref_cfg;
            c.tau = tau;
            for x in linspace(1.8, 34.0, 180) {
                writeln!(
                    fh,
                    "{:.6},{:.6},{:.8}",
                    tau,
                    x,
                    eval_transactional_collapse(x, zeros_128, &c)
                )
                .unwrap();
            }
        }
        fh.flush().unwrap();

        // 3. Temporal AUC pulse tau in [-0.25, 0.25]
        let mut ft = BufWriter::new(File::create(out_dir.join("trace_tau_pulse.csv")).unwrap());
        writeln!(ft, "tau,transactional_auc,linear_auc,transactional_snr").unwrap();
        for tau in linspace(-0.25, 0.25, 161) {
            let mut c = ref_cfg;
            c.tau = tau;
            let mt = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, zeros_128, &c));
            let ml = eval_range_discrimination(2, 100, |x| eval_linear_counterprop(x, zeros_128, &c).2);
            writeln!(
                ft,
                "{:.6},{:.8},{:.8},{:.8}",
                tau, mt.prime_power_auc, ml.prime_power_auc, mt.prime_power_snr
            )
            .unwrap();
        }
        ft.flush().unwrap();

        // 4. Optimizer convergence trajectory (Suite C2)
        let mut fo = BufWriter::new(File::create(out_dir.join("trace_config_search.csv")).unwrap());
        writeln!(fo, "eval,full_obj,tau,k_sum,delta_phi").unwrap();
        for &(ev, obj, tau, ks, dp) in &res_opt.history {
            writeln!(fo, "{},{:.8},{:.8},{:.8},{:.8}", ev, obj, tau, ks, dp).unwrap();
        }
        fo.flush().unwrap();

        // 5. Blind Guinand-Weil resonance spectrum & discovered zeros (Suite C4)
        let mut fg = BufWriter::new(File::create(out_dir.join("trace_guinand_weil.csv")).unwrap());
        writeln!(fg, "omega,resonance").unwrap();
        for w in linspace(10.0, 82.0, 1441) {
            let (r, _) = gw_field.eval_and_grad(w);
            writeln!(fg, "{:.6},{:.8}", w, r).unwrap();
        }
        fg.flush().unwrap();

        let mut fz = BufWriter::new(File::create(out_dir.join("trace_discovered_zeros.csv")).unwrap());
        writeln!(fz, "idx,exact_gamma,dyson_blind,norep_blind").unwrap();
        for i in 0..20 {
            writeln!(fz, "{},{:.8},{:.8},{:.8}", i + 1, zeros_20[i], disc_dyson[i], disc_norep[i]).unwrap();
        }
        fz.flush().unwrap();

        // 6. Out-of-sample Chebyshev psi(x) - x prediction trace on [601, 1200]
        let mut fc = BufWriter::new(File::create(out_dir.join("trace_chebyshev_extrap.csv")).unwrap());
        writeln!(fc, "x,true_delta_psi,exact20_pred,dyson_blind_pred,norep_blind_pred").unwrap();
        for n in (601..=1200).step_by(2) {
            let xf = n as f64;
            writeln!(
                fc,
                "{},{:.6},{:.6},{:.6},{:.6}",
                n,
                psi_table[n] - xf,
                predict_chebyshev_fluctuation(xf, zeros_20),
                predict_chebyshev_fluctuation(xf, &disc_dyson),
                predict_chebyshev_fluctuation(xf, &disc_norep)
            )
            .unwrap();
        }
        fc.flush().unwrap();
    }
}
