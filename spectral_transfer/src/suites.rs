//! Suites T1..T4 execution engine:
//!   - Suite T1: Domain A autonomous spectral organization, unsupervised relation discovery,
//!               unseen inference, Past x Future abduction, and 2-hop composition.
//!   - Suite T2: Causal dissection of Frequencies (Omega), Phases (phi), and Pairings (pi/interferometric).
//!   - Suite T3: Frozen-core cross-domain transfer A -> B (Continuous Coupled Physical Oscillators, R^12).
//!   - Suite T4: Frozen-core cross-domain transfer A -> C (Discrete Relational Graph Z_5 x {0..3}, R^16).

use crate::domains::{generate_domain_a, generate_domain_b, generate_domain_c, DomainData};
use crate::metrics::{eval_mlp, eval_spectral, spectral_spacing_r, EvalSummary};
use crate::models::{
    MlpCore, MlpModel, PairingMode, SpectralCore, SpectralModel, SpectrumKind, TrainConfig,
};
use crate::rng::Rng64;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn domain_a_cfg(freeze_core: bool, use_ssl: bool) -> TrainConfig {
    TrainConfig {
        steps: 800,
        batch_size: 64,
        lr_encoder: 0.018,
        lr_core: 0.012,
        lr_head: 0.018,
        wd_encoder: 5e-4,
        wd_core: 1.5e-3,
        wd_head: 2e-3,
        w_mse: 1.0,
        w_ce: 1.0,
        w_ssl: if use_ssl { 0.60 } else { 0.0 },
        lambda_vec: 0.005,
        lambda_rad: 0.018,
        lambda_alias: 0.015,
        lambda_bw: 10.0,
        lambda_var: 0.08,
        lambda_cov: 0.15,
        latent_norm_reg: 0.04,
        latent_jitter: 0.05,
        freeze_core,
    }
}

fn target_transfer_cfg(domain: &DomainData, freeze_core: bool) -> TrainConfig {
    let has_reg = domain.reg_dim > 0;
    TrainConfig {
        steps: 500,
        batch_size: 32,
        lr_encoder: 0.022,
        lr_core: 0.010,
        lr_head: 0.022,
        wd_encoder: 1.2e-2,
        wd_core: 2.0e-3,
        wd_head: 1.2e-2,
        w_mse: if has_reg { 1.0 } else { 0.0 },
        w_ce: if has_reg { 0.9 } else { 1.0 },
        w_ssl: 0.0,
        lambda_vec: 0.005,
        lambda_rad: 0.012,
        lambda_alias: 0.015,
        lambda_bw: 8.0,
        lambda_var: 0.0,
        lambda_cov: 0.0,
        latent_norm_reg: 0.05,
        latent_jitter: 0.0,
        freeze_core,
    }
}

/// Evaluate regularized training objective on a given train subset (used strictly for fair
/// multi-restart model selection across ALL arms equally — zero access to unseen test pairs!).
fn spectral_train_loss(model: &SpectralModel, domain: &DomainData, train_idx: &[usize], cfg: &TrainConfig) -> f64 {
    let mut tot = 0.0;
    for &idx in train_idx {
        let p = &domain.pairs[idx];
        let (reg, logits, _, _) = model.predict_pair(&domain.obs[p.a], &domain.obs[p.b]);
        if domain.reg_dim > 0 && cfg.w_mse > 0.0 {
            let inv_r = 1.0 / (domain.reg_dim as f64);
            for c in 0..domain.reg_dim {
                let e = reg[c] - p.reg[c];
                tot += cfg.w_mse * 0.5 * e * e * inv_r;
            }
        }
        if domain.cls_dim > 0 && cfg.w_ce > 0.0 {
            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_e: f64 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            let prob = ((logits[p.cls] - max_l).exp() / sum_e).max(1e-15);
            tot += -cfg.w_ce * prob.ln();
        }
    }
    let reg_pen = 0.5 * cfg.wd_encoder * model.encoder.w.iter().map(|x| x * x).sum::<f64>()
        + 0.5 * cfg.wd_head * (model.head.w_reg.iter().map(|x| x * x).sum::<f64>()
            + model.head.w_cls.iter().map(|x| x * x).sum::<f64>());
    tot / (train_idx.len().max(1) as f64) + reg_pen
}

fn mlp_train_loss(model: &MlpModel, domain: &DomainData, train_idx: &[usize], cfg: &TrainConfig) -> f64 {
    let mut tot = 0.0;
    for &idx in train_idx {
        let p = &domain.pairs[idx];
        let (reg, logits, _) = model.predict_pair(&domain.obs[p.a], &domain.obs[p.b]);
        if domain.reg_dim > 0 && cfg.w_mse > 0.0 {
            let inv_r = 1.0 / (domain.reg_dim as f64);
            for c in 0..domain.reg_dim {
                let e = reg[c] - p.reg[c];
                tot += cfg.w_mse * 0.5 * e * e * inv_r;
            }
        }
        if domain.cls_dim > 0 && cfg.w_ce > 0.0 {
            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_e: f64 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            let prob = ((logits[p.cls] - max_l).exp() / sum_e).max(1e-15);
            tot += -cfg.w_ce * prob.ln();
        }
    }
    let reg_pen = 0.5 * cfg.wd_encoder * model.encoder.w.iter().map(|x| x * x).sum::<f64>()
        + 0.5 * cfg.wd_head * (model.head.w_reg.iter().map(|x| x * x).sum::<f64>()
            + model.head.w_cls.iter().map(|x| x * x).sum::<f64>());
    tot / (train_idx.len().max(1) as f64) + reg_pen
}

/// Fit target adapter (2 deterministic restarts selecting lowest regularized training objective;
/// strictly bit-exact frozen core when `freeze_core == true`).
fn fit_spectral_target(
    domain: &DomainData,
    train_idx: &[usize],
    base_core: &SpectralCore,
    freeze_core: bool,
    rng: &mut Rng64,
) -> (SpectralModel, f64, bool) {
    let cfg = target_transfer_cfg(domain, freeze_core);
    let pre_hash = base_core.checksum();
    let t0 = Instant::now();
    let mut best_model: Option<SpectralModel> = None;
    let mut best_loss = f64::INFINITY;
    for r in 0..2 {
        let mut r_rng = rng.fork(0x7700 + r as u64);
        let core_clone = base_core.clone();
        let mut model = SpectralModel::new(domain.obs_dim, domain.reg_dim, domain.cls_dim, core_clone, &mut r_rng);
        model.train(domain, train_idx, &cfg, &mut r_rng);
        let tr_loss = spectral_train_loss(&model, domain, train_idx, &cfg);
        if tr_loss < best_loss {
            best_loss = tr_loss;
            best_model = Some(model);
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let m = best_model.unwrap();
    let checksum_ok = if freeze_core {
        m.core.checksum() == pre_hash
    } else {
        true
    };
    (m, elapsed_ms, checksum_ok)
}

fn fit_mlp_target(
    domain: &DomainData,
    train_idx: &[usize],
    base_core: &MlpCore,
    freeze_core: bool,
    rng: &mut Rng64,
) -> (MlpModel, f64, bool) {
    let cfg = target_transfer_cfg(domain, freeze_core);
    let pre_hash = base_core.checksum();
    let t0 = Instant::now();
    let mut best_model: Option<MlpModel> = None;
    let mut best_loss = f64::INFINITY;
    for r in 0..2 {
        let mut r_rng = rng.fork(0x8800 + r as u64);
        let core_clone = base_core.clone();
        let mut model = MlpModel::new(domain.obs_dim, domain.reg_dim, domain.cls_dim, core_clone, &mut r_rng);
        model.train(domain, train_idx, &cfg, &mut r_rng);
        let tr_loss = mlp_train_loss(&model, domain, train_idx, &cfg);
        if tr_loss < best_loss {
            best_loss = tr_loss;
            best_model = Some(model);
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let m = best_model.unwrap();
    let checksum_ok = if freeze_core {
        m.core.checksum() == pre_hash
    } else {
        true
    };
    (m, elapsed_ms, checksum_ok)
}

pub fn run_all_suites_for_seed(seed: u64, out_dir: &Path, write_trace: bool) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;
    let root_rng = Rng64::new(seed);

    // Generate the 3 genuinely independent domains with orthogonal PRNG forks.
    let mut rng_a = root_rng.fork(0xA001);
    let mut rng_b = root_rng.fork(0xB002);
    let mut rng_c = root_rng.fork(0xC003);
    let dom_a = generate_domain_a(&mut rng_a);
    let dom_b = generate_domain_b(&mut rng_b);
    let dom_c = generate_domain_c(&mut rng_c);
    let dom_a_noise = dom_a.make_shuffled_label_control(&mut root_rng.fork(0xA999));

    let train_a = dom_a.train_indices(dom_a.default_train_n);
    let test_a = dom_a.test_indices(dom_a.default_train_n);
    let cfg_a = domain_a_cfg(false, true);
    let cfg_a_noise = domain_a_cfg(false, false);

    // =========================================================================
    // SUITE T1: Domain A Autonomous Spectral Learning, Organization & Inference
    // =========================================================================
    let mut t1_w = BufWriter::new(File::create(out_dir.join("metrics_t1.csv"))?);
    writeln!(
        t1_w,
        "seed,arm,core_params,uses_riemann,spec_r_init,spec_r_final,cls_acc,macro_f1,reg_rmse,nmi_unsup,ari_unsup,abd_top1,psr,ghost,cfr,two_hop_acc,spec_mu,phase_odd_frac,train_ms"
    )?;

    let t1_spec_arms: [(&str, SpectrumKind, u64); 5] = [
        ("adaptive_spectral", SpectrumKind::Adaptive, 0x101),
        ("riemann_fixed", SpectrumKind::RiemannFixed, 0x102),
        ("fourier_fixed", SpectrumKind::FourierFixed, 0x103),
        ("poisson_fixed", SpectrumKind::PoissonFixed, 0x104),
        ("dyson_fixed", SpectrumKind::DysonFixed, 0x105),
    ];

    let mut trained_spec_models: Vec<(&str, SpectralModel, f64, EvalSummary)> = Vec::new();
    let mut init_adaptive_omega = Vec::new();
    let mut init_adaptive_phi = Vec::new();

    for &(arm_name, kind, salt) in &t1_spec_arms {
        let mut rng_m = root_rng.fork(salt);
        let core = SpectralCore::new(kind, PairingMode::FullInterferometric, false, &mut rng_m);
        if kind == SpectrumKind::Adaptive {
            init_adaptive_omega = core.omega.clone();
            init_adaptive_phi = core.phi_diff.clone();
        }
        let spec_r_init = spectral_spacing_r(&core.omega);
        let mut model = SpectralModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, core, &mut rng_m);
        let t0 = Instant::now();
        model.train(&dom_a, train_a, &cfg_a, &mut rng_m);
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut eval_rng = root_rng.fork(0xE000 + salt);
        let ev = eval_spectral(&model, &dom_a, test_a, &mut eval_rng);
        writeln!(
            t1_w,
            "0x{:X},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
            seed,
            arm_name,
            model.core.param_count(),
            if model.core.uses_riemann_zeros { 1 } else { 0 },
            spec_r_init,
            ev.spec_r,
            ev.cls_acc,
            ev.macro_f1,
            ev.reg_rmse,
            ev.nmi_unsup,
            ev.ari_unsup,
            ev.abd_top1,
            ev.psr,
            ev.ghost,
            ev.cfr,
            ev.two_hop_acc,
            ev.spec_mu,
            ev.phase_odd_frac,
            train_ms
        )?;
        trained_spec_models.push((arm_name, model, spec_r_init, ev));
    }

    // Matched-capacity MLP baseline on Domain A
    let mut rng_mlp = root_rng.fork(0x106);
    let mlp_core_init = MlpCore::new(&mut rng_mlp);
    let mut mlp_a = MlpModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, mlp_core_init, &mut rng_mlp);
    let t0_mlp = Instant::now();
    mlp_a.train(&dom_a, train_a, &cfg_a, &mut rng_mlp);
    let mlp_ms = t0_mlp.elapsed().as_secs_f64() * 1000.0;
    let ev_mlp_a = eval_mlp(&mlp_a, &dom_a, test_a, &mut root_rng.fork(0xE106));
    writeln!(
        t1_w,
        "0x{:X},mlp_conventional,{},0,0.000000,0.000000,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},0.000000,0.000000,{:.2}",
        seed,
        mlp_a.core.param_count(),
        ev_mlp_a.cls_acc,
        ev_mlp_a.macro_f1,
        ev_mlp_a.reg_rmse,
        ev_mlp_a.nmi_unsup,
        ev_mlp_a.ari_unsup,
        ev_mlp_a.abd_top1,
        ev_mlp_a.psr,
        ev_mlp_a.ghost,
        ev_mlp_a.cfr,
        ev_mlp_a.two_hop_acc,
        mlp_ms
    )?;
    t1_w.flush()?;

    // Extract trained cores for downstream suites T2, T3, T4:
    let adaptive_a_model = &trained_spec_models[0].1;
    let riemann_a_model = &trained_spec_models[1].1;
    let fourier_a_model = &trained_spec_models[2].1;
    let poisson_a_model = &trained_spec_models[3].1;
    let dyson_a_model = &trained_spec_models[4].1;

    // Train the shuffled-label negative control core on Domain A (zero genuine relational signal!)
    let mut rng_noise = root_rng.fork(0x199);
    let noise_core_init = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, false, &mut rng_noise);
    let mut noise_a_model = SpectralModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, noise_core_init, &mut rng_noise);
    noise_a_model.train(&dom_a_noise, train_a, &cfg_a_noise, &mut rng_noise);

    // =========================================================================
    // SUITE T2: Causal Dissection of Frequencies, Phases, and Pairings (Domain A)
    // =========================================================================
    let mut t2_w = BufWriter::new(File::create(out_dir.join("metrics_t2.csv"))?);
    writeln!(
        t2_w,
        "seed,arm,axis,cls_acc,acc_sym,acc_asym,reg_rmse,nmi_unsup,abd_top1,psr,ghost,cfr,spec_r,spec_mu,phase_odd_frac"
    )?;

    let write_t2_row = |w: &mut BufWriter<File>, arm: &str, axis: &str, ev: &EvalSummary| -> std::io::Result<()> {
        writeln!(
            w,
            "0x{:X},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            arm,
            axis,
            ev.cls_acc,
            ev.acc_sym,
            ev.acc_asym,
            ev.reg_rmse,
            ev.nmi_unsup,
            ev.abd_top1,
            ev.psr,
            ev.ghost,
            ev.cfr,
            ev.spec_r,
            ev.spec_mu,
            ev.phase_odd_frac
        )
    };

    // 1. Reference full_adaptive
    write_t2_row(&mut t2_w, "full_adaptive", "reference", &trained_spec_models[0].3)?;

    // 2. Phase ablation A (inference-time scramble of trained phases on exact same model)
    let mut m_phase_scr = adaptive_a_model.clone();
    m_phase_scr.core = adaptive_a_model.core.with_scrambled_phases(&mut root_rng.fork(0x201));
    let ev_phase_scr = eval_spectral(&m_phase_scr, &dom_a, test_a, &mut root_rng.fork(0xE201));
    write_t2_row(&mut t2_w, "ablate_phase_scramble", "phases", &ev_phase_scr)?;

    // 3. Phase ablation B (trained from scratch with phi == 0 locked)
    let mut rng_zp = root_rng.fork(0x202);
    let zp_core = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, true, &mut rng_zp);
    let mut m_zp = SpectralModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, zp_core, &mut rng_zp);
    m_zp.train(&dom_a, train_a, &cfg_a, &mut rng_zp);
    let ev_zp = eval_spectral(&m_zp, &dom_a, test_a, &mut root_rng.fork(0xE202));
    write_t2_row(&mut t2_w, "train_zero_phase", "phases", &ev_zp)?;

    // 4. Pairing ablation A (trained with AdditiveOnly superposition — zero cross-interference!)
    let mut rng_add = root_rng.fork(0x203);
    let add_core = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::AdditiveOnly, false, &mut rng_add);
    let mut m_add = SpectralModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, add_core, &mut rng_add);
    m_add.train(&dom_a, train_a, &cfg_a, &mut rng_add);
    let ev_add = eval_spectral(&m_add, &dom_a, test_a, &mut root_rng.fork(0xE203));
    write_t2_row(&mut t2_w, "ablate_pairing_additive", "pairings", &ev_add)?;

    // 5. Pairing ablation B (exact learned omega/phi/W_pair, ONLY breaking mode pairing pi(m) != m!)
    let mut m_perm = adaptive_a_model.clone();
    m_perm.core.pairing = PairingMode::PermutedPairing;
    let ev_perm = eval_spectral(&m_perm, &dom_a, test_a, &mut root_rng.fork(0xE204));
    write_t2_row(&mut t2_w, "ablate_pairing_permuted", "pairings", &ev_perm)?;

    // 6-8. Frequency ablations (reuse exact trained fixed-frequency arms from T1!)
    write_t2_row(&mut t2_w, "ablate_freq_riemann", "frequencies", &trained_spec_models[1].3)?;
    write_t2_row(&mut t2_w, "ablate_freq_fourier", "frequencies", &trained_spec_models[2].3)?;
    write_t2_row(&mut t2_w, "ablate_freq_poisson", "frequencies", &trained_spec_models[3].3)?;
    t2_w.flush()?;

    // Prepare frozen cores for cross-domain transfer T3 (Domain B) & T4 (Domain C):
    let scrambled_a_core = adaptive_a_model.core.with_scrambled_phases(&mut root_rng.fork(0x309));
    let untrained_poisson_core =
        SpectralCore::new(SpectrumKind::PoissonFixed, PairingMode::FullInterferometric, false, &mut root_rng.fork(0x30A));

    // =========================================================================
    // SUITE T3: Cross-Domain Transfer A -> B (Continuous Coupled Oscillators, R^12)
    // =========================================================================
    let mut t3_w = BufWriter::new(File::create(out_dir.join("metrics_t3.csv"))?);
    writeln!(
        t3_w,
        "seed,n_few,arm,core_frozen,adapt_params,checksum_ok,cls_acc,macro_f1,reg_rmse,nmi_unsup,ari_unsup,abd_top1,psr,ghost,cfr,train_ms"
    )?;

    let mut trace_b_models: Option<(SpectralModel, MlpModel, SpectralModel, SpectralModel)> = None;

    for &n_few in &[24usize, 48, 96] {
        let tr_b = dom_b.train_indices(n_few);
        let te_b = dom_b.test_indices(n_few);

        let frozen_spec_arms: [(&str, &SpectralCore, u64); 8] = [
            ("transfer_spectral_A", &adaptive_a_model.core, 0x310),
            ("riemann_A", &riemann_a_model.core, 0x311),
            ("fourier_A", &fourier_a_model.core, 0x312),
            ("poisson_A", &poisson_a_model.core, 0x313),
            ("dyson_A", &dyson_a_model.core, 0x314),
            ("neg_ctrl_noise_A", &noise_a_model.core, 0x315),
            ("neg_ctrl_phase_scramble", &scrambled_a_core, 0x316),
            ("untrained_poisson", &untrained_poisson_core, 0x317),
        ];

        let mut saved_tr_spec: Option<SpectralModel> = None;
        let mut saved_scr_spec: Option<SpectralModel> = None;
        let mut saved_ph_scr: Option<SpectralModel> = None;
        let mut saved_tr_mlp: Option<MlpModel> = None;

        for &(arm, core_ref, salt) in &frozen_spec_arms {
            let mut r_arm = root_rng.fork(salt + (n_few as u64) * 100);
            let (m, ms, ok) = fit_spectral_target(&dom_b, tr_b, core_ref, true, &mut r_arm);
            let ev = eval_spectral(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.head.param_count();
            writeln!(
                t3_w,
                "0x{:X},{},{},1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                arm,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.reg_rmse,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
            if write_trace && n_few == 48 {
                if arm == "transfer_spectral_A" {
                    saved_tr_spec = Some(m);
                } else if arm == "neg_ctrl_phase_scramble" {
                    saved_ph_scr = Some(m);
                }
            }
        }

        // Frozen-trunk MLP transfer A -> B (exact 171 adaptable params!)
        {
            let mut r_arm = root_rng.fork(0x318 + (n_few as u64) * 100);
            let (m, ms, ok) = fit_mlp_target(&dom_b, tr_b, &mlp_a.core, true, &mut r_arm);
            let ev = eval_mlp(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.head.param_count();
            writeln!(
                t3_w,
                "0x{:X},{},transfer_mlp_A,1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.reg_rmse,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
            if write_trace && n_few == 48 {
                saved_tr_mlp = Some(m);
            }
        }

        // From-scratch Spectral on Domain B
        {
            let mut r_arm = root_rng.fork(0x319 + (n_few as u64) * 100);
            let fresh_core = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, false, &mut r_arm);
            let (m, ms, ok) = fit_spectral_target(&dom_b, tr_b, &fresh_core, false, &mut r_arm);
            let ev = eval_spectral(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.core.param_count() + m.head.param_count();
            writeln!(
                t3_w,
                "0x{:X},{},scratch_spectral,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.reg_rmse,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
            if write_trace && n_few == 48 {
                saved_scr_spec = Some(m);
            }
        }

        // From-scratch MLP on Domain B
        {
            let mut r_arm = root_rng.fork(0x31A + (n_few as u64) * 100);
            let fresh_mlp = MlpCore::new(&mut r_arm);
            let (m, ms, ok) = fit_mlp_target(&dom_b, tr_b, &fresh_mlp, false, &mut r_arm);
            let ev = eval_mlp(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.core.param_count() + m.head.param_count();
            writeln!(
                t3_w,
                "0x{:X},{},scratch_mlp,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.reg_rmse,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
        }

        if write_trace && n_few == 48 {
            trace_b_models = Some((
                saved_tr_spec.unwrap(),
                saved_tr_mlp.unwrap(),
                saved_scr_spec.unwrap(),
                saved_ph_scr.unwrap(),
            ));
        }
    }
    t3_w.flush()?;

    // =========================================================================
    // SUITE T4: Cross-Domain Transfer A -> C (Discrete Relational Graph, R^16)
    // =========================================================================
    let mut t4_w = BufWriter::new(File::create(out_dir.join("metrics_t4.csv"))?);
    writeln!(
        t4_w,
        "seed,n_few,arm,core_frozen,adapt_params,checksum_ok,cls_acc,macro_f1,nmi_unsup,ari_unsup,alg_consistency,abd_top1,psr,ghost,cfr,train_ms"
    )?;

    let mut trace_c_models: Option<(SpectralModel, MlpModel, SpectralModel, SpectralModel)> = None;

    for &n_few in &[30usize, 60, 120] {
        let tr_c = dom_c.train_indices(n_few);
        let te_c = dom_c.test_indices(n_few);

        let frozen_spec_arms: [(&str, &SpectralCore, u64); 8] = [
            ("transfer_spectral_A", &adaptive_a_model.core, 0x410),
            ("riemann_A", &riemann_a_model.core, 0x411),
            ("fourier_A", &fourier_a_model.core, 0x412),
            ("poisson_A", &poisson_a_model.core, 0x413),
            ("dyson_A", &dyson_a_model.core, 0x414),
            ("neg_ctrl_noise_A", &noise_a_model.core, 0x415),
            ("neg_ctrl_phase_scramble", &scrambled_a_core, 0x416),
            ("untrained_poisson", &untrained_poisson_core, 0x417),
        ];

        let mut saved_tr_spec: Option<SpectralModel> = None;
        let mut saved_scr_spec: Option<SpectralModel> = None;
        let mut saved_ph_scr: Option<SpectralModel> = None;
        let mut saved_tr_mlp: Option<MlpModel> = None;

        for &(arm, core_ref, salt) in &frozen_spec_arms {
            let mut r_arm = root_rng.fork(salt + (n_few as u64) * 100);
            let (m, ms, ok) = fit_spectral_target(&dom_c, tr_c, core_ref, true, &mut r_arm);
            let ev = eval_spectral(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.head.param_count();
            writeln!(
                t4_w,
                "0x{:X},{},{},1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                arm,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.alg_consistency,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
            if write_trace && n_few == 60 {
                if arm == "transfer_spectral_A" {
                    saved_tr_spec = Some(m);
                } else if arm == "neg_ctrl_phase_scramble" {
                    saved_ph_scr = Some(m);
                }
            }
        }

        // Frozen-trunk MLP transfer A -> C (exact 136 adaptable params!)
        {
            let mut r_arm = root_rng.fork(0x418 + (n_few as u64) * 100);
            let (m, ms, ok) = fit_mlp_target(&dom_c, tr_c, &mlp_a.core, true, &mut r_arm);
            let ev = eval_mlp(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.head.param_count();
            writeln!(
                t4_w,
                "0x{:X},{},transfer_mlp_A,1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.alg_consistency,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
            if write_trace && n_few == 60 {
                saved_tr_mlp = Some(m);
            }
        }

        // From-scratch Spectral on Domain C
        {
            let mut r_arm = root_rng.fork(0x419 + (n_few as u64) * 100);
            let fresh_core = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, false, &mut r_arm);
            let (m, ms, ok) = fit_spectral_target(&dom_c, tr_c, &fresh_core, false, &mut r_arm);
            let ev = eval_spectral(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.core.param_count() + m.head.param_count();
            writeln!(
                t4_w,
                "0x{:X},{},scratch_spectral,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.alg_consistency,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
            if write_trace && n_few == 60 {
                saved_scr_spec = Some(m);
            }
        }

        // From-scratch MLP on Domain C
        {
            let mut r_arm = root_rng.fork(0x41A + (n_few as u64) * 100);
            let fresh_mlp = MlpCore::new(&mut r_arm);
            let (m, ms, ok) = fit_mlp_target(&dom_c, tr_c, &fresh_mlp, false, &mut r_arm);
            let ev = eval_mlp(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            let adapt_p = m.encoder.param_count() + m.core.param_count() + m.head.param_count();
            writeln!(
                t4_w,
                "0x{:X},{},scratch_mlp,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_p,
                if ok { 1 } else { 0 },
                ev.cls_acc,
                ev.macro_f1,
                ev.nmi_unsup,
                ev.ari_unsup,
                ev.alg_consistency,
                ev.abd_top1,
                ev.psr,
                ev.ghost,
                ev.cfr,
                ms
            )?;
        }

        if write_trace && n_few == 60 {
            trace_c_models = Some((
                saved_tr_spec.unwrap(),
                saved_tr_mlp.unwrap(),
                saved_scr_spec.unwrap(),
                saved_ph_scr.unwrap(),
            ));
        }
    }
    t4_w.flush()?;

    // Optional detailed spectral & transfer trace for reference seed (0xD100 / 0xE100)
    if write_trace {
        let mut tr_w = BufWriter::new(File::create(out_dir.join("trace_spectra_and_collapse.csv"))?);
        writeln!(
            tr_w,
            "section,idx,label,val1,val2,val3,val4,val5,val6"
        )?;
        // Section 1: 48 learned vs fixed radial frequencies & phases
        for m in 0..48 {
            let rad = |om: &[f64]| -> f64 {
                let s = &om[m * 4..(m + 1) * 4];
                s.iter().map(|x| x * x).sum::<f64>().sqrt()
            };
            writeln!(
                tr_w,
                "spectrum_mode,{},mode_{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                m,
                m,
                rad(&init_adaptive_omega),
                rad(&adaptive_a_model.core.omega),
                rad(&riemann_a_model.core.omega),
                rad(&fourier_a_model.core.omega),
                init_adaptive_phi[m],
                adaptive_a_model.core.phi_diff[m],
            )?;
        }
        // Section 2: 16 concrete unseen examples in Domain B (N=48)
        if let Some((ref tr_spec, ref tr_mlp, ref scr_spec, ref ph_scr)) = trace_b_models {
            let te_b = dom_b.test_indices(48);
            for (k, &idx) in te_b.iter().take(16).enumerate() {
                let p = &dom_b.pairs[idx];
                let (r_sp, l_sp, _, _) = tr_spec.predict_pair(&dom_b.obs[p.a], &dom_b.obs[p.b]);
                let (r_ml, l_ml, _) = tr_mlp.predict_pair(&dom_b.obs[p.a], &dom_b.obs[p.b]);
                let (r_sc, l_sc, _, _) = scr_spec.predict_pair(&dom_b.obs[p.a], &dom_b.obs[p.b]);
                let (_, l_ph, _, _) = ph_scr.predict_pair(&dom_b.obs[p.a], &dom_b.obs[p.b]);
                let err_sp = ((0..3).map(|c| (r_sp[c] - p.reg[c]).powi(2)).sum::<f64>() / 3.0).sqrt();
                let err_ml = ((0..3).map(|c| (r_ml[c] - p.reg[c]).powi(2)).sum::<f64>() / 3.0).sqrt();
                let err_sc = ((0..3).map(|c| (r_sc[c] - p.reg[c]).powi(2)).sum::<f64>() / 3.0).sqrt();
                writeln!(
                    tr_w,
                    "transfer_b_ex,{},pair_{}_{},{},{},{},{:.5},{:.5},{:.5}",
                    k,
                    p.a,
                    p.b,
                    p.cls,
                    crate::metrics::argmax(&l_sp),
                    crate::metrics::argmax(&l_ml),
                    err_sp,
                    err_ml,
                    if crate::metrics::argmax(&l_ph) == p.cls { 1.0 } else { 0.0 } + 0.0 * (err_sc + crate::metrics::argmax(&l_sc) as f64)
                )?;
            }
        }
        // Section 3: 16 concrete unseen examples in Domain C (N=60)
        if let Some((ref tr_spec, ref tr_mlp, ref scr_spec, ref ph_scr)) = trace_c_models {
            let te_c = dom_c.test_indices(60);
            for (k, &idx) in te_c.iter().take(16).enumerate() {
                let p = &dom_c.pairs[idx];
                let (_, l_sp, _, _) = tr_spec.predict_pair(&dom_c.obs[p.a], &dom_c.obs[p.b]);
                let (_, l_ml, _) = tr_mlp.predict_pair(&dom_c.obs[p.a], &dom_c.obs[p.b]);
                let (_, l_sc, _, _) = scr_spec.predict_pair(&dom_c.obs[p.a], &dom_c.obs[p.b]);
                let (_, l_ph, _, _) = ph_scr.predict_pair(&dom_c.obs[p.a], &dom_c.obs[p.b]);
                writeln!(
                    tr_w,
                    "transfer_c_ex,{},edge_{}_{},{},{},{},{},{},{:.4}",
                    k,
                    p.a,
                    p.b,
                    p.cls,
                    crate::metrics::argmax(&l_sp),
                    crate::metrics::argmax(&l_ml),
                    crate::metrics::argmax(&l_sc),
                    crate::metrics::argmax(&l_ph),
                    crate::metrics::softmax_prob(&l_sp, p.cls)
                )?;
            }
        }
        tr_w.flush()?;
    }

    Ok(())
}
