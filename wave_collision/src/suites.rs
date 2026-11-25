//! Suites T1..T4 execution engine for the 768-wave self-calibrating colliding-wave network.

use crate::domains::{generate_domain_a, generate_domain_b, generate_domain_c, DomainData};
use crate::metrics::{eval_mlp, eval_wave, spectral_spacing_r, EvalSummary};
use crate::models::{
    CoreTrainMode, MlpCore, MlpModel, PairingMode, SpectrumKind, TrainConfig, WaveCollisionCore,
    WaveModel, N_COLLISIONS,
};
use crate::rng::{riemann_zeros_256, Rng64};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn domain_a_cfg(use_ssl: bool) -> TrainConfig {
    TrainConfig {
        steps: 800,
        batch_size: 48,
        lr_encoder: 0.018,
        lr_core: 0.018,
        lr_autocal: 0.0,
        lr_head: 0.018,
        wd_encoder: 5e-4,
        wd_core: 4.0e-3,
        wd_autocal: 0.0,
        wd_head: 3.0e-3,
        w_mse: 1.0,
        w_ce: 1.15,
        w_ssl: if use_ssl { 0.55 } else { 0.0 },
        lambda_vec: 0.025,
        lambda_rad: 0.025,
        lambda_bw: 10.0,
        lambda_var: 0.08,
        lambda_cov: 0.22,
        latent_norm_reg: 0.04,
        latent_jitter: 0.05,
        core_mode: CoreTrainMode::FullTrain,
    }
}

fn target_cfg(domain: &DomainData, core_mode: CoreTrainMode) -> TrainConfig {
    let has_reg = domain.reg_dim > 0;
    TrainConfig {
        steps: 360,
        batch_size: 32,
        lr_encoder: 0.020,
        lr_core: 0.012,
        lr_autocal: 0.045,
        lr_head: 0.020,
        wd_encoder: 2.0e-2,
        wd_core: 4.0e-3,
        wd_autocal: 0.80,
        wd_head: 2.5e-2,
        w_mse: if has_reg { 1.0 } else { 0.0 },
        w_ce: if has_reg { 0.9 } else { 1.0 },
        w_ssl: 0.0,
        lambda_vec: 0.025,
        lambda_rad: 0.014,
        lambda_bw: 8.0,
        lambda_var: 0.0,
        lambda_cov: 0.0,
        latent_norm_reg: 0.05,
        latent_jitter: 0.0,
        core_mode,
    }
}

fn wave_train_loss(model: &WaveModel, domain: &DomainData, train_idx: &[usize], cfg: &TrainConfig) -> f64 {
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
        + 0.5
            * cfg.wd_head
            * (model.head.w_reg.iter().map(|x| x * x).sum::<f64>()
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
        + 0.5
            * cfg.wd_head
            * (model.head.w_reg.iter().map(|x| x * x).sum::<f64>()
                + model.head.w_cls.iter().map(|x| x * x).sum::<f64>());
    tot / (train_idx.len().max(1) as f64) + reg_pen
}

fn fit_wave_target(
    domain: &DomainData,
    train_idx: &[usize],
    base_core: &WaveCollisionCore,
    mode: CoreTrainMode,
    rng: &mut Rng64,
) -> (WaveModel, f64, bool) {
    let cfg = target_cfg(domain, mode);
    let pre_waves = base_core.frozen_waves_checksum();
    let pre_full = base_core.full_checksum();
    let t0 = Instant::now();
    let mut best_model: Option<WaveModel> = None;
    let mut best_loss = f64::INFINITY;
    for r in 0..2 {
        let mut r_rng = rng.fork(0x7700 + r as u64);
        let mut core_clone = base_core.clone();
        core_clone.reset_autocal();
        let mut model = WaveModel::new(domain.obs_dim, domain.reg_dim, domain.cls_dim, core_clone, &mut r_rng);
        model.train(domain, train_idx, &cfg, &mut r_rng);
        let loss = wave_train_loss(&model, domain, train_idx, &cfg);
        if loss < best_loss {
            best_loss = loss;
            best_model = Some(model);
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let m = best_model.unwrap();
    let checksum_ok = match mode {
        CoreTrainMode::AutoCalOnly => m.core.frozen_waves_checksum() == pre_waves,
        CoreTrainMode::FullyFrozen => m.core.full_checksum() == pre_full,
        CoreTrainMode::FullTrain => true,
    };
    (m, elapsed_ms, checksum_ok)
}

fn fit_mlp_target(
    domain: &DomainData,
    train_idx: &[usize],
    base_core: &MlpCore,
    mode: CoreTrainMode,
    rng: &mut Rng64,
) -> (MlpModel, f64, bool) {
    let cfg = target_cfg(domain, mode);
    let pre_trunk = base_core.frozen_trunk_checksum();
    let pre_full = base_core.full_checksum();
    let t0 = Instant::now();
    let mut best_model: Option<MlpModel> = None;
    let mut best_loss = f64::INFINITY;
    for r in 0..2 {
        let mut r_rng = rng.fork(0x8800 + r as u64);
        let mut core_clone = base_core.clone();
        core_clone.reset_autocal();
        let mut model = MlpModel::new(domain.obs_dim, domain.reg_dim, domain.cls_dim, core_clone, &mut r_rng);
        model.train(domain, train_idx, &cfg, &mut r_rng);
        let loss = mlp_train_loss(&model, domain, train_idx, &cfg);
        if loss < best_loss {
            best_loss = loss;
            best_model = Some(model);
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let m = best_model.unwrap();
    let checksum_ok = match mode {
        CoreTrainMode::AutoCalOnly => m.core.frozen_trunk_checksum() == pre_trunk,
        CoreTrainMode::FullyFrozen => m.core.full_checksum() == pre_full,
        CoreTrainMode::FullTrain => true,
    };
    (m, elapsed_ms, checksum_ok)
}

fn adapt_param_count_wave(m: &WaveModel, mode: CoreTrainMode) -> usize {
    let base = m.encoder.param_count() + m.head.param_count();
    match mode {
        CoreTrainMode::FullyFrozen => base,
        CoreTrainMode::AutoCalOnly => base + m.core.autocal_param_count(),
        CoreTrainMode::FullTrain => base + m.core.core_param_count(),
    }
}

fn adapt_param_count_mlp(m: &MlpModel, mode: CoreTrainMode) -> usize {
    let base = m.encoder.param_count() + m.head.param_count();
    match mode {
        CoreTrainMode::FullyFrozen => base,
        CoreTrainMode::AutoCalOnly => base + m.core.autocal_param_count(),
        CoreTrainMode::FullTrain => base + m.core.core_param_count(),
    }
}

pub fn run_all_suites_for_seed(seed: u64, out_dir: &Path, write_trace: bool) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;
    // Warm Riemann 256 cache once before timing arms
    let _ = riemann_zeros_256();
    let root_rng = Rng64::new(seed);

    let dom_a = generate_domain_a(&mut root_rng.fork(0xA001));
    let dom_b = generate_domain_b(&mut root_rng.fork(0xB002));
    let dom_c = generate_domain_c(&mut root_rng.fork(0xC003));
    let dom_a_noise = dom_a.make_shuffled_label_control(&mut root_rng.fork(0xA999));

    let train_a = dom_a.train_indices(dom_a.default_train_n);
    let test_a = dom_a.test_indices(dom_a.default_train_n);
    let cfg_a = domain_a_cfg(true);
    let cfg_a_noise = domain_a_cfg(false);

    // =========================================================================
    // SUITE T1: Domain A (768 Colliding Waves, 0 Riemann zeros, 0 W_pair)
    // =========================================================================
    let mut t1_w = BufWriter::new(File::create(out_dir.join("metrics_t1.csv"))?);
    writeln!(
        t1_w,
        "seed,arm,core_params,uses_riemann,spec_r_init,spec_r_final,cls_acc,macro_f1,reg_rmse,nmi_unsup,ari_unsup,abd_top1,psr,ghost,cfr,two_hop_acc,phase_odd_frac,train_ms"
    )?;

    let t1_arms: [(&str, SpectrumKind, u64); 5] = [
        ("adaptive_wave", SpectrumKind::Adaptive, 0x101),
        ("riemann_fixed", SpectrumKind::RiemannFixed, 0x102),
        ("fourier_fixed", SpectrumKind::FourierFixed, 0x103),
        ("poisson_fixed", SpectrumKind::PoissonFixed, 0x104),
        ("dyson_fixed", SpectrumKind::DysonFixed, 0x105),
    ];

    let mut trained_wave_models: Vec<(&str, WaveModel, f64, EvalSummary)> = Vec::new();
    let mut init_adapt_omega1 = Vec::new();
    let mut init_adapt_phi1 = Vec::new();

    for &(arm_name, kind, salt) in &t1_arms {
        let mut rng_m = root_rng.fork(salt);
        let core = WaveCollisionCore::new(kind, PairingMode::FullCollision, false, false, &mut rng_m);
        if kind == SpectrumKind::Adaptive {
            init_adapt_omega1 = core.omega1.clone();
            init_adapt_phi1 = core.phi1.clone();
        }
        let spec_r_init = spectral_spacing_r(&core.omega1);
        let mut model = WaveModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, core, &mut rng_m);
        let t0 = Instant::now();
        model.train(&dom_a, train_a, &cfg_a, &mut rng_m);
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ev = eval_wave(&model, &dom_a, test_a, &mut root_rng.fork(0xE000 + salt));
        writeln!(
            t1_w,
            "0x{:X},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
            seed,
            arm_name,
            model.core.core_param_count(),
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
            ev.phase_odd_frac,
            train_ms
        )?;
        trained_wave_models.push((arm_name, model, spec_r_init, ev));
    }

    // Matched-capacity MLP baseline (3,832 params)
    let mut rng_mlp = root_rng.fork(0x106);
    let mlp_core_init = MlpCore::new(&mut rng_mlp);
    let mut mlp_a = MlpModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, mlp_core_init, &mut rng_mlp);
    let t0_mlp = Instant::now();
    mlp_a.train(&dom_a, train_a, &cfg_a, &mut rng_mlp);
    let mlp_ms = t0_mlp.elapsed().as_secs_f64() * 1000.0;
    let ev_mlp_a = eval_mlp(&mlp_a, &dom_a, test_a, &mut root_rng.fork(0xE106));
    writeln!(
        t1_w,
        "0x{:X},mlp_conventional,{},0,0.000000,0.000000,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},0.000000,{:.2}",
        seed,
        mlp_a.core.core_param_count(),
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

    let adapt_a = &trained_wave_models[0].1;
    let riemann_a = &trained_wave_models[1].1;
    let fourier_a = &trained_wave_models[2].1;
    let poisson_a = &trained_wave_models[3].1;
    let dyson_a = &trained_wave_models[4].1;

    // Shuffled-label negative control trained on Domain A with zero relational signal
    let mut rng_noise = root_rng.fork(0x199);
    let noise_core = WaveCollisionCore::new(SpectrumKind::Adaptive, PairingMode::FullCollision, false, false, &mut rng_noise);
    let mut noise_a = WaveModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, noise_core, &mut rng_noise);
    noise_a.train(&dom_a_noise, train_a, &cfg_a_noise, &mut rng_noise);

    // =========================================================================
    // SUITE T2: Causal Dissection (Frequencies, Phases, Wave-Collision, Stage-2)
    // =========================================================================
    let mut t2_w = BufWriter::new(File::create(out_dir.join("metrics_t2.csv"))?);
    writeln!(
        t2_w,
        "seed,arm,axis,cls_acc,acc_sym,acc_asym,reg_rmse,nmi_unsup,abd_top1,psr,ghost,cfr,spec_r,phase_odd_frac"
    )?;
    let write_t2 = |w: &mut BufWriter<File>, arm: &str, axis: &str, ev: &EvalSummary| -> std::io::Result<()> {
        writeln!(
            w,
            "0x{:X},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
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
            ev.phase_odd_frac
        )
    };

    write_t2(&mut t2_w, "full_adaptive", "reference", &trained_wave_models[0].3)?;

    let mut m_ph = adapt_a.clone();
    m_ph.core = adapt_a.core.with_scrambled_phases(&mut root_rng.fork(0x201));
    write_t2(&mut t2_w, "ablate_phase_scramble", "phases", &eval_wave(&m_ph, &dom_a, test_a, &mut root_rng.fork(0xE201)))?;

    let mut r_zp = root_rng.fork(0x202);
    let mut m_zp = WaveModel::new(
        dom_a.obs_dim,
        dom_a.reg_dim,
        dom_a.cls_dim,
        WaveCollisionCore::new(SpectrumKind::Adaptive, PairingMode::FullCollision, true, false, &mut r_zp),
        &mut r_zp,
    );
    m_zp.train(&dom_a, train_a, &cfg_a, &mut r_zp);
    write_t2(&mut t2_w, "train_zero_phase", "phases", &eval_wave(&m_zp, &dom_a, test_a, &mut root_rng.fork(0xE202)))?;

    let mut r_add = root_rng.fork(0x203);
    let mut m_add = WaveModel::new(
        dom_a.obs_dim,
        dom_a.reg_dim,
        dom_a.cls_dim,
        WaveCollisionCore::new(SpectrumKind::Adaptive, PairingMode::AdditiveOnly, false, true, &mut r_add),
        &mut r_add,
    );
    m_add.train(&dom_a, train_a, &cfg_a, &mut r_add);
    write_t2(&mut t2_w, "ablate_pairing_additive", "pairings", &eval_wave(&m_add, &dom_a, test_a, &mut root_rng.fork(0xE203)))?;

    let mut m_perm = adapt_a.clone();
    m_perm.core.pairing = PairingMode::PermutedPairing;
    write_t2(&mut t2_w, "ablate_pairing_permuted", "pairings", &eval_wave(&m_perm, &dom_a, test_a, &mut root_rng.fork(0xE204)))?;

    // Ablate Stage-2 secondary wave-wave collision (train from scratch with 512 primary waves only)
    let mut r_ns2 = root_rng.fork(0x205);
    let mut m_ns2 = WaveModel::new(
        dom_a.obs_dim,
        dom_a.reg_dim,
        dom_a.cls_dim,
        WaveCollisionCore::new(SpectrumKind::Adaptive, PairingMode::FullCollision, false, true, &mut r_ns2),
        &mut r_ns2,
    );
    m_ns2.train(&dom_a, train_a, &cfg_a, &mut r_ns2);
    write_t2(&mut t2_w, "ablate_no_stage2", "wave_collision_s2", &eval_wave(&m_ns2, &dom_a, test_a, &mut root_rng.fork(0xE205)))?;

    write_t2(&mut t2_w, "ablate_freq_riemann", "frequencies", &trained_wave_models[1].3)?;
    write_t2(&mut t2_w, "ablate_freq_fourier", "frequencies", &trained_wave_models[2].3)?;
    write_t2(&mut t2_w, "ablate_freq_poisson", "frequencies", &trained_wave_models[3].3)?;
    t2_w.flush()?;

    let scrambled_a_core = adapt_a.core.with_scrambled_phases(&mut root_rng.fork(0x309));
    let untrained_poisson_core = WaveCollisionCore::new(
        SpectrumKind::PoissonFixed,
        PairingMode::FullCollision,
        false,
        false,
        &mut root_rng.fork(0x30A),
    );

    // =========================================================================
    // SUITE T3: Cross-Domain Transfer A -> B (Continuous Physical Oscillators)
    // =========================================================================
    let mut t3_w = BufWriter::new(File::create(out_dir.join("metrics_t3.csv"))?);
    writeln!(
        t3_w,
        "seed,n_few,arm,waves_frozen,adapt_params,checksum_ok,cls_acc,macro_f1,reg_rmse,nmi_unsup,ari_unsup,abd_top1,psr,ghost,cfr,train_ms"
    )?;

    for &n_few in &[24usize, 48, 96] {
        let tr_b = dom_b.train_indices(n_few);
        let te_b = dom_b.test_indices(n_few);

        let wave_arms: [(&str, &WaveCollisionCore, CoreTrainMode, u64); 9] = [
            ("transfer_autocal_A", &adapt_a.core, CoreTrainMode::AutoCalOnly, 0x310),
            ("transfer_frozen_A", &adapt_a.core, CoreTrainMode::FullyFrozen, 0x311),
            ("riemann_A", &riemann_a.core, CoreTrainMode::AutoCalOnly, 0x312),
            ("fourier_A", &fourier_a.core, CoreTrainMode::AutoCalOnly, 0x313),
            ("poisson_A", &poisson_a.core, CoreTrainMode::AutoCalOnly, 0x314),
            ("dyson_A", &dyson_a.core, CoreTrainMode::AutoCalOnly, 0x315),
            ("neg_ctrl_noise_A", &noise_a.core, CoreTrainMode::AutoCalOnly, 0x316),
            ("neg_ctrl_phase_scramble", &scrambled_a_core, CoreTrainMode::AutoCalOnly, 0x317),
            ("untrained_poisson", &untrained_poisson_core, CoreTrainMode::AutoCalOnly, 0x318),
        ];

        for &(arm, core_ref, mode, salt) in &wave_arms {
            let mut r_arm = root_rng.fork(salt + (n_few as u64) * 100);
            let (m, ms, ok) = fit_wave_target(&dom_b, tr_b, core_ref, mode, &mut r_arm);
            let ev = eval_wave(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            writeln!(
                t3_w,
                "0x{:X},{},{},1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                arm,
                adapt_param_count_wave(&m, mode),
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

        // Matched MLP transfer with exact +48 FiLM auto-calibration knobs (219 params!)
        {
            let mut r_arm = root_rng.fork(0x319 + (n_few as u64) * 100);
            let (m, ms, ok) = fit_mlp_target(&dom_b, tr_b, &mlp_a.core, CoreTrainMode::AutoCalOnly, &mut r_arm);
            let ev = eval_mlp(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            writeln!(
                t3_w,
                "0x{:X},{},transfer_mlp_A,1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_param_count_mlp(&m, CoreTrainMode::AutoCalOnly),
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

        // From-scratch 768-wave model on Domain B
        {
            let mut r_arm = root_rng.fork(0x31A + (n_few as u64) * 100);
            let fresh = WaveCollisionCore::new(SpectrumKind::Adaptive, PairingMode::FullCollision, false, false, &mut r_arm);
            let (m, ms, ok) = fit_wave_target(&dom_b, tr_b, &fresh, CoreTrainMode::FullTrain, &mut r_arm);
            let ev = eval_wave(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            writeln!(
                t3_w,
                "0x{:X},{},scratch_wave,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_param_count_wave(&m, CoreTrainMode::FullTrain),
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

        // From-scratch MLP on Domain B
        {
            let mut r_arm = root_rng.fork(0x31B + (n_few as u64) * 100);
            let fresh = MlpCore::new(&mut r_arm);
            let (m, ms, ok) = fit_mlp_target(&dom_b, tr_b, &fresh, CoreTrainMode::FullTrain, &mut r_arm);
            let ev = eval_mlp(&m, &dom_b, te_b, &mut r_arm.fork(0xEE));
            writeln!(
                t3_w,
                "0x{:X},{},scratch_mlp,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_param_count_mlp(&m, CoreTrainMode::FullTrain),
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
    }
    t3_w.flush()?;

    // =========================================================================
    // SUITE T4: Cross-Domain Transfer A -> C (Discrete Relational Graph Z5 x 4)
    // =========================================================================
    let mut t4_w = BufWriter::new(File::create(out_dir.join("metrics_t4.csv"))?);
    writeln!(
        t4_w,
        "seed,n_few,arm,waves_frozen,adapt_params,checksum_ok,cls_acc,macro_f1,nmi_unsup,ari_unsup,alg_consistency,abd_top1,psr,ghost,cfr,train_ms"
    )?;

    let mut trace_cal_c: Option<WaveModel> = None;

    for &n_few in &[30usize, 60, 120] {
        let tr_c = dom_c.train_indices(n_few);
        let te_c = dom_c.test_indices(n_few);

        let wave_arms: [(&str, &WaveCollisionCore, CoreTrainMode, u64); 9] = [
            ("transfer_autocal_A", &adapt_a.core, CoreTrainMode::AutoCalOnly, 0x410),
            ("transfer_frozen_A", &adapt_a.core, CoreTrainMode::FullyFrozen, 0x411),
            ("riemann_A", &riemann_a.core, CoreTrainMode::AutoCalOnly, 0x412),
            ("fourier_A", &fourier_a.core, CoreTrainMode::AutoCalOnly, 0x413),
            ("poisson_A", &poisson_a.core, CoreTrainMode::AutoCalOnly, 0x414),
            ("dyson_A", &dyson_a.core, CoreTrainMode::AutoCalOnly, 0x415),
            ("neg_ctrl_noise_A", &noise_a.core, CoreTrainMode::AutoCalOnly, 0x416),
            ("neg_ctrl_phase_scramble", &scrambled_a_core, CoreTrainMode::AutoCalOnly, 0x417),
            ("untrained_poisson", &untrained_poisson_core, CoreTrainMode::AutoCalOnly, 0x418),
        ];

        for &(arm, core_ref, mode, salt) in &wave_arms {
            let mut r_arm = root_rng.fork(salt + (n_few as u64) * 100);
            let (m, ms, ok) = fit_wave_target(&dom_c, tr_c, core_ref, mode, &mut r_arm);
            let ev = eval_wave(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            writeln!(
                t4_w,
                "0x{:X},{},{},1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                arm,
                adapt_param_count_wave(&m, mode),
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
            if write_trace && n_few == 60 && arm == "transfer_autocal_A" {
                trace_cal_c = Some(m);
            }
        }

        // Matched MLP transfer with exact +48 FiLM auto-calibration knobs (184 params!)
        {
            let mut r_arm = root_rng.fork(0x419 + (n_few as u64) * 100);
            let (m, ms, ok) = fit_mlp_target(&dom_c, tr_c, &mlp_a.core, CoreTrainMode::AutoCalOnly, &mut r_arm);
            let ev = eval_mlp(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            writeln!(
                t4_w,
                "0x{:X},{},transfer_mlp_A,1,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_param_count_mlp(&m, CoreTrainMode::AutoCalOnly),
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

        // From-scratch 768-wave model on Domain C
        {
            let mut r_arm = root_rng.fork(0x41A + (n_few as u64) * 100);
            let fresh = WaveCollisionCore::new(SpectrumKind::Adaptive, PairingMode::FullCollision, false, false, &mut r_arm);
            let (m, ms, ok) = fit_wave_target(&dom_c, tr_c, &fresh, CoreTrainMode::FullTrain, &mut r_arm);
            let ev = eval_wave(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            writeln!(
                t4_w,
                "0x{:X},{},scratch_wave,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_param_count_wave(&m, CoreTrainMode::FullTrain),
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

        // From-scratch MLP on Domain C
        {
            let mut r_arm = root_rng.fork(0x41B + (n_few as u64) * 100);
            let fresh = MlpCore::new(&mut r_arm);
            let (m, ms, ok) = fit_mlp_target(&dom_c, tr_c, &fresh, CoreTrainMode::FullTrain, &mut r_arm);
            let ev = eval_mlp(&m, &dom_c, te_c, &mut r_arm.fork(0xEE));
            writeln!(
                t4_w,
                "0x{:X},{},scratch_mlp,0,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
                seed,
                n_few,
                adapt_param_count_mlp(&m, CoreTrainMode::FullTrain),
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
    }
    t4_w.flush()?;

    if write_trace {
        let mut tr_w = BufWriter::new(File::create(out_dir.join("trace_wave_collision.csv"))?);
        writeln!(tr_w, "section,idx,val1,val2,val3,val4,val5,val6")?;
        for m in 0..N_COLLISIONS {
            let rad = |om: &[f64]| -> f64 {
                let s = &om[m * 4..(m + 1) * 4];
                s.iter().map(|x| x * x).sum::<f64>().sqrt()
            };
            writeln!(
                tr_w,
                "spectrum_mode,{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                m,
                rad(&init_adapt_omega1),
                rad(&adapt_a.core.omega1),
                rad(&riemann_a.core.omega1),
                rad(&fourier_a.core.omega1),
                init_adapt_phi1[m],
                adapt_a.core.phi1[m]
            )?;
        }
        if let Some(ref mc) = trace_cal_c {
            for g in 0..16 {
                writeln!(
                    tr_w,
                    "autocal_bank_c,{},{:.6},{:.6},{:.6},0,0,0",
                    g, mc.core.cal_scale[g], mc.core.cal_phase[g], mc.core.cal_rho[g]
                )?;
            }
        }
        tr_w.flush()?;
    }

    Ok(())
}
