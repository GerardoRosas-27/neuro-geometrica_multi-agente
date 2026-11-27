//! Suites P1..P4 for Prueba A (Validating the Nervous-System Wave Propagation Topology):
//!   - P1: Physical wave propagation diagnostics (causality, arrival time, speed, energy, localization, collision coincidence)
//!   - P2: Relational inference & multi-hop composition on Domain A & Domain C (6 topologies)
//!   - P3: Neurological resilience under 15%/30% axonal lesions and sigma=0.50 sensory noise (zero retraining)
//!   - P4: Cross-domain A -> C transfer with bit-exact frozen synapses + 64 nodal myelin calibration knobs

use crate::domains::{generate_domain_a, generate_domain_c, DomainData};
use crate::rng::Rng64;
use crate::topology::{TopologyKind, HUB_START, N_HUBS, N_NODES};
use crate::wave_net::{
    NervousWaveCore, NervousWaveModel, TrainConfig, WaveTrainMode, N_STEPS, REL_DIM,
};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct DomainEval {
    pub cls_acc: f64,
    pub reg_rmse: f64,
    pub abd_top1: f64,
    pub two_hop_acc: f64,
    pub alg_consistency: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WavePhysicsMetrics {
    pub e_hub_t0: f64,
    pub t_arrival: f64,
    pub mean_hop_dist: f64,
    pub e_final: f64,
    pub localization_snr: f64,
    pub collision_coinc: f64,
}

#[inline]
fn argmax(xs: &[f64]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

#[inline]
fn softmax_prob(logits: &[f64], target: usize) -> f64 {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut num = 0.0;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max_l).exp();
        sum += e;
        if i == target {
            num = e;
        }
    }
    num / sum.max(1e-15)
}

fn eval_model(
    model: &NervousWaveModel,
    domain: &DomainData,
    test_indices: &[usize],
    edge_mask: Option<&[bool]>,
) -> DomainEval {
    let n_ent = domain.n_entities;
    let n_pairs = domain.pairs.len();
    let mut all_reg: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
    let mut all_logits: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
    for p in &domain.pairs {
        let (reg, logits, _z) = model.predict_pair(&domain.obs[p.a], &domain.obs[p.b], edge_mask);
        all_reg.push(reg);
        all_logits.push(logits);
    }

    let mut is_test = vec![false; n_pairs];
    let mut ok = 0usize;
    let mut sq = 0.0;
    let mut rcnt = 0usize;
    for &idx in test_indices {
        is_test[idx] = true;
        let p = &domain.pairs[idx];
        if argmax(&all_logits[idx]) == p.cls {
            ok += 1;
        }
        for c in 0..domain.reg_dim {
            let e = all_reg[idx][c] - p.reg[c];
            sq += e * e;
            rcnt += 1;
        }
    }
    let cls_acc = ok as f64 / (test_indices.len().max(1) as f64);
    let reg_rmse = if rcnt > 0 { (sq / (rcnt as f64)).sqrt() } else { 0.0 };

    // Abduction Top-1
    let mut abd_ok = 0usize;
    let mut abd_tot = 0usize;
    if domain.reg_dim > 0 {
        let inv_2tau2 = 1.0 / (2.0 * 0.45 * 0.45);
        for &idx in test_indices {
            let a = domain.pairs[idx].a;
            let b = domain.pairs[idx].b;
            let y_obs = &domain.pairs[idx].reg;
            let mut best_k = usize::MAX;
            let mut best_s = f64::NEG_INFINITY;
            for k in 0..n_ent {
                if k == a {
                    continue;
                }
                let kidx = domain.pair_lookup[a * n_ent + k];
                let y_hat = &all_reg[kidx];
                let mut d2 = 0.0;
                for c in 0..domain.reg_dim {
                    let e = y_hat[c] - y_obs[c];
                    d2 += e * e;
                }
                let s = (-d2 * inv_2tau2).exp();
                if s > best_s {
                    best_s = s;
                    best_k = k;
                }
            }
            if best_k == b {
                abd_ok += 1;
            }
            abd_tot += 1;
        }
    } else {
        for u in 0..n_ent {
            for r_star in 0..domain.cls_dim {
                let mut cands = Vec::new();
                let mut has_true = false;
                for v in 0..n_ent {
                    if v == u {
                        continue;
                    }
                    let idx = domain.pair_lookup[u * n_ent + v];
                    if is_test[idx] {
                        cands.push(idx);
                        if domain.pairs[idx].cls == r_star {
                            has_true = true;
                        }
                    }
                }
                if !has_true || cands.len() < 2 {
                    continue;
                }
                let mut best_idx = 0usize;
                let mut best_p = f64::NEG_INFINITY;
                for &idx in &cands {
                    let p = softmax_prob(&all_logits[idx], r_star);
                    if p > best_p {
                        best_p = p;
                        best_idx = idx;
                    }
                }
                if domain.pairs[best_idx].cls == r_star {
                    abd_ok += 1;
                }
                abd_tot += 1;
            }
        }
    }
    let abd_top1 = abd_ok as f64 / (abd_tot.max(1) as f64);

    let two_hop_acc = if !domain.two_hop_queries.is_empty() {
        let ok2 = domain
            .two_hop_queries
            .iter()
            .filter(|q| argmax(&all_logits[q.pair_ac_idx]) == domain.pairs[q.pair_ac_idx].cls)
            .count();
        ok2 as f64 / (domain.two_hop_queries.len() as f64)
    } else {
        0.0
    };

    let alg_consistency = if !domain.node_tiers.is_empty() {
        let (mut s_ok, mut s_tot, mut i_ok, mut i_tot, mut c_ok, mut c_tot) = (0, 0, 0, 0, 0, 0);
        for &idx in test_indices {
            let u = domain.pairs[idx].a;
            let v = domain.pairs[idx].b;
            let rev = domain.pair_lookup[v * n_ent + u];
            let (tu, tv) = (domain.pairs[idx].cls, domain.pairs[rev].cls);
            let (pu, pv) = (argmax(&all_logits[idx]), argmax(&all_logits[rev]));
            if tu == 0 && tv == 0 {
                s_tot += 1;
                if pu == 0 && pv == 0 {
                    s_ok += 1;
                }
            }
            if tu == 1 && tv == 3 {
                i_tot += 1;
                if pu == 1 && pv == 3 {
                    i_ok += 1;
                }
            }
            if tu == 2 {
                c_tot += 1;
                if pu == 2 && pv != 2 {
                    c_ok += 1;
                }
            }
        }
        let (mut t_ok, mut t_tot) = (0, 0);
        for u in 0..n_ent {
            for v in 0..n_ent {
                if v == u {
                    continue;
                }
                let uv = domain.pair_lookup[u * n_ent + v];
                if domain.pairs[uv].cls != 1 {
                    continue;
                }
                for w in 0..n_ent {
                    if w == u || w == v {
                        continue;
                    }
                    let vw = domain.pair_lookup[v * n_ent + w];
                    let uw = domain.pair_lookup[u * n_ent + w];
                    if is_test[uw] && domain.pairs[vw].cls == 1 && domain.pairs[uw].cls == 1 {
                        t_tot += 1;
                        if argmax(&all_logits[uv]) == 1
                            && argmax(&all_logits[vw]) == 1
                            && argmax(&all_logits[uw]) == 1
                        {
                            t_ok += 1;
                        }
                    }
                }
            }
        }
        0.25 * (s_ok as f64 / s_tot.max(1) as f64
            + i_ok as f64 / i_tot.max(1) as f64
            + c_ok as f64 / c_tot.max(1) as f64
            + t_ok as f64 / t_tot.max(1) as f64)
    } else {
        0.0
    };

    DomainEval {
        cls_acc,
        reg_rmse,
        abd_top1,
        two_hop_acc,
        alg_consistency,
    }
}

/// Suite P1: Measure physical traveling-wave propagation on a trained model across 24 test pairs.
fn measure_wave_physics(
    model: &NervousWaveModel,
    domain: &DomainData,
    test_indices: &[usize],
) -> WavePhysicsMetrics {
    let n_probe = test_indices.len().min(24);
    let mut e_hub_t0_sum = 0.0;
    let mut t_arr_sum = 0.0;
    let mut e_final_sum = 0.0;
    let mut loc_snr_sum = 0.0;
    let mut coinc_sum = 0.0;

    let past_src: Vec<usize> = (0..8).collect();
    let fut_src: Vec<usize> = (8..16).collect();
    let mean_hop_dist = 0.5
        * (model.core.topo.mean_hop_distance_to_hubs(&past_src)
            + model.core.topo.mean_hop_distance_to_hubs(&fut_src));

    for &idx in &test_indices[..n_probe] {
        let p = &domain.pairs[idx];
        let u = model.encoder.forward(&domain.obs[p.a]);
        let v = model.encoder.forward(&domain.obs[p.b]);

        let c_p = model.core.forward_impulse(true, false, u, v);
        let c_f = model.core.forward_impulse(false, true, u, v);
        let c_pf = model.core.forward_impulse(true, true, u, v);

        // 1. Exact hub energy at t = 0 (causality check: must be strictly 0.0)
        let mut e0 = 0.0;
        for h in HUB_START..(HUB_START + N_HUBS) {
            e0 += c_pf.psi_re[0][h].powi(2) + c_pf.psi_im[0][h].powi(2);
        }
        e_hub_t0_sum += e0 / (N_HUBS as f64);

        // 2. Wavefront arrival time at convergent hubs (first step t where mean hub energy > 0.05)
        let mut t_arr = (N_STEPS + 1) as f64;
        for t in 0..=N_STEPS {
            let mut eh = 0.0;
            for h in HUB_START..(HUB_START + N_HUBS) {
                eh += c_pf.psi_re[t][h].powi(2) + c_pf.psi_im[t][h].powi(2);
            }
            if eh / (N_HUBS as f64) > 0.05 {
                t_arr = t as f64;
                break;
            }
        }
        t_arr_sum += t_arr;

        // 3. Final mean wave energy per node across the collision window t=N_STEPS
        let mut ef = 0.0;
        for i in 0..N_NODES {
            ef += c_pf.psi_re[N_STEPS][i].powi(2) + c_pf.psi_im[N_STEPS][i].powi(2);
        }
        e_final_sum += ef / (N_NODES as f64);

        // 4. Spatial localization / pathway SNR at t = 1 for Past-only pulse:
        //    energy in Past-receiving ganglia (G0/G1: 16..32) vs Future-side (G2/G3: 32..48) + hubs
        let mut e_on = 0.0;
        let mut e_off = 0.0;
        for i in 16..32 {
            e_on += c_p.psi_re[1][i].powi(2) + c_p.psi_im[1][i].powi(2);
        }
        for i in 32..64 {
            e_off += c_p.psi_re[1][i].powi(2) + c_p.psi_im[1][i].powi(2);
        }
        loc_snr_sum += (e_on / 16.0) / ((e_on / 16.0) + (e_off / 32.0) + 1e-9);

        // 5. Past x Future interferometric collision coincidence at convergent hubs (t = 6..=8):
        //    || |psi_PF|^2 - |psi_P|^2 - |psi_F|^2 ||_2 / (|| |psi_P|^2 + |psi_F|^2 ||_2 + eps)
        let mut num2 = 0.0;
        let mut den2 = 0.0;
        for t in 6..=N_STEPS {
            for h in HUB_START..(HUB_START + N_HUBS) {
                let ip = c_p.psi_re[t][h].powi(2) + c_p.psi_im[t][h].powi(2);
                let ifut = c_f.psi_re[t][h].powi(2) + c_f.psi_im[t][h].powi(2);
                let ipf = c_pf.psi_re[t][h].powi(2) + c_pf.psi_im[t][h].powi(2);
                let cross = ipf - ip - ifut;
                let base = ip + ifut;
                num2 += cross * cross;
                den2 += base * base;
            }
        }
        coinc_sum += num2.sqrt() / (den2.sqrt() + 1e-9);
    }

    let nf = n_probe.max(1) as f64;
    WavePhysicsMetrics {
        e_hub_t0: e_hub_t0_sum / nf,
        t_arrival: t_arr_sum / nf,
        mean_hop_dist,
        e_final: e_final_sum / nf,
        localization_snr: loc_snr_sum / nf,
        collision_coinc: coinc_sum / nf,
    }
}

fn domain_a_cfg() -> TrainConfig {
    TrainConfig {
        steps: 650,
        batch_size: 48,
        lr_encoder: 0.018,
        lr_core: 0.015,
        lr_myelin: 0.0,
        lr_head: 0.018,
        wd_encoder: 8e-4,
        wd_core: 2.0e-3,
        wd_myelin: 0.0,
        wd_head: 3.0e-3,
        w_mse: 1.0,
        w_ce: 1.1,
        latent_norm_reg: 0.04,
        latent_jitter: 0.05,
        mode: WaveTrainMode::FullTrain,
    }
}

fn domain_c_scratch_cfg() -> TrainConfig {
    TrainConfig {
        steps: 500,
        batch_size: 32,
        lr_encoder: 0.020,
        lr_core: 0.015,
        lr_myelin: 0.0,
        lr_head: 0.020,
        wd_encoder: 2.0e-3,
        wd_core: 2.0e-3,
        wd_myelin: 0.0,
        wd_head: 4.0e-3,
        w_mse: 0.0,
        w_ce: 1.0,
        latent_norm_reg: 0.04,
        latent_jitter: 0.03,
        mode: WaveTrainMode::FullTrain,
    }
}

fn transfer_myelin_cfg() -> TrainConfig {
    TrainConfig {
        steps: 360,
        batch_size: 32,
        lr_encoder: 0.022,
        lr_core: 0.0,
        lr_myelin: 0.040,
        lr_head: 0.022,
        wd_encoder: 1.5e-2,
        wd_core: 0.0,
        wd_myelin: 0.60,
        wd_head: 2.0e-2,
        w_mse: 0.0,
        w_ce: 1.0,
        latent_norm_reg: 0.05,
        latent_jitter: 0.0,
        mode: WaveTrainMode::MyelinCalOnly,
    }
}

fn model_train_loss(
    model: &NervousWaveModel,
    domain: &DomainData,
    train_idx: &[usize],
    cfg: &TrainConfig,
) -> f64 {
    let mut tot = 0.0;
    for &idx in train_idx {
        let p = &domain.pairs[idx];
        let (_reg, logits, _) = model.predict_pair(&domain.obs[p.a], &domain.obs[p.b], None);
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_e: f64 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let prob = ((logits[p.cls] - max_l).exp() / sum_e).max(1e-15);
        tot += -prob.ln();
    }
    let reg_pen = 0.5 * cfg.wd_encoder * model.encoder.w.iter().map(|x| x * x).sum::<f64>()
        + 0.5 * cfg.wd_head * model.head.w_cls.iter().map(|x| x * x).sum::<f64>();
    tot / (train_idx.len().max(1) as f64) + reg_pen
}

fn fit_myelin_transfer(
    domain: &DomainData,
    train_idx: &[usize],
    base_core: &NervousWaveCore,
    rng: &mut Rng64,
) -> (NervousWaveModel, f64, bool) {
    let cfg = transfer_myelin_cfg();
    let pre_hash = base_core.frozen_synapses_checksum();
    let t0 = Instant::now();
    let mut best: Option<NervousWaveModel> = None;
    let mut best_loss = f64::INFINITY;
    for r in 0..2 {
        let mut r_rng = rng.fork(0x6600 + r as u64);
        let mut c = base_core.clone();
        c.reset_myelin();
        let mut m = NervousWaveModel::new(domain.obs_dim, domain.reg_dim, domain.cls_dim, c, &mut r_rng);
        m.train(domain, train_idx, &cfg, &mut r_rng);
        let l = model_train_loss(&m, domain, train_idx, &cfg);
        if l < best_loss {
            best_loss = l;
            best = Some(m);
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let m = best.unwrap();
    let ok = m.core.frozen_synapses_checksum() == pre_hash;
    (m, ms, ok)
}

pub fn run_all_suites_for_seed(seed: u64, out_dir: &Path, write_trace: bool) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;
    let root_rng = Rng64::new(seed);
    let dom_a = generate_domain_a(&mut root_rng.fork(0xA001));
    let dom_c = generate_domain_c(&mut root_rng.fork(0xC003));
    let dom_a_noisy = dom_a.with_sensory_noise(0.50, &mut root_rng.fork(0xA555));

    let train_a = dom_a.train_indices(dom_a.default_train_n);
    let test_a = dom_a.test_indices(dom_a.default_train_n);
    let train_c120 = dom_c.train_indices(120);
    let test_c120 = dom_c.test_indices(120);
    let train_c60 = dom_c.train_indices(60);
    let test_c60 = dom_c.test_indices(60);

    let cfg_a = domain_a_cfg();
    let cfg_c = domain_c_scratch_cfg();

    let arms: [(TopologyKind, u64); 6] = [
        (TopologyKind::NervousSystem, 0x101),
        (TopologyKind::GridLocalLattice, 0x102),
        (TopologyKind::RandomErdosRenyi, 0x103),
        (TopologyKind::AllToAllFlat, 0x104),
        (TopologyKind::NervousNoDelay, 0x105),
        (TopologyKind::NervousNoInhibition, 0x106),
    ];

    let mut p1_w = BufWriter::new(File::create(out_dir.join("metrics_p1_physics.csv"))?);
    writeln!(
        p1_w,
        "seed,arm,n_edges,e_hub_t0,t_arrival,mean_hop_dist,e_final,localization_snr,collision_coinc"
    )?;

    let mut p2_w = BufWriter::new(File::create(out_dir.join("metrics_p2_inference.csv"))?);
    writeln!(
        p2_w,
        "seed,arm,n_edges,acc_a,rmse_a,abd_a,two_hop_a,acc_c,cons_c,abd_c,train_ms_a,train_ms_c"
    )?;

    let mut p3_w = BufWriter::new(File::create(out_dir.join("metrics_p3_resilience.csv"))?);
    writeln!(
        p3_w,
        "seed,arm,n_edges,acc_clean,acc_lesion15,ret_lesion15,acc_lesion30,ret_lesion30,acc_noise50,ret_noise50"
    )?;

    let mut trained_a: Vec<(TopologyKind, NervousWaveModel)> = Vec::new();

    for &(kind, salt) in &arms {
        // Train on Domain A
        let mut r_a = root_rng.fork(salt);
        let core_a = NervousWaveCore::new(kind, &mut r_a);
        let mut model_a = NervousWaveModel::new(dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, core_a, &mut r_a);
        let t0_a = Instant::now();
        model_a.train(&dom_a, train_a, &cfg_a, &mut r_a);
        let ms_a = t0_a.elapsed().as_secs_f64() * 1000.0;
        let ev_a = eval_model(&model_a, &dom_a, test_a, None);

        // P1: Physical Wave Propagation Diagnostics
        let phys = measure_wave_physics(&model_a, &dom_a, test_a);
        writeln!(
            p1_w,
            "0x{:X},{},{},{:.8},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            kind.name(),
            model_a.core.edge_count(),
            phys.e_hub_t0,
            phys.t_arrival,
            phys.mean_hop_dist,
            phys.e_final,
            phys.localization_snr,
            phys.collision_coinc
        )?;

        // Train from scratch on Domain C (N=120)
        let mut r_c = root_rng.fork(salt + 0x500);
        let core_c = NervousWaveCore::new(kind, &mut r_c);
        let mut model_c = NervousWaveModel::new(dom_c.obs_dim, dom_c.reg_dim, dom_c.cls_dim, core_c, &mut r_c);
        let t0_c = Instant::now();
        model_c.train(&dom_c, train_c120, &cfg_c, &mut r_c);
        let ms_c = t0_c.elapsed().as_secs_f64() * 1000.0;
        let ev_c = eval_model(&model_c, &dom_c, test_c120, None);

        writeln!(
            p2_w,
            "0x{:X},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2},{:.2}",
            seed,
            kind.name(),
            model_a.core.edge_count(),
            ev_a.cls_acc,
            ev_a.reg_rmse,
            ev_a.abd_top1,
            ev_a.two_hop_acc,
            ev_c.cls_acc,
            ev_c.alg_consistency,
            ev_c.abd_top1,
            ms_a,
            ms_c
        )?;

        // P3: Neurological Resilience (8 deterministic lesion masks @ 15% & 30%, + sigma=0.50 sensory noise)
        let mut acc_l15 = 0.0;
        let mut acc_l30 = 0.0;
        for m_i in 0..8u64 {
            let mk15 = model_a.core.topo.with_lesion(0.15, &mut root_rng.fork(0x3100 + salt * 17 + m_i));
            acc_l15 += eval_model(&model_a, &dom_a, test_a, Some(&mk15)).cls_acc;
            let mk30 = model_a.core.topo.with_lesion(0.30, &mut root_rng.fork(0x3200 + salt * 17 + m_i));
            acc_l30 += eval_model(&model_a, &dom_a, test_a, Some(&mk30)).cls_acc;
        }
        acc_l15 /= 8.0;
        acc_l30 /= 8.0;
        let acc_noise = eval_model(&model_a, &dom_a_noisy, test_a, None).cls_acc;
        let clean = ev_a.cls_acc.max(1e-9);
        writeln!(
            p3_w,
            "0x{:X},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            kind.name(),
            model_a.core.edge_count(),
            ev_a.cls_acc,
            acc_l15,
            acc_l15 / clean,
            acc_l30,
            acc_l30 / clean,
            acc_noise,
            acc_noise / clean
        )?;

        trained_a.push((kind, model_a));
    }
    p1_w.flush()?;
    p2_w.flush()?;
    p3_w.flush()?;

    // =========================================================================
    // SUITE P4: Cross-Domain A -> C Myelin Auto-Calibration Transfer (N_few = 60)
    // =========================================================================
    let mut p4_w = BufWriter::new(File::create(out_dir.join("metrics_p4_transfer.csv"))?);
    writeln!(
        p4_w,
        "seed,arm,n_edges,myelin_knobs,checksum_ok,cls_acc,alg_consistency,abd_top1,train_ms"
    )?;

    let nerv_perm_core = trained_a[0]
        .1
        .core
        .with_permuted_synapses(&mut root_rng.fork(0x4999));

    let transfer_arms: [(&str, &NervousWaveCore, u64); 5] = [
        ("nervous_transfer", &trained_a[0].1.core, 0x410),
        ("grid_transfer", &trained_a[1].1.core, 0x411),
        ("random_transfer", &trained_a[2].1.core, 0x412),
        ("flat_transfer", &trained_a[3].1.core, 0x413),
        ("ctrl_permuted_synapses", &nerv_perm_core, 0x414),
    ];

    for &(arm_name, base_core, salt) in &transfer_arms {
        let mut r_tr = root_rng.fork(salt);
        let (m, ms, ok) = fit_myelin_transfer(&dom_c, train_c60, base_core, &mut r_tr);
        let ev = eval_model(&m, &dom_c, test_c60, None);
        writeln!(
            p4_w,
            "0x{:X},{},{},{},{},{:.6},{:.6},{:.6},{:.2}",
            seed,
            arm_name,
            m.core.edge_count(),
            m.core.autocal_param_count(),
            if ok { 1 } else { 0 },
            ev.cls_acc,
            ev.alg_consistency,
            ev.abd_top1,
            ms
        )?;
    }
    p4_w.flush()?;

    // Write full spacetime wave-propagation snapshot for visualization on trace seed
    if write_trace {
        let mut tr_w = BufWriter::new(File::create(out_dir.join("trace_nervous_wave.csv"))?);
        writeln!(tr_w, "arm,t,node,x,y,amp,phase,tier")?;
        let probe_pair = &dom_a.pairs[test_a[0]];
        for idx_arm in [0usize, 1, 2, 3] {
            let (kind, ref model) = trained_a[idx_arm];
            let u = model.encoder.forward(&dom_a.obs[probe_pair.a]);
            let v = model.encoder.forward(&dom_a.obs[probe_pair.b]);
            let cache = model.core.forward(u, v);
            for t in 0..=N_STEPS {
                for i in 0..N_NODES {
                    let pr = cache.psi_re[t][i];
                    let pi = cache.psi_im[t][i];
                    let amp = (pr * pr + pi * pi).sqrt();
                    let phase = pi.atan2(pr);
                    let (x, y) = model.core.topo.coords[i];
                    let tier = if i < 8 {
                        0 // past receptor
                    } else if i < 16 {
                        1 // future receptor
                    } else if i < 48 {
                        2 // ganglion interneuron
                    } else {
                        3 // convergent hub
                    };
                    writeln!(
                        tr_w,
                        "{},{},{},{:.4},{:.4},{:.6},{:.6},{}",
                        kind.name(),
                        t,
                        i,
                        x,
                        y,
                        amp,
                        phase,
                        tier
                    )?;
                }
            }
        }
        // Also verify readout dimension constant used
        let _ = REL_DIM;
        tr_w.flush()?;
    }

    Ok(())
}
