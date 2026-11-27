//! Suites B1..B4 for Prueba B (Scaled 128-node/1024-edge Nervous-System Wave Topology
//! with Kuramoto Phase-Coherence Selective Lateral Inhibition & Resonant Spring Transfer).

use crate::domains::{generate_domain_a, generate_domain_b, generate_domain_c, DomainData};
use crate::rng::Rng64;
use crate::topology::{GraphTopology, TopologyKind};
use crate::wave_net::{NervousWaveNet, Perturbation, StepParamCache, TrainMode, READOUT_START, T_STEPS};
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
    pub hub_r_coh: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PhysicsMetrics {
    pub mean_hops: f64,
    pub hub_energy_t0: f64,
    pub hub_energy_t1: f64,
    pub arrival_time: f64,
    pub final_hub_energy: f64,
    pub localization_snr: f64,
    pub coincidence: f64,
    pub hub_r_coh: f64,
    pub hub_d_incoh: f64,
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

fn eval_domain(
    net: &NervousWaveNet,
    domain: &DomainData,
    test_indices: &[usize],
    pert: &Perturbation,
    obs_noise_std: f64,
    noise_rng: Option<&mut Rng64>,
) -> DomainEval {
    let n_ent = domain.n_entities;
    let n_pairs = domain.pairs.len();

    let noisy_obs: Vec<Vec<f64>> = if obs_noise_std > 0.0 {
        let rng = noise_rng.expect("rng required when obs_noise_std > 0");
        domain
            .obs
            .iter()
            .map(|row| row.iter().map(|&x| x + obs_noise_std * rng.normal()).collect())
            .collect()
    } else {
        Vec::new()
    };
    let obs_ref: &[Vec<f64>] = if obs_noise_std > 0.0 {
        &noisy_obs
    } else {
        &domain.obs
    };

    let mut all_reg: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
    let mut all_logits: Vec<Vec<f64>> = Vec::with_capacity(n_pairs);
    let mut r_coh_sum = 0.0;
    let cache = StepParamCache::build(net, pert);
    for p in &domain.pairs {
        let tr = net.forward_cached(&obs_ref[p.a], &obs_ref[p.b], &cache);
        r_coh_sum += tr.hub_r_coh;
        all_reg.push(tr.reg_pred);
        all_logits.push(tr.cls_logits);
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
    let reg_rmse = if rcnt > 0 {
        (sq / rcnt as f64).sqrt()
    } else {
        0.0
    };

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
        ok2 as f64 / domain.two_hop_queries.len() as f64
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
        hub_r_coh: r_coh_sum / n_pairs.max(1) as f64,
    }
}

/// Suite B1: Physical traveling-wave propagation, causality, isolation, collision, & Kuramoto coherence.
pub fn measure_physics(
    net: &NervousWaveNet,
    domain: &DomainData,
    test_indices: &[usize],
) -> PhysicsMetrics {
    let n_probe = test_indices.len().min(32);
    let pert = Perturbation::none();
    let cache = StepParamCache::build(net, &pert);

    let mut e_t0_sum = 0.0;
    let mut e_t1_sum = 0.0;
    let mut t_arr_sum = 0.0;
    let mut e_fin_sum = 0.0;
    let mut loc_snr_sum = 0.0;
    let mut coinc_sum = 0.0;
    let mut r_coh_sum = 0.0;
    let mut d_inc_sum = 0.0;

    for &idx in &test_indices[..n_probe] {
        let p = &domain.pairs[idx];
        let oa = &domain.obs[p.a];
        let ob = &domain.obs[p.b];

        let tr_both = net.forward_masked(oa, ob, &cache, true, true);
        let tr_past = net.forward_masked(oa, ob, &cache, true, false);
        let tr_fut = net.forward_masked(oa, ob, &cache, false, true);

        let hub_e = |t: usize| -> f64 {
            let mut s = 0.0;
            for &h in &net.topo.hubs {
                let pr = tr_both.psi_re[t][h];
                let pi = tr_both.psi_im[t][h];
                s += pr * pr + pi * pi;
            }
            s / net.topo.hubs.len() as f64
        };

        let e0 = hub_e(0);
        let e1 = hub_e(1);
        let efin = hub_e(T_STEPS);
        e_t0_sum += e0;
        e_t1_sum += e1;
        e_fin_sum += efin;

        let mut peak = 0.0;
        let mut energies = [0.0; T_STEPS + 1];
        for t in 0..=T_STEPS {
            energies[t] = hub_e(t);
            if energies[t] > peak {
                peak = energies[t];
            }
        }
        let thresh = 0.10 * peak.max(1e-12);
        let mut t_arr = T_STEPS as f64;
        for t in 1..=T_STEPS {
            if energies[t] >= thresh {
                t_arr = t as f64;
                break;
            }
        }
        t_arr_sum += t_arr;

        // Spatial isolation SNR at t = 1 under Past-only stimulus
        let mut ep_past = 0.0;
        let mut ep_fut = 0.0;
        for &g in &net.topo.past_ganglia {
            let pr = tr_past.psi_re[1][g];
            let pi = tr_past.psi_im[1][g];
            ep_past += pr * pr + pi * pi;
        }
        for &g in &net.topo.future_ganglia {
            let pr = tr_past.psi_re[1][g];
            let pi = tr_past.psi_im[1][g];
            ep_fut += pr * pr + pi * pi;
        }
        loc_snr_sum += ep_past / (ep_past + ep_fut + 1e-12);

        // Past-Future collision coincidence at hubs over readout window:
        // geometric mean of hub amplitude from Past-only and Future-only waves normalized by total
        let mut co_num = 0.0;
        let mut co_den = 0.0;
        for t in READOUT_START..=T_STEPS {
            for &h in &net.topo.hubs {
                let ap = (tr_past.psi_re[t][h].powi(2) + tr_past.psi_im[t][h].powi(2)).sqrt();
                let af = (tr_fut.psi_re[t][h].powi(2) + tr_fut.psi_im[t][h].powi(2)).sqrt();
                co_num += 2.0 * ap * af;
                co_den += ap * ap + af * af;
            }
        }
        coinc_sum += co_num / (co_den + 1e-12);

        r_coh_sum += tr_both.hub_r_coh;

        let mut d_sum = 0.0;
        let mut d_cnt = 0usize;
        for t in (READOUT_START - 1)..T_STEPS {
            for &h in &net.topo.hubs {
                d_sum += tr_both.d_incoh[t][h];
                d_cnt += 1;
            }
        }
        d_inc_sum += d_sum / d_cnt.max(1) as f64;
    }

    let inv = 1.0 / n_probe.max(1) as f64;
    PhysicsMetrics {
        mean_hops: net.topo.mean_receptor_to_hub_hops(),
        hub_energy_t0: e_t0_sum * inv,
        hub_energy_t1: e_t1_sum * inv,
        arrival_time: t_arr_sum * inv,
        final_hub_energy: e_fin_sum * inv,
        localization_snr: loc_snr_sum * inv,
        coincidence: coinc_sum * inv,
        hub_r_coh: r_coh_sum * inv,
        hub_d_incoh: d_inc_sum * inv,
    }
}

/// Apply phase noise perturbation; for phase-disabled no_delay arm, apply real projection cos(noise).
fn build_phase_noise_pert(net: &NervousWaveNet, sigma: f64, rng: &mut Rng64) -> Perturbation {
    let e_len = net.topo.edges.len();
    let mut pnoise = Vec::with_capacity(e_len);
    let mut mask = Vec::new();
    for _ in 0..e_len {
        let eps = sigma * rng.normal();
        pnoise.push(eps);
    }
    if !net.phase_enabled {
        // Real-axis projection of e^{i * eps} = cos(eps) so no_delay experiences exact real-projected noise
        mask = pnoise.iter().map(|&eps| eps.cos()).collect();
        pnoise.clear();
    }
    Perturbation {
        edge_mask: mask,
        phase_noise: pnoise,
    }
}

fn build_lesion_pert(e_len: usize, drop_frac: f64, rng: &mut Rng64) -> Perturbation {
    let n_drop = ((e_len as f64) * drop_frac).round() as usize;
    let mut order: Vec<usize> = (0..e_len).collect();
    rng.shuffle(&mut order);
    let mut mask = vec![1.0; e_len];
    for &ei in &order[..n_drop.min(e_len)] {
        mask[ei] = 0.0;
    }
    Perturbation {
        edge_mask: mask,
        phase_noise: Vec::new(),
    }
}

pub fn run_seed(seed: u64, split: &str, out_dir: &Path) -> std::io::Result<()> {
    let t0 = Instant::now();
    fs::create_dir_all(out_dir)?;

    let mut dom_a_rng = Rng64::new(seed ^ 0xA11A_0001);
    let mut dom_b_rng = Rng64::new(seed ^ 0xB22B_0002);
    let mut dom_c_rng = Rng64::new(seed ^ 0xC33C_0003);
    let dom_a = generate_domain_a(&mut dom_a_rng);
    let dom_b = generate_domain_b(&mut dom_b_rng);
    let dom_c = generate_domain_c(&mut dom_c_rng);

    let a_train = dom_a.train_indices(dom_a.default_train_n);
    let a_test = dom_a.test_indices(dom_a.default_train_n);
    let c_train = dom_c.train_indices(dom_c.default_train_n);
    let c_test = dom_c.test_indices(dom_c.default_train_n);

    let mut f_b1 = BufWriter::new(File::create(out_dir.join("b1_physics.csv"))?);
    writeln!(
        f_b1,
        "seed,arm,n_nodes,n_edges,mean_hops,hub_energy_t0,hub_energy_t1,arrival_time,final_hub_energy,localization_snr,coincidence,hub_r_coh,hub_d_incoh"
    )?;

    let mut f_b2 = BufWriter::new(File::create(out_dir.join("b2_inference.csv"))?);
    writeln!(
        f_b2,
        "seed,arm,n_nodes,n_edges,domA_acc,domA_rmse,domA_abduction,domA_twohop,domC_acc,domC_consistency,domA_hub_rcoh"
    )?;

    let mut f_b3 = BufWriter::new(File::create(out_dir.join("b3_resilience.csv"))?);
    writeln!(
        f_b3,
        "seed,arm,n_nodes,n_edges,clean_acc,lesion15_acc,lesion15_ret,lesion30_acc,lesion30_ret,noise50_acc,noise50_ret,noise50_rcoh,sens_noise50_acc"
    )?;

    let mut f_b4 = BufWriter::new(File::create(out_dir.join("b4_transfer.csv"))?);
    writeln!(
        f_b4,
        "seed,arm,target_domain,n_train,n_nodes,n_edges,frozen_ok,acc,rmse,consistency,hub_r_coh"
    )?;

    let mut trained_dom_a: Vec<(TopologyKind, NervousWaveNet)> = Vec::new();
    let pert_none = Perturbation::none();

    for (arm_idx, &kind) in TopologyKind::all().iter().enumerate() {
        let topo = GraphTopology::build(kind, seed);
        let n_nodes = topo.n_nodes;
        let n_edges = topo.edges.len();

        // Train on Domain A (130 epochs, batch 24, lr 0.015)
        let mut init_rng = Rng64::new(seed ^ 0x1000_0000 ^ (arm_idx as u64 * 0x9E37_79B9));
        let mut net_a = NervousWaveNet::new(
            topo.clone(),
            dom_a.obs_dim,
            dom_a.reg_dim,
            dom_a.cls_dim,
            &mut init_rng,
        );
        let mut tr_rng = Rng64::new(seed ^ 0x2000_0000 ^ (arm_idx as u64 * 0x517C_C1B7));
        net_a.train(&dom_a, a_train, TrainMode::Full, 150, 24, 0.018, 1.0, 1.0, &mut tr_rng);

        // Suite B1: Physics on trained Domain A network
        let phys = measure_physics(&net_a, &dom_a, a_test);
        writeln!(
            f_b1,
            "{},{},{},{},{:.6},{:.8},{:.8},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            kind.name(),
            n_nodes,
            n_edges,
            phys.mean_hops,
            phys.hub_energy_t0,
            phys.hub_energy_t1,
            phys.arrival_time,
            phys.final_hub_energy,
            phys.localization_snr,
            phys.coincidence,
            phys.hub_r_coh,
            phys.hub_d_incoh
        )?;

        // Suite B2: Evaluate Domain A & train/evaluate scratch Domain C
        let eval_a = eval_domain(&net_a, &dom_a, a_test, &pert_none, 0.0, None);

        let mut init_c_rng = Rng64::new(seed ^ 0x3000_0000 ^ (arm_idx as u64 * 0x9E37_79B9));
        let mut net_c = NervousWaveNet::new(
            topo.clone(),
            dom_c.obs_dim,
            dom_c.reg_dim,
            dom_c.cls_dim,
            &mut init_c_rng,
        );
        let mut tr_c_rng = Rng64::new(seed ^ 0x4000_0000 ^ (arm_idx as u64 * 0x517C_C1B7));
        net_c.train(&dom_c, c_train, TrainMode::Full, 150, 20, 0.018, 0.0, 1.0, &mut tr_c_rng);
        let eval_c = eval_domain(&net_c, &dom_c, c_test, &pert_none, 0.0, None);

        writeln!(
            f_b2,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            kind.name(),
            n_nodes,
            n_edges,
            eval_a.cls_acc,
            eval_a.reg_rmse,
            eval_a.abd_top1,
            eval_a.two_hop_acc,
            eval_c.cls_acc,
            eval_c.alg_consistency,
            eval_a.hub_r_coh
        )?;

        // Suite B3: Zero-retraining Resilience (6 masks/realizations each)
        let n_trials = 6usize;
        let mut l15_sum = 0.0;
        let mut l30_sum = 0.0;
        let mut pn50_sum = 0.0;
        let mut pn50_rcoh_sum = 0.0;
        let mut sn50_sum = 0.0;

        for r in 0..n_trials {
            let mut r15 = Rng64::new(seed ^ 0x5100_0000 ^ (r as u64 * 0x1337) ^ (arm_idx as u64 * 7));
            let p15 = build_lesion_pert(n_edges, 0.15, &mut r15);
            l15_sum += eval_domain(&net_a, &dom_a, a_test, &p15, 0.0, None).cls_acc;

            let mut r30 = Rng64::new(seed ^ 0x5200_0000 ^ (r as u64 * 0x1337) ^ (arm_idx as u64 * 7));
            let p30 = build_lesion_pert(n_edges, 0.30, &mut r30);
            l30_sum += eval_domain(&net_a, &dom_a, a_test, &p30, 0.0, None).cls_acc;

            let mut rpn = Rng64::new(seed ^ 0x5300_0000 ^ (r as u64 * 0x1337) ^ (arm_idx as u64 * 7));
            let ppn = build_phase_noise_pert(&net_a, 0.50, &mut rpn);
            let ev_pn = eval_domain(&net_a, &dom_a, a_test, &ppn, 0.0, None);
            pn50_sum += ev_pn.cls_acc;
            pn50_rcoh_sum += ev_pn.hub_r_coh;

            let mut rsn = Rng64::new(seed ^ 0x5400_0000 ^ (r as u64 * 0x1337));
            sn50_sum += eval_domain(&net_a, &dom_a, a_test, &pert_none, 0.50, Some(&mut rsn)).cls_acc;
        }

        let l15 = l15_sum / n_trials as f64;
        let l30 = l30_sum / n_trials as f64;
        let pn50 = pn50_sum / n_trials as f64;
        let pn50_rcoh = pn50_rcoh_sum / n_trials as f64;
        let sn50 = sn50_sum / n_trials as f64;
        let clean = eval_a.cls_acc.max(1e-9);

        writeln!(
            f_b3,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            seed,
            kind.name(),
            n_nodes,
            n_edges,
            eval_a.cls_acc,
            l15,
            l15 / clean,
            l30,
            l30 / clean,
            pn50,
            pn50 / clean,
            pn50_rcoh,
            sn50
        )?;

        trained_dom_a.push((kind, net_a));
    }

    // Suite B4: Frozen-synapse Resonant Spring Auto-Calibration Transfer (A -> B and A -> C)
    let mut frozen_all_ok = true;
    let b_train_n = 48usize;
    let c_train_n = 60usize;
    let b_train = dom_b.train_indices(b_train_n);
    let b_test = dom_b.test_indices(b_train_n);
    let c_tr_train = dom_c.train_indices(c_train_n);
    let c_tr_test = dom_c.test_indices(c_train_n);

    let transfer_specs: [(&str, TopologyKind, bool); 5] = [
        ("nervous_transfer", TopologyKind::NervousCoherenceB, false),
        ("grid_transfer", TopologyKind::GridLocal128, false),
        ("random_transfer", TopologyKind::RandomEr128, false),
        ("flat_transfer", TopologyKind::FlatDense4096, false),
        ("ctrl_permuted_synapses", TopologyKind::NervousCoherenceB, true),
    ];

    for (t_idx, &(t_name, src_kind, permute)) in transfer_specs.iter().enumerate() {
        let base_net = &trained_dom_a
            .iter()
            .find(|(k, _)| *k == src_kind)
            .expect("base arm must exist")
            .1;

        // A -> B (continuous oscillators R^12, N=48)
        let mut init_rng_b = Rng64::new(seed ^ 0x6100_0000 ^ (t_idx as u64 * 0x9E37_79B9));
        let mut net_ab = base_net.clone_for_transfer(
            dom_b.obs_dim,
            dom_b.reg_dim,
            dom_b.cls_dim,
            permute,
            &mut init_rng_b,
        );
        let ck_b_before = net_ab.frozen_tissue_checksum();
        let mut tr_rng_b = Rng64::new(seed ^ 0x6200_0000 ^ (t_idx as u64 * 0x517C_C1B7));
        net_ab.train(
            &dom_b,
            b_train,
            TrainMode::ResonantCalOnly,
            140,
            16,
            0.018,
            1.0,
            1.0,
            &mut tr_rng_b,
        );
        let ck_b_after = net_ab.frozen_tissue_checksum();
        let ok_b = ck_b_before == ck_b_after;
        if !ok_b {
            frozen_all_ok = false;
        }
        let ev_ab = eval_domain(&net_ab, &dom_b, b_test, &pert_none, 0.0, None);
        writeln!(
            f_b4,
            "{},{},domain_b_oscillators,{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            seed,
            t_name,
            b_train_n,
            net_ab.topo.n_nodes,
            net_ab.topo.edges.len(),
            if ok_b { 1 } else { 0 },
            ev_ab.cls_acc,
            ev_ab.reg_rmse,
            ev_ab.alg_consistency,
            ev_ab.hub_r_coh
        )?;

        // A -> C (discrete relational graph Z_5 x {0..3}, N=60)
        let mut init_rng_c = Rng64::new(seed ^ 0x7100_0000 ^ (t_idx as u64 * 0x9E37_79B9));
        let mut net_ac = base_net.clone_for_transfer(
            dom_c.obs_dim,
            dom_c.reg_dim,
            dom_c.cls_dim,
            permute,
            &mut init_rng_c,
        );
        let ck_c_before = net_ac.frozen_tissue_checksum();
        let mut tr_rng_c = Rng64::new(seed ^ 0x7200_0000 ^ (t_idx as u64 * 0x517C_C1B7));
        net_ac.train(
            &dom_c,
            c_tr_train,
            TrainMode::ResonantCalOnly,
            140,
            16,
            0.018,
            0.0,
            1.0,
            &mut tr_rng_c,
        );
        let ck_c_after = net_ac.frozen_tissue_checksum();
        let ok_c = ck_c_before == ck_c_after;
        if !ok_c {
            frozen_all_ok = false;
        }
        let ev_ac = eval_domain(&net_ac, &dom_c, c_tr_test, &pert_none, 0.0, None);
        writeln!(
            f_b4,
            "{},{},domain_c_graph,{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            seed,
            t_name,
            c_train_n,
            net_ac.topo.n_nodes,
            net_ac.topo.edges.len(),
            if ok_c { 1 } else { 0 },
            ev_ac.cls_acc,
            ev_ac.reg_rmse,
            ev_ac.alg_consistency,
            ev_ac.hub_r_coh
        )?;
    }

    f_b1.flush()?;
    f_b2.flush()?;
    f_b3.flush()?;
    f_b4.flush()?;

    let elapsed_ms = t0.elapsed().as_millis();
    let mut f_meta = File::create(out_dir.join("seed_meta.json"))?;
    writeln!(
        f_meta,
        "{{\n  \"seed\": {},\n  \"seed_hex\": \"0x{:X}\",\n  \"split\": \"{}\",\n  \"elapsed_ms\": {},\n  \"frozen_all_ok\": {}\n}}",
        seed, seed, split, elapsed_ms, frozen_all_ok
    )?;

    Ok(())
}
