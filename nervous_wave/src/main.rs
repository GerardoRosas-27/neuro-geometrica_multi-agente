mod domains;
mod rng;
mod suites;
mod topology;
mod wave_net;

use std::env;
use std::path::PathBuf;

fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("Invalid hex seed")
    } else {
        s.parse::<u64>().expect("Invalid decimal seed")
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: nervous_wave run-seed --seed <0xHEX|DEC> --out <DIR> [--trace]");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "run-seed" => {
            let mut seed: Option<u64> = None;
            let mut out: Option<PathBuf> = None;
            let mut trace = false;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--seed" => {
                        i += 1;
                        seed = Some(parse_seed(&args[i]));
                    }
                    "--out" => {
                        i += 1;
                        out = Some(PathBuf::from(&args[i]));
                    }
                    "--trace" => {
                        trace = true;
                    }
                    other => {
                        eprintln!("Unknown flag: {other}");
                        std::process::exit(1);
                    }
                }
                i += 1;
            }
            let s = seed.expect("--seed is required");
            let o = out.expect("--out is required");
            suites::run_all_suites_for_seed(s, &o, trace).expect("Suite execution failed");
        }
        other => {
            eprintln!("Unknown subcommand: {other}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::domains::generate_domain_c;
    use crate::rng::Rng64;
    use crate::topology::{TopologyKind, WaveTopology, HUB_START, N_HUBS, N_NODES};
    use crate::wave_net::{
        NervousWaveCore, NervousWaveGrads, NervousWaveModel, TrainConfig, WaveTrainMode, LATENT_D,
        REL_DIM,
    };

    #[test]
    fn test_exact_topology_edge_counts_and_hop_distances() {
        let mut rng = Rng64::new(42);
        let nerv = WaveTopology::build(TopologyKind::NervousSystem, &mut rng);
        let grid = WaveTopology::build(TopologyKind::GridLocalLattice, &mut rng);
        let rand = WaveTopology::build(TopologyKind::RandomErdosRenyi, &mut rng);
        let flat = WaveTopology::build(TopologyKind::AllToAllFlat, &mut rng);
        let nodl = WaveTopology::build(TopologyKind::NervousNoDelay, &mut rng);
        let noin = WaveTopology::build(TopologyKind::NervousNoInhibition, &mut rng);

        assert_eq!(nerv.edges.len(), 256);
        assert_eq!(grid.edges.len(), 256);
        assert_eq!(rand.edges.len(), 256);
        assert_eq!(nodl.edges.len(), 256);
        assert_eq!(noin.edges.len(), 256);
        assert_eq!(flat.edges.len(), 4096);

        // Verify lateral inhibition presence in NervousSystem and absence in NervousNoInhibition
        let inh_nerv = nerv.edges.iter().filter(|e| e.sign < 0.0).count();
        let inh_noin = noin.edges.iter().filter(|e| e.sign < 0.0).count();
        assert_eq!(inh_nerv, 24, "Expected 4 ganglia x 2 inh neurons x 3 edges = 24 inhibitory edges");
        assert_eq!(inh_noin, 0);

        // Verify finite propagation distance: Nervous hubs reachable in 2 hops; Grid hubs take >= 3 hops
        let past: Vec<usize> = (0..8).collect();
        let fut: Vec<usize> = (8..16).collect();
        assert!((nerv.mean_hop_distance_to_hubs(&past) - 2.0).abs() < 1e-9);
        assert!((nerv.mean_hop_distance_to_hubs(&fut) - 2.0).abs() < 1e-9);
        assert!(grid.mean_hop_distance_to_hubs(&past) >= 3.0);
        assert!(grid.mean_hop_distance_to_hubs(&fut) >= 3.0);
    }

    #[test]
    fn test_physical_wavefront_causality_and_energy_bounds() {
        let mut rng = Rng64::new(77);
        let core = NervousWaveCore::new(TopologyKind::NervousSystem, &mut rng);
        let u = [0.6, -0.4, 0.3, 0.8];
        let v = [-0.5, 0.7, -0.2, 0.4];
        let cache = core.forward(u, v);

        // Exact vacuum causality at t = 0 and t = 1 on convergent hubs (2 hops away!)
        for h in HUB_START..(HUB_START + N_HUBS) {
            assert_eq!(cache.psi_re[0][h], 0.0);
            assert_eq!(cache.psi_im[0][h], 0.0);
            assert_eq!(cache.psi_re[1][h], 0.0);
            assert_eq!(cache.psi_im[1][h], 0.0);
        }
        // Wavefront MUST arrive at convergent hubs at t = 2
        let e_t2: f64 = (HUB_START..(HUB_START + N_HUBS))
            .map(|h| cache.psi_re[2][h].powi(2) + cache.psi_im[2][h].powi(2))
            .sum();
        assert!(e_t2 > 0.1, "Wavefront failed to arrive at hubs at t=2: e_t2={e_t2}");

        // Saturating phase-preserving oscillator guarantees |psi|^2 < 1/BETA_SAT = 2.0 everywhere
        for t in 0..=8 {
            for i in 0..N_NODES {
                let e = cache.psi_re[t][i].powi(2) + cache.psi_im[t][i].powi(2);
                assert!(e < 2.0, "Wave energy exploded at t={t} i={i}: e={e}");
            }
        }
    }

    #[test]
    fn test_bptt_analytical_gradients_all_topologies_and_myelin() {
        let eps = 1e-6;
        for &kind in &[
            TopologyKind::NervousSystem,
            TopologyKind::GridLocalLattice,
            TopologyKind::RandomErdosRenyi,
            TopologyKind::NervousNoDelay,
            TopologyKind::NervousNoInhibition,
        ] {
            let mut rng = Rng64::new(1000 + kind as u64);
            let mut core = NervousWaveCore::new(kind, &mut rng);
            for i in 0..N_NODES {
                core.myelin_omega[i] = -0.2 + 0.007 * (i as f64);
            }
            let u = [0.42, -0.61, 0.33, -0.55];
            let v = [-0.37, 0.52, -0.48, 0.66];
            let dz: [f64; REL_DIM] = std::array::from_fn(|i| ((i as f64) * 0.39 + 0.1).sin());

            let obj = |c: &NervousWaveCore, uu: [f64; LATENT_D], vv: [f64; LATENT_D]| -> f64 {
                let ca = c.forward(uu, vv);
                (0..REL_DIM).map(|i| ca.z_rel[i] * dz[i]).sum()
            };

            let cache = core.forward(u, v);
            let mut g_full = NervousWaveGrads::zeros(core.edge_count());
            let (du, dv) = core.backward(&cache, &dz, WaveTrainMode::FullTrain, Some(&mut g_full));
            let mut g_my = NervousWaveGrads::zeros(core.edge_count());
            let (du_m, dv_m) = core.backward(&cache, &dz, WaveTrainMode::MyelinCalOnly, Some(&mut g_my));
            for d in 0..LATENT_D {
                assert!((du[d] - du_m[d]).abs() < 1e-12);
                assert!((dv[d] - dv_m[d]).abs() < 1e-12);
            }

            for d in 0..LATENT_D {
                let mut up = u;
                up[d] += eps;
                let mut um = u;
                um[d] -= eps;
                assert!(((obj(&core, up, v) - obj(&core, um, v)) / (2.0 * eps) - du[d]).abs() < 1e-5, "du mismatch {kind:?}");
                let mut vp = v;
                vp[d] += eps;
                let mut vm = v;
                vm[d] -= eps;
                assert!(((obj(&core, u, vp) - obj(&core, u, vm)) / (2.0 * eps) - dv[d]).abs() < 1e-5, "dv mismatch {kind:?}");
            }
            for i in (0..core.w_in.len()).step_by(11) {
                let o = core.w_in[i];
                core.w_in[i] = o + eps;
                let lp = obj(&core, u, v);
                core.w_in[i] = o - eps;
                let lm = obj(&core, u, v);
                core.w_in[i] = o;
                assert!(((lp - lm) / (2.0 * eps) - g_full.w_in[i]).abs() < 1e-5, "w_in mismatch {kind:?}");
            }
            for i in (0..core.alpha_e.len()).step_by(29) {
                let o = core.alpha_e[i];
                core.alpha_e[i] = o + eps;
                let lp = obj(&core, u, v);
                core.alpha_e[i] = o - eps;
                let lm = obj(&core, u, v);
                core.alpha_e[i] = o;
                assert!(((lp - lm) / (2.0 * eps) - g_full.alpha_e[i]).abs() < 1e-5, "alpha_e mismatch {kind:?}");

                if kind.is_complex_wave() {
                    let ot = core.theta_e[i];
                    core.theta_e[i] = ot + eps;
                    let lpt = obj(&core, u, v);
                    core.theta_e[i] = ot - eps;
                    let lmt = obj(&core, u, v);
                    core.theta_e[i] = ot;
                    assert!(((lpt - lmt) / (2.0 * eps) - g_full.theta_e[i]).abs() < 1e-5, "theta_e mismatch {kind:?}");
                }
            }
            for i in (0..N_NODES).step_by(9) {
                let og = core.gamma_logit_n[i];
                core.gamma_logit_n[i] = og + eps;
                let lp = obj(&core, u, v);
                core.gamma_logit_n[i] = og - eps;
                let lm = obj(&core, u, v);
                core.gamma_logit_n[i] = og;
                assert!(((lp - lm) / (2.0 * eps) - g_full.gamma_logit_n[i]).abs() < 1e-5, "gamma mismatch {kind:?}");

                if kind.is_complex_wave() {
                    let om = core.myelin_omega[i];
                    core.myelin_omega[i] = om + eps;
                    let lpm = obj(&core, u, v);
                    core.myelin_omega[i] = om - eps;
                    let lmm = obj(&core, u, v);
                    core.myelin_omega[i] = om;
                    assert!(((lpm - lmm) / (2.0 * eps) - g_my.myelin_omega[i]).abs() < 1e-5, "myelin mismatch {kind:?}");
                }
            }
        }
    }

    #[test]
    fn test_bit_exact_frozen_synapses_during_myelin_transfer() {
        let mut rng = Rng64::new(0xCAFE);
        let dom_c = generate_domain_c(&mut rng.fork(3));
        let core = NervousWaveCore::new(TopologyKind::NervousSystem, &mut rng);
        let h_pre = core.frozen_synapses_checksum();
        let mut model = NervousWaveModel::new(dom_c.obs_dim, dom_c.reg_dim, dom_c.cls_dim, core, &mut rng);
        let cfg = TrainConfig {
            steps: 24,
            batch_size: 16,
            lr_encoder: 0.02,
            lr_core: 0.0,
            lr_myelin: 0.04,
            lr_head: 0.02,
            wd_encoder: 1e-2,
            wd_core: 0.0,
            wd_myelin: 0.5,
            wd_head: 1e-2,
            w_mse: 0.0,
            w_ce: 1.0,
            latent_norm_reg: 0.05,
            latent_jitter: 0.0,
            mode: WaveTrainMode::MyelinCalOnly,
        };
        model.train(&dom_c, dom_c.train_indices(60), &cfg, &mut rng);
        assert_eq!(model.core.frozen_synapses_checksum(), h_pre);
        assert_eq!(model.core.autocal_param_count(), 64);
    }
}
