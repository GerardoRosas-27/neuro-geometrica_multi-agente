#[allow(dead_code)]
mod domains;
#[allow(dead_code)]
mod rng;
mod suites;
mod topology;
mod wave_net;

use std::env;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut seed: u64 = 0xB100;
    let mut split = String::from("dev");
    let mut out_dir = PathBuf::from("out_b100");

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                let s = &args[i];
                seed = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                    u64::from_str_radix(hex, 16).expect("invalid hex seed")
                } else {
                    s.parse::<u64>().expect("invalid u64 seed")
                };
            }
            "--split" => {
                i += 1;
                split = args[i].clone();
            }
            "--out-dir" => {
                i += 1;
                out_dir = PathBuf::from(&args[i]);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(2);
            }
        }
        i += 1;
    }

    suites::run_seed(seed, &split, &out_dir).expect("run_seed failed");
}

#[cfg(test)]
mod tests {
    use crate::domains::{generate_domain_a, generate_domain_b, generate_domain_c};
    use crate::rng::Rng64;
    use crate::topology::{GraphTopology, TopologyKind, E_FLAT, E_SPARSE, N_FLAT, N_SPARSE};
    use crate::wave_net::{GradAccum, NervousWaveNet, Perturbation, TrainMode};
    use std::collections::HashSet;

    #[test]
    fn test_topology_counts_uniqueness_and_hops() {
        for &kind in &TopologyKind::all() {
            let topo = GraphTopology::build(kind, 0xB100);
            if kind == TopologyKind::FlatDense4096 {
                assert_eq!(topo.n_nodes, N_FLAT);
                assert_eq!(topo.edges.len(), E_FLAT);
            } else {
                assert_eq!(topo.n_nodes, N_SPARSE);
                assert_eq!(topo.edges.len(), E_SPARSE);
                let mut seen = HashSet::new();
                for e in &topo.edges {
                    assert_ne!(e.src, e.dst, "self-loop in {:?}", kind);
                    assert!(
                        seen.insert((e.src, e.dst)),
                        "duplicate edge ({},{}) in {:?}",
                        e.src,
                        e.dst,
                        kind
                    );
                }
            }
        }
        let nervous = GraphTopology::build(TopologyKind::NervousCoherenceB, 0xB100);
        let grid = GraphTopology::build(TopologyKind::GridLocal128, 0xB100);
        let h_nerv = nervous.mean_receptor_to_hub_hops();
        let h_grid = grid.mean_receptor_to_hub_hops();
        assert!((h_nerv - 2.25).abs() < 1e-9, "nervous hops={}", h_nerv);
        assert!((h_grid - 4.125).abs() < 1e-9, "grid hops={}", h_grid);
        assert!(h_grid - h_nerv >= 1.8);
    }

    #[test]
    fn test_causality_and_kuramoto_incoherent_dispersion() {
        let mut rng = Rng64::new(0xB100);
        let dom = generate_domain_a(&mut rng);
        let topo = GraphTopology::build(TopologyKind::NervousCoherenceB, 0xB100);
        let net = NervousWaveNet::new(topo, dom.obs_dim, dom.reg_dim, dom.cls_dim, &mut rng);
        let p = &dom.pairs[0];
        let tr = net.forward(&dom.obs[p.a], &dom.obs[p.b], &Perturbation::none());

        // Exact causality: hubs strictly zero at t=0 and t=1; nonzero by t=2
        let e0: f64 = net
            .topo
            .hubs
            .iter()
            .map(|&h| tr.psi_re[0][h].powi(2) + tr.psi_im[0][h].powi(2))
            .sum();
        let e1: f64 = net
            .topo
            .hubs
            .iter()
            .map(|&h| tr.psi_re[1][h].powi(2) + tr.psi_im[1][h].powi(2))
            .sum();
        let e2: f64 = net
            .topo
            .hubs
            .iter()
            .map(|&h| tr.psi_re[2][h].powi(2) + tr.psi_im[2][h].powi(2))
            .sum();
        assert_eq!(e0, 0.0, "hub energy must be strictly 0 at t=0");
        assert_eq!(e1, 0.0, "hub energy must be strictly 0 at t=1");
        assert!(e2 > 1e-9, "wavefront must reach hubs by t=2");

        // All Kuramoto incoherent dispersion energies strictly >= 0 and finite
        for row in &tr.d_incoh {
            for &d in row {
                assert!(d.is_finite() && d >= 0.0, "invalid d_incoh={}", d);
            }
        }
        assert!(tr.hub_r_coh >= 0.0 && tr.hub_r_coh <= 1.0001);
    }

    #[test]
    fn test_exact_analytical_bptt_gradients() {
        let mut rng = Rng64::new(0xC0DE_B001);
        let dom = generate_domain_a(&mut rng);
        let p = &dom.pairs[0];
        let oa = &dom.obs[p.a];
        let ob = &dom.obs[p.b];
        let eps = 1e-6;

        for &kind in &[
            TopologyKind::NervousCoherenceB,
            TopologyKind::NervousNoCoherenceGate,
            TopologyKind::NervousNoDelay,
            TopologyKind::FlatDense4096,
        ] {
            let topo = GraphTopology::build(kind, 0xB100);
            let mut net = NervousWaveNet::new(topo, dom.obs_dim, dom.reg_dim, dom.cls_dim, &mut rng);
            let pert = Perturbation::none();
            let cache = crate::wave_net::StepParamCache::build(&net, &pert);
            let tr = net.forward_cached(oa, ob, &cache);
            let mut grads = GradAccum::zeros_like(&net);
            let _ = net.backward_cached(oa, ob, &p.reg, p.cls, 1.0, 1.0, &cache, &tr, &mut grads);

            let eval_loss = |n: &NervousWaveNet| -> f64 {
                let c = crate::wave_net::StepParamCache::build(n, &pert);
                let t = n.forward_cached(oa, ob, &c);
                let mut dummy = GradAccum::zeros_like(n);
                n.backward_cached(oa, ob, &p.reg, p.cls, 1.0, 1.0, &c, &t, &mut dummy)
            };

            let check_param = |val_analytic: f64, f_plus: f64, f_minus: f64, label: &str| {
                let fd = (f_plus - f_minus) / (2.0 * eps);
                let err = (val_analytic - fd).abs();
                let denom = val_analytic.abs().max(fd.abs()).max(1.0);
                assert!(
                    err / denom < 1e-5,
                    "BPTT mismatch on {:?} {}: analytic={} fd={} err={}",
                    kind,
                    label,
                    val_analytic,
                    fd,
                    err
                );
            };

            for &ei in &[0usize, 17, 260, 777, net.alpha_e.len() - 1] {
                let orig = net.alpha_e[ei];
                net.alpha_e[ei] = orig + eps;
                let fp = eval_loss(&net);
                net.alpha_e[ei] = orig - eps;
                let fm = eval_loss(&net);
                net.alpha_e[ei] = orig;
                check_param(grads.alpha_e[ei], fp, fm, "alpha_e");

                let orig_t = net.theta_e[ei];
                net.theta_e[ei] = orig_t + eps;
                let fp = eval_loss(&net);
                net.theta_e[ei] = orig_t - eps;
                let fm = eval_loss(&net);
                net.theta_e[ei] = orig_t;
                check_param(grads.theta_e[ei], fp, fm, "theta_e");
            }

            for &ni in &[0usize, 19, 35, net.topo.n_nodes - 1] {
                let o1 = net.omega_n[ni];
                net.omega_n[ni] = o1 + eps;
                let fp = eval_loss(&net);
                net.omega_n[ni] = o1 - eps;
                let fm = eval_loss(&net);
                net.omega_n[ni] = o1;
                check_param(grads.omega_n[ni], fp, fm, "omega_n");

                let o2 = net.gamma_logit_n[ni];
                net.gamma_logit_n[ni] = o2 + eps;
                let fp = eval_loss(&net);
                net.gamma_logit_n[ni] = o2 - eps;
                let fm = eval_loss(&net);
                net.gamma_logit_n[ni] = o2;
                check_param(grads.gamma_logit_n[ni], fp, fm, "gamma_logit_n");

                let o3 = net.tau_n[ni];
                net.tau_n[ni] = o3 + eps;
                let fp = eval_loss(&net);
                net.tau_n[ni] = o3 - eps;
                let fm = eval_loss(&net);
                net.tau_n[ni] = o3;
                check_param(grads.tau_n[ni], fp, fm, "tau_n");

                let o4 = net.myelin_omega[ni];
                net.myelin_omega[ni] = o4 + eps;
                let fp = eval_loss(&net);
                net.myelin_omega[ni] = o4 - eps;
                let fm = eval_loss(&net);
                net.myelin_omega[ni] = o4;
                check_param(grads.myelin_omega[ni], fp, fm, "myelin_omega");
            }

            for &ri in &[0usize, 7, 15] {
                let os = net.sensory_scale[ri];
                net.sensory_scale[ri] = os + eps;
                let fp = eval_loss(&net);
                net.sensory_scale[ri] = os - eps;
                let fm = eval_loss(&net);
                net.sensory_scale[ri] = os;
                check_param(grads.sensory_scale[ri], fp, fm, "sensory_scale");

                let op = net.sensory_phase[ri];
                net.sensory_phase[ri] = op + eps;
                let fp = eval_loss(&net);
                net.sensory_phase[ri] = op - eps;
                let fm = eval_loss(&net);
                net.sensory_phase[ri] = op;
                check_param(grads.sensory_phase[ri], fp, fm, "sensory_phase");
            }
        }
    }

    #[test]
    fn test_frozen_tissue_bit_exact_during_resonant_transfer() {
        let mut rng = Rng64::new(0xB100);
        let dom_a = generate_domain_a(&mut rng);
        let dom_b = generate_domain_b(&mut rng);
        let dom_c = generate_domain_c(&mut rng);
        let topo = GraphTopology::build(TopologyKind::NervousCoherenceB, 0xB100);
        let base = NervousWaveNet::new(topo, dom_a.obs_dim, dom_a.reg_dim, dom_a.cls_dim, &mut rng);

        let mut tr_b = base.clone_for_transfer(dom_b.obs_dim, dom_b.reg_dim, dom_b.cls_dim, false, &mut rng);
        let ck_b0 = tr_b.frozen_tissue_checksum();
        tr_b.train(&dom_b, dom_b.train_indices(32), TrainMode::ResonantCalOnly, 12, 16, 0.02, 1.0, 1.0, &mut rng);
        assert_eq!(ck_b0, tr_b.frozen_tissue_checksum(), "A->B modified frozen tissue!");
        assert!(tr_b.myelin_omega.iter().any(|&v| v.abs() > 1e-6), "myelin spring did not calibrate");

        let mut tr_c = base.clone_for_transfer(dom_c.obs_dim, dom_c.reg_dim, dom_c.cls_dim, false, &mut rng);
        let ck_c0 = tr_c.frozen_tissue_checksum();
        tr_c.train(&dom_c, dom_c.train_indices(40), TrainMode::ResonantCalOnly, 12, 16, 0.02, 0.0, 1.0, &mut rng);
        assert_eq!(ck_c0, tr_c.frozen_tissue_checksum(), "A->C modified frozen tissue!");
    }
}
