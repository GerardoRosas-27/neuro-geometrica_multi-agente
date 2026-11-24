//! Isolated experiment crate: `Riemann Wave Inference -> Collapse -> Adaptive State Learning`
//! (`exp/adaptive-spectral-collapse`).

pub mod mlp_model;
pub mod rng;
pub mod spectra;
pub mod spectral_model;
pub mod suites;
pub mod world;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::mlp_model::MlpModel;
    use crate::rng::Rng;
    use crate::spectra::compute_riemann_zeros;
    use crate::spectral_model::{CausalReadout, SpectralArm, SpectralModel, Wave1D, Wave3D, M_1D, M_3D};
    use crate::world::{UnknownWorld, K_BASE, K_NEW, K_TOT, N_TRAIN_BASE_PAIRS};

    #[test]
    fn test_riemann_zeros_sub_1e9() {
        let zeros = compute_riemann_zeros(12);
        assert_eq!(zeros.len(), 12);
        let expected = [
            (0, 14.134_725_141_734_695),
            (1, 21.022_039_638_771_556),
            (9, 49.773_832_477_672_3),
        ];
        for (i, v) in expected {
            assert!((zeros[i] - v).abs() < 1e-9);
        }
    }

    #[test]
    fn test_world_strict_disjoint_splits_and_connectivity() {
        let w = UnknownWorld::generate(0xC100);
        assert_eq!(w.xs.len(), K_TOT);
        assert_eq!(w.train_pairs.len(), N_TRAIN_BASE_PAIRS);
        assert_eq!(w.unseen_pairs.len(), K_BASE * (K_BASE - 1) / 2 - N_TRAIN_BASE_PAIRS);

        let tr_set: HashSet<(usize, usize)> = w.train_pairs.iter().copied().collect();
        let un_set: HashSet<(usize, usize)> = w.unseen_pairs.iter().copied().collect();
        assert!(tr_set.is_disjoint(&un_set), "TRAIN_PAIRS and UNSEEN_PAIRS must be strictly disjoint");

        let mut deg = [0usize; K_BASE];
        for &(a, b) in &w.train_pairs {
            deg[a] += 1;
            deg[b] += 1;
        }
        assert!(
            deg.iter().all(|&d| d >= 6),
            "Every base concept must appear >= 6 times in TRAIN_PAIRS, got {:?}",
            deg
        );

        assert_eq!(w.novel_train_pairs.len(), K_NEW * 4);
        assert_eq!(w.novel_old_unseen.len(), K_NEW * (K_BASE - 4));
        assert_eq!(w.novel_novel_unseen.len(), K_NEW * (K_NEW - 1) / 2);
    }

    #[test]
    fn test_parameter_counts_match_plan() {
        assert_eq!(SpectralModel::TOTAL_PARAMS, 962);
        assert_eq!(MlpModel::param_count(K_BASE), 1118);
    }

    #[test]
    fn test_adaptive_spectral_uses_zero_riemann_zeros() {
        let mut rng = Rng::from_seed(0x42, 1);
        let model = SpectralModel::init(SpectralArm::AdaptiveSpectral, None, &mut rng);
        assert!(model.arm.adapts_frequencies());
        assert_eq!(model.w_enc.len(), 3);
    }

    #[test]
    fn test_wave1d_and_wave3d_analytical_gradients_vs_finite_diff() {
        let eps = 1e-6_f64;
        let mut rng = Rng::from_seed(0x99, 1);

        // 1D wave gradient check
        let mut w1 = Wave1D { omega: [0.0; M_1D], amp: [0.0; M_1D], phi: [0.0; M_1D] };
        for m in 0..M_1D {
            w1.omega[m] = rng.uniform(1.5, 15.0);
            w1.amp[m] = rng.normal();
            w1.phi[m] = rng.uniform(0.0, 6.0);
        }
        let u = 0.73;
        let mut gw = [0.0; M_1D];
        let mut ga = [0.0; M_1D];
        let mut gp = [0.0; M_1D];
        w1.accum_grad(u, 1.0, &mut gw, &mut ga, &mut gp, true);
        for m in 0..M_1D {
            let mut wp = w1.clone();
            let mut wm = w1.clone();
            wp.omega[m] += eps;
            wm.omega[m] -= eps;
            let fd_w = (wp.eval(u) - wm.eval(u)) / (2.0 * eps);
            assert!((fd_w - gw[m]).abs() < 1e-5, "1D omega grad mismatch: {} vs {}", fd_w, gw[m]);

            let mut wp = w1.clone();
            let mut wm = w1.clone();
            wp.amp[m] += eps;
            wm.amp[m] -= eps;
            let fd_a = (wp.eval(u) - wm.eval(u)) / (2.0 * eps);
            assert!((fd_a - ga[m]).abs() < 1e-5);

            let mut wp = w1.clone();
            let mut wm = w1.clone();
            wp.phi[m] += eps;
            wm.phi[m] -= eps;
            let fd_p = (wp.eval(u) - wm.eval(u)) / (2.0 * eps);
            assert!((fd_p - gp[m]).abs() < 1e-5);
        }

        // 3D wave gradient check
        let mut w3 = Wave3D { k: [[0.0; 3]; M_3D], amp: [0.0; M_3D], phi: [0.0; M_3D] };
        for m in 0..M_3D {
            w3.k[m] = [rng.normal() * 3.0, rng.normal() * 3.0, rng.normal() * 3.0];
            w3.amp[m] = rng.normal();
            w3.phi[m] = rng.uniform(0.0, 6.0);
        }
        let u3 = [0.41, -0.82, 1.15];
        let mut gk = [[0.0; 3]; M_3D];
        let mut ga3 = [0.0; M_3D];
        let mut gp3 = [0.0; M_3D];
        w3.accum_grad(&u3, 1.0, &mut gk, &mut ga3, &mut gp3, true);
        for m in 0..4 {
            for d in 0..3 {
                let mut wp = w3.clone();
                let mut wm = w3.clone();
                wp.k[m][d] += eps;
                wm.k[m][d] -= eps;
                let fd = (wp.eval(&u3) - wm.eval(&u3)) / (2.0 * eps);
                assert!((fd - gk[m][d]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_causal_readout_and_mlp_analytical_gradients_vs_finite_diff() {
        let eps = 1e-6_f64;
        let world = UnknownWorld::generate(0xC101);
        let mut rng = Rng::from_seed(0x77, 1);

        // CausalReadout gradient check
        let cr = CausalReadout::new(K_BASE, &mut rng);
        let mut grads = vec![0.0; cr.params.len()];
        let (ty1, ty2, _) = world.eval_outcome(0, 1);
        cr.loss_and_grad_pair(0, 1, ty1, ty2, &mut grads, 1.0, None);
        for idx in (0..cr.params.len()).step_by(7) {
            let mut p_plus = cr.clone();
            let mut p_minus = cr.clone();
            p_plus.params[idx] += eps;
            p_minus.params[idx] -= eps;
            let mut dummy = vec![0.0; cr.params.len()];
            let lp = p_plus.loss_and_grad_pair(0, 1, ty1, ty2, &mut dummy, 1.0, None);
            let lm = p_minus.loss_and_grad_pair(0, 1, ty1, ty2, &mut dummy, 1.0, None);
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - grads[idx]).abs() < 1e-5,
                "CausalReadout grad[{}]: fd={} analytical={}",
                idx,
                fd,
                grads[idx]
            );
        }

        // MlpModel gradient check (checking non-ReLU-kink parameters)
        let mlp = MlpModel::new(K_BASE, &mut rng);
        let x_obs = world.observe_pair(2, 5, 0.0, &mut rng);
        let (my1, my2, _) = world.eval_outcome(2, 5);
        let mut mgr = vec![0.0; mlp.params.len()];
        mlp.loss_and_grad_sample(&x_obs, &[2, 5], Some((my1, my2)), &mut mgr, 1.0);
        for idx in (0..mlp.params.len()).step_by(19) {
            let mut mp = mlp.clone();
            let mut mm = mlp.clone();
            mp.params[idx] += eps;
            mm.params[idx] -= eps;
            let mut dummy = vec![0.0; mlp.params.len()];
            let lp = mp.loss_and_grad_sample(&x_obs, &[2, 5], Some((my1, my2)), &mut dummy, 1.0);
            let lm = mm.loss_and_grad_sample(&x_obs, &[2, 5], Some((my1, my2)), &mut dummy, 1.0);
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - mgr[idx]).abs() < 1e-4,
                "MLP grad[{}]: fd={} analytical={}",
                idx,
                fd,
                mgr[idx]
            );
        }
    }

    #[test]
    fn test_mlp_baseline_genuinely_fits_training_pairs() {
        let world = UnknownWorld::generate(0xC102);
        let mut r_init = Rng::from_seed(0xC102, 199);
        let mut r_tr = Rng::from_seed(0xC102, 299);
        let mut mlp = MlpModel::new(K_BASE, &mut r_init);
        mlp.train_base(&world, &mut r_tr);

        let mut eval_rng = Rng::from_seed(0xC102, 500);
        let mut hits = 0usize;
        for &(a, b) in &world.train_pairs {
            let x = world.observe_pair(a, b, 0.0, &mut eval_rng);
            let ((pa, pb), _) = mlp.infer(&x);
            if pa == a && pb == b {
                hits += 1;
            }
        }
        let acc = (hits as f64) / (world.train_pairs.len() as f64);
        assert!(
            acc >= 0.90,
            "MLP baseline must genuinely converge on TRAIN_PAIRS (got {:.3}), not be a strawman",
            acc
        );
    }
}
