mod domains;
mod metrics;
mod models;
mod rng;
mod suites;

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
        eprintln!("Usage: spectral_transfer run-seed --seed <0xHEX|DEC> --out <DIR> [--trace]");
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
    use crate::domains::{generate_domain_a, generate_domain_b, generate_domain_c};
    use crate::metrics::spectral_spacing_r;
    use crate::models::{
        DomainEncoder, DomainHead, MlpCore, MlpCoreGrads, MlpModel, PairingMode, SpectralCore,
        SpectralCoreGrads, SpectralModel, SpectrumKind, TrainConfig, LATENT_D, N_MODES, REL_DIM,
    };
    use crate::rng::{Rng64, RIEMANN_ZEROS_48};

    #[test]
    fn test_riemann_zeros_reference_accuracy() {
        // Verify Odlyzko reference zeros to < 1e-9 and strict monotonicity / GUE spacing
        assert!((RIEMANN_ZEROS_48[0] - 14.134725141734693).abs() < 1e-12);
        assert!((RIEMANN_ZEROS_48[1] - 21.022039638771555).abs() < 1e-12);
        assert!((RIEMANN_ZEROS_48[9] - 49.773832477672302).abs() < 1e-12);
        assert!((RIEMANN_ZEROS_48[47] - 139.736208952121389).abs() < 1e-12);
        for i in 0..(RIEMANN_ZEROS_48.len() - 1) {
            assert!(RIEMANN_ZEROS_48[i + 1] > RIEMANN_ZEROS_48[i]);
        }
        let mut rng = Rng64::new(42);
        let rc = SpectralCore::new(SpectrumKind::RiemannFixed, PairingMode::FullInterferometric, false, &mut rng);
        let r_stat = spectral_spacing_r(&rc.omega);
        assert!(r_stat > 0.55 && r_stat < 0.70, "Unexpected Riemann <r> = {r_stat}");
    }

    #[test]
    fn test_exact_parameter_counts() {
        let mut rng = Rng64::new(123);
        let sc = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, false, &mut rng);
        let mc = MlpCore::new(&mut rng);
        assert_eq!(sc.param_count(), 1936);
        assert_eq!(mc.param_count(), 2008);
        let enc_b = DomainEncoder::new(12, &mut rng);
        let head_b = DomainHead::new(3, 4, &mut rng);
        assert_eq!(enc_b.param_count() + head_b.param_count(), 171);
        let enc_c = DomainEncoder::new(16, &mut rng);
        let head_c = DomainHead::new(0, 4, &mut rng);
        assert_eq!(enc_c.param_count() + head_c.param_count(), 136);
    }

    #[test]
    #[should_panic(expected = "PROTOCOL VIOLATION")]
    fn test_adaptive_rejects_riemann_injection() {
        let mut rng = Rng64::new(77);
        let sc = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, false, &mut rng);
        sc.assert_no_riemann_injection(true);
    }

    #[test]
    fn test_spectral_core_gradients_all_pairing_modes() {
        let eps = 1e-6;
        for &mode in &[
            PairingMode::FullInterferometric,
            PairingMode::AdditiveOnly,
            PairingMode::PermutedPairing,
        ] {
            let mut rng = Rng64::new(901 + (mode as u64));
            let mut core = SpectralCore::new(SpectrumKind::Adaptive, mode, false, &mut rng);
            let mut enc = DomainEncoder::new(6, &mut rng);
            let mut head = DomainHead::new(3, 4, &mut rng);
            let xa: Vec<f64> = (0..6).map(|_| rng.normal()).collect();
            let xb: Vec<f64> = (0..6).map(|_| rng.normal()).collect();
            let y_true = vec![0.4, -0.7, 0.25];
            let cls_true = 2usize;

            let eval_loss = |e: &DomainEncoder, c: &SpectralCore, h: &DomainHead| -> f64 {
                let u = e.forward(&xa);
                let v = e.forward(&xb);
                let cache = c.forward(u, v);
                let (reg, logits) = h.forward(&cache.z_rel);
                let mut tmp_wr = vec![0.0; h.w_reg.len()];
                let mut tmp_br = vec![0.0; h.b_reg.len()];
                let mut tmp_wc = vec![0.0; h.w_cls.len()];
                let mut tmp_bc = vec![0.0; h.b_cls.len()];
                let (l, _) = h.loss_and_backward(
                    &cache.z_rel,
                    &reg,
                    &logits,
                    &y_true,
                    cls_true,
                    1.0,
                    0.8,
                    1.0,
                    &mut tmp_wr,
                    &mut tmp_br,
                    &mut tmp_wc,
                    &mut tmp_bc,
                );
                l
            };

            let u = enc.forward(&xa);
            let v = enc.forward(&xb);
            let cache = core.forward(u, v);
            let (reg, logits) = head.forward(&cache.z_rel);
            let mut g_wr = vec![0.0; head.w_reg.len()];
            let mut g_br = vec![0.0; head.b_reg.len()];
            let mut g_wc = vec![0.0; head.w_cls.len()];
            let mut g_bc = vec![0.0; head.b_cls.len()];
            let (_, dz) = head.loss_and_backward(
                &cache.z_rel,
                &reg,
                &logits,
                &y_true,
                cls_true,
                1.0,
                0.8,
                1.0,
                &mut g_wr,
                &mut g_br,
                &mut g_wc,
                &mut g_bc,
            );
            let mut gc = SpectralCoreGrads::zeros();
            let (du, dv) = core.backward(&cache, &dz, true, Some(&mut gc));
            let mut g_ew = vec![0.0; enc.w.len()];
            let mut g_eb = vec![0.0; enc.b.len()];
            enc.backward(&xa, &du, &mut g_ew, &mut g_eb);
            enc.backward(&xb, &dv, &mut g_ew, &mut g_eb);

            // Check encoder weights
            for i in 0..enc.w.len() {
                let orig = enc.w[i];
                enc.w[i] = orig + eps;
                let lp = eval_loss(&enc, &core, &head);
                enc.w[i] = orig - eps;
                let lm = eval_loss(&enc, &core, &head);
                enc.w[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                assert!(
                    (num - g_ew[i]).abs() < 1e-5,
                    "Encoder grad mismatch mode={mode:?} i={i}: num={num} ana={}",
                    g_ew[i]
                );
            }
            // Check omega, phi_diff, phi_sum, alpha_diff, alpha_sum, w_pair (sampled indices)
            for i in (0..core.omega.len()).step_by(7) {
                let orig = core.omega[i];
                core.omega[i] = orig + eps;
                let lp = eval_loss(&enc, &core, &head);
                core.omega[i] = orig - eps;
                let lm = eval_loss(&enc, &core, &head);
                core.omega[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                assert!(
                    (num - gc.omega[i]).abs() < 1e-5,
                    "Omega grad mismatch mode={mode:?} i={i}: num={num} ana={}",
                    gc.omega[i]
                );
            }
            for i in (0..N_MODES).step_by(5) {
                let orig = core.phi_diff[i];
                core.phi_diff[i] = orig + eps;
                let lp = eval_loss(&enc, &core, &head);
                core.phi_diff[i] = orig - eps;
                let lm = eval_loss(&enc, &core, &head);
                core.phi_diff[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                assert!((num - gc.phi_diff[i]).abs() < 1e-5, "phi_diff mismatch");

                let orig = core.alpha_sum[i];
                core.alpha_sum[i] = orig + eps;
                let lp = eval_loss(&enc, &core, &head);
                core.alpha_sum[i] = orig - eps;
                let lm = eval_loss(&enc, &core, &head);
                core.alpha_sum[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                assert!((num - gc.alpha_sum[i]).abs() < 1e-5, "alpha_sum mismatch");
            }
            for i in (0..core.w_pair.len()).step_by(53) {
                let orig = core.w_pair[i];
                core.w_pair[i] = orig + eps;
                let lp = eval_loss(&enc, &core, &head);
                core.w_pair[i] = orig - eps;
                let lm = eval_loss(&enc, &core, &head);
                core.w_pair[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                assert!((num - gc.w_pair[i]).abs() < 1e-5, "w_pair mismatch");
            }
            for i in (0..head.w_cls.len()).step_by(7) {
                let orig = head.w_cls[i];
                head.w_cls[i] = orig + eps;
                let lp = eval_loss(&enc, &core, &head);
                head.w_cls[i] = orig - eps;
                let lm = eval_loss(&enc, &core, &head);
                head.w_cls[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                assert!((num - g_wc[i]).abs() < 1e-5, "head w_cls mismatch");
            }
        }
    }

    #[test]
    fn test_spectral_regularizer_and_mlp_gradients() {
        let eps = 1e-6;
        let mut rng = Rng64::new(555);
        let mut core = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, false, &mut rng);
        // Push one mode out of annulus to test bandwidth branch too
        core.omega[0] = 3.5;
        core.omega[4] = 0.2;
        let mut g_om = vec![0.0; core.omega.len()];
        let _ = core.spectral_regularizer_grad(0.04, 0.025, 0.05, 0.08, &mut g_om);
        for i in (0..core.omega.len()).step_by(5) {
            let orig = core.omega[i];
            core.omega[i] = orig + eps;
            let mut dummy = vec![0.0; core.omega.len()];
            let lp = core.spectral_regularizer_grad(0.04, 0.025, 0.05, 0.08, &mut dummy);
            core.omega[i] = orig - eps;
            dummy.fill(0.0);
            let lm = core.spectral_regularizer_grad(0.04, 0.025, 0.05, 0.08, &mut dummy);
            core.omega[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (num - g_om[i]).abs() < 1e-5,
                "Regularizer grad mismatch i={i}: num={num} ana={}",
                g_om[i]
            );
        }

        // Verify MlpCore gradients
        let mut mlp = MlpCore::new(&mut rng);
        let u = [0.3, -0.5, 0.7, -0.2];
        let v = [-0.4, 0.6, -0.1, 0.8];
        let dz: [f64; REL_DIM] = std::array::from_fn(|i| ((i as f64) * 0.37).sin());
        let cache = mlp.forward(u, v);
        let mut gm = MlpCoreGrads::zeros();
        let (du, dv) = mlp.backward(&cache, &dz, true, Some(&mut gm));

        let mlp_obj = |m: &MlpCore, uu: [f64; LATENT_D], vv: [f64; LATENT_D]| -> f64 {
            let c = m.forward(uu, vv);
            (0..REL_DIM).map(|i| c.z_rel[i] * dz[i]).sum()
        };
        for d in 0..LATENT_D {
            let mut up = u;
            up[d] += eps;
            let mut um = u;
            um[d] -= eps;
            let num = (mlp_obj(&mlp, up, v) - mlp_obj(&mlp, um, v)) / (2.0 * eps);
            assert!((num - du[d]).abs() < 1e-5, "MLP du mismatch");
            let mut vp = v;
            vp[d] += eps;
            let mut vm = v;
            vm[d] -= eps;
            let num_v = (mlp_obj(&mlp, u, vp) - mlp_obj(&mlp, u, vm)) / (2.0 * eps);
            assert!((num_v - dv[d]).abs() < 1e-5, "MLP dv mismatch");
        }
        for i in (0..mlp.w2.len()).step_by(37) {
            let orig = mlp.w2[i];
            mlp.w2[i] = orig + eps;
            let lp = mlp_obj(&mlp, u, v);
            mlp.w2[i] = orig - eps;
            let lm = mlp_obj(&mlp, u, v);
            mlp.w2[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!((num - gm.w2[i]).abs() < 1e-5, "MLP w2 mismatch");
        }
    }

    #[test]
    fn test_bit_exact_frozen_core_invariance_in_transfer() {
        let mut rng = Rng64::new(0xBEEF);
        let dom_a = generate_domain_a(&mut rng.fork(1));
        let dom_b = generate_domain_b(&mut rng.fork(2));
        let dom_c = generate_domain_c(&mut rng.fork(3));

        // Verify class diversity across all 3 domains
        for dom in &[&dom_a, &dom_b, &dom_c] {
            let mut counts = vec![0usize; dom.cls_dim];
            for p in &dom.pairs {
                counts[p.cls] += 1;
            }
            for (c, &cnt) in counts.iter().enumerate() {
                assert!(cnt >= 30, "Domain {} class {c} under-represented: {cnt}", dom.name);
            }
        }

        let cfg = TrainConfig {
            steps: 30,
            batch_size: 16,
            lr_encoder: 0.02,
            lr_core: 0.02,
            lr_head: 0.02,
            wd_encoder: 1e-3,
            wd_core: 1e-3,
            wd_head: 1e-3,
            w_mse: 1.0,
            w_ce: 0.8,
            w_ssl: 0.0,
            lambda_vec: 0.008,
            lambda_rad: 0.015,
            lambda_alias: 0.015,
            lambda_bw: 0.25,
            lambda_var: 0.12,
            lambda_cov: 0.20,
            latent_norm_reg: 0.04,
            latent_jitter: 0.0,
            freeze_core: true,
        };

        let sc = SpectralCore::new(SpectrumKind::Adaptive, PairingMode::FullInterferometric, false, &mut rng);
        let h_sc = sc.checksum();
        let mut sm = SpectralModel::new(dom_b.obs_dim, dom_b.reg_dim, dom_b.cls_dim, sc, &mut rng);
        sm.train(&dom_b, dom_b.train_indices(48), &cfg, &mut rng);
        assert_eq!(sm.core.checksum(), h_sc, "SpectralCore mutated during frozen transfer!");

        let mc = MlpCore::new(&mut rng);
        let h_mc = mc.checksum();
        let mut mm = MlpModel::new(dom_c.obs_dim, dom_c.reg_dim, dom_c.cls_dim, mc, &mut rng);
        let mut cfg_c = cfg.clone();
        cfg_c.w_mse = 0.0;
        mm.train(&dom_c, dom_c.train_indices(60), &cfg_c, &mut rng);
        assert_eq!(mm.core.checksum(), h_mc, "MlpCore mutated during frozen transfer!");
    }

    #[test]
    fn test_batch_var_cov_gradients() {
        use crate::models::batch_var_cov_loss_and_grad;
        let eps = 1e-6;
        let mut rng = Rng64::new(777);
        let mut zs: Vec<[f64; REL_DIM]> = (0..8)
            .map(|_| std::array::from_fn(|_| rng.normal() * 0.8))
            .collect();
        let mut dzs = vec![[0.0; REL_DIM]; zs.len()];
        let _ = batch_var_cov_loss_and_grad(&zs, 0.12, 0.20, &mut dzs);
        for i in 0..zs.len() {
            for r in (0..REL_DIM).step_by(3) {
                let orig = zs[i][r];
                zs[i][r] = orig + eps;
                let mut dummy = vec![[0.0; REL_DIM]; zs.len()];
                let lp = batch_var_cov_loss_and_grad(&zs, 0.12, 0.20, &mut dummy);
                zs[i][r] = orig - eps;
                let lm = batch_var_cov_loss_and_grad(&zs, 0.12, 0.20, &mut dummy);
                zs[i][r] = orig;
                let num = (lp - lm) / (2.0 * eps);
                assert!(
                    (num - dzs[i][r]).abs() < 1e-5,
                    "VICReg grad mismatch i={i} r={r}: num={num} ana={}",
                    dzs[i][r]
                );
            }
        }
    }
}
