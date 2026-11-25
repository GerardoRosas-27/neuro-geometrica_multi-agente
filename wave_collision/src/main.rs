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
        eprintln!("Usage: wave_collision run-seed --seed <0xHEX|DEC> --out <DIR> [--trace]");
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
    use crate::domains::{generate_domain_b, generate_domain_c};
    use crate::metrics::spectral_spacing_r;
    use crate::models::{
        batch_var_cov_loss_and_grad, CoreTrainMode, DomainEncoder, DomainHead, MlpCore,
        MlpCoreGrads, MlpModel, PairingMode, SpectrumKind, TrainConfig, WaveCollisionCore,
        WaveCoreGrads, WaveModel, LATENT_D, N_COLLISIONS, REL_DIM,
    };
    use crate::rng::{riemann_zeros_256, Rng64, ODLYZKO_CHECKPOINTS};

    #[test]
    fn test_riemann_zeros_256_accuracy() {
        let rz = riemann_zeros_256();
        for &(idx, exact) in &ODLYZKO_CHECKPOINTS {
            let err = (rz[idx] - exact).abs();
            assert!(
                err < 1e-10,
                "Riemann zero #{idx} mismatch: got {} exact {exact} err={err}",
                rz[idx]
            );
        }
        for i in 0..(rz.len() - 1) {
            assert!(rz[i + 1] > rz[i], "Riemann zeros not strictly increasing at {i}");
        }
        for (i, &g) in rz.iter().enumerate() {
            assert!(crate::rng::hardy_z(g).abs() < 1e-10, "Non-zero |Z(gamma_{i})|");
        }
        let mut rng = Rng64::new(42);
        let rc = WaveCollisionCore::new(
            SpectrumKind::RiemannFixed,
            PairingMode::FullCollision,
            false,
            false,
            &mut rng,
        );
        let r_stat = spectral_spacing_r(&rc.omega1);
        assert!(r_stat > 0.57 && r_stat < 0.66, "Unexpected Riemann-256 <r> = {r_stat}");
    }

    #[test]
    fn test_exact_parameter_counts_and_parity() {
        let mut rng = Rng64::new(123);
        let wc = WaveCollisionCore::new(
            SpectrumKind::Adaptive,
            PairingMode::FullCollision,
            false,
            false,
            &mut rng,
        );
        let mc = MlpCore::new(&mut rng);
        assert_eq!(wc.core_param_count(), 3328);
        assert_eq!(wc.autocal_param_count(), 48);
        assert_eq!(mc.core_param_count(), 3832);
        assert_eq!(mc.autocal_param_count(), 48);

        let enc_b = DomainEncoder::new(12, &mut rng);
        let head_b = DomainHead::new(3, 4, &mut rng);
        assert_eq!(enc_b.param_count() + head_b.param_count() + wc.autocal_param_count(), 219);
        assert_eq!(enc_b.param_count() + head_b.param_count() + mc.autocal_param_count(), 219);

        let enc_c = DomainEncoder::new(16, &mut rng);
        let head_c = DomainHead::new(0, 4, &mut rng);
        assert_eq!(enc_c.param_count() + head_c.param_count() + wc.autocal_param_count(), 184);
        assert_eq!(enc_c.param_count() + head_c.param_count() + mc.autocal_param_count(), 184);
    }

    #[test]
    #[should_panic(expected = "PROTOCOL VIOLATION")]
    fn test_adaptive_rejects_riemann_injection() {
        let mut rng = Rng64::new(99);
        let wc = WaveCollisionCore::new(
            SpectrumKind::Adaptive,
            PairingMode::FullCollision,
            false,
            false,
            &mut rng,
        );
        wc.assert_no_riemann_injection(true);
    }

    #[test]
    fn test_wave_collision_gradients_all_modes_and_autocal() {
        let eps = 1e-6;
        for &mode in &[
            PairingMode::FullCollision,
            PairingMode::AdditiveOnly,
            PairingMode::PermutedPairing,
        ] {
            let mut rng = Rng64::new(901 + mode as u64);
            let mut core = WaveCollisionCore::new(SpectrumKind::Adaptive, mode, false, false, &mut rng);
            // Set non-trivial autocal values to verify all 48 autocal derivatives!
            for g in 0..REL_DIM {
                core.cal_scale[g] = 0.85 + 0.03 * (g as f64);
                core.cal_phase[g] = -0.4 + 0.06 * (g as f64);
                core.cal_rho[g] = 0.7 + 0.04 * (g as f64);
            }
            let u = [0.35, -0.55, 0.72, -0.21];
            let v = [-0.42, 0.63, -0.18, 0.81];
            let dz: [f64; REL_DIM] = std::array::from_fn(|i| ((i as f64) * 0.43 + 0.2).sin());

            let obj = |c: &WaveCollisionCore, uu: [f64; LATENT_D], vv: [f64; LATENT_D]| -> f64 {
                let cache = c.forward(uu, vv);
                (0..REL_DIM).map(|i| cache.z_rel[i] * dz[i]).sum()
            };

            let cache = core.forward(u, v);
            let mut g_full = WaveCoreGrads::zeros();
            let (du, dv) = core.backward(&cache, &dz, CoreTrainMode::FullTrain, Some(&mut g_full));
            let mut g_cal = WaveCoreGrads::zeros();
            let (du_c, dv_c) = core.backward(&cache, &dz, CoreTrainMode::AutoCalOnly, Some(&mut g_cal));
            for d in 0..LATENT_D {
                assert!((du[d] - du_c[d]).abs() < 1e-12);
                assert!((dv[d] - dv_c[d]).abs() < 1e-12);
            }

            // Check du, dv
            for d in 0..LATENT_D {
                let mut up = u;
                up[d] += eps;
                let mut um = u;
                um[d] -= eps;
                let num = (obj(&core, up, v) - obj(&core, um, v)) / (2.0 * eps);
                assert!((num - du[d]).abs() < 1e-5, "du mismatch mode={mode:?} d={d}");
                let mut vp = v;
                vp[d] += eps;
                let mut vm = v;
                vm[d] -= eps;
                let num_v = (obj(&core, u, vp) - obj(&core, u, vm)) / (2.0 * eps);
                assert!((num_v - dv[d]).abs() < 1e-5, "dv mismatch mode={mode:?} d={d}");
            }
            // Check omega1, omega2, phi1, phi2, beta, kappa, psi
            for i in (0..core.omega1.len()).step_by(37) {
                let orig = core.omega1[i];
                core.omega1[i] = orig + eps;
                let lp = obj(&core, u, v);
                core.omega1[i] = orig - eps;
                let lm = obj(&core, u, v);
                core.omega1[i] = orig;
                assert!(((lp - lm) / (2.0 * eps) - g_full.omega1[i]).abs() < 1e-5, "omega1 mismatch mode={mode:?}");

                let orig2 = core.omega2[i];
                core.omega2[i] = orig2 + eps;
                let lp2 = obj(&core, u, v);
                core.omega2[i] = orig2 - eps;
                let lm2 = obj(&core, u, v);
                core.omega2[i] = orig2;
                assert!(((lp2 - lm2) / (2.0 * eps) - g_full.omega2[i]).abs() < 1e-5, "omega2 mismatch mode={mode:?}");
            }
            for i in (0..N_COLLISIONS).step_by(19) {
                let o1 = core.phi1[i];
                core.phi1[i] = o1 + eps;
                let lp = obj(&core, u, v);
                core.phi1[i] = o1 - eps;
                let lm = obj(&core, u, v);
                core.phi1[i] = o1;
                assert!(((lp - lm) / (2.0 * eps) - g_full.phi1[i]).abs() < 1e-5, "phi1 mismatch");

                let ob = core.beta[i];
                core.beta[i] = ob + eps;
                let lpb = obj(&core, u, v);
                core.beta[i] = ob - eps;
                let lmb = obj(&core, u, v);
                core.beta[i] = ob;
                assert!(((lpb - lmb) / (2.0 * eps) - g_full.beta[i]).abs() < 1e-5, "beta mismatch");
            }
            if mode != PairingMode::AdditiveOnly {
                for i in (0..core.kappa.len()).step_by(23) {
                    let ok = core.kappa[i];
                    core.kappa[i] = ok + eps;
                    let lp = obj(&core, u, v);
                    core.kappa[i] = ok - eps;
                    let lm = obj(&core, u, v);
                    core.kappa[i] = ok;
                    assert!(((lp - lm) / (2.0 * eps) - g_full.kappa[i]).abs() < 1e-5, "kappa mismatch");

                    let op = core.psi[i];
                    core.psi[i] = op + eps;
                    let lpp = obj(&core, u, v);
                    core.psi[i] = op - eps;
                    let lmp = obj(&core, u, v);
                    core.psi[i] = op;
                    assert!(((lpp - lmp) / (2.0 * eps) - g_full.psi[i]).abs() < 1e-5, "psi mismatch");
                }
            }
            // Check all 3 autocal knob vectors (cal_scale, cal_phase, cal_rho)
            for g in (0..REL_DIM).step_by(3) {
                let os = core.cal_scale[g];
                core.cal_scale[g] = os + eps;
                let lp = obj(&core, u, v);
                core.cal_scale[g] = os - eps;
                let lm = obj(&core, u, v);
                core.cal_scale[g] = os;
                assert!(((lp - lm) / (2.0 * eps) - g_cal.cal_scale[g]).abs() < 1e-5, "cal_scale mismatch");

                let op = core.cal_phase[g];
                core.cal_phase[g] = op + eps;
                let lpp = obj(&core, u, v);
                core.cal_phase[g] = op - eps;
                let lmp = obj(&core, u, v);
                core.cal_phase[g] = op;
                assert!(((lpp - lmp) / (2.0 * eps) - g_cal.cal_phase[g]).abs() < 1e-5, "cal_phase mismatch");

                let or_ = core.cal_rho[g];
                core.cal_rho[g] = or_ + eps;
                let lpr = obj(&core, u, v);
                core.cal_rho[g] = or_ - eps;
                let lmr = obj(&core, u, v);
                core.cal_rho[g] = or_;
                assert!(((lpr - lmr) / (2.0 * eps) - g_cal.cal_rho[g]).abs() < 1e-5, "cal_rho mismatch");
            }
        }
    }

    #[test]
    fn test_mlp_film_and_regularizer_and_vicreg_gradients() {
        let eps = 1e-6;
        let mut rng = Rng64::new(555);
        // 1. Spectral regularizer
        let mut core = WaveCollisionCore::new(
            SpectrumKind::Adaptive,
            PairingMode::FullCollision,
            false,
            false,
            &mut rng,
        );
        core.omega1[0] = 2.0;
        core.omega2[4] = 0.12;
        let mut g1 = vec![0.0; core.omega1.len()];
        let mut g2 = vec![0.0; core.omega2.len()];
        let _ = core.spectral_regularizer_grad(0.004, 0.018, 10.0, &mut g1, &mut g2);
        for i in (0..core.omega1.len()).step_by(41) {
            let orig = core.omega1[i];
            core.omega1[i] = orig + eps;
            let mut d1 = vec![0.0; core.omega1.len()];
            let mut d2 = vec![0.0; core.omega2.len()];
            let lp = core.spectral_regularizer_grad(0.004, 0.018, 10.0, &mut d1, &mut d2);
            core.omega1[i] = orig - eps;
            d1.fill(0.0);
            d2.fill(0.0);
            let lm = core.spectral_regularizer_grad(0.004, 0.018, 10.0, &mut d1, &mut d2);
            core.omega1[i] = orig;
            assert!(((lp - lm) / (2.0 * eps) - g1[i]).abs() < 1e-5, "reg omega1 mismatch i={i}");
        }

        // 2. MLP + 48 FiLM knobs
        let mut mlp = MlpCore::new(&mut rng);
        for i in 0..8 {
            mlp.film_in_scale[i] = 0.8 + 0.05 * (i as f64);
            mlp.film_in_shift[i] = -0.2 + 0.04 * (i as f64);
        }
        for i in 0..REL_DIM {
            mlp.film_z_scale[i] = 1.1 - 0.02 * (i as f64);
            mlp.film_z_shift[i] = 0.15 - 0.02 * (i as f64);
        }
        let u = [0.3, -0.5, 0.7, -0.2];
        let v = [-0.4, 0.6, -0.1, 0.8];
        let dz: [f64; REL_DIM] = std::array::from_fn(|i| ((i as f64) * 0.37).sin());
        let cache = mlp.forward(u, v);
        let mut gm = MlpCoreGrads::zeros();
        let _ = mlp.backward(&cache, &dz, CoreTrainMode::AutoCalOnly, Some(&mut gm));
        let obj = |m: &MlpCore| -> f64 {
            let c = m.forward(u, v);
            (0..REL_DIM).map(|i| c.z_rel[i] * dz[i]).sum()
        };
        for i in 0..8 {
            let o = mlp.film_in_scale[i];
            mlp.film_in_scale[i] = o + eps;
            let lp = obj(&mlp);
            mlp.film_in_scale[i] = o - eps;
            let lm = obj(&mlp);
            mlp.film_in_scale[i] = o;
            assert!(((lp - lm) / (2.0 * eps) - gm.film_in_scale[i]).abs() < 1e-5);
        }
        for i in (0..REL_DIM).step_by(3) {
            let o = mlp.film_z_scale[i];
            mlp.film_z_scale[i] = o + eps;
            let lp = obj(&mlp);
            mlp.film_z_scale[i] = o - eps;
            let lm = obj(&mlp);
            mlp.film_z_scale[i] = o;
            assert!(((lp - lm) / (2.0 * eps) - gm.film_z_scale[i]).abs() < 1e-5);
        }

        // 3. VICReg covariance
        let mut zs: Vec<[f64; REL_DIM]> = (0..8)
            .map(|_| std::array::from_fn(|_| rng.normal() * 0.8))
            .collect();
        let mut dzs = vec![[0.0; REL_DIM]; zs.len()];
        let _ = batch_var_cov_loss_and_grad(&zs, 0.08, 0.15, &mut dzs);
        for i in 0..zs.len() {
            for r in (0..REL_DIM).step_by(5) {
                let orig = zs[i][r];
                zs[i][r] = orig + eps;
                let mut dummy = vec![[0.0; REL_DIM]; zs.len()];
                let lp = batch_var_cov_loss_and_grad(&zs, 0.08, 0.15, &mut dummy);
                zs[i][r] = orig - eps;
                let lm = batch_var_cov_loss_and_grad(&zs, 0.08, 0.15, &mut dummy);
                zs[i][r] = orig;
                assert!(((lp - lm) / (2.0 * eps) - dzs[i][r]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_bit_exact_frozen_waves_during_autocal_and_frozen_transfer() {
        let mut rng = Rng64::new(0xBEEF);
        let dom_b = generate_domain_b(&mut rng.fork(2));
        let dom_c = generate_domain_c(&mut rng.fork(3));

        let mut cfg = TrainConfig {
            steps: 20,
            batch_size: 16,
            lr_encoder: 0.02,
            lr_core: 0.02,
            lr_autocal: 0.02,
            lr_head: 0.02,
            wd_encoder: 1e-2,
            wd_core: 1e-3,
            wd_autocal: 1e-2,
            wd_head: 1e-2,
            w_mse: 1.0,
            w_ce: 0.9,
            w_ssl: 0.0,
            lambda_vec: 0.004,
            lambda_rad: 0.012,
            lambda_bw: 8.0,
            lambda_var: 0.0,
            lambda_cov: 0.0,
            latent_norm_reg: 0.05,
            latent_jitter: 0.0,
            core_mode: CoreTrainMode::AutoCalOnly,
        };

        let wc = WaveCollisionCore::new(
            SpectrumKind::Adaptive,
            PairingMode::FullCollision,
            false,
            false,
            &mut rng,
        );
        let h_waves = wc.frozen_waves_checksum();
        let mut wm = WaveModel::new(dom_b.obs_dim, dom_b.reg_dim, dom_b.cls_dim, wc.clone(), &mut rng);
        wm.train(&dom_b, dom_b.train_indices(48), &cfg, &mut rng);
        assert_eq!(wm.core.frozen_waves_checksum(), h_waves, "Core waves mutated during AutoCalOnly!");

        cfg.core_mode = CoreTrainMode::FullyFrozen;
        let h_full = wc.full_checksum();
        let mut wm_f = WaveModel::new(dom_b.obs_dim, dom_b.reg_dim, dom_b.cls_dim, wc, &mut rng);
        wm_f.train(&dom_b, dom_b.train_indices(48), &cfg, &mut rng);
        assert_eq!(wm_f.core.full_checksum(), h_full, "Core mutated during FullyFrozen!");

        cfg.core_mode = CoreTrainMode::AutoCalOnly;
        cfg.w_mse = 0.0;
        let mc = MlpCore::new(&mut rng);
        let h_trunk = mc.frozen_trunk_checksum();
        let mut mm = MlpModel::new(dom_c.obs_dim, dom_c.reg_dim, dom_c.cls_dim, mc, &mut rng);
        mm.train(&dom_c, dom_c.train_indices(60), &cfg, &mut rng);
        assert_eq!(mm.core.frozen_trunk_checksum(), h_trunk, "MLP trunk mutated during AutoCalOnly!");
    }
}
