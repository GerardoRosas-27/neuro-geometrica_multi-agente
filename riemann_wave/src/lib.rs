pub mod dyson;
pub mod linalg;
pub mod riemann;
pub mod rng;
pub mod suites;

#[cfg(test)]
mod tests {
    use crate::dyson::{
        consolidate_dyson, consolidate_ginibre_2d_rff, ks_distance, poisson_cdf, wigner_gue_cdf,
    };
    use crate::linalg::{hermitian_eigenvalues, Mat, RlsState, ThinSvd};
    use crate::riemann::{
        compute_riemann_zeros, evaluate_prime_wave_discrimination, hardy_z_complex, C64,
    };
    use crate::rng::{erf, inv_normal_cdf, normal_cdf, Rng};

    #[test]
    fn test_riemann_zeros_match_odlyzko() {
        // First 10 non-trivial zeros of zeta(1/2 + i*gamma) to 10 decimal places (Odlyzko table)
        let ref_zeros = [
            14.134_725_141_7,
            21.022_039_638_8,
            25.010_857_580_1,
            30.424_876_125_9,
            32.935_061_587_7,
            37.586_178_158_8,
            40.918_719_012_1,
            43.327_073_280_9,
            48.005_150_881_2,
            49.773_832_477_7,
        ];
        let zeros = compute_riemann_zeros(16);
        assert_eq!(zeros.len(), 16);
        for (i, (&got, &expected)) in zeros.iter().zip(ref_zeros.iter()).enumerate() {
            let err = (got - expected).abs();
            assert!(
                err < 1e-9,
                "Zero #{} mismatch: got {:.12}, expected {:.12}, err={:.3e}",
                i + 1,
                got,
                expected,
                err
            );
            // Verify Hardy Z(t) imaginary part vanishes (functional equation consistency)
            let zc = hardy_z_complex(got + 0.37);
            assert!(
                zc.im.abs() < 1e-11,
                "Hardy Z(t) imaginary part not zero at t={}: {:.3e}",
                got + 0.37,
                zc.im
            );
        }
    }

    #[test]
    fn test_prime_wave_interference_discriminates_prime_powers() {
        let zeros = compute_riemann_zeros(128);
        let (auc, snr) = evaluate_prime_wave_discrimination(&zeros, 60);
        assert!(auc > 0.96, "Expected prime-power AUC > 0.96, got {}", auc);
        assert!(snr > 3.0, "Expected prime-power SNR > 3.0, got {}", snr);
    }

    #[test]
    fn test_erf_and_inv_normal_cdf_accuracy() {
        assert!((erf(0.0)).abs() < 1e-15);
        let refs = [
            (0.5, 0.520_499_877_813_046_52),
            (1.0, 0.842_700_792_949_714_89),
            (2.5, 0.999_593_047_982_554_99),
            (3.5, 0.999_999_256_901_627_61),
            (3.8, 0.999_999_922_996_072_54),
            (5.0, 0.999_999_999_998_462_56),
        ];
        for &(x, expected) in &refs {
            let got = erf(x);
            assert!(
                (got - expected).abs() < 1e-13,
                "erf({}) mismatch: got {:.16}, expected {:.16}",
                x,
                got,
                expected
            );
            assert!((erf(-x) + got).abs() < 1e-15);
        }
        assert!((wigner_gue_cdf(0.0)).abs() < 1e-15);
        assert!((wigner_gue_cdf(10.0) - 1.0).abs() < 1e-12);
        for &p in &[1e-5, 0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995, 1.0 - 1e-5] {
            let x = inv_normal_cdf(p);
            let p_back = normal_cdf(x);
            assert!(
                (p_back - p).abs() < 1e-12,
                "inv_normal_cdf roundtrip failed at p={}: err={:.3e}",
                p,
                (p_back - p).abs()
            );
        }
    }

    #[test]
    fn test_hermitian_jacobi_eigensolver_exactness() {
        let mut rng = Rng::from_seed(0xCAFE, 1);
        let n = 16;
        let mut h = vec![C64::new(0.0, 0.0); n * n];
        for r in 0..n {
            h[r * n + r] = C64::new(rng.normal(), 0.0);
            for c in (r + 1)..n {
                let re = rng.normal();
                let im = rng.normal();
                h[r * n + c] = C64::new(re, im);
                h[c * n + r] = C64::new(re, -im);
            }
        }
        let trace_in: f64 = (0..n).map(|i| h[i * n + i].re).sum();
        let (evals, v) = hermitian_eigenvalues(&h, n);
        let trace_out: f64 = evals.iter().sum();
        assert!((trace_in - trace_out).abs() < 1e-10);

        // Verify V^H V = I and H V = V diag(evals)
        let mut unit_err = 0.0;
        let mut eigen_err = 0.0;
        for c1 in 0..n {
            for c2 in 0..n {
                let mut dot = C64::new(0.0, 0.0);
                for r in 0..n {
                    dot = dot.add(v[r * n + c1].conj().mul(v[r * n + c2]));
                }
                let target = if c1 == c2 { 1.0 } else { 0.0 };
                unit_err += (dot.re - target).powi(2) + dot.im.powi(2);
            }
            for r in 0..n {
                let mut hv = C64::new(0.0, 0.0);
                for k in 0..n {
                    hv = hv.add(h[r * n + k].mul(v[k * n + c1]));
                }
                let lv = v[r * n + c1].scale(evals[c1]);
                let d = hv.sub(lv);
                eigen_err += d.norm_sq();
            }
        }
        assert!(unit_err.sqrt() < 1e-10, "Unitarity err={}", unit_err.sqrt());
        assert!(eigen_err.sqrt() < 1e-10, "Eigen residual={}", eigen_err.sqrt());
    }

    #[test]
    fn test_thin_svd_ols_ridge_and_cond() {
        let mut rng = Rng::from_seed(0xBEEF, 2);
        let m = 40;
        let n = 10;
        let mut a = Mat::zeros(m, n);
        for r in 0..m {
            for c in 0..n {
                a.set(r, c, rng.normal());
            }
        }
        // Make column 9 nearly collinear with column 0 to test high condition number
        for r in 0..m {
            let v = a.get(r, 0) + 1e-5 * rng.normal();
            a.set(r, n - 1, v);
        }
        let w_true: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.25).collect();
        let y = a.mul_vec(&w_true);

        let svd = ThinSvd::compute(&a);
        assert!(svd.cond() > 1e4, "Expected high cond, got {}", svd.cond());
        let w_ols = svd.solve_ols(&y);
        for i in 0..n {
            assert!(
                (w_ols[i] - w_true[i]).abs() < 1e-6,
                "OLS recovery mismatch at {}: {} vs {}",
                i,
                w_ols[i],
                w_true[i]
            );
        }

        // Verify Ridge satisfies normal equations (A^T A + lam I) w = A^T y
        let lam = 0.25;
        let w_ridge = svd.solve_ridge(&y, lam);
        let aw = a.mul_vec(&w_ridge);
        for c in 0..n {
            let mut lhs = lam * w_ridge[c];
            let mut rhs = 0.0;
            for r in 0..m {
                lhs += a.get(r, c) * aw[r];
                rhs += a.get(r, c) * y[r];
            }
            assert!(
                (lhs - rhs).abs() < 1e-10,
                "Ridge normal eq mismatch at {}: {:.3e}",
                c,
                (lhs - rhs).abs()
            );
        }
    }

    #[test]
    fn test_rls_matches_batch_ridge() {
        let mut rng = Rng::from_seed(0x1234, 3);
        let m = 30;
        let n = 6;
        let alpha = 10.0; // P_0 = alpha I <=> lambda = 1/alpha = 0.1
        let mut a = Mat::zeros(m, n);
        let mut y = vec![0.0; m];
        let mut rls = RlsState::new(n, alpha);
        for r in 0..m {
            let mut row = vec![0.0; n];
            for c in 0..n {
                row[c] = rng.normal();
                a.set(r, c, row[c]);
            }
            y[r] = rng.normal();
            rls.update(&row, y[r]);
        }
        let svd = ThinSvd::compute(&a);
        let w_batch = svd.solve_ridge(&y, 1.0 / alpha);
        for i in 0..n {
            assert!(
                (rls.w[i] - w_batch[i]).abs() < 1e-10,
                "RLS != batch Ridge at {}: {} vs {}",
                i,
                rls.w[i],
                w_batch[i]
            );
        }
    }

    #[test]
    fn test_dyson_and_ginibre_stability_and_determinism() {
        let mut r1 = Rng::from_seed(0xA700, 10);
        let mut r2 = Rng::from_seed(0xA700, 10);
        let d1 = consolidate_dyson(40, 2000, 0.002, 2.0, false, &mut r1);
        let d2 = consolidate_dyson(40, 2000, 0.002, 2.0, false, &mut r2);
        assert_eq!(d1.frequencies, d2.frequencies, "Dyson must be bit-exact deterministic");
        assert!(d1.frequencies.iter().all(|v| v.is_finite()));
        let ks_gue = ks_distance(&d1.equilibrium_spacings, wigner_gue_cdf);
        let ks_poi = ks_distance(&d1.equilibrium_spacings, poisson_cdf);
        assert!(
            ks_gue < ks_poi,
            "Dyson beta=2 must be closer to Wigner GUE ({}) than Poisson ({})",
            ks_gue,
            ks_poi
        );

        let g1 = consolidate_ginibre_2d_rff(32, 1.0, 500, 0.004, &mut r1);
        let g2 = consolidate_ginibre_2d_rff(32, 1.0, 500, 0.004, &mut r2);
        assert_eq!(g1, g2, "Ginibre 2D must be bit-exact deterministic");
        assert!(g1.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
    }
}
