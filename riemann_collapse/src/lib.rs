pub mod collapse;
pub mod riemann;
pub mod rng;
pub mod suites;

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::collapse::{
        eval_future_wave_alone, eval_naive_bilinear, eval_past_wave_alone,
        eval_range_discrimination, eval_transactional_collapse, eval_transactional_explicit_complex,
        CollapseConfig, GuinandWeilField,
    };
    use crate::riemann::{compute_riemann_zeros, hardy_z_complex, is_prime_int, von_mangoldt_table};

    #[test]
    fn test_riemann_zeros_match_odlyzko_1e9() {
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
        for (i, (&got, &exp)) in zeros.iter().zip(ref_zeros.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "Zero #{} mismatch: got {:.12} vs {:.12}",
                i + 1,
                got,
                exp
            );
            assert!(hardy_z_complex(got + 0.31).im.abs() < 1e-11);
        }
    }

    #[test]
    fn test_transactional_complex_overlap_identity() {
        let zeros = compute_riemann_zeros(32);
        let cfg = CollapseConfig {
            tau: 0.07,
            k_sum: 1.03,
            delta_phi: 2.1,
            sigma_tot: 0.42,
            beta_disp: 0.004,
            p_win: 1.3,
            v_asym: -0.15,
        };
        for &x in &[2.0, 7.0, 19.3, 47.0, 97.0] {
            let a = eval_transactional_collapse(x, &zeros, &cfg);
            let b = eval_transactional_explicit_complex(x, &zeros, &cfg);
            assert!(
                (a - b).abs() < 1e-11,
                "Complex transactional identity mismatch at x={}: {} vs {}",
                x,
                a,
                b
            );
        }
    }

    #[test]
    fn test_strict_emergence_past_future_neither_alone_has_primes() {
        let zeros = compute_riemann_zeros(128);
        let cfg = CollapseConfig::analytic_ref(0.5);
        let m_col = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, &zeros, &cfg));
        let m_past = eval_range_discrimination(2, 100, |x| eval_past_wave_alone(x, &zeros, &cfg));
        let m_fut = eval_range_discrimination(2, 100, |x| eval_future_wave_alone(x, &zeros, &cfg));
        let m_naive = eval_range_discrimination(2, 100, |x| eval_naive_bilinear(x, &zeros, &cfg));

        assert!(m_col.prime_only_auc > 0.99, "Collapse prime AUC={}", m_col.prime_only_auc);
        assert!(m_past.prime_only_auc < 0.55, "Past alone AUC={}", m_past.prime_only_auc);
        assert!(m_fut.prime_only_auc < 0.55, "Future alone AUC={}", m_fut.prime_only_auc);
        assert!(m_naive.prime_power_auc < 0.60, "Naive bilinear AUC={}", m_naive.prime_power_auc);

        // Phase inversion check: delta_phi = 0 flips all prime peaks into exact troughs
        let mut inv_cfg = cfg;
        inv_cfg.delta_phi = 0.0;
        let m_inv = eval_range_discrimination(2, 100, |x| eval_transactional_collapse(x, &zeros, &inv_cfg));
        assert!(
            ((m_col.prime_power_auc + m_inv.prime_power_auc) - 1.0).abs() < 1e-12,
            "Exact phase inversion symmetry failed: {} + {}",
            m_col.prime_power_auc,
            m_inv.prime_power_auc
        );
        assert!(m_inv.prime_power_auc < 0.03, "Expected inverted AUC < 0.03, got {}", m_inv.prime_power_auc);
        assert!((cfg.delta_phi - PI).abs() < 1e-12);
    }

    #[test]
    fn test_guinand_weil_gradient_and_blind_peaks_on_true_zeros() {
        let field = GuinandWeilField::from_sieve(600);
        let h = 1e-6;
        for &w in &[14.13, 20.5, 31.2, 55.7, 77.0] {
            let (_, grad_analytic) = field.eval_and_grad(w);
            let (vp, _) = field.eval_and_grad(w + h);
            let (vm, _) = field.eval_and_grad(w - h);
            let grad_fd = (vp - vm) / (2.0 * h);
            assert!(
                (grad_analytic - grad_fd).abs() < 1e-7,
                "GW grad mismatch at w={}: analytic={} fd={}",
                w,
                grad_analytic,
                grad_fd
            );
        }

        // Verify pure prime-sieve R(w) peaks tightly around the true first 10 Riemann zeros
        let zeros = compute_riemann_zeros(10);
        for (i, &gz) in zeros.iter().enumerate() {
            let (val_at_z, _) = field.eval_and_grad(gz);
            let (val_left, _) = field.eval_and_grad(gz - 0.5);
            let (val_right, _) = field.eval_and_grad(gz + 0.5);
            assert!(
                val_at_z > 2.0 && val_at_z > val_left && val_at_z > val_right,
                "Expected blind GW peak at zero #{} ({:.4}): val={:.3}, left={:.3}, right={:.3}",
                i + 1,
                gz,
                val_at_z,
                val_left,
                val_right
            );
        }
    }

    #[test]
    fn test_sieve_and_von_mangoldt_exactness() {
        let lam = von_mangoldt_table(30);
        assert_eq!(lam[1], 0.0);
        assert!((lam[2] - 2.0_f64.ln()).abs() < 1e-14);
        assert!((lam[4] - 2.0_f64.ln()).abs() < 1e-14);
        assert!((lam[8] - 2.0_f64.ln()).abs() < 1e-14);
        assert!((lam[9] - 3.0_f64.ln()).abs() < 1e-14);
        assert_eq!(lam[6], 0.0);
        assert_eq!(lam[12], 0.0);
        assert_eq!(lam[30], 0.0);
        assert!(is_prime_int(29));
        assert!(!is_prime_int(27));
    }
}
