//! Isolated experiment crate: Retrodictive origin tracing & bidirectional inference
//! via Riemann prime-wave collapse vs random collapse, Dyson GUE, and linear Fourier.

pub mod inference;
pub mod riemann;
pub mod rng;
pub mod suites;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::inference::{
        all_distinct_pairs, eval_causal_outcome, min_prime_product_log_gap, trace_origin,
        CollapseArm, SmoothPriorField, WaveField, N_MODES,
    };
    use crate::riemann::{compute_riemann_zeros, first_k_primes};
    use crate::rng::Rng;

    #[test]
    fn test_riemann_zeros_accuracy_sub_1e9() {
        let zeros = compute_riemann_zeros(20);
        assert_eq!(zeros.len(), 20);
        let expected = [
            (0, 14.134_725_141_734_695),
            (1, 21.022_039_638_771_556),
            (2, 25.010_857_580_145_69),
            (9, 49.773_832_477_672_3),
            (19, 77.144_840_068_874_8),
        ];
        for (idx, val) in expected {
            let err = (zeros[idx] - val).abs();
            assert!(
                err < 1e-9,
                "zero[{}] = {}, expected {}, err = {}",
                idx,
                zeros[idx],
                val,
                err
            );
        }
    }

    #[test]
    fn test_unique_prime_products_vs_linear_collisions() {
        let primes = first_k_primes(12);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]);
        let pairs = all_distinct_pairs(12);
        assert_eq!(pairs.len(), 66);

        let prime_products: HashSet<usize> = pairs
            .iter()
            .map(|&(a, b)| primes[a] * primes[b])
            .collect();
        assert_eq!(
            prime_products.len(),
            66,
            "All 66 prime pairs must have unique products (0 collisions)"
        );

        let linear_sums: HashSet<usize> = pairs
            .iter()
            .map(|&(a, b)| (a + 1) + (b + 1))
            .collect();
        assert_eq!(
            linear_sums.len(),
            21,
            "66 linear pairs in 1..=12 must collapse into only 21 sums"
        );
    }

    #[test]
    fn test_noiseless_riemann_origin_recovery_ge_98pct() {
        let zeros = compute_riemann_zeros(N_MODES);
        let mut rng_b = Rng::from_seed(0x1234, 1);
        let field = WaveField::build(CollapseArm::RiemannPrime, 12, &zeros, &mut rng_b);
        let pairs = all_distinct_pairs(12);

        let mut tie_rng = Rng::from_seed(0x1234, 2);
        let mut hits = 0usize;
        for &(a, b) in &pairs {
            let u_obs = field.compose_pair(a, b);
            let res = trace_origin(&field, u_obs, Some((a, b)), None, &mut tie_rng);
            assert!(res.peak_score.is_finite() && res.peak_margin_snr.is_finite());
            if res.pred_a == a && res.pred_b == b {
                hits += 1;
            }
        }
        let acc = (hits as f64) / (pairs.len() as f64);
        assert!(
            acc >= 0.98,
            "Noiseless Riemann prime collapse origin acc = {:.4} ({}/66), expected >= 0.98",
            acc,
            hits
        );
    }

    #[test]
    fn test_wave_interpolation_matches_direct_sum() {
        let zeros = compute_riemann_zeros(N_MODES);
        let mut rng = Rng::from_seed(0x5678, 1);
        let field = WaveField::build(CollapseArm::RiemannPrime, 12, &zeros, &mut rng);

        for &u in &[0.693_147, 1.609_438, 2.397_895, 3.610_918] {
            let direct: f64 = -2.0
                * (0..N_MODES)
                    .map(|n| (1.0 - (n as f64) / (N_MODES as f64)) * (zeros[n] * u).cos())
                    .sum::<f64>();
            let interp = field.eval_raw(u);
            let err = (direct - interp).abs();
            assert!(
                err < 0.05,
                "Interpolation drift at u={}: direct={}, interp={}, err={}",
                u,
                direct,
                interp,
                err
            );
        }
    }

    #[test]
    fn test_smooth_prior_is_bounded_and_spike_free() {
        let primes = first_k_primes(12);
        let coords: Vec<f64> = primes.iter().map(|&p| (p as f64).ln()).collect();
        let mut probs: Vec<f64> = (1..=12).map(|k| (k as f64).powf(-1.4)).collect();
        let s: f64 = probs.iter().sum();
        for p in &mut probs {
            *p /= s;
        }
        let min_p = probs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let field = SmoothPriorField::new(&coords, &probs, 0.03);
        for i in 0..500 {
            let u = 0.55 + (i as f64) * (3.15 / 499.0);
            let v = field.eval_single(u);
            assert!(
                v >= min_p - 1e-12 && v <= max_p + 1e-12,
                "SmoothPriorField out of [min_p, max_p] at u={}: {}",
                u,
                v
            );
        }
    }

    #[test]
    fn test_capacity_min_log_gap_shrinks_monotonically() {
        let g8 = min_prime_product_log_gap(8);
        let g12 = min_prime_product_log_gap(12);
        let g18 = min_prime_product_log_gap(18);
        assert!(g8 > g12 && g12 > g18 && g18 > 0.0, "gaps: {} {} {}", g8, g12, g18);
    }

    #[test]
    fn test_deterministic_replay_and_causal_law_finite() {
        let zeros = compute_riemann_zeros(N_MODES);
        let mut r1 = Rng::from_seed(0xA900, 99);
        let mut r2 = Rng::from_seed(0xA900, 99);
        let f1 = WaveField::build(CollapseArm::Random, 12, &zeros, &mut r1);
        let f2 = WaveField::build(CollapseArm::Random, 12, &zeros, &mut r2);

        for k in 0..12 {
            let u = f1.coords[k];
            assert_eq!(f1.eval_cavity(u).to_bits(), f2.eval_cavity(u).to_bits());
        }
        for &(a, b) in &all_distinct_pairs(12) {
            let (y1, y2, cls) = eval_causal_outcome(a, b);
            assert!(y1.is_finite() && y2.is_finite() && cls < 4);
        }
    }
}
