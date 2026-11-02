#![allow(clippy::needless_range_loop)]

//! Crate Bio-Constelación: Arquitectura Híbrida Neurobiológica y Geométrica.
//!
//! Fusiona los operadores dinámicos de Koopman y predicción residual de Constelación v4
//! con el microcircuito piramidal compartimental, plasticidad BTSP, memoria hipocampal CLS
//! y consolidación en sueño SWR de Neocortex BioBrain.

pub mod dale_stp_synapse;
pub mod dendritic_koopman;
pub mod hippocampus_cdt_sleep;
pub mod hybrid_brain;
pub mod protocol_cst4;
pub mod rng;

#[cfg(test)]
mod tests {
    use super::*;
    use dale_stp_synapse::DaleSynapseLayer;
    use dendritic_koopman::DendriticKoopmanNeuron;
    use hybrid_brain::{BioConstelacion, HybridConfig};
    use rng::Xoshiro256StarStar;

    #[test]
    fn test_strict_dale_law_compliance() {
        let mut rng = Xoshiro256StarStar::new(42);
        let mut layer = DaleSynapseLayer::new(2, 10, 0.8, 1.0, 0.2, &mut rng);

        // Actualizar con gradientes fuertes
        let grad = vec![5.0; 20];
        layer.update_dale_three_factor(&grad, 1.0, 0.1, 1e-4, 1.0, 5.0, 0.1, 10.0, true);

        let (violations, total) = layer.audit_dale_law();
        assert_eq!(
            violations, 0,
            "Se detectaron violaciones a la Ley de Dale en la sinapsis"
        );
        assert!(total > 0);
    }

    #[test]
    fn test_koopman_contraction_bound() {
        let mut rng = Xoshiro256StarStar::new(101);
        let neuron = DendriticKoopmanNeuron::new(&mut rng, 4, 8, 0.05);

        let max_scale = neuron.spectral_radius();
        assert!(
            max_scale <= 1.0,
            "El radio espectral del bloque Koopman supera 1.0: {}",
            max_scale
        );
    }

    #[test]
    fn test_hippocampus_dg_sparsity() {
        let mut rng = Xoshiro256StarStar::new(202);
        let hippo = hippocampus_cdt_sleep::HybridHippocampusCdt::new(64, 32, 128, 0.01, &mut rng);
        let context = [0.5; 10];

        let k = 8;
        let dg_sparse = hippo.encode_dg(&context, k);

        let non_zeros = dg_sparse.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert_eq!(
            non_zeros, k,
            "El Giro Dentado debe producir exactamente k={} activaciones no nulas",
            k
        );
    }

    #[test]
    fn test_hybrid_brain_rollout_stability() {
        let mut rng = Xoshiro256StarStar::new(303);
        let config = HybridConfig::default();
        let mut brain = BioConstelacion::new(config, &mut rng);

        let q0 = vec![
            0.5, 0.0, 0.25, 0.43, -0.25, 0.43, -0.5, 0.0, -0.25, -0.43, 0.25, -0.43,
        ];
        let traj = brain.rollout(&q0, 16);

        assert_eq!(traj.len(), 16);
        for frame in &traj {
            assert_eq!(frame.len(), 12);
            for val in frame {
                assert!(
                    val.is_finite(),
                    "Se detectó valor NaN o Infinito en rollout"
                );
                assert!(val.abs() < 50.0, "La dinámica residual explotó: {}", val);
            }
        }
    }
}
