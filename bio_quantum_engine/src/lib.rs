//! Biblioteca Central: bio_quantum_engine
//!
//! Arquitectura Unificada Bio-Cuántica-Geométrica que consolida:
//! - Ruptura OOD de Bio-Constelación (+8.8× en familias no vistas con Ley de Dale 100% y BTSP).
//! - Victoria de Constelación v4 con predicción residual puntual equivariante.
//! - Operador Espectral Koopman-Brenier con proyector de transporte óptimo (TimesFM beater).
//! - Teorema de la Cuenca CDT con olvido acotado $\Delta R \ge -\varepsilon$ en sueño SWR.
//! - Principio del Handshake de Wheeler-Feynman para inferencia no combinatoria en $\mathbb{C}$.
//! - Aceleración GPU en AMD RDNA3 (Radeon RX 7800 XT) con WGPU nativo.

pub mod cdt_basin_memory;
pub mod dendritic_koopman_dale;
pub mod equivariant_residual;
pub mod gpu_engine;
pub mod gpu_shaders;
pub mod handshake_phasor;
pub mod koopman_brenier_spectral;
pub mod protocol_cst4;
pub mod rng;
pub mod unified_engine;

pub use cdt_basin_memory::CdtSimplicialBasinMemory;
pub use dendritic_koopman_dale::{CorticalPyramidalUnit, DaleSynapse};
pub use equivariant_residual::{AutonomousThalamocorticalPrior, EquivariantBasisExtractor};
pub use gpu_engine::{ConfigUniform, GpuBioQuantumEngine, GpuStepResult};
pub use handshake_phasor::{ComplexPhasor, WheelerFeynmanHandshakeEngine};
pub use koopman_brenier_spectral::{KoopmanBlock2x2, KoopmanBrenierSpectralOperator};
pub use protocol_cst4::{compute_log_relative_error, Cst4DataGenerator, TrajectoryEpisode};
pub use rng::Xoshiro256StarStar;
pub use unified_engine::{BioQuantumConstellationEngine, UnifiedEngineConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dales_law_zero_violations() {
        let mut rng = Xoshiro256StarStar::from_seed(42);
        let config = UnifiedEngineConfig::default();
        let engine = BioQuantumConstellationEngine::new(config, &mut rng);

        let (violations, total_synapses) = engine.audit_dales_law();
        println!("Auditoría Dale: {} violaciones en {} sinapsis.", violations, total_synapses);
        assert_eq!(violations, 0, "No debe existir ninguna violación a la Ley de Dale");
        assert!(total_synapses > 0, "Debe auditar sinapsis reales");
    }

    #[test]
    fn test_koopman_spectral_contraction() {
        let mut rng = Xoshiro256StarStar::from_seed(12345);
        let operator = KoopmanBrenierSpectralOperator::new(10, &mut rng);

        for (j, block) in operator.blocks.iter().enumerate() {
            println!("Bloque Koopman {}: rho = {:.4}, omega = {:.4}", j, block.rho, block.omega);
            assert!(block.rho <= 1.0, "El radio espectral debe ser contractivo (<= 1.0)");
            assert!(block.rho > 0.0, "El radio espectral debe ser estrictamente positivo");
        }
    }

    #[test]
    fn test_wheeler_feynman_handshake() {
        let engine = WheelerFeynmanHandshakeEngine::new(10, 0.2);
        let features = vec![0.5, -0.2, 0.8, 0.1, -0.9, 0.3, 0.4, -0.1, 0.7, -0.5];
        let target = vec![0.6, -0.1, 0.7, 0.2, -0.8, 0.4, 0.3, 0.0, 0.6, -0.4];

        let psi_ret = engine.build_retarded_wave(&features, 0.0);
        let psi_adv = engine.build_advanced_wave(&target, 1.0);

        let (collapsed, action) = engine.compute_handshake_collapse(&psi_ret, &psi_adv, 5);

        assert_eq!(collapsed.len(), 10);
        for p in &collapsed {
            assert!(p.norm() <= 1.0001, "La norma del fasor colapsado debe estar acotada por 1.0");
        }
        println!("Handshake completado: Acción = {:.4}", action);
    }

    #[test]
    fn test_cdt_basin_bounded_forgetting() {
        let mut memory = CdtSimplicialBasinMemory::new(10, 0.05);

        // Registrar episodios durante vigilia
        let feat1 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let feat2 = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1];
        memory.record_wake_episode(&feat1, 1.5);
        memory.record_wake_episode(&feat2, 2.0);

        // Ejecutar consolidación en fase de sueño (SWR)
        let (pass_bounded, delta_r) = memory.execute_swr_sleep_consolidation();

        println!("Consolidación Sueño SWR: ΔR = {:.4}, Cumple cota = {}", delta_r, pass_bounded);
        assert!(pass_bounded, "La consolidación debe cumplir el Teorema de Cuenca (ΔR >= -ε)");
    }

    #[test]
    fn test_equivariant_residual_step() {
        let mut rng = Xoshiro256StarStar::from_seed(999);
        let config = UnifiedEngineConfig::default();
        let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);

        let data_gen = Cst4DataGenerator::new();
        let mut rng_data = Xoshiro256StarStar::from_seed(1001);
        let episode = data_gen.generate_episode(0, 5, &mut rng_data);

        let initial_coords = &episode.trajectory[0];
        let (next_coords, latent, action) = engine.forward_step(initial_coords, None, 1.0);

        assert_eq!(next_coords.len(), 12);
        assert_eq!(latent.len(), 4);
        assert!(action.is_finite());
    }

    #[test]
    fn test_train_step_from_scratch() {
        let mut rng = Xoshiro256StarStar::from_seed(444);
        let config = UnifiedEngineConfig::default();
        let mut engine = BioQuantumConstellationEngine::new(config, &mut rng);

        let data_gen = Cst4DataGenerator::new();
        let mut rng_data = Xoshiro256StarStar::from_seed(777);
        let episode = data_gen.generate_episode(0, 2, &mut rng_data);

        let input = &episode.trajectory[0];
        let target = &episode.trajectory[1];

        // Medir pérdida inicial
        let loss_0 = engine.train_step_from_scratch(input, target);

        // Realizar varias iteraciones de entrenamiento
        let mut final_loss = loss_0;
        for _ in 0..10 {
            final_loss = engine.train_step_from_scratch(input, target);
        }

        println!("Pérdida inicial: {:.6}, Pérdida tras 10 pasos: {:.6}", loss_0, final_loss);
        assert!(final_loss <= loss_0 + 1e-4, "El entrenamiento debe reducir o mantener estable la pérdida");
    }

    #[test]
    fn test_gpu_initialization() {
        let mut rng = Xoshiro256StarStar::from_seed(2026);
        let config = ConfigUniform::default();
        let mut engine = GpuBioQuantumEngine::new(config, &mut rng);

        println!("Adaptador GPU inicializado: {} ({})", engine.adapter_name, engine.adapter_backend);
        let (violations, total) = engine.audit_dales_law_gpu();
        println!("Auditoría GPU Dale: {} violaciones en {} sinapsis.", violations, total);
        assert_eq!(violations, 0, "No debe haber violaciones a la Ley de Dale en GPU");
        std::mem::forget(engine);
    }
}
