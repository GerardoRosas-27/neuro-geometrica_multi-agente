//! # MARCUS-I: Cognitive Neuro-Symbolic Architecture
//!
//! Sustrato cognitivo neuro-simbólico inspirado en las teorías del Dr. Gary Marcus:
//! - "The Algebraic Mind: Integrating Connectionism and Cognitive Science" (MIT Press, 2001)
//! - "Rebooting AI: Building Artificial Intelligence We Can Trust" (Pantheon, 2019)
//! - "Taming Silicon Valley: How We Can Organize to Constrain the Future of AI" (MIT Press, 2024)
//! - "The Next Decade in AI: Four Steps Towards Robust Artificial Intelligence" (2020)
//!
//! ## Módulos:
//! 1. `algebraic_mind`: Variables libres, ligamiento dinámico, HRR/VSA, sintaxis composicional y distinción Tipo-Token.
//! 2. `core_knowledge`: Andamiaje innato Spelke (física espacio-temporal, permanencia, gravedad) e inferencia causal Pearl (do-calculus).
//! 3. `dual_process`: Acoplamiento dialéctico entre Sistema 1 (proponente heurístico asociativo) y Sistema 2 (deliberador formal y veto).
//! 4. `epistemic_vigilance`: Estado del mundo continuo, trazabilidad de procedencia y centinela anti-alucinaciones.
//! 5. `benchmarks`: Batería de 5 experimentos preregistrados de validación empírica.

pub mod algebraic_mind;
pub mod core_knowledge;
pub mod dual_process;
pub mod epistemic_vigilance;
pub mod benchmarks;
pub mod connectors;

pub use algebraic_mind::*;
pub use core_knowledge::*;
pub use dual_process::*;
pub use epistemic_vigilance::*;
pub use benchmarks::*;
pub use connectors::*;

/// Motor Cognitivo Unificado MARCUS-I (Facade de Alto Nivel).
pub struct MarcusEngine {
    pub type_registry: TypeTokenRegistry,
    pub physics_engine: SpelkePhysicsEngine,
    pub causal_engine: CausalEngine,
    pub world_state: WorldStateTracker,
    pub system1: System1Proposer,
    pub coordinator: DualProcessCoordinator,
}

impl Default for MarcusEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MarcusEngine {
    pub fn new() -> Self {
        Self {
            type_registry: TypeTokenRegistry::new(),
            physics_engine: SpelkePhysicsEngine::new(),
            causal_engine: CausalEngine::new(),
            world_state: WorldStateTracker::new(),
            system1: System1Proposer::new(),
            coordinator: DualProcessCoordinator::new(),
        }
    }

    /// Procesa una consulta mediante el ciclo completo neuro-simbólico Sistema 1 + Sistema 2.
    pub fn process_query(&mut self, query: &str) -> DualProcessResult {
        let mut deliberator = System2Deliberator::new(
            &self.physics_engine,
            &mut self.causal_engine,
            &self.type_registry,
        );
        self.coordinator.execute_cycle(query, &self.system1, &mut deliberator)
    }

    /// Evalúa la consistencia física continua en el intervalo dt.
    pub fn verify_physics(
        &mut self,
        objects: &[SpatiotemporalObject],
        dt: f64,
    ) -> Vec<PhysicalViolation> {
        self.physics_engine.step_and_verify(objects, dt)
    }

    /// Evalúa una afirmación contra el centinela anti-alucinaciones.
    pub fn audit_claim(
        &self,
        claim_id: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> HallucinationGateDecision {
        let sentinel = HallucinationSentinel::new(&self.world_state);
        sentinel.evaluate_claim(claim_id, subject, predicate, object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marcus_engine_end_to_end() {
        let mut engine = MarcusEngine::new();

        // 1. Registrar hecho en el estado del mundo
        engine.world_state.insert_fact(
            "TamingSiliconValley",
            "Author",
            "GaryMarcus",
            EpistemicCertainty::ObservedFact,
        );

        // 2. Comprobar auditoría del centinela
        let decision = engine.audit_claim(
            "test_1",
            "TamingSiliconValley",
            "Author",
            "GaryMarcus",
        );
        assert!(matches!(decision, HallucinationGateDecision::Approved { .. }));

        // 3. Comprobar que veta alucinación
        let fake_decision = engine.audit_claim(
            "test_2",
            "TamingSiliconValley",
            "Author",
            "UnknownEntity",
        );
        assert!(matches!(fake_decision, HallucinationGateDecision::Vetoed { .. }));
    }
}
