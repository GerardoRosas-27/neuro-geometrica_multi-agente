//! Módulo de Arquitectura de Energía Libre e Inferencia Activa (Free Energy Principle & Active Inference).
//!
//! Implementación rigurosa basada 100% en:
//! 1. Maximización de la Energía Libre de Evidencia (ELBO = -F(q, o)).
//! 2. Reducción de la Sorpresa Estructurada (Perceptual, Algebraica, Física y Causal).
//! 3. Inferencia Activa mediante selección de acciones causales do(X) que minimizan G(pi).
//! 4. Síntesis total con los principios cognitivos de Gary Marcus (The Algebraic Mind, Rebooting AI, Taming Silicon Valley).

pub mod variational_free_energy;
pub mod surprise_engine;
pub mod precision_dynamics;
pub mod active_inference;
pub mod fep_coordinator;
pub mod fep_substrate;

pub use variational_free_energy::*;
pub use surprise_engine::*;
pub use precision_dynamics::*;
pub use active_inference::*;
pub use fep_coordinator::*;
pub use fep_substrate::*;
