//! Módulo Core Knowledge (Andamiaje Innato Spelke-Marcus).
//!
//! Reúne los pilares de sentido común ontológico:
//! 1. Física espacio-temporal y permanencia (spatiotemporal_physics.rs).
//! 2. Inferencia causal intervencionista Pearl-Marcus (causal_engine.rs).
//! 3. Psicología intencional y racionalidad teleológica (agency.rs).

pub mod spatiotemporal_physics;
pub mod causal_engine;
pub mod agency;

pub use spatiotemporal_physics::{
    PhysicalViolation, SpatiotemporalObject, SpelkePhysicsEngine, Vec3D,
};
pub use causal_engine::{CausalEngine, CausalVar, StructuralEquation};
pub use agency::{AgentGoal, CognitiveAgent};
