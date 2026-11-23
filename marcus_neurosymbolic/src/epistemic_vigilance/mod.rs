//! Módulo de Vigilancia Epistémica (Epistemic Vigilance).
//!
//! Basado en Gary Marcus (Taming Silicon Valley, 2024):
//! - world_state: Estado continuo de la verdad objetiva.
//! - provenance: Árbol de trazabilidad y auditoría de decisiones.
//! - hallucination_filter: Centinela de veto anti-alucinaciones.

pub mod world_state;
pub mod provenance;
pub mod hallucination_filter;

pub use world_state::{EpistemicCertainty, GroundTruthFact, WorldStateTracker};
pub use provenance::{EpistemicProvenanceTree, ProvenanceStep};
pub use hallucination_filter::{HallucinationGateDecision, HallucinationSentinel};
