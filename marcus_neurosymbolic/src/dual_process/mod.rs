//! Módulo Dual Process (Sistema 1 + Sistema 2).
//!
//! Integra el proponente asociativo rápido y el deliberador simbólico causal:
//! - system1_proposer: Generador de hipótesis asociativas.
//! - system2_deliberator: Motor formal de verificación y veto de restricciones.
//! - coordinator: Orquestador del ciclo neuro-simbólico.

pub mod system1_proposer;
pub mod system2_deliberator;
pub mod coordinator;

pub use system1_proposer::{CandidateHypothesis, System1Proposer};
pub use system2_deliberator::{DeliberationVerdict, System2Deliberator};
pub use coordinator::{DualProcessCoordinator, DualProcessResult};
