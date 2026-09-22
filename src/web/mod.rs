//! App web agentica: chat + telemetría líquido / CDT / RQM.
//!
//! Activar con `--features web`. El binario `agentic_web` la consume.
//! Arquitectura: Líquido = inferencia rápida; CDT = memoria post-sueño;
//! RQM = índice relacional / fallback; LLM = solo periferia texto↔concepto
//! (los token_ids **nunca** entran a FieldState).

pub mod api;
pub mod llm_periphery;
pub mod state;
pub mod telemetry;

pub use api::router;
pub use state::AppState;
