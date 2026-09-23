//! App web agentica: chat + telemetría líquido / CDT / RQM + entrenamiento en vivo.
//!
//! Activar con `--features web`. El binario `agentic_web` la consume.
//! Arquitectura: Líquido = inferencia rápida (`WavePredictCore`); CDT = memoria
//! post-sueño por lotes; RQM = índice relacional / fallback;
//! LLM = **decoder only** del campo (+ generación de dataset en periferia).
//! Los token_ids **nunca** entran a FieldState.

pub mod api;
pub mod experiment_suite;
pub mod field_eval;
pub mod llm_periphery;
pub mod process_job;
pub mod sleep_optimize;
pub mod state;
pub mod telemetry;
pub mod train_job;

pub use api::router;
pub use state::AppState;
