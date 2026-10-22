//! App web agentica sobre el campo v4: chat (decoder del campo / LLM crudo),
//! entrenamiento (dataset sellado → CDT → teacher → Dφ §16), sueño
//! (consolidación → destilación → borrado de la CDT), pruebas (smoke v4),
//! Modelos/API (Gemma local + APIs OpenAI-compatibles), Prompt Spider y
//! ÓRBITA (`orbita_job`, por proceso; degrada si faltan Python/pesos/binario).
//!
//! Activar con `--features web`. El binario `agentic_web` la consume. El LLM
//! es periferia: verbaliza el estado del campo, nunca escribe en Dφ.

pub mod api;
pub mod auth;
pub mod llm_periphery;
pub mod llm_provider;
pub mod model_fetch;
pub mod orbita_job;
pub mod process_job;
pub mod spider_field_job;
pub mod spider_job;
pub mod state;
pub mod telemetry;
pub mod train_job;

pub use api::router;
pub use state::AppState;
