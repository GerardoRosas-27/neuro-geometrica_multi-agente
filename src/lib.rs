//! neuro-geométrica · arquitectura v4 (única en `main`).
//!
//! * **Campo v4** — `field_autonomy_v4` (harness E31–E45), `v4_*` (dataset
//!   sellado, controles, Dφ, procedencia, métricas, destilación §16, lenguaje
//!   E43/E44) y `field_v4` (motor que usa la app).
//! * **Periferia LLM** — `native_gemma2*` + `field_gemma_probe` (Gemma 2 GGUF
//!   congelado; sólo verbaliza / responde en modo crudo).
//! * **Prompt Spider** — `prompt_spider`, `spider_field*` y su router de campo
//!   (que conserva su sustrato líquido/CDT/RQM propio: `liquid_cdt_*`,
//!   `wave_predict_core`, `native_thermo*`, `entanglement`, `relational_field`).
//! * **Web** (`--features web`) — app agentica Axum.

// Campo v4
pub mod field_autonomy_v4;
pub mod field_v4;
pub mod v4_controls;
pub mod v4_dataset;
pub mod v4_distill;
pub mod v4_e45;
pub mod v4_language;
pub mod v4_metrics;
pub mod v4_provenance;
#[allow(clippy::all)]
pub mod v4_qwen2_hidden;
pub mod v4_stability;

// Periferia LLM (Gemma 2 GGUF)
pub mod field_gemma_probe;
pub mod native_gemma2;
pub mod native_gemma2_runtime;

// Prompt Spider + sustrato de su router de campo
pub mod entanglement;
pub mod liquid_cdt_memory;
pub mod liquid_cdt_rqm_fuse;
pub mod native_rng;
pub mod native_thermo_rqm_epr;
pub mod native_thermodynamic_cdt;
pub mod prompt_spider;
pub mod relational_field;
pub mod spider_field;
pub mod spider_field_experiment;
pub mod wave_predict_core;

#[cfg(feature = "web")]
pub mod web;
