//! neuro-geométrica · arquitectura v4 (única en `main`).
//!
//! * **Campo v4** — `field_autonomy_v4` (harness E31–E45), `v4_*` (dataset
//!   sellado, controles, Dφ, procedencia, métricas, destilación §16, lenguaje
//!   E43/E44) y `field_v4` (motor que usa la app).
//! * **Periferia LLM** — `native_gemma2*` + `field_gemma_probe` (Gemma 2 GGUF
//!   congelado; sólo verbaliza / responde en modo crudo).
//! * **Prompt Spider** — `prompt_spider` y `spider_field*`: su router de campo
//!   también es v4 (CDT → teacher → Dφ gated con replay de confianza).
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
/// Infra: deterministic intra-seed training checkpoints (v5 design).
pub mod train_ckpt;
/// v6: hipótesis H1–H7 (docs/hipotesis_math.md) sobre la receta v4 §16.
pub mod v6_hypo;

// Periferia LLM (Gemma 2 GGUF)
pub mod field_gemma_probe;
pub mod native_gemma2;
pub mod native_gemma2_runtime;

// Prompt Spider (router de campo v4)
pub mod prompt_spider;
pub mod spider_field;
pub mod spider_field_experiment;

#[cfg(feature = "web")]
pub mod web;
