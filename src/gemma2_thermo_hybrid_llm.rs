//! LLM híbrido semántico: Gemma 2 como periférico lingüístico + motor CTP nativo.
//!
//! ```text
//! Token ──> W_emb·√d ──> [RFF + Softmax CTP] ──> Motor termodinámico ──> Φ_colapso
//!                                                              │
//!                    Softmax(tanh(Φ·W_emb^T / cap)·cap) <──────┘
//! ```
//!
//! La GPU/Candle ejecuta embeddings, proyección de salida y Softmax del vocabulario.
//! El núcleo termodinámico (CPU) resuelve contexto, interferencia y consolidación CDT.

use crate::hybrid_thermo_attention::{
    HybridThermoAttention, HybridThermoAttentionConfig, HybridThermoAttentionError,
    HybridThermoAttentionReport, PhasorRffConfig, ThermoAttentionLearnedState,
};
use crate::native_gemma2::{Gemma2Tokenizer, QuantizedGemma2};
use crate::native_hybrid_phasor_cdt_engine::NativeHybridConfig;
use crate::native_phasor_thermodynamic_engine::NativePhasorMinimizerConfig;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use serde::{Deserialize, Serialize};
use std::fmt;

const DEFAULT_THERMO_WINDOW: usize = 64;
const DEFAULT_RFF_FEATURES_CAP: usize = 512;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gemma2ThermoHybridLearnedState {
    pub config: Gemma2ThermoHybridConfig,
    pub thermo: ThermoAttentionLearnedState,
    pub tokens_processed: u64,
    pub sleep_cycles: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Gemma2ThermoHybridConfig {
    /// Ventana deslizante de embeddings enviada al motor CTP por paso.
    pub thermo_window: usize,
    /// Tope de características RFF (escala con d_model pero acotado en CPU).
    pub rff_features_cap: usize,
    /// Nodos del grafo CDT/fasorial.
    pub cdt_nodes: usize,
    /// Consolidación sleep cada N tokens generados (0 = desactivada).
    pub sleep_every_tokens: usize,
    pub seed: u64,
}

impl Default for Gemma2ThermoHybridConfig {
    fn default() -> Self {
        Self {
            thermo_window: DEFAULT_THERMO_WINDOW,
            rff_features_cap: DEFAULT_RFF_FEATURES_CAP,
            cdt_nodes: 512,
            sleep_every_tokens: 32,
            seed: 0x4745_4D4D_4154_4845,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Gemma2ThermoStepReport {
    pub thermo: HybridThermoAttentionReport,
    pub context_length: usize,
    pub phi_norm: f32,
    pub logits_max: f32,
    pub slept: bool,
}

#[derive(Clone, Debug)]
pub struct Gemma2ThermoGenerationReport {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub sleep_cycles: usize,
    pub mean_phi_norm: f32,
    pub mean_softmax_entropy: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma2ThermoHybridError {
    Thermo(HybridThermoAttentionError),
    Candle(String),
    EmptyContext,
}

impl fmt::Display for Gemma2ThermoHybridError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Thermo(error) => write!(formatter, "motor termodinámico: {error}"),
            Self::Candle(message) => write!(formatter, "Candle: {message}"),
            Self::EmptyContext => write!(formatter, "contexto vacío"),
        }
    }
}

impl std::error::Error for Gemma2ThermoHybridError {}

impl From<HybridThermoAttentionError> for Gemma2ThermoHybridError {
    fn from(error: HybridThermoAttentionError) -> Self {
        Self::Thermo(error)
    }
}

pub struct Gemma2ThermoHybridLlm {
    thermo: HybridThermoAttention,
    config: Gemma2ThermoHybridConfig,
    context: Vec<Vec<f32>>,
    tokens_processed: u64,
    sleep_cycles: u64,
}

impl Gemma2ThermoHybridLlm {
    /// Construye el híbrido dimensionado para un modelo Gemma 2 cargado.
    pub fn for_gemma(model: &QuantizedGemma2, config: Gemma2ThermoHybridConfig) -> Result<Self, Gemma2ThermoHybridError> {
        let d_model = model.embedding_length();
        let thermo = HybridThermoAttention::new(thermo_config_for_gemma(
            d_model,
            config.rff_features_cap,
            config.cdt_nodes,
            config.seed,
        ))?;
        Ok(Self {
            thermo,
            config,
            context: Vec::new(),
            tokens_processed: 0,
            sleep_cycles: 0,
        })
    }

    pub fn config(&self) -> &Gemma2ThermoHybridConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut Gemma2ThermoHybridConfig {
        &mut self.config
    }

    pub fn thermo_engine(&self) -> &HybridThermoAttention {
        &self.thermo
    }

    pub fn thermo_engine_mut(&mut self) -> &mut HybridThermoAttention {
        &mut self.thermo
    }

    pub fn context_len(&self) -> usize {
        self.context.len()
    }

    pub fn sleep_cycles(&self) -> u64 {
        self.sleep_cycles
    }

    pub fn tokens_processed(&self) -> u64 {
        self.tokens_processed
    }

    pub fn reset(&mut self) {
        self.context.clear();
        self.tokens_processed = 0;
        self.sleep_cycles = 0;
    }

    pub fn export_learned_state(&self) -> Gemma2ThermoHybridLearnedState {
        Gemma2ThermoHybridLearnedState {
            config: self.config,
            thermo: self.thermo.export_learned_state(),
            tokens_processed: self.tokens_processed,
            sleep_cycles: self.sleep_cycles,
        }
    }

    pub fn apply_learned_state(
        &mut self,
        state: &Gemma2ThermoHybridLearnedState,
    ) -> Result<(), Gemma2ThermoHybridError> {
        if state.config.cdt_nodes != self.config.cdt_nodes
            || state.config.rff_features_cap != self.config.rff_features_cap
            || state.config.seed != self.config.seed
        {
            return Err(Gemma2ThermoHybridError::Thermo(
                HybridThermoAttentionError::DimensionMismatch {
                    expected: self.config.cdt_nodes,
                    got: state.config.cdt_nodes,
                },
            ));
        }
        self.config.thermo_window = state.config.thermo_window;
        self.config.sleep_every_tokens = state.config.sleep_every_tokens;
        self.thermo.apply_learned_state(&state.thermo)?;
        self.tokens_processed = state.tokens_processed;
        self.sleep_cycles = state.sleep_cycles;
        Ok(())
    }

    /// Prefill: ingesta embeddings del prompt (desde W_emb·√d) al motor CTP.
    pub fn prefill_embeddings(&mut self, embeddings: &[Vec<f32>]) -> Result<(), Gemma2ThermoHybridError> {
        for embedding in embeddings {
            self.push_context(embedding.clone());
        }
        Ok(())
    }

    /// Prefill desde IDs de token usando el periférico Gemma 2.
    pub fn prefill_tokens(
        &mut self,
        model: &QuantizedGemma2,
        token_ids: &[u32],
    ) -> Result<(), Gemma2ThermoHybridError> {
        for &token_id in token_ids {
            let embedding = embed_token_to_vec(model, token_id)?;
            self.push_context(embedding);
        }
        Ok(())
    }

    /// Muestra el siguiente token desde el contexto actual (post-prefill o post-paso).
    pub fn sample_next(
        &mut self,
        model: &QuantizedGemma2,
        processor: &mut LogitsProcessor,
    ) -> Result<(u32, Gemma2ThermoStepReport), Gemma2ThermoHybridError> {
        let (phi, thermo_report) = self.run_thermo_step()?;
        let next_token = sample_phi(model, &phi, processor)?;
        let embedding = embed_token_to_vec(model, next_token)?;
        self.push_context(embedding);

        self.tokens_processed = self.tokens_processed.saturating_add(1);
        let mut slept = false;
        if self.config.sleep_every_tokens > 0
            && self.tokens_processed.is_multiple_of(self.config.sleep_every_tokens as u64)
        {
            let _ = self.thermo.sleep_consolidate()?;
            self.sleep_cycles = self.sleep_cycles.saturating_add(1);
            slept = true;
        }

        let phi_norm = vector_norm(&phi);
        Ok((
            next_token,
            Gemma2ThermoStepReport {
                thermo: thermo_report,
                context_length: self.context.len(),
                phi_norm,
                logits_max: 0.0, // filled below if needed
                slept,
            },
        ))
    }

    /// Un paso autoregresivo: añade token entrante al contexto y devuelve el siguiente.
    pub fn step(
        &mut self,
        model: &QuantizedGemma2,
        input_token: u32,
        processor: &mut LogitsProcessor,
    ) -> Result<(u32, Gemma2ThermoStepReport), Gemma2ThermoHybridError> {
        let embedding = embed_token_to_vec(model, input_token)?;
        self.push_context(embedding);
        self.sample_next(model, processor)
    }

    /// Generación autoregresiva con callback por token (streaming).
    pub fn generate_streaming(
        &mut self,
        model: &QuantizedGemma2,
        prompt_tokens: &[u32],
        max_tokens: usize,
        stop_tokens: &[u32],
        processor: &mut LogitsProcessor,
        mut on_token: impl FnMut(u32, &Gemma2ThermoStepReport) -> bool,
    ) -> Result<(Vec<u32>, Gemma2ThermoGenerationReport), Gemma2ThermoHybridError> {
        self.reset();
        if prompt_tokens.is_empty() {
            return Err(Gemma2ThermoHybridError::EmptyContext);
        }
        self.prefill_tokens(model, prompt_tokens)?;

        let mut generated = Vec::with_capacity(max_tokens);
        let mut phi_norm_sum = 0.0f32;
        let mut entropy_sum = 0.0f32;

        for _ in 0..max_tokens {
            let (next, report) = self.sample_next(model, processor)?;
            phi_norm_sum += report.phi_norm;
            entropy_sum += report.thermo.softmax_entropy;
            generated.push(next);
            let stop = stop_tokens.contains(&next) || on_token(next, &report);
            if stop {
                break;
            }
        }

        let generated_len = generated.len();
        Ok((
            generated,
            Gemma2ThermoGenerationReport {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens: generated_len,
                sleep_cycles: self.sleep_cycles as usize,
                mean_phi_norm: phi_norm_sum / generated_len.max(1) as f32,
                mean_softmax_entropy: entropy_sum / generated_len.max(1) as f32,
            },
        ))
    }

    /// Fuerza un ciclo sleep/consolidación CDT inmediato.
    pub fn force_sleep(
        &mut self,
    ) -> Result<
        crate::native_hybrid_phasor_cdt_engine::NativeHybridSleepReport,
        Gemma2ThermoHybridError,
    > {
        self.thermo
            .sleep_consolidate()
            .map_err(Gemma2ThermoHybridError::from)
    }

    pub fn supervised_align_step(
        &mut self,
        embeddings: &[Vec<f32>],
        teacher_last: &[f32],
        learning_rate: f32,
    ) -> Result<f32, Gemma2ThermoHybridError> {
        self.thermo
            .supervised_align_step(embeddings, teacher_last, learning_rate)
            .map_err(Gemma2ThermoHybridError::from)
    }

    /// Generación autoregresiva completa sobre un prompt tokenizado.
    pub fn generate(
        &mut self,
        model: &QuantizedGemma2,
        prompt_tokens: &[u32],
        max_tokens: usize,
        mut processor: LogitsProcessor,
    ) -> Result<(Vec<u32>, Gemma2ThermoGenerationReport), Gemma2ThermoHybridError> {
        self.reset();
        if prompt_tokens.is_empty() {
            return Err(Gemma2ThermoHybridError::EmptyContext);
        }
        self.prefill_tokens(model, prompt_tokens)?;

        let mut generated = Vec::with_capacity(max_tokens);
        let mut phi_norm_sum = 0.0f32;
        let mut entropy_sum = 0.0f32;

        for _ in 0..max_tokens {
            let (next, report) = self.sample_next(model, &mut processor)?;
            phi_norm_sum += report.phi_norm;
            entropy_sum += report.thermo.softmax_entropy;
            generated.push(next);
        }

        let generated_len = generated.len();
        Ok((
            generated,
            Gemma2ThermoGenerationReport {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens: generated_len,
                sleep_cycles: self.sleep_cycles as usize,
                mean_phi_norm: phi_norm_sum / generated_len.max(1) as f32,
                mean_softmax_entropy: entropy_sum / generated_len.max(1) as f32,
            },
        ))
    }

    /// Generación con tokenizer (texto → tokens → híbrido → texto).
    pub fn generate_text(
        &mut self,
        model: &QuantizedGemma2,
        tokenizer: &Gemma2Tokenizer,
        prompt: &str,
        max_tokens: usize,
        processor: LogitsProcessor,
    ) -> Result<(String, Gemma2ThermoGenerationReport), Gemma2ThermoHybridError> {
        let prompt_tokens = tokenizer
            .encode(prompt)
            .map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))?;
        let (generated, report) = self.generate(model, &prompt_tokens, max_tokens, processor)?;
        let mut all = prompt_tokens;
        all.extend(&generated);
        let text = tokenizer
            .decode(&all, true)
            .map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))?;
        Ok((text, report))
    }

    fn push_context(&mut self, embedding: Vec<f32>) {
        self.context.push(embedding);
        let window = self.config.thermo_window.max(1);
        if self.context.len() > window {
            let drop = self.context.len() - window;
            self.context.drain(0..drop);
        }
    }

    fn run_thermo_step(&mut self) -> Result<(Vec<f32>, HybridThermoAttentionReport), Gemma2ThermoHybridError> {
        if self.context.is_empty() {
            return Err(Gemma2ThermoHybridError::EmptyContext);
        }
        let values = self.context.clone();
        let (out, report) = self.thermo.forward(&self.context, &values)?;
        let phi = out.last().cloned().unwrap_or_default();
        Ok((phi, report))
    }
}

/// Configura HybridThermoAttention para las dimensiones de Gemma 2.
pub fn thermo_config_for_gemma(
    d_model: usize,
    rff_cap: usize,
    cdt_nodes: usize,
    seed: u64,
) -> HybridThermoAttentionConfig {
    let rff_features = (d_model * 2).min(rff_cap).max(32);
    HybridThermoAttentionConfig {
        d_model,
        d_v: d_model,
        rff: PhasorRffConfig {
            features: rff_features,
            sigma: 1.0 / (d_model as f32).sqrt(),
            seed: seed ^ 0x5246_46,
        },
        cdt_nodes,
        cdt_spatial_degree: 4,
        cdt_seed: seed ^ 0x4354_50,
        thermo_blend: 0.55,
        ctp_coupling_alpha: 0.4,
        hybrid: NativeHybridConfig {
            minimizer: NativePhasorMinimizerConfig {
                max_iterations: 80,
                residual_tolerance: 5.0e-3,
                handshake_strength: 0.65,
                attention_strength: 0.5,
                ..Default::default()
            },
            minimum_relative_energy_drop: 0.0,
            maximum_residual: 8.0e-2,
            minimum_magnetic_coherence: 0.5,
            cue_as_boundary: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Decodificación periférica CPU pura (tests y validación sin GGUF).
pub fn decode_phi_with_softcap(phi: &[f32], w_emb: &[Vec<f32>], cap: f32) -> Vec<f32> {
    w_emb
        .iter()
        .map(|row| {
            let raw: f32 = row.iter().zip(phi.iter()).map(|(&w, &p)| w * p).sum();
            cap * (raw / cap).tanh()
        })
        .collect()
}

pub fn embed_token_with_scale(row: &[f32], d_model: usize) -> Vec<f32> {
    let scale = (d_model as f32).sqrt();
    row.iter().map(|v| v * scale).collect()
}

fn sample_phi(
    model: &QuantizedGemma2,
    phi: &[f32],
    processor: &mut LogitsProcessor,
) -> Result<u32, Gemma2ThermoHybridError> {
    let device = model.device();
    let phi_tensor = f32_vec_to_tensor(phi, device)?;
    let logits = model
        .logits_from_hidden(&phi_tensor)
        .map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))?
        .squeeze(0)
        .map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))?;
    processor
        .sample(&logits)
        .map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))
}

fn embed_token_to_vec(model: &QuantizedGemma2, token_id: u32) -> Result<Vec<f32>, Gemma2ThermoHybridError> {
    model
        .embed_token(token_id)
        .map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))
}

fn f32_vec_to_tensor(values: &[f32], device: &candle_core::Device) -> Result<Tensor, Gemma2ThermoHybridError> {
    Tensor::new(values, device).map_err(|e| Gemma2ThermoHybridError::Candle(e.to_string()))
}

fn vector_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_rng::signed_unit;

    fn synthetic_w_emb(vocab: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        (0..vocab)
            .map(|t| {
                (0..d)
                    .map(|j| signed_unit(seed ^ ((t as u64) << 12) ^ j as u64) * 0.02)
                    .collect()
            })
            .collect()
    }

    fn synthetic_embedding(d: usize, seed: u64) -> Vec<f32> {
        (0..d)
            .map(|j| signed_unit(seed ^ j as u64))
            .collect()
    }

    #[test]
    fn peripheral_softcap_matches_gemma_formula() {
        let cap = 30.0f32;
        let phi = vec![0.5, -0.3, 0.8, 0.1];
        let w = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let logits = decode_phi_with_softcap(&phi, &w, cap);
        for (i, &logit) in logits.iter().enumerate() {
            let raw = phi[i];
            let expected = cap * (raw / cap).tanh();
            assert!((logit - expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn embed_scale_matches_gemma_sqrt_d() {
        let d = 16;
        let row = vec![0.1; d];
        let scaled = embed_token_with_scale(&row, d);
        let expected_scale = (d as f32).sqrt();
        assert!((scaled[0] - 0.1 * expected_scale).abs() < 1.0e-5);
    }

    #[test]
    fn thermo_hybrid_llm_prefill_and_step_without_gguf() {
        let d = 8;
        let config = Gemma2ThermoHybridConfig {
            thermo_window: 8,
            rff_features_cap: 16,
            cdt_nodes: 64,
            sleep_every_tokens: 0,
            ..Default::default()
        };
        let thermo = HybridThermoAttention::new(thermo_config_for_gemma(
            d,
            config.rff_features_cap,
            config.cdt_nodes,
            config.seed,
        ))
        .expect("thermo");
        let mut llm = Gemma2ThermoHybridLlm {
            thermo,
            config,
            context: Vec::new(),
            tokens_processed: 0,
            sleep_cycles: 0,
        };

        let embeddings: Vec<Vec<f32>> = (0..4)
            .map(|i| synthetic_embedding(d, 100 + i))
            .collect();
        llm.prefill_embeddings(&embeddings).expect("prefill");
        assert_eq!(llm.context_len(), 4);
        let (phi, _) = llm.run_thermo_step().expect("thermo");
        assert_eq!(phi.len(), d);
    }

    #[test]
    fn decode_phi_produces_finite_logits() {
        let w = synthetic_w_emb(32, 8, 42);
        let phi = synthetic_embedding(8, 99);
        let logits = decode_phi_with_softcap(&phi, &w, 30.0);
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|l| l.is_finite()));
        assert!(logits.iter().all(|&l| l.abs() <= 30.0 + 1.0e-4));
    }

    #[test]
    fn thermo_config_scales_with_gemma_dimensions() {
        let cfg = thermo_config_for_gemma(2048, 512, 1024, 1);
        assert_eq!(cfg.d_model, 2048);
        assert_eq!(cfg.d_v, 2048);
        assert_eq!(cfg.rff.features, 512);
        assert_eq!(cfg.cdt_nodes, 1024);
    }

    #[test]
    fn sliding_window_trims_context() {
        let d = 4;
        let config = Gemma2ThermoHybridConfig {
            thermo_window: 3,
            rff_features_cap: 8,
            cdt_nodes: 32,
            sleep_every_tokens: 0,
            ..Default::default()
        };
        let thermo = HybridThermoAttention::new(thermo_config_for_gemma(
            d,
            config.rff_features_cap,
            config.cdt_nodes,
            config.seed,
        ))
        .unwrap();
        let mut llm = Gemma2ThermoHybridLlm {
            thermo,
            config,
            context: Vec::new(),
            tokens_processed: 0,
            sleep_cycles: 0,
        };
        for i in 0..6 {
            llm.push_context(synthetic_embedding(d, i));
        }
        assert_eq!(llm.context_len(), 3);
    }

    #[test]
    fn hybrid_llm_forward_produces_phi_for_decode() {
        let d = 8;
        let config = Gemma2ThermoHybridConfig {
            thermo_window: 6,
            rff_features_cap: 16,
            cdt_nodes: 64,
            sleep_every_tokens: 0,
            ..Default::default()
        };
        let thermo = HybridThermoAttention::new(thermo_config_for_gemma(
            d,
            config.rff_features_cap,
            config.cdt_nodes,
            config.seed,
        ))
        .unwrap();
        let mut llm = Gemma2ThermoHybridLlm {
            thermo,
            config,
            context: Vec::new(),
            tokens_processed: 0,
            sleep_cycles: 0,
        };
        let embeddings: Vec<Vec<f32>> = (0..5)
            .map(|i| synthetic_embedding(d, 200 + i))
            .collect();
        llm.prefill_embeddings(&embeddings).unwrap();
        let (phi, report) = llm.run_thermo_step().unwrap();
        assert_eq!(phi.len(), d);
        assert!(report.ctp_bias_norm > 0.0);
        let w = synthetic_w_emb(16, d, 300);
        let logits = decode_phi_with_softcap(&phi, &w, 30.0);
        assert!(!logits.is_empty());
    }
}
