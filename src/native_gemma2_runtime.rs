//! Runtime autoregresivo compartido para Gemma 2.
//!
//! Conserva la KV cache entre turnos cuando el nuevo prompt extiende exactamente
//! el prefijo ya procesado y usa el decoder incremental de `tokenizers`, evitando
//! reconstruir toda la salida después de cada token.

use crate::native_gemma2::{Gemma2Tokenizer, LayerExecutionMask, QuantizedGemma2};
use candle_core::{Result, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Gemma2GenerationConfig {
    pub max_tokens: usize,
    pub context_limit: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma2GenerationMetrics {
    pub cache_reused: bool,
    pub prefill_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_seconds: f64,
    pub time_to_first_token_seconds: f64,
    pub decode_seconds: f64,
}

impl Gemma2GenerationMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        self.prefill_tokens as f64 / self.prefill_seconds.max(f64::EPSILON)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_seconds.max(f64::EPSILON)
    }
}

#[derive(Clone, Debug)]
pub struct Gemma2Generation {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub metrics: Gemma2GenerationMetrics,
}

#[derive(Debug, Default)]
pub struct Gemma2Session {
    cached_tokens: Vec<u32>,
    last_logits: Option<Tensor>,
    active_mask: Option<LayerExecutionMask>,
}

impl Gemma2Session {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cached_tokens(&self) -> &[u32] {
        &self.cached_tokens
    }

    pub fn reset(&mut self, model: &mut QuantizedGemma2) {
        model.clear_kv_cache();
        self.cached_tokens.clear();
        self.last_logits = None;
        self.active_mask = None;
    }

    /// Adopta una KV cache que otro pipeline acaba de rellenar con el mismo
    /// modelo. Se usa en la ruta adaptativa tras verificar una máscara.
    pub fn adopt_prefill(
        &mut self,
        prompt_tokens: &[u32],
        mask: Option<&LayerExecutionMask>,
        logits: Tensor,
    ) -> Result<()> {
        self.cached_tokens.clear();
        self.cached_tokens.extend_from_slice(prompt_tokens);
        self.active_mask = mask.cloned();
        self.last_logits = Some(logits.squeeze(0)?);
        Ok(())
    }

    pub fn generate(
        &mut self,
        model: &mut QuantizedGemma2,
        tokenizer: &Gemma2Tokenizer,
        prompt_tokens: &[u32],
        mask: Option<&LayerExecutionMask>,
        config: Gemma2GenerationConfig,
        on_text: impl FnMut(&str),
    ) -> Result<Gemma2Generation> {
        self.generate_until(
            model,
            tokenizer,
            prompt_tokens,
            mask,
            config,
            on_text,
            |_| false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_until(
        &mut self,
        model: &mut QuantizedGemma2,
        tokenizer: &Gemma2Tokenizer,
        prompt_tokens: &[u32],
        mask: Option<&LayerExecutionMask>,
        config: Gemma2GenerationConfig,
        on_text: impl FnMut(&str),
        should_stop: impl FnMut(&str) -> bool,
    ) -> Result<Gemma2Generation> {
        self.generate_observed(
            model,
            tokenizer,
            prompt_tokens,
            mask,
            config,
            on_text,
            |_, _| {},
            should_stop,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_observed(
        &mut self,
        model: &mut QuantizedGemma2,
        tokenizer: &Gemma2Tokenizer,
        prompt_tokens: &[u32],
        mask: Option<&LayerExecutionMask>,
        config: Gemma2GenerationConfig,
        mut on_text: impl FnMut(&str),
        mut on_token: impl FnMut(u32, usize),
        mut should_stop: impl FnMut(&str) -> bool,
    ) -> Result<Gemma2Generation> {
        if prompt_tokens.is_empty() {
            candle_core::bail!("Gemma 2 requiere al menos un token de prompt");
        }
        if config.max_tokens == 0 || config.context_limit == 0 {
            candle_core::bail!("max_tokens y context_limit deben ser mayores que cero");
        }
        let started = Instant::now();
        let same_mask = self.active_mask.as_ref() == mask;
        let extends_cache = same_mask
            && !self.cached_tokens.is_empty()
            && prompt_tokens.starts_with(&self.cached_tokens);
        if !extends_cache {
            self.reset(model);
            self.active_mask = mask.cloned();
        }
        let prefix = if extends_cache {
            self.cached_tokens.len()
        } else {
            0
        };
        let suffix = &prompt_tokens[prefix..];
        let prefill_started = Instant::now();
        let mut logits = if suffix.is_empty() {
            self.last_logits
                .clone()
                .ok_or_else(|| candle_core::Error::Msg("sesión Gemma sin logits".to_string()))?
        } else {
            let input = Tensor::new(suffix, model.device())?.unsqueeze(0)?;
            let logits = model
                .forward_with_mask(&input, prefix, mask, false, false)?
                .logits
                .squeeze(0)?;
            self.cached_tokens.extend_from_slice(suffix);
            self.last_logits = Some(logits.clone());
            logits
        };
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();

        let mut sampler =
            LogitsProcessor::new(config.seed, Some(config.temperature), Some(config.top_p));
        let mut generated = Vec::with_capacity(config.max_tokens);
        let mut decoder = tokenizer.decode_stream(true);
        let mut streamed_text = String::new();
        let decode_started = Instant::now();
        let mut first_token_seconds = None;
        for _ in 0..config.max_tokens {
            let token = sampler.sample(&logits)?;
            if first_token_seconds.is_none() {
                first_token_seconds = Some(started.elapsed().as_secs_f64());
            }
            if token == tokenizer.eos_id || Some(token) == tokenizer.end_of_turn_id {
                break;
            }
            generated.push(token);
            on_token(token, self.cached_tokens.len());
            if let Some(fragment) = decoder.step(token)? {
                streamed_text.push_str(&fragment);
                on_text(&fragment);
                if should_stop(&streamed_text) {
                    break;
                }
            }
            if self.cached_tokens.len() >= config.context_limit {
                break;
            }
            let next = Tensor::new(&[token], model.device())?.unsqueeze(0)?;
            logits = model
                .forward_with_mask(&next, self.cached_tokens.len(), mask, false, false)?
                .logits
                .squeeze(0)?;
            self.cached_tokens.push(token);
            self.last_logits = Some(logits.clone());
        }
        let decode_seconds = decode_started.elapsed().as_secs_f64();
        let text = tokenizer.decode(&generated, true)?.trim().to_string();
        Ok(Gemma2Generation {
            text,
            token_ids: generated.clone(),
            metrics: Gemma2GenerationMetrics {
                cache_reused: extends_cache,
                prefill_tokens: suffix.len(),
                generated_tokens: generated.len(),
                prefill_seconds,
                time_to_first_token_seconds: first_token_seconds
                    .unwrap_or_else(|| started.elapsed().as_secs_f64()),
                decode_seconds,
            },
        })
    }
}

pub fn chat_tokens(
    tokenizer: &Gemma2Tokenizer,
    history: &[(String, String)],
    input: &str,
    limit: usize,
) -> Result<Vec<u32>> {
    for skip in 0..=history.len() {
        let mut prompt = String::new();
        for (user, assistant) in &history[skip..] {
            prompt.push_str("<start_of_turn>user\n");
            prompt.push_str(user);
            prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
            prompt.push_str(assistant);
            prompt.push_str("<end_of_turn>\n");
        }
        prompt.push_str("<start_of_turn>user\n");
        prompt.push_str(input);
        prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(&prompt)?);
        if tokens.len() <= limit {
            return Ok(tokens);
        }
    }
    candle_core::bail!("el mensaje actual excede el límite ({limit} tokens)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_prefix_extension_is_required_for_kv_reuse() {
        let cached = [1, 2, 3, 4];
        assert!([1, 2, 3, 4, 5].starts_with(&cached));
        assert!(![1, 2, 9, 4, 5].starts_with(&cached));
        assert!(![1, 2, 3].starts_with(&cached));
    }

    #[test]
    fn metric_rates_are_finite_for_zero_duration() {
        let metrics = Gemma2GenerationMetrics {
            prefill_tokens: 32,
            generated_tokens: 8,
            ..Gemma2GenerationMetrics::default()
        };
        assert!(metrics.prefill_tokens_per_second().is_finite());
        assert!(metrics.decode_tokens_per_second().is_finite());
    }
}
