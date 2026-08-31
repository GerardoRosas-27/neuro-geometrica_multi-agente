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
    pub model_decode_seconds: f64,
    pub logits_processing_seconds: f64,
    pub text_decode_seconds: f64,
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

    pub fn active_mask(&self) -> Option<&LayerExecutionMask> {
        self.active_mask.as_ref()
    }

    pub fn last_logits(&self) -> Option<&Tensor> {
        self.last_logits.as_ref()
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
        on_text: impl FnMut(&str),
        on_token: impl FnMut(u32, usize),
        should_stop: impl FnMut(&str) -> bool,
    ) -> Result<Gemma2Generation> {
        self.generate_observed_with_logits(
            model,
            tokenizer,
            prompt_tokens,
            mask,
            config,
            on_text,
            on_token,
            |_, _| Ok(()),
            should_stop,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_observed_with_logits(
        &mut self,
        model: &mut QuantizedGemma2,
        tokenizer: &Gemma2Tokenizer,
        prompt_tokens: &[u32],
        mask: Option<&LayerExecutionMask>,
        config: Gemma2GenerationConfig,
        mut on_text: impl FnMut(&str),
        mut on_token: impl FnMut(u32, usize),
        mut on_logits: impl FnMut(&mut Tensor, usize) -> Result<()>,
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
                .take()
                .ok_or_else(|| candle_core::Error::Msg("sesión Gemma sin logits".to_string()))?
        } else {
            let input = Tensor::new(suffix, model.device())?.unsqueeze(0)?;
            let logits = model
                .forward_with_mask(&input, prefix, mask, false, false)?
                .logits
                .squeeze(0)?;
            self.cached_tokens.extend_from_slice(suffix);
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
        let mut model_decode_seconds = 0.0;
        let mut logits_processing_seconds = 0.0;
        let mut text_decode_seconds = 0.0;
        for _ in 0..config.max_tokens {
            let logits_started = Instant::now();
            on_logits(&mut logits, generated.len())?;
            let token = sampler.sample(&logits)?;
            logits_processing_seconds += logits_started.elapsed().as_secs_f64();
            if first_token_seconds.is_none() {
                first_token_seconds = Some(started.elapsed().as_secs_f64());
            }
            if token == tokenizer.eos_id || Some(token) == tokenizer.end_of_turn_id {
                break;
            }
            generated.push(token);
            on_token(token, self.cached_tokens.len());
            let text_started = Instant::now();
            if let Some(fragment) = decoder.step(token)? {
                streamed_text.push_str(&fragment);
                on_text(&fragment);
                if should_stop(&streamed_text) {
                    text_decode_seconds += text_started.elapsed().as_secs_f64();
                    break;
                }
            }
            text_decode_seconds += text_started.elapsed().as_secs_f64();
            if self.cached_tokens.len() >= config.context_limit {
                break;
            }
            let next = Tensor::new(&[token], model.device())?.unsqueeze(0)?;
            let model_started = Instant::now();
            logits = model
                .forward_with_mask(&next, self.cached_tokens.len(), mask, false, false)?
                .logits
                .squeeze(0)?;
            model_decode_seconds += model_started.elapsed().as_secs_f64();
            self.cached_tokens.push(token);
        }
        let decode_seconds = decode_started.elapsed().as_secs_f64();
        self.last_logits = Some(logits);
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
                model_decode_seconds,
                logits_processing_seconds,
                text_decode_seconds,
            },
        })
    }
}

/// Idioma forzado del chat. La identidad Dyamon vive aquí, no en el historial
/// de usuario.
pub const GEMMA2_FORCED_LANGUAGE: &str = "es";
pub const GEMMA2_SYSTEM_INSTRUCTION: &str = concat!(
    "Responde siempre en español. Eres un asistente técnico de ingeniería. ",
    "Tu nombre interno es Dyamon y no forma parte del historial del usuario; ",
    "no lo menciones salvo que te lo pregunten. No digas que eres un large ",
    "language model. Si no hay un resultado del motor, abstente en lugar de ",
    "inventar."
);

pub fn gemma2_system_prefix() -> String {
    let mut prefix = String::from("<start_of_turn>user\n");
    prefix.push_str(GEMMA2_SYSTEM_INSTRUCTION);
    prefix.push_str("<end_of_turn>\n<start_of_turn>model\n");
    prefix.push_str("Entendido. Responderé en español.<end_of_turn>\n");
    prefix
}

pub fn render_chat_prompt(history: &[(String, String)], input: &str) -> String {
    let mut prompt = gemma2_system_prefix();
    for (user, assistant) in history {
        prompt.push_str("<start_of_turn>user\n");
        prompt.push_str(user);
        prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
        prompt.push_str(assistant);
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>user\n");
    prompt.push_str(input);
    prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
    prompt
}

/// Construye el siguiente prompt sobre los IDs exactos ya presentes en KV.
/// Evita `encode(decode(ids))`, que no garantiza recuperar la tokenización
/// original y puede provocar un prefill completo silencioso.
pub fn chat_tokens_with_cache(
    tokenizer: &Gemma2Tokenizer,
    history: &[(String, String)],
    input: &str,
    limit: usize,
    cached_tokens: &[u32],
) -> Result<Vec<u32>> {
    if !cached_tokens.is_empty() {
        let suffix = tokenizer.encode(&format!(
            "<end_of_turn>\n<start_of_turn>user\n{input}<end_of_turn>\n\
             <start_of_turn>model\n"
        ))?;
        if cached_tokens.len() + suffix.len() <= limit {
            let mut tokens = Vec::with_capacity(cached_tokens.len() + suffix.len());
            tokens.extend_from_slice(cached_tokens);
            tokens.extend(suffix);
            return Ok(tokens);
        }
    }
    chat_tokens(tokenizer, history, input, limit)
}

pub fn chat_tokens(
    tokenizer: &Gemma2Tokenizer,
    history: &[(String, String)],
    input: &str,
    limit: usize,
) -> Result<Vec<u32>> {
    for skip in 0..=history.len() {
        let prompt = render_chat_prompt(&history[skip..], input);
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
    fn second_turn_reuses_prompt_plus_generated_prefix() {
        let prompt_turn_1 = [1u32, 2, 3, 4];
        let generated = [10u32, 11];
        let mut cached = prompt_turn_1.to_vec();
        cached.extend_from_slice(&generated);
        let mut prompt_turn_2 = cached.clone();
        prompt_turn_2.extend_from_slice(&[20, 21, 22]);
        assert!(prompt_turn_2.starts_with(&cached));
        assert_eq!(&prompt_turn_2[cached.len()..], &[20, 21, 22]);
    }

    #[test]
    fn cached_prompt_builder_preserves_the_exact_prefix() {
        let cached = [2u32, 10, 11, 12];
        let suffix = [107u32, 108, 20, 21];
        let mut next = Vec::from(cached);
        next.extend(suffix);
        assert!(next.starts_with(&cached));
        assert_eq!(&next[cached.len()..], &suffix);
    }

    #[test]
    fn system_prompt_keeps_dyamon_out_of_user_history() {
        let prompt = render_chat_prompt(
            &[("hola".to_string(), "buenas".to_string())],
            "quién eres",
        );
        assert!(prompt.contains("Responde siempre en español"));
        assert!(prompt.contains("Dyamon"));
        assert!(prompt.contains("<start_of_turn>user\nhola"));
        assert!(!prompt.contains("<start_of_turn>user\nDyamon"));
        assert_eq!(GEMMA2_FORCED_LANGUAGE, "es");
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
