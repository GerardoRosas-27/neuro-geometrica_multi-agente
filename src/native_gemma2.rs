//! Inferencia Gemma 2 cuantizada directamente desde GGUF.
//!
//! Esta implementación usa únicamente kernels Rust de Candle. No inicia ni
//! consulta el servidor de Ollama; el archivo GGUF se abre como un archivo local.

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::kv_cache::{KvCache, RotatingKvCache};
use candle_transformers::quantized_nn::RmsNorm;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::env;
use std::fs;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::{AddedToken, Tokenizer};

const DEFAULT_MAX_CONTEXT: usize = 8_192;
const DEFAULT_ROPE_FREQUENCY: f32 = 10_000.0;
const DEFAULT_ATTENTION_SOFTCAP: f64 = 50.0;
const DEFAULT_FINAL_SOFTCAP: f64 = 30.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerExecutionMask {
    enabled: Vec<bool>,
}

impl LayerExecutionMask {
    pub fn all(layer_count: usize) -> Self {
        Self {
            enabled: vec![true; layer_count],
        }
    }

    pub fn from_enabled(enabled: Vec<bool>) -> Self {
        Self { enabled }
    }

    pub fn layer_count(&self) -> usize {
        self.enabled.len()
    }

    pub fn executes(&self, layer: usize) -> bool {
        self.enabled.get(layer).copied().unwrap_or(false)
    }

    pub fn executed_count(&self) -> usize {
        self.enabled.iter().filter(|enabled| **enabled).count()
    }

    pub fn enabled_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.enabled
            .iter()
            .enumerate()
            .filter_map(|(index, enabled)| enabled.then_some(index))
    }

    pub fn encode(&self) -> Vec<u32> {
        let words = self.enabled.len().div_ceil(32);
        let mut encoded = Vec::with_capacity(words + 2);
        encoded.push(0x474D_5232);
        encoded.push(self.enabled.len() as u32);
        for word in 0..words {
            let mut bits = 0u32;
            for bit in 0..32 {
                let index = word * 32 + bit;
                if self.enabled.get(index).copied().unwrap_or(false) {
                    bits |= 1u32 << bit;
                }
            }
            encoded.push(bits);
        }
        encoded
    }

    pub fn decode(encoded: &[u32]) -> Option<Self> {
        if encoded.len() < 2 || encoded[0] != 0x474D_5232 {
            return None;
        }
        let layer_count = encoded[1] as usize;
        let words = layer_count.div_ceil(32);
        if layer_count == 0 || encoded.len() != words + 2 {
            return None;
        }
        let enabled = (0..layer_count)
            .map(|index| encoded[2 + index / 32] & (1u32 << (index % 32)) != 0)
            .collect();
        Some(Self { enabled })
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct LayerActivationSummary {
    pub layer: usize,
    pub executed: bool,
    pub input_rms: f32,
    pub output_rms: f32,
    pub delta_rms: f32,
    /// Atencion de ventana deslizante (local). `false` = global.
    #[serde(default)]
    pub sliding_window: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Gemma2ForwardTrace {
    pub layers: Vec<LayerActivationSummary>,
    pub executed_layers: usize,
    pub skipped_layers: usize,
}

pub struct Gemma2ForwardOutput {
    pub logits: Tensor,
    pub trace: Gemma2ForwardTrace,
    /// Estados ocultos `[batch, seq, d_model]` antes de norm/logits (sólo si se pidió).
    pub sequence_hidden: Option<Tensor>,
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?.gelu()?;
        let up = self.up.forward(xs)?;
        self.down.forward(&(gate * up)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_context: usize, frequency: f32, device: &Device) -> Result<Self> {
        let inverse_frequency = (0..head_dim)
            .step_by(2)
            .map(|index| 1.0 / frequency.powf(index as f32 / head_dim as f32))
            .collect::<Vec<_>>();
        let inverse_frequency = Tensor::new(inverse_frequency, device)?;
        let positions = Tensor::arange(0u32, max_context as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_context, 1))?;
        let frequencies = positions.matmul(&inverse_frequency.reshape((1, head_dim / 2))?)?;
        Ok(Self {
            sin: frequencies.sin()?,
            cos: frequencies.cos()?,
        })
    }

    fn apply(&self, query: &Tensor, key: &Tensor, position: usize) -> Result<(Tensor, Tensor)> {
        let sequence_length = query.dim(2)?;
        let cos = self.cos.narrow(0, position, sequence_length)?;
        let sin = self.sin.narrow(0, position, sequence_length)?;
        let query = candle_nn::rotary_emb::rope(&query.contiguous()?, &cos, &sin)?;
        let key = candle_nn::rotary_emb::rope(&key.contiguous()?, &cos, &sin)?;
        Ok((query, key))
    }
}

#[derive(Debug, Clone)]
enum LayerKvCache {
    Full(KvCache),
    Sliding(RotatingKvCache),
}

impl LayerKvCache {
    fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Full(cache) => cache.append(key, value),
            Self::Sliding(cache) => cache.append(key, value),
        }
    }

    fn key_positions(&self, sequence_length: usize) -> Option<Vec<usize>> {
        match self {
            Self::Full(_) => None,
            Self::Sliding(cache) => Some(cache.positions(sequence_length)),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Full(cache) => cache.reset(),
            Self::Sliding(cache) => cache.reset(),
        }
    }
}

#[derive(Debug, Clone)]
struct Layer {
    query: QMatMul,
    key: QMatMul,
    value: QMatMul,
    output: QMatMul,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: Mlp,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_scale: f64,
    attention_softcap: f64,
    sliding_window: Option<usize>,
    rotary: RotaryEmbedding,
    kv_cache: LayerKvCache,
}

impl Layer {
    fn key_positions(&self, sequence_length: usize) -> Option<Vec<usize>> {
        self.kv_cache.key_positions(sequence_length)
    }

    fn attention(
        &mut self,
        xs: &Tensor,
        position: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, sequence_length, _) = xs.dims3()?;
        let query = self
            .query
            .forward(xs)?
            .reshape((batch, sequence_length, self.heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = self
            .key
            .forward(xs)?
            .reshape((batch, sequence_length, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value = self
            .value
            .forward(xs)?
            .reshape((batch, sequence_length, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query, key) = self.rotary.apply(&query, &key, position)?;
        let (key, value) = self
            .kv_cache
            .append(&key.contiguous()?, &value.contiguous()?)?;
        let attended = grouped_query_attention(
            &query,
            &key,
            &value,
            self.heads / self.kv_heads,
            self.query_scale,
            self.attention_softcap,
            attention_mask,
        )?
        .transpose(1, 2)?
        .reshape((batch, sequence_length, self.heads * self.head_dim))?;
        self.output.forward(&attended)
    }
}

/// Transformer Gemma 2 cuantizado cargado directamente desde un GGUF.
#[derive(Debug, Clone)]
pub struct QuantizedGemma2 {
    embeddings: QMatMul,
    embedding_length: usize,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    final_softcap: f64,
    max_context: usize,
    active_mask: Option<LayerExecutionMask>,
    device: Device,
}

impl QuantizedGemma2 {
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let architecture = metadata_string(&content, "general.architecture")?;
        if architecture != "gemma2" {
            candle_core::bail!("se esperaba arquitectura gemma2, se recibió {architecture}");
        }
        let heads = metadata_usize(&content, "gemma2.attention.head_count")?;
        let kv_heads = metadata_usize(&content, "gemma2.attention.head_count_kv")?;
        let block_count = metadata_usize(&content, "gemma2.block_count")?;
        let embedding_length = metadata_usize(&content, "gemma2.embedding_length")?;
        let head_dim = metadata_usize_optional(&content, "gemma2.attention.key_length")
            .unwrap_or(embedding_length / heads);
        let max_context = metadata_usize_optional(&content, "gemma2.context_length")
            .unwrap_or(DEFAULT_MAX_CONTEXT);
        let sliding_window =
            metadata_usize_optional(&content, "gemma2.attention.sliding_window").unwrap_or(4_096);
        let rms_epsilon = metadata_f64(&content, "gemma2.attention.layer_norm_rms_epsilon")?;
        let rope_frequency = metadata_f64_optional(&content, "gemma2.rope.freq_base")
            .unwrap_or(DEFAULT_ROPE_FREQUENCY as f64) as f32;
        let query_pre_attention_scalar =
            metadata_f64_optional(&content, "gemma2.attention.query_pre_attn_scalar")
                .unwrap_or(head_dim as f64);
        let attention_softcap = metadata_f64_optional(&content, "gemma2.attention.logit_softcap")
            .or_else(|| metadata_f64_optional(&content, "gemma2.attention.logit_softcapping"))
            .unwrap_or(DEFAULT_ATTENTION_SOFTCAP);
        let final_softcap = metadata_f64_optional(&content, "gemma2.final_logit_softcap")
            .or_else(|| metadata_f64_optional(&content, "gemma2.final_logit_softcapping"))
            .unwrap_or(DEFAULT_FINAL_SOFTCAP);

        let embeddings =
            QMatMul::from_qtensor(content.tensor(reader, "token_embd.weight", device)?)?;
        let output = match content.tensor(reader, "output.weight", device) {
            Ok(output) => QMatMul::from_qtensor(output)?,
            Err(_) => embeddings.clone(),
        };
        let norm = RmsNorm::from_qtensor(
            content.tensor(reader, "output_norm.weight", device)?,
            rms_epsilon,
        )?;
        let rotary = RotaryEmbedding::new(head_dim, max_context, rope_frequency, device)?;
        let mut layers = Vec::with_capacity(block_count);
        for layer_index in 0..block_count {
            let prefix = format!("blk.{layer_index}");
            let tensor = |name: &str, reader: &mut R| {
                content.tensor(reader, &format!("{prefix}.{name}.weight"), device)
            };
            let query = QMatMul::from_qtensor(tensor("attn_q", reader)?)?;
            let key = QMatMul::from_qtensor(tensor("attn_k", reader)?)?;
            let value = QMatMul::from_qtensor(tensor("attn_v", reader)?)?;
            let output = QMatMul::from_qtensor(tensor("attn_output", reader)?)?;
            let attention_norm = RmsNorm::from_qtensor(tensor("attn_norm", reader)?, rms_epsilon)?;
            let post_attention_norm =
                RmsNorm::from_qtensor(tensor("post_attention_norm", reader)?, rms_epsilon)?;
            let ffn_norm = RmsNorm::from_qtensor(tensor("ffn_norm", reader)?, rms_epsilon)?;
            let post_ffn_norm =
                RmsNorm::from_qtensor(tensor("post_ffw_norm", reader)?, rms_epsilon)?;
            let mlp = Mlp {
                gate: QMatMul::from_qtensor(tensor("ffn_gate", reader)?)?,
                up: QMatMul::from_qtensor(tensor("ffn_up", reader)?)?,
                down: QMatMul::from_qtensor(tensor("ffn_down", reader)?)?,
            };
            layers.push(Layer {
                query,
                key,
                value,
                output,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                heads,
                kv_heads,
                head_dim,
                query_scale: 1.0 / query_pre_attention_scalar.sqrt(),
                attention_softcap,
                sliding_window: (layer_index % 2 == 0).then_some(sliding_window),
                rotary: rotary.clone(),
                kv_cache: if layer_index % 2 == 0 {
                    LayerKvCache::Sliding(RotatingKvCache::new(2, sliding_window))
                } else {
                    LayerKvCache::Full(KvCache::new(2, max_context))
                },
            });
        }
        Ok(Self {
            embeddings,
            embedding_length,
            layers,
            norm,
            output,
            final_softcap,
            max_context,
            active_mask: None,
            device: device.clone(),
        })
    }

    pub fn forward(&mut self, token_ids: &Tensor, position: usize) -> Result<Tensor> {
        Ok(self
            .forward_with_mask(token_ids, position, None, false, false)?
            .logits)
    }

    pub fn forward_with_mask(
        &mut self,
        token_ids: &Tensor,
        position: usize,
        mask: Option<&LayerExecutionMask>,
        capture_trace: bool,
        capture_sequence_hidden: bool,
    ) -> Result<Gemma2ForwardOutput> {
        let (_, sequence_length) = token_ids.dims2()?;
        if position + sequence_length > self.max_context {
            candle_core::bail!(
                "contexto Gemma 2 excedido: {} > {}",
                position + sequence_length,
                self.max_context
            );
        }
        let requested_mask = mask
            .cloned()
            .unwrap_or_else(|| LayerExecutionMask::all(self.layers.len()));
        if requested_mask.layer_count() != self.layers.len() {
            candle_core::bail!(
                "máscara Gemma 2 incompatible: {} capas para modelo de {}",
                requested_mask.layer_count(),
                self.layers.len()
            );
        }
        if position == 0 {
            self.active_mask = Some(requested_mask.clone());
        } else if self.active_mask.as_ref() != Some(&requested_mask) {
            candle_core::bail!(
                "la máscara de capas debe permanecer fija durante una generación; limpia KV cache para cambiarla"
            );
        }
        let mut hidden =
            (self.embeddings.embedding(token_ids)? * (self.embedding_length as f64).sqrt())?;
        let global_mask = if sequence_length == 1 {
            None
        } else {
            let key_positions = (0..position + sequence_length).collect::<Vec<_>>();
            build_attention_mask(
                hidden.dim(0)?,
                sequence_length,
                position,
                &key_positions,
                None,
                hidden.dtype(),
                hidden.device(),
            )?
        };
        let local_mask = if sequence_length == 1 {
            None
        } else {
            let local_key_positions = self
                .layers
                .iter()
                .enumerate()
                .find(|(index, layer)| {
                    requested_mask.executes(*index) && layer.sliding_window.is_some()
                })
                .and_then(|(_, layer)| layer.key_positions(sequence_length));
            match local_key_positions {
                Some(key_positions) => build_attention_mask(
                    hidden.dim(0)?,
                    sequence_length,
                    position,
                    &key_positions,
                    self.layers.iter().find_map(|layer| layer.sliding_window),
                    hidden.dtype(),
                    hidden.device(),
                )?,
                None => None,
            }
        };
        let mut trace = Gemma2ForwardTrace {
            layers: Vec::with_capacity(self.layers.len()),
            executed_layers: 0,
            skipped_layers: 0,
        };
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            if !requested_mask.executes(layer_index) {
                trace.skipped_layers += 1;
                if capture_trace {
                    trace.layers.push(LayerActivationSummary {
                        layer: layer_index,
                        executed: false,
                        sliding_window: layer.sliding_window.is_some(),
                        ..LayerActivationSummary::default()
                    });
                }
                continue;
            }
            let layer_input = hidden.clone();
            let input_rms = capture_trace.then(|| tensor_rms(&hidden)).transpose()?;
            let residual = &hidden;
            let normalized = layer.attention_norm.forward(&hidden)?;
            let attention_mask = if layer.sliding_window.is_some() {
                local_mask.as_ref()
            } else {
                global_mask.as_ref()
            };
            let attended = layer.attention(&normalized, position, attention_mask)?;
            let attended = layer.post_attention_norm.forward(&attended)?;
            hidden = (&attended + residual)?;
            let residual = &hidden;
            let normalized = layer.ffn_norm.forward(&hidden)?;
            let projected = layer.mlp.forward(&normalized)?;
            let projected = layer.post_ffn_norm.forward(&projected)?;
            hidden = (&projected + residual)?;
            trace.executed_layers += 1;
            if capture_trace {
                let output_rms = tensor_rms(&hidden)?;
                let delta_rms = tensor_rms(&(&hidden - &layer_input)?)?;
                trace.layers.push(LayerActivationSummary {
                    layer: layer_index,
                    executed: true,
                    input_rms: input_rms.unwrap_or_default(),
                    output_rms,
                    delta_rms,
                    sliding_window: layer.sliding_window.is_some(),
                });
            }
        }
        let sequence_hidden = if capture_sequence_hidden {
            Some(hidden.clone())
        } else {
            None
        };
        let last = hidden.i((.., sequence_length - 1, ..))?;
        let logits = self.output.forward(&self.norm.forward(&last)?)?;
        let logits = (&logits / self.final_softcap)?.tanh()? * self.final_softcap;
        Ok(Gemma2ForwardOutput {
            logits: logits?,
            trace,
            sequence_hidden,
        })
    }

    /// Estados ocultos post-Transformer (pre-norm/logits) para cada token del prompt.
    pub fn prefill_teacher_hiddens(&mut self, token_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.clear_kv_cache();
        let ids = Tensor::new(token_ids, self.device())?.unsqueeze(0)?;
        let output = self.forward_with_mask(&ids, 0, None, false, true)?;
        let hidden = output
            .sequence_hidden
            .ok_or_else(|| candle_core::Error::Msg("prefill sin hidden".to_string()))?;
        let sequence_length = token_ids.len();
        let mut hiddens = Vec::with_capacity(sequence_length);
        for index in 0..sequence_length {
            hiddens.push(hidden.i((0, index, ..))?.to_vec1::<f32>()?);
        }
        Ok(hiddens)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache.reset();
        }
        self.active_mask = None;
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Gemma 2 alterna local (ventana) y global. Las pares son locales.
    pub fn layer_uses_sliding_window(&self, layer: usize) -> bool {
        self.layers
            .get(layer)
            .is_some_and(|layer| layer.sliding_window.is_some())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn embedding_storage_bytes(&self) -> usize {
        match &self.embeddings {
            QMatMul::QTensor(tensor) => tensor.storage_size_in_bytes(),
            QMatMul::Tensor(tensor) | QMatMul::TensorF16(tensor) => {
                tensor.elem_count() * tensor.dtype().size_in_bytes()
            }
        }
    }

    pub fn embedding_logical_f32_bytes(&self) -> usize {
        match &self.embeddings {
            QMatMul::QTensor(tensor) => tensor.shape().elem_count() * std::mem::size_of::<f32>(),
            QMatMul::Tensor(tensor) | QMatMul::TensorF16(tensor) => {
                tensor.elem_count() * std::mem::size_of::<f32>()
            }
        }
    }

    pub fn embedding_length(&self) -> usize {
        self.embedding_length
    }

    pub fn final_softcap(&self) -> f64 {
        self.final_softcap
    }

    /// Periférico de ingesta: W_emb[k] · √d_model (escalación nativa Gemma 2).
    pub fn embed_token_ids(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embeddings.embedding(token_ids)? * (self.embedding_length as f64).sqrt()
    }

    /// Embedding de un único token como vector [d_model].
    pub fn embed_token(&self, token_id: u32) -> Result<Tensor> {
        let ids = Tensor::new(&[[token_id]], self.device())?;
        self.embed_token_ids(&ids)?.squeeze(0)?.squeeze(0)
    }

    /// Periférico de decodificación: norm → W_unemb → softcap tanh (Gemma 2).
    ///
    /// `hidden` debe tener forma `[d_model]` o `[batch, d_model]`.
    pub fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = if hidden.rank() == 1 {
            hidden.unsqueeze(0)?
        } else {
            hidden.clone()
        };
        let logits = self.output.forward(&self.norm.forward(&hidden)?)?;
        (&logits / self.final_softcap)?.tanh()? * self.final_softcap
    }
}

#[allow(clippy::too_many_arguments)]
fn build_attention_mask(
    batch: usize,
    query_length: usize,
    position: usize,
    key_positions: &[usize],
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if query_length == 1 {
        return Ok(None);
    }
    let mut masked = false;
    let mut values = Vec::with_capacity(query_length * key_positions.len());
    for query_index in 0..query_length {
        let absolute_query = position + query_index;
        for &absolute_key in key_positions {
            let causal = absolute_key <= absolute_query;
            let in_window =
                sliding_window.is_none_or(|window| absolute_key + window > absolute_query);
            let value = if causal && in_window {
                0.0f32
            } else {
                masked = true;
                f32::NEG_INFINITY
            };
            values.push(value);
        }
    }
    if !masked {
        return Ok(None);
    }
    Tensor::from_vec(values, (query_length, key_positions.len()), device)?
        .expand((batch, 1, query_length, key_positions.len()))?
        .to_dtype(dtype)
        .map(Some)
}

fn grouped_query_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    groups: usize,
    scale: f64,
    softcap: f64,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (batch, query_heads, query_length, head_dim) = query.dims4()?;
    let (_, key_value_heads, key_length, key_head_dim) = key.dims4()?;
    if key_head_dim != head_dim || query_heads != key_value_heads * groups {
        candle_core::bail!(
            "GQA incompatible: query_heads={query_heads}, kv_heads={key_value_heads}, \
             groups={groups}, query_dim={head_dim}, key_dim={key_head_dim}"
        );
    }
    if query_length > 4 {
        // Prefill: un matmul batched amortiza la copia contigua.
        let repeated_key = repeat_kv(key.clone(), groups)?.contiguous()?;
        let repeated_value = repeat_kv(value.clone(), groups)?.contiguous()?;
        let mut weights = (query.matmul(&repeated_key.transpose(2, 3)?)? * scale)?;
        weights = ((&weights / softcap)?.tanh()? * softcap)?;
        if let Some(mask) = attention_mask {
            weights = weights.broadcast_add(mask)?;
        }
        return candle_nn::ops::softmax_last_dim(&weights)?.matmul(&repeated_value);
    }

    // Decode/sufijo corto: procesa juntas las cabezas Q que comparten K/V.
    // Evita tanto repetir todo el contexto como lanzar un matmul por Q-head.
    let mut attended = Vec::with_capacity(key_value_heads);
    for key_value_head in 0..key_value_heads {
        let query_group = query.narrow(1, key_value_head * groups, groups)?;
        let key_head = key
            .narrow(1, key_value_head, 1)?
            .expand((batch, groups, key_length, head_dim))?;
        let value_head = value
            .narrow(1, key_value_head, 1)?
            .expand((batch, groups, key_length, head_dim))?;
        let mut weights = (query_group.matmul(&key_head.transpose(2, 3)?)? * scale)?;
        weights = ((&weights / softcap)?.tanh()? * softcap)?;
        if let Some(mask) = attention_mask {
            weights = weights.broadcast_add(mask)?;
        }
        attended.push(candle_nn::ops::softmax_last_dim(&weights)?.matmul(&value_head)?);
    }
    Tensor::cat(&attended.iter().collect::<Vec<_>>(), 1)
}

fn tensor_rms(tensor: &Tensor) -> Result<f32> {
    tensor
        .to_dtype(DType::F32)?
        .sqr()?
        .mean_all()?
        .sqrt()?
        .to_scalar::<f32>()
}

/// Tokenizador SentencePiece/Unigram reconstruido desde los metadatos GGUF.
pub struct Gemma2Tokenizer {
    tokenizer: Tokenizer,
    pub bos_id: u32,
    pub eos_id: u32,
    pub end_of_turn_id: Option<u32>,
}

pub struct Gemma2DecodeStream<'a> {
    stepper: Box<dyn FnMut(u32) -> Result<Option<String>> + 'a>,
}

impl Gemma2DecodeStream<'_> {
    pub fn step(&mut self, token: u32) -> Result<Option<String>> {
        (self.stepper)(token)
    }
}

impl Gemma2Tokenizer {
    pub fn from_gguf(content: &gguf_file::Content) -> Result<Self> {
        let tokens = metadata_array(content, "tokenizer.ggml.tokens")?
            .iter()
            .map(|value| value.to_string().map(ToOwned::to_owned))
            .collect::<Result<Vec<_>>>()?;
        let scores = metadata_array(content, "tokenizer.ggml.scores")?
            .iter()
            .map(value_f64)
            .collect::<Result<Vec<_>>>()?;
        if tokens.len() != scores.len() {
            candle_core::bail!(
                "tokenizador GGUF inconsistente: tokens={} scores={}",
                tokens.len(),
                scores.len()
            );
        }
        let unknown_id = metadata_u32_optional(content, "tokenizer.ggml.unknown_token_id")
            .or_else(|| metadata_u32_optional(content, "tokenizer.ggml.unk_token_id"))
            .map(|value| value as usize);
        let unigram = Unigram::from(
            tokens.iter().cloned().zip(scores).collect(),
            unknown_id,
            metadata_bool_optional(content, "tokenizer.ggml.byte_fallback").unwrap_or(true),
        )
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(unigram);
        let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
        tokenizer.with_pre_tokenizer(Some(metaspace.clone()));
        tokenizer.with_decoder(Some(metaspace));

        if let Ok(types) = metadata_array(content, "tokenizer.ggml.token_type") {
            let specials = types
                .iter()
                .enumerate()
                .filter_map(|(index, value)| {
                    let kind = value_u32(value).ok()?;
                    if matches!(kind, 2..=5) {
                        tokens
                            .get(index)
                            .map(|token| AddedToken::from(token.clone(), true))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            tokenizer.add_special_tokens(&specials);
        }
        let bos_id = metadata_u32(content, "tokenizer.ggml.bos_token_id")?;
        let eos_id = metadata_u32(content, "tokenizer.ggml.eos_token_id")?;
        let end_of_turn_id = tokenizer.token_to_id("<end_of_turn>");
        Ok(Self {
            tokenizer,
            bos_id,
            eos_id,
            end_of_turn_id,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }

    pub fn decode_stream(&self, skip_special_tokens: bool) -> Gemma2DecodeStream<'_> {
        let mut stream = self.tokenizer.decode_stream(skip_special_tokens);
        Gemma2DecodeStream {
            stepper: Box::new(move |token| {
                stream
                    .step(token)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            }),
        }
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
}

/// Resuelve el GGUF local sin invocar el proceso ni la API de Ollama.
pub fn resolve_gemma2_model_path(
    explicit: Option<&Path>,
) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
        return Err(format!("GGUF no encontrado: {}", path.display()).into());
    }
    if let Ok(path) = env::var("GEMMA2_GGUF") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }
    let paged_manifest = Path::new("data/native_gemma2_paged_thermo/manifest.txt");
    if let Ok(contents) = fs::read_to_string(paged_manifest) {
        if let Some(path) = contents
            .lines()
            .find_map(|line| line.strip_prefix("source="))
        {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Ok(path);
            }
        }
    }
    let mut stores = Vec::new();
    if let Ok(path) = env::var("OLLAMA_MODELS") {
        stores.push(PathBuf::from(path));
    }
    stores.push(PathBuf::from("ollama-models"));
    if let Some(home) = env::var_os("USERPROFILE") {
        stores.push(PathBuf::from(home).join(".ollama").join("models"));
    }
    for store in stores {
        if let Some(path) = gemma2_from_oci_store(&store)? {
            return Ok(path);
        }
    }
    Err("no se encontró Gemma 2; usa --model RUTA_GGUF o GEMMA2_GGUF".into())
}

pub fn resolve_gemma2_device(requested: &str) -> Result<Device> {
    let normalized = requested.trim().to_ascii_lowercase();
    if normalized == "cpu" {
        return Ok(Device::Cpu);
    }
    if normalized == "cuda" || normalized.starts_with("cuda:") {
        #[cfg(feature = "cuda")]
        {
            let ordinal = normalized
                .split_once(':')
                .map_or(Ok(0usize), |(_, value)| value.parse::<usize>())
                .map_err(|error| {
                    candle_core::Error::Msg(format!("ordinal CUDA inválido: {error}"))
                })?;
            return Device::new_cuda(ordinal);
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_core::bail!(
                "CUDA no está compilado; reconstruye con `cargo build --release --features cuda`"
            );
        }
    }
    candle_core::bail!("dispositivo Gemma desconocido `{requested}`; usa cpu o cuda[:N]")
}

fn gemma2_from_oci_store(
    store: &Path,
) -> std::result::Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let manifest = store
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join("gemma2")
        .join("2b");
    let Ok(body) = fs::read(&manifest) else {
        return Ok(None);
    };
    let document: JsonValue = serde_json::from_slice(&body)?;
    let digest = document["layers"]
        .as_array()
        .and_then(|layers| {
            layers.iter().find(|layer| {
                layer["mediaType"]
                    .as_str()
                    .is_some_and(|kind| kind.ends_with(".image.model"))
            })
        })
        .and_then(|layer| layer["digest"].as_str())
        .and_then(|digest| digest.strip_prefix("sha256:"));
    let Some(digest) = digest else {
        return Ok(None);
    };
    let path = store.join("blobs").join(format!("sha256-{digest}"));
    Ok(path.is_file().then_some(path))
}

fn repeat_kv(xs: Tensor, repetitions: usize) -> Result<Tensor> {
    if repetitions == 1 {
        return Ok(xs);
    }
    let (batch, heads, sequence, head_dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((batch, heads, repetitions, sequence, head_dim))?
        .reshape((batch, heads * repetitions, sequence, head_dim))
}

fn metadata_value<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("falta metadata GGUF `{key}`")))
}

fn metadata_array<'a>(
    content: &'a gguf_file::Content,
    key: &str,
) -> Result<&'a Vec<gguf_file::Value>> {
    metadata_value(content, key)?.to_vec()
}

fn metadata_string(content: &gguf_file::Content, key: &str) -> Result<String> {
    metadata_value(content, key)?
        .to_string()
        .map(ToOwned::to_owned)
}

fn metadata_usize(content: &gguf_file::Content, key: &str) -> Result<usize> {
    Ok(value_u32(metadata_value(content, key)?)? as usize)
}

fn metadata_usize_optional(content: &gguf_file::Content, key: &str) -> Option<usize> {
    metadata_value(content, key)
        .ok()
        .and_then(|value| value_u32(value).ok())
        .map(|value| value as usize)
}

fn metadata_u32(content: &gguf_file::Content, key: &str) -> Result<u32> {
    value_u32(metadata_value(content, key)?)
}

fn metadata_u32_optional(content: &gguf_file::Content, key: &str) -> Option<u32> {
    metadata_value(content, key)
        .ok()
        .and_then(|value| value_u32(value).ok())
}

fn metadata_f64(content: &gguf_file::Content, key: &str) -> Result<f64> {
    value_f64(metadata_value(content, key)?)
}

fn metadata_f64_optional(content: &gguf_file::Content, key: &str) -> Option<f64> {
    metadata_value(content, key)
        .ok()
        .and_then(|value| value_f64(value).ok())
}

fn metadata_bool_optional(content: &gguf_file::Content, key: &str) -> Option<bool> {
    metadata_value(content, key)
        .ok()
        .and_then(|value| value.to_bool().ok())
}

fn value_u32(value: &gguf_file::Value) -> Result<u32> {
    use gguf_file::Value;
    match value {
        Value::U8(value) => Ok(*value as u32),
        Value::I8(value) => Ok(*value as u32),
        Value::U16(value) => Ok(*value as u32),
        Value::I16(value) => Ok(*value as u32),
        Value::U32(value) => Ok(*value),
        Value::I32(value) => Ok(*value as u32),
        Value::U64(value) => Ok(*value as u32),
        Value::I64(value) => Ok(*value as u32),
        _ => candle_core::bail!("se esperaba entero GGUF, se recibió {value:?}"),
    }
}

fn value_f64(value: &gguf_file::Value) -> Result<f64> {
    use gguf_file::Value;
    match value {
        Value::F32(value) => Ok(*value as f64),
        Value::F64(value) => Ok(*value),
        _ => value_u32(value).map(|value| value as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_grouped_key_value_heads() {
        let tensor = Tensor::arange(0f32, 8f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 2, 2))
            .unwrap();
        let repeated = repeat_kv(tensor, 2).unwrap();
        assert_eq!(repeated.dims(), &[1, 4, 2, 2]);
    }

    #[test]
    fn grouped_attention_matches_materialized_kv_heads() {
        let query = Tensor::arange(0f32, 24f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 4, 2, 3))
            .unwrap();
        let key = (Tensor::arange(0f32, 30f32, &Device::Cpu).unwrap() / 10.0)
            .unwrap()
            .reshape((1, 2, 5, 3))
            .unwrap();
        let value = (Tensor::arange(0f32, 30f32, &Device::Cpu).unwrap() / 7.0)
            .unwrap()
            .reshape((1, 2, 5, 3))
            .unwrap();
        let mask = Tensor::new(
            &[
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            &Device::Cpu,
        )
        .unwrap()
        .reshape((1, 1, 2, 5))
        .unwrap();
        let grouped =
            grouped_query_attention(&query, &key, &value, 2, 0.5, 50.0, Some(&mask)).unwrap();

        let repeated_key = repeat_kv(key, 2).unwrap().contiguous().unwrap();
        let repeated_value = repeat_kv(value, 2).unwrap().contiguous().unwrap();
        let weights = (query
            .matmul(&repeated_key.transpose(2, 3).unwrap())
            .unwrap()
            * 0.5)
            .unwrap();
        let weights = ((&weights / 50.0).unwrap().tanh().unwrap() * 50.0)
            .unwrap()
            .broadcast_add(&mask)
            .unwrap();
        let expected = candle_nn::ops::softmax_last_dim(&weights)
            .unwrap()
            .matmul(&repeated_value)
            .unwrap();
        let difference = (&grouped - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(difference < 1.0e-5, "diferencia GQA={difference}");
    }

    #[test]
    fn shared_sliding_mask_uses_absolute_ring_positions() {
        let mask = build_attention_mask(1, 2, 5, &[4, 5, 6], Some(2), DType::F32, &Device::Cpu)
            .unwrap()
            .unwrap()
            .reshape((2, 3))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(mask[0], vec![0.0, 0.0, f32::NEG_INFINITY]);
        assert_eq!(mask[1], vec![f32::NEG_INFINITY, 0.0, 0.0]);
    }

    #[test]
    fn converts_numeric_metadata_values() {
        assert_eq!(value_u32(&gguf_file::Value::U16(42)).unwrap(), 42);
        assert_eq!(value_f64(&gguf_file::Value::F32(0.5)).unwrap(), 0.5);
    }

    #[test]
    fn layer_masks_round_trip_compactly() {
        let mask = LayerExecutionMask::from_enabled(
            (0..67).map(|index| index % 3 != 0).collect::<Vec<_>>(),
        );
        assert_eq!(LayerExecutionMask::decode(&mask.encode()), Some(mask));
    }
}
