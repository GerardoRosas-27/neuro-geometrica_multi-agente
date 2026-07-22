//! Inferencia Gemma 2 cuantizada directamente desde GGUF.
//!
//! Esta implementación usa únicamente kernels Rust de Candle. No inicia ni
//! consulta el servidor de Ollama; el archivo GGUF se abre como un archivo local.

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
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
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Layer {
    fn attention_mask(
        &self,
        batch: usize,
        query_length: usize,
        position: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let key_length = position + query_length;
        if query_length == 1
            && self
                .sliding_window
                .is_none_or(|window| key_length <= window)
        {
            return Ok(None);
        }
        let mut values = Vec::with_capacity(query_length * key_length);
        for query_index in 0..query_length {
            let absolute_query = position + query_index;
            for key_index in 0..key_length {
                let causal = key_index <= absolute_query;
                let in_window = self
                    .sliding_window
                    .is_none_or(|window| key_index + window > absolute_query);
                values.push(if causal && in_window {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        let mask = Tensor::from_vec(values, (query_length, key_length), device)?
            .expand((batch, 1, query_length, key_length))?
            .to_dtype(dtype)?;
        Ok(Some(mask))
    }

    fn attention(&mut self, xs: &Tensor, position: usize) -> Result<Tensor> {
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
        let (key, value) = match &self.kv_cache {
            Some((cached_key, cached_value)) if position > 0 => (
                Tensor::cat(&[cached_key, &key], 2)?,
                Tensor::cat(&[cached_value, &value], 2)?,
            ),
            _ => (key, value),
        };
        self.kv_cache = Some((key.clone(), value.clone()));
        let key = repeat_kv(key, self.heads / self.kv_heads)?.contiguous()?;
        let value = repeat_kv(value, self.heads / self.kv_heads)?.contiguous()?;
        let mut weights = (query.matmul(&key.transpose(2, 3)?)? * self.query_scale)?;
        weights = ((&weights / self.attention_softcap)?.tanh()? * self.attention_softcap)?;
        if let Some(mask) = self.attention_mask(
            batch,
            sequence_length,
            position,
            weights.dtype(),
            weights.device(),
        )? {
            weights = weights.broadcast_add(&mask)?;
        }
        let weights = candle_nn::ops::softmax_last_dim(&weights)?;
        let attended = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            sequence_length,
            self.heads * self.head_dim,
        ))?;
        self.output.forward(&attended)
    }
}

/// Transformer Gemma 2 cuantizado cargado directamente desde un GGUF.
#[derive(Debug, Clone)]
pub struct QuantizedGemma2 {
    embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    final_softcap: f64,
    max_context: usize,
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

        let embeddings_quantized = content.tensor(reader, "token_embd.weight", device)?;
        let embeddings = embeddings_quantized.dequantize(device)?;
        let output_quantized = content
            .tensor(reader, "output.weight", device)
            .unwrap_or(embeddings_quantized);
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
                kv_cache: None,
            });
        }
        Ok(Self {
            embeddings: Embedding::new(embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output_quantized)?,
            final_softcap,
            max_context,
        })
    }

    pub fn forward(&mut self, token_ids: &Tensor, position: usize) -> Result<Tensor> {
        let (_, sequence_length) = token_ids.dims2()?;
        if position + sequence_length > self.max_context {
            candle_core::bail!(
                "contexto Gemma 2 excedido: {} > {}",
                position + sequence_length,
                self.max_context
            );
        }
        let mut hidden =
            (self.embeddings.forward(token_ids)? * (self.embedding_length as f64).sqrt())?;
        for layer in &mut self.layers {
            let residual = &hidden;
            let normalized = layer.attention_norm.forward(&hidden)?;
            let attended = layer.attention(&normalized, position)?;
            let attended = layer.post_attention_norm.forward(&attended)?;
            hidden = (&attended + residual)?;
            let residual = &hidden;
            let normalized = layer.ffn_norm.forward(&hidden)?;
            let projected = layer.mlp.forward(&normalized)?;
            let projected = layer.post_ffn_norm.forward(&projected)?;
            hidden = (&projected + residual)?;
        }
        let last = hidden.i((.., sequence_length - 1, ..))?;
        let logits = self.output.forward(&self.norm.forward(&last)?)?;
        (&logits / self.final_softcap)?.tanh()? * self.final_softcap
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
        }
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }
}

/// Tokenizador SentencePiece/Unigram reconstruido desde los metadatos GGUF.
pub struct Gemma2Tokenizer {
    tokenizer: Tokenizer,
    pub bos_id: u32,
    pub eos_id: u32,
    pub end_of_turn_id: Option<u32>,
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
    fn converts_numeric_metadata_values() {
        assert_eq!(value_u32(&gguf_file::Value::U16(42)).unwrap(), 42);
        assert_eq!(value_f64(&gguf_file::Value::F32(0.5)).unwrap(), 0.5);
    }
}
