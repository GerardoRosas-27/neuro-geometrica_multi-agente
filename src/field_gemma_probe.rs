//! Sonda Gemma 2 congelada: capa lingüística del encoder.
//!
//! Texto → tokenizer + forward con máscara T2.1 → `LinguisticPacket`.
//! Los `token_ids` viven solo aquí. `into_field_features()` los tira.
//! El sustrato recibe features/`z`, nunca `Vec<u32>`.
//!
//! Pesos GGUF congelados. Solo se entrena el projector a `z`.

use crate::field_encoder::{LayerSkipMask, EXPENSIVE_LAYERS, GEMMA2_LAYER_COUNT};
use crate::field_linguistic_layer::{
    FrozenLinguisticProbe, LinguisticPacket, HIDDEN_DIM, STEM_DIM,
};
use crate::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, LayerExecutionMask,
    QuantizedGemma2,
};
use crate::native_gemma2_runtime::{Gemma2GenerationConfig, Gemma2Session};
use candle_core::{Device, Tensor};
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

const STEM_HASH_SALT: u64 = 0x57E4_CAFE;

/// Gemma 2 cargada desde GGUF. Fallar al construir si no hay modelo.
pub struct FrozenGemma2Probe {
    /// Pesos compartidos con [`RawGemmaHandle`] (chat crudo) para no cargar el
    /// GGUF dos veces en RAM.
    model: Arc<Mutex<QuantizedGemma2>>,
    tokenizer: Arc<Gemma2Tokenizer>,
    device: Device,
    mask: LayerExecutionMask,
}

/// Acceso al Gemma 2 congelado **original** (denso, sin máscara T2.1, sin campo)
/// para generación de texto autoregresiva plana: prompt → plantilla chat → respuesta.
/// Comparte pesos con la sonda. No toca FieldState ni el fuse líquido/CDT/RQM.
#[derive(Clone)]
pub struct RawGemmaHandle {
    model: Arc<Mutex<QuantizedGemma2>>,
    tokenizer: Arc<Gemma2Tokenizer>,
}

/// Resultado de una generación cruda.
#[derive(Clone, Debug)]
pub struct RawGemmaReply {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub seconds: f64,
}

/// Plantilla chat oficial de Gemma 2 (sin system prompt, sin campo).
pub fn render_raw_gemma_prompt(history: &[(String, String)], input: &str) -> String {
    let mut prompt = String::new();
    for (user, model) in history {
        prompt.push_str("<start_of_turn>user\n");
        prompt.push_str(user);
        prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
        prompt.push_str(model);
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>user\n");
    prompt.push_str(input);
    prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
    prompt
}

impl RawGemmaHandle {
    /// Genera respuesta con el Gemma 2 original (todas las capas).
    /// Bloqueante (CPU): llamar desde `spawn_blocking`.
    pub fn generate_chat(
        &self,
        history: &[(String, String)],
        input: &str,
        config: Gemma2GenerationConfig,
    ) -> Result<RawGemmaReply, String> {
        let started = std::time::Instant::now();
        // Recorta historial hasta que quepa (deja margen para generar).
        let budget = config
            .context_limit
            .saturating_sub(config.max_tokens)
            .max(16);
        let mut tokens = None;
        for skip in 0..=history.len() {
            let prompt = render_raw_gemma_prompt(&history[skip..], input);
            let mut t = vec![self.tokenizer.bos_id];
            t.extend(self.tokenizer.encode(&prompt).map_err(|e| e.to_string())?);
            if t.len() <= budget {
                tokens = Some(t);
                break;
            }
        }
        let tokens =
            tokens.ok_or_else(|| format!("el mensaje excede el contexto ({budget} tokens)"))?;
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        // Sesión nueva: la sonda comparte KV cache, así que siempre prefill completo.
        let mut session = Gemma2Session::new();
        let out = session
            .generate(&mut model, &self.tokenizer, &tokens, None, config, |_| {})
            .map_err(|e| e.to_string());
        model.clear_kv_cache();
        let out = out?;
        Ok(RawGemmaReply {
            text: out.text,
            prompt_tokens: tokens.len(),
            generated_tokens: out.metrics.generated_tokens,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

impl FrozenGemma2Probe {
    pub fn try_open(explicit: Option<&Path>) -> Result<Self, String> {
        let path = resolve_gemma2_model_path(explicit).map_err(|e| e.to_string())?;
        let device = resolve_gemma2_device("cpu").map_err(|e| e.to_string())?;
        let mut file = File::open(&path).map_err(|e| e.to_string())?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| e.to_string())?;
        let tokenizer = Gemma2Tokenizer::from_gguf(&content).map_err(|e| e.to_string())?;
        let model =
            QuantizedGemma2::from_gguf(content, &mut file, &device).map_err(|e| e.to_string())?;
        let skip = LayerSkipMask::from_t21_expensive_skip();
        let mask = LayerExecutionMask::from_enabled(skip.on.clone());
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            mask,
        })
    }

    /// Handle al Gemma original (denso) para chat crudo; comparte pesos.
    pub fn raw_handle(&self) -> RawGemmaHandle {
        RawGemmaHandle {
            model: Arc::clone(&self.model),
            tokenizer: Arc::clone(&self.tokenizer),
        }
    }

    fn stem_bag(text: &str) -> Vec<f64> {
        let mut bag = vec![0.0; STEM_DIM];
        let lower = text.to_lowercase();
        for raw in lower.split(|c: char| !c.is_alphabetic()) {
            if raw.is_empty() {
                continue;
            }
            let mut h = STEM_HASH_SALT;
            for &b in raw.as_bytes() {
                h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(b as u64);
            }
            bag[(h as usize) % STEM_DIM] += 1.0;
        }
        let n = bag.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for v in bag.iter_mut() {
            *v /= n;
        }
        bag
    }

    fn pool_hidden(hidden: &Tensor) -> Result<Vec<f64>, String> {
        // sequence_hidden: [batch, seq, d_model] → mean over seq, then block-average to HIDDEN_DIM.
        // Block-mean preserva geometría semántica mejor que la proyección sinusoïdal fija
        // (crítica para E12: frontera lineal ES perro/gato generaliza a dog/chien/犬).
        let dims = hidden.dims();
        if dims.len() != 3 {
            return Err(format!("hidden rank esperado 3, got {dims:?}"));
        }
        let mean = hidden
            .mean(1)
            .map_err(|e| e.to_string())?
            .squeeze(0)
            .map_err(|e| e.to_string())?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| e.to_string())?
            .to_vec1::<f32>()
            .map_err(|e| e.to_string())?;
        let mut out = vec![0.0; HIDDEN_DIM];
        if mean.is_empty() {
            return Ok(out);
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let start = i * mean.len() / HIDDEN_DIM;
            let end = ((i + 1) * mean.len() / HIDDEN_DIM).max(start + 1);
            let slice = &mean[start..end.min(mean.len())];
            *slot = slice.iter().map(|&v| f64::from(v)).sum::<f64>() / slice.len().max(1) as f64;
        }
        let n = out.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for v in out.iter_mut() {
            *v /= n;
        }
        Ok(out)
    }
}

impl FrozenLinguisticProbe for FrozenGemma2Probe {
    fn analyze(&mut self, text: &str) -> LinguisticPacket {
        let tokens = self.tokenizer.encode(text).unwrap_or_default();
        let stem_bag = Self::stem_bag(text);
        if tokens.is_empty() {
            return LinguisticPacket::from_parts(
                Vec::new(),
                vec![0.0; HIDDEN_DIM],
                vec![0.0; GEMMA2_LAYER_COUNT],
                stem_bag,
            );
        }
        let ids = Tensor::new(tokens.as_slice(), &self.device).and_then(|t| t.unsqueeze(0));
        let Ok(ids) = ids else {
            return LinguisticPacket::from_parts(
                tokens,
                vec![0.0; HIDDEN_DIM],
                vec![0.0; GEMMA2_LAYER_COUNT],
                stem_bag,
            );
        };
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        model.clear_kv_cache();
        let out = model.forward_with_mask(&ids, 0, Some(&self.mask), None, true, true);
        drop(model);
        let Ok(out) = out else {
            return LinguisticPacket::from_parts(
                tokens,
                vec![0.0; HIDDEN_DIM],
                vec![0.0; GEMMA2_LAYER_COUNT],
                stem_bag,
            );
        };
        let mut layer_rms = vec![0.0; GEMMA2_LAYER_COUNT];
        for layer in &out.trace.layers {
            if layer.layer < GEMMA2_LAYER_COUNT && layer.executed {
                layer_rms[layer.layer] = f64::from(layer.output_rms);
            }
        }
        // Capas caras de T2.1 quedan en 0 aunque el trace las mencione.
        for &i in &EXPENSIVE_LAYERS {
            if i < layer_rms.len() {
                layer_rms[i] = 0.0;
            }
        }
        let hidden = out
            .sequence_hidden
            .as_ref()
            .and_then(|h| Self::pool_hidden(h).ok())
            .unwrap_or_else(|| vec![0.0; HIDDEN_DIM]);
        LinguisticPacket::from_parts(tokens, hidden, layer_rms, stem_bag)
    }

    fn name(&self) -> &'static str {
        "gemma2-frozen-gguf"
    }
}

/// Elige Gemma real si hay GGUF; si no, la sonda con forma de Gemma.
pub fn open_best_probe(_seed: u64) -> Result<Box<dyn FrozenLinguisticProbe>, String> {
    match FrozenGemma2Probe::try_open(None) {
        Ok(p) => Ok(Box::new(p)),
        Err(e) => Err(format!(
            "GGUF no disponible ({e}); usa GemmaShapedLexicon en tests o exporta GEMMA2_GGUF"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_linguistic_layer::{linguistic_feature_dim, LinguisticFieldCodec};
    use crate::field_substrate::{ComplexT, FieldState};

    #[test]
    fn gemma_probe_optional_or_skip() {
        match FrozenGemma2Probe::try_open(None) {
            Ok(mut probe) => {
                let codec = LinguisticFieldCodec::new(3);
                let mut psi = FieldState::new(ComplexT::octahedron());
                let packet = probe.analyze("el sol calienta la plaza");
                assert!(packet.token_count() > 0);
                let n = packet.token_count();
                codec.write_text(&mut probe, "el sol calienta la plaza", &mut psi);
                assert!(!psi.has_token_ids());
                assert_eq!(psi.n(), 12);
                // Tokens no sobrevivieron al write.
                assert!(n >= 1);
                assert_eq!(
                    linguistic_feature_dim(),
                    GEMMA2_LAYER_COUNT + HIDDEN_DIM + STEM_DIM
                );
            }
            Err(reason) => {
                eprintln!("skip GGUF: {reason}");
            }
        }
    }
}
