//! Periferia LLM: texto ↔ features/conceptos. Los tokens no cruzan al campo.
//!
//! Roles:
//! - **Encoder** (sonda): texto → features → `concept_id` (firewall tira tokens).
//! - **Decoder only**: `concept_id` / campo → texto (`decode_field_concept`).
//! - **Dataset gen** (periferia aparte): `generate_train_batch` para entrenamiento
//!   tokenless; no escribe en FieldState.
//!
//! `open_best_probe()` intenta `FrozenGemma2Probe` (env `GEMMA2_GGUF` o rutas
//! comunes). Si no hay GGUF → `GemmaShapedLexicon` + `LabelDecoder` (siempre arranca).

use crate::field_gemma_probe::FrozenGemma2Probe;
use crate::field_hybrid_infer::{features_to_concept_id, LabelDecoder, PeripheralDecoder};
use crate::field_linguistic_layer::{FrozenLinguisticProbe, GemmaShapedLexicon};
use std::env;
use std::path::Path;

/// Número de conceptos discretos del chat agentico (alineado a fuse N=8).
pub const NUM_CONCEPTS: usize = 8;

/// Etiquetas en español para decode sin GGUF.
const ES_LABELS: [&str; 8] = [
    "alfa", "beta", "gamma", "delta", "épsilon", "zeta", "eta", "theta",
];

/// Pares curriculum ES → concepto (dataset sintético determinista).
const CURRICULUM: &[(&str, usize)] = &[
    ("hola campo líquido", 0),
    ("sueño consolidar memoria", 1),
    ("onda predictiva rápida", 2),
    ("engrama termo durable", 3),
    ("relación cue etiqueta", 4),
    ("inferencia sin tokens", 5),
    ("núcleo líquido wave", 6),
    ("decodificar concepto campo", 7),
    ("entrenar lote periferia", 0),
    ("consolidación cdt por lotes", 1),
    ("ruta líquida preferente", 2),
    ("fallback relacional rqm", 3),
    ("features a concept id", 4),
    ("firewall sin token ids", 5),
    ("telemetría en vivo", 6),
    ("vista previa decodificada", 7),
];

/// Ejemplo de dataset para entrenamiento tokenless (periferia).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainExample {
    pub text: String,
    pub concept: usize,
}

/// Modo de la periferia lingüística.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmMode {
    GemmaGguf,
    Lexicon,
}

impl LlmMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GemmaGguf => "gemma_gguf",
            Self::Lexicon => "lexicon",
        }
    }
}

/// Sonda periférica: Gemma real o léxico con forma de Gemma.
#[allow(clippy::large_enum_variant)]
pub enum PeripheralProbe {
    Gemma(FrozenGemma2Probe),
    Lexicon(GemmaShapedLexicon),
}

impl PeripheralProbe {
    pub fn mode(&self) -> LlmMode {
        match self {
            Self::Gemma(_) => LlmMode::GemmaGguf,
            Self::Lexicon(_) => LlmMode::Lexicon,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemma(p) => p.name(),
            Self::Lexicon(p) => p.name(),
        }
    }

    /// Texto → features (firewall tira tokens) → concept_id. Nunca toca FieldState.
    pub fn encode_concept(&mut self, text: &str) -> usize {
        let features = match self {
            Self::Gemma(p) => p.analyze(text).into_field_features(),
            Self::Lexicon(p) => p.analyze(text).into_field_features(),
        };
        features_to_concept_id(&features, NUM_CONCEPTS)
    }
}

/// Abre la mejor sonda disponible. **Siempre** OK: fallback a léxico.
pub fn open_best_probe(seed: u64) -> PeripheralProbe {
    let explicit = env::var("GEMMA2_GGUF").ok();
    let path_ref = explicit.as_deref().map(Path::new);
    match FrozenGemma2Probe::try_open(path_ref) {
        Ok(p) => {
            tracing::info!(probe = p.name(), "periferia LLM: Gemma GGUF");
            PeripheralProbe::Gemma(p)
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "GGUF no disponible; periferia = GemmaShapedLexicon (Railway default)"
            );
            PeripheralProbe::Lexicon(GemmaShapedLexicon::new(seed))
        }
    }
}

/// Genera un lote de entrenamiento en periferia.
///
/// - `gemma_available = true` → fuente `gemma` (curriculum etiquetado; encode real
///   con sonda Gemma en el loop de train).
/// - sin GGUF → `lexicon_synth` determinista.
///
/// Nunca escribe tokens ni FieldState.
pub fn generate_train_batch(
    batch_size: usize,
    seed: u64,
    gemma_available: bool,
) -> (Vec<TrainExample>, &'static str) {
    let n = batch_size.clamp(1, 64);
    let source = if gemma_available {
        "gemma"
    } else {
        "lexicon_synth"
    };
    let mut out = Vec::with_capacity(n);
    let len = CURRICULUM.len();
    for i in 0..n {
        let idx = ((seed as usize).wrapping_add(i).wrapping_mul(7)) % len;
        let (text, concept) = CURRICULUM[idx];
        // Variante ligera del prompt para diversidad sin LLM generate pesado.
        let text = if gemma_available {
            format!("{text} · lote {i}")
        } else {
            text.to_string()
        };
        out.push(TrainExample {
            text,
            concept: concept % NUM_CONCEPTS,
        });
    }
    (out, source)
}

/// Decodificador periférico (concepto → texto). **Solo decoder** del modelo de campo.
/// No escribe en FieldState.
pub struct ConceptDecoder {
    labels: LabelDecoder,
    mode: LlmMode,
}

impl ConceptDecoder {
    pub fn new(mode: LlmMode) -> Self {
        let mut labels = LabelDecoder::concepts8();
        // Sobrescribe con etiquetas ES para UI en español.
        labels.labels = ES_LABELS.to_vec();
        Self { labels, mode }
    }

    pub fn mode(&self) -> LlmMode {
        self.mode
    }

    pub fn decode(&self, concept: usize) -> String {
        self.labels.decode_concept(concept)
    }

    /// Decoder-only explícito: concepto/campo → texto (Gemma labels o LabelDecoder ES).
    pub fn decode_field_concept(&self, concept: usize) -> String {
        let label = self.decode(concept);
        match self.mode {
            LlmMode::GemmaGguf => {
                format!("[decoder·gemma] concepto {concept} → «{label}»")
            }
            LlmMode::Lexicon => label,
        }
    }

    /// Respuesta agentica en español: inferencia de campo → **decode LLM only**.
    pub fn agent_reply(
        &self,
        user_msg: &str,
        concept_in: usize,
        concept_out: usize,
        route: &str,
        liquid_score: f64,
    ) -> String {
        let label_in = self.decode_field_concept(concept_in);
        let label_out = self.decode_field_concept(concept_out);
        let mode = self.mode.as_str();
        match self.mode {
            LlmMode::GemmaGguf => {
                format!(
                    "Campo→texto (decoder only): «{user_msg}». \
                     {concept_in} ({label_in}) → {concept_out} ({label_out}) \
                     vía {route} (líquido={liquid_score:.3}, modo={mode})."
                )
            }
            LlmMode::Lexicon => {
                format!(
                    "Decoder de campo (léxico): de «{label_in}» a «{label_out}» \
                     por ruta {route} (score líquido {liquid_score:.3}). \
                     Mensaje: {user_msg}"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexicon_probe_always_opens() {
        let probe = PeripheralProbe::Lexicon(GemmaShapedLexicon::new(42));
        assert_eq!(probe.mode(), LlmMode::Lexicon);
        let mut p = probe;
        let c = p.encode_concept("hola mundo agentico");
        assert!(c < NUM_CONCEPTS);
    }

    #[test]
    fn decoder_spanish_labels() {
        let d = ConceptDecoder::new(LlmMode::Lexicon);
        assert_eq!(d.decode(0), "alfa");
        assert_eq!(d.decode(3), "delta");
        assert!(!d.decode_field_concept(2).is_empty());
    }

    #[test]
    fn encode_never_needs_gguf() {
        let mut p = open_best_probe(7);
        let _ = p.encode_concept("entrena el campo");
        assert!(!p.name().is_empty());
    }

    #[test]
    fn generate_train_batch_n_and_source() {
        let (items, src) = generate_train_batch(6, 99, false);
        assert_eq!(items.len(), 6);
        assert_eq!(src, "lexicon_synth");
        let (g_items, g_src) = generate_train_batch(4, 1, true);
        assert_eq!(g_items.len(), 4);
        assert_eq!(g_src, "gemma");
    }

    #[test]
    fn decoder_returns_non_empty() {
        let d = ConceptDecoder::new(LlmMode::Lexicon);
        for c in 0..NUM_CONCEPTS {
            assert!(!d.decode_field_concept(c).is_empty());
        }
    }
}
