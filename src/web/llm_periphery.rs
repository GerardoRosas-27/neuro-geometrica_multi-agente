//! Periferia LLM: texto ↔ features/conceptos. Los tokens no cruzan al campo.
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

/// Decodificador periférico (concepto → texto). No escribe en FieldState.
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

    pub fn decode(&self, concept: usize) -> String {
        self.labels.decode_concept(concept)
    }

    /// Respuesta agentica en español a partir del concepto de salida + contexto.
    pub fn agent_reply(
        &self,
        user_msg: &str,
        concept_in: usize,
        concept_out: usize,
        route: &str,
        liquid_score: f64,
    ) -> String {
        let label_in = self.decode(concept_in);
        let label_out = self.decode(concept_out);
        let mode = self.mode.as_str();
        match self.mode {
            LlmMode::GemmaGguf => {
                // Sin generate completo en hot path (GGUF es pesado): plantilla
                // informada por etiquetas. El GGUF ya se usó en encode.
                format!(
                    "Entendido («{user_msg}»). Concepto {concept_in} ({label_in}) → \
                     {concept_out} ({label_out}) vía {route} (líquido={liquid_score:.3}, modo={mode})."
                )
            }
            LlmMode::Lexicon => {
                format!(
                    "Respuesta agentica (léxico): de «{label_in}» a «{label_out}» \
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
        // Forzar fallback: ruta inexistente vía env no debería panic.
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
    }

    #[test]
    fn encode_never_needs_gguf() {
        let mut p = open_best_probe(7);
        // Funciona con o sin GGUF.
        let _ = p.encode_concept("entrena el campo");
        assert!(!p.name().is_empty());
    }
}
