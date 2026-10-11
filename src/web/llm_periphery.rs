//! Periferia LLM de la app (Gemma 2 local o API externa).
//!
//! El LLM es **periferia**: verbaliza el estado del campo v4 (modo ON,
//! decoder-only) o conversa sin campo (modo OFF). Nunca escribe en Dφ ni en la
//! CDT. `open_best_probe()` carga el GGUF si existe (env `GEMMA2_GGUF` o
//! `models/gemma-2-2b-it-Q3_K_L.gguf`); si no, la app arranca igual.

use crate::field_gemma_probe::{FrozenGemma2Probe, RawGemmaHandle};
use crate::native_gemma2_runtime::Gemma2GenerationConfig;
use std::env;
use std::path::Path;

/// Estado del LLM local.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmMode {
    GemmaGguf,
    Unloaded,
}

impl LlmMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GemmaGguf => "gemma_gguf",
            Self::Unloaded => "sin_gguf",
        }
    }
}

/// Gemma local cargado (o no).
#[allow(clippy::large_enum_variant)]
pub enum PeripheralProbe {
    Gemma(FrozenGemma2Probe),
    Unloaded,
}

impl PeripheralProbe {
    pub fn mode(&self) -> LlmMode {
        match self {
            Self::Gemma(_) => LlmMode::GemmaGguf,
            Self::Unloaded => LlmMode::Unloaded,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemma(_) => "gemma2-frozen-gguf",
            Self::Unloaded => "sin-gguf",
        }
    }

    /// Handle al Gemma 2 original (`None` si no hay GGUF).
    pub fn raw_handle(&self) -> Option<RawGemmaHandle> {
        match self {
            Self::Gemma(p) => Some(p.raw_handle()),
            Self::Unloaded => None,
        }
    }
}

/// Modo del chat (bandera UI «Decoder del campo»).
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatMode {
    /// ON: el campo v4 (Dφ) produce el estado; el LLM solo lo verbaliza (decoder-only).
    FieldDecoder,
    /// OFF: Gemma 2 congelado original como LLM plano. Sin campo.
    GemmaRaw,
}

impl ChatMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FieldDecoder => "field_decoder",
            Self::GemmaRaw => "gemma_raw",
        }
    }

    /// Resuelve el modo desde el request. `mode` (string) tiene prioridad sobre
    /// `field_decoder` (bool). Ausente → `FieldDecoder` (comportamiento histórico).
    pub fn resolve(mode: Option<&str>, field_decoder: Option<bool>) -> Result<Self, String> {
        if let Some(m) = mode {
            let m = m.trim().to_ascii_lowercase();
            return match m.as_str() {
                "" => Ok(field_decoder
                    .map(Self::from_flag)
                    .unwrap_or(Self::FieldDecoder)),
                "field_decoder" | "field" | "decoder" | "campo" => Ok(Self::FieldDecoder),
                "gemma_raw" | "raw" | "gemma" | "llm" => Ok(Self::GemmaRaw),
                other => Err(format!(
                    "modo de chat desconocido: «{other}» (usa \"field_decoder\" o \"gemma_raw\")"
                )),
            };
        }
        Ok(field_decoder
            .map(Self::from_flag)
            .unwrap_or(Self::FieldDecoder))
    }

    fn from_flag(on: bool) -> Self {
        if on {
            Self::FieldDecoder
        } else {
            Self::GemmaRaw
        }
    }
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Config de generación para chat crudo. Env: `RAW_CHAT_MAX_TOKENS` (def 160),
/// `RAW_CHAT_TEMPERATURE` (def 0.7), `RAW_CHAT_TOP_P` (def 0.9), `RAW_CHAT_CONTEXT` (def 2048).
pub fn raw_chat_config(seed: u64) -> Gemma2GenerationConfig {
    Gemma2GenerationConfig {
        max_tokens: env_or("RAW_CHAT_MAX_TOKENS", 160usize).clamp(1, 2048),
        context_limit: env_or("RAW_CHAT_CONTEXT", 2048usize).clamp(128, 8192),
        temperature: env_or("RAW_CHAT_TEMPERATURE", 0.7f64).clamp(0.0, 2.0),
        top_p: env_or("RAW_CHAT_TOP_P", 0.9f64).clamp(0.05, 1.0),
        seed,
    }
}

/// Config del decoder del campo (ON). Respuestas cortas y más deterministas.
/// Env: `FIELD_DECODER_MAX_TOKENS` (def 96), `FIELD_DECODER_TEMPERATURE` (def 0.4).
pub fn field_decoder_config(seed: u64) -> Gemma2GenerationConfig {
    Gemma2GenerationConfig {
        max_tokens: env_or("FIELD_DECODER_MAX_TOKENS", 96usize).clamp(1, 1024),
        context_limit: 1024,
        temperature: env_or("FIELD_DECODER_TEMPERATURE", 0.4f64).clamp(0.0, 2.0),
        top_p: 0.9,
        seed,
    }
}

/// Plazo de generación por mensaje (s). Env `CHAT_TIMEOUT_SECS` (def 75).
/// Al vencer se corta la generación y se devuelve texto parcial o error claro.
pub fn chat_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(env_or("CHAT_TIMEOUT_SECS", 75u64).clamp(5, 600))
}

/// Intenta abrir Gemma desde una ruta concreta.
pub fn open_gemma_probe(path: &Path) -> Result<PeripheralProbe, String> {
    FrozenGemma2Probe::try_open(Some(path)).map(PeripheralProbe::Gemma)
}

/// Abre Gemma local **sin descargar**. Siempre OK: sin GGUF → `Unloaded`.
/// Ruta: `GEMMA2_GGUF` o `models/gemma-2-2b-it-Q3_K_L.gguf` (ver `model_fetch`).
/// En tests solo se usa `GEMMA2_GGUF` explícito (no cargar 1 GB por test).
pub fn open_best_probe(seed: u64) -> PeripheralProbe {
    let explicit = env::var("GEMMA2_GGUF")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let path = if cfg!(test) {
        explicit.map(std::path::PathBuf::from)
    } else {
        Some(crate::web::model_fetch::ModelConfig::from_env().path)
    };
    let result = match &path {
        Some(p) if p.is_file() => open_gemma_probe(p),
        Some(p) => Err(format!("GGUF no encontrado: {}", p.display())),
        None => Err("GEMMA2_GGUF no definido".into()),
    };
    match result {
        Ok(p) => {
            tracing::info!(probe = p.name(), "periferia LLM: Gemma GGUF");
            p
        }
        Err(e) => {
            tracing::warn!(error = %e, "GGUF no disponible (aún); chat OFF/decoder sin Gemma local");
            let _ = seed;
            PeripheralProbe::Unloaded
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_mode_parses_strings_and_flag() {
        assert_eq!(
            ChatMode::resolve(Some("gemma_raw"), None).unwrap(),
            ChatMode::GemmaRaw
        );
        assert_eq!(
            ChatMode::resolve(Some(" RAW "), None).unwrap(),
            ChatMode::GemmaRaw
        );
        assert_eq!(
            ChatMode::resolve(Some("field_decoder"), None).unwrap(),
            ChatMode::FieldDecoder
        );
        assert_eq!(
            ChatMode::resolve(None, Some(false)).unwrap(),
            ChatMode::GemmaRaw
        );
        assert_eq!(
            ChatMode::resolve(None, Some(true)).unwrap(),
            ChatMode::FieldDecoder
        );
        // `mode` gana sobre el bool.
        assert_eq!(
            ChatMode::resolve(Some("gemma_raw"), Some(true)).unwrap(),
            ChatMode::GemmaRaw
        );
        assert!(ChatMode::resolve(Some("otro"), None).is_err());
        assert_eq!(ChatMode::GemmaRaw.as_str(), "gemma_raw");
    }

    #[test]
    fn unloaded_probe_has_no_raw_handle() {
        let probe = PeripheralProbe::Unloaded;
        assert!(probe.raw_handle().is_none());
        assert_eq!(probe.mode(), LlmMode::Unloaded);
    }

    #[test]
    fn raw_chat_config_is_sane() {
        let c = raw_chat_config(1);
        assert!(c.max_tokens >= 1 && c.max_tokens <= 2048);
        assert!(c.context_limit > c.max_tokens);
        assert!(c.temperature >= 0.0);
    }

    #[test]
    fn raw_prompt_uses_gemma_template_without_field() {
        let p = crate::field_gemma_probe::render_raw_gemma_prompt(
            &[("hola".into(), "¡Hola!".into())],
            "¿qué es un campo?",
        );
        assert!(p.starts_with("<start_of_turn>user\nhola<end_of_turn>"));
        assert!(p.ends_with(
            "<start_of_turn>user\n¿qué es un campo?<end_of_turn>\n<start_of_turn>model\n"
        ));
        assert!(!p.contains("decoder"));
    }
}
