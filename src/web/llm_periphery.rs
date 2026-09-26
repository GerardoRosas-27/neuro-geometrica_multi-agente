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

use crate::field_gemma_probe::{FrozenGemma2Probe, RawGemmaHandle};
use crate::field_hybrid_infer::{features_to_concept_id, LabelDecoder, PeripheralDecoder};
use crate::field_linguistic_layer::{FrozenLinguisticProbe, GemmaShapedLexicon};
use crate::native_gemma2_runtime::Gemma2GenerationConfig;
use std::env;
use std::path::Path;

/// Número de conceptos discretos del chat agentico (alineado a fuse N=8).
pub const NUM_CONCEPTS: usize = 8;

/// Etiquetas en español para decode sin GGUF.
const ES_LABELS: [&str; 8] = [
    "alfa", "beta", "gamma", "delta", "épsilon", "zeta", "eta", "theta",
];

/// Familia de curriculum para rotación indefinida (Infinito).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetFamily {
    CoreAgentic,
    LiquidE8E10,
    FieldE11E17,
    AutonomyE18E30,
}

impl DatasetFamily {
    pub const ALL: [DatasetFamily; 4] = [
        Self::CoreAgentic,
        Self::LiquidE8E10,
        Self::FieldE11E17,
        Self::AutonomyE18E30,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CoreAgentic => "core_agentic",
            Self::LiquidE8E10 => "liquid_e8_e10",
            Self::FieldE11E17 => "field_e11_e17",
            Self::AutonomyE18E30 => "autonomy_e18_e30",
        }
    }

    pub fn experiment_ids(self) -> &'static [&'static str] {
        match self {
            Self::CoreAgentic => &["core"],
            Self::LiquidE8E10 => &["E8", "E9", "E10"],
            Self::FieldE11E17 => &["E11", "E12", "E13", "E14", "E15", "E16", "E17"],
            Self::AutonomyE18E30 => &[
                "E18", "E19", "E20", "E21", "E22", "E23", "E24", "E25", "E26", "E27", "E28", "E29",
                "E30",
            ],
        }
    }

    pub fn from_seed(seed: u64) -> Self {
        Self::ALL[(seed as usize) % Self::ALL.len()]
    }
}

/// Meta de un lote generado (familia + experimentos + fuente).
#[derive(Clone, Debug, serde::Serialize)]
pub struct TrainBatchMeta {
    pub source: &'static str,
    pub dataset_family: &'static str,
    pub experiment_ids: Vec<&'static str>,
}

/// Pares curriculum ES → concepto por familia experimental.
const CURRICULUM_CORE: &[(&str, usize)] = &[
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

const CURRICULUM_LIQUID: &[(&str, usize)] = &[
    ("invariancia stem perro perritos", 0),
    ("paráfrasis atractor concepto", 1),
    ("crosslingual ood rechazo", 2),
    ("composición multi hop A a C", 3),
    ("abstención margen bajo", 4),
    ("prediccion futura trayectoria", 5),
    ("bifurcacion onda liquida", 6),
    ("perturbacion compose wave", 7),
    ("ranked scores top1 top2", 0),
    ("energia surrogate ln score", 1),
];

const CURRICULUM_FIELD: &[(&str, usize)] = &[
    ("encoder theta campo entrenable", 0),
    ("dinamica phi rollout un paso", 1),
    ("geometria holdout E13", 2),
    ("separacion estructural E15", 3),
    ("margen ranking campo vs llm", 4),
    ("olvido adversarial E14", 5),
    ("extrapolacion geometrica E17", 6),
    ("crosslingual gguf E12", 7),
    ("field only sin contaminacion", 0),
    ("knn silhouette intra inter", 2),
];

const CURRICULUM_AUTONOMY: &[(&str, usize)] = &[
    ("regla consolidada experiencia", 0),
    ("extrapolacion misma regla", 1),
    ("clean room random init", 2),
    ("anti contamination seal test", 3),
    ("hiperparam lock smoke", 4),
    ("experiencia cdt vacia inicio", 5),
    ("transferencia regla A a B", 6),
    ("retencion tras interferencia", 7),
    ("rollback corrupcion parcial", 0),
    ("fase A scaffold field only", 1),
    ("veredicto POSITIVE PARTIAL NULL", 3),
    ("sin seeds confirmation B300", 4),
];

fn curriculum_for(family: DatasetFamily) -> &'static [(&'static str, usize)] {
    match family {
        DatasetFamily::CoreAgentic => CURRICULUM_CORE,
        DatasetFamily::LiquidE8E10 => CURRICULUM_LIQUID,
        DatasetFamily::FieldE11E17 => CURRICULUM_FIELD,
        DatasetFamily::AutonomyE18E30 => CURRICULUM_AUTONOMY,
    }
}

/// Variantes léxicas por seed (sin LLM generate; GGUF solo encode en el loop).
const GEMMA_VARIANTS: &[&str] = &[
    "narrativa",
    "glosa",
    "prompt",
    "ejemplo",
    "caso",
    "muestra",
    "instancia",
    "patrón",
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

    /// Handle al Gemma 2 original para chat crudo (`None` si no hay GGUF).
    pub fn raw_handle(&self) -> Option<RawGemmaHandle> {
        match self {
            Self::Gemma(p) => Some(p.raw_handle()),
            Self::Lexicon(_) => None,
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

/// Modo del chat (bandera UI «Decoder del campo»).
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatMode {
    /// ON: campo (líquido/CDT/RQM) produce el estado; Gemma solo interpreta (decoder-only).
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

/// Config de generación para chat crudo. Env: `RAW_CHAT_MAX_TOKENS` (def 256),
/// `RAW_CHAT_TEMPERATURE` (def 0.7), `RAW_CHAT_TOP_P` (def 0.9), `RAW_CHAT_CONTEXT` (def 2048).
pub fn raw_chat_config(seed: u64) -> Gemma2GenerationConfig {
    fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
        env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    Gemma2GenerationConfig {
        max_tokens: env_or("RAW_CHAT_MAX_TOKENS", 256usize).clamp(1, 2048),
        context_limit: env_or("RAW_CHAT_CONTEXT", 2048usize).clamp(128, 8192),
        temperature: env_or("RAW_CHAT_TEMPERATURE", 0.7f64).clamp(0.0, 2.0),
        top_p: env_or("RAW_CHAT_TOP_P", 0.9f64).clamp(0.05, 1.0),
        seed,
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
/// Rota familias E8–E30 / core según `seed` (round-robin vía `DatasetFamily::from_seed`).
/// - `gemma_available = true` → fuente `gemma`: textos **variables** por lote/seed
///   (plantillas experimentales + variantes; encode real con sonda Gemma en train).
///   Nota: FrozenGemma2Probe no expone generate; diversidad = curriculum ampliado.
/// - sin GGUF → `lexicon_synth` determinista alineado a la misma familia.
///
/// Nunca escribe tokens ni FieldState.
pub fn generate_train_batch(
    batch_size: usize,
    seed: u64,
    gemma_available: bool,
) -> (Vec<TrainExample>, TrainBatchMeta) {
    let family = DatasetFamily::from_seed(seed);
    generate_train_batch_family(batch_size, seed, gemma_available, family)
}

/// Igual que [`generate_train_batch`] fijando la familia (tests / UI selectiva).
pub fn generate_train_batch_family(
    batch_size: usize,
    seed: u64,
    gemma_available: bool,
    family: DatasetFamily,
) -> (Vec<TrainExample>, TrainBatchMeta) {
    let n = batch_size.clamp(1, 64);
    let source = if gemma_available {
        "gemma"
    } else {
        "lexicon_synth"
    };
    let curriculum = curriculum_for(family);
    let len = curriculum.len().max(1);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let idx = ((seed as usize).wrapping_add(i).wrapping_mul(7)) % len;
        let (base, concept) = curriculum[idx];
        let text = if gemma_available {
            let var = GEMMA_VARIANTS
                [(seed as usize).wrapping_add(i).wrapping_mul(13) % GEMMA_VARIANTS.len()];
            let tone = (seed ^ (i as u64).wrapping_mul(0x9E37)) % 5;
            match tone {
                0 => format!(
                    "{base} · {var} seed={seed:#x} i={i} familia={}",
                    family.as_str()
                ),
                1 => format!(
                    "[{var}/{i}] {base} (exp {})",
                    family.experiment_ids().first().unwrap_or(&"?")
                ),
                2 => format!("dataset {var}: {base} · lote∞ {i}@{seed}"),
                3 => format!("{base} | variante {var}#{i} · campo tokenless"),
                _ => format!("gemma·{var} «{base}» batch={i} fam={}", family.as_str()),
            }
        } else {
            base.to_string()
        };
        out.push(TrainExample {
            text,
            concept: concept % NUM_CONCEPTS,
        });
    }
    let meta = TrainBatchMeta {
        source,
        dataset_family: family.as_str(),
        experiment_ids: family.experiment_ids().to_vec(),
    };
    (out, meta)
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
    fn chat_mode_default_is_field_decoder() {
        assert_eq!(
            ChatMode::resolve(None, None).unwrap(),
            ChatMode::FieldDecoder
        );
        assert_eq!(
            ChatMode::resolve(Some(""), None).unwrap(),
            ChatMode::FieldDecoder
        );
    }

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
    fn lexicon_probe_has_no_raw_handle() {
        let probe = PeripheralProbe::Lexicon(GemmaShapedLexicon::new(7));
        assert!(probe.raw_handle().is_none());
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
        let (items, meta) = generate_train_batch(6, 99, false);
        assert_eq!(items.len(), 6);
        assert_eq!(meta.source, "lexicon_synth");
        assert!(!meta.dataset_family.is_empty());
        assert!(!meta.experiment_ids.is_empty());
        let (g_items, g_meta) = generate_train_batch(4, 1, true);
        assert_eq!(g_items.len(), 4);
        assert_eq!(g_meta.source, "gemma");
        // Textos variables (no solo "· lote i").
        assert!(g_items.iter().any(|e| e.text.contains("familia=")
            || e.text.contains("gemma·")
            || e.text.contains("dataset ")));
    }

    #[test]
    fn generate_train_batch_covers_all_families() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0u64..16 {
            let (_items, meta) = generate_train_batch(3, seed, false);
            seen.insert(meta.dataset_family);
        }
        for f in DatasetFamily::ALL {
            assert!(seen.contains(f.as_str()), "missing family {}", f.as_str());
        }
    }

    #[test]
    fn family_curriculum_non_empty() {
        for f in DatasetFamily::ALL {
            let (items, meta) = generate_train_batch_family(4, 42, false, f);
            assert_eq!(items.len(), 4);
            assert_eq!(meta.dataset_family, f.as_str());
            assert!(!items[0].text.is_empty());
        }
    }

    #[test]
    fn decoder_returns_non_empty() {
        let d = ConceptDecoder::new(LlmMode::Lexicon);
        for c in 0..NUM_CONCEPTS {
            assert!(!d.decode_field_concept(c).is_empty());
        }
    }
}
