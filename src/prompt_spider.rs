//! # Prompt Spider — «cada palabra, un fork»
//!
//! Recreación (experimental) de un experimento visto en TikTok: en lugar de
//! apuntar un crawler a la web, se «rastrea» el propio prompt. Cada palabra es
//! un **punto de decisión** que se evalúa con cuatro preguntas fijas:
//!
//! 1. `relevant`  — ¿es relevante para la tarea?
//! 2. `grounded`  — ¿está fundamentada (su afirmación tiene fuente en el prompt)?
//! 3. `ambiguous` — ¿es ambigua?
//! 4. `approval`  — ¿requiere aprobación humana?
//!
//! Cada pregunta produce una probabilidad (`P(sí)`), calculada con heurísticas
//! **deterministas y documentadas** (ver [`score_token`] y
//! `docs/experimento_prompt_spider.md`). El enrutado no tiene opinión propia:
//!
//! - `P(aprobación) ≥ 0.5` → **you** (cola de aprobación humana; nunca se
//!   resuelve sola);
//! - confianza `p ≥ umbral` (0.95 por defecto) → **code** (ramas `if`, sin LLM);
//! - resto → **llm**: se escala al LLM activo de la app, por lotes, pidiendo un
//!   veredicto JSON. Si el LLM falla, no responde o duda, la palabra queda
//!   **pendiente** para el humano: nunca se inventa un veredicto.
//!
//! Este módulo no depende de la feature `web` (corre en el smoke de CI). El
//! job asíncrono, la persistencia y los endpoints viven en
//! `web::spider_job`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Umbral por defecto para el camino determinista (`p ≥ umbral → code`).
pub const DEFAULT_THRESHOLD: f64 = 0.95;
/// Confianza mínima de un veredicto del LLM para darlo por resuelto; por
/// debajo, la palabra queda pendiente para el humano.
pub const DEFAULT_LLM_FLOOR: f64 = 0.70;
/// `P(aprobación)` a partir de la cual la palabra va a la cola humana.
pub const APPROVAL_CUTOFF: f64 = 0.5;

/// Prompt de ejemplo (original, en el espíritu del vídeo: chief of staff de
/// un estudio de contenido con secciones `<role>` / `<objective>`).
pub const SAMPLE_PROMPT: &str = "<prompt>
<role>
You are the chief of staff for a small content studio, powered by one reasoning model.
Turn a raw source packet into a launch-ready release with clear owners,
traceable claims and drafts I can approve.
</role>
<objective>
Ship one complete release: three short posts, one vertical video script,
a cover image and a manifest. Start with one manual run, inspect the
results, then repeat it.
</objective>
<sources>
Use only the files under /packet: interview transcript, brand voice notes
and the 2026 sales sheet. Every claim must cite a source line.
</sources>
<steps>
1. Researcher - extract claims with a quote and source line.
2. Writer - draft posts in the brand voice, max 280 characters.
3. Editor - cut, fact-check and flag anything vague.
4. Publisher - format the video at 1080x1920, 45 seconds, then ask
me before you publish or send anything.
</steps>
<constraints>
Never invent numbers. If something seems ambiguous, mark it pending
instead of guessing. Approval is required for spending, publishing
and external emails. This should be the best release we ship.
</constraints>
</prompt>";

// ---------------------------------------------------------------------------
// Tokenizador
// ---------------------------------------------------------------------------

/// Tipo de token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenKind {
    /// Palabra (letras, con `'`, `-`, `_` internos).
    Word,
    /// Número o token de formato (`2026`, `1080x1920`, `45`).
    Number,
    /// Etiqueta estructural `<role>` / `</role>`: se lee, no es un fork.
    Tag,
    /// Marcador de lista (`1.`, `-`, `*`, `•`): se lee, no es un fork.
    ListMarker,
}

impl TokenKind {
    /// ¿Es un punto de decisión (fork)?
    pub fn is_fork(self) -> bool {
        matches!(self, TokenKind::Word | TokenKind::Number)
    }
}

/// Token con su posición en **caracteres** (no bytes): coincide con los
/// índices de `String` en JS para texto BMP (español/inglés).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub index: usize,
    pub text: String,
    pub lower: String,
    pub kind: TokenKind,
    /// Offset en caracteres (inicio inclusive, fin exclusivo).
    pub start: usize,
    pub end: usize,
    /// Línea (0-based).
    pub line: usize,
    /// Sección más interna abierta (`role`, `objective`, …; `""` = ninguna).
    pub section: String,
    /// Inicio de frase (tras `.`/`!`/`?`/`:`, salto de línea, etiqueta o
    /// marcador de lista): una mayúscula aquí no indica nombre propio.
    #[serde(default)]
    pub sentence_start: bool,
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

fn is_joiner(c: char) -> bool {
    matches!(c, '\'' | '’' | '-' | '_')
}

/// Intenta leer una etiqueta `<name>` / `</name>` en `chars[i..]`.
/// Devuelve (nombre, cierre, longitud).
fn read_tag(chars: &[char], i: usize) -> Option<(String, bool, usize)> {
    if chars.get(i) != Some(&'<') {
        return None;
    }
    let mut j = i + 1;
    let closing = chars.get(j) == Some(&'/');
    if closing {
        j += 1;
    }
    let name_start = j;
    while j < chars.len()
        && (chars[j].is_ascii_alphanumeric() || chars[j] == '_' || chars[j] == '-')
    {
        j += 1;
    }
    if j == name_start || !chars[name_start].is_ascii_alphabetic() {
        return None;
    }
    let name: String = chars[name_start..j].iter().collect();
    while j < chars.len() && chars[j] == ' ' {
        j += 1;
    }
    if chars.get(j) != Some(&'>') {
        return None;
    }
    Some((name.to_lowercase(), closing, j + 1 - i))
}

/// Tokeniza un prompt en palabras/números/etiquetas/marcadores de lista con
/// spans en caracteres y la sección `<…>` que los contiene.
pub fn tokenize(text: &str) -> Vec<Token> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Token> = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut line = 0usize;
    let mut line_has_content = false;
    let mut i = 0usize;
    let push =
        |out: &mut Vec<Token>, kind: TokenKind, s: usize, e: usize, line: usize, section: &str| {
            let t: String = chars[s..e].iter().collect();
            let lower = t.to_lowercase();
            let before = chars[..s].iter().rev().find(|c| **c != ' ' && **c != '\t');
            let sentence_start = match before {
                None => true,
                Some(c) => matches!(
                    c,
                    '.' | '!' | '?' | ':' | '\n' | '>' | '-' | '*' | '•' | ')'
                ),
            };
            out.push(Token {
                index: out.len(),
                text: t,
                lower,
                kind,
                start: s,
                end: e,
                line,
                section: section.to_string(),
                sentence_start,
            });
        };
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            line += 1;
            line_has_content = false;
            i += 1;
            continue;
        }
        if let Some((name, closing, len)) = read_tag(&chars, i) {
            let section = stack.last().cloned().unwrap_or_default();
            if closing {
                if let Some(pos) = stack.iter().rposition(|s| *s == name) {
                    stack.truncate(pos);
                }
                push(&mut out, TokenKind::Tag, i, i + len, line, &section);
            } else {
                stack.push(name.clone());
                push(&mut out, TokenKind::Tag, i, i + len, line, &name);
            }
            line_has_content = true;
            i += len;
            continue;
        }
        let section = stack.last().cloned().unwrap_or_default();
        // Marcadores de lista al inicio de línea: `1.`, `2)`, `-`, `*`, `•`.
        if !line_has_content && !c.is_whitespace() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let numbered = j > i
                && j < chars.len()
                && matches!(chars[j], '.' | ')')
                && chars.get(j + 1).is_some_and(|c| c.is_whitespace());
            if numbered {
                push(&mut out, TokenKind::ListMarker, i, j + 1, line, &section);
                line_has_content = true;
                i = j + 1;
                continue;
            }
            if matches!(c, '-' | '*' | '•') && chars.get(i + 1).is_some_and(|c| c.is_whitespace())
            {
                push(&mut out, TokenKind::ListMarker, i, i + 1, line, &section);
                line_has_content = true;
                i += 1;
                continue;
            }
        }
        if !c.is_whitespace() {
            line_has_content = true;
        }
        if is_word_char(c) {
            let s = i;
            let mut j = i + 1;
            loop {
                if j < chars.len() && is_word_char(chars[j]) {
                    j += 1;
                } else if j + 1 < chars.len() && is_joiner(chars[j]) && is_word_char(chars[j + 1]) {
                    j += 2;
                } else {
                    break;
                }
            }
            let kind = if chars[s].is_ascii_digit() {
                TokenKind::Number
            } else {
                TokenKind::Word
            };
            push(&mut out, kind, s, j, line, &section);
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Léxicos (todos en minúsculas; inglés + español)
// ---------------------------------------------------------------------------

/// Palabras estructurales: el código las resuelve sin pensar.
pub const STOPWORDS: &[&str] = &[
    // en
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on", "for", "with", "by", "at",
    "from", "as", "into", "onto", "under", "over", "about", "is", "are", "was", "were", "be",
    "been", "being", "am", "do", "does", "did", "has", "have", "had", "i", "you", "we", "me", "my",
    "your", "our", "us", "he", "she", "his", "her", "its", "their", "then", "than", "so", "if",
    "else", "not", "no", "can", "could", "will", "would", "should", "must", "may", "might",
    "shall", "one", "each", "per", "via", "up", "out", "before", "after", "only", "also", "just",
    "what", "which", "who", "whom", "when", "where", "why", "how", "there", "here", "all", "any",
    "both", "own", "same", "too", "very", "because", "while", "until", "through", "during",
    "without", "within", "instead", "max", "min", // es
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "u", "e", "ni", "de", "del",
    "al", "en", "con", "por", "para", "sin", "sobre", "entre", "hasta", "desde", "que", "se", "su",
    "sus", "es", "son", "ser", "fue", "era", "está", "están", "estar", "yo", "tú", "tu", "mi",
    "mis", "nos", "nosotros", "le", "les", "lo", "como", "pero", "si", "no", "más", "menos",
    "cada", "solo", "sólo", "también", "luego", "antes", "después", "cuando", "donde", "porque",
    "debe", "puede", "muy", "ya",
];

/// Marcadores de ambigüedad / deixis: «¿es específico?» queda en duda.
pub const AMBIGUITY_MARKERS: &[&str] = &[
    "it",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "something",
    "anything",
    "everything",
    "stuff",
    "things",
    "thing",
    "maybe",
    "perhaps",
    "some",
    "several",
    "few",
    "many",
    "soon",
    "later",
    "appropriate",
    "relevant",
    "good",
    "nice",
    "better",
    "quick",
    "quickly",
    "etc",
    "whatever",
    "somehow",
    "probably",
    "usually",
    "similar",
    "various",
    "seems",
    "vague",
    "kind",
    "short",
    "clear",
    "esto",
    "eso",
    "algo",
    "quizás",
    "quizá",
    "pronto",
    "algunos",
    "algunas",
    "varios",
    "bueno",
    "adecuado",
    "cosas",
    "etc.",
];

/// Afirmaciones sin fuente: «¿está fundamentada?» queda en duda.
pub const CLAIM_MARKERS: &[&str] = &[
    "best",
    "guaranteed",
    "always",
    "never",
    "proven",
    "viral",
    "everyone",
    "every",
    "fastest",
    "perfect",
    "complete",
    "launch-ready",
    "mejor",
    "garantizado",
    "siempre",
    "nunca",
    "todos",
    "viralizar",
    "perfecto",
];

/// Verbos/sustantivos que comprometen hacia fuera: `P(aprobación)`.
/// `ask` lleva 0.544 a propósito (el ejemplo del vídeo): justo por encima del
/// corte, así que queda **pendiente** en vez de resolverse solo.
pub const APPROVAL_TERMS: &[(&str, f64)] = &[
    ("approve", 0.97),
    ("approval", 0.96),
    ("publish", 0.95),
    ("publishing", 0.93),
    ("send", 0.92),
    ("ship", 0.90),
    ("accept", 0.90),
    ("sign", 0.88),
    ("pay", 0.94),
    ("purchase", 0.94),
    ("buy", 0.93),
    ("spend", 0.92),
    ("spending", 0.90),
    ("delete", 0.93),
    ("post", 0.66),
    ("deploy", 0.92),
    ("merge", 0.85),
    ("submit", 0.84),
    ("emails", 0.62),
    ("email", 0.62),
    ("ask", 0.544),
    ("request", 0.53),
    ("confirm", 0.58),
    ("decide", 0.52),
    ("aprobar", 0.97),
    ("aprueba", 0.96),
    ("aprobación", 0.96),
    ("publicar", 0.95),
    ("publica", 0.94),
    ("enviar", 0.92),
    ("envía", 0.92),
    ("aceptar", 0.90),
    ("firmar", 0.88),
    ("pagar", 0.94),
    ("comprar", 0.93),
    ("borrar", 0.93),
    ("eliminar", 0.92),
    ("preguntar", 0.544),
    ("pregunta", 0.544),
    ("confirmar", 0.58),
];

/// Entidades conocidas (plataformas, formatos, herramientas): relevantes y
/// fundamentadas por sí mismas.
pub const KNOWN_ENTITIES: &[&str] = &[
    "tiktok",
    "youtube",
    "instagram",
    "linkedin",
    "twitter",
    "x",
    "facebook",
    "threads",
    "mp4",
    "mov",
    "srt",
    "json",
    "csv",
    "markdown",
    "md",
    "pdf",
    "png",
    "jpg",
    "slack",
    "notion",
    "gmail",
    "drive",
    "figma",
    "canva",
    "sonnet",
    "claude",
    "gemma",
    "openai",
    "gpt",
    "llm",
    "api",
    "url",
    "github",
    "manifest",
    "transcript",
    "packet",
];

fn in_list(list: &[&str], w: &str) -> bool {
    list.contains(&w)
}

/// `P(aprobación)` léxica para una palabra en minúsculas (0 si no aplica).
pub fn approval_prior(lower: &str) -> f64 {
    APPROVAL_TERMS
        .iter()
        .find(|(t, _)| *t == lower)
        .map(|(_, p)| *p)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Contexto del prompt + scorer
// ---------------------------------------------------------------------------

/// Hechos del prompt completo que necesita el scorer.
#[derive(Clone, Debug, Default)]
pub struct PromptContext {
    /// Frecuencia (minúsculas) de cada palabra/número.
    pub freq: HashMap<String, usize>,
    /// Palabras que aparecen dentro de `<objective>`.
    pub objective: HashSet<String>,
    /// Palabras que aparecen dentro de `<role>`.
    pub role: HashSet<String>,
}

impl PromptContext {
    pub fn new(tokens: &[Token]) -> Self {
        let mut c = Self::default();
        for t in tokens.iter().filter(|t| t.kind.is_fork()) {
            *c.freq.entry(t.lower.clone()).or_default() += 1;
            match t.section.as_str() {
                "objective" | "objetivo" | "goal" => {
                    c.objective.insert(t.lower.clone());
                }
                "role" | "rol" => {
                    c.role.insert(t.lower.clone());
                }
                _ => {}
            }
        }
        c
    }
}

/// Las cuatro preguntas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Question {
    Relevant,
    Grounded,
    Ambiguous,
    Approval,
}

/// `P(sí)` para cada pregunta.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub relevant: f64,
    pub grounded: f64,
    pub ambiguous: f64,
    pub needs_approval: f64,
}

impl Scores {
    fn get(&self, q: Question) -> f64 {
        match q {
            Question::Relevant => self.relevant,
            Question::Grounded => self.grounded,
            Question::Ambiguous => self.ambiguous,
            Question::Approval => self.needs_approval,
        }
    }
}

/// Certeza de una respuesta binaria: `max(q, 1-q)`.
pub fn certainty(q: f64) -> f64 {
    q.max(1.0 - q)
}

/// Evaluación determinista de una palabra.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Assessment {
    pub scores: Scores,
    /// Pregunta decisiva (la menos segura, o `approval` si va al humano).
    pub question: Question,
    /// Confianza de la decisión (la certeza de la pregunta decisiva; para
    /// `approval` es `P(aprobación)`).
    pub p: f64,
    /// Reglas que dispararon (transparencia).
    pub reasons: Vec<String>,
}

fn r3(x: f64) -> f64 {
    (x.clamp(0.0, 1.0) * 1000.0).round() / 1000.0
}

/// Heurísticas deterministas (ver docs). Resumen:
///
/// - **estructural** (stopword): relevante 0.03, fundamentada 0.99, ambigua
///   0.03 → p≈0.97 → `code`.
/// - **número/formato**: relevante 0.97, fundamentada 0.99, ambigua 0.02.
/// - **palabra de contenido**: relevancia base 0.93, +0.04 si está en
///   `<objective>`, +0.03 en `<role>`, +0.02 por repetición (máx +0.04),
///   +0.05 si es entidad conocida, +0.02 si tiene ≥5 letras y +0.01 más si
///   tiene ≥7 (palabras cortas y sueltas quedan en duda); fundamentada
///   0.97 si se repite / es entidad / está en role/objective, si no 0.955;
///   ambigua 0.04.
/// - **marcador de ambigüedad**: ambigua 0.38 (certeza 0.62) → `llm`.
/// - **afirmación sin fuente** (`best`, `every`, `never`…): fundamentada
///   0.40 si aparece una sola vez fuera de role/objective, si no 0.80.
/// - **nombre propio desconocido** (mayúscula a mitad de frase, no entidad,
///   aparece una vez): fundamentada 0.80.
/// - **aprobación**: `P(aprobación)` léxica (`approve` 0.97 … `ask` 0.544).
pub fn score_token(tok: &Token, ctx: &PromptContext, prev: Option<&Token>) -> Assessment {
    let w = tok.lower.as_str();
    let mut reasons = Vec::new();
    let freq = ctx.freq.get(w).copied().unwrap_or(1);
    let in_obj = ctx.objective.contains(w);
    let in_role = ctx.role.contains(w);
    let entity = in_list(KNOWN_ENTITIES, w);
    let mut s = Scores {
        relevant: 0.5,
        grounded: 0.5,
        ambiguous: 0.5,
        needs_approval: approval_prior(w),
    };
    if s.needs_approval > 0.0 {
        reasons.push(format!("verbo de aprobación ({:.3})", s.needs_approval));
    }
    if tok.kind == TokenKind::Number {
        s.relevant = 0.97;
        s.grounded = 0.99;
        s.ambiguous = 0.02;
        reasons.push("número/formato literal".into());
    } else if in_list(STOPWORDS, w) && !in_list(AMBIGUITY_MARKERS, w) {
        s.relevant = 0.03;
        s.grounded = 0.99;
        s.ambiguous = 0.03;
        reasons.push("token estructural (stopword)".into());
    } else {
        let mut r = 0.93;
        if in_obj {
            r += 0.04;
            reasons.push("en <objective>".into());
        }
        if in_role {
            r += 0.03;
            reasons.push("en <role>".into());
        }
        if freq > 1 {
            r += (0.02 * (freq - 1) as f64).min(0.04);
            reasons.push(format!("repetida ×{freq}"));
        }
        if entity {
            r += 0.05;
            reasons.push("entidad conocida".into());
        }
        let len = w.chars().count();
        if len >= 5 {
            r += 0.02;
        }
        if len >= 7 {
            r += 0.01;
        }
        s.relevant = r.min(0.99);
        s.grounded = if freq > 1 || entity || in_obj || in_role {
            0.97
        } else {
            0.955
        };
        s.ambiguous = 0.04;
        // Nombre propio desconocido a mitad de frase.
        let capital = tok.text.chars().next().is_some_and(char::is_uppercase);
        let sentence_start = tok.sentence_start
            || prev.is_none_or(|p| {
                p.line != tok.line || p.kind == TokenKind::Tag || p.kind == TokenKind::ListMarker
            });
        if capital && !sentence_start && !entity && freq == 1 && w != "i" {
            s.grounded = 0.80;
            reasons.push("nombre propio sin definir".into());
        }
    }
    if in_list(AMBIGUITY_MARKERS, w) {
        s.relevant = s.relevant.max(0.5);
        s.ambiguous = 0.38;
        s.grounded = s.grounded.min(0.9);
        reasons.push("marcador de ambigüedad".into());
    }
    if in_list(CLAIM_MARKERS, w) {
        s.grounded = if freq == 1 && !in_obj && !in_role {
            0.40
        } else {
            0.80
        };
        reasons.push("afirmación sin fuente".into());
    }
    s = Scores {
        relevant: r3(s.relevant),
        grounded: r3(s.grounded),
        ambiguous: r3(s.ambiguous),
        needs_approval: r3(s.needs_approval),
    };
    if s.needs_approval >= APPROVAL_CUTOFF {
        return Assessment {
            scores: s,
            question: Question::Approval,
            p: s.needs_approval,
            reasons,
        };
    }
    let mut question = Question::Relevant;
    let mut p = 1.0;
    for q in [
        Question::Relevant,
        Question::Grounded,
        Question::Ambiguous,
        Question::Approval,
    ] {
        let c = certainty(s.get(q));
        if c < p {
            p = c;
            question = q;
        }
    }
    Assessment {
        scores: s,
        question,
        p: r3(p),
        reasons,
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Ruta de una decisión.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// `p ≥ umbral`: ramas if, sin LLM.
    Code,
    /// Escalada al LLM activo.
    Llm,
    /// Solo aprobación humana.
    You,
}

/// Enrutado determinista (sin opinión): aprobación → `you`; `p ≥ umbral` →
/// `code`; resto → `llm`.
pub fn route(a: &Assessment, threshold: f64) -> Route {
    if a.scores.needs_approval >= APPROVAL_CUTOFF {
        Route::You
    } else if a.p >= threshold {
        Route::Code
    } else {
        Route::Llm
    }
}

/// Umbral saneado a `[0.5, 0.999]`.
pub fn clamp_threshold(t: Option<f64>) -> f64 {
    match t {
        Some(x) if x.is_finite() => x.clamp(0.5, 0.999),
        _ => DEFAULT_THRESHOLD,
    }
}

// ---------------------------------------------------------------------------
// Veredicto del LLM
// ---------------------------------------------------------------------------

/// Veredicto JSON del LLM para una palabra escalada.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmVerdict {
    pub index: usize,
    pub relevant: bool,
    pub grounded: bool,
    pub ambiguous: bool,
    pub needs_approval: bool,
    pub p: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

/// Resultado de aplicar un veredicto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerdictOutcome {
    /// Resuelto por el LLM.
    Resolved,
    /// El LLM pide aprobación humana o duda (p < piso): pendiente.
    Pending,
}

/// Qué hacer con el veredicto: el LLM solo resuelve si no pide aprobación,
/// no marca la palabra como ambigua y su confianza llega al piso.
pub fn verdict_outcome(v: &LlmVerdict, floor: f64) -> VerdictOutcome {
    if v.needs_approval || v.ambiguous || v.p < floor {
        VerdictOutcome::Pending
    } else {
        VerdictOutcome::Resolved
    }
}

/// Elemento de un lote escalado.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EscalationItem {
    pub index: usize,
    pub word: String,
    pub context: String,
    pub question: Question,
    pub p: f64,
}

/// Contexto ±`radius` palabras alrededor del token `idx`.
pub fn word_context(tokens: &[Token], idx: usize, radius: usize) -> String {
    let lo = idx.saturating_sub(radius);
    let hi = (idx + radius + 1).min(tokens.len());
    tokens[lo..hi]
        .iter()
        .filter(|t| t.kind != TokenKind::Tag)
        .map(|t| {
            if t.index == idx {
                format!("[{}]", t.text)
            } else {
                t.text.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Marca del system prompt (los mocks la usan para reconocer la petición).
pub const LLM_SYSTEM_MARK: &str = "Prompt Spider";

/// System prompt para el LLM.
pub fn llm_system_prompt() -> String {
    format!(
        "Eres el único modelo de razonamiento de {LLM_SYSTEM_MARK}. Un crawler determinista \
         recorre un prompt palabra por palabra; solo te llegan las palabras que el código no \
         pudo resolver con confianza. Para cada una responde: relevant (¿relevante para la \
         tarea?), grounded (¿su afirmación tiene fuente en el prompt?), ambiguous (¿es ambigua?), \
         needs_approval (¿requiere aprobación humana?) y p (tu confianza 0-1). Si dudas, baja p: \
         no inventes. Responde SOLO JSON: {{\"verdicts\":[{{\"i\":0,\"relevant\":true,\
         \"grounded\":true,\"ambiguous\":false,\"needs_approval\":false,\"p\":0.9,\"note\":\"…\"}}]}}"
    )
}

/// Mensaje de usuario con el prompt (recortado) y el lote.
pub fn llm_user_prompt(prompt: &str, items: &[EscalationItem]) -> String {
    let mut p: String = prompt.chars().take(3000).collect();
    if prompt.chars().count() > 3000 {
        p.push('…');
    }
    let mut s = format!(
        "PROMPT:\n{p}\n\nPALABRAS A EVALUAR (i | palabra | contexto | duda | p heurística):\n"
    );
    for it in items {
        s.push_str(&format!(
            "{} | {} | {} | {:?} | {:.3}\n",
            it.index, it.word, it.context, it.question, it.p
        ));
    }
    s.push_str("\nDevuelve un veredicto por cada i, en JSON.");
    s
}

fn as_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|x| x >= 0.5),
        Value::String(s) => match s.trim().to_lowercase().as_str() {
            "true" | "yes" | "sí" | "si" | "y" | "1" => Some(true),
            "false" | "no" | "n" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn as_prob(v: &Value) -> Option<f64> {
    let x = match v {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().trim_end_matches('%').trim().parse::<f64>().ok()?,
        _ => return None,
    };
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    let x = if x > 1.0 && x <= 100.0 { x / 100.0 } else { x };
    (x <= 1.0).then_some(x)
}

fn as_index(v: &Value) -> Option<usize> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| x as usize),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Extrae el primer valor JSON (objeto o array) de un texto con ruido
/// (`<think>`, bloques ```json, prosa alrededor).
pub fn extract_json(text: &str) -> Option<Value> {
    let mut t = text.to_string();
    while let Some(a) = t.find("<think>") {
        match t[a..].find("</think>") {
            Some(b) => t.replace_range(a..a + b + "</think>".len(), ""),
            None => t.truncate(a),
        }
    }
    let t = t.trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        if v.is_object() || v.is_array() {
            return Some(v);
        }
    }
    // Recorre cada `{` / `[` y prueba con el deserializador en streaming.
    for (i, c) in t.char_indices() {
        if c == '{' || c == '[' {
            let mut it = serde_json::Deserializer::from_str(&t[i..]).into_iter::<Value>();
            if let Some(Ok(v)) = it.next() {
                if v.is_object() || v.is_array() {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Parsea la respuesta del LLM. Acepta `{"verdicts":[…]}`, un array suelto,
/// `{"results"|"items"|"words":[…]}` o un objeto indexado `{"12":{…}}`.
/// Campos tolerantes (`"yes"`, `"0.9"`, `"90%"`, `confidence` como alias de
/// `p`, `index`/`id` como alias de `i`). Un elemento sin `p` válido o sin
/// índice esperado se descarta (nunca se inventa). `Err` si no hay JSON o
/// ningún veredicto utilizable.
pub fn parse_llm_verdicts(text: &str, expected: &[usize]) -> Result<Vec<LlmVerdict>, String> {
    let v =
        extract_json(text).ok_or_else(|| "la respuesta del LLM no contiene JSON".to_string())?;
    let mut items: Vec<(Option<usize>, Value)> = Vec::new();
    let arr = match &v {
        Value::Array(a) => Some(a.clone()),
        Value::Object(o) => ["verdicts", "results", "items", "words", "veredictos"]
            .iter()
            .find_map(|k| o.get(*k).and_then(Value::as_array).cloned()),
        _ => None,
    };
    match (arr, &v) {
        (Some(a), _) => items.extend(a.into_iter().map(|x| (None, x))),
        (None, Value::Object(o)) => {
            if o.contains_key("p") || o.contains_key("confidence") {
                items.push((None, v.clone()));
            } else {
                for (k, x) in o {
                    items.push((k.trim().parse().ok(), x.clone()));
                }
            }
        }
        _ => {}
    }
    let expected_set: HashSet<usize> = expected.iter().copied().collect();
    let mut out: Vec<LlmVerdict> = Vec::new();
    for (pos, (key_idx, it)) in items.iter().enumerate() {
        let Some(o) = it.as_object() else { continue };
        let idx = key_idx
            .or_else(|| {
                ["i", "index", "id", "idx"]
                    .iter()
                    .find_map(|k| o.get(*k).and_then(as_index))
            })
            .or_else(|| (items.len() == expected.len()).then(|| expected[pos]));
        let Some(idx) = idx.filter(|i| expected_set.contains(i)) else {
            continue;
        };
        if out.iter().any(|x| x.index == idx) {
            continue;
        }
        let Some(p) = ["p", "confidence", "confianza", "prob"]
            .iter()
            .find_map(|k| o.get(*k).and_then(as_prob))
        else {
            continue;
        };
        let b = |keys: &[&str], d: bool| {
            keys.iter()
                .find_map(|k| o.get(*k).and_then(as_bool))
                .unwrap_or(d)
        };
        out.push(LlmVerdict {
            index: idx,
            relevant: b(&["relevant", "relevante"], true),
            grounded: b(&["grounded", "fundamentada", "sourced"], false),
            ambiguous: b(&["ambiguous", "ambigua"], false),
            needs_approval: b(
                &["needs_approval", "approval", "requiere_aprobacion"],
                false,
            ),
            p: r3(p),
            note: o
                .get("note")
                .or_else(|| o.get("reason"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .chars()
                .take(160)
                .collect(),
        });
    }
    if out.is_empty() {
        return Err("el JSON del LLM no trae veredictos utilizables".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Corrida síncrona (sin LLM) — útil para tests y para el resumen
// ---------------------------------------------------------------------------

/// Decisión determinista de un token fork.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DryDecision {
    pub index: usize,
    pub word: String,
    pub assessment: Assessment,
    pub route: Route,
}

/// Evalúa y enruta todo el prompt sin llamar al LLM.
pub fn dry_run(prompt: &str, threshold: f64) -> (Vec<Token>, Vec<DryDecision>) {
    let tokens = tokenize(prompt);
    let ctx = PromptContext::new(&tokens);
    let mut out = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        if !t.kind.is_fork() {
            continue;
        }
        let a = score_token(t, &ctx, i.checked_sub(1).map(|j| &tokens[j]));
        let r = route(&a, threshold);
        out.push(DryDecision {
            index: t.index,
            word: t.text.clone(),
            assessment: a,
            route: r,
        });
    }
    (tokens, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_spans_sections_and_kinds() {
        let p = "<prompt>\n<role>\nYou are a chief-of-staff.\n</role>\n<objective>\n1. Ship 3 posts, at 1080x1920.\n- ask me\n</objective>\n</prompt>";
        let toks = tokenize(p);
        let chars: Vec<char> = p.chars().collect();
        for t in &toks {
            let s: String = chars[t.start..t.end].iter().collect();
            assert_eq!(s, t.text);
        }
        let words: Vec<&str> = toks
            .iter()
            .filter(|t| t.kind.is_fork())
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(
            words,
            [
                "You",
                "are",
                "a",
                "chief-of-staff",
                "Ship",
                "3",
                "posts",
                "at",
                "1080x1920",
                "ask",
                "me"
            ]
        );
        let tags = toks.iter().filter(|t| t.kind == TokenKind::Tag).count();
        assert_eq!(tags, 6);
        let markers: Vec<&str> = toks
            .iter()
            .filter(|t| t.kind == TokenKind::ListMarker)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(markers, ["1.", "-"]);
        let you = toks.iter().find(|t| t.text == "You").unwrap();
        assert_eq!(you.section, "role");
        assert_eq!(you.line, 2);
        let ship = toks.iter().find(|t| t.text == "Ship").unwrap();
        assert_eq!(ship.section, "objective");
        assert_eq!(
            toks.iter().find(|t| t.text == "1080x1920").unwrap().kind,
            TokenKind::Number
        );
        // Índices consecutivos.
        assert!(toks.iter().enumerate().all(|(i, t)| t.index == i));
    }

    #[test]
    fn tokenizer_handles_unicode_and_non_tags() {
        let toks = tokenize("Publica el vídeo a < 5 min, sin «hype» — a<b");
        let w: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            w,
            ["Publica", "el", "vídeo", "a", "5", "min", "sin", "hype", "a", "b"]
        );
        let v = toks.iter().find(|t| t.text == "vídeo").unwrap();
        assert_eq!(v.end - v.start, 5); // caracteres, no bytes
    }

    fn score_word(prompt: &str, word: &str) -> Assessment {
        let toks = tokenize(prompt);
        let ctx = PromptContext::new(&toks);
        let i = toks.iter().position(|t| t.text == word).unwrap();
        score_token(&toks[i], &ctx, i.checked_sub(1).map(|j| &toks[j]))
    }

    #[test]
    fn scorer_heuristics_are_deterministic_and_documented() {
        let a = score_word(SAMPLE_PROMPT, "the");
        assert!(a.p >= 0.95 && a.scores.relevant < 0.1, "{a:?}");
        let a = score_word(SAMPLE_PROMPT, "1080x1920");
        assert!(a.p >= 0.95, "{a:?}");
        // `ask` → aprobación 0.544 (justo encima del corte).
        let a = score_word(SAMPLE_PROMPT, "ask");
        assert_eq!(a.question, Question::Approval);
        assert!((a.p - 0.544).abs() < 1e-9);
        // `release` aparece en role y objective → camino código.
        let a = score_word(SAMPLE_PROMPT, "release");
        assert!(a.p >= 0.95, "{a:?}");
        // marcador de ambigüedad → duda en «¿específico?».
        let a = score_word(SAMPLE_PROMPT, "anything");
        assert_eq!(a.question, Question::Ambiguous);
        assert!(a.p < 0.95);
        // afirmación sin fuente.
        let a = score_word(SAMPLE_PROMPT, "best");
        assert_eq!(a.question, Question::Grounded);
        assert!(a.p < 0.7);
        // Determinismo.
        assert_eq!(
            score_word(SAMPLE_PROMPT, "ask"),
            score_word(SAMPLE_PROMPT, "ask")
        );
    }

    #[test]
    fn router_is_deterministic_with_threshold() {
        let mk = |p: f64, appr: f64| Assessment {
            scores: Scores {
                relevant: 0.9,
                grounded: 0.9,
                ambiguous: 0.1,
                needs_approval: appr,
            },
            question: Question::Relevant,
            p,
            reasons: vec![],
        };
        assert_eq!(route(&mk(0.97, 0.0), 0.95), Route::Code);
        assert_eq!(route(&mk(0.95, 0.0), 0.95), Route::Code);
        assert_eq!(route(&mk(0.949, 0.0), 0.95), Route::Llm);
        assert_eq!(route(&mk(0.99, 0.544), 0.95), Route::You);
        assert_eq!(route(&mk(0.949, 0.0), 0.90), Route::Code);
        assert_eq!(clamp_threshold(None), DEFAULT_THRESHOLD);
        assert_eq!(clamp_threshold(Some(f64::NAN)), DEFAULT_THRESHOLD);
        assert_eq!(clamp_threshold(Some(2.0)), 0.999);
        assert_eq!(clamp_threshold(Some(0.1)), 0.5);
    }

    #[test]
    fn sample_prompt_mix_is_plausible() {
        let (toks, d) = dry_run(SAMPLE_PROMPT, DEFAULT_THRESHOLD);
        let n = d.len();
        let code = d.iter().filter(|x| x.route == Route::Code).count();
        let llm = d.iter().filter(|x| x.route == Route::Llm).count();
        let you = d.iter().filter(|x| x.route == Route::You).count();
        assert_eq!(code + llm + you, n);
        assert!((150..=200).contains(&n), "forks {n}");
        assert!(toks.len() > n);
        assert!(code * 100 / n >= 60, "code {code}/{n}");
        assert!(llm >= 5 && you >= 5, "llm {llm} you {you}");
        // Umbral más bajo → más código, nunca menos.
        let (_, d2) = dry_run(SAMPLE_PROMPT, 0.90);
        let code2 = d2.iter().filter(|x| x.route == Route::Code).count();
        assert!(code2 >= code);
    }

    #[test]
    fn llm_json_parsing_is_robust() {
        let exp = [3, 7, 9];
        let t = "<think>hmm {no}</think>Claro:\n```json\n{\"verdicts\":[{\"i\":3,\"relevant\":true,\"grounded\":\"yes\",\"ambiguous\":false,\"needs_approval\":false,\"p\":0.91},{\"index\":\"7\",\"relevant\":\"no\",\"grounded\":1,\"ambiguous\":\"sí\",\"needs_approval\":false,\"confidence\":\"85%\"},{\"i\":99,\"p\":0.9},{\"i\":9,\"relevant\":true}]}\n```";
        let v = parse_llm_verdicts(t, &exp).unwrap();
        assert_eq!(v.len(), 2); // 99 no esperado; 9 sin p → descartado
        assert_eq!(v[0].index, 3);
        assert!(v[0].grounded && !v[0].ambiguous);
        assert_eq!(v[1].index, 7);
        assert!((v[1].p - 0.85).abs() < 1e-9);
        assert!(v[1].ambiguous && !v[1].relevant);
        // Array suelto, por posición.
        let v = parse_llm_verdicts("[{\"p\":0.8},{\"p\":0.99},{\"p\":0.7}]", &exp).unwrap();
        assert_eq!(v.iter().map(|x| x.index).collect::<Vec<_>>(), [3, 7, 9]);
        // Objeto indexado.
        let v = parse_llm_verdicts("{\"7\":{\"p\":0.96,\"needs_approval\":true}}", &exp).unwrap();
        assert!(v[0].needs_approval && v[0].index == 7);
        // Basura → Err (nunca inventa).
        assert!(parse_llm_verdicts("no sé", &exp).is_err());
        assert!(parse_llm_verdicts("{\"verdicts\":[]}", &exp).is_err());
        assert!(parse_llm_verdicts("{\"verdicts\":[{\"i\":3,\"p\":7.5e3}]}", &exp).is_err());
    }

    #[test]
    fn verdict_outcome_never_resolves_doubt_or_approval() {
        let v = LlmVerdict {
            index: 1,
            relevant: true,
            grounded: true,
            ambiguous: false,
            needs_approval: false,
            p: 0.9,
            note: String::new(),
        };
        assert_eq!(verdict_outcome(&v, 0.7), VerdictOutcome::Resolved);
        let low = LlmVerdict {
            p: 0.544,
            ..v.clone()
        };
        assert_eq!(verdict_outcome(&low, 0.7), VerdictOutcome::Pending);
        let appr = LlmVerdict {
            needs_approval: true,
            ..v.clone()
        };
        assert_eq!(verdict_outcome(&appr, 0.7), VerdictOutcome::Pending);
        let amb = LlmVerdict {
            ambiguous: true,
            ..v
        };
        assert_eq!(verdict_outcome(&amb, 0.7), VerdictOutcome::Pending);
    }

    #[test]
    fn llm_prompts_contain_batch_and_mark() {
        let toks = tokenize(SAMPLE_PROMPT);
        let i = toks.iter().position(|t| t.text == "vague").unwrap();
        let item = EscalationItem {
            index: i,
            word: "vague".into(),
            context: word_context(&toks, i, 3),
            question: Question::Ambiguous,
            p: 0.62,
        };
        assert!(item.context.contains("[vague]"));
        let u = llm_user_prompt(SAMPLE_PROMPT, &[item]);
        assert!(u.contains(&format!("{i} | vague |")));
        assert!(llm_system_prompt().contains(LLM_SYSTEM_MARK));
    }
}
