//! **Decoder del campo para Prompt Spider** (router del campo).
//!
//! Sustituye al LLM en las palabras **escaladas** del Prompt Spider cuando el
//! interruptor «Decoder del campo» está activo, igual que en el chat: el campo
//! produce el estado y la decisión; ningún LLM opina.
//!
//! Pipeline (arquitectura v4, receta §16):
//!
//! 1. **Codificación** ([`encode_decision`] → [`decision_context`]): rasgos
//!    del token (tipo, léxicos del scorer, mayúscula, longitud, frecuencia,
//!    sección, puntuación vecina, hashing), media de la ventana ±3 y rasgos
//!    globales de la decisión (4 puntuaciones heurísticas, `p`, pregunta,
//!    ruta inicial).
//! 2. **Vigilia**: cada ejemplo etiquetado entra en la CDT del Spider
//!    (experiencia) y en el reservorio de repetición.
//! 3. **Sueño = consolidación → teacher**: [`Readout`] logístico por cabeza
//!    (relevant, grounded, ambiguous, needs_approval, ok) sobre la
//!    experiencia; la **máscara de confianza** marca los pares
//!    ejemplo×cabeza donde el teacher acierta con margen (|P−y| < τ).
//! 4. **Dφ** (`v4_controls::Dphi::gated_h`, cabeza local-lineal con
//!    compuerta, H136): el estado de cada cabeza es un punto 2-D del campo
//!    `q₀ = (0.5, 1)`; Dφ lo transporta en un paso, condicionado por el
//!    contexto, hasta `(P, 1)`. Entrenamiento one-step (Adam, coseno) con
//!    50 % etiquetas reales y 50 % **replay de confianza** del teacher. El
//!    teacher y la CDT se descartan tras el sueño.
//! 5. **Calibración**: temperatura por cabeza en un split por prompt.
//! 6. **Guardia de novedad**: radio estandarizado del contexto contra el
//!    cuantil 99.5 % visto en el sueño (solo estadísticas, sin ejemplos).
//! 7. **Router**: una escalada se resuelve solo si `P(ok) ≥ piso`,
//!    `P(aprob) < 0.5`, `P(ambigua) < 0.5` y dentro de distribución. Si no,
//!    **pendiente** (nunca se inventa). La inferencia no consulta memoria.
//!
//! Ver `docs/prompt_spider_decoder_campo.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::prompt_spider::{
    approval_prior, route, score_token, tokenize, Assessment, LlmVerdict, PromptContext, Question,
    Route, Token, TokenKind, AMBIGUITY_MARKERS, CLAIM_MARKERS, KNOWN_ENTITIES, STOPWORDS,
};
use crate::v4_controls::{Dphi, TrainBudget};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

/// Cabezas: las 4 preguntas + `ok` (ruta: la palabra está bien → resolver).
pub const HEADS: usize = 5;
pub const HEAD_NAMES: [&str; HEADS] = ["relevant", "grounded", "ambiguous", "needs_approval", "ok"];
pub const H_REL: usize = 0;
pub const H_GRD: usize = 1;
pub const H_AMB: usize = 2;
pub const H_APR: usize = 3;
pub const H_OK: usize = 4;
/// Clases de engrama/RQM: cabeza × {no, sí}.
pub const CLASSES: usize = HEADS * 2;
/// Radio de la ventana de contexto (±3 tokens).
pub const WINDOW_RADIUS: usize = 3;
pub const WINDOW: usize = 2 * WINDOW_RADIUS + 1;
/// Rasgos locales por token.
pub const LOCAL_DIM: usize = 53;
/// Rasgos globales de la decisión.
pub const GLOBAL_DIM: usize = 12;
/// Rasgos del núcleo de chat (acoplamiento B/C).
/// Versión del formato del modelo persistido.
pub const MODEL_VERSION: u32 = 2;

/// Marcadores de fuente (para el rasgo «cue de fuente» en la ventana).
pub const SOURCE_CUES: &[&str] = &[
    "per",
    "according",
    "source",
    "sources",
    "report",
    "data",
    "survey",
    "study",
    "brief",
    "según",
    "fuente",
    "estudio",
    "informe",
    "datos",
];

const SECTIONS: [&str; 5] = ["objective", "role", "context", "steps", "constraints"];

// ---------------------------------------------------------------------------
// RNG determinista (splitmix64) — sin dependencias
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Sm(pub u64);

impl Sm {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// Ejemplos y etiquetas
// ---------------------------------------------------------------------------

/// Origen de las etiquetas de un ejemplo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelSource {
    /// Prompt sintético con verdad por construcción.
    Synthetic,
    /// Scorer determinista en palabras de alta confianza (ruta `code` / `you`).
    Scorer,
    /// Aprobar / Rechazar del usuario en corridas guardadas.
    User,
    /// Veredicto de un LLM (Gemma / docker-llm) como maestro.
    Teacher,
}

/// Etiquetas opcionales por cabeza (`None` = sin supervisión en esa cabeza).
pub type Labels = [Option<bool>; HEADS];

/// Un ejemplo codificado (no se persiste: se recalcula desde el prompt).
#[derive(Clone, Debug)]
pub struct SpiderExample {
    pub index: usize,
    pub word: String,
    /// `WINDOW × LOCAL_DIM`, aplanado.
    pub window: Vec<f32>,
    pub global: Vec<f32>,
    pub labels: Labels,
    pub source: LabelSource,
    pub first_route: Route,
    pub question: Question,
    /// `P(ok)` según la heurística (baseline): puntuación «buena» de la
    /// pregunta decisiva.
    pub heur_ok: f64,
    pub heur_scores: [f64; 4],
    /// Grupo (prompt) para splits sin fuga.
    pub group: u64,
    /// Cue del núcleo de chat (hash léxico mod 8) para B/C.
    pub core_cue: usize,
}

impl SpiderExample {
    pub fn center_local(&self) -> &[f32] {
        &self.window[WINDOW_RADIUS * LOCAL_DIM..(WINDOW_RADIUS + 1) * LOCAL_DIM]
    }
}

/// `P(sí)` heurística orientada a «la palabra está bien» para la pregunta.
pub fn heuristic_ok(a: &Assessment) -> f64 {
    let s = &a.scores;
    match a.question {
        Question::Relevant => s.relevant,
        Question::Grounded => s.grounded,
        Question::Ambiguous => 1.0 - s.ambiguous,
        Question::Approval => 1.0 - s.needs_approval,
    }
}

/// `ok` a partir de las 4 respuestas: estructural o relevante, fundamentada,
/// no ambigua y sin aprobación.
pub fn ok_from(
    relevant: bool,
    structural: bool,
    grounded: bool,
    ambiguous: bool,
    approval: bool,
) -> bool {
    (relevant || structural) && grounded && !ambiguous && !approval
}

/// Etiquetas del scorer para palabras de alta confianza: ruta `code` →
/// respuestas = puntuaciones ≥ 0.5 y `ok`; ruta `you` → aprobación y no-ok.
/// Escaladas → `None` (el scorer no sabe).
pub fn scorer_labels(a: &Assessment, r: Route) -> Option<Labels> {
    let s = &a.scores;
    match r {
        Route::Code => Some([
            Some(s.relevant >= 0.5),
            Some(s.grounded >= 0.5),
            Some(s.ambiguous >= 0.5),
            Some(false),
            Some(true),
        ]),
        Route::You => Some([None, None, None, Some(true), Some(false)]),
        Route::Llm => None,
    }
}

/// Etiquetas a partir de una decisión del usuario (Aprobar / Rechazar).
/// En palabras de aprobación la decisión no cambia que requiera aprobación:
/// solo se etiqueta `needs_approval = sí`, `ok = no`.
pub fn user_labels(question: Question, approve: bool) -> Labels {
    let mut l: Labels = [None; HEADS];
    match question {
        Question::Approval => {
            l[H_APR] = Some(true);
            l[H_OK] = Some(false);
        }
        Question::Relevant => {
            l[H_REL] = Some(approve);
            l[H_OK] = Some(approve);
        }
        Question::Grounded => {
            l[H_GRD] = Some(approve);
            l[H_OK] = Some(approve);
        }
        Question::Ambiguous => {
            l[H_AMB] = Some(!approve);
            l[H_OK] = Some(approve);
        }
    }
    l
}

/// Etiquetas a partir de un veredicto de LLM maestro (`p ≥ piso`, no
/// ambigua, sin aprobación → ok). Con Gemma local compacto solo `p` es
/// informativo: se etiqueta `ok` y la pregunta decisiva.
pub fn teacher_labels(question: Question, v: &LlmVerdict, floor: f64, compact: bool) -> Labels {
    let ok = !v.needs_approval && !v.ambiguous && v.p >= floor;
    let mut l: Labels = [None; HEADS];
    l[H_OK] = Some(ok);
    if compact {
        match question {
            Question::Relevant => l[H_REL] = Some(ok),
            Question::Grounded => l[H_GRD] = Some(ok),
            Question::Ambiguous => l[H_AMB] = Some(!ok),
            Question::Approval => l[H_APR] = Some(!ok),
        }
    } else {
        l[H_REL] = Some(v.relevant);
        l[H_GRD] = Some(v.grounded);
        l[H_AMB] = Some(v.ambiguous);
        l[H_APR] = Some(v.needs_approval);
    }
    l
}

// ---------------------------------------------------------------------------
// Codificación de una decisión
// ---------------------------------------------------------------------------

/// Texto entre el token anterior y `tokens[i]` (puntuación, saltos).
fn gap_before(chars: &[char], tokens: &[Token], i: usize) -> String {
    let start = if i == 0 { 0 } else { tokens[i - 1].end };
    let end = tokens[i].start.min(chars.len());
    chars[start.min(end)..end].iter().collect()
}

fn gap_after(chars: &[char], tokens: &[Token], i: usize) -> String {
    let start = tokens[i].end.min(chars.len());
    let end = if i + 1 < tokens.len() {
        tokens[i + 1].start.min(chars.len())
    } else {
        chars.len()
    };
    chars[start..end.max(start)].iter().collect()
}

fn hash_into(v: &mut [f32], off: usize, n: usize, key: &str) {
    let h = fnv(key);
    let slot = (h % n as u64) as usize;
    let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
    v[off + slot] += sign;
}

/// Rasgos locales de `tokens[i]` (`LOCAL_DIM`).
fn local_features(
    chars: &[char],
    tokens: &[Token],
    i: usize,
    ctx: &PromptContext,
    scores: Option<&Assessment>,
) -> [f32; LOCAL_DIM] {
    let mut v = [0f32; LOCAL_DIM];
    let t = &tokens[i];
    let w = t.lower.as_str();
    match t.kind {
        TokenKind::Word => v[1] = 1.0,
        TokenKind::Number => v[2] = 1.0,
        TokenKind::Tag => v[3] = 1.0,
        TokenKind::ListMarker => v[4] = 1.0,
    }
    let b = |x: bool| if x { 1.0 } else { 0.0 };
    if t.kind.is_fork() {
        v[5] = b(STOPWORDS.contains(&w));
        v[6] = b(AMBIGUITY_MARKERS.contains(&w));
        v[7] = b(CLAIM_MARKERS.contains(&w));
        v[8] = approval_prior(w) as f32;
        v[9] = b(KNOWN_ENTITIES.contains(&w));
        v[10] = b(SOURCE_CUES.contains(&w));
        v[11] = b(t.text.chars().next().is_some_and(char::is_uppercase));
        v[12] = b(t.sentence_start);
        let len = w.chars().count();
        v[13] = (len as f32 / 12.0).min(1.5);
        v[14] = b(len < 5);
        let f = ctx.freq.get(w).copied().unwrap_or(1);
        v[15] = ((f as f32).ln() / 2.0).min(1.5);
        v[16] = b(ctx.objective.contains(w));
        v[17] = b(ctx.role.contains(w));
    }
    let gb = gap_before(chars, tokens, i);
    let ga = gap_after(chars, tokens, i);
    v[18] = b(gb.contains('('));
    v[19] = b(gb.contains(','));
    v[20] = b(gb.contains(':'));
    v[21] = b(gb.contains(['.', '!', '?', ';']));
    v[22] = b(gb.contains('\n'));
    v[23] = b(ga.contains('('));
    v[24] = b(ga.contains(')'));
    v[25] = b(ga.contains(','));
    v[26] = b(ga.contains(':'));
    v[27] = b(ga.contains(['.', '!', '?', ';']));
    v[28] = b(ga.contains('\n'));
    if let Some(a) = scores {
        v[29] = a.scores.relevant as f32;
        v[30] = a.scores.grounded as f32;
        v[31] = a.scores.ambiguous as f32;
        v[32] = a.scores.needs_approval as f32;
    }
    // Sección (5 + otra).
    match SECTIONS.iter().position(|s| *s == t.section.as_str()) {
        Some(k) => v[33 + k] = 1.0,
        None => v[38] = 1.0,
    }
    // Identidad (8) y sufijo de 2 letras (6) por hashing con signo.
    if t.kind.is_fork() {
        hash_into(&mut v, 39, 8, w);
        let cs: Vec<char> = w.chars().collect();
        let suf: String = cs[cs.len().saturating_sub(2)..].iter().collect();
        hash_into(&mut v, 47, 6, &format!("suf:{suf}"));
    }
    v
}

/// Rasgos globales de la decisión (`GLOBAL_DIM`).
fn global_features(a: &Assessment, r: Route) -> [f32; GLOBAL_DIM] {
    let mut g = [0f32; GLOBAL_DIM];
    g[0] = a.scores.relevant as f32;
    g[1] = a.scores.grounded as f32;
    g[2] = a.scores.ambiguous as f32;
    g[3] = a.scores.needs_approval as f32;
    g[4] = a.p as f32;
    let q = match a.question {
        Question::Relevant => 0,
        Question::Grounded => 1,
        Question::Ambiguous => 2,
        Question::Approval => 3,
    };
    g[5 + q] = 1.0;
    let rr = match r {
        Route::Code => 0,
        Route::Llm => 1,
        Route::You => 2,
    };
    g[9 + rr] = 1.0;
    g
}

/// Cue léxico del núcleo de chat (8 conceptos), determinista.
pub fn core_cue(word_lower: &str) -> usize {
    (fnv(word_lower) % 8) as usize
}

/// Prompt ya analizado (tokens, contexto, evaluaciones y rutas por fork).
pub struct AnalyzedPrompt {
    pub text: String,
    pub chars: Vec<char>,
    pub tokens: Vec<Token>,
    pub ctx: PromptContext,
    /// Por posición de token: evaluación + ruta (solo forks).
    pub assess: Vec<Option<(Assessment, Route)>>,
}

impl AnalyzedPrompt {
    pub fn new(text: &str, threshold: f64) -> Self {
        Self::from_tokens(text, tokenize(text), threshold)
    }

    /// Igual que [`Self::new`] reutilizando tokens ya calculados.
    pub fn from_tokens(text: &str, tokens: Vec<Token>, threshold: f64) -> Self {
        let ctx = PromptContext::new(&tokens);
        let assess = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| {
                t.kind.is_fork().then(|| {
                    let a = score_token(t, &ctx, i.checked_sub(1).map(|j| &tokens[j]));
                    let r = route(&a, threshold);
                    (a, r)
                })
            })
            .collect();
        Self {
            text: text.to_string(),
            chars: text.chars().collect(),
            tokens,
            ctx,
            assess,
        }
    }

    /// Posición en `tokens` del token con `index`.
    pub fn pos_of(&self, index: usize) -> Option<usize> {
        self.tokens.iter().position(|t| t.index == index)
    }

    /// Codifica el fork en la posición `i`.
    pub fn encode(
        &self,
        i: usize,
        labels: Labels,
        source: LabelSource,
        group: u64,
    ) -> Option<SpiderExample> {
        let (a, r) = self.assess.get(i)?.as_ref()?;
        Some(encode_decision(self, i, a, *r, labels, source, group))
    }
}

/// Codifica la decisión de `tokens[i]` (ventana ±3 + globales).
pub fn encode_decision(
    p: &AnalyzedPrompt,
    i: usize,
    a: &Assessment,
    r: Route,
    labels: Labels,
    source: LabelSource,
    group: u64,
) -> SpiderExample {
    let mut window = vec![0f32; WINDOW * LOCAL_DIM];
    for k in 0..WINDOW {
        let j = i as isize + k as isize - WINDOW_RADIUS as isize;
        let dst = &mut window[k * LOCAL_DIM..(k + 1) * LOCAL_DIM];
        if j < 0 || j as usize >= p.tokens.len() {
            dst[0] = 1.0; // relleno
            continue;
        }
        let j = j as usize;
        let sc = p.assess[j].as_ref().map(|(a, _)| a);
        dst.copy_from_slice(&local_features(&p.chars, &p.tokens, j, &p.ctx, sc));
    }
    let t = &p.tokens[i];
    SpiderExample {
        index: t.index,
        word: t.text.clone(),
        window,
        global: global_features(a, r).to_vec(),
        labels,
        source,
        first_route: r,
        question: a.question,
        heur_ok: heuristic_ok(a),
        heur_scores: [
            a.scores.relevant,
            a.scores.grounded,
            a.scores.ambiguous,
            a.scores.needs_approval,
        ],
        group,
        core_cue: core_cue(&t.lower),
    }
}

// ---------------------------------------------------------------------------
// Contexto de la decisión (entrada de Dφ)
// ---------------------------------------------------------------------------

/// Dimensión del contexto: centro ⊕ globales ⊕ media de la ventana vecina.
pub const CTX_DIM: usize = LOCAL_DIM + GLOBAL_DIM + LOCAL_DIM;

/// Rasgos de la decisión: token central, rasgos globales y media de los
/// vecinos de la ventana ±3 (contexto local sin estado recurrente).
pub fn decision_context(ex: &SpiderExample) -> Vec<f32> {
    let mut v = ex.center_local().to_vec();
    v.extend_from_slice(&ex.global);
    let mut nb = [0f32; LOCAL_DIM];
    for k in (0..WINDOW).filter(|k| *k != WINDOW_RADIUS) {
        for d in 0..LOCAL_DIM {
            nb[d] += ex.window[k * LOCAL_DIM + d] / (WINDOW - 1) as f32;
        }
    }
    v.extend_from_slice(&nb);
    v
}

// ---------------------------------------------------------------------------
// Lectura (cabezas logísticas + temperatura) = teacher consolidado
// ---------------------------------------------------------------------------

/// Cabezas logísticas con estandarización y temperatura por cabeza.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Readout {
    pub dim: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// `HEADS × dim`.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub temp: Vec<f32>,
    /// Ejemplos con etiqueta por cabeza en el último ajuste.
    pub n_labeled: Vec<usize>,
}

impl Readout {
    fn logit(&self, h: usize, z: &[f32]) -> f32 {
        let w = &self.w[h * self.dim..(h + 1) * self.dim];
        let mut s = self.b[h];
        for d in 0..self.dim.min(z.len()) {
            s += w[d] * (z[d] - self.mean[d]) / self.std[d];
        }
        s
    }

    /// `P(sí)` calibrada por cabeza (0.5 en cabezas sin datos).
    pub fn probs(&self, z: &[f32]) -> [f64; HEADS] {
        let mut out = [0.5; HEADS];
        if self.w.is_empty() {
            return out;
        }
        for (h, o) in out.iter_mut().enumerate() {
            if self.n_labeled.get(h).copied().unwrap_or(0) == 0 {
                continue;
            }
            *o = sigmoid((self.logit(h, z) / self.temp[h].max(0.05)) as f64);
        }
        out
    }

    /// Ajuste por Adam (full batch por cabeza con máscara de etiquetas) +
    /// temperatura en el split de calibración (`cal[i] = true`).
    pub fn fit(zs: &[Vec<f32>], labels: &[Labels], cal: &[bool], epochs: usize, l2: f32) -> Self {
        let n = zs.len();
        let dim = zs.first().map(|z| z.len()).unwrap_or(0);
        let mut mean = vec![0f32; dim];
        let mut std = vec![0f32; dim];
        for z in zs {
            for d in 0..dim {
                mean[d] += z[d];
            }
        }
        for m in mean.iter_mut() {
            *m /= n.max(1) as f32;
        }
        for z in zs {
            for d in 0..dim {
                std[d] += (z[d] - mean[d]).powi(2);
            }
        }
        for s in std.iter_mut() {
            *s = (*s / n.max(1) as f32).sqrt().max(1e-3);
        }
        let xs: Vec<Vec<f32>> = zs
            .iter()
            .map(|z| (0..dim).map(|d| (z[d] - mean[d]) / std[d]).collect())
            .collect();
        let mut w = vec![0f32; HEADS * dim];
        let mut b = vec![0f32; HEADS];
        let mut temp = vec![1f32; HEADS];
        let mut n_labeled = vec![0usize; HEADS];
        for h in 0..HEADS {
            let idx: Vec<usize> = (0..n)
                .filter(|&i| !cal[i] && labels[i][h].is_some())
                .collect();
            n_labeled[h] = idx.len();
            if idx.is_empty() {
                continue;
            }
            let pos = idx.iter().filter(|&&i| labels[i][h] == Some(true)).count();
            let prior = (pos as f32 + 1.0) / (idx.len() as f32 + 2.0);
            let mut bh = (prior / (1.0 - prior)).ln();
            let (mut mw, mut vw) = (vec![0f32; dim], vec![0f32; dim]);
            let (mut mb, mut vb) = (0f32, 0f32);
            let (lr, b1, b2) = (0.05f32, 0.9f32, 0.999f32);
            let wh = &mut w[h * dim..(h + 1) * dim];
            let m = idx.len() as f32;
            for ep in 1..=epochs {
                let mut gw = vec![0f32; dim];
                let mut gb = 0f32;
                for &i in &idx {
                    let x = &xs[i];
                    let mut s = bh;
                    for d in 0..dim {
                        s += wh[d] * x[d];
                    }
                    let p = sigmoid(s as f64) as f32;
                    let y = if labels[i][h] == Some(true) { 1.0 } else { 0.0 };
                    let g = p - y;
                    for d in 0..dim {
                        gw[d] += g * x[d];
                    }
                    gb += g;
                }
                let t = ep as i32;
                let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
                for d in 0..dim {
                    let g = gw[d] / m + l2 * wh[d];
                    mw[d] = b1 * mw[d] + (1.0 - b1) * g;
                    vw[d] = b2 * vw[d] + (1.0 - b2) * g * g;
                    wh[d] -= lr * (mw[d] / c1) / ((vw[d] / c2).sqrt() + 1e-8);
                }
                let g = gb / m;
                mb = b1 * mb + (1.0 - b1) * g;
                vb = b2 * vb + (1.0 - b2) * g * g;
                bh -= lr * (mb / c1) / ((vb / c2).sqrt() + 1e-8);
            }
            b[h] = bh;
            // Temperatura en calibración (rejilla 0.25–6, NLL).
            let cidx: Vec<usize> = (0..n)
                .filter(|&i| cal[i] && labels[i][h].is_some())
                .collect();
            if cidx.len() >= 10 {
                let logits: Vec<(f32, bool)> = cidx
                    .iter()
                    .map(|&i| {
                        let mut s = bh;
                        for d in 0..dim {
                            s += wh[d] * xs[i][d];
                        }
                        (s, labels[i][h] == Some(true))
                    })
                    .collect();
                let mut best = (f64::INFINITY, 1f32);
                let mut t = 0.25f32;
                while t <= 6.0 {
                    let nll: f64 = logits
                        .iter()
                        .map(|(s, y)| {
                            let p = sigmoid((*s / t) as f64).clamp(1e-6, 1.0 - 1e-6);
                            if *y {
                                -p.ln()
                            } else {
                                -(1.0 - p).ln()
                            }
                        })
                        .sum();
                    if nll < best.0 {
                        best = (nll, t);
                    }
                    t += 0.05;
                }
                temp[h] = best.1;
            }
        }
        Self {
            dim,
            mean,
            std,
            w,
            b,
            temp,
            n_labeled,
        }
    }
}

// ---------------------------------------------------------------------------
// Router del campo v4
// ---------------------------------------------------------------------------

/// Configuración del router (receta §16 + controles del experimento).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldConfig {
    pub seed: u64,
    /// `false` = control sin campo: decide el teacher logístico directamente.
    pub dphi: bool,
    /// Máscara de confianza en el replay (§16). `false` = replay completo.
    pub trusted: bool,
    /// Guardia de novedad/OOD.
    pub ood: bool,
    /// Ancho oculto de Dφ (§16: 136).
    pub hidden: usize,
    /// Pasos de Adam por sueño (batch 16).
    pub steps: usize,
    /// Umbral de confianza: |P_teacher − etiqueta| < τ.
    pub tau: f64,
    /// Épocas del teacher logístico.
    pub epochs: usize,
    pub l2: f32,
    /// Ejemplos máximos en el buffer de repetición (reservorio).
    pub replay_cap: usize,
}

impl FieldConfig {
    /// Router v4 por defecto: Dφ gated H136 + replay de confianza + OOD.
    pub fn v4(seed: u64) -> Self {
        Self {
            seed,
            dphi: true,
            trusted: true,
            ood: true,
            hidden: crate::v4_distill::E_HIDDEN,
            steps: 3000,
            tau: 0.25,
            epochs: 120,
            l2: 1e-3,
            replay_cap: 24_000,
        }
    }
}

/// Predicción del campo para una palabra.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldPrediction {
    /// `P(sí)` calibrada por cabeza.
    pub probs: [f64; HEADS],
    pub novelty: f64,
}

impl FieldPrediction {
    pub fn p_ok(&self) -> f64 {
        self.probs[H_OK]
    }
}

/// Decisión del router del campo para una escalada.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldDecision {
    pub resolve: bool,
    pub verdict: LlmVerdict,
    pub prediction: FieldPrediction,
    /// Motivo legible (resuelta / pendiente por …).
    pub reason: String,
    pub micros: f64,
}

/// Informe de un sueño del router.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FieldSleepReport {
    /// Experiencias de vigilia consolidadas (CDT del Spider, luego borrada).
    pub episodes: usize,
    /// Pares ejemplo×cabeza del replay y cuántos son de confianza.
    pub replay_pairs: usize,
    pub trusted_pairs: usize,
    /// Pérdida final de Dφ (MSE de un paso).
    pub dphi_loss: f64,
    pub replay: usize,
    pub ms: f64,
}

/// Modelo persistido: Dφ (checkpoint bit-exacto) + calibración + guardia.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldModelFile {
    pub version: u32,
    pub cfg: FieldConfig,
    pub dphi: Option<String>,
    /// Solo en el control sin campo (`cfg.dphi = false`).
    pub teacher: Option<Readout>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub temp: Vec<f32>,
    pub n_labeled: Vec<usize>,
    pub novelty_max: f64,
    pub examples_seen: u64,
    pub sleeps: u64,
    #[serde(default)]
    pub holdout: Option<serde_json::Value>,
}

/// Router del campo v4 para Prompt Spider.
///
/// Vigilia: cada ejemplo etiquetado entra en la CDT del Spider (experiencias
/// pendientes) y en el reservorio de repetición. Sueño: consolida la
/// repetición en un teacher logístico por cabeza → máscara de confianza →
/// destila Dφ (cabeza local-lineal con compuerta) con replay de confianza →
/// calibra por temperatura → borra la CDT. Inferencia: solo Dφ + temperatura
/// + guardia de novedad (0 consultas a memoria).
#[derive(Clone, Debug)]
pub struct SpiderFieldRouter {
    pub cfg: FieldConfig,
    pub dphi: Option<Dphi>,
    teacher: Option<Readout>,
    mean: Vec<f32>,
    std: Vec<f32>,
    temp: Vec<f32>,
    n_labeled: Vec<usize>,
    wake: usize,
    replay: Vec<SpiderExample>,
    replay_seen: u64,
    rng: Sm,
    pub novelty_max: f64,
    pub examples_seen: u64,
    pub sleeps: u64,
    /// Métricas hold-out del último entrenamiento (para la UI).
    pub holdout: Option<serde_json::Value>,
}

/// Estado inicial de cada cabeza (punto 2-D del campo): (P previa, 1).
const Q0: [f64; 2] = [0.5, 1.0];

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-3, 1.0 - 1e-3);
    (p / (1.0 - p)).ln()
}

impl SpiderFieldRouter {
    pub fn new(cfg: FieldConfig) -> Self {
        Self {
            dphi: None,
            teacher: None,
            mean: Vec::new(),
            std: Vec::new(),
            temp: vec![1.0; HEADS],
            n_labeled: vec![0; HEADS],
            wake: 0,
            replay: Vec::new(),
            replay_seen: 0,
            rng: Sm(cfg.seed ^ 0xBEEF),
            novelty_max: f64::INFINITY,
            examples_seen: 0,
            sleeps: 0,
            holdout: None,
            cfg,
        }
    }

    pub fn is_trained(&self) -> bool {
        self.n_labeled.get(H_OK).copied().unwrap_or(0) > 0
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Episodios en vigilia pendientes de sueño.
    pub fn wake_len(&self) -> usize {
        self.wake
    }

    fn standardize(&self, ctx: &[f32]) -> Vec<f64> {
        ctx.iter()
            .enumerate()
            .map(|(d, x)| ((x - self.mean[d]) / self.std[d]).clamp(-6.0, 6.0) as f64)
            .collect()
    }

    fn head_ctx(x: &[f64], h: usize) -> Vec<f64> {
        let mut c = x.to_vec();
        c.extend((0..HEADS).map(|k| if k == h { 1.0 } else { 0.0 }));
        c
    }

    fn novelty_of(x: &[f64]) -> f64 {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    /// Logit sin calibrar por cabeza (Dφ, o el teacher en el control).
    fn raw_logits(&self, ex: &SpiderExample) -> ([f64; HEADS], f64) {
        let ctx = decision_context(ex);
        let mut out = [0.0; HEADS];
        if self.mean.is_empty() {
            return (out, 0.0);
        }
        let x = self.standardize(&ctx);
        let nov = Self::novelty_of(&x);
        if let Some(d) = &self.dphi {
            for (h, o) in out.iter_mut().enumerate() {
                *o = logit(d.step(&Q0, &Self::head_ctx(&x, h))[0]);
            }
        } else if let Some(t) = &self.teacher {
            let p = t.probs(&ctx);
            for h in 0..HEADS {
                out[h] = logit(p[h]);
            }
        }
        (out, nov)
    }

    /// Vigilia: observa un ejemplo etiquetado (CDT + repetición).
    pub fn observe(&mut self, ex: SpiderExample) {
        self.wake += 1;
        self.examples_seen += 1;
        self.replay_seen += 1;
        if self.replay.len() < self.cfg.replay_cap {
            self.replay.push(ex);
        } else {
            let j = (self.rng.next_u64() % self.replay_seen) as usize;
            if j < self.replay.len() {
                self.replay[j] = ex;
            }
        }
    }

    /// Sueño: consolidación → teacher → Dφ (§16) → calibración; borra la CDT.
    pub fn sleep(&mut self) -> FieldSleepReport {
        let t0 = Instant::now();
        let mut rep = FieldSleepReport {
            episodes: self.wake,
            ..Default::default()
        };
        let (pairs, trusted, loss) = self.refit();
        rep.replay_pairs = pairs;
        rep.trusted_pairs = trusted;
        rep.dphi_loss = loss;
        self.wake = 0;
        self.sleeps += 1;
        rep.replay = self.replay.len();
        rep.ms = t0.elapsed().as_secs_f64() * 1e3;
        rep
    }

    /// Consolida la repetición y re-destila Dφ. Devuelve (pares, de
    /// confianza, pérdida final).
    pub fn refit(&mut self) -> (usize, usize, f64) {
        if self.replay.is_empty() {
            return (0, 0, 0.0);
        }
        let ctxs: Vec<Vec<f32>> = self.replay.iter().map(decision_context).collect();
        let labels: Vec<Labels> = self.replay.iter().map(|e| e.labels).collect();
        let mut cal: Vec<bool> = self
            .replay
            .iter()
            .map(|e| fnv(&format!("cal{}", e.group)).is_multiple_of(5))
            .collect();
        if cal.iter().all(|c| *c) || cal.iter().all(|c| !*c) {
            cal.iter_mut().for_each(|c| *c = false);
        }
        // 1. Consolidación: teacher logístico por cabeza sobre la experiencia.
        let teacher = Readout::fit(&ctxs, &labels, &cal, self.cfg.epochs, self.cfg.l2);
        self.mean = teacher.mean.clone();
        self.std = teacher.std.clone();
        self.n_labeled = teacher.n_labeled.clone();
        let xs: Vec<Vec<f64>> = ctxs.iter().map(|c| self.standardize(c)).collect();
        // 2. Guardia de novedad: cuantil 99.5 % del radio estandarizado.
        let mut novs: Vec<f64> = xs.iter().map(|x| Self::novelty_of(x)).collect();
        novs.sort_by(|a, b| a.total_cmp(b));
        let k = ((novs.len() as f64) * 0.995) as usize;
        self.novelty_max = novs[k.min(novs.len() - 1)] * 1.05 + 0.02;
        if !self.cfg.dphi {
            self.temp = teacher.temp.clone();
            self.teacher = Some(teacher);
            return (0, 0, 0.0);
        }
        // 3. Máscara de confianza + pares (ejemplo, cabeza) de entrenamiento.
        let mut real: Vec<(usize, usize, f64)> = Vec::new();
        let mut replay: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..xs.len() {
            if cal[i] {
                continue;
            }
            let pt = teacher.probs(&ctxs[i]);
            for h in 0..HEADS {
                if let Some(y) = labels[i][h] {
                    let yf = y as u8 as f64;
                    real.push((i, h, yf));
                    if !self.cfg.trusted || (pt[h] - yf).abs() < self.cfg.tau {
                        replay.push((i, h, pt[h]));
                    }
                }
            }
        }
        drop(teacher);
        if real.is_empty() {
            return (0, 0, 0.0);
        }
        // 4. Dφ gated local-lineal + replay de confianza (§16).
        let mut d = self.dphi.take().unwrap_or_else(|| {
            Dphi::gated_h(2, CTX_DIM + HEADS, self.cfg.hidden, true, self.cfg.seed)
        });
        let budget = TrainBudget {
            steps: self.cfg.steps,
            batch: 16,
            lr: 3e-3,
        };
        let any = !replay.is_empty();
        let loss = d.train_onestep(
            budget,
            self.cfg.seed ^ self.sleeps ^ 0x7A,
            crate::v4_distill::ONESTEP_LR_FINAL,
            false,
            |r| {
                let (i, h, t) = if any && r.gen::<f64>() < crate::field_autonomy_v4::REPLAY_FRACTION
                {
                    replay[r.gen_range(0..replay.len())]
                } else {
                    real[r.gen_range(0..real.len())]
                };
                (Q0.to_vec(), Self::head_ctx(&xs[i], h), vec![t, 1.0])
            },
        );
        // 5. Calibración por temperatura en el split de calibración.
        let mut temp = vec![1f32; HEADS];
        for h in 0..HEADS {
            let rows: Vec<(f64, bool)> = (0..xs.len())
                .filter(|&i| cal[i])
                .filter_map(|i| {
                    labels[i][h].map(|y| (logit(d.step(&Q0, &Self::head_ctx(&xs[i], h))[0]), y))
                })
                .collect();
            if rows.len() < 10 {
                continue;
            }
            let mut best = (f64::INFINITY, 1f32);
            let mut t = 0.25f32;
            while t <= 6.0 {
                let nll: f64 = rows
                    .iter()
                    .map(|(s, y)| {
                        let p = sigmoid(s / t as f64).clamp(1e-6, 1.0 - 1e-6);
                        if *y {
                            -p.ln()
                        } else {
                            -(1.0 - p).ln()
                        }
                    })
                    .sum();
                if nll < best.0 {
                    best = (nll, t);
                }
                t += 0.05;
            }
            temp[h] = best.1;
        }
        self.temp = temp;
        self.dphi = Some(d);
        (real.len(), replay.len(), loss)
    }

    pub fn predict(&self, ex: &SpiderExample) -> FieldPrediction {
        let (lg, nov) = self.raw_logits(ex);
        let mut probs = [0.5; HEADS];
        for h in 0..HEADS {
            if self.n_labeled.get(h).copied().unwrap_or(0) > 0 {
                probs[h] = sigmoid(lg[h] / self.temp[h].max(0.05) as f64);
            }
        }
        FieldPrediction {
            probs,
            novelty: nov,
        }
    }

    /// Decide una escalada: resuelve solo si `P(ok) ≥ piso`, sin aprobación,
    /// no ambigua y dentro de distribución. Si no, pendiente (no inventa).
    pub fn decide(&self, ex: &SpiderExample, floor: f64) -> FieldDecision {
        let t0 = Instant::now();
        let pred = self.predict(ex);
        let micros = t0.elapsed().as_secs_f64() * 1e6;
        let r3 = |x: f64| (x.clamp(0.0, 1.0) * 1000.0).round() / 1000.0;
        let pr = pred.probs;
        let p_ok = r3(pred.p_ok());
        let ood = self.cfg.ood && pred.novelty > self.novelty_max;
        let trained = self.is_trained();
        let needs_approval = pr[H_APR] >= 0.5;
        let ambiguous = pr[H_AMB] >= 0.5;
        let resolve = trained && !ood && !needs_approval && !ambiguous && p_ok >= floor;
        let reason = if !trained {
            "decoder del campo sin entrenar".to_string()
        } else if ood {
            format!(
                "fuera de distribución (novedad {:.3} > {:.3})",
                pred.novelty, self.novelty_max
            )
        } else if needs_approval {
            format!("el campo pide aprobación (P={:.3})", pr[H_APR])
        } else if ambiguous {
            format!("el campo la marca ambigua (P={:.3})", pr[H_AMB])
        } else if p_ok < floor {
            format!("el campo duda (P(ok)={p_ok:.3} < {floor:.2})")
        } else {
            format!("P(ok)={p_ok:.3}")
        };
        let verdict = LlmVerdict {
            index: ex.index,
            relevant: pr[H_REL] >= 0.5,
            grounded: pr[H_GRD] >= 0.5,
            ambiguous: ambiguous || ood || !trained,
            needs_approval,
            p: p_ok,
            note: format!(
                "campo v4 · rel {:.2} · fund {:.2} · amb {:.2} · aprob {:.2} · ok {:.3} · novedad {:.3}",
                pr[H_REL], pr[H_GRD], pr[H_AMB], pr[H_APR], p_ok, pred.novelty
            ),
        };
        FieldDecision {
            resolve,
            verdict,
            prediction: pred,
            reason,
            micros,
        }
    }

    pub fn to_file(&self) -> FieldModelFile {
        FieldModelFile {
            version: MODEL_VERSION,
            cfg: self.cfg,
            dphi: self.dphi.as_ref().map(|d| d.to_ckpt()),
            teacher: self.teacher.clone(),
            mean: self.mean.clone(),
            std: self.std.clone(),
            temp: self.temp.clone(),
            n_labeled: self.n_labeled.clone(),
            novelty_max: self.novelty_max,
            examples_seen: self.examples_seen,
            sleeps: self.sleeps,
            holdout: self.holdout.clone(),
        }
    }

    /// Reconstruye el router desde un modelo guardado (sin la repetición:
    /// el siguiente entrenamiento la rellena).
    pub fn from_file(f: FieldModelFile) -> Result<Self, String> {
        if f.version != MODEL_VERSION {
            return Err(format!("versión de modelo {} no soportada", f.version));
        }
        let mut r = Self::new(f.cfg);
        r.dphi = match f.dphi {
            Some(s) => Some(Dphi::from_ckpt(&s).ok_or("checkpoint Dφ inválido")?),
            None => None,
        };
        r.teacher = f.teacher;
        r.mean = f.mean;
        r.std = f.std;
        r.temp = f.temp;
        r.n_labeled = f.n_labeled;
        r.novelty_max = f.novelty_max;
        r.examples_seen = f.examples_seen;
        r.sleeps = f.sleeps;
        r.holdout = f.holdout;
        Ok(r)
    }
}

// ---------------------------------------------------------------------------
// Prompts sintéticos con verdad por construcción
// ---------------------------------------------------------------------------

/// Verdad por construcción de una palabra sintética.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Truth {
    pub relevant: bool,
    pub grounded: bool,
    pub ambiguous: bool,
    pub approval: bool,
    pub structural: bool,
}

impl Truth {
    pub fn content() -> Self {
        Self {
            relevant: true,
            grounded: true,
            ambiguous: false,
            approval: false,
            structural: false,
        }
    }
    pub fn structural() -> Self {
        Self {
            relevant: false,
            structural: true,
            ..Self::content()
        }
    }
    pub fn ok(&self) -> bool {
        ok_from(
            self.relevant,
            self.structural,
            self.grounded,
            self.ambiguous,
            self.approval,
        )
    }
    pub fn labels(&self) -> Labels {
        [
            Some(self.relevant),
            Some(self.grounded),
            Some(self.ambiguous),
            Some(self.approval),
            Some(self.ok()),
        ]
    }
}

/// Vocabulario del generador (dos particiones: entrenamiento y desplazado).
pub struct Vocab {
    pub long: &'static [&'static str],
    pub short: &'static [&'static str],
    pub filler: &'static [&'static str],
    pub names: &'static [&'static str],
    pub roles: &'static [&'static str],
}

/// Vocabulario de entrenamiento / hold-out en distribución.
pub const VOCAB_TRAIN: Vocab = Vocab {
    long: &[
        "summarize",
        "transcript",
        "deliverables",
        "highlights",
        "storyboard",
        "captions",
        "thumbnail",
        "schedule",
        "research",
        "outline",
        "headline",
        "reviewer",
        "checklist",
        "dashboard",
        "segment",
        "audience",
        "metrics",
        "workflow",
        "subtitles",
        "voiceover",
        "keywords",
        "timeline",
        "chapters",
        "glossary",
        "invoice",
        "interview",
    ],
    short: &[
        "use", "cut", "flag", "tag", "trim", "list", "note", "edit", "clip", "map", "plan", "copy",
        "file", "logo", "font", "crop", "sort", "fix", "mark", "rank",
    ],
    filler: &[
        "really",
        "basically",
        "actually",
        "honestly",
        "literally",
        "totally",
        "simply",
        "kinda",
        "sorta",
        "anyway",
        "like",
        "well",
        "okay",
        "whatnot",
    ],
    names: &[
        "Zorvex", "Kelmar", "Quintara", "Brovik", "Ostrel", "Valdis", "Tremont", "Halvo",
        "Peloria", "Nimbra", "Castor", "Ilvane",
    ],
    roles: &[
        "editor", "producer", "client", "designer", "analyst", "host",
    ],
};

/// Vocabulario desplazado (palabras nunca vistas en entrenamiento).
pub const VOCAB_SHIFT: Vocab = Vocab {
    long: &[
        "proposal",
        "estimate",
        "manuscript",
        "itinerary",
        "inventory",
        "agenda",
        "contract",
        "translation",
        "abstract",
        "portfolio",
        "newsletter",
        "roadmap",
        "spreadsheet",
    ],
    short: &[
        "add", "pin", "log", "dub", "loop", "mute", "slot", "nest", "size", "pad",
    ],
    filler: &[
        "truly",
        "merely",
        "frankly",
        "seriously",
        "um",
        "uh",
        "yeah",
        "meh",
    ],
    names: &[
        "Dravon", "Elmquist", "Fenwyck", "Gorlan", "Hespera", "Juvik",
    ],
    roles: &["auditor", "translator", "planner", "manager"],
};

const AMBIG_RESOLVABLE: &[(&str, &str)] = &[
    ("short", "under {n} seconds"),
    ("soon", "by November {n}"),
    ("some", "{n} items"),
    ("few", "{n} at most"),
    ("several", "{n} total"),
    ("quick", "within {n} minutes"),
    ("later", "after {n} pm"),
    ("many", "{n} or more"),
];
const AMBIG_DISTRACTOR: &[&str] = &["see notes", "as usual", "you know"];
const CLAIMS: &[&str] = &[
    "best",
    "fastest",
    "perfect",
    "complete",
    "guaranteed",
    "proven",
    "viral",
];
const SOURCES: &[&str] = &[
    "per the 2025 report",
    "according to the brief",
    "source: survey data",
    "per the client study",
];
const DEICTIC: &[&str] = &["this", "that", "it"];

/// Prompt sintético: texto + verdad por palabra (en orden de forks).
#[derive(Clone, Debug)]
pub struct SynthPrompt {
    pub seed: u64,
    pub text: String,
    pub truths: Vec<Truth>,
}

struct Builder<'a> {
    text: String,
    truths: Vec<Truth>,
    rng: Sm,
    v: &'a Vocab,
}

fn auto_truth(w: &str) -> Truth {
    let lw = w.to_lowercase();
    if STOPWORDS.contains(&lw.as_str()) && !AMBIGUITY_MARKERS.contains(&lw.as_str()) {
        Truth::structural()
    } else if approval_prior(&lw) >= 0.5 {
        Truth {
            approval: true,
            ..Truth::content()
        }
    } else {
        Truth::content()
    }
}

impl Builder<'_> {
    fn raw(&mut self, s: &str) {
        self.text.push_str(s);
    }
    fn word(&mut self, w: &str, t: Truth) {
        if !self.text.is_empty() && !self.text.ends_with(['\n', ' ', '(']) {
            self.text.push(' ');
        }
        self.text.push_str(w);
        self.truths.push(t);
    }
    /// Palabra con verdad automática (estructural / aprobación / contenido).
    fn auto(&mut self, w: &str) {
        self.word(w, auto_truth(w));
    }
    fn words(&mut self, s: &str) {
        for w in s.split_whitespace() {
            self.auto(w);
        }
    }
    /// Paréntesis con palabras automáticas (`source:` conserva los dos puntos).
    fn paren(&mut self, s: &str) {
        if !self.text.ends_with([' ', '\n']) {
            self.text.push(' ');
        }
        self.text.push('(');
        for (k, w) in s.split_whitespace().enumerate() {
            let (core, colon) = match w.strip_suffix(':') {
                Some(c) => (c, true),
                None => (w, false),
            };
            if k > 0 {
                self.text.push(' ');
            }
            self.text.push_str(core);
            self.truths.push(auto_truth(core));
            if colon {
                self.text.push(':');
            }
        }
        self.text.push(')');
    }
    fn n(&mut self) -> String {
        (2 + self.rng.below(58)).to_string()
    }

    /// Un evento «difícil» (lo que la heurística suele escalar).
    fn event(&mut self, noun: &str) {
        match self.rng.below(7) {
            0 => {
                // Marcador de ambigüedad resuelto (paréntesis con número) o no.
                let (m, tpl) = *self.rng.pick(AMBIG_RESOLVABLE);
                let resolved = self.rng.chance(0.5);
                self.words("keep the");
                self.auto(noun);
                self.word(
                    m,
                    Truth {
                        ambiguous: !resolved,
                        ..Truth::content()
                    },
                );
                if resolved {
                    let n = self.n();
                    self.paren(&tpl.replace("{n}", &n));
                } else if self.rng.chance(0.4) {
                    let d = *self.rng.pick(AMBIG_DISTRACTOR);
                    self.paren(d);
                }
            }
            1 => {
                // Deíctico con o sin referente inmediato.
                let d = *self.rng.pick(DEICTIC);
                let with_noun = d != "it" && self.rng.chance(0.5);
                let verb = *self.rng.pick(&["review", "check", "update", "polish"]);
                self.auto(verb);
                if with_noun {
                    self.word(d, Truth::content());
                    self.auto(noun);
                } else {
                    self.word(
                        d,
                        Truth {
                            ambiguous: true,
                            ..Truth::content()
                        },
                    );
                }
            }
            2 => {
                // Afirmación con o sin fuente.
                let c = *self.rng.pick(CLAIMS);
                let sourced = self.rng.chance(0.5);
                self.words("deliver the");
                self.word(
                    c,
                    Truth {
                        grounded: sourced,
                        ..Truth::content()
                    },
                );
                self.auto(noun);
                if sourced {
                    let s = *self.rng.pick(SOURCES);
                    self.paren(s);
                }
            }
            3 => {
                // Nombre propio definido (aposición) o no.
                let name = *self.rng.pick(self.v.names);
                let role = *self.rng.pick(self.v.roles);
                let defined = self.rng.chance(0.5);
                self.words("share the");
                self.auto(noun);
                self.words("with");
                self.word(
                    name,
                    Truth {
                        grounded: defined,
                        ..Truth::content()
                    },
                );
                if defined {
                    self.raw(",");
                    self.words("our");
                    self.auto(role);
                    self.raw(",");
                }
            }
            4 => {
                // Relleno (irrelevante).
                let f = *self.rng.pick(self.v.filler);
                self.word(
                    f,
                    Truth {
                        relevant: false,
                        ..Truth::content()
                    },
                );
                let s = *self.rng.pick(self.v.short);
                self.auto(s);
                self.words("the");
                self.auto(noun);
            }
            5 => {
                // Verbo corto de contenido (relevante, claro).
                let s = *self.rng.pick(self.v.short);
                self.auto(s);
                self.words("the");
                self.auto(noun);
            }
            _ => {
                // Instrucción «never/always + verbo» (regla, no afirmación).
                let c = *self.rng.pick(&["never", "always"]);
                let verb = *self.rng.pick(&["skip", "rename", "reorder", "hide"]);
                self.word(c, Truth::content());
                self.auto(verb);
                self.words("the");
                self.auto(noun);
            }
        }
    }
}

fn capitalize(w: &str) -> String {
    let mut cs = w.chars();
    match cs.next() {
        Some(c) => c.to_uppercase().collect::<String>() + cs.as_str(),
        None => String::new(),
    }
}

/// Genera un prompt sintético (`<prompt>` con role/objective/context/steps/
/// constraints) con verdad por construcción para cada palabra.
pub fn synth_prompt(seed: u64, vocab: &Vocab) -> SynthPrompt {
    let mut b = Builder {
        text: String::new(),
        truths: Vec::new(),
        rng: Sm(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x5EED),
        v: vocab,
    };
    let v = vocab;
    b.raw("<prompt>\n<role>\n");
    let role = *b.rng.pick(v.roles);
    b.words("You are the");
    b.auto(role);
    b.words("for a");
    let org = *b.rng.pick(v.long);
    b.auto(org);
    b.words("team");
    b.raw(".\n</role>\n<objective>\n");
    let obj1 = *b.rng.pick(v.long);
    let obj2 = *b.rng.pick(v.long);
    b.words("Prepare the");
    b.auto(obj1);
    b.words("and the");
    b.auto(obj2);
    b.words("for");
    let n = b.n();
    b.word(&n, Truth::content());
    b.words("episodes");
    b.raw(".\n</objective>\n<context>\n");
    if b.rng.chance(0.7) {
        let noun = *b.rng.pick(v.long);
        b.words("The");
        b.auto(noun);
        b.words("is due on");
        let n = b.n();
        b.word(&n, Truth::content());
        b.words("March");
        b.raw(".\n");
    }
    b.raw("</context>\n<steps>\n");
    let steps = 3 + b.rng.below(4);
    for s in 0..steps {
        b.raw(&format!("{}. ", s + 1));
        let lead = capitalize(b.rng.pick(v.short));
        b.auto(&lead);
        b.words("the");
        let noun = *b.rng.pick(v.long);
        b.auto(noun);
        b.raw(",");
        b.words("then");
        let noun2 = *b.rng.pick(v.long);
        b.event(noun2);
        if b.rng.chance(0.5) {
            b.raw(";");
            b.words("then");
            let noun3 = *b.rng.pick(v.long);
            b.event(noun3);
        }
        if b.rng.chance(0.2) {
            b.raw(",");
            b.words("then ask me before you publish");
        }
        b.raw(".\n");
    }
    b.raw("</steps>\n<constraints>\n");
    let noun = *b.rng.pick(v.long);
    b.words("Never invent numbers in the");
    b.auto(noun);
    b.raw(".\n");
    if b.rng.chance(0.6) {
        let noun = *b.rng.pick(v.long);
        b.words("Also");
        b.event(noun);
        b.raw(".\n");
    }
    b.raw("</constraints>\n</prompt>");
    SynthPrompt {
        seed,
        text: b.text,
        truths: b.truths,
    }
}

/// Ejemplos de un prompt sintético (todas las palabras, verdad por
/// construcción). `noise` = prob. de invertir cada etiqueta (supervisión
/// ruidosa; 0 en hold-out).
pub fn synth_examples(
    sp: &SynthPrompt,
    threshold: f64,
    noise: f64,
    rng: &mut Sm,
) -> Vec<SpiderExample> {
    let ap = AnalyzedPrompt::new(&sp.text, threshold);
    let forks: Vec<usize> = (0..ap.tokens.len())
        .filter(|&i| ap.tokens[i].kind.is_fork())
        .collect();
    assert_eq!(
        forks.len(),
        sp.truths.len(),
        "alineación sintética rota (seed {}): {}",
        sp.seed,
        sp.text
    );
    forks
        .iter()
        .zip(sp.truths.iter())
        .filter_map(|(&i, t)| {
            let mut l = t.labels();
            if noise > 0.0 {
                for x in l.iter_mut() {
                    if let Some(b) = *x {
                        if rng.chance(noise) {
                            *x = Some(!b);
                        }
                    }
                }
            }
            ap.encode(i, l, LabelSource::Synthetic, sp.seed)
        })
        .collect()
}

/// Ejemplos con etiquetas del scorer (solo palabras `code` / `you`).
pub fn scorer_examples(text: &str, threshold: f64, group: u64) -> Vec<SpiderExample> {
    let ap = AnalyzedPrompt::new(text, threshold);
    (0..ap.tokens.len())
        .filter_map(|i| {
            let (a, r) = ap.assess[i].as_ref()?;
            let l = scorer_labels(a, *r)?;
            ap.encode(i, l, LabelSource::Scorer, group)
        })
        .collect()
}

/// Grupo estable para un texto (prompts reales).
pub fn text_group(text: &str) -> u64 {
    fnv(text)
}

// ---------------------------------------------------------------------------
// Métricas
// ---------------------------------------------------------------------------

/// Métricas binarias de una cabeza.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BinMetrics {
    pub n: usize,
    pub accuracy: f64,
    pub ece: f64,
    pub brier: f64,
}

/// Exactitud (umbral 0.5), ECE (10 bins de igual ancho) y Brier.
pub fn bin_metrics(pairs: &[(f64, bool)]) -> BinMetrics {
    let n = pairs.len();
    if n == 0 {
        return BinMetrics::default();
    }
    let acc = pairs.iter().filter(|(p, y)| (*p >= 0.5) == *y).count() as f64 / n as f64;
    let brier = pairs
        .iter()
        .map(|(p, y)| (p - if *y { 1.0 } else { 0.0 }).powi(2))
        .sum::<f64>()
        / n as f64;
    let mut bins = [(0f64, 0f64, 0usize); 10];
    for (p, y) in pairs {
        let k = ((p * 10.0) as usize).min(9);
        bins[k].0 += p;
        bins[k].1 += if *y { 1.0 } else { 0.0 };
        bins[k].2 += 1;
    }
    let ece = bins
        .iter()
        .filter(|b| b.2 > 0)
        .map(|b| (b.2 as f64 / n as f64) * ((b.0 - b.1) / b.2 as f64).abs())
        .sum();
    BinMetrics {
        n,
        accuracy: acc,
        ece,
        brier,
    }
}

/// Métricas del router sobre las escaladas.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RouterMetrics {
    pub escalated: usize,
    /// % resueltas por el campo.
    pub resolved_pct: f64,
    /// Precisión de lo resuelto (verdad ok entre las resueltas).
    pub resolved_precision: f64,
    /// % de escaladas resueltas erróneamente (resueltas con verdad no-ok).
    pub wrong_resolved_pct: f64,
    /// % pendientes.
    pub pending_pct: f64,
    pub ok: BinMetrics,
}

/// Filas `(P(ok), resuelve, verdad_ok)` → métricas del router.
pub fn router_metrics(rows: &[(f64, bool, bool)]) -> RouterMetrics {
    let n = rows.len();
    if n == 0 {
        return RouterMetrics::default();
    }
    let res: Vec<&(f64, bool, bool)> = rows.iter().filter(|r| r.1).collect();
    let good = res.iter().filter(|r| r.2).count();
    RouterMetrics {
        escalated: n,
        resolved_pct: 100.0 * res.len() as f64 / n as f64,
        resolved_precision: if res.is_empty() {
            0.0
        } else {
            good as f64 / res.len() as f64
        },
        wrong_resolved_pct: 100.0 * (res.len() - good) as f64 / n as f64,
        pending_pct: 100.0 * (n - res.len()) as f64 / n as f64,
        ok: bin_metrics(&rows.iter().map(|r| (r.0, r.2)).collect::<Vec<_>>()),
    }
}

/// Evaluación hold-out del router (lo que muestra la UI y guardan los
/// checkpoints).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FieldEval {
    /// Por cabeza (orden [`HEAD_NAMES`]) en todas las palabras de H1.
    pub heads_h1: Vec<BinMetrics>,
    /// `ok` por ruta inicial (code, llm, you) en H1.
    pub ok_by_route_h1: Vec<BinMetrics>,
    /// Router en las escaladas de H1 (vocabulario de entrenamiento) y H2
    /// (vocabulario nuevo).
    pub router_h1: RouterMetrics,
    pub router_h2: RouterMetrics,
    /// Baseline heurístico (scorer + piso) en las mismas escaladas.
    pub heuristic_h1: RouterMetrics,
    pub heuristic_h2: RouterMetrics,
    /// Latencia mediana por decisión (µs).
    pub latency_us: f64,
    pub h1_words: usize,
    pub h2_words: usize,
}

/// Evalúa `router` en H1/H2 (etiquetas limpias) con el piso dado.
pub fn evaluate_router(
    router: &SpiderFieldRouter,
    h1: &[SpiderExample],
    h2: &[SpiderExample],
    floor: f64,
) -> FieldEval {
    let mut per_head: Vec<Vec<(f64, bool)>> = vec![Vec::new(); HEADS];
    let mut by_route: Vec<Vec<(f64, bool)>> = vec![Vec::new(); 3];
    let mut lat = Vec::new();
    let mut rows_h1 = Vec::new();
    let mut heur_h1 = Vec::new();
    for ex in h1 {
        let p = router.predict(ex);
        for h in 0..HEADS {
            if let Some(y) = ex.labels[h] {
                per_head[h].push((p.probs[h], y));
            }
        }
        let r = match ex.first_route {
            Route::Code => 0,
            Route::Llm => 1,
            Route::You => 2,
        };
        let truth = ex.labels[H_OK].unwrap_or(false);
        by_route[r].push((p.p_ok(), truth));
        if ex.first_route == Route::Llm {
            let d = router.decide(ex, floor);
            lat.push(d.micros);
            rows_h1.push((d.prediction.p_ok(), d.resolve, truth));
            heur_h1.push((ex.heur_ok, ex.heur_ok >= floor, truth));
        }
    }
    let mut rows_h2 = Vec::new();
    let mut heur_h2 = Vec::new();
    for ex in h2.iter().filter(|e| e.first_route == Route::Llm) {
        let d = router.decide(ex, floor);
        let truth = ex.labels[H_OK].unwrap_or(false);
        rows_h2.push((d.prediction.p_ok(), d.resolve, truth));
        heur_h2.push((ex.heur_ok, ex.heur_ok >= floor, truth));
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    FieldEval {
        heads_h1: per_head.iter().map(|v| bin_metrics(v)).collect(),
        ok_by_route_h1: by_route.iter().map(|v| bin_metrics(v)).collect(),
        router_h1: router_metrics(&rows_h1),
        router_h2: router_metrics(&rows_h2),
        heuristic_h1: router_metrics(&heur_h1),
        heuristic_h2: router_metrics(&heur_h2),
        latency_us: if lat.is_empty() {
            0.0
        } else {
            lat[lat.len() / 2]
        },
        h1_words: h1.len(),
        h2_words: h2.len(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_spider::{DEFAULT_LLM_FLOOR, DEFAULT_THRESHOLD, SAMPLE_PROMPT};

    #[test]
    fn synthetic_prompts_align_with_tokenizer() {
        for seed in 0..200 {
            for v in [&VOCAB_TRAIN, &VOCAB_SHIFT] {
                let sp = synth_prompt(seed, v);
                let ex = synth_examples(&sp, DEFAULT_THRESHOLD, 0.0, &mut Sm(1));
                assert_eq!(ex.len(), sp.truths.len());
            }
        }
    }

    #[test]
    fn synthetic_prompts_produce_escalations_of_both_kinds() {
        let (mut esc_ok, mut esc_bad) = (0, 0);
        for seed in 0..40 {
            let sp = synth_prompt(seed, &VOCAB_TRAIN);
            for ex in synth_examples(&sp, DEFAULT_THRESHOLD, 0.0, &mut Sm(1)) {
                if ex.first_route == Route::Llm {
                    if ex.labels[H_OK] == Some(true) {
                        esc_ok += 1;
                    } else {
                        esc_bad += 1;
                    }
                }
            }
        }
        assert!(esc_ok > 50 && esc_bad > 50, "ok={esc_ok} bad={esc_bad}");
    }

    #[test]
    fn encoding_is_deterministic_and_sized() {
        let ap = AnalyzedPrompt::new(SAMPLE_PROMPT, DEFAULT_THRESHOLD);
        let i = ap.tokens.iter().position(|t| t.kind.is_fork()).unwrap();
        let a = ap.encode(i, [None; HEADS], LabelSource::Scorer, 0).unwrap();
        let b = ap.encode(i, [None; HEADS], LabelSource::Scorer, 0).unwrap();
        assert_eq!(a.window.len(), WINDOW * LOCAL_DIM);
        assert_eq!(a.global.len(), GLOBAL_DIM);
        assert_eq!(a.window, b.window);
        assert_eq!(decision_context(&a), decision_context(&b));
        assert_eq!(decision_context(&a).len(), CTX_DIM);
    }

    #[test]
    fn untrained_router_never_resolves() {
        let r = SpiderFieldRouter::new(FieldConfig::v4(1));
        let ap = AnalyzedPrompt::new(SAMPLE_PROMPT, DEFAULT_THRESHOLD);
        for i in 0..ap.tokens.len() {
            if let Some(ex) = ap.encode(i, [None; HEADS], LabelSource::Scorer, 0) {
                let d = r.decide(&ex, DEFAULT_LLM_FLOOR);
                assert!(!d.resolve);
                assert!(d.reason.contains("sin entrenar"));
            }
        }
    }

    #[test]
    fn user_and_teacher_labels_follow_rules() {
        let l = user_labels(Question::Ambiguous, true);
        assert_eq!(l[H_AMB], Some(false));
        assert_eq!(l[H_OK], Some(true));
        let l = user_labels(Question::Approval, true);
        assert_eq!(l[H_APR], Some(true));
        assert_eq!(l[H_OK], Some(false));
        let v = LlmVerdict {
            index: 0,
            relevant: true,
            grounded: true,
            ambiguous: false,
            needs_approval: false,
            p: 0.9,
            note: String::new(),
        };
        let l = teacher_labels(Question::Grounded, &v, 0.7, true);
        assert_eq!(l[H_OK], Some(true));
        assert_eq!(l[H_GRD], Some(true));
        assert_eq!(l[H_REL], None);
        let v2 = LlmVerdict { p: 0.4, ..v };
        assert_eq!(
            teacher_labels(Question::Relevant, &v2, 0.7, true)[H_OK],
            Some(false)
        );
    }

    #[test]
    fn metrics_basics() {
        let m = bin_metrics(&[(0.9, true), (0.1, false), (0.8, false), (0.3, true)]);
        assert!((m.accuracy - 0.5).abs() < 1e-9);
        assert!(m.brier > 0.0 && m.ece > 0.0);
        let r = router_metrics(&[(0.9, true, true), (0.8, true, false), (0.2, false, false)]);
        assert!((r.resolved_pct - 66.666).abs() < 0.1);
        assert!((r.resolved_precision - 0.5).abs() < 1e-9);
    }

    /// Entrena un router pequeño: aprende, consolida CDT/RQM y el modelo
    /// guardado reproduce las predicciones.
    #[test]
    fn small_router_learns_and_roundtrips() {
        let mut cfg = FieldConfig::v4(3);
        cfg.epochs = 60;
        cfg.steps = 1500;
        let mut r = SpiderFieldRouter::new(cfg);
        let mut rng = Sm(9);
        for seed in 0..40 {
            let sp = synth_prompt(seed, &VOCAB_TRAIN);
            for ex in synth_examples(&sp, DEFAULT_THRESHOLD, 0.0, &mut rng) {
                r.observe(ex);
            }
        }
        let rep = r.sleep();
        assert!(
            rep.trusted_pairs > 0 && rep.trusted_pairs <= rep.replay_pairs,
            "{rep:?}"
        );
        assert_eq!(r.wake_len(), 0, "el sueño borra la CDT");
        assert!(r.is_trained());
        let mut rows = Vec::new();
        let mut hold = Vec::new();
        for seed in 5000..5015 {
            let sp = synth_prompt(seed, &VOCAB_TRAIN);
            for ex in synth_examples(&sp, DEFAULT_THRESHOLD, 0.0, &mut rng) {
                if ex.first_route == Route::Llm {
                    let p = r.predict(&ex).p_ok();
                    rows.push((p, ex.labels[H_OK].unwrap()));
                    hold.push(ex);
                }
            }
        }
        let m = bin_metrics(&rows);
        assert!(m.accuracy > 0.75, "{m:?}");
        let json = serde_json::to_string(&r.to_file()).unwrap();
        let r2 = SpiderFieldRouter::from_file(serde_json::from_str(&json).unwrap()).unwrap();
        for ex in hold.iter().take(30) {
            let (a, b) = (r.predict(ex), r2.predict(ex));
            for h in 0..HEADS {
                assert!((a.probs[h] - b.probs[h]).abs() < 1e-5);
            }
        }
    }
}
