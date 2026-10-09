//! **Decoder del campo para Prompt Spider** (router del campo).
//!
//! Sustituye al LLM en las palabras **escaladas** del Prompt Spider cuando el
//! interruptor «Decoder del campo» está activo, igual que en el chat: el campo
//! produce el estado y la decisión; ningún LLM opina.
//!
//! Pipeline de una decisión (palabra = fork):
//!
//! 1. **Codificación** ([`encode_decision`]): rasgos del token (tipo, léxicos
//!    del scorer, mayúscula/inicio de frase, longitud, frecuencia, sección,
//!    puntuación vecina, hashing de identidad y sufijo), ventana de contexto
//!    ±3 tokens y rasgos globales de la decisión (las 4 puntuaciones
//!    heurísticas, `p`, pregunta decisiva y ruta inicial).
//! 2. **Inferencia líquida** ([`LiquidEncoder`]): reservorio de constante de
//!    tiempo dependiente de la entrada (LTC) que lee la ventana token a token;
//!    el estado (centro ⊕ final) es el estado del campo. Pesos fijos por
//!    semilla (no se entrenan).
//! 3. **CDT** ([`CdtBasinMemory`]): engramas por clase (pregunta × sí/no)
//!    consolidados **solo en el sueño** sobre un `NativeThermoCdtSubstrate`
//!    (bloques de nodos disjuntos, piloto amp/fase del prototipo, 8 pasos
//!    termodinámicos, plantilla amp/cos/sin). En inferencia solo se leen las
//!    plantillas: similitudes + **novedad** (guardia fuera de distribución).
//! 4. **RQM/EPR** ([`RqmIndex`]): índice relacional celda→etiqueta en un
//!    `NativeThermoRqmEprSubstrate` (entrelazamiento EPR incluido). Celda =
//!    hash LSH (8 bits) del estado líquido. Se escribe en el sueño y se
//!    destila a una tabla de priors; la inferencia lee la tabla (sin mutar).
//! 5. **Lectura** ([`Readout`]): 5 cabezas logísticas (relevant, grounded,
//!    ambiguous, needs_approval, ok) + escalado de temperatura por cabeza
//!    (calibración en un split de calibración por prompt).
//! 6. **Router**: `ok` es la cabeza de ruta (resolver vs. pendiente). Una
//!    escalada se resuelve por el campo solo si `P(ok) ≥ piso`, `P(aprob) <
//!    0.5`, `P(ambigua) < 0.5` y la novedad está dentro de lo visto en
//!    entrenamiento. Si no, **pendiente** (nunca se inventa).
//!
//! Ver `docs/prompt_spider_decoder_campo.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::entanglement::EntanglementConfig;
use crate::liquid_cdt_rqm_fuse::FusedLiquidCdt;
use crate::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use crate::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use crate::prompt_spider::{
    approval_prior, route, score_token, tokenize, Assessment, LlmVerdict, PromptContext, Question,
    Route, Token, TokenKind, AMBIGUITY_MARKERS, CLAIM_MARKERS, KNOWN_ENTITIES, STOPWORDS,
};
use crate::relational_field::ObserverId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
pub const CORE_DIM: usize = 12;
/// Versión del formato del modelo persistido.
pub const MODEL_VERSION: u32 = 1;

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
// Inferencia líquida (reservorio LTC)
// ---------------------------------------------------------------------------

/// Reservorio líquido de constante de tiempo dependiente de la entrada:
/// `τ_i(u) = 1 + 3·σ(a_i·u)`, `h ← h + (−h + tanh(W_in u + W h + b)) / τ`.
#[derive(Clone, Debug)]
pub struct LiquidEncoder {
    pub n: usize,
    pub in_dim: usize,
    w_in: Vec<f32>,
    w_rec: Vec<f32>,
    w_tau: Vec<f32>,
    bias: Vec<f32>,
}

impl LiquidEncoder {
    pub fn new(n: usize, seed: u64) -> Self {
        let in_dim = LOCAL_DIM + GLOBAL_DIM;
        let mut rng = Sm(seed ^ 0x11C0_1D00);
        let s_in = 1.0 / (in_dim as f64).sqrt();
        let w_in = (0..n * in_dim)
            .map(|_| (rng.normal() * s_in * 1.5) as f32)
            .collect();
        let density = 0.2;
        let s_rec = 0.9 / (n as f64 * density).sqrt();
        let w_rec = (0..n * n)
            .map(|_| {
                if rng.chance(density) {
                    (rng.normal() * s_rec) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let w_tau = (0..n * in_dim)
            .map(|_| (rng.normal() * s_in) as f32)
            .collect();
        let bias = (0..n).map(|_| (rng.normal() * 0.1) as f32).collect();
        Self {
            n,
            in_dim,
            w_in,
            w_rec,
            w_tau,
            bias,
        }
    }

    /// Dimensión del estado (centro ⊕ final).
    pub fn state_dim(&self) -> usize {
        2 * self.n
    }

    /// Lee la ventana token a token; devuelve `[h_centro, h_final]`.
    pub fn run(&self, ex: &SpiderExample) -> Vec<f32> {
        let n = self.n;
        let mut h = vec![0f32; n];
        let mut u = vec![0f32; self.in_dim];
        let mut out = vec![0f32; 2 * n];
        let mut nh = vec![0f32; n];
        for k in 0..WINDOW {
            u[..LOCAL_DIM].copy_from_slice(&ex.window[k * LOCAL_DIM..(k + 1) * LOCAL_DIM]);
            u[LOCAL_DIM..].copy_from_slice(&ex.global);
            for i in 0..n {
                let wi = &self.w_in[i * self.in_dim..(i + 1) * self.in_dim];
                let wt = &self.w_tau[i * self.in_dim..(i + 1) * self.in_dim];
                let mut pre = self.bias[i];
                let mut at = 0f32;
                for d in 0..self.in_dim {
                    pre += wi[d] * u[d];
                    at += wt[d] * u[d];
                }
                let wr = &self.w_rec[i * n..(i + 1) * n];
                for j in 0..n {
                    pre += wr[j] * h[j];
                }
                let tau = 1.0 + 3.0 * sigmoid(at as f64) as f32;
                nh[i] = h[i] + (-h[i] + pre.tanh()) / tau;
            }
            h.copy_from_slice(&nh);
            if k == WINDOW_RADIUS {
                out[..n].copy_from_slice(&h);
            }
        }
        out[n..].copy_from_slice(&h);
        out
    }
}

// ---------------------------------------------------------------------------
// CDT: engramas por clase consolidados en el sueño
// ---------------------------------------------------------------------------

const CDT_BLOCK: usize = 12;
const CDT_STEPS: usize = 8;

/// Estado persistible de la memoria CDT.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CdtState {
    pub protos: Vec<Vec<f32>>,
    pub counts: Vec<u64>,
    pub templates: Vec<Vec<f32>>,
}

/// Memoria CDT: prototipos por clase → plantillas tras relajación
/// termodinámica (solo en el sueño). La inferencia solo lee plantillas.
#[derive(Clone, Debug)]
pub struct CdtBasinMemory {
    pub substrate: NativeThermoCdtSubstrate,
    dim: usize,
    p1: Vec<f32>,
    p2: Vec<f32>,
    pub state: CdtState,
    pending: Vec<(usize, Vec<f32>)>,
    pub sleep_steps: u64,
}

impl CdtBasinMemory {
    pub fn new(dim: usize, seed: u64) -> Self {
        let nodes = CLASSES * CDT_BLOCK;
        let cfg = NativeThermoCdtConfig {
            slices: 2,
            nodes_per_slice: nodes.div_ceil(2).max(64),
            spatial_degree: 3,
            temporal_degree: 1,
            temperature: 0.2,
            seed: seed ^ 0xCD7_5F1D,
            ..NativeThermoCdtConfig::default()
        };
        let mut rng = Sm(seed ^ 0xCD7_0001);
        let s = 1.0 / (dim as f64).sqrt();
        let p1 = (0..CDT_BLOCK * dim)
            .map(|_| (rng.normal() * s) as f32)
            .collect();
        let p2 = (0..CDT_BLOCK * dim)
            .map(|_| (rng.normal() * s) as f32)
            .collect();
        Self {
            substrate: NativeThermoCdtSubstrate::new(cfg),
            dim,
            p1,
            p2,
            state: CdtState {
                protos: vec![vec![0.0; dim]; CLASSES],
                counts: vec![0; CLASSES],
                templates: vec![Vec::new(); CLASSES],
            },
            pending: Vec::new(),
            sleep_steps: 0,
        }
    }

    /// Proyección (amp, fase) por nodo del bloque.
    fn pilot(&self, s: &[f32]) -> Vec<(f32, f32)> {
        (0..CDT_BLOCK)
            .map(|j| {
                let (mut v, mut w) = (0f32, 0f32);
                let r1 = &self.p1[j * self.dim..(j + 1) * self.dim];
                let r2 = &self.p2[j * self.dim..(j + 1) * self.dim];
                for d in 0..self.dim.min(s.len()) {
                    v += r1[d] * s[d];
                    w += r2[d] * s[d];
                }
                (0.5 + 0.5 * v.tanh(), std::f32::consts::PI * w.tanh())
            })
            .collect()
    }

    fn signature(&self, s: &[f32]) -> Vec<f32> {
        let mut sig = Vec::with_capacity(CDT_BLOCK * 3);
        for (amp, ph) in self.pilot(s) {
            sig.push(amp - 0.5);
            sig.push(ph.cos());
            sig.push(ph.sin());
        }
        sig
    }

    /// Vigilia: bufferiza el estado bajo sus clases etiquetadas.
    pub fn buffer(&mut self, state: &[f32], labels: &Labels) {
        for (h, l) in labels.iter().enumerate() {
            if let Some(b) = l {
                self.pending.push((2 * h + *b as usize, state.to_vec()));
            }
        }
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Sueño: actualiza prototipos (incremental, sin borrar engramas) y
    /// re-codifica las clases tocadas en el sustrato termodinámico.
    pub fn consolidate(&mut self) -> usize {
        let pend = std::mem::take(&mut self.pending);
        let mut touched = [false; CLASSES];
        for (c, s) in pend.iter() {
            let n = self.state.counts[*c] as f32;
            let p = &mut self.state.protos[*c];
            for d in 0..self.dim.min(s.len()) {
                p[d] = (p[d] * n + s[d]) / (n + 1.0);
            }
            self.state.counts[*c] += 1;
            touched[*c] = true;
        }
        for c in 0..CLASSES {
            if !touched[c] {
                continue;
            }
            let nodes: Vec<usize> = (c * CDT_BLOCK..(c + 1) * CDT_BLOCK).collect();
            let target = self.pilot(&self.state.protos[c].clone());
            for (k, &node) in nodes.iter().enumerate() {
                if node < self.substrate.node_count() {
                    self.substrate.amplitude[node] = 0.5;
                    self.substrate.phase[node] = 0.0;
                    self.substrate.thermal_state[node] = 0.0;
                    self.substrate.pilot_force[node] = 0.0;
                    let (amp, ph) = target[k];
                    self.substrate.inject_local_node(node, amp, ph, 1.0);
                }
            }
            for _ in 0..CDT_STEPS {
                let _ = self.substrate.step();
                self.sleep_steps += 1;
            }
            let mut tpl = Vec::with_capacity(CDT_BLOCK * 3);
            for &node in &nodes {
                let (a, ph) = if node < self.substrate.node_count() {
                    (self.substrate.amplitude[node], self.substrate.phase[node])
                } else {
                    (0.5, 0.0)
                };
                tpl.push(a.clamp(0.0, 1.0) - 0.5);
                tpl.push(ph.cos());
                tpl.push(ph.sin());
            }
            self.state.templates[c] = tpl;
        }
        pend.len()
    }

    /// Similitud coseno a cada plantilla (0 si la clase no existe) + novedad
    /// (`1 − máx` sobre las clases con plantilla).
    pub fn read(&self, state: &[f32]) -> (Vec<f32>, f32) {
        let sig = self.signature(state);
        let mut sims = vec![0f32; CLASSES];
        let mut best = f32::NEG_INFINITY;
        for c in 0..CLASSES {
            let t = &self.state.templates[c];
            if t.is_empty() {
                continue;
            }
            let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
            for k in 0..t.len().min(sig.len()) {
                dot += sig[k] * t[k];
                na += sig[k] * sig[k];
                nb += t[k] * t[k];
            }
            let s = dot / (na.sqrt() * nb.sqrt()).max(1e-6);
            sims[c] = s;
            best = best.max(s);
        }
        let novelty = if best.is_finite() { 1.0 - best } else { 1.0 };
        (sims, novelty)
    }

    pub fn restore(&mut self, st: CdtState) {
        if st.protos.len() == CLASSES && st.templates.len() == CLASSES {
            self.state = st;
        }
    }
}

// ---------------------------------------------------------------------------
// RQM/EPR: índice relacional celda → etiqueta
// ---------------------------------------------------------------------------

const RQM_BITS: usize = 8;
const RQM_CUE_BASE: usize = 16;
const RQM_OBSERVER: ObserverId = ObserverId(0x5_91DE_F1E1);

/// Estado persistible del índice RQM.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RqmState {
    /// celda → conteo por clase.
    pub counts: HashMap<u32, Vec<u32>>,
    /// celda → (prior P(sí) por cabeza, soporte).
    pub cache: HashMap<u32, Vec<f32>>,
}

/// Índice relacional (RQM + EPR) de celdas del estado líquido a clases.
#[derive(Clone, Debug)]
pub struct RqmIndex {
    pub substrate: NativeThermoRqmEprSubstrate,
    dim: usize,
    lsh: Vec<f32>,
    pub state: RqmState,
    pending: Vec<(u32, Labels)>,
    pub train_calls: u64,
    pub query_calls: u64,
}

fn rqm_substrate(seed: u64) -> NativeThermoRqmEprSubstrate {
    let nodes = RQM_CUE_BASE + (1 << RQM_BITS);
    let thermal = NativeThermoCdtConfig {
        slices: 2,
        nodes_per_slice: nodes,
        seed: seed ^ 0x5F1E_0D00,
        ..NativeThermoCdtConfig::default()
    };
    let cfg = NativeThermoRqmConfig {
        max_candidates: CLASSES * 2,
        thermal_steps_per_train: 1,
        thermal_steps_per_query: 2,
        thermal_activation_margin: 0.05,
        collect_query_diagnostics: false,
        ..Default::default()
    };
    let epr = EntanglementConfig {
        create_threshold: 0.4,
        max_syncs_per_step: 64,
        ..EntanglementConfig::default()
    };
    NativeThermoRqmEprSubstrate::new(thermal, cfg, epr)
}

impl RqmIndex {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = Sm(seed ^ 0x5_0001);
        let lsh = (0..RQM_BITS * dim).map(|_| rng.normal() as f32).collect();
        Self {
            substrate: rqm_substrate(seed),
            dim,
            lsh,
            state: RqmState::default(),
            pending: Vec::new(),
            train_calls: 0,
            query_calls: 0,
        }
    }

    pub fn cell(&self, s: &[f32]) -> u32 {
        let mut c = 0u32;
        for bit in 0..RQM_BITS {
            let r = &self.lsh[bit * self.dim..(bit + 1) * self.dim];
            let mut v = 0f32;
            for d in 0..self.dim.min(s.len()) {
                v += r[d] * s[d];
            }
            if v > 0.0 {
                c |= 1 << bit;
            }
        }
        c
    }

    pub fn buffer(&mut self, state: &[f32], labels: &Labels) {
        let c = self.cell(state);
        self.pending.push((c, *labels));
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    fn train_one(&mut self, cell: u32, class: usize) {
        self.substrate.train_observed_transition(
            RQM_OBSERVER,
            0.0,
            &[RQM_CUE_BASE + cell as usize],
            &[class],
            0.95,
        );
        self.train_calls += 1;
    }

    fn refresh(&mut self, cell: u32) {
        let rep = self
            .substrate
            .query(RQM_OBSERVER, 0.0, &[RQM_CUE_BASE + cell as usize]);
        self.query_calls += 1;
        let mut sc = [0f32; CLASSES];
        for c in rep.candidates.iter() {
            if c.agent < CLASSES {
                sc[c.agent] = c.score.max(0.0);
            }
        }
        let mut out = Vec::with_capacity(HEADS + 1);
        let mut support = 0f32;
        for h in 0..HEADS {
            let (no, yes) = (sc[2 * h], sc[2 * h + 1]);
            out.push((yes + 1e-3) / (yes + no + 2e-3));
            support += yes + no;
        }
        out.push((support / HEADS as f32).tanh());
        self.state.cache.insert(cell, out);
    }

    /// Sueño: entrena el índice con lo bufferizado y destila la tabla de priors.
    pub fn consolidate(&mut self) -> usize {
        let pend = std::mem::take(&mut self.pending);
        let mut touched: Vec<u32> = Vec::new();
        for (cell, labels) in pend.iter() {
            for (h, l) in labels.iter().enumerate() {
                if let Some(b) = l {
                    let class = 2 * h + *b as usize;
                    self.train_one(*cell, class);
                    let e = self
                        .state
                        .counts
                        .entry(*cell)
                        .or_insert_with(|| vec![0; CLASSES]);
                    e[class] += 1;
                }
            }
            touched.push(*cell);
        }
        touched.sort_unstable();
        touched.dedup();
        for c in touched {
            self.refresh(c);
        }
        pend.len()
    }

    /// Priors (P(sí) por cabeza + soporte); celda no vista → 0.5 / 0.
    pub fn read(&self, state: &[f32]) -> Vec<f32> {
        let c = self.cell(state);
        self.state.cache.get(&c).cloned().unwrap_or_else(|| {
            let mut v = vec![0.5; HEADS];
            v.push(0.0);
            v
        })
    }

    /// Reconstruye el sustrato desde los conteos (al cargar un modelo).
    pub fn restore(&mut self, st: RqmState) {
        let mut cells: Vec<(u32, Vec<u32>)> = st.counts.clone().into_iter().collect();
        cells.sort_by_key(|(c, _)| *c);
        for (cell, counts) in cells {
            for (class, n) in counts.iter().enumerate() {
                for _ in 0..(*n).min(4) {
                    self.train_one(cell, class);
                }
            }
        }
        self.state = st;
    }
}

// ---------------------------------------------------------------------------
// Acoplamiento con el núcleo de chat (brazos B/C del experimento)
// ---------------------------------------------------------------------------

/// Respuesta del núcleo de chat (`FusedLiquidCdt`) para cada cue 0..8:
/// one-hot del concepto predicho + líquido + RQM + margen + ruta.
pub fn core_response_table(fuse: &mut FusedLiquidCdt) -> Vec<[f32; CORE_DIM]> {
    let cands: Vec<usize> = (0..fuse.num_labels).collect();
    (0..8)
        .map(|cue| {
            let r = fuse.infer(cue, &cands);
            let mut v = [0f32; CORE_DIM];
            v[r.predicted % 8] = 1.0;
            v[8] = r.liquid_score as f32;
            v[9] = r.rqm_score.unwrap_or(0.0) as f32;
            v[10] = r.margin.clamp(-5.0, 5.0) as f32;
            v[11] = if matches!(r.route, crate::liquid_cdt_rqm_fuse::InferRoute::RqmFallback) {
                1.0
            } else {
                0.0
            };
            v
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Lectura (cabezas logísticas + temperatura)
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
// Router del campo
// ---------------------------------------------------------------------------

/// Cómo se acopla el router al núcleo de chat existente.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coupling {
    /// (A) sustrato en blanco: solo datos del Spider.
    Blank,
    /// (B) núcleo de chat congelado (solo lectura) + capa externa.
    FrozenCore,
    /// (C) ajuste fino del núcleo compartido (escribe en el campo del chat).
    SharedCore,
}

/// Configuración del router (componentes activos + entrenamiento).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldConfig {
    pub seed: u64,
    pub liquid: bool,
    pub cdt: bool,
    pub rqm: bool,
    pub coupling: Coupling,
    pub n_liquid: usize,
    pub epochs: usize,
    pub l2: f32,
    /// Ejemplos máximos en el buffer de repetición (reservorio).
    pub replay_cap: usize,
}

impl FieldConfig {
    /// (A) por defecto: líquido + CDT + RQM en sustrato en blanco.
    pub fn blank(seed: u64) -> Self {
        Self {
            seed,
            liquid: true,
            cdt: true,
            rqm: true,
            coupling: Coupling::Blank,
            n_liquid: 48,
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
    pub cell: Option<u32>,
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
    pub episodes: usize,
    pub cdt_classes: usize,
    pub rqm_cells: usize,
    pub replay: usize,
    pub ms: f64,
}

/// Modelo persistido (pesos líquidos = semilla; CDT/RQM = estado; lectura).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldModelFile {
    pub version: u32,
    pub cfg: FieldConfig,
    pub readout: Readout,
    pub cdt: Option<CdtState>,
    pub rqm: Option<RqmState>,
    pub novelty_max: f64,
    pub examples_seen: u64,
    pub sleeps: u64,
    #[serde(default)]
    pub holdout: Option<serde_json::Value>,
}

/// Router del campo para Prompt Spider.
#[derive(Clone, Debug)]
pub struct SpiderFieldRouter {
    pub cfg: FieldConfig,
    pub liquid: LiquidEncoder,
    pub cdt: Option<CdtBasinMemory>,
    pub rqm: Option<RqmIndex>,
    /// Tabla de respuesta del núcleo de chat por cue (B/C).
    pub core: Option<Vec<[f32; CORE_DIM]>>,
    pub readout: Readout,
    replay: Vec<SpiderExample>,
    replay_seen: u64,
    rng: Sm,
    pub novelty_max: f64,
    pub examples_seen: u64,
    pub sleeps: u64,
    /// Métricas hold-out del último entrenamiento (para la UI).
    pub holdout: Option<serde_json::Value>,
}

impl SpiderFieldRouter {
    pub fn new(cfg: FieldConfig) -> Self {
        let liquid = LiquidEncoder::new(cfg.n_liquid, cfg.seed);
        let enc_dim = if cfg.liquid {
            liquid.state_dim()
        } else {
            LOCAL_DIM + GLOBAL_DIM
        };
        Self {
            cdt: cfg.cdt.then(|| CdtBasinMemory::new(enc_dim, cfg.seed)),
            rqm: cfg.rqm.then(|| RqmIndex::new(enc_dim, cfg.seed)),
            liquid,
            core: None,
            readout: Readout::default(),
            replay: Vec::new(),
            replay_seen: 0,
            rng: Sm(cfg.seed ^ 0xBEEF),
            novelty_max: 1.0,
            examples_seen: 0,
            sleeps: 0,
            holdout: None,
            cfg,
        }
    }

    pub fn is_trained(&self) -> bool {
        self.readout.n_labeled.get(H_OK).copied().unwrap_or(0) > 0
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Episodios en vigilia pendientes de sueño.
    pub fn wake_len(&self) -> usize {
        self.cdt
            .as_ref()
            .map(|c| c.pending_len())
            .or_else(|| self.rqm.as_ref().map(|r| r.pending_len()))
            .unwrap_or(0)
    }

    /// Estado del encoder: líquido o rasgos crudos (centro + globales).
    pub fn encode_state(&self, ex: &SpiderExample) -> Vec<f32> {
        if self.cfg.liquid {
            self.liquid.run(ex)
        } else {
            let mut v = ex.center_local().to_vec();
            v.extend_from_slice(&ex.global);
            v
        }
    }

    /// Vector de lectura: salto crudo ⊕ estado ⊕ CDT ⊕ RQM ⊕ núcleo.
    fn features(&self, ex: &SpiderExample, state: &[f32]) -> (Vec<f32>, f32, Option<u32>) {
        let mut z = ex.center_local().to_vec();
        z.extend_from_slice(&ex.global);
        if self.cfg.liquid {
            z.extend_from_slice(state);
        }
        let mut novelty = 0f32;
        if let Some(c) = &self.cdt {
            let (sims, nov) = c.read(state);
            z.extend_from_slice(&sims);
            z.push(nov);
            novelty = nov;
        }
        let mut cell = None;
        if let Some(r) = &self.rqm {
            z.extend_from_slice(&r.read(state));
            cell = Some(r.cell(state));
        }
        if self.cfg.coupling != Coupling::Blank {
            match &self.core {
                Some(t) => z.extend_from_slice(&t[ex.core_cue % t.len()]),
                None => z.extend_from_slice(&[0f32; CORE_DIM]),
            }
        }
        (z, novelty, cell)
    }

    /// Vigilia: observa un ejemplo etiquetado (buffer CDT/RQM + repetición).
    pub fn observe(&mut self, ex: SpiderExample) {
        let state = self.encode_state(&ex);
        if let Some(c) = self.cdt.as_mut() {
            c.buffer(&state, &ex.labels);
        }
        if let Some(r) = self.rqm.as_mut() {
            r.buffer(&state, &ex.labels);
        }
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

    /// Sueño: CDT + RQM y reajuste de las cabezas sobre la repetición.
    pub fn sleep(&mut self) -> FieldSleepReport {
        let t0 = Instant::now();
        let mut rep = FieldSleepReport {
            episodes: self.wake_len(),
            ..Default::default()
        };
        if let Some(c) = self.cdt.as_mut() {
            c.consolidate();
            rep.cdt_classes = c.state.templates.iter().filter(|t| !t.is_empty()).count();
        }
        if let Some(r) = self.rqm.as_mut() {
            r.consolidate();
            rep.rqm_cells = r.state.cache.len();
        }
        self.refit();
        self.sleeps += 1;
        rep.replay = self.replay.len();
        rep.ms = t0.elapsed().as_secs_f64() * 1e3;
        rep
    }

    /// Reajusta las cabezas (calibración = ~20 % de los prompts, por grupo).
    pub fn refit(&mut self) {
        if self.replay.is_empty() {
            return;
        }
        let mut zs = Vec::with_capacity(self.replay.len());
        let mut labels = Vec::with_capacity(self.replay.len());
        let mut cal = Vec::with_capacity(self.replay.len());
        let mut novs = Vec::new();
        for ex in &self.replay {
            let s = self.encode_state(ex);
            let (z, nov, _) = self.features(ex, &s);
            zs.push(z);
            labels.push(ex.labels);
            cal.push(fnv(&format!("cal{}", ex.group)).is_multiple_of(5));
            novs.push(nov as f64);
        }
        if cal.iter().all(|c| *c) || cal.iter().all(|c| !*c) {
            cal.iter_mut().for_each(|c| *c = false);
        }
        self.readout = Readout::fit(&zs, &labels, &cal, self.cfg.epochs, self.cfg.l2);
        if self.cdt.is_some() && !novs.is_empty() {
            novs.sort_by(|a, b| a.total_cmp(b));
            let k = ((novs.len() as f64) * 0.995) as usize;
            self.novelty_max = novs[k.min(novs.len() - 1)] + 0.02;
        }
    }

    pub fn predict(&self, ex: &SpiderExample) -> FieldPrediction {
        let s = self.encode_state(ex);
        let (z, nov, cell) = self.features(ex, &s);
        FieldPrediction {
            probs: self.readout.probs(&z),
            novelty: nov as f64,
            cell,
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
        let ood = self.cdt.is_some() && pred.novelty > self.novelty_max;
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
                "campo · rel {:.2} · fund {:.2} · amb {:.2} · aprob {:.2} · ok {:.3} · novedad {:.3}",
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

    /// Instala la tabla de respuesta del núcleo de chat (B/C).
    pub fn set_core_table(&mut self, t: Vec<[f32; CORE_DIM]>) {
        self.core = Some(t);
    }

    pub fn to_file(&self) -> FieldModelFile {
        FieldModelFile {
            version: MODEL_VERSION,
            cfg: self.cfg,
            readout: self.readout.clone(),
            cdt: self.cdt.as_ref().map(|c| c.state.clone()),
            rqm: self.rqm.as_ref().map(|r| r.state.clone()),
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
        if let (Some(c), Some(st)) = (r.cdt.as_mut(), f.cdt) {
            c.restore(st);
        }
        if let (Some(q), Some(st)) = (r.rqm.as_mut(), f.rqm) {
            q.restore(st);
        }
        r.readout = f.readout;
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
        let enc = LiquidEncoder::new(16, 7);
        assert_eq!(enc.run(&a), enc.run(&b));
        assert_eq!(enc.run(&a).len(), 32);
    }

    #[test]
    fn untrained_router_never_resolves() {
        let r = SpiderFieldRouter::new(FieldConfig::blank(1));
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
        let mut cfg = FieldConfig::blank(3);
        cfg.n_liquid = 24;
        cfg.epochs = 60;
        let mut r = SpiderFieldRouter::new(cfg);
        let mut rng = Sm(9);
        for seed in 0..40 {
            let sp = synth_prompt(seed, &VOCAB_TRAIN);
            for ex in synth_examples(&sp, DEFAULT_THRESHOLD, 0.0, &mut rng) {
                r.observe(ex);
            }
        }
        let rep = r.sleep();
        assert!(rep.cdt_classes >= 8, "{rep:?}");
        assert!(rep.rqm_cells > 0);
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
