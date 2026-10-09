//! Prompt Spider: job asíncrono, cola de aprobación, persistencia y endpoints
//! `/api/spider/*` (protegidos por la sesión como el resto de `/api/*`).
//!
//! La lógica pura (tokenizador, scorer, router, parseo del JSON del LLM) está
//! en [`crate::prompt_spider`]; aquí solo se orquesta: el crawler camina el
//! prompt con un ritmo (`pace_ms`) para que la UI pueda animarlo, las palabras
//! dudosas se agrupan en lotes y un worker las escala al **LLM activo** de la
//! app (Gemma local o API OpenAI-compatible). Si el LLM falla, duda o pide
//! aprobación, la palabra queda **pendiente** para el humano.

use crate::field_gemma_probe::RawGemmaHandle;
use crate::prompt_spider::{
    clamp_threshold, compact_gemma_prompt, default_mode, gemma_choice_verdict, llm_system_prompt,
    llm_user_prompt, parse_llm_verdicts, resolve_profile, route, score_token, task_excerpt,
    tokenize, verdict_outcome, word_context, Assessment, EscalationItem, LlmKind, LlmVerdict,
    PromptContext, Question, Route, RunMode, RunProfile, Scores, Token, VerdictOutcome,
    DEFAULT_LLM_FLOOR, DEFAULT_THRESHOLD, SAMPLE_PROMPT,
};
use crate::web::api::SharedState;
use crate::web::llm_provider::{ChatMessage, ExternalGenConfig, OpenAiClient, ProviderConfig};
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Etiqueta de quien resuelve cuando decide el decoder del campo.
pub const FIELD_DECIDER: &str = "campo";

/// Máximo de caracteres del prompt.
pub const MAX_PROMPT_CHARS: usize = 20_000;
/// Eventos retenidos en memoria.
const MAX_EVENTS: usize = 4000;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn env_var(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

/// Perfil efectivo (defaults por proveedor + env) para un tipo de LLM.
pub fn profile_for(kind: LlmKind, mode: RunMode) -> RunProfile {
    resolve_profile(kind, mode, &env_var)
}

/// Directorio de corridas. Env `SPIDER_RUNS_DIR`; si no, `data/spider_runs`.
pub fn default_runs_dir() -> PathBuf {
    match std::env::var("SPIDER_RUNS_DIR") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p),
        _ => PathBuf::from("data/spider_runs"),
    }
}

/// Estado de una decisión.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Resuelta (código o LLM).
    Resolved,
    /// En cola del LLM.
    Escalating,
    /// Esperando al humano.
    Pending,
    Approved,
    Rejected,
}

/// Decisión de una palabra (fork).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    /// Índice del token.
    pub index: usize,
    pub word: String,
    pub question: Question,
    /// Confianza heurística (determinista).
    pub p: f64,
    pub scores: Scores,
    pub reasons: Vec<String>,
    /// Ruta inicial del router determinista.
    pub first_route: Route,
    /// Ruta final (`llm` resuelta por el LLM; `you` si quedó para el humano).
    pub route: Route,
    pub status: Status,
    /// Confianza final (la del LLM si resolvió).
    pub final_p: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<LlmVerdict>,
    /// Quién resolvió: `code`, etiqueta del LLM o `you`.
    pub resolved_by: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// No se escaló: se alcanzó el límite de la corrida corta.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub capped: bool,
    /// Lectura del decoder del campo (solo si decidió el campo).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<FieldReading>,
}

/// Lo que leyó el decoder del campo para una escalada.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FieldReading {
    /// `P(sí)` calibrada por cabeza: relevant, grounded, ambiguous,
    /// needs_approval, ok.
    pub probs: [f64; 5],
    pub p_ok: f64,
    pub novelty: f64,
    /// Latencia de la decisión (µs).
    pub micros: f64,
    pub resolved: bool,
    pub reason: String,
}

impl Decision {
    fn new(tok: &Token, a: Assessment, r: Route) -> Self {
        let (status, resolved_by, note) = match r {
            Route::Code => (Status::Resolved, "code".to_string(), String::new()),
            Route::Llm => (Status::Escalating, String::new(), String::new()),
            Route::You => (
                Status::Pending,
                String::new(),
                format!("requiere aprobación (p={:.3})", a.p),
            ),
        };
        Self {
            index: tok.index,
            word: tok.text.clone(),
            question: a.question,
            p: a.p,
            scores: a.scores,
            reasons: a.reasons,
            first_route: r,
            route: r,
            status,
            final_p: a.p,
            verdict: None,
            resolved_by,
            note,
            capped: false,
            field: None,
        }
    }
}

/// LLM que atendió las escalaciones.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LlmInfo {
    pub id: String,
    pub label: String,
    /// `local` | `openai_compatible`.
    pub kind: String,
    pub model: String,
    /// Motivo si no está disponible (p. ej. GGUF ausente).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable: Option<String>,
}

/// Registro de una llamada (lote) al LLM.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmCall {
    pub batch: usize,
    pub words: Vec<usize>,
    pub ok: bool,
    pub seconds: f64,
    pub verdicts: usize,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Quién atendió la llamada (etiqueta del LLM).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub llm: String,
    /// La llamada la atendió el respaldo (Gemma local).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fallback: bool,
}

/// Cambio a respaldo durante una corrida.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FallbackInfo {
    /// Etiqueta del LLM que falló.
    pub from: String,
    /// Etiqueta del respaldo.
    pub to: String,
    pub reason: String,
    /// Texto para la UI: «docker-llm no disponible → Gemma local».
    pub message: String,
}

/// Resumen numérico de una corrida.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SpiderSummary {
    pub tokens_total: usize,
    pub words_read: usize,
    pub forks_total: usize,
    pub forks: usize,
    pub code: usize,
    /// Palabras escaladas al LLM (incluye las que luego fueron al humano).
    pub llm_escalated: usize,
    /// Resueltas por el LLM.
    pub llm_resolved: usize,
    pub escalating: usize,
    /// Ruta final `you` (aprobación humana o duda del LLM).
    pub you: usize,
    pub pending: usize,
    pub approved: usize,
    pub rejected: usize,
    /// code + llm_resolved.
    pub auto_resolved: usize,
    pub avg_confidence: f64,
    pub coverage: f64,
    pub llm_calls: usize,
    pub llm_failures: usize,
    /// Palabras enviadas al LLM (en llamadas, incluidas las fallidas).
    #[serde(default)]
    pub llm_sent: usize,
    /// Escaladas no enviadas por el límite de la corrida corta.
    #[serde(default)]
    pub llm_capped: usize,
    /// Quién decidió las escaladas: `llm` | `campo`.
    #[serde(default)]
    pub decider: String,
    /// Escaladas que leyó el decoder del campo.
    #[serde(default)]
    pub field_escalated: usize,
    /// Resueltas por el decoder del campo (`resolved_by = campo`).
    #[serde(default)]
    pub field_resolved: usize,
    /// Latencia mediana del campo por escalada (µs).
    #[serde(default)]
    pub field_latency_us: f64,
}

/// Una corrida.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpiderRun {
    pub id: String,
    pub started_ms: u64,
    pub finished_ms: Option<u64>,
    /// `running` | `done` | `stopped`.
    pub state: String,
    pub prompt: String,
    pub threshold: f64,
    pub llm_floor: f64,
    pub batch_size: usize,
    pub pace_ms: u64,
    pub llm: LlmInfo,
    /// `corta` | `completa`.
    #[serde(default = "default_run_mode")]
    pub mode: RunMode,
    /// Perfil efectivo del LLM que atiende (cambia si entra el respaldo).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<RunProfile>,
    /// Respaldo activado (API externa caída → Gemma local).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<FallbackInfo>,
    pub tokens: Vec<Token>,
    /// Tokens leídos (cursor del crawler).
    pub cursor: usize,
    pub decisions: Vec<Decision>,
    pub llm_calls: Vec<LlmCall>,
    /// Quién decide las escaladas: `llm` (por defecto) | `campo`.
    #[serde(default = "default_decider")]
    pub decider: String,
    /// Modelo del decoder del campo al iniciar (si `decider = campo`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_model: Option<serde_json::Value>,
}

fn default_decider() -> String {
    "llm".into()
}

fn default_run_mode() -> RunMode {
    RunMode::Completa
}

impl SpiderRun {
    pub fn summary(&self) -> SpiderSummary {
        let d = &self.decisions;
        let c = |f: &dyn Fn(&Decision) -> bool| d.iter().filter(|x| f(x)).count();
        let code = c(&|x| x.route == Route::Code);
        let llm_resolved = c(&|x| x.route == Route::Llm && x.status == Status::Resolved);
        let forks = d.len();
        let avg = if forks == 0 {
            0.0
        } else {
            d.iter().map(|x| x.final_p).sum::<f64>() / forks as f64
        };
        let total = self.tokens.len();
        let mut lat: Vec<f64> = d
            .iter()
            .filter_map(|x| x.field.as_ref().map(|f| f.micros))
            .collect();
        lat.sort_by(|a, b| a.total_cmp(b));
        SpiderSummary {
            tokens_total: total,
            words_read: self.cursor.min(total),
            forks_total: self.tokens.iter().filter(|t| t.kind.is_fork()).count(),
            forks,
            code,
            llm_escalated: c(&|x| x.first_route == Route::Llm),
            llm_resolved,
            escalating: c(&|x| x.status == Status::Escalating),
            you: c(&|x| x.route == Route::You),
            pending: c(&|x| x.status == Status::Pending),
            approved: c(&|x| x.status == Status::Approved),
            rejected: c(&|x| x.status == Status::Rejected),
            auto_resolved: code + llm_resolved,
            avg_confidence: (avg * 1000.0).round() / 1000.0,
            coverage: if total == 0 {
                0.0
            } else {
                ((self.cursor.min(total) as f64 / total as f64) * 1000.0).round() / 1000.0
            },
            llm_calls: self.llm_calls.len(),
            llm_failures: self.llm_calls.iter().filter(|c| !c.ok).count(),
            llm_sent: self.llm_calls.iter().map(|c| c.words.len()).sum(),
            llm_capped: c(&|x| x.capped),
            decider: self.decider.clone(),
            field_escalated: lat.len(),
            field_resolved: c(&|x| x.resolved_by == FIELD_DECIDER && x.status == Status::Resolved),
            field_latency_us: if lat.is_empty() {
                0.0
            } else {
                (lat[lat.len() / 2] * 10.0).round() / 10.0
            },
        }
    }

    fn decision_mut(&mut self, index: usize) -> Option<&mut Decision> {
        self.decisions.iter_mut().find(|d| d.index == index)
    }
}

/// Evento para la UI (SSE / sondeo).
#[derive(Clone, Debug, Serialize)]
pub struct SpiderEvent {
    pub seq: u64,
    /// `start` | `read` | `decision` | `update` | `llm_progress` | `llm_call`
    /// | `fallback` | `done`.
    pub kind: String,
    pub run_id: String,
    pub cursor: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<Decision>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub message: String,
}

/// Estado del módulo dentro de `AppState`.
pub struct SpiderJobs {
    pub run: Option<SpiderRun>,
    pub running: bool,
    pub stop: Arc<AtomicBool>,
    pub events: Vec<SpiderEvent>,
    pub seq: u64,
    /// Directorio de persistencia (None = no persiste; tests).
    pub dir: Option<PathBuf>,
    /// Sustituto de Gemma local cuando no hay GGUF cargado (solo tests).
    #[doc(hidden)]
    pub local_stub: Option<LocalStub>,
}

/// LLM local sustituto (tests): recibe el prompt compacto renderizado y
/// devuelve `P(sí)` normalizada, como haría Gemma con sus logits.
pub type LocalStub = Arc<dyn Fn(&str) -> Result<f64, String> + Send + Sync>;

impl Default for SpiderJobs {
    fn default() -> Self {
        Self::new(None)
    }
}

impl SpiderJobs {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            run: None,
            running: false,
            stop: Arc::new(AtomicBool::new(false)),
            events: Vec::new(),
            seq: 0,
            dir,
            local_stub: None,
        }
    }

    fn push(&mut self, kind: &str, index: Option<usize>, decision: Option<Decision>, msg: String) {
        self.seq += 1;
        let (run_id, cursor) = self
            .run
            .as_ref()
            .map(|r| (r.id.clone(), r.cursor))
            .unwrap_or_default();
        self.events.push(SpiderEvent {
            seq: self.seq,
            kind: kind.into(),
            run_id,
            cursor,
            index,
            decision,
            message: msg,
        });
        if self.events.len() > MAX_EVENTS {
            let n = self.events.len() - MAX_EVENTS;
            self.events.drain(0..n);
        }
    }

    pub fn events_after(&self, after: u64) -> Vec<SpiderEvent> {
        self.events
            .iter()
            .filter(|e| e.seq > after)
            .take(500)
            .cloned()
            .collect()
    }

    /// Guarda la corrida actual (JSON con resumen). Errores solo se registran.
    pub fn save(&self) {
        let (Some(dir), Some(run)) = (&self.dir, &self.run) else {
            return;
        };
        if let Err(e) = save_run(dir, run) {
            tracing::warn!(error = %e, "prompt spider: no se pudo guardar la corrida");
        }
    }
}

fn export_value(run: &SpiderRun) -> serde_json::Value {
    json!({ "run": run, "summary": run.summary() })
}

fn save_run(dir: &Path, run: &SpiderRun) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.json", run.id));
    let tmp = dir.join(format!("{}.json.tmp", run.id));
    let txt = serde_json::to_string_pretty(&export_value(run)).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, txt).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

/// Un LLM concreto que puede atender escaladas.
#[derive(Clone)]
pub enum SpiderLlm {
    External(Box<ProviderConfig>),
    Gemma(RawGemmaHandle),
    /// Sustituto local (tests).
    Stub(LocalStub),
    Unavailable(String),
}

impl SpiderLlm {
    fn kind(&self) -> LlmKind {
        match self {
            Self::External(_) => LlmKind::External,
            _ => LlmKind::Local,
        }
    }
}

/// Backend de escalado (resuelto al iniciar la corrida): el LLM activo y,
/// si el activo es una API externa, Gemma local como respaldo (si está
/// cargado).
#[derive(Clone)]
pub struct SpiderBackend {
    pub primary: SpiderLlm,
    /// Nombre corto del proveedor (p. ej. `docker-llm`) para los avisos.
    pub name: String,
    pub info: LlmInfo,
    pub fallback: Option<(SpiderLlm, LlmInfo)>,
}

/// Respuesta de una llamada: (veredictos, modelo, segundos).
type CallResult = Result<(Vec<LlmVerdict>, String, f64), String>;

/// Opciones (primer token) para `P(sí)` / `P(no)` en el prompt compacto.
const YES_NO: [&[&str]; 2] = [&["sí", "si", "Sí"], &["no", "No"]];

/// Veredicto compacto de un LLM local: `P(sí)` con las dos órdenes de
/// opciones (`gen(prompt) → P(sí) normalizada`), `p = min` de ambas.
fn compact_choice(
    prompt: &str,
    items: &[EscalationItem],
    gen: &dyn Fn(&str) -> Result<f64, String>,
) -> Result<Vec<LlmVerdict>, String> {
    let task = task_excerpt(prompt, 120);
    items
        .iter()
        .map(|it| {
            let a = gen(&compact_gemma_prompt(&task, it, true))?;
            let b = gen(&compact_gemma_prompt(&task, it, false))?;
            Ok(gemma_choice_verdict(it.index, a, b))
        })
        .collect()
}

/// Llama al LLM (bloqueante) con el perfil dado y parsea los veredictos.
fn call_llm_blocking(
    llm: &SpiderLlm,
    prompt: &str,
    items: &[EscalationItem],
    profile: RunProfile,
) -> CallResult {
    let idx: Vec<usize> = items.iter().map(|i| i.index).collect();
    let timeout = Duration::from_secs(profile.timeout_secs);
    let t0 = Instant::now();
    match llm {
        SpiderLlm::External(cfg) => {
            let r = OpenAiClient::new(cfg, timeout).chat(
                &[
                    ChatMessage::new("system", llm_system_prompt()),
                    ChatMessage::new("user", llm_user_prompt(prompt, items)),
                ],
                ExternalGenConfig {
                    max_tokens: profile.max_tokens,
                    temperature: 0.0,
                    top_p: 1.0,
                },
            )?;
            parse_llm_verdicts(&r.text, &idx).map(|v| (v, r.model, r.seconds))
        }
        SpiderLlm::Gemma(h) if profile.compact => {
            let deadline = t0 + timeout;
            let gen = |rendered: &str| {
                if Instant::now() >= deadline {
                    return Err(format!("timeout ({}s)", profile.timeout_secs));
                }
                let (pr, _, _) = h.choice_probs(rendered, &YES_NO, 768, Some(deadline))?;
                let tot = pr[0] + pr[1];
                if tot < 1e-6 {
                    return Err("Gemma no puso masa en «sí»/«no»".into());
                }
                Ok(pr[0] / tot)
            };
            compact_choice(prompt, items, &gen)
                .map(|v| (v, "gemma2-gguf·P(sí)".into(), t0.elapsed().as_secs_f64()))
        }
        SpiderLlm::Gemma(h) => {
            let mut cfg = crate::web::llm_periphery::raw_chat_config(0x5_91DE);
            cfg.max_tokens = profile.max_tokens;
            cfg.temperature = 0.0;
            cfg.top_p = 1.0;
            let r = h.generate_chat(
                &[],
                &format!(
                    "{}\n\n{}",
                    llm_system_prompt(),
                    llm_user_prompt(prompt, items)
                ),
                cfg,
                Some(t0 + timeout),
            )?;
            parse_llm_verdicts(&r.text, &idx).map(|v| (v, "gemma2-gguf".into(), r.seconds))
        }
        SpiderLlm::Stub(f) => compact_choice(prompt, items, &|r: &str| f(r))
            .map(|v| (v, "stub-local".into(), t0.elapsed().as_secs_f64())),
        SpiderLlm::Unavailable(why) => Err(format!("LLM no disponible: {why}")),
    }
}

/// ¿El error indica que la API externa no está disponible (y conviene el
/// respaldo local)? Conexión, timeout, 5xx, 401/403/404, sin modelo. Un JSON
/// malo **no** cuenta: la API respondió.
pub fn is_unavailable_error(e: &str) -> bool {
    let l = e.to_lowercase();
    [
        "no se pudo conectar",
        "tiempo de espera",
        "timeout",
        "timed out",
        "401",
        "403",
        "404 no encontrado",
        "http 5",
        "no disponible",
        "no lista ningún modelo",
        "connection",
        "tarea falló",
    ]
    .iter()
    .any(|k| l.contains(k))
}

fn local_info(g: &crate::web::AppState) -> LlmInfo {
    LlmInfo {
        id: crate::web::llm_provider::LOCAL_ID.into(),
        label: crate::web::state::LOCAL_LABEL.into(),
        kind: "local".into(),
        model: g.probe.name().into(),
        unavailable: None,
    }
}

/// Gemma local (o el sustituto de tests), si está cargado.
fn local_llm(g: &crate::web::AppState) -> Option<SpiderLlm> {
    g.probe
        .raw_handle()
        .map(SpiderLlm::Gemma)
        .or_else(|| g.spider.local_stub.clone().map(SpiderLlm::Stub))
}

fn llm_info_and_backend(g: &crate::web::AppState) -> SpiderBackend {
    let local = local_llm(g);
    match g.active_external() {
        Some(p) => SpiderBackend {
            name: p.name.clone(),
            info: LlmInfo {
                id: p.id.clone(),
                label: p.label(),
                kind: "openai_compatible".into(),
                model: p.model.clone(),
                unavailable: None,
            },
            primary: SpiderLlm::External(Box::new(p)),
            fallback: local.map(|l| {
                let mut info = local_info(g);
                info.label = format!("{} (respaldo)", crate::web::state::LOCAL_LABEL);
                (l, info)
            }),
        },
        None => {
            let mut info = local_info(g);
            match local {
                Some(l) => SpiderBackend {
                    name: info.label.clone(),
                    primary: l,
                    info,
                    fallback: None,
                },
                None => {
                    let why = g.model_unavailable_reason();
                    info.unavailable = Some(why.clone());
                    SpiderBackend {
                        name: info.label.clone(),
                        primary: SpiderLlm::Unavailable(why),
                        info,
                        fallback: None,
                    }
                }
            }
        }
    }
}

fn lock(st: &SharedState) -> std::sync::MutexGuard<'_, crate::web::AppState> {
    st.lock().unwrap_or_else(|e| e.into_inner())
}

/// Pasa palabras en escalado a pendiente con una nota (falla del LLM,
/// detenido, límite de corrida corta…). `capped` marca el límite.
fn mark_pending(st: &SharedState, items: &[usize], note: &str, capped: bool) {
    let mut g = lock(st);
    let mut changed = Vec::new();
    if let Some(run) = g.spider.run.as_mut() {
        for &i in items {
            if let Some(d) = run.decision_mut(i) {
                if d.status == Status::Escalating {
                    d.route = Route::You;
                    d.status = Status::Pending;
                    d.note = note.to_string();
                    d.capped = capped;
                    changed.push(d.clone());
                }
            }
        }
    }
    for d in changed {
        g.spider
            .push("update", Some(d.index), Some(d), String::new());
    }
}

/// Aplica los veredictos de una llamada correcta a las decisiones.
fn apply_verdicts(
    run: &mut SpiderRun,
    idx: &[usize],
    verdicts: &[LlmVerdict],
    floor: f64,
    label: &str,
) -> Vec<Decision> {
    let mut updates = Vec::new();
    for &i in idx {
        let Some(d) = run.decision_mut(i) else {
            continue;
        };
        match verdicts.iter().find(|v| v.index == i) {
            Some(v) => {
                d.verdict = Some(v.clone());
                d.final_p = v.p;
                match verdict_outcome(v, floor) {
                    VerdictOutcome::Resolved => {
                        d.route = Route::Llm;
                        d.status = Status::Resolved;
                        d.resolved_by = label.to_string();
                        d.note = v.note.clone();
                    }
                    VerdictOutcome::Pending => {
                        d.route = Route::You;
                        d.status = Status::Pending;
                        d.note = if v.needs_approval {
                            format!("{label} pide aprobación humana")
                        } else if v.ambiguous {
                            format!("{label} la marca ambigua (p={:.3})", v.p)
                        } else {
                            format!("{label} duda (p={:.3} < {floor:.2})", v.p)
                        };
                    }
                }
            }
            None => {
                d.route = Route::You;
                d.status = Status::Pending;
                d.note = format!("{label} no devolvió veredicto: pendiente");
            }
        }
        updates.push(d.clone());
    }
    updates
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Ejecuta una llamada en `spawn_blocking` con plazo externo. Gemma corta la
/// generación en su propio plazo; el margen extra cubre el prefill (no
/// interrumpible) para no soltar el lock del modelo a medias.
async fn run_call(
    llm: &SpiderLlm,
    prompt: &str,
    items: &[EscalationItem],
    profile: RunProfile,
) -> CallResult {
    let slack = match llm.kind() {
        LlmKind::Local => Duration::from_secs(20),
        LlmKind::External => Duration::from_secs(2),
    };
    let timeout = Duration::from_secs(profile.timeout_secs);
    let (l, p, it) = (llm.clone(), prompt.to_string(), items.to_vec());
    let call = tokio::task::spawn_blocking(move || call_llm_blocking(&l, &p, &it, profile));
    match tokio::time::timeout(timeout + slack, call).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => Err(format!("tarea falló: {e}")),
        Err(_) => Err(format!("timeout ({}s)", profile.timeout_secs)),
    }
}

/// Maestro LLM para «Entrenamiento Prompt Spider»: consulta el LLM activo
/// (y su respaldo si la API externa no responde) sobre `items` de `prompt`.
/// Devuelve (veredictos, etiqueta del LLM, compacto = solo `P(sí)`).
pub(crate) async fn teacher_call(
    st: &SharedState,
    prompt: &str,
    items: &[EscalationItem],
) -> Result<(Vec<LlmVerdict>, String, bool), String> {
    let backend = {
        let g = lock(st);
        llm_info_and_backend(&g)
    };
    let mut tries = vec![(backend.primary.clone(), backend.info.label.clone())];
    if let Some((fb, info)) = backend.fallback.clone() {
        tries.push((fb, info.label));
    }
    let mut last = String::from("sin LLM");
    for (llm, label) in tries {
        let profile = profile_for(llm.kind(), RunMode::Completa);
        let compact = llm.kind() == LlmKind::Local;
        match run_call(&llm, prompt, items, profile).await {
            Ok((v, _, _)) => return Ok((v, label, compact)),
            Err(e) => last = e,
        }
    }
    Err(last)
}

/// Aplica la decisión del decoder del campo a una escalada: resuelve con
/// `resolved_by = campo` o la deja pendiente con el motivo (no inventa).
fn apply_field_decision(d: &mut Decision, fd: &crate::spider_field::FieldDecision) {
    d.verdict = Some(fd.verdict.clone());
    d.final_p = fd.verdict.p;
    d.field = Some(FieldReading {
        probs: fd.prediction.probs.map(|x| (x * 1000.0).round() / 1000.0),
        p_ok: fd.verdict.p,
        novelty: (fd.prediction.novelty * 1000.0).round() / 1000.0,
        micros: (fd.micros * 10.0).round() / 10.0,
        resolved: fd.resolve,
        reason: fd.reason.clone(),
    });
    if fd.resolve {
        d.route = Route::Llm;
        d.status = Status::Resolved;
        d.resolved_by = FIELD_DECIDER.into();
        d.note = format!("campo: {}", fd.reason);
    } else {
        d.route = Route::You;
        d.status = Status::Pending;
        d.note = format!("campo: {}; pendiente (no se inventa)", fd.reason);
    }
}

/// Worker de escaladas: agrupa según el perfil del LLM que atiende, aplica
/// el límite de la corrida corta, emite progreso por llamada y, si la API
/// externa no está disponible, cambia a Gemma local (respaldo etiquetado)
/// para el resto de la corrida. Fallos → pendiente con motivo (nunca inventa).
#[allow(clippy::too_many_arguments)]
async fn llm_worker(
    st: SharedState,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EscalationItem>,
    backend: SpiderBackend,
    mode: RunMode,
    mode_auto: bool,
    profile: RunProfile,
    prompt: String,
    floor: f64,
    stop: Arc<AtomicBool>,
) {
    let mut llm = backend.primary.clone();
    let mut label = backend.info.label.clone();
    let mut profile = profile;
    let mut fallback = backend.fallback.clone();
    let mut on_fallback = false;
    let mut queue: VecDeque<EscalationItem> = VecDeque::new();
    let mut closed = false;
    let mut call_no = 0usize;
    let mut sent = 0usize;
    let mut consecutive_failures = 0usize;
    let mut gave_up: Option<String> = None;
    loop {
        // Llenar la cola hasta un lote (o hasta que el crawler termine).
        while !closed && queue.len() < profile.batch_size {
            match rx.recv().await {
                Some(it) => queue.push_back(it),
                None => closed = true,
            }
            while let Ok(it) = rx.try_recv() {
                queue.push_back(it);
            }
        }
        if queue.is_empty() {
            if closed {
                break;
            }
            continue;
        }
        let room = profile
            .max_escalations
            .map(|c| c.saturating_sub(sent))
            .unwrap_or(usize::MAX);
        if room == 0 && !stop.load(Ordering::Relaxed) && gave_up.is_none() {
            let cap = profile.max_escalations.unwrap_or(0);
            let over: Vec<usize> = queue.drain(..).map(|i| i.index).collect();
            mark_pending(
                &st,
                &over,
                &format!(
                    "límite de corrida corta ({cap} escaladas al LLM); pendiente sin consultar"
                ),
                true,
            );
            continue;
        }
        let take = profile.batch_size.min(queue.len()).min(room.max(1));
        let items: Vec<EscalationItem> = queue.drain(..take).collect();
        let idx: Vec<usize> = items.iter().map(|i| i.index).collect();
        if stop.load(Ordering::Relaxed) {
            mark_pending(&st, &idx, "detenido antes del veredicto del LLM", false);
            continue;
        }
        if let Some(why) = &gave_up {
            mark_pending(&st, &idx, why, false);
            continue;
        }
        call_no += 1;
        sent += idx.len();
        {
            let words: Vec<String> = items.iter().map(|i| format!("«{}»", i.word)).collect();
            let cap = profile
                .max_escalations
                .map(|c| format!("{sent}/{c}"))
                .unwrap_or_else(|| format!("{sent}"));
            let mut g = lock(&st);
            g.spider.push(
                "llm_progress",
                idx.first().copied(),
                None,
                format!(
                    "llamada {call_no} → {label}: {} · escaladas {cap} · plazo {}s",
                    words.join(" "),
                    profile.timeout_secs
                ),
            );
        }
        let t0 = Instant::now();
        let res = run_call(&llm, &prompt, &items, profile).await;
        // API externa caída → respaldo Gemma local para esta y las siguientes.
        if let Err(e) = &res {
            if !on_fallback && is_unavailable_error(e) {
                if let Some((fb, fb_info)) = fallback.take() {
                    let e: String = e.chars().take(300).collect();
                    let from = backend.info.label.clone();
                    let message = format!(
                        "{} no disponible → {}",
                        backend.name,
                        crate::web::state::LOCAL_LABEL
                    );
                    let mut g = lock(&st);
                    if let Some(run) = g.spider.run.as_mut() {
                        run.llm_calls.push(LlmCall {
                            batch: call_no,
                            words: idx.clone(),
                            ok: false,
                            seconds: r2(t0.elapsed().as_secs_f64()),
                            verdicts: 0,
                            model: String::new(),
                            error: Some(e.clone()),
                            llm: label.clone(),
                            fallback: false,
                        });
                        run.fallback = Some(FallbackInfo {
                            from,
                            to: fb_info.label.clone(),
                            reason: e.clone(),
                            message: message.clone(),
                        });
                    }
                    llm = fb;
                    label = fb_info.label;
                    on_fallback = true;
                    // Modo auto: con Gemma local, corrida corta.
                    let fb_mode = if mode_auto {
                        default_mode(llm.kind())
                    } else {
                        mode
                    };
                    profile = profile_for(llm.kind(), fb_mode);
                    if let Some(run) = g.spider.run.as_mut() {
                        run.mode = fb_mode;
                        run.profile = Some(profile);
                        run.batch_size = profile.batch_size;
                    }
                    g.spider
                        .push("fallback", None, None, format!("{message} ({e})"));
                    drop(g);
                    // Reintenta estas palabras con el respaldo (lotes nuevos).
                    sent = 0;
                    for it in items.into_iter().rev() {
                        queue.push_front(it);
                    }
                    continue;
                }
            }
        }
        let mut g = lock(&st);
        let mut updates = Vec::new();
        let call_log = match res {
            Ok((verdicts, model, secs)) => {
                consecutive_failures = 0;
                if let Some(run) = g.spider.run.as_mut() {
                    updates = apply_verdicts(run, &idx, &verdicts, floor, &label);
                }
                LlmCall {
                    batch: call_no,
                    words: idx.clone(),
                    ok: true,
                    seconds: r2(secs),
                    verdicts: verdicts.len(),
                    model,
                    error: None,
                    llm: label.clone(),
                    fallback: on_fallback,
                }
            }
            Err(e) => {
                let e: String = e.chars().take(300).collect();
                consecutive_failures += 1;
                if let Some(run) = g.spider.run.as_mut() {
                    for &i in &idx {
                        if let Some(d) = run.decision_mut(i) {
                            d.route = Route::You;
                            d.status = Status::Pending;
                            d.note = format!("{label} falló; pendiente (no se inventa): {e}");
                            updates.push(d.clone());
                        }
                    }
                }
                LlmCall {
                    batch: call_no,
                    words: idx.clone(),
                    ok: false,
                    seconds: r2(t0.elapsed().as_secs_f64()),
                    verdicts: 0,
                    model: String::new(),
                    error: Some(e),
                    llm: label.clone(),
                    fallback: on_fallback,
                }
            }
        };
        if consecutive_failures >= profile.max_consecutive_failures && gave_up.is_none() {
            gave_up = Some(format!(
                "{label} falló {consecutive_failures} veces seguidas; pendiente sin consultar (no se inventa)"
            ));
        }
        let msg = if call_log.ok {
            format!(
                "llamada {} → {}: {} palabra(s), {} veredicto(s), {:.2}s",
                call_log.batch,
                label,
                call_log.words.len(),
                call_log.verdicts,
                call_log.seconds
            )
        } else {
            format!(
                "llamada {} → {} falló ({:.1}s): {}",
                call_log.batch,
                label,
                call_log.seconds,
                call_log.error.clone().unwrap_or_default()
            )
        };
        if let Some(run) = g.spider.run.as_mut() {
            run.llm_calls.push(call_log);
        }
        for d in updates {
            g.spider
                .push("update", Some(d.index), Some(d), String::new());
        }
        g.spider.push("llm_call", None, None, msg);
    }
}

/// Camina el prompt, decide cada palabra y pasa las dudosas al worker.
async fn run_spider(
    st: SharedState,
    stop: Arc<AtomicBool>,
    backend: SpiderBackend,
    profile: RunProfile,
    mode_auto: bool,
    field: Option<Arc<crate::spider_field::SpiderFieldRouter>>,
) {
    let (tokens, prompt, pace, floor, threshold, mode) = {
        let g = lock(&st);
        let r = g.spider.run.as_ref().expect("run");
        (
            r.tokens.clone(),
            r.prompt.clone(),
            r.pace_ms,
            r.llm_floor,
            r.threshold,
            r.mode,
        )
    };
    let ctx = PromptContext::new(&tokens);
    // Decoder del campo: el prompt se analiza una vez (mismas rutas que el
    // scorer) y cada escalada se codifica por su posición.
    let analyzed = field.as_ref().map(|_| {
        crate::spider_field::AnalyzedPrompt::from_tokens(&prompt, tokens.clone(), threshold)
    });
    let group = crate::spider_field::text_group(&prompt);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<EscalationItem>();
    let worker = tokio::spawn(llm_worker(
        st.clone(),
        rx,
        backend,
        mode,
        mode_auto,
        profile,
        prompt,
        floor,
        stop.clone(),
    ));
    for (i, tok) in tokens.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        {
            let mut g = lock(&st);
            if let Some(run) = g.spider.run.as_mut() {
                run.cursor = i + 1;
            }
            if tok.kind.is_fork() {
                let a = score_token(tok, &ctx, i.checked_sub(1).map(|j| &tokens[j]));
                let r = route(&a, threshold);
                let mut d = Decision::new(tok, a, r);
                if let (Route::Llm, Some(router), Some(ap)) = (r, &field, &analyzed) {
                    let ex = ap.encode(
                        i,
                        [None; crate::spider_field::HEADS],
                        crate::spider_field::LabelSource::Teacher,
                        group,
                    );
                    match ex {
                        Some(ex) => apply_field_decision(&mut d, &router.decide(&ex, floor)),
                        None => {
                            d.route = Route::You;
                            d.status = Status::Pending;
                            d.note = "campo: no se pudo codificar; pendiente".into();
                        }
                    }
                } else if r == Route::Llm {
                    let _ = tx.send(EscalationItem {
                        index: tok.index,
                        word: tok.text.clone(),
                        context: word_context(&tokens, i, 6),
                        question: d.question,
                        p: d.p,
                    });
                }
                if let Some(run) = g.spider.run.as_mut() {
                    run.decisions.push(d.clone());
                }
                g.spider
                    .push("decision", Some(tok.index), Some(d), String::new());
            } else {
                g.spider.push("read", Some(tok.index), None, String::new());
            }
        }
        if pace > 0 {
            tokio::time::sleep(Duration::from_millis(pace)).await;
        }
    }
    drop(tx);
    let _ = worker.await;
    let stopped = stop.load(Ordering::Relaxed);
    let pending_ids: Vec<usize> = {
        let g = lock(&st);
        g.spider
            .run
            .as_ref()
            .map(|r| {
                r.decisions
                    .iter()
                    .filter(|d| d.status == Status::Escalating)
                    .map(|d| d.index)
                    .collect()
            })
            .unwrap_or_default()
    };
    mark_pending(
        &st,
        &pending_ids,
        "sin veredicto del LLM (corrida detenida)",
        false,
    );
    let mut g = lock(&st);
    let msg = if let Some(run) = g.spider.run.as_mut() {
        run.state = if stopped { "stopped" } else { "done" }.into();
        run.finished_ms = Some(now_ms());
        let s = run.summary();
        let who = match &run.fallback {
            Some(f) => format!("{} [{}]", f.to, f.message),
            None => run.llm.label.clone(),
        };
        let field_txt = if run.decider == FIELD_DECIDER {
            format!(
                " · campo {}/{} resueltas · mediana {:.0} µs",
                s.field_resolved, s.field_escalated, s.field_latency_us
            )
        } else {
            String::new()
        };
        let capped = if s.llm_capped > 0 {
            format!(" · {} por límite de corrida corta", s.llm_capped)
        } else {
            String::new()
        };
        format!(
            "{} ({}): {} forks · code {} · {} {} de {} enviadas ({} escaladas{}) · you {} · p̄ {:.3}{}",
            run.state,
            run.mode.as_str(),
            s.forks,
            s.code,
            who,
            s.llm_resolved,
            s.llm_sent,
            s.llm_escalated,
            capped,
            s.you,
            s.avg_confidence,
            field_txt
        )
    } else {
        String::new()
    };
    g.spider.running = false;
    g.spider.push("done", None, None, msg);
    g.spider.save();
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/// Rutas `/api/spider/*` (se fusionan dentro del router protegido).
pub fn routes() -> Router<SharedState> {
    Router::new()
        .route("/api/spider/sample", get(sample))
        .route("/api/spider/start", post(start))
        .route("/api/spider/stop", post(stop_run))
        .route("/api/spider/status", get(status))
        .route("/api/spider/events", get(events))
        .route("/api/spider/stream", get(stream_events))
        .route("/api/spider/decide", post(decide))
        .route("/api/spider/runs", get(runs))
        .route("/api/spider/export", get(export))
}

async fn sample() -> Json<serde_json::Value> {
    Json(json!({
        "prompt": SAMPLE_PROMPT,
        "threshold": DEFAULT_THRESHOLD,
        "llm_floor": DEFAULT_LLM_FLOOR,
    }))
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct StartRequest {
    pub prompt: Option<String>,
    pub threshold: Option<f64>,
    pub llm_floor: Option<f64>,
    pub pace_ms: Option<u64>,
    pub batch_size: Option<usize>,
    /// `corta` | `completa` (por defecto: corta con Gemma local, completa con API).
    pub mode: Option<String>,
    /// Override del límite de escaladas (solo `corta`; 0 = sin límite).
    pub max_escalations: Option<usize>,
    /// «Decoder del campo»: las escaladas las decide el campo entrenado en
    /// vez del LLM (por debajo de la confianza → pendiente).
    pub field_decoder: Option<bool>,
}

fn bad(status: StatusCode, msg: &str) -> axum::response::Response {
    (status, Json(json!({ "ok": false, "error": msg }))).into_response()
}

async fn start(
    State(st): State<SharedState>,
    Json(b): Json<StartRequest>,
) -> axum::response::Response {
    let prompt = b.prompt.unwrap_or_else(|| SAMPLE_PROMPT.to_string());
    if prompt.trim().is_empty() {
        return bad(StatusCode::BAD_REQUEST, "prompt vacío");
    }
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return bad(
            StatusCode::BAD_REQUEST,
            "prompt demasiado largo (máx 20000 caracteres)",
        );
    }
    let tokens = tokenize(&prompt);
    if !tokens.iter().any(|t| t.kind.is_fork()) {
        return bad(StatusCode::BAD_REQUEST, "el prompt no tiene palabras");
    }
    let threshold = clamp_threshold(b.threshold);
    let floor = b
        .llm_floor
        .filter(|x| x.is_finite())
        .map(|x| x.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_LLM_FLOOR);
    let pace = b.pace_ms.unwrap_or(60).min(2000);
    let mut g = lock(&st);
    if g.spider.running {
        return bad(StatusCode::CONFLICT, "ya hay una corrida en curso");
    }
    let use_field = b.field_decoder.unwrap_or(false);
    let backend = llm_info_and_backend(&g);
    let field = use_field.then(|| Arc::new(g.spider_field.router.clone()));
    let field_model = field.as_ref().map(|r| g.spider_field.model_summary(r));
    let info = if use_field {
        LlmInfo {
            id: FIELD_DECIDER.into(),
            label: FIELD_DECIDER.into(),
            kind: "field".into(),
            model: crate::web::spider_field_job::model_name(&g.spider_field.router),
            unavailable: (!g.spider_field.router.is_trained())
                .then(|| "decoder del campo sin entrenar: las escaladas quedan pendientes".into()),
        }
    } else {
        backend.info.clone()
    };
    let kind = backend.primary.kind();
    let mode_auto = matches!(
        b.mode.as_deref().map(str::trim),
        None | Some("") | Some("auto")
    );
    let mode = match b.mode.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        None | Some("auto") => default_mode(kind),
        Some(m) => match RunMode::parse(m) {
            Some(m) => m,
            None => return bad(StatusCode::BAD_REQUEST, "modo inválido (corta | completa)"),
        },
    };
    let mut profile = profile_for(kind, mode);
    if let Some(bs) = b.batch_size {
        profile.batch_size = bs.clamp(1, 64);
    }
    if let (RunMode::Corta, Some(n)) = (mode, b.max_escalations) {
        profile.max_escalations = (n > 0).then_some(n.min(10_000));
    }
    let batch_size = profile.batch_size;
    let id = uuid::Uuid::new_v4().to_string();
    let forks = tokens.iter().filter(|t| t.kind.is_fork()).count();
    let total = tokens.len();
    let run = SpiderRun {
        id: id.clone(),
        started_ms: now_ms(),
        finished_ms: None,
        state: "running".into(),
        prompt,
        threshold,
        llm_floor: floor,
        batch_size,
        pace_ms: pace,
        llm: info.clone(),
        mode,
        profile: Some(profile),
        fallback: None,
        tokens,
        cursor: 0,
        decisions: Vec::new(),
        llm_calls: Vec::new(),
        decider: if use_field { FIELD_DECIDER } else { "llm" }.into(),
        field_model,
    };
    let stop = Arc::new(AtomicBool::new(false));
    g.spider.stop = stop.clone();
    g.spider.running = true;
    g.spider.run = Some(run);
    g.spider.events.clear();
    let fb = backend
        .fallback
        .as_ref()
        .map(|(_, i)| format!(" · respaldo {}", i.label))
        .unwrap_or_default();
    let cap = profile
        .max_escalations
        .map(|c| format!("máx {c} escaladas"))
        .unwrap_or_else(|| "sin límite".into());
    let msg = if use_field {
        format!(
            "corrida {} {id} · umbral {threshold:.2} · escaladas → decoder del campo ({}){}",
            mode.as_str(),
            info.model,
            info.unavailable
                .as_ref()
                .map(|u| format!(" · {u}"))
                .unwrap_or_default()
        )
    } else {
        format!(
            "corrida {} {id} · umbral {threshold:.2} · LLM {}{fb} · lotes de {} · plazo {}s · {cap}",
            mode.as_str(),
            info.label,
            profile.batch_size,
            profile.timeout_secs
        )
    };
    g.spider.push("start", None, None, msg);
    let seq = g.spider.seq;
    drop(g);
    let has_fallback = backend.fallback.is_some();
    // El worker arranca con el perfil resuelto aquí (con overrides de la
    // petición); si entra el respaldo, usa el perfil de Gemma local.
    tokio::spawn(run_spider(
        st.clone(),
        stop,
        backend,
        profile,
        mode_auto,
        field,
    ));
    Json(json!({
        "ok": true, "run_id": id, "threshold": threshold, "llm_floor": floor,
        "tokens_total": total, "forks_total": forks, "llm": info, "seq": seq,
        "mode": mode, "profile": profile, "fallback_available": has_fallback,
        "decider": if use_field { FIELD_DECIDER } else { "llm" },
    }))
    .into_response()
}

async fn stop_run(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let g = lock(&st);
    g.spider.stop.store(true, Ordering::Relaxed);
    Json(json!({ "ok": true, "running": g.spider.running }))
}

async fn status(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let g = lock(&st);
    let backend = llm_info_and_backend(&g);
    let kind = backend.primary.kind();
    Json(json!({
        "running": g.spider.running,
        "seq": g.spider.seq,
        "run": g.spider.run,
        "summary": g.spider.run.as_ref().map(SpiderRun::summary),
        "llm_active": backend.info,
        "llm_fallback": backend.fallback.as_ref().map(|(_, i)| i),
        "field": g.spider_field.status_value(),
        "defaults": {
            "threshold": DEFAULT_THRESHOLD,
            "llm_floor": DEFAULT_LLM_FLOOR,
            "mode": default_mode(kind),
            "corta": profile_for(kind, RunMode::Corta),
            "completa": profile_for(kind, RunMode::Completa),
        },
    }))
}

#[derive(Deserialize)]
pub struct AfterQuery {
    pub after: Option<u64>,
}

async fn events(
    State(st): State<SharedState>,
    Query(q): Query<AfterQuery>,
) -> Json<serde_json::Value> {
    let g = lock(&st);
    let ev = g.spider.events_after(q.after.unwrap_or(0));
    Json(json!({
        "events": ev,
        "seq": g.spider.seq,
        "running": g.spider.running,
        "summary": g.spider.run.as_ref().map(SpiderRun::summary),
    }))
}

/// SSE (mismo patrón que `/api/train/stream`): evento `spider` con un array
/// de eventos; cierra cuando la corrida terminó y no quedan eventos.
async fn stream_events(
    State(st): State<SharedState>,
    Query(q): Query<AfterQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let after0 = q.after.unwrap_or(0);
    let s = stream::unfold((st, after0, false), |(st, mut after, closed)| async move {
        if closed {
            return None;
        }
        loop {
            let (evs, running, seq) = {
                let g = lock(&st);
                (g.spider.events_after(after), g.spider.running, g.spider.seq)
            };
            if !evs.is_empty() {
                after = evs.last().map(|e| e.seq).unwrap_or(after);
                let payload = serde_json::to_string(&evs).unwrap_or_else(|_| "[]".into());
                return Some((
                    Ok(Event::default().event("spider").data(payload)),
                    (st, after, false),
                ));
            }
            if !running && after >= seq {
                let ev = Event::default().event("idle").data("{}");
                return Some((Ok(ev), (st, after, true)));
            }
            tokio::time::sleep(Duration::from_millis(120)).await;
        }
    });
    Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

#[derive(Deserialize)]
pub struct DecideRequest {
    pub run_id: Option<String>,
    pub index: usize,
    pub approve: bool,
}

async fn decide(
    State(st): State<SharedState>,
    Json(b): Json<DecideRequest>,
) -> axum::response::Response {
    let mut g = lock(&st);
    let Some(run) = g.spider.run.as_mut() else {
        return bad(StatusCode::NOT_FOUND, "no hay corrida");
    };
    if b.run_id.as_deref().is_some_and(|id| id != run.id) {
        return bad(StatusCode::CONFLICT, "la corrida ya no es la actual");
    }
    let Some(d) = run.decision_mut(b.index) else {
        return bad(StatusCode::NOT_FOUND, "palabra no encontrada");
    };
    if d.status != Status::Pending {
        return bad(StatusCode::CONFLICT, "la palabra no está pendiente");
    }
    d.status = if b.approve {
        Status::Approved
    } else {
        Status::Rejected
    };
    d.resolved_by = "you".into();
    let d = d.clone();
    let summary = run.summary();
    // Vigilia del decoder del campo: la decisión humana queda como episodio
    // etiquetado; se consolida en el siguiente sueño (Sueño o entrenamiento).
    let example = crate::web::spider_field_job::user_example(run, &d, b.approve);
    if let Some(ex) = example {
        g.spider_field.observe_online(ex);
    }
    let word = d.word.clone();
    g.spider.push(
        "update",
        Some(d.index),
        Some(d.clone()),
        format!(
            "«{word}» {}",
            if b.approve { "aprobada" } else { "rechazada" }
        ),
    );
    if !g.spider.running {
        g.spider.save();
    }
    Json(json!({ "ok": true, "decision": d, "summary": summary })).into_response()
}

async fn runs(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let dir = lock(&st).spider.dir.clone();
    let mut list = Vec::new();
    if let Some(dir) = dir {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                let Ok(txt) = std::fs::read_to_string(&p) else {
                    continue;
                };
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
                    continue;
                };
                list.push(json!({
                    "id": v.pointer("/run/id"),
                    "started_ms": v.pointer("/run/started_ms"),
                    "state": v.pointer("/run/state"),
                    "llm": v.pointer("/run/llm/label"),
                    "threshold": v.pointer("/run/threshold"),
                    "summary": v.get("summary"),
                }));
            }
        }
    }
    list.sort_by_key(|v| std::cmp::Reverse(v["started_ms"].as_u64().unwrap_or(0)));
    list.truncate(50);
    Json(json!({ "runs": list }))
}

#[derive(Deserialize)]
pub struct ExportQuery {
    pub id: Option<String>,
}

async fn export(
    State(st): State<SharedState>,
    Query(q): Query<ExportQuery>,
) -> axum::response::Response {
    let (cur, dir) = {
        let g = lock(&st);
        (g.spider.run.clone(), g.spider.dir.clone())
    };
    let value = match (&q.id, cur) {
        (None, Some(r)) => export_value(&r),
        (Some(id), Some(r)) if *id == r.id => export_value(&r),
        (Some(id), _) if valid_id(id) => {
            let Some(dir) = dir else {
                return bad(StatusCode::NOT_FOUND, "corrida no encontrada");
            };
            match std::fs::read_to_string(dir.join(format!("{id}.json")))
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            {
                Some(v) => v,
                None => return bad(StatusCode::NOT_FOUND, "corrida no encontrada"),
            }
        }
        (Some(_), _) => return bad(StatusCode::BAD_REQUEST, "id inválido"),
        (None, None) => return bad(StatusCode::NOT_FOUND, "no hay corrida"),
    };
    let id = value
        .pointer("/run/id")
        .and_then(|v| v.as_str())
        .unwrap_or("run")
        .to_string();
    (
        [
            (header::CONTENT_TYPE, "application/json".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"prompt_spider_{id}.json\""),
            ),
        ],
        serde_json::to_string_pretty(&value).unwrap_or_default(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Etiquetas de maestro con Gemma local (P(sí) en ambas órdenes) para el
    /// experimento del decoder del campo: escaladas del prompt de ejemplo +
    /// escaladas del hold-out H1. Ignorada en CI.
    /// `GEMMA2_GGUF=… N=60 OUT=docs/data/spider_field_teacher_gemma.json`.
    #[test]
    #[ignore]
    fn real_gemma_teacher_labels() {
        use crate::spider_field_experiment::{holdout_seed, TeacherFile, TeacherItem};
        let path = std::env::var("GEMMA2_GGUF").expect("GEMMA2_GGUF");
        let n_max: usize = std::env::var("N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60);
        let probe = crate::web::llm_periphery::open_gemma_probe(Path::new(&path)).unwrap();
        let h = probe.raw_handle().unwrap();
        let gen = |r: &str| -> f64 {
            let (pr, _, _) = h.choice_probs(r, &YES_NO, 768, None).unwrap();
            pr[0] / (pr[0] + pr[1]).max(1e-12)
        };
        let mut prompts: Vec<(String, u64, String)> =
            vec![("sample".into(), 0, SAMPLE_PROMPT.to_string())];
        for k in 0..40 {
            let sp = crate::spider_field::synth_prompt(
                holdout_seed(k),
                &crate::spider_field::VOCAB_TRAIN,
            );
            prompts.push(("holdout".into(), sp.seed, sp.text));
        }
        let mut out = TeacherFile {
            model: Path::new(&path)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default(),
            floor: DEFAULT_LLM_FLOOR,
            items: Vec::new(),
        };
        'outer: for (set, seed, text) in prompts {
            let (toks, dry) = crate::prompt_spider::dry_run(&text, DEFAULT_THRESHOLD);
            for d in dry.iter().filter(|d| d.route == Route::Llm) {
                if out.items.len() >= n_max {
                    break 'outer;
                }
                let it = EscalationItem {
                    index: d.index,
                    word: d.word.clone(),
                    context: word_context(&toks, d.index, 6),
                    question: d.assessment.question,
                    p: d.assessment.p,
                };
                let task = task_excerpt(&text, 120);
                let t0 = Instant::now();
                let a = gen(&compact_gemma_prompt(&task, &it, true));
                let b = gen(&compact_gemma_prompt(&task, &it, false));
                let v = gemma_choice_verdict(it.index, a, b);
                println!("{set} {seed} «{}» p={:.3}", it.word, v.p);
                out.items.push(TeacherItem {
                    set: set.clone(),
                    seed,
                    index: it.index,
                    word: it.word,
                    p_yes_first: a,
                    p_no_first: b,
                    p: v.p,
                    seconds: t0.elapsed().as_secs_f64(),
                });
            }
        }
        let dst = std::env::var("OUT").unwrap_or_else(|_| "spider_teacher.json".into());
        std::fs::write(&dst, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }

    /// Depuración del prompt compacto con el GGUF real (ignorada en CI).
    #[test]
    #[ignore]
    fn real_gemma_compact_raw_outputs() {
        let path = std::env::var("GEMMA2_GGUF").expect("GEMMA2_GGUF");
        let probe = crate::web::llm_periphery::open_gemma_probe(Path::new(&path)).unwrap();
        let h = probe.raw_handle().unwrap();
        let (toks, dry) = crate::prompt_spider::dry_run(SAMPLE_PROMPT, DEFAULT_THRESHOLD);
        let items: Vec<EscalationItem> = dry
            .iter()
            .filter(|d| d.route == Route::Llm)
            .take(
                std::env::var("N")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4),
            )
            .map(|d| EscalationItem {
                index: d.index,
                word: d.word.clone(),
                context: word_context(&toks, d.index, 6),
                question: d.assessment.question,
                p: d.assessment.p,
            })
            .collect();
        let gen = |r: &str| {
            let (pr, ntok, secs) = h.choice_probs(r, &YES_NO, 768, None)?;
            println!(
                "  prompt_tok={ntok} {secs:.2}s P(sí)={:.3} P(no)={:.3}",
                pr[0], pr[1]
            );
            Ok(pr[0] / (pr[0] + pr[1]).max(1e-12))
        };
        for it in &items {
            let t0 = Instant::now();
            let v = compact_choice(SAMPLE_PROMPT, std::slice::from_ref(it), &gen).unwrap();
            println!(
                "«{}» {:?} heur_p={:.2} → p={:.3} ({}) {:.2}s",
                it.word,
                it.question,
                it.p,
                v[0].p,
                v[0].note,
                t0.elapsed().as_secs_f64()
            );
        }
    }
}
