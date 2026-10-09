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
    clamp_threshold, llm_system_prompt, llm_user_prompt, parse_llm_verdicts, route, score_token,
    tokenize, verdict_outcome, word_context, Assessment, EscalationItem, LlmVerdict, PromptContext,
    Question, Route, Scores, Token, VerdictOutcome, DEFAULT_LLM_FLOOR, DEFAULT_THRESHOLD,
    SAMPLE_PROMPT,
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
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(d)
}

/// Directorio de corridas. Env `SPIDER_RUNS_DIR`; si no, `data/spider_runs`.
pub fn default_runs_dir() -> PathBuf {
    match std::env::var("SPIDER_RUNS_DIR") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p),
        _ => PathBuf::from("data/spider_runs"),
    }
}

/// Plazo por lote escalado al LLM. Env `SPIDER_LLM_TIMEOUT_SECS` (def 60).
pub fn llm_timeout() -> Duration {
    Duration::from_secs(env_u64("SPIDER_LLM_TIMEOUT_SECS", 60).clamp(2, 600))
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
    pub tokens: Vec<Token>,
    /// Tokens leídos (cursor del crawler).
    pub cursor: usize,
    pub decisions: Vec<Decision>,
    pub llm_calls: Vec<LlmCall>,
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
    /// `start` | `read` | `decision` | `update` | `llm_call` | `done`.
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
}

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

/// Backend de escalado (resuelto al iniciar la corrida).
#[derive(Clone)]
pub enum SpiderBackend {
    External(Box<ProviderConfig>),
    Gemma(RawGemmaHandle),
    Unavailable(String),
}

/// Llama al LLM (bloqueante). Devuelve (texto, modelo, segundos).
fn call_llm_blocking(
    backend: &SpiderBackend,
    system: &str,
    user: &str,
    timeout: Duration,
) -> Result<(String, String, f64), String> {
    match backend {
        SpiderBackend::External(cfg) => {
            let r = OpenAiClient::new(cfg, timeout).chat(
                &[
                    ChatMessage::new("system", system),
                    ChatMessage::new("user", user),
                ],
                ExternalGenConfig {
                    max_tokens: env_u64("SPIDER_LLM_MAX_TOKENS", 1200).clamp(64, 8192) as usize,
                    temperature: 0.0,
                    top_p: 1.0,
                },
            )?;
            Ok((r.text, r.model, r.seconds))
        }
        SpiderBackend::Gemma(h) => {
            let mut cfg = crate::web::llm_periphery::raw_chat_config(0x5_91DE);
            cfg.max_tokens = env_u64("SPIDER_LLM_MAX_TOKENS", 512).clamp(64, 2048) as usize;
            cfg.temperature = 0.0;
            let r = h.generate_chat(
                &[],
                &format!("{system}\n\n{user}"),
                cfg,
                Some(Instant::now() + timeout),
            )?;
            Ok((r.text, "gemma2-gguf".into(), r.seconds))
        }
        SpiderBackend::Unavailable(why) => Err(format!("LLM no disponible: {why}")),
    }
}

fn llm_info_and_backend(g: &crate::web::AppState) -> (LlmInfo, SpiderBackend) {
    match g.active_external() {
        Some(p) => (
            LlmInfo {
                id: p.id.clone(),
                label: p.label(),
                kind: "openai_compatible".into(),
                model: p.model.clone(),
                unavailable: None,
            },
            SpiderBackend::External(Box::new(p)),
        ),
        None => {
            let mut info = LlmInfo {
                id: crate::web::llm_provider::LOCAL_ID.into(),
                label: crate::web::state::LOCAL_LABEL.into(),
                kind: "local".into(),
                model: g.probe.name().into(),
                unavailable: None,
            };
            match g.probe.raw_handle() {
                Some(h) => (info, SpiderBackend::Gemma(h)),
                None => {
                    let why = g.model_unavailable_reason();
                    info.unavailable = Some(why.clone());
                    (info, SpiderBackend::Unavailable(why))
                }
            }
        }
    }
}

fn lock(st: &SharedState) -> std::sync::MutexGuard<'_, crate::web::AppState> {
    st.lock().unwrap_or_else(|e| e.into_inner())
}

/// Pasa un lote a pendiente con una nota (falla del LLM, detenido…).
fn mark_pending(st: &SharedState, items: &[usize], note: &str) {
    let mut g = lock(st);
    let mut changed = Vec::new();
    if let Some(run) = g.spider.run.as_mut() {
        for &i in items {
            if let Some(d) = run.decision_mut(i) {
                if d.status == Status::Escalating {
                    d.route = Route::You;
                    d.status = Status::Pending;
                    d.note = note.to_string();
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

async fn llm_worker(
    st: SharedState,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Vec<EscalationItem>>,
    backend: SpiderBackend,
    prompt: String,
    floor: f64,
    label: String,
    stop: Arc<AtomicBool>,
) {
    let system = llm_system_prompt();
    let mut batch_no = 0usize;
    while let Some(items) = rx.recv().await {
        batch_no += 1;
        let idx: Vec<usize> = items.iter().map(|i| i.index).collect();
        if stop.load(Ordering::Relaxed) {
            mark_pending(&st, &idx, "detenido antes del veredicto del LLM");
            continue;
        }
        let user = llm_user_prompt(&prompt, &items);
        let timeout = llm_timeout();
        let b = backend.clone();
        let sys = system.clone();
        let t0 = Instant::now();
        let call = tokio::task::spawn_blocking(move || call_llm_blocking(&b, &sys, &user, timeout));
        let res = match tokio::time::timeout(timeout + Duration::from_secs(2), call).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => Err(format!("tarea falló: {e}")),
            Err(_) => Err(format!("timeout ({}s)", timeout.as_secs())),
        };
        let parsed = res.and_then(|(text, model, secs)| {
            parse_llm_verdicts(&text, &idx).map(|v| (v, model, secs))
        });
        let mut g = lock(&st);
        let mut updates = Vec::new();
        let call_log = match parsed {
            Ok((verdicts, model, secs)) => {
                if let Some(run) = g.spider.run.as_mut() {
                    for &i in &idx {
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
                                        d.resolved_by = label.clone();
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
                }
                LlmCall {
                    batch: batch_no,
                    words: idx.clone(),
                    ok: true,
                    seconds: (secs * 100.0).round() / 100.0,
                    verdicts: verdicts.len(),
                    model,
                    error: None,
                }
            }
            Err(e) => {
                let e: String = e.chars().take(300).collect();
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
                    batch: batch_no,
                    words: idx.clone(),
                    ok: false,
                    seconds: (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
                    verdicts: 0,
                    model: String::new(),
                    error: Some(e),
                }
            }
        };
        let msg = if call_log.ok {
            format!(
                "lote {} → {}: {} palabras, {} veredictos, {:.2}s",
                call_log.batch,
                label,
                call_log.words.len(),
                call_log.verdicts,
                call_log.seconds
            )
        } else {
            format!(
                "lote {} → {} falló: {}",
                call_log.batch,
                label,
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

/// Camina el prompt, decide cada palabra y escala por lotes.
async fn run_spider(st: SharedState, stop: Arc<AtomicBool>, backend: SpiderBackend) {
    let (tokens, prompt, pace, batch_size, floor, threshold, label) = {
        let g = lock(&st);
        let r = g.spider.run.as_ref().expect("run");
        (
            r.tokens.clone(),
            r.prompt.clone(),
            r.pace_ms,
            r.batch_size,
            r.llm_floor,
            r.threshold,
            r.llm.label.clone(),
        )
    };
    let ctx = PromptContext::new(&tokens);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<EscalationItem>>();
    let worker = tokio::spawn(llm_worker(
        st.clone(),
        rx,
        backend,
        prompt,
        floor,
        label,
        stop.clone(),
    ));
    let mut batch: Vec<EscalationItem> = Vec::new();
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
                let d = Decision::new(tok, a, r);
                if r == Route::Llm {
                    batch.push(EscalationItem {
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
        if batch.len() >= batch_size {
            let _ = tx.send(std::mem::take(&mut batch));
        }
        if pace > 0 {
            tokio::time::sleep(Duration::from_millis(pace)).await;
        }
    }
    if !batch.is_empty() {
        let _ = tx.send(std::mem::take(&mut batch));
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
    );
    let mut g = lock(&st);
    let msg = if let Some(run) = g.spider.run.as_mut() {
        run.state = if stopped { "stopped" } else { "done" }.into();
        run.finished_ms = Some(now_ms());
        let s = run.summary();
        format!(
            "{}: {} forks · code {} · {} {} ({} escaladas) · you {} · p̄ {:.3}",
            run.state,
            s.forks,
            s.code,
            run.llm.label,
            s.llm_resolved,
            s.llm_escalated,
            s.you,
            s.avg_confidence
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
    let batch_size = b.batch_size.unwrap_or(8).clamp(1, 64);
    let mut g = lock(&st);
    if g.spider.running {
        return bad(StatusCode::CONFLICT, "ya hay una corrida en curso");
    }
    let (info, backend) = llm_info_and_backend(&g);
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
        tokens,
        cursor: 0,
        decisions: Vec::new(),
        llm_calls: Vec::new(),
    };
    let stop = Arc::new(AtomicBool::new(false));
    g.spider.stop = stop.clone();
    g.spider.running = true;
    g.spider.run = Some(run);
    g.spider.events.clear();
    let msg = format!("corrida {id} · umbral {threshold:.2} · LLM {}", info.label);
    g.spider.push("start", None, None, msg);
    let seq = g.spider.seq;
    drop(g);
    tokio::spawn(run_spider(st.clone(), stop, backend));
    Json(json!({
        "ok": true, "run_id": id, "threshold": threshold, "llm_floor": floor,
        "tokens_total": total, "forks_total": forks, "llm": info, "seq": seq,
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
    let (info, _) = llm_info_and_backend(&g);
    Json(json!({
        "running": g.spider.running,
        "seq": g.spider.seq,
        "run": g.spider.run,
        "summary": g.spider.run.as_ref().map(SpiderRun::summary),
        "llm_active": info,
        "defaults": { "threshold": DEFAULT_THRESHOLD, "llm_floor": DEFAULT_LLM_FLOOR },
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
