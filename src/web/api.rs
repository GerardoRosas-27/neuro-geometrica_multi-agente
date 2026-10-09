//! REST API JSON + estáticos + SSE de entrenamiento en vivo.

use crate::web::auth::{self, AuthConfig, SharedAuth};
use crate::web::field_eval::FieldEvalReport;
use crate::web::process_job::{ProcessesSnapshot, SleepJobSnapshot, TestsJobSnapshot};
use crate::web::sleep_optimize::{SleepOptimizeOpts, SleepOptimizeReport};
use crate::web::state::{AppState, ChatRequest};
use crate::web::telemetry::{FuseReportDto, SleepReportDto, TelemetrySnapshot};
use crate::web::train_job::{LiveTrainStartRequest, TrainJobSnapshot, TrainLiveEvent};
use axum::extract::{Query, State};
use axum::http::{header, HeaderValue};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::trace::TraceLayer;

pub type SharedState = Arc<Mutex<AppState>>;

#[derive(Clone, Debug, Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub llm_mode: String,
    pub engrams: usize,
    pub wake_buffer: usize,
    pub training: bool,
    pub sleeping: bool,
    pub testing: bool,
    pub active_processes: Vec<String>,
    pub probe: String,
    /// True si el Gemma 2 original (GGUF) está cargado → chat crudo disponible.
    pub raw_gemma_available: bool,
    pub raw_chat_calls: u64,
    /// Estado del GGUF: ready | downloading | loading | missing | error | disabled.
    pub model: crate::web::model_fetch::ModelStatus,
    /// LLM activo de la app (Gemma local o API externa).
    pub llm_active: LlmActive,
}

/// LLM activo (para badges de la UI).
#[derive(Clone, Debug, Serialize)]
pub struct LlmActive {
    pub id: String,
    pub label: String,
    /// `local` | `openai_compatible`.
    pub kind: String,
    pub model: String,
}

fn llm_active_of(g: &AppState) -> LlmActive {
    match g.active_external() {
        Some(p) => LlmActive {
            id: p.id.clone(),
            label: p.label(),
            kind: "openai_compatible".into(),
            model: p.model.clone(),
        },
        None => LlmActive {
            id: crate::web::llm_provider::LOCAL_ID.into(),
            label: crate::web::state::LOCAL_LABEL.into(),
            kind: "local".into(),
            model: g.probe.name().into(),
        },
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FullTelemetry {
    pub metrics: TelemetrySnapshot,
    pub liquid: LiquidSlice,
    pub cdt: CdtSlice,
    pub rqm: RqmSlice,
    pub train: TrainSlice,
    pub last_fuse: Option<FuseReportDto>,
    pub last_sleep: Option<SleepReportDto>,
    pub train_log_tail: Vec<crate::web::state::TrainEvent>,
    pub llm_mode: String,
    pub live_job: TrainJobSnapshot,
    pub last_sleep_optimize: Option<SleepOptimizeReport>,
    pub last_field_eval: Option<FieldEvalReport>,
    pub sleep_job: SleepJobSnapshot,
    pub tests_job: TestsJobSnapshot,
    pub processes: ProcessesSnapshot,
}

#[derive(Clone, Debug, Serialize)]
pub struct LiquidSlice {
    pub queries: u64,
    pub score_last: f64,
    pub score_avg: f64,
    pub latency_us_last: f64,
    pub route_pct: f64,
    pub last_route: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CdtSlice {
    pub engram_count: usize,
    pub wake_buffer: usize,
    pub sleeps: u64,
    pub last_sleep: Option<SleepReportDto>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RqmSlice {
    pub infer_calls: u64,
    pub train_calls: u64,
    pub route_pct: f64,
    pub relational_cues: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainSlice {
    pub training: bool,
    pub epochs_done: u64,
    pub accuracy: Option<f64>,
    pub events_tail: Vec<crate::web::state::TrainEvent>,
    pub live: TrainJobSnapshot,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EventsQuery {
    pub after: Option<u64>,
}

/// Lanza el loop de lotes en background (libera el Mutex entre lotes).
pub fn spawn_live_train_loop(state: SharedState) {
    tokio::spawn(async move {
        loop {
            let cont = {
                let st = state.clone();
                match tokio::task::spawn_blocking(move || {
                    // API externa activa: el dataset se genera por HTTP **sin**
                    // sostener el lock (chat/telemetría siguen respondiendo).
                    let plan = st
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .external_dataset_plan();
                    let pre = plan.map(|(cfg, batch_size, seed)| {
                        let r = crate::web::llm_periphery::generate_train_batch_external(
                            &cfg,
                            batch_size,
                            seed,
                            crate::web::llm_periphery::chat_timeout(),
                        );
                        (seed, r)
                    });
                    let mut g = st.lock().unwrap_or_else(|e| e.into_inner());
                    g.run_one_live_batch_with(pre)
                })
                .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        if let Ok(mut g) = state.lock() {
                            g.train_job.push_event(
                                "error",
                                format!("worker panic: {e}"),
                                None,
                                None,
                                json!({}),
                            );
                            g.training = false;
                            g.train_job.running = false;
                        }
                        false
                    }
                }
            };
            if !cont {
                break;
            }
            // Ceder al event loop para chat/telemetry.
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    });
}

pub fn spawn_sleep_job(state: SharedState) {
    tokio::spawn(async move {
        loop {
            let cont = {
                let st = state.clone();
                match tokio::task::spawn_blocking(move || {
                    let mut g = st.lock().unwrap();
                    g.run_one_sleep_cycle()
                })
                .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        if let Ok(mut g) = state.lock() {
                            g.sleep_job.push_event(
                                "error",
                                format!("worker panic: {e}"),
                                json!({}),
                            );
                            g.sleep_job.running = false;
                            g.sleep_job.phase = "error".into();
                            g.sleep_job.finished_ms = Some(crate::web::train_job::now_ms());
                        }
                        false
                    }
                }
            };
            if !cont {
                break;
            }
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    });
}

pub fn spawn_tests_job(state: SharedState) {
    tokio::spawn(async move {
        loop {
            let cont = {
                let st = state.clone();
                match tokio::task::spawn_blocking(move || {
                    let mut g = st.lock().unwrap();
                    g.run_one_tests_cycle()
                })
                .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        if let Ok(mut g) = state.lock() {
                            g.tests_job.push_event(
                                "error",
                                format!("worker panic: {e}"),
                                json!({}),
                            );
                            g.tests_job.running = false;
                            g.field_eval_running = false;
                            g.tests_job.phase = "error".into();
                            g.tests_job.finished_ms = Some(crate::web::train_job::now_ms());
                        }
                        false
                    }
                }
            };
            if !cont {
                break;
            }
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    });
}

/// Router con autenticación leída del entorno (`MASTER_SECRET`, …).
pub fn router(state: SharedState, static_dir: PathBuf) -> Router {
    router_with_auth(state, static_dir, Arc::new(AuthConfig::from_env()))
}

/// Router con una configuración de acceso explícita. **Todo** `/api/*` exige
/// sesión salvo `/api/auth/{login,logout,status}`; `/health` y los estáticos
/// son públicos (el middleware envuelve también el fallback).
pub fn router_with_auth(state: SharedState, static_dir: PathBuf, auth: SharedAuth) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let public = Router::new()
        .route("/health", get(health))
        .route("/api/auth/login", post(auth::login))
        .route("/api/auth/logout", post(auth::logout))
        .route("/api/auth/status", get(auth::status))
        .with_state(auth.clone());

    let api = Router::new()
        .route("/api/status", get(app_status))
        .route("/api/chat", post(chat))
        .route("/api/train/start", post(train_start))
        .route("/api/train/stop", post(train_stop))
        .route("/api/train/status", get(train_status))
        .route("/api/train/events", get(train_events))
        .route("/api/train/stream", get(train_stream))
        .route("/api/sleep", post(sleep))
        .route("/api/sleep/start", post(sleep_start))
        .route("/api/sleep/stop", post(sleep_stop))
        .route("/api/sleep/status", get(sleep_status))
        .route("/api/sleep/events", get(sleep_events))
        .route("/api/tests/run", post(tests_run))
        .route("/api/tests/start", post(tests_start))
        .route("/api/tests/stop", post(tests_stop))
        .route("/api/tests/last", get(tests_last))
        .route("/api/tests/status", get(tests_status_full))
        .route("/api/tests/events", get(tests_events))
        .route("/api/processes", get(processes))
        .route("/api/telemetry", get(telemetry))
        .route("/api/telemetry/liquid", get(telemetry_liquid))
        .route("/api/telemetry/cdt", get(telemetry_cdt))
        .route("/api/telemetry/rqm", get(telemetry_rqm))
        .route("/api/telemetry/train", get(telemetry_train))
        .route("/api/chat/history", get(chat_history))
        .route("/api/chat/reset", post(chat_reset))
        .route(
            "/api/llm/providers",
            get(llm_providers_list).post(llm_providers_save),
        )
        .route("/api/llm/providers/test", post(llm_providers_test))
        .route(
            "/api/llm/providers/{id}",
            axum::routing::delete(llm_providers_delete),
        )
        .route("/api/llm/active", get(llm_active_get).post(llm_active_set))
        .route("/api/llm/parse-curl", post(llm_parse_curl))
        .merge(crate::web::spider_job::routes())
        .with_state(state);

    // Estáticos: no-cache para que index.html siempre pida app.js/css frescos
    // (cache-bust ?v=N en index.html es la defensa principal; esto refuerza).
    let static_svc = ServiceBuilder::new()
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache, must-revalidate"),
        ))
        .service(ServeDir::new(static_dir).append_index_html_on_directories(true));

    Router::new()
        .merge(public)
        .merge(api)
        .fallback_service(static_svc)
        .layer(axum::middleware::from_fn_with_state(
            auth,
            auth::require_auth,
        ))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}

/// Salud pública: sin datos internos (modelo, rutas, procesos, LLM). Solo si el
/// proceso responde y el estado del secreto maestro (sin revelarlo).
async fn health(State(auth): State<SharedAuth>) -> impl IntoResponse {
    Json(json!({
        "ok": true,
        "service": "neuro-geometrica",
        "auth": {
            "mode": auth.mode(),
            "master_configured": auth.configured(),
            "session_ttl_hours": auth.ttl_secs as f64 / 3600.0,
            "warnings": auth.warnings(),
        },
    }))
}

/// Estado detallado de la app (antes `/health`); requiere sesión.
async fn app_status(State(st): State<SharedState>) -> impl IntoResponse {
    let mut g = st.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = g.model_cfg.clone();
    g.model_status.refresh_progress(&cfg);
    let procs = g.processes_snapshot();
    Json(HealthResponse {
        ok: true,
        llm_mode: g.llm_mode().as_str().into(),
        engrams: g.fuse.engram_count(),
        wake_buffer: g.fuse.wake_buffer_len(),
        training: g.training || g.train_job.running,
        sleeping: g.sleep_job.running,
        testing: g.tests_job.running || g.field_eval_running,
        active_processes: procs.active,
        probe: g.probe.name().into(),
        raw_gemma_available: g.probe.raw_handle().is_some(),
        raw_chat_calls: g.raw_chat_calls,
        model: g.model_status.clone(),
        llm_active: llm_active_of(&g),
    })
}

/// Arranque del modelo: si falta el GGUF y la descarga automática está activa,
/// lo descarga en segundo plano (curl) y hace hot-swap de la sonda a Gemma.
/// El servidor sigue sirviendo (léxico) mientras tanto.
pub fn spawn_model_bootstrap(state: SharedState) {
    let (cfg, needs) = {
        let g = state.lock().unwrap_or_else(|e| e.into_inner());
        (
            g.model_cfg.clone(),
            g.probe.raw_handle().is_none() && g.model_status.state == "missing",
        )
    };
    if !needs {
        return;
    }
    tokio::spawn(async move {
        use crate::web::model_fetch::{download_gguf, ModelStatus};
        if !cfg.path.is_file() {
            state.lock().unwrap_or_else(|e| e.into_inner()).model_status =
                ModelStatus::new("downloading", &cfg, "descargando GGUF");
            let c = cfg.clone();
            let r = tokio::task::spawn_blocking(move || download_gguf(&c))
                .await
                .unwrap_or_else(|e| Err(format!("tarea de descarga falló: {e}")));
            if let Err(e) = r {
                tracing::error!(error = %e, "descarga GGUF falló");
                state.lock().unwrap_or_else(|e| e.into_inner()).model_status =
                    ModelStatus::new("error", &cfg, e);
                return;
            }
        }
        state.lock().unwrap_or_else(|e| e.into_inner()).model_status =
            ModelStatus::new("loading", &cfg, "cargando GGUF");
        let c = cfg.clone();
        let r = tokio::task::spawn_blocking(move || {
            crate::web::llm_periphery::open_gemma_probe(&c.path)
        })
        .await
        .unwrap_or_else(|e| Err(format!("tarea de carga falló: {e}")));
        let mut g = state.lock().unwrap_or_else(|e| e.into_inner());
        match r {
            Ok(probe) => {
                tracing::info!(path = %cfg.path.display(), "Gemma cargado (hot-swap)");
                g.install_probe(probe);
            }
            Err(e) => {
                tracing::error!(error = %e, "carga GGUF falló");
                g.model_status = ModelStatus::new("error", &cfg, e);
            }
        }
    });
}

async fn chat(
    State(st): State<SharedState>,
    Json(body): Json<ChatRequest>,
) -> axum::response::Response {
    use crate::web::llm_periphery::ChatMode;
    use axum::http::StatusCode;
    let mode = match body.chat_mode() {
        Ok(m) => m,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "ok": false, "error": e })),
            )
                .into_response();
        }
    };
    match mode {
        ChatMode::FieldDecoder => chat_field_decoder(st, body.message).await,
        ChatMode::GemmaRaw => chat_gemma_raw(st, body.message).await,
    }
}

/// ON: campo (líquido/CDT/RQM) produce el estado bajo el lock (en hilo
/// bloqueante, no en el event loop); Gemma lo **decodifica** a texto fuera del
/// lock con plazo. Si Gemma no está o falla → decoder léxico. Siempre responde.
async fn chat_field_decoder(st: SharedState, message: String) -> axum::response::Response {
    use crate::web::llm_periphery::{chat_timeout, field_decoder_config};
    use axum::http::StatusCode;
    if message.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "mode": "field_decoder", "error": "mensaje vacío" })),
        )
            .into_response();
    }
    let deadline = std::time::Instant::now() + chat_timeout();
    let t0 = std::time::Instant::now();
    let st2 = st.clone();
    let msg = message.clone();
    let field = tokio::task::spawn_blocking(move || {
        let mut g = st2.lock().unwrap_or_else(|e| e.into_inner());
        let was_running = g.train_job.running;
        let resp = g.handle_chat(&msg);
        let should_spawn = resp.route == "train" && g.train_job.running && !was_running;
        let job = g.field_decode_job(&msg, &resp);
        let pending_note = (job.is_none()
            && matches!(resp.route.as_str(), "Liquid" | "RqmFallback")
            && g.model_status.state != "ready")
            .then(|| g.model_unavailable_reason());
        (resp, should_spawn, job, pending_note)
    })
    .await;
    let (mut resp, should_spawn, job, pending_note) = match field {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "ok": false, "mode": "field_decoder",
                              "error": format!("pipeline de campo falló: {e}") })),
            )
                .into_response();
        }
    };
    let field_secs = t0.elapsed().as_secs_f64();
    if should_spawn {
        spawn_live_train_loop(st.clone());
    }
    if let Some(job) = job {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xDEC0);
        let cfg = field_decoder_config(seed);
        let outcome = tokio::task::spawn_blocking(move || run_decode_job(job, cfg, deadline))
            .await
            .unwrap_or_else(|e| DecodeOutcome {
                result: Err(format!("tarea del decoder falló: {e}")),
                who: "decoder".into(),
                llm_id: String::new(),
                llm: "decoder".into(),
                note: None,
                fallback: false,
            });
        tracing::info!(
            field_secs,
            decoder_secs = t0.elapsed().as_secs_f64() - field_secs,
            ok = outcome.result.is_ok(),
            llm = %outcome.llm,
            fallback = outcome.fallback,
            "chat field_decoder"
        );
        resp = st
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply_field_decoded_by(
                resp,
                outcome.result,
                &outcome.who,
                &outcome.llm_id,
                &outcome.llm,
                outcome.note,
            );
        resp.fallback = outcome.fallback;
    } else if let Some(note) = pending_note {
        resp.reply.push_str(&format!(
            "\n\n(decoder Gemma aún no disponible: {note}; respuesta del decoder léxico)"
        ));
    }
    if resp.llm.is_empty() && matches!(resp.route.as_str(), "Liquid" | "RqmFallback") {
        resp.llm = "decoder léxico".into();
        resp.llm_id = "lexicon".into();
    }
    Json(resp).into_response()
}

/// Resultado del decoder (quién respondió + nota de respaldo).
struct DecodeOutcome {
    result: Result<crate::field_gemma_probe::RawGemmaReply, String>,
    who: String,
    llm_id: String,
    llm: String,
    note: Option<String>,
    fallback: bool,
}

fn remaining(deadline: std::time::Instant) -> Duration {
    deadline
        .saturating_duration_since(std::time::Instant::now())
        .max(Duration::from_secs(1))
}

fn external_to_raw(
    r: crate::web::llm_provider::ExternalReply,
) -> crate::field_gemma_probe::RawGemmaReply {
    crate::field_gemma_probe::RawGemmaReply {
        text: r.text,
        prompt_tokens: r.prompt_tokens,
        generated_tokens: r.completion_tokens,
        seconds: r.seconds,
    }
}

/// Ejecuta el decoder (bloqueante). API externa → si falla y Gemma local está
/// cargado y queda plazo (≥ 10 s), respaldo local **etiquetado**.
fn run_decode_job(
    job: crate::web::state::DecodeJob,
    cfg: crate::native_gemma2_runtime::Gemma2GenerationConfig,
    deadline: std::time::Instant,
) -> DecodeOutcome {
    use crate::web::llm_provider::{external_decoder_config, field_decoder_messages, OpenAiClient};
    use crate::web::state::{DecodeBackend, LOCAL_LABEL};
    let local = |handle: crate::field_gemma_probe::RawGemmaHandle| {
        handle.generate_prompt(&job.gemma_prompt, cfg, Some(deadline))
    };
    match job.backend {
        DecodeBackend::Gemma(handle) => DecodeOutcome {
            result: local(handle),
            who: "gemma2 decoder".into(),
            llm_id: crate::web::llm_provider::LOCAL_ID.into(),
            llm: LOCAL_LABEL.into(),
            note: None,
            fallback: false,
        },
        DecodeBackend::External { cfg: p, fallback } => {
            let client = OpenAiClient::new(&p, remaining(deadline));
            let r = client
                .chat(
                    &field_decoder_messages(&job.user_msg, &job.field_state),
                    external_decoder_config(),
                )
                .map(external_to_raw);
            let who = format!(
                "api decoder·{}",
                if p.model.is_empty() {
                    &p.name
                } else {
                    &p.model
                }
            );
            match (r, fallback) {
                (Ok(reply), _) => DecodeOutcome {
                    result: Ok(reply),
                    who,
                    llm_id: p.id.clone(),
                    llm: p.label(),
                    note: None,
                    fallback: false,
                },
                (Err(e), Some(handle)) if remaining(deadline) >= Duration::from_secs(10) => {
                    tracing::warn!(provider = %p.name, error = %e, "API externa falló; respaldo Gemma local");
                    DecodeOutcome {
                        result: local(handle),
                        who: "gemma2 decoder (respaldo)".into(),
                        llm_id: crate::web::llm_provider::LOCAL_ID.into(),
                        llm: format!("{LOCAL_LABEL} (respaldo)"),
                        note: Some(format!(
                            "{} falló: {e} · respondió Gemma local como respaldo",
                            p.label()
                        )),
                        fallback: true,
                    }
                }
                (Err(e), _) => DecodeOutcome {
                    result: Err(e),
                    who,
                    llm_id: p.id.clone(),
                    llm: p.label(),
                    note: None,
                    fallback: false,
                },
            }
        }
    }
}

/// OFF: Gemma 2 congelado original como LLM plano. Nunca cae al campo.
/// La generación corre en `spawn_blocking` sin sostener el lock de AppState.
async fn chat_gemma_raw(st: SharedState, message: String) -> axum::response::Response {
    use crate::web::llm_periphery::{chat_timeout, raw_chat_config};
    use axum::http::StatusCode;
    let msg = message.trim().to_string();
    if msg.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "mode": "gemma_raw", "error": "mensaje vacío" })),
        )
            .into_response();
    }
    let deadline = std::time::Instant::now() + chat_timeout();
    let st2 = st.clone();
    let prepared = tokio::task::spawn_blocking(move || {
        st2.lock()
            .unwrap_or_else(|e| e.into_inner())
            .raw_chat_backend()
    })
    .await
    .unwrap_or_else(|e| Err(format!("tarea falló: {e}")));
    let (backend, history) = match prepared {
        Ok(v) => v,
        Err(e) => {
            let resp = st
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .record_raw_chat(&msg, &Err(e));
            return (StatusCode::SERVICE_UNAVAILABLE, Json(resp)).into_response();
        }
    };
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x6E33A);
    let cfg = raw_chat_config(seed);
    let input = msg.clone();
    let (result, external) = match backend {
        crate::web::state::RawBackend::Gemma(handle) => (
            tokio::task::spawn_blocking(move || {
                handle.generate_chat(&history, &input, cfg, Some(deadline))
            })
            .await
            .unwrap_or_else(|e| Err(format!("tarea de generación falló: {e}"))),
            None,
        ),
        crate::web::state::RawBackend::External { cfg: p, fallback } => {
            use crate::web::llm_provider::{external_raw_config, raw_chat_messages, OpenAiClient};
            // Con respaldo local, la API no puede comerse todo el plazo.
            let reserve = Duration::from_secs(
                std::env::var("RAW_CHAT_FALLBACK_RESERVE_SECS")
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(30u64)
                    .min(300),
            );
            let api_budget = if fallback.is_some() {
                remaining(deadline)
                    .saturating_sub(reserve)
                    .max(Duration::from_secs(5))
            } else {
                remaining(deadline)
            };
            let (p2, hist2, input2) = (p.clone(), history.clone(), input.clone());
            let r = tokio::task::spawn_blocking(move || {
                OpenAiClient::new(&p2, api_budget)
                    .chat(&raw_chat_messages(&hist2, &input2), external_raw_config())
                    .map(external_to_raw)
            })
            .await
            .unwrap_or_else(|e| Err(format!("tarea de generación falló: {e}")));
            match (r, fallback) {
                (Err(e), Some(handle))
                    if crate::web::spider_job::is_unavailable_error(&e)
                        && remaining(deadline) >= Duration::from_secs(10) =>
                {
                    tracing::warn!(provider = %p.name, error = %e, "chat crudo: API externa no disponible; respaldo Gemma local");
                    let local = tokio::task::spawn_blocking(move || {
                        handle.generate_chat(&history, &input, cfg, Some(deadline))
                    })
                    .await
                    .unwrap_or_else(|e| Err(format!("tarea de generación falló: {e}")));
                    let ok = local.is_ok();
                    let resp = st
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .record_raw_chat_fallback(&msg, &local, &p, &e);
                    let status = if ok {
                        StatusCode::OK
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    };
                    return (status, Json(resp)).into_response();
                }
                (r, _) => (r, Some(p)),
            }
        }
    };
    let ok = result.is_ok();
    if let (Err(e), Some(p)) = (&result, &external) {
        tracing::warn!(provider = %p.name, error = %e, "chat crudo: API externa falló");
    }
    let resp = st
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .record_raw_chat_by(&msg, &result, external.as_ref());
    let status = match (ok, external.is_some()) {
        (true, _) => StatusCode::OK,
        (false, true) => StatusCode::BAD_GATEWAY,
        (false, false) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(resp)).into_response()
}

async fn train_start(
    State(st): State<SharedState>,
    Json(body): Json<LiveTrainStartRequest>,
) -> impl IntoResponse {
    use crate::web::train_job::parse_batches_field;
    // missing / null / 0 / "infinite" → infinito (default).
    let batches = parse_batches_field(body.batches.as_ref());
    let batch_size = body.batch_size.unwrap_or(8);
    let epochs = body.epochs.unwrap_or(1);
    let started = {
        let mut g = st.lock().unwrap();
        g.begin_live_train(batches, batch_size, epochs)
    };
    if started.ok {
        spawn_live_train_loop(st);
    }
    Json(started)
}

async fn train_stop(State(st): State<SharedState>) -> impl IntoResponse {
    let mut g = st.lock().unwrap();
    let ok = g.request_train_stop();
    Json(json!({ "ok": ok, "cancelled": g.train_job.cancelled }))
}

async fn train_status(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(g.train_job.snapshot_for_api())
}

async fn train_events(
    State(st): State<SharedState>,
    Query(q): Query<EventsQuery>,
) -> impl IntoResponse {
    let after = q.after.unwrap_or(0);
    let g = st.lock().unwrap();
    let events = g.live_events_after(after);
    Json(json!({
        "after": after,
        "event_seq": g.train_job.event_seq,
        "events": events,
        "running": g.train_job.running,
    }))
}

async fn train_stream(
    State(st): State<SharedState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::unfold((st, 0u64), |(st, mut after)| async move {
        loop {
            let (events, running, seq): (Vec<TrainLiveEvent>, bool, u64) = {
                let g = st.lock().unwrap();
                let ev = g.live_events_after(after);
                (ev, g.train_job.running, g.train_job.event_seq)
            };
            if !events.is_empty() {
                after = events.last().map(|e| e.seq).unwrap_or(after);
                let payload = serde_json::to_string(&events).unwrap_or_else(|_| "[]".into());
                let event = Event::default().event("train").data(payload);
                return Some((Ok(event), (st, after)));
            }
            if !running && after >= seq {
                let event = Event::default()
                    .event("train")
                    .data(r#"[{"kind":"done","message":"stream idle"}]"#);
                return Some((Ok(event), (st, after)));
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
            // Si no hay job activo y no hay eventos nuevos, cerrar tras un ciclo.
            let still = {
                let g = st.lock().unwrap();
                g.train_job.running || g.train_job.event_seq > after
            };
            if !still {
                return None;
            }
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

#[derive(Clone, Debug, Deserialize)]
pub struct SleepRequest {
    pub prune_intensity: Option<f64>,
    pub compact_intensity: Option<f64>,
    pub consolidate_first: Option<bool>,
    /// Si true (o cycles null) → ciclos hasta Detener.
    pub infinite: Option<bool>,
    /// Número de ciclos; null/0/"infinite" = infinito.
    pub cycles: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TestsStartRequest {
    pub infinite: Option<bool>,
    pub cycles: Option<serde_json::Value>,
}

fn sleep_opts_from_req(req: SleepRequest) -> SleepOptimizeOpts {
    SleepOptimizeOpts {
        prune_intensity: req.prune_intensity.unwrap_or(0.55),
        compact_intensity: req.compact_intensity.unwrap_or(0.55),
        consolidate_first: req.consolidate_first.unwrap_or(true),
    }
}

/// Alias de `/api/sleep/start` (async). Devuelve job_id; el informe final está en status.
async fn sleep(
    State(st): State<SharedState>,
    body: Option<Json<SleepRequest>>,
) -> impl IntoResponse {
    sleep_start(State(st), body).await
}

async fn sleep_start(
    State(st): State<SharedState>,
    body: Option<Json<SleepRequest>>,
) -> impl IntoResponse {
    use crate::web::process_job::resolve_infinite_cycles;
    let req = body.map(|j| j.0).unwrap_or(SleepRequest {
        prune_intensity: None,
        compact_intensity: None,
        consolidate_first: None,
        infinite: None,
        cycles: None,
    });
    let (infinite, total_cycles) = resolve_infinite_cycles(req.infinite, req.cycles.as_ref());
    let opts = sleep_opts_from_req(SleepRequest {
        prune_intensity: req.prune_intensity,
        compact_intensity: req.compact_intensity,
        consolidate_first: req.consolidate_first,
        infinite: None,
        cycles: None,
    });
    let started = {
        let mut g = st.lock().unwrap();
        g.begin_sleep_job(opts, infinite, total_cycles)
    };
    if started.ok {
        spawn_sleep_job(st);
    }
    Json(started)
}

async fn sleep_stop(State(st): State<SharedState>) -> impl IntoResponse {
    let mut g = st.lock().unwrap();
    let ok = g.request_sleep_stop();
    Json(json!({ "ok": ok, "cancelled": g.sleep_job.cancelled }))
}

async fn sleep_status(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(g.sleep_job.snapshot_for_api())
}

async fn sleep_events(
    State(st): State<SharedState>,
    Query(q): Query<EventsQuery>,
) -> impl IntoResponse {
    let after = q.after.unwrap_or(0);
    let g = st.lock().unwrap();
    let events = g.sleep_job.events_after(after);
    Json(json!({
        "after": after,
        "event_seq": g.sleep_job.event_seq,
        "events": events,
        "running": g.sleep_job.running,
    }))
}

/// Arranca batería async (compat: antes era síncrono).
async fn tests_run(
    State(st): State<SharedState>,
    body: Option<Json<TestsStartRequest>>,
) -> impl IntoResponse {
    tests_start(State(st), body).await
}

async fn tests_start(
    State(st): State<SharedState>,
    body: Option<Json<TestsStartRequest>>,
) -> impl IntoResponse {
    use crate::web::process_job::resolve_infinite_cycles;
    let req = body.map(|j| j.0).unwrap_or(TestsStartRequest {
        infinite: None,
        cycles: None,
    });
    let (infinite, total_cycles) = resolve_infinite_cycles(req.infinite, req.cycles.as_ref());
    let started = {
        let mut g = st.lock().unwrap();
        g.begin_tests_job(infinite, total_cycles)
    };
    if started.ok {
        spawn_tests_job(st);
    }
    Json(started)
}

async fn tests_stop(State(st): State<SharedState>) -> impl IntoResponse {
    let mut g = st.lock().unwrap();
    let ok = g.request_tests_stop();
    Json(json!({ "ok": ok, "cancelled": g.tests_job.cancelled }))
}

async fn tests_last(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(g.tests_status())
}

async fn tests_status_full(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(g.tests_job.snapshot_for_api())
}

async fn tests_events(
    State(st): State<SharedState>,
    Query(q): Query<EventsQuery>,
) -> impl IntoResponse {
    let after = q.after.unwrap_or(0);
    let g = st.lock().unwrap();
    let events = g.tests_job.events_after(after);
    Json(json!({
        "after": after,
        "event_seq": g.tests_job.event_seq,
        "events": events,
        "running": g.tests_job.running,
    }))
}

async fn processes(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(g.processes_snapshot())
}

fn build_full(g: &AppState) -> FullTelemetry {
    let last_route = g.last_fuse_dto().map(|f| f.route);
    let live = g.train_job.snapshot_for_api();
    FullTelemetry {
        metrics: g.metrics.clone(),
        liquid: LiquidSlice {
            queries: g.metrics.liquid_queries,
            score_last: g.metrics.liquid_score_last,
            score_avg: g.metrics.liquid_score_avg(),
            latency_us_last: g.metrics.liquid_latency_us_last,
            route_pct: g.metrics.liquid_route_pct(),
            last_route,
        },
        cdt: CdtSlice {
            engram_count: g.fuse.engram_count(),
            wake_buffer: g.fuse.wake_buffer_len(),
            sleeps: g.metrics.sleeps,
            last_sleep: g.last_sleep_dto(),
        },
        rqm: RqmSlice {
            infer_calls: g.fuse.rqm_infer_calls,
            train_calls: g.fuse.rqm_train_calls,
            route_pct: g.metrics.rqm_route_pct(),
            relational_cues: g.fuse.relational_cues.len(),
        },
        train: TrainSlice {
            training: g.training,
            epochs_done: g.metrics.train_epochs_done,
            accuracy: g.metrics.train_accuracy(),
            events_tail: g.train_log.iter().rev().take(30).cloned().collect(),
            live: live.clone(),
        },
        last_fuse: g.last_fuse_dto(),
        last_sleep: g.last_sleep_dto(),
        train_log_tail: g.train_log.iter().rev().take(50).cloned().collect(),
        llm_mode: g.llm_mode().as_str().into(),
        live_job: live,
        last_sleep_optimize: g.last_sleep_optimize.clone(),
        last_field_eval: g.last_field_eval.clone(),
        sleep_job: g.sleep_job.snapshot_for_api(),
        tests_job: g.tests_job.snapshot_for_api(),
        processes: g.processes_snapshot(),
    }
}

async fn telemetry(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(build_full(&g))
}

async fn telemetry_liquid(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(build_full(&g).liquid)
}

async fn telemetry_cdt(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(build_full(&g).cdt)
}

async fn telemetry_rqm(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(build_full(&g).rqm)
}

async fn telemetry_train(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(build_full(&g).train)
}

/// «Nuevo chat»: vacía el historial del servidor (y el contexto crudo de OFF).
async fn chat_reset(State(st): State<SharedState>) -> impl IntoResponse {
    let cleared = st.lock().unwrap_or_else(|e| e.into_inner()).reset_chat();
    Json(json!({ "ok": true, "cleared": cleared }))
}

async fn chat_history(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(json!({ "turns": g.chat_log }))
}

// ------------------------------------------------------------ proveedores LLM

fn llm_bad(status: axum::http::StatusCode, e: impl Into<String>) -> axum::response::Response {
    (status, Json(json!({ "ok": false, "error": e.into() }))).into_response()
}

fn llm_list_json(g: &AppState) -> serde_json::Value {
    let active = llm_active_of(g);
    json!({
        "ok": true,
        "active": active.id,
        "active_label": active.label,
        "active_info": active,
        "local": {
            "id": crate::web::llm_provider::LOCAL_ID,
            "name": crate::web::state::LOCAL_LABEL,
            "kind": "local",
            "available": g.probe.raw_handle().is_some(),
            "probe": g.probe.name(),
            "model_state": g.model_status.state,
        },
        "providers": g.llm.public_list(),
        "persisted_to": g.llm.path.as_ref().map(|p| p.display().to_string()),
    })
}

async fn llm_providers_list(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap_or_else(|e| e.into_inner());
    Json(llm_list_json(&g))
}

/// Cuerpo de guardar/probar: campos manuales y/o `curl` pegado.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct LlmProviderBody {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub curl: Option<String>,
    /// Guardar y activar en un paso.
    #[serde(default)]
    pub activate: Option<bool>,
}

fn non_empty(v: &Option<String>) -> Option<String> {
    v.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Combina: guardado (si `id`) ← curl ← campos manuales (los manuales ganan).
fn resolve_provider_body(
    g: &AppState,
    body: &LlmProviderBody,
) -> Result<crate::web::llm_provider::ProviderInput, String> {
    use crate::web::llm_provider::{parse_curl, ProviderInput};
    let mut out = ProviderInput {
        id: non_empty(&body.id),
        ..Default::default()
    };
    if let Some(id) = &out.id {
        if let Some(p) = g.llm.get(id) {
            out.name = p.name.clone();
            out.base_url = p.base_url.clone();
            out.api_key = p.api_key.clone();
            out.model = p.model.clone();
        }
    }
    if let Some(c) = non_empty(&body.curl) {
        let parsed = parse_curl(&c)?;
        if !parsed.base_url.is_empty() {
            out.base_url = parsed.base_url;
        }
        if !parsed.api_key.is_empty() && !parsed.api_key.contains('$') {
            out.api_key = parsed.api_key;
        }
        if !parsed.model.is_empty() {
            out.model = parsed.model;
        }
        if out.name.is_empty() {
            out.name = parsed.name;
        }
    }
    if let Some(v) = non_empty(&body.name) {
        out.name = v;
    }
    if let Some(v) = non_empty(&body.base_url) {
        out.base_url = v;
    }
    if let Some(v) = non_empty(&body.api_key) {
        out.api_key = v;
    }
    if let Some(v) = non_empty(&body.model) {
        out.model = v;
    }
    Ok(out)
}

async fn llm_providers_save(
    State(st): State<SharedState>,
    Json(body): Json<LlmProviderBody>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    let mut g = st.lock().unwrap_or_else(|e| e.into_inner());
    let mut input = match resolve_provider_body(&g, &body) {
        Ok(v) => v,
        Err(e) => return llm_bad(StatusCode::BAD_REQUEST, e),
    };
    // Al editar, `upsert` conserva la clave si viene vacía: no reenviar la guardada.
    if input.id.is_some() && non_empty(&body.api_key).is_none() && non_empty(&body.curl).is_none() {
        input.api_key.clear();
    }
    let saved = match g.llm.upsert(input) {
        Ok(p) => p,
        Err(e) => return llm_bad(StatusCode::BAD_REQUEST, e),
    };
    tracing::info!(id = %saved.id, name = %saved.name, base = %saved.base_url, model = %saved.model, "proveedor LLM guardado");
    if body.activate == Some(true) {
        if let Err(e) = g.llm.set_active(&saved.id) {
            return llm_bad(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    }
    let mut v = llm_list_json(&g);
    v["provider"] = serde_json::to_value(saved.public()).unwrap_or_default();
    Json(v).into_response()
}

async fn llm_providers_delete(
    State(st): State<SharedState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    let mut g = st.lock().unwrap_or_else(|e| e.into_inner());
    match g.llm.remove(&id) {
        Ok(true) => {
            tracing::info!(id = %id, "proveedor LLM borrado");
            let mut v = llm_list_json(&g);
            v["removed"] = json!(true);
            Json(v).into_response()
        }
        Ok(false) => llm_bad(
            StatusCode::NOT_FOUND,
            format!("no existe el proveedor «{id}»"),
        ),
        Err(e) => llm_bad(StatusCode::BAD_REQUEST, e),
    }
}

async fn llm_providers_test(
    State(st): State<SharedState>,
    Json(body): Json<LlmProviderBody>,
) -> axum::response::Response {
    use crate::web::llm_provider::{normalize_base_url, test_provider, ProviderConfig};
    use axum::http::StatusCode;
    let input = {
        let g = st.lock().unwrap_or_else(|e| e.into_inner());
        match resolve_provider_body(&g, &body) {
            Ok(v) => v,
            Err(e) => return llm_bad(StatusCode::BAD_REQUEST, e),
        }
    };
    let base_url = match normalize_base_url(&input.base_url) {
        Ok(b) => b,
        Err(e) => return llm_bad(StatusCode::BAD_REQUEST, e),
    };
    let cfg = ProviderConfig {
        id: input.id.clone().unwrap_or_else(|| "test".into()),
        name: input.name.clone(),
        base_url,
        api_key: input.api_key.clone(),
        model: input.model.clone(),
        created_ms: 0,
        source: "test".into(),
    };
    let timeout = crate::web::llm_periphery::chat_timeout();
    let rep = tokio::task::spawn_blocking(move || test_provider(&cfg, timeout))
        .await
        .map_err(|e| e.to_string());
    match rep {
        Ok(r) => {
            tracing::info!(ok = r.ok, base = %r.base_url, models_ms = r.models_ms, chat_ms = r.chat_ms, "prueba proveedor LLM");
            Json(r).into_response()
        }
        Err(e) => llm_bad(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct LlmActiveBody {
    pub id: String,
}

async fn llm_active_get(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap_or_else(|e| e.into_inner());
    Json(llm_active_of(&g))
}

async fn llm_active_set(
    State(st): State<SharedState>,
    Json(body): Json<LlmActiveBody>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    let mut g = st.lock().unwrap_or_else(|e| e.into_inner());
    match g.llm.set_active(&body.id) {
        Ok(()) => {
            let a = llm_active_of(&g);
            tracing::info!(active = %a.id, label = %a.label, "LLM activo cambiado");
            let mut v = llm_list_json(&g);
            v["ok"] = json!(true);
            Json(v).into_response()
        }
        Err(e) => {
            let code = if e.starts_with("no existe") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            llm_bad(code, e)
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ParseCurlBody {
    pub curl: String,
}

/// Interpreta un curl pegado. Devuelve la clave tal cual (viene en la propia
/// petición del usuario) para rellenar el formulario; no se guarda ni se registra.
async fn llm_parse_curl(Json(body): Json<ParseCurlBody>) -> axum::response::Response {
    use axum::http::StatusCode;
    match crate::web::llm_provider::parse_curl(&body.curl) {
        Ok(p) => {
            let masked = crate::web::llm_provider::mask_key(&p.api_key);
            Json(json!({ "ok": true, "parsed": p, "api_key_masked": masked })).into_response()
        }
        Err(e) => llm_bad(StatusCode::BAD_REQUEST, e),
    }
}

/// Helper de tests: router con un secreto de prueba y una sesión válida
/// inyectada como `Authorization: Bearer` (si la petición no trae otra).
#[cfg(test)]
pub fn test_router(state: SharedState) -> Router {
    use axum::body::Body;
    use axum::http::Request;
    let auth = Arc::new(AuthConfig::new(
        Some("test-master-secret-0123456789-abcdefghij".into()),
        3600,
    ));
    let (tok, _) = auth.issue_session().expect("sesión de test");
    let bearer = HeaderValue::from_str(&format!("Bearer {tok}")).unwrap();
    router_with_auth(state, PathBuf::from("web/static"), auth).layer(
        tower::util::MapRequestLayer::new(move |mut req: Request<Body>| {
            if !req.headers().contains_key(header::AUTHORIZATION) {
                req.headers_mut()
                    .insert(header::AUTHORIZATION, bearer.clone());
            }
            req
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn lex_state() -> SharedState {
        let mut s = AppState::new();
        s.probe = crate::web::llm_periphery::PeripheralProbe::Lexicon(
            crate::field_linguistic_layer::GemmaShapedLexicon::new(11),
        );
        s.decoder = crate::web::llm_periphery::ConceptDecoder::new(
            crate::web::llm_periphery::LlmMode::Lexicon,
        );
        Arc::new(Mutex::new(s))
    }

    // ------------------------------------------------------------------
    // Acceso con MASTER_SECRET

    const SECRET: &str = "secreto-maestro-de-prueba-0123456789abcdef";

    fn auth_cfg(secret: Option<&str>) -> SharedAuth {
        let mut a = AuthConfig::new(secret.map(String::from), 3600);
        a.fail_delay = Duration::from_millis(0);
        Arc::new(a)
    }

    fn app_with(auth: SharedAuth) -> Router {
        router_with_auth(lex_state(), PathBuf::from("web/static"), auth)
    }

    /// Todas las familias protegidas (+ una ruta /api inexistente).
    const PROTECTED: &[(&str, &str)] = &[
        ("GET", "/api/status"),
        ("POST", "/api/chat"),
        ("GET", "/api/chat/history"),
        ("POST", "/api/chat/reset"),
        ("POST", "/api/train/start"),
        ("POST", "/api/train/stop"),
        ("GET", "/api/train/status"),
        ("GET", "/api/train/events?after=0"),
        ("GET", "/api/train/stream"),
        ("POST", "/api/sleep"),
        ("POST", "/api/sleep/start"),
        ("POST", "/api/sleep/stop"),
        ("GET", "/api/sleep/status"),
        ("GET", "/api/sleep/events?after=0"),
        ("POST", "/api/tests/run"),
        ("POST", "/api/tests/start"),
        ("POST", "/api/tests/stop"),
        ("GET", "/api/tests/last"),
        ("GET", "/api/tests/status"),
        ("GET", "/api/tests/events?after=0"),
        ("GET", "/api/processes"),
        ("GET", "/api/telemetry"),
        ("GET", "/api/telemetry/liquid"),
        ("GET", "/api/telemetry/cdt"),
        ("GET", "/api/telemetry/rqm"),
        ("GET", "/api/telemetry/train"),
        ("GET", "/api/llm/providers"),
        ("POST", "/api/llm/providers"),
        ("POST", "/api/llm/providers/test"),
        ("DELETE", "/api/llm/providers/x"),
        ("GET", "/api/llm/active"),
        ("POST", "/api/llm/active"),
        ("POST", "/api/llm/parse-curl"),
        ("GET", "/api/spider/sample"),
        ("POST", "/api/spider/start"),
        ("POST", "/api/spider/stop"),
        ("GET", "/api/spider/status"),
        ("GET", "/api/spider/events?after=0"),
        ("GET", "/api/spider/stream"),
        ("POST", "/api/spider/decide"),
        ("GET", "/api/spider/runs"),
        ("GET", "/api/spider/export"),
        ("GET", "/api/no-existe"),
    ];

    fn req(method: &str, uri: &str, headers: &[(&str, &str)], body: &str) -> Request<Body> {
        let mut b = Request::builder()
            .method(method)
            .uri(uri)
            .header("host", "app.test")
            .header("content-type", "application/json");
        for (k, v) in headers {
            b = b.header(*k, *v);
        }
        b.body(Body::from(body.to_string())).unwrap()
    }

    async fn send(app: &Router, r: Request<Body>) -> (StatusCode, HeaderMap, serde_json::Value) {
        let res = app.clone().oneshot(r).await.unwrap();
        let status = res.status();
        let headers = res.headers().clone();
        let is_sse = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        if is_sse {
            return (status, headers, serde_json::Value::Null);
        }
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, headers, v)
    }

    async fn login_as(
        app: &Router,
        secret: &str,
        ip: &str,
    ) -> (StatusCode, HeaderMap, serde_json::Value) {
        let body = json!({ "secret": secret }).to_string();
        send(
            app,
            req("POST", "/api/auth/login", &[("x-forwarded-for", ip)], &body),
        )
        .await
    }

    fn cookie_of(h: &HeaderMap) -> String {
        let c = h.get("set-cookie").unwrap().to_str().unwrap();
        c.split(';').next().unwrap().to_string()
    }

    use axum::http::HeaderMap;

    #[tokio::test]
    async fn auth_unauthenticated_every_protected_route_rejected() {
        let auth = auth_cfg(Some(SECRET));
        let app = app_with(auth.clone());
        let other = AuthConfig::new(Some("otro-secreto-distinto-0123456789abcdef".into()), 3600);
        let (old_secret_tok, _) = other.issue_session().unwrap();
        let (expired_tok, _) = auth
            .issue_session_at(crate::web::train_job::now_ms() / 1000 - 7200, 3600)
            .unwrap();
        let (good, _) = auth.issue_session().unwrap();
        let tampered = format!("{}x", &good[..good.len() - 1]);
        let bad_tokens = [
            None,
            Some("ngs1.basura.basura".to_string()),
            Some(tampered),
            Some(expired_tok),
            Some(old_secret_tok),
        ];
        for (m, path) in PROTECTED {
            for tok in &bad_tokens {
                for via_cookie in [false, true] {
                    let hv;
                    let headers: Vec<(&str, &str)> = match tok {
                        None => vec![],
                        Some(t) if via_cookie => {
                            hv = format!("ngs_session={t}");
                            vec![("cookie", hv.as_str())]
                        }
                        Some(t) => {
                            hv = format!("Bearer {t}");
                            vec![("authorization", hv.as_str())]
                        }
                    };
                    let (st, h, v) = send(&app, req(m, path, &headers, "{}")).await;
                    assert_eq!(st, StatusCode::UNAUTHORIZED, "{m} {path} tok={tok:?}");
                    assert_eq!(v["code"], "unauthorized", "{m} {path}");
                    assert_eq!(h.get("www-authenticate").unwrap(), "Bearer");
                }
            }
        }
        // Con sesión válida: la ruta ya no es 401.
        let bearer = format!("Bearer {good}");
        for (m, path) in [
            ("GET", "/api/status"),
            ("GET", "/api/telemetry"),
            ("GET", "/api/llm/providers"),
            ("GET", "/api/train/stream"),
        ] {
            let (st, _, _) = send(&app, req(m, path, &[("authorization", &bearer)], "")).await;
            assert_eq!(st, StatusCode::OK, "{m} {path}");
        }
    }

    #[tokio::test]
    async fn auth_no_secret_fails_closed() {
        let app = app_with(auth_cfg(None));
        for (m, path) in PROTECTED {
            let (st, _, v) = send(&app, req(m, path, &[], "{}")).await;
            assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{m} {path}");
            assert_eq!(v["code"], "master_secret_not_configured");
        }
        let (st, _, v) = login_as(&app, "", "1.1.1.1").await;
        assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["code"], "master_secret_not_configured");
        let (st, _, v) = send(&app, req("GET", "/health", &[], "")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["auth"]["master_configured"], false);
        assert_eq!(v["auth"]["mode"], "unconfigured");
        assert!(!v["auth"]["warnings"].as_array().unwrap().is_empty());
        let (st, _, v) = send(&app, req("GET", "/api/auth/status", &[], "")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["master_configured"], false);
        // La UI (estáticos) sigue sirviéndose para mostrar el aviso.
        let res = app.clone().oneshot(req("GET", "/", &[], "")).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_health_is_public_and_minimal() {
        let app = app_with(auth_cfg(Some("corto")));
        let (st, _, v) = send(&app, req("GET", "/health", &[], "")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["ok"], true);
        for k in [
            "engrams",
            "model",
            "llm_active",
            "probe",
            "active_processes",
            "llm_mode",
        ] {
            assert!(v.get(k).is_none(), "/health expone {k}");
        }
        let w = v["auth"]["warnings"].to_string();
        assert!(w.contains("corto"), "{w}");
        assert!(!v.to_string().contains("\"corto\""));
        let app = app_with(auth_cfg(Some(SECRET)));
        let (_, _, v) = send(&app, req("GET", "/health", &[], "")).await;
        assert!(v["auth"]["warnings"].as_array().unwrap().is_empty());
        assert!(!v.to_string().contains(SECRET));
    }

    #[tokio::test]
    async fn auth_login_ok_cookie_bearer_and_sse() {
        let app = app_with(auth_cfg(Some(SECRET)));
        let (st, h, v) = login_as(&app, SECRET, "2.2.2.2").await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["ok"], true);
        assert!(v["expires_at"].as_u64().unwrap() > 0);
        assert!(!v.to_string().contains(SECRET));
        let raw = h.get("set-cookie").unwrap().to_str().unwrap().to_string();
        for part in ["HttpOnly", "SameSite=Strict", "Path=/api", "Max-Age=3600"] {
            assert!(raw.contains(part), "{raw}");
        }
        assert_eq!(h.get("cache-control").unwrap(), "no-store");
        let cookie = cookie_of(&h);
        let tok = v["token"].as_str().unwrap().to_string();
        assert!(tok.starts_with("ngs1."));
        // Cookie (navegador, incl. EventSource) y Bearer (scripts).
        let (st, _, v) = send(&app, req("GET", "/api/status", &[("cookie", &cookie)], "")).await;
        assert_eq!(st, StatusCode::OK);
        assert!(v["llm_mode"].as_str().is_some());
        let bearer = format!("Bearer {tok}");
        let (st, _, _) = send(
            &app,
            req("GET", "/api/telemetry", &[("authorization", &bearer)], ""),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        let (st, h, _) = send(
            &app,
            req("GET", "/api/train/stream", &[("cookie", &cookie)], ""),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert!(h["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let (_, _, v) = send(
            &app,
            req("GET", "/api/auth/status", &[("cookie", &cookie)], ""),
        )
        .await;
        assert_eq!(v["authenticated"], true);
        assert_eq!(v["session"]["via"], "cookie");
        let (_, _, v) = send(&app, req("GET", "/api/auth/status", &[], "")).await;
        assert_eq!(v["authenticated"], false);
        // Secreto con espacios alrededor se acepta (trim), como en docker-llm.
        let (st, _, _) = login_as(&app, &format!("  {SECRET}\n"), "2.2.2.3").await;
        assert_eq!(st, StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_login_fail_and_lockout() {
        let app = app_with(auth_cfg(Some(SECRET)));
        for i in 0..5 {
            let (st, h, v) = login_as(&app, "incorrecto", "3.3.3.3").await;
            assert_eq!(st, StatusCode::UNAUTHORIZED, "intento {i}");
            assert_eq!(v["code"], "invalid_secret");
            assert!(h.get("set-cookie").is_none());
        }
        // Bloqueada: ni el secreto correcto entra desde esa IP.
        let (st, h, v) = login_as(&app, SECRET, "3.3.3.3").await;
        assert_eq!(st, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(v["code"], "rate_limited");
        let wait: u64 = h["retry-after"].to_str().unwrap().parse().unwrap();
        assert!((29..=30).contains(&wait), "{wait}");
        // Otra IP no está bloqueada.
        let (st, _, _) = login_as(&app, SECRET, "4.4.4.4").await;
        assert_eq!(st, StatusCode::OK);
        // Cuerpo sin secreto = fallo normal.
        let (st, _, _) = send(
            &app,
            req(
                "POST",
                "/api/auth/login",
                &[("x-forwarded-for", "5.5.5.5")],
                "{}",
            ),
        )
        .await;
        assert_eq!(st, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_global_cap_limits_distributed_attempts() {
        let app = app_with(auth_cfg(Some(SECRET)));
        for i in 0..30 {
            let (st, _, _) = login_as(&app, "no", &format!("10.0.0.{i}")).await;
            assert_eq!(st, StatusCode::UNAUTHORIZED);
        }
        let (st, _, _) = login_as(&app, "no", "10.0.1.1").await;
        assert_eq!(st, StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn auth_logout_revokes_server_side() {
        let app = app_with(auth_cfg(Some(SECRET)));
        let (_, h, v) = login_as(&app, SECRET, "6.6.6.6").await;
        let cookie = cookie_of(&h);
        let bearer = format!("Bearer {}", v["token"].as_str().unwrap());
        let (st, _, _) = send(&app, req("GET", "/api/status", &[("cookie", &cookie)], "")).await;
        assert_eq!(st, StatusCode::OK);
        let (st, h, v) = send(
            &app,
            req("POST", "/api/auth/logout", &[("cookie", &cookie)], ""),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["revoked"], true);
        assert!(h["set-cookie"].to_str().unwrap().contains("Max-Age=0"));
        // Ni la cookie ni el mismo token por Bearer sirven ya.
        let (st, _, _) = send(&app, req("GET", "/api/status", &[("cookie", &cookie)], "")).await;
        assert_eq!(st, StatusCode::UNAUTHORIZED);
        let (st, _, _) = send(
            &app,
            req("GET", "/api/status", &[("authorization", &bearer)], ""),
        )
        .await;
        assert_eq!(st, StatusCode::UNAUTHORIZED);
        let (st, _, v) = send(&app, req("POST", "/api/auth/logout", &[], "")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["revoked"], false);
    }

    #[tokio::test]
    async fn auth_rotating_secret_invalidates_sessions() {
        let a = auth_cfg(Some(SECRET));
        let app_a = app_with(a.clone());
        let (_, h, _) = login_as(&app_a, SECRET, "7.7.7.7").await;
        let cookie = cookie_of(&h);
        let app_b = app_with(auth_cfg(Some("nuevo-secreto-rotado-0123456789abcdefgh")));
        let (st, _, _) = send(
            &app_b,
            req("GET", "/api/status", &[("cookie", &cookie)], ""),
        )
        .await;
        assert_eq!(st, StatusCode::UNAUTHORIZED);
        // El secreto antiguo tampoco entra en la app rotada.
        let (st, _, _) = login_as(&app_b, SECRET, "7.7.7.7").await;
        assert_eq!(st, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_cookie_writes_require_same_origin() {
        let app = app_with(auth_cfg(Some(SECRET)));
        let (_, h, v) = login_as(&app, SECRET, "8.8.8.8").await;
        let cookie = cookie_of(&h);
        let (st, _, v2) = send(
            &app,
            req(
                "POST",
                "/api/chat/reset",
                &[("cookie", &cookie), ("origin", "https://evil.example")],
                "{}",
            ),
        )
        .await;
        assert_eq!(st, StatusCode::FORBIDDEN);
        assert_eq!(v2["code"], "cross_origin_rejected");
        let (st, _, _) = send(
            &app,
            req(
                "POST",
                "/api/chat/reset",
                &[("cookie", &cookie), ("origin", "http://app.test")],
                "{}",
            ),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        // Bearer (scripts) no depende de Origin.
        let bearer = format!("Bearer {}", v["token"].as_str().unwrap());
        let (st, _, _) = send(
            &app,
            req(
                "POST",
                "/api/chat/reset",
                &[
                    ("authorization", &bearer),
                    ("origin", "https://evil.example"),
                ],
                "{}",
            ),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
    }

    #[tokio::test]
    async fn health_ok_lexicon() {
        let app = test_router(lex_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);
        assert!(v["llm_mode"].as_str().is_some());
    }

    async fn post_chat(app: Router, body: &str) -> (StatusCode, serde_json::Value) {
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn chat_default_mode_is_field_decoder() {
        let app = test_router(lex_state());
        let (st, v) = post_chat(app, r#"{"message":"hola campo"}"#).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["mode"], "field_decoder");
        let (st2, v2) = post_chat(
            test_router(lex_state()),
            r#"{"message":"hola campo","mode":"field_decoder"}"#,
        )
        .await;
        assert_eq!(st2, StatusCode::OK);
        assert_eq!(v2["mode"], "field_decoder");
    }

    #[tokio::test]
    async fn chat_raw_without_gguf_errors_and_skips_field() {
        let state = lex_state();
        let (liq0, rqm0) = {
            let g = state.lock().unwrap();
            (g.metrics.liquid_queries, g.fuse.rqm_infer_calls)
        };
        let app = test_router(state.clone());
        let (st, v) = post_chat(app.clone(), r#"{"message":"hola","mode":"gemma_raw"}"#).await;
        assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["mode"], "gemma_raw");
        assert_eq!(v["route"], "gemma_raw_error");
        assert!(v["reply"].as_str().unwrap().contains("GGUF"));
        // Bool alternativo: field_decoder=false → crudo.
        let (st2, v2) = post_chat(app, r#"{"message":"hola","field_decoder":false}"#).await;
        assert_eq!(st2, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v2["mode"], "gemma_raw");
        let g = state.lock().unwrap();
        // No cae al campo: sin infer líquido/RQM.
        assert_eq!(g.metrics.liquid_queries, liq0);
        assert_eq!(g.fuse.rqm_infer_calls, rqm0);
        assert_eq!(g.raw_chat_calls, 2);
        assert!(g
            .chat_log
            .iter()
            .all(|t| t.mode.as_deref() == Some("gemma_raw")));
    }

    #[tokio::test]
    async fn chat_reset_clears_history_and_raw_context() {
        let st = lex_state();
        let app = test_router(st.clone());
        let _ = post_chat(app.clone(), r#"{"message":"hola campo"}"#).await;
        {
            let mut g = st.lock().unwrap();
            // Simula un par crudo previo (contexto que OFF reenvía).
            let ok = Ok(crate::field_gemma_probe::RawGemmaReply {
                text: "¡Hola!".into(),
                prompt_tokens: 3,
                generated_tokens: 2,
                seconds: 0.1,
            });
            g.record_raw_chat("hola", &ok);
            assert_eq!(g.raw_history().len(), 1);
            assert!(g.chat_log.len() >= 4);
        }
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/reset")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);
        assert!(v["cleared"].as_u64().unwrap() >= 4);
        let g = st.lock().unwrap();
        assert!(g.chat_log.is_empty());
        assert!(g.raw_history().is_empty());
    }

    #[tokio::test]
    async fn chat_unknown_mode_is_bad_request() {
        let app = test_router(lex_state());
        let (st, v) = post_chat(app, r#"{"message":"hola","mode":"xyz"}"#).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(v["ok"], false);
    }

    #[tokio::test]
    async fn chat_and_telemetry_shape() {
        let app = test_router(lex_state());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"message":"hola líquido"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["reply"].as_str().unwrap().len() > 0);
        assert!(v.get("concept_in").is_some());
        assert!(v.get("liquid_score").is_some());
        assert!(v.get("decoded").is_some());

        let res2 = app
            .oneshot(
                Request::builder()
                    .uri("/api/telemetry")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res2.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res2.into_body(), usize::MAX)
            .await
            .unwrap();
        let t: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(t.get("liquid").is_some());
        assert!(t.get("cdt").is_some());
        assert!(t.get("rqm").is_some());
        assert!(t.get("train").is_some());
        assert!(t.get("live_job").is_some());
    }

    #[tokio::test]
    async fn train_start_async_increases_engrams() {
        let st = lex_state();
        let before = st.lock().unwrap().fuse.engram_count();
        let app = test_router(st.clone());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/train/start")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"batches":2,"batch_size":4,"epochs":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);
        assert!(v["job_id"].as_str().unwrap().len() > 0);

        // Esperar a que el worker termine.
        for _ in 0..200 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let g = st.lock().unwrap();
            if !g.train_job.running {
                break;
            }
        }
        let after = st.lock().unwrap().fuse.engram_count();
        assert!(after > before, "engrams {after} <= {before}");

        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/train/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(status.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(status.into_body(), usize::MAX)
            .await
            .unwrap();
        let s: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(s["running"], false);
        assert!(s["events"].as_array().unwrap().len() > 0);
        assert!(s.get("engrams").is_some());

        let ev = app
            .oneshot(
                Request::builder()
                    .uri("/api/train/events?after=0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ev.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn train_start_omitted_batches_is_infinite_then_stop() {
        let st = lex_state();
        let app = test_router(st.clone());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/train/start")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"batch_size":3,"epochs":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["infinite"], true);

        // Dejar correr al menos un lote.
        for _ in 0..80 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let g = st.lock().unwrap();
            if g.train_job.current_batch >= 1 || g.train_job.datasets_saved >= 1 {
                break;
            }
        }
        {
            let g = st.lock().unwrap();
            assert!(g.train_job.infinite);
            assert!(g.train_job.total_batches.is_none());
            assert!(g.train_job.running || g.train_job.datasets_saved >= 1);
        }

        let stop = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/train/stop")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stop.status(), StatusCode::OK);

        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let g = st.lock().unwrap();
            if !g.train_job.running {
                break;
            }
        }
        let g = st.lock().unwrap();
        assert!(!g.train_job.running);
        assert!(g.train_job.cancelled || g.train_job.datasets_saved >= 1);
    }

    #[tokio::test]
    async fn sleep_job_async_status_events_and_report() {
        let st = lex_state();
        {
            let mut g = st.lock().unwrap();
            let cands = g.candidates();
            for i in 0..4 {
                g.fuse.observe(i, &cands);
                g.fuse.teach_relation(i, (i + 1) % 8);
            }
            let _ = g.fuse.sleep_consolidate();
            g.ever_trained = true;
        }
        let app = test_router(st.clone());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sleep/start")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"prune_intensity":0.5,"compact_intensity":0.5,"infinite":false,"cycles":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);
        assert!(v["job_id"].as_str().unwrap().len() > 0);

        for _ in 0..200 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if !st.lock().unwrap().sleep_job.running {
                break;
            }
        }
        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/sleep/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(status.into_body(), usize::MAX)
            .await
            .unwrap();
        let s: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(s["running"], false);
        assert_eq!(s["infinite"], false);
        assert_eq!(s["current_cycle"], 1);
        assert_eq!(s["total_cycles"], 1);
        assert!(s["events"].as_array().unwrap().len() > 0);
        assert!(s.get("last_report").is_some());
        assert!(s["last_report"].get("free_energy_before").is_some());

        let ev = app
            .oneshot(
                Request::builder()
                    .uri("/api/sleep/events?after=0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ev.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn tests_job_async_status_events_and_last() {
        let st = lex_state();
        let app = test_router(st.clone());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/tests/start")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"infinite":false,"cycles":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);

        for _ in 0..200 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if !st.lock().unwrap().tests_job.running {
                break;
            }
        }
        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/tests/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(status.into_body(), usize::MAX)
            .await
            .unwrap();
        let s: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(s["running"], false);
        assert!(s["events"].as_array().unwrap().len() > 0);
        assert!(s["last_report"].get("identity_accuracy").is_some());

        let last = app
            .oneshot(
                Request::builder()
                    .uri("/api/tests/last")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(last.into_body(), usize::MAX)
            .await
            .unwrap();
        let l: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(l["running"], false);
        assert!(l["last"].get("identity_accuracy").is_some());
    }

    #[tokio::test]
    async fn sleep_infinite_runs_until_stop() {
        let st = lex_state();
        {
            let mut g = st.lock().unwrap();
            let cands = g.candidates();
            for i in 0..4 {
                g.fuse.observe(i, &cands);
                g.fuse.teach_relation(i, (i + 1) % 8);
            }
            let _ = g.fuse.sleep_consolidate();
            g.ever_trained = true;
        }
        let app = test_router(st.clone());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sleep/start")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"prune_intensity":0.4,"compact_intensity":0.4,"infinite":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["infinite"], true);

        for _ in 0..120 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let g = st.lock().unwrap();
            if g.sleep_job.current_cycle >= 1 || g.sleep_job.event_seq > 2 {
                break;
            }
        }
        {
            let g = st.lock().unwrap();
            assert!(g.sleep_job.infinite);
            assert!(g.sleep_job.total_cycles.is_none());
            assert!(g.sleep_job.running || g.sleep_job.current_cycle >= 1);
        }

        let stop = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sleep/stop")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stop.status(), StatusCode::OK);

        for _ in 0..120 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if !st.lock().unwrap().sleep_job.running {
                break;
            }
        }
        let g = st.lock().unwrap();
        assert!(!g.sleep_job.running);
        assert!(g.sleep_job.cancelled || g.sleep_job.current_cycle >= 1);
    }

    #[tokio::test]
    async fn tests_infinite_runs_until_stop() {
        let st = lex_state();
        let app = test_router(st.clone());
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/tests/start")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"infinite":true,"cycles":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["infinite"], true);

        for _ in 0..120 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let g = st.lock().unwrap();
            if g.tests_job.current_cycle >= 1 || g.tests_job.event_seq > 2 {
                break;
            }
        }
        {
            let g = st.lock().unwrap();
            assert!(g.tests_job.infinite);
            assert!(g.tests_job.running || g.tests_job.current_cycle >= 1);
        }

        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/tests/stop")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();

        for _ in 0..120 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if !st.lock().unwrap().tests_job.running {
                break;
            }
        }
        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/tests/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(status.into_body(), usize::MAX)
            .await
            .unwrap();
        let s: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(s["running"], false);
        assert_eq!(s["infinite"], true);
    }

    #[tokio::test]
    async fn train_reconnect_status_keeps_running_with_events() {
        let st = lex_state();
        let app = test_router(st.clone());
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/train/start")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"batches":null,"batch_size":3,"epochs":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Esperar al menos un evento / lote.
        for _ in 0..80 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let g = st.lock().unwrap();
            if g.train_job.event_seq > 0 {
                break;
            }
        }

        // "Reconnect": status debe seguir running (o haber avanzado) con buffer.
        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/train/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(status.into_body(), usize::MAX)
            .await
            .unwrap();
        let s: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(s["events"].as_array().unwrap().len() > 0);
        assert!(s["running"] == true || s["event_seq"].as_u64().unwrap() > 0);

        let procs = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/processes")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(procs.into_body(), usize::MAX)
            .await
            .unwrap();
        let p: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(p.get("train").is_some());
        assert!(p.get("sleep").is_some());
        assert!(p.get("tests").is_some());
        assert!(p.get("active").is_some());

        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(health.into_body(), usize::MAX)
            .await
            .unwrap();
        let h: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(h.get("sleeping").is_some());
        assert!(h.get("testing").is_some());
        assert!(h.get("active_processes").is_some());

        let _ = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/train/stop")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if !st.lock().unwrap().train_job.running {
                break;
            }
        }
    }

    #[tokio::test]
    async fn live_train_start_response_shape() {
        let _ = crate::web::train_job::LiveTrainStartResponse {
            ok: true,
            job_id: "x".into(),
            message: "ok".into(),
            infinite: Some(true),
        };
    }

    // ------------------------------------------------ proveedores LLM (mock OpenAI)

    const GOOD_KEY: &str = "obk1.mockKEY0123456789abcdef";

    /// Servidor OpenAI-compatible mínimo en 127.0.0.1:<libre>. Devuelve la base `/v1`.
    async fn spawn_mock_openai() -> String {
        use axum::http::HeaderMap;
        async fn authed(h: &HeaderMap) -> bool {
            h.get("authorization").and_then(|v| v.to_str().ok())
                == Some(&format!("Bearer {GOOD_KEY}"))
        }
        fn unauthorized() -> axum::response::Response {
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": {"code": "invalid_api_key", "message": "API key inválida"}})),
            )
                .into_response()
        }
        async fn models(h: HeaderMap) -> axum::response::Response {
            if !authed(&h).await {
                return unauthorized();
            }
            Json(json!({"object": "list", "data": [{"id": "mock-model", "object": "model"}]}))
                .into_response()
        }
        async fn chat(h: HeaderMap, Json(b): Json<serde_json::Value>) -> axum::response::Response {
            if !authed(&h).await {
                return unauthorized();
            }
            let msgs = b["messages"].as_array().cloned().unwrap_or_default();
            let sys = msgs
                .iter()
                .find(|m| m["role"] == "system")
                .and_then(|m| m["content"].as_str())
                .unwrap_or("")
                .to_string();
            let last = msgs
                .last()
                .and_then(|m| m["content"].as_str())
                .unwrap_or("")
                .to_string();
            if last.contains("lento") {
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
            let content = if sys.contains("Generas datos") {
                let n = last
                    .lines()
                    .filter(|l| l.chars().next().is_some_and(|c| c.is_ascii_digit()))
                    .count();
                (1..=n)
                    .map(|i| format!("{i}. frase externa {i}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else if sys.contains("decoder de un modelo de campo") {
                format!("<think>x</think>MOCK-DECODER: {last}")
            } else if sys.contains(crate::prompt_spider::LLM_SYSTEM_MARK) {
                // Prompt Spider: veredicto JSON por palabra. «vague» → duda;
                // «Every» → pide aprobación; prompt con MOCK-GARBAGE → sin JSON.
                if last.contains("MOCK-GARBAGE") {
                    "no tengo idea, lo siento".to_string()
                } else {
                    let verdicts: Vec<serde_json::Value> = last
                        .lines()
                        .filter_map(|l| {
                            let mut parts = l.split(" | ");
                            let i: usize = parts.next()?.trim().parse().ok()?;
                            let w = parts.next()?.trim().to_string();
                            Some(match w.as_str() {
                                "vague" => json!({"i": i, "relevant": true, "grounded": true,
                                    "ambiguous": true, "needs_approval": false, "p": 0.55}),
                                "Every" => json!({"i": i, "relevant": true, "grounded": false,
                                    "ambiguous": false, "needs_approval": true, "p": 0.8}),
                                _ => json!({"i": i, "relevant": true, "grounded": "yes",
                                    "ambiguous": false, "needs_approval": false, "p": 0.91,
                                    "note": format!("mock ok {w}")}),
                            })
                        })
                        .collect();
                    format!(
                        "<think>pensando</think>```json\n{}\n```",
                        json!({ "verdicts": verdicts })
                    )
                }
            } else {
                format!("MOCK-RAW({} msgs): {last}", msgs.len())
            };
            Json(json!({
                "id": "c1", "object": "chat.completion", "model": b["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
            }))
            .into_response()
        }
        let app = Router::new()
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}/v1")
    }

    async fn call_json(
        app: Router,
        method: &str,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(match body {
                Some(b) => Body::from(b.to_string()),
                None => Body::empty(),
            })
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn llm_provider_crud_test_select_chat_and_train_via_mock() {
        let base = spawn_mock_openai().await;
        let state = lex_state();
        let app = test_router(state.clone());

        // Default: Gemma local.
        let (st, v) = call_json(app.clone(), "GET", "/api/llm/providers", None).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["active"], "gemma_local");
        assert_eq!(v["providers"].as_array().unwrap().len(), 0);

        // Parse curl (docker-llm, con export).
        let curl = format!(
            "export BASE={}\nexport API_KEY=\"{GOOD_KEY}\"\ncurl $BASE/v1/chat/completions \\\n  -H \"Authorization: Bearer $API_KEY\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{{\"model\":\"mock-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hola\"}}]}}'",
            base.trim_end_matches("/v1")
        );
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/llm/parse-curl",
            Some(json!({ "curl": curl })),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["parsed"]["base_url"], base);
        assert_eq!(v["parsed"]["model"], "mock-model");
        assert_eq!(v["parsed"]["api_key"], GOOD_KEY);

        // Probar conexión (sin guardar) OK y con clave mala → 401 claro.
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/llm/providers/test",
            Some(json!({ "curl": curl })),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["ok"], true, "{v}");
        assert_eq!(v["model_listed"], true);
        assert!(v["reply_preview"].as_str().unwrap().contains("MOCK-RAW"));
        let (_, v) = call_json(
            app.clone(),
            "POST",
            "/api/llm/providers/test",
            Some(json!({ "base_url": base, "api_key": "obk1.bogus-key-000000000", "model": "mock-model" })),
        )
        .await;
        assert_eq!(v["ok"], false);
        let err = v["error"].as_str().unwrap();
        assert!(err.contains("401") && err.contains("API key"), "{err}");
        assert!(!err.contains("bogus-key"));

        // Guardar desde curl + nombre manual; la respuesta no trae la clave.
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/llm/providers",
            Some(json!({ "curl": curl, "name": "mock" })),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert!(!v.to_string().contains(GOOD_KEY));
        let id = v["provider"]["id"].as_str().unwrap().to_string();
        assert_eq!(v["provider"]["name"], "mock");
        assert_eq!(v["active"], "gemma_local");

        // Probar guardado por id.
        let (_, v) = call_json(
            app.clone(),
            "POST",
            "/api/llm/providers/test",
            Some(json!({ "id": id })),
        )
        .await;
        assert_eq!(v["ok"], true, "{v}");

        // Activar.
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/llm/active",
            Some(json!({ "id": id })),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["active"], id);
        let (st, _) = call_json(
            app.clone(),
            "POST",
            "/api/llm/active",
            Some(json!({ "id": "nope" })),
        )
        .await;
        assert_eq!(st, StatusCode::NOT_FOUND);
        let (_, h) = call_json(app.clone(), "GET", "/api/status", None).await;
        assert_eq!(h["llm_active"]["id"], id);
        assert_eq!(h["llm_active"]["kind"], "openai_compatible");

        // Chat crudo (OFF) → API externa, con historial.
        let (st, v) = post_chat(app.clone(), r#"{"message":"hola api","mode":"gemma_raw"}"#).await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["mode"], "gemma_raw");
        assert!(v["reply"]
            .as_str()
            .unwrap()
            .starts_with("MOCK-RAW(1 msgs): hola api"));
        assert_eq!(v["llm_id"], id);
        assert!(v["llm"].as_str().unwrap().contains("mock"));
        let (_, v) = post_chat(app.clone(), r#"{"message":"segunda","mode":"gemma_raw"}"#).await;
        assert!(
            v["reply"].as_str().unwrap().starts_with("MOCK-RAW(3 msgs)"),
            "{v}"
        );

        // Chat ON (decoder del campo) → el campo decide y la API verbaliza.
        let (liq0, _) = {
            let g = state.lock().unwrap();
            (g.metrics.liquid_queries, 0)
        };
        let (st, v) = post_chat(
            app.clone(),
            r#"{"message":"hola campo","mode":"field_decoder"}"#,
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        let reply = v["reply"].as_str().unwrap();
        assert!(reply.starts_with("MOCK-DECODER: hola campo"), "{reply}");
        assert!(reply.contains("Estado del campo") && reply.contains("[campo] ruta="));
        assert!(!reply.contains("<think>"));
        assert!(v["decoded"]
            .as_str()
            .unwrap()
            .starts_with("api decoder·mock-model"));
        assert!(state.lock().unwrap().metrics.liquid_queries > liq0);

        // Entrenamiento: dataset generado por la API externa.
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/train/start",
            Some(json!({ "batches": 1, "batch_size": 4, "epochs": 1 })),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["ok"], true);
        for _ in 0..400 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if !state.lock().unwrap().train_job.running {
                break;
            }
        }
        {
            let g = state.lock().unwrap();
            assert_eq!(g.train_job.dataset_source.as_deref(), Some("api:mock"));
            assert!(g
                .train_job
                .events
                .iter()
                .any(|e| e.kind == "dataset" && e.message.contains("source=api:mock")));
        }

        // Nuevo chat limpia el contexto que se reenvía a la API.
        let (st, _) = call_json(app.clone(), "POST", "/api/chat/reset", Some(json!({}))).await;
        assert_eq!(st, StatusCode::OK);
        let (_, v) = post_chat(
            app.clone(),
            r#"{"message":"tras reset","mode":"gemma_raw"}"#,
        )
        .await;
        assert!(
            v["reply"].as_str().unwrap().starts_with("MOCK-RAW(1 msgs)"),
            "{v}"
        );

        // Borrar el activo → vuelve a Gemma local.
        let (st, v) = call_json(
            app.clone(),
            "DELETE",
            &format!("/api/llm/providers/{id}"),
            None,
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["active"], "gemma_local");
        let (st, _) = call_json(
            app.clone(),
            "DELETE",
            &format!("/api/llm/providers/{id}"),
            None,
        )
        .await;
        assert_eq!(st, StatusCode::NOT_FOUND);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn llm_external_failure_is_clear_error_and_field_mode_falls_back() {
        let base = spawn_mock_openai().await;
        let state = lex_state();
        {
            let mut g = state.lock().unwrap();
            let p = g
                .llm
                .upsert(crate::web::llm_provider::ProviderInput {
                    id: None,
                    name: "mala".into(),
                    base_url: base.clone(),
                    api_key: "obk1.revocada-000000000000".into(),
                    model: "mock-model".into(),
                })
                .unwrap();
            g.llm.set_active(&p.id).unwrap();
        }
        let app = test_router(state.clone());
        let (st, v) = post_chat(app.clone(), r#"{"message":"hola","mode":"gemma_raw"}"#).await;
        assert_eq!(st, StatusCode::BAD_GATEWAY);
        assert_eq!(v["route"], "gemma_raw_error");
        let r = v["reply"].as_str().unwrap();
        assert!(r.contains("401") && r.contains("API · mala"), "{r}");
        // Error no entra al historial reenviado.
        assert!(state.lock().unwrap().raw_history().is_empty());
        // ON: sin Gemma local → decoder léxico con el motivo.
        let (st, v) = post_chat(
            app.clone(),
            r#"{"message":"hola campo","mode":"field_decoder"}"#,
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        let r = v["reply"].as_str().unwrap();
        assert!(r.contains("no respondió") && r.contains("401"), "{r}");
        assert_eq!(v["llm"], "decoder léxico");
    }

    #[test]
    fn openai_client_timeout_and_connection_errors_are_clear() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let base = rt.block_on(spawn_mock_openai());
        let cfg = crate::web::llm_provider::ProviderConfig {
            id: "t".into(),
            name: "t".into(),
            base_url: base,
            api_key: GOOD_KEY.into(),
            model: "mock-model".into(),
            created_ms: 0,
            source: "test".into(),
        };
        let c = crate::web::llm_provider::OpenAiClient::new(&cfg, Duration::from_secs(1));
        let gen = crate::web::llm_provider::ExternalGenConfig {
            max_tokens: 8,
            temperature: 0.0,
            top_p: 1.0,
        };
        let e = c
            .chat(
                &[crate::web::llm_provider::ChatMessage::new("user", "lento")],
                gen,
            )
            .unwrap_err();
        assert!(e.contains("tiempo de espera"), "{e}");
        assert_eq!(c.list_models().unwrap(), vec!["mock-model".to_string()]);
        // Modelo vacío → usa el primero de /v1/models.
        let mut no_model = cfg.clone();
        no_model.model.clear();
        let r = crate::web::llm_provider::OpenAiClient::new(&no_model, Duration::from_secs(5))
            .chat(
                &[crate::web::llm_provider::ChatMessage::new("user", "x")],
                gen,
            )
            .unwrap();
        assert_eq!(r.model, "mock-model");
        assert_eq!(r.completion_tokens, 7);
        let mut down = cfg;
        down.base_url = "http://127.0.0.1:9/v1".into();
        let e = crate::web::llm_provider::OpenAiClient::new(&down, Duration::from_secs(2))
            .list_models()
            .unwrap_err();
        assert!(e.contains("no se pudo conectar"), "{e}");
        assert!(!e.contains(GOOD_KEY));
    }

    async fn spider_wait_done(app: &Router) -> serde_json::Value {
        for _ in 0..400 {
            let (_, v) = call_json(app.clone(), "GET", "/api/spider/status", None).await;
            if v["running"] == false {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("la corrida del spider no terminó");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_spider_run_with_mock_llm_escalation_and_approval() {
        use crate::prompt_spider::{dry_run, Route, DEFAULT_THRESHOLD, SAMPLE_PROMPT};
        let base = spawn_mock_openai().await;
        let state = lex_state();
        let app = test_router(state.clone());

        // 1) Gemma local sin GGUF: las escaladas quedan pendientes (no inventa).
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "pace_ms": 0 })),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["llm"]["kind"], "local");
        let v = spider_wait_done(&app).await;
        let s = &v["summary"];
        let (_, dry) = dry_run(SAMPLE_PROMPT, DEFAULT_THRESHOLD);
        let n_llm = dry.iter().filter(|d| d.route == Route::Llm).count() as u64;
        let n_you = dry.iter().filter(|d| d.route == Route::You).count() as u64;
        let n_code = dry.iter().filter(|d| d.route == Route::Code).count() as u64;
        assert_eq!(s["forks"].as_u64().unwrap(), dry.len() as u64);
        assert_eq!(s["code"].as_u64().unwrap(), n_code);
        assert_eq!(s["llm_escalated"].as_u64().unwrap(), n_llm);
        assert_eq!(s["llm_resolved"], 0);
        assert_eq!(s["you"].as_u64().unwrap(), n_llm + n_you);
        assert_eq!(s["coverage"], 1.0);
        assert!(s["llm_failures"].as_u64().unwrap() >= 1);
        let run = &v["run"];
        assert_eq!(run["state"], "done");
        assert!(run["llm_calls"][0]["error"]
            .as_str()
            .unwrap()
            .contains("no disponible"));

        // 2) API externa (mock) activa: escaladas resueltas por el LLM.
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/llm/providers",
            Some(json!({ "base_url": base, "api_key": GOOD_KEY, "model": "mock-model", "name": "mock" })),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        let id = v["provider"]["id"].as_str().unwrap().to_string();
        let (st, _) = call_json(
            app.clone(),
            "POST",
            "/api/llm/active",
            Some(json!({ "id": id })),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "pace_ms": 0, "batch_size": 6, "threshold": 0.95 })),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["llm"]["kind"], "openai_compatible");
        let run_id = v["run_id"].as_str().unwrap().to_string();
        // Una segunda corrida simultánea → 409 (o ya terminó → 200).
        let v = spider_wait_done(&app).await;
        let s = &v["summary"];
        assert_eq!(s["llm_escalated"].as_u64().unwrap(), n_llm);
        // vague (duda) + Every (pide aprobación) quedan para el humano.
        assert_eq!(s["llm_resolved"].as_u64().unwrap(), n_llm - 2, "{s}");
        assert_eq!(s["you"].as_u64().unwrap(), n_you + 2);
        assert_eq!(s["llm_failures"], 0);
        assert_eq!(
            s["llm_calls"].as_u64().unwrap(),
            n_llm.div_ceil(6),
            "lotes de 6"
        );
        let decisions = v["run"]["decisions"].as_array().unwrap();
        let by_word = |w: &str| decisions.iter().find(|d| d["word"] == w).cloned().unwrap();
        let vague = by_word("vague");
        assert_eq!(vague["status"], "pending");
        assert_eq!(vague["first_route"], "llm");
        assert!(vague["note"].as_str().unwrap().contains("ambigua"));
        let resolved = decisions
            .iter()
            .find(|d| d["route"] == "llm" && d["status"] == "resolved")
            .unwrap();
        assert!(resolved["resolved_by"].as_str().unwrap().contains("mock"));
        assert_eq!(resolved["final_p"], 0.91);
        let ask = by_word("ask");
        assert_eq!(ask["route"], "you");
        assert_eq!(ask["p"], 0.544);

        // Eventos: start … decision … update … done.
        let (_, ev) = call_json(app.clone(), "GET", "/api/spider/events?after=0", None).await;
        let kinds: Vec<&str> = ev["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["kind"].as_str().unwrap())
            .collect();
        assert_eq!(kinds.first(), Some(&"start"));
        assert_eq!(kinds.last(), Some(&"done"));
        assert!(kinds.contains(&"update") && kinds.contains(&"llm_call"));

        // Aprobar «ask», rechazar «vague»; repetir → 409.
        let ask_i = ask["index"].as_u64().unwrap();
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/spider/decide",
            Some(json!({ "run_id": run_id, "index": ask_i, "approve": true })),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["decision"]["status"], "approved");
        assert_eq!(v["summary"]["approved"], 1);
        let (st, _) = call_json(
            app.clone(),
            "POST",
            "/api/spider/decide",
            Some(json!({ "run_id": run_id, "index": ask_i, "approve": false })),
        )
        .await;
        assert_eq!(st, StatusCode::CONFLICT);
        let (st, v) = call_json(
            app.clone(),
            "POST",
            "/api/spider/decide",
            Some(json!({ "index": vague["index"], "approve": false })),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["summary"]["rejected"], 1);

        // Export JSON de la corrida actual.
        let (st, v) = call_json(app.clone(), "GET", "/api/spider/export", None).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["run"]["id"], run_id);
        assert_eq!(v["summary"]["approved"], 1);
        assert!(!v.to_string().contains(GOOD_KEY));
        let (st, _) = call_json(app.clone(), "GET", "/api/spider/export?id=../x", None).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);

        // 3) LLM devuelve basura → pendiente, nunca inventa.
        let (st, _) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "pace_ms": 0, "prompt": "<objective>Ship MOCK-GARBAGE posts, maybe soon.</objective>" })),
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        let v = spider_wait_done(&app).await;
        let s = &v["summary"];
        assert!(s["llm_escalated"].as_u64().unwrap() >= 2, "{s}");
        assert_eq!(s["llm_resolved"], 0);
        assert!(s["llm_failures"].as_u64().unwrap() >= 1);
        assert!(v["run"]["llm_calls"][0]["error"]
            .as_str()
            .unwrap()
            .contains("JSON"));

        // Validación.
        let (st, _) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "prompt": "   " })),
        )
        .await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
    }

    async fn spider_wait_done_secs(app: &Router, secs: u64) -> serde_json::Value {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(secs) {
            let (_, v) = call_json(app.clone(), "GET", "/api/spider/status", None).await;
            if v["running"] == false {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("la corrida del spider no terminó en {secs}s");
    }

    /// Gemma local sustituto: `P(sí)` normalizada por prompt (dos órdenes).
    fn stub_gemma(calls: Arc<std::sync::atomic::AtomicUsize>) -> crate::web::spider_job::LocalStub {
        Arc::new(move |prompt: &str| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert!(prompt.contains(crate::prompt_spider::LLM_SYSTEM_MARK));
            assert!(prompt.starts_with("<start_of_turn>user\n"));
            assert!(prompt.ends_with("\"v\":\""), "el modelo continúa el primer");
            // «vague»: duda solo con «no» primero (sesgo de orden) → pendiente.
            if prompt.contains("«vague»") && prompt.contains("<no|sí|dudo>") {
                Ok(0.4)
            } else {
                Ok(0.9)
            }
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_spider_external_down_falls_back_to_gemma_local_short_run() {
        use crate::prompt_spider::{dry_run, Route, DEFAULT_THRESHOLD, SAMPLE_PROMPT};
        let (_, dry) = dry_run(SAMPLE_PROMPT, DEFAULT_THRESHOLD);
        let n_llm = dry.iter().filter(|d| d.route == Route::Llm).count();
        assert!(
            n_llm > 6,
            "el ejemplo debe superar el límite corto: {n_llm}"
        );
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = lex_state();
        {
            let mut g = state.lock().unwrap();
            g.spider.local_stub = Some(stub_gemma(calls.clone()));
            let p = g
                .llm
                .upsert(crate::web::llm_provider::ProviderInput {
                    id: None,
                    name: "docker-llm".into(),
                    // Puerto discard: conexión rechazada al instante.
                    base_url: "http://127.0.0.1:9/v1".into(),
                    api_key: "dk-local-000000000000".into(),
                    model: "qwen".into(),
                })
                .unwrap();
            g.llm.set_active(&p.id).unwrap();
        }
        let app = test_router(state.clone());
        let (_, st) = call_json(app.clone(), "GET", "/api/spider/status", None).await;
        assert_eq!(st["llm_active"]["kind"], "openai_compatible");
        assert!(st["llm_fallback"]["label"]
            .as_str()
            .unwrap()
            .contains("Gemma local"));
        assert_eq!(st["defaults"]["mode"], "completa");

        // 1) Modo auto: la API cae → Gemma local en corrida corta.
        let (code, v) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "pace_ms": 0 })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{v}");
        assert_eq!(v["fallback_available"], true);
        let v = spider_wait_done_secs(&app, 30).await;
        let run = &v["run"];
        let s = &v["summary"];
        let fb = &run["fallback"];
        assert!(
            fb["message"]
                .as_str()
                .unwrap()
                .contains("docker-llm no disponible → Gemma local"),
            "{fb}"
        );
        assert!(fb["reason"]
            .as_str()
            .unwrap()
            .contains("no se pudo conectar"));
        assert_eq!(run["mode"], "corta");
        let llm_calls = run["llm_calls"].as_array().unwrap();
        assert_eq!(llm_calls[0]["ok"], false);
        assert_ne!(llm_calls[0]["fallback"], true);
        let local: Vec<_> = llm_calls[1..].iter().collect();
        assert_eq!(local.len(), 6, "corta: máx 6 llamadas de 1 palabra");
        assert!(local
            .iter()
            .all(|c| c["fallback"] == true && c["words"].as_array().unwrap().len() == 1));
        // Dos lecturas de P(sí) por palabra (sí primero / no primero).
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 12);
        assert_eq!(s["llm_escalated"].as_u64().unwrap() as usize, n_llm);
        assert_eq!(s["llm_capped"].as_u64().unwrap() as usize, n_llm - 6);
        assert!(s["llm_resolved"].as_u64().unwrap() >= 4, "{s}");
        assert_eq!(s["escalating"], 0);
        let decisions = run["decisions"].as_array().unwrap();
        let resolved = decisions
            .iter()
            .find(|d| d["route"] == "llm" && d["status"] == "resolved")
            .unwrap();
        assert!(resolved["resolved_by"]
            .as_str()
            .unwrap()
            .contains("respaldo"));
        assert_eq!(resolved["final_p"], 0.9);
        let capped: Vec<_> = decisions.iter().filter(|d| d["capped"] == true).collect();
        assert_eq!(capped.len(), n_llm - 6);
        assert!(capped.iter().all(|d| d["status"] == "pending"
            && d["note"]
                .as_str()
                .unwrap()
                .contains("límite de corrida corta")));
        let (_, ev) = call_json(app.clone(), "GET", "/api/spider/events?after=0", None).await;
        let evs = ev["events"].as_array().unwrap();
        assert!(evs.iter().any(|e| e["kind"] == "fallback"));
        assert_eq!(
            evs.iter().filter(|e| e["kind"] == "llm_progress").count(),
            7,
            "progreso por llamada (1 API + 6 locales)"
        );

        // 2) Completa explícita: lotes de 2 en Gemma local, sin límite.
        calls.store(0, std::sync::atomic::Ordering::SeqCst);
        let (code, _) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "pace_ms": 0, "mode": "completa" })),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let v = spider_wait_done_secs(&app, 30).await;
        let s = &v["summary"];
        assert_eq!(v["run"]["mode"], "completa");
        assert_eq!(s["llm_capped"], 0);
        // 1.ª llamada (API, lote de 8, falla) + todas las palabras en Gemma local.
        assert_eq!(
            s["llm_sent"].as_u64().unwrap() as usize,
            n_llm + n_llm.min(8),
            "{s}"
        );
        let local: Vec<_> = v["run"]["llm_calls"].as_array().unwrap()[1..].to_vec();
        assert_eq!(local.len(), n_llm.div_ceil(2));
        assert!(local
            .iter()
            .all(|c| c["words"].as_array().unwrap().len() <= 2));

        // 3) Modo inválido → 400.
        let (code, _) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "mode": "eterna" })),
        )
        .await;
        assert_eq!(code, StatusCode::BAD_REQUEST);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_spider_local_failures_go_pending_and_short_cap_override() {
        let state = lex_state();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c2 = calls.clone();
        state.lock().unwrap().spider.local_stub = Some(Arc::new(move |_p: &str| {
            c2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err("timeout (45s)".to_string())
        }));
        let app = test_router(state.clone());
        let (_, st) = call_json(app.clone(), "GET", "/api/spider/status", None).await;
        assert_eq!(st["llm_active"]["kind"], "local");
        assert_eq!(st["defaults"]["mode"], "corta");
        assert_eq!(st["defaults"]["corta"]["batch_size"], 1);
        let (code, v) = call_json(
            app.clone(),
            "POST",
            "/api/spider/start",
            Some(json!({ "pace_ms": 0, "max_escalations": 5 })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{v}");
        assert_eq!(v["mode"], "corta");
        assert_eq!(v["profile"]["max_escalations"], 5);
        let v = spider_wait_done_secs(&app, 30).await;
        let s = &v["summary"];
        // Dos fallos seguidos → resto pendiente sin consultar (no inventa).
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(s["llm_resolved"], 0);
        assert_eq!(s["llm_failures"], 2);
        assert_eq!(s["escalating"], 0);
        let decisions = v["run"]["decisions"].as_array().unwrap();
        let notes: Vec<&str> = decisions
            .iter()
            .filter(|d| d["first_route"] == "llm")
            .map(|d| d["note"].as_str().unwrap())
            .collect();
        assert!(notes
            .iter()
            .any(|n| n.contains("falló; pendiente (no se inventa)")));
        assert!(notes.iter().any(|n| n.contains("veces seguidas")));
        assert!(decisions
            .iter()
            .filter(|d| d["first_route"] == "llm")
            .all(|d| d["status"] == "pending"));
    }

    /// Medición real con el GGUF (ignorada en CI):
    /// `GEMMA2_GGUF=models/gemma-2-2b-it-Q3_K_L.gguf cargo test --release
    /// --features web --lib real_gemma_spider -- --ignored --nocapture`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn real_gemma_spider_short_and_full_runs() {
        let path = std::env::var("GEMMA2_GGUF").expect("GEMMA2_GGUF");
        let probe = crate::web::llm_periphery::open_gemma_probe(std::path::Path::new(&path))
            .expect("abrir GGUF");
        let state = lex_state();
        state.lock().unwrap().install_probe(probe);
        let app = test_router(state.clone());
        let modes = std::env::var("SPIDER_REAL_MODES").unwrap_or_else(|_| "corta,completa".into());
        for mode in modes.split(',') {
            let t0 = std::time::Instant::now();
            let (code, v) = call_json(
                app.clone(),
                "POST",
                "/api/spider/start",
                Some(json!({ "pace_ms": 0, "mode": mode })),
            )
            .await;
            assert_eq!(code, StatusCode::OK, "{v}");
            println!("[{mode}] profile {}", v["profile"]);
            let v = spider_wait_done_secs(&app, 3600).await;
            let wall = t0.elapsed().as_secs_f64();
            println!("[{mode}] summary {}", v["summary"]);
            for c in v["run"]["llm_calls"].as_array().unwrap() {
                let words: Vec<String> = c["words"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| {
                        v["run"]["tokens"][i.as_u64().unwrap() as usize]["text"]
                            .as_str()
                            .unwrap()
                            .to_string()
                    })
                    .collect();
                println!(
                    "[{mode}] call {} ok={} {:.2}s words={:?} verdicts={} err={}",
                    c["batch"],
                    c["ok"],
                    c["seconds"].as_f64().unwrap_or(0.0),
                    words,
                    c["verdicts"],
                    c["error"]
                );
            }
            for d in v["run"]["decisions"].as_array().unwrap() {
                if d["first_route"] == "llm" {
                    println!(
                        "[{mode}] «{}» {} p={} final_p={} note={}",
                        d["word"].as_str().unwrap(),
                        d["status"],
                        d["p"],
                        d["final_p"],
                        d["note"]
                    );
                }
            }
            println!("[{mode}] wall {wall:.1}s");
        }
    }
}
