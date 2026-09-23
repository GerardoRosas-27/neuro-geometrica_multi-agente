//! REST API JSON + estáticos + SSE de entrenamiento en vivo.

use crate::web::field_eval::FieldEvalReport;
use crate::web::process_job::{ProcessesSnapshot, SleepJobSnapshot, TestsJobSnapshot};
use crate::web::sleep_optimize::{SleepOptimizeOpts, SleepOptimizeReport};
use crate::web::state::{AppState, ChatRequest};
use crate::web::telemetry::{FuseReportDto, SleepReportDto, TelemetrySnapshot};
use crate::web::train_job::{LiveTrainStartRequest, TrainJobSnapshot, TrainLiveEvent};
use axum::extract::{Query, State};
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
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
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
                    let mut g = st.lock().unwrap();
                    g.run_one_live_batch()
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

pub fn router(state: SharedState, static_dir: PathBuf) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let api = Router::new()
        .route("/health", get(health))
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
        .with_state(state);

    Router::new()
        .merge(api)
        .fallback_service(ServeDir::new(static_dir).append_index_html_on_directories(true))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}

async fn health(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
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
    })
}

async fn chat(State(st): State<SharedState>, Json(body): Json<ChatRequest>) -> impl IntoResponse {
    let (resp, should_spawn) = {
        let mut g = st.lock().unwrap();
        let was_running = g.train_job.running;
        let resp = g.handle_chat(&body.message);
        let should_spawn = resp.route == "train" && g.train_job.running && !was_running;
        (resp, should_spawn)
    };
    if should_spawn {
        spawn_live_train_loop(st);
    }
    Json(resp)
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

async fn chat_history(State(st): State<SharedState>) -> impl IntoResponse {
    let g = st.lock().unwrap();
    Json(json!({ "turns": g.chat_log }))
}

/// Helper de tests: router sin estáticos (ruta inexistente OK).
pub fn test_router(state: SharedState) -> Router {
    router(state, PathBuf::from("web/static"))
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

    #[tokio::test]
    async fn health_ok_lexicon() {
        let app = test_router(lex_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/health")
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
                    .uri("/health")
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
}
