//! REST API JSON + estáticos para la UI agentica.

use crate::web::state::{AppState, ChatRequest, TrainStartRequest};
use crate::web::telemetry::{FuseReportDto, SleepReportDto, TelemetrySnapshot};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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
        .route("/api/sleep", post(sleep))
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
    Json(HealthResponse {
        ok: true,
        llm_mode: g.llm_mode().as_str().into(),
        engrams: g.fuse.engram_count(),
        wake_buffer: g.fuse.wake_buffer_len(),
        training: g.training,
        probe: g.probe.name().into(),
    })
}

async fn chat(State(st): State<SharedState>, Json(body): Json<ChatRequest>) -> impl IntoResponse {
    let mut g = st.lock().unwrap();
    Json(g.handle_chat(&body.message))
}

async fn train_start(
    State(st): State<SharedState>,
    Json(body): Json<TrainStartRequest>,
) -> impl IntoResponse {
    let mut g = st.lock().unwrap();
    Json(g.train_start(body.epochs, body.concepts))
}

async fn sleep(State(st): State<SharedState>) -> impl IntoResponse {
    let mut g = st.lock().unwrap();
    let s = g.sleep_now();
    Json(SleepReportDto::from(&s))
}

fn build_full(g: &AppState) -> FullTelemetry {
    let last_route = g.last_fuse_dto().map(|f| f.route);
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
        },
        last_fuse: g.last_fuse_dto(),
        last_sleep: g.last_sleep_dto(),
        train_log_tail: g.train_log.iter().rev().take(50).cloned().collect(),
        llm_mode: g.llm_mode().as_str().into(),
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
    }

    #[tokio::test]
    async fn train_start_increases_engrams() {
        let st = lex_state();
        let before = st.lock().unwrap().fuse.engram_count();
        let app = test_router(st.clone());
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/train/start")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"epochs":2,"concepts":[0,1,2]}"#))
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
        let after = v["engrams"].as_u64().unwrap() as usize;
        assert!(after > before);
    }
}
