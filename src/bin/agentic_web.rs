//! Binario agentico para Railway / local.
//!
//! ```bash
//! cargo run --features web --bin agentic_web
//! ```
//!
//! Escucha `0.0.0.0:$PORT` (default 8080). UI en `/`. GGUF opcional vía `GEMMA2_GGUF`.

use cdt_rqm_epr::web::{router, AppState};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info".into()),
        )
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let static_dir = resolve_static_dir();
    tracing::info!(?static_dir, "sirviendo UI estática");

    let state = Arc::new(Mutex::new(AppState::new()));
    {
        let g = state.lock().unwrap();
        tracing::info!(
            llm_mode = g.llm_mode().as_str(),
            probe = g.probe.name(),
            "AppState listo (tokens nunca entran a FieldState)"
        );
    }

    let app = router(state, static_dir);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!(%addr, "agentic_web escuchando");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind PORT");
    axum::serve(listener, app).await.expect("serve");
}

fn resolve_static_dir() -> PathBuf {
    // Prefer cwd/web/static (desarrollo y Docker COPY).
    let candidates = [
        PathBuf::from("web/static"),
        PathBuf::from("/app/web/static"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("web/static"),
    ];
    for c in &candidates {
        if c.join("index.html").is_file() {
            return c.clone();
        }
    }
    PathBuf::from("web/static")
}
