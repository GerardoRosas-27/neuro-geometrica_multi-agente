//! Backend ÓRBITA para la GUI (`/api/orbita/*`).
//!
//! ÓRBITA vive en el crate aparte `orbita/` (binario, sin API de biblioteca) y
//! su pila completa (Gemma capa 13 → D2 + numnorm → ÓRBITA-v1 → yo →
//! re-observación) se invoca por proceso: `python orbita/llm/demo.py ... --json`.
//! Nada de esto está en la imagen Docker/Railway (no hay Python, pesos, GGUF ni
//! binario `orbita`), así que [`availability`] comprueba cada pieza y la GUI
//! muestra «ÓRBITA no disponible» con la lista de faltantes en vez de romperse.
//!
//! Configuración (entorno; los valores por defecto son los del box de
//! investigación):
//! - `ORBITA_PYTHON` (def. `python3`): intérprete con numpy + torch.
//! - `ORBITA_ROOT` (def. directorio actual): raíz del repo (contiene `orbita/llm/demo.py`).
//! - `ORBITA_BIN` (def. `$ORBITA_ROOT/orbita/target/release/orbita`).
//! - `ORBITA_DATA` (def. `/workspace/orbita-data`): pesos por seed, encoder D2, calibración del yo.
//! - `GEMMA_GGUF_PATH` (o, si falta, `GEMMA2_GGUF` de la GUI): Gemma 2 2B Q4_K_M (encoder capa 13).
//! - `ORBITA_TIMEOUT_S` (def. 180): plazo por petición.
//!
//! LAT/LAT_EP en vivo necesita un servicio con estado persistente; aquí sólo
//! hay un stub documentado (`/api/orbita/lat` → 501).

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::web::api::SharedState;

/// Familias en el orden del encoder (`orbita/llm/make_texts.py`, `FAM["es"]`).
pub const FAMILIES: [&str; 8] = [
    "traslación",
    "rotación",
    "reflexión",
    "escalado",
    "cizalla",
    "transformación afín",
    "rotación con escalado",
    "deformación radial",
];

/// Estado inicial por defecto de `demo.py` (hexágono, círculo k=6, r=0,8).
pub const X0_DEFAULT: [f64; 12] = [
    0.8, 0.0, 0.4, 0.69282, -0.4, 0.69282, -0.8, 0.0, -0.4, -0.69282, 0.4, -0.69282,
];

/// Rango de entrenamiento: `numnorm.parse2` recorta a [-1, 1].
const MAX_ABS_PARAM: f64 = 1.0;
const MAX_ABS_COORD: f64 = 100.0;
const MAX_TEXT: usize = 500;

#[derive(Debug, Clone)]
pub struct OrbitaConfig {
    pub python: String,
    pub root: PathBuf,
    pub bin: PathBuf,
    pub data: PathBuf,
    pub gemma: PathBuf,
    pub timeout: Duration,
}

impl OrbitaConfig {
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let root = env("ORBITA_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let bin = env("ORBITA_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("orbita/target/release/orbita"));
        Self {
            python: env("ORBITA_PYTHON").unwrap_or_else(|| "python3".into()),
            bin,
            data: PathBuf::from(
                env("ORBITA_DATA").unwrap_or_else(|| "/workspace/orbita-data".into()),
            ),
            gemma: PathBuf::from(
                env("GEMMA_GGUF_PATH")
                    .or_else(|| env("GEMMA2_GGUF"))
                    .unwrap_or_else(|| {
                        "/workspace/neuro-geometrica_multi-agente/models/gemma-2-2b-it-Q4_K_M.gguf"
                            .into()
                    }),
            ),
            timeout: Duration::from_secs(
                env("ORBITA_TIMEOUT_S")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(180),
            ),
            root,
        }
    }

    fn demo_py(&self) -> PathBuf {
        self.root.join("orbita/llm/demo.py")
    }
}

/// `dev` para seeds 0xA4xx, `confirm` para 0xB4xx (igual que `demo.py`).
fn phase(seed: &str) -> &'static str {
    if seed.to_ascii_uppercase().starts_with("0XA4") {
        "dev"
    } else {
        "confirm"
    }
}

/// Seeds con pesos publicados: 0xA400–0xA40F (DEV) y 0xB400–0xB40F (CONFIRM).
pub fn valid_seed(seed: &str) -> bool {
    let s = seed.to_ascii_uppercase();
    s.len() == 6
        && (s.starts_with("0XA40") || s.starts_with("0XB40"))
        && s.as_bytes()[5].is_ascii_hexdigit()
}

/// Piezas necesarias para una petición con `seed`. Devuelve los faltantes.
pub fn missing(cfg: &OrbitaConfig, seed: &str) -> Vec<String> {
    let ph = phase(seed);
    let mut out = Vec::new();
    let mut need = |p: &Path, what: &str| {
        if !p.exists() {
            out.push(format!("{what}: {}", p.display()));
        }
    };
    need(&cfg.demo_py(), "script demo.py (ORBITA_ROOT)");
    need(
        &cfg.bin,
        "binario orbita (cd orbita && cargo build --release; ORBITA_BIN)",
    );
    need(&cfg.gemma, "Gemma 2 2B Q4_K_M GGUF (GEMMA_GGUF_PATH)");
    need(
        &cfg.data.join(format!("{ph}/w_{seed}.json")),
        "pesos ÓRBITA por seed (ORBITA_DATA)",
    );
    need(
        &cfg.data.join("llm_d2/stats.npz"),
        "estadísticas del encoder D2 (ORBITA_DATA)",
    );
    need(
        &cfg.data.join(format!("llm_d2/{ph}/enc_D2_{seed}.pt")),
        "encoder D2 por seed (ORBITA_DATA)",
    );
    need(
        &cfg.data.join(format!("yo/{ph}/yo_dev_{seed}.csv")),
        "calibración del yo (ORBITA_DATA)",
    );
    out
}

/// Comprueba el intérprete (numpy + torch) con plazo corto.
async fn python_ok(cfg: &OrbitaConfig) -> Result<(), String> {
    let fut = tokio::process::Command::new(&cfg.python)
        .args(["-c", "import numpy, torch"])
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(Duration::from_secs(20), fut).await {
        Ok(Ok(o)) if o.status.success() => Ok(()),
        Ok(Ok(o)) => Err(format!(
            "Python ({}) sin numpy/torch: {}",
            cfg.python,
            String::from_utf8_lossy(&o.stderr)
                .lines()
                .last()
                .unwrap_or("")
        )),
        Ok(Err(e)) => Err(format!(
            "Python no encontrado ({}): {e} (ORBITA_PYTHON)",
            cfg.python
        )),
        Err(_) => Err(format!("Python ({}) no respondió en 20 s", cfg.python)),
    }
}

/// Disponibilidad completa (archivos + intérprete) para la seed por defecto.
pub async fn availability(cfg: &OrbitaConfig, seed: &str) -> Value {
    let mut miss = missing(cfg, seed);
    if let Err(e) = python_ok(cfg).await {
        miss.push(e);
    }
    json!({
        "available": miss.is_empty(),
        "missing": miss,
        "seed": seed,
        "config": {
            "python": cfg.python,
            "root": cfg.root.display().to_string(),
            "bin": cfg.bin.display().to_string(),
            "data": cfg.data.display().to_string(),
            "gemma": cfg.gemma.display().to_string(),
            "timeout_s": cfg.timeout.as_secs(),
        },
        "families": FAMILIES,
        "x0_default": X0_DEFAULT,
        "lat": "no disponible en línea: requiere servicio con estado (ver docs/ORBITA_GUI.md)",
    })
}

/// Petición de la GUI: familia + 2 parámetros (o texto libre que los
/// describa) y estado de 12 números; opcionalmente k observaciones.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct OrbitaRequest {
    /// Índice 0–7 en [`FAMILIES`]. Obligatorio si no hay `text`.
    pub family: Option<usize>,
    pub p1: Option<f64>,
    pub p2: Option<f64>,
    /// Texto que describe la transformación. Si falta, se genera con la plantilla.
    pub text: Option<String>,
    /// 12 números (6 puntos 2D). Si falta, hexágono por defecto.
    pub x0: Option<Vec<f64>>,
    /// Estados observados (cada uno 12 números), k = 0–2.
    #[serde(default)]
    pub observed: Vec<Vec<f64>>,
    /// Simular k observaciones con la familia/parámetros dados (0–2).
    #[serde(default)]
    pub simulate_obs: usize,
    pub seed: Option<String>,
}

/// Plantilla en español con coma decimal (la que entiende `numnorm`).
pub fn template_text(family: usize, p1: f64, p2: f64) -> String {
    let f = |v: f64| format!("{v}").replace('.', ",");
    format!(
        "Aplica {} {} con parámetros {} y {}.",
        if family == 3 { "un" } else { "una" },
        FAMILIES[family],
        f(p1),
        f(p2)
    )
}

fn check_state(v: &[f64], what: &str) -> Result<(), String> {
    if v.len() != 12 {
        return Err(format!(
            "{what}: se esperan 12 números (6 puntos 2D), llegaron {}",
            v.len()
        ));
    }
    if v.iter().any(|x| !x.is_finite() || x.abs() > MAX_ABS_COORD) {
        return Err(format!(
            "{what}: valores no finitos o fuera de ±{MAX_ABS_COORD}"
        ));
    }
    Ok(())
}

/// Valida y construye los argumentos de `demo.py` (sin `python` ni script).
pub fn build_args(req: &OrbitaRequest) -> Result<(String, Vec<String>), String> {
    let seed = req.seed.clone().unwrap_or_else(|| "0xA400".into());
    if !valid_seed(&seed) {
        return Err("seed inválida: usa 0xA400–0xA40F (DEV) o 0xB400–0xB40F (CONFIRM)".into());
    }
    if let Some(f) = req.family {
        if f >= FAMILIES.len() {
            return Err(format!("familia fuera de rango (0–{})", FAMILIES.len() - 1));
        }
    }
    for (n, p) in [("p1", req.p1), ("p2", req.p2)] {
        if let Some(p) = p {
            if !p.is_finite() || p.abs() > MAX_ABS_PARAM {
                return Err(format!("{n} debe ser finito y |{n}| ≤ {MAX_ABS_PARAM}"));
            }
        }
    }
    let structured = match (req.family, req.p1, req.p2) {
        (Some(f), Some(a), Some(b)) => Some((f, a, b)),
        (None, None, None) => None,
        _ => return Err("familia, p1 y p2 van juntos (los tres o ninguno)".into()),
    };
    let text = match req.text.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => {
            if t.chars().count() > MAX_TEXT {
                return Err(format!("texto demasiado largo (máx. {MAX_TEXT} caracteres)"));
            }
            t.replace(['\n', '\r'], " ")
        }
        None => match structured {
            Some((f, a, b)) => template_text(f, a, b),
            None => {
                return Err(
                    "describe la transformación: familia + p1 + p2, o un texto con familia y 2 parámetros"
                        .into(),
                )
            }
        },
    };
    let mut args = vec!["--text".into(), text, "--seed".into(), seed.clone()];
    if let Some(x0) = &req.x0 {
        check_state(x0, "x0")?;
        args.push("--x0".into());
        args.push(join(x0));
    }
    if !req.observed.is_empty() && req.simulate_obs > 0 {
        return Err("usa observaciones reales o simuladas, no ambas".into());
    }
    if req.observed.len() > 2 || req.simulate_obs > 2 {
        return Err("re-observación: k = 1 o 2".into());
    }
    if !req.observed.is_empty() {
        for (i, y) in req.observed.iter().enumerate() {
            check_state(y, &format!("observación {}", i + 1))?;
        }
        args.push("--obs".into());
        args.push(req.observed.len().to_string());
        args.push("--observed".into());
        args.push(
            req.observed
                .iter()
                .map(|y| join(y))
                .collect::<Vec<_>>()
                .join("|"),
        );
    } else if req.simulate_obs > 0 {
        let Some((f, a, b)) = structured else {
            return Err("simular observaciones requiere familia, p1 y p2".into());
        };
        args.push("--obs".into());
        args.push(req.simulate_obs.to_string());
        args.push("--simulate".into());
        args.push(format!("{f},{a},{b}"));
    }
    args.push("--no-explain".into());
    args.push("--json".into());
    Ok((seed, args))
}

fn join(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn err(code: StatusCode, v: Value) -> Response {
    (code, Json(v)).into_response()
}

async fn status(State(_st): State<SharedState>) -> Json<Value> {
    let cfg = OrbitaConfig::from_env();
    Json(availability(&cfg, "0xA400").await)
}

async fn run(State(_st): State<SharedState>, Json(req): Json<OrbitaRequest>) -> Response {
    let cfg = OrbitaConfig::from_env();
    let (seed, args) = match build_args(&req) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, json!({ "ok": false, "error": e })),
    };
    let miss = missing(&cfg, &seed);
    if !miss.is_empty() {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({ "ok": false, "available": false,
                    "error": "ÓRBITA no está disponible en este servidor", "missing": miss }),
        );
    }
    let t0 = std::time::Instant::now();
    let fut = tokio::process::Command::new(&cfg.python)
        .arg(cfg.demo_py())
        .args(&args)
        .current_dir(&cfg.root)
        .env("PYTHONPATH", cfg.root.join("orbita"))
        .env("ORBITA_BIN", &cfg.bin)
        .env("ORBITA_DATA", &cfg.data)
        .env("GEMMA_GGUF_PATH", &cfg.gemma)
        .kill_on_drop(true)
        .output();
    let out = match tokio::time::timeout(cfg.timeout, fut).await {
        Err(_) => {
            return err(
                StatusCode::GATEWAY_TIMEOUT,
                json!({ "ok": false, "error": format!("ÓRBITA superó el plazo de {} s", cfg.timeout.as_secs()) }),
            )
        }
        Ok(Err(e)) => {
            return err(
                StatusCode::SERVICE_UNAVAILABLE,
                json!({ "ok": false, "available": false,
                        "error": format!("no se pudo lanzar Python ({}): {e}", cfg.python) }),
            )
        }
        Ok(Ok(o)) => o,
    };
    let stderr = String::from_utf8_lossy(&out.stderr);
    let tail: String = stderr
        .lines()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    if !out.status.success() {
        return err(
            StatusCode::BAD_GATEWAY,
            json!({ "ok": false, "error": "demo.py falló", "stderr": tail }),
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'));
    match line.and_then(|l| serde_json::from_str::<Value>(l).ok()) {
        Some(result) => Json(json!({
            "ok": true,
            "elapsed_ms": t0.elapsed().as_millis() as u64,
            "args": args,
            "result": result,
        }))
        .into_response(),
        None => err(
            StatusCode::BAD_GATEWAY,
            json!({ "ok": false, "error": "salida de demo.py sin JSON", "stderr": tail }),
        ),
    }
}

/// Stub: LAT/LAT_EP sólo existe como evaluación (`orbita lat-eval`,
/// `orbita/llm/latlp.py`). En línea haría falta un servicio con estado.
async fn lat_stub() -> Response {
    err(
        StatusCode::NOT_IMPLEMENTED,
        json!({ "ok": false, "error": "LAT en vivo no implementado: requiere un servicio con estado persistente (ver docs/ORBITA_GUI.md)" }),
    )
}

pub fn routes() -> Router<SharedState> {
    Router::new()
        .route("/api/orbita/status", get(status))
        .route("/api/orbita/run", post(run))
        .route("/api/orbita/lat", post(lat_stub))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(f: usize, a: f64, b: f64) -> OrbitaRequest {
        OrbitaRequest {
            family: Some(f),
            p1: Some(a),
            p2: Some(b),
            ..Default::default()
        }
    }

    #[test]
    fn template_matches_demo_text() {
        assert_eq!(
            template_text(1, 0.5, -0.2),
            "Aplica una rotación con parámetros 0,5 y -0,2."
        );
    }

    #[test]
    fn template_article() {
        assert_eq!(
            template_text(3, 0.75, 0.8),
            "Aplica un escalado con parámetros 0,75 y 0,8."
        );
    }

    #[test]
    fn build_args_template_and_simulate() {
        let mut r = req(1, 0.5, -0.2);
        r.simulate_obs = 1;
        let (seed, a) = build_args(&r).unwrap();
        assert_eq!(seed, "0xA400");
        assert_eq!(a[1], "Aplica una rotación con parámetros 0,5 y -0,2.");
        assert!(a
            .windows(2)
            .any(|w| w[0] == "--simulate" && w[1] == "1,0.5,-0.2"));
        assert!(a.contains(&"--json".to_string()));
    }

    #[test]
    fn validation_errors() {
        assert!(build_args(&OrbitaRequest::default()).is_err());
        assert!(build_args(&req(8, 0.1, 0.1)).is_err());
        assert!(build_args(&req(0, f64::NAN, 0.1)).is_err());
        assert!(build_args(&req(0, 1.2, 0.1)).is_err());
        let mut r = req(0, 0.1, 0.1);
        r.x0 = Some(vec![0.0; 11]);
        assert!(build_args(&r).is_err());
        let mut r = req(0, 0.1, 0.1);
        r.observed = vec![vec![0.0; 12]; 3];
        assert!(build_args(&r).is_err());
        let mut r = req(0, 0.1, 0.1);
        r.seed = Some("0xC400".into());
        assert!(build_args(&r).is_err());
        let r = OrbitaRequest {
            family: Some(0),
            ..Default::default()
        };
        assert!(build_args(&r).is_err());
    }

    #[test]
    fn observed_joined_with_pipe() {
        let mut r = req(3, 0.9, 0.8);
        r.observed = vec![vec![1.0; 12], vec![2.0; 12]];
        let (_, a) = build_args(&r).unwrap();
        let i = a.iter().position(|s| s == "--observed").unwrap();
        assert_eq!(a[i + 1].matches('|').count(), 1);
        assert!(a.windows(2).any(|w| w[0] == "--obs" && w[1] == "2"));
    }

    #[test]
    fn missing_reports_everything_when_absent() {
        let cfg = OrbitaConfig {
            python: "python3".into(),
            root: PathBuf::from("/nonexistent"),
            bin: PathBuf::from("/nonexistent/orbita"),
            data: PathBuf::from("/nonexistent/data"),
            gemma: PathBuf::from("/nonexistent/g.gguf"),
            timeout: Duration::from_secs(1),
        };
        assert_eq!(missing(&cfg, "0xA400").len(), 7);
    }
}
