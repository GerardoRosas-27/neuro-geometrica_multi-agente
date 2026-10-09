//! Modelo GGUF ligero: ruta configurable + descarga automática si falta.
//!
//! Env:
//! - `GEMMA2_GGUF`: ruta local del GGUF (def [`DEFAULT_GGUF_PATH`]).
//! - `GEMMA2_GGUF_URL`: URL pública (Hugging Face, sin token) de la que descargar
//!   si la ruta no existe (def [`DEFAULT_GGUF_URL`]).
//! - `GEMMA2_GGUF_SHA256`: sha256 esperado; vacío = no verificar
//!   (def [`DEFAULT_GGUF_SHA256`] solo cuando se usa la URL por defecto).
//! - `GEMMA2_AUTO_DOWNLOAD`: `0` desactiva la descarga en arranque (def `1`).
//!
//! La descarga usa `curl` (presente en la imagen Docker) en un hilo aparte: el
//! servidor arranca enseguida (healthcheck OK) en modo léxico y hace hot-swap a
//! Gemma cuando el GGUF queda descargado y cargado.

use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Gemma 2 2B-it en **Q3_K_L** (~1.55 GB, bartowski). El cargador nativo solo
/// acepta arquitectura `gemma2` y candle no soporta IQ*; Q2_K (~1.23 GB) se
/// probó y degenera (bucles/texto incoherente), así que Q3_K_L es la opción
/// más ligera usable. Cambiable con `GEMMA2_GGUF_URL` + `GEMMA2_GGUF_SHA256`.
pub const DEFAULT_GGUF_PATH: &str = "models/gemma-2-2b-it-Q3_K_L.gguf";
/// Revisión fijada (commit) para que el sha256 sea estable.
pub const DEFAULT_GGUF_URL: &str = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/855f67caed130e1befc571b52bd181be2e858883/gemma-2-2b-it-Q3_K_L.gguf";
pub const DEFAULT_GGUF_SHA256: &str =
    "d14b920ed8025a03e0764bdaeca6fe553dfb559569c79b5b83bb5b28fc364984";
pub const DEFAULT_GGUF_BYTES: u64 = 1_550_436_192;

/// Configuración resuelta del modelo.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub path: PathBuf,
    pub url: String,
    pub sha256: Option<String>,
    pub auto_download: bool,
    pub expected_bytes: Option<u64>,
}

impl ModelConfig {
    pub fn from_env() -> Self {
        let path = env::var("GEMMA2_GGUF")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_GGUF_PATH));
        let custom_url = env::var("GEMMA2_GGUF_URL")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let url = custom_url
            .clone()
            .unwrap_or_else(|| DEFAULT_GGUF_URL.to_string());
        let sha256 = match env::var("GEMMA2_GGUF_SHA256") {
            Ok(v) if v.trim().is_empty() => None,
            Ok(v) => Some(v.trim().to_ascii_lowercase()),
            Err(_) if custom_url.is_none() => Some(DEFAULT_GGUF_SHA256.to_string()),
            Err(_) => None,
        };
        let auto_download = env::var("GEMMA2_AUTO_DOWNLOAD")
            .map(|v| !matches!(v.trim(), "0" | "false" | "no" | "off"))
            .unwrap_or(true);
        let expected_bytes = custom_url.is_none().then_some(DEFAULT_GGUF_BYTES);
        Self {
            path,
            url,
            sha256,
            auto_download,
            expected_bytes,
        }
    }

    pub fn part_path(&self) -> PathBuf {
        let mut p = self.path.clone().into_os_string();
        p.push(".part");
        PathBuf::from(p)
    }
}

/// Estado del modelo expuesto en `/health` y usado por la UI.
#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    /// `ready` | `downloading` | `loading` | `missing` | `error` | `disabled`.
    pub state: String,
    pub path: String,
    pub url: String,
    pub detail: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

impl ModelStatus {
    pub fn new(state: &str, cfg: &ModelConfig, detail: impl Into<String>) -> Self {
        Self {
            state: state.into(),
            path: cfg.path.display().to_string(),
            url: cfg.url.clone(),
            detail: detail.into(),
            downloaded_bytes: 0,
            total_bytes: cfg.expected_bytes,
        }
    }

    /// Refresca bytes descargados (tamaño del `.part`) mientras descarga.
    pub fn refresh_progress(&mut self, cfg: &ModelConfig) {
        if self.state == "downloading" {
            self.downloaded_bytes = std::fs::metadata(cfg.part_path())
                .map(|m| m.len())
                .unwrap_or(0);
        }
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let out = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|e| format!("sha256sum no disponible: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "sha256sum falló: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase())
}

/// Descarga el GGUF a `cfg.path` (vía `.part` + rename). Bloqueante.
/// Reanuda descargas parciales (`curl -C -`). Verifica sha256 si está configurado.
pub fn download_gguf(cfg: &ModelConfig) -> Result<(), String> {
    if let Some(dir) = cfg.path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        }
    }
    let part = cfg.part_path();
    tracing::info!(url = %cfg.url, dest = %cfg.path.display(), "descargando GGUF");
    let status = Command::new("curl")
        .args([
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "30",
            "-sS",
            "-C",
            "-",
            "-o",
        ])
        .arg(&part)
        .arg(&cfg.url)
        .status()
        .map_err(|e| format!("no se pudo ejecutar curl: {e}"))?;
    if !status.success() {
        return Err(format!("curl terminó con {status} descargando {}", cfg.url));
    }
    if let Some(expected) = &cfg.sha256 {
        let got = sha256_file(&part)?;
        if &got != expected {
            let _ = std::fs::remove_file(&part);
            return Err(format!(
                "sha256 no coincide (esperado {expected}, obtenido {got})"
            ));
        }
    }
    std::fs::rename(&part, &cfg.path).map_err(|e| format!("rename: {e}"))?;
    tracing::info!(dest = %cfg.path.display(), "GGUF descargado");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_points_to_light_q2k() {
        let cfg = ModelConfig {
            path: PathBuf::from(DEFAULT_GGUF_PATH),
            url: DEFAULT_GGUF_URL.into(),
            sha256: Some(DEFAULT_GGUF_SHA256.into()),
            auto_download: true,
            expected_bytes: Some(DEFAULT_GGUF_BYTES),
        };
        assert!(cfg.url.starts_with("https://huggingface.co/"));
        assert!(cfg.url.contains("Q3_K_L"));
        assert!(cfg.part_path().to_string_lossy().ends_with(".gguf.part"));
        let st = ModelStatus::new("missing", &cfg, "x");
        assert_eq!(st.total_bytes, Some(DEFAULT_GGUF_BYTES));
    }
}
