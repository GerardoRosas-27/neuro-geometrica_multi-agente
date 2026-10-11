//! Job de entrenamiento en vivo del campo v4.
//!
//! Por lote: dataset v4 sellado (generador geométrico, TEST con SHA-256) →
//! experiencia en la CDT de vigilia → consolidación (teacher) → Dφ con la
//! receta §16 (cabeza local-lineal con compuerta + replay de confianza) →
//! evaluación limpia en el TEST sellado del lote → checkpoint bit-exacto de Dφ
//! en `data/checkpoints/v4/`.
//!
//! Por defecto el entrenamiento es **infinito** (solo para con cancel/stop).

use crate::field_v4::TrainBatchReport;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

const MAX_EVENTS: usize = 400;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainLiveEvent {
    pub seq: u64,
    pub ts_ms: u64,
    pub kind: String,
    pub message: String,
    pub batch: Option<usize>,
    #[serde(default)]
    pub metrics: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub path: String,
    pub batch: usize,
    pub ts_ms: u64,
    pub accuracy: Option<f64>,
    pub weights_hash: String,
}

#[derive(Clone, Debug)]
pub struct TrainJob {
    pub job_id: String,
    pub running: bool,
    pub cancelled: bool,
    /// Lotes completados (0, 1, 2…).
    pub current_batch: usize,
    /// `None` cuando `infinite == true`.
    pub total_batches: Option<usize>,
    pub infinite: bool,
    /// Pasos de optimización de Dφ por lote.
    pub steps: usize,
    pub events: VecDeque<TrainLiveEvent>,
    pub last_checkpoint: Option<CheckpointMeta>,
    pub last_report: Option<TrainBatchReport>,
    pub event_seq: u64,
    pub checkpoints_saved: usize,
}

impl Default for TrainJob {
    fn default() -> Self {
        Self {
            job_id: String::new(),
            running: false,
            cancelled: false,
            current_batch: 0,
            total_batches: None,
            infinite: true,
            steps: crate::field_v4::APP_TRAIN_STEPS,
            events: VecDeque::new(),
            last_checkpoint: None,
            last_report: None,
            event_seq: 0,
            checkpoints_saved: 0,
        }
    }
}

impl TrainJob {
    pub fn push_event(
        &mut self,
        kind: &str,
        message: impl Into<String>,
        batch: Option<usize>,
        metrics: Value,
    ) {
        self.event_seq = self.event_seq.wrapping_add(1);
        self.events.push_back(TrainLiveEvent {
            seq: self.event_seq,
            ts_ms: now_ms(),
            kind: kind.into(),
            message: message.into(),
            batch,
            metrics,
        });
        while self.events.len() > MAX_EVENTS {
            self.events.pop_front();
        }
    }

    pub fn events_after(&self, after: u64) -> Vec<TrainLiveEvent> {
        self.events
            .iter()
            .filter(|e| e.seq > after)
            .cloned()
            .collect()
    }

    pub fn snapshot_for_api(&self) -> TrainJobSnapshot {
        TrainJobSnapshot {
            job_id: self.job_id.clone(),
            running: self.running,
            cancelled: self.cancelled,
            current_batch: self.current_batch,
            total_batches: self.total_batches,
            infinite: self.infinite,
            steps: self.steps,
            events: self.events.iter().cloned().collect(),
            last_checkpoint: self.last_checkpoint.clone(),
            last_report: self.last_report.clone(),
            event_seq: self.event_seq,
            checkpoints_saved: self.checkpoints_saved,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainJobSnapshot {
    pub job_id: String,
    pub running: bool,
    pub cancelled: bool,
    pub current_batch: usize,
    pub total_batches: Option<usize>,
    pub infinite: bool,
    pub steps: usize,
    pub events: Vec<TrainLiveEvent>,
    pub last_checkpoint: Option<CheckpointMeta>,
    pub last_report: Option<TrainBatchReport>,
    pub event_seq: u64,
    pub checkpoints_saved: usize,
}

/// Request de arranque. `batches` acepta número, null, 0 o `"infinite"`.
#[derive(Clone, Debug, Deserialize)]
pub struct LiveTrainStartRequest {
    #[serde(default)]
    pub batches: Option<Value>,
    /// Pasos de Dφ por lote (def. [`crate::field_v4::APP_TRAIN_STEPS`]).
    pub steps: Option<usize>,
}

/// Interpreta el campo `batches` del body.
/// `None` = entrenamiento infinito.
pub fn parse_batches_field(batches: Option<&Value>) -> Option<usize> {
    match batches {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) => {
            let u = n.as_u64().unwrap_or(0) as usize;
            if u == 0 {
                None
            } else {
                Some(u)
            }
        }
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty()
                || t.eq_ignore_ascii_case("infinite")
                || t == "∞"
                || t.eq_ignore_ascii_case("inf")
            {
                None
            } else {
                match t.parse::<usize>() {
                    Ok(0) | Err(_) => None,
                    Ok(n) => Some(n),
                }
            }
        }
        Some(Value::Bool(false)) => None,
        Some(_) => None,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LiveTrainStartResponse {
    pub ok: bool,
    pub job_id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infinite: Option<bool>,
}

/// Checkpoint bit-exacto de Dφ + informe del lote que lo produjo.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DphiCheckpointFile {
    pub architecture: String,
    pub job_id: String,
    pub batch: usize,
    pub ts_ms: u64,
    pub report: Option<TrainBatchReport>,
    /// `Dphi::to_ckpt()` (f64 como bits u64).
    pub dphi: String,
}

pub fn checkpoints_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NGM_CHECKPOINT_DIR") {
        return PathBuf::from(d);
    }
    for c in ["data/checkpoints/v4", "/app/data/checkpoints/v4"] {
        let p = PathBuf::from(c);
        if p.parent().map(|p| p.exists()).unwrap_or(false) || p.exists() {
            return p;
        }
    }
    PathBuf::from("data/checkpoints/v4")
}

pub fn write_dphi_checkpoint(file: &DphiCheckpointFile) -> std::io::Result<PathBuf> {
    let dir = checkpoints_dir();
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("dphi_{}_b{}.json", file.ts_ms, file.batch));
    let json = serde_json::to_vec(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    fs::write(&path, json)?;
    let latest =
        json!({ "path": path.display().to_string(), "batch": file.batch, "ts_ms": file.ts_ms });
    fs::write(dir.join("latest.json"), serde_json::to_vec_pretty(&latest)?)?;
    Ok(path)
}

/// Último checkpoint de Dφ (persistencia entre reinicios, como E38).
pub fn read_latest_dphi() -> Option<(PathBuf, DphiCheckpointFile)> {
    let dir = checkpoints_dir();
    let idx: Value = serde_json::from_slice(&fs::read(dir.join("latest.json")).ok()?).ok()?;
    let path = PathBuf::from(idx.get("path")?.as_str()?);
    let file: DphiCheckpointFile = serde_json::from_slice(&fs::read(&path).ok()?).ok()?;
    Some((path, file))
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dphi_checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ngm_ckpt_{}", now_ms()));
        std::env::set_var("NGM_CHECKPOINT_DIR", &dir);
        let d = crate::field_v4::new_dphi(5);
        let f = DphiCheckpointFile {
            architecture: "v4".into(),
            job_id: "t".into(),
            batch: 0,
            ts_ms: now_ms(),
            report: None,
            dphi: d.to_ckpt(),
        };
        let p = write_dphi_checkpoint(&f).unwrap();
        assert!(p.exists());
        let (_, back) = read_latest_dphi().unwrap();
        let d2 = crate::field_v4::from_checkpoint(&back.dphi).unwrap();
        assert_eq!(d.weights_hash(), d2.weights_hash());
        std::env::remove_var("NGM_CHECKPOINT_DIR");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn parse_batches_infinite_cases() {
        assert_eq!(parse_batches_field(None), None);
        assert_eq!(parse_batches_field(Some(&Value::Null)), None);
        assert_eq!(parse_batches_field(Some(&json!(0))), None);
        assert_eq!(
            parse_batches_field(Some(&Value::String("infinite".into()))),
            None
        );
        assert_eq!(parse_batches_field(Some(&json!(2))), Some(2));
        assert_eq!(
            parse_batches_field(Some(&Value::String("3".into()))),
            Some(3)
        );
    }

    #[test]
    fn events_after_filters() {
        let mut job = TrainJob::default();
        job.push_event("dataset", "a", Some(0), json!({}));
        job.push_event("batch_start", "b", Some(0), json!({}));
        let after = job.events_after(1);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].kind, "batch_start");
    }
}
