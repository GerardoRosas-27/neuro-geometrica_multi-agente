//! Job de entrenamiento en vivo: dataset LLM (periferia) → líquido → CDT por lotes.
//!
//! Pipeline por lote/dataset:
//! 1. Generar dataset (Gemma o curriculum `lexicon_synth`) — solo periferia.
//! 2. Encode texto → features → concept_id (firewall; sin tokens en FieldState).
//! 3. `fuse.observe` / `fuse.infer` (ruta líquida / WavePredictCore).
//! 4. `sleep_consolidate` → engramas CDT + checkpoint **por dataset** en
//!    `data/checkpoints/datasets/`.
//! 5. Decode opcional concepto→texto (LLM decoder only).
//!
//! Por defecto el entrenamiento es **infinito** (solo para con cancel/stop).

use crate::web::llm_periphery::{generate_train_batch, TrainExample, NUM_CONCEPTS};

use crate::web::telemetry::SleepReportDto;
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
    pub engrams: Option<usize>,
    #[serde(default)]
    pub metrics: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub path: String,
    pub batch: usize,
    pub engrams: usize,
    pub ts_ms: u64,
    pub accuracy: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainJob {
    pub job_id: String,
    pub running: bool,
    pub cancelled: bool,
    /// Contador monotónico de datasets/lotes completados (0, 1, 2…).
    pub current_batch: usize,
    /// `None` cuando `infinite == true`.
    pub total_batches: Option<usize>,
    pub infinite: bool,
    pub epochs: usize,
    pub batch_size: usize,
    pub events: VecDeque<TrainLiveEvent>,
    pub last_checkpoint: Option<CheckpointMeta>,
    pub dataset_size: usize,
    pub accuracy: Option<f64>,
    pub engrams: usize,
    pub last_decoded: Option<String>,
    pub dataset_source: Option<String>,
    pub event_seq: u64,
    pub correct: u64,
    pub total: u64,
    pub datasets_saved: usize,
    pub last_dataset_path: Option<String>,
    /// Última familia de dataset generada (E8–E30 / core).
    pub last_dataset_family: Option<String>,
    /// IDs de experimento del último lote.
    pub last_experiment_ids: Vec<String>,
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
            epochs: 1,
            batch_size: 8,
            events: VecDeque::new(),
            last_checkpoint: None,
            dataset_size: 0,
            accuracy: None,
            engrams: 0,
            last_decoded: None,
            dataset_source: None,
            event_seq: 0,
            correct: 0,
            total: 0,
            datasets_saved: 0,
            last_dataset_path: None,
            last_dataset_family: None,
            last_experiment_ids: Vec::new(),
        }
    }
}

impl TrainJob {
    pub fn push_event(
        &mut self,
        kind: &str,
        message: impl Into<String>,
        batch: Option<usize>,
        engrams: Option<usize>,
        metrics: Value,
    ) {
        self.event_seq = self.event_seq.wrapping_add(1);
        let ev = TrainLiveEvent {
            seq: self.event_seq,
            ts_ms: now_ms(),
            kind: kind.into(),
            message: message.into(),
            batch,
            engrams,
            metrics,
        };
        self.events.push_back(ev);
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
            epochs: self.epochs,
            batch_size: self.batch_size,
            events: self.events.iter().cloned().collect(),
            last_checkpoint: self.last_checkpoint.clone(),
            dataset_size: self.dataset_size,
            accuracy: self.accuracy,
            engrams: self.engrams,
            last_decoded: self.last_decoded.clone(),
            dataset_source: self.dataset_source.clone(),
            event_seq: self.event_seq,
            datasets_saved: self.datasets_saved,
            last_dataset_path: self.last_dataset_path.clone(),
            last_dataset_family: self.last_dataset_family.clone(),
            last_experiment_ids: self.last_experiment_ids.clone(),
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
    pub epochs: usize,
    pub batch_size: usize,
    pub events: Vec<TrainLiveEvent>,
    pub last_checkpoint: Option<CheckpointMeta>,
    pub dataset_size: usize,
    pub accuracy: Option<f64>,
    pub engrams: usize,
    pub last_decoded: Option<String>,
    pub dataset_source: Option<String>,
    pub event_seq: u64,
    pub datasets_saved: usize,
    pub last_dataset_path: Option<String>,
    pub last_dataset_family: Option<String>,
    pub last_experiment_ids: Vec<String>,
}

/// Request de arranque. `batches` acepta número, null, 0 o `"infinite"`.
#[derive(Clone, Debug, Deserialize)]
pub struct LiveTrainStartRequest {
    #[serde(default)]
    pub batches: Option<Value>,
    pub batch_size: Option<usize>,
    pub epochs: Option<usize>,
    /// Compat con API antigua.
    pub concepts: Option<Vec<usize>>,
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

#[derive(Clone, Debug, Serialize)]
pub struct CheckpointFile {
    pub job_id: String,
    pub batch: usize,
    pub engrams: usize,
    pub accuracy: Option<f64>,
    pub dataset_size: usize,
    pub dataset_source: Option<String>,
    pub sleep: Option<SleepReportDto>,
    pub relational_cues: Vec<usize>,
    pub events_tail: Vec<TrainLiveEvent>,
    pub ts_ms: u64,
}

/// Checkpoint rico **por dataset** (ejemplos + métricas líquido + sueño + decoder).
#[derive(Clone, Debug, Serialize)]
pub struct DatasetCheckpointFile {
    pub dataset_id: String,
    pub job_id: String,
    pub batch: usize,
    pub source: String,
    #[serde(default)]
    pub dataset_family: Option<String>,
    #[serde(default)]
    pub experiment_ids: Vec<String>,
    pub examples: Vec<TrainExample>,
    pub liquid: LiquidDatasetMetrics,
    pub sleep: Option<SleepReportDto>,
    pub decoder_preview: Option<String>,
    pub engrams: usize,
    pub accuracy: Option<f64>,
    pub ts_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LiquidDatasetMetrics {
    pub scores: Vec<f64>,
    pub preds: Vec<usize>,
    pub correct: u64,
    pub total: u64,
    pub last_score: f64,
    pub last_pred: usize,
}

pub fn checkpoints_dir() -> PathBuf {
    let candidates = [
        PathBuf::from("data/checkpoints"),
        PathBuf::from("./data/checkpoints"),
        PathBuf::from("/app/data/checkpoints"),
    ];
    for c in &candidates {
        if c.parent().map(|p| p.exists()).unwrap_or(false) || c.exists() {
            return c.clone();
        }
    }
    PathBuf::from("data/checkpoints")
}

pub fn datasets_dir() -> PathBuf {
    checkpoints_dir().join("datasets")
}

pub fn ensure_checkpoints_dir() -> std::io::Result<PathBuf> {
    let dir = checkpoints_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn ensure_datasets_dir() -> std::io::Result<PathBuf> {
    let dir = datasets_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn write_checkpoint(file: &CheckpointFile) -> std::io::Result<PathBuf> {
    let dir = ensure_checkpoints_dir()?;
    let name = format!("train_{}_batch_{}.json", file.ts_ms, file.batch);
    let path = dir.join(name);
    let json = serde_json::to_vec_pretty(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    fs::write(&path, json)?;
    Ok(path)
}

pub fn write_dataset_checkpoint(file: &DatasetCheckpointFile) -> std::io::Result<PathBuf> {
    let dir = ensure_datasets_dir()?;
    let name = format!("train_{}_ds_{}.json", file.ts_ms, file.batch);
    let path = dir.join(name);
    let json = serde_json::to_vec_pretty(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    fs::write(&path, json)?;
    Ok(path)
}

/// Índice pequeño apuntando al último checkpoint de dataset.
pub fn write_latest_index(
    dataset_path: &str,
    job_id: &str,
    batch: usize,
    datasets_saved: usize,
) -> std::io::Result<PathBuf> {
    let dir = ensure_checkpoints_dir()?;
    let path = dir.join("latest.json");
    let body = json!({
        "last_dataset_path": dataset_path,
        "job_id": job_id,
        "batch": batch,
        "datasets_saved": datasets_saved,
        "ts_ms": now_ms(),
    });
    fs::write(&path, serde_json::to_vec_pretty(&body)?)?;
    Ok(path)
}

/// Curriculum sintético público para tests (mismo que `generate_train_batch` léxico).
pub fn synth_dataset(batch_size: usize, seed: u64) -> Vec<TrainExample> {
    let (items, _) = generate_train_batch(batch_size, seed, false);
    // meta descartada: solo ejemplos para tests unitarios
    items
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Relación ligera para ejercitar buffer + RQM en sueño (tokenless).
pub fn relation_target(concept: usize) -> usize {
    (concept + 1) % NUM_CONCEPTS
}

pub fn batch_metrics(
    source: &str,
    examples: usize,
    correct: u64,
    total: u64,
    liquid_score: f64,
) -> Value {
    json!({
        "source": source,
        "examples": examples,
        "correct": correct,
        "total": total,
        "liquid_score": liquid_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_generates_n_items() {
        let items = synth_dataset(5, 42);
        assert_eq!(items.len(), 5);
        for it in &items {
            assert!(!it.text.is_empty());
            assert!(it.concept < NUM_CONCEPTS);
        }
    }

    #[test]
    fn checkpoint_file_written() {
        let dir = ensure_checkpoints_dir().expect("mkdir");
        let file = CheckpointFile {
            job_id: "test-job".into(),
            batch: 0,
            engrams: 3,
            accuracy: Some(0.5),
            dataset_size: 8,
            dataset_source: Some("lexicon_synth".into()),
            sleep: None,
            relational_cues: vec![0, 1],
            events_tail: vec![],
            ts_ms: 1_700_000_000_000,
        };
        let path = write_checkpoint(&file).expect("write");
        assert!(path.exists());
        assert!(path.starts_with(&dir) || path.to_string_lossy().contains("checkpoints"));
        let raw = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["job_id"], "test-job");
        assert_eq!(v["engrams"], 3);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn dataset_checkpoint_and_latest() {
        let file = DatasetCheckpointFile {
            dataset_id: "ds_test_0".into(),
            job_id: "test-job".into(),
            batch: 0,
            source: "lexicon_synth".into(),
            dataset_family: Some("core_agentic".into()),
            experiment_ids: vec!["core".into()],
            examples: vec![TrainExample {
                text: "hola".into(),
                concept: 1,
            }],
            liquid: LiquidDatasetMetrics {
                scores: vec![0.9],
                preds: vec![1],
                correct: 1,
                total: 1,
                last_score: 0.9,
                last_pred: 1,
            },
            sleep: None,
            decoder_preview: Some("hola".into()),
            engrams: 2,
            accuracy: Some(1.0),
            ts_ms: 1_700_000_000_001,
        };
        let path = write_dataset_checkpoint(&file).expect("write ds");
        assert!(path.exists());
        assert!(path.to_string_lossy().contains("datasets"));
        let latest = write_latest_index(path.to_str().unwrap(), "test-job", 0, 1).expect("latest");
        assert!(latest.exists());
        let raw = fs::read_to_string(&latest).unwrap();
        let v: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["datasets_saved"], 1);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&latest);
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
        job.push_event("dataset", "a", Some(0), Some(0), json!({}));
        job.push_event("batch_start", "b", Some(0), Some(0), json!({}));
        let after = job.events_after(1);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].kind, "batch_start");
    }
}
