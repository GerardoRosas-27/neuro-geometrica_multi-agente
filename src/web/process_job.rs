//! Registro unificado de jobs en memoria (sueño / pruebas) para reconexión UI.
//!
//! Patrón análogo a `train_job`: ring buffer de eventos con `seq`, estado
//! `running`/`cancelled`, y snapshot completo en status para que un refresh
//! reconstruya la consola sin reiniciar el trabajo.

use crate::web::field_eval::FieldEvalReport;
use crate::web::sleep_optimize::{SleepOptimizeOpts, SleepOptimizeReport};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;

const MAX_EVENTS: usize = 300;

/// Evento de consola compartido (sueño / pruebas).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessEvent {
    pub seq: u64,
    pub ts_ms: u64,
    pub kind: String,
    pub message: String,
    #[serde(default)]
    pub metrics: Value,
}

fn now_ms() -> u64 {
    crate::web::train_job::now_ms()
}

fn push_into(
    events: &mut VecDeque<ProcessEvent>,
    event_seq: &mut u64,
    kind: &str,
    message: impl Into<String>,
    metrics: Value,
) {
    *event_seq = event_seq.wrapping_add(1);
    events.push_back(ProcessEvent {
        seq: *event_seq,
        ts_ms: now_ms(),
        kind: kind.into(),
        message: message.into(),
        metrics,
    });
    while events.len() > MAX_EVENTS {
        events.pop_front();
    }
}

fn events_after(events: &VecDeque<ProcessEvent>, after: u64) -> Vec<ProcessEvent> {
    events.iter().filter(|e| e.seq > after).cloned().collect()
}

/// Interpreta `cycles` del body (misma semántica que train `batches`).
/// `None` = infinito.
pub fn parse_cycles_field(cycles: Option<&Value>) -> Option<usize> {
    crate::web::train_job::parse_batches_field(cycles)
}

/// Resuelve `infinite` + `cycles` → (infinite, total_cycles).
/// Default sin body: infinito (checkbox marcado por defecto en la UI).
pub fn resolve_infinite_cycles(
    infinite_flag: Option<bool>,
    cycles: Option<&Value>,
) -> (bool, Option<usize>) {
    match infinite_flag {
        Some(true) => (true, None),
        Some(false) => {
            let n = parse_cycles_field(cycles).unwrap_or(1).max(1);
            (false, Some(n))
        }
        None => match parse_cycles_field(cycles) {
            None => (true, None),
            Some(n) => (false, Some(n.max(1))),
        },
    }
}

/// Job async de sueño / optimización.
#[derive(Clone, Debug, Serialize)]
pub struct SleepJob {
    pub job_id: String,
    pub running: bool,
    pub cancelled: bool,
    pub phase: String,
    pub prune_intensity: f64,
    pub compact_intensity: f64,
    pub consolidate_first: bool,
    /// Ciclos de optimización completados (se incrementa al terminar cada uno).
    pub current_cycle: usize,
    /// `None` cuando `infinite == true`.
    pub total_cycles: Option<usize>,
    pub infinite: bool,
    pub events: VecDeque<ProcessEvent>,
    pub event_seq: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub last_report: Option<SleepOptimizeReport>,
}

impl Default for SleepJob {
    fn default() -> Self {
        Self {
            job_id: String::new(),
            running: false,
            cancelled: false,
            phase: "idle".into(),
            prune_intensity: 0.55,
            compact_intensity: 0.55,
            consolidate_first: true,
            current_cycle: 0,
            total_cycles: None,
            infinite: true,
            events: VecDeque::new(),
            event_seq: 0,
            started_ms: None,
            finished_ms: None,
            last_report: None,
        }
    }
}

impl SleepJob {
    pub fn push_event(&mut self, kind: &str, message: impl Into<String>, metrics: Value) {
        push_into(
            &mut self.events,
            &mut self.event_seq,
            kind,
            message,
            metrics,
        );
    }

    pub fn events_after(&self, after: u64) -> Vec<ProcessEvent> {
        events_after(&self.events, after)
    }

    pub fn opts(&self) -> SleepOptimizeOpts {
        SleepOptimizeOpts {
            prune_intensity: self.prune_intensity,
            compact_intensity: self.compact_intensity,
            // Solo consolidar wake buffer en el primer ciclo.
            consolidate_first: self.consolidate_first && self.current_cycle == 0,
        }
    }

    pub fn snapshot_for_api(&self) -> SleepJobSnapshot {
        SleepJobSnapshot {
            job_id: self.job_id.clone(),
            running: self.running,
            cancelled: self.cancelled,
            phase: self.phase.clone(),
            prune_intensity: self.prune_intensity,
            compact_intensity: self.compact_intensity,
            consolidate_first: self.consolidate_first,
            current_cycle: self.current_cycle,
            total_cycles: self.total_cycles,
            infinite: self.infinite,
            current_batch: self.current_cycle,
            total_batches: self.total_cycles,
            events: self.events.iter().cloned().collect(),
            event_seq: self.event_seq,
            started_ms: self.started_ms,
            finished_ms: self.finished_ms,
            last_report: self.last_report.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SleepJobSnapshot {
    pub job_id: String,
    pub running: bool,
    pub cancelled: bool,
    pub phase: String,
    pub prune_intensity: f64,
    pub compact_intensity: f64,
    pub consolidate_first: bool,
    pub current_cycle: usize,
    pub total_cycles: Option<usize>,
    pub infinite: bool,
    /// Alias train-like para la barra de progreso.
    pub current_batch: usize,
    pub total_batches: Option<usize>,
    pub events: Vec<ProcessEvent>,
    pub event_seq: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub last_report: Option<SleepOptimizeReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SleepStartResponse {
    pub ok: bool,
    pub job_id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infinite: Option<bool>,
}

/// Job async de batería de pruebas.
#[derive(Clone, Debug, Serialize)]
pub struct TestsJob {
    pub job_id: String,
    pub running: bool,
    pub cancelled: bool,
    pub phase: String,
    pub step: usize,
    pub total_steps: usize,
    /// Baterías de eval completadas.
    pub current_cycle: usize,
    /// `None` cuando `infinite == true`.
    pub total_cycles: Option<usize>,
    pub infinite: bool,
    pub events: VecDeque<ProcessEvent>,
    pub event_seq: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub last_report: Option<FieldEvalReport>,
}

impl Default for TestsJob {
    fn default() -> Self {
        Self {
            job_id: String::new(),
            running: false,
            cancelled: false,
            phase: "idle".into(),
            step: 0,
            total_steps: 0,
            current_cycle: 0,
            total_cycles: Some(1),
            infinite: false,
            events: VecDeque::new(),
            event_seq: 0,
            started_ms: None,
            finished_ms: None,
            last_report: None,
        }
    }
}

impl TestsJob {
    pub fn push_event(&mut self, kind: &str, message: impl Into<String>, metrics: Value) {
        push_into(
            &mut self.events,
            &mut self.event_seq,
            kind,
            message,
            metrics,
        );
    }

    pub fn events_after(&self, after: u64) -> Vec<ProcessEvent> {
        events_after(&self.events, after)
    }

    pub fn snapshot_for_api(&self) -> TestsJobSnapshot {
        TestsJobSnapshot {
            job_id: self.job_id.clone(),
            running: self.running,
            cancelled: self.cancelled,
            phase: self.phase.clone(),
            step: self.step,
            total_steps: self.total_steps,
            current_cycle: self.current_cycle,
            total_cycles: self.total_cycles,
            infinite: self.infinite,
            current_batch: self.current_cycle,
            total_batches: self.total_cycles,
            events: self.events.iter().cloned().collect(),
            event_seq: self.event_seq,
            started_ms: self.started_ms,
            finished_ms: self.finished_ms,
            last_report: self.last_report.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TestsJobSnapshot {
    pub job_id: String,
    pub running: bool,
    pub cancelled: bool,
    pub phase: String,
    pub step: usize,
    pub total_steps: usize,
    pub current_cycle: usize,
    pub total_cycles: Option<usize>,
    pub infinite: bool,
    pub current_batch: usize,
    pub total_batches: Option<usize>,
    pub events: Vec<ProcessEvent>,
    pub event_seq: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub last_report: Option<FieldEvalReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TestsStartResponse {
    pub ok: bool,
    pub job_id: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infinite: Option<bool>,
}

/// Snapshot global para reconexión tras refresh.
#[derive(Clone, Debug, Serialize)]
pub struct ProcessesSnapshot {
    pub train: crate::web::train_job::TrainJobSnapshot,
    pub sleep: SleepJobSnapshot,
    pub tests: TestsJobSnapshot,
    pub active: Vec<String>,
}

pub fn process_tail_metrics(kind: &str, extra: Value) -> Value {
    let mut m = json!({ "kind": kind });
    if let Some(obj) = extra.as_object() {
        if let Some(dst) = m.as_object_mut() {
            for (k, v) in obj {
                dst.insert(k.clone(), v.clone());
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleep_events_ring_and_after() {
        let mut job = SleepJob::default();
        job.push_event("start", "a", json!({}));
        job.push_event("prune", "b", json!({ "n": 1 }));
        assert_eq!(job.event_seq, 2);
        assert_eq!(job.events_after(1).len(), 1);
        assert_eq!(job.events_after(1)[0].kind, "prune");
        let snap = job.snapshot_for_api();
        assert_eq!(snap.events.len(), 2);
    }

    #[test]
    fn tests_job_defaults_idle() {
        let job = TestsJob::default();
        assert!(!job.running);
        assert_eq!(job.phase, "idle");
    }

    #[test]
    fn resolve_infinite_cycles_cases() {
        assert_eq!(resolve_infinite_cycles(Some(true), None), (true, None));
        assert_eq!(resolve_infinite_cycles(Some(false), None), (false, Some(1)));
        assert_eq!(
            resolve_infinite_cycles(Some(false), Some(&json!(3))),
            (false, Some(3))
        );
        assert_eq!(resolve_infinite_cycles(None, None), (true, None));
        assert_eq!(
            resolve_infinite_cycles(None, Some(&Value::Null)),
            (true, None)
        );
        assert_eq!(
            resolve_infinite_cycles(None, Some(&json!(2))),
            (false, Some(2))
        );
    }

    #[test]
    fn sleep_snapshot_exposes_cycle_aliases() {
        let mut job = SleepJob::default();
        job.current_cycle = 2;
        job.infinite = true;
        let snap = job.snapshot_for_api();
        assert_eq!(snap.current_batch, 2);
        assert!(snap.infinite);
        assert!(snap.total_batches.is_none());
    }
}
