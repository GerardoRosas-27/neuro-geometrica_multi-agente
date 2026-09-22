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
            consolidate_first: self.consolidate_first,
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
}
