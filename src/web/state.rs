//! Estado compartido de la app agentica (Arc<Mutex<AppState>>).

use crate::field_hybrid_infer::FieldHybridInfer;
use crate::liquid_cdt_memory::SleepReport;
use crate::liquid_cdt_rqm_fuse::{FuseReport, FusedLiquidCdt, InferRoute};
use crate::web::llm_periphery::{
    generate_train_batch, open_best_probe, ConceptDecoder, LlmMode, PeripheralProbe, NUM_CONCEPTS,
};
use crate::web::telemetry::{FuseReportDto, SleepReportDto, TelemetrySnapshot};
use crate::web::train_job::{
    batch_metrics, relation_target, write_checkpoint, CheckpointFile, CheckpointMeta, TrainJob,
    TrainLiveEvent,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainEvent {
    pub t_ms: u64,
    pub kind: String,
    pub detail: String,
    pub engrams: usize,
    pub epoch: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    pub text: String,
    pub route: Option<String>,
    pub concept_in: Option<usize>,
    pub concept_out: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatResponse {
    pub reply: String,
    pub route: String,
    pub concept_in: usize,
    pub concept_out: usize,
    pub liquid_score: f64,
    pub rqm_score: Option<f64>,
    pub engrams: usize,
    /// Texto del **LLM decoder only** (concepto de campo → texto).
    pub decoded: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

/// Compat: respuesta síncrona legacy (tests / intents). Preferir job async.
#[derive(Clone, Debug, Serialize)]
pub struct TrainStartResponse {
    pub ok: bool,
    pub epochs: u64,
    pub events: Vec<TrainEvent>,
    pub engrams: usize,
    pub accuracy: Option<f64>,
    pub sleep: Option<SleepReportDto>,
    pub job_id: Option<String>,
}

pub struct AppState {
    pub fuse: FusedLiquidCdt,
    pub field_hybrid: Option<FieldHybridInfer>,
    pub probe: PeripheralProbe,
    pub decoder: ConceptDecoder,
    pub train_log: Vec<TrainEvent>,
    pub chat_log: Vec<ChatTurn>,
    pub last_fuse: Option<FuseReport>,
    pub last_sleep: Option<SleepReport>,
    pub metrics: TelemetrySnapshot,
    pub training: bool,
    pub started_ms: u64,
    pub train_job: TrainJob,
}

impl AppState {
    pub fn new() -> Self {
        let probe = open_best_probe(0x0A6E_471C);
        let mode = probe.mode();
        let fuse = FusedLiquidCdt::new(NUM_CONCEPTS);
        Self {
            fuse,
            field_hybrid: Some(FieldHybridInfer::new(0xF1E1D)),
            probe,
            decoder: ConceptDecoder::new(mode),
            train_log: Vec::new(),
            chat_log: Vec::new(),
            last_fuse: None,
            last_sleep: None,
            metrics: TelemetrySnapshot::default(),
            training: false,
            started_ms: now_ms(),
            train_job: TrainJob::default(),
        }
    }

    pub fn llm_mode(&self) -> LlmMode {
        self.probe.mode()
    }

    pub fn candidates(&self) -> Vec<usize> {
        (0..self.fuse.num_labels).collect()
    }

    fn push_train(&mut self, kind: &str, detail: impl Into<String>, epoch: Option<u64>) {
        let detail = detail.into();
        let ev = TrainEvent {
            t_ms: now_ms().saturating_sub(self.started_ms),
            kind: kind.into(),
            detail: detail.clone(),
            engrams: self.fuse.engram_count(),
            epoch,
        };
        self.train_log.push(ev);
        if self.train_log.len() > 500 {
            let drain = self.train_log.len() - 500;
            self.train_log.drain(0..drain);
        }
    }

    /// Chat agentico: intents sueltos ES/EN + encode→fuse→decode (decoder only).
    pub fn handle_chat(&mut self, message: &str) -> ChatResponse {
        let msg = message.trim();
        let lower = msg.to_lowercase();

        if looks_like_train(&lower) {
            let started = self.begin_live_train(4, 8, 1);
            let reply = if started.ok {
                format!(
                    "Entrenamiento en vivo iniciado (job {}). Lotes async + CDT por lote. \
                     Mira el panel Entrenamiento.",
                    started.job_id
                )
            } else {
                format!("No se pudo iniciar: {}", started.message)
            };
            self.chat_log.push(ChatTurn {
                role: "user".into(),
                text: msg.into(),
                route: None,
                concept_in: None,
                concept_out: None,
            });
            self.chat_log.push(ChatTurn {
                role: "agent".into(),
                text: reply.clone(),
                route: Some("train".into()),
                concept_in: None,
                concept_out: None,
            });
            return ChatResponse {
                reply,
                route: "train".into(),
                concept_in: 0,
                concept_out: 0,
                liquid_score: 0.0,
                rqm_score: None,
                engrams: self.fuse.engram_count(),
                decoded: self.decoder.decode_field_concept(0),
            };
        }

        if looks_like_sleep(&lower) {
            let sleep = self.sleep_now();
            let reply = format!(
                "Sueño consolidado: {} episodios → engramas {}→{} ({} ms).",
                sleep.episodes_consolidated,
                sleep.engrams_before,
                sleep.engrams_after,
                sleep.sleep_ms
            );
            self.chat_log.push(ChatTurn {
                role: "user".into(),
                text: msg.into(),
                route: None,
                concept_in: None,
                concept_out: None,
            });
            self.chat_log.push(ChatTurn {
                role: "agent".into(),
                text: reply.clone(),
                route: Some("sleep".into()),
                concept_in: None,
                concept_out: None,
            });
            return ChatResponse {
                reply,
                route: "sleep".into(),
                concept_in: 0,
                concept_out: 0,
                liquid_score: 0.0,
                rqm_score: None,
                engrams: self.fuse.engram_count(),
                decoded: self.decoder.decode_field_concept(1),
            };
        }

        if looks_like_status(&lower) {
            let reply = format!(
                "Estado: modo={}, engramas={}, wake={}, RQM infer={}, train={}, \
                 Liquid%={:.1}, training={}, job={}",
                self.llm_mode().as_str(),
                self.fuse.engram_count(),
                self.fuse.wake_buffer_len(),
                self.fuse.rqm_infer_calls,
                self.fuse.rqm_train_calls,
                self.metrics.liquid_route_pct(),
                self.training,
                if self.train_job.job_id.is_empty() {
                    "—"
                } else {
                    &self.train_job.job_id
                }
            );
            self.chat_log.push(ChatTurn {
                role: "user".into(),
                text: msg.into(),
                route: None,
                concept_in: None,
                concept_out: None,
            });
            self.chat_log.push(ChatTurn {
                role: "agent".into(),
                text: reply.clone(),
                route: Some("status".into()),
                concept_in: None,
                concept_out: None,
            });
            return ChatResponse {
                reply,
                route: "status".into(),
                concept_in: 0,
                concept_out: 0,
                liquid_score: self.metrics.liquid_score_last,
                rqm_score: None,
                engrams: self.fuse.engram_count(),
                decoded: self.decoder.decode_field_concept(2),
            };
        }

        // Flujo normal: periferia encode → fuse.infer (líquido) → LLM decoder only.
        let concept_in = self.probe.encode_concept(msg);
        let cands = self.candidates();
        let t0 = Instant::now();
        let report = self.fuse.infer(concept_in, &cands);
        let latency_us = t0.elapsed().as_secs_f64() * 1e6;
        self.metrics.record_fuse(&report, latency_us);
        self.metrics.sync_fuse_counters(
            self.fuse.rqm_infer_calls,
            self.fuse.rqm_train_calls,
            self.fuse.engram_count(),
            self.fuse.wake_buffer_len(),
        );
        // Observación tokenless (buffer para sueño); no pasa token ids.
        self.fuse.observe(concept_in, &cands);

        let route = match report.route {
            InferRoute::Liquid => "Liquid",
            InferRoute::RqmFallback => "RqmFallback",
        };
        let decoded = self.decoder.decode_field_concept(report.predicted);
        let reply = self.decoder.agent_reply(
            msg,
            concept_in,
            report.predicted,
            route,
            report.liquid_score,
        );

        self.last_fuse = Some(report.clone());
        self.chat_log.push(ChatTurn {
            role: "user".into(),
            text: msg.into(),
            route: None,
            concept_in: Some(concept_in),
            concept_out: None,
        });
        self.chat_log.push(ChatTurn {
            role: "agent".into(),
            text: reply.clone(),
            route: Some(route.into()),
            concept_in: Some(concept_in),
            concept_out: Some(report.predicted),
        });
        if self.chat_log.len() > 200 {
            let drain = self.chat_log.len() - 200;
            self.chat_log.drain(0..drain);
        }

        ChatResponse {
            reply,
            route: route.into(),
            concept_in,
            concept_out: report.predicted,
            liquid_score: report.liquid_score,
            rqm_score: report.rqm_score,
            engrams: self.fuse.engram_count(),
            decoded,
        }
    }

    /// Prepara job async. El caller debe `spawn` `run_live_train_loop`.
    pub fn begin_live_train(
        &mut self,
        batches: usize,
        batch_size: usize,
        epochs: usize,
    ) -> crate::web::train_job::LiveTrainStartResponse {
        use crate::web::train_job::LiveTrainStartResponse;
        if self.training || self.train_job.running {
            return LiveTrainStartResponse {
                ok: false,
                job_id: self.train_job.job_id.clone(),
                message: "ya hay un entrenamiento en curso".into(),
            };
        }
        let batches = batches.clamp(1, 64);
        let batch_size = batch_size.clamp(1, 64);
        let epochs = epochs.clamp(1, 16);
        let job_id = Uuid::new_v4().to_string();
        self.train_job = TrainJob {
            job_id: job_id.clone(),
            running: true,
            cancelled: false,
            current_batch: 0,
            total_batches: batches,
            epochs,
            batch_size,
            events: std::collections::VecDeque::new(),
            last_checkpoint: self.train_job.last_checkpoint.clone(),
            dataset_size: 0,
            accuracy: None,
            engrams: self.fuse.engram_count(),
            last_decoded: None,
            dataset_source: None,
            event_seq: 0,
            correct: 0,
            total: 0,
        };
        self.training = true;
        self.train_job.push_event(
            "dataset",
            format!("job {job_id}: {batches} lotes × {batch_size}, épocas={epochs}"),
            None,
            Some(self.fuse.engram_count()),
            json!({ "batches": batches, "batch_size": batch_size, "epochs": epochs }),
        );
        self.push_train(
            "start",
            format!("live job={job_id} batches={batches}"),
            None,
        );
        LiveTrainStartResponse {
            ok: true,
            job_id,
            message: "entrenamiento en vivo iniciado".into(),
        }
    }

    pub fn request_train_stop(&mut self) -> bool {
        if !self.train_job.running && !self.training {
            return false;
        }
        self.train_job.cancelled = true;
        self.train_job.push_event(
            "error",
            "cancelación solicitada",
            Some(self.train_job.current_batch),
            Some(self.fuse.engram_count()),
            json!({ "cancelled": true }),
        );
        true
    }

    /// Un lote: dataset LLM → encode → líquido → CDT consolidate → checkpoint.
    /// Devuelve `false` si cancelado o terminó.
    pub fn run_one_live_batch(&mut self) -> bool {
        if !self.train_job.running || self.train_job.cancelled {
            self.finish_live_train(self.train_job.cancelled);
            return false;
        }
        let batch = self.train_job.current_batch;
        if batch >= self.train_job.total_batches {
            self.finish_live_train(false);
            return false;
        }

        let batch_size = self.train_job.batch_size;
        let epochs = self.train_job.epochs;
        let gemma = matches!(self.probe.mode(), LlmMode::GemmaGguf);
        let seed = now_ms().wrapping_add(batch as u64 * 17);

        self.train_job.push_event(
            "batch_start",
            format!("lote {batch}/{}", self.train_job.total_batches),
            Some(batch),
            Some(self.fuse.engram_count()),
            json!({}),
        );

        let (examples, source) = generate_train_batch(batch_size, seed, gemma);
        self.train_job.dataset_source = Some(source.into());
        self.train_job.dataset_size = self.train_job.dataset_size.saturating_add(examples.len());
        self.train_job.push_event(
            "dataset",
            format!(
                "dataset lote {batch}: {} ejemplos (source={source})",
                examples.len()
            ),
            Some(batch),
            Some(self.fuse.engram_count()),
            json!({ "source": source, "n": examples.len() }),
        );

        let cands = self.candidates();
        let mut last_pred = 0usize;
        let mut last_score = 0.0f64;

        for _epoch in 0..epochs {
            for ex in &examples {
                // Encode periferia (firewall); target de curriculum para teach.
                let encoded = self.probe.encode_concept(&ex.text);
                let cue = encoded % NUM_CONCEPTS;
                let target = ex.concept % NUM_CONCEPTS;
                // Inferencia líquida (WavePredictCore vía FusedLiquidCdt).
                let report = self.fuse.infer(cue, &cands);
                self.metrics.record_fuse(&report, 0.0);
                self.fuse.observe(cue, &cands);
                let rel = relation_target(target);
                self.fuse.teach_relation(cue, rel);

                self.train_job.total = self.train_job.total.wrapping_add(1);
                if report.predicted == cue || report.predicted == target || report.predicted == rel
                {
                    self.train_job.correct = self.train_job.correct.wrapping_add(1);
                }
                last_pred = report.predicted;
                last_score = report.liquid_score;
                self.last_fuse = Some(report);
            }
        }

        self.train_job.push_event(
            "infer",
            format!("líquido lote {batch}: score={last_score:.3} pred={last_pred}"),
            Some(batch),
            Some(self.fuse.engram_count()),
            batch_metrics(
                source,
                examples.len(),
                self.train_job.correct,
                self.train_job.total,
                last_score,
            ),
        );

        // CDT consolidation por lote.
        let sleep = self.sleep_now();
        self.train_job.push_event(
            "cdt_consolidate",
            format!(
                "sueño lote {batch}: {} eps → engramas {}",
                sleep.episodes_consolidated, sleep.engrams_after
            ),
            Some(batch),
            Some(sleep.engrams_after),
            json!({
                "episodes": sleep.episodes_consolidated,
                "engrams_before": sleep.engrams_before,
                "engrams_after": sleep.engrams_after,
                "sleep_ms": sleep.sleep_ms,
            }),
        );

        self.metrics.train_correct = self.train_job.correct;
        self.metrics.train_total = self.train_job.total;
        self.metrics.train_epochs_done = self.metrics.train_epochs_done.wrapping_add(1);
        self.train_job.accuracy = self.metrics.train_accuracy();
        self.train_job.engrams = self.fuse.engram_count();

        // Decode preview (LLM decoder only).
        let decoded = self.decoder.decode_field_concept(last_pred);
        self.train_job.last_decoded = Some(decoded.clone());
        self.train_job.push_event(
            "decode",
            format!("decoder: {decoded}"),
            Some(batch),
            Some(self.fuse.engram_count()),
            json!({ "concept": last_pred, "decoded": decoded }),
        );

        // Checkpoint a disco.
        let mut cues: Vec<usize> = self.fuse.relational_cues.iter().copied().collect();
        cues.sort_unstable();
        let ts = crate::web::train_job::now_ms();
        let ck = CheckpointFile {
            job_id: self.train_job.job_id.clone(),
            batch,
            engrams: self.fuse.engram_count(),
            accuracy: self.train_job.accuracy,
            dataset_size: self.train_job.dataset_size,
            dataset_source: self.train_job.dataset_source.clone(),
            sleep: Some(SleepReportDto::from(&sleep)),
            relational_cues: cues,
            events_tail: self
                .train_job
                .events
                .iter()
                .rev()
                .take(40)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect(),
            ts_ms: ts,
        };
        match write_checkpoint(&ck) {
            Ok(path) => {
                let path_s = path.display().to_string();
                self.train_job.last_checkpoint = Some(CheckpointMeta {
                    path: path_s.clone(),
                    batch,
                    engrams: ck.engrams,
                    ts_ms: ts,
                    accuracy: ck.accuracy,
                });
                self.train_job.push_event(
                    "checkpoint",
                    format!("guardado {path_s}"),
                    Some(batch),
                    Some(ck.engrams),
                    json!({ "path": path_s }),
                );
            }
            Err(e) => {
                self.train_job.push_event(
                    "error",
                    format!("checkpoint falló: {e}"),
                    Some(batch),
                    Some(self.fuse.engram_count()),
                    json!({ "error": e.to_string() }),
                );
            }
        }

        self.push_train(
            "batch",
            format!("lote {batch} ok eng={}", self.fuse.engram_count()),
            Some(batch as u64),
        );

        self.train_job.current_batch = batch + 1;
        if self.train_job.current_batch >= self.train_job.total_batches || self.train_job.cancelled
        {
            self.finish_live_train(self.train_job.cancelled);
            return false;
        }
        true
    }

    fn finish_live_train(&mut self, cancelled: bool) {
        self.train_job.running = false;
        self.training = false;
        self.train_job.engrams = self.fuse.engram_count();
        self.train_job.accuracy = self.metrics.train_accuracy();
        let kind = if cancelled { "error" } else { "done" };
        let msg = if cancelled {
            "entrenamiento detenido"
        } else {
            "entrenamiento completado"
        };
        self.train_job.push_event(
            kind,
            msg,
            Some(self.train_job.current_batch),
            Some(self.fuse.engram_count()),
            json!({
                "cancelled": cancelled,
                "accuracy": self.train_job.accuracy,
                "engrams": self.train_job.engrams,
            }),
        );
        self.push_train(kind, msg, None);
        self.metrics.sync_fuse_counters(
            self.fuse.rqm_infer_calls,
            self.fuse.rqm_train_calls,
            self.fuse.engram_count(),
            self.fuse.wake_buffer_len(),
        );
    }

    /// Entrenamiento síncrono legacy (tests / compat). Usa el mismo pipeline por lotes.
    pub fn train_start(
        &mut self,
        epochs: Option<u64>,
        concepts: Option<Vec<usize>>,
    ) -> TrainStartResponse {
        let _ = concepts;
        if self.training || self.train_job.running {
            return TrainStartResponse {
                ok: false,
                epochs: 0,
                events: self.train_log.iter().rev().take(20).cloned().collect(),
                engrams: self.fuse.engram_count(),
                accuracy: self.metrics.train_accuracy(),
                sleep: None,
                job_id: None,
            };
        }
        let epochs_u = epochs.unwrap_or(2).clamp(1, 8) as usize;
        let started = self.begin_live_train(epochs_u, 4, 1);
        if !started.ok {
            return TrainStartResponse {
                ok: false,
                epochs: 0,
                events: self.train_log.iter().rev().take(20).cloned().collect(),
                engrams: self.fuse.engram_count(),
                accuracy: self.metrics.train_accuracy(),
                sleep: None,
                job_id: Some(started.job_id),
            };
        }
        while self.run_one_live_batch() {}
        TrainStartResponse {
            ok: true,
            epochs: epochs_u as u64,
            events: self.train_log.iter().rev().take(40).cloned().collect(),
            engrams: self.fuse.engram_count(),
            accuracy: self.metrics.train_accuracy(),
            sleep: self.last_sleep.as_ref().map(SleepReportDto::from),
            job_id: Some(started.job_id),
        }
    }

    pub fn sleep_now(&mut self) -> SleepReport {
        let sleep = self.fuse.sleep_consolidate();
        self.metrics.record_sleep(
            &sleep,
            self.fuse.engram_count(),
            self.fuse.wake_buffer_len(),
        );
        self.metrics.sync_fuse_counters(
            self.fuse.rqm_infer_calls,
            self.fuse.rqm_train_calls,
            self.fuse.engram_count(),
            self.fuse.wake_buffer_len(),
        );
        self.last_sleep = Some(sleep.clone());
        sleep
    }

    pub fn last_fuse_dto(&self) -> Option<FuseReportDto> {
        self.last_fuse.as_ref().map(FuseReportDto::from)
    }

    pub fn last_sleep_dto(&self) -> Option<SleepReportDto> {
        self.last_sleep.as_ref().map(SleepReportDto::from)
    }

    pub fn live_events_after(&self, after: u64) -> Vec<TrainLiveEvent> {
        self.train_job.events_after(after)
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

fn now_ms() -> u64 {
    crate::web::train_job::now_ms()
}

fn looks_like_train(lower: &str) -> bool {
    lower.contains("entrena")
        || lower.contains("entrenar")
        || lower.contains("start training")
        || lower.contains("train now")
        || lower == "train"
}

fn looks_like_sleep(lower: &str) -> bool {
    lower.contains("sueño")
        || lower.contains("sueno")
        || lower.contains("dormir")
        || lower.contains("consolid")
        || lower.contains("sleep")
}

fn looks_like_status(lower: &str) -> bool {
    lower.contains("estado")
        || lower.contains("status")
        || lower.contains("telemetr")
        || lower == "stats"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_maps_text_to_concept_lexicon() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(99));
        s.decoder = ConceptDecoder::new(LlmMode::Lexicon);
        let r = s.handle_chat("hola campo líquido");
        assert!(r.concept_in < NUM_CONCEPTS);
        assert!(r.concept_out < NUM_CONCEPTS);
        assert!(!r.reply.is_empty());
        assert!(!r.decoded.is_empty());
        assert!(matches!(r.route.as_str(), "Liquid" | "RqmFallback"));
    }

    #[test]
    fn train_increases_engrams_after_sleep() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(1));
        let before = s.fuse.engram_count();
        let r = s.train_start(Some(2), Some(vec![0, 1, 2, 3]));
        assert!(r.ok);
        assert!(r.engrams > before, "engrams {} <= {}", r.engrams, before);
        assert!(r.sleep.is_some());
        assert!(!r.events.is_empty());
        assert!(!s.train_job.events.is_empty());
    }

    #[test]
    fn one_batch_increases_engrams_and_events() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(3));
        let before = s.fuse.engram_count();
        let start = s.begin_live_train(1, 4, 1);
        assert!(start.ok);
        let cont = s.run_one_live_batch();
        assert!(!cont, "un solo lote debe terminar");
        assert!(s.fuse.engram_count() > before);
        assert!(s
            .train_job
            .events
            .iter()
            .any(|e| e.kind == "cdt_consolidate"));
        assert!(s.train_job.last_checkpoint.is_some());
        assert!(s
            .train_job
            .last_decoded
            .as_ref()
            .map(|d| !d.is_empty())
            .unwrap_or(false));
    }

    #[test]
    fn intent_sleep_and_status() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(2));
        let _ = s.handle_chat("hola");
        let sleep = s.handle_chat("sueño por favor");
        assert_eq!(sleep.route, "sleep");
        let st = s.handle_chat("estado");
        assert_eq!(st.route, "status");
    }

    #[test]
    fn train_status_json_shape() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(5));
        let _ = s.begin_live_train(1, 2, 1);
        let snap = s.train_job.snapshot_for_api();
        let v = serde_json::to_value(&snap).unwrap();
        assert!(v.get("job_id").is_some());
        assert!(v.get("running").is_some());
        assert!(v.get("current_batch").is_some());
        assert!(v.get("total_batches").is_some());
        assert!(v.get("events").is_some());
        assert!(v.get("engrams").is_some());
        while s.run_one_live_batch() {}
    }
}
