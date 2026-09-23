//! Estado compartido de la app agentica (Arc<Mutex<AppState>>).

use crate::field_hybrid_infer::FieldHybridInfer;
use crate::liquid_cdt_memory::SleepReport;
use crate::liquid_cdt_rqm_fuse::{FuseReport, FusedLiquidCdt, InferRoute};
use crate::web::experiment_suite::{run_ui_experiment_suite, ExperimentSuiteReport};
use crate::web::field_eval::{run_field_eval_with_progress, FieldEvalReport, FieldEvalStatus};
use crate::web::llm_periphery::{
    generate_train_batch, open_best_probe, ConceptDecoder, LlmMode, PeripheralProbe, NUM_CONCEPTS,
};
use crate::web::process_job::{
    ProcessesSnapshot, SleepJob, SleepStartResponse, TestsJob, TestsStartResponse,
};
use crate::web::sleep_optimize::{
    run_sleep_optimize_with_progress, SleepOptimizeOpts, SleepOptimizeReport,
};
use crate::web::telemetry::{FuseReportDto, SleepReportDto, TelemetrySnapshot};
use crate::web::train_job::{
    batch_metrics, relation_target, write_checkpoint, write_dataset_checkpoint, write_latest_index,
    CheckpointFile, CheckpointMeta, DatasetCheckpointFile, LiquidDatasetMetrics, TrainJob,
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
    pub sleep_job: SleepJob,
    pub tests_job: TestsJob,
    pub last_sleep_optimize: Option<SleepOptimizeReport>,
    pub last_field_eval: Option<FieldEvalReport>,
    pub field_eval_running: bool,
    /// True tras completar ≥1 lote de train (o si hay engramas/datasets).
    pub ever_trained: bool,
    /// Última suite de experimentos (pestaña Pruebas).
    pub last_experiment_suite: Option<ExperimentSuiteReport>,
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
            sleep_job: SleepJob::default(),
            tests_job: TestsJob::default(),
            last_sleep_optimize: None,
            last_field_eval: None,
            field_eval_running: false,
            ever_trained: false,
            last_experiment_suite: None,
        }
    }

    /// Evidencia de entrenamiento previo para gate de sueño.
    pub fn has_training_evidence(&self) -> bool {
        self.ever_trained || self.train_job.datasets_saved > 0 || self.fuse.engram_count() > 0
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
            let started = self.begin_live_train(None, 8, 1);
            let reply = if started.ok {
                format!(
                    "Entrenamiento en vivo iniciado (job {}). Modo infinito · CDT por dataset. \
                     Mira el panel Entrenamiento. Usa Detener para parar.",
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
            if !self.has_training_evidence() {
                let reply = "Necesitas entrenar antes de dormir (al menos 1 lote o engramas/datasets_saved > 0). Usa la pestaña Entrenamiento.".to_string();
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
        let eng = self.fuse.engram_count();
        let reply = format!(
            "{reply}\n\n[memoria de campo] engramas={eng} · concepto recordado={} · interpretación decoder-only del modelo de campo (no chat genérico).",
            decoded
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

    /// Prepara job async. El caller debe `spawn` el loop de lotes.
    /// `batches = None` (o 0) → entrenamiento **infinito** hasta cancel/stop.
    pub fn begin_live_train(
        &mut self,
        batches: Option<usize>,
        batch_size: usize,
        epochs: usize,
    ) -> crate::web::train_job::LiveTrainStartResponse {
        use crate::web::train_job::LiveTrainStartResponse;
        if self.training || self.train_job.running {
            return LiveTrainStartResponse {
                ok: false,
                job_id: self.train_job.job_id.clone(),
                message: "ya hay un entrenamiento en curso".into(),
                infinite: None,
            };
        }
        let infinite = batches.is_none() || batches == Some(0);
        let total_batches = if infinite {
            None
        } else {
            Some(batches.unwrap_or(1).clamp(1, 64))
        };
        let batch_size = batch_size.clamp(1, 64);
        let epochs = epochs.clamp(1, 16);
        let job_id = Uuid::new_v4().to_string();
        self.train_job = TrainJob {
            job_id: job_id.clone(),
            running: true,
            cancelled: false,
            current_batch: 0,
            total_batches,
            infinite,
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
            datasets_saved: 0,
            last_dataset_path: self.train_job.last_dataset_path.clone(),
            last_dataset_family: None,
            last_experiment_ids: Vec::new(),
        };
        self.training = true;
        let mode_msg = if infinite {
            format!("job {job_id}: ∞ infinito × {batch_size}/lote, épocas={epochs}")
        } else {
            format!(
                "job {job_id}: {} lotes × {batch_size}, épocas={epochs}",
                total_batches.unwrap_or(0)
            )
        };
        self.train_job.push_event(
            "dataset",
            mode_msg,
            None,
            Some(self.fuse.engram_count()),
            json!({
                "batches": total_batches,
                "infinite": infinite,
                "batch_size": batch_size,
                "epochs": epochs
            }),
        );
        self.push_train(
            "start",
            format!(
                "live job={job_id} {}",
                if infinite {
                    "infinite".into()
                } else {
                    format!("batches={}", total_batches.unwrap_or(0))
                }
            ),
            None,
        );
        LiveTrainStartResponse {
            ok: true,
            job_id,
            message: if infinite {
                "entrenamiento infinito iniciado".into()
            } else {
                "entrenamiento en vivo iniciado".into()
            },
            infinite: Some(infinite),
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

    /// Un lote/dataset: LLM → encode → líquido → CDT → checkpoint por dataset.
    /// Devuelve `false` solo si cancelado o (modo finito) se alcanzó el tope.
    /// En modo infinito nunca termina por conteo de lotes.
    pub fn run_one_live_batch(&mut self) -> bool {
        if !self.train_job.running || self.train_job.cancelled {
            self.finish_live_train(self.train_job.cancelled);
            return false;
        }
        let batch = self.train_job.current_batch;
        if !self.train_job.infinite {
            if let Some(total) = self.train_job.total_batches {
                if batch >= total {
                    self.finish_live_train(false);
                    return false;
                }
            }
        }

        let batch_size = self.train_job.batch_size;
        let epochs = self.train_job.epochs;
        let gemma = matches!(self.probe.mode(), LlmMode::GemmaGguf);
        let seed = now_ms().wrapping_add(batch as u64 * 17);
        let engrams_before = self.fuse.engram_count();

        let batch_label = if self.train_job.infinite {
            format!("lote {batch} (∞)")
        } else {
            format!("lote {batch}/{}", self.train_job.total_batches.unwrap_or(0))
        };
        self.train_job.push_event(
            "batch_start",
            batch_label,
            Some(batch),
            Some(engrams_before),
            json!({ "infinite": self.train_job.infinite }),
        );

        let (examples, meta) = generate_train_batch(batch_size, seed, gemma);
        let source = meta.source;
        let family = meta.dataset_family;
        let exp_ids: Vec<String> = meta
            .experiment_ids
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        self.train_job.dataset_source = Some(source.into());
        self.train_job.last_dataset_family = Some(family.into());
        self.train_job.last_experiment_ids = exp_ids.clone();
        self.train_job.dataset_size = self.train_job.dataset_size.saturating_add(examples.len());
        self.train_job.push_event(
            "dataset",
            format!(
                "dataset lote {batch}: {} ejemplos (source={source}, familia={family}, exps={:?})",
                examples.len(),
                meta.experiment_ids
            ),
            Some(batch),
            Some(self.fuse.engram_count()),
            json!({
                "source": source,
                "n": examples.len(),
                "source_tag": "llm_dataset_decoupled",
                "dataset_family": family,
                "experiment_ids": meta.experiment_ids,
            }),
        );

        let cands = self.candidates();
        let mut last_pred = 0usize;
        let mut last_score = 0.0f64;
        let mut scores: Vec<f64> = Vec::with_capacity(examples.len() * epochs);
        let mut preds: Vec<usize> = Vec::with_capacity(examples.len() * epochs);

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
                scores.push(report.liquid_score);
                preds.push(report.predicted);
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

        // CDT consolidation por dataset.
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

        let ts = crate::web::train_job::now_ms();
        let dataset_id = format!("ds_{}_{}", self.train_job.job_id, batch);
        let ds_file = DatasetCheckpointFile {
            dataset_id: dataset_id.clone(),
            job_id: self.train_job.job_id.clone(),
            batch,
            source: source.into(),
            dataset_family: Some(family.into()),
            experiment_ids: exp_ids.clone(),
            examples: examples.clone(),
            liquid: LiquidDatasetMetrics {
                scores,
                preds,
                correct: self.train_job.correct,
                total: self.train_job.total,
                last_score,
                last_pred,
            },
            sleep: Some(SleepReportDto::from(&sleep)),
            decoder_preview: Some(decoded.clone()),
            engrams: self.fuse.engram_count(),
            accuracy: self.train_job.accuracy,
            ts_ms: ts,
        };
        match write_dataset_checkpoint(&ds_file) {
            Ok(path) => {
                let path_s = path.display().to_string();
                self.train_job.datasets_saved = self.train_job.datasets_saved.saturating_add(1);
                self.ever_trained = true;
                self.train_job.last_dataset_path = Some(path_s.clone());
                let _ = write_latest_index(
                    &path_s,
                    &self.train_job.job_id,
                    batch,
                    self.train_job.datasets_saved,
                );
                self.train_job.push_event(
                    "checkpoint",
                    format!("dataset guardado {path_s}"),
                    Some(batch),
                    Some(ds_file.engrams),
                    json!({
                        "path": path_s,
                        "dataset_id": dataset_id,
                        "kind": "dataset"
                    }),
                );
            }
            Err(e) => {
                self.train_job.push_event(
                    "error",
                    format!("dataset checkpoint falló: {e}"),
                    Some(batch),
                    Some(self.fuse.engram_count()),
                    json!({ "error": e.to_string() }),
                );
            }
        }

        // Resumen de lote (compat).
        let mut cues: Vec<usize> = self.fuse.relational_cues.iter().copied().collect();
        cues.sort_unstable();
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
                    format!("resumen lote {path_s}"),
                    Some(batch),
                    Some(ck.engrams),
                    json!({ "path": path_s, "kind": "batch_summary" }),
                );
            }
            Err(e) => {
                self.train_job.push_event(
                    "error",
                    format!("checkpoint resumen falló: {e}"),
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
        // Lote completado (aunque falle checkpoint): evidencia de train para sueño.
        self.ever_trained = true;

        if self.train_job.cancelled {
            self.finish_live_train(true);
            return false;
        }
        if !self.train_job.infinite {
            if let Some(total) = self.train_job.total_batches {
                if self.train_job.current_batch >= total {
                    self.finish_live_train(false);
                    return false;
                }
            }
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
        let started = self.begin_live_train(Some(epochs_u), 4, 1);
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
        let _ = self.sleep_optimize(SleepOptimizeOpts::default());
        self.last_sleep.clone().unwrap_or(SleepReport {
            episodes_consolidated: 0,
            engrams_before: 0,
            engrams_after: self.fuse.engram_count(),
            used_rqm: false,
            rqm_relations_trained: 0,
            sleep_ms: 0.0,
        })
    }

    /// Sueño + poda/compactación + minimización de energía libre (síncrono / tests).
    pub fn sleep_optimize(&mut self, opts: SleepOptimizeOpts) -> SleepOptimizeReport {
        self.sleep_optimize_with_progress(opts, |_phase, _msg| {})
    }

    fn sleep_optimize_with_progress(
        &mut self,
        opts: SleepOptimizeOpts,
        mut on_progress: impl FnMut(&str, &str),
    ) -> SleepOptimizeReport {
        let cfg = self
            .field_hybrid
            .as_ref()
            .map(|h| h.cfg)
            .unwrap_or_default();
        if self.field_hybrid.is_none() {
            self.field_hybrid = Some(FieldHybridInfer::new(0x51EE_0001));
        }
        let field = &mut self.field_hybrid.as_mut().unwrap().field;
        let report =
            run_sleep_optimize_with_progress(&mut self.fuse, field, &cfg, opts, &mut on_progress);
        let sleep = SleepReport {
            episodes_consolidated: report.episodes_consolidated,
            engrams_before: report.engrams_before,
            engrams_after: report.engrams_after,
            used_rqm: report.used_rqm,
            rqm_relations_trained: report.rqm_relations_trained,
            sleep_ms: report.sleep_ms,
        };
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
        self.last_sleep = Some(sleep);
        self.last_sleep_optimize = Some(report.clone());
        report
    }

    /// Arranca job async de sueño si no hay otro en curso.
    /// `infinite` / `total_cycles = None` → ciclos hasta Detener.
    pub fn begin_sleep_job(
        &mut self,
        opts: SleepOptimizeOpts,
        infinite: bool,
        total_cycles: Option<usize>,
    ) -> SleepStartResponse {
        if self.sleep_job.running {
            return SleepStartResponse {
                ok: false,
                job_id: self.sleep_job.job_id.clone(),
                message: "ya hay un sueño en curso".into(),
                infinite: None,
            };
        }
        if !self.has_training_evidence() {
            return SleepStartResponse {
                ok: false,
                job_id: self.sleep_job.job_id.clone(),
                message: "necesitas entrenar antes de dormir (al menos 1 lote o engramas/datasets_saved > 0)".into(),
                infinite: None,
            };
        }
        let infinite = infinite || total_cycles.is_none() || total_cycles == Some(0);
        let total_cycles = if infinite {
            None
        } else {
            Some(total_cycles.unwrap_or(1).max(1))
        };
        let job_id = Uuid::new_v4().to_string();
        let keep_report = self.sleep_job.last_report.clone();
        self.sleep_job = SleepJob {
            job_id: job_id.clone(),
            running: true,
            cancelled: false,
            phase: "queued".into(),
            prune_intensity: opts.prune_intensity,
            compact_intensity: opts.compact_intensity,
            consolidate_first: opts.consolidate_first,
            current_cycle: 0,
            total_cycles,
            infinite,
            events: Default::default(),
            event_seq: 0,
            started_ms: Some(now_ms()),
            finished_ms: None,
            last_report: keep_report,
        };
        let mode = if infinite {
            "∞ infinito".to_string()
        } else {
            format!("{} ciclo(s)", total_cycles.unwrap_or(1))
        };
        self.sleep_job.push_event(
            "start",
            format!(
                "sueño job {job_id} ({mode}; prune={:.2} compact={:.2})",
                opts.prune_intensity, opts.compact_intensity
            ),
            serde_json::json!({ "infinite": infinite, "total_cycles": total_cycles }),
        );
        SleepStartResponse {
            ok: true,
            job_id,
            message: if infinite {
                "sueño infinito iniciado".into()
            } else {
                "sueño iniciado".into()
            },
            infinite: Some(infinite),
        }
    }

    pub fn request_sleep_stop(&mut self) -> bool {
        if !self.sleep_job.running {
            return false;
        }
        self.sleep_job.cancelled = true;
        self.sleep_job.phase = "cancelling".into();
        self.sleep_job
            .push_event("cancel", "cancelación solicitada", serde_json::json!({}));
        true
    }

    /// Un ciclo de sueño. Devuelve `true` si el spawn debe continuar.
    pub fn run_one_sleep_cycle(&mut self) -> bool {
        if !self.sleep_job.running {
            return false;
        }
        if self.sleep_job.cancelled {
            self.finish_sleep_job(true);
            return false;
        }
        if !self.sleep_job.infinite {
            if let Some(total) = self.sleep_job.total_cycles {
                if self.sleep_job.current_cycle >= total {
                    self.finish_sleep_job(false);
                    return false;
                }
            }
        }

        let cycle = self.sleep_job.current_cycle;
        let opts = self.sleep_job.opts();
        let label = if self.sleep_job.infinite {
            format!("ciclo {cycle} (∞)")
        } else {
            format!("ciclo {cycle}/{}", self.sleep_job.total_cycles.unwrap_or(1))
        };
        self.sleep_job.phase = "running".into();
        self.sleep_job.push_event(
            "cycle",
            format!("inicio {label}"),
            serde_json::json!({
                "cycle": cycle,
                "infinite": self.sleep_job.infinite,
            }),
        );

        let mut progress: Vec<(String, String)> = Vec::new();
        let report = self.sleep_optimize_with_progress(opts, |phase, msg| {
            progress.push((phase.to_string(), msg.to_string()));
        });
        for (phase, msg) in &progress {
            self.sleep_job.phase = phase.clone();
            self.sleep_job.push_event(
                phase,
                msg.clone(),
                serde_json::json!({
                    "cycle": cycle,
                    "free_energy_after": report.free_energy_after,
                    "symmetry_after": report.symmetry_after,
                }),
            );
            if self.sleep_job.cancelled {
                break;
            }
        }
        self.sleep_job.last_report = Some(report.clone());
        self.sleep_job.push_event(
            "cycle_done",
            format!(
                "{label} listo ΔF={:.4} Δsym={:.4}",
                report.free_energy_after - report.free_energy_before,
                report.symmetry_after - report.symmetry_before
            ),
            serde_json::json!({
                "cycle": cycle,
                "free_energy_after": report.free_energy_after,
                "symmetry_after": report.symmetry_after,
                "handshake": report.handshake,
            }),
        );
        self.sleep_job.current_cycle = cycle.saturating_add(1);

        if self.sleep_job.cancelled {
            self.finish_sleep_job(true);
            return false;
        }
        if !self.sleep_job.infinite {
            if let Some(total) = self.sleep_job.total_cycles {
                if self.sleep_job.current_cycle >= total {
                    self.finish_sleep_job(false);
                    return false;
                }
            }
        }
        true
    }

    fn finish_sleep_job(&mut self, cancelled: bool) {
        self.sleep_job.running = false;
        self.sleep_job.finished_ms = Some(now_ms());
        if cancelled {
            self.sleep_job.phase = "cancelled".into();
            self.sleep_job
                .push_event("done", "sueño cancelado", serde_json::json!({}));
        } else {
            self.sleep_job.phase = "done".into();
            self.sleep_job
                .push_event("done", "sueño completado", serde_json::json!({}));
        }
    }

    /// Compat: ejecuta todos los ciclos finitos en el hilo actual.
    pub fn execute_sleep_job(&mut self) {
        while self.run_one_sleep_cycle() {}
    }

    /// Suite de pruebas síncrona (compat / unit tests).
    pub fn run_tests(&mut self) -> FieldEvalReport {
        self.field_eval_running = true;
        let report = self.run_tests_with_progress(|_s, _t, _m| {});
        self.field_eval_running = false;
        report
    }

    fn run_tests_with_progress(
        &mut self,
        mut on_progress: impl FnMut(usize, usize, &str),
    ) -> FieldEvalReport {
        let cfg = self
            .field_hybrid
            .as_ref()
            .map(|h| h.cfg)
            .unwrap_or_default();
        let field_owned = self.field_hybrid.as_ref().map(|h| h.field.clone());
        let last_sleep = self.last_sleep_optimize.clone();
        let report = run_field_eval_with_progress(
            &mut self.fuse,
            field_owned.as_ref(),
            &cfg,
            last_sleep.as_ref(),
            &mut on_progress,
        );
        self.last_field_eval = Some(report.clone());
        report
    }

    pub fn begin_tests_job(
        &mut self,
        infinite: bool,
        total_cycles: Option<usize>,
    ) -> TestsStartResponse {
        if self.tests_job.running || self.field_eval_running {
            return TestsStartResponse {
                ok: false,
                job_id: self.tests_job.job_id.clone(),
                message: "ya hay pruebas en curso".into(),
                infinite: None,
            };
        }
        let infinite = infinite || total_cycles.is_none() || total_cycles == Some(0);
        let total_cycles = if infinite {
            None
        } else {
            Some(total_cycles.unwrap_or(1).max(1))
        };
        let job_id = Uuid::new_v4().to_string();
        let keep = self
            .tests_job
            .last_report
            .clone()
            .or_else(|| self.last_field_eval.clone());
        let total = self.fuse.num_labels.max(1);
        self.tests_job = TestsJob {
            job_id: job_id.clone(),
            running: true,
            cancelled: false,
            phase: "queued".into(),
            step: 0,
            total_steps: total,
            current_cycle: 0,
            total_cycles,
            infinite,
            events: Default::default(),
            event_seq: 0,
            started_ms: Some(now_ms()),
            finished_ms: None,
            last_report: keep,
        };
        self.field_eval_running = true;
        let mode = if infinite {
            "∞ infinito".to_string()
        } else {
            format!("{} batería(s)", total_cycles.unwrap_or(1))
        };
        self.tests_job.push_event(
            "start",
            format!("pruebas job {job_id} ({mode})"),
            serde_json::json!({
                "total": total,
                "infinite": infinite,
                "total_cycles": total_cycles,
            }),
        );
        TestsStartResponse {
            ok: true,
            job_id,
            message: if infinite {
                "pruebas infinitas iniciadas".into()
            } else {
                "pruebas iniciadas".into()
            },
            infinite: Some(infinite),
        }
    }

    pub fn request_tests_stop(&mut self) -> bool {
        if !self.tests_job.running && !self.field_eval_running {
            return false;
        }
        self.tests_job.cancelled = true;
        self.tests_job.phase = "cancelling".into();
        self.tests_job
            .push_event("cancel", "cancelación solicitada", serde_json::json!({}));
        true
    }

    /// Una batería de eval. Devuelve `true` para continuar el loop.
    pub fn run_one_tests_cycle(&mut self) -> bool {
        if !self.tests_job.running {
            return false;
        }
        if self.tests_job.cancelled {
            self.finish_tests_job(true);
            return false;
        }
        if !self.tests_job.infinite {
            if let Some(total) = self.tests_job.total_cycles {
                if self.tests_job.current_cycle >= total {
                    self.finish_tests_job(false);
                    return false;
                }
            }
        }

        let cycle = self.tests_job.current_cycle;
        let label = if self.tests_job.infinite {
            format!("batería {cycle} (∞)")
        } else {
            format!(
                "batería {cycle}/{}",
                self.tests_job.total_cycles.unwrap_or(1)
            )
        };
        self.tests_job.phase = "running".into();
        self.tests_job.push_event(
            "cycle",
            format!("inicio {label}"),
            serde_json::json!({
                "cycle": cycle,
                "infinite": self.tests_job.infinite,
            }),
        );

        // 1) Suite de experimentos (siempre; harness independientes).
        self.tests_job.phase = "experiment_suite".into();
        self.tests_job.push_event(
            "suite_start",
            "suite experimentos E8–E10 / E13·E15 / Clean-Room v2 smoke",
            serde_json::json!({ "cycle": cycle }),
        );
        let suite = run_ui_experiment_suite();
        let suite_summary = suite.summarize_counts();
        self.last_experiment_suite = Some(suite.clone());
        self.tests_job.push_event(
            "suite_done",
            format!(
                "suite: {} filas · {} · {:.0} ms",
                suite.rows.len(),
                suite_summary,
                suite.elapsed_ms
            ),
            serde_json::json!({
                "cycle": cycle,
                "verdict_counts": suite.verdict_counts,
                "elapsed_ms": suite.elapsed_ms,
                "rows": suite.rows,
                "notes": suite.notes,
            }),
        );
        if self.tests_job.cancelled {
            self.finish_tests_job(true);
            return false;
        }

        // 2) field_eval del modelo (condicionado a engramas/sueño → notas honestas).
        let mut progress: Vec<(usize, usize, String)> = Vec::new();
        let mut report = self.run_tests_with_progress(|step, total, msg| {
            progress.push((step, total, msg.to_string()));
        });
        if !report.had_engrams {
            report.notes.push(
                "field_eval sin engramas/sin sueño consolidado: se reporta estado actual; suite de experimentos sí corrió"
                    .into(),
            );
        } else if self.last_sleep_optimize.is_none() && self.last_sleep.is_none() {
            report
                .notes
                .push("había engramas pero no hay informe de sueño optimizado reciente".into());
        }
        report.experiment_suite = Some(suite);
        for (step, total, msg) in progress {
            self.tests_job.step = step;
            self.tests_job.total_steps = total;
            self.tests_job.phase = "running".into();
            self.tests_job.push_event(
                "step",
                msg,
                serde_json::json!({
                    "step": step,
                    "total": total,
                    "cycle": cycle,
                }),
            );
            if self.tests_job.cancelled {
                break;
            }
        }
        self.tests_job.last_report = Some(report.clone());
        self.last_field_eval = Some(report);
        self.tests_job.push_event(
            "cycle_done",
            format!("{label} lista · suite {suite_summary}"),
            serde_json::json!({ "cycle": cycle, "suite_summary": suite_summary }),
        );
        self.tests_job.current_cycle = cycle.saturating_add(1);

        if self.tests_job.cancelled {
            self.finish_tests_job(true);
            return false;
        }
        if !self.tests_job.infinite {
            if let Some(total) = self.tests_job.total_cycles {
                if self.tests_job.current_cycle >= total {
                    self.finish_tests_job(false);
                    return false;
                }
            }
        }
        true
    }

    fn finish_tests_job(&mut self, cancelled: bool) {
        self.tests_job.running = false;
        self.field_eval_running = false;
        self.tests_job.finished_ms = Some(now_ms());
        if cancelled {
            self.tests_job.phase = "cancelled".into();
            self.tests_job
                .push_event("done", "pruebas canceladas", serde_json::json!({}));
        } else {
            self.tests_job.phase = "done".into();
            self.tests_job
                .push_event("done", "pruebas completadas", serde_json::json!({}));
        }
    }

    /// Compat: ejecuta todas las baterías finitas en el hilo actual.
    pub fn execute_tests_job(&mut self) {
        while self.run_one_tests_cycle() {}
    }

    pub fn tests_status(&self) -> FieldEvalStatus {
        FieldEvalStatus {
            running: self.tests_job.running || self.field_eval_running,
            last: self
                .tests_job
                .last_report
                .clone()
                .or_else(|| self.last_field_eval.clone()),
        }
    }

    pub fn processes_snapshot(&self) -> ProcessesSnapshot {
        let mut active = Vec::new();
        if self.train_job.running || self.training {
            active.push("train".into());
        }
        if self.sleep_job.running {
            active.push("sleep".into());
        }
        if self.tests_job.running || self.field_eval_running {
            active.push("tests".into());
        }
        ProcessesSnapshot {
            train: self.train_job.snapshot_for_api(),
            sleep: self.sleep_job.snapshot_for_api(),
            tests: self.tests_job.snapshot_for_api(),
            active,
        }
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
        let start = s.begin_live_train(Some(1), 4, 1);
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
        let blocked = s.handle_chat("sueño por favor");
        assert_eq!(blocked.route, "sleep");
        assert!(
            blocked.reply.to_lowercase().contains("entrenar"),
            "sin train debe pedir entrenar: {}",
            blocked.reply
        );
        // Tras un lote, el sueño de chat sí consolida.
        let start = s.begin_live_train(Some(1), 2, 1);
        assert!(start.ok);
        while s.run_one_live_batch() {}
        let sleep = s.handle_chat("sueño por favor");
        assert_eq!(sleep.route, "sleep");
        assert!(sleep.reply.to_lowercase().contains("consolid"));
        let st = s.handle_chat("estado");
        assert_eq!(st.route, "status");
    }

    #[test]
    fn sleep_job_rejects_without_training() {
        let mut s = AppState::new();
        assert!(!s.has_training_evidence());
        let r = s.begin_sleep_job(SleepOptimizeOpts::default(), false, Some(1));
        assert!(!r.ok);
        assert!(
            r.message.to_lowercase().contains("entrenar"),
            "msg={}",
            r.message
        );
    }

    #[test]
    fn sleep_job_allows_after_one_batch() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(11));
        let start = s.begin_live_train(Some(1), 2, 1);
        assert!(start.ok);
        while s.run_one_live_batch() {}
        assert!(s.has_training_evidence());
        assert!(s.ever_trained);
        let r = s.begin_sleep_job(SleepOptimizeOpts::default(), false, Some(1));
        assert!(r.ok, "msg={}", r.message);
        assert!(s.run_one_sleep_cycle() || !s.sleep_job.running);
    }

    #[test]
    fn train_status_json_shape() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(5));
        let _ = s.begin_live_train(Some(1), 2, 1);
        let snap = s.train_job.snapshot_for_api();
        let v = serde_json::to_value(&snap).unwrap();
        assert!(v.get("job_id").is_some());
        assert!(v.get("running").is_some());
        assert!(v.get("current_batch").is_some());
        assert!(v.get("total_batches").is_some());
        assert!(v.get("infinite").is_some());
        assert!(v.get("datasets_saved").is_some());
        assert!(v.get("events").is_some());
        assert!(v.get("engrams").is_some());
        while s.run_one_live_batch() {}
    }

    #[test]
    fn infinite_train_runs_until_cancel_saves_datasets() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(7));
        let before = s.fuse.engram_count();
        let start = s.begin_live_train(None, 4, 1);
        assert!(start.ok);
        assert_eq!(start.infinite, Some(true));
        assert!(s.train_job.infinite);
        assert!(s.train_job.total_batches.is_none());

        assert!(s.run_one_live_batch());
        assert!(s.run_one_live_batch());
        assert!(s.run_one_live_batch());
        assert!(s.train_job.running);
        assert_eq!(s.train_job.current_batch, 3);
        assert_eq!(s.train_job.datasets_saved, 3);
        assert!(s.fuse.engram_count() > before);
        assert!(s.train_job.last_dataset_path.is_some());

        let ds_dir = crate::web::train_job::datasets_dir();
        let n_files = std::fs::read_dir(&ds_dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                    .count()
            })
            .unwrap_or(0);
        assert!(
            n_files >= 3,
            "expected ≥3 dataset files in {:?}, got {n_files}",
            ds_dir
        );

        assert!(s.request_train_stop());
        let cont = s.run_one_live_batch();
        assert!(!cont, "tras cancel el siguiente lote debe devolver false");
        assert!(!s.train_job.running);
    }

    #[test]
    fn chat_does_not_feed_train_dataset() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(3));
        s.decoder = ConceptDecoder::new(LlmMode::Lexicon);
        let _ = s.handle_chat("hola campo de prueba único xyz");
        assert!(
            s.train_job.events.iter().all(|e| e.kind != "dataset"),
            "chat must not create train dataset events"
        );
        assert!(s.chat_log.len() >= 2);
    }

    #[test]
    fn sleep_optimize_report_improves() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(9));
        let cands = s.candidates();
        for i in 0..8 {
            s.fuse.observe(i, &cands);
            s.fuse.teach_relation(i, (i + 2) % 8);
        }
        let report = s.sleep_optimize(SleepOptimizeOpts {
            prune_intensity: 0.5,
            compact_intensity: 0.6,
            consolidate_first: true,
        });
        let energy_ok = report.free_energy_after <= report.free_energy_before + 1e-6;
        let sym_ok = report.symmetry_after + 1e-9 >= report.symmetry_before;
        assert!(energy_ok || sym_ok);
        assert!(serde_json::to_value(&report)
            .unwrap()
            .get("routes_pruned")
            .is_some());
    }

    #[test]
    fn field_eval_runs_without_engrams() {
        let mut s = AppState::new();
        let r = s.run_tests();
        assert_eq!(r.identity_total, NUM_CONCEPTS);
        assert!(!r.had_engrams);
    }

    #[test]
    fn batches_zero_means_infinite() {
        let mut s = AppState::new();
        s.probe =
            PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(11));
        let start = s.begin_live_train(Some(0), 2, 1);
        assert!(start.ok);
        assert!(s.train_job.infinite);
        assert!(s.request_train_stop());
        assert!(!s.run_one_live_batch());
    }
}
