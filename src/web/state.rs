//! Estado compartido de la app agentica (Arc<Mutex<AppState>>).

use crate::field_hybrid_infer::FieldHybridInfer;
use crate::liquid_cdt_memory::SleepReport;
use crate::liquid_cdt_rqm_fuse::{FuseReport, FusedLiquidCdt, InferRoute};
use crate::web::llm_periphery::{
    open_best_probe, ConceptDecoder, LlmMode, PeripheralProbe, NUM_CONCEPTS,
};
use crate::web::telemetry::{FuseReportDto, SleepReportDto, TelemetrySnapshot};
use serde::{Deserialize, Serialize};
use std::time::Instant;

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
    pub decoded: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrainStartRequest {
    pub epochs: Option<u64>,
    pub concepts: Option<Vec<usize>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainStartResponse {
    pub ok: bool,
    pub epochs: u64,
    pub events: Vec<TrainEvent>,
    pub engrams: usize,
    pub accuracy: Option<f64>,
    pub sleep: Option<SleepReportDto>,
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
}

impl AppState {
    pub fn new() -> Self {
        let probe = open_best_probe(0xA6E4_71C);
        let mode = probe.mode();
        let mut fuse = FusedLiquidCdt::new(NUM_CONCEPTS);
        // Candidatos 0..N-1 para inferencia.
        let _ = &mut fuse;
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
        }
    }

    pub fn llm_mode(&self) -> LlmMode {
        self.probe.mode()
    }

    pub fn candidates(&self) -> Vec<usize> {
        (0..self.fuse.num_labels).collect()
    }

    fn push_train(&mut self, kind: &str, detail: impl Into<String>, epoch: Option<u64>) {
        let ev = TrainEvent {
            t_ms: now_ms().saturating_sub(self.started_ms),
            kind: kind.into(),
            detail: detail.into(),
            engrams: self.fuse.engram_count(),
            epoch,
        };
        self.train_log.push(ev);
        if self.train_log.len() > 500 {
            let drain = self.train_log.len() - 500;
            self.train_log.drain(0..drain);
        }
    }

    /// Chat agentico: intents sueltos ES/EN + encode→fuse→decode.
    pub fn handle_chat(&mut self, message: &str) -> ChatResponse {
        let msg = message.trim();
        let lower = msg.to_lowercase();

        if looks_like_train(&lower) {
            let r = self.train_start(Some(2), None);
            let reply = format!(
                "Entrenamiento tokenless iniciado ({} épocas). Engramas: {}. Acc: {:?}.",
                r.epochs,
                r.engrams,
                r.accuracy
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
                decoded: "entrenamiento".into(),
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
                decoded: "sueño".into(),
            };
        }

        if looks_like_status(&lower) {
            let reply = format!(
                "Estado: modo={}, engramas={}, wake={}, RQM infer={}, train={}, \
                 Liquid%={:.1}, training={}",
                self.llm_mode().as_str(),
                self.fuse.engram_count(),
                self.fuse.wake_buffer_len(),
                self.fuse.rqm_infer_calls,
                self.fuse.rqm_train_calls,
                self.metrics.liquid_route_pct(),
                self.training
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
                decoded: "estado".into(),
            };
        }

        // Flujo normal: periferia → concepto → fuse.infer → decode.
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
        let decoded = self.decoder.decode(report.predicted);
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

    /// Entrenamiento tokenless: observe/teach loops + sueño opcional.
    pub fn train_start(
        &mut self,
        epochs: Option<u64>,
        concepts: Option<Vec<usize>>,
    ) -> TrainStartResponse {
        if self.training {
            return TrainStartResponse {
                ok: false,
                epochs: 0,
                events: self.train_log.iter().rev().take(20).cloned().collect(),
                engrams: self.fuse.engram_count(),
                accuracy: self.metrics.train_accuracy(),
                sleep: None,
            };
        }
        self.training = true;
        let epochs = epochs.unwrap_or(3).clamp(1, 32);
        let concepts: Vec<usize> = concepts.unwrap_or_else(|| (0..NUM_CONCEPTS).collect());
        let cands = self.candidates();
        let mut correct = 0u64;
        let mut total = 0u64;

        self.push_train(
            "start",
            format!("épocas={epochs} conceptos={concepts:?}"),
            None,
        );

        for epoch in 0..epochs {
            for &c in &concepts {
                let cue = c % NUM_CONCEPTS;
                // Identidad tokenless.
                self.fuse.observe(cue, &cands);
                // Relación ligera cue → (cue+1)%N para ejercitar RQM en sueño.
                let target = (cue + 1) % NUM_CONCEPTS;
                self.fuse.teach_relation(cue, target);

                let report = self.fuse.infer(cue, &cands);
                total += 1;
                if report.predicted == cue || report.predicted == target {
                    correct += 1;
                }
                self.metrics.record_fuse(&report, 0.0);
            }
            self.push_train(
                "epoch",
                format!("epoch {epoch} done, wake={}", self.fuse.wake_buffer_len()),
                Some(epoch),
            );
            self.metrics.train_epochs_done = self.metrics.train_epochs_done.wrapping_add(1);
        }

        self.metrics.train_correct = self.metrics.train_correct.wrapping_add(correct);
        self.metrics.train_total = self.metrics.train_total.wrapping_add(total);

        let sleep = self.sleep_now();
        self.push_train(
            "sleep",
            format!(
                "consolidó {} → engramas {}",
                sleep.episodes_consolidated, sleep.engrams_after
            ),
            None,
        );

        self.training = false;
        self.metrics.sync_fuse_counters(
            self.fuse.rqm_infer_calls,
            self.fuse.rqm_train_calls,
            self.fuse.engram_count(),
            self.fuse.wake_buffer_len(),
        );

        TrainStartResponse {
            ok: true,
            epochs,
            events: self.train_log.iter().rev().take(40).cloned().collect(),
            engrams: self.fuse.engram_count(),
            accuracy: self.metrics.train_accuracy(),
            sleep: Some(SleepReportDto::from(&sleep)),
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
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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
        // Forzar léxico para determinismo sin GGUF.
        s.probe = PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(
            99,
        ));
        s.decoder = ConceptDecoder::new(LlmMode::Lexicon);
        let r = s.handle_chat("hola campo líquido");
        assert!(r.concept_in < NUM_CONCEPTS);
        assert!(r.concept_out < NUM_CONCEPTS);
        assert!(!r.reply.is_empty());
        assert!(!r.decoded.is_empty());
        assert!(matches!(
            r.route.as_str(),
            "Liquid" | "RqmFallback"
        ));
    }

    #[test]
    fn train_increases_engrams_after_sleep() {
        let mut s = AppState::new();
        s.probe = PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(
            1,
        ));
        let before = s.fuse.engram_count();
        let r = s.train_start(Some(2), Some(vec![0, 1, 2, 3]));
        assert!(r.ok);
        assert!(r.engrams > before, "engrams {} <= {}", r.engrams, before);
        assert!(r.sleep.is_some());
        assert!(!r.events.is_empty());
    }

    #[test]
    fn intent_sleep_and_status() {
        let mut s = AppState::new();
        s.probe = PeripheralProbe::Lexicon(crate::field_linguistic_layer::GemmaShapedLexicon::new(
            2,
        ));
        let _ = s.handle_chat("hola");
        let sleep = s.handle_chat("sueño por favor");
        assert_eq!(sleep.route, "sleep");
        let st = s.handle_chat("estado");
        assert_eq!(st.route, "status");
    }
}
