//! Estado compartido de la app agentica (Arc<Mutex<AppState>>).
//!
//! El único motor de campo es v4 ([`crate::field_v4`]): Dφ (campo residual
//! local-lineal con compuerta) + CDT de vigilia + teacher consolidado que se
//! borra tras aprender. El cómputo pesado (lotes, sueño, pruebas) corre **fuera
//! del lock**: se toma un plan, se calcula y se aplica el resultado.

use crate::field_gemma_probe::{RawGemmaHandle, RawGemmaReply};
use crate::field_v4::{
    self, ChatField, SleepReport, SmokeReport, TrainBatchReport, APP_SLEEP_STEPS, APP_TRAIN_STEPS,
};
use crate::v4_controls::Dphi;
use crate::v4_dataset::Example;
use crate::web::llm_periphery::{open_best_probe, LlmMode, PeripheralProbe};
use crate::web::llm_provider::{ProviderConfig, ProviderStore, LOCAL_ID};
use crate::web::model_fetch::{ModelConfig, ModelStatus};
use crate::web::process_job::{
    ProcessesSnapshot, SleepJob, SleepStartResponse, TestsJob, TestsStartResponse,
};
use crate::web::telemetry::TelemetrySnapshot;
use crate::web::train_job::{
    read_latest_dphi, write_dphi_checkpoint, CheckpointMeta, DphiCheckpointFile,
    LiveTrainStartResponse, TrainJob, TrainLiveEvent,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatTurn {
    /// `field_decoder` | `gemma_raw` (badge UI). `None` en logs antiguos.
    #[serde(default)]
    pub mode: Option<String>,
    pub role: String,
    pub text: String,
    pub route: Option<String>,
    /// LLM que generó el texto (Gemma local / API externa). `None` = campo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatResponse {
    /// Modo que respondió: `field_decoder` (campo v4 + LLM decoder) | `gemma_raw`.
    pub mode: String,
    pub reply: String,
    /// `field_v4` | `train` | `sleep` | `status` | `gemma_raw` | `gemma_raw_error`.
    pub route: String,
    /// Estado del campo v4 para este mensaje (solo modo ON).
    pub field: Option<ChatField>,
    /// Descripción corta de quién verbalizó / métricas del decoder.
    pub decoded: String,
    /// LLM que produjo el texto: `Gemma local`, `API · nombre (modelo)` o
    /// `campo (sin LLM)`. Vacío en intents sin LLM (train/sleep…).
    pub llm: String,
    /// Id del proveedor que respondió (`gemma_local` o id de API).
    pub llm_id: String,
    /// True si la API externa falló y respondió el respaldo local.
    pub fallback: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    /// `"field_decoder"` (def) | `"gemma_raw"`.
    #[serde(default)]
    pub mode: Option<String>,
    /// Alternativa booleana a `mode`: true = decoder del campo, false = LLM crudo.
    #[serde(default)]
    pub field_decoder: Option<bool>,
}

impl ChatRequest {
    pub fn chat_mode(&self) -> Result<crate::web::llm_periphery::ChatMode, String> {
        crate::web::llm_periphery::ChatMode::resolve(self.mode.as_deref(), self.field_decoder)
    }
}

const FIELD_MODE: &str = "field_decoder";
const RAW_MODE: &str = "gemma_raw";
pub const FIELD_ROUTE: &str = "field_v4";
/// Pares (user, model) de chat crudo que se reenvían como contexto.
const RAW_HISTORY_PAIRS: usize = 4;

/// Motor de campo v4 de la app.
pub struct FieldEngine {
    /// Dφ actual (§16: cabeza local-lineal con compuerta, H = 136).
    pub dphi: Dphi,
    /// True si Dφ vio al menos un lote (o se cargó de checkpoint).
    pub trained: bool,
    /// CDT de vigilia: experiencias TRAIN pendientes de consolidar en sueño.
    pub wake: Vec<Example>,
    /// Origen del Dφ actual (`init` | `checkpoint:<ruta>` | `train` | `sleep`).
    pub origin: String,
    pub batches_total: usize,
    pub sleeps_total: usize,
    pub last_train: Option<TrainBatchReport>,
    pub last_sleep: Option<SleepReport>,
    pub last_smoke: Option<SmokeReport>,
}

impl FieldEngine {
    pub fn new(load_checkpoint: bool) -> Self {
        let mut e = Self {
            dphi: field_v4::new_dphi(0x0A11_D0F1),
            trained: false,
            wake: Vec::new(),
            origin: "init".into(),
            batches_total: 0,
            sleeps_total: 0,
            last_train: None,
            last_sleep: None,
            last_smoke: None,
        };
        if load_checkpoint {
            if let Some((path, f)) = read_latest_dphi() {
                if let Some(d) = field_v4::from_checkpoint(&f.dphi) {
                    tracing::info!(path = %path.display(), "Dφ v4 restaurado de checkpoint");
                    e.dphi = d;
                    e.trained = true;
                    e.origin = format!("checkpoint:{}", path.display());
                    e.last_train = f.report;
                }
            }
        }
        e
    }

    pub fn status_json(&self) -> serde_json::Value {
        json!({
            "architecture": "v4 · Dφ residual local-lineal con compuerta (§16) + CDT/teacher borrados antes de TEST",
            "trained": self.trained,
            "origin": self.origin,
            "params": crate::v4_controls::Predictor::params(&self.dphi),
            "weights_hash": self.dphi.weights_hash(),
            "wake_experiences": self.wake.len(),
            "batches_total": self.batches_total,
            "sleeps_total": self.sleeps_total,
            "last_train": self.last_train,
            "last_sleep": self.last_sleep,
        })
    }
}

/// Plan de un lote (tomado bajo el lock, ejecutado fuera).
pub struct TrainPlan {
    pub job_id: String,
    pub batch: usize,
    pub seed: u64,
    pub steps: usize,
    pub dphi: Dphi,
    pub wake: Vec<Example>,
}

pub struct TrainDone {
    pub plan: TrainPlan,
    pub report: TrainBatchReport,
}

pub struct SleepPlan {
    pub cycle: usize,
    pub seed: u64,
    pub steps: usize,
    pub dphi: Dphi,
    pub wake: Vec<Example>,
}

pub struct SleepDone {
    pub plan: SleepPlan,
    pub report: Option<SleepReport>,
}

pub struct TestsPlan {
    pub cycle: usize,
    pub model: Option<Dphi>,
}

pub struct AppState {
    pub field: FieldEngine,
    pub probe: PeripheralProbe,
    pub chat_log: Vec<ChatTurn>,
    pub metrics: TelemetrySnapshot,
    pub started_ms: u64,
    pub train_job: TrainJob,
    pub sleep_job: SleepJob,
    pub tests_job: TestsJob,
    /// Escribir checkpoints a disco (false en tests).
    pub persist: bool,
    /// Llamadas al chat crudo (OFF).
    pub raw_chat_calls: u64,
    /// Config del GGUF (ruta/URL/sha) y estado de descarga/carga.
    pub model_cfg: ModelConfig,
    pub model_status: ModelStatus,
    /// Proveedores LLM (Gemma local + APIs externas) y el activo.
    pub llm: ProviderStore,
    /// Prompt Spider: corrida actual, eventos y cola de aprobación.
    pub spider: crate::web::spider_job::SpiderJobs,
    /// Decoder del campo del Prompt Spider (router + entrenamiento).
    pub spider_field: crate::web::spider_field_job::SpiderFieldState,
}

/// Backend del decoder del campo (ON).
pub enum DecodeBackend {
    /// Gemma 2 local (GGUF).
    Gemma(RawGemmaHandle),
    /// API externa OpenAI-compatible; `fallback` = Gemma local si está cargado.
    External {
        cfg: ProviderConfig,
        fallback: Option<RawGemmaHandle>,
    },
}

/// Trabajo de decodificación preparado bajo el lock y ejecutado fuera.
pub struct DecodeJob {
    pub backend: DecodeBackend,
    /// Prompt con plantilla Gemma (backend local / respaldo).
    pub gemma_prompt: String,
    pub user_msg: String,
    /// Estado del campo ya decidido (texto) para el decoder externo.
    pub field_state: String,
}

/// Backend del chat crudo (OFF).
pub enum RawBackend {
    Gemma(RawGemmaHandle),
    /// API externa; `fallback` = Gemma local si está cargado (respaldo
    /// etiquetado si la API no está disponible).
    External {
        cfg: ProviderConfig,
        fallback: Option<RawGemmaHandle>,
    },
}

/// Etiqueta del LLM local.
pub const LOCAL_LABEL: &str = "Gemma local";
/// Etiqueta cuando ningún LLM verbalizó (se muestra el estado del campo).
pub const FIELD_ONLY_LABEL: &str = "campo (sin LLM)";

impl AppState {
    pub fn new() -> Self {
        let model_cfg = ModelConfig::from_env();
        let probe = open_best_probe(0x0A6E_471C);
        let mode = probe.mode();
        let persist = !cfg!(test);
        Self {
            field: FieldEngine::new(persist),
            probe,
            chat_log: Vec::new(),
            metrics: TelemetrySnapshot::default(),
            started_ms: now_ms(),
            train_job: TrainJob::default(),
            sleep_job: SleepJob::default(),
            tests_job: TestsJob::default(),
            persist,
            raw_chat_calls: 0,
            model_status: if matches!(mode, LlmMode::GemmaGguf) {
                ModelStatus::new("ready", &model_cfg, "GGUF cargado")
            } else if model_cfg.auto_download {
                ModelStatus::new("missing", &model_cfg, "GGUF ausente; se descargará")
            } else {
                ModelStatus::new(
                    "disabled",
                    &model_cfg,
                    "GGUF ausente y GEMMA2_AUTO_DOWNLOAD=0",
                )
            },
            model_cfg,
            llm: if cfg!(test) {
                ProviderStore::in_memory()
            } else {
                ProviderStore::load_default()
            },
            spider: crate::web::spider_job::SpiderJobs::new(if cfg!(test) {
                None
            } else {
                Some(crate::web::spider_job::default_runs_dir())
            }),
            spider_field: if cfg!(test) {
                crate::web::spider_field_job::SpiderFieldState::new(None, None)
            } else {
                crate::web::spider_field_job::SpiderFieldState::new(
                    Some(crate::web::spider_field_job::default_field_dir()),
                    Some(crate::web::spider_field_job::default_ckpt_dir()),
                )
            },
        }
    }

    /// Proveedor externo activo (None = Gemma local).
    pub fn active_external(&self) -> Option<ProviderConfig> {
        self.llm.active_external()
    }

    /// (id, etiqueta) del LLM activo para UI/respuestas.
    pub fn active_llm_label(&self) -> (String, String) {
        match self.active_external() {
            Some(p) => (p.id.clone(), p.label()),
            None => (LOCAL_ID.into(), LOCAL_LABEL.into()),
        }
    }

    /// «Nuevo chat»: borra el historial del servidor. No toca el campo.
    pub fn reset_chat(&mut self) -> usize {
        let n = self.chat_log.len();
        self.chat_log.clear();
        n
    }

    /// Hot-swap: instala Gemma recién descargado/cargado.
    pub fn install_probe(&mut self, probe: PeripheralProbe) {
        self.probe = probe;
        self.model_status = ModelStatus::new("ready", &self.model_cfg, "GGUF cargado");
    }

    /// Texto explicativo cuando el LLM aún no está disponible.
    pub fn model_unavailable_reason(&self) -> String {
        let st = &self.model_status;
        match st.state.as_str() {
            "downloading" => {
                let done = std::fs::metadata(self.model_cfg.part_path())
                    .map(|m| m.len())
                    .unwrap_or(st.downloaded_bytes);
                let mb = done as f64 / 1e6;
                match st.total_bytes {
                    Some(t) if t > 0 => format!(
                        "el modelo se está descargando ({mb:.0}/{:.0} MB, {:.0}%); reintenta en un momento",
                        t as f64 / 1e6,
                        100.0 * done as f64 / t as f64
                    ),
                    _ => format!("el modelo se está descargando ({mb:.0} MB); reintenta en un momento"),
                }
            }
            "loading" => "el modelo se está cargando en memoria; reintenta en unos segundos".into(),
            "error" => format!("no se pudo preparar el modelo: {}", st.detail),
            "disabled" => format!(
                "no hay GGUF en {} y la descarga automática está desactivada (GEMMA2_AUTO_DOWNLOAD=0)",
                st.path
            ),
            _ => format!("no se encontró el GGUF ({})", st.path),
        }
    }

    /// ON: si la respuesta viene del campo y hay LLM disponible (API externa
    /// activa o Gemma cargado), prepara el trabajo de decodificación.
    pub fn field_decode_job(&self, message: &str, resp: &ChatResponse) -> Option<DecodeJob> {
        if resp.route != FIELD_ROUTE {
            return None;
        }
        let local = self.probe.raw_handle();
        let backend = match self.active_external() {
            Some(cfg) => DecodeBackend::External {
                cfg,
                fallback: local,
            },
            None => DecodeBackend::Gemma(local?),
        };
        let state = resp.field.as_ref()?.summary();
        Some(DecodeJob {
            backend,
            gemma_prompt: crate::field_gemma_probe::render_field_decoder_prompt(
                message.trim(),
                &state,
            ),
            user_msg: message.trim().to_string(),
            field_state: state,
        })
    }

    /// Aplica el texto del LLM decoder a la respuesta de campo ya registrada.
    /// Si el LLM falló, conserva el estado del campo y anota el motivo: ON
    /// **siempre** responde.
    pub fn apply_field_decoded(
        &mut self,
        resp: ChatResponse,
        result: Result<RawGemmaReply, String>,
    ) -> ChatResponse {
        self.apply_field_decoded_by(resp, result, "gemma2 decoder", LOCAL_ID, LOCAL_LABEL, None)
    }

    pub fn apply_field_decoded_by(
        &mut self,
        mut resp: ChatResponse,
        result: Result<RawGemmaReply, String>,
        who: &str,
        llm_id: &str,
        llm: &str,
        note: Option<String>,
    ) -> ChatResponse {
        let ok_text = matches!(&result, Ok(r) if !r.text.trim().is_empty());
        let state = resp.field.as_ref().map(|f| f.summary()).unwrap_or_default();
        match result {
            Ok(r) if !r.text.trim().is_empty() => {
                resp.reply = format!("{}\n\n[campo v4] {state}", r.text.trim());
                resp.decoded = format!("{who} · {} tok · {:.1}s", r.generated_tokens, r.seconds);
            }
            Ok(_) => resp.reply.push_str(&format!(
                "\n\n(decoder {llm} devolvió texto vacío; se muestra el estado del campo)"
            )),
            Err(e) => resp.reply.push_str(&format!(
                "\n\n(decoder {llm} no respondió: {e}; se muestra el estado del campo)"
            )),
        }
        if let Some(n) = note {
            resp.reply.push_str(&format!("\n\n({n})"));
        }
        if ok_text {
            resp.llm = llm.to_string();
            resp.llm_id = llm_id.to_string();
        } else {
            resp.llm = FIELD_ONLY_LABEL.into();
            resp.llm_id = "field".into();
        }
        let turn_llm = resp.llm.clone();
        if let Some(last) = self
            .chat_log
            .iter_mut()
            .rev()
            .find(|t| t.role == "agent" && t.mode.as_deref() == Some(FIELD_MODE))
        {
            last.text = resp.reply.clone();
            last.llm = Some(turn_llm);
        }
        resp
    }

    /// Prepara chat crudo (OFF): handle al Gemma original + historial crudo.
    /// `Err` si no hay GGUF. No toca fuse/campo/telemetría líquida.
    pub fn raw_chat_prepare(&self) -> Result<(RawGemmaHandle, Vec<(String, String)>), String> {
        let handle = self.probe.raw_handle().ok_or_else(|| {
            format!(
                "Gemma 2 original no disponible: {}. \
                 Mientras tanto puedes usar «Decoder del campo».",
                self.model_unavailable_reason()
            )
        })?;
        Ok((handle, self.raw_history()))
    }

    /// Backend del chat crudo según el proveedor activo (API externa o Gemma
    /// local) + historial crudo. `Err` si el activo es Gemma y no hay GGUF.
    pub fn raw_chat_backend(&self) -> Result<(RawBackend, Vec<(String, String)>), String> {
        if let Some(cfg) = self.active_external() {
            let fallback = self.probe.raw_handle();
            return Ok((RawBackend::External { cfg, fallback }, self.raw_history()));
        }
        let (h, hist) = self.raw_chat_prepare()?;
        Ok((RawBackend::Gemma(h), hist))
    }

    /// Últimos pares (user, model) del chat crudo exitoso.
    pub fn raw_history(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        let turns = &self.chat_log;
        let mut i = 0;
        while i + 1 < turns.len() {
            let (u, a) = (&turns[i], &turns[i + 1]);
            if u.role == "user"
                && a.role == "agent"
                && u.mode.as_deref() == Some(RAW_MODE)
                && a.mode.as_deref() == Some(RAW_MODE)
                && a.route.as_deref() == Some(RAW_MODE)
            {
                pairs.push((u.text.clone(), a.text.clone()));
                i += 2;
            } else {
                i += 1;
            }
        }
        let n = pairs.len();
        pairs.split_off(n.saturating_sub(RAW_HISTORY_PAIRS))
    }

    /// Registra el turno crudo (éxito o error) y construye la respuesta.
    /// No toca el campo; sí `raw_chat_calls`.
    pub fn record_raw_chat(
        &mut self,
        message: &str,
        result: &Result<RawGemmaReply, String>,
    ) -> ChatResponse {
        self.record_raw_chat_by(message, result, None)
    }

    /// Registra un turno crudo indicando el proveedor externo (None = Gemma local).
    pub fn record_raw_chat_by(
        &mut self,
        message: &str,
        result: &Result<RawGemmaReply, String>,
        external: Option<&ProviderConfig>,
    ) -> ChatResponse {
        self.raw_chat_calls = self.raw_chat_calls.wrapping_add(1);
        let (llm_id, llm) = match external {
            Some(p) => (p.id.clone(), p.label()),
            None => (LOCAL_ID.to_string(), LOCAL_LABEL.to_string()),
        };
        let (reply, route) = match (result, external) {
            (Ok(r), _) => (r.text.clone(), RAW_MODE),
            (Err(e), None) => (format!("Error Gemma 2 original: {e}"), "gemma_raw_error"),
            (Err(e), Some(_)) => (format!("Error {llm}: {e}"), "gemma_raw_error"),
        };
        for (role, text, route) in [
            ("user", message.trim().to_string(), None),
            ("agent", reply.clone(), Some(route.to_string())),
        ] {
            self.chat_log.push(ChatTurn {
                mode: Some(RAW_MODE.into()),
                role: role.into(),
                text,
                route,
                llm: (role == "agent").then(|| llm.clone()),
            });
        }
        if self.chat_log.len() > 200 {
            let drain = self.chat_log.len() - 200;
            self.chat_log.drain(0..drain);
        }
        let who = match external {
            Some(p) => format!(
                "api·{}",
                if p.model.is_empty() {
                    &p.name
                } else {
                    &p.model
                }
            ),
            None => "gemma2-original".into(),
        };
        let decoded = match result {
            Ok(r) => format!(
                "{who} · {} tok prompt · {} tok gen · {:.1}s",
                r.prompt_tokens, r.generated_tokens, r.seconds
            ),
            Err(_) => format!("{who} · error"),
        };
        ChatResponse {
            llm,
            llm_id,
            fallback: false,
            mode: RAW_MODE.into(),
            reply,
            route: route.into(),
            field: None,
            decoded,
        }
    }

    /// Registra un turno crudo respondido por Gemma local como **respaldo**
    /// tras fallar la API externa `failed` con `error`. El historial guarda el
    /// texto limpio; la respuesta lleva la nota «X no disponible → Gemma local».
    pub fn record_raw_chat_fallback(
        &mut self,
        message: &str,
        result: &Result<RawGemmaReply, String>,
        failed: &ProviderConfig,
        error: &str,
    ) -> ChatResponse {
        let mut resp = self.record_raw_chat_by(message, result, None);
        let label = format!("{LOCAL_LABEL} (respaldo)");
        let note = format!(
            "{} no disponible → {LOCAL_LABEL} ({})",
            failed.name,
            error.chars().take(200).collect::<String>()
        );
        if let Some(last) = self
            .chat_log
            .iter_mut()
            .rev()
            .find(|t| t.role == "agent" && t.mode.as_deref() == Some(RAW_MODE))
        {
            last.llm = Some(label.clone());
        }
        resp.reply.push_str(&format!("\n\n({note})"));
        resp.decoded = format!("{} · respaldo: {note}", resp.decoded);
        resp.llm = label;
        resp.fallback = true;
        resp
    }

    /// Evidencia de entrenamiento previo para el gate de sueño: hay
    /// experiencias en la CDT de vigilia.
    pub fn has_training_evidence(&self) -> bool {
        !self.field.wake.is_empty()
    }

    pub fn llm_mode(&self) -> LlmMode {
        self.probe.mode()
    }

    fn push_chat_pair(&mut self, msg: &str, reply: &str, route: &str) {
        for (role, text, route) in [
            ("user", msg.to_string(), None),
            ("agent", reply.to_string(), Some(route.to_string())),
        ] {
            self.chat_log.push(ChatTurn {
                mode: Some(FIELD_MODE.into()),
                role: role.into(),
                text,
                route,
                llm: None,
            });
        }
        if self.chat_log.len() > 200 {
            let drain = self.chat_log.len() - 200;
            self.chat_log.drain(0..drain);
        }
    }

    fn intent_response(&mut self, msg: &str, reply: String, route: &str) -> ChatResponse {
        self.push_chat_pair(msg, &reply, route);
        ChatResponse {
            mode: FIELD_MODE.into(),
            reply,
            route: route.into(),
            field: None,
            decoded: String::new(),
            llm: String::new(),
            llm_id: String::new(),
            fallback: false,
        }
    }

    /// Chat ON: intents (entrenar / dormir / estado) o paso del campo v4.
    pub fn handle_chat(&mut self, message: &str) -> ChatResponse {
        let msg = message.trim();
        let lower = msg.to_lowercase();

        if looks_like_train(&lower) {
            let started = self.begin_live_train(None, None);
            let reply = if started.ok {
                format!(
                    "Entrenamiento v4 iniciado (job {}). Modo infinito: dataset sellado → CDT → teacher → Dφ (§16). Mira la pestaña Entrenamiento; Detener para parar.",
                    started.job_id
                )
            } else {
                format!("No se pudo iniciar: {}", started.message)
            };
            return self.intent_response(msg, reply, "train");
        }

        if looks_like_sleep(&lower) {
            let started = self.begin_sleep_job(None, false, Some(1));
            let reply = if started.ok {
                format!(
                    "Sueño iniciado (job {}): consolidación de la CDT de vigilia ({} experiencias) en el teacher → destilación en Dφ → borrado de la CDT. Mira la pestaña Sueño.",
                    started.job_id,
                    self.field.wake.len()
                )
            } else {
                started.message
            };
            return self.intent_response(msg, reply, "sleep");
        }

        if looks_like_status(&lower) {
            let f = &self.field;
            let reply = format!(
                "Estado v4: Dφ {} ({} params, hash {}), origen {}, CDT de vigilia {} experiencias, lotes {}, sueños {}, último acc TEST {}, LLM {}, train={}, sueño={}, pruebas={}",
                if f.trained { "entrenado" } else { "sin entrenar" },
                crate::v4_controls::Predictor::params(&f.dphi),
                &f.dphi.weights_hash()[..12],
                f.origin,
                f.wake.len(),
                f.batches_total,
                f.sleeps_total,
                f.last_train
                    .as_ref()
                    .map(|r| format!("{:.3}", r.metrics.acc))
                    .unwrap_or_else(|| "—".into()),
                self.llm_mode().as_str(),
                self.train_job.running,
                self.sleep_job.running,
                self.tests_job.running,
            );
            return self.intent_response(msg, reply, "status");
        }

        let t0 = Instant::now();
        let field = field_v4::chat_field(&self.field.dphi, self.field.trained, msg);
        let latency_us = t0.elapsed().as_secs_f64() * 1e6;
        self.metrics
            .record_field(field.rel_err, field.correct, latency_us);
        let reply = format!("[campo v4] {}", field.summary());
        self.push_chat_pair(msg, &reply, FIELD_ROUTE);
        ChatResponse {
            mode: FIELD_MODE.into(),
            reply,
            route: FIELD_ROUTE.into(),
            decoded: format!("Dφ · {:.0} µs", latency_us),
            field: Some(field),
            llm: String::new(),
            llm_id: String::new(),
            fallback: false,
        }
    }

    // ───────────────────────── entrenamiento ─────────────────────────

    /// Prepara el job async. `batches = None` (o 0) → infinito hasta Detener.
    pub fn begin_live_train(
        &mut self,
        batches: Option<usize>,
        steps: Option<usize>,
    ) -> LiveTrainStartResponse {
        if self.train_job.running {
            return LiveTrainStartResponse {
                ok: false,
                job_id: self.train_job.job_id.clone(),
                message: "ya hay un entrenamiento en curso".into(),
                infinite: None,
            };
        }
        if self.sleep_job.running {
            return LiveTrainStartResponse {
                ok: false,
                job_id: String::new(),
                message: "hay un sueño en curso; espera a que termine".into(),
                infinite: None,
            };
        }
        let infinite = batches.is_none() || batches == Some(0);
        let total_batches = (!infinite).then(|| batches.unwrap_or(1).clamp(1, 256));
        let steps = steps.unwrap_or(APP_TRAIN_STEPS).clamp(10, 64_000);
        let job_id = Uuid::new_v4().to_string();
        self.train_job = TrainJob {
            job_id: job_id.clone(),
            running: true,
            cancelled: false,
            current_batch: 0,
            total_batches,
            infinite,
            steps,
            events: Default::default(),
            last_checkpoint: self.train_job.last_checkpoint.clone(),
            last_report: self.train_job.last_report.clone(),
            event_seq: 0,
            checkpoints_saved: 0,
        };
        self.train_job.push_event(
            "start",
            format!(
                "job {job_id}: {} · {steps} pasos Dφ/lote · receta §16 (gated H136 + replay de confianza τ=1%)",
                if infinite {
                    "∞ infinito".to_string()
                } else {
                    format!("{} lote(s)", total_batches.unwrap_or(1))
                }
            ),
            None,
            json!({ "batches": total_batches, "infinite": infinite, "steps": steps }),
        );
        LiveTrainStartResponse {
            ok: true,
            job_id,
            message: if infinite {
                "entrenamiento infinito iniciado".into()
            } else {
                "entrenamiento iniciado".into()
            },
            infinite: Some(infinite),
        }
    }

    pub fn request_train_stop(&mut self) -> bool {
        if !self.train_job.running {
            return false;
        }
        self.train_job.cancelled = true;
        self.train_job.push_event(
            "cancel",
            "cancelación solicitada (termina el lote actual)",
            Some(self.train_job.current_batch),
            json!({ "cancelled": true }),
        );
        true
    }

    /// Siguiente lote a calcular fuera del lock (`None` = el job terminó).
    pub fn next_train_plan(&mut self) -> Option<TrainPlan> {
        if !self.train_job.running {
            return None;
        }
        if self.train_job.cancelled {
            self.finish_live_train(true);
            return None;
        }
        let batch = self.train_job.current_batch;
        if let Some(total) = self.train_job.total_batches {
            if batch >= total {
                self.finish_live_train(false);
                return None;
            }
        }
        let seed = now_ms()
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(batch as u64);
        self.train_job.push_event(
            "batch_start",
            format!(
                "lote {batch}{}: dataset v4 sellado seed=0x{seed:X}",
                if self.train_job.infinite {
                    " (∞)".to_string()
                } else {
                    format!("/{}", self.train_job.total_batches.unwrap_or(0))
                }
            ),
            Some(batch),
            json!({ "seed": format!("0x{seed:X}") }),
        );
        Some(TrainPlan {
            job_id: self.train_job.job_id.clone(),
            batch,
            seed,
            steps: self.train_job.steps,
            dphi: self.field.dphi.clone(),
            wake: self.field.wake.clone(),
        })
    }

    /// Ejecuta un plan (sin lock).
    pub fn compute_train(mut plan: TrainPlan) -> TrainDone {
        let report = field_v4::train_batch(&mut plan.dphi, &mut plan.wake, plan.seed, plan.steps);
        TrainDone { plan, report }
    }

    /// Aplica el resultado de un lote. Devuelve `true` si el loop sigue.
    pub fn apply_train(&mut self, done: TrainDone) -> bool {
        let TrainDone { plan, report } = done;
        if plan.job_id != self.train_job.job_id {
            return false;
        }
        let batch = plan.batch;
        let r = &report;
        self.train_job.push_event(
            "dataset",
            format!(
                "dataset {} ({}): {} ejemplos TRAIN → CDT de vigilia = {} experiencias",
                &r.dataset_hash[..12],
                r.audit_status,
                r.n_train_batch,
                r.cdt_experiences
            ),
            Some(batch),
            json!({ "dataset_hash": r.dataset_hash, "audit": r.audit_status, "cdt": r.cdt_experiences }),
        );
        self.train_job.push_event(
            "teacher",
            format!(
                "teacher consolidado · familias de confianza (residuo <1%): {}",
                if r.trusted_families.is_empty() {
                    "ninguna".to_string()
                } else {
                    r.trusted_families.join(", ")
                }
            ),
            Some(batch),
            json!({ "fit_residual": r.fit_residual, "trusted": r.trusted_families }),
        );
        self.train_job.push_event(
            "dphi",
            format!(
                "Dφ {} pasos en {:.1}s · TEST sellado: acc={:.3} rel_err₁={:.4} rollout h{}={:.3} · queries={} leaked={}",
                r.steps,
                r.train_ms / 1e3,
                r.metrics.acc,
                r.metrics.rel_err_1step,
                r.metrics.rollout_h,
                r.metrics.rollout_acc,
                r.metrics.test_queries,
                r.metrics.leaked
            ),
            Some(batch),
            serde_json::to_value(&r.metrics).unwrap_or_default(),
        );
        self.field.dphi = plan.dphi;
        self.field.wake = plan.wake;
        self.field.trained = true;
        self.field.origin = "train".into();
        self.field.batches_total += 1;
        self.metrics.train_batches += 1;
        let ts = now_ms();
        if self.persist {
            let file = DphiCheckpointFile {
                architecture: "v4_dphi_gated_h136_trusted".into(),
                job_id: self.train_job.job_id.clone(),
                batch,
                ts_ms: ts,
                report: Some(report.clone()),
                dphi: self.field.dphi.to_ckpt(),
            };
            match write_dphi_checkpoint(&file) {
                Ok(p) => {
                    let path = p.display().to_string();
                    self.train_job.checkpoints_saved += 1;
                    self.train_job.last_checkpoint = Some(CheckpointMeta {
                        path: path.clone(),
                        batch,
                        ts_ms: ts,
                        accuracy: Some(report.metrics.acc),
                        weights_hash: report.weights_hash.clone(),
                    });
                    self.train_job.push_event(
                        "checkpoint",
                        format!("checkpoint Dφ {path}"),
                        Some(batch),
                        json!({ "path": path, "weights_hash": report.weights_hash }),
                    );
                }
                Err(e) => self.train_job.push_event(
                    "error",
                    format!("checkpoint falló: {e}"),
                    Some(batch),
                    json!({}),
                ),
            }
        }
        self.field.last_train = Some(report.clone());
        self.train_job.last_report = Some(report);
        self.train_job.current_batch = batch + 1;
        if self.train_job.cancelled {
            self.finish_live_train(true);
            return false;
        }
        if let Some(total) = self.train_job.total_batches {
            if self.train_job.current_batch >= total {
                self.finish_live_train(false);
                return false;
            }
        }
        true
    }

    /// Compat/tests: ejecuta lotes en el hilo actual hasta terminar.
    pub fn run_one_live_batch(&mut self) -> bool {
        match self.next_train_plan() {
            Some(p) => {
                let d = Self::compute_train(p);
                self.apply_train(d)
            }
            None => false,
        }
    }

    pub fn fail_train(&mut self, why: String) {
        self.train_job.push_event("error", why, None, json!({}));
        self.train_job.running = false;
    }

    fn finish_live_train(&mut self, cancelled: bool) {
        self.train_job.running = false;
        let (kind, msg) = if cancelled {
            ("done", "entrenamiento detenido")
        } else {
            ("done", "entrenamiento completado")
        };
        self.train_job.push_event(
            kind,
            msg,
            Some(self.train_job.current_batch),
            json!({ "cancelled": cancelled, "batches": self.train_job.current_batch }),
        );
    }

    // ───────────────────────── sueño ─────────────────────────

    pub fn begin_sleep_job(
        &mut self,
        steps: Option<usize>,
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
        if self.train_job.running {
            return SleepStartResponse {
                ok: false,
                job_id: String::new(),
                message: "hay un entrenamiento en curso; detenlo antes de dormir".into(),
                infinite: None,
            };
        }
        if !self.has_training_evidence() {
            return SleepStartResponse {
                ok: false,
                job_id: self.sleep_job.job_id.clone(),
                message: "necesitas entrenar antes de dormir: la CDT de vigilia está vacía (cada sueño la consolida y la borra)".into(),
                infinite: None,
            };
        }
        let infinite = infinite || total_cycles.is_none() || total_cycles == Some(0);
        let total_cycles = (!infinite).then(|| total_cycles.unwrap_or(1).max(1));
        let job_id = Uuid::new_v4().to_string();
        let keep = self.sleep_job.last_report.clone();
        self.sleep_job = SleepJob {
            job_id: job_id.clone(),
            running: true,
            cancelled: false,
            phase: "queued".into(),
            steps: steps.unwrap_or(APP_SLEEP_STEPS).clamp(10, 128_000),
            current_cycle: 0,
            total_cycles,
            infinite,
            events: Default::default(),
            event_seq: 0,
            started_ms: Some(now_ms()),
            finished_ms: None,
            last_report: keep,
        };
        let steps = self.sleep_job.steps;
        self.sleep_job.push_event(
            "start",
            format!(
                "sueño job {job_id}: CDT de vigilia {} experiencias · {steps} pasos de destilación",
                self.field.wake.len()
            ),
            json!({ "infinite": infinite, "total_cycles": total_cycles, "steps": steps }),
        );
        SleepStartResponse {
            ok: true,
            job_id,
            message: "sueño iniciado".into(),
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
            .push_event("cancel", "cancelación solicitada", json!({}));
        true
    }

    pub fn next_sleep_plan(&mut self) -> Option<SleepPlan> {
        if !self.sleep_job.running {
            return None;
        }
        if self.sleep_job.cancelled {
            self.finish_sleep_job(true);
            return None;
        }
        let cycle = self.sleep_job.current_cycle;
        if let Some(total) = self.sleep_job.total_cycles {
            if cycle >= total {
                self.finish_sleep_job(false);
                return None;
            }
        }
        if self.field.wake.is_empty() {
            self.sleep_job.push_event(
                "idle",
                "CDT de vigilia vacía: nada que consolidar (entrena más lotes)",
                json!({}),
            );
            self.finish_sleep_job(false);
            return None;
        }
        self.sleep_job.phase = "consolidate".into();
        self.sleep_job.push_event(
            "consolidate",
            format!(
                "ciclo {cycle}: consolidando {} experiencias en el teacher",
                self.field.wake.len()
            ),
            json!({ "cycle": cycle }),
        );
        Some(SleepPlan {
            cycle,
            seed: now_ms()
                .wrapping_mul(0x5851_F42D)
                .wrapping_add(cycle as u64),
            steps: self.sleep_job.steps,
            dphi: self.field.dphi.clone(),
            wake: std::mem::take(&mut self.field.wake),
        })
    }

    pub fn compute_sleep(mut plan: SleepPlan) -> SleepDone {
        let report =
            field_v4::sleep_consolidate(&mut plan.dphi, &mut plan.wake, plan.seed, plan.steps);
        SleepDone { plan, report }
    }

    pub fn apply_sleep(&mut self, done: SleepDone) -> bool {
        let SleepDone { plan, report } = done;
        let cycle = plan.cycle;
        if let Some(r) = &report {
            self.field.dphi = plan.dphi;
            self.field.trained = true;
            self.field.origin = "sleep".into();
            self.field.sleeps_total += 1;
            self.metrics.sleeps += 1;
            self.sleep_job.phase = "distill".into();
            self.sleep_job.push_event(
                "distill",
                format!(
                    "teacher {} números · confianza: {} · Dφ destilado {} pasos en {:.1}s",
                    r.teacher_numbers,
                    if r.trusted_families.is_empty() {
                        "ninguna".to_string()
                    } else {
                        r.trusted_families.join(", ")
                    },
                    r.steps,
                    r.sleep_ms / 1e3
                ),
                json!({ "fit_residual": r.fit_residual }),
            );
            self.sleep_job.push_event(
                "cycle_done",
                format!(
                    "ciclo {cycle}: CDT borrada · TEST nuevo acc {:.3}→{:.3} · rel_err₁ {:.4}→{:.4} · queries={} leaked={}",
                    r.before.acc,
                    r.after.acc,
                    r.before.rel_err_1step,
                    r.after.rel_err_1step,
                    r.after.test_queries,
                    r.after.leaked
                ),
                json!({ "before": r.before, "after": r.after, "cycle": cycle }),
            );
            if self.persist {
                let file = DphiCheckpointFile {
                    architecture: "v4_dphi_gated_h136_trusted".into(),
                    job_id: self.sleep_job.job_id.clone(),
                    batch: cycle,
                    ts_ms: now_ms(),
                    report: self.field.last_train.clone(),
                    dphi: self.field.dphi.to_ckpt(),
                };
                if let Err(e) = write_dphi_checkpoint(&file) {
                    self.sleep_job
                        .push_event("error", format!("checkpoint falló: {e}"), json!({}));
                }
            }
            self.field.last_sleep = Some(r.clone());
            self.sleep_job.last_report = Some(r.clone());
        } else {
            // Nada que consolidar: devolvemos la (vacía) vigilia.
            self.field.wake = plan.wake;
        }
        // Decoder del campo del Spider: el mismo sueño consolida su vigilia
        // (decisiones Aprobar / Rechazar) y reajusta sus cabezas.
        match self.spider_field.sleep_if_ready() {
            Some(Ok(rep)) => self.sleep_job.push_event(
                "spider_field",
                format!(
                    "decoder del campo (Spider): {} episodio(s) consolidados",
                    rep.episodes
                ),
                json!({ "episodes": rep.episodes, "ms": rep.ms }),
            ),
            Some(Err(why)) => self.sleep_job.push_event(
                "spider_field",
                format!("decoder del campo (Spider): {why}"),
                json!({}),
            ),
            None => {}
        }
        self.sleep_job.current_cycle = cycle + 1;
        if self.sleep_job.cancelled {
            self.finish_sleep_job(true);
            return false;
        }
        true
    }

    pub fn fail_sleep(&mut self, why: String) {
        self.sleep_job.push_event("error", why, json!({}));
        self.sleep_job.running = false;
        self.sleep_job.phase = "error".into();
        self.sleep_job.finished_ms = Some(now_ms());
    }

    fn finish_sleep_job(&mut self, cancelled: bool) {
        self.sleep_job.running = false;
        self.sleep_job.finished_ms = Some(now_ms());
        self.sleep_job.phase = if cancelled { "cancelled" } else { "done" }.into();
        self.sleep_job.push_event(
            "done",
            if cancelled {
                "sueño cancelado"
            } else {
                "sueño completado"
            },
            json!({}),
        );
    }

    /// Compat/tests: ciclos de sueño en el hilo actual.
    pub fn execute_sleep_job(&mut self) {
        while let Some(p) = self.next_sleep_plan() {
            let d = Self::compute_sleep(p);
            if !self.apply_sleep(d) {
                break;
            }
        }
    }

    // ───────────────────────── pruebas ─────────────────────────

    pub fn begin_tests_job(
        &mut self,
        infinite: bool,
        total_cycles: Option<usize>,
    ) -> TestsStartResponse {
        if self.tests_job.running {
            return TestsStartResponse {
                ok: false,
                job_id: self.tests_job.job_id.clone(),
                message: "ya hay pruebas en curso".into(),
                infinite: None,
            };
        }
        let infinite = infinite || total_cycles.is_none() || total_cycles == Some(0);
        let total_cycles = (!infinite).then(|| total_cycles.unwrap_or(1).max(1));
        let job_id = Uuid::new_v4().to_string();
        let keep = self.tests_job.last_report.clone();
        self.tests_job = TestsJob {
            job_id: job_id.clone(),
            running: true,
            cancelled: false,
            phase: "queued".into(),
            step: 0,
            total_steps: 1,
            current_cycle: 0,
            total_cycles,
            infinite,
            events: Default::default(),
            event_seq: 0,
            started_ms: Some(now_ms()),
            finished_ms: None,
            last_report: keep,
        };
        self.tests_job.push_event(
            "start",
            format!(
                "pruebas job {job_id}: smoke v4 (E35 A vs B ×{} seeds + auditoría memoria cero + error a un paso + rollout E33 h{})",
                field_v4::SMOKE_SEEDS.len(),
                field_v4::ROLLOUT_H
            ),
            json!({ "infinite": infinite, "total_cycles": total_cycles }),
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
        if !self.tests_job.running {
            return false;
        }
        self.tests_job.cancelled = true;
        self.tests_job.phase = "cancelling".into();
        self.tests_job
            .push_event("cancel", "cancelación solicitada", json!({}));
        true
    }

    pub fn next_tests_plan(&mut self) -> Option<TestsPlan> {
        if !self.tests_job.running {
            return None;
        }
        if self.tests_job.cancelled {
            self.finish_tests_job(true);
            return None;
        }
        let cycle = self.tests_job.current_cycle;
        if let Some(total) = self.tests_job.total_cycles {
            if cycle >= total {
                self.finish_tests_job(false);
                return None;
            }
        }
        self.tests_job.phase = "running".into();
        self.tests_job.push_event(
            "cycle",
            format!("batería {cycle}: smoke v4 en curso…"),
            json!({ "cycle": cycle }),
        );
        Some(TestsPlan {
            cycle,
            model: self.field.trained.then(|| self.field.dphi.clone()),
        })
    }

    pub fn compute_tests(plan: &TestsPlan) -> SmokeReport {
        field_v4::run_smoke(plan.model.as_ref(), 3000)
    }

    pub fn apply_tests(&mut self, plan: TestsPlan, rep: SmokeReport) -> bool {
        for r in &rep.rows {
            self.tests_job.push_event(
                "step",
                format!(
                    "{} [{}] {} · {} → {}",
                    r.test, r.seed, r.value, r.threshold, r.verdict
                ),
                json!({ "row": r }),
            );
        }
        self.tests_job.push_event(
            "cycle_done",
            format!(
                "batería {}: {} PASS / {} FAIL · E35 A={:.3} B={:.3} · {:.1}s",
                plan.cycle,
                rep.pass,
                rep.fail,
                rep.e35_mean_a,
                rep.e35_mean_b,
                rep.elapsed_ms / 1e3
            ),
            json!({ "pass": rep.pass, "fail": rep.fail }),
        );
        self.field.last_smoke = Some(rep.clone());
        self.tests_job.last_report = Some(rep);
        self.tests_job.step = 1;
        self.tests_job.current_cycle = plan.cycle + 1;
        if self.tests_job.cancelled {
            self.finish_tests_job(true);
            return false;
        }
        true
    }

    pub fn fail_tests(&mut self, why: String) {
        self.tests_job.push_event("error", why, json!({}));
        self.tests_job.running = false;
        self.tests_job.phase = "error".into();
        self.tests_job.finished_ms = Some(now_ms());
    }

    fn finish_tests_job(&mut self, cancelled: bool) {
        self.tests_job.running = false;
        self.tests_job.finished_ms = Some(now_ms());
        self.tests_job.phase = if cancelled { "cancelled" } else { "done" }.into();
        self.tests_job.push_event(
            "done",
            if cancelled {
                "pruebas canceladas"
            } else {
                "pruebas completadas"
            },
            json!({}),
        );
    }

    /// Compat/tests: baterías finitas en el hilo actual.
    pub fn execute_tests_job(&mut self) {
        while let Some(p) = self.next_tests_plan() {
            let r = Self::compute_tests(&p);
            if !self.apply_tests(p, r) {
                break;
            }
        }
    }

    pub fn processes_snapshot(&self) -> ProcessesSnapshot {
        let mut active = Vec::new();
        if self.train_job.running {
            active.push("train".into());
        }
        if self.sleep_job.running {
            active.push("sleep".into());
        }
        if self.tests_job.running {
            active.push("tests".into());
        }
        ProcessesSnapshot {
            train: self.train_job.snapshot_for_api(),
            sleep: self.sleep_job.snapshot_for_api(),
            tests: self.tests_job.snapshot_for_api(),
            active,
        }
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
        || lower.contains("start training")
        || lower.contains("train now")
        || lower == "train"
}

fn looks_like_sleep(lower: &str) -> bool {
    lower.contains("sueño")
        || lower.contains("sueno")
        || lower.contains("dormir")
        || lower.contains("consolida")
        || lower == "sleep"
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
    fn chat_on_runs_field_v4_without_llm() {
        let mut s = AppState::new();
        let r = s.handle_chat("rota este objeto 0.4");
        assert_eq!(r.route, FIELD_ROUTE);
        let f = r.field.as_ref().unwrap();
        assert_eq!(f.family, "rotate");
        assert!(r.reply.contains("Dφ v4"));
        assert!(s.field_decode_job("x", &r).is_none() || s.active_external().is_some());
        assert_eq!(s.metrics.field_queries, 1);
    }

    #[test]
    fn intents_status_and_sleep_gate() {
        let mut s = AppState::new();
        assert_eq!(s.handle_chat("estado").route, "status");
        let r = s.handle_chat("dormir");
        assert_eq!(r.route, "sleep");
        assert!(!s.sleep_job.running, "sin CDT de vigilia no hay sueño");
    }

    #[test]
    fn train_sleep_tests_pipeline_v4() {
        let mut s = AppState::new();
        let st = s.begin_live_train(Some(1), Some(60));
        assert!(st.ok);
        while s.run_one_live_batch() {}
        assert!(!s.train_job.running);
        let rep = s.train_job.last_report.clone().expect("informe de lote");
        assert_eq!(rep.metrics.test_queries, 0);
        assert_eq!(rep.metrics.leaked, 0);
        assert!(s.field.trained && !s.field.wake.is_empty());
        let sl = s.begin_sleep_job(Some(60), false, Some(1));
        assert!(sl.ok, "{}", sl.message);
        s.execute_sleep_job();
        let r = s.sleep_job.last_report.clone().expect("informe de sueño");
        assert!(r.cdt_dropped && s.field.wake.is_empty());
        assert_eq!(r.after.test_queries, 0);
        assert!(!s.begin_sleep_job(None, false, Some(1)).ok);
    }

    #[test]
    fn batches_zero_means_infinite_and_stop() {
        let mut s = AppState::new();
        let st = s.begin_live_train(Some(0), Some(20));
        assert_eq!(st.infinite, Some(true));
        assert!(s.run_one_live_batch());
        assert!(s.request_train_stop());
        assert!(!s.run_one_live_batch());
        assert!(!s.train_job.running);
    }
}
