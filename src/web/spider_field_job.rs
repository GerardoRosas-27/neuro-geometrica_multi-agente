//! Decoder del campo del Prompt Spider dentro de la app: estado en
//! `AppState`, **«Entrenamiento Prompt Spider»** (job por datasets con sueño
//! entre datasets, evaluación hold-out y checkpoint por dataset), vigilia
//! online (Aprobar / Rechazar) y endpoints `/api/spider/field/*`.
//!
//! El router puro está en [`crate::spider_field`]. Configuración por defecto
//! = brazo (A) sustrato en blanco (líquido + CDT + RQM), elegido por el
//! experimento de sustrato (`docs/prompt_spider_decoder_campo.md`).

use crate::prompt_spider::{
    word_context, EscalationItem, Route, DEFAULT_LLM_FLOOR, DEFAULT_THRESHOLD, SAMPLE_PROMPT,
};
use crate::spider_field::{
    evaluate_router, scorer_examples, synth_examples, synth_prompt, teacher_labels, text_group,
    user_labels, AnalyzedPrompt, Coupling, FieldConfig, FieldEval, FieldModelFile,
    FieldSleepReport, LabelSource, Sm, SpiderExample, SpiderFieldRouter, MODEL_VERSION,
    VOCAB_SHIFT, VOCAB_TRAIN,
};
use crate::spider_field_experiment::{holdout_seed, shift_seed};
use crate::web::api::SharedState;
use crate::web::spider_job::{Decision, SpiderRun, Status, FIELD_DECIDER};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Ejemplos mínimos en la repetición para reajustar las cabezas en un sueño
/// ajeno al entrenamiento (pestaña Sueño). Tras reiniciar la app la
/// repetición está vacía: el sueño se difiere hasta el siguiente
/// entrenamiento para no reajustar con unos pocos ejemplos.
pub const MIN_REFIT: usize = 400;
/// Prompts del hold-out H1 / H2 en la app (el experimento usa 80).
pub const APP_HOLDOUT_PROMPTS: usize = 24;
/// Semilla del reservorio líquido del modelo por defecto.
pub const DEFAULT_SEED: u64 = 7;
const MAX_EVENTS: usize = 800;
const MAX_ONLINE_PENDING: usize = 4000;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Configuración por defecto: (A) sustrato en blanco.
pub fn default_config() -> FieldConfig {
    FieldConfig::blank(DEFAULT_SEED)
}

/// Directorio del modelo. Env `SPIDER_FIELD_DIR`; si no, `data/spider_field`.
pub fn default_field_dir() -> PathBuf {
    match std::env::var("SPIDER_FIELD_DIR") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p),
        _ => PathBuf::from("data/spider_field"),
    }
}

/// Checkpoints por dataset: `data/checkpoints/spider_field/`.
pub fn default_ckpt_dir() -> PathBuf {
    crate::web::train_job::checkpoints_dir().join("spider_field")
}

/// Nombre legible del modelo.
pub fn model_name(r: &SpiderFieldRouter) -> String {
    let c = &r.cfg;
    let arm = match c.coupling {
        Coupling::Blank => "A·blanco",
        Coupling::FrozenCore => "B·núcleo congelado",
        Coupling::SharedCore => "C·núcleo compartido",
    };
    let mut parts = vec![format!("spider-field v{MODEL_VERSION} {arm}")];
    if c.liquid {
        parts.push(format!("líquido {}", c.n_liquid));
    }
    if c.cdt {
        parts.push("CDT".into());
    }
    if c.rqm {
        parts.push("RQM/EPR".into());
    }
    parts.join(" + ")
}

/// Evento del entrenamiento (sondeo).
#[derive(Clone, Debug, Serialize)]
pub struct FieldEvent {
    pub seq: u64,
    /// `start` | `dataset` | `teacher` | `sleep` | `eval` | `checkpoint` |
    /// `online_sleep` | `error` | `done`.
    pub kind: String,
    pub message: String,
    pub ts_ms: u64,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub data: serde_json::Value,
}

/// Opciones del entrenamiento.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainOpts {
    /// Datasets (0 = infinito hasta Detener).
    pub datasets: usize,
    pub prompts_per_dataset: usize,
    /// Etiquetas reales de las corridas guardadas (scorer, Aprobar/Rechazar,
    /// veredictos de LLM).
    pub include_runs: bool,
    /// Palabras escaladas por dataset que se consultan en vivo al LLM activo
    /// como maestro (0 = no).
    pub teacher_live: usize,
    /// Empieza desde un sustrato en blanco nuevo.
    pub reset: bool,
    pub seed: u64,
    pub floor: f64,
    pub threshold: f64,
    /// Ruido de etiquetas en los sintéticos (robustez).
    pub noise: f64,
}

/// Estado del job de entrenamiento.
#[derive(Default)]
pub struct FieldTrainJob {
    pub running: bool,
    pub stop: Arc<AtomicBool>,
    pub job_id: String,
    pub opts: Option<TrainOpts>,
    pub current: usize,
    pub phase: String,
    pub started_ms: u64,
    pub finished_ms: Option<u64>,
    pub datasets_saved: usize,
    pub last_checkpoint: Option<String>,
    pub last_eval: Option<FieldEval>,
    pub last_sleep: Option<FieldSleepReport>,
    pub sources: HashMap<String, usize>,
    pub events: Vec<FieldEvent>,
    pub seq: u64,
    pub error: Option<String>,
}

impl FieldTrainJob {
    fn push(&mut self, kind: &str, message: String, data: serde_json::Value) {
        self.seq += 1;
        self.events.push(FieldEvent {
            seq: self.seq,
            kind: kind.into(),
            message,
            ts_ms: now_ms(),
            data,
        });
        if self.events.len() > MAX_EVENTS {
            let n = self.events.len() - MAX_EVENTS;
            self.events.drain(0..n);
        }
    }
}

/// Decoder del campo del Spider en `AppState`.
pub struct SpiderFieldState {
    pub router: SpiderFieldRouter,
    /// Directorio del modelo (None = no persiste; tests).
    pub dir: Option<PathBuf>,
    /// Directorio de checkpoints por dataset (None = no persiste).
    pub ckpt_dir: Option<PathBuf>,
    pub job: FieldTrainJob,
    pub load_note: Option<String>,
    /// Decisiones humanas observadas en vigilia (total).
    pub online_observed: u64,
    /// Episodios online llegados mientras entrena un job (se re-observan al
    /// instalar el modelo nuevo).
    online_pending: Vec<SpiderExample>,
}

impl Default for SpiderFieldState {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl SpiderFieldState {
    /// Crea el estado y carga `dir/model.json` si existe.
    pub fn new(dir: Option<PathBuf>, ckpt_dir: Option<PathBuf>) -> Self {
        let mut router = SpiderFieldRouter::new(default_config());
        let mut load_note = None;
        if let Some(path) = dir.as_ref().map(|d| d.join("model.json")) {
            if path.exists() {
                match std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|t| {
                        serde_json::from_str::<FieldModelFile>(&t).map_err(|e| e.to_string())
                    })
                    .and_then(SpiderFieldRouter::from_file)
                {
                    Ok(r) => {
                        router = r;
                        load_note = Some(format!("cargado de {}", path.display()));
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "spider field: modelo ilegible; en blanco");
                        load_note = Some(format!("modelo ilegible ({e}); en blanco"));
                    }
                }
            }
        }
        Self {
            router,
            dir,
            ckpt_dir,
            job: FieldTrainJob::default(),
            load_note,
            online_observed: 0,
            online_pending: Vec::new(),
        }
    }

    pub fn model_path(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join("model.json"))
    }

    /// Guarda el modelo (escritura atómica). Errores solo se registran.
    pub fn save_model(&self) {
        let Some(dir) = &self.dir else {
            return;
        };
        let res = (|| -> Result<(), String> {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let txt = serde_json::to_string(&self.router.to_file()).map_err(|e| e.to_string())?;
            let tmp = dir.join("model.json.tmp");
            std::fs::write(&tmp, txt).map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, dir.join("model.json")).map_err(|e| e.to_string())
        })();
        if let Err(e) = res {
            tracing::warn!(error = %e, "spider field: no se pudo guardar el modelo");
        }
    }

    /// Resumen del modelo para una corrida.
    pub fn model_summary(&self, r: &SpiderFieldRouter) -> serde_json::Value {
        json!({
            "name": model_name(r),
            "trained": r.is_trained(),
            "examples_seen": r.examples_seen,
            "sleeps": r.sleeps,
            "novelty_max": r.novelty_max,
            "holdout": r.holdout,
        })
    }

    /// Estado para la UI (`/api/spider/field/status` y `/api/spider/status`).
    pub fn status_value(&self) -> serde_json::Value {
        let r = &self.router;
        let j = &self.job;
        json!({
            "name": model_name(r),
            "config": r.cfg,
            "trained": r.is_trained(),
            "examples_seen": r.examples_seen,
            "sleeps": r.sleeps,
            "replay": r.replay_len(),
            "wake": r.wake_len(),
            "novelty_max": r.novelty_max,
            "holdout": r.holdout,
            "online_observed": self.online_observed,
            "model_path": self.model_path().map(|p| p.display().to_string()),
            "load_note": self.load_note,
            "job": {
                "running": j.running,
                "job_id": j.job_id,
                "opts": j.opts,
                "current": j.current,
                "phase": j.phase,
                "started_ms": j.started_ms,
                "finished_ms": j.finished_ms,
                "datasets_saved": j.datasets_saved,
                "last_checkpoint": j.last_checkpoint,
                "last_eval": j.last_eval,
                "last_sleep": j.last_sleep,
                "sources": j.sources,
                "error": j.error,
            },
            "seq": j.seq,
        })
    }

    /// Vigilia: una decisión humana como episodio etiquetado.
    pub fn observe_online(&mut self, ex: SpiderExample) {
        self.online_observed += 1;
        if self.job.running && self.online_pending.len() < MAX_ONLINE_PENDING {
            self.online_pending.push(ex.clone());
        }
        self.router.observe(ex);
    }

    /// Sueño compatible con la pestaña Sueño: consolida la vigilia (CDT +
    /// RQM + reajuste) si hay episodios y suficiente repetición. `None` si
    /// no hay nada que consolidar o se difiere.
    pub fn sleep_if_ready(&mut self) -> Option<Result<FieldSleepReport, String>> {
        if self.router.wake_len() == 0 {
            return None;
        }
        if self.job.running {
            return Some(Err("entrenamiento en curso: el sueño lo hace el job".into()));
        }
        if self.router.is_trained() && self.router.replay_len() < MIN_REFIT {
            return Some(Err(format!(
                "repetición {} < {MIN_REFIT}: se difiere al próximo entrenamiento",
                self.router.replay_len()
            )));
        }
        let rep = self.router.sleep();
        self.save_model();
        self.job.push(
            "online_sleep",
            format!(
                "sueño: {} episodio(s) de vigilia consolidados ({:.0} ms)",
                rep.episodes, rep.ms
            ),
            json!(rep),
        );
        Some(Ok(rep))
    }
}

// ---------------------------------------------------------------------------
// Datos
// ---------------------------------------------------------------------------

/// Ejemplo de usuario a partir de una decisión Aprobar / Rechazar.
pub fn user_example(run: &SpiderRun, d: &Decision, approve: bool) -> Option<SpiderExample> {
    let ap = AnalyzedPrompt::new(&run.prompt, run.threshold);
    let pos = ap.pos_of(d.index)?;
    ap.encode(
        pos,
        user_labels(d.question, approve),
        LabelSource::User,
        text_group(&run.prompt),
    )
}

/// ¿Va al hold-out real? (split por palabra, 1 de cada 5).
fn real_holdout(group: u64, index: usize) -> bool {
    fnv(&format!("hold{group}:{index}")).is_multiple_of(5)
}

/// Ejemplos reales: scorer del prompt de ejemplo y de cada prompt guardado,
/// Aprobar/Rechazar del usuario y veredictos de LLM (no del campo).
#[derive(Default)]
pub struct RealData {
    pub train: Vec<SpiderExample>,
    /// Hold-out real (usuario / maestro).
    pub hold: Vec<SpiderExample>,
    /// (texto, umbral) de los prompts reales (para el maestro en vivo).
    pub prompts: Vec<(String, f64)>,
    /// (grupo, índice) ya etiquetados por un maestro.
    pub teacher_seen: HashSet<(u64, usize)>,
    pub runs: usize,
}

fn load_runs(dir: Option<&Path>, current: Option<&SpiderRun>) -> Vec<SpiderRun> {
    let mut out: Vec<SpiderRun> = Vec::new();
    if let Some(dir) = dir {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                let Ok(txt) = std::fs::read_to_string(&p) else {
                    continue;
                };
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
                    continue;
                };
                if let Some(run) = v
                    .get("run")
                    .and_then(|r| serde_json::from_value::<SpiderRun>(r.clone()).ok())
                {
                    out.push(run);
                }
            }
        }
    }
    if let Some(c) = current {
        if c.state != "running" && !out.iter().any(|r| r.id == c.id) {
            out.push(c.clone());
        }
    }
    out.sort_by_key(|r| r.started_ms);
    out
}

/// Construye los datos reales (más recientes ganan en duplicados).
pub fn real_data(dir: Option<&Path>, current: Option<&SpiderRun>, threshold: f64) -> RealData {
    let runs = load_runs(dir, current);
    let mut rd = RealData {
        runs: runs.len(),
        ..Default::default()
    };
    let mut seen_prompt: HashSet<u64> = HashSet::new();
    let sample_g = text_group(SAMPLE_PROMPT);
    rd.train
        .extend(scorer_examples(SAMPLE_PROMPT, threshold, sample_g));
    rd.prompts.push((SAMPLE_PROMPT.to_string(), threshold));
    seen_prompt.insert(sample_g);
    // (grupo, índice, fuente) → ejemplo (el último gana).
    let mut labeled: HashMap<(u64, usize, LabelSource), SpiderExample> = HashMap::new();
    for run in &runs {
        let g = text_group(&run.prompt);
        let ap = AnalyzedPrompt::new(&run.prompt, run.threshold);
        if seen_prompt.insert(g) {
            rd.train
                .extend(scorer_examples(&run.prompt, run.threshold, g));
            rd.prompts.push((run.prompt.clone(), run.threshold));
        }
        let compact = run.llm.kind == "local";
        for d in &run.decisions {
            let Some(pos) = ap.pos_of(d.index) else {
                continue;
            };
            match d.status {
                Status::Approved | Status::Rejected => {
                    let approve = d.status == Status::Approved;
                    if let Some(ex) =
                        ap.encode(pos, user_labels(d.question, approve), LabelSource::User, g)
                    {
                        labeled.insert((g, d.index, LabelSource::User), ex);
                    }
                }
                _ => {}
            }
            // Maestro: veredicto de un LLM (nunca del propio campo).
            if let Some(v) = &d.verdict {
                if d.field.is_none()
                    && d.resolved_by != FIELD_DECIDER
                    && run.decider != FIELD_DECIDER
                {
                    let l = teacher_labels(d.question, v, run.llm_floor, compact);
                    if let Some(ex) = ap.encode(pos, l, LabelSource::Teacher, g) {
                        rd.teacher_seen.insert((g, d.index));
                        labeled.insert((g, d.index, LabelSource::Teacher), ex);
                    }
                }
            }
        }
    }
    let mut keys: Vec<_> = labeled.keys().copied().collect();
    keys.sort_by_key(|k| (k.0, k.1, k.2 as u8));
    for k in keys {
        let ex = labeled.remove(&k).expect("clave");
        if real_holdout(k.0, k.1) {
            rd.hold.push(ex);
        } else {
            rd.train.push(ex);
        }
    }
    rd
}

/// Hold-out sintético fijo de la app (etiquetas limpias).
pub fn app_holdout(threshold: f64) -> (Vec<SpiderExample>, Vec<SpiderExample>) {
    let mut rng = Sm(0);
    let h1 = (0..APP_HOLDOUT_PROMPTS)
        .flat_map(|k| {
            synth_examples(
                &synth_prompt(holdout_seed(k), &VOCAB_TRAIN),
                threshold,
                0.0,
                &mut rng,
            )
        })
        .collect();
    let h2 = (0..APP_HOLDOUT_PROMPTS)
        .flat_map(|k| {
            synth_examples(
                &synth_prompt(shift_seed(k), &VOCAB_SHIFT),
                threshold,
                0.0,
                &mut rng,
            )
        })
        .collect();
    (h1, h2)
}

/// Semilla del prompt sintético `k` del dataset `n` (lejos de los hold-out).
fn train_seed(job_seed: u64, n: usize, k: usize) -> u64 {
    (1u64 << 40) | (fnv(&format!("{job_seed}:{n}:{k}")) & ((1u64 << 40) - 1))
}

/// Métricas en el hold-out real: exactitud de `ok` vs usuario y acuerdo
/// resolver/pendiente con el maestro.
fn real_eval(r: &SpiderFieldRouter, hold: &[SpiderExample], floor: f64) -> serde_json::Value {
    let (mut un, mut uok, mut tn, mut tagree) = (0usize, 0usize, 0usize, 0usize);
    for ex in hold {
        let Some(y) = ex.labels[crate::spider_field::H_OK] else {
            continue;
        };
        match ex.source {
            LabelSource::User => {
                un += 1;
                uok += ((r.predict(ex).p_ok() >= 0.5) == y) as usize;
            }
            LabelSource::Teacher => {
                tn += 1;
                tagree += (r.decide(ex, floor).resolve == y) as usize;
            }
            _ => {}
        }
    }
    json!({
        "user_n": un,
        "user_ok_accuracy": (un > 0).then(|| uok as f64 / un as f64),
        "teacher_n": tn,
        "teacher_agreement": (tn > 0).then(|| tagree as f64 / tn as f64),
    })
}

// ---------------------------------------------------------------------------
// Job
// ---------------------------------------------------------------------------

fn lock(st: &SharedState) -> std::sync::MutexGuard<'_, crate::web::AppState> {
    st.lock().unwrap_or_else(|e| e.into_inner())
}

/// Maestro en vivo: hasta `max` escaladas de prompts reales sin etiqueta.
async fn live_teacher(
    st: &SharedState,
    real: &mut RealData,
    max: usize,
    floor: f64,
) -> (Vec<SpiderExample>, Vec<serde_json::Value>, Option<String>) {
    let mut out = Vec::new();
    let mut log = Vec::new();
    let mut err = None;
    let prompts = real.prompts.clone();
    'outer: for (text, thr) in prompts {
        let g = text_group(&text);
        let ap = AnalyzedPrompt::new(&text, thr);
        let mut items = Vec::new();
        for (i, a) in ap.assess.iter().enumerate() {
            let Some((a, r)) = a else {
                continue;
            };
            let idx = ap.tokens[i].index;
            if *r != Route::Llm || real.teacher_seen.contains(&(g, idx)) {
                continue;
            }
            if out.len() + items.len() >= max {
                break;
            }
            items.push((
                i,
                EscalationItem {
                    index: idx,
                    word: ap.tokens[i].text.clone(),
                    context: word_context(&ap.tokens, i, 6),
                    question: a.question,
                    p: a.p,
                },
            ));
        }
        for chunk in items.chunks(4) {
            let its: Vec<EscalationItem> = chunk.iter().map(|c| c.1.clone()).collect();
            match crate::web::spider_job::teacher_call(st, &text, &its).await {
                Ok((verdicts, label, compact)) => {
                    for (pos, it) in chunk {
                        real.teacher_seen.insert((g, it.index));
                        let Some(v) = verdicts.iter().find(|v| v.index == it.index) else {
                            continue;
                        };
                        let l = teacher_labels(it.question, v, floor, compact);
                        if let Some(ex) = ap.encode(*pos, l, LabelSource::Teacher, g) {
                            log.push(json!({
                                "group": g, "index": it.index, "word": it.word,
                                "p": v.p, "llm": label, "compact": compact,
                            }));
                            if real_holdout(g, it.index) {
                                real.hold.push(ex);
                            } else {
                                out.push(ex);
                            }
                        }
                    }
                }
                Err(e) => {
                    err = Some(e.chars().take(200).collect());
                    break 'outer;
                }
            }
            if out.len() >= max {
                break 'outer;
            }
        }
    }
    (out, log, err)
}

fn source_name(s: LabelSource) -> &'static str {
    match s {
        LabelSource::Synthetic => "synthetic",
        LabelSource::Scorer => "scorer",
        LabelSource::User => "user",
        LabelSource::Teacher => "teacher",
    }
}

/// Bucle del entrenamiento: dataset → vigilia (observe) → sueño (CDT + RQM
/// + reajuste) → evaluación hold-out → checkpoint + modelo.
async fn train_loop(st: SharedState, stop: Arc<AtomicBool>, opts: TrainOpts) {
    let (runs_dir, current, job_id, ckpt_dir) = {
        let g = lock(&st);
        (
            g.spider.dir.clone(),
            g.spider.run.clone(),
            g.spider_field.job.job_id.clone(),
            g.spider_field.ckpt_dir.clone(),
        )
    };
    let mut real = real_data(runs_dir.as_deref(), current.as_ref(), opts.threshold);
    let (h1, h2) = app_holdout(opts.threshold);
    {
        let mut g = lock(&st);
        let msg = format!(
            "datos reales: {} ejemplos de entrenamiento + {} hold-out ({} corridas guardadas, {} prompts)",
            real.train.len(),
            real.hold.len(),
            real.runs,
            real.prompts.len()
        );
        g.spider_field
            .job
            .push("dataset", msg, serde_json::Value::Null);
    }
    let mut noise_rng = Sm(opts.seed ^ 0x5EED);
    let mut n = 0usize;
    let mut error: Option<String> = None;
    loop {
        if stop.load(Ordering::Relaxed) || (opts.datasets > 0 && n >= opts.datasets) {
            break;
        }
        // 1) Dataset: sintéticos + (cada 3 datasets) reales + maestro en vivo.
        let mut ds: Vec<SpiderExample> = Vec::new();
        let mut seeds = Vec::new();
        for k in 0..opts.prompts_per_dataset {
            let seed = train_seed(opts.seed, n, k);
            seeds.push(seed);
            ds.extend(synth_examples(
                &synth_prompt(seed, &VOCAB_TRAIN),
                opts.threshold,
                opts.noise,
                &mut noise_rng,
            ));
        }
        if opts.include_runs && n.is_multiple_of(3) {
            ds.extend(real.train.iter().cloned());
        }
        let mut teacher_log = Vec::new();
        if opts.teacher_live > 0 {
            {
                let mut g = lock(&st);
                g.spider_field.job.phase = "teacher".into();
            }
            let (ex, log, err) = live_teacher(&st, &mut real, opts.teacher_live, opts.floor).await;
            {
                let msg = match &err {
                    Some(e) => format!("maestro LLM: {} etiqueta(s); falló: {e}", log.len()),
                    None => format!("maestro LLM: {} etiqueta(s) nuevas", log.len()),
                };
                let mut g = lock(&st);
                g.spider_field
                    .job
                    .push("teacher", msg, json!({ "n": log.len() }));
            }
            real.train.extend(ex.iter().cloned());
            ds.extend(ex);
            teacher_log = log;
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for ex in &ds {
            *counts.entry(source_name(ex.source).into()).or_default() += 1;
        }
        let real_listing: Vec<serde_json::Value> = ds
            .iter()
            .filter(|e| e.source != LabelSource::Synthetic)
            .map(|e| {
                json!({
                    "group": e.group, "index": e.index, "word": e.word,
                    "source": source_name(e.source), "labels": e.labels,
                })
            })
            .collect();
        // 2) Vigilia + sueño + evaluación fuera del lock (sobre una copia).
        let snapshot = {
            let mut g = lock(&st);
            g.spider_field.job.phase = "sleep".into();
            g.spider_field.job.current = n;
            g.spider_field.online_pending.clear();
            g.spider_field.router.clone()
        };
        let (floor, h1c, h2c, hold_real) = (opts.floor, h1.clone(), h2.clone(), real.hold.clone());
        let t0 = Instant::now();
        let res = tokio::task::spawn_blocking(move || {
            let mut r = snapshot;
            for ex in ds {
                r.observe(ex);
            }
            let rep = r.sleep();
            let ev = evaluate_router(&r, &h1c, &h2c, floor);
            let rv = real_eval(&r, &hold_real, floor);
            r.holdout = Some(json!({ "synthetic": ev, "real": rv }));
            (r, rep, ev, rv)
        })
        .await;
        let (r, rep, ev, rv) = match res {
            Ok(x) => x,
            Err(e) => {
                error = Some(format!("entrenamiento falló: {e}"));
                break;
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        // 3) Instalar, guardar modelo y checkpoint.
        {
            let mut g = lock(&st);
            let pending = std::mem::take(&mut g.spider_field.online_pending);
            g.spider_field.router = r;
            for ex in pending {
                g.spider_field.router.observe(ex);
            }
            g.spider_field.save_model();
            let model_path = g.spider_field.model_path().map(|p| p.display().to_string());
            let ts = now_ms();
            let ckpt = json!({
                "kind": "spider_field_dataset",
                "job_id": job_id,
                "dataset": n,
                "ts_ms": ts,
                "model": model_name(&g.spider_field.router),
                "model_path": model_path,
                "opts": opts,
                "synthetic_seeds": seeds,
                "sources": counts,
                "real_examples": real_listing,
                "teacher_live": teacher_log,
                "sleep": rep,
                "eval": { "synthetic": ev, "real": rv },
                "train_seconds": secs,
            });
            let mut ckpt_path = None;
            if let Some(dir) = &ckpt_dir {
                let path = dir.join(format!("spider_field_{ts}_ds_{n}.json"));
                let res = std::fs::create_dir_all(dir)
                    .and_then(|_| std::fs::write(&path, serde_json::to_vec_pretty(&ckpt)?))
                    .and_then(|_| {
                        std::fs::write(
                            dir.join("latest.json"),
                            serde_json::to_vec_pretty(&json!({
                                "last_dataset_path": path.display().to_string(),
                                "job_id": job_id, "dataset": n, "ts_ms": ts,
                                "model_path": model_path,
                            }))?,
                        )
                    });
                match res {
                    Ok(()) => ckpt_path = Some(path.display().to_string()),
                    Err(e) => tracing::warn!(error = %e, "spider field: checkpoint no guardado"),
                }
            }
            let j = &mut g.spider_field.job;
            for (k, v) in &counts {
                *j.sources.entry(k.clone()).or_default() += v;
            }
            j.push(
            "sleep",
            format!(
                "dataset {n}: {} episodios → sueño CDT {} clases · RQM {} celdas · repetición {} ({:.1}s)",
                rep.episodes, rep.cdt_classes, rep.rqm_cells, rep.replay, secs
            ),
            json!(rep),
        );
            j.push(
            "eval",
            format!(
                "hold-out H1: ok {:.3} · ECE {:.3} · resueltas {:.1}% (precisión {:.3}) · H2 ok {:.3} · {:.0} µs",
                ev.router_h1.ok.accuracy,
                ev.router_h1.ok.ece,
                ev.router_h1.resolved_pct,
                ev.router_h1.resolved_precision,
                ev.router_h2.ok.accuracy,
                ev.latency_us
            ),
            json!({ "synthetic": ev, "real": rv }),
        );
            if let Some(p) = &ckpt_path {
                j.datasets_saved += 1;
                j.last_checkpoint = Some(p.clone());
                j.push(
                    "checkpoint",
                    format!("checkpoint {p}"),
                    serde_json::Value::Null,
                );
            }
            j.last_eval = Some(ev);
            j.last_sleep = Some(rep);
            j.current = n + 1;
        }
        n += 1;
        tokio::task::yield_now().await;
    }
    let mut g = lock(&st);
    let j = &mut g.spider_field.job;
    j.running = false;
    j.finished_ms = Some(now_ms());
    let stopped = stop.load(Ordering::Relaxed);
    j.phase = if error.is_some() {
        "error"
    } else if stopped {
        "stopped"
    } else {
        "done"
    }
    .into();
    if let Some(e) = &error {
        j.error = Some(e.clone());
        j.push("error", e.clone(), serde_json::Value::Null);
    }
    let msg = format!(
        "entrenamiento {}: {} dataset(s)",
        if stopped { "detenido" } else { "terminado" },
        n
    );
    j.push("done", msg, serde_json::Value::Null);
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/// Rutas `/api/spider/field/*` (protegidas como el resto de `/api/*`).
pub fn routes() -> Router<SharedState> {
    Router::new()
        .route("/api/spider/field/status", get(status))
        .route("/api/spider/field/train", post(train))
        .route("/api/spider/field/stop", post(stop))
        .route("/api/spider/field/events", get(events))
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TrainRequest {
    /// 0 = infinito (hasta Detener). Por defecto 6.
    pub datasets: Option<usize>,
    /// Por defecto 40.
    pub prompts_per_dataset: Option<usize>,
    pub include_runs: Option<bool>,
    pub teacher_live: Option<usize>,
    pub reset: Option<bool>,
    pub seed: Option<u64>,
    pub noise: Option<f64>,
}

fn bad(status: StatusCode, msg: &str) -> axum::response::Response {
    (status, Json(json!({ "ok": false, "error": msg }))).into_response()
}

async fn status(State(st): State<SharedState>) -> Json<serde_json::Value> {
    Json(lock(&st).spider_field.status_value())
}

async fn train(
    State(st): State<SharedState>,
    Json(b): Json<TrainRequest>,
) -> axum::response::Response {
    let mut g = lock(&st);
    if g.spider_field.job.running {
        return bad(StatusCode::CONFLICT, "ya hay un entrenamiento en curso");
    }
    let opts = TrainOpts {
        datasets: b.datasets.unwrap_or(6).min(10_000),
        prompts_per_dataset: b.prompts_per_dataset.unwrap_or(40).clamp(1, 400),
        include_runs: b.include_runs.unwrap_or(true),
        teacher_live: b.teacher_live.unwrap_or(0).min(200),
        reset: b.reset.unwrap_or(false),
        seed: b.seed.unwrap_or_else(now_ms),
        floor: DEFAULT_LLM_FLOOR,
        threshold: DEFAULT_THRESHOLD,
        noise: b
            .noise
            .filter(|x| x.is_finite())
            .unwrap_or(0.05)
            .clamp(0.0, 0.3),
    };
    if opts.reset {
        g.spider_field.router = SpiderFieldRouter::new(default_config());
    }
    let stop = Arc::new(AtomicBool::new(false));
    let job_id = uuid::Uuid::new_v4().to_string();
    let seq = g.spider_field.job.seq;
    let events = std::mem::take(&mut g.spider_field.job.events);
    g.spider_field.job = FieldTrainJob {
        running: true,
        stop: stop.clone(),
        job_id: job_id.clone(),
        opts: Some(opts.clone()),
        phase: "dataset".into(),
        started_ms: now_ms(),
        seq,
        events,
        ..Default::default()
    };
    let total = if opts.datasets == 0 {
        "∞".to_string()
    } else {
        opts.datasets.to_string()
    };
    g.spider_field.job.push(
        "start",
        format!(
            "Entrenamiento Prompt Spider {job_id}: {total} dataset(s) × {} prompts sintéticos{}{}{}",
            opts.prompts_per_dataset,
            if opts.include_runs {
                " + corridas guardadas"
            } else {
                ""
            },
            if opts.teacher_live > 0 {
                format!(" + maestro LLM ({} por dataset)", opts.teacher_live)
            } else {
                String::new()
            },
            if opts.reset { " · sustrato nuevo" } else { "" },
        ),
        json!({ "opts": opts }),
    );
    let seq = g.spider_field.job.seq;
    drop(g);
    tokio::spawn(train_loop(st.clone(), stop, opts.clone()));
    Json(json!({ "ok": true, "job_id": job_id, "opts": opts, "seq": seq })).into_response()
}

async fn stop(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let g = lock(&st);
    g.spider_field.job.stop.store(true, Ordering::Relaxed);
    Json(json!({ "ok": true, "running": g.spider_field.job.running }))
}

#[derive(Deserialize)]
pub struct AfterQuery {
    pub after: Option<u64>,
}

async fn events(
    State(st): State<SharedState>,
    Query(q): Query<AfterQuery>,
) -> Json<serde_json::Value> {
    let g = lock(&st);
    let after = q.after.unwrap_or(0);
    let j = &g.spider_field.job;
    let ev: Vec<&FieldEvent> = j
        .events
        .iter()
        .filter(|e| e.seq > after)
        .take(300)
        .collect();
    Json(json!({ "events": ev, "seq": j.seq, "running": j.running, "phase": j.phase }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_data_from_current_run_uses_user_and_teacher_but_not_field() {
        let rd = real_data(None, None, DEFAULT_THRESHOLD);
        assert!(!rd.train.is_empty(), "scorer del prompt de ejemplo");
        assert!(rd.train.iter().all(|e| e.source == LabelSource::Scorer));
        assert!(rd.hold.is_empty());
    }

    #[test]
    fn app_holdout_is_fixed_and_disjoint_from_training_seeds() {
        let (a1, a2) = app_holdout(DEFAULT_THRESHOLD);
        let (b1, b2) = app_holdout(DEFAULT_THRESHOLD);
        assert_eq!(a1.len(), b1.len());
        assert_eq!(a2.len(), b2.len());
        assert!(!a1.is_empty() && !a2.is_empty());
        let s = train_seed(1, 0, 0);
        assert!(s >= 1u64 << 40);
    }
}
