//! Puente circadiano: vigilia con Gemma adaptativo + sueño entrenando el núcleo CTP.

use crate::adaptive_gemma2::{AdaptiveThermoMemory, SleepConsolidationReport};
use crate::gemma2_thermo_hybrid_llm::{Gemma2ThermoHybridConfig, Gemma2ThermoHybridLlm};
use crate::gemma2_thermo_hybrid_session::{
    chat_session_path, load_chat_session, restore_hybrid_from_session, save_chat_session,
    sanitize_chat_name, unix_now,
};
use crate::native_checkpoint::atomic_write;
use crate::native_gemma2::QuantizedGemma2;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub const DEFAULT_CIRCADIAN_ROOT: &str = "data/native_gemma2_circadian";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WakeTurnRecord {
    pub turn: u64,
    pub unix: u64,
    pub user: String,
    pub assistant: String,
    pub prompt_tokens: Vec<u32>,
    pub response_tokens: Vec<u32>,
    pub quality: f32,
    pub executed_layers: usize,
    pub sleep_trained: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SleepDatasetEntry {
    pub turn: u64,
    pub user: String,
    pub assistant: String,
    pub sequence_tokens: Vec<u32>,
    pub quality: f32,
    pub executed_layers: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ThermoSleepTrainingReport {
    pub sequences: usize,
    pub windows: usize,
    pub mean_alignment_mse: f32,
    pub attractors_after: usize,
    pub sleep_cycles: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CircadianSleepReport {
    pub adaptive: SleepConsolidationReport,
    pub dataset_entries: usize,
    pub thermo: ThermoSleepTrainingReport,
}

#[derive(Clone, Debug)]
pub struct CircadianPaths {
    pub root: PathBuf,
    pub chat_name: String,
    pub adaptive_root: PathBuf,
    pub thermo_session: PathBuf,
    pub wake_journal: PathBuf,
    pub sleep_dataset: PathBuf,
    pub sleep_training: PathBuf,
}

#[derive(Clone, Debug)]
pub struct CircadianSleepConfig {
    pub min_quality: f32,
    pub max_sequences: usize,
    pub learning_rate: f32,
    pub consolidate_every: usize,
}

impl Default for CircadianSleepConfig {
    fn default() -> Self {
        Self {
            min_quality: 0.05,
            max_sequences: 64,
            learning_rate: 0.02,
            consolidate_every: 16,
        }
    }
}

impl CircadianPaths {
    pub fn for_chat(root: impl AsRef<Path>, chat_name: &str) -> Result<Self, String> {
        let chat_name = sanitize_chat_name(chat_name)?;
        let root = root.as_ref().join(&chat_name);
        Ok(Self {
            adaptive_root: root.join("adaptive"),
            thermo_session: chat_session_path(&root, "thermo"),
            wake_journal: root.join("wake").join("journal.jsonl"),
            sleep_dataset: root.join("sleep").join("dataset.jsonl"),
            sleep_training: root.join("sleep").join("last_training.json"),
            chat_name,
            root,
        })
    }
}

pub struct WakeJournal {
    path: PathBuf,
}

impl WakeJournal {
    pub fn open(path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        Ok(Self { path })
    }

    pub fn append(&self, record: &WakeTurnRecord) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| error.to_string())?;
        let line = serde_json::to_string(record).map_err(|error| error.to_string())?;
        writeln!(file, "{line}").map_err(|error| error.to_string())
    }

    pub fn load_all(&self) -> Result<Vec<WakeTurnRecord>, String> {
        if !self.path.is_file() {
            return Ok(Vec::new());
        }
        let file = fs::File::open(&self.path).map_err(|error| error.to_string())?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|error| error.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            records.push(serde_json::from_str(&line).map_err(|error| error.to_string())?);
        }
        Ok(records)
    }

    pub fn rewrite(&self, records: &[WakeTurnRecord]) -> Result<(), String> {
        let body = records
            .iter()
            .map(|record| serde_json::to_string(record))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?
            .join("\n");
        let payload = if body.is_empty() {
            Vec::new()
        } else {
            format!("{body}\n").into_bytes()
        };
        atomic_write(&self.path, &payload)
    }
}

pub fn load_or_create_hybrid(
    model: &QuantizedGemma2,
    paths: &CircadianPaths,
    hybrid_config: Gemma2ThermoHybridConfig,
    history: &[(String, String)],
    turns: u64,
    created_at_unix: u64,
) -> Result<(Gemma2ThermoHybridLlm, u64), String> {
    if paths.thermo_session.is_file() {
        let saved = load_chat_session(&paths.thermo_session)?;
        let hybrid = restore_hybrid_from_session(model, hybrid_config, &saved)?;
        Ok((hybrid, saved.created_at_unix))
    } else {
        let hybrid = Gemma2ThermoHybridLlm::for_gemma(model, hybrid_config)
            .map_err(|error| error.to_string())?;
        save_chat_session(
            &paths.thermo_session,
            &paths.chat_name,
            created_at_unix,
            turns,
            history,
            &hybrid,
        )?;
        Ok((hybrid, created_at_unix))
    }
}

pub fn persist_hybrid_session(
    paths: &CircadianPaths,
    created_at_unix: u64,
    turns: u64,
    history: &[(String, String)],
    hybrid: &mut Gemma2ThermoHybridLlm,
) -> Result<(), String> {
    let _ = hybrid.force_sleep();
    save_chat_session(
        &paths.thermo_session,
        &paths.chat_name,
        created_at_unix,
        turns,
        history,
        hybrid,
    )
}

pub fn export_sleep_dataset(records: &[WakeTurnRecord]) -> Vec<SleepDatasetEntry> {
    records
        .iter()
        .filter(|record| !record.sleep_trained && record.quality >= 0.05)
        .map(|record| {
            let mut sequence_tokens = record.prompt_tokens.clone();
            sequence_tokens.extend(&record.response_tokens);
            SleepDatasetEntry {
                turn: record.turn,
                user: record.user.clone(),
                assistant: record.assistant.clone(),
                sequence_tokens,
                quality: record.quality,
                executed_layers: record.executed_layers,
            }
        })
        .collect()
}

pub fn write_sleep_dataset(path: &Path, entries: &[SleepDatasetEntry]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let body = entries
        .iter()
        .map(|entry| serde_json::to_string(entry))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?
        .join("\n");
    let payload = if body.is_empty() {
        Vec::new()
    } else {
        format!("{body}\n").into_bytes()
    };
    atomic_write(path, &payload)
}

fn embed_sequence(model: &QuantizedGemma2, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    tokens
        .iter()
        .map(|token| {
            model
                .embed_token(*token)
                .map_err(|error| error.to_string())?
                .to_vec1::<f32>()
                .map_err(|error| error.to_string())
        })
        .collect()
}

pub fn train_thermo_from_dataset(
    model: &mut QuantizedGemma2,
    hybrid: &mut Gemma2ThermoHybridLlm,
    entries: &[SleepDatasetEntry],
    config: &CircadianSleepConfig,
) -> Result<ThermoSleepTrainingReport, String> {
    let window = hybrid.config().thermo_window.max(4);
    let mut report = ThermoSleepTrainingReport::default();
    let mut mse_sum = 0.0f32;

    for entry in entries.iter().take(config.max_sequences) {
        if entry.sequence_tokens.len() < 2 || entry.quality < config.min_quality {
            continue;
        }
        let embeddings = embed_sequence(model, &entry.sequence_tokens)?;
        let teachers = model
            .prefill_teacher_hiddens(&entry.sequence_tokens)
            .map_err(|error| error.to_string())?;
        if teachers.len() != embeddings.len() {
            continue;
        }
        report.sequences = report.sequences.saturating_add(1);
        for end in window..=embeddings.len() {
            let start = end.saturating_sub(window);
            let slice = &embeddings[start..end];
            let teacher = &teachers[end - 1];
            let mse = hybrid
                .supervised_align_step(slice, teacher, config.learning_rate)
                .map_err(|error| error.to_string())?;
            mse_sum += mse;
            report.windows = report.windows.saturating_add(1);
            if config.consolidate_every > 0
                && report.windows.is_multiple_of(config.consolidate_every)
            {
                let _ = hybrid.force_sleep();
                report.sleep_cycles = hybrid.sleep_cycles();
            }
        }
    }

    let _ = hybrid.force_sleep();
    report.sleep_cycles = hybrid.sleep_cycles();
    report.attractors_after = hybrid
        .thermo_engine()
        .hybrid_engine()
        .attractors()
        .len();
    if report.windows > 0 {
        report.mean_alignment_mse = mse_sum / report.windows as f32;
    }
    Ok(report)
}

pub fn run_sleep_phase(
    model: &mut QuantizedGemma2,
    hybrid: &mut Gemma2ThermoHybridLlm,
    memory: &mut AdaptiveThermoMemory,
    journal: &WakeJournal,
    paths: &CircadianPaths,
    sleep_config: &CircadianSleepConfig,
) -> Result<CircadianSleepReport, String> {
    let adaptive = memory.consolidate_sleep()?;
    let mut records = journal.load_all()?;
    let dataset = export_sleep_dataset(&records);
    write_sleep_dataset(&paths.sleep_dataset, &dataset)?;

    let thermo = train_thermo_from_dataset(model, hybrid, &dataset, sleep_config)?;

    for record in &mut records {
        if !record.sleep_trained && record.quality >= sleep_config.min_quality {
            record.sleep_trained = true;
        }
    }
    journal.rewrite(&records)?;

    let report = CircadianSleepReport {
        adaptive,
        dataset_entries: dataset.len(),
        thermo,
    };
    atomic_write(
        &paths.sleep_training,
        &serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )?;
    Ok(report)
}

pub fn new_wake_record(
    turn: u64,
    user: &str,
    assistant: &str,
    prompt_tokens: &[u32],
    response_tokens: &[u32],
    quality: f32,
    executed_layers: usize,
) -> WakeTurnRecord {
    WakeTurnRecord {
        turn,
        unix: unix_now(),
        user: user.to_string(),
        assistant: assistant.to_string(),
        prompt_tokens: prompt_tokens.to_vec(),
        response_tokens: response_tokens.to_vec(),
        quality,
        executed_layers,
        sleep_trained: false,
    }
}
