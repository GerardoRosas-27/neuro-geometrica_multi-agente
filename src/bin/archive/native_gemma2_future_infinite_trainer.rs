//! Entrenamiento infinito: una entrada presente, futuros de Gemma, selección
//! por energía libre y consolidación CDT.
//!
//! Guarda `latest.json` atómicamente por tiempo o ciclos, crea hitos históricos
//! y vuelve a guardar al recibir Ctrl+C. Al reiniciar, restaura geometría,
//! atractores y contadores antes de continuar.

use cdt_rqm_epr::future_guided_training::{
    future_training_engine, FutureGuidedTrainer, FutureGuidedTrainingConfig, FutureTrainingMode,
};
use cdt_rqm_epr::gemma_future_generator::GemmaFutureGenerator;
use cdt_rqm_epr::native_checkpoint::atomic_write;
use cdt_rqm_epr::native_gemma2::resolve_gemma2_model_path;
use cdt_rqm_epr::native_gemma2_runtime::Gemma2GenerationConfig;
use cdt_rqm_epr::native_hybrid_phasor_cdt_engine::{ConsolidatedCdtAttractor, NativePhasorCue};
use num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const VERSION: u32 = 1;
const DEFAULT_ROOT: &str = "data/native_gemma2_future_infinite_training";

#[derive(Clone, Debug)]
struct Config {
    root: PathBuf,
    input: String,
    nodes: usize,
    proposals: usize,
    iterations: usize,
    max_tokens: usize,
    context: usize,
    device: String,
    model: Option<PathBuf>,
    save_every: Duration,
    save_cycles: u64,
    milestone_cycles: u64,
    retry_delay: Duration,
    max_cycles: Option<u64>,
    seed: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Totals {
    cycles: u64,
    proposals: u64,
    parse_failures: u64,
    gates_passed: u64,
    consolidated: u64,
    rejected_by_efficiency: u64,
    energy_evaluations: u64,
    attention_ignitions: u64,
    last_free_energy: f32,
    best_free_energy: f32,
    last_residual: f32,
    last_handshake: f32,
    last_phi: f32,
    generated_tokens: u64,
    decode_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CoreCheckpoint {
    thermal_state: Vec<f32>,
    amplitude: Vec<f32>,
    phase: Vec<f32>,
    temperature: Vec<f32>,
    energy: Vec<f32>,
    activation: Vec<f32>,
    edge_weight: Vec<f32>,
    edge_phase: Vec<f32>,
    edge_stability: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AttractorCheckpoint {
    id: usize,
    prototype: Vec<[f32; 2]>,
    free_energy: f32,
    confidence: f32,
    consolidations: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    saved_unix_seconds: u64,
    input: String,
    nodes: usize,
    seed: u64,
    totals: Totals,
    core: CoreCheckpoint,
    attractors: Vec<AttractorCheckpoint>,
}

struct LockGuard {
    path: PathBuf,
    _file: File,
}

impl LockGuard {
    fn acquire(root: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        fs::create_dir_all(root)?;
        let path = root.join("training.lock");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                format!(
                    "ya existe {} ({error}); otro entrenador puede estar activo",
                    path.display()
                )
            })?;
        writeln!(file, "pid={}", std::process::id())?;
        file.sync_all()?;
        Ok(Self { path, _file: file })
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let _lock = LockGuard::acquire(&config.root)?;
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        signal.store(true, Ordering::SeqCst);
    })?;

    let cue = parse_input(&config.input, config.nodes)?;
    let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
    let training_config = FutureGuidedTrainingConfig {
        nodes: config.nodes,
        proposals_per_input: config.proposals,
        candidate_iterations: config.iterations,
        seed: config.seed,
        ..FutureGuidedTrainingConfig::default()
    };
    let engine = future_training_engine(
        training_config,
        FutureTrainingMode::PredictedFutureAdaptive,
        config.seed,
    )?;
    let generator = GemmaFutureGenerator::from_gguf(
        &model_path,
        &config.device,
        Gemma2GenerationConfig {
            max_tokens: config.max_tokens,
            context_limit: config.context,
            temperature: 0.75,
            top_p: 0.90,
            seed: config.seed,
        },
    )?;
    let mut trainer = FutureGuidedTrainer::new(engine, generator, config.proposals);
    let latest = config.root.join("latest.json");
    let mut totals = Totals {
        best_free_energy: f32::MAX,
        last_residual: f32::MAX,
        ..Totals::default()
    };
    if latest.exists() {
        let checkpoint: Checkpoint = serde_json::from_slice(&fs::read(&latest)?)?;
        restore_checkpoint(&config, &mut trainer, &mut totals, checkpoint)?;
    }

    println!(
        "trainer=gemma_future_infinite estado={} cycle={} modelo={} root={}",
        if totals.cycles == 0 {
            "nuevo"
        } else {
            "reanudado"
        },
        totals.cycles,
        model_path.display(),
        config.root.display()
    );
    println!(
        "entrada={} nodos={} propuestas={} iteraciones={} save_s={} save_cycles={}",
        config.input,
        config.nodes,
        config.proposals,
        config.iterations,
        config.save_every.as_secs(),
        config.save_cycles
    );
    println!("detener=Ctrl+C guardado_final=atomico");

    let started = Instant::now();
    let mut last_save = Instant::now();
    let mut last_milestone = totals.cycles / config.milestone_cycles;
    loop {
        if stop.load(Ordering::SeqCst)
            || config
                .max_cycles
                .is_some_and(|maximum| totals.cycles >= maximum)
        {
            let reason = if stop.load(Ordering::SeqCst) {
                "ctrl_c"
            } else {
                "max_cycles"
            };
            save_checkpoint(&config, &trainer, &totals, true)?;
            println!("event=finished reason={reason} cycle={}", totals.cycles);
            break;
        }

        let cycle = totals.cycles.saturating_add(1);
        let seed = config.seed ^ cycle.rotate_left(23);
        let cycle_started = Instant::now();
        match trainer.learn_from_input(&cue, seed)? {
            Some(episode) => {
                totals.cycles = cycle;
                totals.proposals = totals
                    .proposals
                    .saturating_add(episode.proposals_generated as u64);
                totals.gates_passed += u64::from(episode.gate_passed);
                totals.consolidated = totals
                    .consolidated
                    .saturating_add(episode.consolidated as u64);
                totals.rejected_by_efficiency = totals
                    .rejected_by_efficiency
                    .saturating_add(episode.rejected_by_efficiency as u64);
                totals.energy_evaluations = totals
                    .energy_evaluations
                    .saturating_add(episode.energy_evaluations as u64);
                totals.attention_ignitions = totals
                    .attention_ignitions
                    .saturating_add(episode.attention_ignitions as u64);
                totals.last_free_energy = episode.selected_free_energy;
                totals.best_free_energy = totals.best_free_energy.min(episode.selected_free_energy);
                totals.last_residual = episode.final_residual;
                totals.last_handshake = episode.handshake_coherence;
                totals.last_phi = episode.integrated_information;
                totals.generated_tokens = totals
                    .generated_tokens
                    .saturating_add(trainer.generator.last_metrics.generated_tokens as u64);
                totals.decode_seconds += trainer.generator.last_metrics.decode_seconds;
                println!(
                    "cycle={} proposals={} F={:.6} residual={:.3e} coherence={:.6} \
                     handshake={:.6} phi={:.6} gate={} consolidated={} memories={} \
                     gemma_tokens={} cycle_s={:.2} elapsed_s={:.1}",
                    cycle,
                    episode.proposals_generated,
                    episode.selected_free_energy,
                    episode.final_residual,
                    episode.phase_coherence,
                    episode.handshake_coherence,
                    episode.integrated_information,
                    episode.gate_passed,
                    episode.consolidated,
                    trainer.engine.attractors().len(),
                    trainer.generator.last_metrics.generated_tokens,
                    cycle_started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64()
                );
            }
            None => {
                totals.cycles = cycle;
                totals.parse_failures = totals.parse_failures.saturating_add(1);
                totals.generated_tokens = totals
                    .generated_tokens
                    .saturating_add(trainer.generator.last_metrics.generated_tokens as u64);
                totals.decode_seconds += trainer.generator.last_metrics.decode_seconds;
                println!(
                    "cycle={} event=no_parseable_future tokens={} output={:?} error={:?}",
                    cycle,
                    trainer.generator.last_metrics.generated_tokens,
                    trainer.generator.last_text,
                    trainer.generator.last_error
                );
                thread::sleep(config.retry_delay);
            }
        }

        atomic_write(
            &config.root.join("latest_gemma_output.txt"),
            trainer.generator.last_text.as_bytes(),
        )?;
        let periodic = totals.cycles.is_multiple_of(config.save_cycles)
            || last_save.elapsed() >= config.save_every;
        if periodic {
            save_checkpoint(&config, &trainer, &totals, false)?;
            last_save = Instant::now();
            println!(
                "event=checkpoint cycle={} consolidated={} failures={}",
                totals.cycles, totals.consolidated, totals.parse_failures
            );
        }
        let milestone = totals.cycles / config.milestone_cycles;
        if milestone > last_milestone {
            save_checkpoint(&config, &trainer, &totals, true)?;
            last_milestone = milestone;
            println!("event=milestone cycle={}", totals.cycles);
        }
    }
    Ok(())
}

fn capture_checkpoint(
    config: &Config,
    trainer: &FutureGuidedTrainer<GemmaFutureGenerator>,
    totals: &Totals,
) -> Checkpoint {
    let core = &trainer.engine.core;
    Checkpoint {
        version: VERSION,
        saved_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        input: config.input.clone(),
        nodes: config.nodes,
        seed: config.seed,
        totals: totals.clone(),
        core: CoreCheckpoint {
            thermal_state: core.thermal_state.clone(),
            amplitude: core.amplitude.clone(),
            phase: core.phase.clone(),
            temperature: core.temperature.clone(),
            energy: core.energy.clone(),
            activation: core.activation.clone(),
            edge_weight: core.edge_weight.clone(),
            edge_phase: core.edge_phase.clone(),
            edge_stability: core.edge_stability.clone(),
        },
        attractors: trainer
            .engine
            .attractors()
            .iter()
            .map(|attractor| AttractorCheckpoint {
                id: attractor.id,
                prototype: attractor
                    .prototype
                    .iter()
                    .map(|value| [value.re, value.im])
                    .collect(),
                free_energy: attractor.free_energy,
                confidence: attractor.confidence,
                consolidations: attractor.consolidations,
            })
            .collect(),
    }
}

fn restore_checkpoint(
    config: &Config,
    trainer: &mut FutureGuidedTrainer<GemmaFutureGenerator>,
    totals: &mut Totals,
    checkpoint: Checkpoint,
) -> Result<(), Box<dyn std::error::Error>> {
    if checkpoint.version != VERSION
        || checkpoint.nodes != config.nodes
        || checkpoint.seed != config.seed
        || checkpoint.input != config.input
    {
        return Err("checkpoint incompatible con entrada, nodos, seed o versión".into());
    }
    let core = &mut trainer.engine.core;
    copy_exact(
        &mut core.thermal_state,
        checkpoint.core.thermal_state,
        "thermal_state",
    )?;
    copy_exact(&mut core.amplitude, checkpoint.core.amplitude, "amplitude")?;
    copy_exact(&mut core.phase, checkpoint.core.phase, "phase")?;
    copy_exact(
        &mut core.temperature,
        checkpoint.core.temperature,
        "temperature",
    )?;
    copy_exact(&mut core.energy, checkpoint.core.energy, "energy")?;
    copy_exact(
        &mut core.activation,
        checkpoint.core.activation,
        "activation",
    )?;
    copy_exact(
        &mut core.edge_weight,
        checkpoint.core.edge_weight,
        "edge_weight",
    )?;
    copy_exact(
        &mut core.edge_phase,
        checkpoint.core.edge_phase,
        "edge_phase",
    )?;
    copy_exact(
        &mut core.edge_stability,
        checkpoint.core.edge_stability,
        "edge_stability",
    )?;
    trainer.engine.phasor.recompile_from_core(core)?;
    trainer.engine.phasor.synchronize_state_from_core(core)?;
    trainer.engine.restore_attractors(
        checkpoint
            .attractors
            .into_iter()
            .map(|attractor| ConsolidatedCdtAttractor {
                id: attractor.id,
                prototype: attractor
                    .prototype
                    .into_iter()
                    .map(|value| Complex32::new(value[0], value[1]))
                    .collect(),
                free_energy: attractor.free_energy,
                confidence: attractor.confidence,
                consolidations: attractor.consolidations,
            })
            .collect(),
    )?;
    *totals = checkpoint.totals;
    Ok(())
}

fn copy_exact(
    destination: &mut Vec<f32>,
    source: Vec<f32>,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if destination.len() != source.len() {
        return Err(format!(
            "checkpoint {name} incompatible: {} != {}",
            source.len(),
            destination.len()
        )
        .into());
    }
    *destination = source;
    Ok(())
}

fn save_checkpoint(
    config: &Config,
    trainer: &FutureGuidedTrainer<GemmaFutureGenerator>,
    totals: &Totals,
    milestone: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = capture_checkpoint(config, trainer, totals);
    let body = serde_json::to_vec_pretty(&checkpoint)?;
    atomic_write(&config.root.join("latest.json"), &body)?;
    if milestone {
        atomic_write(
            &config
                .root
                .join("checkpoints")
                .join(format!("cycle-{:012}.json", totals.cycles)),
            &body,
        )?;
    }
    Ok(())
}

fn parse_input(
    input: &str,
    nodes: usize,
) -> Result<Vec<NativePhasorCue>, Box<dyn std::error::Error>> {
    let mut cue = Vec::new();
    for field in input.split(',') {
        let (node, sign) = field
            .trim()
            .split_once(':')
            .ok_or_else(|| format!("entrada inválida: {field}"))?;
        let node = node.trim().parse::<usize>()?;
        if node >= nodes {
            return Err(format!("nodo {node} fuera de 0..{nodes}").into());
        }
        let phase = match sign.trim() {
            "+" | "1" => 0.0,
            "-" | "0" => std::f32::consts::PI,
            other => return Err(format!("signo inválido: {other}").into()),
        };
        cue.push(NativePhasorCue {
            node,
            amplitude: 1.0,
            phase,
        });
    }
    if cue.is_empty() {
        return Err("la entrada no puede estar vacía".into());
    }
    Ok(cue)
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config {
        root: PathBuf::from(DEFAULT_ROOT),
        input: String::new(),
        nodes: 64,
        proposals: 3,
        iterations: 256,
        max_tokens: 192,
        context: 1_024,
        device: "cpu".to_string(),
        model: None,
        save_every: Duration::from_secs(300),
        save_cycles: 5,
        milestone_cycles: 100,
        retry_delay: Duration::from_secs(2),
        max_cycles: None,
        seed: 0xF077_2026,
    };
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--root" => config.root = PathBuf::from(required(&mut args, "--root")?),
            "--input" => config.input = required(&mut args, "--input")?,
            "--nodes" => config.nodes = required(&mut args, "--nodes")?.parse::<usize>()?.max(16),
            "--proposals" => {
                config.proposals = required(&mut args, "--proposals")?.parse::<usize>()?.max(1)
            }
            "--iterations" => {
                config.iterations = required(&mut args, "--iterations")?
                    .parse::<usize>()?
                    .max(4)
            }
            "--max-tokens" => {
                config.max_tokens = required(&mut args, "--max-tokens")?
                    .parse::<usize>()?
                    .max(16)
            }
            "--context" => {
                config.context = required(&mut args, "--context")?.parse::<usize>()?.max(64)
            }
            "--device" => config.device = required(&mut args, "--device")?,
            "--model" => config.model = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--save-seconds" => {
                config.save_every = Duration::from_secs(
                    required(&mut args, "--save-seconds")?
                        .parse::<u64>()?
                        .max(1),
                )
            }
            "--save-cycles" => {
                config.save_cycles = required(&mut args, "--save-cycles")?.parse::<u64>()?.max(1)
            }
            "--milestone-cycles" => {
                config.milestone_cycles = required(&mut args, "--milestone-cycles")?
                    .parse::<u64>()?
                    .max(1)
            }
            "--retry-delay-ms" => {
                config.retry_delay =
                    Duration::from_millis(required(&mut args, "--retry-delay-ms")?.parse()?)
            }
            "--max-cycles" => {
                config.max_cycles = Some(required(&mut args, "--max-cycles")?.parse()?)
            }
            "--seed" => config.seed = required(&mut args, "--seed")?.parse()?,
            "--help" | "-h" => {
                println!(
                    "Uso: native_gemma2_future_infinite_trainer --input \"0:+,3:-\" \
                     [--root DIR] [--nodes N] [--proposals N] [--iterations N] \
                     [--max-tokens N] [--context N] [--device cpu|cuda] [--model GGUF] \
                     [--save-seconds N] [--save-cycles N] [--milestone-cycles N] \
                     [--retry-delay-ms N] [--max-cycles N] [--seed N]"
                );
                return Err("ayuda solicitada".into());
            }
            _ => return Err(format!("argumento desconocido: {argument}").into()),
        }
    }
    if config.input.is_empty() {
        return Err("falta --input \"0:+,3:-,...\"".into());
    }
    if config.context <= config.max_tokens + 32 {
        return Err("--context debe superar --max-tokens en al menos 32".into());
    }
    Ok(config)
}

fn required(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("falta valor para {name}"))
}
