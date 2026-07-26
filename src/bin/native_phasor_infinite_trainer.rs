//! Entrenamiento fasorial infinito con ciclos wake/sleep y checkpoints.
//!
//! Toda inferencia de este bin usa la ruta L0 de `NativeMultiOperatorCore`,
//! que delega en `NativePhasorThermodynamicEngine::minimize_free_energy`.

use cdt_rqm_epr::native_checkpoint::atomic_write;
use cdt_rqm_epr::native_cognitive_closed_loop::{
    episode_from_solution, generate_dream_recipes, record_episode,
};
use cdt_rqm_epr::native_multi_operator_core::{
    NativeMultiOperatorCore, OperatorDeltaSnapshot, OperatorRecipe, OperatorSolution, PairFactor,
    RequestedOperator, UnaryFactor, VariableDomain, VariableSpec,
};
use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    DEFAULT_PHASOR_NODES_PER_SLICE, DEFAULT_PHASOR_STARTUP_SLICES,
};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const STATE_VERSION: u32 = 1;
const DEFAULT_ROOT: &str = "data/native_phasor_infinite_training";

#[derive(Clone, Debug)]
struct Config {
    root: PathBuf,
    sleep_every: u64,
    save_every_sleeps: u64,
    concepts: usize,
    nodes_per_concept: usize,
    learning_rate: f32,
    dreams_per_sleep: usize,
    dream_learning_rate_scale: f32,
    max_cycles: Option<u64>,
    seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainingState {
    version: u32,
    cycle: u64,
    wake_cycles: u64,
    sleep_cycles: u64,
    accepted_wake: u64,
    rejected_wake: u64,
    consolidated: u64,
    rejected_sleep: u64,
    last_initial_energy: f32,
    last_final_energy: f32,
    best_final_energy: f32,
    last_residual: f32,
    #[serde(default)]
    dream_generated: u64,
    #[serde(default)]
    dream_consolidated: u64,
    #[serde(default)]
    dream_rejected: u64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            version: STATE_VERSION,
            cycle: 0,
            wake_cycles: 0,
            sleep_cycles: 0,
            accepted_wake: 0,
            rejected_wake: 0,
            consolidated: 0,
            rejected_sleep: 0,
            last_initial_energy: 0.0,
            last_final_energy: 0.0,
            best_final_energy: f32::MAX,
            last_residual: f32::MAX,
            dream_generated: 0,
            dream_consolidated: 0,
            dream_rejected: 0,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    fs::create_dir_all(&config.root)?;
    let state_path = config.root.join("latest.state.json");
    let operator_path = config.root.join("latest.operator_deltas.json");
    let mut state = load_state(&state_path)?;
    let mut global_core = build_global_core(config.seed);
    let mut engine = NativeMultiOperatorCore::default();
    if operator_path.exists() {
        engine.snapshot = OperatorDeltaSnapshot::load(&operator_path)?;
        engine.snapshot.apply_to_core(&mut global_core, 1.0)?;
    }
    println!(
        "trainer=phasor_infinite estado={} ciclo={} core_nodos={} core_aristas={} \
         conceptos={} nodos_concepto={} sleep_cada={} snapshot={}",
        if state.cycle == 0 {
            "nuevo"
        } else {
            "reanudado"
        },
        state.cycle,
        global_core.node_count(),
        global_core.edge_count(),
        config.concepts,
        config.nodes_per_concept,
        config.sleep_every,
        config.root.display()
    );
    println!("inferencia=L0_fasorial objetivo=mínimo_energía_libre backend=CPU_sparse");

    let mut pending = VecDeque::<OperatorRecipe>::new();
    let started = Instant::now();
    loop {
        if config
            .max_cycles
            .is_some_and(|maximum| state.cycle >= maximum)
        {
            if !pending.is_empty() {
                sleep_phase(
                    &config,
                    &mut state,
                    &mut engine,
                    &mut global_core,
                    &mut pending,
                )?;
            }
            save_checkpoint(&state_path, &operator_path, &state, &engine)?;
            println!("event=finished cycle={} reason=max_cycles", state.cycle);
            return Ok(());
        }

        state.cycle = state.cycle.saturating_add(1);
        state.wake_cycles = state.wake_cycles.saturating_add(1);
        let concept = (state.cycle.saturating_sub(1) as usize) % config.concepts;
        let recipe = concept_recipe(concept, config.nodes_per_concept, state.cycle, config.seed);
        let wake_started = Instant::now();
        let solved = engine.solve(&recipe, &global_core)?;
        let (initial_energy, final_energy, residual, accepted) = match &solved.solution {
            OperatorSolution::L0(solution) => (
                solution.initial_energy,
                solution.final_energy,
                solution.residual,
                solution.verified,
            ),
            _ => return Err("el entrenador fasorial seleccionó un operador no L0".into()),
        };
        state.last_initial_energy = initial_energy;
        state.last_final_energy = final_energy;
        state.last_residual = residual;
        state.best_final_energy = state.best_final_energy.min(final_energy);
        if accepted {
            state.accepted_wake = state.accepted_wake.saturating_add(1);
            pending.push_back(recipe);
        } else {
            state.rejected_wake = state.rejected_wake.saturating_add(1);
        }
        println!(
            "phase=wake cycle={} concept={} accepted={} F={:.6}->{:.6} residual={:.3e} \
             pending={} wake_ms={:.3} elapsed_s={:.1}",
            state.cycle,
            concept,
            accepted,
            initial_energy,
            final_energy,
            residual,
            pending.len(),
            wake_started.elapsed().as_secs_f64() * 1_000.0,
            started.elapsed().as_secs_f64()
        );

        if state.cycle % config.sleep_every == 0 {
            sleep_phase(
                &config,
                &mut state,
                &mut engine,
                &mut global_core,
                &mut pending,
            )?;
            if state.sleep_cycles % config.save_every_sleeps == 0 {
                save_checkpoint(&state_path, &operator_path, &state, &engine)?;
                println!(
                    "event=checkpoint cycle={} sleep={} recipes={} nodos_delta={} aristas_delta={}",
                    state.cycle,
                    state.sleep_cycles,
                    engine.snapshot.accepted_recipes.len(),
                    engine.snapshot.node_deltas.len(),
                    engine.snapshot.edge_deltas.len()
                );
            }
        }
    }
}

fn sleep_phase(
    config: &Config,
    state: &mut TrainingState,
    engine: &mut NativeMultiOperatorCore,
    global_core: &mut NativeThermoCdtSubstrate,
    pending: &mut VecDeque<OperatorRecipe>,
) -> Result<(), Box<dyn std::error::Error>> {
    state.sleep_cycles = state.sleep_cycles.saturating_add(1);
    let started = Instant::now();
    let before = pending.len();
    let mut accepted = 0u64;
    let mut rejected = 0u64;
    while let Some(recipe) = pending.pop_front() {
        let revalidated = engine.solve(&recipe, global_core)?;
        if revalidated.solution.verified() {
            engine.consolidate(&recipe, &revalidated, global_core, config.learning_rate)?;
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    state.consolidated = state.consolidated.saturating_add(accepted);
    state.rejected_sleep = state.rejected_sleep.saturating_add(rejected);
    let dreams = generate_dream_recipes(
        &engine.snapshot,
        state.sleep_cycles,
        config.dreams_per_sleep,
        config.seed,
    );
    let dream_candidates = dreams.len() as u64;
    let mut dream_accepted = 0u64;
    let mut dream_rejected = 0u64;
    for dream in dreams {
        let solved = engine.solve(&dream, global_core)?;
        if solved.solution.verified() {
            engine.consolidate(
                &dream,
                &solved,
                global_core,
                config.learning_rate * config.dream_learning_rate_scale,
            )?;
            let prompt = format!("sueño fasorial derivado {}", dream.name);
            record_episode(
                &mut engine.snapshot,
                episode_from_solution(&prompt, &dream, &solved),
            );
            dream_accepted += 1;
        } else {
            dream_rejected += 1;
        }
    }
    state.dream_generated = state.dream_generated.saturating_add(dream_candidates);
    state.dream_consolidated = state.dream_consolidated.saturating_add(dream_accepted);
    state.dream_rejected = state.dream_rejected.saturating_add(dream_rejected);
    println!(
        "phase=sleep sleep={} candidates={} consolidated={} rejected={} dreams={} \
         dreams_consolidated={} dreams_rejected={} core_edges={} \
         sleep_ms={:.3}",
        state.sleep_cycles,
        before,
        accepted,
        rejected,
        dream_candidates,
        dream_accepted,
        dream_rejected,
        global_core.edge_count(),
        started.elapsed().as_secs_f64() * 1_000.0
    );
    Ok(())
}

fn concept_recipe(concept: usize, nodes: usize, cycle: u64, seed: u64) -> OperatorRecipe {
    let variables = (0..nodes)
        .map(|node| VariableSpec {
            name: format!("concept_{concept}_node_{node}"),
            domain: VariableDomain::Phasor,
        })
        .collect::<Vec<_>>();
    let target_phase = |node: usize| {
        if deterministic_bit(seed ^ concept as u64, node as u64) {
            std::f32::consts::PI
        } else {
            0.0
        }
    };
    let mut pair_factors = Vec::with_capacity(nodes * 2);
    for node in 0..nodes {
        for hop in [1usize, 5usize] {
            let other = (node + hop) % nodes;
            if node < other || (node + hop) >= nodes {
                pair_factors.push(PairFactor {
                    a: variables[node].name.clone(),
                    b: variables[other].name.clone(),
                    weight: 1.0,
                    phase: (target_phase(other) - target_phase(node))
                        .rem_euclid(std::f32::consts::TAU),
                });
            }
        }
    }
    let unary_factors = (0..nodes)
        .step_by(16)
        .map(|node| {
            let corrupted = deterministic_bit(cycle ^ seed, node as u64) && node % 5 == 0;
            UnaryFactor {
                variable: variables[node].name.clone(),
                weight: 1.5,
                phase: target_phase(node) + if corrupted { std::f32::consts::PI } else { 0.0 },
            }
        })
        .collect();
    OperatorRecipe {
        name: format!("phasor_concept_{concept}"),
        requested_operator: RequestedOperator::L0,
        variables,
        unary_factors,
        pair_factors,
        oriented_faces: Vec::new(),
        flow_demands: Vec::new(),
        max_working_set: 8_192,
        ridge: 1.0e-3,
    }
}

fn deterministic_bit(seed: u64, value: u64) -> bool {
    let mut mixed = seed ^ value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;
    mixed & 1 == 1
}

fn build_global_core(seed: u64) -> NativeThermoCdtSubstrate {
    NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices: DEFAULT_PHASOR_STARTUP_SLICES,
        nodes_per_slice: DEFAULT_PHASOR_NODES_PER_SLICE,
        temperature: 0.0,
        seed,
        ..NativeThermoCdtConfig::default()
    })
}

fn load_state(path: &Path) -> Result<TrainingState, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(TrainingState::default());
    }
    let state: TrainingState = serde_json::from_slice(&fs::read(path)?)?;
    if state.version != STATE_VERSION {
        return Err(format!(
            "checkpoint incompatible: versión {} != {}",
            state.version, STATE_VERSION
        )
        .into());
    }
    Ok(state)
}

fn save_checkpoint(
    state_path: &Path,
    operator_path: &Path,
    state: &TrainingState,
    engine: &NativeMultiOperatorCore,
) -> Result<(), Box<dyn std::error::Error>> {
    engine.snapshot.save(operator_path)?;
    let body = serde_json::to_vec_pretty(state)?;
    atomic_write(state_path, &body)?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config {
        root: PathBuf::from(DEFAULT_ROOT),
        sleep_every: 4,
        save_every_sleeps: 1,
        concepts: 64,
        nodes_per_concept: 256,
        learning_rate: 0.18,
        dreams_per_sleep: 1,
        dream_learning_rate_scale: 0.35,
        max_cycles: None,
        seed: 0x1AF1_117E_5EED,
    };
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--root" => config.root = PathBuf::from(required(&mut args, "--root")?),
            "--sleep-every" => {
                config.sleep_every = required(&mut args, "--sleep-every")?.parse::<u64>()?.max(1)
            }
            "--save-every-sleeps" => {
                config.save_every_sleeps = required(&mut args, "--save-every-sleeps")?
                    .parse::<u64>()?
                    .max(1)
            }
            "--concepts" => {
                config.concepts = required(&mut args, "--concepts")?.parse::<usize>()?.max(1)
            }
            "--nodes-per-concept" => {
                config.nodes_per_concept = required(&mut args, "--nodes-per-concept")?
                    .parse::<usize>()?
                    .clamp(8, 8_192)
            }
            "--learning-rate" => {
                config.learning_rate = required(&mut args, "--learning-rate")?.parse::<f32>()?
            }
            "--dreams-per-sleep" => {
                config.dreams_per_sleep =
                    required(&mut args, "--dreams-per-sleep")?.parse::<usize>()?
            }
            "--dream-learning-rate-scale" => {
                config.dream_learning_rate_scale =
                    required(&mut args, "--dream-learning-rate-scale")?.parse::<f32>()?
            }
            "--max-cycles" => {
                config.max_cycles = Some(required(&mut args, "--max-cycles")?.parse::<u64>()?)
            }
            "--seed" => config.seed = required(&mut args, "--seed")?.parse::<u64>()?,
            "--help" | "-h" => {
                println!(
                    "Uso: native_phasor_infinite_trainer [--root DIR] [--sleep-every N] \
                     [--save-every-sleeps N] [--concepts N] [--nodes-per-concept N] \
                     [--learning-rate X] [--dreams-per-sleep N] \
                     [--dream-learning-rate-scale X] [--max-cycles N] [--seed N]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("argumento desconocido: {argument}").into()),
        }
    }
    if !(0.0..=1.0).contains(&config.learning_rate) {
        return Err("--learning-rate debe pertenecer a [0,1]".into());
    }
    if !(0.0..=1.0).contains(&config.dream_learning_rate_scale) {
        return Err("--dream-learning-rate-scale debe pertenecer a [0,1]".into());
    }
    Ok(config)
}

fn required(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("falta valor para {flag}").into())
}
