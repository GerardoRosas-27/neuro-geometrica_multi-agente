//! Gemma 2 → OperatorRecipe → solver sparse L0/QUBO/L1 → CDT.

use candle_core::quantized::gguf_file;
use candle_core::Device;
use cdt_rqm_epr::gemma_operator_bridge::{
    compile_simple_qubo_expression, generate_operator_recipe, GemmaRecipeGenerationConfig,
};
use cdt_rqm_epr::native_gemma2::{resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2};
use cdt_rqm_epr::native_multi_operator_core::{
    NativeMultiOperatorCore, OperatorDeltaSnapshot, OperatorRecipe, OperatorSolution,
};
use cdt_rqm_epr::native_phasor_thermodynamic_engine::{
    DEFAULT_PHASOR_NODES_PER_SLICE, DEFAULT_PHASOR_STARTUP_NODES, DEFAULT_PHASOR_STARTUP_SLICES,
};
use cdt_rqm_epr::native_thermodynamic_cdt::{NativeThermoCdtConfig, NativeThermoCdtSubstrate};
use std::env;
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_SNAPSHOT: &str = "data/native_multi_operator/operator_deltas.json";

#[derive(Debug)]
struct Config {
    prompt: Option<String>,
    recipe: Option<PathBuf>,
    model: Option<PathBuf>,
    snapshot: PathBuf,
    nodes: usize,
    max_tokens: usize,
    learning_rate: f32,
    no_consolidate: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let recipe = if let Some(path) = &config.recipe {
        let body = fs::read(path)?;
        let recipe: OperatorRecipe = serde_json::from_slice(&body)?;
        recipe.validate()?;
        recipe
    } else {
        let prompt = config
            .prompt
            .as_deref()
            .ok_or("se requiere --prompt TEXTO o --recipe ARCHIVO.json")?;
        if let Some(recipe) = compile_simple_qubo_expression(prompt) {
            eprintln!("receta compilada sin invocar Gemma (origen=DeterministicQuboFallback)");
            recipe
        } else {
            let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
            eprintln!("cargando Gemma 2 desde {}", model_path.display());
            let device = Device::Cpu;
            let mut file = File::open(model_path)?;
            let content = gguf_file::Content::read(&mut file)?;
            let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
            let mut model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
            let started = Instant::now();
            let generated = generate_operator_recipe(
                &mut model,
                &tokenizer,
                prompt,
                &device,
                GemmaRecipeGenerationConfig {
                    max_tokens: config.max_tokens,
                    ..GemmaRecipeGenerationConfig::default()
                },
            )?;
            eprintln!(
                "receta generada en {:.2}s ({} bytes, origen={:?})",
                started.elapsed().as_secs_f64(),
                generated.raw_model_output.len(),
                generated.origin
            );
            generated.recipe
        }
    };

    let mut global_core = build_global_core(config.nodes);
    let mut engine = NativeMultiOperatorCore::default();
    if config.snapshot.exists() {
        engine.snapshot = OperatorDeltaSnapshot::load(&config.snapshot)?;
        engine.snapshot.apply_to_core(&mut global_core, 1.0)?;
        eprintln!(
            "snapshot restaurado: recetas={} nodos_delta={} aristas_delta={}",
            engine.snapshot.accepted_recipes.len(),
            engine.snapshot.node_deltas.len(),
            engine.snapshot.edge_deltas.len()
        );
    }

    println!("{}", serde_json::to_string_pretty(&recipe)?);
    let started = Instant::now();
    let solved = if config.no_consolidate {
        engine.solve(&recipe, &global_core)?
    } else {
        engine.solve_and_consolidate(&recipe, &mut global_core, config.learning_rate)?
    };
    print_solution(&solved.solution, started.elapsed().as_secs_f64());
    println!(
        "operador={:?} working_set={}/{} verificado={}",
        solved.operator,
        solved.working_set.global_nodes.len(),
        global_core.node_count(),
        solved.solution.verified()
    );

    if !config.no_consolidate {
        engine.snapshot.save(&config.snapshot)?;
        println!("snapshot={}", config.snapshot.display());
    }
    Ok(())
}

fn build_global_core(nodes: usize) -> NativeThermoCdtSubstrate {
    let (slices, nodes_per_slice) = if nodes == DEFAULT_PHASOR_STARTUP_NODES {
        (
            DEFAULT_PHASOR_STARTUP_SLICES,
            DEFAULT_PHASOR_NODES_PER_SLICE,
        )
    } else {
        (1, nodes.max(1))
    };
    NativeThermoCdtSubstrate::new(NativeThermoCdtConfig {
        slices,
        nodes_per_slice,
        temperature: 0.0,
        seed: 0x0FEB_A70A_C07E,
        ..NativeThermoCdtConfig::default()
    })
}

fn print_solution(solution: &OperatorSolution, elapsed_seconds: f64) {
    match solution {
        OperatorSolution::L0(solution) => println!(
            "L0 F={:.6}->{:.6} residuo={:.3e} tiempo={elapsed_seconds:.4}s",
            solution.initial_energy, solution.final_energy, solution.residual
        ),
        OperatorSolution::Qubo(solution) => println!(
            "QUBO energía={:.6} bits={:?} starts={} exacto={} óptimo_local={} \
             tiempo={elapsed_seconds:.4}s",
            solution.energy, solution.bits, solution.starts, solution.exact, solution.local_optimum
        ),
        OperatorSolution::L1(solution) => println!(
            "L1 aristas={} residuo={:.3e} iteraciones={} tiempo={elapsed_seconds:.4}s",
            solution.edge_flows.len(),
            solution.residual,
            solution.iterations
        ),
    }
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config {
        prompt: None,
        recipe: None,
        model: None,
        snapshot: PathBuf::from(DEFAULT_SNAPSHOT),
        nodes: DEFAULT_PHASOR_STARTUP_NODES,
        max_tokens: 384,
        learning_rate: 0.18,
        no_consolidate: false,
    };
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--prompt" => config.prompt = Some(required(&mut args, "--prompt")?),
            "--recipe" => config.recipe = Some(PathBuf::from(required(&mut args, "--recipe")?)),
            "--model" => config.model = Some(PathBuf::from(required(&mut args, "--model")?)),
            "--snapshot" => config.snapshot = PathBuf::from(required(&mut args, "--snapshot")?),
            "--nodes" => config.nodes = required(&mut args, "--nodes")?.parse::<usize>()?.max(1),
            "--max-tokens" => {
                config.max_tokens = required(&mut args, "--max-tokens")?
                    .parse::<usize>()?
                    .max(1)
            }
            "--learning-rate" => {
                config.learning_rate = required(&mut args, "--learning-rate")?.parse::<f32>()?
            }
            "--no-consolidate" => config.no_consolidate = true,
            "--help" | "-h" => {
                println!(
                    "Uso: native_gemma2_multi_operator (--prompt TEXTO | --recipe JSON) \
                     [--model GGUF] [--snapshot RUTA] [--nodes N] \
                     [--max-tokens N] [--learning-rate X] [--no-consolidate]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("argumento desconocido: {argument}").into()),
        }
    }
    if !(0.0..=1.0).contains(&config.learning_rate) {
        return Err("--learning-rate debe pertenecer a [0,1]".into());
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
