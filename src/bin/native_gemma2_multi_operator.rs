//! Gemma 2 → OperatorRecipe → solver sparse L0/QUBO/L1 → CDT.

use candle_core::quantized::gguf_file;
use cdt_rqm_epr::gemma_operator_bridge::{
    compile_simple_qubo_expression, generate_operator_recipe_with_memory,
    generate_solution_explanation, GemmaRecipeGenerationConfig,
};
use cdt_rqm_epr::native_cognitive_closed_loop::{
    episode_from_solution, memory_context, record_episode, retrieve_episodes, summarize_solution,
};
use cdt_rqm_epr::native_gemma2::{
    resolve_gemma2_device, resolve_gemma2_model_path, Gemma2Tokenizer, QuantizedGemma2,
};
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
    feedback_tokens: usize,
    memory_limit: usize,
    learning_rate: f32,
    no_consolidate: bool,
    no_gemma_feedback: bool,
    device: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
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
    let problem = config
        .prompt
        .clone()
        .unwrap_or_else(|| "resolver la receta estructurada proporcionada".to_string());
    let memories = retrieve_episodes(&mut engine.snapshot, &problem, config.memory_limit);
    if !memories.is_empty() {
        eprintln!(
            "memoria_recuperada={} contexto:\n{}",
            memories.len(),
            memory_context(&memories)
        );
    }
    let deterministic_recipe = config
        .prompt
        .as_deref()
        .and_then(compile_simple_qubo_expression);
    let needs_model =
        config.prompt.is_some() && (deterministic_recipe.is_none() || !config.no_gemma_feedback);
    let device = resolve_gemma2_device(&config.device)?;
    let mut gemma = if needs_model {
        let model_path = resolve_gemma2_model_path(config.model.as_deref())?;
        eprintln!("cargando Gemma 2 desde {}", model_path.display());
        let mut file = File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let tokenizer = Gemma2Tokenizer::from_gguf(&content)?;
        let model = QuantizedGemma2::from_gguf(content, &mut file, &device)?;
        Some((model, tokenizer))
    } else {
        None
    };
    let recipe = if let Some(path) = &config.recipe {
        let body = fs::read(path)?;
        let recipe: OperatorRecipe = serde_json::from_slice(&body)?;
        recipe.validate()?;
        recipe
    } else if let Some(recipe) = deterministic_recipe {
        eprintln!("receta compilada sin invocar Gemma (origen=DeterministicQuboFallback)");
        recipe
    } else {
        let (model, tokenizer) = gemma
            .as_mut()
            .ok_or("Gemma es necesaria para compilar esta tarea")?;
        let started = Instant::now();
        let generated = generate_operator_recipe_with_memory(
            model,
            tokenizer,
            &problem,
            &memories,
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
    };

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

    let deterministic_feedback = summarize_solution(&solved);
    let explanation = if !config.no_gemma_feedback {
        if let Some((model, tokenizer)) = gemma.as_mut() {
            match generate_solution_explanation(
                model,
                tokenizer,
                &problem,
                &recipe,
                &solved,
                &memories,
                &device,
                GemmaRecipeGenerationConfig {
                    max_tokens: config.feedback_tokens,
                    temperature: 0.05,
                    ..GemmaRecipeGenerationConfig::default()
                },
            ) {
                Ok(explanation) if !explanation.is_empty() => explanation,
                Ok(_) => deterministic_feedback.clone(),
                Err(error) => {
                    eprintln!("feedback Gemma falló; usando resumen verificable: {error}");
                    deterministic_feedback.clone()
                }
            }
        } else {
            deterministic_feedback.clone()
        }
    } else {
        deterministic_feedback.clone()
    };
    println!("resultado_verificado:\n{deterministic_feedback}");
    if explanation != deterministic_feedback {
        println!("interpretacion_gemma_validada:\n{explanation}");
    }

    if !config.no_consolidate {
        if solved.solution.verified() {
            record_episode(
                &mut engine.snapshot,
                episode_from_solution(&problem, &recipe, &solved),
            );
        }
        engine.snapshot.save(&config.snapshot)?;
        println!(
            "snapshot={} episodios={}",
            config.snapshot.display(),
            engine.snapshot.episodes.len()
        );
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
        feedback_tokens: 256,
        memory_limit: 3,
        learning_rate: 0.18,
        no_consolidate: false,
        no_gemma_feedback: false,
        device: "cpu".to_string(),
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
            "--feedback-tokens" => {
                config.feedback_tokens = required(&mut args, "--feedback-tokens")?
                    .parse::<usize>()?
                    .max(1)
            }
            "--memory-limit" => {
                config.memory_limit = required(&mut args, "--memory-limit")?.parse::<usize>()?
            }
            "--learning-rate" => {
                config.learning_rate = required(&mut args, "--learning-rate")?.parse::<f32>()?
            }
            "--no-consolidate" => config.no_consolidate = true,
            "--no-gemma-feedback" => config.no_gemma_feedback = true,
            "--device" => config.device = required(&mut args, "--device")?,
            "--help" | "-h" => {
                println!(
                    "Uso: native_gemma2_multi_operator (--prompt TEXTO | --recipe JSON) \
                     [--model GGUF] [--device cpu|cuda:N] [--snapshot RUTA] [--nodes N] \
                     [--max-tokens N] [--feedback-tokens N] [--memory-limit N] \
                     [--learning-rate X] [--no-consolidate] [--no-gemma-feedback]"
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
