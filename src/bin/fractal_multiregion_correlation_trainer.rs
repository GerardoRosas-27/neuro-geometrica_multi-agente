use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::thread;
use std::time::Duration;

const DEFAULT_BASE_STATE: &str =
    "data/snga_fractal_multiregion_substrate_linguistic_compressed.snga";
const DEFAULT_STATE_PATH: &str = "data/snga_fractal_multiregion_correlations.snga";
const DEFAULT_PROGRESS_PATH: &str = "data/snga_fractal_multiregion_correlations.progress";
const REGION_SIZE: usize = 11_520;
const REGION_COUNT: usize = 8;
const PATTERN_SIZE: usize = 12;

#[derive(Clone, Copy)]
enum BrainRegion {
    Linguistic = 0,
    Visual = 1,
    Auditory = 2,
    Motor = 3,
    Parietal = 4,
    Hippocampal = 5,
    Prefrontal = 6,
    BasalGanglia = 7,
}

#[derive(Clone, Copy)]
struct CorrelationChain {
    label: &'static str,
    cue: &'static str,
    a: &'static str,
    b: &'static str,
    c: &'static str,
}

#[derive(Default)]
struct Progress {
    batches: usize,
    lessons: usize,
}

fn main() {
    let batch_size = arg_value("--batch-size")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32)
        .max(1);
    let max_batches = arg_value("--batches").and_then(|value| value.parse::<usize>().ok());
    let sleep_ms = arg_value("--sleep-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let relax_every = arg_value("--relax-every")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5);
    let base_state = arg_value("--base").unwrap_or_else(|| DEFAULT_BASE_STATE.to_string());
    let state_path = arg_value("--state").unwrap_or_else(|| DEFAULT_STATE_PATH.to_string());
    let progress_path =
        arg_value("--progress").unwrap_or_else(|| DEFAULT_PROGRESS_PATH.to_string());

    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    let loaded = if Path::new(&state_path).exists() {
        network.load_persistent_state(&state_path).is_ok()
    } else {
        network.load_persistent_state(&base_state).is_ok()
    };
    network.enable_neural_oscillations();
    let mut progress = load_progress(&progress_path).unwrap_or_default();

    println!("SNGA multiregion correlation trainer");
    println!(
        "loaded={} base={} state={} progress={} batch_size={} batches={:?}",
        loaded, base_state, state_path, progress_path, batch_size, max_batches
    );

    loop {
        if let Some(limit) = max_batches {
            if progress.batches >= limit {
                break;
            }
        }

        for idx in 0..batch_size {
            let chain = chains()[(progress.lessons + idx) % chains().len()];
            train_chain(&mut network, chain);
        }
        progress.batches += 1;
        progress.lessons += batch_size;

        let eval = evaluate(&network);
        if relax_every > 0 && progress.batches % relax_every == 0 {
            network.anneal_active_edge_rest_lengths(0.12, 1.05);
        }
        save_all(&network, &progress, &state_path, &progress_path, "batch");
        let stats = network.plasticity_stats();
        println!(
            "batch={} lessons={} transitive_hits={}/{} cue_hits={}/{} conf={:.3} edges={} assoc={} causal={} energy={:.1}",
            progress.batches,
            progress.lessons,
            eval.transitive_hits,
            eval.total,
            eval.cue_hits,
            eval.total,
            eval.confidence,
            stats.active_edges,
            stats.associative_edges,
            stats.causal_edges,
            network.total_free_energy()
        );

        if sleep_ms > 0 {
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    save_all(&network, &progress, &state_path, &progress_path, "final");
}

fn train_chain(network: &mut SimplicialNetwork, chain: CorrelationChain) {
    let cue = intent_cue_pattern(chain.label, chain.cue, network.agents.len());
    let language = linguistic_pattern(chain.cue, network.agents.len());
    let goal = goal_pattern(chain.label, chain.cue, network.agents.len());
    let context = context_pattern(chain.label, chain.a, network.agents.len());
    let a = event_pattern("a", chain.a, network.agents.len());
    let b = event_pattern("b", chain.b, network.agents.len());
    let c = event_pattern("c", chain.c, network.agents.len());
    let label = relation_pattern(chain.label, network.agents.len());

    // Puente lingüístico-control: la frase amplia se comprime en una intención
    // prefrontal limpia. No entrenamos language/cue -> C.
    network.learn_transition(&language, &cue);
    for _ in 0..3 {
        network.learn_transition(&cue, &goal);
        network.learn_transition(&goal, &context);
        network.learn_transition(&context, &a);
    }
    for _ in 0..2 {
        network.learn_transition(&cue, &a);
    }

    // Dinámica relacional interna: A -> B -> C. No entrenamos A -> C.
    network.learn_transition(&a, &b);
    network.learn_transition(&b, &c);
    network.learn_transition(&label, &a);

    reinforce_local(network, &language, &cue, 0.035);
    reinforce_bridge(network, &cue, &goal, &a, 0.065);
    reinforce_local(network, &goal, &context, 0.055);
    reinforce_local(network, &context, &a, 0.06);
    reinforce_local(network, &a, &b, 0.045);
    reinforce_local(network, &b, &c, 0.045);
}

fn reinforce_local(network: &mut SimplicialNetwork, left: &[usize], right: &[usize], lr: f32) {
    let mut fused = Vec::new();
    fused.extend(left.iter().copied());
    fused.extend(right.iter().copied());
    fused.sort_unstable();
    fused.dedup();
    network.reinforce_coactivation_if_useful(&fused, lr, 0.9);
}

fn reinforce_bridge(
    network: &mut SimplicialNetwork,
    cue: &[usize],
    goal: &[usize],
    a: &[usize],
    lr: f32,
) {
    let mut fused = Vec::new();
    fused.extend(cue.iter().copied());
    fused.extend(goal.iter().copied());
    fused.extend(a.iter().copied());
    fused.sort_unstable();
    fused.dedup();
    network.reinforce_coactivation_if_useful(&fused, lr, 0.9);
}

#[derive(Default)]
struct EvalReport {
    total: usize,
    transitive_hits: usize,
    cue_hits: usize,
    confidence: f32,
}

fn evaluate(network: &SimplicialNetwork) -> EvalReport {
    let mut report = EvalReport::default();
    for chain in chains() {
        let cue = intent_cue_pattern(chain.label, chain.cue, network.agents.len());
        let a = event_pattern("a", chain.a, network.agents.len());
        let c = event_pattern("c", chain.c, network.agents.len());

        let transitive = network.infer_transitive_from(&a, 2, 64);
        let cue_prediction = network.infer_transitive_from(&cue, 4, 512);
        let transitive_ids = transitive.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let cue_ids = cue_prediction
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>();

        report.total += 1;
        report.transitive_hits += usize::from(overlap_ratio(&transitive_ids, &c) > 0.0);
        report.cue_hits += usize::from(overlap_ratio(&cue_ids, &c) > 0.0);
        report.confidence += transitive.iter().map(|(_, score)| *score).sum::<f32>() / 64.0;
    }
    if report.total > 0 {
        report.confidence /= report.total as f32;
    }
    report
}

fn chains() -> &'static [CorrelationChain] {
    &[
        CorrelationChain {
            label: "objeto cae",
            cue: "objeto cae repetidamente",
            a: "objeto queda sin soporte",
            b: "objeto desciende",
            c: "objeto toca el suelo",
        },
        CorrelationChain {
            label: "tomar taza roja",
            cue: "agarra la taza roja",
            a: "visual taza roja presente",
            b: "parietal taza alcanzable",
            c: "motor preparar agarre",
        },
        CorrelationChain {
            label: "buscar taza correcta",
            cue: "no agarres la taza azul",
            a: "visual taza azul presente",
            b: "prefrontal no coincide objetivo",
            c: "basal ganglia inhibir agarre",
        },
        CorrelationChain {
            label: "causa lluvia",
            cue: "si llueve el suelo se moja",
            a: "lluvia ocurre",
            b: "agua cae al suelo",
            c: "suelo queda mojado",
        },
        CorrelationChain {
            label: "familia abuelo",
            cue: "juan ana luis familia",
            a: "juan padre de ana",
            b: "ana madre de luis",
            c: "juan relacionado con luis",
        },
        CorrelationChain {
            label: "semilla crece",
            cue: "semilla con agua crece",
            a: "semilla recibe agua",
            b: "semilla germina",
            c: "planta empieza a crecer",
        },
        CorrelationChain {
            label: "fuego vapor",
            cue: "fuego calienta agua",
            a: "fuego calienta recipiente",
            b: "agua alcanza hervor",
            c: "vapor aparece",
        },
        CorrelationChain {
            label: "accion por objetivo",
            cue: "objetivo mover mano",
            a: "prefrontal mantiene objetivo",
            b: "basal ganglia selecciona ruta",
            c: "motor ejecuta movimiento",
        },
        CorrelationChain {
            label: "alarma sonora",
            cue: "escucha alarma y atiende",
            a: "auditivo alarma suena",
            b: "hippocampal evento relevante",
            c: "prefrontal dirige atencion",
        },
    ]
}

fn linguistic_pattern(text: &str, nodes: usize) -> Vec<usize> {
    regional_pattern(BrainRegion::Linguistic, "linguistic", text, nodes)
}

fn intent_cue_pattern(label: &str, cue: &str, nodes: usize) -> Vec<usize> {
    regional_pattern(
        BrainRegion::Prefrontal,
        "intent_cue",
        &format!("{label}_{cue}"),
        nodes,
    )
}

fn goal_pattern(label: &str, cue: &str, nodes: usize) -> Vec<usize> {
    regional_pattern(
        BrainRegion::Prefrontal,
        "goal",
        &format!("{label}_{cue}"),
        nodes,
    )
}

fn context_pattern(label: &str, event: &str, nodes: usize) -> Vec<usize> {
    regional_pattern(
        BrainRegion::Hippocampal,
        "context",
        &format!("{label}_{event}"),
        nodes,
    )
}

fn event_pattern(role: &str, text: &str, nodes: usize) -> Vec<usize> {
    let region = if text.contains("visual") {
        BrainRegion::Visual
    } else if text.contains("auditivo") || text.contains("sonido") || text.contains("alarma") {
        BrainRegion::Auditory
    } else if text.contains("parietal") || text.contains("alcanzable") {
        BrainRegion::Parietal
    } else if text.contains("motor") || text.contains("agarre") || text.contains("movimiento") {
        BrainRegion::Motor
    } else if text.contains("basal") || text.contains("inhibir") {
        BrainRegion::BasalGanglia
    } else if text.contains("prefrontal") || text.contains("objetivo") {
        BrainRegion::Prefrontal
    } else {
        BrainRegion::Hippocampal
    };
    regional_pattern(region, role, text, nodes)
}

fn relation_pattern(text: &str, nodes: usize) -> Vec<usize> {
    regional_pattern(BrainRegion::Prefrontal, "relation", text, nodes)
}

fn regional_pattern(region: BrainRegion, prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    let start = region as usize * REGION_SIZE;
    let len = REGION_SIZE.min(nodes.saturating_sub(start)).max(1);
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            prefix.hash(&mut hasher);
            normalize_text(value).hash(&mut hasher);
            offset.hash(&mut hasher);
            start + (hasher.finish() as usize % len)
        })
        .collect()
}

fn overlap_ratio(left: &[usize], right: &[usize]) -> f32 {
    let hits = left.iter().filter(|idx| right.contains(idx)).count();
    hits as f32 / right.len().max(1) as f32
}

fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| match ch {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            other => other,
        })
        .filter(|ch| ch.is_ascii_alphanumeric() || ch.is_whitespace())
        .collect()
}

fn save_all(
    network: &SimplicialNetwork,
    progress: &Progress,
    state: &str,
    progress_path: &str,
    label: &str,
) {
    match network.save_persistent_state(state) {
        Ok(report) => {
            if let Err(err) = save_progress(progress_path, progress) {
                eprintln!("{label}: estado guardado, progreso fallo: {err}");
            }
            println!(
                "{label}: saved agents={} edges={} causal={} batches={} lessons={}",
                report.agents,
                report.edges,
                report.causal_edges,
                progress.batches,
                progress.lessons
            );
        }
        Err(err) => eprintln!("{label}: fallo guardando: {err}"),
    }
}

fn load_progress(path: &str) -> Option<Progress> {
    let text = fs::read_to_string(path).ok()?;
    let mut progress = Progress::default();
    for line in text.lines() {
        let (key, value) = line.split_once('=')?;
        match key {
            "batches" => progress.batches = value.parse().ok()?,
            "lessons" => progress.lessons = value.parse().ok()?,
            _ => {}
        }
    }
    Some(progress)
}

fn save_progress(path: &str, progress: &Progress) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
        format!(
            "batches={}\nlessons={}\n",
            progress.batches, progress.lessons
        ),
    )
}

fn fractal_mesh_config() -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: REGION_SIZE * REGION_COUNT,
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    }
}

fn config() -> SimplicialConfig {
    SimplicialConfig {
        width: 72,
        height: 40,
        spacing: 6.5,
        elasticity: 0.005,
        damping: 0.86,
        activation_threshold: 0.64,
        simplex_area_weight: 0.00012,
        max_active_agents: 384,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 16,
        rhythm_amplitude: 0.04,
        forgetting_rate: 0.0,
        prune_below_weight: 0.02,
        consolidate_after: 3,
        consolidated_forgetting_scale: 0.1,
        max_episodes: 2048,
        replay_interval: 8,
        replay_batch: 12,
        replay_learning_rate: 0.05,
        causal_learning_rate: 0.18,
        contradiction_learning_rate: 0.2,
        contradiction_energy_weight: 1.0,
        simplex3_weight: 0.00008,
        hyperbolic_curvature: 0.0,
        seed: 401,
    }
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}
