use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::hash::{Hash, Hasher};

const DEFAULT_STATE_PATH: &str = "data/snga_fractal_multiregion_correlations.snga";
const BASE_STATE_PATH: &str = "data/snga_fractal_multiregion_substrate_linguistic_compressed.snga";
const REGION_SIZE: usize = 11_520;
const REGION_COUNT: usize = 8;
const PATTERN_SIZE: usize = 12;
const TOP_K: usize = 64;

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
struct EvalStats {
    total: usize,
    a_to_c_hits: usize,
    intent_to_c_hits: usize,
    cue_to_c_hits: usize,
    a_to_c_overlap: f32,
    intent_to_c_overlap: f32,
    cue_to_c_overlap: f32,
    a_to_c_conf: f32,
    intent_to_c_conf: f32,
    cue_to_c_conf: f32,
}

fn main() {
    println!("SNGA multiregion correlation probe");
    let state =
        env::var("SNGA_CORRELATION_STATE").unwrap_or_else(|_| DEFAULT_STATE_PATH.to_string());

    let mut trained = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    match trained.load_persistent_state(&state) {
        Ok(report) => println!(
            "trained_loaded=true path={} agents={} edges={} causal_edges={}",
            state, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("trained_loaded=false error={err}");
            return;
        }
    }

    let mut baseline = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config());
    match baseline.load_persistent_state(BASE_STATE_PATH) {
        Ok(report) => println!(
            "baseline_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => println!("baseline_loaded=false error={err}"),
    }

    let trained_stats = evaluate("trained", &trained);
    let baseline_stats = evaluate("baseline", &baseline);
    print_summary("trained", trained_stats);
    print_summary("baseline", baseline_stats);
    print_network("trained", &trained);
    print_network("baseline", &baseline);
}

fn evaluate(label: &str, network: &SimplicialNetwork) -> EvalStats {
    let mut stats = EvalStats::default();
    for chain in chains() {
        let cue = linguistic_pattern(chain.cue, network.agents.len());
        let intent = intent_cue_pattern(chain.label, chain.cue, network.agents.len());
        let a = event_pattern("a", chain.a, network.agents.len());
        let b = event_pattern("b", chain.b, network.agents.len());
        let c = event_pattern("c", chain.c, network.agents.len());

        let a_to_c = network.infer_transitive_from(&a, 2, TOP_K);
        let intent_to_c = network.infer_transitive_from(&intent, 4, 512);
        let b_to_c = network.infer_transitive_from(&b, 1, TOP_K);
        let cue_to_c = network.infer_transitive_from(&cue, 4, 512);
        let a_ids = a_to_c.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let intent_ids = intent_to_c.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let cue_ids = cue_to_c.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
        let a_overlap = overlap_ratio(&a_ids, &c);
        let intent_overlap = overlap_ratio(&intent_ids, &c);
        let cue_overlap = overlap_ratio(&cue_ids, &c);
        let a_conf = confidence(&a_to_c);
        let intent_conf = confidence(&intent_to_c);
        let _b_conf = confidence(&b_to_c);
        let cue_conf = confidence(&cue_to_c);

        stats.total += 1;
        stats.a_to_c_hits += usize::from(a_overlap > 0.0);
        stats.intent_to_c_hits += usize::from(intent_overlap > 0.0);
        stats.cue_to_c_hits += usize::from(cue_overlap > 0.0);
        stats.a_to_c_overlap += a_overlap;
        stats.intent_to_c_overlap += intent_overlap;
        stats.cue_to_c_overlap += cue_overlap;
        stats.a_to_c_conf += a_conf;
        stats.intent_to_c_conf += intent_conf;
        stats.cue_to_c_conf += cue_conf;

        println!(
            "case {label} label={:?} A={:?} C={:?} a_to_c_overlap={:.1}% intent_to_c_overlap={:.1}% cue_to_c_overlap={:.1}% a_conf={:.3} intent_conf={:.3} cue_conf={:.3}",
            chain.label,
            chain.a,
            chain.c,
            a_overlap * 100.0,
            intent_overlap * 100.0,
            cue_overlap * 100.0,
            a_conf,
            intent_conf,
            cue_conf
        );
    }

    if stats.total > 0 {
        let total = stats.total as f32;
        stats.a_to_c_overlap /= total;
        stats.intent_to_c_overlap /= total;
        stats.cue_to_c_overlap /= total;
        stats.a_to_c_conf /= total;
        stats.intent_to_c_conf /= total;
        stats.cue_to_c_conf /= total;
    }
    stats
}

fn print_summary(label: &str, stats: EvalStats) {
    println!(
        "{label}_summary: a_to_c_hits={}/{} intent_to_c_hits={}/{} cue_to_c_hits={}/{} a_to_c_overlap={:.1}% intent_to_c_overlap={:.1}% cue_to_c_overlap={:.1}% a_conf={:.3} intent_conf={:.3} cue_conf={:.3}",
        stats.a_to_c_hits,
        stats.total,
        stats.intent_to_c_hits,
        stats.total,
        stats.cue_to_c_hits,
        stats.total,
        stats.a_to_c_overlap * 100.0,
        stats.intent_to_c_overlap * 100.0,
        stats.cue_to_c_overlap * 100.0,
        stats.a_to_c_conf,
        stats.intent_to_c_conf,
        stats.cue_to_c_conf
    );
}

fn print_network(label: &str, network: &SimplicialNetwork) {
    let stats = network.plasticity_stats();
    println!(
        "{label}_network: nodes={} edges={} associative={} consolidated={} causal={} energy={:.1}",
        network.agents.len(),
        stats.active_edges,
        stats.associative_edges,
        stats.consolidated_edges,
        stats.causal_edges,
        network.total_free_energy()
    );
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

fn confidence(predicted: &[(usize, f32)]) -> f32 {
    predicted.iter().map(|(_, score)| *score).sum::<f32>() / TOP_K as f32
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
