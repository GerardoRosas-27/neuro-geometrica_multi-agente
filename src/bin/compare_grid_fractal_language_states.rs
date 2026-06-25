use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

const GRID_STATE_PATH: &str = "data/snga_scaled_gemma_language.snga";
const FRACTAL_STATE_PATH: &str = "data/snga_scaled_gemma_language_fractal_compressed.snga";
const AGENT_COUNT: usize = 5_760;
const PATTERN_SIZE: usize = 12;
const TOP_K: usize = 32;

fn main() {
    println!("SNGA grid vs fractal compressed language comparison");

    let mut grid = SimplicialNetwork::grid_3d(config(), 2);
    match grid.load_persistent_state(GRID_STATE_PATH) {
        Ok(report) => println!(
            "grid_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("grid_loaded=false error={err}");
            return;
        }
    }

    let fractal_config = FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes: AGENT_COUNT,
        base_radius: 0.0,
        lateral_link_weight: 0.35,
        parent_link_weight: 1.0,
    };
    let mut fractal = SimplicialNetwork::fractal_3d(config(), fractal_config);
    match fractal.load_persistent_state(FRACTAL_STATE_PATH) {
        Ok(report) => println!(
            "fractal_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("fractal_loaded=false error={err}");
            return;
        }
    }

    let comparison = compare_predictions(&grid, &fractal);
    print_network_summary("grid_original", &grid);
    print_network_summary("fractal_compressed", &fractal);
    println!(
        "knowledge: cases={} grid_nonzero={} fractal_nonzero={} both_nonzero={} avg_grid_conf={:.3} avg_fractal_conf={:.3} avg_overlap={:.1}% exact_topk_matches={}/{}",
        comparison.cases,
        comparison.grid_nonzero,
        comparison.fractal_nonzero,
        comparison.both_nonzero,
        comparison.grid_confidence,
        comparison.fractal_confidence,
        comparison.average_overlap * 100.0,
        comparison.exact_matches,
        comparison.cases
    );

    let grid_energy = grid.total_free_energy();
    let fractal_energy = fractal.total_free_energy();
    println!(
        "optimization: energy_delta_fractal_minus_grid={:.1} energy_ratio_fractal_over_grid={:.3}",
        fractal_energy - grid_energy,
        fractal_energy / grid_energy.max(1.0)
    );
    println!(
        "lectura: {}",
        if comparison.same_knowledge() && fractal_energy <= grid_energy {
            "ambas versiones conservan el mismo conocimiento y la fractal es mas optima energeticamente"
        } else if comparison.same_knowledge() {
            "ambas versiones conservan el mismo conocimiento; la fractal comprimida gana en tamano/red pero la grilla aun tiene menor energia"
        } else {
            "la version fractal no conserva exactamente la misma senal predictiva que la grilla original"
        }
    );
}

struct Comparison {
    cases: usize,
    grid_nonzero: usize,
    fractal_nonzero: usize,
    both_nonzero: usize,
    exact_matches: usize,
    grid_confidence: f32,
    fractal_confidence: f32,
    average_overlap: f32,
}

impl Comparison {
    fn same_knowledge(&self) -> bool {
        self.grid_nonzero == self.cases
            && self.fractal_nonzero == self.cases
            && self.exact_matches == self.cases
    }
}

fn compare_predictions(grid: &SimplicialNetwork, fractal: &SimplicialNetwork) -> Comparison {
    let prompts = comparison_prompts();
    let mut comparison = Comparison {
        cases: prompts.len(),
        grid_nonzero: 0,
        fractal_nonzero: 0,
        both_nonzero: 0,
        exact_matches: 0,
        grid_confidence: 0.0,
        fractal_confidence: 0.0,
        average_overlap: 0.0,
    };

    for (prefix, prompt) in prompts {
        let grid_cue = pattern(prefix, prompt, grid.agents.len());
        let fractal_cue = pattern(prefix, prompt, fractal.agents.len());
        let grid_predicted = grid.predict_next_pattern(&grid_cue, 1, TOP_K);
        let fractal_predicted = fractal.predict_next_pattern(&fractal_cue, 1, TOP_K);
        let grid_ids = grid_predicted
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>();
        let fractal_ids = fractal_predicted
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>();
        let grid_conf = confidence(&grid_predicted);
        let fractal_conf = confidence(&fractal_predicted);
        let overlap = overlap_ratio(&grid_ids, &fractal_ids);

        if !grid_predicted.is_empty() {
            comparison.grid_nonzero += 1;
        }
        if !fractal_predicted.is_empty() {
            comparison.fractal_nonzero += 1;
        }
        if !grid_predicted.is_empty() && !fractal_predicted.is_empty() {
            comparison.both_nonzero += 1;
        }
        if grid_ids == fractal_ids {
            comparison.exact_matches += 1;
        }
        comparison.grid_confidence += grid_conf;
        comparison.fractal_confidence += fractal_conf;
        comparison.average_overlap += overlap;

        println!(
            "case {prefix}={prompt:?}: grid_pred={} grid_conf={:.3} fractal_pred={} fractal_conf={:.3} overlap={:.1}% exact={}",
            grid_predicted.len(),
            grid_conf,
            fractal_predicted.len(),
            fractal_conf,
            overlap * 100.0,
            grid_ids == fractal_ids
        );
    }

    let cases = comparison.cases.max(1) as f32;
    comparison.grid_confidence /= cases;
    comparison.fractal_confidence /= cases;
    comparison.average_overlap /= cases;
    comparison
}

fn print_network_summary(label: &str, network: &SimplicialNetwork) {
    let stats = network.plasticity_stats();
    println!(
        "{label}: nodes={} edges={} associative={} consolidated={} causal={} energy={:.1}",
        network.agents.len(),
        stats.active_edges,
        stats.associative_edges,
        stats.consolidated_edges,
        stats.causal_edges,
        network.total_free_energy()
    );
}

fn confidence(predicted: &[(usize, f32)]) -> f32 {
    predicted.iter().map(|(_, score)| *score).sum::<f32>() / TOP_K as f32
}

fn overlap_ratio(left: &[usize], right: &[usize]) -> f32 {
    let right = right.iter().copied().collect::<HashSet<_>>();
    let hits = left.iter().filter(|idx| right.contains(idx)).count();
    hits as f32 / left.len().max(right.len()).max(1) as f32
}

fn comparison_prompts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("topic", "lenguaje"),
        ("topic", "concepto"),
        ("topic", "causalidad"),
        ("topic", "memoria"),
        ("topic", "planificacion"),
        ("topic", "mundo"),
        ("topic", "herramienta"),
        ("topic", "emocion"),
        ("topic", "sociedad"),
        ("topic", "fisica simple"),
        ("topic", "objeto"),
        ("topic", "categoria"),
        ("question", "que es una palabra"),
        ("question", "como se agrupan ideas"),
        ("question", "que hace una causa"),
        ("question", "para que sirve un plan"),
        ("question", "como cambia un objeto"),
        ("question", "que es una emocion"),
        ("relation", "simbolo-significado"),
        ("relation", "rasgo-categoria"),
        ("relation", "causa-efecto"),
        ("relation", "objetivo-ruta"),
        ("relation", "pregunta-respuesta"),
    ]
}

fn pattern(prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = DefaultHasher::new();
            prefix.hash(&mut hasher);
            value.hash(&mut hasher);
            offset.hash(&mut hasher);
            hasher.finish() as usize % nodes
        })
        .collect()
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
