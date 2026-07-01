use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;

const DEFAULT_INPUT_STATE: &str = "data/snga_fractal_gemma_spanish_curriculum_expanded.snga";
const DEFAULT_OUTPUT_STATE: &str = "data/snga_fractal_multiregion_substrate.snga";
const DEFAULT_REGION_SIZE: usize = 11_520;
const REGION_COUNT: usize = 8;
const BRIDGE_HUBS: usize = 16;

#[derive(Clone, Copy)]
struct BrainRegion {
    name: &'static str,
    description: &'static str,
}

const REGIONS: [BrainRegion; REGION_COUNT] = [
    BrainRegion {
        name: "linguistic",
        description: "simbolos, letras, silabas, palabras, frases y oraciones ya entrenadas",
    },
    BrainRegion {
        name: "visual",
        description: "rasgos visuales futuros: forma, color, textura, objetos",
    },
    BrainRegion {
        name: "auditory",
        description: "rasgos sonoros futuros: fonemas, timbre, ritmo, fuentes",
    },
    BrainRegion {
        name: "motor",
        description: "acciones posibles futuras y programas motores",
    },
    BrainRegion {
        name: "parietal",
        description: "espacio, orden, cantidad, comparacion y ubicacion",
    },
    BrainRegion {
        name: "hippocampal",
        description: "episodios, mapas relacionales rapidos y contexto",
    },
    BrainRegion {
        name: "prefrontal",
        description: "objetivos, reglas activas, pasos intermedios y plan temporal",
    },
    BrainRegion {
        name: "basal_ganglia",
        description: "seleccion, gating e inhibicion de rutas de accion",
    },
];

fn main() {
    let input = env::var("SNGA_MULTI_INPUT").unwrap_or_else(|_| DEFAULT_INPUT_STATE.to_string());
    let output = env::var("SNGA_MULTI_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT_STATE.to_string());
    let region_size = env::var("SNGA_REGION_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REGION_SIZE)
        .max(DEFAULT_REGION_SIZE);
    let total_nodes = region_size * REGION_COUNT;

    println!("SNGA fractal multiregion substrate");
    println!("input={input} output={output} region_size={region_size} total_nodes={total_nodes}");

    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes));
    match network.load_persistent_memory_state(&input) {
        Ok(report) => println!(
            "linguistic_memory_loaded=true agents={} edges={} causal_edges={}",
            report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("linguistic_memory_loaded=false error={err}");
            return;
        }
    }

    add_region_bridges(&mut network, region_size);
    let adjusted = network.anneal_active_edge_rest_lengths(1.0, 0.0);

    match network.save_persistent_state(&output) {
        Ok(report) => println!(
            "saved=true agents={} edges={} causal_edges={} adjusted={} energy={:.1}",
            report.agents,
            report.edges,
            report.causal_edges,
            adjusted,
            network.total_free_energy()
        ),
        Err(err) => {
            println!("saved=false error={err}");
            return;
        }
    }

    let region_path = output.replace(".snga", ".regions.txt");
    if let Err(err) = fs::write(&region_path, region_manifest(region_size)) {
        println!("regions_saved=false error={err}");
    } else {
        println!("regions_saved=true path={region_path}");
    }
}

fn add_region_bridges(network: &mut SimplicialNetwork, region_size: usize) {
    let bridges = [
        ("visual", "parietal"),
        ("parietal", "prefrontal"),
        ("prefrontal", "motor"),
        ("linguistic", "prefrontal"),
        ("hippocampal", "prefrontal"),
        ("prefrontal", "basal_ganglia"),
        ("basal_ganglia", "motor"),
        ("prefrontal", "linguistic"),
        ("parietal", "hippocampal"),
    ];

    for (source, target) in bridges {
        let source_idx = region_index(source);
        let target_idx = region_index(target);
        let source_pattern = region_hubs(source_idx, region_size);
        let target_pattern = region_hubs(target_idx, region_size);
        network.learn_transition(&source_pattern, &target_pattern);

        for pair in source_pattern.iter().zip(target_pattern.iter()) {
            network.reinforce_coactivation_if_useful(&[*pair.0, *pair.1], 0.025, 0.90);
        }
    }
}

fn region_hubs(region_idx: usize, region_size: usize) -> Vec<usize> {
    let start = region_idx * region_size;
    let stride = (region_size / (BRIDGE_HUBS + 1)).max(1);
    (1..=BRIDGE_HUBS).map(|idx| start + idx * stride).collect()
}

fn region_index(name: &str) -> usize {
    REGIONS
        .iter()
        .position(|region| region.name == name)
        .expect("region conocida")
}

fn region_manifest(region_size: usize) -> String {
    let mut out = String::new();
    out.push_str("SNGA_MULTIREGION_SUBSTRATE_V1\n");
    out.push_str(&format!("region_size={region_size}\n"));
    out.push_str(&format!("total_nodes={}\n\n", region_size * REGION_COUNT));
    for (idx, region) in REGIONS.iter().enumerate() {
        let start = idx * region_size;
        let end = start + region_size - 1;
        out.push_str(&format!(
            "{}={}..{} # {}\n",
            region.name, start, end, region.description
        ));
    }
    out.push_str("\nflows:\n");
    out.push_str("visual -> parietal -> prefrontal -> motor\n");
    out.push_str("linguistic -> prefrontal\n");
    out.push_str("hippocampal -> prefrontal\n");
    out.push_str("prefrontal -> basal_ganglia -> motor\n");
    out.push_str("prefrontal -> linguistic\n");
    out
}

fn fractal_mesh_config(target_nodes: usize) -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 5,
        target_dimension: 2.65,
        target_nodes,
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
