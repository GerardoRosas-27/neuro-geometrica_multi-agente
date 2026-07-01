use snga::geometry::Vec2;
use snga::mesh_engine::FractalMeshConfig;
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};

const DEFAULT_OUTPUT_STATE: &str = "data/snga_fractal_semantic_executive_substrate.snga";
const DEFAULT_REGION_SIZE: usize = 8_192;
const REGION_COUNT: usize = 12;
const BRIDGE_HUBS: usize = 16;
const PATTERN_SIZE: usize = 12;

#[derive(Clone, Copy)]
struct BrainRegion {
    name: &'static str,
    ring: RegionRing,
    description: &'static str,
}

#[derive(Clone, Copy)]
enum RegionRing {
    Core,
    Peripheral,
}

#[derive(Clone, Copy)]
enum Region {
    SemanticHubAtl = 0,
    ConceptBinder = 1,
    SemanticControl = 2,
    ExecutiveLogicDlpfc = 3,
    WorkingMemory = 4,
    Planner = 5,
    ControlGate = 6,
    VisualSlot = 7,
    AuditorySlot = 8,
    SomatosensorySlot = 9,
    LinguisticSlot = 10,
    EpisodicSlot = 11,
}

const REGIONS: [BrainRegion; REGION_COUNT] = [
    BrainRegion {
        name: "semantic_hub_atl",
        ring: RegionRing::Core,
        description: "hub tipo ATL: integra rasgos distribuidos en conceptos abstractos",
    },
    BrainRegion {
        name: "concept_binder",
        ring: RegionRing::Core,
        description: "ensambla nombre, forma, sonido, textura, categoria y affordance",
    },
    BrainRegion {
        name: "semantic_control",
        ring: RegionRing::Core,
        description: "control semantico fronto-temporal: selecciona significado por contexto",
    },
    BrainRegion {
        name: "executive_logic_dlpfc",
        ring: RegionRing::Core,
        description: "logica, reglas activas, restricciones y comparacion de alternativas",
    },
    BrainRegion {
        name: "working_memory",
        ring: RegionRing::Core,
        description: "pantalla mental para mantener metas, conceptos y pasos intermedios",
    },
    BrainRegion {
        name: "planner",
        ring: RegionRing::Core,
        description: "secuenciacion de planes y preparacion de acciones futuras",
    },
    BrainRegion {
        name: "control_gate",
        ring: RegionRing::Core,
        description: "gating e inhibicion funcional para significados o acciones no pertinentes",
    },
    BrainRegion {
        name: "visual_slot",
        ring: RegionRing::Peripheral,
        description: "slot periferico futuro para color, forma, textura visual y objetos",
    },
    BrainRegion {
        name: "auditory_slot",
        ring: RegionRing::Peripheral,
        description: "slot periferico futuro para fonemas, timbre, ritmo y habla percibida",
    },
    BrainRegion {
        name: "somatosensory_slot",
        ring: RegionRing::Peripheral,
        description: "slot periferico futuro para tacto, sabor, cuerpo y textura",
    },
    BrainRegion {
        name: "linguistic_slot",
        ring: RegionRing::Peripheral,
        description: "slot periferico futuro para simbolos, palabras, frases y lectura",
    },
    BrainRegion {
        name: "episodic_slot",
        ring: RegionRing::Peripheral,
        description: "slot periferico futuro para contexto episodico y memoria rapida",
    },
];

#[derive(Clone, Copy)]
struct ConceptSeed {
    name: &'static str,
    word: &'static str,
    visual: &'static str,
    sound: &'static str,
    somatic: &'static str,
    category: &'static str,
    use_hint: &'static str,
}

#[derive(Clone, Copy)]
struct PlanSeed {
    goal: &'static str,
    constraint: &'static str,
    request: &'static str,
    accepted_concept: &'static str,
    rejected_concept: &'static str,
    plan_step: &'static str,
}

fn main() {
    let output =
        env::var("SNGA_SEMEXEC_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT_STATE.to_string());
    let region_size = env::var("SNGA_SEMEXEC_REGION_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_REGION_SIZE)
        .max(BRIDGE_HUBS + PATTERN_SIZE + 1);
    let warmup_rounds = env::var("SNGA_SEMEXEC_WARMUP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6)
        .max(1);
    let total_nodes = region_size * REGION_COUNT;
    let cfg = config();

    println!("SNGA fractal semantic-executive substrate");
    println!(
        "from_scratch=true output={output} region_size={region_size} total_nodes={total_nodes} warmup_rounds={warmup_rounds}"
    );

    let mut network = SimplicialNetwork::fractal_3d(cfg.clone(), fractal_mesh_config(total_nodes));
    layout_regions(&mut network, region_size, &cfg);
    add_core_bridges(&mut network, region_size);
    seed_semantic_executive_priors(&mut network, warmup_rounds);
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
    if let Err(err) = fs::write(&region_path, region_manifest(region_size, warmup_rounds)) {
        println!("regions_saved=false error={err}");
    } else {
        println!("regions_saved=true path={region_path}");
    }
}

fn add_core_bridges(network: &mut SimplicialNetwork, region_size: usize) {
    let bridges = [
        (Region::VisualSlot, Region::SemanticHubAtl),
        (Region::AuditorySlot, Region::SemanticHubAtl),
        (Region::SomatosensorySlot, Region::SemanticHubAtl),
        (Region::LinguisticSlot, Region::SemanticHubAtl),
        (Region::EpisodicSlot, Region::SemanticHubAtl),
        (Region::SemanticHubAtl, Region::ConceptBinder),
        (Region::ConceptBinder, Region::SemanticHubAtl),
        (Region::SemanticHubAtl, Region::SemanticControl),
        (Region::SemanticControl, Region::SemanticHubAtl),
        (Region::SemanticControl, Region::ExecutiveLogicDlpfc),
        (Region::ExecutiveLogicDlpfc, Region::SemanticControl),
        (Region::ExecutiveLogicDlpfc, Region::WorkingMemory),
        (Region::WorkingMemory, Region::Planner),
        (Region::Planner, Region::ControlGate),
        (Region::ControlGate, Region::SemanticControl),
    ];

    for (source, target) in bridges {
        let source_pattern = region_hubs(source, region_size);
        let target_pattern = region_hubs(target, region_size);
        network.learn_transition(&source_pattern, &target_pattern);

        for (left, right) in source_pattern.iter().zip(target_pattern.iter()) {
            network.reinforce_coactivation_if_useful(&[*left, *right], 0.03, 0.92);
        }
    }
}

fn seed_semantic_executive_priors(network: &mut SimplicialNetwork, warmup_rounds: usize) {
    for _ in 0..warmup_rounds {
        for seed in concept_seeds() {
            train_concept(network, *seed);
        }
        for seed in plan_seeds() {
            train_plan(network, *seed);
        }
        train_ambiguity_control(network);
    }
}

fn train_concept(network: &mut SimplicialNetwork, seed: ConceptSeed) {
    let word = pattern(
        Region::LinguisticSlot,
        "word",
        seed.word,
        network.agents.len(),
    );
    let visual = pattern(
        Region::VisualSlot,
        "visual_feature",
        seed.visual,
        network.agents.len(),
    );
    let sound = pattern(
        Region::AuditorySlot,
        "auditory_feature",
        seed.sound,
        network.agents.len(),
    );
    let somatic = pattern(
        Region::SomatosensorySlot,
        "somatic_feature",
        seed.somatic,
        network.agents.len(),
    );
    let hub = pattern(
        Region::SemanticHubAtl,
        "concept",
        seed.name,
        network.agents.len(),
    );
    let binder = pattern(
        Region::ConceptBinder,
        "binding",
        &format!("{} {}", seed.category, seed.use_hint),
        network.agents.len(),
    );

    for input in [&word, &visual, &sound, &somatic] {
        network.learn_transition(input, &hub);
    }
    network.learn_transition(&hub, &binder);
    network.learn_transition(&binder, &hub);
    reinforce_fused(
        network,
        [&word, &visual, &sound, &somatic, &hub, &binder],
        0.055,
    );
}

fn train_plan(network: &mut SimplicialNetwork, seed: PlanSeed) {
    let goal = pattern(Region::Planner, "goal", seed.goal, network.agents.len());
    let constraint = pattern(
        Region::WorkingMemory,
        "constraint",
        seed.constraint,
        network.agents.len(),
    );
    let request = pattern(
        Region::ExecutiveLogicDlpfc,
        "semantic_request",
        seed.request,
        network.agents.len(),
    );
    let control = pattern(
        Region::SemanticControl,
        "context_filter",
        seed.constraint,
        network.agents.len(),
    );
    let accepted = pattern(
        Region::SemanticHubAtl,
        "accepted_concept",
        seed.accepted_concept,
        network.agents.len(),
    );
    let rejected = pattern(
        Region::SemanticHubAtl,
        "rejected_concept",
        seed.rejected_concept,
        network.agents.len(),
    );
    let gate = pattern(
        Region::ControlGate,
        "gate_reject",
        seed.rejected_concept,
        network.agents.len(),
    );
    let step = pattern(
        Region::Planner,
        "plan_step",
        seed.plan_step,
        network.agents.len(),
    );

    network.learn_transition(&goal, &constraint);
    network.learn_transition(&constraint, &request);
    network.learn_transition(&request, &control);
    network.learn_transition(&control, &accepted);
    network.learn_transition(&accepted, &step);
    network.learn_transition(&rejected, &gate);
    network.learn_transition(&gate, &control);

    reinforce_fused(
        network,
        [&goal, &constraint, &request, &control, &accepted, &step],
        0.06,
    );
    reinforce_fused(network, [&rejected, &gate, &control], 0.045);
}

fn train_ambiguity_control(network: &mut SimplicialNetwork) {
    let word = pattern(
        Region::LinguisticSlot,
        "word",
        "banco",
        network.agents.len(),
    );
    let money_context = pattern(
        Region::WorkingMemory,
        "context",
        "pagar dinero",
        network.agents.len(),
    );
    let park_context = pattern(
        Region::WorkingMemory,
        "context",
        "paseo parque sentarse",
        network.agents.len(),
    );
    let money_control = pattern(
        Region::SemanticControl,
        "select_meaning",
        "banco financiero",
        network.agents.len(),
    );
    let seat_control = pattern(
        Region::SemanticControl,
        "select_meaning",
        "banco para sentarse",
        network.agents.len(),
    );
    let money_meaning = pattern(
        Region::SemanticHubAtl,
        "concept",
        "institucion financiera",
        network.agents.len(),
    );
    let seat_meaning = pattern(
        Region::SemanticHubAtl,
        "concept",
        "asiento del parque",
        network.agents.len(),
    );

    network.learn_transition(&word, &money_meaning);
    network.learn_transition(&money_context, &money_control);
    network.learn_transition(&park_context, &seat_control);
    network.learn_transition(&money_control, &money_meaning);
    network.learn_transition(&seat_control, &seat_meaning);
    reinforce_fused(
        network,
        [&word, &money_context, &money_control, &money_meaning],
        0.05,
    );
    reinforce_fused(
        network,
        [&word, &park_context, &seat_control, &seat_meaning],
        0.05,
    );
}

fn reinforce_fused<const N: usize>(
    network: &mut SimplicialNetwork,
    parts: [&Vec<usize>; N],
    learning_rate: f32,
) {
    let mut fused = Vec::new();
    for part in parts {
        fused.extend(part.iter().copied());
    }
    fused.sort_unstable();
    fused.dedup();
    network.reinforce_coactivation_if_useful(&fused, learning_rate, 0.93);
}

fn pattern(region: Region, prefix: &str, value: &str, nodes: usize) -> Vec<usize> {
    let region_size = inferred_region_size(nodes);
    let start = region as usize * region_size;
    let len = region_size.min(nodes.saturating_sub(start)).max(1);
    let normalized = normalize_text(value);
    (0..PATTERN_SIZE)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (region as usize).hash(&mut hasher);
            prefix.hash(&mut hasher);
            normalized.hash(&mut hasher);
            offset.hash(&mut hasher);
            start + (hasher.finish() as usize % len)
        })
        .collect()
}

fn inferred_region_size(nodes: usize) -> usize {
    (nodes / REGION_COUNT).max(BRIDGE_HUBS + PATTERN_SIZE + 1)
}

fn region_hubs(region: Region, region_size: usize) -> Vec<usize> {
    let start = region as usize * region_size;
    let stride = (region_size / (BRIDGE_HUBS + 1)).max(1);
    (1..=BRIDGE_HUBS).map(|idx| start + idx * stride).collect()
}

fn layout_regions(network: &mut SimplicialNetwork, region_size: usize, cfg: &SimplicialConfig) {
    let center = Vec2::new(
        cfg.width as f32 * cfg.spacing * 0.5,
        cfg.height as f32 * cfg.spacing * 0.5,
    );
    let core_radius = cfg.spacing * 9.0;
    let peripheral_radius = cfg.spacing * 27.0;
    let local_core_radius = cfg.spacing * 3.6;
    let local_peripheral_radius = cfg.spacing * 5.4;
    let golden_angle = 2.399_963_1_f32;

    for region_idx in 0..REGION_COUNT {
        let region = REGIONS[region_idx];
        let (region_center, local_radius, base_depth) = match region.ring {
            RegionRing::Core => {
                let angle = region_idx as f32 * std::f32::consts::TAU / 7.0;
                let offset_radius = if region_idx == 0 {
                    0.0
                } else {
                    core_radius * 0.62
                };
                (
                    center + Vec2::new(angle.cos(), angle.sin()) * offset_radius,
                    local_core_radius,
                    -cfg.spacing * 1.2,
                )
            }
            RegionRing::Peripheral => {
                let peripheral_idx = region_idx - 7;
                let angle = peripheral_idx as f32 * std::f32::consts::TAU / 5.0;
                (
                    center + Vec2::new(angle.cos(), angle.sin()) * peripheral_radius,
                    local_peripheral_radius,
                    cfg.spacing * 1.8,
                )
            }
        };

        let start = region_idx * region_size;
        let end = ((region_idx + 1) * region_size).min(network.agents.len());
        let count = end.saturating_sub(start).max(1) as f32;
        for (local_idx, agent_idx) in (start..end).enumerate() {
            let t = (local_idx as f32 + 0.5) / count;
            let radius = local_radius * t.sqrt();
            let angle = local_idx as f32 * golden_angle;
            network.agents[agent_idx].position =
                region_center + Vec2::new(angle.cos(), angle.sin()) * radius;
            network.agents[agent_idx].depth = base_depth + (angle * 0.37).sin() * cfg.spacing * 0.8;
        }
    }
}

fn region_manifest(region_size: usize, warmup_rounds: usize) -> String {
    let mut out = String::new();
    out.push_str("SNGA_SEMANTIC_EXECUTIVE_SUBSTRATE_V1\n");
    out.push_str("from_scratch=true\n");
    out.push_str("layout=central_semantic_executive_core_with_peripheral_future_slots\n");
    out.push_str(&format!("region_size={region_size}\n"));
    out.push_str(&format!("total_nodes={}\n", region_size * REGION_COUNT));
    out.push_str(&format!("warmup_rounds={warmup_rounds}\n\n"));

    for (idx, region) in REGIONS.iter().enumerate() {
        let start = idx * region_size;
        let end = start + region_size - 1;
        let ring = match region.ring {
            RegionRing::Core => "core",
            RegionRing::Peripheral => "peripheral_future_slot",
        };
        out.push_str(&format!(
            "{}={}..{} ring={} # {}\n",
            region.name, start, end, ring, region.description
        ));
    }

    out.push_str("\ncentral_flows:\n");
    out.push_str(
        "peripheral sensory/linguistic/episodic slots -> semantic_hub_atl -> concept_binder\n",
    );
    out.push_str("semantic_hub_atl <-> semantic_control <-> executive_logic_dlpfc\n");
    out.push_str(
        "executive_logic_dlpfc -> working_memory -> planner -> control_gate -> semantic_control\n",
    );
    out.push_str("semantic_control resolves ambiguous meanings before planner commits a step\n");
    out
}

fn concept_seeds() -> &'static [ConceptSeed] {
    &[
        ConceptSeed {
            name: "manzana",
            word: "manzana",
            visual: "rojo redondo brillante",
            sound: "sonido palabra manzana",
            somatic: "crujiente dulce jugosa",
            category: "fruta vegetal comestible",
            use_hint: "se puede comer",
        },
        ConceptSeed {
            name: "lechuga",
            word: "lechuga",
            visual: "verde hojas capas",
            sound: "sonido palabra lechuga",
            somatic: "fresco crujiente ligero",
            category: "vegetal comestible",
            use_hint: "sirve para ensalada",
        },
        ConceptSeed {
            name: "filete",
            word: "filete",
            visual: "pieza carne plato",
            sound: "sonido palabra filete",
            somatic: "fibroso salado",
            category: "carne animal",
            use_hint: "no vegetariano",
        },
        ConceptSeed {
            name: "banco asiento",
            word: "banco",
            visual: "asiento largo parque",
            sound: "sonido palabra banco",
            somatic: "superficie dura sentarse",
            category: "mueble lugar",
            use_hint: "sentarse en parque",
        },
        ConceptSeed {
            name: "banco financiero",
            word: "banco",
            visual: "edificio dinero ventanilla",
            sound: "sonido palabra banco",
            somatic: "tramite documento",
            category: "institucion dinero",
            use_hint: "guardar o pagar dinero",
        },
    ]
}

fn plan_seeds() -> &'static [PlanSeed] {
    &[
        PlanSeed {
            goal: "preparar cena vegetariana",
            constraint: "sin carne",
            request: "buscar alimentos permitidos",
            accepted_concept: "lechuga lentejas arroz manzana",
            rejected_concept: "filete carne",
            plan_step: "combinar lentejas con arroz",
        },
        PlanSeed {
            goal: "elegir significado de banco",
            constraint: "contexto paseo parque",
            request: "resolver palabra ambigua banco",
            accepted_concept: "banco asiento",
            rejected_concept: "banco financiero",
            plan_step: "buscar lugar para sentarse",
        },
    ]
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

fn fractal_mesh_config(target_nodes: usize) -> FractalMeshConfig {
    FractalMeshConfig {
        levels: 7,
        branches_per_region: 6,
        target_dimension: 2.72,
        target_nodes,
        base_radius: 0.0,
        lateral_link_weight: 0.32,
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
        activation_threshold: 0.63,
        simplex_area_weight: 0.00012,
        max_active_agents: 448,
        inhibition_decay: 0.035,
        max_spikes_per_step: 1024,
        local_inhibition_decay: 0.78,
        refractory_ticks: 0,
        rhythm_period: 14,
        rhythm_amplitude: 0.045,
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
        seed: 727,
    }
}
