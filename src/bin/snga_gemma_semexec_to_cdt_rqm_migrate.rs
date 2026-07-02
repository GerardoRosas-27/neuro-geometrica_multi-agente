use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::mesh_engine::FractalMeshConfig;
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use snga::simplicial::{SimplicialConfig, SimplicialNetwork};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};

const DEFAULT_STATE_PATH: &str = "data/snga_fractal_semantic_executive_gemma_adapter.snga";
const DEFAULT_TEACHINGS_PATH: &str = "data/snga_semantic_executive_console_teachings.tsv";
const DEFAULT_OUTPUT_PATH: &str = "data/cdt_rqm_gemma_semexec_consolidated.cdt_rqm";
const DEFAULT_REGION_SIZE: usize = 8_192;
const REGION_COUNT: usize = 12;
const PATTERN_SIZE: usize = 12;
const LETTER_PATTERN_SIZE: usize = 7;

#[allow(dead_code)]
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

#[derive(Clone, Debug)]
struct LearnedTeaching {
    prompt: String,
    teaching: String,
    response: String,
}

fn main() {
    let state_path =
        env::var("SNGA_SEMEXEC_ADAPTER_STATE").unwrap_or_else(|_| DEFAULT_STATE_PATH.to_string());
    let teachings_path =
        env::var("SNGA_SEMEXEC_TEACHINGS").unwrap_or_else(|_| DEFAULT_TEACHINGS_PATH.to_string());
    let output_path =
        env::var("SNGA_CDT_RQM_GEMMA_OUTPUT").unwrap_or_else(|_| DEFAULT_OUTPUT_PATH.to_string());
    let progress_path =
        env::var("SNGA_CDT_RQM_PROGRESS").unwrap_or_else(|_| format!("{output_path}.progress"));
    let max_causal = env_usize("SNGA_CDT_RQM_MAX_CAUSAL", 200_000);
    let min_weight = env_f32("SNGA_CDT_RQM_MIN_CAUSAL_WEIGHT", 0.05);
    let anneal_attempts = env_usize("SNGA_CDT_RQM_ANNEAL_ATTEMPTS", 48);
    let cdt_nodes_per_slice = env_usize("SNGA_CDT_RQM_NODES_PER_SLICE", 16_384);
    let progress_every = env_usize("SNGA_CDT_RQM_PROGRESS_EVERY", 1_000).max(1);
    let checkpoint_every = env_usize("SNGA_CDT_RQM_CHECKPOINT_EVERY", 25_000);

    let total_nodes = total_nodes();
    let mut network = SimplicialNetwork::fractal_3d(config(), fractal_mesh_config(total_nodes));
    match network.load_persistent_state(&state_path) {
        Ok(report) => println!(
            "loaded_snga=true state={} agents={} edges={} causal_edges={}",
            state_path, report.agents, report.edges, report.causal_edges
        ),
        Err(err) => {
            println!("loaded_snga=false state={} error={err}", state_path);
            return;
        }
    }

    let teachings = load_teachings(&teachings_path);
    println!(
        "loaded_teachings={} path={} max_causal={} min_weight={:.3} progress={} checkpoint_every={}",
        teachings.len(),
        teachings_path,
        max_causal,
        min_weight,
        progress_path,
        checkpoint_every
    );

    let mut substrate = CdtRqmUniverseSubstrate::new(cdt_rqm_config(cdt_nodes_per_slice));
    write_progress(
        &progress_path,
        &format!(
            "stage=init percent=0.00 migrated_causal=0 max_causal={} teachings={} output={}\n",
            max_causal,
            teachings.len(),
            output_path
        ),
    );
    checkpoint_state(&substrate, &output_path, checkpoint_every, 0, "init");

    let progress = ProgressConfig {
        progress_path: progress_path.clone(),
        output_path: output_path.clone(),
        progress_every,
        checkpoint_every,
    };
    println!(
        "progress stage=select_causal percent=0.50 message=seleccionando_top_causal_edges max_causal={} min_weight={:.3}",
        max_causal, min_weight
    );
    write_progress(
        &progress_path,
        &format!(
            "stage=select_causal percent=0.50 message=seleccionando_top_causal_edges max_causal={} min_weight={:.3}\n",
            max_causal, min_weight
        ),
    );
    let migrated_causal =
        migrate_causal_edges(&network, &mut substrate, max_causal, min_weight, &progress);
    let (teaching_relations, validation) = migrate_teachings(
        &teachings,
        &mut substrate,
        total_nodes,
        cdt_nodes_per_slice,
        &progress,
    );
    let anneal = if validation.is_empty() {
        None
    } else {
        println!(
            "progress stage=anneal percent=95.00 attempts={} validation={}",
            anneal_attempts,
            validation.len()
        );
        write_progress(
            &progress_path,
            &format!(
                "stage=anneal percent=95.00 attempts={} validation={} migrated_causal={} teaching_relations={}\n",
                anneal_attempts,
                validation.len(),
                migrated_causal,
                teaching_relations
            ),
        );
        checkpoint_state(&substrate, &output_path, 1, 1, "pre_anneal");
        Some(substrate.anneal_after_migration(&validation, anneal_attempts))
    };

    match substrate.save_consolidated_state(&output_path) {
        Ok(()) => {
            println!("saved_cdt_rqm=true output={output_path}");
            write_progress(
                &progress_path,
                &format!(
                    "stage=done percent=100.00 migrated_causal={} teaching_relations={} rqm_relations={} active_edges={} regge={:.3} causality_violations={}\n",
                    migrated_causal,
                    teaching_relations,
                    substrate.relation_count(),
                    substrate.hardware.edges.iter().filter(|edge| edge.active).count(),
                    substrate.hardware.regge_action(),
                    substrate.hardware.causality_violations()
                ),
            );
            println!(
                "summary: migrated_causal={} teaching_relations={} rqm_relations={} active_edges={} regge={:.3} causality_violations={}",
                migrated_causal,
                teaching_relations,
                substrate.relation_count(),
                substrate.hardware.edges.iter().filter(|edge| edge.active).count(),
                substrate.hardware.regge_action(),
                substrate.hardware.causality_violations()
            );
            if let Some(report) = anneal {
                println!(
                    "anneal: attempts={} accepted={} accuracy={:.1}%->{:.1}% leakage={:.1}%->{:.1}% regge={:.3}->{:.3} edges={} -> {}",
                    report.attempts,
                    report.accepted,
                    report.initial_accuracy * 100.0,
                    report.final_accuracy * 100.0,
                    report.initial_leakage * 100.0,
                    report.final_leakage * 100.0,
                    report.initial_regge,
                    report.final_regge,
                    report.initial_edges,
                    report.final_edges
                );
            } else {
                println!("anneal=skipped reason=no_teachings_validation");
            }
        }
        Err(err) => println!("saved_cdt_rqm=false output={} error={err}", output_path),
    }
}

struct ProgressConfig {
    progress_path: String,
    output_path: String,
    progress_every: usize,
    checkpoint_every: usize,
}

fn migrate_causal_edges(
    network: &SimplicialNetwork,
    substrate: &mut CdtRqmUniverseSubstrate,
    max_causal: usize,
    min_weight: f32,
    progress: &ProgressConfig,
) -> usize {
    let selected = network.causal_edges_snapshot_limited(max_causal, min_weight);
    println!(
        "progress stage=select_causal percent=1.00 selected={} max_causal={}",
        selected.len(),
        max_causal
    );
    write_progress(
        &progress.progress_path,
        &format!(
            "stage=select_causal percent=1.00 selected={} max_causal={} output={}\n",
            selected.len(),
            max_causal,
            progress.output_path
        ),
    );
    let total = selected.len().max(1);

    let mut migrated = 0;
    for (idx, (source, target, weight)) in selected.into_iter().enumerate() {
        let source_region = source / inferred_region_size(network.agents.len());
        let target_region = target / inferred_region_size(network.agents.len());
        let source_cdt = map_semexec_node(
            source,
            source_region,
            0,
            substrate.config.cdt.nodes_per_slice,
        );
        let target_cdt = map_semexec_node(
            target,
            target_region,
            1,
            substrate.config.cdt.nodes_per_slice,
        );
        let observer = observer_for_regions(source_region, target_region);
        let phase = phase_for_region(target_region);
        substrate.software.reinforce_relation(
            observer,
            source_cdt,
            target_cdt,
            phase,
            weight.min(1.0),
        );
        if substrate
            .hardware
            .reinforce_temporal_link(source_cdt, target_cdt, weight.min(1.0))
        {
            migrated += 1;
        }
        let done = idx + 1;
        if done % progress.progress_every == 0 || done == total {
            let percent = done as f32 / total as f32 * 85.0;
            println!(
                "progress stage=causal percent={:.2} scanned={} migrated={} total={}",
                percent, done, migrated, total
            );
            write_progress(
                &progress.progress_path,
                &format!(
                    "stage=causal percent={:.2} scanned={} migrated={} total={} rqm_relations={} active_edges={} output={}\n",
                    percent,
                    done,
                    migrated,
                    total,
                    substrate.relation_count(),
                    substrate.hardware.edges.iter().filter(|edge| edge.active).count(),
                    progress.output_path
                ),
            );
        }
        checkpoint_state(
            substrate,
            &progress.output_path,
            progress.checkpoint_every,
            done,
            "causal",
        );
    }
    migrated
}

fn migrate_teachings(
    teachings: &[LearnedTeaching],
    substrate: &mut CdtRqmUniverseSubstrate,
    source_nodes: usize,
    cdt_nodes_per_slice: usize,
    progress: &ProgressConfig,
) -> (
    usize,
    Vec<(ObserverId, f32, Vec<usize>, Vec<usize>, Vec<usize>)>,
) {
    let mut relations = 0;
    let mut validation = Vec::new();
    let total = teachings.len().max(1);
    for (idx, teaching) in teachings.iter().enumerate() {
        let input = linguistic_text_pattern("teach_input", &teaching.prompt, source_nodes);
        let meaning = regional_pattern(
            Region::SemanticHubAtl,
            "teach_meaning",
            &teaching.teaching,
            PATTERN_SIZE,
            source_nodes,
        );
        let frame = regional_pattern(
            Region::Planner,
            "teach_response_frame",
            &teaching.response,
            PATTERN_SIZE,
            source_nodes,
        );

        let cue = map_pattern(&input, 0, cdt_nodes_per_slice);
        let expected = map_pattern(&meaning, 1, cdt_nodes_per_slice);
        let response = map_pattern(&frame, 2, cdt_nodes_per_slice);
        let observer = ObserverId(20_000 + stable_hash(&teaching.prompt) % 20_000);
        let phase = phase_for_region(Region::SemanticHubAtl as usize);

        relations += reinforce_links(substrate, observer, phase, &cue, &expected, 1.0);
        relations += reinforce_links(substrate, observer, phase, &expected, &response, 0.95);
        validation.push((observer, phase, cue, expected, response));
        let done = idx + 1;
        let percent = 85.0 + done as f32 / total as f32 * 10.0;
        println!(
            "progress stage=teachings percent={:.2} migrated_teachings={} total={} teaching_relations={}",
            percent, done, total, relations
        );
        write_progress(
            &progress.progress_path,
            &format!(
                "stage=teachings percent={:.2} migrated_teachings={} total={} teaching_relations={} rqm_relations={} active_edges={} output={}\n",
                percent,
                done,
                total,
                relations,
                substrate.relation_count(),
                substrate.hardware.edges.iter().filter(|edge| edge.active).count(),
                progress.output_path
            ),
        );
        checkpoint_state(
            substrate,
            &progress.output_path,
            progress.checkpoint_every,
            done,
            "teachings",
        );
    }
    (relations, validation)
}

fn write_progress(path: &str, contents: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(err) = fs::write(path, contents) {
        eprintln!("progress_write=false path={} error={}", path, err);
    }
}

fn checkpoint_state(
    substrate: &CdtRqmUniverseSubstrate,
    output_path: &str,
    checkpoint_every: usize,
    step: usize,
    stage: &str,
) {
    if checkpoint_every == 0 || step % checkpoint_every != 0 {
        return;
    }
    match substrate.save_consolidated_state(output_path) {
        Ok(()) => println!(
            "checkpoint_saved=true stage={} step={} output={}",
            stage, step, output_path
        ),
        Err(err) => eprintln!(
            "checkpoint_saved=false stage={} step={} output={} error={}",
            stage, step, output_path, err
        ),
    }
}

fn reinforce_links(
    substrate: &mut CdtRqmUniverseSubstrate,
    observer: ObserverId,
    phase: f32,
    sources: &[usize],
    targets: &[usize],
    strength: f32,
) -> usize {
    let mut count = 0;
    for &source in sources.iter().take(12) {
        for &target in targets.iter().take(12) {
            for _ in 0..3 {
                substrate
                    .software
                    .reinforce_relation(observer, source, target, phase, strength);
            }
            if source / substrate.config.cdt.nodes_per_slice + 1
                == target / substrate.config.cdt.nodes_per_slice
                && substrate
                    .hardware
                    .reinforce_temporal_link(source, target, strength)
            {
                count += 1;
            }
        }
    }
    count
}

fn map_pattern(pattern: &[usize], slice: usize, nodes_per_slice: usize) -> Vec<usize> {
    let mut mapped = pattern
        .iter()
        .map(|idx| map_semexec_node(*idx, idx / DEFAULT_REGION_SIZE, slice, nodes_per_slice))
        .collect::<Vec<_>>();
    mapped.sort_unstable();
    mapped.dedup();
    mapped
}

fn map_semexec_node(node: usize, region: usize, slice: usize, nodes_per_slice: usize) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node.hash(&mut hasher);
    region.hash(&mut hasher);
    slice.hash(&mut hasher);
    slice * nodes_per_slice + (hasher.finish() as usize % nodes_per_slice)
}

fn observer_for_regions(source_region: usize, target_region: usize) -> ObserverId {
    ObserverId(10_000 + source_region * REGION_COUNT + target_region)
}

fn phase_for_region(region: usize) -> f32 {
    match region % 4 {
        0 => 0.0,
        1 => std::f32::consts::FRAC_PI_2,
        2 => std::f32::consts::PI,
        _ => -std::f32::consts::FRAC_PI_2,
    }
}

fn linguistic_text_pattern(prefix: &str, text: &str, nodes: usize) -> Vec<usize> {
    let mut out = regional_pattern(Region::LinguisticSlot, prefix, text, PATTERN_SIZE, nodes);
    let normalized = normalize_text(text);
    for (pos, ch) in normalized.chars().enumerate().take(32) {
        out.extend(regional_pattern(
            Region::LinguisticSlot,
            "letter",
            &format!("{ch}_{pos}"),
            LETTER_PATTERN_SIZE,
            nodes,
        ));
    }
    for word in normalized.split_whitespace().take(16) {
        out.extend(regional_pattern(
            Region::LinguisticSlot,
            "word",
            word,
            PATTERN_SIZE,
            nodes,
        ));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn regional_pattern(
    region: Region,
    prefix: &str,
    value: &str,
    size: usize,
    nodes: usize,
) -> Vec<usize> {
    let region_size = inferred_region_size(nodes);
    let start = region as usize * region_size;
    let len = region_size.min(nodes.saturating_sub(start)).max(1);
    let normalized = normalize_text(value);
    (0..size)
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

fn load_teachings(path: &str) -> Vec<LearnedTeaching> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter_map(|line| {
            let parts = line.split('\t').collect::<Vec<_>>();
            (parts.len() == 3).then(|| LearnedTeaching {
                prompt: unescape_field(parts[0]),
                teaching: unescape_field(parts[1]),
                response: unescape_field(parts[2]),
            })
        })
        .collect()
}

fn unescape_field(value: &str) -> String {
    let mut out = String::new();
    let mut escaped = false;
    for ch in value.chars() {
        if escaped {
            match ch {
                't' => out.push('\t'),
                'n' => out.push('\n'),
                '\\' => out.push('\\'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    out
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

fn stable_hash(value: &str) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    normalize_text(value).hash(&mut hasher);
    hasher.finish() as usize
}

fn inferred_region_size(nodes: usize) -> usize {
    (nodes / REGION_COUNT).max(DEFAULT_REGION_SIZE)
}

fn total_nodes() -> usize {
    env_usize("SNGA_SEMEXEC_REGION_SIZE", DEFAULT_REGION_SIZE).max(DEFAULT_REGION_SIZE)
        * REGION_COUNT
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

fn cdt_rqm_config(nodes_per_slice: usize) -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice,
            initial_spatial_connectivity: 0.08,
            initial_temporal_connectivity: 0.02,
            target_spatial_degree: 5,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 20_727,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.09,
            phase_learning_rate: 0.22,
            coherence_learning_rate: 0.12,
            uncertainty_learning_rate: 0.10,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.025,
        },
        max_quantum_candidates: 64,
        rqm_feedback_gain: 0.40,
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

fn env_usize(name: &str, fallback: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn env_f32(name: &str, fallback: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(fallback)
}
