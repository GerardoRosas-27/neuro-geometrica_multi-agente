use cdt_rqm_epr::cdt_graphity::CdtGraphityConfig;
use cdt_rqm_epr::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use cdt_rqm_epr::entanglement::EntanglementConfig;
use cdt_rqm_epr::relational_field::{ObserverId, RelationalFieldConfig};
use std::hash::{Hash, Hasher};

const NODES_PER_SLICE: usize = 384;
const PAIRS: usize = 8;
const OBSERVER: ObserverId = ObserverId(800_001);

#[derive(Default)]
struct Profile {
    latency_sum: usize,
    leakage_sum: f32,
    active_edges: usize,
    regge: f32,
    relations: usize,
    epr_links: usize,
    epr_coherence: f32,
    epr_entropy: f32,
    causality_violations: usize,
    conflict_pruned: usize,
}

fn main() {
    let pairs = pairs();
    let mut baseline = CdtRqmUniverseSubstrate::new(config());
    let mut epr = CdtRqmUniverseSubstrate::new(config());
    epr.enable_entanglement(EntanglementConfig {
        create_threshold: 1.0,
        max_links_per_node: 8,
        max_syncs_per_step: 512,
        contradiction_gain: 0.55,
        max_entropy: 0.9,
        max_heat: 0.9,
        ..EntanglementConfig::default()
    });

    for (idx, (local, remote, _distractor)) in pairs.iter().enumerate() {
        // Baseline learns only half the long-range relations, emulating a purely
        // metric substrate where distant correlations arrive late or sparsely.
        if idx % 2 == 0 {
            for _ in 0..4 {
                baseline.train_observed_transition(OBSERVER, 0.0, local, remote, 0.55);
            }
        }
        for (&a, &b) in local.iter().zip(remote.iter()) {
            if let Some(field) = epr.entanglement.as_mut() {
                field.create_or_reinforce(a, b);
                field.create_or_reinforce(a, b);
                field.create_or_reinforce(a, b);
            }
        }
    }

    let baseline_profile = profile(&mut baseline, &pairs, false);
    let mut epr_profile = profile(&mut epr, &pairs, true);
    let before_conflict = epr.entanglement_summary().unwrap();
    for (local, remote, _) in pairs.iter().take(4) {
        if let Some(report) = epr.inject_entanglement_conflict(local[0], remote[0]) {
            epr_profile.conflict_pruned += report.pruned;
        }
    }
    let after_conflict = epr.entanglement_summary().unwrap();

    println!("CDT-RQM EPR efficiency profile");
    print_profile("baseline_cdt_rqm", &baseline_profile, pairs.len());
    print_profile("epr_cdt_rqm", &epr_profile, pairs.len());
    println!(
        "latency_gain_avg={:.2} baseline_latency_avg={:.2} epr_latency_avg={:.2}",
        baseline_profile.latency_sum as f32 / pairs.len() as f32
            - epr_profile.latency_sum as f32 / pairs.len() as f32,
        baseline_profile.latency_sum as f32 / pairs.len() as f32,
        epr_profile.latency_sum as f32 / pairs.len() as f32
    );
    println!(
        "geometry_delta: active_edges {} -> {} regge {:.3} -> {:.3}",
        baseline_profile.active_edges,
        epr_profile.active_edges,
        baseline_profile.regge,
        epr_profile.regge
    );
    println!(
        "epr_fuse: before_links={} after_links={} pruned={} coherence={:.3}->{:.3} entropy={:.3}->{:.3}",
        before_conflict.active_links,
        after_conflict.active_links,
        epr_profile.conflict_pruned,
        before_conflict.mean_coherence,
        after_conflict.mean_coherence,
        before_conflict.mean_entropy,
        after_conflict.mean_entropy
    );
    println!(
        "lectura: {}",
        if epr_profile.latency_sum < baseline_profile.latency_sum
            && epr_profile.causality_violations == 0
            && epr_profile.active_edges <= baseline_profile.active_edges + 256
        {
            "EPR mejora latencia con bajo costo estructural y conserva causalidad CDT"
        } else {
            "EPR mejora parcialmente, pero su costo estructural requiere mas poda o presupuesto"
        }
    );
}

fn profile(
    substrate: &mut CdtRqmUniverseSubstrate,
    pairs: &[(Vec<usize>, Vec<usize>, Vec<usize>)],
    include_epr: bool,
) -> Profile {
    let mut profile = Profile::default();
    for (local, remote, distractor) in pairs {
        profile.latency_sum += latency(substrate, local, remote, 8);
        profile.leakage_sum += leakage(substrate, local, remote, distractor);
    }
    profile.active_edges = substrate
        .hardware
        .edges
        .iter()
        .filter(|edge| edge.active)
        .count();
    profile.regge = substrate.hardware.regge_action();
    profile.relations = substrate.relation_count();
    profile.causality_violations = substrate.hardware.causality_violations();
    if include_epr {
        if let Some(report) = substrate.entanglement_summary() {
            profile.epr_links = report.active_links;
            profile.epr_coherence = report.mean_coherence;
            profile.epr_entropy = report.mean_entropy;
        }
    }
    profile
}

fn latency(
    substrate: &mut CdtRqmUniverseSubstrate,
    local: &[usize],
    remote: &[usize],
    max_steps: usize,
) -> usize {
    for step in 1..=max_steps {
        substrate.hardware.clear_activity();
        substrate.hardware.inject_pattern(local, 1.0);
        let report = substrate.step_from_boundary(OBSERVER, 0.0, local);
        let hits = report
            .expected_from_rqm
            .iter()
            .filter(|idx| remote.contains(idx))
            .count();
        if hits >= remote.len().min(4) {
            return step;
        }
    }
    max_steps + 1
}

fn leakage(
    substrate: &mut CdtRqmUniverseSubstrate,
    local: &[usize],
    remote: &[usize],
    distractor: &[usize],
) -> f32 {
    substrate.hardware.clear_activity();
    substrate.hardware.inject_pattern(local, 1.0);
    let report = substrate.step_from_boundary(OBSERVER, 0.0, local);
    let expected = report
        .collapse
        .candidates
        .iter()
        .filter(|candidate| remote.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum::<f32>();
    let leak = report
        .collapse
        .candidates
        .iter()
        .filter(|candidate| distractor.contains(&candidate.agent))
        .map(|candidate| candidate.score)
        .sum::<f32>();
    leak / (expected + leak).max(0.0001)
}

fn print_profile(label: &str, profile: &Profile, cases: usize) {
    println!(
        "{}: latency_avg={:.2} leakage_avg={:.1}% active_edges={} regge={:.3} relations={} epr_links={} epr_coherence={:.3} epr_entropy={:.3} causality_violations={}",
        label,
        profile.latency_sum as f32 / cases.max(1) as f32,
        profile.leakage_sum / cases.max(1) as f32 * 100.0,
        profile.active_edges,
        profile.regge,
        profile.relations,
        profile.epr_links,
        profile.epr_coherence,
        profile.epr_entropy,
        profile.causality_violations
    );
}

fn pairs() -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    (0..PAIRS)
        .map(|idx| {
            (
                pattern(&format!("local_{idx}"), 0),
                pattern(&format!("remote_{idx}"), 1),
                pattern(&format!("distractor_{idx}"), 1),
            )
        })
        .collect()
}

fn pattern(label: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..12)
        .map(|offset| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut hasher);
            offset.hash(&mut hasher);
            slice * NODES_PER_SLICE + (hasher.finish() as usize % NODES_PER_SLICE)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn config() -> CdtRqmConfig {
    CdtRqmConfig {
        cdt: CdtGraphityConfig {
            slices: 4,
            nodes_per_slice: NODES_PER_SLICE,
            initial_spatial_connectivity: 0.0001,
            initial_temporal_connectivity: 0.00005,
            target_spatial_degree: 4,
            target_temporal_degree: 3,
            target_tetrahedra_per_edge: 4,
            cooling_rate: 0.055,
            heating_rate: 0.12,
            reinforcement_rate: 0.11,
            prune_threshold: 0.055,
            max_new_edges_per_step: 8,
            seed: 88_991,
        },
        rqm: RelationalFieldConfig {
            amplitude_learning_rate: 0.10,
            phase_learning_rate: 0.24,
            coherence_learning_rate: 0.13,
            uncertainty_learning_rate: 0.11,
            amplitude_decay: 0.001,
            coherence_decay: 0.0005,
            uncertainty_recovery: 0.002,
            activation_threshold: 0.02,
        },
        max_quantum_candidates: 128,
        rqm_feedback_gain: 0.42,
    }
}
