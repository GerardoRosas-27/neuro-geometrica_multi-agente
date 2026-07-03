use snga::cdt_graphity::CdtGraphityConfig;
use snga::cdt_rqm::{CdtRqmConfig, CdtRqmUniverseSubstrate};
use snga::entanglement::EntanglementConfig;
use snga::relational_field::{ObserverId, RelationalFieldConfig};
use std::hash::{Hash, Hasher};

const NODES_PER_SLICE: usize = 512;
const OBSERVER: ObserverId = ObserverId(700_001);

fn main() {
    let local = pattern("local_attractor", 0);
    let remote = pattern("remote_attractor", 1);
    let distractor = pattern("distractor", 1);

    let baseline = CdtRqmUniverseSubstrate::new(config());
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

    for _ in 0..10 {
        for (&a, &b) in local.iter().zip(remote.iter()) {
            epr.observe_entanglement_correlation(a, b, 0.35);
        }
    }

    let baseline_latency = sync_latency(&mut baseline.clone(), &local, &remote, 6);
    let epr_latency = sync_latency(&mut epr.clone(), &local, &remote, 6);
    let baseline_leakage = leakage(&mut baseline.clone(), &local, &remote, &distractor);
    let epr_leakage = leakage(&mut epr.clone(), &local, &remote, &distractor);
    let before_conflict = epr.entanglement_summary().unwrap();
    let conflict = epr
        .inject_entanglement_conflict(local[0], remote[0])
        .unwrap();
    let after_conflict = epr.entanglement_summary().unwrap();

    println!("CDT-RQM temporal EPR entanglement benchmark");
    println!(
        "baseline_latency_steps={} epr_latency_steps={} latency_gain={}",
        baseline_latency,
        epr_latency,
        baseline_latency.saturating_sub(epr_latency)
    );
    println!(
        "baseline_leakage={:.1}% epr_leakage={:.1}% causality_violations={}",
        baseline_leakage * 100.0,
        epr_leakage * 100.0,
        epr.hardware.causality_violations()
    );
    println!(
        "epr_before_conflict: active_links={} coherence={:.3} entropy={:.3}",
        before_conflict.active_links, before_conflict.mean_coherence, before_conflict.mean_entropy
    );
    println!(
        "epr_conflict: conflicts={} pruned={} active_after={} coherence={:.3} entropy={:.3}",
        conflict.conflicts,
        conflict.pruned,
        after_conflict.active_links,
        after_conflict.mean_coherence,
        after_conflict.mean_entropy
    );
    println!(
        "lectura: {}",
        if epr_latency < baseline_latency
            && epr.hardware.causality_violations() == 0
            && after_conflict.active_links <= before_conflict.active_links
        {
            "EPR crea una autopista logica temporal, baja latencia y el fusible entropico conserva la causalidad CDT"
        } else {
            "EPR ejecuta, pero requiere ajustar umbrales de creacion/fusible para mejorar latencia con seguridad"
        }
    );
}

fn sync_latency(
    substrate: &mut CdtRqmUniverseSubstrate,
    local: &[usize],
    remote: &[usize],
    max_steps: usize,
) -> usize {
    for step in 1..=max_steps {
        substrate.hardware.clear_activity();
        substrate.hardware.inject_pattern(local, 1.0);
        let report = substrate.step_from_boundary(OBSERVER, 0.0, local);
        if report
            .expected_from_rqm
            .iter()
            .filter(|idx| remote.contains(idx))
            .count()
            >= remote.len().min(4)
        {
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

fn pattern(label: &str, slice: usize) -> Vec<usize> {
    let mut out = (0..16)
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
            seed: 88_771,
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
