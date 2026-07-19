//! Comparación pareada entre el motor unificado y CDT–RQM–EPR legacy.

use crate::entanglement::EntanglementConfig;
use crate::matrix_free_cognitive_substrate::LatentConceptId;
use crate::native_thermo_rqm_epr::{NativeThermoRqmConfig, NativeThermoRqmEprSubstrate};
use crate::native_thermodynamic_cdt::NativeThermoCdtConfig;
use crate::relational_field::ObserverId;
use crate::unified_spin_cognitive_engine::{
    UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngineComparisonReport {
    pub trials: usize,
    pub new_direct_accuracy: f64,
    pub legacy_direct_accuracy: f64,
    pub new_orbit_accuracy: f64,
    pub legacy_orbit_accuracy: f64,
    pub new_composition_accuracy: f64,
    pub legacy_composition_accuracy: f64,
    pub new_retention_accuracy: f64,
    pub legacy_retention_accuracy: f64,
    pub new_ood_abstention: f64,
    pub legacy_ood_abstention: f64,
    pub new_topology_gate_accuracy: f64,
    pub legacy_topology_gate_accuracy: f64,
    pub new_mean_knowledge: f64,
    pub new_mean_relations: f64,
    pub legacy_mean_relations: f64,
    pub new_mean_epr_links: f64,
    pub legacy_mean_epr_links: f64,
    pub new_total_ms: f64,
    pub legacy_total_ms: f64,
    pub new_has_quantum_entanglement: bool,
    pub new_has_tensor_network_backend: bool,
    pub legacy_has_knowledge_gate: bool,
}

pub fn compare_unified_against_legacy(trials: usize) -> EngineComparisonReport {
    let trials = trials.max(1);
    let mut new_direct = 0;
    let mut legacy_direct = 0;
    let mut new_orbit = 0;
    let mut legacy_orbit = 0;
    let mut new_composition = 0;
    let mut legacy_composition = 0;
    let mut new_retention = 0;
    let mut legacy_retention = 0;
    let mut new_ood = 0;
    let mut legacy_ood = 0;
    let mut new_gate = 0;
    let mut legacy_gate = 0;
    let mut new_knowledge_sum = 0;
    let mut new_relations_sum = 0;
    let mut legacy_relations_sum = 0;
    let mut new_epr_sum = 0;
    let mut legacy_epr_sum = 0;
    let mut new_total_ms = 0.0;
    let mut legacy_total_ms = 0.0;
    let mut quantum_entanglement = false;

    for trial in 0..trials {
        let observer = ObserverId(996_000 + trial);
        let phase = (trial as f64 * 0.137).rem_euclid(std::f64::consts::TAU);
        let orbit = [(LatentConceptId(3), LatentConceptId(4))];

        let new_started = Instant::now();
        let mut unified = UnifiedSpinCognitiveEngine::periodic_pyrochlore(
            2,
            1,
            1,
            UnifiedSpinCognitiveConfig {
                bootstrap_cooling_steps: 120,
                cooling_steps_per_observation: 1,
                real_steps_per_observation: 0,
                ..UnifiedSpinCognitiveConfig::default()
            },
        )
        .expect("unified fixture");
        unified.train_relation(
            observer,
            LatentConceptId(0),
            LatentConceptId(1),
            phase,
            1.0,
            1.0,
            &orbit,
            24,
        );
        unified.train_relation(
            observer,
            LatentConceptId(1),
            LatentConceptId(2),
            phase,
            1.0,
            0.0,
            &[],
            24,
        );
        new_direct += usize::from(
            unified
                .cognition
                .workspace
                .query(observer, LatentConceptId(0), phase)
                .first()
                .is_some_and(|candidate| candidate.concept == LatentConceptId(1)),
        );
        new_orbit += usize::from(
            unified
                .cognition
                .workspace
                .query(observer, LatentConceptId(3), phase)
                .first()
                .is_some_and(|candidate| candidate.concept == LatentConceptId(4)),
        );
        new_composition += usize::from(
            unified
                .infer(observer, LatentConceptId(0), phase, 2)
                .is_some_and(|inference| {
                    inference.path
                        == vec![LatentConceptId(0), LatentConceptId(1), LatentConceptId(2)]
                }),
        );
        unified.train_relation(
            observer,
            LatentConceptId(5),
            LatentConceptId(6),
            phase,
            1.0,
            0.0,
            &[],
            24,
        );
        new_retention += usize::from(
            unified
                .infer(observer, LatentConceptId(0), phase, 2)
                .is_some(),
        );
        new_ood += usize::from(
            unified
                .infer(observer, LatentConceptId(7), phase, 2)
                .is_none(),
        );
        let knowledge_before_lesion = unified.knowledge.len();
        unified.spin_liquid.bonds.pop();
        let lesion = unified.train_relation(
            observer,
            LatentConceptId(8),
            LatentConceptId(9),
            phase,
            1.0,
            1.0,
            &[],
            24,
        );
        new_gate +=
            usize::from(!lesion.gate.passed && unified.knowledge.len() == knowledge_before_lesion);
        let new_report = unified.report();
        quantum_entanglement |= new_report.quantum.entangled_edges > 0;
        new_knowledge_sum += unified.knowledge.len();
        new_relations_sum += new_report.rqm_relations;
        new_epr_sum += new_report.epr_links;
        new_total_ms += new_started.elapsed().as_secs_f64() * 1_000.0;

        let legacy_started = Instant::now();
        let mut legacy = legacy_fixture();
        train_legacy(&mut legacy, observer, phase as f32, 0, 1);
        train_legacy(&mut legacy, observer, phase as f32, 1, 2);
        legacy_direct += usize::from(has_candidate(&mut legacy, observer, phase as f32, 0, 1));
        legacy_orbit += usize::from(has_candidate(&mut legacy, observer, phase as f32, 3, 4));
        let first_hop = top_candidate(&mut legacy, observer, phase as f32, 0);
        let second_hop =
            first_hop.and_then(|middle| top_candidate(&mut legacy, observer, phase as f32, middle));
        legacy_composition += usize::from(first_hop == Some(1) && second_hop == Some(2));
        train_legacy(&mut legacy, observer, phase as f32, 5, 6);
        legacy_retention += usize::from(has_candidate(&mut legacy, observer, phase as f32, 0, 1));
        legacy_ood += usize::from(
            legacy
                .query(observer, phase as f32, &[7])
                .candidates
                .is_empty(),
        );
        let relations_before_lesion = legacy.relation_count();
        train_legacy(&mut legacy, observer, phase as f32, 8, 9);
        // El legacy aprende pese a no recibir/validar estado topológico.
        legacy_gate += usize::from(legacy.relation_count() == relations_before_lesion);
        legacy_relations_sum += legacy.relation_count();
        legacy_epr_sum += legacy.entanglement.active_count();
        legacy_total_ms += legacy_started.elapsed().as_secs_f64() * 1_000.0;
    }

    let denominator = trials as f64;
    EngineComparisonReport {
        trials,
        new_direct_accuracy: new_direct as f64 / denominator,
        legacy_direct_accuracy: legacy_direct as f64 / denominator,
        new_orbit_accuracy: new_orbit as f64 / denominator,
        legacy_orbit_accuracy: legacy_orbit as f64 / denominator,
        new_composition_accuracy: new_composition as f64 / denominator,
        legacy_composition_accuracy: legacy_composition as f64 / denominator,
        new_retention_accuracy: new_retention as f64 / denominator,
        legacy_retention_accuracy: legacy_retention as f64 / denominator,
        new_ood_abstention: new_ood as f64 / denominator,
        legacy_ood_abstention: legacy_ood as f64 / denominator,
        new_topology_gate_accuracy: new_gate as f64 / denominator,
        legacy_topology_gate_accuracy: legacy_gate as f64 / denominator,
        new_mean_knowledge: new_knowledge_sum as f64 / denominator,
        new_mean_relations: new_relations_sum as f64 / denominator,
        legacy_mean_relations: legacy_relations_sum as f64 / denominator,
        new_mean_epr_links: new_epr_sum as f64 / denominator,
        legacy_mean_epr_links: legacy_epr_sum as f64 / denominator,
        new_total_ms,
        legacy_total_ms,
        new_has_quantum_entanglement: quantum_entanglement,
        new_has_tensor_network_backend: true,
        legacy_has_knowledge_gate: false,
    }
}

fn legacy_fixture() -> NativeThermoRqmEprSubstrate {
    NativeThermoRqmEprSubstrate::new(
        NativeThermoCdtConfig {
            slices: 1,
            nodes_per_slice: 16,
            temperature: 0.0,
            ..NativeThermoCdtConfig::default()
        },
        NativeThermoRqmConfig {
            thermal_steps_per_train: 0,
            thermal_steps_per_query: 0,
            collect_query_diagnostics: false,
            ..NativeThermoRqmConfig::default()
        },
        EntanglementConfig {
            create_threshold: 0.75,
            ..EntanglementConfig::default()
        },
    )
}

fn train_legacy(
    legacy: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    phase: f32,
    source: usize,
    target: usize,
) {
    for _ in 0..24 {
        legacy.train_observed_transition(observer, phase, &[source], &[target], 1.0);
    }
}

fn has_candidate(
    legacy: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    phase: f32,
    source: usize,
    expected: usize,
) -> bool {
    legacy
        .query(observer, phase, &[source])
        .candidates
        .iter()
        .any(|candidate| candidate.agent == expected)
}

fn top_candidate(
    legacy: &mut NativeThermoRqmEprSubstrate,
    observer: ObserverId,
    phase: f32,
    source: usize,
) -> Option<usize> {
    legacy
        .query(observer, phase, &[source])
        .candidates
        .first()
        .map(|candidate| candidate.agent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_engine_improves_transfer_and_knowledge_gating() {
        let report = compare_unified_against_legacy(24);
        println!("{report:#?}");
        assert_eq!(report.new_direct_accuracy, 1.0);
        assert_eq!(report.legacy_direct_accuracy, 1.0);
        assert_eq!(report.new_orbit_accuracy, 1.0);
        assert_eq!(report.legacy_orbit_accuracy, 0.0);
        assert_eq!(report.new_composition_accuracy, 1.0);
        assert_eq!(report.legacy_composition_accuracy, 1.0);
        assert_eq!(report.new_retention_accuracy, 1.0);
        assert_eq!(report.legacy_retention_accuracy, 1.0);
        assert_eq!(report.new_ood_abstention, 1.0);
        assert_eq!(report.legacy_ood_abstention, 1.0);
        assert_eq!(report.new_topology_gate_accuracy, 1.0);
        assert_eq!(report.legacy_topology_gate_accuracy, 0.0);
        assert!(report.new_has_quantum_entanglement);
        assert!(report.new_total_ms > report.legacy_total_ms);
    }
}
