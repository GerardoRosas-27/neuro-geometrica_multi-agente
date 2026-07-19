//! Ciclo causal para probar cognición operacional y consolidación por simetría.

use crate::matrix_free_cognitive_substrate::LatentConceptId;
use crate::relational_field::ObserverId;
use crate::unified_spin_cognitive_engine::{
    KnowledgeKey, UnifiedSpinCognitiveConfig, UnifiedSpinCognitiveEngine,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EmergentCognitionReport {
    pub trials: usize,
    pub symmetry_without_relations_abstention: f64,
    pub composition_absent_after_one_rule: f64,
    pub novel_composition_after_two_rules: f64,
    pub direct_composed_relation_absent: f64,
    pub symmetry_orbit_transfer: f64,
    pub no_symmetry_orbit_transfer: f64,
    pub intact_consolidation: f64,
    pub lesion_blocks_consolidation: f64,
    pub repair_restores_consolidation: f64,
    pub ood_abstention: f64,
    pub mean_knowledge_stage0: f64,
    pub mean_knowledge_stage1: f64,
    pub mean_knowledge_stage2: f64,
    pub mean_knowledge_final: f64,
    pub evidence_pass: bool,
}

pub fn run_emergent_cognition_cycle(trials: usize) -> EmergentCognitionReport {
    let trials = trials.max(1);
    let mut no_content = 0;
    let mut absent_after_one = 0;
    let mut composition = 0;
    let mut no_direct_ac = 0;
    let mut orbit_transfer = 0;
    let mut ablated_orbit = 0;
    let mut intact = 0;
    let mut lesion_block = 0;
    let mut repair_restore = 0;
    let mut ood = 0;
    let mut knowledge0 = 0;
    let mut knowledge1 = 0;
    let mut knowledge2 = 0;
    let mut knowledge_final = 0;

    for trial in 0..trials {
        let observer = ObserverId(997_000 + trial);
        let phase = (trial as f64 * 0.113).rem_euclid(std::f64::consts::TAU);
        let mut engine = fixture();
        knowledge0 += engine.knowledge.len();
        no_content += usize::from(
            engine
                .infer(observer, LatentConceptId(0), phase, 2)
                .is_none(),
        );

        let first = engine.train_relation(
            observer,
            LatentConceptId(0),
            LatentConceptId(1),
            phase,
            1.0,
            0.0,
            &[],
            24,
        );
        intact += usize::from(first.gate.passed);
        knowledge1 += engine.knowledge.len();
        absent_after_one += usize::from(
            engine
                .infer(observer, LatentConceptId(0), phase, 2)
                .is_some_and(|inference| inference.path.last() != Some(&LatentConceptId(2))),
        );

        engine.train_relation(
            observer,
            LatentConceptId(1),
            LatentConceptId(2),
            phase,
            1.0,
            0.0,
            &[],
            24,
        );
        knowledge2 += engine.knowledge.len();
        no_direct_ac += usize::from(
            engine
                .cognition
                .workspace
                .relation(observer, LatentConceptId(0), LatentConceptId(2))
                .is_none(),
        );
        composition += usize::from(
            engine
                .infer(observer, LatentConceptId(0), phase, 2)
                .is_some_and(|inference| {
                    inference.path
                        == vec![LatentConceptId(0), LatentConceptId(1), LatentConceptId(2)]
                }),
        );

        let orbit = [(LatentConceptId(5), LatentConceptId(6))];
        engine.train_relation(
            observer,
            LatentConceptId(3),
            LatentConceptId(4),
            phase,
            1.0,
            1.0,
            &orbit,
            24,
        );
        orbit_transfer += usize::from(engine.knowledge.contains_key(&KnowledgeKey {
            observer: observer.0,
            source: LatentConceptId(5),
            target: LatentConceptId(6),
        }));
        ood += usize::from(
            engine
                .infer(observer, LatentConceptId(7), phase, 2)
                .is_none(),
        );

        let removed_bond = engine.spin_liquid.bonds.pop().expect("periodic bonds");
        let before_lesion = engine.knowledge.len();
        let lesion = engine.train_relation(
            observer,
            LatentConceptId(8),
            LatentConceptId(9),
            phase,
            1.0,
            1.0,
            &[],
            24,
        );
        lesion_block += usize::from(!lesion.gate.passed && engine.knowledge.len() == before_lesion);
        engine.spin_liquid.bonds.push(removed_bond);
        let repaired = engine.train_relation(
            observer,
            LatentConceptId(8),
            LatentConceptId(9),
            phase,
            1.0,
            1.0,
            &[],
            24,
        );
        repair_restore += usize::from(
            repaired.gate.passed
                && engine.knowledge.contains_key(&KnowledgeKey {
                    observer: observer.0,
                    source: LatentConceptId(8),
                    target: LatentConceptId(9),
                }),
        );
        knowledge_final += engine.knowledge.len();

        let mut ablated = fixture();
        ablated.train_relation(
            observer,
            LatentConceptId(3),
            LatentConceptId(4),
            phase,
            1.0,
            0.0,
            &orbit,
            24,
        );
        ablated_orbit += usize::from(
            ablated
                .cognition
                .workspace
                .query(observer, LatentConceptId(5), phase)
                .is_empty(),
        );
    }

    let n = trials as f64;
    let mut report = EmergentCognitionReport {
        trials,
        symmetry_without_relations_abstention: no_content as f64 / n,
        composition_absent_after_one_rule: absent_after_one as f64 / n,
        novel_composition_after_two_rules: composition as f64 / n,
        direct_composed_relation_absent: no_direct_ac as f64 / n,
        symmetry_orbit_transfer: orbit_transfer as f64 / n,
        no_symmetry_orbit_transfer: ablated_orbit as f64 / n,
        intact_consolidation: intact as f64 / n,
        lesion_blocks_consolidation: lesion_block as f64 / n,
        repair_restores_consolidation: repair_restore as f64 / n,
        ood_abstention: ood as f64 / n,
        mean_knowledge_stage0: knowledge0 as f64 / n,
        mean_knowledge_stage1: knowledge1 as f64 / n,
        mean_knowledge_stage2: knowledge2 as f64 / n,
        mean_knowledge_final: knowledge_final as f64 / n,
        evidence_pass: false,
    };
    report.evidence_pass = report.symmetry_without_relations_abstention == 1.0
        && report.composition_absent_after_one_rule == 1.0
        && report.novel_composition_after_two_rules == 1.0
        && report.direct_composed_relation_absent == 1.0
        && report.symmetry_orbit_transfer == 1.0
        && report.no_symmetry_orbit_transfer == 1.0
        && report.intact_consolidation == 1.0
        && report.lesion_blocks_consolidation == 1.0
        && report.repair_restores_consolidation == 1.0
        && report.ood_abstention == 1.0;
    report
}

fn fixture() -> UnifiedSpinCognitiveEngine {
    UnifiedSpinCognitiveEngine::periodic_pyrochlore(
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
    .expect("emergent cognition fixture")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_cycle_supports_operational_emergent_cognition() {
        let report = run_emergent_cognition_cycle(32);
        println!("{report:#?}");
        assert!(report.evidence_pass);
        assert_eq!(report.mean_knowledge_stage0, 0.0);
        assert_eq!(report.mean_knowledge_stage1, 1.0);
        assert_eq!(report.mean_knowledge_stage2, 2.0);
        assert!(report.mean_knowledge_final > report.mean_knowledge_stage2);
    }
}
