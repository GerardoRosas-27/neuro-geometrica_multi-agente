//! Controlador estable de aprendizaje continuo del motor termodinámico.
//!
//! Mantiene separadas las etapas:
//! 1. limpieza/contabilidad del residuo;
//! 2. fluctuación geométrica temporal;
//! 3. propuestas de puente EPR verificadas;
//! 4. consolidación y gate transaccional.

use crate::native_thermo_rqm_epr::NativeThermoRqmEprSubstrate;
use crate::native_thermodynamic_engine::{
    evaluate_native_suite, native_sleep_consolidate, EngineMetrics, Lesson,
};
use crate::residue_budget::{native_sleep_residue, ResidueBudgetConfig, ResidueSleepReport};
use crate::residue_vacuum_bridge::{
    native_residue_vacuum_bridge, VacuumBridgeConfig, VacuumBridgeReport,
};
use crate::residue_vacuum_fluctuation::{
    native_vacuum_fluctuation, VacuumCycleReport, VacuumFluctuationConfig,
};

#[derive(Clone, Copy, Debug)]
pub struct PlasticityConfig {
    pub cleanup: bool,
    pub fluctuation: bool,
    pub bridges: bool,
    pub final_consolidation_attempts: usize,
    pub final_consolidation_replay_passes: usize,
    pub max_accuracy_drop: f32,
    pub max_leak_drift: f32,
    pub residue: ResidueBudgetConfig,
    pub vacuum: VacuumFluctuationConfig,
    pub bridge: VacuumBridgeConfig,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        let mut bridge = VacuumBridgeConfig::default();
        bridge.apply_vacuum = false;
        Self {
            cleanup: true,
            fluctuation: true,
            bridges: true,
            final_consolidation_attempts: 3,
            final_consolidation_replay_passes: 1,
            max_accuracy_drop: 0.0,
            max_leak_drift: 0.002,
            residue: ResidueBudgetConfig::default(),
            vacuum: VacuumFluctuationConfig::default(),
            bridge,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlasticityReport {
    pub before: EngineMetrics,
    pub after: EngineMetrics,
    pub cleanup: Option<ResidueSleepReport>,
    pub fluctuation: Option<VacuumCycleReport>,
    pub bridge: Option<VacuumBridgeReport>,
    pub final_consolidation_accepted: usize,
    pub accepted: bool,
    pub decision: &'static str,
}

pub fn run_plasticity_cycle(
    substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    config: PlasticityConfig,
) -> (NativeThermoRqmEprSubstrate, PlasticityReport) {
    let original = substrate.clone();
    let before = evaluate_native_suite(&substrate, lessons);
    let mut candidate = substrate;

    let cleanup = if config.cleanup {
        let (next, report) = native_sleep_residue(candidate, lessons, config.residue);
        candidate = next;
        Some(report)
    } else {
        None
    };

    let fluctuation = if config.fluctuation {
        let (next, report) = native_vacuum_fluctuation(candidate, lessons, config.vacuum);
        candidate = next;
        Some(report)
    } else {
        None
    };

    let bridge = if config.bridges {
        let mut bridge_config = config.bridge;
        // La fluctuación ya ocurrió en la etapa anterior.
        bridge_config.apply_vacuum = !config.fluctuation;
        let (next, report) =
            native_residue_vacuum_bridge(candidate, lessons, config.vacuum, bridge_config);
        candidate = next;
        Some(report)
    } else {
        None
    };

    let (candidate, final_sleep) = native_sleep_consolidate(
        candidate,
        lessons,
        config.final_consolidation_attempts,
        config.final_consolidation_replay_passes,
    );
    let after = evaluate_native_suite(&candidate, lessons);
    let preserves_accuracy =
        after.accuracy() + config.max_accuracy_drop + 1.0e-6 >= before.accuracy();
    let preserves_leakage = after.leakage() <= before.leakage() + config.max_leak_drift;
    let accepted = preserves_accuracy && preserves_leakage;
    let decision = if accepted {
        "plasticity_cycle_accept"
    } else if !preserves_accuracy {
        "plasticity_cycle_reject_accuracy"
    } else {
        "plasticity_cycle_reject_leakage"
    };

    let report = PlasticityReport {
        before,
        after,
        cleanup,
        fluctuation,
        bridge,
        final_consolidation_accepted: final_sleep.accepted,
        accepted,
        decision,
    };
    if accepted {
        (candidate, report)
    } else {
        (original, report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_thermodynamic_engine::{
        canonical_lessons, fresh_native_substrate, train_canonical,
    };

    #[test]
    fn plasticity_cycle_is_transactional() {
        let lessons = canonical_lessons();
        let mut substrate = fresh_native_substrate();
        train_canonical(&mut substrate, &lessons, 2);
        let before = evaluate_native_suite(&substrate, &lessons);
        let (after, report) =
            run_plasticity_cycle(substrate, &lessons, PlasticityConfig::default());
        let metrics = evaluate_native_suite(&after, &lessons);
        assert!(metrics.accuracy() + 1.0e-6 >= before.accuracy());
        if !report.accepted {
            assert!((metrics.leakage() - before.leakage()).abs() < 1.0e-5);
        }
    }
}
