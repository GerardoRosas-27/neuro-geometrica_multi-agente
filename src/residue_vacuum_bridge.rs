//! Conversión de fluctuaciones de residuo en puentes EPR prospectivos.
//!
//! Secuencia: pulso local de vacío -> detectar A→B→C coherente -> reservar capacidad
//! desalojando el enlace EPR menos útil -> entrenar el puente con fuerza baja -> Sueño A.
//! Los cambios solo persisten cuando pasan el gate de memoria y utilidad.

use crate::native_thermo_rqm_epr::NativeThermoRqmEprSubstrate;
use crate::native_thermodynamic_engine::{
    evaluate_native_suite, native_sleep_consolidate, EngineMetrics, Lesson, DEFAULT_OBSERVER,
};
use crate::residue_budget::ResidueBudgetConfig;
use crate::residue_vacuum_fluctuation::{
    measure_dynamics, pulse_lessons_vacuum, SubstrateDynamics, VacuumFluctuationConfig,
    VacuumPulseReport,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct VacuumBridgeConfig {
    pub apply_vacuum: bool,
    pub max_bridges: usize,
    pub min_relation_score: f32,
    pub min_phase_alignment: f32,
    pub bridge_success: f32,
    /// Capacidad reservada por nodo para enlaces prospectivos.
    pub prospective_slots_per_node: usize,
    pub consolidate_attempts: usize,
    pub consolidate_replay_passes: usize,
    pub max_leak_drift: f32,
}

impl Default for VacuumBridgeConfig {
    fn default() -> Self {
        Self {
            apply_vacuum: true,
            max_bridges: 6,
            min_relation_score: 0.20,
            min_phase_alignment: 0.70,
            bridge_success: 0.24,
            prospective_slots_per_node: 1,
            consolidate_attempts: 4,
            consolidate_replay_passes: 2,
            max_leak_drift: 0.005,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BridgeSeed {
    a: usize,
    c: usize,
    prior: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VacuumBridgeReport {
    pub seeds_detected: usize,
    pub bridges_trained: usize,
    pub epr_before: usize,
    pub epr_after: usize,
    pub epr_created: usize,
    pub epr_evicted: usize,
    pub bridge_accuracy_before: f32,
    pub bridge_accuracy_after: f32,
    pub base_before: EngineMetrics,
    pub base_after: EngineMetrics,
    pub dynamics_before: SubstrateDynamics,
    pub dynamics_after: SubstrateDynamics,
    pub vacuum: VacuumPulseReport,
    pub consolidate_accepted: usize,
    pub accepted: bool,
    pub decision: &'static str,
}

pub fn native_residue_vacuum_bridge(
    substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    vacuum_config: VacuumFluctuationConfig,
    config: VacuumBridgeConfig,
) -> (NativeThermoRqmEprSubstrate, VacuumBridgeReport) {
    let base_before = evaluate_native_suite(&substrate, lessons);
    let dynamics_before = measure_dynamics(&substrate);
    let epr_before = substrate.entanglement.active_count();
    let original = substrate.clone();

    let mut candidate = substrate;
    let account_config = ResidueBudgetConfig {
        min_residue: vacuum_config.min_residue,
        ..ResidueBudgetConfig::default()
    };
    let vacuum = if config.apply_vacuum {
        pulse_lessons_vacuum(&mut candidate, lessons, &account_config, &vacuum_config)
    } else {
        VacuumPulseReport::default()
    };
    let seeds = detect_bridge_seeds(&candidate, lessons, config);
    let bridge_accuracy_before = bridge_accuracy(&candidate, &seeds, lessons);

    let mut epr_evicted = 0;
    let mut bridges_trained = 0;
    for seed in &seeds {
        epr_evicted += reserve_epr_capacity(
            &mut candidate,
            seed.a,
            seed.c,
            config.prospective_slots_per_node,
        );
        candidate.train_observed_transition(
            DEFAULT_OBSERVER,
            0.0,
            &[seed.a],
            &[seed.c],
            (config.bridge_success * (0.75 + 0.25 * seed.prior)).clamp(0.10, 0.40),
        );
        candidate.entanglement.create_or_reinforce(seed.a, seed.c);
        bridges_trained += 1;
    }

    let bridge_accuracy_after = bridge_accuracy(&candidate, &seeds, lessons);
    let epr_after_training = candidate.entanglement.active_count();
    let (candidate, sleep) = native_sleep_consolidate(
        candidate,
        lessons,
        config.consolidate_attempts,
        config.consolidate_replay_passes,
    );
    let base_after = evaluate_native_suite(&candidate, lessons);
    let dynamics_after = measure_dynamics(&candidate);
    let epr_after = candidate.entanglement.active_count();
    let epr_created = epr_after_training.saturating_sub(epr_before);
    let preserves_base = base_after.accuracy() + 1.0e-6 >= base_before.accuracy()
        && base_after.leakage() <= base_before.leakage() + config.max_leak_drift;
    let bridge_improves = bridge_accuracy_after > bridge_accuracy_before + 1.0e-6;
    let accepted = preserves_base && epr_created > 0 && bridge_improves;

    let decision = if accepted {
        "vacuum_bridge_accept"
    } else if !preserves_base {
        "vacuum_bridge_reject_memory"
    } else if seeds.is_empty() {
        "vacuum_bridge_reject_no_seed"
    } else if epr_created == 0 {
        "vacuum_bridge_reject_no_capacity"
    } else {
        "vacuum_bridge_reject_no_bridge_gain"
    };

    let report = VacuumBridgeReport {
        seeds_detected: seeds.len(),
        bridges_trained,
        epr_before,
        epr_after,
        epr_created,
        epr_evicted,
        bridge_accuracy_before,
        bridge_accuracy_after,
        base_before,
        base_after,
        dynamics_before,
        dynamics_after,
        vacuum,
        consolidate_accepted: sleep.accepted,
        accepted,
        decision,
    };
    if accepted {
        (candidate, report)
    } else {
        (original, report)
    }
}

fn detect_bridge_seeds(
    substrate: &NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    config: VacuumBridgeConfig,
) -> Vec<BridgeSeed> {
    let mut outgoing = HashMap::<usize, Vec<(usize, f32, f32)>>::new();
    let mut direct = HashMap::<(usize, usize), f32>::new();
    for (_, source, target, amplitude, phase, coherence, uncertainty, _) in
        substrate.relation_entries()
    {
        let score = amplitude * amplitude * coherence * (1.0 - uncertainty);
        if score >= config.min_relation_score {
            outgoing
                .entry(source)
                .or_default()
                .push((target, score, phase));
            direct
                .entry((source, target))
                .and_modify(|value| *value = value.max(score))
                .or_insert(score);
        }
    }
    let anchors = lessons
        .iter()
        .flat_map(|lesson| lesson.local.iter().chain(lesson.remote.iter()))
        .copied()
        .collect::<HashSet<_>>();
    let mut seeds = Vec::new();
    let mut seen = HashSet::new();
    for (&a, first) in &outgoing {
        for &(b, score_ab, phase_ab) in first {
            let Some(second) = outgoing.get(&b) else {
                continue;
            };
            for &(c, score_bc, phase_bc) in second {
                if a == c
                    || direct.get(&(a, c)).copied().unwrap_or(0.0) >= config.min_relation_score
                {
                    continue;
                }
                let relation_alignment = (phase_ab - phase_bc).cos().max(0.0);
                let node_alignment = substrate
                    .thermal
                    .phase
                    .get(a)
                    .zip(substrate.thermal.phase.get(c))
                    .map(|(left, right)| (*left - *right).cos().max(0.0))
                    .unwrap_or(0.0);
                let alignment = 0.60 * relation_alignment + 0.40 * node_alignment;
                if alignment < config.min_phase_alignment {
                    continue;
                }
                if !(anchors.contains(&a) || anchors.contains(&c)) || !seen.insert((a, c)) {
                    continue;
                }
                let activation = substrate
                    .thermal
                    .activation
                    .get(a)
                    .copied()
                    .unwrap_or(0.0)
                    .max(substrate.thermal.activation.get(c).copied().unwrap_or(0.0));
                seeds.push(BridgeSeed {
                    a,
                    c,
                    prior: (score_ab.min(score_bc) * alignment * (1.0 + 0.25 * activation))
                        .clamp(0.0, 1.0),
                });
            }
        }
    }
    seeds.sort_by(|left, right| right.prior.total_cmp(&left.prior));
    seeds.truncate(config.max_bridges);
    seeds
}

fn reserve_epr_capacity(
    substrate: &mut NativeThermoRqmEprSubstrate,
    a: usize,
    c: usize,
    prospective_slots: usize,
) -> usize {
    substrate
        .entanglement
        .reserve_pair_capacity(a, c, prospective_slots)
}

fn bridge_accuracy(
    substrate: &NativeThermoRqmEprSubstrate,
    seeds: &[BridgeSeed],
    lessons: &[Lesson],
) -> f32 {
    if seeds.is_empty() {
        return 0.0;
    }
    let mut trial = substrate.clone();
    let mut correct = 0;
    for seed in seeds {
        let report = trial.query(DEFAULT_OBSERVER, 0.0, &[seed.a]);
        let expected = report
            .candidates
            .iter()
            .find(|candidate| candidate.agent == seed.c)
            .map(|candidate| candidate.score)
            .unwrap_or(0.0);
        let distractor = lessons
            .iter()
            .flat_map(|lesson| lesson.distractor.iter())
            .filter_map(|node| {
                report
                    .candidates
                    .iter()
                    .find(|candidate| candidate.agent == *node)
                    .map(|candidate| candidate.score)
            })
            .fold(0.0_f32, f32::max);
        correct += usize::from(expected > distractor);
    }
    correct as f32 / seeds.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_thermodynamic_engine::{
        canonical_lessons, fresh_native_substrate, train_canonical,
    };

    #[test]
    fn bridge_gate_preserves_canonical_memory() {
        let lessons = canonical_lessons();
        let mut substrate = fresh_native_substrate();
        train_canonical(&mut substrate, &lessons, 3);
        let (consolidated, _) = native_sleep_consolidate(substrate, &lessons, 4, 2);
        let before = evaluate_native_suite(&consolidated, &lessons);
        let (after, report) = native_residue_vacuum_bridge(
            consolidated,
            &lessons,
            VacuumFluctuationConfig::default(),
            VacuumBridgeConfig::default(),
        );
        let metrics = evaluate_native_suite(&after, &lessons);
        assert!(metrics.accuracy() + 1.0e-6 >= before.accuracy());
        assert!(metrics.leakage() <= before.leakage() + 0.01);
        assert!(!report.decision.is_empty());
    }
}
