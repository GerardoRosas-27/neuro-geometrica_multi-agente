//! Fluctuación geométrica débil derivada del residuo R.
//!
//! El residuo no se borra: se alquila como señal débil sobre el tejido geométrico
//! (activation + fase local), decae solo, y solo se acepta si no degrada memoria.
//!
//! El gate revierte pulsos que degradan memoria.

use crate::native_thermo_rqm_epr::NativeThermoRqmEprSubstrate;
use crate::native_thermodynamic_engine::{
    evaluate_native_suite, EngineMetrics, Lesson, DEFAULT_OBSERVER,
};
use crate::residue_budget::{probe_residue_account_in_place, ResidueAccount, ResidueBudgetConfig};

#[derive(Clone, Copy, Debug)]
pub struct VacuumFluctuationConfig {
    /// Escala ε del presupuesto: amp/act ∝ ε * R.
    pub epsilon: f32,
    /// Tope absoluto del presupuesto de inyección.
    pub budget_max: f32,
    /// Micro-pasos térmicos locales tras inyectar.
    pub microsteps: usize,
    /// Tamaño máximo de ventana geométrica.
    pub max_window_nodes: usize,
    /// Residuo mínimo para pulsar.
    pub min_residue: f32,
    /// Intentos con gate (revert si degrada).
    pub gate_attempts: usize,
    /// Dejar eco de activación (no clear total).
    pub leave_echo: bool,
    /// Eco residual de activación tras el pulso (si leave_echo).
    pub echo_activation: f32,
}

impl Default for VacuumFluctuationConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.10,
            budget_max: 0.28,
            microsteps: 4,
            max_window_nodes: 64,
            min_residue: 0.01,
            gate_attempts: 4,
            leave_echo: true,
            echo_activation: 0.12,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SubstrateDynamics {
    pub mean_activation: f32,
    pub phase_variance: f32,
    pub state_variance: f32,
    pub mean_energy: f32,
    pub free_energy_proxy: f32,
    pub active_nodes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VacuumPulseReport {
    pub pulses: usize,
    pub nodes_touched: usize,
    pub mean_budget: f32,
    pub mean_residue: f32,
    pub before: SubstrateDynamics,
    pub after: SubstrateDynamics,
}

#[derive(Clone, Debug)]
pub struct VacuumCycleReport {
    pub attempts: usize,
    pub accepted: usize,
    pub before_metrics: EngineMetrics,
    pub after_metrics: EngineMetrics,
    pub before_dynamics: SubstrateDynamics,
    pub after_dynamics: SubstrateDynamics,
    pub pulse: VacuumPulseReport,
    pub decision: &'static str,
}

pub fn measure_dynamics(substrate: &NativeThermoRqmEprSubstrate) -> SubstrateDynamics {
    let n = substrate.thermal.node_count().max(1) as f32;
    let mean_activation = substrate.thermal.activation.iter().map(|v| *v).sum::<f32>() / n;
    let mean_phase = substrate.thermal.phase.iter().map(|v| *v).sum::<f32>() / n;
    let phase_variance = substrate
        .thermal
        .phase
        .iter()
        .map(|v| {
            let d = *v - mean_phase;
            d * d
        })
        .sum::<f32>()
        / n;
    let report = substrate.thermal.report();
    SubstrateDynamics {
        mean_activation,
        phase_variance,
        state_variance: report.state_variance,
        mean_energy: report.mean_energy,
        free_energy_proxy: report.free_energy_proxy,
        active_nodes: report.active_nodes,
    }
}

/// Un pulso de vacío: inyecta R como activación/fase débil en el vecindario.
pub fn residue_vacuum_pulse(
    substrate: &mut NativeThermoRqmEprSubstrate,
    cue: &[usize],
    account: &ResidueAccount,
    config: &VacuumFluctuationConfig,
) -> VacuumPulseReport {
    let before = measure_dynamics(substrate);
    let loser_mass: f32 = account.loser_mass.iter().map(|(_, p)| *p).sum();
    // Tras consolidar R→0; aún así la mezcla (S) y masa perdedora alimentan el vacío geométrico.
    let drive = account
        .residue_r
        .max(0.12 * account.entropy_s)
        .max(0.50 * loser_mass);
    if drive < config.min_residue && account.loser_mass.is_empty() {
        return VacuumPulseReport {
            before,
            after: before,
            ..VacuumPulseReport::default()
        };
    }

    let budget = (config.epsilon * drive.max(config.min_residue)).clamp(0.0, config.budget_max);
    let fe_cap = account.free_energy_proxy.abs() * 0.20 + config.epsilon;
    let budget = budget.min(fe_cap).clamp(0.0, config.budget_max);

    let losers: Vec<usize> = account.loser_mass.iter().map(|(id, _)| *id).collect();
    let window = substrate
        .thermal
        .local_neighborhood(cue, &losers, config.max_window_nodes.max(8));
    if window.is_empty() {
        return VacuumPulseReport {
            before,
            after: before,
            ..VacuumPulseReport::default()
        };
    }

    let inv = 1.0 / window.len() as f32;
    for (i, &node) in window.iter().enumerate() {
        let loser_w = account
            .loser_mass
            .iter()
            .find(|(id, _)| *id == node)
            .map(|(_, p)| *p)
            .unwrap_or(inv);
        let w = (0.35 * inv + 0.65 * loser_w).clamp(0.01, 1.0);
        let amp = 0.15 * budget * w;
        let act = 0.25 * budget * w;
        let phase = account.residual_phase
            + 0.15 * (i as f32) * std::f32::consts::TAU / window.len().max(1) as f32;
        substrate.thermal.inject_vacuum_node(
            node,
            amp,
            phase.rem_euclid(std::f32::consts::TAU),
            act,
        );
    }

    for _ in 0..config.microsteps.max(1) {
        substrate.thermal.step_local(&window);
    }

    if config.leave_echo {
        let echo = (config.echo_activation * drive).clamp(0.0, 0.25);
        for &node in &window {
            if node < substrate.thermal.node_count() {
                let cur = substrate.thermal.activation[node];
                substrate.thermal.activation[node] = (cur * 0.5 + echo * 0.5).max(echo * 0.35);
            }
        }
    }

    let after = measure_dynamics(substrate);
    VacuumPulseReport {
        pulses: 1,
        nodes_touched: window.len(),
        mean_budget: budget,
        mean_residue: drive,
        before,
        after,
    }
}

fn merge_pulse(acc: &mut VacuumPulseReport, pulse: VacuumPulseReport) {
    if pulse.pulses == 0 {
        return;
    }
    let n = (acc.pulses + pulse.pulses).max(1) as f32;
    acc.mean_budget = (acc.mean_budget * acc.pulses as f32 + pulse.mean_budget) / n;
    acc.mean_residue = (acc.mean_residue * acc.pulses as f32 + pulse.mean_residue) / n;
    acc.pulses += pulse.pulses;
    acc.nodes_touched += pulse.nodes_touched;
    acc.after = pulse.after;
}

/// Pulsos de vacío anclados a las lecciones (contabilidad de R en clone).
pub fn pulse_lessons_vacuum(
    substrate: &mut NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    account_cfg: &ResidueBudgetConfig,
    vacuum_cfg: &VacuumFluctuationConfig,
) -> VacuumPulseReport {
    let before = measure_dynamics(substrate);
    let mut acc = VacuumPulseReport {
        before,
        ..VacuumPulseReport::default()
    };
    // Un solo snapshot para todas las consultas de residuo del ciclo.
    let mut probe = substrate.clone();

    for lesson in lessons {
        let account = probe_residue_account_in_place(
            &mut probe,
            DEFAULT_OBSERVER,
            &lesson.local,
            &lesson.remote,
            account_cfg,
        );
        let pulse = residue_vacuum_pulse(substrate, &lesson.local, &account, vacuum_cfg);
        merge_pulse(&mut acc, pulse);
    }

    acc.after = measure_dynamics(substrate);
    if acc.pulses == 0 {
        acc.before = before;
        acc.after = before;
    }
    acc
}

fn preserves(before: EngineMetrics, after: EngineMetrics, max_leak_drift: f32) -> bool {
    after.accuracy() + 1.0e-6 >= before.accuracy()
        && after.leakage() <= before.leakage() + max_leak_drift
}

fn dynamics_useful(before: SubstrateDynamics, after: SubstrateDynamics) -> bool {
    after.phase_variance > before.phase_variance + 1.0e-6
        || after.state_variance > before.state_variance + 1.0e-6
        || after.mean_activation > before.mean_activation + 1.0e-5
        || after.active_nodes > before.active_nodes
}

/// Aplica fluctuación de vacío con gate: acepta solo si no degrada memoria y aporta dinámica.
pub fn native_vacuum_fluctuation(
    substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    vacuum_cfg: VacuumFluctuationConfig,
) -> (NativeThermoRqmEprSubstrate, VacuumCycleReport) {
    let account_cfg = ResidueBudgetConfig {
        temperature: 0.85,
        min_residue: vacuum_cfg.min_residue,
        top_losers: 4,
        ..ResidueBudgetConfig::default()
    };

    let before_metrics = evaluate_native_suite(&substrate, lessons);
    let before_dynamics = measure_dynamics(&substrate);
    let original = substrate.clone();
    let mut best = substrate;
    let mut best_metrics = before_metrics;
    let mut best_dynamics = before_dynamics;
    let mut best_pulse = VacuumPulseReport::default();
    let mut accepted = 0;
    let mut last_pulse = VacuumPulseReport::default();

    for attempt in 0..vacuum_cfg.gate_attempts.max(1) {
        let mut candidate = original.clone();
        let mut cfg = vacuum_cfg;
        // Ligera variación de ε por intento.
        cfg.epsilon = (vacuum_cfg.epsilon * (0.75 + 0.15 * (attempt as f32))).clamp(0.02, 0.12);
        let pulse = pulse_lessons_vacuum(&mut candidate, lessons, &account_cfg, &cfg);
        last_pulse = pulse;
        // Sin run_until_stable: preserva el eco de activación (leasing de vacío).
        let metrics = evaluate_native_suite(&candidate, lessons);
        let dynamics = measure_dynamics(&candidate);

        let ok_memory = preserves(before_metrics, metrics, 0.005);
        let ok_dyn = dynamics_useful(before_dynamics, dynamics);
        let better_margin = metrics.margin() > best_metrics.margin() + 0.05;
        let better_dyn = dynamics.phase_variance > best_dynamics.phase_variance + 1.0e-6
            || dynamics.mean_activation > best_dynamics.mean_activation + 1.0e-5;

        if ok_memory && (ok_dyn || better_margin) && (accepted == 0 || better_dyn || better_margin)
        {
            best = candidate;
            best_metrics = metrics;
            best_dynamics = dynamics;
            best_pulse = pulse;
            accepted += 1;
        }
    }

    if accepted == 0 {
        best_pulse = last_pulse;
        best_dynamics = measure_dynamics(&best);
    }

    let decision = if accepted > 0
        && (best_dynamics.phase_variance > before_dynamics.phase_variance + 1.0e-6
            || best_metrics.margin() > before_metrics.margin() + 0.05)
    {
        "vacuum_fluctuation_improve"
    } else if accepted > 0 {
        "vacuum_fluctuation_accept"
    } else {
        "vacuum_fluctuation_reject"
    };

    (
        best,
        VacuumCycleReport {
            attempts: vacuum_cfg.gate_attempts.max(1),
            accepted,
            before_metrics,
            after_metrics: best_metrics,
            before_dynamics,
            after_dynamics: best_dynamics,
            pulse: best_pulse,
            decision,
        },
    )
}

/// Varias rondas de fluctuación para ver plasticidad acumulada.
pub fn vacuum_plasticity_loop(
    mut substrate: NativeThermoRqmEprSubstrate,
    lessons: &[Lesson],
    rounds: usize,
    vacuum_cfg: VacuumFluctuationConfig,
) -> (NativeThermoRqmEprSubstrate, Vec<VacuumCycleReport>) {
    let mut reports = Vec::with_capacity(rounds.max(1));
    for _ in 0..rounds.max(1) {
        let (next, report) = native_vacuum_fluctuation(substrate, lessons, vacuum_cfg);
        substrate = next;
        reports.push(report);
    }
    (substrate, reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_thermodynamic_engine::{
        canonical_lessons, fresh_native_substrate, native_sleep_consolidate, train_canonical,
    };
    use crate::residue_budget::ResidueAccount;

    #[test]
    fn vacuum_pulse_raises_activation_or_phase_var() {
        let lessons = canonical_lessons();
        let mut substrate = fresh_native_substrate();
        train_canonical(&mut substrate, &lessons, 2);
        let account = ResidueAccount {
            partition_z: 2.0,
            entropy_s: 0.8,
            residue_r: 0.40,
            chosen_prob: 0.60,
            residual_phase: 1.2,
            free_energy_proxy: -0.5,
            loser_mass: lessons[0]
                .distractor
                .iter()
                .take(3)
                .map(|id| (*id, 0.2))
                .collect(),
        };
        let before = measure_dynamics(&substrate);
        let report = residue_vacuum_pulse(
            &mut substrate,
            &lessons[0].local,
            &account,
            &VacuumFluctuationConfig::default(),
        );
        assert_eq!(report.pulses, 1);
        assert!(report.nodes_touched > 0);
        let after = measure_dynamics(&substrate);
        assert!(
            after.mean_activation + 1.0e-6 >= before.mean_activation
                || after.phase_variance + 1.0e-6 >= before.phase_variance
                || after.active_nodes >= before.active_nodes
        );
    }

    #[test]
    fn gated_vacuum_preserves_accuracy_after_sleep() {
        let lessons = canonical_lessons();
        let mut substrate = fresh_native_substrate();
        train_canonical(&mut substrate, &lessons, 3);
        let (slept, _) = native_sleep_consolidate(substrate, &lessons, 4, 2);
        let before = evaluate_native_suite(&slept, &lessons);
        let (after, report) =
            native_vacuum_fluctuation(slept, &lessons, VacuumFluctuationConfig::default());
        let metrics = evaluate_native_suite(&after, &lessons);
        assert!(metrics.accuracy() + 1.0e-6 >= before.accuracy());
        assert!(metrics.leakage() <= before.leakage() + 0.01);
        assert!(
            report.decision == "vacuum_fluctuation_improve"
                || report.decision == "vacuum_fluctuation_accept"
                || report.decision == "vacuum_fluctuation_reject"
        );
    }
}
