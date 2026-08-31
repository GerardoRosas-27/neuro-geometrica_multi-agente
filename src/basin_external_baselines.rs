//! Baselines externos sobre el mismo fixture de cuenca.
//!
//! Misma topología (32 nodos), mismas cues y el mismo presupuesto de
//! iteraciones que `consolidation_basin_experiment`. Hopfield clásico y
//! Hopfield moderno son ajenos al crate; la relajación fasorial sin
//! consolidación CDT formaliza el brazo pre; Hebb escribe fases en las
//! mismas aristas sin el gate de sueño.

use crate::consolidation_basin_experiment::{
    balanced_target, corrupted_phases, evaluate_basin, training_engine, BasinLevelMetrics,
    ConsolidationBasinConfig,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtSubstrate;
use serde::Serialize;
use std::time::Instant;

const ITERATION_BUDGET: usize = 300;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BaselineMethodReport {
    pub method: &'static str,
    pub foreign_to_crate: bool,
    pub wall_clock_seconds: f64,
    pub mean_model_energy: f32,
    pub mean_success_rate: f32,
    pub mean_accuracy: f32,
    pub mean_saturation: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BasinBaselineTable {
    pub nodes: usize,
    pub trials_per_corruption: usize,
    pub corruption_fractions: Vec<f32>,
    pub iteration_budget: usize,
    pub methods: Vec<BaselineMethodReport>,
}

/// Compara Hopfield, Hopfield moderno, fasores sin CDT y Hebb de arista.
pub fn run_basin_external_baselines(
    config: ConsolidationBasinConfig,
) -> Result<BasinBaselineTable, crate::native_hybrid_phasor_cdt_engine::NativeHybridError> {
    let nodes = config.nodes.max(8);
    let target = balanced_target(nodes, config.seed);
    let cues = collect_cues(&target, &config);
    let mut methods = Vec::new();

    methods.push(evaluate_hopfield(
        &target,
        &cues,
        config.success_accuracy,
        false,
    ));
    methods.push(evaluate_hopfield(
        &target,
        &cues,
        config.success_accuracy,
        true,
    ));

    let engine = training_engine(nodes, config.seed)?;
    let pre_started = Instant::now();
    let pre_levels = evaluate_basin(&engine.core, &target, &config);
    methods.push(summarize_phasor_levels(
        "fasorial_sin_consolidacion_cdt",
        false,
        &pre_levels,
        pre_started.elapsed().as_secs_f64(),
    ));

    let mut hebb_core = engine.core.clone();
    hebb_write_edges(&mut hebb_core, &target);
    let hebb_started = Instant::now();
    let hebb_levels = evaluate_basin(&hebb_core, &target, &config);
    methods.push(summarize_phasor_levels(
        "hebb_aristas",
        false,
        &hebb_levels,
        hebb_started.elapsed().as_secs_f64(),
    ));

    Ok(BasinBaselineTable {
        nodes,
        trials_per_corruption: config.trials_per_corruption,
        corruption_fractions: config.corruption_fractions.clone(),
        iteration_budget: ITERATION_BUDGET,
        methods,
    })
}

fn collect_cues(target: &[i8], config: &ConsolidationBasinConfig) -> Vec<Vec<i8>> {
    let mut cues = Vec::new();
    for (level, corruption) in config.corruption_fractions.iter().copied().enumerate() {
        for trial in 0..config.trials_per_corruption {
            let phases = corrupted_phases(
                target,
                corruption,
                config.phase_jitter,
                config.seed ^ (level as u64).rotate_left(17),
                trial,
            );
            cues.push(
                phases
                    .into_iter()
                    .map(|phase| if phase.cos() >= 0.0 { 1 } else { -1 })
                    .collect(),
            );
        }
    }
    cues
}

fn evaluate_hopfield(
    target: &[i8],
    cues: &[Vec<i8>],
    success_accuracy: f32,
    modern: bool,
) -> BaselineMethodReport {
    let started = Instant::now();
    let mut energy_sum = 0.0f32;
    let mut accuracy_sum = 0.0f32;
    let mut saturation_sum = 0.0f32;
    let mut successes = 0usize;
    for cue in cues {
        let (state, energy) = if modern {
            modern_hopfield_retrieve(target, cue)
        } else {
            hopfield_retrieve(target, cue, ITERATION_BUDGET)
        };
        let accuracy = bit_accuracy(&state, target);
        energy_sum += energy;
        accuracy_sum += accuracy;
        saturation_sum += saturation(&state, target);
        if accuracy >= success_accuracy {
            successes += 1;
        }
    }
    let n = cues.len().max(1) as f32;
    BaselineMethodReport {
        method: if modern {
            "hopfield_moderno"
        } else {
            "hopfield"
        },
        foreign_to_crate: true,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        mean_model_energy: energy_sum / n,
        mean_success_rate: successes as f32 / n,
        mean_accuracy: accuracy_sum / n,
        mean_saturation: saturation_sum / n,
    }
}

fn hopfield_retrieve(target: &[i8], cue: &[i8], max_iterations: usize) -> (Vec<i8>, f32) {
    let n = target.len();
    let mut state = cue.to_vec();
    for _ in 0..max_iterations {
        let mut next = vec![0i8; n];
        let mut changed = false;
        for i in 0..n {
            let mut field = 0i32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                field += i32::from(target[i]) * i32::from(target[j]) * i32::from(state[j]);
            }
            next[i] = if field >= 0 { 1 } else { -1 };
            changed |= next[i] != state[i];
        }
        state = next;
        if !changed {
            break;
        }
    }
    let energy = hopfield_energy(target, &state);
    (state, energy)
}

fn modern_hopfield_retrieve(target: &[i8], cue: &[i8]) -> (Vec<i8>, f32) {
    // Un patrón almacenado: ξ ← x softmax(β x·q) colapsa al patrón x.
    let overlap = target
        .iter()
        .zip(cue)
        .map(|(bit, query)| f32::from(*bit) * f32::from(*query))
        .sum::<f32>();
    let state = if overlap >= 0.0 {
        target.to_vec()
    } else {
        target.iter().map(|bit| -*bit).collect()
    };
    let energy = -overlap.abs() / target.len().max(1) as f32;
    (state, energy)
}

fn hopfield_energy(target: &[i8], state: &[i8]) -> f32 {
    let n = target.len();
    let mut energy = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let weight = f32::from(target[i]) * f32::from(target[j]);
            energy -= 0.5 * weight * f32::from(state[i]) * f32::from(state[j]);
        }
    }
    energy
}

fn hebb_write_edges(core: &mut NativeThermoCdtSubstrate, target: &[i8]) {
    for edge in 0..core.edge_phase.len() {
        let left = core.edge_a[edge];
        let right = core.edge_b[edge];
        core.edge_phase[edge] = if target[left] == target[right] {
            0.0
        } else {
            std::f32::consts::PI
        };
    }
}

fn summarize_phasor_levels(
    method: &'static str,
    foreign_to_crate: bool,
    levels: &[BasinLevelMetrics],
    wall_clock_seconds: f64,
) -> BaselineMethodReport {
    let n = levels.len().max(1) as f32;
    BaselineMethodReport {
        method,
        foreign_to_crate,
        wall_clock_seconds,
        mean_model_energy: levels
            .iter()
            .map(|level| level.mean_final_energy)
            .sum::<f32>()
            / n,
        mean_success_rate: levels.iter().map(|level| level.success_rate).sum::<f32>() / n,
        mean_accuracy: levels.iter().map(|level| level.mean_accuracy).sum::<f32>() / n,
        mean_saturation: levels.iter().map(|level| level.mean_accuracy).sum::<f32>() / n,
    }
}

fn bit_accuracy(state: &[i8], target: &[i8]) -> f32 {
    let matches = state
        .iter()
        .zip(target)
        .filter(|(observed, expected)| *observed == *expected)
        .count();
    matches as f32 / target.len().max(1) as f32
}

fn saturation(state: &[i8], target: &[i8]) -> f32 {
    let overlap = state
        .iter()
        .zip(target)
        .map(|(observed, expected)| i32::from(*observed) * i32::from(*expected))
        .sum::<i32>();
    overlap.abs() as f32 / target.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hopfield_recovers_a_single_stored_pattern_below_majority_corruption() {
        let target = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let mut cue = target.clone();
        cue[0] = -cue[0];
        cue[1] = -cue[1];
        let (state, _) = hopfield_retrieve(&target, &cue, 16);
        assert_eq!(state, target);
        let (modern, _) = modern_hopfield_retrieve(&target, &cue);
        assert_eq!(modern, target);
    }

    #[test]
    fn scientific_basin_baselines_publish_two_foreign_methods() {
        let table = run_basin_external_baselines(ConsolidationBasinConfig {
            trials_per_corruption: 4,
            corruption_fractions: vec![0.20, 0.40],
            ..ConsolidationBasinConfig::default()
        })
        .unwrap();
        let foreign = table
            .methods
            .iter()
            .filter(|method| method.foreign_to_crate)
            .map(|method| method.method)
            .collect::<Vec<_>>();
        assert!(
            foreign.contains(&"hopfield") && foreign.contains(&"hopfield_moderno"),
            "{table:#?}"
        );
        assert_eq!(table.iteration_budget, 300, "{table:#?}");
        assert_eq!(table.nodes, 32, "{table:#?}");
        assert!(
            table
                .methods
                .iter()
                .any(|method| method.method == "fasorial_sin_consolidacion_cdt"),
            "{table:#?}"
        );
        eprintln!(
            "scientific_baselines nodes={} budget={} trials={}",
            table.nodes, table.iteration_budget, table.trials_per_corruption
        );
        for method in &table.methods {
            eprintln!(
                "  {} foreign={} wall={:.4}s energy={:.4} succ={:.3} acc={:.3} sat={:.3}",
                method.method,
                method.foreign_to_crate,
                method.wall_clock_seconds,
                method.mean_model_energy,
                method.mean_success_rate,
                method.mean_accuracy,
                method.mean_saturation
            );
        }
    }
}
