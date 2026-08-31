//! Baselines externos sobre el mismo fixture de cuenca.
//!
//! Misma topología, mismas cues y el mismo presupuesto de iteraciones que
//! `consolidation_basin_experiment`. Hopfield clásico y Hopfield exponencial
//! son ajenos al crate; Hebb de aristas es el acumulador R1; el fasorial
//! sin consolidación formaliza el brazo pre. Las energías viven dentro del
//! funcional de cada método y no se comparan entre sí.

use crate::consolidation_basin_experiment::{
    balanced_target, commit_with_retention, corrupted_phases, evaluate_basin, mean_success,
    training_engine, BasinLevelMetrics, BoundedMemoryConfig, BoundedPhasorMemory,
    ConsolidationBasinConfig,
};
use crate::native_hybrid_phasor_cdt_engine::{
    apply_pattern_additive, clear_pattern_accumulator, NativeHybridError,
};
use crate::native_thermodynamic_cdt::NativeThermoCdtSubstrate;
use serde::Serialize;
use std::time::Instant;

const ITERATION_BUDGET: usize = 300;
const EXPONENTIAL_BETA: f32 = 8.0;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BaselineMethodReport {
    pub method: &'static str,
    pub foreign_to_crate: bool,
    pub wall_clock_seconds: f64,
    pub mean_model_energy: f32,
    pub mean_success_rate: f32,
    pub mean_accuracy: f32,
    /// Saturación = |solapamiento| / N, no una copia de la exactitud directa.
    pub mean_saturation: f32,
    pub note: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BasinBaselineTable {
    pub nodes: usize,
    pub trials_per_corruption: usize,
    pub corruption_fractions: Vec<f32>,
    pub iteration_budget: usize,
    pub methods: Vec<BaselineMethodReport>,
}

/// Compara Hopfield, Hopfield exponencial, fasores sin consolidación y Hebb acumulativo.
pub fn run_basin_external_baselines(
    config: ConsolidationBasinConfig,
) -> Result<BasinBaselineTable, NativeHybridError> {
    let nodes = config.nodes.max(8);
    let target = balanced_target(nodes, config.seed);
    let patterns = [target.clone()];
    let cues = collect_cues(&target, &config);
    let mut methods = Vec::new();

    methods.push(evaluate_hopfield_family(
        &patterns,
        &cues,
        config.success_accuracy,
        HopfieldFamily::Classical,
    ));
    methods.push(evaluate_hopfield_family(
        &patterns,
        &cues,
        config.success_accuracy,
        HopfieldFamily::Exponential {
            beta: EXPONENTIAL_BETA,
        },
    ));

    let engine = training_engine(nodes, config.seed)?;
    let pre_started = Instant::now();
    let pre_levels = evaluate_basin(&engine.core, &target, &config);
    methods.push(summarize_phasor_levels(
        "fasorial_sin_consolidacion",
        false,
        &pre_levels,
        pre_started.elapsed().as_secs_f64(),
        "cdt_consolidation_steps=0; no se etiqueta CDT",
    ));

    let mut hebb_core = engine.core.clone();
    clear_pattern_accumulator(&mut hebb_core);
    apply_pattern_additive(&mut hebb_core, &target, 0.0, None, None);
    let hebb_started = Instant::now();
    let hebb_levels = evaluate_basin(&hebb_core, &target, &config);
    methods.push(summarize_phasor_levels(
        "hebb_aristas",
        false,
        &hebb_levels,
        hebb_started.elapsed().as_secs_f64(),
        "acumulador R1; con un patrón coincide con la asignación",
    ));

    Ok(BasinBaselineTable {
        nodes,
        trials_per_corruption: config.trials_per_corruption,
        corruption_fractions: config.corruption_fractions.clone(),
        iteration_budget: ITERATION_BUDGET,
        methods,
    })
}

#[derive(Clone, Debug)]
pub struct CapacityCurveConfig {
    pub nodes: usize,
    pub pattern_counts: Vec<usize>,
    pub corruption_fractions: Vec<f32>,
    pub trials_per_corruption: usize,
    pub seeds: Vec<u64>,
    pub epsilon: f32,
    pub theta_candidate: f32,
    pub occupancy_floor: f32,
    pub exponential_beta: f32,
}

impl Default for CapacityCurveConfig {
    fn default() -> Self {
        Self {
            nodes: 32,
            pattern_counts: (1..=8).collect(),
            corruption_fractions: vec![0.20, 0.30],
            trials_per_corruption: 4,
            seeds: vec![
                0xCA12_0001,
                0xCA12_0002,
                0xCA12_0003,
                0xCA12_0004,
                0xCA12_0005,
                0xCA12_0006,
                0xCA12_0007,
                0xCA12_0008,
            ],
            epsilon: 0.10,
            theta_candidate: 0.80,
            occupancy_floor: 0.50,
            exponential_beta: EXPONENTIAL_BETA,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapacityCurveRow {
    pub nodes: usize,
    pub patterns: usize,
    pub corruption_fraction: f32,
    pub method: &'static str,
    pub recovery_mean: f32,
    pub recovery_std: f32,
    pub delta_r_mean: f32,
    pub delta_r_std: f32,
    pub wall_clock_seconds: f64,
    pub mean_model_energy: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapacityCurveReport {
    pub nodes: usize,
    pub k_max_phasor: usize,
    pub rows: Vec<CapacityCurveRow>,
    pub wall_clock_seconds: f64,
}

/// Curva K(N, ρ) del fasorial con commit R1, Hebb acumulativo, Hopfield
/// clásico y Hopfield exponencial (softmax sobre K, β versionado).
pub fn run_capacity_curve(
    config: CapacityCurveConfig,
) -> Result<CapacityCurveReport, NativeHybridError> {
    let started = Instant::now();
    let nodes = config.nodes.max(8);
    let mut rows = Vec::new();
    for &rho in &config.corruption_fractions {
        let eval = ConsolidationBasinConfig {
            nodes,
            trials_per_corruption: config.trials_per_corruption.max(1),
            corruption_fractions: vec![rho],
            seed: config.seeds.first().copied().unwrap_or(0xCA12_0001),
            ..ConsolidationBasinConfig::default()
        };
        for &k in &config.pattern_counts {
            let k = k.max(1);
            rows.extend(capacity_at_k(&config, &eval, nodes, k, rho)?);
        }
    }
    let k_max_phasor = k_max_phasor_from_rows(&rows, config.epsilon);
    Ok(CapacityCurveReport {
        nodes,
        k_max_phasor,
        rows,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
    })
}

fn capacity_at_k(
    config: &CapacityCurveConfig,
    eval: &ConsolidationBasinConfig,
    nodes: usize,
    k: usize,
    rho: f32,
) -> Result<Vec<CapacityCurveRow>, NativeHybridError> {
    let mut phasor_rec = Vec::new();
    let mut phasor_dr = Vec::new();
    let mut phasor_energy = Vec::new();
    let mut hebb_rec = Vec::new();
    let mut hebb_dr = Vec::new();
    let mut hebb_energy = Vec::new();
    let mut hop_rec = Vec::new();
    let mut hop_dr = Vec::new();
    let mut hop_energy = Vec::new();
    let mut exp_rec = Vec::new();
    let mut exp_dr = Vec::new();
    let mut exp_energy = Vec::new();
    let phasor_started = Instant::now();
    let mut hebb_time = 0.0;
    let mut hop_time = 0.0;
    let mut exp_time = 0.0;

    for (seed_index, &seed) in config.seeds.iter().enumerate() {
        let patterns: Vec<Vec<i8>> = (0..k)
            .map(|index| {
                balanced_target(
                    nodes,
                    seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )
            })
            .collect();
        let eval_seeded = ConsolidationBasinConfig {
            seed: seed ^ (seed_index as u64).rotate_left(11),
            ..eval.clone()
        };

        let mem_config = BoundedMemoryConfig {
            epsilon: config.epsilon,
            theta_candidate: config.theta_candidate,
            max_patterns: None,
            replay_rounds: 1,
            occupancy_floor: config.occupancy_floor,
            replay_write_rate: 0.0,
            evaluation: eval_seeded.clone(),
        };
        let engine = training_engine(nodes, seed)?;
        let mut memory = BoundedPhasorMemory::from_core(engine.core.clone());
        let mut last_delta = 0.0f32;
        for pattern in &patterns {
            let commit = commit_with_retention(&mut memory, pattern, &mem_config)?;
            last_delta = commit.delta_r;
            if commit.decision != "accepted" {
                break;
            }
        }
        let (rec, energy) = recovery_on_core(&memory.core, &patterns, &eval_seeded);
        phasor_rec.push(rec);
        phasor_dr.push(last_delta);
        phasor_energy.push(energy);

        let hebb_t0 = Instant::now();
        let mut hebb_core = engine.core.clone();
        clear_pattern_accumulator(&mut hebb_core);
        let mut hebb_delta = 0.0f32;
        for (index, pattern) in patterns.iter().enumerate() {
            let before = if index == 0 {
                0.0
            } else {
                mean_success_patterns(&hebb_core, &patterns[..index], &eval_seeded)
            };
            apply_pattern_additive(&mut hebb_core, pattern, 0.0, None, None);
            if index > 0 {
                let after = mean_success_patterns(&hebb_core, &patterns[..index], &eval_seeded);
                hebb_delta = after - before;
            }
        }
        hebb_time += hebb_t0.elapsed().as_secs_f64();
        let (rec, energy) = recovery_on_core(&hebb_core, &patterns, &eval_seeded);
        hebb_rec.push(rec);
        hebb_dr.push(hebb_delta);
        hebb_energy.push(energy);

        let hop_t0 = Instant::now();
        let (rec, energy, delta) =
            hopfield_capacity_metrics(&patterns, &eval_seeded, HopfieldFamily::Classical);
        hop_time += hop_t0.elapsed().as_secs_f64();
        hop_rec.push(rec);
        hop_dr.push(delta);
        hop_energy.push(energy);

        let exp_t0 = Instant::now();
        let (rec, energy, delta) = hopfield_capacity_metrics(
            &patterns,
            &eval_seeded,
            HopfieldFamily::Exponential {
                beta: config.exponential_beta,
            },
        );
        exp_time += exp_t0.elapsed().as_secs_f64();
        exp_rec.push(rec);
        exp_dr.push(delta);
        exp_energy.push(energy);
    }

    Ok(vec![
        summarize_capacity_row(
            nodes,
            k,
            rho,
            "fasorial_commit",
            &phasor_rec,
            &phasor_dr,
            &phasor_energy,
            phasor_started.elapsed().as_secs_f64(),
        ),
        summarize_capacity_row(
            nodes,
            k,
            rho,
            "hebb_acumulativo",
            &hebb_rec,
            &hebb_dr,
            &hebb_energy,
            hebb_time,
        ),
        summarize_capacity_row(
            nodes,
            k,
            rho,
            "hopfield",
            &hop_rec,
            &hop_dr,
            &hop_energy,
            hop_time,
        ),
        summarize_capacity_row(
            nodes,
            k,
            rho,
            "hopfield_exponencial",
            &exp_rec,
            &exp_dr,
            &exp_energy,
            exp_time,
        ),
    ])
}

fn recovery_on_core(
    core: &NativeThermoCdtSubstrate,
    patterns: &[Vec<i8>],
    eval: &ConsolidationBasinConfig,
) -> (f32, f32) {
    if patterns.is_empty() {
        return (0.0, 0.0);
    }
    let mut rec = 0.0;
    let mut energy = 0.0;
    for pattern in patterns {
        let levels = evaluate_basin(core, pattern, eval);
        rec += mean_success(&levels);
        energy += levels
            .iter()
            .map(|level| level.mean_final_energy)
            .sum::<f32>()
            / levels.len().max(1) as f32;
    }
    let n = patterns.len() as f32;
    (rec / n, energy / n)
}

fn mean_success_patterns(
    core: &NativeThermoCdtSubstrate,
    patterns: &[Vec<i8>],
    eval: &ConsolidationBasinConfig,
) -> f32 {
    if patterns.is_empty() {
        return 0.0;
    }
    patterns
        .iter()
        .map(|pattern| mean_success(&evaluate_basin(core, pattern, eval)))
        .sum::<f32>()
        / patterns.len() as f32
}

fn hopfield_capacity_metrics(
    patterns: &[Vec<i8>],
    eval: &ConsolidationBasinConfig,
    family: HopfieldFamily,
) -> (f32, f32, f32) {
    if patterns.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let retained = if patterns.len() <= 1 {
        &[][..]
    } else {
        &patterns[..patterns.len() - 1]
    };
    let rec_before = hopfield_mean_recovery(retained, eval, family);
    let rec_after = hopfield_mean_recovery(patterns, eval, family);
    let delta = if retained.is_empty() {
        0.0
    } else {
        hopfield_mean_recovery_of(patterns, retained, eval, family) - rec_before
    };
    let energy = hopfield_mean_energy(patterns, eval, family);
    (rec_after, energy, delta)
}

fn hopfield_mean_recovery(
    stored: &[Vec<i8>],
    eval: &ConsolidationBasinConfig,
    family: HopfieldFamily,
) -> f32 {
    hopfield_mean_recovery_of(stored, stored, eval, family)
}

fn hopfield_mean_recovery_of(
    stored: &[Vec<i8>],
    query_patterns: &[Vec<i8>],
    eval: &ConsolidationBasinConfig,
    family: HopfieldFamily,
) -> f32 {
    if stored.is_empty() || query_patterns.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for target in query_patterns {
        let cues = collect_cues(target, eval);
        for cue in &cues {
            let (state, _) = retrieve(stored, cue, family);
            let accuracy = bit_accuracy(&state, target);
            if accuracy >= eval.success_accuracy {
                total += 1.0;
            }
            count += 1.0;
        }
    }
    if count <= 0.0 {
        0.0
    } else {
        total / count
    }
}

fn hopfield_mean_energy(
    stored: &[Vec<i8>],
    eval: &ConsolidationBasinConfig,
    family: HopfieldFamily,
) -> f32 {
    if stored.is_empty() {
        return 0.0;
    }
    let mut energy = 0.0;
    let mut count = 0.0;
    for target in stored {
        let cues = collect_cues(target, eval);
        for cue in &cues {
            let (_, e) = retrieve(stored, cue, family);
            energy += e;
            count += 1.0;
        }
    }
    if count <= 0.0 {
        0.0
    } else {
        energy / count
    }
}

#[allow(clippy::too_many_arguments)]
fn summarize_capacity_row(
    nodes: usize,
    patterns: usize,
    rho: f32,
    method: &'static str,
    rec: &[f32],
    delta: &[f32],
    energy: &[f32],
    wall_clock_seconds: f64,
) -> CapacityCurveRow {
    let (recovery_mean, recovery_std) = mean_std(rec);
    let (delta_r_mean, delta_r_std) = mean_std(delta);
    let (mean_model_energy, _) = mean_std(energy);
    CapacityCurveRow {
        nodes,
        patterns,
        corruption_fraction: rho,
        method,
        recovery_mean,
        recovery_std,
        delta_r_mean,
        delta_r_std,
        wall_clock_seconds,
        mean_model_energy,
    }
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let var = values
        .iter()
        .map(|value| (*value - mean).powi(2))
        .sum::<f32>()
        / (values.len() - 1) as f32;
    (mean, var.max(0.0).sqrt())
}

fn k_max_phasor_from_rows(rows: &[CapacityCurveRow], epsilon: f32) -> usize {
    let mut k_max = 0usize;
    for k in 1..=8 {
        let phasor: Vec<&CapacityCurveRow> = rows
            .iter()
            .filter(|row| row.patterns == k && row.method == "fasorial_commit")
            .collect();
        if phasor.is_empty() {
            continue;
        }
        let hopfield: Vec<&CapacityCurveRow> = rows
            .iter()
            .filter(|row| row.patterns == k && row.method == "hopfield")
            .collect();
        let meets = phasor
            .iter()
            .all(|row| row.delta_r_mean + 1.0e-6 >= -epsilon)
            && phasor.iter().any(|row| {
                let hop = hopfield
                    .iter()
                    .find(|other| {
                        (other.corruption_fraction - row.corruption_fraction).abs() < 1.0e-6
                    })
                    .map(|other| other.recovery_mean)
                    .unwrap_or(row.recovery_mean);
                row.recovery_mean + 1.0e-6 >= hop - 0.10
            });
        if meets {
            k_max = k;
        }
    }
    k_max
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

#[derive(Clone, Copy, Debug)]
enum HopfieldFamily {
    Classical,
    Exponential { beta: f32 },
}

fn evaluate_hopfield_family(
    patterns: &[Vec<i8>],
    cues: &[Vec<i8>],
    success_accuracy: f32,
    family: HopfieldFamily,
) -> BaselineMethodReport {
    let started = Instant::now();
    let target = &patterns[0];
    let mut energy_sum = 0.0f32;
    let mut accuracy_sum = 0.0f32;
    let mut saturation_sum = 0.0f32;
    let mut successes = 0usize;
    for cue in cues {
        let (state, energy) = retrieve(patterns, cue, family);
        let accuracy = bit_accuracy(&state, target);
        energy_sum += energy;
        accuracy_sum += accuracy;
        saturation_sum += saturation(&state, target);
        if accuracy >= success_accuracy {
            successes += 1;
        }
    }
    let n = cues.len().max(1) as f32;
    let (method, note) = match family {
        HopfieldFamily::Classical => ("hopfield", "funcional −½ ξᵀWξ; no comparar con F fasorial"),
        HopfieldFamily::Exponential { .. } => (
            "hopfield_exponencial",
            "softmax sobre K recuerdos, β versionado; con K=1 colapsa al único patrón",
        ),
    };
    BaselineMethodReport {
        method,
        foreign_to_crate: true,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        mean_model_energy: energy_sum / n,
        mean_success_rate: successes as f32 / n,
        mean_accuracy: accuracy_sum / n,
        mean_saturation: saturation_sum / n,
        note,
    }
}

fn retrieve(patterns: &[Vec<i8>], cue: &[i8], family: HopfieldFamily) -> (Vec<i8>, f32) {
    match family {
        HopfieldFamily::Classical => hopfield_retrieve(patterns, cue, ITERATION_BUDGET),
        HopfieldFamily::Exponential { beta } => exponential_hopfield_retrieve(patterns, cue, beta),
    }
}

fn hopfield_retrieve(patterns: &[Vec<i8>], cue: &[i8], max_iterations: usize) -> (Vec<i8>, f32) {
    if patterns.is_empty() {
        return (cue.to_vec(), 0.0);
    }
    let n = cue.len();
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
                let mut weight = 0i32;
                for pattern in patterns {
                    weight += i32::from(pattern[i]) * i32::from(pattern[j]);
                }
                field += weight * i32::from(state[j]);
            }
            next[i] = if field >= 0 { 1 } else { -1 };
            changed |= next[i] != state[i];
        }
        state = next;
        if !changed {
            break;
        }
    }
    let energy = hopfield_energy(patterns, &state);
    (state, energy)
}

fn exponential_hopfield_retrieve(patterns: &[Vec<i8>], cue: &[i8], beta: f32) -> (Vec<i8>, f32) {
    if patterns.is_empty() {
        return (cue.to_vec(), 0.0);
    }
    let n = cue.len().max(1) as f32;
    let logits: Vec<f32> = patterns
        .iter()
        .map(|pattern| {
            let overlap = pattern
                .iter()
                .zip(cue)
                .map(|(bit, query)| f32::from(*bit) * f32::from(*query))
                .sum::<f32>();
            beta * overlap / n
        })
        .collect();
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect();
    let z = exps.iter().sum::<f32>().max(1.0e-12);
    let mut mixed = vec![0.0f32; cue.len()];
    for (pattern, weight) in patterns.iter().zip(&exps) {
        let p = *weight / z;
        for (slot, bit) in mixed.iter_mut().zip(pattern) {
            *slot += p * f32::from(*bit);
        }
    }
    let state = mixed
        .iter()
        .map(|value| if *value >= 0.0 { 1 } else { -1 })
        .collect::<Vec<_>>();
    let energy = -(z.ln() + max_logit);
    (state, energy)
}

fn hopfield_energy(patterns: &[Vec<i8>], state: &[i8]) -> f32 {
    let n = state.len();
    let mut energy = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut weight = 0.0f32;
            for pattern in patterns {
                weight += f32::from(pattern[i]) * f32::from(pattern[j]);
            }
            energy -= 0.5 * weight * f32::from(state[i]) * f32::from(state[j]);
        }
    }
    energy
}

fn summarize_phasor_levels(
    method: &'static str,
    foreign_to_crate: bool,
    levels: &[BasinLevelMetrics],
    wall_clock_seconds: f64,
    note: &'static str,
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
        mean_saturation: levels
            .iter()
            .map(|level| level.mean_gauge_invariant_accuracy)
            .sum::<f32>()
            / n,
        note,
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
        let (state, _) = hopfield_retrieve(std::slice::from_ref(&target), &cue, 16);
        assert_eq!(state, target);
        let (modern, _) = exponential_hopfield_retrieve(std::slice::from_ref(&target), &cue, 8.0);
        assert_eq!(modern, target);
    }

    #[test]
    fn exponential_hopfield_selects_among_two_memories() {
        let first = vec![1, 1, 1, 1, -1, -1, -1, -1];
        let second = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let mut cue = first.clone();
        cue[0] = -cue[0];
        let (state, _) = exponential_hopfield_retrieve(&[first.clone(), second], &cue, 8.0);
        assert_eq!(state, first);
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
            foreign.contains(&"hopfield") && foreign.contains(&"hopfield_exponencial"),
            "{table:#?}"
        );
        assert_eq!(table.iteration_budget, 300, "{table:#?}");
        assert_eq!(table.nodes, 32, "{table:#?}");
        assert!(
            table
                .methods
                .iter()
                .any(|method| method.method == "fasorial_sin_consolidacion"),
            "{table:#?}"
        );
        let phasor = table
            .methods
            .iter()
            .find(|method| method.method == "fasorial_sin_consolidacion")
            .unwrap();
        assert!(
            (phasor.mean_saturation - phasor.mean_accuracy).abs() > 1.0e-9
                || phasor.mean_success_rate < 0.5,
            "la saturación no debe copiar la exactitud del brazo al azar: {table:#?}"
        );
        eprintln!(
            "scientific_baselines nodes={} budget={} trials={}",
            table.nodes, table.iteration_budget, table.trials_per_corruption
        );
        for method in &table.methods {
            eprintln!(
                "  {} foreign={} wall={:.4}s energy={:.4} succ={:.3} acc={:.3} sat={:.3} note={}",
                method.method,
                method.foreign_to_crate,
                method.wall_clock_seconds,
                method.mean_model_energy,
                method.mean_success_rate,
                method.mean_accuracy,
                method.mean_saturation,
                method.note
            );
        }
    }

    #[test]
    fn scientific_capacity_curve_publishes_four_methods() {
        let report = run_capacity_curve(CapacityCurveConfig {
            nodes: 32,
            pattern_counts: vec![1, 2],
            corruption_fractions: vec![0.20],
            trials_per_corruption: 2,
            seeds: vec![0xCA12_0001, 0xCA12_0002],
            epsilon: 0.10,
            theta_candidate: 0.80,
            occupancy_floor: 0.50,
            exponential_beta: 8.0,
        })
        .unwrap();
        eprintln!(
            "scientific_capacity k_max={} rows={} wall={:.3}s",
            report.k_max_phasor,
            report.rows.len(),
            report.wall_clock_seconds
        );
        for row in &report.rows {
            eprintln!(
                "  N={} K={} ρ={:.2} {} rec={:.3}±{:.3} dR={:.3}±{:.3} E={:.4} wall={:.3}s",
                row.nodes,
                row.patterns,
                row.corruption_fraction,
                row.method,
                row.recovery_mean,
                row.recovery_std,
                row.delta_r_mean,
                row.delta_r_std,
                row.mean_model_energy,
                row.wall_clock_seconds
            );
        }
        let methods: Vec<_> = report.rows.iter().map(|row| row.method).collect();
        assert!(methods.contains(&"fasorial_commit"), "{report:#?}");
        assert!(methods.contains(&"hebb_acumulativo"), "{report:#?}");
        assert!(methods.contains(&"hopfield"), "{report:#?}");
        assert!(methods.contains(&"hopfield_exponencial"), "{report:#?}");
        assert!(
            report.rows.iter().any(|row| row.patterns >= 2),
            "{report:#?}"
        );
    }
}
