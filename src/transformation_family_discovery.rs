//! Selección de familia de transformación con penalización de complejidad.
//!
//! Las familias candidatas están definidas por el benchmark, pero tanto la
//! familia como sus parámetros son desconocidos para el selector. La energía de
//! hipótesis combina error robusto y longitud descriptiva.

use crate::matrix_free_cognitive_substrate::SparsePattern;
use crate::native_rng::{splitmix64, unit_f64_from_u64};
use rayon::prelude::*;
use serde::Serialize;
use std::time::Instant;

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum TransformationFamily {
    Translation,
    Rotation,
    Reflection,
    Permutation,
    Composition,
}

#[derive(Clone, Debug)]
pub struct FamilyExample {
    pub source: SparsePattern,
    pub target: SparsePattern,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TransformationHypothesis {
    pub family: TransformationFamily,
    pub parameter_a: i32,
    pub parameter_b: i32,
    pub mapping: Vec<usize>,
    pub robust_data_error: f64,
    pub complexity: f64,
    pub energy: f64,
}

impl TransformationHypothesis {
    pub fn predict(&self, source: &SparsePattern) -> SparsePattern {
        apply_mapping(source, &self.mapping)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilySelectionReport {
    pub selected: Option<TransformationHypothesis>,
    pub runner_up_energy: f64,
    pub energy_margin: f64,
    pub hypotheses_evaluated: usize,
    pub ambiguous: bool,
}

pub fn discover_transformation_family(
    examples: &[FamilyExample],
    grid_size: usize,
    complexity_weight: f64,
    maximum_data_error: f64,
    minimum_energy_margin: f64,
) -> FamilySelectionReport {
    if grid_size < 2 || examples.len() < 3 {
        return FamilySelectionReport {
            selected: None,
            runner_up_energy: f64::INFINITY,
            energy_margin: 0.0,
            hypotheses_evaluated: 0,
            ambiguous: true,
        };
    }
    let channels = grid_size * grid_size;
    let mut hypotheses = candidate_hypotheses(examples, grid_size);
    for hypothesis in &mut hypotheses {
        hypothesis.robust_data_error =
            robust_mapping_error(&hypothesis.mapping, examples, channels);
        hypothesis.energy =
            hypothesis.robust_data_error + complexity_weight.max(0.0) * hypothesis.complexity;
    }
    hypotheses.sort_by(|left, right| {
        left.energy
            .total_cmp(&right.energy)
            .then_with(|| left.complexity.total_cmp(&right.complexity))
            .then_with(|| family_rank(left.family).cmp(&family_rank(right.family)))
            .then_with(|| left.parameter_a.cmp(&right.parameter_a))
            .then_with(|| left.parameter_b.cmp(&right.parameter_b))
    });
    let Some(best) = hypotheses.first().cloned() else {
        return FamilySelectionReport {
            selected: None,
            runner_up_energy: f64::INFINITY,
            energy_margin: 0.0,
            hypotheses_evaluated: 0,
            ambiguous: true,
        };
    };
    let runner_up_energy = hypotheses.get(1).map_or(f64::INFINITY, |item| item.energy);
    let energy_margin = (runner_up_energy - best.energy).max(0.0);
    let ambiguous = best.robust_data_error > maximum_data_error.max(0.0)
        || energy_margin < minimum_energy_margin.max(0.0);
    FamilySelectionReport {
        selected: (!ambiguous).then_some(best),
        runner_up_energy,
        energy_margin,
        hypotheses_evaluated: hypotheses.len(),
        ambiguous,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FamilyDiscoveryBenchmarkConfig {
    pub trials_per_family: usize,
    pub complexity_weight: f64,
    pub minimum_rate: f64,
    /// Error de datos máximo para aceptar una hipótesis seleccionada.
    pub maximum_data_error: f64,
    /// Margen mínimo de energía sobre la hipótesis rival para no abstenerse.
    pub minimum_energy_margin: f64,
    /// Umbral de error robusto para contar una recuperación ruidosa como éxito.
    pub robust_error_threshold: f64,
    /// Error máximo del detector de evidencia ambigua (estricto: casi todo
    /// empate debe producir abstención).
    pub ambiguity_maximum_error: f64,
}

impl Default for FamilyDiscoveryBenchmarkConfig {
    fn default() -> Self {
        Self {
            trials_per_family: 10,
            complexity_weight: 1.0e-4,
            minimum_rate: 0.90,
            maximum_data_error: 5.0e-4,
            minimum_energy_margin: 2.0e-5,
            robust_error_threshold: 5.0e-4,
            ambiguity_maximum_error: 1.0e-12,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct FamilyDiscoveryBenchmarkReport {
    pub trials: usize,
    pub families: usize,
    pub family_identification_accuracy: f64,
    pub translation_identification_accuracy: f64,
    pub rotation_identification_accuracy: f64,
    pub reflection_identification_accuracy: f64,
    pub permutation_identification_accuracy: f64,
    pub composition_identification_accuracy: f64,
    pub parameter_mapping_accuracy: f64,
    pub heldout_transfer_accuracy: f64,
    pub noisy_outlier_robustness: f64,
    pub minimum_complexity_preference: f64,
    pub ambiguous_evidence_abstention: f64,
    pub mean_robust_data_error: f64,
    pub mean_selected_complexity: f64,
    pub mean_mdl_advantage_over_permutation: f64,
    pub mean_selected_energy_margin: f64,
    /// Tiempo de pared total del benchmark (segundos).
    pub wall_clock_seconds: f64,
    pub decision: &'static str,
}

pub fn run_family_discovery_benchmark(
    config: FamilyDiscoveryBenchmarkConfig,
) -> FamilyDiscoveryBenchmarkReport {
    let started = Instant::now();
    let trials_per_family = config.trials_per_family.max(1);
    let families = [
        TransformationFamily::Translation,
        TransformationFamily::Rotation,
        TransformationFamily::Reflection,
        TransformationFamily::Permutation,
        TransformationFamily::Composition,
    ];

    // Los ensayos (familia, trial) son independientes y su semilla deriva de
    // ambos índices: se calculan en paralelo. La reducción en orden conserva
    // la semántica original, incluidos los trials sin selección (contribuyen
    // cero a todas las sumas, como hacía el `continue`).
    let outcomes = (0..families.len() * trials_per_family)
        .into_par_iter()
        .map(|flat| {
            let family_index = flat / trials_per_family;
            let trial = flat % trials_per_family;
            run_family_trial(
                families[family_index],
                &families,
                family_index,
                trial,
                &config,
            )
        })
        .collect::<Vec<_>>();

    let mut family_successes = 0;
    let mut family_successes_by_kind = [0usize; 5];
    let mut mapping_successes = 0;
    let mut transfer_successes = 0;
    let mut robust_successes = 0;
    let mut simplicity_successes = 0;
    let mut simplicity_trials = 0;
    let mut ambiguity_successes = 0;
    let mut margin_sum = 0.0;
    let mut data_error_sum = 0.0;
    let mut complexity_sum = 0.0;
    let mut mdl_advantage_sum = 0.0;
    for (flat, outcome) in outcomes.iter().enumerate() {
        let family_index = flat / trials_per_family;
        family_successes += outcome.family_successes;
        family_successes_by_kind[family_index] += outcome.family_successes;
        mapping_successes += outcome.mapping_successes;
        transfer_successes += outcome.transfer_successes;
        robust_successes += outcome.robust_successes;
        simplicity_successes += outcome.simplicity_successes;
        simplicity_trials += outcome.simplicity_trials;
        ambiguity_successes += outcome.ambiguity_successes;
        margin_sum += outcome.margin_sum;
        data_error_sum += outcome.data_error_sum;
        complexity_sum += outcome.complexity_sum;
        mdl_advantage_sum += outcome.mdl_advantage_sum;
    }

    let total = (families.len() * trials_per_family) as f64;
    let minimum_rate = config.minimum_rate.clamp(0.0, 1.0);
    let mut report = FamilyDiscoveryBenchmarkReport {
        trials: total as usize,
        families: families.len(),
        family_identification_accuracy: family_successes as f64 / total,
        translation_identification_accuracy: family_successes_by_kind[0] as f64
            / trials_per_family as f64,
        rotation_identification_accuracy: family_successes_by_kind[1] as f64
            / trials_per_family as f64,
        reflection_identification_accuracy: family_successes_by_kind[2] as f64
            / trials_per_family as f64,
        permutation_identification_accuracy: family_successes_by_kind[3] as f64
            / trials_per_family as f64,
        composition_identification_accuracy: family_successes_by_kind[4] as f64
            / trials_per_family as f64,
        parameter_mapping_accuracy: mapping_successes as f64 / total,
        heldout_transfer_accuracy: transfer_successes as f64 / total,
        noisy_outlier_robustness: robust_successes as f64 / total,
        minimum_complexity_preference: simplicity_successes as f64
            / simplicity_trials.max(1) as f64,
        ambiguous_evidence_abstention: ambiguity_successes as f64 / total,
        mean_robust_data_error: data_error_sum / total,
        mean_selected_complexity: complexity_sum / total,
        mean_mdl_advantage_over_permutation: mdl_advantage_sum / simplicity_trials.max(1) as f64,
        mean_selected_energy_margin: margin_sum / total,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        decision: "transformation_family_discovery_not_demonstrated",
    };
    if report.family_identification_accuracy >= minimum_rate
        && report.translation_identification_accuracy >= minimum_rate
        && report.rotation_identification_accuracy >= minimum_rate
        && report.reflection_identification_accuracy >= minimum_rate
        && report.permutation_identification_accuracy >= minimum_rate
        && report.composition_identification_accuracy >= minimum_rate
        && report.parameter_mapping_accuracy >= minimum_rate
        && report.heldout_transfer_accuracy >= minimum_rate
        && report.noisy_outlier_robustness >= minimum_rate
        && report.minimum_complexity_preference >= minimum_rate
        && report.ambiguous_evidence_abstention >= minimum_rate
    {
        report.decision = "family_parameter_mdl_discovery_pass";
    }
    report
}

#[derive(Clone, Copy, Debug, Default)]
struct FamilyTrialOutcome {
    family_successes: usize,
    mapping_successes: usize,
    transfer_successes: usize,
    robust_successes: usize,
    simplicity_successes: usize,
    simplicity_trials: usize,
    ambiguity_successes: usize,
    margin_sum: f64,
    data_error_sum: f64,
    complexity_sum: f64,
    mdl_advantage_sum: f64,
}

fn run_family_trial(
    family: TransformationFamily,
    families: &[TransformationFamily],
    family_index: usize,
    trial: usize,
    config: &FamilyDiscoveryBenchmarkConfig,
) -> FamilyTrialOutcome {
    let mut outcome = FamilyTrialOutcome::default();
    let seed = 0xFA61_1E00 ^ (family_index as u64).rotate_left(17) ^ trial as u64;
    let grid_size = 5;
    let channels = grid_size * grid_size;
    let expected = actual_hypothesis(family, grid_size, seed);
    let outlier = actual_hypothesis(
        families[(family_index + 2) % families.len()],
        grid_size,
        seed.rotate_left(23),
    );
    let mut examples = Vec::new();
    for example in 0..4 {
        let source = coverage_pattern(channels, example, seed);
        examples.push(FamilyExample {
            target: noisy_target(
                &apply_mapping(&source, &expected.mapping),
                seed ^ example as u64,
            ),
            source,
        });
    }
    let outlier_source = coverage_pattern(channels, 4, seed);
    examples.push(FamilyExample {
        target: noisy_target(
            &apply_mapping(&outlier_source, &outlier.mapping),
            seed.rotate_left(31),
        ),
        source: outlier_source,
    });

    let selection = discover_transformation_family(
        &examples,
        grid_size,
        config.complexity_weight,
        config.maximum_data_error,
        config.minimum_energy_margin,
    );
    let Some(selected) = selection.selected else {
        return outcome;
    };
    outcome.margin_sum += selection.energy_margin;
    outcome.family_successes += usize::from(selected.family == family);
    outcome.data_error_sum += selected.robust_data_error;
    outcome.complexity_sum += selected.complexity;
    outcome.mapping_successes += usize::from(selected.mapping == expected.mapping);
    outcome.robust_successes += usize::from(
        selected.mapping == expected.mapping
            && selected.robust_data_error <= config.robust_error_threshold,
    );

    let heldout = heldout_pattern(channels, seed.rotate_left(41));
    let expected_output = apply_mapping(&heldout, &expected.mapping);
    let predicted = selected.predict(&heldout);
    outcome.transfer_successes +=
        usize::from(pattern_error(&predicted, &expected_output, channels) <= EPSILON);

    if family != TransformationFamily::Permutation {
        outcome.simplicity_trials += 1;
        let permutation = inferred_permutation_hypothesis(&examples, channels);
        let permutation_error = robust_mapping_error(&permutation.mapping, &examples, channels);
        let permutation_energy =
            permutation_error + config.complexity_weight * permutation.complexity;
        outcome.simplicity_successes += usize::from(
            selected.complexity < permutation.complexity && selected.energy < permutation_energy,
        );
        outcome.mdl_advantage_sum += permutation_energy - selected.energy;
    }

    let symmetric = symmetric_ambiguous_examples(channels);
    let ambiguous = discover_transformation_family(
        &symmetric,
        grid_size,
        config.complexity_weight,
        config.ambiguity_maximum_error,
        config.minimum_energy_margin,
    );
    outcome.ambiguity_successes += usize::from(ambiguous.ambiguous && ambiguous.selected.is_none());
    outcome
}

fn candidate_hypotheses(
    examples: &[FamilyExample],
    grid_size: usize,
) -> Vec<TransformationHypothesis> {
    let mut hypotheses = Vec::new();
    for dx in 0..grid_size {
        for dy in 0..grid_size {
            hypotheses.push(hypothesis(
                TransformationFamily::Translation,
                dx as i32,
                dy as i32,
                translation_mapping(grid_size, dx, dy),
                1.0,
            ));
        }
    }
    for quarter_turns in 1..=3 {
        hypotheses.push(hypothesis(
            TransformationFamily::Rotation,
            quarter_turns,
            0,
            rotation_mapping(grid_size, quarter_turns as usize),
            1.2,
        ));
    }
    for reflection in 0..4 {
        hypotheses.push(hypothesis(
            TransformationFamily::Reflection,
            reflection,
            0,
            reflection_mapping(grid_size, reflection as usize),
            1.2,
        ));
    }
    let translations = (0..grid_size)
        .flat_map(|dx| (0..grid_size).map(move |dy| (dx, dy)))
        .filter(|&(dx, dy)| dx != 0 || dy != 0)
        .collect::<Vec<_>>();
    for quarter_turns in 1..=3 {
        for &(dx, dy) in &translations {
            hypotheses.push(hypothesis(
                TransformationFamily::Composition,
                quarter_turns,
                (dx * grid_size + dy) as i32,
                compose_mapping(
                    &rotation_mapping(grid_size, quarter_turns as usize),
                    &translation_mapping(grid_size, dx, dy),
                ),
                3.0,
            ));
        }
    }
    for reflection in 0..4 {
        for &(dx, dy) in &translations {
            hypotheses.push(hypothesis(
                TransformationFamily::Composition,
                -(reflection + 1),
                (dx * grid_size + dy) as i32,
                compose_mapping(
                    &reflection_mapping(grid_size, reflection as usize),
                    &translation_mapping(grid_size, dx, dy),
                ),
                3.0,
            ));
        }
    }
    hypotheses.push(inferred_permutation_hypothesis(
        examples,
        grid_size * grid_size,
    ));
    hypotheses
}

fn actual_hypothesis(
    family: TransformationFamily,
    grid_size: usize,
    seed: u64,
) -> TransformationHypothesis {
    match family {
        TransformationFamily::Translation => {
            let dx = 1 + splitmix64(seed) as usize % (grid_size - 1);
            let dy = splitmix64(seed.rotate_left(7)) as usize % grid_size;
            hypothesis(
                family,
                dx as i32,
                dy as i32,
                translation_mapping(grid_size, dx, dy),
                1.0,
            )
        }
        TransformationFamily::Rotation => {
            let turns = 1 + splitmix64(seed) as usize % 3;
            hypothesis(
                family,
                turns as i32,
                0,
                rotation_mapping(grid_size, turns),
                1.2,
            )
        }
        TransformationFamily::Reflection => {
            let reflection = splitmix64(seed) as usize % 4;
            hypothesis(
                family,
                reflection as i32,
                0,
                reflection_mapping(grid_size, reflection),
                1.2,
            )
        }
        TransformationFamily::Permutation => hypothesis(
            family,
            0,
            0,
            random_permutation(grid_size * grid_size, seed),
            8.0,
        ),
        TransformationFamily::Composition => {
            let turns = 1 + splitmix64(seed) as usize % 3;
            let dx = 1 + splitmix64(seed.rotate_left(11)) as usize % (grid_size - 1);
            let dy = 1 + splitmix64(seed.rotate_left(29)) as usize % (grid_size - 1);
            hypothesis(
                family,
                turns as i32,
                (dx * grid_size + dy) as i32,
                compose_mapping(
                    &rotation_mapping(grid_size, turns),
                    &translation_mapping(grid_size, dx, dy),
                ),
                3.0,
            )
        }
    }
}

fn hypothesis(
    family: TransformationFamily,
    parameter_a: i32,
    parameter_b: i32,
    mapping: Vec<usize>,
    complexity: f64,
) -> TransformationHypothesis {
    TransformationHypothesis {
        family,
        parameter_a,
        parameter_b,
        mapping,
        robust_data_error: f64::INFINITY,
        complexity,
        energy: f64::INFINITY,
    }
}

fn inferred_permutation_hypothesis(
    examples: &[FamilyExample],
    channels: usize,
) -> TransformationHypothesis {
    let mut votes = vec![vec![0usize; channels]; channels];
    for example in examples {
        for &(source_channel, source_value) in &example.source.components {
            let target = example
                .target
                .components
                .iter()
                .min_by(|left, right| {
                    (left.1 - source_value)
                        .abs()
                        .total_cmp(&(right.1 - source_value).abs())
                })
                .map(|&(channel, _)| channel);
            if source_channel < channels {
                if let Some(target_channel) = target.filter(|channel| *channel < channels) {
                    votes[source_channel][target_channel] += 1;
                }
            }
        }
    }
    let mut mapping = (0..channels).collect::<Vec<_>>();
    let mut used = vec![false; channels];
    let mut ranked_sources = (0..channels)
        .map(|source| {
            let best_votes = votes[source].iter().copied().max().unwrap_or(0);
            (best_votes, source)
        })
        .collect::<Vec<_>>();
    ranked_sources.sort_by(|left, right| right.cmp(left));
    for (_, source) in ranked_sources {
        let target = (0..channels)
            .filter(|target| !used[*target])
            .max_by_key(|target| (votes[source][*target], usize::MAX - *target));
        if let Some(target) = target {
            mapping[source] = target;
            used[target] = true;
        }
    }
    let moved = mapping
        .iter()
        .enumerate()
        .filter(|(source, target)| *source != **target)
        .count();
    hypothesis(
        TransformationFamily::Permutation,
        moved as i32,
        0,
        mapping,
        8.0 + moved as f64 * 0.2,
    )
}

fn robust_mapping_error(mapping: &[usize], examples: &[FamilyExample], channels: usize) -> f64 {
    let mut errors = examples
        .iter()
        .map(|example| {
            pattern_error(
                &apply_mapping(&example.source, mapping),
                &example.target,
                channels,
            )
        })
        .collect::<Vec<_>>();
    errors.sort_by(f64::total_cmp);
    errors[errors.len() / 2]
}

fn translation_mapping(grid_size: usize, dx: usize, dy: usize) -> Vec<usize> {
    (0..grid_size * grid_size)
        .map(|index| {
            let x = index % grid_size;
            let y = index / grid_size;
            ((y + dy) % grid_size) * grid_size + (x + dx) % grid_size
        })
        .collect()
}

fn rotation_mapping(grid_size: usize, quarter_turns: usize) -> Vec<usize> {
    (0..grid_size * grid_size)
        .map(|index| {
            let mut x = index % grid_size;
            let mut y = index / grid_size;
            for _ in 0..quarter_turns % 4 {
                (x, y) = (grid_size - 1 - y, x);
            }
            y * grid_size + x
        })
        .collect()
}

fn reflection_mapping(grid_size: usize, reflection: usize) -> Vec<usize> {
    (0..grid_size * grid_size)
        .map(|index| {
            let x = index % grid_size;
            let y = index / grid_size;
            let (next_x, next_y) = match reflection % 4 {
                0 => (grid_size - 1 - x, y),
                1 => (x, grid_size - 1 - y),
                2 => (y, x),
                _ => (grid_size - 1 - y, grid_size - 1 - x),
            };
            next_y * grid_size + next_x
        })
        .collect()
}

fn compose_mapping(first: &[usize], second: &[usize]) -> Vec<usize> {
    first.iter().map(|&index| second[index]).collect()
}

fn random_permutation(channels: usize, seed: u64) -> Vec<usize> {
    let mut ranked = (0..channels)
        .map(|channel| (splitmix64(seed ^ channel as u64), channel))
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let mut mapping = vec![0; channels];
    for (target, (_, source)) in ranked.into_iter().enumerate() {
        mapping[source] = target;
    }
    mapping
}

fn coverage_pattern(channels: usize, omitted_group: usize, seed: u64) -> SparsePattern {
    SparsePattern::new((0..channels).filter_map(|channel| {
        (channel % 5 != omitted_group % 5).then_some((
            channel,
            0.75 + channel as f64 / channels as f64
                + 0.01 * unit_f64_from_u64(splitmix64(seed ^ channel as u64)),
        ))
    }))
}

fn heldout_pattern(channels: usize, seed: u64) -> SparsePattern {
    SparsePattern::new((0..channels).filter_map(|channel| {
        (!splitmix64(seed ^ channel as u64).is_multiple_of(3)).then_some((
            channel,
            0.65 + channel as f64 / channels as f64
                + 0.02 * unit_f64_from_u64(splitmix64(seed.rotate_left(17) ^ channel as u64)),
        ))
    }))
}

fn noisy_target(pattern: &SparsePattern, seed: u64) -> SparsePattern {
    SparsePattern::new(
        pattern
            .components
            .iter()
            .enumerate()
            .map(|(index, &(channel, value))| {
                let noise =
                    0.004 * (2.0 * unit_f64_from_u64(splitmix64(seed ^ index as u64)) - 1.0);
                (channel, value + noise)
            }),
    )
}

fn symmetric_ambiguous_examples(channels: usize) -> Vec<FamilyExample> {
    let pattern = SparsePattern::new((0..channels).map(|channel| (channel, 1.0)));
    (0..3)
        .map(|_| FamilyExample {
            source: pattern.clone(),
            target: pattern.clone(),
        })
        .collect()
}

fn apply_mapping(pattern: &SparsePattern, mapping: &[usize]) -> SparsePattern {
    SparsePattern::new(
        pattern
            .components
            .iter()
            .filter_map(|&(channel, value)| mapping.get(channel).map(|&target| (target, value))),
    )
}

fn pattern_error(left: &SparsePattern, right: &SparsePattern, channels: usize) -> f64 {
    let mut left_dense = vec![0.0; channels];
    let mut right_dense = vec![0.0; channels];
    for &(channel, value) in &left.components {
        if channel < channels {
            left_dense[channel] = value;
        }
    }
    for &(channel, value) in &right.components {
        if channel < channels {
            right_dense[channel] = value;
        }
    }
    left_dense
        .iter()
        .zip(right_dense)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / channels.max(1) as f64
}

fn family_rank(family: TransformationFamily) -> usize {
    match family {
        TransformationFamily::Translation => 0,
        TransformationFamily::Rotation => 1,
        TransformationFamily::Reflection => 2,
        TransformationFamily::Composition => 3,
        TransformationFamily::Permutation => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_family_parameters_and_prefers_minimum_description() {
        let report = run_family_discovery_benchmark(FamilyDiscoveryBenchmarkConfig {
            trials_per_family: 3,
            ..FamilyDiscoveryBenchmarkConfig::default()
        });
        assert_eq!(
            report.decision, "family_parameter_mdl_discovery_pass",
            "{report:#?}"
        );
    }
}
